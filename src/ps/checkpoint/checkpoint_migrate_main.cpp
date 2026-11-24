// Offline CLI for checkpoint_migrate: see checkpoint_migrate.h for the file
// format and the safety assumptions.
//
//   checkpoint_migrate --input <checkpoint dir> --output <new dir> \
//       --optimizer <SGD|RowWiseAdagrad|AdamW> [--table-id N] \
//       [--table-ids 0,1,2] [--shards N] [--hash city_hash|simple_mod] \
//       [--no-verify]
//
// The tool never edits a checkpoint in place: --output must not exist. Keys are
// repartitioned with the same hash the client uses, so --hash has to match
// distributed_client.hash_method of the run that will read the result.

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "ps/checkpoint/checkpoint_migrate.h"

namespace {

namespace mig = recstore::checkpoint_migrate;

void PrintUsage(std::ostream& out) {
  out << "usage: checkpoint_migrate --input <dir> --output <dir> "
         "--optimizer <SGD|RowWiseAdagrad|AdamW>\n"
         "                         [--table-id N] [--table-ids 0,1,2] "
         "[--shards N]\n"
         "                         [--hash city_hash|simple_mod] "
         "[--no-verify]\n";
}

bool ParseUint(const std::string& text, uint64_t* value) {
  if (text.empty()) {
    return false;
  }
  uint64_t parsed = 0;
  for (const char c : text) {
    if (c < '0' || c > '9') {
      return false;
    }
    parsed = parsed * 10 + static_cast<uint64_t>(c - '0');
  }
  *value = parsed;
  return true;
}

bool ParseTableIds(const std::string& text, std::vector<uint64_t>* ids) {
  size_t start = 0;
  while (start <= text.size()) {
    const size_t comma     = text.find(',', start);
    const std::string part = text.substr(
        start, comma == std::string::npos ? std::string::npos : comma - start);
    uint64_t id = 0;
    if (!ParseUint(part, &id)) {
      return false;
    }
    ids->push_back(id);
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }
  return !ids->empty();
}

} // namespace

int main(int argc, char** argv) {
  std::string input;
  std::string output;
  std::string optimizer_name;
  std::vector<uint64_t> table_ids;
  mig::Options options;
  bool table_id_given = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next             = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "error: " << name << " requires a value\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--input") {
      input = next("--input");
    } else if (arg == "--output") {
      output = next("--output");
    } else if (arg == "--optimizer") {
      optimizer_name = next("--optimizer");
    } else if (arg == "--table-id") {
      const std::string value = next("--table-id");
      if (!ParseUint(value, &options.table_id)) {
        std::cerr << "error: --table-id must be a non-negative integer\n";
        return 2;
      }
      table_id_given = true;
    } else if (arg == "--table-ids") {
      if (!ParseTableIds(next("--table-ids"), &table_ids)) {
        std::cerr << "error: --table-ids must be a comma separated list\n";
        return 2;
      }
    } else if (arg == "--shards") {
      uint64_t value = 0;
      if (!ParseUint(next("--shards"), &value) || value == 0 || value > 4096) {
        std::cerr << "error: --shards must be between 1 and 4096\n";
        return 2;
      }
      options.num_shards = static_cast<int>(value);
    } else if (arg == "--hash") {
      const std::string value = next("--hash");
      if (value == "city_hash") {
        options.hash = mig::HashMethod::kCityHash;
      } else if (value == "simple_mod") {
        options.hash = mig::HashMethod::kSimpleMod;
      } else {
        std::cerr << "error: --hash must be city_hash or simple_mod\n";
        return 2;
      }
    } else if (arg == "--no-verify") {
      options.verify = false;
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(std::cout);
      return 0;
    } else {
      std::cerr << "error: unknown argument " << arg << "\n";
      PrintUsage(std::cerr);
      return 2;
    }
  }

  if (input.empty() || output.empty() || optimizer_name.empty()) {
    PrintUsage(std::cerr);
    return 2;
  }
  if (optimizer_name == "SGD") {
    options.optimizer = mig::Optimizer::kSgd;
  } else if (optimizer_name == "RowWiseAdagrad" ||
             optimizer_name == "AdaGrad") {
    options.optimizer = mig::Optimizer::kRowWiseAdagrad;
  } else if (optimizer_name == "AdamW") {
    options.optimizer = mig::Optimizer::kAdamW;
  } else {
    std::cerr << "error: --optimizer must be SGD, RowWiseAdagrad or AdamW\n";
    return 2;
  }
  if (options.optimizer == mig::Optimizer::kAdamW) {
    if (!table_id_given) {
      // Old tags did not record the table, so there is nothing to infer from.
      std::cerr << "error: AdamW checkpoints require --table-id: the old key "
                   "layout did not record which table wrote the shard\n";
      return 2;
    }
    if (!table_ids.empty()) {
      std::cerr << "error: --table-ids only applies to SGD/RowWiseAdagrad\n";
      return 2;
    }
  } else if (table_id_given) {
    std::cerr << "error: --table-id only applies to AdamW\n";
    return 2;
  }
  options.table_ids = table_ids;

  std::vector<mig::ShardReport> reports;
  std::string error;
  if (!mig::MigrateCheckpoint(input, output, options, &reports, &error)) {
    std::cerr << "error: " << error << "\n";
    return 1;
  }

  std::cout << "migrated " << reports.size() << " shard(s) from " << input
            << " to " << output << " as "
            << mig::OptimizerName(options.optimizer);
  if (options.optimizer == mig::Optimizer::kAdamW) {
    std::cout << " (table_id " << options.table_id << ")";
  }
  std::cout << "\n";
  for (const auto& report : reports) {
    std::cout << "  " << report.name << ": records=" << report.records
              << " step_rows=" << report.step_rows
              << " bytes=" << report.bytes_written << "\n";
    std::cout << "    tags before:";
    for (const auto& [tag, count] : report.tags_before) {
      std::cout << " " << tag << "=" << count;
    }
    std::cout << "\n    tags after: ";
    for (const auto& [tag, count] : report.tags_after) {
      std::cout << " " << tag << "=" << count;
    }
    std::cout << "\n";
  }
  std::cout << "done; point cache_ps.checkpoint.restore_dir at " << output
            << "\n";
  return 0;
}
