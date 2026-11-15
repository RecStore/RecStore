#include <gtest/gtest.h>
#include "base/report/report_client.h"
#include "base/timer.h"
#include <sys/wait.h>
#include <unistd.h>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <mutex>
#include <thread>
#include <string>
#include <iostream>
#include <memory>
#include <vector>
#include <nlohmann/json.hpp>

using namespace std::chrono_literals;

namespace {

std::mutex& EnvTestMutex() {
  static std::mutex mutex;
  return mutex;
}

class ScopedEnvVar {
public:
  ScopedEnvVar(const char* key, const char* value) : key_(key) {
    const char* original = std::getenv(key_);
    if (original != nullptr) {
      had_original_   = true;
      original_value_ = original;
    }

    if (value != nullptr) {
      setenv(key_, value, 1);
    } else {
      unsetenv(key_);
    }
  }

  ~ScopedEnvVar() {
    if (had_original_) {
      setenv(key_, original_value_.c_str(), 1);
    } else {
      unsetenv(key_);
    }
  }

private:
  const char* key_;
  bool had_original_ = false;
  std::string original_value_;
};

class ScopedCurrentPath {
public:
  explicit ScopedCurrentPath(const std::filesystem::path& path)
      : original_(std::filesystem::current_path()) {
    std::filesystem::current_path(path);
  }

  ~ScopedCurrentPath() { std::filesystem::current_path(original_); }

private:
  std::filesystem::path original_;
};

} // namespace

TEST(ReportTest, TimelineBasic) {
  {
    ReportTimeline tl("cpp_easy_test_timeline_map", "timeline_test_2");
    std::cout << "Running phase ... " << std::endl;
    std::this_thread::sleep_for(1s);
  }

  std::this_thread::sleep_for(500ms);
  SUCCEED();
}

TEST(ReportTest, ReportModeCanBeDisabledByEnvVar) {
  ResetReportConfigForTest();
  std::lock_guard<std::mutex> lock(EnvTestMutex());
  ScopedEnvVar report_mode("RECSTORE_REPORT_MODE", "local");
  xmh::Timer::Init();

  EXPECT_FALSE(is_report_remote_enabled_for_test());
  EXPECT_TRUE(
      report("cpp_easy_test_report_map", "env_local_test", "duration_us", 1.0));
  EXPECT_DOUBLE_EQ(
      xmh::Timer::ManualQuery("cpp_easy_test_report_map.duration_us"), 1000.0);
}

TEST(ReportTest, EmptyReportApiDisablesRemoteReporting) {
  ResetReportConfigForTest();
  std::lock_guard<std::mutex> lock(EnvTestMutex());
  ScopedEnvVar report_mode("RECSTORE_REPORT_MODE", nullptr);

  const std::filesystem::path temp_dir =
      std::filesystem::temp_directory_path() / "recstore_test_report_empty_api";
  std::filesystem::create_directories(temp_dir);

  const std::filesystem::path config_path = temp_dir / "recstore_config.json";
  std::ofstream ofs(config_path);
  ofs << "{\n"
         "  \"report_API\": \"\"\n"
         "}\n";
  ofs.close();

  ScopedCurrentPath cwd_guard(temp_dir);
  EXPECT_FALSE(is_report_remote_enabled_for_test());
  EXPECT_TRUE(report(
      "cpp_easy_test_report_map", "config_local_test", "duration_us", 2.0));

  std::filesystem::remove(config_path);
  std::filesystem::remove(temp_dir);
}

TEST(ReportTest, JsonlSinkWritesStructuredEvent) {
  ResetReportConfigForTest();
  std::lock_guard<std::mutex> lock(EnvTestMutex());
  ScopedEnvVar report_mode("RECSTORE_REPORT_MODE", "local");
  ScopedEnvVar sink_mode("RECSTORE_REPORT_LOCAL_SINK", "both");

  const std::filesystem::path temp_dir =
      std::filesystem::temp_directory_path() /
      "recstore_test_report_jsonl_sink";
  std::filesystem::create_directories(temp_dir);
  const std::filesystem::path output_path = temp_dir / "report_events.jsonl";
  ScopedEnvVar output_file("RECSTORE_REPORT_JSONL_PATH", output_path.c_str());

  EXPECT_TRUE(report(
      "cpp_easy_test_report_map", "jsonl_sink_test", "latency_ns", 3000.0));

  std::ifstream ifs(output_path);
  ASSERT_TRUE(ifs.is_open());
  std::string line;
  ASSERT_TRUE(std::getline(ifs, line));
  ASSERT_FALSE(line.empty());

  auto event = nlohmann::json::parse(line);
  EXPECT_EQ(event["table_name"], "cpp_easy_test_report_map");
  EXPECT_EQ(event["unique_id"], "jsonl_sink_test");
  EXPECT_EQ(event["metric_name"], "latency_ns");
  EXPECT_DOUBLE_EQ(event["metric_value"], 3000.0);
  EXPECT_EQ(event["source"], "report");
  EXPECT_TRUE(event.contains("timestamp_us"));

  std::filesystem::remove(output_path);
  std::filesystem::remove(temp_dir);
}

TEST(ReportTest, JsonlSinkEscapesStructuredEventStrings) {
  ResetReportConfigForTest();
  std::lock_guard<std::mutex> lock(EnvTestMutex());
  ScopedEnvVar report_mode("RECSTORE_REPORT_MODE", "local");
  ScopedEnvVar sink_mode("RECSTORE_REPORT_LOCAL_SINK", "jsonl");

  const std::filesystem::path temp_dir =
      std::filesystem::temp_directory_path() /
      "recstore_test_report_jsonl_escape";
  std::filesystem::create_directories(temp_dir);
  const std::filesystem::path output_path = temp_dir / "report_events.jsonl";
  ScopedEnvVar output_file("RECSTORE_REPORT_JSONL_PATH", output_path.c_str());

  const char* unique_id = "uid\"with\\escapes\nline";
  EXPECT_TRUE(report("cpp_easy_test_report_map", unique_id, "latency_us", 4.0));

  std::ifstream ifs(output_path);
  ASSERT_TRUE(ifs.is_open());
  std::string line;
  ASSERT_TRUE(std::getline(ifs, line));
  ASSERT_FALSE(line.empty());

  auto event = nlohmann::json::parse(line);
  EXPECT_EQ(event["table_name"], "cpp_easy_test_report_map");
  EXPECT_EQ(event["unique_id"], unique_id);
  EXPECT_EQ(event["metric_name"], "latency_us");
  EXPECT_DOUBLE_EQ(event["metric_value"], 4.0);

  std::filesystem::remove(output_path);
  std::filesystem::remove(temp_dir);
}

TEST(ReportTest, FlameGraphData) {
  FlameGraphData root     = {"main", 0.0, 0, 100.0, 10.0};
  FlameGraphData child1   = {"do_work", 10.0, 1, 60.0, 20.0};
  FlameGraphData child2   = {"do_io", 70.0, 1, 30.0, 30.0};
  FlameGraphData child1_1 = {"compute", 20.0, 2, 40.0, 40.0};

  report_flame_graph("cpp_easy_test_flame_map", "flame_test_2", root);
  report_flame_graph("cpp_easy_test_flame_map", "flame_test_2", child1);
  report_flame_graph("cpp_easy_test_flame_map", "flame_test_2", child2);
  report_flame_graph("cpp_easy_test_flame_map", "flame_test_2", child1_1);

  std::this_thread::sleep_for(500ms);
  SUCCEED();
}

namespace {

// Minimal CSV reader: handles the quoting the sink emits (doubled quotes
// inside a quoted field). Enough to verify the long-format rows.
std::vector<std::string> SplitCsvLine(const std::string& line) {
  std::vector<std::string> fields;
  std::string current;
  bool quoted = false;
  for (size_t i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (quoted) {
      if (c == '"' && i + 1 < line.size() && line[i + 1] == '"') {
        current.push_back('"');
        ++i;
      } else if (c == '"') {
        quoted = false;
      } else {
        current.push_back(c);
      }
    } else if (c == '"') {
      quoted = true;
    } else if (c == ',') {
      fields.push_back(current);
      current.clear();
    } else {
      current.push_back(c);
    }
  }
  fields.push_back(current);
  return fields;
}

std::vector<std::string> ReadLines(const std::filesystem::path& path) {
  std::vector<std::string> lines;
  std::ifstream ifs(path);
  std::string line;
  while (std::getline(ifs, line)) {
    lines.push_back(line);
  }
  return lines;
}

struct CapturingSink : public recstore::MetricSink {
  void Emit(const recstore::MetricRecord& record) override {
    table     = record.table_name;
    unique_id = record.unique_id;
    metric    = record.metric_name;
    value     = record.metric_value;
    source    = record.source;
    has_host  = record.host != nullptr && record.host[0] != '\0';
    has_pid   = record.pid > 0;
    ++calls;
  }
  void Flush() override { ++flushes; }

  static int calls;
  static int flushes;
  static std::string table;
  static std::string unique_id;
  static std::string metric;
  static std::string source;
  static double value;
  static bool has_host;
  static bool has_pid;
};

int CapturingSink::calls   = 0;
int CapturingSink::flushes = 0;
std::string CapturingSink::table;
std::string CapturingSink::unique_id;
std::string CapturingSink::metric;
std::string CapturingSink::source;
double CapturingSink::value  = 0.0;
bool CapturingSink::has_host = false;
bool CapturingSink::has_pid  = false;

// The CSV sink keeps one file per process, so tests point it at a unique path.
class ScopedCsvTestEnv {
public:
  ScopedCsvTestEnv(const std::filesystem::path& path,
                   const char* sinks  = "csv",
                   const char* run_id = "csv-test-run")
      : sinks_("RECSTORE_REPORT_SINKS", sinks),
        csv_path_("RECSTORE_REPORT_CSV_PATH", path.c_str()),
        run_id_("RECSTORE_RUN_ID", run_id),
        rank_("RANK", nullptr),
        local_rank_("LOCAL_RANK", nullptr),
        shard_("RECSTORE_SHARD_ID", nullptr),
        legacy_shard_("SHARD_ID", nullptr) {
    ResetReportConfigForTest();
  }

  ~ScopedCsvTestEnv() { ResetReportConfigForTest(); }

private:
  ScopedEnvVar sinks_;
  ScopedEnvVar csv_path_;
  ScopedEnvVar run_id_;
  ScopedEnvVar rank_;
  ScopedEnvVar local_rank_;
  ScopedEnvVar shard_;
  ScopedEnvVar legacy_shard_;
};

} // namespace

TEST(ReportTest, CsvSinkWritesLongFormatRows) {
  std::lock_guard<std::mutex> lock(EnvTestMutex());
  const std::filesystem::path temp_dir =
      std::filesystem::temp_directory_path() / "recstore_test_report_csv";
  std::filesystem::remove_all(temp_dir);
  std::filesystem::create_directories(temp_dir);
  const std::filesystem::path output_path = temp_dir / "metrics.csv";

  {
    ScopedCsvTestEnv env(output_path);
    EXPECT_TRUE(is_report_csv_enabled_for_test());
    EXPECT_FALSE(is_report_remote_enabled_for_test());

    EXPECT_TRUE(report("csv_table", "uid_one", "duration_us", 1.5));
    EXPECT_TRUE(report("csv_table", "uid,two\"quoted", "latency_ns", 2000.0));
  }

  const auto lines = ReadLines(output_path);
  ASSERT_EQ(lines.size(), 3U);
  EXPECT_EQ(lines[0],
            "timestamp_us,host,pid,run_id,rank,shard,table_name,unique_id,"
            "metric_name,metric_value,source");

  const auto first = SplitCsvLine(lines[1]);
  ASSERT_EQ(first.size(), 11U);
  EXPECT_FALSE(first[0].empty());
  EXPECT_FALSE(first[1].empty());
  EXPECT_EQ(first[3], "csv-test-run");
  EXPECT_EQ(first[4], "-1");
  EXPECT_EQ(first[5], "-1");
  EXPECT_EQ(first[6], "csv_table");
  EXPECT_EQ(first[7], "uid_one");
  EXPECT_EQ(first[8], "duration_us");
  EXPECT_EQ(first[9], "1.5");
  EXPECT_EQ(first[10], "report");

  const auto second = SplitCsvLine(lines[2]);
  ASSERT_EQ(second.size(), 11U);
  EXPECT_EQ(second[7], "uid,two\"quoted");
  EXPECT_EQ(second[8], "latency_ns");
  EXPECT_EQ(second[9], "2000");

  std::filesystem::remove_all(temp_dir);
}

TEST(ReportTest, CsvSinkAppendsWithoutRepeatingHeader) {
  std::lock_guard<std::mutex> lock(EnvTestMutex());
  const std::filesystem::path temp_dir =
      std::filesystem::temp_directory_path() /
      "recstore_test_report_csv_append";
  std::filesystem::remove_all(temp_dir);
  std::filesystem::create_directories(temp_dir);
  const std::filesystem::path output_path = temp_dir / "metrics.csv";
  {
    std::ofstream seed(output_path);
    seed << "timestamp_us,host,pid,run_id,rank,shard,table_name,unique_id,"
            "metric_name,metric_value,source\n";
  }

  {
    ScopedCsvTestEnv env(output_path);
    EXPECT_TRUE(report("csv_table", "uid_append", "duration_us", 3.0));
  }

  const auto lines = ReadLines(output_path);
  ASSERT_EQ(lines.size(), 2U);
  EXPECT_EQ(lines[0].rfind("timestamp_us,", 0), 0U);

  std::filesystem::remove_all(temp_dir);
}

TEST(ReportTest, CsvSinkKeepsLinesWholeAcrossProcesses) {
  std::lock_guard<std::mutex> lock(EnvTestMutex());
  const std::filesystem::path temp_dir =
      std::filesystem::temp_directory_path() /
      "recstore_test_report_csv_shared";
  std::filesystem::remove_all(temp_dir);
  std::filesystem::create_directories(temp_dir);
  const std::filesystem::path output_path = temp_dir / "metrics.csv";

  constexpr int kChildren = 3;
  constexpr int kPerChild = 200;

  {
    ScopedCsvTestEnv env(output_path);

    std::vector<pid_t> children;
    for (int child = 0; child < kChildren; ++child) {
      const pid_t pid = ::fork();
      ASSERT_GE(pid, 0);
      if (pid == 0) {
        for (int i = 0; i < kPerChild; ++i) {
          if (!report("csv_shared", "uid_shared", "duration_us", 1.0)) {
            ::_exit(1);
          }
        }
        ::_exit(0);
      }
      children.push_back(pid);
    }

    for (const pid_t pid : children) {
      int status = 0;
      ASSERT_EQ(::waitpid(pid, &status, 0), pid);
      ASSERT_TRUE(WIFEXITED(status));
      ASSERT_EQ(WEXITSTATUS(status), 0);
    }
  }

  const auto lines = ReadLines(output_path);
  ASSERT_EQ(lines.size(), 1U + static_cast<size_t>(kChildren * kPerChild));
  EXPECT_EQ(lines[0].rfind("timestamp_us,", 0), 0U);
  for (size_t i = 1; i < lines.size(); ++i) {
    EXPECT_EQ(SplitCsvLine(lines[i]).size(), 11U) << "line " << i;
  }

  std::filesystem::remove_all(temp_dir);
}

TEST(ReportTest, CsvSinkReadsObservabilityConfigBlock) {
  std::lock_guard<std::mutex> lock(EnvTestMutex());
  const std::filesystem::path temp_dir =
      std::filesystem::temp_directory_path() /
      "recstore_test_report_csv_config";
  std::filesystem::remove_all(temp_dir);
  std::filesystem::create_directories(temp_dir);
  const std::filesystem::path output_path = temp_dir / "from_config.csv";

  {
    std::ofstream ofs(temp_dir / "recstore_config.json");
    ofs << "{\n"
           "  \"observability\": {\n"
           "    \"sinks\": [\"csv\"],\n"
           "    \"csv\": {\"path\": \""
        << output_path.string()
        << "\"}\n"
           "  }\n"
           "}\n";
  }

  {
    ScopedEnvVar sinks("RECSTORE_REPORT_SINKS", nullptr);
    ScopedEnvVar mode("RECSTORE_REPORT_MODE", nullptr);
    ScopedEnvVar local_sink("RECSTORE_REPORT_LOCAL_SINK", nullptr);
    ScopedEnvVar csv_path("RECSTORE_REPORT_CSV_PATH", nullptr);
    ScopedCurrentPath cwd(temp_dir);
    ResetReportConfigForTest();

    EXPECT_TRUE(is_report_csv_enabled_for_test());
    EXPECT_FALSE(is_report_remote_enabled_for_test());
    EXPECT_FALSE(is_report_local_jsonl_enabled_for_test());
    EXPECT_TRUE(report("config_table", "uid_config", "metric", 7.0));
  }
  ResetReportConfigForTest();

  const auto lines = ReadLines(output_path);
  ASSERT_EQ(lines.size(), 2U);
  EXPECT_EQ(SplitCsvLine(lines[1])[6], "config_table");

  std::filesystem::remove_all(temp_dir);
}

TEST(ReportTest, EnvSinkListOverridesConfigFile) {
  std::lock_guard<std::mutex> lock(EnvTestMutex());
  const std::filesystem::path temp_dir =
      std::filesystem::temp_directory_path() / "recstore_test_report_sink_env";
  std::filesystem::remove_all(temp_dir);
  std::filesystem::create_directories(temp_dir);

  {
    std::ofstream ofs(temp_dir / "recstore_config.json");
    ofs << "{\n"
           "  \"observability\": {\"sinks\": [\"csv\"]}\n"
           "}\n";
  }

  {
    ScopedEnvVar sinks("RECSTORE_REPORT_SINKS", "jsonl");
    ScopedEnvVar jsonl_path(
        "RECSTORE_REPORT_JSONL_PATH", (temp_dir / "events.jsonl").c_str());
    ScopedCurrentPath cwd(temp_dir);
    ResetReportConfigForTest();

    EXPECT_FALSE(is_report_csv_enabled_for_test());
    EXPECT_TRUE(is_report_local_jsonl_enabled_for_test());
    EXPECT_TRUE(report("env_table", "uid_env", "metric", 1.0));
  }
  ResetReportConfigForTest();

  std::filesystem::remove_all(temp_dir);
}

TEST(ReportTest, RegisteredSinkReceivesFullRecord) {
  std::lock_guard<std::mutex> lock(EnvTestMutex());
  CapturingSink::calls = 0;

  recstore::RegisterMetricSink(std::make_unique<CapturingSink>());

  const std::filesystem::path temp_dir =
      std::filesystem::temp_directory_path() / "recstore_test_report_user_sink";
  std::filesystem::remove_all(temp_dir);
  std::filesystem::create_directories(temp_dir);
  {
    ScopedEnvVar sinks("RECSTORE_REPORT_SINKS", "none");
    ResetReportConfigForTest();
    EXPECT_TRUE(report("user_table", "uid_user", "duration_us", 42.0));
  }
  ResetReportConfigForTest();

  EXPECT_EQ(CapturingSink::calls, 1);
  EXPECT_EQ(CapturingSink::table, "user_table");
  EXPECT_EQ(CapturingSink::unique_id, "uid_user");
  EXPECT_EQ(CapturingSink::metric, "duration_us");
  EXPECT_DOUBLE_EQ(CapturingSink::value, 42.0);
  EXPECT_EQ(CapturingSink::source, "report");
  EXPECT_TRUE(CapturingSink::has_host);
  EXPECT_TRUE(CapturingSink::has_pid);

  std::filesystem::remove_all(temp_dir);
}

TEST(ReportTest, CsvSinkIsOffByDefault) {
  std::lock_guard<std::mutex> lock(EnvTestMutex());
  const std::filesystem::path temp_dir =
      std::filesystem::temp_directory_path() / "recstore_test_report_csv_off";
  std::filesystem::remove_all(temp_dir);
  std::filesystem::create_directories(temp_dir);

  {
    ScopedEnvVar sinks("RECSTORE_REPORT_SINKS", nullptr);
    ScopedEnvVar mode("RECSTORE_REPORT_MODE", "local");
    ScopedEnvVar local_sink("RECSTORE_REPORT_LOCAL_SINK", nullptr);
    ScopedEnvVar csv_path("RECSTORE_REPORT_CSV_PATH", nullptr);
    ScopedCurrentPath cwd(temp_dir);
    ResetReportConfigForTest();
    EXPECT_FALSE(is_report_csv_enabled_for_test());
  }
  ResetReportConfigForTest();

  std::filesystem::remove_all(temp_dir);
}
