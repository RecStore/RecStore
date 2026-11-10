#pragma once
#include <cstdint>
#include <memory>
#include <string>

extern "C" {

bool report(const char* table_name,
            const char* unique_id,
            const char* metric_name,
            double metric_value);
}

bool send_json_request(const std::string& json_payload);

bool is_report_remote_enabled_for_test();
bool is_report_local_jsonl_enabled_for_test();
bool is_report_csv_enabled_for_test();

// Drops the cached observability config. Tests that rewrite
// recstore_config.json or change the working directory call this between cases.
void ResetReportConfigForTest();

namespace recstore {
extern thread_local uint64_t g_trace_id;

// One telemetry sample. report() builds this and passes it to every configured
// sink; a sink reads only the fields it needs.
struct MetricRecord {
  int64_t timestamp_us    = 0;
  const char* host        = "";
  int32_t pid             = 0;
  const char* run_id      = "";
  int32_t rank            = -1;
  int32_t shard           = -1;
  const char* table_name  = "";
  const char* unique_id   = "";
  const char* metric_name = "";
  double metric_value     = 0.0;
  const char* source      = "report";
};

// Extension point for a platform-specific exporter. Registered sinks receive
// every sample in addition to the built-in ones selected by
// `observability.sinks`. Implementations must be thread-safe.
class MetricSink {
public:
  virtual ~MetricSink()                         = default;
  virtual void Emit(const MetricRecord& record) = 0;
  virtual void Flush()                          = 0;
};

// Takes ownership. Register before the first report() call to see every sample.
void RegisterMetricSink(std::unique_ptr<MetricSink> sink);
} // namespace recstore

struct FlameGraphData {
  std::string label;
  double start;
  int level;
  double value;
  double self;
};

extern "C" {
bool report_flame_graph(
    const char* table_name, const char* unique_id, const FlameGraphData& data);
}

enum TimelineState { STATE_RUNNING = 1, STATE_SUCCESS = 2, STATE_FAILED = 3 };

class ReportTimeline {
public:
  ReportTimeline(const std::string& table_name,
                 const std::string& unique_id,
                 const std::string& metric_name = "state");

  ~ReportTimeline();

  void done(TimelineState state = STATE_SUCCESS);

  ReportTimeline(const ReportTimeline&)            = delete;
  ReportTimeline& operator=(const ReportTimeline&) = delete;

private:
  std::string table_name_;
  std::string unique_id_;
  std::string metric_name_;
  bool is_done_;
};
