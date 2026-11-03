#pragma once

// Opt-in measurements. Each metric has exactly one writer; summarize after all
// producers and the Web worker have joined. No allocation or logging in record().
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <nlohmann/json.hpp>
#include <vector>

namespace debug {

enum class BenchmarkMetric : size_t {
  Algorithm, FrameLoop, FramePeriod, FrameAge, DebugPrepare, FrameSubmit,
  FrameCopy, Encode, PlanWork, PlanPeriod, TelemetrySubmit, Count
};

class BenchmarkSession {
public:
  using Clock = std::chrono::steady_clock;
  static constexpr size_t capacity = 65536;

  BenchmarkSession(double warmup_seconds, double measure_seconds)
      : begin_(Clock::now() + std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(warmup_seconds))),
        end_(begin_ + std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(measure_seconds))),
        duration_seconds_(measure_seconds) {}

  bool finished(Clock::time_point now = Clock::now()) const { return now >= end_; }
  void record(BenchmarkMetric metric, double ms) {
    const auto now = Clock::now();
    if (now < begin_ || now >= end_ || !std::isfinite(ms)) return;
    auto & d = data_[static_cast<size_t>(metric)];
    ++d.total;
    if (d.count < capacity) d.values[d.count++] = ms;
    else ++d.overflow;
  }

  nlohmann::json summary(Clock::time_point stopped_at = Clock::now()) const {
    static constexpr const char * names[] = {
      "algorithm_ms", "frame_loop_ms", "frame_period_ms", "frame_age_ms",
      "debug_prepare_ms", "frame_submit_ms", "frame_copy_ms", "encode_ms",
      "plan_work_ms", "plan_period_ms", "telemetry_submit_ms"};
    nlohmann::json result;
    for (size_t i = 0; i < data_.size(); ++i) {
      const auto & d = data_[i];
      auto & j = result[names[i]];
      j = {{"count", d.total}, {"retained", d.count}, {"overflow", d.overflow}};
      if (!d.count) continue;
      std::vector<double> values(d.values.begin(), d.values.begin() + d.count);
      std::sort(values.begin(), values.end());
      double sum = 0;
      for (double value : values) sum += value;
      const auto percentile = [&](double p) {
        return values[static_cast<size_t>(std::ceil(p * values.size())) - 1];
      };
      j["mean"] = sum / values.size();
      j["p50"] = percentile(0.50);
      j["p95"] = percentile(0.95);
      j["p99"] = percentile(0.99);
      j["max"] = values.back();
    }
    result["measurement_seconds"] = duration_seconds_;
    result["measurement_complete"] = stopped_at >= end_;
    const double elapsed = std::clamp(
        std::chrono::duration<double>(stopped_at - begin_).count(), 0.0, duration_seconds_);
    result["elapsed_seconds"] = elapsed;
    result["measurement_start_monotonic"] = std::chrono::duration<double>(
        begin_.time_since_epoch()).count();
    result["measurement_end_monotonic"] = std::chrono::duration<double>(
        std::min(stopped_at, end_).time_since_epoch()).count();
    result["measured_frames"] = data_[static_cast<size_t>(BenchmarkMetric::FrameLoop)].total;
    result["vision_fps"] =
        elapsed > 0 ? data_[static_cast<size_t>(BenchmarkMetric::FrameLoop)].total / elapsed : 0;
    return result;
  }

private:
  struct Distribution {
    std::array<double, capacity> values{};
    size_t count = 0;
    size_t total = 0;
    size_t overflow = 0;
  };
  Clock::time_point begin_, end_;
  double duration_seconds_;
  std::array<Distribution, static_cast<size_t>(BenchmarkMetric::Count)> data_{};
};

inline std::atomic<BenchmarkSession *> active_benchmark{nullptr};

inline void benchmark_record(BenchmarkMetric metric, double ms) {
  if (auto * session = active_benchmark.load(std::memory_order_acquire))
    session->record(metric, ms);
}

inline double benchmark_ms(BenchmarkSession::Clock::duration duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

}  // namespace debug
