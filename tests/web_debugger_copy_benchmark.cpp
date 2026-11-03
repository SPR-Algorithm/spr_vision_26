// Controlled, headless allocator comparison. Reports measurements; no performance gates.
// Source images and reusable destinations are allocated before any measurements.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

namespace {
using Clock = std::chrono::steady_clock;
constexpr int kWidth = 1440, kHeight = 1080;
constexpr size_t kSources = 8, kRawBuffers = 3, kPaths = 4;
constexpr size_t kWarmup = 100, kRounds = 3, kBlocks = 60, kBlockSamples = 50;
constexpr size_t kSamples = kBlocks * kBlockSamples;
constexpr uint32_t kSeed = 20261103;
const std::array<const char *, kPaths> kNames{{
  "legacy_resize_640", "fresh_raw_copy", "reuse_one_raw_buffer", "reuse_three_raw_buffers",
}};

double milliseconds(Clock::duration duration)
{
  return std::chrono::duration<double, std::milli>(duration).count();
}

void require(bool condition, const char * message)
{
  if (!condition) throw std::runtime_error(message);
}

struct Measurements {
  // Fixed capacity throughout collection: recording never allocates.
  std::vector<double> copy_ms = std::vector<double>(kRounds * kSamples);
  std::vector<double> copy_release_ms = std::vector<double>(kRounds * kSamples);
  size_t count = 0;
};

nlohmann::json distribution(const std::vector<double> & samples, size_t begin, size_t end)
{
  std::vector<double> sorted(samples.begin() + begin, samples.begin() + end);
  require(!sorted.empty(), "empty benchmark distribution");
  std::sort(sorted.begin(), sorted.end());
  const auto percentile = [&](double fraction) {
    return sorted[static_cast<size_t>(std::ceil(fraction * sorted.size())) - 1];
  };
  return {
    {"count", sorted.size()},
    {"mean", std::accumulate(sorted.begin(), sorted.end(), 0.0) / sorted.size()},
    {"p50", percentile(0.50)}, {"p95", percentile(0.95)},
    {"p99", percentile(0.99)}, {"max", sorted.back()},
  };
}

nlohmann::json summarize(const Measurements & measurements, size_t begin, size_t end)
{
  return {{"copy_ms", distribution(measurements.copy_ms, begin, end)},
          {"copy_plus_release_ms", distribution(measurements.copy_release_ms, begin, end)}};
}

class Comparison {
public:
  Comparison()
  {
    cv::RNG random(kSeed);
    for (auto & source : sources_) {
      source.create(kHeight, kWidth, CV_8UC3);
      random.fill(source, cv::RNG::UNIFORM, 0, 256);
    }
    for (auto & buffer : raw_pool_) buffer.create(kHeight, kWidth, CV_8UC3);
  }

  void sanity()
  {
    // Full image reads and correctness checks stay outside the timed samples.
    const auto & source = sources_.front();
    cv::Mat fresh, resized;
    source.copyTo(fresh);
    source.copyTo(raw_pool_.front());
    require(fresh.data != source.data && raw_pool_.front().data != source.data,
      "raw destination aliases its source");
    require(fresh.size() == source.size() && fresh.type() == source.type() &&
      cv::norm(source, fresh, cv::NORM_INF) == 0 &&
      cv::norm(source, raw_pool_.front(), cv::NORM_INF) == 0, "raw copy is incorrect");
    const double scale = 640.0 / kWidth;
    cv::resize(source, resized, {}, scale, scale);
    require(resized.cols == 640 && resized.rows == 480 && resized.type() == CV_8UC3,
      "legacy resize dimensions are incorrect");
    require(cv::mean(resized)[0] > 0, "legacy resize produced empty pixels");
  }

  void sample(size_t path, Measurements * result)
  {
    const auto & source = sources_[source_indices_[path]++ % kSources];
    cv::Mat fresh;
    auto & destination = path >= 2
      ? raw_pool_[path == 2 ? 0 : pool_index_++ % kRawBuffers] : fresh;
    const auto start = Clock::now();
    if (path == 0) {
      const double scale = 640.0 / kWidth;
      cv::resize(source, destination, {}, scale, scale);
    } else {
      source.copyTo(destination);
    }
    const auto copied = Clock::now();
    if (path < 2) destination.release();
    const auto released = Clock::now();
    if (result) {
      const size_t index = result->count++;
      result->copy_ms[index] = milliseconds(copied - start);
      result->copy_release_ms[index] = milliseconds(released - start);
    }
  }

private:
  std::array<cv::Mat, kSources> sources_;
  std::array<cv::Mat, kRawBuffers> raw_pool_;
  std::array<size_t, kPaths> source_indices_{};
  size_t pool_index_ = 0;
};
}  // namespace

int main()
{
  try {
    Comparison comparison;
    std::array<Measurements, kPaths> measurements;
    comparison.sanity();
    for (size_t path = 0; path < kPaths; ++path)
      for (size_t i = 0; i < kWarmup; ++i) comparison.sample(path, nullptr);

    nlohmann::json result = {
      {"source", {{"width", kWidth}, {"height", kHeight}, {"type", "CV_8UC3"},
                   {"bytes", kWidth * kHeight * 3}, {"ring_images", kSources}}},
      {"settings", {{"rounds", kRounds}, {"blocks_per_path_per_round", kBlocks},
                    {"samples_per_block", kBlockSamples}, {"warmup_per_path", kWarmup},
                    {"raw_pool_buffers", kRawBuffers}, {"shuffle_seed", kSeed},
                    {"opencv_threads", cv::getNumThreads()}}},
      {"timing_notes", "copy_ms includes destination allocation or reuse and the image operation; "
                        "copy_plus_release_ms also includes release for fresh destinations and one "
                        "intermediate clock read. Reused destinations stay allocated. "
                        "No CPU affinity or OpenCV thread settings are changed."},
      {"rounds", nlohmann::json::array()},
    };
    const auto all_start = Clock::now();
    std::mt19937 random(kSeed);
    for (size_t round = 0; round < kRounds; ++round) {
      std::array<size_t, kPaths * kBlocks> schedule;
      for (size_t path = 0; path < kPaths; ++path)
        std::fill_n(schedule.begin() + path * kBlocks, kBlocks, path);
      std::shuffle(schedule.begin(), schedule.end(), random);
      const auto round_start = Clock::now();
      for (const size_t path : schedule)
        for (size_t i = 0; i < kBlockSamples; ++i)
          comparison.sample(path, &measurements[path]);
      nlohmann::json report = {{"round", round + 1},
        {"measurement_seconds", milliseconds(Clock::now() - round_start) / 1000.0}};
      for (size_t path = 0; path < kPaths; ++path)
        report["paths"][kNames[path]] = summarize(measurements[path], round * kSamples,
                                                (round + 1) * kSamples);
      result["rounds"].push_back(std::move(report));
    }
    result["elapsed_seconds"] = milliseconds(Clock::now() - all_start) / 1000.0;
    for (size_t path = 0; path < kPaths; ++path) {
      require(measurements[path].count == kRounds * kSamples, "sample accounting mismatch");
      result["aggregate"][kNames[path]] = summarize(measurements[path], 0, measurements[path].count);
    }
    comparison.sanity();
    std::cout << result.dump(2) << '\n';
    return 0;
  } catch (const std::exception & error) {
    std::cerr << "web_debugger_copy_benchmark: " << error.what() << '\n';
    return 1;
  }
}
