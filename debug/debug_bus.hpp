#pragma once
// debug_bus.hpp — 调试事件总线
// 用法:
//   DebugBus::load_config(config);
//   DebugBus::post({frame, &armors, reprojs, proc_ms, capture_time});

#include <array>
#include <chrono>
#include <list>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include "web_debugger.hpp"

namespace auto_aim {
struct Armor;
}

namespace debug {

struct ReprojectionData;
struct DetectionData;

// 只在 post() 调用期间有效：sink 需要的数据必须在 push_frame 内复制，不能保存指针或浅拷贝的 Mat
struct FrameDebugData {
  cv::Mat frame;
  const std::list<auto_aim::Armor>* armors = nullptr;
  std::vector<ReprojectionData> reprojections;
  double proc_ms = 0.0;  // 从拿到图像到 post 的处理耗时
  std::chrono::steady_clock::time_point capture_time;
};

class IDebugSink {
public:
  virtual ~IDebugSink() = default;
  virtual void push_frame(const FrameDebugData& data) = 0;
  virtual FramePermit try_begin_frame() noexcept { return FramePermit(this); }
  virtual void push_frame(FramePermit&& permit, const FrameDebugData& data) {
    if (permit.consume(this)) push_frame(data);
  }
  virtual void post_telemetry(const TelemetrySample&) noexcept {}
  virtual DebugStats stats() const noexcept { return {}; }
  virtual void stop() {}
};

class DebugBus {
public:
  // Fixed reservation storage; sink membership is immutable while producers run.
  class FrameTicket {
  public:
    FrameTicket() = default;
    FrameTicket(const FrameTicket&) = delete;
    FrameTicket& operator=(const FrameTicket&) = delete;
    FrameTicket(FrameTicket&& other) noexcept
      : owner_(std::exchange(other.owner_, nullptr)),
        entries_(std::move(other.entries_)), count_(std::exchange(other.count_, 0)) {}
    FrameTicket& operator=(FrameTicket&& other) noexcept {
      owner_ = std::exchange(other.owner_, nullptr);
      entries_ = std::move(other.entries_);
      count_ = std::exchange(other.count_, 0);
      return *this;
    }
    explicit operator bool() const noexcept { return owner_ != nullptr && count_ > 0; }
  private:
    struct Entry { IDebugSink* sink = nullptr; FramePermit permit; };
    const DebugBus* owner_ = nullptr;
    std::array<Entry, 16> entries_{};
    size_t count_ = 0;
    friend class DebugBus;
  };

  static DebugBus& instance();
  void load_config(const YAML::Node& config);
  void shutdown();
  FrameTicket try_begin_frame() noexcept;
  void post(FrameTicket&& ticket, const FrameDebugData& data);
  void post(const FrameDebugData& data);
  void post_telemetry(const TelemetrySample& sample) noexcept;
  DebugStats stats() const noexcept;
  void add_sink(std::unique_ptr<IDebugSink> sink);
  bool has_sinks() const { return !sinks_.empty(); }

private:
  DebugBus() = default;
  std::vector<std::unique_ptr<IDebugSink>> sinks_;
  DebugStats final_stats_;
};

}  // namespace debug
