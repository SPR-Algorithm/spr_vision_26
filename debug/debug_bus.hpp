#pragma once
// debug_bus.hpp — 调试事件总线
// 用法:
//   DebugBus::load_config(config);
//   DebugBus::post({frame, &armors, reprojs, latency_ms});

#include <list>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

namespace auto_aim {
struct Armor;
}

namespace debug {

struct ReprojectionData;
struct DetectionData;

struct FrameDebugData {
  cv::Mat frame;
  const std::list<auto_aim::Armor>* armors = nullptr;
  std::vector<ReprojectionData> reprojections;
  double latency_ms = 0.0;
};

class IDebugSink {
public:
  virtual ~IDebugSink() = default;
  virtual void push_frame(const FrameDebugData& data) = 0;
};

class DebugBus {
public:
  static DebugBus& instance();
  void load_config(const YAML::Node& config);
  void shutdown();
  void post(const FrameDebugData& data);
  void add_sink(std::unique_ptr<IDebugSink> sink);
  bool has_sinks() const { return !sinks_.empty(); }

private:
  DebugBus() = default;
  std::vector<std::unique_ptr<IDebugSink>> sinks_;
};

}  // namespace debug
