#pragma once
#include "../debug_bus.hpp"
#include "../web_debugger.hpp"

namespace debug {

class WebSink : public IDebugSink {
public:
  WebSink(int port, const std::string& bind);
  void push_frame(const FrameDebugData& data) override;

private:
  WebDebugger web_;
  static std::vector<DetectionData> to_detections(const std::list<auto_aim::Armor>& armors);
};

}  // namespace debug
