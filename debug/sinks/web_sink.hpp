#pragma once
#include "../debug_bus.hpp"
#include "../web_debugger.hpp"

namespace debug {

class WebSink : public IDebugSink {
public:
  WebSink(int port, const std::string& bind, const StreamConfig& stream = {});
  FramePermit try_begin_frame() noexcept override { return web_.try_begin_frame(); }
  void push_frame(FramePermit&& permit, const FrameDebugData& data) override;
  void push_frame(const FrameDebugData& data) override;
  void post_telemetry(const TelemetrySample& sample) noexcept override { web_.post_telemetry(sample); }
  DebugStats stats() const noexcept override { return web_.stats(); }
  void stop() override { web_.stop(); }

private:
  WebDebugger web_;
  static std::vector<DetectionData> to_detections(const std::list<auto_aim::Armor>& armors);
};

}  // namespace debug
