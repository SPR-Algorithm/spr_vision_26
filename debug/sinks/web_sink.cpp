#include "web_sink.hpp"

#include "tasks/auto_aim/armor.hpp"
#include "tools/logger.hpp"

namespace debug {

static int to_color(auto_aim::Color c) {
  if (c == auto_aim::Color::blue) return 0;
  if (c == auto_aim::Color::red) return 1;
  if (c == auto_aim::Color::purple) return 3;
  return 2;
}

static int to_number(auto_aim::ArmorName n) {
  switch (n) {
    case auto_aim::ArmorName::sentry: return 0;
    case auto_aim::ArmorName::one: return 1;
    case auto_aim::ArmorName::two: return 2;
    case auto_aim::ArmorName::three: return 3;
    case auto_aim::ArmorName::four: return 4;
    case auto_aim::ArmorName::five: return 5;
    case auto_aim::ArmorName::outpost: return 6;
    case auto_aim::ArmorName::base: return 7;
    default: return 8;
  }
}

std::vector<DetectionData> WebSink::to_detections(
    const std::list<auto_aim::Armor>& armors) {
  std::vector<DetectionData> dets;
  dets.reserve(armors.size());
  for (const auto& a : armors) {
    dets.push_back({a.points, to_color(a.color), to_number(a.name),
                    static_cast<float>(a.confidence)});
  }
  return dets;
}

WebSink::WebSink(int port, const std::string& bind, const StreamConfig& stream)
    : web_(port, bind, stream) {
  if (web_.start()) tools::logger()->info("Web debugger: http://{}:{}", bind, port);
}

void WebSink::push_frame(const FrameDebugData& data) {
  push_frame(try_begin_frame(), data);
}

void WebSink::push_frame(FramePermit&& permit, const FrameDebugData& data) {
  if (!permit) return;
  std::vector<DetectionData> dets;
  if (data.armors) dets = to_detections(*data.armors);
  web_.push(std::move(permit), data.frame, dets, data.reprojections, data.proc_ms, data.capture_time);
}

}  // namespace debug
