#include "web_sink.hpp"

#include "../param_tuner.hpp"
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

WebSink::WebSink(int port, const std::string& bind) : web_(port, bind) {
  web_.start();
  web_.push_param_sets();
  tools::logger()->info("Web debugger: http://{}:{}", bind, port);
}

void WebSink::push_frame(const FrameDebugData& data) {
  std::vector<DetectionData> dets;
  if (data.armors) dets = to_detections(*data.armors);
  web_.push(data.frame, dets, data.reprojections, data.latency_ms);

  const auto ekf_state = ParamTuner::instance().latest_ekf_state();
  if (ekf_state.timestamp > 0) {
    web_.push_ekf_state(ekf_state);
  }
}

}  // namespace debug
