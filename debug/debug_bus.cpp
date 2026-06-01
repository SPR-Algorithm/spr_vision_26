#include "debug_bus.hpp"
#include "sinks/web_sink.hpp"
#include "tools/logger.hpp"

namespace debug {

DebugBus& DebugBus::instance() {
  static DebugBus bus;
  return bus;
}

void DebugBus::load_config(const YAML::Node& config) {
  auto dbg = config["debug_bus"];
  if (!dbg) return;

  auto web = dbg["sinks"] ? dbg["sinks"]["web"] : YAML::Node();
  if (web && web["enabled"] && web["enabled"].as<bool>()) {
    int port = web["port"] ? web["port"].as<int>() : 8080;
    std::string bind = web["bind"] ? web["bind"].as<std::string>() : "0.0.0.0";
    sinks_.push_back(std::make_unique<WebSink>(port, bind));
  }
}

void DebugBus::shutdown() { sinks_.clear(); }

void DebugBus::post(const FrameDebugData& data) {
  for (auto& sink : sinks_) sink->push_frame(data);
}

void DebugBus::add_sink(std::unique_ptr<IDebugSink> sink) {
  sinks_.push_back(std::move(sink));
}

}  // namespace debug
