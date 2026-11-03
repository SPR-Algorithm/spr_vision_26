#include "debug_bus.hpp"
#include <algorithm>
#include <stdexcept>
#include "sinks/web_sink.hpp"
#include "tools/logger.hpp"

namespace debug {

DebugBus& DebugBus::instance() {
  static DebugBus bus;
  return bus;
}

void DebugBus::load_config(const YAML::Node& config) {
  final_stats_ = {};
  auto dbg = config["debug_bus"];
  if (!dbg) return;

  auto web = dbg["sinks"] ? dbg["sinks"]["web"] : YAML::Node();
  if (web && web["enabled"] && web["enabled"].as<bool>()) {
    int port = web["port"] ? web["port"].as<int>() : 8080;
    std::string bind = web["bind"] ? web["bind"].as<std::string>() : "0.0.0.0";
    StreamConfig stream;
    if (web["max_fps"]) stream.max_fps = web["max_fps"].as<int>();
    if (web["jpeg_quality"]) stream.jpeg_quality = web["jpeg_quality"].as<int>();
    if (web["max_width"]) stream.max_width = web["max_width"].as<int>();
    add_sink(std::make_unique<WebSink>(port, bind, stream));
  }
}

void DebugBus::shutdown() {
  if (sinks_.empty()) return;
  for (auto& sink : sinks_) sink->stop();
  final_stats_ = stats();
  sinks_.clear();
}

DebugBus::FrameTicket DebugBus::try_begin_frame() noexcept {
  FrameTicket ticket;
  ticket.owner_ = this;
  for (auto& sink : sinks_) {
    auto permit = sink->try_begin_frame();
    if (!permit) continue;
    auto& entry = ticket.entries_[ticket.count_++];
    entry.sink = sink.get();
    entry.permit = std::move(permit);
  }
  return ticket;
}

void DebugBus::post(FrameTicket&& ticket, const FrameDebugData& data) {
  if (std::exchange(ticket.owner_, nullptr) != this) return;
  const size_t count = std::exchange(ticket.count_, 0);
  for (size_t i = 0; i < count; ++i) {
    auto& entry = ticket.entries_[i];
    entry.sink->push_frame(std::move(entry.permit), data);
  }
}

void DebugBus::post(const FrameDebugData& data) {
  post(try_begin_frame(), data);
}

void DebugBus::post_telemetry(const TelemetrySample& sample) noexcept {
  for (auto& sink : sinks_) sink->post_telemetry(sample);
}

DebugStats DebugBus::stats() const noexcept {
  if (sinks_.empty()) return final_stats_;
  DebugStats result;
  for (const auto& sink : sinks_) {
    const auto s = sink->stats();
#define ADD_COUNTER(name) result.name += s.name
    ADD_COUNTER(video_clients); ADD_COUNTER(telemetry_clients);
    ADD_COUNTER(frames_accepted); ADD_COUNTER(frames_overwritten); ADD_COUNTER(frames_busy);
    ADD_COUNTER(frames_sent); ADD_COUNTER(video_client_dropped);
    ADD_COUNTER(telemetry_accepted); ADD_COUNTER(telemetry_full_dropped);
    ADD_COUNTER(telemetry_expired_dropped); ADD_COUNTER(telemetry_client_dropped);
    ADD_COUNTER(telemetry_batches); ADD_COUNTER(frame_copy_count); ADD_COUNTER(frame_copy_total_ns);
    ADD_COUNTER(raw_frame_allocations); ADD_COUNTER(raw_frame_reuses);
    ADD_COUNTER(encode_count); ADD_COUNTER(encode_total_ns);
#undef ADD_COUNTER
    result.frame_copy_max_ns = std::max(result.frame_copy_max_ns, s.frame_copy_max_ns);
    result.encode_max_ns = std::max(result.encode_max_ns, s.encode_max_ns);
  }
  return result;
}

void DebugBus::add_sink(std::unique_ptr<IDebugSink> sink) {
  if (!sink) return;
  if (sinks_.empty()) final_stats_ = {};
  if (sinks_.size() >= 16) throw std::length_error("DebugBus supports at most 16 sinks");
  sinks_.push_back(std::move(sink));
}

}  // namespace debug
