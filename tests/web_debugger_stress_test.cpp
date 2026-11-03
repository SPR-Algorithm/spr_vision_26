// 无硬件回归：预采样、图像快照、独立遥测、有界队列及慢客户端。
// 使用真实 loopback WebSocket；队列满载与过期边界使用确定性的单元测试。
#include <arpa/inet.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "debug/web_debugger.hpp"
#include "debug/telemetry.hpp"
#include "debug/debug_bus.hpp"
#include "tools/logger.hpp"

namespace {
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

double milliseconds(Clock::duration d)
{
  return std::chrono::duration<double, std::milli>(d).count();
}

void require(bool condition, const std::string & message)
{
  if (!condition) throw std::runtime_error(message);
}

struct Frame {
  int opcode;
  std::string payload;
};

class Client {
public:
  explicit Client(int port, const std::string & path = "/ws")
  {
    fd_ = socket(AF_INET, SOCK_STREAM, 0);
    require(fd_ >= 0, "socket failed");
    int rcvbuf = 64 * 1024;
    setsockopt(fd_, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(fd_, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0) {
      close(fd_);
      fd_ = -1;
      throw std::runtime_error("connect failed");
    }
    send_all("GET " + path + " HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\n"
             "Connection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
             "Sec-WebSocket-Version: 13\r\n\r\n");
    const auto deadline = Clock::now() + 2s;
    while (buffer_.find("\r\n\r\n") == std::string::npos)
      require(read_more(deadline), "WebSocket handshake timed out");
    const auto end = buffer_.find("\r\n\r\n") + 4;
    require(buffer_.substr(0, end).find(" 101 ") != std::string::npos, "WebSocket upgrade failed");
    buffer_.erase(0, end);
  }
  ~Client() { disconnect(); }
  Client(const Client &) = delete;
  Client & operator=(const Client &) = delete;
  int fd() const { return fd_; }

  void disconnect()
  {
    if (fd_ < 0) return;
    shutdown(fd_, SHUT_RDWR);
    close(fd_);
    fd_ = -1;
  }

  void send_text(const std::string & payload)
  {
    require(payload.size() < 126, "test payload too long");
    std::string frame{static_cast<char>(0x81), static_cast<char>(0x80 | payload.size())};
    // Zero mask is valid and leaves the payload unchanged.
    frame.append(4, '\0');
    frame += payload;
    send_all(frame);
  }

  std::optional<Frame> receive(Clock::time_point deadline)
  {
    while (Clock::now() < deadline) {
      while (buffer_.size() >= 2) {
        const auto * p = reinterpret_cast<const unsigned char *>(buffer_.data());
        uint64_t length = p[1] & 0x7f;
        size_t header = 2;
        if (length == 126) {
          if (buffer_.size() < 4) break;
          length = (p[2] << 8) | p[3];
          header = 4;
        } else if (length == 127) {
          if (buffer_.size() < 10) break;
          length = 0;
          for (int i = 2; i < 10; ++i) length = (length << 8) | p[i];
          header = 10;
        }
        require(length <= 10 * 1024 * 1024, "unexpectedly large WebSocket frame");
        if (buffer_.size() < header + length) break;
        const int opcode = p[0] & 0x0f;
        const auto payload = buffer_.substr(header, length);
        buffer_.erase(0, header + length);
        return Frame{opcode, payload};
      }
      if (!read_more(deadline)) return std::nullopt;
    }
    return std::nullopt;
  }

  std::optional<nlohmann::json> wait_for_json(const std::string & type,
    std::chrono::milliseconds timeout = 2s)
  {
    const auto deadline = Clock::now() + timeout;
    while (auto frame = receive(deadline)) {
      if (frame->opcode != 1) continue;
      auto json = nlohmann::json::parse(frame->payload, nullptr, false);
      if (json.is_object() && json.value("type", std::string{}) == type) return json;
    }
    return std::nullopt;
  }

  bool wait_for_type(const std::string & type)
  {
    return wait_for_json(type).has_value();
  }

  std::optional<Frame> wait_for_binary()
  {
    const auto deadline = Clock::now() + 2s;
    while (auto frame = receive(deadline)) {
      if (frame->opcode == 2) return frame;
      if (frame->opcode == 1) {
        const auto json = nlohmann::json::parse(frame->payload, nullptr, false);
        require(!json.is_object() || json.value("type", std::string{}) != "telemetry",
          "telemetry leaked onto the video endpoint");
      }
    }
    return std::nullopt;
  }

  void require_no_opcode(int opcode, std::chrono::milliseconds duration)
  {
    const auto deadline = Clock::now() + duration;
    while (auto frame = receive(deadline)) {
      require(frame->opcode != opcode, "unexpected WebSocket opcode on endpoint");
    }
  }

  void require_no_type(const std::string & type, std::chrono::milliseconds duration)
  {
    const auto deadline = Clock::now() + duration;
    while (auto frame = receive(deadline)) {
      if (frame->opcode != 1) continue;
      const auto json = nlohmann::json::parse(frame->payload, nullptr, false);
      require(!json.is_object() || json.value("type", std::string{}) != type,
        "message leaked onto the wrong endpoint: " + type);
    }
  }

private:
  int fd_ = -1;
  std::string buffer_;

  void send_all(const std::string & bytes)
  {
    size_t offset = 0;
    while (offset < bytes.size()) {
      const auto n = send(fd_, bytes.data() + offset, bytes.size() - offset, MSG_NOSIGNAL);
      if (n < 0 && errno == EINTR) continue;
      require(n > 0, "send failed");
      offset += n;
    }
  }
  bool read_more(Clock::time_point deadline)
  {
    while (Clock::now() < deadline) {
      pollfd pfd{fd_, POLLIN, 0};
      const int timeout = std::max(1, static_cast<int>(milliseconds(deadline - Clock::now())));
      int result = poll(&pfd, 1, timeout);
      if (result < 0 && errno == EINTR) continue;
      if (result <= 0) return false;
      char data[65536];
      const auto n = recv(fd_, data, sizeof(data), 0);
      if (n <= 0) return false;
      buffer_.append(data, n);
      return true;
    }
    return false;
  }
};

// Bind an ephemeral port first so parallel invocations rarely conflict.
int available_port()
{
  const int fd = socket(AF_INET, SOCK_STREAM, 0);
  require(fd >= 0, "port probe socket failed");
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  require(bind(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0, "port probe bind failed");
  socklen_t length = sizeof(address);
  require(getsockname(fd, reinterpret_cast<sockaddr *>(&address), &length) == 0, "getsockname failed");
  close(fd);
  return ntohs(address.sin_port);
}

template <typename Predicate>
void wait_until(Predicate ready, const std::string & message)
{
  const auto deadline = Clock::now() + 2s;
  while (!ready()) {
    require(Clock::now() < deadline, message);
    std::this_thread::sleep_for(1ms);
  }
}

void test_bus_preflight()
{
  struct Record {
    bool accept = false;
    int preflight = 0, frames = 0, telemetry = 0, stopped = 0;
    double yaw = 0;
  } record;
  class Sink : public debug::IDebugSink {
  public:
    explicit Sink(Record & record) : record_(record) {}
    debug::FramePermit try_begin_frame() noexcept override
    {
      ++record_.preflight;
      return record_.accept ? IDebugSink::try_begin_frame() : debug::FramePermit{};
    }
    void push_frame(const debug::FrameDebugData & data) override
    {
      require(data.proc_ms == 42, "bus changed the reserved snapshot");
      ++record_.frames;
    }
    void post_telemetry(const debug::TelemetrySample & sample) noexcept override
    {
      ++record_.telemetry;
      record_.yaw = sample.gimbal_yaw;
    }
    debug::DebugStats stats() const noexcept override
    {
      debug::DebugStats stats;
      stats.frames_accepted = record_.frames;
      return stats;
    }
    void stop() override { ++record_.stopped; }
  private:
    Record & record_;
  };
  auto & bus = debug::DebugBus::instance();
  bus.shutdown();
  require(!bus.try_begin_frame(), "empty bus requested debug construction");
  bus.add_sink(std::make_unique<Sink>(record));
  int constructions = 0;
  if (auto ticket = bus.try_begin_frame()) ++constructions;
  require(constructions == 0 && record.preflight == 1, "bus ignored a sink's preflight rejection");
  record.accept = true;
  auto ticket = bus.try_begin_frame();
  require(static_cast<bool>(ticket), "bus failed to reserve an interested sink");
  auto moved = std::move(ticket);
  require(!ticket && static_cast<bool>(moved), "bus ticket move did not transfer ownership");
  record.accept = false;  // Reservation must survive until submission without a second check.
  debug::FrameDebugData snapshot;
  snapshot.proc_ms = 42;
  bus.post(std::move(moved), snapshot);
  bus.post(std::move(moved), snapshot);
  require(!moved && record.preflight == 2 && record.frames == 1,
    "bus resampled or reused a one-use ticket");
  record.accept = true;
  bus.post(snapshot);
  require(record.preflight == 3 && record.frames == 2,
    "legacy bus post does not share the same single preflight path");
  record.accept = false;
  debug::TelemetrySample sample;
  sample.gimbal_yaw = -0.75;
  bus.post_telemetry(sample);
  require(record.telemetry == 1 && record.yaw == -0.75,
    "bus telemetry depended on a video reservation");
  bus.shutdown();
  bus.shutdown();
  require(record.stopped == 1 && bus.stats().frames_accepted == 2,
    "bus shutdown lost final statistics or stopped twice");
  std::cout << "DebugBus: construction gate, moved single-use ticket, legacy adapter and independent telemetry passed\n";
}

void test_bounded_queue()
{
  static_assert(std::is_trivially_copyable<debug::TelemetrySample>::value,
    "telemetry snapshot must remain fixed storage");
  static_assert(noexcept(std::declval<debug::WebDebugger &>().post_telemetry(
    std::declval<const debug::TelemetrySample &>())), "telemetry submission must not throw");
  debug::SpscQueue<debug::StampedTelemetry, 256> queue;
  debug::StampedTelemetry value;
  require(!queue.try_pop(value), "new telemetry queue is not empty");
  // Repeatedly wrap both indices, using all 256 slots and testing overflow recovery.
  for (uint64_t round = 0; round < 8; ++round) {
    for (uint64_t i = 0; i < 256; ++i) {
      value.seq = round * 256 + i;
      value.sample.gimbal_yaw = static_cast<double>(value.seq);
      require(queue.try_push(value), "queue rejected one of its 256 usable entries");
    }
    require(queue.size() == 256, "queue capacity differs from 256");
    require(!queue.try_push(value), "full telemetry queue accepted an extra entry");
    for (uint64_t i = 0; i < 256; ++i) {
      require(queue.try_pop(value), "queue lost a retained entry after overflow");
      require(value.seq == round * 256 + i && value.sample.gimbal_yaw == value.seq,
        "queue changed FIFO order or snapshot payload");
    }
    require(queue.size() == 0 && !queue.try_pop(value), "drained queue retained entries");
  }
  value.monotonic_ms = 1000;
  require(!debug::telemetry_is_expired(value, 1249.9), "fresh telemetry expired");
  require(!debug::telemetry_is_expired(value, 1250), "250 ms boundary expired early");
  require(debug::telemetry_is_expired(value, 1250.1), "stale telemetry was retained");

  // Exercise the actual release/acquire exchange across two CPU threads.
  constexpr uint64_t samples = 65536;
  std::atomic<bool> done{false}, failed{false};
  const auto deadline = Clock::now() + 5s;
  std::thread producer([&] {
    for (uint64_t i = 0; i < samples; ++i) {
      debug::StampedTelemetry sample;
      sample.seq = i;
      sample.sample.gimbal_pitch = static_cast<double>(i);
      while (!queue.try_push(sample)) {
        if (Clock::now() >= deadline) { failed = true; done = true; return; }
        std::this_thread::yield();
      }
    }
    done = true;
  });
  uint64_t received = 0;
  while ((!done || queue.size() != 0) && Clock::now() < deadline) {
    if (!queue.try_pop(value)) { std::this_thread::yield(); continue; }
    if (value.seq != received || value.sample.gimbal_pitch != received) failed = true;
    ++received;
  }
  producer.join();
  require(!failed && received == samples, "concurrent SPSC transfer lost or tore a snapshot");
  std::cout << "bounded queue: 256 slots, overflow recovery, expiration and concurrent FIFO passed\n";
}

struct DecodedVideo {
  nlohmann::json meta;
  cv::Mat image;
};

DecodedVideo decode_video(const Frame & frame)
{
  require(frame.opcode == 2 && frame.payload.size() >= 4, "invalid video packet");
  const auto * bytes = reinterpret_cast<const unsigned char *>(frame.payload.data());
  uint32_t length = 0;
  for (int i = 0; i < 4; ++i) length |= uint32_t(bytes[i]) << (8 * i);
  require(length <= frame.payload.size() - 4, "invalid little-endian metadata length");
  auto meta = nlohmann::json::parse(frame.payload.substr(4, length));
  std::vector<unsigned char> jpeg(frame.payload.begin() + 4 + length, frame.payload.end());
  auto image = cv::imdecode(jpeg, cv::IMREAD_COLOR);
  require(!image.empty(), "video JPEG could not be decoded");
  require(meta.at("jpeg_bytes") == jpeg.size(), "JPEG length metadata differs from payload");
  require(meta.at("w") == image.cols && meta.at("h") == image.rows,
    "video dimensions differ from metadata");
  return {std::move(meta), std::move(image)};
}

void test_video_snapshot()
{
  const int port = available_port();
  debug::WebDebugger debugger(port, "127.0.0.1", {1, 90, 640});
  require(debugger.start(), "snapshot debugger failed to start");
  require(!debugger.try_begin_frame(), "frame preflight accepted without video clients");
  cv::Mat image(721, 1281, CV_8UC3, cv::Scalar(20, 80, 200));
  debugger.push(image, {}, {}, 5, Clock::now());
  require(debugger.stats().frame_copy_count == 0, "legacy no-client push copied an image");
  Client client(port);
  require(client.wait_for_type("param_sets"), "legacy parameter greeting missing");
  auto permit = debugger.try_begin_frame();
  require(static_cast<bool>(permit), "video client did not enable preflight");
  require(!debugger.try_begin_frame(), "preflight ignored one-fps sampling interval");
  std::vector<debug::DetectionData> detections{
    {{{100, 101}, {1100, 601}}, 1, 4, 0.875f}};
  std::vector<debug::ReprojectionData> reprojections{{{{500, 501}}}};
  debugger.push(std::move(permit), image, detections, reprojections, 7.25, Clock::now());
  require(!permit, "consumed frame permit remains usable");
  // The producer is free to recycle all inputs immediately after submission.
  image.setTo(cv::Scalar(0, 0, 0));
  detections.front().pts.front() = {-999, -999};
  reprojections.clear();
  debugger.push(std::move(permit), image, {}, {}, 0, Clock::now());
  require(debugger.stats().frames_accepted == 1 && debugger.stats().frame_copy_count == 1,
    "permit was sampled twice or reused after consumption");
  const auto packet = client.wait_for_binary();
  require(packet.has_value(), "accepted permit was throttled again at submission");
  const auto video = decode_video(*packet);
  require(video.image.cols == 640, "large raw image was not resized by consumer");
  const auto mean = cv::mean(video.image);
  require(std::abs(mean[0] - 20) < 5 && std::abs(mean[1] - 80) < 5 &&
    std::abs(mean[2] - 200) < 5, "snapshot retained a recycled source buffer");
  const double sx = static_cast<double>(video.image.cols) / image.cols;
  const double sy = static_cast<double>(video.image.rows) / image.rows;
  require(std::abs(video.meta.at("detections")[0]["pts"][0][0].get<double>() - 100 * sx) < 1e-3 &&
    std::abs(video.meta.at("detections")[0]["pts"][0][1].get<double>() - 101 * sy) < 1e-3,
    "detection coordinates do not match rounded output dimensions");
  require(std::abs(video.meta.at("reprojections")[0]["pts"][0][1].get<double>() - 501 * sy) < 1e-3,
    "reprojection snapshot or scaling is incorrect");
  require(video.meta.at("proc_ms") == 7.25, "producer metadata changed");
  client.require_no_opcode(2, 100ms);
  client.disconnect();
  wait_until([&] { return debugger.stats().video_clients == 0; }, "video disconnect not observed");
  require(!debugger.try_begin_frame(), "disconnected video client kept capture enabled");
  debugger.stop();
  require(!debugger.is_running() && !debugger.try_begin_frame(), "stopped debugger accepts frames");
  std::cout << "video preflight: one-use sampling, copied raw frame and rounded coordinate scales passed\n";
}

void test_frame_pool()
{
  const int port = available_port();
  debug::WebDebugger debugger(port, "127.0.0.1", {120, 90, 640});
  require(debugger.start(), "pool debugger failed to start");
  Client client(port);
  require(client.wait_for_type("param_sets"), "pool test video client greeting missing");
  uint64_t expected_seq = 0;
  const auto submit = [&](cv::Size size, int index) {
    const cv::Scalar color(20 + index, 80, 200 - index);
    cv::Mat source(size, CV_8UC3, color);
    debug::FramePermit permit;
    wait_until([&] {
      permit = debugger.try_begin_frame();
      return static_cast<bool>(permit);
    }, "pool frame did not reach its sampling interval");
    std::vector<debug::DetectionData> detections{
      {{{100, 101}}, 1, 4, 0.875f}};
    const auto accepted = debugger.stats().frames_accepted;
    debugger.push(std::move(permit), source, detections, {}, index, Clock::now());
    require(debugger.stats().frames_accepted == accepted + 1, "sequential pool submission was dropped");
    source.setTo(cv::Scalar(0, 0, 0));
    detections[0].pts[0] = {-999, -999};
    const auto packet = client.wait_for_binary();
    require(packet.has_value(), "pooled frame was not delivered");
    auto decoded = decode_video(*packet);
    const int width = std::min(size.width, 640);
    const int height = cvRound(size.height * static_cast<double>(width) / size.width);
    require(decoded.image.cols == width && decoded.image.rows == height,
      "recycling raw storage corrupted output dimensions");
    require(decoded.meta.at("seq").get<uint64_t>() == expected_seq++, "pooled frame sequence changed");
    const auto mean = cv::mean(decoded.image);
    require(std::abs(mean[0] - color[0]) < 5 && std::abs(mean[1] - color[1]) < 5 &&
      std::abs(mean[2] - color[2]) < 5, "pooled pixels aliased recycled source or another frame");
    require(std::abs(decoded.meta.at("detections")[0]["pts"][0][0].get<double>() -
        100.0 * width / size.width) < 1e-3 &&
      std::abs(decoded.meta.at("detections")[0]["pts"][0][1].get<double>() -
        101.0 * height / size.height) < 1e-3, "pooled frame lost its metadata or coordinate scale");
  };
  constexpr int steady_frames = 24;
  for (int i = 0; i < steady_frames; ++i) submit({1280, 720}, i);
  const auto steady = debugger.stats();
  require(steady.frame_copy_count == steady_frames && steady.raw_frame_allocations <= 3 &&
    steady.raw_frame_reuses >= steady_frames - 3,
    "stable shape did not reuse bounded raw buffers across sequential frames");
  // Exercise the no-resize branch: the worker recycles f.img before frame_meta().
  submit({319, 239}, 24);
  // Recycled storage with a different shape must be replaced rather than reused blindly.
  const std::array<cv::Size, 6> shapes{{
    {1281, 721}, {320, 241}, {800, 601}, {320, 241}, {1281, 721}, {319, 239},
  }};
  for (size_t i = 0; i < shapes.size(); ++i) submit(shapes[i], 25 + i);
  require(debugger.stats().raw_frame_allocations > steady.raw_frame_allocations,
    "shape changes did not report raw storage replacement");
  debugger.stop();
  std::cout << "raw pool: bounded stable-shape reuse, source ownership, no-resize dimensions and shape changes passed\n";
}

enum class Validity { Always, Target, TargetPlan, Plan, Outpost };
struct TelemetryField {
  const char * key;
  double debug::TelemetrySample::* member;
  Validity validity;
};

const std::array<TelemetryField, 28> telemetry_fields{{
  {"t", &debug::TelemetrySample::t, Validity::Always},
  {"gimbal_yaw", &debug::TelemetrySample::gimbal_yaw, Validity::Always},
  {"gimbal_yaw_vel", &debug::TelemetrySample::gimbal_yaw_vel, Validity::Always},
  {"gimbal_pitch", &debug::TelemetrySample::gimbal_pitch, Validity::Always},
  {"gimbal_pitch_vel", &debug::TelemetrySample::gimbal_pitch_vel, Validity::Always},
  {"target_yaw", &debug::TelemetrySample::target_yaw, Validity::TargetPlan},
  {"target_pitch", &debug::TelemetrySample::target_pitch, Validity::TargetPlan},
  {"plan_yaw", &debug::TelemetrySample::plan_yaw, Validity::Plan},
  {"plan_yaw_vel", &debug::TelemetrySample::plan_yaw_vel, Validity::Plan},
  {"plan_yaw_acc", &debug::TelemetrySample::plan_yaw_acc, Validity::Plan},
  {"plan_pitch", &debug::TelemetrySample::plan_pitch, Validity::Plan},
  {"plan_pitch_vel", &debug::TelemetrySample::plan_pitch_vel, Validity::Plan},
  {"plan_pitch_acc", &debug::TelemetrySample::plan_pitch_acc, Validity::Plan},
  {"fire", &debug::TelemetrySample::fire, Validity::Always},
  {"fired", &debug::TelemetrySample::fired, Validity::Always},
  {"target_z", &debug::TelemetrySample::target_z, Validity::Target},
  {"target_vz", &debug::TelemetrySample::target_vz, Validity::Target},
  {"target_last_id", &debug::TelemetrySample::target_last_id, Validity::Outpost},
  {"target_update_count", &debug::TelemetrySample::target_update_count, Validity::Outpost},
  {"target_jumped", &debug::TelemetrySample::target_jumped, Validity::Outpost},
  {"target_z0", &debug::TelemetrySample::target_z0, Validity::Outpost},
  {"target_z1", &debug::TelemetrySample::target_z1, Validity::Outpost},
  {"target_z2", &debug::TelemetrySample::target_z2, Validity::Outpost},
  {"target_z0_var", &debug::TelemetrySample::target_z0_var, Validity::Outpost},
  {"target_z1_var", &debug::TelemetrySample::target_z1_var, Validity::Outpost},
  {"target_z2_var", &debug::TelemetrySample::target_z2_var, Validity::Outpost},
  {"reprojection_aim_id", &debug::TelemetrySample::reprojection_aim_id, Validity::Outpost},
  {"w", &debug::TelemetrySample::w, Validity::Target},
}};

template <typename Predicate>
nlohmann::json wait_for_sample(Client & client, Predicate wanted)
{
  const auto deadline = Clock::now() + 3s;
  while (auto frame = client.receive(deadline)) {
    require(frame->opcode == 1, "video or control opcode leaked onto telemetry endpoint");
    const auto json = nlohmann::json::parse(frame->payload);
    if (json.at("type") == "telemetry_hello") {
      require(json.at("capacity") == 256 && json.at("batch_ms") == 50,
        "telemetry greeting advertised unexpected bounds");
      continue;
    }
    require(json.at("type") == "telemetry", "video/control message leaked onto telemetry endpoint");
    require(json.at("samples").is_array() && json.at("samples").size() <= 256,
      "telemetry batch is not bounded to queue capacity");
    for (const auto & key : {"full", "expired", "client"})
      require(json.at("dropped").at(key).is_number_unsigned(), "missing telemetry drop counter");
    for (const auto & sample : json.at("samples")) if (wanted(sample)) return sample;
  }
  throw std::runtime_error("expected telemetry sample did not arrive without video");
}

void check_telemetry_fields(const nlohmann::json & actual, const debug::TelemetrySample & expected)
{
  require(actual.at("seq").is_number_unsigned(), "telemetry sequence is missing");
  require(actual.at("monotonic_ms").is_number() && actual.at("monotonic_ms").get<double>() > 0,
    "monotonic sample timestamp is missing");
  require(actual.at("target_valid") == expected.target_valid && actual.at("control") == expected.control &&
    actual.at("outpost_valid") == expected.outpost_valid, "telemetry validity flags changed");
  for (const auto & field : telemetry_fields) {
    const bool available = field.validity == Validity::Always ||
      (field.validity == Validity::Target && expected.target_valid) ||
      (field.validity == Validity::TargetPlan && expected.target_valid && expected.control) ||
      (field.validity == Validity::Plan && expected.control) ||
      (field.validity == Validity::Outpost && expected.target_valid && expected.outpost_valid);
    const bool valid = available && std::isfinite(expected.*(field.member));
    if (!valid) {
      require(actual.at(field.key).is_null(), std::string("invalid field was not null: ") + field.key);
    } else {
      require(actual.at(field.key).is_number() &&
        actual.at(field.key).get<double>() == expected.*(field.member),
        std::string("telemetry field/unit/sign changed: ") + field.key);
    }
  }
}

void test_telemetry()
{
  const int port = available_port();
  debug::WebDebugger debugger(port, "127.0.0.1", {120, 60, 640});
  require(debugger.start(), "telemetry debugger failed to start");
  debug::TelemetrySample sample;
  for (size_t i = 0; i < telemetry_fields.size(); ++i)
    sample.*(telemetry_fields[i].member) = static_cast<double>(i) + 0.125;
  debugger.post_telemetry(sample);
  require(debugger.stats().telemetry_accepted == 0, "no-client telemetry was enqueued");
  Client telemetry(port, "/ws/telemetry");
  wait_until([&] { return debugger.stats().telemetry_clients == 1; }, "telemetry connection not counted");
  require(debugger.stats().video_clients == 0 && !debugger.try_begin_frame(),
    "pure telemetry enabled video capture");
  cv::Mat image(720, 1280, CV_8UC3, cv::Scalar(5, 10, 15));
  debugger.push(image, {}, {}, 1, Clock::now());
  require(debugger.stats().frame_copy_count == 0, "telemetry-only client triggered image copy");

  uint64_t last_seq = 0;
  // Valid target -> target lost -> target recovered without control -> normal armor -> outpost.
  const std::array<std::array<bool, 3>, 6> validity{{
    {{true, true, true}}, {{false, false, false}}, {{true, false, false}},
    {{false, true, true}}, {{true, true, false}}, {{true, true, true}},
  }};
  for (size_t i = 0; i < validity.size(); ++i) {
    sample.t = 100 + i;
    sample.target_valid = validity[i][0];
    sample.control = validity[i][1];
    sample.outpost_valid = validity[i][2];
    debugger.post_telemetry(sample);
    const auto received = wait_for_sample(telemetry, [&](const auto & value) {
      return value.at("t") == sample.t;
    });
    check_telemetry_fields(received, sample);
    if (i != 0) require(received.at("seq").get<uint64_t>() == last_seq + 1,
      "normal telemetry samples acquired unexpected sequence gaps");
    last_seq = received.at("seq").get<uint64_t>();
  }
  sample.t = 150;
  const double yaw = sample.gimbal_yaw;
  sample.gimbal_yaw = std::numeric_limits<double>::quiet_NaN();
  debugger.post_telemetry(sample);
  check_telemetry_fields(wait_for_sample(telemetry, [](const auto & value) { return value.at("t") == 150; }), sample);
  sample.gimbal_yaw = yaw;

  Client video(port);
  require(video.wait_for_type("param_sets"), "video control greeting missing");
  sample.t = 200;
  debugger.post_telemetry(sample);
  debugger.push(image, {}, {}, 1, Clock::now());
  require(video.wait_for_binary().has_value(), "video did not arrive alongside telemetry");
  wait_for_sample(telemetry, [](const auto & value) { return value.at("t") == 200; });
  video.require_no_type("telemetry", 100ms);
  telemetry.require_no_opcode(2, 100ms);

  // Production remains allocation-free and bounded while its consumer drains at 20 Hz.
  // The exact capacity/order is verified above without depending on this scheduling race.
  for (int i = 0; i < 100000; ++i) {
    sample.t = 300 + i;
    debugger.post_telemetry(sample);
  }
  require(debugger.stats().telemetry_full_dropped > 0, "saturated telemetry did not report overflow");
  sample.t = 999999;
  const auto accepted_before_recovery = debugger.stats().telemetry_accepted;
  wait_until([&] {
    debugger.post_telemetry(sample);
    return debugger.stats().telemetry_accepted > accepted_before_recovery;
  }, "full telemetry queue failed to accept after draining");
  const auto recovered = wait_for_sample(telemetry, [](const auto & value) { return value.at("t") == 999999; });
  require(recovered.at("seq").get<uint64_t>() > last_seq + 1, "overflow did not leave visible sequence gaps");
  last_seq = recovered.at("seq").get<uint64_t>();
  check_telemetry_fields(recovered, sample);

  video.disconnect();
  wait_until([&] { return debugger.stats().video_clients == 0; }, "video disconnect not counted");
  const auto copies = debugger.stats().frame_copy_count;
  sample.t = 1000000;
  debugger.post_telemetry(sample);
  debugger.push(image, {}, {}, 1, Clock::now());
  wait_for_sample(telemetry, [](const auto & value) { return value.at("t") == 1000000; });
  require(debugger.stats().frame_copy_count == copies, "paused video still copied frames");

  telemetry.disconnect();
  wait_until([&] { return debugger.stats().telemetry_clients == 0; }, "telemetry disconnect not counted");
  const auto accepted = debugger.stats().telemetry_accepted;
  debugger.post_telemetry(sample);
  require(debugger.stats().telemetry_accepted == accepted, "disconnected telemetry still enqueued");
  Client reconnected(port, "/ws/telemetry");
  sample.t = 1000001;
  debugger.post_telemetry(sample);
  const auto resumed = wait_for_sample(reconnected, [](const auto & value) { return value.at("t") == 1000001; });
  require(resumed.at("seq").get<uint64_t>() > last_seq, "reconnect reused sequence numbers");
  const auto stopping = Clock::now();
  debugger.stop();
  require(!debugger.is_running() && milliseconds(Clock::now() - stopping) < 2000,
    "debugger failed to exit with a live telemetry connection");
  require(debugger.stats().video_clients == 0 && debugger.stats().telemetry_clients == 0,
    "stop retained endpoint client counts");
  std::cout << "telemetry: all 28 fields, validity/nulls, independent endpoints, overflow/gaps and reconnect passed\n";
}

struct Result { double fps; double max_push_ms; };
Result run_loop(debug::WebDebugger & debugger, const cv::Mat & image, const char * name)
{
  constexpr int frames = 300;
  constexpr auto period = 10ms;
  const auto start = Clock::now();
  std::vector<double> pushes;
  pushes.reserve(frames);
  for (int i = 0; i < frames; ++i) {
    std::this_thread::sleep_until(start + i * period);
    const auto before = Clock::now();
    debugger.push(image, {}, {}, 5.0, before);
    debug::TelemetrySample sample;
    sample.t = i * 0.01;
    sample.gimbal_yaw = i * 0.001;
    sample.target_valid = sample.control = true;
    debugger.post_telemetry(sample);
    pushes.push_back(milliseconds(Clock::now() - before));
  }
  std::this_thread::sleep_until(start + frames * period);
  const double fps = frames * 1000.0 / milliseconds(Clock::now() - start);
  std::sort(pushes.begin(), pushes.end());
  std::cout << name << ": " << fps << " fps, push P50/P95/P99/max "
    << pushes[frames / 2] << '/' << pushes[frames * 95 / 100] << '/'
    << pushes[frames * 99 / 100] << '/' << pushes.back() << " ms\n";
  return {fps, pushes.back()};
}

// RAII keeps failures in the producer from leaving a joinable thread behind.
class BackgroundReader {
public:
  enum class Mode { Video, Telemetry, LimitedBytes };
  BackgroundReader(Client & client, Mode mode) : client_(client), mode_(mode), thread_([this] { run(); }) {}
  ~BackgroundReader() { finish(); }
  void finish()
  {
    stop_ = true;
    if (thread_.joinable()) thread_.join();
  }
  uint64_t count() const { return count_.load(); }
  const std::string & error() const { return error_; }  // Read after finish().
private:
  Client & client_;
  Mode mode_;
  std::atomic<bool> stop_{false};
  std::atomic<uint64_t> count_{0};
  std::string error_;
  std::thread thread_;

  void run()
  {
    try {
      uint64_t last_seq = 0;
      bool have_seq = false;
      while (!stop_) {
        if (mode_ == Mode::LimitedBytes) {
          char data[20 * 1024];
          pollfd pfd{client_.fd(), POLLIN, 0};
          if (poll(&pfd, 1, 100) <= 0) continue;
          const auto n = recv(client_.fd(), data, sizeof(data), 0);
          if (n <= 0) break;
          count_ += n;
          std::this_thread::sleep_for(10ms);  // 20 KiB / 10 ms ≈ 2 MB/s.
          continue;
        }
        const auto frame = client_.receive(Clock::now() + 100ms);
        if (!frame) continue;
        if (mode_ == Mode::Video) {
          if (frame->opcode == 2) { ++count_; continue; }
          if (frame->opcode == 1) {
            const auto json = nlohmann::json::parse(frame->payload);
            require(json.at("type") != "telemetry", "telemetry leaked to normal video reader");
          }
        } else {
          require(frame->opcode == 1, "binary frame leaked to normal telemetry reader");
          const auto json = nlohmann::json::parse(frame->payload);
          if (json.at("type") == "telemetry_hello") continue;
          require(json.at("type") == "telemetry", "control message leaked to telemetry reader");
          for (const auto & sample : json.at("samples")) {
            const auto seq = sample.at("seq").get<uint64_t>();
            require(!have_seq || seq > last_seq, "normal telemetry reader received reordered samples");
            have_seq = true;
            last_seq = seq;
            ++count_;
          }
        }
      }
    } catch (const std::exception & e) { error_ = e.what(); }
  }
};
}  // namespace

int main()
{
  try {
    tools::logger()->set_level(spdlog::level::warn);
    debug::ParamTuner::instance().load_from_yaml("configs/standard4.yaml");
    test_bus_preflight();
    test_bounded_queue();
    test_video_snapshot();
    test_frame_pool();
    test_telemetry();
    const int port = available_port();
    debug::WebDebugger debugger(port, "127.0.0.1", {100, 95, 640});
    require(debugger.start(), "debugger failed to start");

    cv::Mat image(720, 1280, CV_8UC3);
    cv::RNG rng(12345);
    rng.fill(image, cv::RNG::UNIFORM, 0, 256);
    const auto baseline = run_loop(debugger, image, "no client");

    Client stalled(port);
    run_loop(debugger, image, "stalled client");

    Client slow(port);
    Client normal_video(port);
    Client stalled_telemetry(port, "/ws/telemetry");
    Client normal_telemetry(port, "/ws/telemetry");
    BackgroundReader reader(slow, BackgroundReader::Mode::LimitedBytes);
    BackgroundReader video_reader(normal_video, BackgroundReader::Mode::Video);
    BackgroundReader telemetry_reader(normal_telemetry, BackgroundReader::Mode::Telemetry);
    const auto limited = run_loop(debugger, image, "stalled + 2 MB/s + normal video/telemetry");
    reader.finish();
    video_reader.finish();
    telemetry_reader.finish();
    std::cout << "limited client received " << reader.count() << " bytes, normal readers "
      << video_reader.count() << " video frames / " << telemetry_reader.count() << " telemetry samples\n";
    require(reader.error().empty() && video_reader.error().empty() && telemetry_reader.error().empty(),
      "background reader failed: " + reader.error() + video_reader.error() + telemetry_reader.error());
    require(reader.count() > 0, "limited client received no data");
    require(video_reader.count() > 100 && telemetry_reader.count() > 200,
      "normal readers were starved by stalled clients");
    require(limited.fps >= 95.0 && limited.fps >= baseline.fps * 0.95,
      "2 MB/s client reduced main loop fps");
    require(debugger.stats().video_client_dropped > 0 && debugger.stats().telemetry_client_dropped > 0,
      "stalled endpoint backpressure was not observable");

    Client control(port);
    require(control.wait_for_type("param_sets"), "initial parameter snapshot missing");
    control.send_text(R"({"cmd":"update_param","index":0,"v1":null,"v2":0.5})");
    require(control.wait_for_type("error"), "null parameter did not receive error");
    control.send_text(R"({"cmd":"set_stream","max_fps":null})");
    require(control.wait_for_type("error"), "null stream configuration did not receive error");
    control.send_text(R"({"cmd":"update_param","index":0,"v1":12.5,"v2":0.6})");
    require(control.wait_for_type("param_sets"), "server did not accept valid update after error");
    require(debug::ParamTuner::instance().select(0).v1 == 12.5, "valid update was not applied");
    std::cout << "null rejected; subsequent valid update succeeded\n";
    debugger.stop();
    return 0;
  } catch (const std::exception & e) {
    std::cerr << "web_debugger_stress_test: " << e.what() << '\n';
    return 1;
  }
}
