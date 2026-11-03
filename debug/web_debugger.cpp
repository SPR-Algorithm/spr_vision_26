#include "web_debugger.hpp"

#include <arpa/inet.h>
#include <linux/sockios.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/eventfd.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <nlohmann/json.hpp>
#include <sstream>

#include "tools/logger.hpp"
#include "benchmark.hpp"

namespace debug
{
namespace
{

constexpr size_t kMaxHttpRequest = 16 * 1024;
constexpr size_t kMaxWsMessage = 1 << 20;
constexpr size_t kMaxPendingBytes = 8 << 20;
constexpr size_t kMaxConns = 32;
constexpr int kSendBufferBytes = 256 * 1024;
// 内核发送队列里尚未发出的字节超过该值，说明对端跟不上，本帧直接丢弃
constexpr int kKernelBacklogLimit = 16 * 1024;

void record_timing(std::atomic<uint64_t>& count, std::atomic<uint64_t>& total,
                   std::atomic<uint64_t>& maximum, std::chrono::steady_clock::duration duration)
{
  const uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
  count.fetch_add(1, std::memory_order_relaxed);
  total.fetch_add(ns, std::memory_order_relaxed);
  // Each distribution has a single writer; statistics readers use atomic snapshots.
  if (ns > maximum.load(std::memory_order_relaxed)) maximum.store(ns, std::memory_order_relaxed);
}

double steady_ms()
{
  return std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}

constexpr uint8_t kOpContinuation = 0x0;
constexpr uint8_t kOpText = 0x1;
constexpr uint8_t kOpBinary = 0x2;
constexpr uint8_t kOpClose = 0x8;
constexpr uint8_t kOpPing = 0x9;
constexpr uint8_t kOpPong = 0xA;

// ── SHA-1 (RFC 3174)，用于 WebSocket 握手，避免引入 OpenSSL ───────────────────
std::array<uint8_t, 20> sha1(const std::string & input)
{
  uint32_t h[5] = {0x67452301,0xEFCDAB89,0x98BADCFE,0x10325476,0xC3D2E1F0};
  std::vector<uint8_t> msg(input.begin(), input.end());
  uint64_t bit_len = msg.size() * 8;
  msg.push_back(0x80);
  while (msg.size() % 64 != 56) msg.push_back(0);
  for (int i = 7; i >= 0; --i) msg.push_back((bit_len >> (i * 8)) & 0xFF);

  for (size_t i = 0; i < msg.size(); i += 64) {
    uint32_t w[80];
    for (int j = 0; j < 16; ++j)
      w[j] = (msg[i+j*4]<<24)|(msg[i+j*4+1]<<16)|(msg[i+j*4+2]<<8)|msg[i+j*4+3];
    for (int j = 16; j < 80; ++j) {
      uint32_t v = w[j-3]^w[j-8]^w[j-14]^w[j-16];
      w[j] = (v<<1)|(v>>31);
    }
    uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4];
    for (int j = 0; j < 80; ++j) {
      uint32_t f,k;
      if      (j<20){f=(b&c)|((~b)&d);k=0x5A827999;}
      else if (j<40){f=b^c^d;         k=0x6ED9EBA1;}
      else if (j<60){f=(b&c)|(b&d)|(c&d);k=0x8F1BBCDC;}
      else          {f=b^c^d;         k=0xCA62C1D6;}
      uint32_t tmp=((a<<5)|(a>>27))+f+e+k+w[j];
      e=d;d=c;c=(b<<30)|(b>>2);b=a;a=tmp;
    }
    h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;
  }
  std::array<uint8_t,20> out;
  for (int i=0;i<5;++i) for (int j=3;j>=0;--j) out[i*4+(3-j)]=(h[i]>>(j*8))&0xFF;
  return out;
}

std::string base64_encode(const uint8_t * data, size_t len)
{
  static const char * t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve((len + 2) / 3 * 4);
  for (size_t i = 0; i < len; i += 3) {
    uint32_t v = (uint32_t)data[i] << 16;
    if (i+1 < len) v |= (uint32_t)data[i+1] << 8;
    if (i+2 < len) v |= data[i+2];
    out += t[(v>>18)&63]; out += t[(v>>12)&63];
    out += (i+1<len) ? t[(v>>6)&63] : '=';
    out += (i+2<len) ? t[v&63]      : '=';
  }
  return out;
}

// RFC 6455 服务端帧，服务端发送不需要 masking
std::string ws_frame(const std::string & payload, uint8_t opcode)
{
  std::string frame;
  frame.reserve(payload.size() + 10);
  frame += static_cast<char>(0x80 | opcode);
  const size_t len = payload.size();
  if (len <= 125) {
    frame += static_cast<char>(len);
  } else if (len <= 65535) {
    frame += static_cast<char>(126);
    frame += static_cast<char>((len >> 8) & 0xFF);
    frame += static_cast<char>(len & 0xFF);
  } else {
    frame += static_cast<char>(127);
    for (int i = 7; i >= 0; --i) frame += static_cast<char>((len >> (i * 8)) & 0xFF);
  }
  frame += payload;
  return frame;
}

std::string to_lower(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
  return s;
}

std::string trim(const std::string & s)
{
  const auto b = s.find_first_not_of(" \t\r");
  if (b == std::string::npos) return "";
  return s.substr(b, s.find_last_not_of(" \t\r") - b + 1);
}

std::string http_response(const std::string & status, const std::string & type, const std::string & body)
{
  return "HTTP/1.1 " + status + "\r\n"
         "Content-Type: " + type + "\r\n"
         "Cache-Control: no-cache\r\n"
         "Content-Length: " + std::to_string(body.size()) + "\r\n"
         "Connection: close\r\n\r\n" + body;
}

std::string ws_handshake_response(const std::string & key)
{
  static const std::string GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
  const auto digest = sha1(key + GUID);
  return "HTTP/1.1 101 Switching Protocols\r\n"
         "Upgrade: websocket\r\n"
         "Connection: Upgrade\r\n"
         "Sec-WebSocket-Accept: " + base64_encode(digest.data(), digest.size()) + "\r\n\r\n";
}

std::string param_sets_message()
{
  return "{\"type\":\"param_sets\",\"data\":" + ParamTuner::instance().param_sets_to_json() + "}";
}

}  // namespace

// ── 生命周期 ─────────────────────────────────────────────────────────────────
WebDebugger::WebDebugger(int port, const std::string & bind_addr, const StreamConfig & stream)
: port_(port), bind_addr_(bind_addr)
{
  apply_stream_config(stream);
}

void WebDebugger::apply_stream_config(const StreamConfig & cfg)
{
  max_fps_ = std::clamp(cfg.max_fps, 1, 120);
  jpeg_quality_ = std::clamp(cfg.jpeg_quality, 10, 95);
  max_width_ = std::clamp(cfg.max_width, 160, 4096);
}

std::string WebDebugger::stream_message() const
{
  return nlohmann::json{
    {"type", "stream"},
    {"max_fps", max_fps_.load()},
    {"jpeg_quality", jpeg_quality_.load()},
    {"max_width", max_width_.load()}}.dump();
}

WebDebugger::~WebDebugger() { stop(); }

bool WebDebugger::start()
{
  if (running_) return true;

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port_);
  if (inet_pton(AF_INET, bind_addr_.c_str(), &addr.sin_addr) != 1) {
    tools::logger()->error("[WebDebugger] Invalid bind address: {}", bind_addr_);
    return false;
  }

  listen_fd_ = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  if (listen_fd_ < 0) {
    tools::logger()->error("[WebDebugger] socket() failed: {}", std::strerror(errno));
    return false;
  }
  int opt = 1;
  setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
  if (bind(listen_fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0 ||
      listen(listen_fd_, 16) < 0) {
    tools::logger()->error(
      "[WebDebugger] Failed to listen on {}:{}: {}", bind_addr_, port_, std::strerror(errno));
    ::close(listen_fd_);
    listen_fd_ = -1;
    return false;
  }

  wake_fd_ = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  if (wake_fd_ < 0) {
    tools::logger()->error("[WebDebugger] eventfd() failed: {}", std::strerror(errno));
    ::close(listen_fd_);
    listen_fd_ = -1;
    return false;
  }

  index_path_ = find_index_html();
  if (index_path_.empty()) {
    tools::logger()->warn("[WebDebugger] index.html not found");
  } else {
    tools::logger()->info("[WebDebugger] Serving page {}", index_path_);
  }

  next_accept_ = {};
  next_telemetry_flush_ = std::chrono::steady_clock::now() + std::chrono::milliseconds(50);
  running_ = true;
  server_thread_ = std::thread(&WebDebugger::serve, this);
  return true;
}

void WebDebugger::stop()
{
  if (!running_.exchange(false)) return;
  wake();
  if (server_thread_.joinable()) server_thread_.join();
  ::close(listen_fd_);
  ::close(wake_fd_);
  listen_fd_ = wake_fd_ = -1;
}

void WebDebugger::wake()
{
  const uint64_t one = 1;
  [[maybe_unused]] auto r = ::write(wake_fd_, &one, sizeof(one));
}

// ── 主线程接口 ───────────────────────────────────────────────────────────────
FramePermit WebDebugger::try_begin_frame() noexcept
{
  if (!running_.load(std::memory_order_relaxed) || ws_count_.load(std::memory_order_relaxed) == 0)
    return {};
  const auto now = std::chrono::steady_clock::now();
  if (now < next_accept_) return {};
  const auto period = std::chrono::microseconds(1000000 / max_fps_.load(std::memory_order_relaxed));
  next_accept_ += period;
  if (next_accept_ <= now) next_accept_ = now + period;
  return FramePermit(this);
}

void WebDebugger::push(
  const cv::Mat& frame, const std::vector<DetectionData>& detections,
  const std::vector<ReprojectionData>& reprojections, double proc_ms,
  std::chrono::steady_clock::time_point capture_time)
{
  push(try_begin_frame(), frame, detections, reprojections, proc_ms, capture_time);
}

void WebDebugger::push(
  FramePermit&& permit, const cv::Mat& frame, const std::vector<DetectionData>& detections,
  const std::vector<ReprojectionData>& reprojections, double proc_ms,
  std::chrono::steady_clock::time_point capture_time)
{
  if (!permit.consume(this) || !running_.load(std::memory_order_relaxed) ||
      ws_count_.load(std::memory_order_relaxed) == 0 || frame.empty()) return;
  FrameSlot f;
  f.img = std::move(producer_spare_);
  if (f.img.empty()) {
    std::unique_lock<std::mutex> lock(slot_mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
      counters_.frames_busy.fetch_add(1, std::memory_order_relaxed);
      return;  // Never allocate a fourth buffer while a recyclable one may be locked.
    }
    f.img = std::move(recycled_raw_);
  }
  const bool reusable = !f.img.empty() && f.img.size() == frame.size() && f.img.type() == frame.type();
  if (reusable) counters_.raw_frame_reuses.fetch_add(1, std::memory_order_relaxed);
  else counters_.raw_frame_allocations.fetch_add(1, std::memory_order_relaxed);
  const auto copy_start = std::chrono::steady_clock::now();
  frame.copyTo(f.img);
  const auto copy_elapsed = std::chrono::steady_clock::now() - copy_start;
  record_timing(counters_.frame_copy_count, counters_.frame_copy_total_ns,
                counters_.frame_copy_max_ns, copy_elapsed);
  benchmark_record(BenchmarkMetric::FrameCopy, benchmark_ms(copy_elapsed));
  f.detections = detections;
  f.reprojections = reprojections;
  f.proc_ms = proc_ms;
  f.capture_time = capture_time;
  {
    std::unique_lock<std::mutex> lock(slot_mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
      counters_.frames_busy.fetch_add(1, std::memory_order_relaxed);
      producer_spare_ = std::move(f.img);
      return;
    }
    if (slot_full_) counters_.frames_overwritten.fetch_add(1, std::memory_order_relaxed);
    // The old pending frame moves into f; neither pixels nor metadata are freed in the lock.
    std::swap(slot_, f);
    slot_full_ = true;
  }
  producer_spare_ = std::move(f.img);
  counters_.frames_accepted.fetch_add(1, std::memory_order_relaxed);
  wake();
}

void WebDebugger::post_telemetry(const TelemetrySample& sample) noexcept
{
  if (!running_.load(std::memory_order_relaxed) ||
      telemetry_ws_count_.load(std::memory_order_relaxed) == 0) return;
  const StampedTelemetry stamped{sample, next_telemetry_seq_++, steady_ms()};
  if (telemetry_queue_.try_push(stamped))
    counters_.telemetry_accepted.fetch_add(1, std::memory_order_relaxed);
  else
    counters_.telemetry_full_dropped.fetch_add(1, std::memory_order_relaxed);
}

DebugStats WebDebugger::stats() const noexcept
{
  DebugStats result;
  result.video_clients = ws_count_.load(std::memory_order_relaxed);
  result.telemetry_clients = telemetry_ws_count_.load(std::memory_order_relaxed);
#define COPY_COUNTER(name) result.name = counters_.name.load(std::memory_order_relaxed)
  COPY_COUNTER(frames_accepted); COPY_COUNTER(frames_overwritten); COPY_COUNTER(frames_busy);
  COPY_COUNTER(frames_sent); COPY_COUNTER(video_client_dropped);
  COPY_COUNTER(telemetry_accepted); COPY_COUNTER(telemetry_full_dropped);
  COPY_COUNTER(telemetry_expired_dropped); COPY_COUNTER(telemetry_client_dropped);
  COPY_COUNTER(telemetry_batches); COPY_COUNTER(frame_copy_count); COPY_COUNTER(frame_copy_total_ns);
  COPY_COUNTER(frame_copy_max_ns); COPY_COUNTER(encode_count); COPY_COUNTER(encode_total_ns);
  COPY_COUNTER(encode_max_ns); COPY_COUNTER(raw_frame_allocations); COPY_COUNTER(raw_frame_reuses);
#undef COPY_COUNTER
  return result;
}

// ── 服务线程：事件循环 ───────────────────────────────────────────────────────
void WebDebugger::serve()
{
  std::vector<pollfd> fds;
  while (running_) {
    fds.clear();
    fds.push_back({wake_fd_, POLLIN, 0});
    fds.push_back({listen_fd_, POLLIN, 0});
    for (const auto & c : conns_) {
      short events = POLLIN;
      if (c.tx_off < c.tx.size()) events |= POLLOUT;
      fds.push_back({c.fd, events, 0});
    }
    const size_t n_conns = conns_.size();

    const auto until_telemetry_us = std::chrono::duration_cast<std::chrono::microseconds>(
      next_telemetry_flush_ - std::chrono::steady_clock::now()).count();
    const int timeout_ms = static_cast<int>(std::clamp<int64_t>((until_telemetry_us + 999) / 1000, 0, 50));
    if (poll(fds.data(), fds.size(), timeout_ms) < 0) {
      if (errno == EINTR) continue;
      tools::logger()->error("[WebDebugger] poll() failed: {}", std::strerror(errno));
      break;
    }

    if (fds[0].revents & POLLIN) {
      uint64_t v;
      [[maybe_unused]] auto r = ::read(wake_fd_, &v, sizeof(v));
      try {
        publish();
      } catch (const std::exception& e) {
        tools::logger()->warn("[WebDebugger] Dropping invalid debug frame: {}", e.what());
      }
    }

    for (size_t i = 0; i < n_conns; ++i) {
      auto & c = conns_[i];
      const short re = fds[i + 2].revents;
      if (c.dead || re == 0) continue;
      if (re & (POLLERR | POLLNVAL)) {
        c.dead = true;
        continue;
      }
      if ((re & (POLLIN | POLLHUP)) && !read_conn(c)) c.dead = true;
      if (!c.dead && (re & POLLOUT) && !flush(c)) c.dead = true;
    }

    if (fds[1].revents & POLLIN) accept_conns();

    const auto now = std::chrono::steady_clock::now();
    if (now >= next_telemetry_flush_) {
      try {
        publish_telemetry();
      } catch (const std::exception& e) {
        tools::logger()->warn("[WebDebugger] Dropping invalid telemetry batch: {}", e.what());
      }
      next_telemetry_flush_ = now + std::chrono::milliseconds(50);
    }

    for (auto & c : conns_) {
      if (!c.dead) continue;
      ::close(c.fd);
      if (c.is_ws) {
        if (c.is_telemetry) --telemetry_ws_count_;
        else --ws_count_;
      }
    }
    conns_.erase(
      std::remove_if(conns_.begin(), conns_.end(), [](const Conn & c) { return c.dead; }),
      conns_.end());
  }

  for (auto & c : conns_) ::close(c.fd);
  conns_.clear();
  ws_count_ = 0;
  telemetry_ws_count_ = 0;
}

void WebDebugger::accept_conns()
{
  while (true) {
    const int fd = accept4(listen_fd_, nullptr, nullptr, SOCK_NONBLOCK | SOCK_CLOEXEC);
    if (fd < 0) {
      if (errno == EINTR) continue;
      return;
    }
    if (conns_.size() >= kMaxConns) {
      ::close(fd);
      continue;
    }
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    int sndbuf = kSendBufferBytes;
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));
    Conn c;
    c.fd = fd;
    conns_.push_back(std::move(c));
  }
}

bool WebDebugger::read_conn(Conn & c)
{
  char buf[16384];
  while (true) {
    const ssize_t n = recv(c.fd, buf, sizeof(buf), 0);
    if (n > 0) {
      c.rx.append(buf, n);
      if (c.rx.size() > kMaxWsMessage + kMaxHttpRequest) return false;
      continue;
    }
    if (n == 0) return false;
    if (errno == EINTR) continue;
    if (errno == EAGAIN || errno == EWOULDBLOCK) break;
    return false;
  }
  if (!c.is_ws && !process_http(c)) return false;
  return !c.is_ws || process_ws(c);
}

bool WebDebugger::process_http(Conn & c)
{
  if (c.close_after_flush) {
    c.rx.clear();
    return true;
  }
  const auto header_end = c.rx.find("\r\n\r\n");
  if (header_end == std::string::npos) return c.rx.size() <= kMaxHttpRequest;

  std::istringstream ss(c.rx.substr(0, header_end));
  c.rx.erase(0, header_end + 4);

  std::string line, method, path;
  std::getline(ss, line);
  std::istringstream(line) >> method >> path;
  std::map<std::string, std::string> headers;
  while (std::getline(ss, line)) {
    const auto colon = line.find(':');
    if (colon == std::string::npos) continue;
    headers[to_lower(trim(line.substr(0, colon)))] = trim(line.substr(colon + 1));
  }

  if (to_lower(headers["upgrade"]).find("websocket") != std::string::npos) {
    if (method != "GET" || (path != "/ws" && path != "/ws/telemetry")) {
      c.close_after_flush = true;
      send_raw(c, http_response("404 Not Found", "text/plain", "unknown websocket endpoint"));
      return true;
    }
    const auto key = headers.find("sec-websocket-key");
    if (key == headers.end() || key->second.empty()) {
      c.close_after_flush = true;
      send_raw(c, http_response("400 Bad Request", "text/plain", "missing Sec-WebSocket-Key"));
      return true;
    }
    c.is_ws = true;
    c.is_telemetry = path == "/ws/telemetry";
    std::string hello = ws_handshake_response(key->second);
    if (c.is_telemetry) {
      ++telemetry_ws_count_;
      hello += ws_frame("{\"type\":\"telemetry_hello\",\"capacity\":256,\"batch_ms\":50}", kOpText);
    } else {
      ++ws_count_;
      hello += ws_frame(param_sets_message(), kOpText) + ws_frame(stream_message(), kOpText);
      const std::string ekf = ekf_message();
      if (!ekf.empty()) hello += ws_frame(ekf, kOpText);
    }
    send_raw(c, hello);
    return true;
  }

  c.close_after_flush = true;
  if (method == "GET" && (path == "/" || path == "/index.html")) {
    send_raw(c, index_response());
  } else {
    send_raw(c, http_response("404 Not Found", "text/plain", "not found"));
  }
  return true;
}

bool WebDebugger::process_ws(Conn & c)
{
  size_t pos = 0;
  while (c.rx.size() - pos >= 2) {
    const auto * p = reinterpret_cast<const uint8_t *>(c.rx.data() + pos);
    const bool fin = p[0] & 0x80;
    const uint8_t opcode = p[0] & 0x0F;
    const bool masked = p[1] & 0x80;
    uint64_t len = p[1] & 0x7F;
    size_t header = 2;
    if (len == 126) {
      if (c.rx.size() - pos < 4) break;
      len = (uint64_t(p[2]) << 8) | p[3];
      header = 4;
    } else if (len == 127) {
      if (c.rx.size() - pos < 10) break;
      len = 0;
      for (int i = 0; i < 8; ++i) len = (len << 8) | p[2 + i];
      header = 10;
    }
    // 浏览器发来的帧必须带掩码
    if (!masked || len > kMaxWsMessage) return false;
    if (c.rx.size() - pos < header + 4 + len) break;

    const uint8_t * mask = p + header;
    std::string payload(len, '\0');
    for (size_t i = 0; i < len; ++i) payload[i] = p[header + 4 + i] ^ mask[i % 4];
    pos += header + 4 + len;

    switch (opcode) {
      case kOpText:
      case kOpBinary:
        if (fin) {
          handle_message(c, payload);
        } else {
          c.fragment = std::move(payload);
        }
        break;
      case kOpContinuation:
        c.fragment += payload;
        if (c.fragment.size() > kMaxWsMessage) return false;
        if (fin) {
          handle_message(c, c.fragment);
          c.fragment.clear();
        }
        break;
      case kOpClose:
        c.close_after_flush = true;
        send_raw(c, ws_frame("", kOpClose));
        c.rx.clear();
        return true;
      case kOpPing:
        send_raw(c, ws_frame(payload, kOpPong));
        break;
      case kOpPong:
        break;
      default:
        return false;
    }
    if (c.dead) return false;
  }
  c.rx.erase(0, pos);
  return true;
}

void WebDebugger::handle_message(Conn & c, const std::string & msg)
{
  try {
    if (c.is_telemetry) {
      send_raw(c, ws_frame("{\"type\":\"error\",\"msg\":\"telemetry endpoint is read only\"}", kOpText));
      return;
    }
    std::string err;
    const auto j = nlohmann::json::parse(msg, nullptr, false);
    const auto cmd = j.is_object() ? j.find("cmd") : j.end();
    const bool is_set_stream = cmd != j.end() && cmd->is_string() && *cmd == "set_stream";

    if (is_set_stream) {
      StreamConfig cfg{max_fps_, jpeg_quality_, max_width_};
      const std::pair<const char *, int *> fields[] = {
        {"max_fps", &cfg.max_fps}, {"jpeg_quality", &cfg.jpeg_quality}, {"max_width", &cfg.max_width}};
      for (const auto & [key, value] : fields) {
        const auto it = j.find(key);
        if (it == j.end()) continue;
        if (!it->is_number_integer() || *it < std::numeric_limits<int>::min() ||
            *it > std::numeric_limits<int>::max()) {
          err = std::string(key) + " must be an integer in range";
          break;
        }
        *value = it->get<int>();
      }
      if (err.empty()) {
        apply_stream_config(cfg);
        tools::logger()->info(
          "[WebDebugger] Stream set to {} fps, JPEG quality {}, max width {}", max_fps_.load(),
          jpeg_quality_.load(), max_width_.load());
        broadcast(ws_frame(stream_message(), kOpText));
        return;
      }
    } else if (ParamTuner::instance().update_from_json(msg, err)) {
      broadcast(ws_frame(param_sets_message(), kOpText));
      return;
    }

    tools::logger()->warn("[WebDebugger] Rejected message: {}", err);
    send_raw(c, ws_frame(nlohmann::json{{"type", "error"}, {"msg", err}}.dump(), kOpText));
  } catch (const std::exception & e) {
    tools::logger()->error("[WebDebugger] Failed to handle message: {}", e.what());
  }
}

void WebDebugger::recycle_raw(cv::Mat raw)
{
  {
    std::lock_guard<std::mutex> lock(slot_mutex_);
    std::swap(recycled_raw_, raw);
  }
  // Any displaced recycle buffer is destroyed outside the mailbox lock.
}

void WebDebugger::publish()
{
  FrameSlot f;
  {
    std::lock_guard<std::mutex> lk(slot_mutex_);
    if (!slot_full_) return;
    f = std::move(slot_);
    slot_full_ = false;
  }

  if (ws_count_.load(std::memory_order_relaxed) == 0) {
    recycle_raw(std::move(f.img));
    return;
  }
  const int source_width = f.img.cols, source_height = f.img.rows;
  const int max_width = max_width_.load(std::memory_order_relaxed);
  if (source_width > max_width) {
    const int height = std::max(1, cvRound(source_height * static_cast<double>(max_width) / source_width));
    cv::Mat resized;
    cv::resize(f.img, resized, cv::Size(max_width, height));
    // resize has completed the last raw-pixel read; JPEG uses separate resized storage.
    recycle_raw(std::move(f.img));
    f.img = std::move(resized);
    f.scale_x = static_cast<double>(f.img.cols) / source_width;
    f.scale_y = static_cast<double>(f.img.rows) / source_height;
  }
  f.output_width = f.img.cols;
  f.output_height = f.img.rows;
  std::vector<uint8_t> jpeg;
  const auto encode_start = std::chrono::steady_clock::now();
  cv::imencode(".jpg", f.img, jpeg, {cv::IMWRITE_JPEG_QUALITY, jpeg_quality_.load()});
  const auto encode_elapsed = std::chrono::steady_clock::now() - encode_start;
  record_timing(counters_.encode_count, counters_.encode_total_ns, counters_.encode_max_ns, encode_elapsed);
  benchmark_record(BenchmarkMetric::Encode, benchmark_ms(encode_elapsed));
  // Without resize the encoder was the last raw reader; metadata needs dimensions only.
  if (source_width <= max_width) recycle_raw(std::move(f.img));

  std::string ekf_frame;
  const std::string ekf = ekf_message();
  if (!ekf.empty()) {
    ekf_frame = ws_frame(ekf, kOpText);
  }

  // 二进制帧布局：uint32 小端元数据长度 | 元数据 JSON | JPEG
  const uint64_t seq = next_seq_++;
  for (auto & c : conns_) {
    if (!c.is_ws || c.is_telemetry || c.dead) continue;
    if (!ready_for_frame(c)) {
      ++c.dropped;
      counters_.video_client_dropped.fetch_add(1, std::memory_order_relaxed);
      continue;
    }
    const std::string meta = frame_meta(f, jpeg.size(), c.dropped, seq);
    const uint32_t n = meta.size();
    std::string payload;
    payload.reserve(4 + meta.size() + jpeg.size());
    for (int i = 0; i < 4; ++i) payload += static_cast<char>((n >> (8 * i)) & 0xFF);
    payload += meta;
    payload.append(jpeg.begin(), jpeg.end());
    send_raw(c, ws_frame(payload, kOpBinary) + ekf_frame);
    counters_.frames_sent.fetch_add(1, std::memory_order_relaxed);
  }
}

// Timer-driven telemetry is independent of video submission and rate limits.
void WebDebugger::publish_telemetry()
{
  const double now_ms = steady_ms();
  auto samples = nlohmann::json::array();
  StampedTelemetry value;
  for (size_t i = 0; i < 256 && telemetry_queue_.try_pop(value); ++i) {
    if (telemetry_is_expired(value, now_ms)) {
      counters_.telemetry_expired_dropped.fetch_add(1, std::memory_order_relaxed);
      continue;
    }
    if (telemetry_ws_count_.load(std::memory_order_relaxed) == 0) continue;
    const auto& s = value.sample;
    nlohmann::json j = {{"seq", value.seq}, {"monotonic_ms", value.monotonic_ms},
                       {"target_valid", s.target_valid}, {"control", s.control},
                       {"outpost_valid", s.outpost_valid}};
    const auto field = [&](const char* key, double number, bool valid = true) {
      j[key] = valid && std::isfinite(number) ? nlohmann::json(number) : nlohmann::json(nullptr);
    };
    field("t", s.t);
    field("gimbal_yaw", s.gimbal_yaw); field("gimbal_yaw_vel", s.gimbal_yaw_vel);
    field("gimbal_pitch", s.gimbal_pitch); field("gimbal_pitch_vel", s.gimbal_pitch_vel);
    field("target_yaw", s.target_yaw, s.target_valid && s.control);
    field("target_pitch", s.target_pitch, s.target_valid && s.control);
    field("plan_yaw", s.plan_yaw, s.control); field("plan_yaw_vel", s.plan_yaw_vel, s.control);
    field("plan_yaw_acc", s.plan_yaw_acc, s.control);
    field("plan_pitch", s.plan_pitch, s.control); field("plan_pitch_vel", s.plan_pitch_vel, s.control);
    field("plan_pitch_acc", s.plan_pitch_acc, s.control);
    field("fire", s.fire); field("fired", s.fired);
    field("target_z", s.target_z, s.target_valid); field("target_vz", s.target_vz, s.target_valid);
    field("w", s.w, s.target_valid);
    const bool outpost = s.target_valid && s.outpost_valid;
    field("target_last_id", s.target_last_id, outpost);
    field("target_update_count", s.target_update_count, outpost);
    field("target_jumped", s.target_jumped, outpost);
    field("target_z0", s.target_z0, outpost); field("target_z1", s.target_z1, outpost);
    field("target_z2", s.target_z2, outpost);
    field("target_z0_var", s.target_z0_var, outpost); field("target_z1_var", s.target_z1_var, outpost);
    field("target_z2_var", s.target_z2_var, outpost);
    field("reprojection_aim_id", s.reprojection_aim_id, outpost);
    samples.push_back(std::move(j));
  }
  if (samples.empty()) return;
  counters_.telemetry_batches.fetch_add(1, std::memory_order_relaxed);
  for (auto& c : conns_) {
    if (!c.is_ws || !c.is_telemetry || c.dead) continue;
    if (!ready_for_frame(c)) {
      c.dropped += samples.size();
      counters_.telemetry_client_dropped.fetch_add(samples.size(), std::memory_order_relaxed);
      continue;
    }
    const nlohmann::json batch = {
      {"type", "telemetry"}, {"samples", samples},
      {"dropped", {{"full", counters_.telemetry_full_dropped.load(std::memory_order_relaxed)},
                   {"expired", counters_.telemetry_expired_dropped.load(std::memory_order_relaxed)},
                   {"client", c.dropped}}}};
    send_raw(c, ws_frame(batch.dump(), kOpText));
  }
}

// ── 发送 ─────────────────────────────────────────────────────────────────────
void WebDebugger::send_raw(Conn & c, const std::string & bytes)
{
  if (c.dead) return;
  if (c.tx.size() - c.tx_off + bytes.size() > kMaxPendingBytes) {
    tools::logger()->warn("[WebDebugger] Client too slow, disconnecting");
    c.dead = true;
    return;
  }
  c.tx += bytes;
  if (!flush(c)) c.dead = true;
}

void WebDebugger::broadcast(const std::string & bytes)
{
  for (auto & c : conns_) {
    if (c.is_ws && !c.is_telemetry) send_raw(c, bytes);
  }
}

bool WebDebugger::flush(Conn & c)
{
  while (c.tx_off < c.tx.size()) {
    const ssize_t n = ::send(c.fd, c.tx.data() + c.tx_off, c.tx.size() - c.tx_off, MSG_NOSIGNAL);
    if (n > 0) {
      c.tx_off += n;
      continue;
    }
    if (n < 0 && errno == EINTR) continue;
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) break;
    return false;
  }
  if (c.tx_off == c.tx.size()) {
    c.tx.clear();
    c.tx_off = 0;
    return !c.close_after_flush;
  }
  if (c.tx_off > (1u << 20)) {
    c.tx.erase(0, c.tx_off);
    c.tx_off = 0;
  }
  return true;
}

bool WebDebugger::ready_for_frame(const Conn & c) const
{
  if (c.tx_off < c.tx.size()) return false;
  int unsent = 0;
  return ioctl(c.fd, SIOCOUTQNSD, &unsent) != 0 || unsent < kKernelBacklogLimit;
}

// ── 页面 ─────────────────────────────────────────────────────────────────────
std::string WebDebugger::find_index_html()
{
  std::vector<std::string> candidates;
  const std::string src = __FILE__;
  candidates.push_back(src.substr(0, src.rfind('/') + 1) + "static/index.html");

  char exe[4096];
  const ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
  if (n > 0) {
    const std::string exe_path(exe, n);
    candidates.push_back(exe_path.substr(0, exe_path.rfind('/') + 1) + "debug/static/index.html");
  }

  for (const auto & path : candidates) {
    if (std::ifstream(path).good()) return path;
  }
  return "";
}

std::string WebDebugger::index_response() const
{
  std::ifstream f(index_path_);
  if (index_path_.empty() || !f.good()) {
    return http_response("404 Not Found", "text/html; charset=utf-8",
                         "<h1>SPR_Vision_26 Debugger</h1><p>index.html not found</p>");
  }
  std::string body((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  return http_response("200 OK", "text/html; charset=utf-8", body);
}

// ── 帧序列化 ─────────────────────────────────────────────────────────────────
std::string WebDebugger::ekf_message()
{
  const auto s = ParamTuner::instance().latest_ekf_state();
  if (s.timestamp <= 0) return "";
  const double now = std::chrono::duration<double>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
  return "{\"type\":\"ekf_state\",\"age_ms\":" + std::to_string((now - s.timestamp) * 1000) +
         ",\"data\":" + ParamTuner::ekf_state_to_json(s) + "}";
}

std::string WebDebugger::frame_meta(
  const FrameSlot & f, size_t jpeg_bytes, uint64_t dropped, uint64_t seq)
{
  const auto points = [&](const std::vector<cv::Point2f> & pts) {
    auto arr = nlohmann::json::array();
    for (const auto & p : pts) arr.push_back({p.x * f.scale_x, p.y * f.scale_y});
    return arr;
  };

  nlohmann::json j;
  j["seq"] = seq;
  j["w"] = f.output_width;
  j["h"] = f.output_height;
  j["jpeg_bytes"] = jpeg_bytes;
  j["dropped"] = dropped;
  j["proc_ms"] = f.proc_ms;
  if (f.capture_time.time_since_epoch().count() > 0) {
    j["age_ms"] = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - f.capture_time).count();
  }
  j["detections"] = nlohmann::json::array();
  for (const auto & d : f.detections) {
    j["detections"].push_back(
      {{"pts", points(d.pts)}, {"color", d.color}, {"number", d.number}, {"conf", d.conf}});
  }
  j["reprojections"] = nlohmann::json::array();
  for (const auto & r : f.reprojections) j["reprojections"].push_back({{"pts", points(r.pts)}});
  return j.dump();
}

}  // namespace debug
