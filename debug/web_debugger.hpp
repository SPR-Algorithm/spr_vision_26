#pragma once
// web_debugger.hpp — 轻量 HTTP + WebSocket 调试服务器
// 依赖：POSIX socket（Linux/Jetson 原生支持），OpenCV，nlohmann_json
// 用法：
//   WebDebugger dbg(8080);
//   dbg.start();
//   // 每帧推送
//   dbg.push(frame, detections, reprojections, proc_ms, capture_time);

#include <atomic>
#include <utility>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <opencv2/opencv.hpp>

#include "param_tuner.hpp"
#include "telemetry.hpp"

namespace debug
{

class IDebugSink;

// Move-only one-use result of the producer-side client/rate check.
class FramePermit {
public:
  FramePermit() = default;
  FramePermit(const FramePermit&) = delete;
  FramePermit& operator=(const FramePermit&) = delete;
  FramePermit(FramePermit&& other) noexcept : owner_(std::exchange(other.owner_, nullptr)) {}
  FramePermit& operator=(FramePermit&& other) noexcept {
    owner_ = std::exchange(other.owner_, nullptr);
    return *this;
  }
  explicit operator bool() const noexcept { return owner_ != nullptr; }
private:
  explicit FramePermit(const void* owner) noexcept : owner_(owner) {}
  bool consume(const void* owner) noexcept {
    return std::exchange(owner_, nullptr) == owner;
  }
  const void* owner_ = nullptr;
  friend class WebDebugger;
  friend class IDebugSink;
};

struct DetectionData {
  std::vector<cv::Point2f> pts;   // 4 个关键点（图像坐标）
  int color;                       // 0=blue 1=red
  int number;                      // 0=guard 1-5 6=outpost 7-8=base
  float conf;
};

struct ReprojectionData {
  std::vector<cv::Point2f> pts;   // 3D 点投影到图像的坐标
};

// 推流设置，运行时可由网页的 set_stream 命令修改，对所有浏览器同时生效
struct StreamConfig {
  int max_fps = 30;        // 1 ~ 120
  int jpeg_quality = 60;   // 10 ~ 95
  int max_width = 640;     // 160 ~ 4096，宽度超过时等比缩小
};

// 所有 socket 读写和 JPEG 编码都在一个服务线程里完成，调用 push 的线程不会被网络阻塞
class WebDebugger
{
public:
  explicit WebDebugger(
    int port = 8080, const std::string & bind_addr = "0.0.0.0", const StreamConfig & stream = {});
  ~WebDebugger();

  // 启动后台服务线程（非阻塞）；端口被占用等失败情况返回 false 并打印错误
  bool start();
  void stop();

  // One visual producer only. Check before constructing debug-only geometry/metadata.
  // A permit reserves the rate slot and is consumed even if submission drops under contention.
  FramePermit try_begin_frame() noexcept;
  void push(
    FramePermit&& permit,
    const cv::Mat& frame,
    const std::vector<DetectionData>& detections,
    const std::vector<ReprojectionData>& reprojections,
    double proc_ms,
    std::chrono::steady_clock::time_point capture_time);

  // Compatibility path: preflight and submit once. Source pixels are owned on return.
  void push(
    const cv::Mat & frame,
    const std::vector<DetectionData> & detections,
    const std::vector<ReprojectionData> & reprojections,
    double proc_ms,
    std::chrono::steady_clock::time_point capture_time);

  // One independent planning producer; no allocations, locks, syscalls, or socket writes.
  void post_telemetry(const TelemetrySample& sample) noexcept;
  DebugStats stats() const noexcept;

  bool is_running() const { return running_; }
  int port() const { return port_; }

private:
  struct Conn {
    int fd = -1;
    bool is_ws = false;
    bool is_telemetry = false;
    bool close_after_flush = false;
    bool dead = false;
    std::string rx;
    std::string tx;
    size_t tx_off = 0;
    std::string fragment;  // 分片消息的已收部分
    uint64_t dropped = 0;
  };

  int port_;
  std::string bind_addr_;
  std::atomic<bool> running_{false};
  std::thread server_thread_;
  int listen_fd_ = -1;
  int wake_fd_ = -1;
  std::string index_path_;

  struct FrameSlot {
    cv::Mat img;          // Owned original image; resized only by the service thread.
    double scale_x = 1.0, scale_y = 1.0;
    int output_width = 0, output_height = 0;
    std::vector<DetectionData> detections;
    std::vector<ReprojectionData> reprojections;
    double proc_ms = 0.0;
    std::chrono::steady_clock::time_point capture_time;
  };

  std::atomic<int> max_fps_{30};
  std::atomic<int> jpeg_quality_{60};
  std::atomic<int> max_width_{640};

  // 主线程 -> 服务线程的单槽信箱，新帧覆盖旧帧
  std::mutex slot_mutex_;
  FrameSlot slot_;
  bool slot_full_ = false;
  cv::Mat producer_spare_;  // Visual producer only: an overwritten pending frame.
  cv::Mat recycled_raw_;    // Under slot_mutex_: worker has finished every pixel read.
  // Together with the pending slot and in-flight worker this retains at most three raw
  // buffers; shape changes may replace storage. Copying never shares pixels with a reader.
  std::chrono::steady_clock::time_point next_accept_;

  // 以下成员只在服务线程中访问
  std::vector<Conn> conns_;
  uint64_t next_seq_ = 0;
  std::atomic<int> ws_count_{0};
  std::atomic<int> telemetry_ws_count_{0};
  SpscQueue<StampedTelemetry, 256> telemetry_queue_;
  uint64_t next_telemetry_seq_ = 0;  // Planning producer only.
  std::chrono::steady_clock::time_point next_telemetry_flush_;

  struct Counters {
    std::atomic<uint64_t> frames_accepted{0}, frames_overwritten{0}, frames_busy{0}, frames_sent{0};
    std::atomic<uint64_t> video_client_dropped{0};
    std::atomic<uint64_t> telemetry_accepted{0}, telemetry_full_dropped{0};
    std::atomic<uint64_t> telemetry_expired_dropped{0}, telemetry_client_dropped{0}, telemetry_batches{0};
    std::atomic<uint64_t> frame_copy_count{0}, frame_copy_total_ns{0}, frame_copy_max_ns{0};
    std::atomic<uint64_t> raw_frame_allocations{0}, raw_frame_reuses{0};
    std::atomic<uint64_t> encode_count{0}, encode_total_ns{0}, encode_max_ns{0};
  } counters_;

  void wake();
  void serve();
  void accept_conns();
  bool read_conn(Conn & c);
  bool process_http(Conn & c);
  bool process_ws(Conn & c);
  void handle_message(Conn & c, const std::string & msg);
  void publish();
  void recycle_raw(cv::Mat raw);  // Worker only; moves old storage out before freeing it.
  void publish_telemetry();
  void send_raw(Conn & c, const std::string & bytes);
  void broadcast(const std::string & bytes);
  bool flush(Conn & c);
  bool ready_for_frame(const Conn & c) const;
  std::string index_response() const;

  void apply_stream_config(const StreamConfig & cfg);
  std::string stream_message() const;

  static std::string find_index_html();
  static std::string ekf_message();
  static std::string frame_meta(const FrameSlot & f, size_t jpeg_bytes, uint64_t dropped, uint64_t seq);
};

}  // namespace debug
