#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace debug {

// Fixed-size snapshot: names, signs and units match the planning thread's Plotter output.
// Angles are radians, position is metres, and t is seconds from planning-loop startup.
struct TelemetrySample {
  double t = 0.0;
  double gimbal_yaw = 0.0, gimbal_yaw_vel = 0.0;
  double gimbal_pitch = 0.0, gimbal_pitch_vel = 0.0;
  double target_yaw = 0.0, target_pitch = 0.0;
  double plan_yaw = 0.0, plan_yaw_vel = 0.0, plan_yaw_acc = 0.0;
  double plan_pitch = 0.0, plan_pitch_vel = 0.0, plan_pitch_acc = 0.0;
  double fire = 0.0, fired = 0.0;
  double target_z = 0.0, target_vz = 0.0;
  double target_last_id = 0.0, target_update_count = 0.0, target_jumped = 0.0;
  double target_z0 = 0.0, target_z1 = 0.0, target_z2 = 0.0;
  double target_z0_var = 0.0, target_z1_var = 0.0, target_z2_var = 0.0;
  double reprojection_aim_id = 0.0, w = 0.0;
  bool target_valid = false, control = false, outpost_valid = false;
};
static_assert(std::is_trivially_copyable<TelemetrySample>::value, "telemetry must stay allocation-free");

struct StampedTelemetry {
  TelemetrySample sample;
  uint64_t seq = 0;
  double monotonic_ms = 0.0;
};

inline bool telemetry_is_expired(const StampedTelemetry& value, double now_ms) noexcept {
  return now_ms - value.monotonic_ms > 250.0;
}

// Exactly Capacity usable entries. One producer and one consumer; neither waits or allocates.
template <typename T, size_t Capacity>
class SpscQueue {
  static_assert(Capacity > 0, "queue needs storage");
  static_assert(std::is_trivially_copyable<T>::value, "queue entries must be fixed snapshots");
public:
  bool try_push(const T& value) noexcept {
    const size_t head = head_.load(std::memory_order_relaxed);
    if (head - tail_.load(std::memory_order_acquire) >= Capacity) return false;
    storage_[head % Capacity] = value;
    head_.store(head + 1, std::memory_order_release);
    return true;
  }
  bool try_pop(T& value) noexcept {
    const size_t tail = tail_.load(std::memory_order_relaxed);
    if (tail == head_.load(std::memory_order_acquire)) return false;
    value = storage_[tail % Capacity];
    tail_.store(tail + 1, std::memory_order_release);
    return true;
  }
  // An observational estimate for statistics, never used to synchronize producer/consumer.
  size_t size() const noexcept {
    const size_t tail = tail_.load(std::memory_order_acquire);
    const size_t head = head_.load(std::memory_order_acquire);
    return head - tail;
  }
private:
  std::array<T, Capacity> storage_{};
  alignas(64) std::atomic<size_t> head_{0};
  alignas(64) std::atomic<size_t> tail_{0};
};

struct DebugStats {
  uint64_t video_clients = 0, telemetry_clients = 0;
  uint64_t frames_accepted = 0, frames_overwritten = 0, frames_busy = 0, frames_sent = 0;
  uint64_t video_client_dropped = 0;
  uint64_t telemetry_accepted = 0, telemetry_full_dropped = 0, telemetry_expired_dropped = 0;
  uint64_t telemetry_client_dropped = 0, telemetry_batches = 0;
  uint64_t frame_copy_count = 0, frame_copy_total_ns = 0, frame_copy_max_ns = 0;
  uint64_t raw_frame_allocations = 0, raw_frame_reuses = 0;
  uint64_t encode_count = 0, encode_total_ns = 0, encode_max_ns = 0;
};

}  // namespace debug
