#ifndef AUTO_BUFF__BUFF_DECISION_MODULE_HPP
#define AUTO_BUFF__BUFF_DECISION_MODULE_HPP

#include <chrono>

#include "rune_types.hpp"

namespace io
{
struct GimbalState;
}  // namespace io

namespace auto_buff
{
enum class BuffControlMode
{
  IDLE = 0,
  TRACKING = 1,
  FIRE = 2
};

// Domain decision only. BuffProcessor performs the final conversion to the gimbal wire format.
struct BuffDecision
{
  BuffControlMode mode = BuffControlMode::IDLE;
  double yaw = 0.0;
  double yaw_vel = 0.0;
  double yaw_acc = 0.0;
  double pitch = 0.0;
  double pitch_vel = 0.0;
  double pitch_acc = 0.0;
};

class BuffDecisionModule
{
public:
  BuffDecision decide(
    const RuneTarget & target, const io::GimbalState & gimbal,
    std::chrono::steady_clock::time_point now);

  // Hold the measured attitude without firing while a motion model is unavailable.
  BuffDecision track(const io::GimbalState & gimbal) const;

  void reset();
};
}  // namespace auto_buff

#endif  // AUTO_BUFF__BUFF_DECISION_MODULE_HPP
