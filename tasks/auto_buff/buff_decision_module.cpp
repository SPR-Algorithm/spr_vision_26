#include "buff_decision_module.hpp"

#include <cmath>

#include "io/gimbal/gimbal.hpp"

namespace auto_buff
{
BuffDecision BuffDecisionModule::decide(
  const RuneTarget &, const io::GimbalState &, std::chrono::steady_clock::time_point)
{
  // TODO: Implement intercept prediction, ballistic solving and the firing decision state machine.
  // An unimplemented aim solution must not issue tracking angles or grant firing permission.
  return {};
}

BuffDecision BuffDecisionModule::track(const io::GimbalState & gimbal) const
{
  if (!std::isfinite(gimbal.yaw) || !std::isfinite(gimbal.pitch)) return {};

  BuffDecision decision;
  decision.mode = BuffControlMode::TRACKING;
  decision.yaw = gimbal.yaw;
  decision.pitch = gimbal.pitch;
  return decision;
}

void BuffDecisionModule::reset()
{
  // No decision history is retained until the S6 decision state machine is implemented.
}
}  // namespace auto_buff
