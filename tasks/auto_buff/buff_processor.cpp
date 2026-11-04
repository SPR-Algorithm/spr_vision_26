#include "buff_processor.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

#include <yaml-cpp/yaml.h>

namespace auto_buff
{
BuffInput make_buff_input(
  const cv::Mat & img, FrameTimestamp timestamp, const Eigen::Quaterniond & imu_q,
  io::GimbalState gimbal_state, io::GimbalMode gimbal_mode,
  std::vector<BuffObservation> observations)
{
  BuffInput input;
  input.img = img;
  input.timestamp = timestamp;
  input.observations = std::move(observations);
  input.imu_q = imu_q;
  input.gimbal_state = gimbal_state;
  input.gimbal_mode = gimbal_mode;
  return input;
}

BuffProcessor::BuffProcessor(const std::string & camera_config, const std::string & auto_buff_config)
: refiner_(auto_buff_config), reconstructor_(camera_config)
{
  const auto config = YAML::LoadFile(auto_buff_config);
  const auto timeout_ms = config["tracking_timeout_ms"].as<int>();
  if (timeout_ms <= 0) throw std::runtime_error("tracking_timeout_ms must be positive");
  tracking_timeout_ = std::chrono::milliseconds(timeout_ms);
}

io::VisionToGimbal BuffProcessor::process(const BuffInput & input)
{
  if (
    (last_mode_ && *last_mode_ != input.gimbal_mode) ||
    (last_timestamp_ && input.timestamp < *last_timestamp_) ||
    (last_timestamp_ && input.timestamp - *last_timestamp_ > tracking_timeout_) ||
    (last_reconstruction_timestamp_ &&
     input.timestamp - *last_reconstruction_timestamp_ > tracking_timeout_)) {
    reset();
    return {};
  }
  if (!is_valid(input)) {
    reset();
    return {};
  }
  last_mode_ = input.gimbal_mode;
  last_timestamp_ = input.timestamp;

  // S2 is the detector adapter at the caller. Preserve every candidate for joint S4 geometry;
  // observation validation and contour extraction belong exclusively to S3.
  std::vector<RefinedObservation> blades;
  blades.reserve(input.observations.size());
  for (const auto & observation : input.observations) {
    const auto state = resolve_state(input.gimbal_mode, observation.activation);
    if (!state) continue;
    auto refined = refiner_.refine(input.img, observation, *state);
    if (refined.valid()) blades.push_back(std::move(refined));
  }
  if (blades.empty()) return {};
  // S4 carries slot associations and armor centers along with the common rune pose.
  const auto reconstruction = reconstructor_.reconstruct(
    blades, input.imu_q.normalized(), last_reconstruction_pose_);
  if (!reconstruction) return {};
  last_reconstruction_timestamp_ = input.timestamp;
  last_reconstruction_pose_ = reconstruction->pose;

  const ReconstructedBlade * target_blade = nullptr;
  for (const auto & blade : reconstruction->blades) {
    if (
      !blade.observation.valid() || !blade.observation.state ||
      blade.observation.state->activation != ActivationState::INACTIVE) {
      continue;
    }
    if (
      target_blade == nullptr || blade.observation.detection_confidence >
                                   target_blade->observation.detection_confidence) {
      target_blade = &blade;
    }
  }
  if (target_blade == nullptr) return {};

  // S5 absence means geometry exists but motion has not converged: attitude hold only.
  const auto target =
    motion_estimator_.update(reconstruction->pose, *target_blade, input.timestamp);
  if (!target) return map_output(decision_module_.track(input.gimbal_state));

  // S6 owns interception, ballistics and firing timing. Apply the common quality gate before
  // the single conversion to the protocol; a PnP fallback can never authorize firing.
  auto decision = decision_module_.decide(
    *target, input.gimbal_state, std::chrono::steady_clock::now());
  if (
    decision.mode == BuffControlMode::FIRE &&
    (target->quality != TrackingQuality::GOOD ||
     target->pose.source != PoseSource::CHAMFER_REFINED || !target->motion.converged ||
     target->motion.model == MotionModel::UNAVAILABLE)) {
    decision.mode = BuffControlMode::TRACKING;
  }
  return map_output(decision);
}

void BuffProcessor::reset()
{
  motion_estimator_.reset();
  decision_module_.reset();
  last_mode_.reset();
  last_timestamp_.reset();
  last_reconstruction_timestamp_.reset();
  last_reconstruction_pose_.reset();
}

bool BuffProcessor::is_valid(const BuffInput & input)
{
  if (
    input.img.empty() ||
    (input.gimbal_mode != io::GimbalMode::SMALL_BUFF &&
     input.gimbal_mode != io::GimbalMode::BIG_BUFF)) {
    return false;
  }
  const auto quaternion_norm_squared = input.imu_q.squaredNorm();
  if (
    !input.imu_q.coeffs().allFinite() || !std::isfinite(quaternion_norm_squared) ||
    quaternion_norm_squared <= 1e-12) {
    return false;
  }

  const auto & state = input.gimbal_state;
  return std::isfinite(state.yaw) && std::isfinite(state.yaw_vel) && std::isfinite(state.pitch) &&
         std::isfinite(state.pitch_vel) && std::isfinite(state.bullet_speed);
}

std::optional<RuneState> BuffProcessor::resolve_state(
  io::GimbalMode mode, BuffActivation activation)
{
  if (mode != io::GimbalMode::SMALL_BUFF && mode != io::GimbalMode::BIG_BUFF) return std::nullopt;

  const auto kind = mode == io::GimbalMode::SMALL_BUFF ? RuneKind::SMALL : RuneKind::BIG;
  if (activation == BuffActivation::INACTIVE) return RuneState{kind, ActivationState::INACTIVE};
  if (
    (kind == RuneKind::SMALL && activation == BuffActivation::SMALL_ACTIVATED) ||
    (kind == RuneKind::BIG && activation == BuffActivation::BIG_ACTIVATED)) {
    return RuneState{kind, ActivationState::ACTIVATED};
  }
  return std::nullopt;
}

io::VisionToGimbal BuffProcessor::map_output(const BuffDecision & decision)
{
  if (decision.mode != BuffControlMode::TRACKING && decision.mode != BuffControlMode::FIRE) return {};

  const std::array<double, 6> fields = {
    decision.yaw, decision.yaw_vel, decision.yaw_acc,
    decision.pitch, decision.pitch_vel, decision.pitch_acc};
  if (!std::all_of(fields.begin(), fields.end(), [](double value) {
        return std::isfinite(value) && std::abs(value) <= std::numeric_limits<float>::max();
      })) {
    return {};
  }

  io::VisionToGimbal output{};
  output.mode = static_cast<std::uint8_t>(decision.mode);
  output.yaw = static_cast<float>(decision.yaw);
  output.yaw_vel = static_cast<float>(decision.yaw_vel);
  output.yaw_acc = static_cast<float>(decision.yaw_acc);
  output.pitch = static_cast<float>(decision.pitch);
  output.pitch_vel = static_cast<float>(decision.pitch_vel);
  output.pitch_acc = static_cast<float>(decision.pitch_acc);
  return output;
}
}  // namespace auto_buff
