#include "buff_processor.hpp"

#include <cmath>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace auto_buff
{
namespace
{
bool is_finite(float value)
{
  return std::isfinite(value);
}

float cross(const cv::Point2f & a, const cv::Point2f & b, const cv::Point2f & c)
{
  const auto ab = b - a;
  const auto bc = c - b;
  return ab.x * bc.y - ab.y * bc.x;
}
}  // namespace

BuffInput make_buff_input(
  const cv::Mat & img, std::chrono::steady_clock::time_point timestamp,
  const Eigen::Quaterniond & imu_q, io::GimbalState gimbal_state, io::GimbalMode gimbal_mode,
  const std::optional<BuffObservation> & observation)
{
  BuffInput input;
  input.img = img;
  input.timestamp = timestamp;
  input.imu_q = imu_q;
  input.gimbal_state = gimbal_state;
  input.gimbal_mode = gimbal_mode;
  if (observation.has_value()) {
    input.points = observation->points;
    input.activation = observation->activation;
    input.confidence = observation->confidence;
  }
  return input;
}

BuffProcessor::BuffProcessor(const std::string & camera_config, const std::string & auto_buff_config)
{
  (void)camera_config;
  const auto config = YAML::LoadFile(auto_buff_config);
  const auto timeout_ms = config["tracking_timeout_ms"].as<int>();
  if (timeout_ms <= 0) throw std::runtime_error("tracking_timeout_ms must be positive");
  tracking_timeout_ = std::chrono::milliseconds(timeout_ms);
}

io::VisionToGimbal BuffProcessor::process(const BuffInput & input)
{
  if (last_mode_.has_value() && last_mode_.value() != input.gimbal_mode) {
    reset();
    return {};
  }
  if (last_timestamp_.has_value() && input.timestamp < last_timestamp_.value()) {
    reset();
    return {};
  }
  if (last_timestamp_.has_value() && input.timestamp - last_timestamp_.value() > tracking_timeout_) {
    reset();
    return {};
  }

  io::VisionToGimbal output{};
  if (!is_valid(input)) {
    reset();
    return output;
  }

  last_mode_ = input.gimbal_mode;
  last_timestamp_ = input.timestamp;
  output.mode = 1;
  return output;
}

void BuffProcessor::reset()
{
  last_mode_.reset();
  last_timestamp_.reset();
}

bool BuffProcessor::is_valid(const BuffInput & input)
{
  if (input.img.empty() || !std::isfinite(input.confidence) || input.confidence < 0.0F ||
      input.confidence > 1.0F) {
    return false;
  }

  if (!input.imu_q.coeffs().allFinite() || input.imu_q.squaredNorm() <= 1e-12) return false;

  const auto & state = input.gimbal_state;
  if (!is_finite(state.yaw) || !is_finite(state.yaw_vel) || !is_finite(state.pitch) ||
      !is_finite(state.pitch_vel) || !is_finite(state.bullet_speed)) {
    return false;
  }

  return has_valid_points(input) && has_compatible_activation(input);
}

bool BuffProcessor::has_compatible_activation(const BuffInput & input)
{
  switch (input.gimbal_mode) {
    case io::GimbalMode::SMALL_BUFF:
      return input.activation == BuffActivation::INACTIVE ||
             input.activation == BuffActivation::SMALL_ACTIVATED;
    case io::GimbalMode::BIG_BUFF:
      return input.activation == BuffActivation::INACTIVE ||
             input.activation == BuffActivation::BIG_ACTIVATED;
    case io::GimbalMode::IDLE:
    case io::GimbalMode::AUTO_AIM:
      return false;
  }
  return false;
}

bool BuffProcessor::has_valid_points(const BuffInput & input)
{
  for (const auto & point : input.points) {
    if (!is_finite(point.x) || !is_finite(point.y) || point.x < 0.0F || point.y < 0.0F ||
        point.x >= input.img.cols || point.y >= input.img.rows) {
      return false;
    }
  }

  const auto & top = input.points[0];
  const auto & right = input.points[1];
  const auto & bottom = input.points[2];
  const auto & left = input.points[3];
  if (top.y >= right.y || top.y >= left.y || bottom.y <= right.y || bottom.y <= left.y ||
      right.x <= top.x || right.x <= bottom.x || left.x >= top.x || left.x >= bottom.x) {
    return false;
  }

  const std::array<float, 4> turns = {
    cross(top, right, bottom), cross(right, bottom, left), cross(bottom, left, top),
    cross(left, top, right)};
  const bool clockwise = turns[0] < -1e-3F;
  const bool counter_clockwise = turns[0] > 1e-3F;
  if (!clockwise && !counter_clockwise) return false;

  for (const auto turn : turns) {
    if ((clockwise && turn >= -1e-3F) || (counter_clockwise && turn <= 1e-3F)) return false;
  }

  const auto & r_mark = input.points[4];
  for (const auto & corner : {top, right, bottom, left}) {
    if (cv::norm(r_mark - corner) <= 1e-3F) return false;
  }
  if (std::abs(cross(top, right, r_mark)) <= 1e-3F ||
      std::abs(cross(right, bottom, r_mark)) <= 1e-3F ||
      std::abs(cross(bottom, left, r_mark)) <= 1e-3F ||
      std::abs(cross(left, top, r_mark)) <= 1e-3F) {
    return false;
  }
  return true;
}
}  // namespace auto_buff
