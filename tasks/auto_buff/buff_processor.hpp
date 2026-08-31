#ifndef AUTO_BUFF__BUFF_PROCESSOR_HPP
#define AUTO_BUFF__BUFF_PROCESSOR_HPP

#include <Eigen/Geometry>
#include <array>
#include <chrono>
#include <optional>
#include <string>

#include <opencv2/core.hpp>

#include "io/gimbal/gimbal.hpp"

namespace auto_buff
{
enum class BuffActivation
{
  INACTIVE,
  SMALL_ACTIVATED,
  BIG_ACTIVATED,
  INVALID
};

struct BuffInput
{
  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;
  std::array<cv::Point2f, 5> points{};
  BuffActivation activation = BuffActivation::INVALID;
  float confidence = 0.0F;
  Eigen::Quaterniond imu_q = Eigen::Quaterniond::Identity();
  io::GimbalState gimbal_state{};
  io::GimbalMode gimbal_mode = io::GimbalMode::IDLE;
};

struct BuffObservation
{
  std::array<cv::Point2f, 5> points{};
  BuffActivation activation = BuffActivation::INVALID;
  float confidence = 0.0F;
};

BuffInput make_buff_input(
  const cv::Mat & img, std::chrono::steady_clock::time_point timestamp,
  const Eigen::Quaterniond & imu_q, io::GimbalState gimbal_state, io::GimbalMode gimbal_mode,
  const std::optional<BuffObservation> & observation);

class BuffProcessor
{
public:
  BuffProcessor(const std::string & camera_config, const std::string & auto_buff_config);

  io::VisionToGimbal process(const BuffInput & input);
  void reset();

private:
  std::chrono::milliseconds tracking_timeout_{500};

  std::optional<io::GimbalMode> last_mode_;
  std::optional<std::chrono::steady_clock::time_point> last_timestamp_;

  static bool is_valid(const BuffInput & input);
  static bool has_compatible_activation(const BuffInput & input);
  static bool has_valid_points(const BuffInput & input);
};
}  // namespace auto_buff

#endif  // AUTO_BUFF__BUFF_PROCESSOR_HPP
