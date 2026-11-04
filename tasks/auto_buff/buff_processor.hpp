#ifndef AUTO_BUFF__BUFF_PROCESSOR_HPP
#define AUTO_BUFF__BUFF_PROCESSOR_HPP

#include <Eigen/Geometry>
#include <chrono>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "buff_decision_module.hpp"
#include "io/gimbal/gimbal.hpp"
#include "observation_refiner.hpp"
#include "phase_motion_estimator.hpp"
#include "rune_reconstructor.hpp"
#include "rune_types.hpp"

namespace auto_buff
{
struct BuffInput
{
  cv::Mat img;
  FrameTimestamp timestamp{};
  // All S2 detections for this frame, in detector quality order. Image storage must stay
  // unchanged during process(); observations own their keypoints.
  std::vector<BuffObservation> observations;
  Eigen::Quaterniond imu_q = Eigen::Quaterniond::Identity();
  io::GimbalState gimbal_state{};
  io::GimbalMode gimbal_mode = io::GimbalMode::IDLE;
};

BuffInput make_buff_input(
  const cv::Mat & img, FrameTimestamp timestamp,
  const Eigen::Quaterniond & imu_q, io::GimbalState gimbal_state, io::GimbalMode gimbal_mode,
  std::vector<BuffObservation> observations);

class BuffProcessor
{
public:
  BuffProcessor(const std::string & camera_config, const std::string & auto_buff_config);

  io::VisionToGimbal process(const BuffInput & input);
  void reset();

private:
  ObservationRefiner refiner_;
  RuneReconstructor reconstructor_;
  PhaseMotionEstimator motion_estimator_;
  BuffDecisionModule decision_module_;
  std::chrono::milliseconds tracking_timeout_{500};

  std::optional<io::GimbalMode> last_mode_;
  std::optional<FrameTimestamp> last_timestamp_;
  std::optional<FrameTimestamp> last_reconstruction_timestamp_;
  std::optional<RunePose> last_reconstruction_pose_;

  static bool is_valid(const BuffInput & input);
  static std::optional<RuneState> resolve_state(io::GimbalMode mode, BuffActivation activation);
  static io::VisionToGimbal map_output(const BuffDecision & decision);
};
}  // namespace auto_buff

#endif  // AUTO_BUFF__BUFF_PROCESSOR_HPP
