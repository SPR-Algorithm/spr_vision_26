#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "tasks/auto_buff/buff_processor.hpp"

namespace
{
int failures = 0;

void check(bool condition, const char * expression, int line)
{
  if (!condition) {
    std::cerr << "CHECK failed: " << expression << " at " << line << '\n';
    ++failures;
  }
}

#define CHECK(condition) check((condition), #condition, __LINE__)

auto valid_input(io::GimbalMode mode, auto_buff::BuffActivation activation)
{
  auto_buff::BuffInput input;
  input.img = cv::Mat::zeros(480, 640, CV_8UC3);
  input.timestamp = std::chrono::steady_clock::now();
  auto_buff::BuffObservation observation;
  observation.points = {
    cv::Point2f{320.0F, 100.0F}, cv::Point2f{420.0F, 200.0F},
    cv::Point2f{320.0F, 300.0F}, cv::Point2f{220.0F, 200.0F},
    cv::Point2f{320.0F, 200.0F}};
  observation.activation = activation;
  observation.confidence = 0.95F;
  input.observations.push_back(observation);
  input.imu_q = Eigen::Quaterniond::Identity();
  input.gimbal_state = {0.0F, 0.0F, 0.0F, 0.0F, 24.0F, 0};
  input.gimbal_mode = mode;
  return input;
}

void assert_idle_frame(const io::VisionToGimbal & output)
{
  CHECK(output.head[0] == 'S');
  CHECK(output.head[1] == 'P');
  CHECK(output.tail == 0xef);
  CHECK(output.mode == 0);
  CHECK(output.yaw == 0.0F);
  CHECK(output.yaw_vel == 0.0F);
  CHECK(output.yaw_acc == 0.0F);
  CHECK(output.pitch == 0.0F);
  CHECK(output.pitch_vel == 0.0F);
  CHECK(output.pitch_acc == 0.0F);
}

void assert_same_frame(const io::VisionToGimbal & actual, const io::VisionToGimbal & expected)
{
  CHECK(actual.head[0] == expected.head[0]);
  CHECK(actual.head[1] == expected.head[1]);
  CHECK(actual.tail == expected.tail);
  CHECK(actual.mode == expected.mode);
  CHECK(actual.yaw == expected.yaw);
  CHECK(actual.yaw_vel == expected.yaw_vel);
  CHECK(actual.yaw_acc == expected.yaw_acc);
  CHECK(actual.pitch == expected.pitch);
  CHECK(actual.pitch_vel == expected.pitch_vel);
  CHECK(actual.pitch_acc == expected.pitch_acc);
}

auto reconstructed_input(double phase, auto_buff::FrameTimestamp timestamp)
{
  // Project two physical blades through the same vehicle calibration used by the processor.
  // The black image deliberately exercises the valid PnP fallback without contour refinement.
  const auto config = YAML::LoadFile("configs/sentry.yaml");
  auto intrinsics = config["camera_matrix"].as<std::vector<double>>();
  auto coefficients = config["distort_coeffs"].as<std::vector<double>>();
  const cv::Mat camera_matrix(3, 3, CV_64F, intrinsics.data());
  const cv::Mat distortion(1, static_cast<int>(coefficients.size()), CV_64F, coefficients.data());
  const cv::Matx33d front(0.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0);
  cv::Matx33d tilt;
  cv::Rodrigues(cv::Vec3d(0.12, -0.25, 0.18), tilt);
  cv::Matx33d turn;
  cv::Rodrigues(cv::Vec3d(phase, 0.0, 0.0), turn);
  cv::Vec3d rvec;
  cv::Rodrigues(tilt * front * turn, rvec);

  auto_buff::BuffInput input;
  input.img = cv::Mat::zeros(1080, 1440, CV_8UC3);
  input.timestamp = timestamp;
  input.gimbal_state = {0.25F, 0.0F, -0.10F, 0.0F, 24.0F, 0};
  input.gimbal_mode = io::GimbalMode::SMALL_BUFF;
  for (const int slot : {0, 2}) {
    const Eigen::AngleAxisd rotation(slot * 2.0 * CV_PI / 5.0, Eigen::Vector3d::UnitX());
    std::vector<cv::Point3f> model;
    for (const auto & base : std::vector<Eigen::Vector3d>{
           {0.0, 0.0, 0.827}, {0.0, 0.127, 0.7}, {0.0, 0.0, 0.573},
           {0.0, -0.127, 0.7}, {0.0, 0.0, 0.0}}) {
      const Eigen::Vector3d point = rotation * base;
      model.emplace_back(point.x(), point.y(), point.z());
    }
    std::vector<cv::Point2f> image;
    cv::projectPoints(model, rvec, cv::Vec3d(0.18, -0.12, 5.8), camera_matrix, distortion, image);
    auto_buff::BuffObservation observation;
    std::copy(image.begin(), image.end(), observation.points.begin());
    observation.activation = slot == 0 ? auto_buff::BuffActivation::INACTIVE :
                                        auto_buff::BuffActivation::SMALL_ACTIVATED;
    observation.confidence = slot == 0 ? 0.98F : 0.88F;
    input.observations.push_back(observation);
  }
  return input;
}

void assert_no_stage_result(const auto_buff::BuffDebugInfo & debug)
{
  CHECK(debug.refined_observations.empty());
  CHECK(!debug.reconstruction);
  CHECK(!debug.target_slot);
  CHECK(!debug.target);
  CHECK(std::string(debug.motion.status) == "EMPTY");
  CHECK(debug.motion.samples == 0);
  CHECK(!debug.motion.observed_phase_rad);
  CHECK(debug.decision.mode == auto_buff::BuffControlMode::IDLE);
  CHECK(debug.decision.yaw == 0.0);
  CHECK(debug.decision.pitch == 0.0);
}

void debug_snapshot_tests()
{
  auto_buff::BuffProcessor processor("configs/sentry.yaml", "configs/auto_buff.yaml");
  auto_buff::BuffProcessor without_debug("configs/sentry.yaml", "configs/auto_buff.yaml");
  auto_buff::BuffDebugInfo debug;
  const auto run = [&](const auto_buff::BuffInput & input) {
    const auto output = processor.process(input, &debug);
    assert_same_frame(output, without_debug.process(input));
    return output;
  };
  const auto t0 = std::chrono::steady_clock::time_point{} + std::chrono::seconds(10);
  bool converged = false;
  for (int frame = 0; frame < 31; ++frame) {
    const double elapsed_s = frame * 0.04;
    const auto input = reconstructed_input(
      elapsed_s * CV_PI / 3.0, t0 + std::chrono::milliseconds(40 * frame));
    const auto output = run(input);
    CHECK(debug.input_valid);
    CHECK(!debug.reset);
    CHECK(debug.refined_observations.size() == 2);
    CHECK(debug.reconstruction.has_value());
    CHECK(debug.target_slot.has_value());
    CHECK(debug.motion.samples == static_cast<std::size_t>(frame + 1));
    CHECK(std::abs(debug.motion.span_s - elapsed_s) < 1e-9);
    CHECK(debug.motion.observed_phase_rad.has_value());
    if (frame == 0) {
      CHECK(!debug.target);
      CHECK(debug.decision.mode == auto_buff::BuffControlMode::TRACKING);
      CHECK(output.mode == 1);
      CHECK(output.yaw == input.gimbal_state.yaw);
      CHECK(output.pitch == input.gimbal_state.pitch);
    }
    if (debug.target) {
      converged = true;
      CHECK(std::string(debug.motion.status) == "CONVERGED");
      CHECK(std::isfinite(debug.motion.fit_rms_rad));
      CHECK(debug.target->motion.converged);
      CHECK(debug.target->slot == *debug.target_slot);
      CHECK(output.mode == static_cast<unsigned int>(debug.decision.mode));
      CHECK(output.mode != 2);  // PnP-only geometry cannot authorize firing.
    }
  }
  CHECK(converged);

  // Reusing one debug object must clear every earlier stage result on an early return.
  auto missing = reconstructed_input(1.3, t0 + std::chrono::milliseconds(1240));
  missing.observations.clear();
  assert_idle_frame(run(missing));
  CHECK(debug.input_valid);
  CHECK(!debug.reset);
  assert_no_stage_result(debug);

  auto rejected = missing;
  rejected.timestamp += std::chrono::milliseconds(40);
  auto invalid_observation = valid_input(
    io::GimbalMode::SMALL_BUFF, auto_buff::BuffActivation::INACTIVE).observations.front();
  invalid_observation.points[0].y = -1.0F;
  rejected.observations.push_back(invalid_observation);
  assert_idle_frame(run(rejected));
  CHECK(debug.refined_observations.size() == 1);
  if (!debug.refined_observations.empty()) {
    CHECK(!debug.refined_observations.front().valid());
    CHECK(debug.refined_observations.front().reject_reason ==
          auto_buff::ObservationRejectReason::POINT_OUT_OF_BOUNDS);
  }
  CHECK(!debug.reconstruction);
  CHECK(!debug.target);
  CHECK(std::string(debug.motion.status) == "EMPTY");

  auto changed_mode = rejected;
  changed_mode.timestamp += std::chrono::milliseconds(40);
  changed_mode.gimbal_mode = io::GimbalMode::BIG_BUFF;
  assert_idle_frame(run(changed_mode));
  CHECK(debug.reset);
  CHECK(!debug.input_valid);
  assert_no_stage_result(debug);

  auto active_only = reconstructed_input(0.0, changed_mode.timestamp + std::chrono::milliseconds(40));
  active_only.observations.front().activation = auto_buff::BuffActivation::SMALL_ACTIVATED;
  assert_idle_frame(run(active_only));
  CHECK(debug.input_valid);
  CHECK(!debug.reset);
  CHECK(debug.reconstruction.has_value());
  CHECK(!debug.target_slot);
  CHECK(!debug.target);
  CHECK(std::string(debug.motion.status) == "EMPTY");

  auto mixed = active_only;
  mixed.timestamp += std::chrono::milliseconds(40);
  auto conflict = mixed.observations.front();
  conflict.activation = auto_buff::BuffActivation::BIG_ACTIVATED;
  mixed.observations.insert(mixed.observations.begin(), conflict);
  assert_idle_frame(run(mixed));
  CHECK(debug.refined_observations.size() == mixed.observations.size());
  CHECK(!debug.refined_observations.front().valid());
  CHECK(debug.refined_observations.front().reject_reason ==
        auto_buff::ObservationRejectReason::STATE_MODE_CONFLICT);
  CHECK(debug.refined_observations[1].valid());
  CHECK(debug.refined_observations.front().points == conflict.points);
  CHECK(debug.reconstruction.has_value());

  auto invalid = mixed;
  invalid.timestamp += std::chrono::milliseconds(40);
  invalid.img.release();
  assert_idle_frame(run(invalid));
  CHECK(debug.reset);
  CHECK(!debug.input_valid);
  assert_no_stage_result(debug);
}
}  // namespace

int main()
{
  static_assert(sizeof(io::VisionToGimbal) == 28);

  auto_buff::BuffProcessor processor("configs/sentry.yaml", "configs/auto_buff.yaml");
  const auto t0 = std::chrono::steady_clock::time_point{} + std::chrono::seconds(1);

  auto small_input = valid_input(io::GimbalMode::SMALL_BUFF, auto_buff::BuffActivation::SMALL_ACTIVATED);
  small_input.timestamp = t0;
  const auto small = processor.process(small_input);
  // A single observation cannot satisfy joint S4 reconstruction. Detections alone must
  // not produce a tracking command without reconstructed geometry.
  assert_idle_frame(small);
  CHECK(small.head[0] == 'S');
  CHECK(small.head[1] == 'P');
  CHECK(small.tail == 0xef);

  auto big_input = valid_input(io::GimbalMode::BIG_BUFF, auto_buff::BuffActivation::BIG_ACTIVATED);
  big_input.timestamp = t0 + std::chrono::milliseconds(1);
  assert_idle_frame(processor.process(big_input));
  big_input.timestamp += std::chrono::milliseconds(1);
  assert_idle_frame(processor.process(big_input));

  auto conflicting = valid_input(io::GimbalMode::SMALL_BUFF, auto_buff::BuffActivation::BIG_ACTIVATED);
  conflicting.timestamp = big_input.timestamp + std::chrono::milliseconds(1);
  assert_idle_frame(processor.process(conflicting));

  auto invalid = valid_input(io::GimbalMode::SMALL_BUFF, auto_buff::BuffActivation::SMALL_ACTIVATED);
  invalid.timestamp = conflicting.timestamp + std::chrono::milliseconds(1);
  invalid.observations[0].points[0].y = -1.0F;
  assert_idle_frame(processor.process(invalid));

  auto r_degenerate = valid_input(io::GimbalMode::SMALL_BUFF, auto_buff::BuffActivation::SMALL_ACTIVATED);
  r_degenerate.timestamp = invalid.timestamp + std::chrono::milliseconds(1);
  r_degenerate.observations[0].points[4] = r_degenerate.observations[0].points[0];
  assert_idle_frame(processor.process(r_degenerate));

  auto tracked = valid_input(io::GimbalMode::SMALL_BUFF, auto_buff::BuffActivation::SMALL_ACTIVATED);
  tracked.timestamp = t0 + std::chrono::seconds(2);
  assert_idle_frame(processor.process(tracked));
  auto regressed = tracked;
  regressed.timestamp -= std::chrono::milliseconds(1);
  assert_idle_frame(processor.process(regressed));

  auto restarted = tracked;
  restarted.timestamp += std::chrono::milliseconds(1);
  assert_idle_frame(processor.process(restarted));
  auto expired = restarted;
  expired.timestamp += std::chrono::milliseconds(25);
  assert_idle_frame(processor.process(expired));

  debug_snapshot_tests();

  return failures == 0 ? 0 : 1;
}
