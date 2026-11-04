#include <chrono>
#include <iostream>

#include <opencv2/opencv.hpp>

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

  return failures == 0 ? 0 : 1;
}
