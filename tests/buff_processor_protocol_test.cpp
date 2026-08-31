#include <cassert>
#include <chrono>

#include <opencv2/opencv.hpp>

#include "tasks/auto_buff/buff_processor.hpp"

namespace
{
auto valid_input(io::GimbalMode mode, auto_buff::BuffActivation activation)
{
  auto_buff::BuffInput input;
  input.img = cv::Mat::zeros(480, 640, CV_8UC3);
  input.timestamp = std::chrono::steady_clock::now();
  input.points = {
    cv::Point2f{320.0F, 100.0F}, cv::Point2f{420.0F, 200.0F},
    cv::Point2f{320.0F, 300.0F}, cv::Point2f{220.0F, 200.0F},
    cv::Point2f{320.0F, 200.0F}};
  input.activation = activation;
  input.confidence = 0.95F;
  input.imu_q = Eigen::Quaterniond::Identity();
  input.gimbal_state = {0.0F, 0.0F, 0.0F, 0.0F, 24.0F, 0};
  input.gimbal_mode = mode;
  return input;
}

void assert_idle_frame(const io::VisionToGimbal & output)
{
  assert(output.head[0] == 'S');
  assert(output.head[1] == 'P');
  assert(output.tail == 0xef);
  assert(output.mode == 0);
  assert(output.yaw == 0.0F);
  assert(output.yaw_vel == 0.0F);
  assert(output.yaw_acc == 0.0F);
  assert(output.pitch == 0.0F);
  assert(output.pitch_vel == 0.0F);
  assert(output.pitch_acc == 0.0F);
}
}  // namespace

int main()
{
  static_assert(sizeof(io::VisionToGimbal) == 28);

  auto_buff::BuffProcessor processor("", "");

  const auto small = processor.process(
    valid_input(io::GimbalMode::SMALL_BUFF, auto_buff::BuffActivation::SMALL_ACTIVATED));
  assert(small.mode == 1);
  assert(small.head[0] == 'S');
  assert(small.head[1] == 'P');
  assert(small.tail == 0xef);

  const auto big = processor.process(
    valid_input(io::GimbalMode::BIG_BUFF, auto_buff::BuffActivation::BIG_ACTIVATED));
  assert(big.mode == 1);

  auto conflicting = valid_input(io::GimbalMode::SMALL_BUFF, auto_buff::BuffActivation::BIG_ACTIVATED);
  assert_idle_frame(processor.process(conflicting));

  auto invalid = valid_input(io::GimbalMode::SMALL_BUFF, auto_buff::BuffActivation::SMALL_ACTIVATED);
  invalid.points[0].y = -1.0F;
  assert_idle_frame(processor.process(invalid));
}
