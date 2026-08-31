#include <iostream>

#include "tasks/auto_buff/buff_detector.hpp"

int main()
{
  const std::vector<cv::Point2f> points = {
    {320.0F, 100.0F}, {420.0F, 200.0F}, {320.0F, 300.0F}, {220.0F, 200.0F}, {320.0F, 200.0F}};
  const auto blade = auto_buff::FanBlade(points, {320.0F, 200.0F}, auto_buff::_light, 2, 0.73F);
  const auto observation = auto_buff::Buff_Detector::to_observation(blade);
  if (!observation.has_value() || observation->activation != auto_buff::BuffActivation::BIG_ACTIVATED ||
      observation->confidence != 0.73F || observation->points[4] != points[4]) {
    std::cerr << "detector adapter did not preserve the standard observation\n";
    return 1;
  }
  return 0;
}
