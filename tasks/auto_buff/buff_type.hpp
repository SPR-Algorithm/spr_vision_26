#ifndef BUFF__TYPE_HPP
#define BUFF__TYPE_HPP

#include <algorithm>
#include <deque>
#include <eigen3/Eigen/Dense> // 必须在opencv2/core/eigen.hpp上面
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include <vector>

#include "tools/math_tools.hpp"
namespace auto_buff {
const int INF = 1000000;
enum PowerRune_type { SMALL, BIG };
enum FanBlade_type { _target, _unlight, _light };
enum Track_status { TRACK, TEM_LOSE, LOSE };

class FanBlade {
public:
  cv::Point2f center;              // 扇页中心(4个装甲板角点均值)
  std::vector<cv::Point2f> points; // 关键点(4角点 + R标中心)
  double angle, width, height;
  FanBlade_type type; // 类型
  int cls = -1;       // 类别: 0-未激活 1-小符已激活 2-大符已激活 (-1 未知)

  explicit FanBlade() = default;

  explicit FanBlade(const std::vector<cv::Point2f> &kpt,
                    cv::Point2f keypoints_center, FanBlade_type t,
                    int cls = -1);

  explicit FanBlade(FanBlade_type t);
};

class PowerRune {
public:
  cv::Point2f r_center;
  std::vector<FanBlade> fanblades; // 按target开始顺时针

  int light_num;
  PowerRune_type type = SMALL; // 大小符状态(由 target 扇叶类别推断)

  Eigen::Vector3d xyz_in_world; // 单位：m
  Eigen::Vector3d ypr_in_world; // 单位：rad
  Eigen::Vector3d ypd_in_world; // 球坐标系

  Eigen::Vector3d blade_xyz_in_world; // 单位：m
  Eigen::Vector3d blade_ypd_in_world; // 球坐标系, 单位: m

  explicit PowerRune(std::vector<FanBlade> &ts, const cv::Point2f r_center,
                     std::optional<PowerRune> last_powerrune);
  explicit PowerRune() = default;

  FanBlade &target() { return fanblades[0]; };

  bool is_unsolve() const { return unsolvable_; }

private:
  double target_angle_;
  bool unsolvable_ = false;

  double atan_angle(cv::Point2f v) const; // [0, 2CV_PI]
};
} // namespace auto_buff
#endif // BUFF_TYPE_HPP
