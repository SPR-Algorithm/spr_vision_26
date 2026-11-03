#ifndef AUTO_BUFF__SOLVER_HPP
#define AUTO_BUFF__SOLVER_HPP

#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>  // 必须在opencv2/core/eigen.hpp上面
#include <opencv2/core/eigen.hpp>
#include <optional>

#include "buff_type.hpp"
#include "tools/math_tools.hpp"
namespace auto_buff
{
// 旋转角度
const double THETA = 2.0 * CV_PI / 5.0;  // 2/5π

class Solver
{
public:
  explicit Solver(const std::string & config_path);

  Eigen::Matrix3d R_gimbal2world() const;

  void set_R_gimbal2world(const Eigen::Quaterniond & q);

  bool solve(std::optional<PowerRune> & ps) const;

  // 调试用
  cv::Point2f point_buff2pixel(cv::Point3f x);

  std::vector<cv::Point2f> reproject_buff(
    const Eigen::Vector3d & xyz_in_world, double yaw, double row) const;

private:
  cv::Mat camera_matrix_;
  cv::Mat distort_coeffs_;
  Eigen::Matrix3d R_gimbal2imubody_;
  Eigen::Matrix3d R_camera2gimbal_;
  Eigen::Vector3d t_camera2gimbal_;
  Eigen::Matrix3d R_gimbal2world_;

  mutable cv::Vec3d rvec_, tvec_;

  // buff 坐标系 3D 点, 与 detector order_keypoints [上,右,下,左,R标] 角点顺序一致
  // [0] 上  [1] 右  [2] 下  [3] 左  [4] 扇叶装甲中心  [5] 结构点  [6] R标/buff原点
  const std::vector<cv::Point3f> OBJECT_POINTS = {
    cv::Point3f(0, 0, 827e-3),       // [0] 上
    cv::Point3f(0, 127e-3, 700e-3),  // [1] 右
    cv::Point3f(0, 0, 573e-3),       // [2] 下
    cv::Point3f(0, -127e-3, 700e-3), // [3] 左
    cv::Point3f(0, 0, 700e-3),       // [4] 扇叶装甲中心
    cv::Point3f(0, 0, 220e-3),       // [5]
    cv::Point3f(0, 0, 0)};           // [6] R 标旋转中心 / buff 原点

  cv::Matx33f rotation_matrix(double angle) const;

  void compute_rotated_points(std::vector<std::vector<cv::Point3f>> & object_points);
};
}  // namespace auto_buff
#endif  // AUTO_AIM__SOLVER_HPP
