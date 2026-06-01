#ifndef AUTO_AIM__AUX_SOLVER_HPP
#define AUTO_AIM__AUX_SOLVER_HPP

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <string>

#include "armor.hpp"

namespace auto_aim
{
/// 辅助相机 yaw 解算器，仅计算 xyz_in_gimbal 与 yaw
class AuxSolver
{
public:
  explicit AuxSolver(const std::string & config_path);

  /// 对单个 armor 解算，填充 xyz_in_gimbal
  void solve(Armor & armor) const;

  /// 从 xyz_in_gimbal 计算 yaw（弧度），用于云台旋转
  static double yaw_from_armor(const Armor & armor);

private:
  cv::Mat camera_matrix_;
  cv::Mat distort_coeffs_;
  Eigen::Matrix3d R_camera2gimbal_;
  Eigen::Vector3d t_camera2gimbal_;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__AUX_SOLVER_HPP
