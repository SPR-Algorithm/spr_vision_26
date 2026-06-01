#include "aux_solver.hpp"

#include <yaml-cpp/yaml.h>

#include <opencv2/core/eigen.hpp>
#include <stdexcept>
#include <string>

#include "tools/yaml.hpp"

namespace auto_aim
{
constexpr double LIGHTBAR_LENGTH = 56e-3;
constexpr double BIG_ARMOR_WIDTH = 230e-3;
constexpr double SMALL_ARMOR_WIDTH = 135e-3;

const std::vector<cv::Point3f> BIG_ARMOR_POINTS{
  {0, BIG_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
  {0, -BIG_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
  {0, -BIG_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2},
  {0, BIG_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2}};
const std::vector<cv::Point3f> SMALL_ARMOR_POINTS{
  {0, SMALL_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
  {0, -SMALL_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
  {0, -SMALL_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2},
  {0, SMALL_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2}};

AuxSolver::AuxSolver(const std::string & config_path)
{
  auto yaml = YAML::LoadFile(config_path);
  auto aux = yaml["aux_camera_calibration"];
  if (!aux) {
    throw std::runtime_error("[AuxSolver] aux_camera_calibration not found in config");
  }

  auto R_camera2gimbal_data = aux["R_camera2gimbal"].as<std::vector<double>>();
  auto t_camera2gimbal_data = aux["t_camera2gimbal"].as<std::vector<double>>();
  R_camera2gimbal_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_camera2gimbal_data.data());
  t_camera2gimbal_ = Eigen::Matrix<double, 3, 1>(t_camera2gimbal_data.data());

  auto camera_matrix_data = aux["camera_matrix"].as<std::vector<double>>();
  auto distort_coeffs_data = aux["distort_coeffs"].as<std::vector<double>>();
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> camera_matrix(camera_matrix_data.data());
  Eigen::Matrix<double, 1, 5> distort_coeffs(distort_coeffs_data.data());
  cv::eigen2cv(camera_matrix, camera_matrix_);
  cv::eigen2cv(distort_coeffs, distort_coeffs_);
}

void AuxSolver::solve(Armor & armor) const
{
  const auto & object_points =
    (armor.type == ArmorType::big) ? BIG_ARMOR_POINTS : SMALL_ARMOR_POINTS;

  cv::Vec3d rvec, tvec;
  cv::solvePnP(
    object_points, armor.points, camera_matrix_, distort_coeffs_, rvec, tvec, false,
    cv::SOLVEPNP_IPPE);

  Eigen::Vector3d xyz_in_camera;
  cv::cv2eigen(tvec, xyz_in_camera);
  armor.xyz_in_gimbal = R_camera2gimbal_ * xyz_in_camera + t_camera2gimbal_;
}

double AuxSolver::yaw_from_armor(const Armor & armor)
{
  return std::atan2(armor.xyz_in_gimbal.y(), armor.xyz_in_gimbal.x());
}

}  // namespace auto_aim
