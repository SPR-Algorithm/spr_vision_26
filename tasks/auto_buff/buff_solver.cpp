#include "buff_solver.hpp"

#include "tools/logger.hpp"

namespace auto_buff
{
namespace
{
constexpr double REPROJ_ERR_THRESHOLD = 15.0;  // px, 5点 PnP 重投影误差上限

double mean_reprojection_error(
  const std::vector<cv::Point3f> & object_points, const std::vector<cv::Point2f> & image_points,
  const cv::Mat & camera_matrix, const cv::Mat & distort_coeffs, const cv::Vec3d & rvec,
  const cv::Vec3d & tvec)
{
  std::vector<cv::Point2f> projected;
  cv::projectPoints(object_points, rvec, tvec, camera_matrix, distort_coeffs, projected);
  double err = 0.0;
  for (size_t i = 0; i < image_points.size(); ++i) {
    err += cv::norm(projected[i] - image_points[i]);
  }
  return err / static_cast<double>(image_points.size());
}
}  // namespace

cv::Matx33f Solver::rotation_matrix(double angle) const
{
  return cv::Matx33f(
    1, 0, 0, 0, std::cos(angle), -std::sin(angle), 0, std::sin(angle), std::cos(angle));
}

void Solver::compute_rotated_points(std::vector<std::vector<cv::Point3f>> & object_points)
{
  const std::vector<cv::Point3f> & base_points = object_points[0];
  for (int i = 1; i < 5; ++i) {
    double angle = i * THETA;
    cv::Matx33f R = rotation_matrix(angle);
    std::vector<cv::Point3f> rotated_points;
    for (const auto & point : base_points) {
      cv::Vec3f vec(point.x, point.y, point.z);
      cv::Vec3f rotated_vec = R * vec;
      rotated_points.emplace_back(rotated_vec[0], rotated_vec[1], rotated_vec[2]);
    }
    object_points[i] = rotated_points;
  }
}

Solver::Solver(const std::string & config_path) : R_gimbal2world_(Eigen::Matrix3d::Identity())
{
  auto yaml = YAML::LoadFile(config_path);

  auto R_gimbal2imubody_data = yaml["R_gimbal2imubody"].as<std::vector<double>>();
  auto R_camera2gimbal_data = yaml["R_camera2gimbal"].as<std::vector<double>>();
  auto t_camera2gimbal_data = yaml["t_camera2gimbal"].as<std::vector<double>>();
  R_gimbal2imubody_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_gimbal2imubody_data.data());
  R_camera2gimbal_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_camera2gimbal_data.data());
  t_camera2gimbal_ = Eigen::Matrix<double, 3, 1>(t_camera2gimbal_data.data());

  auto camera_matrix_data = yaml["camera_matrix"].as<std::vector<double>>();
  auto distort_coeffs_data = yaml["distort_coeffs"].as<std::vector<double>>();
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> camera_matrix(camera_matrix_data.data());
  Eigen::Matrix<double, 1, 5> distort_coeffs(distort_coeffs_data.data());
  cv::eigen2cv(camera_matrix, camera_matrix_);
  cv::eigen2cv(distort_coeffs, distort_coeffs_);
}

Eigen::Matrix3d Solver::R_gimbal2world() const { return R_gimbal2world_; }

void Solver::set_R_gimbal2world(const Eigen::Quaterniond & q)
{
  Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
  R_gimbal2world_ = R_gimbal2imubody_.transpose() * R_imubody2imuabs * R_gimbal2imubody_;
}

bool Solver::solve(std::optional<PowerRune> & ps) const
{
  if (!ps.has_value() || ps->is_unsolve()) return false;

  PowerRune & p = ps.value();
  const auto & corners = p.target().points;
  if (corners.size() < 4) {
    tools::logger()->debug("[Buff_Solver] target 角点不足: {}", corners.size());
    return false;
  }

  // 5点 PnP: 4角点 + 精修 r_center, 对应 buff 原点 (0,0,0)
  const std::vector<cv::Point2f> image_points = {
    corners[0], corners[1], corners[2], corners[3], p.r_center};
  const std::vector<cv::Point3f> object_points = {
    OBJECT_POINTS[0], OBJECT_POINTS[1], OBJECT_POINTS[2], OBJECT_POINTS[3], OBJECT_POINTS[6]};

  if (
    !cv::solvePnP(
      object_points, image_points, camera_matrix_, distort_coeffs_, rvec_, tvec_, false,
      cv::SOLVEPNP_SQPNP)) {
    tools::logger()->debug("[Buff_Solver] solvePnP 失败");
    return false;
  }

  const double reproj_err = mean_reprojection_error(
    object_points, image_points, camera_matrix_, distort_coeffs_, rvec_, tvec_);
  if (reproj_err > REPROJ_ERR_THRESHOLD) {
    tools::logger()->debug("[Buff_Solver] 重投影误差过大: {:.2f}px", reproj_err);
    return false;
  }

  Eigen::Vector3d t_buff2camera;
  cv::cv2eigen(tvec_, t_buff2camera);
  cv::Mat rmat;
  cv::Rodrigues(rvec_, rmat);
  Eigen::Matrix3d R_buff2camera;
  cv::cv2eigen(rmat, R_buff2camera);

  const Eigen::Vector3d blade_xyz_in_buff(
    OBJECT_POINTS[4].x, OBJECT_POINTS[4].y, OBJECT_POINTS[4].z);

  // buff -> camera; t_buff2camera 为 R 标中心 (buff 原点)
  const Eigen::Vector3d xyz_in_camera = t_buff2camera;
  const Eigen::Vector3d blade_xyz_in_camera = R_buff2camera * blade_xyz_in_buff + t_buff2camera;

  // camera -> gimbal
  const Eigen::Matrix3d R_buff2gimbal = R_camera2gimbal_ * R_buff2camera;
  const Eigen::Vector3d xyz_in_gimbal = R_camera2gimbal_ * xyz_in_camera + t_camera2gimbal_;
  const Eigen::Vector3d blade_xyz_in_gimbal =
    R_camera2gimbal_ * blade_xyz_in_camera + t_camera2gimbal_;

  // gimbal -> world
  const Eigen::Matrix3d R_buff2world = R_gimbal2world_ * R_buff2gimbal;

  p.xyz_in_world = R_gimbal2world_ * xyz_in_gimbal;
  p.ypd_in_world = tools::xyz2ypd(p.xyz_in_world);

  p.blade_xyz_in_world = R_gimbal2world_ * blade_xyz_in_gimbal;
  p.blade_ypd_in_world = tools::xyz2ypd(p.blade_xyz_in_world);

  p.ypr_in_world = tools::eulers(R_buff2world, 2, 1, 0);
  return true;
}

cv::Point2f Solver::point_buff2pixel(cv::Point3f x)
{
  std::vector<cv::Point3d> world_points;
  std::vector<cv::Point2d> image_points;
  world_points.push_back(x);
  cv::projectPoints(world_points, rvec_, tvec_, camera_matrix_, distort_coeffs_, image_points);
  return image_points.back();
}

std::vector<cv::Point2f> Solver::reproject_buff(
  const Eigen::Vector3d & xyz_in_world, double yaw, double row) const
{
  auto R_buff2world = tools::rotation_matrix(Eigen::Vector3d(yaw, 0.0, row));

  const Eigen::Vector3d & t_buff2world = xyz_in_world;
  Eigen::Matrix3d R_buff2camera =
    R_camera2gimbal_.transpose() * R_gimbal2world_.transpose() * R_buff2world;
  Eigen::Vector3d t_buff2camera =
    R_camera2gimbal_.transpose() * (R_gimbal2world_.transpose() * t_buff2world - t_camera2gimbal_);

  cv::Vec3d rvec;
  cv::Mat R_buff2camera_cv;
  cv::eigen2cv(R_buff2camera, R_buff2camera_cv);
  cv::Rodrigues(R_buff2camera_cv, rvec);
  cv::Vec3d tvec(t_buff2camera[0], t_buff2camera[1], t_buff2camera[2]);

  // 调试投影: [0..3] 角点四边形, [4..6] 扇叶中心/结构点/R原点
  std::vector<cv::Point2f> image_points;
  cv::projectPoints(OBJECT_POINTS, rvec, tvec, camera_matrix_, distort_coeffs_, image_points);
  return image_points;
}
}  // namespace auto_buff
