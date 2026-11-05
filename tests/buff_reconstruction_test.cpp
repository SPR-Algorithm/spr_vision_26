#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/calib3d.hpp>

#include "tasks/auto_buff/rune_pose_tools.hpp"
#include "tasks/auto_buff/rune_reconstructor.hpp"

namespace
{
int failures = 0;

void check(bool condition, const char * expression, int line)
{
  if (condition) return;
  std::cerr << "CHECK failed: " << expression << " at " << line << '\n';
  ++failures;
}

#define CHECK(condition) check((condition), #condition, __LINE__)

struct Fixture
{
  cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 1500.0, 0.0, 720.0, 0.0, 1480.0,
                            540.0, 0.0, 0.0, 1.0);
  cv::Mat distortion = (cv::Mat_<double>(1, 5) << -0.08, 0.02, 0.0004, -0.0003, 0.0);
  cv::Vec3d rvec;
  cv::Vec3d tvec{0.18, -0.12, 5.8};
  Eigen::Matrix3d model_to_camera;
  Eigen::Matrix3d camera_to_gimbal;
  Eigen::Vector3d camera_translation{0.04, 0.01, 0.11};
  // This is the existing vehicle's basis convention, including its reflected Z axis.
  Eigen::Matrix3d gimbal_to_imu;
  Eigen::Quaterniond imu_q = Eigen::Quaterniond(
    Eigen::AngleAxisd(0.31, Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(-0.17, Eigen::Vector3d::UnitY()) *
    Eigen::AngleAxisd(0.08, Eigen::Vector3d::UnitX()));
  std::filesystem::path directory;
  std::string config_path;

  explicit Fixture(cv::Vec3d tilt_vector = {0.12, -0.25, 0.18}, double phase = 0.0)
  {
    // The rune's +Z points upwards in the image and its +X normal faces the camera.
    cv::Matx33d front(0.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0);
    cv::Matx33d tilt;
    cv::Rodrigues(tilt_vector, tilt);
    cv::Matx33d phase_rotation;
    cv::Rodrigues(cv::Vec3d(phase, 0.0, 0.0), phase_rotation);
    const cv::Matx33d rotation = tilt * front * phase_rotation;
    cv::Rodrigues(rotation, rvec);
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) model_to_camera(row, col) = rotation(row, col);
    }
    camera_to_gimbal << 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0;
    camera_to_gimbal =
      Eigen::AngleAxisd(0.025, Eigen::Vector3d::UnitY()).toRotationMatrix() * camera_to_gimbal;
    gimbal_to_imu << 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0;
    static unsigned int next_fixture = 0;
    directory = std::filesystem::temp_directory_path() /
                ("buff_reconstruction_test_" + std::to_string(getpid()) + "_" +
                 std::to_string(next_fixture++));
    std::filesystem::create_directories(directory);
    config_path = (directory / "camera.yaml").string();
    write_calibration(config_path, camera_matrix, distortion, camera_to_gimbal);
  }

  ~Fixture()
  {
    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
  }

  void write_calibration(
    const std::string & path, const cv::Mat & intrinsics, const cv::Mat & coefficients,
    const Eigen::Matrix3d & camera_rotation) const
  {
    std::ofstream file(path);
    file << std::setprecision(17);
    const auto write_matrix = [&file](const char * key, const auto & value) {
      file << key << ": [";
      for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
          if (row != 0 || col != 0) file << ", ";
          file << value(row, col);
        }
      }
      file << "]\n";
    };
    write_matrix("camera_matrix", cv::Matx33d(intrinsics));
    file << "distort_coeffs: [";
    for (int i = 0; i < coefficients.cols; ++i) {
      if (i != 0) file << ", ";
      file << coefficients.at<double>(0, i);
    }
    file << "]\n";
    write_matrix("R_camera2gimbal", camera_rotation);
    write_matrix("R_gimbal2imubody", gimbal_to_imu);
    file << "t_camera2gimbal: [" << camera_translation.x() << ", " << camera_translation.y()
         << ", " << camera_translation.z() << "]\n";
  }

  Eigen::Matrix3d gimbal_to_world() const
  {
    return gimbal_to_imu.transpose() * imu_q.toRotationMatrix() * gimbal_to_imu;
  }

  Eigen::Vector3d world_point(const Eigen::Vector3d & model) const
  {
    const Eigen::Vector3d camera = model_to_camera * model +
                                   Eigen::Vector3d(tvec[0], tvec[1], tvec[2]);
    return gimbal_to_world() * (camera_to_gimbal * camera + camera_translation);
  }

  static std::vector<cv::Point3f> model_points(int physical_slot)
  {
    // Independently specified physical dimensions in metres, matching the rune drawing.
    const std::vector<Eigen::Vector3d> base = {
      {0.0, 0.0, 0.827}, {0.0, 0.127, 0.7}, {0.0, 0.0, 0.573},
      {0.0, -0.127, 0.7}, {0.0, 0.0, 0.0}};
    const Eigen::AngleAxisd turn(physical_slot * 2.0 * CV_PI / 5.0, Eigen::Vector3d::UnitX());
    std::vector<cv::Point3f> result;
    for (const auto & point : base) {
      const Eigen::Vector3d rotated = turn * point;
      result.emplace_back(rotated.x(), rotated.y(), rotated.z());
    }
    return result;
  }

  auto_buff::RefinedObservation observation(int slot) const
  {
    std::vector<cv::Point2f> image;
    cv::projectPoints(model_points(slot), rvec, tvec, camera_matrix, distortion, image);
    auto_buff::RefinedObservation result;
    std::copy(image.begin(), image.end(), result.points.begin());
    result.state = auto_buff::RuneState{
      auto_buff::RuneKind::SMALL,
      slot == 0 ? auto_buff::ActivationState::INACTIVE : auto_buff::ActivationState::ACTIVATED};
    result.detection_confidence = 0.98F - 0.05F * slot;
    result.quality = auto_buff::TrackingQuality::GOOD;
    result.geometry_score = 1.0F;
    result.armor_polygon.assign(image.begin(), image.begin() + 4);
    result.roi = cv::Rect(0, 0, 1440, 1080);
    return result;
  }
};

void verify_reconstruction(
  const Fixture & fixture, const auto_buff::RuneReconstruction & reconstruction,
  const std::vector<int> & slots)
{
  const auto & pose = reconstruction.pose;
  const Eigen::Matrix3d expected_rotation = fixture.gimbal_to_world() * fixture.camera_to_gimbal *
                                          fixture.model_to_camera;
  CHECK((pose.position_world_m - fixture.world_point(Eigen::Vector3d::Zero())).norm() < 0.002);
  CHECK((pose.plane_normal_world - expected_rotation.col(0)).norm() < 0.002);
  CHECK((pose.phase_zero_direction_world - expected_rotation.col(2)).norm() < 0.002);
  CHECK((pose.orientation_world.toRotationMatrix() - expected_rotation).norm() < 0.003);
  CHECK(std::abs(pose.plane_normal_world.dot(pose.phase_zero_direction_world)) < 1e-6);
  CHECK(std::abs(pose.blade_radius_m - 0.7) < 1e-9);
  CHECK(pose.source == auto_buff::PoseSource::PNP);
  CHECK(pose.reprojection_error_px < 0.01);
  CHECK(reconstruction.blades.size() == slots.size());
  for (const int slot : slots) {
    const auto expected_observation = fixture.observation(slot);
    const auto found = std::find_if(
      reconstruction.blades.begin(), reconstruction.blades.end(), [&](const auto & blade) {
        return cv::norm(blade.observation.points[0] - expected_observation.points[0]) < 0.001;
      });
    CHECK(found != reconstruction.blades.end());
    if (found == reconstruction.blades.end()) continue;
    CHECK(static_cast<int>(found->slot) == slot + 1);
    const Eigen::Vector3d center = Eigen::AngleAxisd(
      slot * 2.0 * CV_PI / 5.0, Eigen::Vector3d::UnitX()) * Eigen::Vector3d(0.0, 0.0, 0.7);
    CHECK((found->armor_center_world_m - fixture.world_point(center)).norm() < 0.002);
  }
}

void reconstruction_tests(const Fixture & fixture)
{
  auto_buff::RuneReconstructor reconstructor(fixture.config_path);
  for (const auto & slots : std::vector<std::vector<int>>{
         {0, 1}, {0, 2}, {0, 1, 3}, {0, 1, 2, 4}, {0, 1, 2, 3, 4}}) {
    std::vector<auto_buff::RefinedObservation> observations;
    for (const int slot : slots) observations.push_back(fixture.observation(slot));
    const auto result = reconstructor.reconstruct(observations, fixture.imu_q);
    CHECK(result.has_value());
    if (result) verify_reconstruction(fixture, *result, slots);
    std::reverse(observations.begin(), observations.end());
    const auto reordered = reconstructor.reconstruct(observations, fixture.imu_q);
    CHECK(reordered.has_value());
    if (reordered) verify_reconstruction(fixture, *reordered, slots);
  }

  CHECK(!reconstructor.reconstruct({}, fixture.imu_q));
  CHECK(!reconstructor.reconstruct({fixture.observation(0)}, fixture.imu_q));
  const std::vector<auto_buff::RefinedObservation> pair = {
    fixture.observation(0), fixture.observation(2)};
  auto scaled_q = fixture.imu_q;
  scaled_q.coeffs() *= 3.0;
  const auto scaled_result = reconstructor.reconstruct(pair, scaled_q);
  CHECK(scaled_result.has_value());
  if (scaled_result) verify_reconstruction(fixture, *scaled_result, {0, 2});
  CHECK(!reconstructor.reconstruct(pair, Eigen::Quaterniond(0.0, 0.0, 0.0, 0.0)));
  auto invalid_q = fixture.imu_q;
  invalid_q.x() = std::numeric_limits<double>::quiet_NaN();
  CHECK(!reconstructor.reconstruct(pair, invalid_q));
  CHECK(!reconstructor.reconstruct(
    {fixture.observation(0), fixture.observation(0)}, fixture.imu_q));
  auto duplicated = pair;
  duplicated.push_back(fixture.observation(2));
  const auto deduplicated = reconstructor.reconstruct(duplicated, fixture.imu_q);
  CHECK(deduplicated.has_value());
  if (deduplicated) verify_reconstruction(fixture, *deduplicated, {0, 2});
  auto invalid_points = pair;
  invalid_points[0].points[0].x = std::numeric_limits<float>::infinity();
  CHECK(!reconstructor.reconstruct(invalid_points, fixture.imu_q));
}

void pnp_tests(const Fixture & fixture)
{
  using auto_buff::rune_pose_tools::solve_pnp_multi_anchor;
  auto model = Fixture::model_points(0);
  const auto second = Fixture::model_points(2);
  model.insert(model.end(), second.begin(), second.end() - 1);
  std::vector<cv::Point2f> image;
  cv::projectPoints(
    model, fixture.rvec, fixture.tvec, fixture.camera_matrix, fixture.distortion, image);
  const auto solved =
    solve_pnp_multi_anchor(image, model, fixture.camera_matrix, fixture.distortion);
  CHECK(solved.has_value());
  const auto four_anchor_pose = solve_pnp_multi_anchor(
    {image[0], image[1], image[2], image[3]}, {model[0], model[1], model[2], model[3]},
    fixture.camera_matrix, fixture.distortion);
  CHECK(four_anchor_pose.has_value());
  if (four_anchor_pose) CHECK(cv::norm(four_anchor_pose->tvec - fixture.tvec) < 0.001);
  if (solved) {
    CHECK(cv::norm(solved->tvec - fixture.tvec) < 0.001);
    cv::Matx33d recovered;
    cv::Rodrigues(solved->rvec, recovered);
    for (const auto & point : model) {
      CHECK((recovered * cv::Vec3d(point.x, point.y, point.z) + solved->tvec)[2] > 0.0);
    }
  }
  CHECK(!solve_pnp_multi_anchor({}, {}, fixture.camera_matrix, fixture.distortion));
  auto short_image = image;
  short_image.pop_back();
  CHECK(!solve_pnp_multi_anchor(short_image, model, fixture.camera_matrix, fixture.distortion));
  CHECK(!solve_pnp_multi_anchor(
    {image[0], image[1], image[2]}, {model[0], model[1], model[2]}, fixture.camera_matrix,
    fixture.distortion));
  auto invalid_image = image;
  invalid_image[0].x = std::numeric_limits<float>::quiet_NaN();
  CHECK(!solve_pnp_multi_anchor(invalid_image, model, fixture.camera_matrix, fixture.distortion));
  auto invalid_model = model;
  invalid_model[0].z = std::numeric_limits<float>::infinity();
  CHECK(!solve_pnp_multi_anchor(image, invalid_model, fixture.camera_matrix, fixture.distortion));
  CHECK(!solve_pnp_multi_anchor(image, model, cv::Mat(), fixture.distortion));
  auto invalid_camera = fixture.camera_matrix.clone();
  invalid_camera.at<double>(0, 0) = -1.0;
  CHECK(!solve_pnp_multi_anchor(image, model, invalid_camera, fixture.distortion));
  invalid_camera = fixture.camera_matrix.clone();
  invalid_camera.at<double>(2, 2) = 0.0;
  CHECK(!solve_pnp_multi_anchor(image, model, invalid_camera, fixture.distortion));
  auto invalid_distortion = fixture.distortion.clone();
  invalid_distortion.at<double>(0, 0) = std::numeric_limits<double>::infinity();
  CHECK(!solve_pnp_multi_anchor(image, model, fixture.camera_matrix, invalid_distortion));
  CHECK(!solve_pnp_multi_anchor(image, model, fixture.camera_matrix, cv::Mat::zeros(1, 3, CV_64F)));

  const std::vector<cv::Point3f> collinear = {
    {0.0F, 0.0F, 0.0F}, {0.0F, 0.1F, 0.0F}, {0.0F, 0.2F, 0.0F}, {0.0F, 0.3F, 0.0F}};
  std::vector<cv::Point2f> line_image;
  cv::projectPoints(
    collinear, fixture.rvec, fixture.tvec, fixture.camera_matrix, fixture.distortion, line_image);
  CHECK(!solve_pnp_multi_anchor(line_image, collinear, fixture.camera_matrix, fixture.distortion));
  CHECK(!solve_pnp_multi_anchor(
    std::vector<cv::Point2f>(model.size(), cv::Point2f(10.0F, 20.0F)), model,
    fixture.camera_matrix, fixture.distortion));
}

void calibration_tests(const Fixture & fixture)
{
  const auto throws_for = [](const std::string & path) {
    try {
      auto_buff::RuneReconstructor reconstructor(path);
      return false;
    } catch (const std::exception &) {
      return true;
    }
  };
  const auto missing_path = (fixture.directory / "missing.yaml").string();
  {
    std::ofstream missing(missing_path);
    missing << "camera_matrix: [1, 0, 0, 0, 1, 0, 0, 0, 1]\n";
  }
  CHECK(throws_for(missing_path));
  const auto invalid_path = (fixture.directory / "invalid.yaml").string();
  auto invalid_camera = fixture.camera_matrix.clone();
  invalid_camera.at<double>(0, 0) = 0.0;
  fixture.write_calibration(
    invalid_path, invalid_camera, fixture.distortion, fixture.camera_to_gimbal);
  CHECK(throws_for(invalid_path));
  fixture.write_calibration(
    invalid_path, fixture.camera_matrix, fixture.distortion, Eigen::Matrix3d::Zero());
  CHECK(throws_for(invalid_path));
}
}  // namespace

int main()
{
  try {
    const Fixture fixture;
    pnp_tests(fixture);
    reconstruction_tests(fixture);
    calibration_tests(fixture);
    const Fixture second_view({-0.27, 0.40, -0.21}, 1.1);
    reconstruction_tests(second_view);
    const Fixture third_view({0.35, 0.55, 0.20}, -0.7);
    reconstruction_tests(third_view);
  } catch (const std::exception & error) {
    std::cerr << "Unexpected reconstruction test exception: " << error.what() << '\n';
    return 1;
  }
  if (failures != 0) return 1;
  std::cout << "Buff reconstruction: physical projections, world transforms and rejection cases "
               "passed\n";
  return 0;
}
