#include "rune_reconstructor.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace auto_buff
{
namespace
{
constexpr double BLADE_RADIUS_M = 0.7;
constexpr double ARMOR_HALF_DIAGONAL_M = 0.127;
constexpr double SLOT_ANGLE_RAD = 2.0 * CV_PI / 5.0;
constexpr double MAX_REPROJECTION_ERROR_PX = 15.0;
constexpr double MAX_SLOT_ANGLE_ERROR_RAD = SLOT_ANGLE_RAD / 4.0;
constexpr double MAX_RADIUS_RELATIVE_ERROR = 0.35;

bool finite_point(const cv::Point2f & point)
{
  return std::isfinite(point.x) && std::isfinite(point.y);
}

std::array<cv::Point3f, 4> armor_model(int slot)
{
  const double angle = slot * SLOT_ANGLE_RAD;
  const double cosine = std::cos(angle);
  const double sine = std::sin(angle);
  const std::array<cv::Point3f, 4> base = {
    cv::Point3f(0.0F, 0.0F, BLADE_RADIUS_M + ARMOR_HALF_DIAGONAL_M),
    cv::Point3f(0.0F, ARMOR_HALF_DIAGONAL_M, BLADE_RADIUS_M),
    cv::Point3f(0.0F, 0.0F, BLADE_RADIUS_M - ARMOR_HALF_DIAGONAL_M),
    cv::Point3f(0.0F, -ARMOR_HALF_DIAGONAL_M, BLADE_RADIUS_M)};
  auto points = base;
  for (auto & point : points) {
    const double y = point.y;
    const double z = point.z;
    point.y = static_cast<float>(cosine * y - sine * z);
    point.z = static_cast<float>(sine * y + cosine * z);
  }
  return points;
}

std::vector<cv::Point3f> armor_boundary_model(int slot)
{
  const auto corners = armor_model(slot);
  std::vector<cv::Point3f> samples;
  constexpr int samples_per_edge = 32;
  samples.reserve(4 * samples_per_edge);
  // Only the established physical armour diamond is modelled. R and light-arm shapes
  // do not have measured 3D boundaries and cannot contribute invented Chamfer residuals.
  for (std::size_t edge = 0; edge < corners.size(); ++edge) {
    for (int i = 0; i < samples_per_edge; ++i) {
      const float ratio = static_cast<float>(i) / samples_per_edge;
      samples.push_back(corners[edge] * (1.0F - ratio) + corners[(edge + 1) % 4] * ratio);
    }
  }
  return samples;
}

cv::Point3f armor_center_model(int slot)
{
  const double angle = slot * SLOT_ANGLE_RAD;
  return cv::Point3f(0.0F, -BLADE_RADIUS_M * std::sin(angle), BLADE_RADIUS_M * std::cos(angle));
}

// Diagonal intersection, unlike the average of four pixels, preserves the physical centre
// under perspective projection.
std::optional<cv::Point2f> armor_center_pixel(const BuffPoints & points)
{
  const cv::Point2d first = cv::Point2d(points[2]) - cv::Point2d(points[0]);
  const cv::Point2d second = cv::Point2d(points[3]) - cv::Point2d(points[1]);
  const cv::Point2d offset = cv::Point2d(points[1]) - cv::Point2d(points[0]);
  const double denominator = first.x * second.y - first.y * second.x;
  if (std::abs(denominator) < 1e-9) return std::nullopt;
  const double ratio = (offset.x * second.y - offset.y * second.x) / denominator;
  const double other_ratio = (offset.x * first.y - offset.y * first.x) / denominator;
  if (ratio <= 0.0 || ratio >= 1.0 || other_ratio <= 0.0 || other_ratio >= 1.0) {
    return std::nullopt;
  }
  const cv::Point2d center = cv::Point2d(points[0]) + ratio * first;
  const cv::Point2f result(center);
  if (!finite_point(result)) return std::nullopt;
  return result;
}

float median(std::vector<float> values)
{
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  return values.size() % 2 == 0 ? (values[middle - 1] + values[middle]) * 0.5F : values[middle];
}

cv::Point2f fused_r_center(const std::vector<const RefinedObservation *> & blades)
{
  std::vector<float> x_values;
  std::vector<float> y_values;
  std::vector<cv::Point> contour_points;
  double radius_px = 0.0;
  for (const auto * blade : blades) {
    x_values.push_back(blade->points[4].x);
    y_values.push_back(blade->points[4].y);
    radius_px += cv::norm(*armor_center_pixel(blade->points) - blade->points[4]);
    const auto & feature = blade->r_mark_contour;
    if (feature.usable() && feature.support >= 0.45F && feature.contour.size() >= 5) {
      contour_points.insert(contour_points.end(), feature.contour.begin(), feature.contour.end());
    }
  }
  const cv::Point2f network_center(median(x_values), median(y_values));
  if (contour_points.size() >= 5) {
    try {
      const auto ellipse = cv::fitEllipse(contour_points);
      // A bright unrelated component must not move the common R origin away from the network
      // consensus. This remains a fallback PnP observation, rather than contour pose refinement.
      const double maximum_offset = std::max(8.0, 0.15 * radius_px / blades.size());
      if (
        finite_point(ellipse.center) && std::isfinite(ellipse.size.width) &&
        std::isfinite(ellipse.size.height) && ellipse.size.width > 0.0F &&
        ellipse.size.height > 0.0F && cv::norm(ellipse.center - network_center) <= maximum_offset) {
        return ellipse.center;
      }
    } catch (const cv::Exception &) {
      // Degenerate contours cannot suppress the network R fallback.
    }
  }
  return network_center;
}

std::optional<cv::Vec3d> intersect_model_plane(
  const cv::Point2f & pixel, const rune_pose_tools::CameraPose & pose,
  const rune_pose_tools::CameraCalibration & calibration)
{
  std::vector<cv::Point2f> normalized;
  cv::undistortPoints(
    std::vector<cv::Point2f>{pixel}, normalized, calibration.camera_matrix,
    calibration.distortion_coeffs);
  if (normalized.size() != 1 || !finite_point(normalized.front())) return std::nullopt;
  const cv::Vec3d ray(normalized.front().x, normalized.front().y, 1.0);
  cv::Matx33d rotation;
  cv::Rodrigues(pose.rvec, rotation);
  const cv::Vec3d normal(rotation(0, 0), rotation(1, 0), rotation(2, 0));
  const double denominator = normal.dot(ray);
  if (std::abs(denominator) < 1e-9) return std::nullopt;
  const double depth = normal.dot(pose.tvec) / denominator;
  if (!std::isfinite(depth) || depth <= 1e-6) return std::nullopt;
  const cv::Vec3d point = rotation.t() * (depth * ray - pose.tvec);
  if (!std::isfinite(point[0]) || !std::isfinite(point[1]) || !std::isfinite(point[2])) {
    return std::nullopt;
  }
  return point;
}

std::optional<rune_pose_tools::CameraPose> reference_pose_prior(
  const std::optional<rune_pose_tools::CameraPose> & previous_camera_pose,
  const BuffPoints & current_points, const rune_pose_tools::CameraCalibration & calibration)
{
  if (!previous_camera_pose) return std::nullopt;
  const auto center = armor_center_pixel(current_points);
  if (!center) return std::nullopt;
  const auto model_center = intersect_model_plane(*center, *previous_camera_pose, calibration);
  if (!model_center) return std::nullopt;
  const double radius = std::hypot((*model_center)[1], (*model_center)[2]);
  if (std::abs(radius - BLADE_RADIUS_M) > MAX_RADIUS_RELATIVE_ERROR * BLADE_RADIUS_M) {
    return std::nullopt;
  }
  // The previous pose's +Z belongs to its own reference blade. Recover the current blade's
  // angle in that plane, then change the model basis so the current reference is slot zero.
  // The physical normal and center are unchanged by this basis rotation.
  const double angle = std::atan2(-(*model_center)[1], (*model_center)[2]);
  const double cosine = std::cos(angle), sine = std::sin(angle);
  const cv::Matx33d phase_rotation(1.0, 0.0, 0.0, 0.0, cosine, -sine, 0.0, sine, cosine);
  cv::Matx33d previous_rotation;
  cv::Rodrigues(previous_camera_pose->rvec, previous_rotation);
  rune_pose_tools::CameraPose prior;
  cv::Rodrigues(previous_rotation * phase_rotation, prior.rvec);
  prior.tvec = previous_camera_pose->tvec;
  return prior;
}

double reprojection_error(
  const std::vector<cv::Point2f> & image_points, const std::vector<cv::Point3f> & model_points,
  const rune_pose_tools::CameraPose & pose, const rune_pose_tools::CameraCalibration & calibration)
{
  cv::Matx33d rotation;
  cv::Rodrigues(pose.rvec, rotation);
  for (const auto & point : model_points) {
    const cv::Vec3d camera_point = rotation * cv::Vec3d(point.x, point.y, point.z) + pose.tvec;
    if (!std::isfinite(camera_point[2]) || camera_point[2] <= 1e-6) {
      return std::numeric_limits<double>::infinity();
    }
  }
  std::vector<cv::Point2f> projected;
  cv::projectPoints(
    model_points, pose.rvec, pose.tvec, calibration.camera_matrix, calibration.distortion_coeffs,
    projected);
  double error = 0.0;
  for (std::size_t i = 0; i < projected.size(); ++i) {
    if (!finite_point(projected[i])) return std::numeric_limits<double>::infinity();
    error += cv::norm(projected[i] - image_points[i]);
  }
  return error / projected.size();
}

bool valid_rotation(const Eigen::Matrix3d & rotation, bool allow_reflection = false)
{
  return rotation.allFinite() &&
         (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() < 1e-3 &&
         std::abs((allow_reflection ? std::abs(rotation.determinant()) : rotation.determinant()) - 1.0) <
           1e-3;
}

std::vector<double> read_values(const YAML::Node & yaml, const char * key)
{
  const auto values = yaml[key].as<std::vector<double>>();
  if (!std::all_of(values.begin(), values.end(), [](double value) { return std::isfinite(value); })) {
    throw std::invalid_argument(std::string("auto_buff: nonfinite camera calibration field ") + key);
  }
  return values;
}

std::vector<double> read_values(const YAML::Node & yaml, const char * key, std::size_t count)
{
  auto values = read_values(yaml, key);
  if (values.size() != count) {
    throw std::invalid_argument(
      std::string("auto_buff: ") + key + " must contain " + std::to_string(count) + " values");
  }
  return values;
}

rune_pose_tools::CameraCalibration load_calibration(const std::string & camera_config)
{
  const auto yaml = YAML::LoadFile(camera_config);
  const auto intrinsics = read_values(yaml, "camera_matrix", 9);
  const auto distortion = read_values(yaml, "distort_coeffs");
  const auto camera_rotation = read_values(yaml, "R_camera2gimbal", 9);
  const auto camera_translation = read_values(yaml, "t_camera2gimbal", 3);
  const auto imu_rotation = read_values(yaml, "R_gimbal2imubody", 9);

  // These are the coefficient counts accepted by OpenCV's standard pinhole model.
  if (
    distortion.size() != 4 && distortion.size() != 5 && distortion.size() != 8 &&
    distortion.size() != 12 && distortion.size() != 14) {
    throw std::invalid_argument("auto_buff: distort_coeffs must contain 4, 5, 8, 12 or 14 values");
  }
  if (intrinsics[0] <= 0.0 || intrinsics[4] <= 0.0) {
    throw std::invalid_argument("auto_buff: camera focal lengths must be positive");
  }

  rune_pose_tools::CameraCalibration calibration;
  calibration.camera_matrix = cv::Mat(3, 3, CV_64FC1);
  std::copy(intrinsics.begin(), intrinsics.end(), calibration.camera_matrix.ptr<double>());
  calibration.distortion_coeffs = cv::Mat(1, static_cast<int>(distortion.size()), CV_64FC1);
  std::copy(distortion.begin(), distortion.end(), calibration.distortion_coeffs.ptr<double>());

  // Configuration rotations are flattened row-major matrices, with the legacy directions intact.
  using RowMajorMatrix3d = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;
  calibration.R_camera2gimbal = Eigen::Map<const RowMajorMatrix3d>(camera_rotation.data());
  calibration.t_camera2gimbal_m = Eigen::Map<const Eigen::Vector3d>(camera_translation.data());
  calibration.R_gimbal2imubody = Eigen::Map<const RowMajorMatrix3d>(imu_rotation.data());
  if (
    !valid_rotation(calibration.R_camera2gimbal) ||
    !valid_rotation(calibration.R_gimbal2imubody, true)) {
    throw std::invalid_argument("auto_buff: camera and IMU bases must be orthogonal");
  }
  return calibration;
}
}  // namespace

RuneReconstructor::RuneReconstructor(const std::string & camera_config)
: calibration_(load_calibration(camera_config))
{
}

std::optional<RuneReconstruction> RuneReconstructor::reconstruct(
  const std::vector<RefinedObservation> & blades, const Eigen::Quaterniond & imu_q,
  const std::optional<RunePose> & previous_pose) const
{
  if (
    blades.size() < 2 || blades.size() > 5 || !imu_q.coeffs().allFinite() ||
    !std::isfinite(imu_q.squaredNorm()) || imu_q.squaredNorm() <= 1e-12 ||
    !valid_rotation(calibration_.R_camera2gimbal) ||
    !valid_rotation(calibration_.R_gimbal2imubody, true) || !calibration_.t_camera2gimbal_m.allFinite()) {
    return std::nullopt;
  }

  std::vector<const RefinedObservation *> candidates;
  for (const auto & blade : blades) {
    if (
      !blade.valid() || !blade.state || !std::isfinite(blade.detection_confidence) ||
      blade.detection_confidence < 0.0F || blade.detection_confidence > 1.0F ||
      !std::all_of(blade.points.begin(), blade.points.end(), finite_point) ||
      !armor_center_pixel(blade.points)) {
      continue;
    }
    if (!candidates.empty() && candidates.front()->state->kind != blade.state->kind) {
      return std::nullopt;
    }
    candidates.push_back(&blade);
  }
  if (candidates.size() < 2) return std::nullopt;

  try {
    const Eigen::Matrix3d R_gimbal2world = calibration_.R_gimbal2imubody.transpose() *
                                        imu_q.normalized().toRotationMatrix() *
                                        calibration_.R_gimbal2imubody;
    std::optional<rune_pose_tools::CameraPose> previous_camera_pose;
    if (previous_pose && previous_pose->source != PoseSource::UNKNOWN &&
        previous_pose->position_world_m.allFinite() &&
        previous_pose->orientation_world.coeffs().allFinite() &&
        std::isfinite(previous_pose->orientation_world.squaredNorm()) &&
        previous_pose->orientation_world.squaredNorm() > 1e-12) {
      const Eigen::Matrix3d rotation = calibration_.R_camera2gimbal.transpose() *
        R_gimbal2world.transpose() * previous_pose->orientation_world.normalized().toRotationMatrix();
      const Eigen::Vector3d translation = calibration_.R_camera2gimbal.transpose() *
        (R_gimbal2world.transpose() * previous_pose->position_world_m -
         calibration_.t_camera2gimbal_m);
      if (valid_rotation(rotation) && translation.allFinite()) {
        cv::Matx33d rotation_cv;
        for (int row = 0; row < 3; ++row) {
          for (int column = 0; column < 3; ++column) rotation_cv(row, column) = rotation(row, column);
        }
        rune_pose_tools::CameraPose camera_prior;
        cv::Rodrigues(rotation_cv, camera_prior.rvec);
        camera_prior.tvec = cv::Vec3d(translation.x(), translation.y(), translation.z());
        previous_camera_pose = camera_prior;
      }
    }
    const cv::Point2f r_center = fused_r_center(candidates);
    std::optional<RuneReconstruction> best;
    // Each observed blade can supply the initial plane. A poor single-blade estimate must
    // not prevent the other observations from establishing a consistent joint reconstruction.
    for (const auto * reference : candidates) {
      const auto reference_model = armor_model(0);
      std::vector<cv::Point3f> seed_model(reference_model.begin(), reference_model.end());
      std::vector<cv::Point2f> seed_image(reference->points.begin(), reference->points.begin() + 4);
      seed_model.emplace_back(0.0F, 0.0F, 0.0F);
      seed_image.push_back(r_center);
      const auto seed_prior =
        reference_pose_prior(previous_camera_pose, reference->points, calibration_);
      const auto seed = rune_pose_tools::solve_pnp_multi_anchor(
        seed_image, seed_model, calibration_.camera_matrix, calibration_.distortion_coeffs,
        seed_prior);
      if (
        !seed || reprojection_error(seed_image, seed_model, *seed, calibration_) >
                   MAX_REPROJECTION_ERROR_PX) {
        continue;
      }

      struct AssociatedBlade
      {
        const RefinedObservation * observation;
        int slot;
      };
      std::vector<AssociatedBlade> accepted{{reference, 0}};
      std::array<bool, 5> occupied{true, false, false, false, false};
      auto ordered_candidates = candidates;
      std::stable_sort(
        ordered_candidates.begin(), ordered_candidates.end(), [](const auto * left, const auto * right) {
          return left->detection_confidence > right->detection_confidence;
        });
      for (const auto * candidate : ordered_candidates) {
        if (candidate == reference) continue;
        const auto model_center =
          intersect_model_plane(*armor_center_pixel(candidate->points), *seed, calibration_);
        if (!model_center) continue;
        const double radius = std::hypot((*model_center)[1], (*model_center)[2]);
        if (std::abs(radius - BLADE_RADIUS_M) > MAX_RADIUS_RELATIVE_ERROR * BLADE_RADIUS_M) {
          continue;
        }
        // Slot matching occurs in the undistorted model plane. Image polar angles would be
        // distorted by perspective and can assign adjacent blades to the same physical slot.
        const double angle = std::atan2(-(*model_center)[1], (*model_center)[2]);
        const int nearest = static_cast<int>(std::lround(angle / SLOT_ANGLE_RAD));
        const int slot = (nearest % 5 + 5) % 5;
        if (
          occupied[slot] || std::abs(std::remainder(angle - nearest * SLOT_ANGLE_RAD, 2.0 * CV_PI)) >
                              MAX_SLOT_ANGLE_ERROR_RAD) {
          continue;
        }
        occupied[slot] = true;
        accepted.push_back({candidate, slot});
      }
      if (accepted.size() < 2) continue;

      const auto output_reference = std::max_element(
        accepted.begin(), accepted.end(), [](const auto & left, const auto & right) {
          const bool left_inactive = left.observation->state->activation == ActivationState::INACTIVE;
          const bool right_inactive = right.observation->state->activation == ActivationState::INACTIVE;
          if (left_inactive != right_inactive) return !left_inactive;
          return left.observation->detection_confidence < right.observation->detection_confidence;
        });
      const int reference_slot = output_reference->slot;
      for (auto & blade : accepted) blade.slot = (blade.slot - reference_slot + 5) % 5;
      std::sort(accepted.begin(), accepted.end(), [](const auto & left, const auto & right) {
        return left.slot < right.slot;
      });

      std::vector<cv::Point2f> image_points;
      std::vector<cv::Point3f> model_points;
      for (const auto & blade : accepted) {
        const auto model = armor_model(blade.slot);
        model_points.insert(model_points.end(), model.begin(), model.end());
        image_points.insert(
          image_points.end(), blade.observation->points.begin(), blade.observation->points.begin() + 4);
      }
      // A common origin is one anchor, even if the detector returned the same R for every blade.
      model_points.emplace_back(0.0F, 0.0F, 0.0F);
      image_points.push_back(r_center);
      // accepted was sorted after selecting the frame-local reference, so slot zero is first.
      const auto joint_prior = reference_pose_prior(
        previous_camera_pose, accepted.front().observation->points, calibration_);
      auto camera_pose = rune_pose_tools::solve_pnp_multi_anchor(
        image_points, model_points, calibration_.camera_matrix, calibration_.distortion_coeffs,
        joint_prior);
      if (!camera_pose) continue;
      double error = reprojection_error(image_points, model_points, *camera_pose, calibration_);
      if (!std::isfinite(error) || error > MAX_REPROJECTION_ERROR_PX) continue;
      const double pnp_error = error;
      std::vector<rune_pose_tools::ChamferContour> contour_terms;
      for (const auto & blade : accepted) {
        const auto & feature = blade.observation->armor_contour;
        if (!feature.usable() || !std::isfinite(feature.support) || feature.support < 0.45F ||
            feature.contour.size() < 4) continue;
        const auto field = rune_pose_tools::make_distance_transform(
          {feature.contour}, blade.observation->roi);
        if (!field) continue;
        contour_terms.push_back({armor_boundary_model(blade.slot), *field});
      }
      std::optional<rune_pose_tools::ChamferRefinement> contour_refinement;
      if (!contour_terms.empty()) {
        const auto proposal = rune_pose_tools::refine_by_chamfer(
          contour_terms, calibration_, *camera_pose, 0.0);
        if (proposal) {
          const double anchor_error =
            reprojection_error(image_points, model_points, proposal->pose, calibration_);
          // Contour fitting must retain agreement with the independent network/R anchors.
          // Its residual must never be presented as the anchor reprojection error.
          if (std::isfinite(anchor_error) && anchor_error <= MAX_REPROJECTION_ERROR_PX &&
              anchor_error <= pnp_error + 1.0) {
            camera_pose = proposal->pose;
            error = anchor_error;
            contour_refinement = proposal;
          }
        }
      }

      cv::Matx33d rotation_cv;
      cv::Rodrigues(camera_pose->rvec, rotation_cv);
      Eigen::Matrix3d R_model2camera;
      for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
          R_model2camera(row, column) = rotation_cv(row, column);
        }
      }
      const Eigen::Vector3d t_model2camera(
        camera_pose->tvec[0], camera_pose->tvec[1], camera_pose->tvec[2]);
      const Eigen::Matrix3d R_model2world =
        R_gimbal2world * calibration_.R_camera2gimbal * R_model2camera;

      RuneReconstruction reconstruction;
      auto & pose = reconstruction.pose;
      pose.position_world_m = R_gimbal2world *
                              (calibration_.R_camera2gimbal * t_model2camera +
                               calibration_.t_camera2gimbal_m);
      pose.orientation_world = Eigen::Quaterniond(R_model2world).normalized();
      pose.plane_normal_world = R_model2world.col(0).normalized();
      pose.phase_zero_direction_world = R_model2world.col(2).normalized();
      if (previous_camera_pose && previous_pose->plane_normal_world.allFinite() &&
          previous_pose->plane_normal_world.norm() > 1e-6 &&
          pose.plane_normal_world.dot(previous_pose->plane_normal_world.normalized()) <
            std::cos(20.0 * CV_PI / 180.0)) {
        // The rune rotates within its plane. A short-lived prior does not justify a sudden
        // plane tilt, even when an ambiguous planar PnP branch has a small anchor residual.
        // Reject this measurement; the caller expires the prior and reacquires after a gap.
        continue;
      }
      pose.blade_radius_m = BLADE_RADIUS_M;
      pose.reprojection_error_px = error;
      pose.pnp_reprojection_error_px = pnp_error;
      pose.used_temporal_prior = camera_pose->used_temporal_prior;
      pose.source = contour_refinement ? PoseSource::CHAMFER_REFINED : PoseSource::PNP;
      if (contour_refinement) {
        pose.contour_error_px = contour_refinement->contour_error_px;
        pose.contour_sample_count = contour_refinement->sample_count;
      }
      double confidence = 0.0;
      for (const auto & blade : accepted) {
        const auto center = armor_center_model(blade.slot);
        ReconstructedBlade result;
        result.observation = *blade.observation;
        result.slot = static_cast<RuneSlot>(blade.slot + 1);
        result.armor_center_world_m =
          pose.position_world_m + R_model2world * Eigen::Vector3d(center.x, center.y, center.z);
        if (!result.armor_center_world_m.allFinite()) return std::nullopt;
        confidence += blade.observation->detection_confidence;
        reconstruction.blades.push_back(std::move(result));
      }
      pose.confidence = static_cast<float>(
        confidence / accepted.size() * std::max(0.0, 1.0 - error / MAX_REPROJECTION_ERROR_PX));
      if (
        !pose.position_world_m.allFinite() || !pose.orientation_world.coeffs().allFinite() ||
        !pose.plane_normal_world.allFinite() || !pose.phase_zero_direction_world.allFinite()) {
        return std::nullopt;
      }
      if (
        !best || reconstruction.blades.size() > best->blades.size() ||
        (reconstruction.blades.size() == best->blades.size() &&
         reconstruction.pose.reprojection_error_px < best->pose.reprojection_error_px)) {
        best = std::move(reconstruction);
      }
    }
    return best;
  } catch (const cv::Exception &) {
    return std::nullopt;
  }
}
}  // namespace auto_buff
