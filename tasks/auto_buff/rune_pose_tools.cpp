#include "rune_pose_tools.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <ceres/ceres.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace auto_buff::rune_pose_tools
{
namespace
{
bool floating_matrix(const cv::Mat & matrix)
{
  return matrix.channels() == 1 && (matrix.depth() == CV_32F || matrix.depth() == CV_64F) &&
         cv::checkRange(matrix);
}

// Planar geometry is expected here; collinear or collapsed correspondences cannot fix a pose.
bool has_two_dimensions(const cv::Mat & points)
{
  cv::Mat mean;
  cv::reduce(points, mean, 0, cv::REDUCE_AVG);
  const cv::Mat centered = points - cv::repeat(mean, points.rows, 1);
  cv::Mat singular_values;
  cv::SVD::compute(centered, singular_values);
  return singular_values.total() >= 2 && singular_values.at<double>(0) > 1e-9 &&
         singular_values.at<double>(1) > singular_values.at<double>(0) * 1e-6;
}

std::optional<double> pose_error(
  const CameraPose & pose, const std::vector<cv::Point2f> & image_points,
  const std::vector<cv::Point3f> & model_points, const cv::Mat & camera_matrix,
  const cv::Mat & distortion_coeffs)
{
  for (int i = 0; i < 3; ++i) {
    if (!std::isfinite(pose.rvec[i]) || !std::isfinite(pose.tvec[i])) return std::nullopt;
  }
  cv::Matx33d rotation;
  cv::Rodrigues(pose.rvec, rotation);
  for (const auto & point : model_points) {
    const cv::Vec3d camera_point = rotation * cv::Vec3d(point.x, point.y, point.z) + pose.tvec;
    if (!std::isfinite(camera_point[2]) || camera_point[2] <= 1e-9) return std::nullopt;
  }

  std::vector<cv::Point2f> projected;
  cv::projectPoints(
    model_points, pose.rvec, pose.tvec, camera_matrix, distortion_coeffs, projected);
  double squared_error = 0.0;
  for (std::size_t i = 0; i < projected.size(); ++i) {
    if (!std::isfinite(projected[i].x) || !std::isfinite(projected[i].y)) return std::nullopt;
    const auto delta = projected[i] - image_points[i];
    squared_error +=
      static_cast<double>(delta.x) * delta.x + static_cast<double>(delta.y) * delta.y;
  }
  if (!std::isfinite(squared_error)) return std::nullopt;
  return squared_error;
}

bool valid_roi(const cv::Rect & roi)
{
  // Besides rejecting overflow, bound allocations from externally supplied ROIs.
  return roi.x >= 0 && roi.y >= 0 && roi.width >= 2 && roi.height >= 2 &&
         roi.width <= 32768 && roi.height <= 32768 &&
         static_cast<std::int64_t>(roi.x) + roi.width <= std::numeric_limits<int>::max() &&
         static_cast<std::int64_t>(roi.y) + roi.height <= std::numeric_limits<int>::max() &&
         static_cast<std::int64_t>(roi.width) * roi.height <= 16 * 1024 * 1024;
}

bool valid_calibration(const CameraCalibration & calibration)
{
  const auto & matrix = calibration.camera_matrix;
  const auto & distortion = calibration.distortion_coeffs;
  if (matrix.rows != 3 || matrix.cols != 3 || !floating_matrix(matrix)) return false;
  cv::Mat intrinsics;
  matrix.convertTo(intrinsics, CV_64F);
  if (
    intrinsics.at<double>(0, 0) <= 0.0 || intrinsics.at<double>(1, 1) <= 0.0 ||
    std::abs(intrinsics.at<double>(0, 1)) > 1e-9 ||
    std::abs(intrinsics.at<double>(1, 0)) > 1e-9 ||
    std::abs(intrinsics.at<double>(2, 0)) > 1e-9 ||
    std::abs(intrinsics.at<double>(2, 1)) > 1e-9 ||
    std::abs(intrinsics.at<double>(2, 2) - 1.0) > 1e-9) return false;
  const auto count = distortion.total();
  return distortion.empty() ||
         (floating_matrix(distortion) && (distortion.rows == 1 || distortion.cols == 1) &&
          (count == 4 || count == 5 || count == 8 || count == 12 || count == 14));
}

CameraPose corrected_pose(const CameraPose & prior, const double * increment)
{
  const Eigen::Vector3d original(prior.rvec[0], prior.rvec[1], prior.rvec[2]);
  const Eigen::Vector3d delta(increment[0], increment[1], increment[2]);
  const auto quaternion = [](const Eigen::Vector3d & angle_axis) {
    const double angle = angle_axis.norm();
    return angle > 0.0 ? Eigen::Quaterniond(Eigen::AngleAxisd(angle, angle_axis / angle)) :
                         Eigen::Quaterniond::Identity();
  };
  // Avoid the small-angle threshold of matrix-to-Rodrigues conversion, which would flatten
  // finite-difference derivatives around an identity rotation.
  const Eigen::AngleAxisd rotation((quaternion(delta) * quaternion(original)).normalized());
  const Eigen::Vector3d rvec = rotation.angle() * rotation.axis();
  CameraPose result;
  result.rvec = cv::Vec3d(rvec.x(), rvec.y(), rvec.z());
  result.tvec = prior.tvec + cv::Vec3d(increment[3], increment[4], increment[5]);
  result.used_temporal_prior = prior.used_temporal_prior;
  return result;
}

struct PreparedContour
{
  std::vector<cv::Point3d> model_points;
  const DistanceField * field;
};

struct ContourEvaluation
{
  double mean = 0.0;
  double squared_mean = 0.0;
  bool inside_roi = true;
};

// Double-precision projections are needed for finite-difference derivatives near zero pose
// corrections. Float projections would quantize those derivatives to zero.
bool contour_residuals(
  const std::vector<PreparedContour> & contours, const CameraCalibration & calibration,
  const CameraPose & pose, double scale, double * residuals, ContourEvaluation & evaluation)
{
  cv::Matx33d rotation;
  cv::Rodrigues(pose.rvec, rotation);
  std::size_t offset = 0, count = 0;
  for (const auto & contour : contours) {
    for (const auto & point : contour.model_points) {
      const cv::Vec3d camera_point = rotation * cv::Vec3d(point.x, point.y, point.z) + pose.tvec;
      if (!std::isfinite(camera_point[2]) || camera_point[2] <= 1e-6) return false;
    }
    std::vector<cv::Point2d> projected;
    cv::projectPoints(
      contour.model_points, pose.rvec, pose.tvec, calibration.camera_matrix,
      calibration.distortion_coeffs, projected);
    const auto & field = *contour.field;
    for (const auto & point : projected) {
      if (!std::isfinite(point.x) || !std::isfinite(point.y)) return false;
      const double local_x = point.x - field.roi.x;
      const double local_y = point.y - field.roi.y;
      const double x = std::clamp(local_x, 0.0, static_cast<double>(field.roi.width - 1));
      const double y = std::clamp(local_y, 0.0, static_cast<double>(field.roi.height - 1));
      const int x0 = static_cast<int>(std::floor(x));
      const int y0 = static_cast<int>(std::floor(y));
      const int x1 = std::min(x0 + 1, field.roi.width - 1);
      const int y1 = std::min(y0 + 1, field.roi.height - 1);
      const double dx = x - x0, dy = y - y0;
      const double distance =
        (1.0 - dy) * ((1.0 - dx) * field.distances_px.at<float>(y0, x0) +
                      dx * field.distances_px.at<float>(y0, x1)) +
        dy * ((1.0 - dx) * field.distances_px.at<float>(y1, x0) +
              dx * field.distances_px.at<float>(y1, x1));
      if (!std::isfinite(distance)) return false;
      const double outside_x = local_x - x, outside_y = local_y - y;
      evaluation.inside_roi = evaluation.inside_roi &&
                              std::abs(outside_x) <= 1e-6 && std::abs(outside_y) <= 1e-6;
      evaluation.mean += distance;
      evaluation.squared_mean += distance * distance;
      if (residuals) {
        residuals[offset++] = scale * distance;
        // Clamping alone would make the distance flat outside the ROI and permit escape.
        residuals[offset++] = 2.0 * scale * outside_x;
        residuals[offset++] = 2.0 * scale * outside_y;
      }
      ++count;
    }
  }
  evaluation.mean /= count;
  evaluation.squared_mean /= count;
  return true;
}

struct ChamferCost
{
  const std::vector<PreparedContour> & contours;
  const CameraCalibration & calibration;
  CameraPose prior;
  std::size_t count;
  double translation_sigma;

  bool operator()(const double * increment, double * residuals) const
  {
    for (int i = 0; i < 6; ++i) if (!std::isfinite(increment[i])) return false;
    try {
      ContourEvaluation ignored;
      if (!contour_residuals(
            contours, calibration, corrected_pose(prior, increment),
            std::sqrt(32.0 / count), residuals, ignored)) return false;
      // A finite correction prior constrains the under-observed planar pose even when many
      // contour samples are supplied. Its strength is independent of sampling density.
      for (int i = 0; i < 3; ++i) {
        residuals[3 * count + i] = increment[i] / 0.05;
        residuals[3 * count + 3 + i] = increment[3 + i] / translation_sigma;
      }
      return true;
    } catch (const cv::Exception &) {
      return false;
    }
  }
};
}  // namespace

std::optional<CameraPose> solve_pnp_multi_anchor(
  const std::vector<cv::Point2f> & anchors_2d,
  const std::vector<cv::Point3f> & anchors_3d, const cv::Mat & camera_matrix,
  const cv::Mat & distortion_coeffs, const std::optional<CameraPose> & initial_prior)
{
  if (anchors_2d.size() != anchors_3d.size() || anchors_2d.size() < 4) return std::nullopt;
  if (
    camera_matrix.rows != 3 || camera_matrix.cols != 3 || !floating_matrix(camera_matrix)) {
    return std::nullopt;
  }
  const auto distortion_count = distortion_coeffs.total();
  if (
    !distortion_coeffs.empty() &&
    (!floating_matrix(distortion_coeffs) ||
     (distortion_coeffs.rows != 1 && distortion_coeffs.cols != 1) ||
     (distortion_count != 4 && distortion_count != 5 && distortion_count != 8 &&
      distortion_count != 12 && distortion_count != 14))) {
    return std::nullopt;
  }

  try {
    cv::Mat intrinsics, distortion;
    camera_matrix.convertTo(intrinsics, CV_64F);
    distortion_coeffs.convertTo(distortion, CV_64F);
    if (
      intrinsics.at<double>(0, 0) <= 0.0 || intrinsics.at<double>(1, 1) <= 0.0 ||
      std::abs(intrinsics.at<double>(0, 1)) > 1e-9 ||
      std::abs(intrinsics.at<double>(1, 0)) > 1e-9 ||
      std::abs(intrinsics.at<double>(2, 0)) > 1e-9 ||
      std::abs(intrinsics.at<double>(2, 1)) > 1e-9 ||
      std::abs(intrinsics.at<double>(2, 2) - 1.0) > 1e-9) {
      return std::nullopt;
    }

    cv::Mat image_points(static_cast<int>(anchors_2d.size()), 2, CV_64F);
    cv::Mat model_points(static_cast<int>(anchors_3d.size()), 3, CV_64F);
    for (std::size_t i = 0; i < anchors_2d.size(); ++i) {
      const auto & image = anchors_2d[i];
      const auto & model = anchors_3d[i];
      if (
        !std::isfinite(image.x) || !std::isfinite(image.y) || !std::isfinite(model.x) ||
        !std::isfinite(model.y) || !std::isfinite(model.z)) {
        return std::nullopt;
      }
      image_points.at<double>(static_cast<int>(i), 0) = image.x;
      image_points.at<double>(static_cast<int>(i), 1) = image.y;
      model_points.at<double>(static_cast<int>(i), 0) = model.x;
      model_points.at<double>(static_cast<int>(i), 1) = model.y;
      model_points.at<double>(static_cast<int>(i), 2) = model.z;
    }
    if (!has_two_dimensions(image_points) || !has_two_dimensions(model_points)) return std::nullopt;

    std::optional<CameraPose> best;
    double best_error = std::numeric_limits<double>::infinity();
    try {
      CameraPose pose;
      if (cv::solvePnP(
            anchors_3d, anchors_2d, intrinsics, distortion, pose.rvec, pose.tvec, false,
            cv::SOLVEPNP_SQPNP)) {
        const auto initial_error = pose_error(pose, anchors_2d, anchors_3d, intrinsics, distortion);
        if (initial_error) {
          best = pose;
          best_error = *initial_error;
          // LM remains optional: an unusable improvement cannot discard a valid SQPnP result.
          CameraPose refined = pose;
          try {
            cv::solvePnPRefineLM(
              anchors_3d, anchors_2d, intrinsics, distortion, refined.rvec, refined.tvec);
            const auto refined_error =
              pose_error(refined, anchors_2d, anchors_3d, intrinsics, distortion);
            if (refined_error && *refined_error <= best_error) {
              best = refined;
              best_error = *refined_error;
            }
          } catch (const cv::Exception &) {
          }
        }
      }
    } catch (const cv::Exception &) {
      // A usable prior-initialized solve can still recover a failed independent estimate.
    }
    if (initial_prior) {
      const auto prior_error =
        pose_error(*initial_prior, anchors_2d, anchors_3d, intrinsics, distortion);
      if (prior_error) {
        CameraPose refined = *initial_prior;
        try {
          // This minimizes only current-frame anchor residuals. The previous translation and
          // rotation are an initialization, not a regularizer or a fixed world-center constraint.
          cv::solvePnPRefineLM(
            anchors_3d, anchors_2d, intrinsics, distortion, refined.rvec, refined.tvec);
          const auto refined_error =
            pose_error(refined, anchors_2d, anchors_3d, intrinsics, distortion);
          if (
            refined_error && *refined_error <= *prior_error + 1e-9 &&
            std::sqrt(*refined_error / anchors_2d.size()) <=
              std::sqrt(best_error / anchors_2d.size()) + 0.5) {
            refined.used_temporal_prior = true;
            return refined;
          }
        } catch (const cv::Exception &) {
          // Reject the temporal branch rather than letting it suppress independent PnP.
        }
      }
    }
    return best;
  } catch (const cv::Exception &) {
    return std::nullopt;
  }
}

std::optional<DistanceField> make_distance_transform(
  const std::vector<std::vector<cv::Point>> & contours, const cv::Rect & roi)
{
  if (!valid_roi(roi) || contours.empty()) return std::nullopt;
  try {
    cv::Mat boundary(roi.size(), CV_8UC1, cv::Scalar(255));
    bool drew_boundary = false;
    for (const auto & contour : contours) {
      if (contour.size() < 2) continue;
      for (std::size_t i = 0; i < contour.size(); ++i) {
        const auto & first = contour[i];
        const auto & second = contour[(i + 1) % contour.size()];
        cv::Point2l a(static_cast<std::int64_t>(first.x) - roi.x,
                      static_cast<std::int64_t>(first.y) - roi.y);
        cv::Point2l b(static_cast<std::int64_t>(second.x) - roi.x,
                      static_cast<std::int64_t>(second.y) - roi.y);
        if (!cv::clipLine(cv::Size2l(roi.width, roi.height), a, b)) continue;
        cv::line(boundary, cv::Point(a), cv::Point(b), cv::Scalar(0), 1, cv::LINE_8);
        drew_boundary = true;
      }
    }
    if (!drew_boundary) return std::nullopt;
    DistanceField result;
    result.roi = roi;
    cv::distanceTransform(boundary, result.distances_px, cv::DIST_L2, cv::DIST_MASK_PRECISE);
    if (!cv::checkRange(result.distances_px, true, nullptr, 0.0)) return std::nullopt;
    return result;
  } catch (const cv::Exception &) {
    return std::nullopt;
  }
}

std::optional<ChamferRefinement> refine_by_chamfer(
  const std::vector<cv::Point3f> & model_points, const DistanceField & distance_field,
  const CameraCalibration & calibration, const CameraPose & prior_pose,
  double plane_y_constraint_weight)
{
  return refine_by_chamfer(
    std::vector<ChamferContour>{{model_points, distance_field}}, calibration, prior_pose,
    plane_y_constraint_weight);
}

std::optional<ChamferRefinement> refine_by_chamfer(
  const std::vector<ChamferContour> & contours, const CameraCalibration & calibration,
  const CameraPose & prior_pose, double plane_y_constraint_weight)
{
  // The API has no declared world frame for a Y plane; do not silently impose one.
  if (
    !std::isfinite(plane_y_constraint_weight) || plane_y_constraint_weight != 0.0 ||
    contours.empty() || contours.size() > 5 || !valid_calibration(calibration)) return std::nullopt;
  for (int i = 0; i < 3; ++i) {
    if (!std::isfinite(prior_pose.rvec[i]) || !std::isfinite(prior_pose.tvec[i])) return std::nullopt;
  }
  try {
    std::vector<PreparedContour> prepared;
    std::size_t count = 0;
    for (const auto & contour : contours) {
      const auto & field = contour.distance_field;
      if (
        contour.model_points.size() < 4 || contour.model_points.size() > 4096 ||
        !valid_roi(field.roi) || field.distances_px.type() != CV_32FC1 ||
        field.distances_px.size() != field.roi.size() ||
        !cv::checkRange(field.distances_px, true, nullptr, 0.0)) return std::nullopt;
      double minimum;
      cv::minMaxLoc(field.distances_px, &minimum);
      if (minimum > 1e-6) return std::nullopt;
      PreparedContour item;
      item.field = &field;
      cv::Mat points(static_cast<int>(contour.model_points.size()), 3, CV_64F);
      for (std::size_t i = 0; i < contour.model_points.size(); ++i) {
        const auto & p = contour.model_points[i];
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) return std::nullopt;
        item.model_points.emplace_back(p.x, p.y, p.z);
        points.at<double>(static_cast<int>(i), 0) = p.x;
        points.at<double>(static_cast<int>(i), 1) = p.y;
        points.at<double>(static_cast<int>(i), 2) = p.z;
      }
      if (!has_two_dimensions(points)) return std::nullopt;
      count += item.model_points.size();
      prepared.push_back(std::move(item));
    }
    ContourEvaluation before;
    if (!contour_residuals(prepared, calibration, prior_pose, 1.0, nullptr, before)) return std::nullopt;
    // A crop that does not contain the prior model is not evidence for fitting its whole boundary.
    if (!before.inside_roi || before.mean <= 0.05) return std::nullopt;
    const double translation_sigma = std::clamp(0.0075 * cv::norm(prior_pose.tvec), 0.02, 0.06);
    const double maximum_translation = std::clamp(0.02 * cv::norm(prior_pose.tvec), 0.02, 0.12);
    constexpr double maximum_rotation = 8.0 * CV_PI / 180.0;
    double increment[6]{};
    ceres::Problem problem;
    auto * cost = new ceres::NumericDiffCostFunction<ChamferCost, ceres::CENTRAL, ceres::DYNAMIC, 6>(
      new ChamferCost{prepared, calibration, prior_pose, count, translation_sigma},
      ceres::TAKE_OWNERSHIP, static_cast<int>(3 * count + 6));
    problem.AddResidualBlock(cost, nullptr, increment);
    for (int i = 0; i < 6; ++i) {
      const double bound = (i < 3 ? maximum_rotation : maximum_translation) / std::sqrt(3.0);
      problem.SetParameterLowerBound(increment, i, -bound);
      problem.SetParameterUpperBound(increment, i, bound);
    }
    ceres::Solver::Options options;
    options.max_num_iterations = 30;
    options.linear_solver_type = ceres::DENSE_QR;
    options.num_threads = 1;
    options.logging_type = ceres::SILENT;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (!summary.IsSolutionUsable()) return std::nullopt;
    ChamferRefinement result;
    result.pose = corrected_pose(prior_pose, increment);
    ContourEvaluation after;
    if (!contour_residuals(prepared, calibration, result.pose, 1.0, nullptr, after)) return std::nullopt;
    // Gate on the actual distance residual, separately from priors and escape penalties.
    if (
      !after.inside_roi || !std::isfinite(after.mean) ||
      // Improvement alone is insufficient: an edge model that remains far from the real
      // contour must not acquire CHAMFER_REFINED quality (or downstream firing eligibility).
      after.mean > 3.0 || std::sqrt(after.squared_mean) > 5.0 ||
      after.mean > before.mean - std::max(0.05, 0.01 * before.mean) ||
      after.squared_mean >= before.squared_mean ||
      cv::norm(cv::Vec3d(increment[0], increment[1], increment[2])) > maximum_rotation + 1e-9 ||
      cv::norm(result.pose.tvec - prior_pose.tvec) > maximum_translation + 1e-9) return std::nullopt;
    result.contour_error_px = after.mean;
    result.prior_contour_error_px = before.mean;
    result.sample_count = count;
    return result;
  } catch (const cv::Exception &) {
    return std::nullopt;
  }
}
}  // namespace auto_buff::rune_pose_tools
