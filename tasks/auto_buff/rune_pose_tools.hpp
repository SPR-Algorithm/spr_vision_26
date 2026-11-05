#ifndef AUTO_BUFF__RUNE_POSE_TOOLS_HPP
#define AUTO_BUFF__RUNE_POSE_TOOLS_HPP

#include <Eigen/Geometry>

#include <limits>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>

namespace auto_buff::rune_pose_tools
{
struct CameraCalibration
{
  cv::Mat camera_matrix;      // 3 x 3, CV_64FC1.
  cv::Mat distortion_coeffs;  // 1 x N, CV_64FC1, OpenCV distortion ordering.
  Eigen::Matrix3d R_camera2gimbal = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_camera2gimbal_m = Eigen::Vector3d::Zero();
  Eigen::Matrix3d R_gimbal2imubody = Eigen::Matrix3d::Identity();

  // p_gimbal = R_camera2gimbal * p_camera + t_camera2gimbal_m.
  // Preserve the legacy IMU convention:
  // R_gimbal2world = R_gimbal2imubody.transpose() * R_imu * R_gimbal2imubody.
};

struct CameraPose
{
  // OpenCV model -> camera transform; translation is in metres.
  cv::Vec3d rvec{};
  cv::Vec3d tvec{};
  // True only when the accepted anchor solution came from LM initialized with a temporal prior.
  bool used_temporal_prior = false;
};

struct DistanceField
{
  cv::Mat distances_px;  // CV_32FC1, roi.size(); distance to contour boundaries in pixels.
  cv::Rect roi;         // Full-image coordinates; subtract roi.tl() before sampling.
};

struct ChamferRefinement
{
  CameraPose pose;
  // These are mean model-to-contour distances, not anchor reprojection errors.
  double contour_error_px = std::numeric_limits<double>::infinity();
  double prior_contour_error_px = std::numeric_limits<double>::infinity();
  std::size_t sample_count = 0;
};

struct ChamferContour
{
  // A separate field per observed blade prevents matching another blade's contour.
  std::vector<cv::Point3f> model_points;
  DistanceField distance_field;
};

// Corresponding anchors use full-image pixels and model coordinates in metres. An optional
// pose initializes a separate LM candidate; it can select a planar branch only within 0.5 px
// anchor RMS of the best independent SQPnP/LM candidate. No prior is a hard pose constraint.
std::optional<CameraPose> solve_pnp_multi_anchor(
  const std::vector<cv::Point2f> & anchors_2d, const std::vector<cv::Point3f> & anchors_3d,
  const cv::Mat & camera_matrix, const cv::Mat & distortion_coeffs,
  const std::optional<CameraPose> & initial_prior = std::nullopt);

// Input contours use full-image pixels. Segments are clipped to a nonnegative full-image ROI.
// Empty/off-ROI contours and invalid/oversized ROIs have no usable distance field.
std::optional<DistanceField> make_distance_transform(
  const std::vector<std::vector<cv::Point>> & contours, const cv::Rect & roi);

std::optional<ChamferRefinement> refine_by_chamfer(
  const std::vector<cv::Point3f> & model_points, const DistanceField & distance_field,
  const CameraCalibration & calibration, const CameraPose & prior_pose,
  double plane_y_constraint_weight);

// Fits a bounded SE(3) correction with a pose prior and an ROI escape penalty. Only zero
// plane_y_constraint_weight is supported: no world-axis plane constraint is defined yet.
// Success requires positive depth and a measured improvement in the actual contour distances.
std::optional<ChamferRefinement> refine_by_chamfer(
  const std::vector<ChamferContour> & contours, const CameraCalibration & calibration,
  const CameraPose & prior_pose, double plane_y_constraint_weight);
}  // namespace auto_buff::rune_pose_tools

#endif  // AUTO_BUFF__RUNE_POSE_TOOLS_HPP
