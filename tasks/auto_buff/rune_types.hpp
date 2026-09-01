#ifndef AUTO_BUFF__RUNE_TYPES_HPP
#define AUTO_BUFF__RUNE_TYPES_HPP

#include <Eigen/Geometry>

#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>

namespace auto_buff
{
constexpr std::size_t BUFF_POINT_COUNT = 5;
using BuffPoints = std::array<cv::Point2f, BUFF_POINT_COUNT>;

enum class RuneKind
{
  SMALL,
  BIG
};

enum class ActivationState
{
  INACTIVE,
  ACTIVATED
};

struct RuneState
{
  RuneKind kind = RuneKind::SMALL;
  ActivationState activation = ActivationState::INACTIVE;

  bool operator==(const RuneState & other) const
  {
    return kind == other.kind && activation == other.activation;
  }
};

// Detector-side class semantics. RuneState is resolved later with the authoritative gimbal mode.
enum class BuffActivation
{
  INACTIVE,
  SMALL_ACTIVATED,
  BIG_ACTIVATED,
  INVALID
};

struct BuffObservation
{
  BuffPoints points{};  // [top, right, bottom, left, R]
  BuffActivation activation = BuffActivation::INVALID;
  float confidence = 0.0F;
};

enum class TrackingQuality
{
  INVALID,
  DEGRADED,
  GOOD
};

enum class ObservationRejectReason
{
  NONE,
  IMAGE_EMPTY,
  IMAGE_FORMAT_UNSUPPORTED,
  CONFIDENCE_NONFINITE,
  CONFIDENCE_OUT_OF_RANGE,
  POINT_NONFINITE,
  POINT_OUT_OF_BOUNDS,
  POINT_TOO_CLOSE,
  ARMOR_NON_CONVEX_OR_WRONG_WINDING,
  ARMOR_AREA_TOO_SMALL,
  ARMOR_EDGE_TOO_SHORT,
  R_OVERLAPS_CORNER,
  R_ON_ARMOR_BOUNDARY,
  STATE_MODE_CONFLICT
};

enum class ContourRejectReason
{
  NONE,
  NOT_FOUND,
  AREA_MISMATCH,
  LOW_SUPPORT
};

struct ContourFeature
{
  std::vector<cv::Point> contour;  // Full-image coordinates.
  float support = 0.0F;
  ContourRejectReason reject_reason = ContourRejectReason::NOT_FOUND;

  bool usable() const
  {
    return reject_reason == ContourRejectReason::NONE && !contour.empty();
  }
};

struct RefinedObservation
{
  std::optional<RuneState> state;
  BuffPoints points{};
  float detection_confidence = 0.0F;
  cv::Rect roi;
  std::vector<cv::Point2f> armor_polygon;
  ContourFeature armor_contour;
  ContourFeature r_mark_contour;
  ContourFeature light_arm_contour;
  TrackingQuality quality = TrackingQuality::INVALID;
  ObservationRejectReason reject_reason = ObservationRejectReason::NONE;
  float geometry_score = 0.0F;

  bool valid() const { return quality != TrackingQuality::INVALID; }
};

enum class PoseSource
{
  UNKNOWN,
  PNP,
  CHAMFER_REFINED
};

struct RunePose
{
  Eigen::Vector3d position_world_m = Eigen::Vector3d::Zero();
  Eigen::Quaterniond orientation_world = Eigen::Quaterniond::Identity();
  Eigen::Vector3d plane_normal_world = Eigen::Vector3d::UnitX();
  double reprojection_error_px = std::numeric_limits<double>::infinity();
  float confidence = 0.0F;
  PoseSource source = PoseSource::UNKNOWN;
};

enum class RotationDirection
{
  UNKNOWN = 0,
  CLOCKWISE = -1,
  COUNTER_CLOCKWISE = 1
};

struct MotionState
{
  double continuous_phase_rad = 0.0;
  double angular_velocity_rad_s = 0.0;
  RotationDirection direction = RotationDirection::UNKNOWN;
  std::chrono::steady_clock::time_point reference_timestamp{};
  float confidence = 0.0F;
  bool converged = false;
};

struct RuneTarget
{
  std::uint64_t id = 0;
  RefinedObservation observation;
  std::optional<RunePose> pose;
  std::optional<MotionState> motion;
  TrackingQuality quality = TrackingQuality::INVALID;
};

inline const char * to_string(TrackingQuality quality)
{
  switch (quality) {
    case TrackingQuality::INVALID:
      return "INVALID";
    case TrackingQuality::DEGRADED:
      return "DEGRADED";
    case TrackingQuality::GOOD:
      return "GOOD";
  }
  return "UNKNOWN";
}

inline const char * to_string(ObservationRejectReason reason)
{
  switch (reason) {
    case ObservationRejectReason::NONE:
      return "NONE";
    case ObservationRejectReason::IMAGE_EMPTY:
      return "IMAGE_EMPTY";
    case ObservationRejectReason::IMAGE_FORMAT_UNSUPPORTED:
      return "IMAGE_FORMAT_UNSUPPORTED";
    case ObservationRejectReason::CONFIDENCE_NONFINITE:
      return "CONFIDENCE_NONFINITE";
    case ObservationRejectReason::CONFIDENCE_OUT_OF_RANGE:
      return "CONFIDENCE_OUT_OF_RANGE";
    case ObservationRejectReason::POINT_NONFINITE:
      return "POINT_NONFINITE";
    case ObservationRejectReason::POINT_OUT_OF_BOUNDS:
      return "POINT_OUT_OF_BOUNDS";
    case ObservationRejectReason::POINT_TOO_CLOSE:
      return "POINT_TOO_CLOSE";
    case ObservationRejectReason::ARMOR_NON_CONVEX_OR_WRONG_WINDING:
      return "ARMOR_NON_CONVEX_OR_WRONG_WINDING";
    case ObservationRejectReason::ARMOR_AREA_TOO_SMALL:
      return "ARMOR_AREA_TOO_SMALL";
    case ObservationRejectReason::ARMOR_EDGE_TOO_SHORT:
      return "ARMOR_EDGE_TOO_SHORT";
    case ObservationRejectReason::R_OVERLAPS_CORNER:
      return "R_OVERLAPS_CORNER";
    case ObservationRejectReason::R_ON_ARMOR_BOUNDARY:
      return "R_ON_ARMOR_BOUNDARY";
    case ObservationRejectReason::STATE_MODE_CONFLICT:
      return "STATE_MODE_CONFLICT";
  }
  return "UNKNOWN";
}
}  // namespace auto_buff

#endif  // AUTO_BUFF__RUNE_TYPES_HPP
