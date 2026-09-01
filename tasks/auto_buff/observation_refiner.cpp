#include "observation_refiner.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

namespace auto_buff
{
namespace
{
constexpr float EPSILON = 1e-4F;

template<typename T>
void read_if_present(const YAML::Node & node, const char * key, T & value)
{
  if (node[key]) value = node[key].as<T>();
}

float cross(const cv::Point2f & a, const cv::Point2f & b, const cv::Point2f & c)
{
  const cv::Point2f ab = b - a;
  const cv::Point2f bc = c - b;
  return ab.x * bc.y - ab.y * bc.x;
}

float point_to_segment_distance(
  const cv::Point2f & point, const cv::Point2f & start, const cv::Point2f & end)
{
  const cv::Point2f segment = end - start;
  const float length_squared = segment.dot(segment);
  if (length_squared <= EPSILON) return static_cast<float>(cv::norm(point - start));
  const float projection = std::clamp((point - start).dot(segment) / length_squared, 0.0F, 1.0F);
  return static_cast<float>(cv::norm(point - (start + projection * segment)));
}

float contour_solidity(const std::vector<cv::Point> & contour)
{
  if (contour.size() < 3) return 0.0F;
  std::vector<cv::Point> hull;
  cv::convexHull(contour, hull);
  const double hull_area = std::abs(cv::contourArea(hull));
  if (hull_area <= EPSILON) return 0.0F;
  return static_cast<float>(std::clamp(std::abs(cv::contourArea(contour)) / hull_area, 0.0, 1.0));
}

float line_support(
  const std::vector<cv::Point> & contour, const cv::Point2f & start, const cv::Point2f & end,
  int samples)
{
  int inside = 0;
  for (int i = 1; i < samples; ++i) {
    const float ratio = static_cast<float>(i) / static_cast<float>(samples);
    if (cv::pointPolygonTest(contour, start + ratio * (end - start), false) > 0.0) ++inside;
  }
  return static_cast<float>(inside) / static_cast<float>(std::max(1, samples - 1));
}

cv::Rect make_roi(
  const BuffPoints & points, const cv::Size & image_size, const ObservationRefinerConfig & config)
{
  const std::vector<cv::Point2f> all_points(points.begin(), points.end());
  const cv::Rect bounds = cv::boundingRect(all_points);
  const int ratio_margin = static_cast<int>(
    std::ceil(config.roi_margin_ratio * static_cast<float>(std::max(bounds.width, bounds.height))));
  const int margin = std::max(config.roi_margin_px, ratio_margin);
  const cv::Rect expanded(
    bounds.x - margin, bounds.y - margin, bounds.width + 2 * margin, bounds.height + 2 * margin);
  return expanded & cv::Rect(0, 0, image_size.width, image_size.height);
}

cv::Mat make_binary_mask(
  const cv::Mat & image, const cv::Rect & roi, const ObservationRefinerConfig & config)
{
  const cv::Mat view = image(roi);
  cv::Mat signal;
  if (view.channels() == 1) {
    signal = view.clone();
  } else {
    cv::Mat bgr;
    if (view.channels() == 4) {
      cv::cvtColor(view, bgr, cv::COLOR_BGRA2BGR);
    } else {
      bgr = view;
    }
    std::vector<cv::Mat> channels;
    cv::split(bgr, channels);
    cv::absdiff(channels[2], channels[0], signal);
  }

  if (config.gaussian_kernel_size > 1) {
    cv::GaussianBlur(
      signal, signal,
      cv::Size(config.gaussian_kernel_size, config.gaussian_kernel_size), 0.0);
  }

  cv::Mat mask;
  cv::threshold(signal, mask, config.binary_threshold, 255, cv::THRESH_BINARY);
  if (config.morphology_kernel_size > 1) {
    const cv::Mat kernel = cv::getStructuringElement(
      cv::MORPH_ELLIPSE,
      cv::Size(config.morphology_kernel_size, config.morphology_kernel_size));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
  }
  return mask;
}

void set_feature(
  ContourFeature & feature, const std::vector<cv::Point> & contour, float support)
{
  feature.contour = contour;
  feature.support = std::clamp(support, 0.0F, 1.0F);
  feature.reject_reason = ContourRejectReason::NONE;
}

ObservationRejectReason validate_observation(
  const cv::Mat & image, const BuffObservation & observation,
  const ObservationRefinerConfig & config, float & geometry_score)
{
  if (image.empty()) return ObservationRejectReason::IMAGE_EMPTY;
  if (image.depth() != CV_8U || (image.channels() != 1 && image.channels() != 3 && image.channels() != 4)) {
    return ObservationRejectReason::IMAGE_FORMAT_UNSUPPORTED;
  }
  if (!std::isfinite(observation.confidence)) {
    return ObservationRejectReason::CONFIDENCE_NONFINITE;
  }
  if (observation.confidence < 0.0F || observation.confidence > 1.0F) {
    return ObservationRejectReason::CONFIDENCE_OUT_OF_RANGE;
  }

  for (const auto & point : observation.points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
      return ObservationRejectReason::POINT_NONFINITE;
    }
    if (point.x < 0.0F || point.y < 0.0F || point.x >= image.cols || point.y >= image.rows) {
      return ObservationRejectReason::POINT_OUT_OF_BOUNDS;
    }
  }

  float minimum_separation = std::numeric_limits<float>::infinity();
  for (std::size_t i = 0; i < observation.points.size(); ++i) {
    for (std::size_t j = i + 1; j < observation.points.size(); ++j) {
      const float distance = static_cast<float>(cv::norm(observation.points[i] - observation.points[j]));
      minimum_separation = std::min(minimum_separation, distance);
      if (distance < config.min_point_distance_px) {
        if (i == 4 || j == 4) return ObservationRejectReason::R_OVERLAPS_CORNER;
        return ObservationRejectReason::POINT_TOO_CLOSE;
      }
    }
  }

  for (std::size_t i = 0; i < 4; ++i) {
    const float turn = cross(
      observation.points[i], observation.points[(i + 1) % 4], observation.points[(i + 2) % 4]);
    // [top, right, bottom, left] is clockwise in image coordinates (positive y is down).
    if (turn <= EPSILON) {
      return ObservationRejectReason::ARMOR_NON_CONVEX_OR_WRONG_WINDING;
    }
  }

  const std::vector<cv::Point2f> armor_points(
    observation.points.begin(), observation.points.begin() + 4);
  const float armor_area = static_cast<float>(std::abs(cv::contourArea(armor_points)));
  if (armor_area < config.min_armor_area_px2) {
    return ObservationRejectReason::ARMOR_AREA_TOO_SMALL;
  }

  float minimum_edge = std::numeric_limits<float>::infinity();
  for (std::size_t i = 0; i < 4; ++i) {
    minimum_edge = std::min(
      minimum_edge,
      static_cast<float>(cv::norm(observation.points[i] - observation.points[(i + 1) % 4])));
  }
  if (minimum_edge < config.min_armor_edge_px) {
    return ObservationRejectReason::ARMOR_EDGE_TOO_SHORT;
  }

  const cv::Point2f & r_mark = observation.points[4];
  for (std::size_t i = 0; i < 4; ++i) {
    if (
      point_to_segment_distance(r_mark, observation.points[i], observation.points[(i + 1) % 4]) <=
      config.r_boundary_tolerance_px) {
      return ObservationRejectReason::R_ON_ARMOR_BOUNDARY;
    }
  }

  const float area_score = std::clamp(
    armor_area / std::max(config.min_armor_area_px2 * 4.0F, EPSILON), 0.0F, 1.0F);
  const float edge_score = std::clamp(
    minimum_edge / std::max(config.min_armor_edge_px * 4.0F, EPSILON), 0.0F, 1.0F);
  const float separation_score = std::clamp(
    minimum_separation / std::max(config.min_point_distance_px * 4.0F, EPSILON), 0.0F, 1.0F);
  geometry_score = std::min({area_score, edge_score, separation_score});
  return ObservationRejectReason::NONE;
}

void extract_semantic_contours(
  const cv::Mat & image, RefinedObservation & result, const ObservationRefinerConfig & config)
{
  const cv::Mat mask = make_binary_mask(image, result.roi, config);
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(
    mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, result.roi.tl());

  const cv::Point2f armor_center =
    (result.points[0] + result.points[1] + result.points[2] + result.points[3]) * 0.25F;
  const cv::Point2f & r_mark = result.points[4];
  const double expected_armor_area = std::abs(cv::contourArea(result.armor_polygon));

  int armor_index = -1;
  int r_mark_index = -1;
  int light_arm_index = -1;
  float best_armor_error = std::numeric_limits<float>::infinity();
  double best_r_mark_area = std::numeric_limits<double>::infinity();
  double best_light_arm_area = -1.0;
  std::vector<float> solidities(contours.size(), 0.0F);
  std::vector<float> arm_line_support(contours.size(), 0.0F);

  for (std::size_t index = 0; index < contours.size(); ++index) {
    const auto & contour = contours[index];
    const double area = std::abs(cv::contourArea(contour));
    if (area < config.min_contour_area_px2) continue;
    solidities[index] = contour_solidity(contour);

    const bool contains_armor_center =
      cv::pointPolygonTest(contour, armor_center, false) >= 0.0;
    if (contains_armor_center && expected_armor_area > EPSILON) {
      const float relative_error =
        static_cast<float>(std::abs(area - expected_armor_area) / expected_armor_area);
      if (
        relative_error <= config.armor_area_relative_error &&
        relative_error < best_armor_error) {
        armor_index = static_cast<int>(index);
        best_armor_error = relative_error;
      }
    }

    bool contains_other_semantic_point = contains_armor_center;
    for (std::size_t point_index = 0; point_index < 4; ++point_index) {
      contains_other_semantic_point =
        contains_other_semantic_point ||
        cv::pointPolygonTest(contour, result.points[point_index], false) >= 0.0;
    }
    const bool contains_r_mark = cv::pointPolygonTest(contour, r_mark, false) >= 0.0;
    if (contains_r_mark && !contains_other_semantic_point && area < best_r_mark_area) {
      r_mark_index = static_cast<int>(index);
      best_r_mark_area = area;
    }

    if (!contains_armor_center && !contains_r_mark) {
      arm_line_support[index] =
        line_support(contour, armor_center, r_mark, config.light_arm_line_samples);
      if (arm_line_support[index] > 0.02F && area > best_light_arm_area) {
        light_arm_index = static_cast<int>(index);
        best_light_arm_area = area;
      }
    }
  }

  if (armor_index >= 0) {
    const float area_match = 1.0F - std::clamp(best_armor_error, 0.0F, 1.0F);
    set_feature(
      result.armor_contour, contours[armor_index],
      0.5F * solidities[armor_index] + 0.5F * area_match);
  }
  if (r_mark_index >= 0) {
    set_feature(result.r_mark_contour, contours[r_mark_index], solidities[r_mark_index]);
  }
  if (light_arm_index >= 0) {
    const float axial_support = std::min(1.0F, arm_line_support[light_arm_index] * 4.0F);
    set_feature(
      result.light_arm_contour, contours[light_arm_index],
      0.5F * solidities[light_arm_index] + 0.5F * axial_support);
  }
}
}  // namespace

ObservationRefiner::ObservationRefiner(ObservationRefinerConfig config) : config_(std::move(config))
{
  validate_config(config_);
}

ObservationRefiner::ObservationRefiner(const std::string & config_path)
: ObservationRefiner(load_config(config_path))
{
}

RefinedObservation ObservationRefiner::refine(
  const cv::Mat & image, const BuffObservation & observation, const RuneState & state) const
{
  RefinedObservation result;
  result.state = state;
  result.points = observation.points;
  result.detection_confidence = observation.confidence;
  result.armor_polygon.assign(observation.points.begin(), observation.points.begin() + 4);

  result.reject_reason = validate_observation(image, observation, config_, result.geometry_score);
  if (result.reject_reason != ObservationRejectReason::NONE) {
    result.quality = TrackingQuality::INVALID;
    return result;
  }

  result.roi = make_roi(observation.points, image.size(), config_);
  if (result.roi.empty()) {
    result.reject_reason = ObservationRejectReason::POINT_OUT_OF_BOUNDS;
    result.quality = TrackingQuality::INVALID;
    return result;
  }

  extract_semantic_contours(image, result, config_);
  const bool complete =
    result.armor_contour.usable() && result.r_mark_contour.usable() &&
    result.light_arm_contour.usable();
  const float minimum_support = std::min(
    {result.armor_contour.support, result.r_mark_contour.support,
     result.light_arm_contour.support});
  if (
    complete && observation.confidence >= config_.good_confidence &&
    result.geometry_score >= config_.good_geometry_score &&
    minimum_support >= config_.good_contour_support) {
    result.quality = TrackingQuality::GOOD;
  } else {
    result.quality = TrackingQuality::DEGRADED;
  }
  return result;
}

ObservationRefinerConfig ObservationRefiner::load_config(const std::string & config_path)
{
  ObservationRefinerConfig config;
  const YAML::Node root = YAML::LoadFile(config_path);
  const YAML::Node node = root["observation_refiner"];
  if (!node) return config;
  if (!node.IsMap()) throw std::runtime_error("observation_refiner must be a YAML map");

  read_if_present(node, "roi_margin_ratio", config.roi_margin_ratio);
  read_if_present(node, "roi_margin_px", config.roi_margin_px);
  read_if_present(node, "min_point_distance_px", config.min_point_distance_px);
  read_if_present(node, "min_armor_area_px2", config.min_armor_area_px2);
  read_if_present(node, "min_armor_edge_px", config.min_armor_edge_px);
  read_if_present(node, "r_boundary_tolerance_px", config.r_boundary_tolerance_px);
  read_if_present(node, "gaussian_kernel_size", config.gaussian_kernel_size);
  read_if_present(node, "binary_threshold", config.binary_threshold);
  read_if_present(node, "morphology_kernel_size", config.morphology_kernel_size);
  read_if_present(node, "min_contour_area_px2", config.min_contour_area_px2);
  read_if_present(node, "armor_area_relative_error", config.armor_area_relative_error);
  read_if_present(node, "light_arm_line_samples", config.light_arm_line_samples);
  read_if_present(node, "good_confidence", config.good_confidence);
  read_if_present(node, "good_geometry_score", config.good_geometry_score);
  read_if_present(node, "good_contour_support", config.good_contour_support);
  return config;
}

void ObservationRefiner::validate_config(const ObservationRefinerConfig & config)
{
  const auto in_unit_interval = [](float value) {
    return std::isfinite(value) && value >= 0.0F && value <= 1.0F;
  };
  if (!in_unit_interval(config.roi_margin_ratio)) {
    throw std::invalid_argument("roi_margin_ratio must be in [0, 1]");
  }
  if (
    config.roi_margin_px < 0 || config.min_point_distance_px <= 0.0F ||
    config.min_armor_area_px2 <= 0.0F || config.min_armor_edge_px <= 0.0F ||
    config.r_boundary_tolerance_px < 0.0F || config.min_contour_area_px2 <= 0.0F) {
    throw std::invalid_argument("ObservationRefiner geometry thresholds must be positive");
  }
  if (
    config.gaussian_kernel_size <= 0 || config.gaussian_kernel_size % 2 == 0 ||
    config.morphology_kernel_size <= 0 || config.morphology_kernel_size % 2 == 0) {
    throw std::invalid_argument("ObservationRefiner kernel sizes must be positive odd numbers");
  }
  if (config.binary_threshold < 0 || config.binary_threshold > 255) {
    throw std::invalid_argument("binary_threshold must be in [0, 255]");
  }
  if (
    !in_unit_interval(config.armor_area_relative_error) || config.light_arm_line_samples < 2 ||
    !in_unit_interval(config.good_confidence) || !in_unit_interval(config.good_geometry_score) ||
    !in_unit_interval(config.good_contour_support)) {
    throw std::invalid_argument("ObservationRefiner quality thresholds are invalid");
  }
}
}  // namespace auto_buff
