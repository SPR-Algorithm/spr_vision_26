#include "buff_detector.hpp"

#include <algorithm>

namespace auto_buff {
// RuneDetectionModel order: [top, left, R, right, bottom].
// Public observation order: [top, right, bottom, left, R].
constexpr std::array<std::size_t, BUFF_POINT_COUNT> KPT_MAP = {0, 3, 4, 1, 2};

static std::optional<BuffPoints>
order_keypoints(const std::vector<cv::Point2f> &kpt) {
  if (kpt.size() != KPT_MAP.size())
    return std::nullopt;
  BuffPoints ordered;
  for (std::size_t i = 0; i < KPT_MAP.size(); ++i)
    ordered[i] = kpt[KPT_MAP[i]];
  return ordered;
}

static BuffActivation activation_from_class(int cls) {
  switch (cls) {
  case 0:
    return BuffActivation::INACTIVE;
  case 1:
    return BuffActivation::SMALL_ACTIVATED;
  case 2:
    return BuffActivation::BIG_ACTIVATED;
  default:
    return BuffActivation::INVALID;
  }
}

Buff_Detector::Buff_Detector(const std::string &config) : MODE_(config) {}

std::vector<BuffObservation>
Buff_Detector::detect_observations(cv::Mat &bgr_img) {
  std::vector<BuffObservation> observations;

  // 直接取 YOLO 多候选结果, 绕过 PowerRune 的"单 target"跟踪逻辑:
  // 后者在多符片且无历史时会被判 unsolvable 而整体丢弃, 无法覆盖全部符片。
  const std::vector<YOLO11_BUFF::Object> results =
      MODE_.get_multicandidateboxes(bgr_img);
  for (const auto &result : results) {
    const auto observation = to_observation(result);
    if (observation.has_value())
      observations.push_back(observation.value());
  }
  return observations;
}

std::optional<BuffObservation>
Buff_Detector::detect_observation(cv::Mat &bgr_img) {
  // 单符片便捷接口: 取全部检出符片中置信度最高的一片(NMS 结果按 quality 降序)。
  const auto observations = detect_observations(bgr_img);
  if (observations.empty())
    return std::nullopt;
  return observations.front();
}

std::optional<BuffObservation>
Buff_Detector::to_observation(const YOLO11_BUFF::Object &detection) {
  const auto ordered = order_keypoints(detection.kpt);
  if (!ordered.has_value())
    return std::nullopt;

  BuffObservation observation;
  observation.points = ordered.value();
  observation.activation = activation_from_class(detection.label);
  observation.confidence = detection.prob;
  return observation;
}

std::optional<BuffObservation>
Buff_Detector::to_observation(const FanBlade &blade) {
  if (blade.points.size() != 5)
    return std::nullopt;
  BuffObservation observation;
  std::copy_n(blade.points.begin(), observation.points.size(),
              observation.points.begin());
  observation.activation = activation_from_class(blade.cls);
  observation.confidence = blade.confidence;
  return observation;
}

} // namespace auto_buff
