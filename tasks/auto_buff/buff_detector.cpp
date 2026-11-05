#include "buff_detector.hpp"

#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>
#include <yaml-cpp/yaml.h>

#include "yolo11_buff_openvino.hpp"
#ifdef SPR_AUTO_BUFF_TENSORRT
#include "yolo11_buff_tensorrt.hpp"
#endif

namespace auto_buff {
namespace {
std::unique_ptr<Yolo11BuffBackend> select_yolo11_buff_backend(const std::string & config)
{
  const auto yaml = YAML::LoadFile(config);
  const auto settings = yaml["buff_inference"];
  const auto name = settings && settings["backend"] ?
    settings["backend"].as<std::string>() : "openvino";
  if (name == "openvino") {
    const auto model = settings && settings["model"] ? settings["model"] : yaml["model"];
    const auto device = settings && settings["device"] ? settings["device"] : yaml["device"];
    if (!model || model.as<std::string>().empty()) {
      throw std::invalid_argument("[YOLO11_BUFF] Missing OpenVINO model");
    }
    return std::make_unique<Yolo11BuffOpenVINO>(
      model.as<std::string>(), device ? device.as<std::string>() : "CPU");
  }
  if (name == "tensorrt") {
#ifdef SPR_AUTO_BUFF_TENSORRT
    if (!settings["engine"] || settings["engine"].as<std::string>().empty()) {
      throw std::invalid_argument("[YOLO11_BUFF] TensorRT requires buff_inference.engine");
    }
    return std::make_unique<Yolo11BuffTensorRT>(
      settings["engine"].as<std::string>(),
      settings["cuda_device"] ? settings["cuda_device"].as<int>() : 0);
#else
    throw std::runtime_error("[YOLO11_BUFF] TensorRT was not compiled; enable SPR_AUTO_BUFF_TENSORRT");
#endif
  }
  throw std::invalid_argument("[YOLO11_BUFF] Unknown buff_inference.backend: " + name);
}
}  // namespace

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

Buff_Detector::Buff_Detector(const std::string &config)
: MODE_(config, select_yolo11_buff_backend(config)) {}

std::vector<BuffObservation>
Buff_Detector::detect_observations(cv::Mat &bgr_img) {
  std::vector<BuffObservation> observations;

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
Buff_Detector::detect_observation(cv::Mat &bgr_img) {
  // 单符片便捷接口: 取全部检出符片中置信度最高的一片(NMS 结果按 quality 降序)。
  const auto observations = detect_observations(bgr_img);
  if (observations.empty())
    return std::nullopt;
  return observations.front();
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
