#include "yolo11_buff.hpp"
#include "yolo11_buff_backend.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <yaml-cpp/yaml.h>

#include "tools/logger.hpp"

namespace auto_buff {
BuffLetterbox make_buff_letterbox(int source_width, int source_height, int width, int height)
{
  if (source_width <= 0 || source_height <= 0 || width <= 0 || height <= 0) {
    throw std::invalid_argument("[YOLO11_BUFF] Invalid letterbox dimensions");
  }
  BuffLetterbox result;
  result.width = width;
  result.height = height;
  result.scale = std::min(float(width) / source_width, float(height) / source_height);
  result.resized_width = std::max(1, int(std::round(source_width * result.scale)));
  result.resized_height = std::max(1, int(std::round(source_height * result.scale)));
  result.pad_x = (width - result.resized_width) / 2;
  result.pad_y = (height - result.resized_height) / 2;
  return result;
}

namespace {
// RuneDetectionModel 类别定义 (与 buffer-0624.xml / model-0624.onnx 一致)。
constexpr std::array<const char *, 3> class_names = {
  "inactive", "small_activated", "big_activated"};

// 可选读取 yaml 字段, 缺省时保留成员默认值 (与 RP-26 detect.json 对齐)
template <typename T>
void read_threshold_if_present(const YAML::Node &node, const char *key,
                               T &value) {
  if (node[key])
    value = node[key].as<T>();
}
} // namespace

YOLO11_BUFF::YOLO11_BUFF(const std::string &config, std::unique_ptr<Yolo11BuffBackend> backend)
: backend_(std::move(backend)) {
  auto yaml = YAML::LoadFile(config);
  // 后处理阈值: 与 RP-26 detect.json 一致 (conf 0.65 / kconf 0.5 / nms 30 /
  // min_valid 5)
  read_threshold_if_present(yaml, "conf", conf_threshold_);
  if (!yaml["conf"])
    read_threshold_if_present(yaml, "confidence_threshold", conf_threshold_);
  read_threshold_if_present(yaml, "kconf", kpt_conf_threshold_);
  if (!yaml["kconf"])
    read_threshold_if_present(yaml, "keypoint_confidence_threshold",
                              kpt_conf_threshold_);
  read_threshold_if_present(yaml, "nms", nms_dist_threshold_);
  if (!yaml["nms"])
    read_threshold_if_present(yaml, "nms_distance", nms_dist_threshold_);
  if (!yaml["nms"] && !yaml["nms_distance"]) {
    read_threshold_if_present(yaml, "nms_distance_threshold",
                              nms_dist_threshold_);
  }
  read_threshold_if_present(yaml, "min_valid_kpts", min_valid_kpts_);
  if (!yaml["min_valid_kpts"])
    read_threshold_if_present(yaml, "min_valid_keypoints", min_valid_kpts_);

  if (!backend_) throw std::invalid_argument("[YOLO11_BUFF] Missing inference backend");
  diagnostics_.info = backend_->info();
  const auto & input_shape = diagnostics_.info.input_shape;
  if (input_shape.size() != 4 || input_shape[0] != 1 || input_shape[1] != 3 ||
      input_shape[2] <= 0 || input_shape[3] <= 0) {
    throw std::runtime_error("[YOLO11_BUFF] Expected static [1,3,H,W] input");
  }
  input_height_ = static_cast<int>(input_shape[2]);
  input_width_ = static_cast<int>(input_shape[3]);
  const auto & output_shape = diagnostics_.info.output_shape;
  if (output_shape.size() != 3 || output_shape[0] != 1 || output_shape[1] <= 0 || output_shape[2] <= 0) {
    throw std::runtime_error("[YOLO11_BUFF] Expected static batch=1 three-dimensional output");
  }
  if (output_shape[1] == 18 && output_shape[2] != 18) {
    output_is_nca_ = true;
    output_channels_ = static_cast<int>(output_shape[1]);
    num_anchors_ = static_cast<int>(output_shape[2]);
  } else if (output_shape[2] == 18 && output_shape[1] != 18) {
    output_is_nca_ = false;
    num_anchors_ = static_cast<int>(output_shape[1]);
    output_channels_ = static_cast<int>(output_shape[2]);
  } else {
    throw std::runtime_error("[YOLO11_BUFF] Expected unambiguous [1,18,A] or [1,A,18] output");
  }

  // RuneDetectionModel 固定 3 类别 + 5 关键点 * 3 (x,y,conf) = 18 通道 (与
  // RP-26 校验一致)
  if (output_channels_ != num_classes_ + num_keypoints_ * kpt_dim_) {
    throw std::runtime_error(
        "[YOLO11_BUFF] 不支持的输出头: 期望 " +
        std::to_string(num_classes_ + num_keypoints_ * kpt_dim_) +
        " 通道(3类别+5x3关键点), 实际 " + std::to_string(output_channels_));
  }

  tools::logger()->info(
      "[YOLO11_BUFF] 模型加载成功: {} (输入 {}x{}, 输出 [{},{}], layout: {}, "
      "device: {}, conf: {:.2f}, "
      "kconf: {:.2f}, nms: {:.0f}, min_valid_kpts: {})",
      diagnostics_.info.model_path, input_width_, input_height_, output_channels_, num_anchors_,
      output_is_nca_ ? "[1,C,A]" : "[1,A,C]", diagnostics_.info.backend + ":" + diagnostics_.info.device, conf_threshold_,
      kpt_conf_threshold_, nms_dist_threshold_, min_valid_kpts_);
}

cv::Mat YOLO11_BUFF::normalize_input(const cv::Mat &source) {
  if (source.empty()) {
    throw std::invalid_argument("[YOLO11_BUFF] 无法推理空图");
  }
  if (source.type() == CV_8UC3)
    return source;

  cv::Mat bgr;
  if (source.type() == CV_8UC1) {
    cv::cvtColor(source, bgr, cv::COLOR_GRAY2BGR);
  } else if (source.type() == CV_8UC4) {
    cv::cvtColor(source, bgr, cv::COLOR_BGRA2BGR);
  } else {
    throw std::invalid_argument("[YOLO11_BUFF] 期望 8-bit 1/3/4 通道图, 实际 " +
                                std::to_string(source.channels()) + " 通道");
  }
  return bgr;
}

YOLO11_BUFF::~YOLO11_BUFF() = default;

std::vector<YOLO11_BUFF::Object> YOLO11_BUFF::postprocess(const float *output_data, float scale,
                                                          int pad_w, int pad_h,
                                                          int orig_w,
                                                          int orig_h) {
  std::vector<Object> objects;
  objects.reserve(64);

  // 自动按 [1,C,A] 或 [1,A,C] 布局取元素 (与 RP-26 一致)
  const auto value_at = [this, output_data](int channel, int anchor) -> float {
    return output_is_nca_ ? output_data[channel * num_anchors_ + anchor]
                          : output_data[anchor * output_channels_ + channel];
  };

  for (int anchor = 0; anchor < num_anchors_; ++anchor) {
    int best_cls = -1;
    float max_conf = 0.0f;
    for (int c = 0; c < num_classes_; ++c) {
      const float score = value_at(c, anchor);
      if (std::isfinite(score) && score > max_conf) {
        max_conf = score;
        best_cls = c;
      }
    }

    if (best_cls < 0 || max_conf < conf_threshold_)
      continue;

    Object obj;
    obj.label = best_cls;
    obj.prob = max_conf;
    obj.kpt.reserve(num_keypoints_);
    obj.kpt_conf.reserve(num_keypoints_);

    int valid_kpts = 0;
    float kpt_conf_sum = 0.0f;
    cv::Point2f valid_center(0.0f, 0.0f);
    bool has_invalid_kpt = false;

    for (int k = 0; k < num_keypoints_; ++k) {
      const int base = num_classes_ + k * kpt_dim_;
      // 先还原 letterbox, 再判有限性/负坐标 (与 RP-26 一致)
      const float x =
          (value_at(base + 0, anchor) - static_cast<float>(pad_w)) / scale;
      const float y =
          (value_at(base + 1, anchor) - static_cast<float>(pad_h)) / scale;
      const float k_conf = value_at(base + 2, anchor);

      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(k_conf) ||
          x < 0.0f || y < 0.0f) {
        has_invalid_kpt = true;
        break;
      }

      const float x_clamped =
          std::clamp(x, 0.0f, static_cast<float>(orig_w - 1));
      const float y_clamped =
          std::clamp(y, 0.0f, static_cast<float>(orig_h - 1));

      obj.kpt.emplace_back(x_clamped, y_clamped);
      obj.kpt_conf.push_back(k_conf);
      if (k_conf >= kpt_conf_threshold_) {
        ++valid_kpts;
        kpt_conf_sum += k_conf;
        valid_center += cv::Point2f(x_clamped, y_clamped);
      }
    }

    // 只统计关键点置信度达标者; 有效关键点不足则整条丢弃 (与 RP-26 一致)
    if (has_invalid_kpt || valid_kpts < min_valid_kpts_)
      continue;

    obj.center = valid_center * (1.0f / static_cast<float>(valid_kpts));
    obj.quality = obj.prob * (kpt_conf_sum / static_cast<float>(valid_kpts));
    objects.push_back(std::move(obj));
  }

  return nms(objects);
}

std::vector<YOLO11_BUFF::Object>
YOLO11_BUFF::nms(std::vector<Object> &objects) {
  if (objects.empty())
    return {};

  // 排序键 = quality (conf * mean kpt_conf), 与 RP-26 center_distance_nms 一致
  std::sort(
      objects.begin(), objects.end(),
      [](const Object &a, const Object &b) { return a.quality > b.quality; });

  std::vector<Object> result;
  std::vector<bool> suppressed(objects.size(), false);
  const float dist_thresh_sq = nms_dist_threshold_ * nms_dist_threshold_;

  for (size_t i = 0; i < objects.size(); ++i) {
    if (suppressed[i])
      continue;
    result.push_back(objects[i]);
    const auto &c_i = objects[i].center;
    for (size_t j = i + 1; j < objects.size(); ++j) {
      if (suppressed[j])
        continue;
      const float dx = c_i.x - objects[j].center.x;
      const float dy = c_i.y - objects[j].center.y;
      if (dx * dx + dy * dy < dist_thresh_sq)
        suppressed[j] = true;
    }
  }
  return result;
}

std::vector<YOLO11_BUFF::Object>
YOLO11_BUFF::get_multicandidateboxes(cv::Mat &image) {
  diagnostics_.timings = {};
  if (image.empty()) {
    tools::logger()->warn("[YOLO11_BUFF] Empty img!, camera drop!");
    return std::vector<Object>();
  }
  const int64 start = cv::getTickCount();

  // 统一输入为 8-bit BGR (支持 1/3/4 通道), 与 RP-26 一致
  const cv::Mat bgr = normalize_input(image);

  const auto transform = make_buff_letterbox(image.cols, image.rows, input_width_, input_height_);
  const float * output = backend_->infer(bgr, transform, diagnostics_.timings);
  const int64 postprocess_start = cv::getTickCount();
  auto objects = postprocess(output, transform.scale, transform.pad_x, transform.pad_y,
                             image.cols, image.rows);
  diagnostics_.timings.postprocess_ms =
    (cv::getTickCount() - postprocess_start) * 1000.0 / cv::getTickFrequency();
  diagnostics_.timings.total_ms = (cv::getTickCount() - start) * 1000.0 / cv::getTickFrequency();
  const int64 draw_start = cv::getTickCount();

  draw(image, objects);

  const float t =
      (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
  cv::putText(image, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40),
              cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
  diagnostics_.timings.draw_ms = (cv::getTickCount() - draw_start) * 1000.0 / cv::getTickFrequency();

  return objects;
}

void YOLO11_BUFF::draw(cv::Mat &image, const std::vector<Object> &objects) {
  for (const auto &obj : objects) {
    // 绘制关键点
    for (size_t k = 0; k < obj.kpt.size(); ++k) {
      cv::Scalar color = obj.kpt_conf[k] >= kpt_conf_threshold_
                             ? cv::Scalar(0, 255, 0)
                             : cv::Scalar(0, 0, 255);
      cv::circle(image, obj.kpt[k], 3, color, -1);
      cv::circle(image, obj.kpt[k], 3, cv::Scalar(255, 255, 255), 1);
      cv::putText(image, std::to_string(k + 1), obj.kpt[k] + cv::Point2f(5, -5),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1,
                  cv::LINE_AA);
    }

    // 绘制类别标签
    std::string cls_name =
        (obj.label >= 0 && obj.label < static_cast<int>(class_names.size()))
            ? class_names[obj.label]
            : ("class" + std::to_string(obj.label));
    std::string label =
        cls_name + " " + std::to_string(static_cast<int>(obj.prob * 100)) + "%";
    int baseline = 0;
    cv::Size text_size =
        cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    cv::rectangle(
        image, cv::Point(obj.center.x - 2, obj.center.y - text_size.height - 4),
        cv::Point(obj.center.x + text_size.width + 2, obj.center.y + 2),
        cv::Scalar(0, 0, 0), -1);
    cv::putText(image, label, cv::Point(obj.center.x, obj.center.y - 2),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
  }
}

} // namespace auto_buff
