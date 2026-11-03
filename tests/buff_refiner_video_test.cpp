// buff_refiner_video_test.cpp
//
// 针对 ObservationRefiner 的视频测试:
//  1. 读取 assets/test_video/buff_2.avi 作为图片数据源;
//  2. 逐帧用 Buff_Detector 生成 BuffObservation;
//  3. 送入 ObservationRefiner::refine;
//  4. 通过 OpenCV 窗口直观查看生成的 ROI、装甲板/R
//  标/灯臂轮廓效果图与观测质量。
//
// 用法 (带值参数请使用 "=" 形式, 否则数值会被当作视频路径):
//  ./buff_refiner_video_test                              # 默认读取 buff_2.avi
//  并弹窗
//  ./buff_refiner_video_test -b                           # 基准模式: 不弹窗,
//  仅输出统计
//  ./buff_refiner_video_test -s=100 -e=300                # 只处理 100~300 帧
//  ./buff_refiner_video_test --device=CPU                 # 指定 OpenVINO
//  推理设备
//  ./buff_refiner_video_test --save=out.avi               #
//  同时把效果图写为视频

#include <fmt/core.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/observation_refiner.hpp"
#include "tasks/auto_buff/rune_types.hpp"
#include "tools/logger.hpp"

namespace {
// BuffObservation 携带的是检测侧的激活语义(BuffActivation),
// RuneState 需要解析出 kind + activation, 这里按观测激活状态做映射。
auto_buff::RuneState to_rune_state(auto_buff::BuffActivation activation) {
  switch (activation) {
  case auto_buff::BuffActivation::INACTIVE:
    return {auto_buff::RuneKind::SMALL, auto_buff::ActivationState::INACTIVE};
  case auto_buff::BuffActivation::SMALL_ACTIVATED:
    return {auto_buff::RuneKind::SMALL, auto_buff::ActivationState::ACTIVATED};
  case auto_buff::BuffActivation::BIG_ACTIVATED:
    return {auto_buff::RuneKind::BIG, auto_buff::ActivationState::ACTIVATED};
  case auto_buff::BuffActivation::INVALID:
  default:
    return {auto_buff::RuneKind::SMALL, auto_buff::ActivationState::INACTIVE};
  }
}

const char *activation_to_string(auto_buff::BuffActivation activation) {
  switch (activation) {
  case auto_buff::BuffActivation::INACTIVE:
    return "INACTIVE";
  case auto_buff::BuffActivation::SMALL_ACTIVATED:
    return "SMALL_ACTIVATED";
  case auto_buff::BuffActivation::BIG_ACTIVATED:
    return "BIG_ACTIVATED";
  case auto_buff::BuffActivation::INVALID:
    return "INVALID";
  }
  return "UNKNOWN";
}

// YOLO11_BUFF 的 device 从 yaml 读取; 需要覆盖设备(如 GPU->CPU)时,
// 复制原配置并写临时文件, 返回临时文件路径, 否则原样返回 config_path。
std::string make_detector_config(const std::string &config_path,
                                 const std::string &device) {
  if (device.empty())
    return config_path;
  YAML::Node modified = YAML::LoadFile(config_path);
  modified["device"] = device;
  const auto temp_path = (std::filesystem::temp_directory_path() /
                          "buff_refiner_device_override.yaml")
                             .string();
  std::ofstream out(temp_path, std::ios::trunc);
  if (!out) {
    tools::logger()->warn("无法写入临时配置 {}，回退到原始配置 {}", temp_path,
                          config_path);
    return config_path;
  }
  out << modified;
  tools::logger()->info("已生成临时配置(device={}): {}", device, temp_path);
  return temp_path;
}

// 判断字符串是否纯数字(用于捕获把 -e 的值误当作视频路径的常见用法)
bool is_numeric_string(const std::string &text) {
  return !text.empty() && std::all_of(text.begin(), text.end(), [](char c) {
    return c >= '0' && c <= '9';
  });
}

// Panel 1: 原始输入 + 关键点编号 (用于 --save 效果视频, 不单独开窗口)
cv::Mat draw_input_panel(const cv::Mat &image,
                         const auto_buff::BuffObservation &observation) {
  const std::array<cv::Scalar, 5> colors = {
      cv::Scalar(255, 255, 255), cv::Scalar(0, 255, 255),
      cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255),
      cv::Scalar(0, 165, 255)};
  cv::Mat input_panel = image.clone();
  for (std::size_t i = 0; i < observation.points.size(); ++i) {
    cv::circle(input_panel, observation.points[i], 5, colors[i], cv::FILLED,
               cv::LINE_AA);
    cv::putText(input_panel, std::to_string(i),
                observation.points[i] + cv::Point2f{7.0F, -7.0F},
                cv::FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2, cv::LINE_AA);
  }
  cv::putText(input_panel, "Input: [top, right, bottom, left, R]", {15, 28},
              cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 255, 255), 2,
              cv::LINE_AA);
  return input_panel;
}

// Panel 2: ROI + 装甲板/R 标/灯臂轮廓 + 质量信息 (窗口: Overlay)
cv::Mat draw_overlay(const cv::Mat &image,
                     const auto_buff::BuffObservation &observation,
                     const auto_buff::RefinedObservation &result,
                     int frame_count) {
  cv::Mat overlay = image.clone();
  if (result.roi.area() > 0) {
    cv::rectangle(overlay, result.roi, cv::Scalar(255, 255, 255), 2,
                  cv::LINE_AA);
  }
  const auto draw_contour = [&overlay](const auto_buff::ContourFeature &feature,
                                       const cv::Scalar &color) {
    if (!feature.usable())
      return;
    cv::drawContours(overlay,
                     std::vector<std::vector<cv::Point>>{feature.contour}, -1,
                     color, 3, cv::LINE_AA);
  };
  draw_contour(result.armor_contour, cv::Scalar(0, 255, 0));       // 装甲板
  draw_contour(result.r_mark_contour, cv::Scalar(0, 255, 255));    // R 标
  draw_contour(result.light_arm_contour, cv::Scalar(255, 255, 0)); // 灯臂

  const std::string quality_text =
      fmt::format("frame:{} Quality:{}  geo:{:.2f}  conf:{:.2f}  act:{}",
                  frame_count, auto_buff::to_string(result.quality),
                  result.geometry_score, result.detection_confidence,
                  activation_to_string(observation.activation));
  cv::putText(overlay, quality_text, {15, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
              cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
  const std::string reject_text =
      fmt::format("reject:{}", auto_buff::to_string(result.reject_reason));
  cv::putText(overlay, reject_text, {15, 56}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
              cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
  return overlay;
}

// Panel 3: 语义分割填充 (窗口: Semantic Mask)
cv::Mat draw_semantic(const cv::Mat &image,
                      const auto_buff::RefinedObservation &result) {
  cv::Mat semantic = cv::Mat::zeros(image.size(), CV_8UC3);
  const auto fill_contour =
      [&semantic](const auto_buff::ContourFeature &feature,
                  const cv::Scalar &color) {
        if (!feature.usable())
          return;
        cv::drawContours(semantic,
                         std::vector<std::vector<cv::Point>>{feature.contour},
                         -1, color, cv::FILLED);
      };
  fill_contour(result.armor_contour, cv::Scalar(0, 255, 0));
  fill_contour(result.r_mark_contour, cv::Scalar(0, 255, 255));
  fill_contour(result.light_arm_contour, cv::Scalar(255, 255, 0));
  cv::putText(semantic, "green=armor  yellow=R  cyan=arm", {15, 28},
              cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2,
              cv::LINE_AA);
  return semantic;
}

// Panel 4: 裁剪放大 ROI (窗口: ROI Clipped), source 通常传 overlay 以保留轮廓
cv::Mat draw_roi_panel(const cv::Mat &source,
                       const auto_buff::RefinedObservation &result) {
  cv::Mat roi_panel = cv::Mat::zeros(source.size(), CV_8UC3);
  if (result.roi.area() > 0 && result.roi.width > 0 && result.roi.height > 0) {
    const cv::Mat crop = source(result.roi);
    const double scale =
        std::min(static_cast<double>(roi_panel.cols) / crop.cols,
                 static_cast<double>(roi_panel.rows) / crop.rows);
    cv::Mat resized;
    cv::resize(crop, resized, {}, scale, scale, cv::INTER_NEAREST);
    const int x = (roi_panel.cols - resized.cols) / 2;
    const int y = (roi_panel.rows - resized.rows) / 2;
    resized.copyTo(roi_panel(cv::Rect(x, y, resized.cols, resized.rows)));
  }
  cv::putText(roi_panel, "Clipped ROI", {15, 28}, cv::FONT_HERSHEY_SIMPLEX,
              0.65, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
  return roi_panel;
}

// 显示缩放: 保持宽高比缩放到指定最大宽度, 避免 OpenCV 窗口过大
cv::Mat fit_display(const cv::Mat &src, int max_width) {
  if (src.empty() || src.cols <= max_width)
    return src;
  const double scale = static_cast<double>(max_width) / src.cols;
  cv::Mat dst;
  cv::resize(src, dst, {}, scale, scale, cv::INTER_AREA);
  return dst;
}

// 四宫格拼接, 用于 --save 效果视频
cv::Mat compose_save_canvas(const cv::Mat &input, const cv::Mat &overlay,
                            const cv::Mat &semantic, const cv::Mat &roi) {
  cv::Mat top_row;
  cv::Mat bottom_row;
  cv::Mat canvas;
  cv::hconcat(input, overlay, top_row);
  cv::hconcat(semantic, roi, bottom_row);
  cv::vconcat(top_row, bottom_row, canvas);
  return canvas;
}
} // namespace

int main(int argc, char **argv) {
  const cv::String keys =
      "{help h usage ?   |                              | 输出命令行参数说明 }"
      "{config-path c    | configs/standard4.yaml       | buff 检测器配置    }"
      "{refiner-config r | configs/auto_buff.yaml       | ObservationRefiner "
      "配置}"
      "{device d         | CPU                          | OpenVINO "
      "推理设备(GPU/CPU) }"
      "{start-index s    | 0                            | 视频起始帧下标     }"
      "{end-index e      | 0                            | "
      "视频结束帧下标(0=到末尾) }"
      "{save             |                              | 输出效果视频路径   }"
      "{window-width ww  | 640                          | 显示窗口最大宽度(px)}"
      "{benchmark b      | false                        | 基准模式(不显示窗口) "
      "}"
      "{@video_path      | assets/test_video/buff_2.avi | 输入视频路径       }";
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  const auto video_path = cli.get<std::string>(0);
  const auto config_path = cli.get<std::string>("config-path");
  const auto refiner_config = cli.get<std::string>("refiner-config");
  const auto device = cli.get<std::string>("device");
  const auto start_index = cli.get<int>("start-index");
  const auto end_index = cli.get<int>("end-index");
  const auto save_path = cli.get<std::string>("save");
  const auto window_width = cli.get<int>("window-width");
  const auto benchmark = cli.get<bool>("benchmark");

  // 常见误用: ./buff_refiner_video_test -e 300 (空格形式) 会把 "300"
  // 当成视频路径
  if (is_numeric_string(video_path)) {
    tools::logger()->error(
        "视频路径疑似为数字 '{}'。OpenCV 参数解析要求带值参数使用 '=' 形式, "
        "例如 -e=300 / --end-index=300, 而不是 '-e 300'。",
        video_path);
    return 1;
  }

  if (!std::filesystem::exists(video_path)) {
    tools::logger()->error("无法打开视频: {} (请在项目根目录运行)", video_path);
    return 1;
  }
  if (!std::filesystem::exists(config_path)) {
    tools::logger()->error("无法打开检测器配置: {}", config_path);
    return 1;
  }
  if (!std::filesystem::exists(refiner_config)) {
    tools::logger()->error("无法打开 refiner 配置: {}", refiner_config);
    return 1;
  }

  const std::string detector_config = make_detector_config(config_path, device);

  tools::logger()->info(
      "========== ObservationRefiner 视频测试启动 ==========");
  tools::logger()->info("video:        {}", video_path);
  tools::logger()->info("detector cfg: {}", detector_config);
  tools::logger()->info("device:       {}", device);
  tools::logger()->info("refiner cfg:  {}", refiner_config);
  tools::logger()->info("benchmark:    {}", benchmark);

  auto_buff::Buff_Detector detector(detector_config);
  auto_buff::ObservationRefiner refiner(refiner_config);

  cv::VideoCapture video(video_path);
  if (!video.isOpened()) {
    tools::logger()->error("无法打开视频: {}", video_path);
    return 1;
  }

  cv::VideoWriter writer;
  if (!benchmark && !save_path.empty()) {
    const int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    double fps = video.get(cv::CAP_PROP_FPS);
    if (fps <= 0)
      fps = 30.0;
    writer.open(
        save_path, fourcc, fps,
        cv::Size(2 * static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH)),
                 2 * static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT))));
    if (!writer.isOpened()) {
      tools::logger()->error("无法创建输出视频: {}", save_path);
      return 1;
    }
  }

  int frames_read = 0;
  int detect_frames = 0;
  int good_frames = 0;
  int degraded_frames = 0;
  int invalid_frames = 0;
  std::vector<double> refine_ms_samples;

  video.set(cv::CAP_PROP_POS_FRAMES, start_index);
  for (int frame_count = start_index;; ++frame_count) {
    if (end_index > 0 && frame_count > end_index)
      break;

    cv::Mat img;
    video.read(img);
    if (img.empty())
      break;
    ++frames_read;

    // 克隆一份用于检测, 避免污染显示/保存用的干净原图(否则标签会遮挡特征)
    cv::Mat work = img.clone();
    auto observation = detector.detect_observation(work);
    if (!observation.has_value()) {
      if (!benchmark) {
        cv::putText(img, "NO DETECT", {15, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.9,
                    cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        cv::imshow("Original Video", fit_display(img, window_width));
        const int key = cv::waitKey(1);
        if (key == 'q')
          break;
      }
      continue;
    }
    ++detect_frames;

    // 观测 -> RuneState -> ObservationRefiner
    const auto state = to_rune_state(observation->activation);
    const auto t0 = std::chrono::steady_clock::now();
    const auto result = refiner.refine(img, observation.value(), state);
    const auto t1 = std::chrono::steady_clock::now();
    refine_ms_samples.push_back(
        std::chrono::duration<double, std::milli>(t1 - t0).count());

    switch (result.quality) {
    case auto_buff::TrackingQuality::GOOD:
      ++good_frames;
      break;
    case auto_buff::TrackingQuality::DEGRADED:
      ++degraded_frames;
      break;
    case auto_buff::TrackingQuality::INVALID:
      ++invalid_frames;
      break;
    }

    if (!benchmark) {
      // 窗口 1: 原始视频
      cv::imshow("Original Video", fit_display(img, window_width));
      // 窗口 2: ROI + 装甲板/R 标/灯臂轮廓 + 质量信息
      const cv::Mat overlay =
          draw_overlay(img, observation.value(), result, frame_count);
      cv::imshow("Overlay (ROI / armor / R / arm)",
                 fit_display(overlay, window_width));
      // 窗口 3: 语义分割填充
      const cv::Mat semantic = draw_semantic(img, result);
      cv::imshow("Semantic Mask", fit_display(semantic, window_width));
      // 窗口 4: 裁剪放大 ROI
      const cv::Mat roi_panel = draw_roi_panel(overlay, result);
      cv::imshow("ROI Clipped", fit_display(roi_panel, window_width));
      int key = cv::waitKey(1);
      if (key == 'q')
        break;
      while (key == ' ')
        key = cv::waitKey(30);
      if (writer.isOpened()) {
        const cv::Mat canvas =
            compose_save_canvas(draw_input_panel(img, observation.value()),
                                overlay, semantic, roi_panel);
        writer.write(canvas);
      }
    }

    if (detect_frames % 50 == 0) {
      tools::logger()->info(
          "frame {} | read {} | GOOD {} DEGRADED {} INVALID {}", frame_count,
          frames_read, good_frames, degraded_frames, invalid_frames);
    }
  }

  cv::destroyAllWindows();

  const double good_rate =
      detect_frames > 0 ? 100.0 * good_frames / detect_frames : 0.0;
  const double degraded_rate =
      detect_frames > 0 ? 100.0 * degraded_frames / detect_frames : 0.0;

  tools::logger()->info(
      "========== ObservationRefiner 视频测试汇总 ==========");
  tools::logger()->info("读取帧数: {}", frames_read);
  tools::logger()->info("检测到观测帧数: {}", detect_frames);
  tools::logger()->info("GOOD: {} ({:.2f}%)", good_frames, good_rate);
  tools::logger()->info("DEGRADED: {} ({:.2f}%)", degraded_frames,
                        degraded_rate);
  tools::logger()->info("INVALID: {}", invalid_frames);
  if (!refine_ms_samples.empty()) {
    const double sum_ms = std::accumulate(refine_ms_samples.begin(),
                                          refine_ms_samples.end(), 0.0);
    const double avg_ms = sum_ms / refine_ms_samples.size();
    const double min_ms =
        *std::min_element(refine_ms_samples.begin(), refine_ms_samples.end());
    const double max_ms =
        *std::max_element(refine_ms_samples.begin(), refine_ms_samples.end());
    tools::logger()->info(
        "refine 耗时: avg {:.2f} ms | min {:.2f} ms | max {:.2f} ms", avg_ms,
        min_ms, max_ms);
  }
  tools::logger()->info(
      "=====================================================");

  return 0;
}
