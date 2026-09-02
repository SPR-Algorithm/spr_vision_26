#include <fmt/core.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "tasks/auto_buff/buff_detector.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/plotter.hpp"

const std::string keys =
    "{help h usage ? |                                          | "
    "输出命令行参数说明 }"
    "{config-path c  | configs/standard4.yaml                   | "
    "yaml配置文件的路径}"
    "{start-index s  | 0                                        | "
    "视频起始帧下标    }"
    "{end-index e    | 0                                        | "
    "视频结束帧下标    }"
    "{output o       |                                          | "
    "输出标注视频路径  }"
    "{benchmark b    | false                                    | "
    "基准模式(不显示)  }"
    "{warmup w       | 10                                       | GPU预热帧数  "
    "     }"
    "{@video_path    | assets/test_video/buff_2.avi  | 输入视频路径      }";

static cv::Scalar activation_color(auto_buff::BuffActivation activation) {
  switch (activation) {
  case auto_buff::BuffActivation::INACTIVE:
    return cv::Scalar(0, 255, 0); // 绿: 未激活
  case auto_buff::BuffActivation::SMALL_ACTIVATED:
    return cv::Scalar(255, 128, 0); // 蓝: 小符已激活
  case auto_buff::BuffActivation::BIG_ACTIVATED:
    return cv::Scalar(0, 0, 255); // 红: 大符已激活
  case auto_buff::BuffActivation::INVALID:
  default:
    return cv::Scalar(200, 200, 200);
  }
}

// 绘制单个标准五点观测: points[0..3] = 装甲板角点, points[4] = R 标中心
static void draw_observation(cv::Mat &img,
                             const auto_buff::BuffObservation &obs,
                             std::size_t index, bool highlight) {
  const cv::Scalar color =
      highlight ? cv::Scalar(0, 0, 255) : activation_color(obs.activation);
  for (std::size_t i = 0; i < 4; ++i) {
    tools::draw_point(img, obs.points[i], color, i == 0 ? 4 : 3);
  }
  tools::draw_point(img, obs.points[4], {255, 255, 0}, 5); // R 标中心(黄)
  tools::draw_text(img, fmt::format("#{} conf:{:.2f}", index, obs.confidence),
                   obs.points[0] + cv::Point2f(6.0f, -6.0f), color, 0.5, 1);
}

// 显示本帧全部标准五点观测(红色高亮置信度最高的一片)
static void
draw_overlay(cv::Mat &img, int frame_count,
             const std::vector<auto_buff::BuffObservation> &observations) {
  for (std::size_t i = 0; i < observations.size(); ++i) {
    draw_observation(img, observations[i], i, i == 0);
  }
  const auto text =
      fmt::format("frame:{} blades:{}", frame_count, observations.size());
  tools::draw_text(img, text, {10, 30}, {0, 255, 255}, 0.8, 2);
}

int main(int argc, char *argv[]) {
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  const auto video_path = cli.get<std::string>(0);
  const auto config_path = cli.get<std::string>("config-path");
  const auto start_index = cli.get<int>("start-index");
  const auto end_index = cli.get<int>("end-index");
  const auto output_path = cli.get<std::string>("output");
  const auto benchmark = cli.get<bool>("benchmark");
  const auto warmup_frames = cli.get<int>("warmup");

  if (!std::filesystem::exists(config_path)) {
    tools::logger()->error("无法打开配置文件: {} (请在项目根目录运行)",
                           config_path);
    return 1;
  }

  std::string device = "CPU";
  {
    const auto yaml = YAML::LoadFile(config_path);
    if (yaml["device"])
      device = yaml["device"].as<std::string>();
  }

  tools::logger()->info("========== buff_detector 测试启动 ==========");
  tools::logger()->info("config: {}", config_path);
  tools::logger()->info("video:  {}", video_path);
  tools::logger()->info("device: {} | benchmark: {} | warmup: {} 帧", device,
                        benchmark, warmup_frames);
  tools::logger()->info("===========================================");
  tools::Plotter plotter;
  tools::Exiter exiter;

  cv::VideoCapture video(video_path);
  if (!video.isOpened()) {
    tools::logger()->error("无法打开视频: {}", video_path);
    return 1;
  }

  cv::VideoWriter writer;
  if (!benchmark && !output_path.empty()) {
    const int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    double fps = video.get(cv::CAP_PROP_FPS);
    if (fps <= 0)
      fps = 30.0;
    writer.open(
        output_path, fourcc, fps,
        cv::Size(static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH)),
                 static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT))));
    if (!writer.isOpened()) {
      tools::logger()->error("无法创建输出视频: {}", output_path);
      return 1;
    }
  }

  auto_buff::Buff_Detector detector(config_path);

  int total_frames = 0;
  int detected_frames = 0;
  int total_blades = 0;
  std::array<int, 4>
      activation_total{}; // [INACTIVE, SMALL_ACTIVATED, BIG_ACTIVATED, INVALID]
  int warmup_done = 0;
  std::vector<double> detect_ms_samples;

  video.set(cv::CAP_PROP_POS_FRAMES, start_index);

  for (int frame_count = start_index; !exiter.exit(); ++frame_count) {
    if (end_index > 0 && frame_count > end_index)
      break;

    cv::Mat img;
    video.read(img);
    if (img.empty())
      break;

    const auto t0 = std::chrono::steady_clock::now();
    auto observations = detector.detect_observations(img);
    const auto t1 = std::chrono::steady_clock::now();
    const double detect_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    const bool in_warmup = benchmark && warmup_done < warmup_frames;
    if (in_warmup) {
      ++warmup_done;
    } else {
      ++total_frames;
      detect_ms_samples.push_back(detect_ms);
    }
    nlohmann::json data;
    data["frame"] = frame_count;
    data["detect_ms"] = detect_ms;
    data["detected"] = 0;

    if (!observations.empty()) {
      if (!in_warmup) {
        ++detected_frames;
        total_blades += static_cast<int>(observations.size());
        for (const auto &obs : observations) {
          const int idx = static_cast<int>(obs.activation);
          if (idx >= 0 && idx < static_cast<int>(activation_total.size()))
            ++activation_total[idx];
        }
      }

      data["detected"] = 1;
      data["blades"] = static_cast<int>(observations.size());
      const auto &top = observations.front();
      data["top_activation"] = static_cast<int>(top.activation);
      data["top_conf"] = top.confidence;
      for (int i = 0; i < 5; ++i) {
        data[fmt::format("top_pt{}_x", i)] = top.points[i].x;
        data[fmt::format("top_pt{}_y", i)] = top.points[i].y;
      }
    }

    plotter.plot(data);

    if (!benchmark) {
      draw_overlay(img, frame_count, observations);
      if (writer.isOpened())
        writer.write(img);
      cv::imshow("buff_detector_test", img);
      int key = cv::waitKey(1);
      if (key == 'q')
        break;
      while (key == ' ')
        key = cv::waitKey(30);
    }

    if (!in_warmup && total_frames % 100 == 0) {
      tools::logger()->info("frame {} | detected {}/{} ({:.1f}%) | {:.1f} ms",
                            frame_count, detected_frames, total_frames,
                            100.0 * detected_frames / total_frames, detect_ms);
    }
  }

  cv::destroyAllWindows();

  const double detection_rate =
      total_frames > 0 ? 100.0 * detected_frames / total_frames : 0.0;
  const double avg_blades =
      detected_frames > 0 ? static_cast<double>(total_blades) / detected_frames
                          : 0.0;

  tools::logger()->info("========== buff_detector 测试汇总 ==========");
  if (benchmark && warmup_frames > 0) {
    tools::logger()->info("GPU 预热帧数: {}", warmup_done);
  }
  tools::logger()->info("总帧数: {}", total_frames);
  tools::logger()->info("成功检测帧数: {}", detected_frames);
  tools::logger()->info("检测率: {:.2f}%", detection_rate);
  tools::logger()->info("平均符片数: {:.2f}", avg_blades);
  tools::logger()->info(
      "激活累计: inactive={} small_activated={} big_activated={} invalid={}",
      activation_total[0], activation_total[1], activation_total[2],
      activation_total[3]);
  if (!detect_ms_samples.empty()) {
    const double sum_ms = std::accumulate(detect_ms_samples.begin(),
                                          detect_ms_samples.end(), 0.0);
    const double avg_ms = sum_ms / detect_ms_samples.size();
    const double min_ms =
        *std::min_element(detect_ms_samples.begin(), detect_ms_samples.end());
    const double max_ms =
        *std::max_element(detect_ms_samples.begin(), detect_ms_samples.end());
    tools::logger()->info(
        "detect 耗时 (排除预热): avg {:.2f} ms | min {:.2f} ms | max {:.2f} ms",
        avg_ms, min_ms, max_ms);
  }
  tools::logger()->info("===========================================");

  return 0;
}
