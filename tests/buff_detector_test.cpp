#include <fmt/core.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <numeric>
#include <nlohmann/json.hpp>
#include <optional>
#include <opencv2/opencv.hpp>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/buff_type.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/plotter.hpp"

const std::string keys =
  "{help h usage ? |                                          | 输出命令行参数说明 }"
  "{config-path c  | configs/standard4.yaml                   | yaml配置文件的路径}"
  "{start-index s  | 0                                        | 视频起始帧下标    }"
  "{end-index e    | 0                                        | 视频结束帧下标    }"
  "{output o       |                                          | 输出标注视频路径  }"
  "{benchmark b    | false                                    | 基准模式(不显示)  }"
  "{warmup w       | 10                                       | GPU预热帧数       }"
  "{@video_path    | assets/test_video/buff_2.avi  | 输入视频路径      }";

static void draw_fanblade(cv::Mat & img, const auto_buff::FanBlade & blade, bool is_target)
{
  const cv::Scalar color = is_target ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
  if (blade.points.size() >= 4) {
    std::vector<cv::Point2f> corners(blade.points.begin(), blade.points.begin() + 4);
    tools::draw_points(img, corners, color, 2);
  }
  for (size_t i = 0; i < blade.points.size(); ++i) {
    const int radius = (i == 4) ? 4 : 3;
    tools::draw_point(img, blade.points[i], color, radius);
  }
  tools::draw_point(img, blade.center, color, 3);
}

static void draw_powerrune(cv::Mat & img, const auto_buff::PowerRune & rune)
{
  for (const auto & blade : rune.fanblades) {
    if (blade.type == auto_buff::_unlight) continue;
    draw_fanblade(img, blade, blade.type == auto_buff::_target);
  }
  tools::draw_point(img, rune.r_center, {255, 255, 0}, 6);
}

static void draw_overlay(cv::Mat & img, int frame_count, const std::optional<auto_buff::PowerRune> & result)
{
  std::string status = "LOSE";
  int light_num = 0;
  int target_cls = -1;
  std::string rune_type = "N/A";

  if (result.has_value() && !result->is_unsolve()) {
    const auto & rune = result.value();
    status = "TRACK";
    light_num = rune.light_num;
    target_cls = rune.fanblades[0].cls;
    rune_type = (rune.type == auto_buff::BIG) ? "BIG" : "SMALL";
    draw_powerrune(img, rune);
  }

  const auto text = fmt::format(
    "frame:{} status:{} light:{} type:{} cls:{}", frame_count, status, light_num, rune_type,
    target_cls);
  tools::draw_text(img, text, {10, 30}, {0, 255, 255}, 0.8, 2);
}

int main(int argc, char * argv[])
{
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
    tools::logger()->error("无法打开配置文件: {} (请在项目根目录运行)", config_path);
    return 1;
  }

  std::string device = "CPU";
  {
    const auto yaml = YAML::LoadFile(config_path);
    if (yaml["device"]) device = yaml["device"].as<std::string>();
  }

  tools::logger()->info("========== buff_detector 测试启动 ==========");
  tools::logger()->info("config: {}", config_path);
  tools::logger()->info("video:  {}", video_path);
  tools::logger()->info("device: {} | benchmark: {} | warmup: {} 帧", device, benchmark, warmup_frames);
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
    if (fps <= 0) fps = 30.0;
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
  int total_light_num = 0;
  int small_count = 0;
  int big_count = 0;
  int warmup_done = 0;
  std::vector<double> detect_ms_samples;

  video.set(cv::CAP_PROP_POS_FRAMES, start_index);

  for (int frame_count = start_index; !exiter.exit(); ++frame_count) {
    if (end_index > 0 && frame_count > end_index) break;

    cv::Mat img;
    video.read(img);
    if (img.empty()) break;

    const auto t0 = std::chrono::steady_clock::now();
    auto result = detector.detect(img);
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

    if (result.has_value() && !result->is_unsolve()) {
      const auto & rune = result.value();
      if (!in_warmup) {
        ++detected_frames;
        total_light_num += rune.light_num;
        if (rune.type == auto_buff::BIG)
          ++big_count;
        else
          ++small_count;
      }

      data["detected"] = 1;
      data["light_num"] = rune.light_num;
      data["r_center_x"] = rune.r_center.x;
      data["r_center_y"] = rune.r_center.y;
      data["target_cls"] = rune.fanblades[0].cls;
      data["rune_type"] = (rune.type == auto_buff::BIG) ? 1 : 0;
      data["target_center_x"] = rune.fanblades[0].center.x;
      data["target_center_y"] = rune.fanblades[0].center.y;
      for (int i = 0; i < 4; ++i) {
        data[fmt::format("target_pt{}_x", i)] = rune.fanblades[0].points[i].x;
        data[fmt::format("target_pt{}_y", i)] = rune.fanblades[0].points[i].y;
      }
    }

    plotter.plot(data);

    if (!benchmark) {
      draw_overlay(img, frame_count, result);
      if (writer.isOpened()) writer.write(img);
      cv::imshow("buff_detector_test", img);
      int key = cv::waitKey(1);
      if (key == 'q') break;
      while (key == ' ') key = cv::waitKey(30);
    }

    if (!in_warmup && total_frames % 100 == 0) {
      tools::logger()->info(
        "frame {} | detected {}/{} ({:.1f}%) | {:.1f} ms", frame_count, detected_frames,
        total_frames, 100.0 * detected_frames / total_frames, detect_ms);
    }
  }

  cv::destroyAllWindows();

  const double detection_rate =
    total_frames > 0 ? 100.0 * detected_frames / total_frames : 0.0;
  const double avg_light_num =
    detected_frames > 0 ? static_cast<double>(total_light_num) / detected_frames : 0.0;

  tools::logger()->info("========== buff_detector 测试汇总 ==========");
  if (benchmark && warmup_frames > 0) {
    tools::logger()->info("GPU 预热帧数: {}", warmup_done);
  }
  tools::logger()->info("总帧数: {}", total_frames);
  tools::logger()->info("成功检测帧数: {}", detected_frames);
  tools::logger()->info("检测率: {:.2f}%", detection_rate);
  tools::logger()->info("平均亮扇叶数: {:.2f}", avg_light_num);
  tools::logger()->info("小符帧数: {} | 大符帧数: {}", small_count, big_count);
  if (!detect_ms_samples.empty()) {
    const double sum_ms = std::accumulate(detect_ms_samples.begin(), detect_ms_samples.end(), 0.0);
    const double avg_ms = sum_ms / detect_ms_samples.size();
    const double min_ms = *std::min_element(detect_ms_samples.begin(), detect_ms_samples.end());
    const double max_ms = *std::max_element(detect_ms_samples.begin(), detect_ms_samples.end());
    tools::logger()->info(
      "detect 耗时 (排除预热): avg {:.2f} ms | min {:.2f} ms | max {:.2f} ms", avg_ms, min_ms,
      max_ms);
  }
  tools::logger()->info("===========================================");

  return 0;
}
