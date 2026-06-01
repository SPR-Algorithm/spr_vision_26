#include <fmt/core.h>

#include <chrono>
#include <opencv2/opencv.hpp>
#include <string>

#include "io/camera.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/recorder.hpp"

const std::string keys =
    "{help h usage ? |                        | 输出命令行参数说明}"
    "{fps f          | 30                    | 录制帧率          }"
    "{@config-path   | configs/ascento.yaml | 位置参数，yaml配置文件路径 }";

int main(int argc, char *argv[]) {
  tools::Exiter exiter;

  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  double fps = cli.get<double>("fps");

  // 初始化相机
  io::Camera camera(config_path);
  tools::logger()->info("相机初始化完成");

  // 初始化录制器
  tools::Recorder recorder(fps);
  tools::logger()->info("录制器初始化完成，帧率: {} fps", fps);

  cv::Mat img;
  Eigen::Quaterniond q(1.0, 0.0, 0.0,
                       0.0); // 单位四元数（无旋转，仅用于录制接口）
  std::chrono::steady_clock::time_point timestamp;
  std::chrono::steady_clock::time_point last_t =
      std::chrono::steady_clock::now();

  int frame_count = 0;

  tools::logger()->info("开始录制视频，按 'q' 退出");

  while (!exiter.exit()) {
    // 读取相机图像
    camera.read(img, timestamp);
    auto dt = tools::delta_time(timestamp, last_t);
    last_t = timestamp;

    // 计算FPS
    double current_fps = 1.0 / dt;
    frame_count++;

    // 录制帧（使用单位四元数，不记录姿态信息）
    recorder.record(img, q, timestamp);

    // 在图像上显示信息
    cv::Mat display_img = img.clone();
    tools::draw_text(display_img, fmt::format("FPS: {:.2f}", current_fps),
                     {10, 30}, {255, 255, 255});
    tools::draw_text(display_img, fmt::format("Frame: {}", frame_count),
                     {10, 60}, {255, 255, 255});
    tools::draw_text(display_img, "Recording...", {10, 90}, {0, 255, 0});

    // 显示图像
    cv::imshow("Recording", display_img);

    // 每30帧输出一次日志
    if (frame_count % 30 == 0) {
      tools::logger()->info("已录制 {} 帧，当前FPS: {:.2f}", frame_count,
                            current_fps);
    }

    // 检查退出条件
    auto key = cv::waitKey(1);
    if (key == 'q') {
      tools::logger()->info("用户中断录制");
      break;
    }
  }

  cv::destroyAllWindows();
  tools::logger()->info("录制完成！共录制 {} 帧", frame_count);
  tools::logger()->info("视频已保存到 records/ 目录");

  return 0;
}
