#include <chrono>
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/camera.hpp"
#include "io/cboard.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/multithread/commandgener.hpp"
#include "tasks/auto_aim/multithread/mt_detector.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/buff_processor.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"

const std::string keys =
  "{help h usage ? | | 输出命令行参数说明}"
  "{@config-path   | | yaml配置文件路径 }";

using namespace std::chrono_literals;

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  const auto config_path = cli.get<std::string>("@config-path");
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;
  io::Camera camera(config_path);
  io::CBoard cboard(config_path);
  auto_aim::multithread::MultiThreadDetector detector(config_path);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);
  tools::Plotter plotter;
  auto_aim::multithread::CommandGener commandgener(shooter, aimer, cboard, plotter);
  auto_buff::Buff_Detector buff_detector(config_path);
  auto_buff::BuffProcessor buff_processor(config_path, "configs/auto_buff.yaml");

  std::atomic<io::Mode> mode{io::Mode::idle};
  auto last_mode = io::Mode::idle;
  auto detect_thread = std::thread([&]() {
    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp;
    while (!exiter.exit()) {
      if (mode.load() == io::Mode::auto_aim) {
        camera.read(img, timestamp);
        detector.push(img, timestamp);
      }
    }
  });

  while (!exiter.exit()) {
    mode = cboard.mode;
    if (last_mode != mode) {
      tools::logger()->info("Switch to {}", io::MODES[mode]);
      last_mode = mode.load();
    }
    if (mode.load() == io::Mode::auto_aim) {
      auto [img, armors, timestamp] = detector.debug_pop();
      const auto q = cboard.imu_at(timestamp - 1ms);
      solver.set_R_gimbal2world(q);
      const auto ypr = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);
      const auto targets = tracker.track(armors, timestamp);
      commandgener.push(targets, timestamp, cboard.bullet_speed, ypr);
    } else if (mode.load() == io::Mode::small_buff || mode.load() == io::Mode::big_buff) {
      cv::Mat img;
      std::chrono::steady_clock::time_point timestamp;
      camera.read(img, timestamp);
      const auto q = cboard.imu_at(timestamp - 1ms);
      const auto observation = buff_detector.detect_observation(img);
      const auto output = buff_processor.process(auto_buff::make_buff_input(
        img, timestamp, q, cboard.gimbal_state(), cboard.gimbal_mode(), observation));
      cboard.send(output);
    }
  }
  detect_thread.join();
  return 0;
}
