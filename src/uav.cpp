#include <chrono>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/cboard.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/detector.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/buff_processor.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

const std::string keys =
  "{help h usage ? |                  | 输出命令行参数说明}"
  "{@config-path   | configs/uav.yaml | yaml配置文件路径 }";

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
  auto_aim::Detector detector(config_path);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);
  auto_buff::Buff_Detector buff_detector(config_path);
  auto_buff::BuffProcessor buff_processor(config_path, "configs/auto_buff.yaml");

  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;
  auto mode = io::Mode::idle;
  auto last_mode = io::Mode::idle;

  while (!exiter.exit()) {
    camera.read(img, timestamp);
    const auto q = cboard.imu_at(timestamp - 1ms);
    mode = cboard.mode;
    if (last_mode != mode) {
      tools::logger()->info("Switch to {}", io::MODES[mode]);
      last_mode = mode;
    }

    if (mode == io::Mode::auto_aim || mode == io::Mode::outpost) {
      solver.set_R_gimbal2world(q);
      const auto ypr = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);
      auto armors = detector.detect(img);
      const auto targets = tracker.track(armors, timestamp);
      auto command = aimer.aim(targets, timestamp, cboard.bullet_speed);
      command.shoot = shooter.shoot(command, aimer, targets, ypr);
      cboard.send(command);
    } else if (mode == io::Mode::small_buff || mode == io::Mode::big_buff) {
      const auto observation = buff_detector.detect_observation(img);
      const auto output = buff_processor.process(auto_buff::make_buff_input(
        img, timestamp, q, cboard.gimbal_state(), cboard.gimbal_mode(), observation));
      cboard.send(output);
    }
  }
  return 0;
}
