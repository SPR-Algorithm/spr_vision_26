#include <chrono>
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/multithread/mt_detector.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/buff_processor.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/recorder.hpp"
#include "tools/thread_safe_queue.hpp"

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
  tools::Recorder recorder;
  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);
  auto_aim::YOLO yolo(config_path, true);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Planner planner(config_path);
  tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);
  target_queue.push(std::nullopt);

  auto_buff::Buff_Detector buff_detector(config_path);
  auto_buff::BuffProcessor buff_processor(config_path, "configs/auto_buff.yaml");

  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;
  std::atomic<bool> quit = false;
  std::atomic<io::GimbalMode> mode{io::GimbalMode::IDLE};
  auto last_mode = io::GimbalMode::IDLE;

  auto plan_thread = std::thread([&]() {
    while (!quit) {
      if (!target_queue.empty() && mode == io::GimbalMode::AUTO_AIM) {
        const auto target = target_queue.front();
        const auto state = gimbal.state();
        const auto plan = planner.plan(target, state.bullet_speed);
        gimbal.send(
          plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch,
          plan.pitch_vel, plan.pitch_acc);
        std::this_thread::sleep_for(10ms);
      } else {
        std::this_thread::sleep_for(200ms);
      }
    }
  });

  while (!exiter.exit()) {
    mode = gimbal.mode();
    if (last_mode != mode) {
      tools::logger()->info("Switch to {}", gimbal.str(mode));
      last_mode = mode.load();
    }

    camera.read(img, timestamp);
    const auto q = gimbal.q(timestamp);
    const auto state = gimbal.state();
    recorder.record(img, q, timestamp);
    solver.set_R_gimbal2world(q);

    if (mode.load() == io::GimbalMode::AUTO_AIM) {
      auto armors = yolo.detect(img);
      const auto targets = tracker.track(armors, timestamp);
      target_queue.push(targets.empty() ? std::nullopt : std::optional<auto_aim::Target>(targets.front()));
    } else if (
      mode.load() == io::GimbalMode::SMALL_BUFF || mode.load() == io::GimbalMode::BIG_BUFF) {
      const auto observation = buff_detector.detect_observation(img);
      const auto output = buff_processor.process(auto_buff::make_buff_input(
        img, timestamp, q, state, mode.load(), observation));
      gimbal.send(output);
    } else {
      gimbal.send(io::VisionToGimbal{});
    }
  }

  quit = true;
  if (plan_thread.joinable()) plan_thread.join();
  gimbal.send(io::VisionToGimbal{});
  return 0;
}
