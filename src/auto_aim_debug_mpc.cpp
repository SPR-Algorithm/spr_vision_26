#include <fmt/core.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <cmath>
#include <list>
#include <memory>
#include <optional>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <thread>
#include <yaml-cpp/yaml.h>

#include "debug/debug_bus.hpp"
#include "debug/benchmark.hpp"
#include "debug/param_tuner.hpp"
#include "debug/web_debugger.hpp"
#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/thread_safe_queue.hpp"

using namespace std::chrono_literals;

namespace {

bool has_env(const char *name) {
  const char *value = std::getenv(name);
  return value != nullptr && value[0] != '\0';
}

bool has_display_server() {
  return has_env("DISPLAY") || has_env("WAYLAND_DISPLAY");
}

bool is_ssh_session() {
  return has_env("SSH_CONNECTION") || has_env("SSH_CLIENT") ||
         has_env("SSH_TTY");
}

bool yaml_bool_or(const YAML::Node &node, bool default_value) {
  return node ? node.as<bool>() : default_value;
}

} // namespace

const std::string keys =
    "{help h usage ? |                        | 输出命令行参数说明}"
    "{@config-path   | configs/sentry.yaml | 位置参数，yaml配置文件路径 }"
    "{headless       | false | 关闭本地窗口 }"
    "{web            | auto  | Web: auto/on/off }"
    "{benchmark-seconds | 0 | 测量秒数，0=关闭 }"
    "{benchmark-warmup | 5 | 测量前预热秒数 }"
    "{benchmark-output | | 测量结果 JSON 路径 }";

int main(int argc, char *argv[]) {
  tools::Exiter exiter;

  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  const double benchmark_seconds = cli.get<double>("benchmark-seconds");
  const double benchmark_warmup = cli.get<double>("benchmark-warmup");
  const auto benchmark_output = cli.get<std::string>("benchmark-output");
  const auto web_override = cli.get<std::string>("web");
  if (!cli.check() || !std::isfinite(benchmark_seconds) || benchmark_seconds < 0 ||
      !std::isfinite(benchmark_warmup) || benchmark_warmup < 0 ||
      (web_override != "auto" && web_override != "on" && web_override != "off")) {
    cli.printErrors();
    tools::logger()->error("Invalid benchmark or Web options");
    return 1;
  }
  auto config = YAML::LoadFile(config_path);
  if (web_override != "auto")
    config["debug_bus"]["sinks"]["web"]["enabled"] = web_override == "on";
  auto plotter_host = config["plotter"] && config["plotter"]["host"]
                          ? config["plotter"]["host"].as<std::string>()
                          : "127.0.0.1";
  auto plotter_port = config["plotter"] && config["plotter"]["port"]
                          ? config["plotter"]["port"].as<uint16_t>()
                          : 9870;
  tools::Plotter plotter(plotter_host, plotter_port);

  auto debug_display_config = config["debug_display"];
  auto window_config =
      debug_display_config ? debug_display_config["window"] : YAML::Node();

  auto window_enabled = !cli.get<bool>("headless") && yaml_bool_or(window_config["enabled"], true);
  auto window_auto_detect = yaml_bool_or(window_config["auto_detect"], true);
  if (window_enabled && window_auto_detect && !has_display_server()) {
    tools::logger()->warn("{} display server found, local debug window "
                          "disabled. Use the web debugger instead.",
                          is_ssh_session() ? "SSH session without" : "No");
    window_enabled = false;
  }
  if (window_enabled) {
    cv::namedWindow("reprojection", cv::WINDOW_NORMAL);
  }

  debug::ParamTuner::instance().load_from_yaml(config_path);
  debug::DebugBus::instance().load_config(config);

  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);

  auto_aim::YOLO yolo(config_path, window_enabled);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Planner planner(config_path);

  tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);
  target_queue.push(std::nullopt);

  std::unique_ptr<debug::BenchmarkSession> benchmark;
  if (benchmark_seconds > 0) {
    benchmark = std::make_unique<debug::BenchmarkSession>(benchmark_warmup, benchmark_seconds);
    debug::active_benchmark.store(benchmark.get(), std::memory_order_release);
    tools::logger()->info("Benchmark: {} s warmup + {} s measurement", benchmark_warmup, benchmark_seconds);
  }
  std::atomic<bool> quit = false;
  auto plan_thread = std::thread([&]() {
    auto t0 = std::chrono::steady_clock::now();
    uint16_t last_bullet_count = 0;
    auto previous_plan = t0;

    while (!quit) {
      const auto plan_start = std::chrono::steady_clock::now();
      debug::benchmark_record(debug::BenchmarkMetric::PlanPeriod,
                              debug::benchmark_ms(plan_start - previous_plan));
      previous_plan = plan_start;
      auto target = target_queue.front();
      auto gs = gimbal.state();
      auto plan = planner.plan(target, gs.bullet_speed);

      gimbal.send(plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc,
                  plan.pitch, plan.pitch_vel, plan.pitch_acc);

      auto fired = gs.bullet_count > last_bullet_count;
      last_bullet_count = gs.bullet_count;

      debug::TelemetrySample telemetry;
      telemetry.target_valid = target.has_value();
      telemetry.control = plan.control;
      nlohmann::json data;
      data["t"] = telemetry.t = tools::delta_time(std::chrono::steady_clock::now(), t0);

      data["gimbal_yaw"] = telemetry.gimbal_yaw = -gs.yaw;
      data["gimbal_yaw_vel"] = telemetry.gimbal_yaw_vel = gs.yaw_vel;
      data["gimbal_pitch"] = telemetry.gimbal_pitch = gs.pitch;
      data["gimbal_pitch_vel"] = telemetry.gimbal_pitch_vel = gs.pitch_vel;

      data["target_yaw"] = telemetry.target_yaw = plan.target_yaw;
      data["target_pitch"] = telemetry.target_pitch = plan.target_pitch;

      data["plan_yaw"] = telemetry.plan_yaw = plan.yaw;
      data["plan_yaw_vel"] = telemetry.plan_yaw_vel = plan.yaw_vel;
      data["plan_yaw_acc"] = telemetry.plan_yaw_acc = plan.yaw_acc;

      data["plan_pitch"] = telemetry.plan_pitch = plan.pitch;
      data["plan_pitch_vel"] = telemetry.plan_pitch_vel = plan.pitch_vel;
      data["plan_pitch_acc"] = telemetry.plan_pitch_acc = plan.pitch_acc;

      data["fire"] = telemetry.fire = plan.fire ? 1 : 0;
      data["fired"] = telemetry.fired = fired ? 1 : 0;

      if (target.has_value()) {
        const auto x = target->ekf_x();
        data["target_z"] = telemetry.target_z = x[4];  // z
        data["target_vz"] = telemetry.target_vz = x[5]; // vz

        if (target->name == auto_aim::ArmorName::outpost && x.size() >= 13) {
          telemetry.outpost_valid = true;
          data["target_last_id"] = telemetry.target_last_id = target->last_id;
          data["target_update_count"] = telemetry.target_update_count = target->update_count();
          data["target_jumped"] = telemetry.target_jumped = target->jumped ? 1 : 0;
          data["target_z0"] = telemetry.target_z0 = x[4];
          data["target_z1"] = telemetry.target_z1 = x[11];
          data["target_z2"] = telemetry.target_z2 = x[12];
          data["target_z0_var"] = telemetry.target_z0_var = target->ekf().P(4, 4);
          data["target_z1_var"] = telemetry.target_z1_var = target->ekf().P(11, 11);
          data["target_z2_var"] = telemetry.target_z2_var = target->ekf().P(12, 12);
          data["reprojection_aim_id"] = telemetry.reprojection_aim_id = planner.reprojection_aim_id(*target);
        }
      }

      if (target.has_value()) {
        data["w"] = telemetry.w = target->ekf_x()[7];
      } else {
        data["w"] = telemetry.w = 0.0;
      }

      const auto telemetry_start = std::chrono::steady_clock::now();
      debug::DebugBus::instance().post_telemetry(telemetry);
      debug::benchmark_record(debug::BenchmarkMetric::TelemetrySubmit,
                              debug::benchmark_ms(std::chrono::steady_clock::now() - telemetry_start));
      plotter.plot(data);
      debug::benchmark_record(debug::BenchmarkMetric::PlanWork,
                              debug::benchmark_ms(std::chrono::steady_clock::now() - plan_start));

      std::this_thread::sleep_for(10ms);
    }
  });

  cv::Mat img;
  std::chrono::steady_clock::time_point t;

  // FPS 统计
  int frame_count = 0;
  double fps = 0.0;
  std::string fps_text = "FPS: --";
  auto fps_timer = std::chrono::steady_clock::now();

  auto previous_frame = std::chrono::steady_clock::now();
  while (!exiter.exit() && !(benchmark && benchmark->finished())) {
    const auto frame_start = std::chrono::steady_clock::now();
    debug::benchmark_record(debug::BenchmarkMetric::FramePeriod,
                            debug::benchmark_ms(frame_start - previous_frame));
    previous_frame = frame_start;
    camera.read(img, t);
    auto proc_start = std::chrono::steady_clock::now();
    auto q = gimbal.q(t);

    solver.set_R_gimbal2world(q);
    auto armors = yolo.detect(img);
    std::vector<debug::ReprojectionData> reprojections;

    auto targets = tracker.track(armors, t);
    if (!targets.empty())
      target_queue.push(targets.front());
    else
      target_queue.push(std::nullopt);

    const auto debug_start = std::chrono::steady_clock::now();
    debug::benchmark_record(debug::BenchmarkMetric::Algorithm,
                            debug::benchmark_ms(debug_start - proc_start));
    auto frame_ticket = debug::DebugBus::instance().try_begin_frame();
    cv::Mat annotated;
    if (window_enabled) annotated = img.clone();
    if ((window_enabled || frame_ticket) && !targets.empty()) {
      auto target = targets.front();

      // 当前帧target更新后
      std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
      for (const Eigen::Vector4d &xyza : armor_xyza_list) {
        auto image_points = solver.reproject_armor(
            xyza.head(3), xyza[3], target.armor_type, target.name);
        if (frame_ticket) reprojections.push_back({image_points});
        if (window_enabled) tools::draw_points(annotated, image_points, {0, 255, 0});
      }

      int aim_id = planner.reprojection_aim_id(target);
      if (window_enabled && target.name == auto_aim::ArmorName::outpost &&
          armor_xyza_list.size() == 3) {
        if (target.last_id >= 0 &&
            target.last_id < static_cast<int>(armor_xyza_list.size())) {
          const auto &last_xyza = armor_xyza_list[target.last_id];
          auto last_image_points = solver.reproject_armor(
              last_xyza.head(3), last_xyza[3], target.armor_type, target.name);
          tools::draw_points(annotated, last_image_points, {255, 255, 0}, 3);
        }

        const auto x = target.ekf_x();
        tools::draw_text(
            annotated,
            fmt::format("outpost last={} aim={} jumped={} w={:.2f}",
                        target.last_id, aim_id, target.jumped ? 1 : 0, x[7]),
            {20, 35}, {255, 255, 0}, 0.8, 2);
        if (x.size() >= 13) {
          tools::draw_text(
              annotated,
              fmt::format("z0={:.3f} z1={:.3f} z2={:.3f} upd={}", x[4],
                          x[11], x[12], target.update_count()),
              {20, 70}, {255, 255, 0}, 0.8, 2);
        }
      }

      Eigen::Vector4d aim_xyza = armor_xyza_list[aim_id];
      auto image_points = solver.reproject_armor(
          aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
      if (frame_ticket) reprojections.push_back({image_points});
      // 红框最后画，避免 aim_id == last_id 时被青色诊断框盖住。
      if (window_enabled) tools::draw_points(annotated, image_points, {0, 0, 255}, 2);
    }

    auto proc_ms = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - proc_start)
                       .count();

    const auto submit_start = std::chrono::steady_clock::now();
    if (frame_ticket)
      debug::DebugBus::instance().post(std::move(frame_ticket),
          {img, &armors, std::move(reprojections), proc_ms, t});
    const auto debug_end = std::chrono::steady_clock::now();
    debug::benchmark_record(debug::BenchmarkMetric::FrameSubmit,
                            debug::benchmark_ms(debug_end - submit_start));
    debug::benchmark_record(debug::BenchmarkMetric::DebugPrepare,
                            debug::benchmark_ms(debug_end - debug_start));
    debug::benchmark_record(debug::BenchmarkMetric::FrameAge,
                            debug::benchmark_ms(debug_end - t));

    if (window_enabled) {
      cv::Mat display_img;
      cv::resize(annotated, display_img, {}, 0.5, 0.5);  // 显示时缩小图片尺寸

      // FPS 计算（每秒更新一次）
      frame_count++;
      auto now = std::chrono::steady_clock::now();
      auto elapsed = std::chrono::duration<double>(now - fps_timer).count();
      if (elapsed >= 1.0) {
        fps = frame_count / elapsed;
        frame_count = 0;
        fps_timer = now;
        fps_text = fmt::format("FPS:{:.0f}", fps);
      }

      const int cx = display_img.cols / 2;
      const int cy = display_img.rows / 2;
      const int arm = 12;
      const cv::Scalar red(0, 0, 255);
      cv::line(display_img, {cx - arm, cy}, {cx + arm, cy}, red, 1);
      cv::line(display_img, {cx, cy - arm}, {cx, cy + arm}, red, 1);

      cv::putText(
        display_img, fps_text, {display_img.cols - 72, 18},
        cv::FONT_HERSHEY_SIMPLEX, 0.45, {255, 255, 0}, 1, cv::LINE_8);

      cv::imshow("reprojection", display_img);
      auto key = cv::waitKey(1);
      if (key == 'q')
        break;
    }
    debug::benchmark_record(debug::BenchmarkMetric::FrameLoop,
                            debug::benchmark_ms(std::chrono::steady_clock::now() - frame_start));
  }

  const auto measurement_stopped_at = std::chrono::steady_clock::now();
  quit = true;
  if (plan_thread.joinable())
    plan_thread.join();
  gimbal.send(false, false, 0, 0, 0, 0, 0, 0);

  debug::DebugBus::instance().shutdown();
  debug::active_benchmark.store(nullptr, std::memory_order_relaxed);
  if (benchmark) {
    auto result = benchmark->summary(measurement_stopped_at);
    const auto stats = debug::DebugBus::instance().stats();
#define WEB_STAT(name) result["debug_run_counters"][#name] = stats.name
    WEB_STAT(frames_accepted); WEB_STAT(frames_overwritten); WEB_STAT(frames_busy);
    WEB_STAT(frames_sent); WEB_STAT(video_client_dropped); WEB_STAT(telemetry_accepted);
    WEB_STAT(telemetry_full_dropped); WEB_STAT(telemetry_expired_dropped);
    WEB_STAT(telemetry_client_dropped); WEB_STAT(telemetry_batches);
    WEB_STAT(frame_copy_count); WEB_STAT(frame_copy_total_ns); WEB_STAT(frame_copy_max_ns);
    WEB_STAT(encode_count); WEB_STAT(encode_total_ns); WEB_STAT(encode_max_ns);
    WEB_STAT(raw_frame_allocations); WEB_STAT(raw_frame_reuses);
#undef WEB_STAT
    result["image_width"] = img.cols;
    result["image_height"] = img.rows;
    result["config"] = config_path;
    result["web_override"] = web_override;
    result["warmup_seconds"] = benchmark_warmup;
    if (!benchmark_output.empty()) {
      std::ofstream output(benchmark_output);
      if (!output || !(output << result.dump(2) << '\n')) {
        tools::logger()->error("Cannot write benchmark result: {}", benchmark_output);
        return 1;
      }
    }
    tools::logger()->info("Benchmark result: {}", result.dump());
  }
  return 0;
}
