// 实车流水线：相机/云台 -> S2 全片检测 -> S1 编排 S3/S4/S5/S6 -> 串口发送。
// 四个窗口只消费同一次 process 的快照，不重复运行算法，也不暂停控制循环。
// 用法：auto_buff_debug_mpc configs/standard4.yaml [--record]；q/Esc 退出。
#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/buff_processor.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"

namespace
{
using Clock = std::chrono::steady_clock;
constexpr double RAD_TO_DEG = 180.0 / CV_PI;
constexpr double PREVIEW_SECONDS = 0.5;
const std::array<std::string, 4> WINDOW_NAMES = {"kalman", "bullet", "debug", "replay"};

const std::string keys =
  "{help h usage ? | | 输出命令行参数说明}"
  "{@config-path   | | 实车相机、串口和标定 yaml 配置}"
  "{buff-config    | configs/auto_buff.yaml | 大符流水线配置}"
  "{headless       | false | 关闭本地四窗口}"
  "{record         | false | 保存原图 avi 和对应 IMU txt 到 records/}"
  "{frames         | 0 | 处理帧数上限，0 为持续运行}"
  "{window-width   | 800 | 单个窗口宽度}"
  "{window-height  | 500 | 单个窗口高度}";

bool has_display()
{
  const auto present = [](const char * name) {
    const auto value = std::getenv(name);
    return value != nullptr && value[0] != '\0';
  };
  return present("DISPLAY") || present("WAYLAND_DISPLAY");
}

const char * mode_name(io::GimbalMode mode)
{
  switch (mode) {
    case io::GimbalMode::SMALL_BUFF: return "SMALL_BUFF";
    case io::GimbalMode::BIG_BUFF: return "BIG_BUFF";
    case io::GimbalMode::AUTO_AIM: return "AUTO_AIM";
    case io::GimbalMode::IDLE: return "IDLE";
  }
  return "UNKNOWN";
}

const char * activation_name(auto_buff::BuffActivation activation)
{
  switch (activation) {
    case auto_buff::BuffActivation::INACTIVE: return "UNHIT";
    case auto_buff::BuffActivation::SMALL_ACTIVATED: return "Small-A";
    case auto_buff::BuffActivation::BIG_ACTIVATED: return "Big-A";
    case auto_buff::BuffActivation::INVALID: return "INVALID";
  }
  return "UNKNOWN";
}

const char * source_name(auto_buff::PoseSource source)
{
  switch (source) {
    case auto_buff::PoseSource::PNP: return "PNP";
    case auto_buff::PoseSource::CHAMFER_REFINED: return "CHAMFER_REFINED";
    case auto_buff::PoseSource::UNKNOWN: return "UNKNOWN";
  }
  return "UNKNOWN";
}

void label(
  cv::Mat & image, const std::string & text, cv::Point point,
  const cv::Scalar & color = {255, 255, 255}, double scale = 0.55, int thickness = 1)
{
  cv::putText(image, text, point, cv::FONT_HERSHEY_SIMPLEX,
    scale, {0, 0, 0}, thickness + 2, cv::LINE_AA);
  cv::putText(image, text, point, cv::FONT_HERSHEY_SIMPLEX,
    scale, color, thickness, cv::LINE_AA);
}

// 原图通常高于 1000px；先放大图内标注，避免缩放到四窗口后文字变成几个像素。
void frame_label(
  cv::Mat & image, const std::string & text, cv::Point point,
  const cv::Scalar & color = {255, 255, 255}, double scale = 0.55)
{
  const double factor = std::max(1.0, image.rows / 400.0);
  label(image, text, point, color, scale * factor, std::max(1, cvRound(factor)));
}

bool in_image(const cv::Point2f & point, const cv::Mat & image)
{
  return std::isfinite(point.x) && std::isfinite(point.y) && point.x >= 0 && point.y >= 0 &&
         point.x < image.cols && point.y < image.rows;
}

// 与 S4 相同的标定/IMU 约定，只负责将世界坐标调试几何投回当前帧。
class Projector
{
public:
  explicit Projector(const YAML::Node & config)
  {
    const auto intrinsics = config["camera_matrix"].as<std::vector<double>>();
    const auto distortion = config["distort_coeffs"].as<std::vector<double>>();
    const auto rotation = config["R_camera2gimbal"].as<std::vector<double>>();
    const auto translation = config["t_camera2gimbal"].as<std::vector<double>>();
    const auto imu_rotation = config["R_gimbal2imubody"].as<std::vector<double>>();
    if (intrinsics.size() != 9 || rotation.size() != 9 || translation.size() != 3 ||
        imu_rotation.size() != 9) {
      throw std::invalid_argument("Invalid camera calibration sizes");
    }
    camera_matrix_ = cv::Mat(intrinsics).reshape(1, 3).clone();
    distortion_ = cv::Mat(distortion).reshape(1, 1).clone();
    using RowMajor = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;
    camera_to_gimbal_ = Eigen::Map<const RowMajor>(rotation.data());
    translation_ = Eigen::Map<const Eigen::Vector3d>(translation.data());
    gimbal_to_imu_ = Eigen::Map<const RowMajor>(imu_rotation.data());
  }

  void set_imu(const Eigen::Quaterniond & q)
  {
    gimbal_to_world_ =
      gimbal_to_imu_.transpose() * q.normalized().toRotationMatrix() * gimbal_to_imu_;
  }

  std::optional<cv::Point2f> project(const Eigen::Vector3d & world) const
  {
    const Eigen::Vector3d camera = camera_to_gimbal_.transpose() *
      (gimbal_to_world_.transpose() * world - translation_);
    if (!camera.allFinite() || camera.z() <= 1e-6) return std::nullopt;
    std::vector<cv::Point2d> pixels;
    cv::projectPoints(
      std::vector<cv::Point3d>{{camera.x(), camera.y(), camera.z()}}, cv::Vec3d::all(0),
      cv::Vec3d::all(0), camera_matrix_, distortion_, pixels);
    if (!std::isfinite(pixels[0].x) || !std::isfinite(pixels[0].y) ||
        std::abs(pixels[0].x) > 1e6 || std::abs(pixels[0].y) > 1e6) {
      return std::nullopt;
    }
    return cv::Point2f(pixels[0]);
  }

private:
  cv::Mat camera_matrix_, distortion_;
  Eigen::Matrix3d camera_to_gimbal_, gimbal_to_imu_;
  Eigen::Matrix3d gimbal_to_world_ = Eigen::Matrix3d::Identity();
  Eigen::Vector3d translation_;
};

void draw_detections(cv::Mat & image, const std::vector<auto_buff::BuffObservation> & observations)
{
  constexpr std::array<const char *, 5> POINT_NAMES = {"0", "1", "2", "3", "R"};
  const cv::Rect bounds(0, 0, image.cols, image.rows);
  for (std::size_t i = 0; i < observations.size(); ++i) {
    const auto & observation = observations[i];
    const cv::Scalar color = observation.activation == auto_buff::BuffActivation::INACTIVE ?
      cv::Scalar(0, 255, 255) : cv::Scalar(0, 255, 0);
    std::vector<cv::Point2f> points;
    for (std::size_t j = 0; j < observation.points.size(); ++j) {
      const auto & point = observation.points[j];
      if (!in_image(point, image)) continue;
      points.push_back(point);
      cv::circle(image, point, std::max(4, image.rows / 120), color, 2, cv::LINE_AA);
      frame_label(image, POINT_NAMES[j], cv::Point(point) + cv::Point(10, -10), color, 0.45);
    }
    if (points.empty()) continue;
    const auto box = cv::boundingRect(points) & bounds;
    cv::rectangle(image, box, color, 1);
    const int title_offset = 12 + static_cast<int>(i % 3) * std::max(30, image.rows * 30 / 400);
    frame_label(image, fmt::format("#{} {} {:.2f}", i, activation_name(observation.activation),
      observation.confidence), {box.x, std::max(30, box.y - title_offset)}, color);
  }
}

void draw_refined(cv::Mat & image, const std::vector<auto_buff::RefinedObservation> & observations)
{
  for (std::size_t i = 0; i < observations.size(); ++i) {
    const auto & observation = observations[i];
    const cv::Scalar color = observation.valid() ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
    if (!observation.roi.empty()) cv::rectangle(image, observation.roi, color, 1);
    const auto contour = [&](const auto_buff::ContourFeature & feature, const char * name,
                             const cv::Scalar & feature_color) {
      if (!feature.usable()) return;
      cv::drawContours(image, std::vector<std::vector<cv::Point>>{feature.contour}, -1,
        feature_color, 2, cv::LINE_AA);
      frame_label(image, name, feature.contour.front(), feature_color, 0.45);
    };
    contour(observation.armor_contour, "Armor", {0, 255, 0});
    contour(observation.r_mark_contour, "R mark", {255, 0, 255});
    contour(observation.light_arm_contour, "LightArm", {255, 255, 0});
    for (const auto & point : observation.points) {
      if (in_image(point, image)) cv::circle(image, point, 3, color, 1, cv::LINE_AA);
    }
    if (in_image(observation.points[0], image)) {
      frame_label(image, fmt::format("#{} {} {}", i, auto_buff::to_string(observation.quality),
        auto_buff::to_string(observation.reject_reason)),
        cv::Point(observation.points[0]) + cv::Point(6, -10), color, 0.45);
    }
  }
}

void draw_reconstruction(
  cv::Mat & image, const auto_buff::RuneReconstruction & reconstruction,
  const std::optional<auto_buff::RuneSlot> & target_slot, const Projector & projector)
{
  const auto & pose = reconstruction.pose;
  const auto world = [&](const Eigen::Vector3d & local) -> Eigen::Vector3d {
    return pose.position_world_m + pose.orientation_world * local;
  };
  const auto center = projector.project(pose.position_world_m);
  if (center) cv::drawMarker(image, *center, {0, 255, 255}, cv::MARKER_CROSS, 20, 2);

  std::vector<cv::Point> ring;
  for (int slot = 1; slot <= 5; ++slot) {
    const Eigen::AngleAxisd rotation((slot - 1) * 2.0 * CV_PI / 5.0, Eigen::Vector3d::UnitX());
    const auto armor = projector.project(world(rotation * Eigen::Vector3d(0, 0, pose.blade_radius_m)));
    if (!armor) continue;
    ring.emplace_back(*armor);
    const bool selected = target_slot && static_cast<int>(*target_slot) == slot;
    const bool observed = std::any_of(reconstruction.blades.begin(), reconstruction.blades.end(),
      [slot](const auto & blade) { return static_cast<int>(blade.slot) == slot; });
    const cv::Scalar color = selected ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 255, 255);
    constexpr double HALF_DIAGONAL = 0.127;
    const std::array<Eigen::Vector3d, 4> corners = {{
      {0, 0, pose.blade_radius_m + HALF_DIAGONAL},
      {0, HALF_DIAGONAL, pose.blade_radius_m},
      {0, 0, pose.blade_radius_m - HALF_DIAGONAL},
      {0, -HALF_DIAGONAL, pose.blade_radius_m}}};
    std::vector<cv::Point> pixels;
    for (const auto & corner : corners) {
      const auto pixel = projector.project(world(rotation * corner));
      if (!pixel) {
        pixels.clear();
        break;
      }
      pixels.emplace_back(*pixel);
    }
    if (pixels.size() == 4) {
      cv::polylines(image, pixels, true, color, 2, cv::LINE_AA);
      if (selected) {
        constexpr std::array<const char *, 4> NAMES = {"0", "1", "2", "3"};
        for (std::size_t i = 0; i < pixels.size(); ++i) {
          frame_label(image, NAMES[i], pixels[i] + cv::Point(10, -10), {0, 255, 255}, 0.45);
        }
      }
    }
    cv::circle(image, *armor, 4, color, -1, cv::LINE_AA);
    const auto text = fmt::format("slot {}{}{}", slot,
      observed ? "" : " model", selected ? " target" : "");
    const double text_scale = 0.5 * std::max(1.0, image.rows / 400.0);
    int baseline = 0;
    const auto text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX,
      text_scale, std::max(1, image.rows / 400), &baseline);
    // 槽位文字向盘面外侧放置；小目标的相邻槽位也能分别读出。
    const int gap = std::max(12, image.rows / 40);
    const int dx = center && armor->x < center->x ? -text_size.width - gap : gap;
    const int dy = center && armor->y < center->y ? -gap : text_size.height + gap;
    const cv::Point origin(std::clamp(cvRound(armor->x) + dx, 0,
      std::max(0, image.cols - text_size.width)),
      std::clamp(cvRound(armor->y) + dy, text_size.height, image.rows - 1));
    frame_label(image, text, origin, color, 0.5);
    if (selected && center) cv::line(image, *center, *armor, color, 1, cv::LINE_AA);
  }
  if (ring.size() == 5) cv::polylines(image, ring, true, {0, 255, 255}, 1, cv::LINE_AA);
}

// 仅为 S5 模型预览；使用目标自身的稳定相位基，不冒充 S6 的命中点/弹道解。
Eigen::Vector3d preview_position(const auto_buff::RuneTarget & target, double seconds)
{
  const auto & motion = target.motion;
  double phase = motion.continuous_phase_rad + seconds * motion.angular_velocity_rad_s;
  if (motion.model == auto_buff::MotionModel::HARMONIC_PHASE) {
    const double t = std::chrono::duration<double>(
      motion.reference_timestamp - motion.model_epoch).count() + seconds;
    const auto & h = motion.harmonic_phase;
    phase = h.cos_amplitude_rad * std::cos(h.angular_frequency_rad_s * t) +
      h.sin_amplitude_rad * std::sin(h.angular_frequency_rad_s * t) +
      h.linear_rate_rad_s * t + h.phase_offset_rad;
  }
  const auto & pose = target.pose;
  return pose.position_world_m + pose.blade_radius_m *
    (Eigen::AngleAxisd(phase, pose.plane_normal_world) * pose.phase_zero_direction_world);
}

cv::Mat panel(const cv::Mat & image, cv::Size size, const std::vector<std::string> & lines)
{
  const int header = 18 + 23 * static_cast<int>(lines.size());
  cv::Mat result(size, CV_8UC3, cv::Scalar(18, 18, 18));
  const double scale = std::min(static_cast<double>(size.width) / image.cols,
    static_cast<double>(size.height - header) / image.rows);
  const cv::Size fitted(
    std::max(1, cvRound(image.cols * scale)), std::max(1, cvRound(image.rows * scale)));
  const cv::Rect area((size.width - fitted.width) / 2,
    header + (size.height - header - fitted.height) / 2, fitted.width, fitted.height);
  cv::resize(image, result(area), fitted);
  for (std::size_t i = 0; i < lines.size(); ++i) {
    int baseline = 0;
    const auto text_size = cv::getTextSize(lines[i], cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    const double text_scale = 0.5 * std::min(1.0,
      static_cast<double>(size.width - 20) / std::max(1, text_size.width));
    label(result, lines[i], {10, 25 + 23 * static_cast<int>(i)}, {255, 255, 255}, text_scale);
  }
  return result;
}

std::array<cv::Mat, 4> render_views(
  const auto_buff::BuffInput & input, const auto_buff::BuffDebugInfo & debug,
  const io::VisionToGimbal & output, Projector & projector, cv::Size size,
  int frame, double fps, double processing_ms, bool recording)
{
  cv::Mat detection = input.img.clone(), kalman = input.img.clone(), bullet = input.img.clone();
  draw_detections(detection, input.observations);
  draw_refined(kalman, debug.refined_observations);
  if (debug.reconstruction) {
    projector.set_imu(input.imu_q);
    draw_reconstruction(bullet, *debug.reconstruction, debug.target_slot, projector);
  }
  if (debug.target) {
    const auto current = projector.project(preview_position(*debug.target, 0));
    if (current) cv::drawMarker(kalman, *current, {255, 0, 255}, cv::MARKER_CROSS, 20, 2);
    std::optional<cv::Point2f> previous;
    for (int i = 0; i <= 20; ++i) {
      const auto next = projector.project(preview_position(*debug.target, PREVIEW_SECONDS * i / 20));
      if (previous && next) cv::line(bullet, *previous, *next, {0, 255, 255}, 2, cv::LINE_AA);
      previous = next;
    }
    if (previous) {
      cv::drawMarker(bullet, *previous, {0, 255, 255}, cv::MARKER_TILTED_CROSS, 22, 2);
      frame_label(bullet, "S5 +0.50s", cv::Point(*previous) + cv::Point(10, -10), {0, 255, 255});
    }
  }

  const auto valid = std::count_if(debug.refined_observations.begin(), debug.refined_observations.end(),
    [](const auto & observation) { return observation.valid(); });
  const auto & motion = debug.motion;
  const std::string geometry = debug.reconstruction ?
    fmt::format("S4 {} | {} blades | mean error {:.2f}px",
      source_name(debug.reconstruction->pose.source),
      debug.reconstruction->blades.size(), debug.reconstruction->pose.reprojection_error_px) :
    "S4 unavailable (requires >=2 valid blades)";
  const std::string phase = debug.target ?
    fmt::format("phase {:.3f}rad | speed {:.3f}rad/s | {}", debug.target->motion.continuous_phase_rad,
      debug.target->motion.angular_velocity_rad_s,
      debug.target->motion.model == auto_buff::MotionModel::HARMONIC_PHASE ? "HARMONIC" : "KALMAN") :
    "S5 model unavailable";
  const bool buff_mode = input.gimbal_mode == io::GimbalMode::SMALL_BUFF ||
    input.gimbal_mode == io::GimbalMode::BIG_BUFF;
  const std::string status = !buff_mode ? "WAITING FOR BUFF MODE" :
    (debug.reset ? "RESET (input / history)" : "RUNNING");
  const auto common = fmt::format("frame {} | {} | {:.1f} fps | {:.1f}ms",
    frame, mode_name(input.gimbal_mode), fps, processing_ms);
  return {{
    panel(kalman, size, {fmt::format("S3 valid {}/{} | {}",
      valid, debug.refined_observations.size(), geometry),
      fmt::format("S5 {} | n={} span={:.2f}s | fit RMS {}rad",
        motion.status, motion.samples, motion.span_s,
        std::isfinite(motion.fit_rms_rad) ? fmt::format("{:.3f}", motion.fit_rms_rad) : "--"),
      phase, "Armor GREEN | R MAGENTA | LightArm CYAN | filtered target MAGENTA"}),
    panel(bullet, size, {geometry, "S6 intercept / ballistics / firing: NOT IMPLEMENTED",
      fmt::format("TX mode={} | yaw={:.2f} pitch={:.2f}deg | bullet={:.1f}m/s", output.mode,
        output.yaw * RAD_TO_DEG, output.pitch * RAD_TO_DEG, input.gimbal_state.bullet_speed),
      "RED selected | WHITE geometry | YELLOW S5 +0.50s preview (no ballistics)"}),
    panel(detection, size, {"S2 detector | all candidates | 0 top / 1 right / 2 bottom / 3 left / R",
      fmt::format("{} | {} candidates | {}", common, input.observations.size(), status),
      "UNHIT = inactive target | Small-A / Big-A = activated", "q / Esc / close window: exit"}),
    panel(input.img, size, {"Camera original | no detection / contour overlays", common,
      fmt::format("RX yaw={:.2f} pitch={:.2f}deg | bullets={} | IMU norm={:.4f}",
        input.gimbal_state.yaw * RAD_TO_DEG, input.gimbal_state.pitch * RAD_TO_DEG,
        input.gimbal_state.bullet_count, input.imu_q.norm()),
      recording ? "REC: raw AVI + timestamp / IMU TXT in records/" :
        "Recording OFF | --record to save raw frames / IMU"})}};
}

nlohmann::json telemetry(
  const auto_buff::BuffInput & input, const auto_buff::BuffDebugInfo & debug,
  const io::VisionToGimbal & output, double seconds, double processing_ms, double fps)
{
  nlohmann::json data = {
    {"t", seconds}, {"mode", output.mode}, {"gimbal_mode", static_cast<int>(input.gimbal_mode)},
    {"yaw", output.yaw * RAD_TO_DEG}, {"pitch", output.pitch * RAD_TO_DEG},
    {"yaw_vel", output.yaw_vel * RAD_TO_DEG}, {"pitch_vel", output.pitch_vel * RAD_TO_DEG},
    {"gimbal_yaw", input.gimbal_state.yaw * RAD_TO_DEG},
    {"gimbal_pitch", input.gimbal_state.pitch * RAD_TO_DEG},
    {"bullet_speed", input.gimbal_state.bullet_speed}, {"bullet_count", input.gimbal_state.bullet_count},
    {"detections", input.observations.size()},
    {"reconstruction_valid", debug.reconstruction.has_value()},
    {"motion_valid", debug.target.has_value()}, {"motion_samples", debug.motion.samples},
    {"processing_ms", processing_ms}, {"fps", fps}};
  data["reprojection_error_px"] = debug.reconstruction ?
    nlohmann::json(debug.reconstruction->pose.reprojection_error_px) : nlohmann::json(nullptr);
  data["phase_rad"] = debug.target ?
    nlohmann::json(debug.target->motion.continuous_phase_rad) : nlohmann::json(nullptr);
  data["omega_rad_s"] = debug.target ?
    nlohmann::json(debug.target->motion.angular_velocity_rad_s) : nlohmann::json(nullptr);
  return data;
}

// 控制循环退出或栈展开时撤销控制；底层阻塞采集/串口查询仍需先返回。
struct IdleOnExit
{
  io::Gimbal & gimbal;
  bool stopped = false;
  void stop() noexcept
  {
    if (stopped) return;
    stopped = true;
    try {
      gimbal.send(io::VisionToGimbal{});
    } catch (const std::exception & e) {
      tools::logger()->error("Send idle on exit: {}", e.what());
    }
  }
  ~IdleOnExit() { stop(); }
};
}  // namespace

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  const auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }
  const auto buff_config = cli.get<std::string>("buff-config");
  const auto frame_limit = cli.get<int>("frames");
  const cv::Size window_size(cli.get<int>("window-width"), cli.get<int>("window-height"));
  const bool record = cli.get<bool>("record");
  bool window_enabled = !cli.get<bool>("headless");
  if (!cli.check() || frame_limit < 0 || window_size.width < 400 || window_size.height < 240) {
    cli.printErrors();
    tools::logger()->error("frames must be >= 0; window size must be >= 400x240");
    return 1;
  }
  if (window_enabled && !has_display()) {
    tools::logger()->warn("No display server; four debug windows disabled");
    window_enabled = false;
  }

  try {
    const auto config = YAML::LoadFile(config_path);
    tools::Exiter exiter;
    tools::Plotter plotter;
    auto_buff::Buff_Detector detector(config_path);
    auto_buff::BuffProcessor processor(config_path, buff_config);
    Projector projector(config);
    io::Gimbal gimbal(config_path);
    io::Camera camera(config_path);
    std::unique_ptr<tools::Recorder> recorder;
    if (record) recorder = std::make_unique<tools::Recorder>();
    IdleOnExit idle_on_exit{gimbal};

    if (window_enabled) {
      for (std::size_t i = 0; i < WINDOW_NAMES.size(); ++i) {
        cv::namedWindow(WINDOW_NAMES[i], cv::WINDOW_NORMAL);
        cv::resizeWindow(WINDOW_NAMES[i], window_size.width, window_size.height);
        cv::moveWindow(WINDOW_NAMES[i], static_cast<int>(i % 2) * (window_size.width + 10),
          static_cast<int>(i / 2) * (window_size.height + 65));
      }
    }
    tools::logger()->info("Live buff: {} | S2 -> S3 -> S4 -> S5 -> S6 -> gimbal.send", config_path);
    tools::logger()->info("S6 intercept/ballistics is unimplemented; showing actual processor output");
    if (recorder) {
      tools::logger()->info("Recording {} + {}", recorder->video_path(), recorder->text_path());
    }

    const auto start = Clock::now();
    auto fps_start = start;
    int fps_frames = 0, frame = 0;
    double fps = 0.0;
    while (!exiter.exit() && (frame_limit == 0 || frame < frame_limit)) {
      // 采集：图像时间对应 q(t)，电控档位决定大小符；不在实车入口伪造模式。
      cv::Mat image;
      Clock::time_point timestamp;
      camera.read(image, timestamp);
      if (image.empty()) {
        processor.reset();
        gimbal.send(io::VisionToGimbal{});
        tools::logger()->error("Camera returned an empty frame");
        break;
      }
      const auto q = gimbal.q(timestamp);
      const auto state = gimbal.state();
      const auto mode = gimbal.mode();
      const auto process_start = Clock::now();

      // S2 会就地画图；只传副本，S3、原图窗口和录像始终使用干净图像。
      cv::Mat detector_image = image.clone();
      auto observations = detector.detect_observations(detector_image);
      const auto input = auto_buff::make_buff_input(
        image, timestamp, q, state, mode, std::move(observations));

      // S1 保留统一的校验、重置、S3 精化/S4 重建/S5 运动/S6 决策和协议映射。
      auto_buff::BuffDebugInfo debug;
      const auto output = processor.process(input, &debug);
      gimbal.send(output);
      const double processing_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - process_start).count();
      ++frame;
      ++fps_frames;
      const auto now = Clock::now();
      const double fps_seconds = std::chrono::duration<double>(now - fps_start).count();
      if (fps_seconds >= 1.0) {
        fps = fps_frames / fps_seconds;
        fps_frames = 0;
        fps_start = now;
        tools::logger()->info("Buff {} | {} | {:.1f} fps | {} observations | S5 {} | TX {}",
          frame, mode_name(mode), fps, input.observations.size(), debug.motion.status, output.mode);
      }

      // 旁路诊断在发送之后执行；录制帧拥有独立存储，避免异步写盘读到相机复用缓冲。
      plotter.plot(telemetry(input, debug, output,
        std::chrono::duration<double>(timestamp - start).count(), processing_ms, fps));
      if (recorder) recorder->record(image.clone(), q, timestamp);
      if (window_enabled) {
        const auto views = render_views(input, debug, output, projector, window_size,
          frame, fps, processing_ms, record);
        for (std::size_t i = 0; i < WINDOW_NAMES.size(); ++i) cv::imshow(WINDOW_NAMES[i], views[i]);
        const int key = cv::waitKey(1) & 0xff;
        if (key == 'q' || key == 27) break;
        const bool closed = std::any_of(WINDOW_NAMES.begin(), WINDOW_NAMES.end(), [](const auto & name) {
          return cv::getWindowProperty(name, cv::WND_PROP_VISIBLE) < 1;
        });
        if (closed) break;
      }
    }
    idle_on_exit.stop();
    if (window_enabled) cv::destroyAllWindows();
  } catch (const std::exception & e) {
    tools::logger()->error("auto_buff_debug_mpc: {}", e.what());
    if (window_enabled) cv::destroyAllWindows();
    return 1;
  }
  return 0;
}
