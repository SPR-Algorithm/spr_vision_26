#include <Eigen/Geometry>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include <unistd.h>
#include <fcntl.h>
#include <termios.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "debug/web_debugger.hpp"
#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/yolo11_buff_types.hpp"
#include "tasks/auto_buff/buff_processor.hpp"
#include "tasks/auto_buff/observation_refiner.hpp"
#include "tasks/auto_buff/phase_motion_estimator.hpp"
#include "tasks/auto_buff/rune_reconstructor.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/recorder.hpp"

namespace
{
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
constexpr double PI = 3.14159265358979323846;

struct Options
{
  std::string config = "configs/standard4.yaml";
  std::string buff_config = "configs/auto_buff.yaml";
  std::string mode = "serial";
  std::string video, imu, device, log, backend, model, engine;
  std::size_t frames = 0, start_index = 0, warmup = 0;
  std::optional<int> cuda_device;
  bool headless = false, record = false, help = false, motion = false, pose_prior = true;
  bool check_processor = false;
  std::optional<bool> window;
};

void print_help()
{
  std::cout <<
    "Buff S2/S3/S4 and optional S5 debugger (serial receive only).\n"
    "  --config=PATH       Camera/serial/model YAML (configs/standard4.yaml)\n"
    "  --buff-config=PATH  Refiner YAML (configs/auto_buff.yaml)\n"
    "  --mode=serial|small|big  Serial mode is authoritative by default\n"
    "  --frames=N          Stop after N frames (positive integer)\n"
    "  --start-index=N     Start at this zero-based offline video frame\n"
    "  --motion            Enable S5 phase and motion diagnostics; no control output\n"
    "  --pose-prior=false  Disable the previous-pose PnP branch prior for comparison\n"
    "  --check-processor   Check S1 output using live state, without serial transmission\n"
    "  --headless          Disable both web and local window\n"
    "  --window=true|false Override local window setting\n"
    "  --record            Record clean video and synchronized wxyz IMU\n"
    "  --log=PATH          JSONL output (default records/buff_observation_*.jsonl)\n"
    "  --backend=openvino|tensorrt  Override only Buff inference backend\n"
    "  --model=PATH        Override the OpenVINO model\n"
    "  --engine=PATH       Override engine and select TensorRT\n"
    "  --cuda-device=N     TensorRT CUDA device index (nonnegative integer)\n"
    "  --device=CPU        OpenVINO device override; incompatible with TensorRT\n"
    "  --warmup=N          Repeat detection on the first real frame (default 0)\n"
    "  --video=PATH        Offline video; requires --mode=small or big\n"
    "  --imu=PATH          One 'seconds w x y z' row per offline video frame\n"
    "                      Without --imu, identity IMU is explicitly marked\n"
    "  --help              Show this message\n"
    "Press q in the local window/terminal, or Ctrl+C, to stop.\n";
}

bool parse_bool(const std::string & value)
{
  if (value == "true" || value == "1") return true;
  if (value == "false" || value == "0") return false;
  throw std::invalid_argument("Expected true or false, got: " + value);
}

Options parse_options(int argc, char ** argv)
{
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    const auto equal = argument.find('=');
    const std::string key = argument.substr(0, equal);
    if (key == "--help" || key == "-h") { options.help = true; continue; }
    if (key == "--headless" || key == "--record" || key == "--motion" || key == "--pose-prior" ||
        key == "--check-processor") {
      const bool value = equal == std::string::npos || parse_bool(argument.substr(equal + 1));
      if (key == "--motion") options.motion = value;
      else if (key == "--pose-prior") options.pose_prior = value;
      else if (key == "--check-processor") options.check_processor = value;
      else (key == "--headless" ? options.headless : options.record) = value;
      continue;
    }
    const bool known = key == "--config" || key == "--buff-config" || key == "--mode" ||
      key == "--video" || key == "--imu" || key == "--device" || key == "--log" ||
      key == "--frames" || key == "--start-index" || key == "--window" ||
      key == "--backend" || key == "--model" || key == "--engine" ||
      key == "--cuda-device" || key == "--warmup";
    if (!known) throw std::invalid_argument("Unknown argument: " + argument);
    const std::string value = equal != std::string::npos ? argument.substr(equal + 1) :
      (++i < argc ? argv[i] : "");
    if (value.empty()) throw std::invalid_argument("Missing value for " + key);
    if (key == "--config") options.config = value;
    else if (key == "--buff-config") options.buff_config = value;
    else if (key == "--mode") options.mode = value;
    else if (key == "--video") options.video = value;
    else if (key == "--imu") options.imu = value;
    else if (key == "--device") options.device = value;
    else if (key == "--backend") options.backend = value;
    else if (key == "--model") options.model = value;
    else if (key == "--engine") options.engine = value;
    else if (key == "--log") options.log = value;
    else if (key == "--window") options.window = parse_bool(value);
    else {
      if (value.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument(key + " must be a nonnegative integer");
      }
      const auto count = std::stoull(value);
      if (key == "--start-index") options.start_index = count;
      else if (key == "--warmup") options.warmup = count;
      else if (key == "--cuda-device") {
        if (count > static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
          throw std::invalid_argument("--cuda-device exceeds the supported integer range");
        }
        options.cuda_device = static_cast<int>(count);
      }
      else {
        options.frames = count;
        if (count == 0) throw std::invalid_argument("--frames must be positive");
      }
    }
  }
  if (options.help) return options;
  if (!options.backend.empty() && options.backend != "openvino" && options.backend != "tensorrt") {
    throw std::invalid_argument("--backend must be openvino or tensorrt");
  }
  if (!options.engine.empty() && options.backend == "openvino") {
    throw std::invalid_argument("--engine cannot be combined with --backend=openvino");
  }
  if (!options.engine.empty()) options.backend = "tensorrt";
  if (options.mode != "serial" && options.mode != "small" && options.mode != "big") {
    throw std::invalid_argument("--mode must be serial, small or big");
  }
  if (!options.video.empty() && options.mode == "serial") {
    throw std::invalid_argument("Offline --video requires explicit --mode=small or --mode=big");
  }
  if (!options.video.empty() && options.check_processor) {
    throw std::invalid_argument("--check-processor requires live serial state");
  }
  if (options.video.empty() && !options.imu.empty()) {
    throw std::invalid_argument("--imu requires --video");
  }
  if (options.video.empty() && options.start_index != 0) {
    throw std::invalid_argument("--start-index requires --video");
  }
  for (const auto & path : {options.config, options.buff_config, options.video, options.imu,
                            options.model, options.engine}) {
    if (!path.empty() && !std::filesystem::is_regular_file(path)) {
      throw std::invalid_argument("Input file does not exist: " + path);
    }
  }
  return options;
}

// Buff-only overrides live in a temporary file; camera/serial and auto_aim settings stay intact.
class DetectorConfig
{
public:
  DetectorConfig(const Options & options, const YAML::Node & config) : path_(options.config)
  {
    auto copy = YAML::Clone(config);
    const auto inference = copy["buff_inference"];
    const std::string backend = !options.backend.empty() ? options.backend :
      (inference && inference["backend"] ? inference["backend"].as<std::string>() : "openvino");
    if (backend != "openvino" && backend != "tensorrt") {
      throw std::invalid_argument("buff_inference.backend must be openvino or tensorrt");
    }
    if (backend == "tensorrt" && !options.device.empty()) {
      throw std::invalid_argument("--device applies only to OpenVINO; use --cuda-device for TensorRT");
    }
    if (backend == "tensorrt" && !options.model.empty()) {
      throw std::invalid_argument("--model applies only to OpenVINO; use --engine for TensorRT");
    }
    if (backend != "tensorrt" && options.cuda_device) {
      throw std::invalid_argument("--cuda-device requires the TensorRT backend");
    }
    if (options.device.empty() && options.backend.empty() && options.model.empty() &&
        options.engine.empty() && !options.cuda_device) return;
    copy["buff_inference"]["backend"] = backend;
    if (!options.device.empty()) copy["buff_inference"]["device"] = options.device;
    if (!options.model.empty()) copy["buff_inference"]["model"] = options.model;
    if (!options.engine.empty()) copy["buff_inference"]["engine"] = options.engine;
    if (options.cuda_device) copy["buff_inference"]["cuda_device"] = *options.cuda_device;
    auto pattern = (std::filesystem::temp_directory_path() / "buff-observation-XXXXXX").string();
    std::vector<char> name(pattern.begin(), pattern.end());
    name.push_back('\0');
    const int fd = mkstemp(name.data());
    if (fd < 0) throw std::runtime_error("Cannot create temporary detector config");
    ::close(fd);
    path_ = name.data();
    temporary_ = true;
    std::ofstream output(path_);
    output << copy;
    output.close();
    if (!output) {
      std::filesystem::remove(path_);
      throw std::runtime_error("Cannot write temporary detector config");
    }
  }
  ~DetectorConfig()
  {
    if (temporary_) { std::error_code ignored; std::filesystem::remove(path_, ignored); }
  }
  const std::string & path() const { return path_; }
private:
  std::string path_;
  bool temporary_ = false;
};

Json inference_json(const auto_buff::BuffInferenceDiagnostics & diagnostics)
{
  const auto & info = diagnostics.info;
  const auto & timing = diagnostics.timings;
  const auto optional_time = [](const std::optional<double> & value) {
    return value ? Json(*value) : Json(nullptr);
  };
  return {{"backend", info.backend}, {"model_path", info.model_path}, {"device", info.device},
    {"precision", info.precision}, {"model_sha256", info.model_sha256},
    {"input_shape", info.input_shape}, {"output_shape", info.output_shape},
    {"timings", {{"upload_ms", optional_time(timing.upload_ms)},
      {"preprocess_ms", optional_time(timing.preprocess_ms)},
      {"inference_ms", optional_time(timing.inference_ms)},
      {"download_ms", optional_time(timing.download_ms)},
      {"postprocess_ms", timing.postprocess_ms}, {"draw_ms", timing.draw_ms},
      {"total_ms", timing.total_ms}}}};
}

// A local terminal can stop a web-only/headless run without requiring a HighGUI window.
class TerminalQuit
{
public:
  TerminalQuit()
  {
    if (!isatty(STDIN_FILENO) || tcgetattr(STDIN_FILENO, &saved_) != 0) return;
    flags_ = fcntl(STDIN_FILENO, F_GETFL);
    if (flags_ < 0) return;
    auto raw = saved_;
    raw.c_lflag &= ~(ICANON | ECHO);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;
    active_ = tcsetattr(STDIN_FILENO, TCSANOW, &raw) == 0;
    if (active_) fcntl(STDIN_FILENO, F_SETFL, flags_ | O_NONBLOCK);
  }
  ~TerminalQuit()
  {
    if (active_) {
      tcsetattr(STDIN_FILENO, TCSANOW, &saved_);
      fcntl(STDIN_FILENO, F_SETFL, flags_);
    }
  }
  bool pressed() const
  {
    if (!active_) return false;
    char key;
    while (::read(STDIN_FILENO, &key, 1) == 1) if (key == 'q' || key == 'Q') return true;
    return false;
  }
private:
  termios saved_{};
  int flags_ = -1;
  bool active_ = false;
};

struct ImuSample { double seconds; Eigen::Quaterniond q; };

std::vector<ImuSample> read_imu(const std::string & path)
{
  std::vector<ImuSample> samples;
  if (path.empty()) return samples;
  std::ifstream input(path);
  std::string line;
  for (std::size_t row = 1; std::getline(input, line); ++row) {
    if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
    std::istringstream values(line);
    double t, w, x, y, z;
    std::string extra;
    if (!(values >> t >> w >> x >> y >> z) || (values >> extra) || !std::isfinite(t) ||
        (!samples.empty() && t <= samples.back().seconds)) {
      throw std::invalid_argument("Malformed/non-increasing IMU row " + std::to_string(row));
    }
    samples.push_back({t, Eigen::Quaterniond(w, x, y, z)});
  }
  if (input.bad() || samples.empty()) throw std::invalid_argument("Cannot read IMU data: " + path);
  return samples;
}

std::string default_log_path()
{
  const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::tm utc{};
  gmtime_r(&now, &utc);
  char stamp[32];
  std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%SZ", &utc);
  return fmt::format("records/buff_observation_{}_{}.jsonl", stamp, getpid());
}

const char * mode_name(io::GimbalMode mode)
{
  switch (mode) {
    case io::GimbalMode::IDLE: return "IDLE";
    case io::GimbalMode::AUTO_AIM: return "AUTO_AIM";
    case io::GimbalMode::SMALL_BUFF: return "SMALL_BUFF";
    case io::GimbalMode::BIG_BUFF: return "BIG_BUFF";
  }
  return "INVALID";
}

const char * activation_name(auto_buff::BuffActivation activation)
{
  switch (activation) {
    case auto_buff::BuffActivation::INACTIVE: return "INACTIVE";
    case auto_buff::BuffActivation::SMALL_ACTIVATED: return "SMALL_ACTIVATED";
    case auto_buff::BuffActivation::BIG_ACTIVATED: return "BIG_ACTIVATED";
    case auto_buff::BuffActivation::INVALID: return "INVALID";
  }
  return "INVALID";
}

const char * contour_reason(auto_buff::ContourRejectReason reason)
{
  switch (reason) {
    case auto_buff::ContourRejectReason::NONE: return "NONE";
    case auto_buff::ContourRejectReason::NOT_FOUND: return "NOT_FOUND";
    case auto_buff::ContourRejectReason::AREA_MISMATCH: return "AREA_MISMATCH";
    case auto_buff::ContourRejectReason::LOW_SUPPORT: return "LOW_SUPPORT";
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

std::optional<auto_buff::RuneState> resolve_state(
  io::GimbalMode mode, auto_buff::BuffActivation activation)
{
  using namespace auto_buff;
  if (mode != io::GimbalMode::SMALL_BUFF && mode != io::GimbalMode::BIG_BUFF) return std::nullopt;
  const auto kind = mode == io::GimbalMode::SMALL_BUFF ? RuneKind::SMALL : RuneKind::BIG;
  if (activation == BuffActivation::INACTIVE) return RuneState{kind, ActivationState::INACTIVE};
  if ((kind == RuneKind::SMALL && activation == BuffActivation::SMALL_ACTIVATED) ||
      (kind == RuneKind::BIG && activation == BuffActivation::BIG_ACTIVATED)) {
    return RuneState{kind, ActivationState::ACTIVATED};
  }
  return std::nullopt;
}

Json vector_json(const Eigen::Vector3d & v) { return Json::array({v.x(), v.y(), v.z()}); }
Json quaternion_json(const Eigen::Quaterniond & q) { return Json::array({q.w(), q.x(), q.y(), q.z()}); }

Json points_json(const auto_buff::BuffPoints & points)
{
  auto result = Json::array();
  for (const auto & point : points) result.push_back({point.x, point.y});
  return result;
}

Json contour_json(const auto_buff::ContourFeature & contour)
{
  return {{"support", contour.support}, {"reject_reason", contour_reason(contour.reject_reason)},
    {"usable", contour.usable()}, {"point_count", contour.contour.size()}};
}

Json pose_json(const auto_buff::RunePose & pose)
{
  return {{"position_world_m", vector_json(pose.position_world_m)},
    {"orientation_world_wxyz", quaternion_json(pose.orientation_world)},
    {"plane_normal_world", vector_json(pose.plane_normal_world)},
    {"phase_zero_direction_world", vector_json(pose.phase_zero_direction_world)},
    {"blade_radius_m", pose.blade_radius_m}, {"reprojection_error_px", pose.reprojection_error_px},
    {"pnp_reprojection_error_px", pose.pnp_reprojection_error_px ? Json(*pose.pnp_reprojection_error_px) : Json(nullptr)},
    {"contour_error_px", pose.contour_error_px ? Json(*pose.contour_error_px) : Json(nullptr)},
    {"contour_sample_count", pose.contour_sample_count},
    {"used_temporal_prior", pose.used_temporal_prior},
    {"confidence", pose.confidence}, {"source", source_name(pose.source)}};
}

Json reconstruction_json(const std::optional<auto_buff::RuneReconstruction> & reconstruction)
{
  if (!reconstruction) return nullptr;
  Json output = {{"pose", pose_json(reconstruction->pose)}, {"blades", Json::array()}};
  for (const auto & blade : reconstruction->blades) {
    output["blades"].push_back({{"slot", static_cast<int>(blade.slot)},
      {"armor_center_world_m", vector_json(blade.armor_center_world_m)},
      {"points", points_json(blade.observation.points)},
      {"quality", auto_buff::to_string(blade.observation.quality)}});
  }
  return output;
}

const char * model_name(auto_buff::MotionModel model)
{
  switch (model) {
    case auto_buff::MotionModel::CONSTANT_VELOCITY: return "CONSTANT_VELOCITY";
    case auto_buff::MotionModel::HARMONIC_PHASE: return "HARMONIC_PHASE";
    case auto_buff::MotionModel::UNAVAILABLE: return "UNAVAILABLE";
  }
  return "UNAVAILABLE";
}

double predict_phase(const auto_buff::MotionState & state, double future_seconds)
{
  if (state.model == auto_buff::MotionModel::CONSTANT_VELOCITY) {
    return state.continuous_phase_rad + future_seconds * state.angular_velocity_rad_s;
  }
  const double t = std::chrono::duration<double>(
    state.reference_timestamp - state.model_epoch).count() + future_seconds;
  const auto & h = state.harmonic_phase;
  return h.cos_amplitude_rad * std::cos(h.angular_frequency_rad_s * t) +
    h.sin_amplitude_rad * std::sin(h.angular_frequency_rad_s * t) +
    h.linear_rate_rad_s * t + h.phase_offset_rad;
}

Json motion_json(const auto_buff::PhaseMotionDiagnostics & diagnostics,
  const std::optional<auto_buff::RuneTarget> & target, const std::string & unavailable)
{
  Json result = {{"status", unavailable.empty() ? diagnostics.status : unavailable.c_str()},
    {"samples", diagnostics.samples}, {"span_s", diagnostics.span_s},
    {"fit_rms_rad", std::isfinite(diagnostics.fit_rms_rad) ? Json(diagnostics.fit_rms_rad) : Json(nullptr)},
    {"blade_switches", diagnostics.blade_switches}, {"resets", diagnostics.resets},
    {"cumulative_switch_offset_rad", diagnostics.cumulative_switch_offset_rad},
    {"direction_votes", diagnostics.direction_votes},
    {"observed_phase_rad", unavailable.empty() && diagnostics.observed_phase_rad ?
      Json(*diagnostics.observed_phase_rad) : Json(nullptr)},
    {"phase_zero_direction_world", vector_json(diagnostics.phase_zero_direction_world)},
    {"plane_normal_world", vector_json(diagnostics.plane_normal_world)},
    {"converged", target.has_value()}, {"model", "UNAVAILABLE"}};
  if (!target) return result;
  const auto & state = target->motion;
  const auto & h = state.harmonic_phase;
  result["model"] = model_name(state.model);
  result["continuous_phase_rad"] = state.continuous_phase_rad;
  result["angular_velocity_rad_s"] = state.angular_velocity_rad_s;
  result["confidence"] = state.confidence;
  result["quality"] = auto_buff::to_string(target->quality);
  result["slot"] = static_cast<int>(target->slot);
  result["pose"] = pose_json(target->pose);
  result["reference_timestamp_steady_s"] =
    std::chrono::duration<double>(state.reference_timestamp.time_since_epoch()).count();
  result["model_epoch_steady_s"] =
    std::chrono::duration<double>(state.model_epoch.time_since_epoch()).count();
  result["harmonic_phase"] = {{"cos_amplitude_rad", h.cos_amplitude_rad},
    {"sin_amplitude_rad", h.sin_amplitude_rad}, {"linear_rate_rad_s", h.linear_rate_rad_s},
    {"phase_offset_rad", h.phase_offset_rad}, {"angular_frequency_rad_s", h.angular_frequency_rad_s}};
  const double future_phase = predict_phase(state, 0.5);
  result["predicted_phase_0_5s_rad"] = future_phase;
  const auto & pose = target->pose;
  const Eigen::Vector3d future_armor = pose.position_world_m + pose.blade_radius_m *
    (Eigen::AngleAxisd(future_phase, pose.plane_normal_world) * pose.phase_zero_direction_world);
  result["predicted_armor_center_0_5s_world_m"] = vector_json(future_armor);
  return result;
}

// This inverse transform mirrors S4's documented legacy IMU/calibration convention.
class Projector
{
public:
  explicit Projector(const YAML::Node & config)
  {
    const auto intrinsics = config["camera_matrix"].as<std::vector<double>>();
    const auto distortion = config["distort_coeffs"].as<std::vector<double>>();
    const auto camera_rotation = config["R_camera2gimbal"].as<std::vector<double>>();
    const auto translation = config["t_camera2gimbal"].as<std::vector<double>>();
    const auto imu_rotation = config["R_gimbal2imubody"].as<std::vector<double>>();
    if (intrinsics.size() != 9 || camera_rotation.size() != 9 || translation.size() != 3 ||
        imu_rotation.size() != 9) throw std::invalid_argument("Invalid camera calibration sizes");
    camera_matrix_ = cv::Mat(3, 3, CV_64F, const_cast<double *>(intrinsics.data())).clone();
    distortion_ = cv::Mat(distortion).reshape(1, 1).clone();
    using RowMajor = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;
    camera_to_gimbal_ = Eigen::Map<const RowMajor>(camera_rotation.data());
    translation_ = Eigen::Map<const Eigen::Vector3d>(translation.data());
    gimbal_to_imu_ = Eigen::Map<const RowMajor>(imu_rotation.data());
  }
  void set_imu(const Eigen::Quaterniond & q)
  {
    gimbal_to_world_ = gimbal_to_imu_.transpose() * q.normalized().toRotationMatrix() * gimbal_to_imu_;
  }
  std::optional<cv::Point2f> project(const Eigen::Vector3d & world) const
  {
    const Eigen::Vector3d camera = camera_to_gimbal_.transpose() *
      (gimbal_to_world_.transpose() * world - translation_);
    if (!camera.allFinite() || camera.z() <= 1e-6) return std::nullopt;
    std::vector<cv::Point2d> pixels;
    cv::projectPoints(std::vector<cv::Point3d>{{camera.x(), camera.y(), camera.z()}},
      cv::Vec3d::all(0), cv::Vec3d::all(0), camera_matrix_, distortion_, pixels);
    if (!std::isfinite(pixels[0].x) || !std::isfinite(pixels[0].y) ||
        std::abs(pixels[0].x) > 1e6 || std::abs(pixels[0].y) > 1e6) return std::nullopt;
    return cv::Point2f(pixels[0]);
  }
private:
  cv::Mat camera_matrix_, distortion_;
  Eigen::Matrix3d camera_to_gimbal_, gimbal_to_imu_;
  Eigen::Matrix3d gimbal_to_world_ = Eigen::Matrix3d::Identity();
  Eigen::Vector3d translation_;
};

void label(cv::Mat & image, const std::string & text, cv::Point position,
  const cv::Scalar & color = {255, 255, 255}, double scale = 0.55)
{
  cv::putText(image, text, position, cv::FONT_HERSHEY_SIMPLEX, scale, {0, 0, 0}, 3, cv::LINE_AA);
  cv::putText(image, text, position, cv::FONT_HERSHEY_SIMPLEX, scale, color, 1, cv::LINE_AA);
}

void draw_observation(cv::Mat & image, const auto_buff::RefinedObservation & observation,
  std::size_t index, const std::string & state_error)
{
  const cv::Scalar color = observation.valid() ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 80, 255);
  if (!observation.roi.empty()) cv::rectangle(image, observation.roi, color, 1);
  const auto draw_contour = [&](const auto_buff::ContourFeature & feature, cv::Scalar c) {
    if (!feature.contour.empty()) cv::drawContours(image,
      std::vector<std::vector<cv::Point>>{feature.contour}, -1, c, 2, cv::LINE_AA);
  };
  draw_contour(observation.armor_contour, {0, 255, 0});
  draw_contour(observation.r_mark_contour, {255, 0, 255});
  draw_contour(observation.light_arm_contour, {255, 255, 0});
  const char * names[] = {"T", "R", "B", "L", "Rmark"};
  std::optional<cv::Point> anchor;
  for (std::size_t point = 0; point < observation.points.size(); ++point) {
    const auto p = observation.points[point];
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || p.x < 0 || p.y < 0 ||
        p.x >= image.cols || p.y >= image.rows) continue;
    if (!anchor) anchor = cv::Point(p);
    cv::circle(image, p, 4, color, -1, cv::LINE_AA);
    label(image, fmt::format("#{}:{}", index, names[point]), cv::Point(p) + cv::Point(5, -5), color, 0.45);
  }
  const auto reason = state_error.empty() ? auto_buff::to_string(observation.reject_reason) : state_error;
  const auto origin = anchor.value_or(cv::Point(10, 170 + static_cast<int>(index) * 42));
  const int y = std::clamp(origin.y + 25, 20, std::max(20, image.rows - 24));
  const int x = std::clamp(origin.x, 0, std::max(0, image.cols - 400));
  label(image, fmt::format("#{} {:.2f} {} {}", index, observation.detection_confidence,
    auto_buff::to_string(observation.quality), reason), {x, y}, color, 0.45);
  label(image, fmt::format("geom {:.2f} A {:.2f}/{} R {:.2f}/{} L {:.2f}/{}",
    observation.geometry_score, observation.armor_contour.support,
    contour_reason(observation.armor_contour.reject_reason), observation.r_mark_contour.support,
    contour_reason(observation.r_mark_contour.reject_reason), observation.light_arm_contour.support,
    contour_reason(observation.light_arm_contour.reject_reason)), {x, y + 17}, color, 0.40);
}

void draw_reconstruction(cv::Mat & image, const auto_buff::RuneReconstruction & reconstruction,
  const Projector & projector)
{
  const auto & pose = reconstruction.pose;
  const auto world = [&](const Eigen::Vector3d & local) {
    return Eigen::Vector3d(pose.position_world_m + pose.orientation_world * local);
  };
  const auto center = projector.project(pose.position_world_m);
  if (center) {
    cv::drawMarker(image, *center, {0, 255, 255}, cv::MARKER_CROSS, 18, 2, cv::LINE_AA);
    label(image, "3D center", cv::Point(*center) + cv::Point(8, 14), {0, 255, 255});
    const Eigen::Vector3d axes[] = {{0.25, 0, 0}, {0, 0.25, 0}, {0, 0, 0.25}};
    const cv::Scalar colors[] = {{0, 0, 255}, {0, 255, 0}, {255, 100, 0}};
    const char * names[] = {"X normal", "Y", "Z phase0"};
    for (int i = 0; i < 3; ++i) {
      const auto endpoint = projector.project(world(axes[i]));
      if (!endpoint) continue;
      cv::arrowedLine(image, *center, *endpoint, colors[i], 2, cv::LINE_AA);
      label(image, names[i], cv::Point(*endpoint), colors[i], 0.45);
    }
  }
  // The existing model uses 0.127 m from armour center to each diamond corner.
  constexpr double half_diagonal = 0.127;
  for (const auto & blade : reconstruction.blades) {
    const int slot = static_cast<int>(blade.slot);
    const Eigen::AngleAxisd rotation((slot - 1) * 2.0 * PI / 5.0, Eigen::Vector3d::UnitX());
    const Eigen::Vector3d corners[] = {{0, 0, 0.7 + half_diagonal},
      {0, half_diagonal, 0.7}, {0, 0, 0.7 - half_diagonal}, {0, -half_diagonal, 0.7}};
    std::vector<cv::Point> pixels;
    for (const auto & corner : corners) {
      const auto pixel = projector.project(world(rotation * corner));
      if (!pixel) { pixels.clear(); break; }
      pixels.emplace_back(*pixel);
      cv::circle(image, *pixel, 5, {255, 180, 0}, 1, cv::LINE_AA);
    }
    if (pixels.size() == 4) cv::polylines(image, pixels, true, {255, 180, 0}, 2, cv::LINE_AA);
    const auto armor = projector.project(world(rotation * Eigen::Vector3d(0, 0, 0.7)));
    if (armor) {
      label(image, fmt::format("slot {} PNP", slot), cv::Point(*armor), {255, 180, 0});
      if (center) cv::line(image, *center, *armor, {255, 180, 0}, 1, cv::LINE_AA);
    }
  }
}

int run(const Options & options)
{
  const auto config = YAML::LoadFile(options.config);
  const auto imu_samples = read_imu(options.imu);
  auto_buff::ObservationRefiner refiner(options.buff_config);
  auto_buff::RuneReconstructor reconstructor(options.config);
  auto_buff::PhaseMotionEstimator motion_estimator;
  std::unique_ptr<auto_buff::BuffProcessor> processor;
  if (options.check_processor) {
    processor = std::make_unique<auto_buff::BuffProcessor>(options.config, options.buff_config);
  }
  Projector projector(config);
  DetectorConfig detector_config(options, config);
  // Validate all arguments, YAML and log destination before opening real hardware.
  const std::string log_path = options.log.empty() ? default_log_path() : options.log;
  const auto parent = std::filesystem::path(log_path).parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent);
  std::ofstream log(log_path, std::ios::out | std::ios::trunc);
  if (!log) throw std::runtime_error("Cannot open JSONL log: " + log_path);
  // Keep the exact calibration and thresholds alongside each capture/replay for reproducibility.
  const std::string camera_snapshot_path = log_path + ".camera.yaml";
  const std::string buff_snapshot_path = log_path + ".buff.yaml";
  for (const auto & snapshot : std::vector<std::pair<std::string, YAML::Node>>{
         {camera_snapshot_path, YAML::LoadFile(detector_config.path())},
         {buff_snapshot_path, YAML::LoadFile(options.buff_config)}}) {
    std::ofstream output(snapshot.first);
    output << snapshot.second;
    output.close();
    if (!output) throw std::runtime_error("Cannot write configuration snapshot: " + snapshot.first);
  }

  const auto display = config["debug_display"];
  const auto window_config = display ? display["window"] : YAML::Node();
  bool window = options.window.value_or(window_config && window_config["enabled"] ?
    window_config["enabled"].as<bool>() : false);
  if (window && !getenv("DISPLAY") && !getenv("WAYLAND_DISPLAY")) {
    tools::logger()->warn("No display server; local debug window disabled");
    window = false;
  }
  window = window && !options.headless;
  const auto sinks = config["debug_bus"] ? config["debug_bus"]["sinks"] : YAML::Node();
  auto web_config = sinks ? sinks["web"] : YAML::Node();
  if (!web_config && display) web_config = display["web"];
  const bool web_enabled = !options.headless &&
    (!web_config || !web_config["enabled"] || web_config["enabled"].as<bool>());
  const int port = web_config && web_config["port"] ? web_config["port"].as<int>() : 8080;
  if (port < 1 || port > 65535) throw std::invalid_argument("Web port must be in [1,65535]");
  debug::StreamConfig stream;
  if (web_config) {
    if (web_config["max_fps"]) stream.max_fps = web_config["max_fps"].as<int>();
    if (web_config["jpeg_quality"]) stream.jpeg_quality = web_config["jpeg_quality"].as<int>();
    if (web_config["max_width"]) stream.max_width = web_config["max_width"].as<int>();
  }
  std::unique_ptr<debug::WebDebugger> web;
  if (web_enabled) {
    const auto bind = web_config && web_config["bind"] ? web_config["bind"].as<std::string>() : "0.0.0.0";
    web = std::make_unique<debug::WebDebugger>(port, bind, stream);
    if (!web->start()) throw std::runtime_error("Cannot start web debugger");
    tools::logger()->info("Buff observation view: http://127.0.0.1:{}", port);
  }
  if (window) cv::namedWindow("buff_observation", cv::WINDOW_NORMAL);
  auto_buff::Buff_Detector detector(detector_config.path());
  const bool offline = !options.video.empty();
  cv::VideoCapture video;
  std::unique_ptr<io::Gimbal> gimbal;
  std::unique_ptr<io::Camera> camera;
  double fps = 30;
  if (offline) {
    if (!video.open(options.video)) throw std::runtime_error("Cannot open video: " + options.video);
    fps = video.get(cv::CAP_PROP_FPS);
    if (!std::isfinite(fps) || fps <= 0) fps = 30;
    const double video_frames = video.get(cv::CAP_PROP_FRAME_COUNT);
    if ((video_frames > 0 && options.start_index >= video_frames) ||
        (!options.imu.empty() && options.start_index >= imu_samples.size())) {
      throw std::invalid_argument("--start-index lies outside video/IMU data");
    }
    if (options.start_index != 0 && !video.set(cv::CAP_PROP_POS_FRAMES, options.start_index)) {
      throw std::runtime_error("Cannot seek video to --start-index");
    }
  } else {
    gimbal = std::make_unique<io::Gimbal>(options.config);
    camera = std::make_unique<io::Camera>(options.config);
  }
  std::unique_ptr<tools::Recorder> recorder;
  if (options.record) recorder = std::make_unique<tools::Recorder>(fps);
  tools::Exiter exiter;
  TerminalQuit terminal;
  const std::string imu_source = offline ? (options.imu.empty() ? "IDENTITY (NO IMU FILE)" : "FILE") : "SERIAL";
  tools::logger()->info("{}; mode={}, IMU={}, log={}; serial receive only, q/Ctrl+C exits",
    options.motion ? "S2/S3/S4/S5 diagnostics" : "S2/S3/S4 diagnostics",
    options.mode, imu_source, log_path);
  const auto epoch = Clock::now();
  auto last_summary = epoch;
  std::optional<Clock::time_point> previous_timestamp;
  std::optional<io::GimbalMode> previous_mode;
  std::optional<auto_buff::RunePose> previous_pose;
  std::optional<Clock::time_point> previous_pose_timestamp;
  std::size_t frame_count = 0, total_valid = 0, total_poses = 0;
  double total_ms = 0;
  while (!exiter.exit() && !terminal.pressed() &&
         (options.frames == 0 || frame_count < options.frames)) {
    cv::Mat raw;
    auto timestamp = Clock::now();
    Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
    io::GimbalMode serial_mode = io::GimbalMode::IDLE;
    io::GimbalState gimbal_state{};
    const std::size_t input_frame_index = options.start_index + frame_count;
    if (offline) {
      if (!video.read(raw) || raw.empty()) break;
      if (!options.imu.empty()) {
        if (input_frame_index >= imu_samples.size()) throw std::runtime_error("IMU file ended before video");
        const auto & sample = imu_samples[input_frame_index];
        timestamp = epoch + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(sample.seconds));
        q = sample.q;
      } else {
        timestamp = epoch + std::chrono::duration_cast<Clock::duration>(
          std::chrono::duration<double>(static_cast<double>(input_frame_index) / fps));
      }
    } else {
      camera->read(raw, timestamp);
      if (raw.empty()) throw std::runtime_error("Camera returned an empty frame");
      q = gimbal->q(timestamp);
      serial_mode = gimbal->mode();
      gimbal_state = gimbal->state();
    }
    if (frame_count == 0 && options.warmup != 0) {
      tools::logger()->info("Warming up Buff detector on the first real frame: {} iterations", options.warmup);
      for (std::size_t iteration = 0; iteration < options.warmup; ++iteration) {
        if (exiter.exit() || terminal.pressed()) return 0;
        auto warmup_image = raw.clone();
        detector.detect_observations(warmup_image);
      }
    }
    const auto begin = Clock::now();
    const std::optional<double> capture_interval_ms = previous_timestamp ?
      std::optional<double>(std::chrono::duration<double, std::milli>(timestamp - *previous_timestamp).count()) :
      std::nullopt;
    const std::optional<double> capture_fps = capture_interval_ms && *capture_interval_ms > 0 ?
      std::optional<double>(1000.0 / *capture_interval_ms) : std::nullopt;
    previous_timestamp = timestamp;
    const auto mode = options.mode == "serial" ? serial_mode :
      (options.mode == "small" ? io::GimbalMode::SMALL_BUFF : io::GimbalMode::BIG_BUFF);
    if (previous_mode && *previous_mode != mode) {
      motion_estimator.reset();
      if (processor) processor->reset();
      previous_pose.reset();
      previous_pose_timestamp.reset();
    }
    previous_mode = mode;
    const double imu_norm = q.norm();
    const bool imu_valid = q.coeffs().allFinite() && std::isfinite(imu_norm) && imu_norm > 1e-6;
    if (recorder) recorder->record(raw.clone(), q, timestamp);
    cv::Mat detector_image = raw.clone();
    const auto observations = detector.detect_observations(detector_image);
    std::vector<auto_buff::RefinedObservation> refined, valid;
    std::vector<std::string> state_errors;
    auto candidates = Json::array();
    for (const auto & observation : observations) {
      const auto state = resolve_state(mode, observation.activation);
      // Still collect image/contour diagnostics for a state conflict, then reject it for S4.
      auto value = refiner.refine(raw, observation, state.value_or(auto_buff::RuneState{}));
      const auto geometry_reject = value.reject_reason;
      const std::string state_error = state ? "" :
        (observation.activation == auto_buff::BuffActivation::INVALID ? "INVALID_ACTIVATION" :
         (mode == io::GimbalMode::IDLE || mode == io::GimbalMode::AUTO_AIM ? "NOT_BUFF_MODE" : "STATE_MODE_CONFLICT"));
      if (!state) {
        value.state.reset();
        value.quality = auto_buff::TrackingQuality::INVALID;
        value.reject_reason = auto_buff::ObservationRejectReason::STATE_MODE_CONFLICT;
      }
      if (value.valid()) valid.push_back(value);
      candidates.push_back({{"index", refined.size()},
        {"raw", {{"points", points_json(observation.points)},
          {"activation", activation_name(observation.activation)}, {"confidence", observation.confidence}}},
        {"refined", {{"points", points_json(value.points)},
          {"quality", auto_buff::to_string(value.quality)},
          {"reject_reason", auto_buff::to_string(value.reject_reason)},
          {"geometry_reject_reason", auto_buff::to_string(geometry_reject)},
          {"state_reject_reason", state_error}, {"geometry_score", value.geometry_score},
          {"roi", {value.roi.x, value.roi.y, value.roi.width, value.roi.height}},
          {"state", state ? Json{{"kind", state->kind == auto_buff::RuneKind::SMALL ? "SMALL" : "BIG"},
            {"activation", state->activation == auto_buff::ActivationState::INACTIVE ? "INACTIVE" : "ACTIVATED"}} : Json(nullptr)},
          {"armor_contour", contour_json(value.armor_contour)},
          {"r_mark_contour", contour_json(value.r_mark_contour)},
          {"light_arm_contour", contour_json(value.light_arm_contour)}}}});
      refined.push_back(std::move(value));
      state_errors.push_back(state_error);
    }
    if (previous_pose_timestamp && (timestamp <= *previous_pose_timestamp ||
        timestamp - *previous_pose_timestamp > std::chrono::milliseconds(200))) {
      previous_pose.reset();
      previous_pose_timestamp.reset();
    }
    const auto reconstruction = imu_valid ? reconstructor.reconstruct(valid, q.normalized(),
      options.pose_prior ? previous_pose : std::nullopt) : std::nullopt;
    if (reconstruction) {
      previous_pose = reconstruction->pose;
      previous_pose_timestamp = timestamp;
    }
    std::optional<auto_buff::RuneTarget> motion_target;
    std::string motion_unavailable = "NO_GEOMETRY";
    if (options.motion && reconstruction) {
      const auto_buff::ReconstructedBlade * target = nullptr;
      for (const auto & blade : reconstruction->blades) {
        if (!blade.observation.state ||
            blade.observation.state->activation != auto_buff::ActivationState::INACTIVE) continue;
        if (!target || blade.observation.detection_confidence > target->observation.detection_confidence) {
          target = &blade;
        }
      }
      if (target) {
        motion_unavailable.clear();
        motion_target = motion_estimator.update(reconstruction->pose, *target, timestamp);
      } else motion_unavailable = "NO_INACTIVE_BLADE";
    }
    const auto motion_data = options.motion ?
      motion_json(motion_estimator.diagnostics(), motion_target, motion_unavailable) : Json(nullptr);
    Json processor_data = nullptr;
    if (processor) {
      // Evaluate the production chain with measured serial state. Its output is inspected
      // here and deliberately has no connection to Gimbal::send.
      const auto output = processor->process(auto_buff::make_buff_input(
        raw, timestamp, q, gimbal_state, mode, observations));
      const std::array<float, 6> fields = {output.yaw, output.yaw_vel, output.yaw_acc,
        output.pitch, output.pitch_vel, output.pitch_acc};
      const bool finite = std::all_of(fields.begin(), fields.end(), [](float v) { return std::isfinite(v); });
      const bool idle_zero = output.mode != 0 || std::all_of(fields.begin(), fields.end(), [](float v) { return v == 0.0F; });
      const bool hold = output.mode != 1 ||
        (output.yaw == gimbal_state.yaw && output.pitch == gimbal_state.pitch &&
         output.yaw_vel == 0 && output.yaw_acc == 0 && output.pitch_vel == 0 && output.pitch_acc == 0);
      if (!finite || !idle_zero || !hold || output.mode > 1) {
        throw std::runtime_error("S1 passive check failed: invalid idle/hold protocol output");
      }
      processor_data = {{"mode", output.mode}, {"yaw", output.yaw}, {"pitch", output.pitch},
        {"finite", finite}, {"idle_zero", idle_zero}, {"measured_attitude_hold", hold},
        {"transmitted", false}};
    }
    const std::string status = reconstruction ? source_name(reconstruction->pose.source) :
      (!imu_valid ? "INVALID IMU" : (valid.size() < 2 ? "NEED >=2 BLADES" : "PNP FAILED"));
    const double elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
    ++frame_count;
    total_valid += valid.size();
    total_poses += reconstruction.has_value();
    total_ms += elapsed_ms;
    Json row = {{"frame", frame_count}, {"input_frame_index", input_frame_index},
      {"timestamp_steady_s", std::chrono::duration<double>(timestamp.time_since_epoch()).count()},
      {"timestamp_relative_s", std::chrono::duration<double>(timestamp - epoch).count()},
      {"q_wxyz", quaternion_json(q)}, {"imu_norm", imu_norm}, {"imu_valid", imu_valid},
      {"imu_source", imu_source}, {"input", offline ? options.video : "camera"},
      {"gimbal_state", offline ? Json(nullptr) : Json{{"yaw", gimbal_state.yaw},
        {"yaw_vel", gimbal_state.yaw_vel}, {"pitch", gimbal_state.pitch},
        {"pitch_vel", gimbal_state.pitch_vel}, {"bullet_speed", gimbal_state.bullet_speed},
        {"bullet_count", gimbal_state.bullet_count}}},
      {"config", options.config}, {"buff_config", options.buff_config},
      {"camera_config_snapshot", camera_snapshot_path}, {"buff_config_snapshot", buff_snapshot_path},
      {"pose_prior_enabled", options.pose_prior},
      {"recording_video", recorder ? Json(recorder->video_path()) : Json(nullptr)},
      {"recording_imu", recorder ? Json(recorder->text_path()) : Json(nullptr)},
      {"device_override", options.device.empty() ? Json(nullptr) : Json(options.device)},
      {"warmup_iterations", options.warmup},
      {"inference", inference_json(detector.inference_diagnostics())},
      {"mode", {{"selection", options.mode}, {"serial", offline ? Json(nullptr) : Json(mode_name(serial_mode))},
        {"effective", mode_name(mode)}}},
      {"capture_interval_ms", capture_interval_ms ? Json(*capture_interval_ms) : Json(nullptr)},
      {"fps", capture_fps ? Json(*capture_fps) : Json(nullptr)},
      {"processing_ms", elapsed_ms}, {"observations", std::move(candidates)},
      {"valid_blades", valid.size()}, {"reconstruction_status", status},
      {"motion", motion_data},
      {"processor_check", processor_data},
      {"reconstruction", reconstruction_json(reconstruction)}};
    log << row.dump() << '\n';
    if (!log) throw std::runtime_error("Failed to write JSONL log: " + log_path);

    auto permit = web ? web->try_begin_frame() : debug::FramePermit{};
    if (window || permit) {
      cv::Mat view;
      if (raw.channels() == 1) cv::cvtColor(raw, view, cv::COLOR_GRAY2BGR);
      else if (raw.channels() == 4) cv::cvtColor(raw, view, cv::COLOR_BGRA2BGR);
      else view = raw.clone();
      for (std::size_t i = 0; i < refined.size(); ++i) draw_observation(view, refined[i], i, state_errors[i]);
      if (reconstruction) { projector.set_imu(q); draw_reconstruction(view, *reconstruction, projector); }
      label(view, fmt::format("frame {} | fps {} | dt {} ms | proc {:.1f} ms | IMU norm {:.4f} {}",
        frame_count, capture_fps ? fmt::format("{:.1f}", *capture_fps) : "--",
        capture_interval_ms ? fmt::format("{:.1f}", *capture_interval_ms) : "--",
        elapsed_ms, imu_norm, imu_source), {10, 24});
      label(view, fmt::format("serial {} | test {} -> {} | candidates {} valid {}",
        offline ? "N/A" : mode_name(serial_mode), options.mode, mode_name(mode), observations.size(), valid.size()), {10, 47});
      label(view, status, {10, 70}, reconstruction ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 100, 255));
      label(view, options.motion ? "S2/S3/S4/S5 diagnostics | serial receive only | q to exit" :
        "S2/S3/S4 | contours: armor GREEN / R MAGENTA / arm CYAN | q to exit", {10, 93}, {255, 255, 255}, 0.45);
      if (reconstruction) {
        const auto & p = reconstruction->pose;
        label(view, fmt::format("world center [{:.3f}, {:.3f}, {:.3f}] m | error {:.2f} px",
          p.position_world_m.x(), p.position_world_m.y(), p.position_world_m.z(), p.reprojection_error_px), {10, 116});
        label(view, fmt::format("world normal [{:.3f}, {:.3f}, {:.3f}] | {}",
          p.plane_normal_world.x(), p.plane_normal_world.y(), p.plane_normal_world.z(), source_name(p.source)), {10, 139});
      }
      if (options.motion) {
        const auto & d = motion_estimator.diagnostics();
        label(view, fmt::format("S5 {} | n {} span {:.2f}s | fit RMS {} rad | switches {} resets {}",
          motion_data["status"].get<std::string>(), d.samples, d.span_s,
          std::isfinite(d.fit_rms_rad) ? fmt::format("{:.3f}", d.fit_rms_rad) : "--",
          d.blade_switches, d.resets), {10, 162}, {255, 255, 0}, 0.45);
        if (motion_target) {
          const auto & state = motion_target->motion;
          label(view, fmt::format("{} phase {:.3f} rad | speed {:.3f} rad/s | prediction +0.5s",
            model_name(state.model), state.continuous_phase_rad, state.angular_velocity_rad_s),
            {10, 182}, {255, 255, 0}, 0.45);
          projector.set_imu(q);
          const auto point = motion_data["predicted_armor_center_0_5s_world_m"];
          const auto pixel = projector.project(Eigen::Vector3d(point[0].get<double>(),
            point[1].get<double>(), point[2].get<double>()));
          if (pixel) cv::drawMarker(view, *pixel, {0, 255, 255}, cv::MARKER_TILTED_CROSS, 20, 2);
        }
      }
      if (permit) web->push(std::move(permit), view, {}, {}, elapsed_ms, timestamp);
      if (window) { cv::imshow("buff_observation", view); if ((cv::waitKey(1) & 0xff) == 'q') break; }
    }
    const auto now = Clock::now();
    if (frame_count % 30 == 0 || now - last_summary >= std::chrono::seconds(1)) {
      tools::logger()->info("Buff frame {}: {} candidates, {} valid, {}; mean {:.1f} ms, poses {}/{}",
        frame_count, observations.size(), valid.size(), status, total_ms / frame_count, total_poses, frame_count);
      if (options.motion) tools::logger()->info("S5 {}: {} samples / {:.2f}s, converged={}",
        motion_data["status"].get<std::string>(), motion_estimator.diagnostics().samples,
        motion_estimator.diagnostics().span_s, motion_target.has_value());
      log.flush();
      last_summary = now;
    }
  }
  log.flush();
  if (!log) throw std::runtime_error("Failed to flush JSONL log: " + log_path);
  if (window) cv::destroyWindow("buff_observation");
  tools::logger()->info("Finished: {} frames, {} valid observations, {} poses; JSONL {}",
    frame_count, total_valid, total_poses, log_path);
  return 0;
}
}  // namespace

int main(int argc, char ** argv)
{
  try {
    const auto options = parse_options(argc, argv);
    if (options.help) { print_help(); return 0; }
    return run(options);
  } catch (const std::exception & error) {
    std::cerr << "buff_observation_debug: " << error.what() << '\n';
    return 1;
  }
}
