#include "param_tuner.hpp"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <sstream>

#include "tools/logger.hpp"

namespace debug
{

ParamTuner & ParamTuner::instance()
{
  static ParamTuner inst;
  return inst;
}

void ParamTuner::load_from_yaml(const std::string & config_path)
{
  auto yaml = YAML::LoadFile(config_path);

  param_sets_.clear();

  // 默认参数集（如果 YAML 中没有配置的话）
  if (!yaml["filter_param_sets"]) {
    tools::logger()->warn("[ParamTuner] No filter_param_sets in config, using defaults");
    param_sets_ = {
      {"slow",   10.0,  0.5, 4e-3, 4e-3, 9e-2, 2.0},
      {"normal", 100.0, 400.0, 4e-3, 4e-3, 9e-2, 8.0},
      {"fast",   500.0, 1000.0, 2e-3, 2e-3, 5e-2, 999.0},
    };
    return;
  }

  auto sets = yaml["filter_param_sets"];
  for (const auto & s : sets) {
    FilterParamSet ps;
    ps.name = s["name"].as<std::string>();
    ps.v1 = s["v1"].as<double>();
    ps.v2 = s["v2"].as<double>();
    ps.R_yaw = s["R_yaw"].as<double>();
    ps.R_pitch = s["R_pitch"].as<double>();
    ps.R_distance = s["R_distance"].as<double>();
    ps.angular_velocity_threshold = s["angular_velocity_threshold"].as<double>();
    param_sets_.push_back(ps);
  }

  tools::logger()->info("[ParamTuner] Loaded {} filter param sets", param_sets_.size());
  for (const auto & ps : param_sets_) {
    tools::logger()->info(
      "  [{}] v1={:.1f} v2={:.1f} R_yaw={:.1e} R_pitch={:.1e} R_dist={:.1e} thresh={:.1f}",
      ps.name, ps.v1, ps.v2, ps.R_yaw, ps.R_pitch, ps.R_distance,
      ps.angular_velocity_threshold);
  }
}

const FilterParamSet & ParamTuner::select(double angular_velocity) const
{
  std::lock_guard<std::mutex> lk(mutex_);

  if (param_sets_.empty()) {
    static const FilterParamSet default_param{
      "normal", 100.0, 400.0, 4e-3, 4e-3, 9e-2, 999.0};
    tools::logger()->warn("[ParamTuner] No filter param sets loaded, using default normal set");
    active_index_ = 0;
    return default_param;
  }

  double w = std::abs(angular_velocity);

  for (size_t i = 0; i < param_sets_.size(); ++i) {
    if (w < param_sets_[i].angular_velocity_threshold) {
      active_index_ = static_cast<int>(i);
      return param_sets_[i];
    }
  }

  // 如果角速度超过所有阈值，使用最后一个
  active_index_ = static_cast<int>(param_sets_.size()) - 1;
  return param_sets_.back();
}

void ParamTuner::update_param_set(int index, const FilterParamSet & ps)
{
  std::lock_guard<std::mutex> lk(mutex_);
  if (index >= 0 && index < static_cast<int>(param_sets_.size())) {
    param_sets_[index] = ps;
    tools::logger()->info(
      "[ParamTuner] Updated param set [{}]: v1={:.1f} v2={:.1f}", index, ps.v1, ps.v2);
  }

  // 通知回调
  if (update_callback_) {
    update_callback_(ps);
  }
}

void ParamTuner::push_ekf_state(const EKFStateData & state)
{
  std::lock_guard<std::mutex> lk(state_mutex_);
  latest_ekf_state_ = state;
}

EKFStateData ParamTuner::latest_ekf_state() const
{
  std::lock_guard<std::mutex> lk(state_mutex_);
  return latest_ekf_state_;
}

std::string ParamTuner::active_name() const
{
  std::lock_guard<std::mutex> lk(mutex_);
  int idx = active_index_.load();
  if (idx >= 0 && idx < static_cast<int>(param_sets_.size())) {
    return param_sets_[idx].name;
  }
  return "unknown";
}

void ParamTuner::set_update_callback(ParamUpdateCallback cb) { update_callback_ = cb; }

std::string ParamTuner::param_sets_to_json() const
{
  std::lock_guard<std::mutex> lk(mutex_);

  std::ostringstream ss;
  ss << "[";
  for (size_t i = 0; i < param_sets_.size(); ++i) {
    if (i) ss << ",";
    const auto & ps = param_sets_[i];
    ss << "{"
       << "\"index\":" << i << ","
       << "\"name\":\"" << ps.name << "\","
       << "\"v1\":" << ps.v1 << ","
       << "\"v2\":" << ps.v2 << ","
       << "\"R_yaw\":" << ps.R_yaw << ","
       << "\"R_pitch\":" << ps.R_pitch << ","
       << "\"R_distance\":" << ps.R_distance << ","
       << "\"angular_velocity_threshold\":" << ps.angular_velocity_threshold << ","
       << "\"active\":" << (static_cast<int>(i) == active_index_.load() ? "true" : "false")
       << "}";
  }
  ss << "]";
  return ss.str();
}

std::string ParamTuner::ekf_state_to_json() const
{
  EKFStateData s = latest_ekf_state();

  std::ostringstream ss;
  ss << "{"
     << "\"timestamp\":" << s.timestamp << ","
     << "\"x\":" << s.x << ","
     << "\"y\":" << s.y << ","
     << "\"z\":" << s.z << ","
     << "\"vx\":" << s.vx << ","
     << "\"vy\":" << s.vy << ","
     << "\"vz\":" << s.vz << ","
     << "\"yaw\":" << s.yaw << ","
     << "\"w\":" << s.w << ","
     << "\"r\":" << s.r << ","
     << "\"l\":" << s.l << ","
     << "\"h\":" << s.h << ","
     << "\"v1\":" << s.v1 << ","
     << "\"v2\":" << s.v2 << ","
     << "\"R_yaw\":" << s.R_yaw << ","
     << "\"R_pitch\":" << s.R_pitch << ","
     << "\"R_distance\":" << s.R_distance << ","
     << "\"nis\":" << s.nis << ","
     << "\"nis_fail_count\":" << s.nis_fail_count << ","
     << "\"converged\":" << (s.converged ? "true" : "false") << ","
     << "\"active_param_set\":\"" << s.active_param_set << "\","
     << "\"P_diag\":[";
  for (int i = 0; i < s.P_dim && i < 13; ++i) {
    if (i) ss << ",";
    ss << s.P_diag[i];
  }
  ss << "],"
     << "\"P_dim\":" << s.P_dim
     << "}";
  return ss.str();
}

void ParamTuner::update_from_json(const std::string & json)
{
  // 简单 JSON 解析：查找 "cmd":"update_param" 和参数
  // 格式: {"cmd":"update_param","index":0,"v1":100,...}
  // 使用简单的字符串查找而非完整 JSON 解析器

  if (json.find("\"cmd\":\"update_param\"") == std::string::npos &&
      json.find("\"cmd\": \"update_param\"") == std::string::npos) {
    return;
  }

  // 简单的手动解析（避免引入 nlohmann_json 依赖）
  auto find_val = [&](const std::string & key) -> std::string {
    auto pos = json.find("\"" + key + "\":");
    if (pos == std::string::npos) {
      pos = json.find("\"" + key + "\": ");
      if (pos == std::string::npos) return "";
    }
    pos = json.find(":", pos) + 1;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
    auto end = json.find_first_of(",}\n\r", pos);
    return json.substr(pos, end - pos);
  };

  auto index_str = find_val("index");
  if (index_str.empty()) return;

  int index = std::stoi(index_str);

  FilterParamSet ps;
  {
    std::lock_guard<std::mutex> lk(mutex_);
    if (index < 0 || index >= static_cast<int>(param_sets_.size())) return;
    ps = param_sets_[index];
  }

  auto v1_str = find_val("v1");
  auto v2_str = find_val("v2");
  auto R_yaw_str = find_val("R_yaw");
  auto R_pitch_str = find_val("R_pitch");
  auto R_distance_str = find_val("R_distance");
  auto thresh_str = find_val("angular_velocity_threshold");

  if (!v1_str.empty()) ps.v1 = std::stod(v1_str);
  if (!v2_str.empty()) ps.v2 = std::stod(v2_str);
  if (!R_yaw_str.empty()) ps.R_yaw = std::stod(R_yaw_str);
  if (!R_pitch_str.empty()) ps.R_pitch = std::stod(R_pitch_str);
  if (!R_distance_str.empty()) ps.R_distance = std::stod(R_distance_str);
  if (!thresh_str.empty()) ps.angular_velocity_threshold = std::stod(thresh_str);

  update_param_set(index, ps);
}

}  // namespace debug
