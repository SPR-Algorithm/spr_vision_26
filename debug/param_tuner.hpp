#pragma once
// param_tuner.hpp — 动态滤波器参数管理器
// 支持运行时通过 WebSocket 调节 EKF 参数，并基于角速度自动切换参数集

#include <Eigen/Dense>
#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace debug
{

/// @brief 一组滤波器参数（过程噪声 v1/v2、观测噪声 R、初始协方差 P0）
struct FilterParamSet {
  std::string name;          // 参数集名称，如 "slow", "normal", "fast"
  double v1;                 // 位置过程噪声
  double v2;                 // 角度过程噪声
  double R_yaw;              // yaw 观测噪声因子
  double R_pitch;            // pitch 观测噪声因子
  double R_distance;         // distance 观测噪声因子

  // 角速度阈值：当 |w| < threshold 时使用此参数集（最后一个可以为无穷大）
  double angular_velocity_threshold;  // rad/s
};

/// @brief EKF 实时状态数据（用于 GUI 显示）
struct EKFStateData {
  double timestamp;                    // 时间戳
  double x, y, z;                      // 位置
  double vx, vy, vz;                   // 速度
  double yaw, w;                       // 角度、角速度
  double r, l, h;                      // 半径、长短轴差、高度差
  double P_diag[13];                   // P 矩阵对角线（最多13维）
  int P_dim;                           // P 矩阵维度
  double v1, v2;                       // 当前使用的过程噪声
  double R_yaw, R_pitch, R_distance;   // 当前使用的观测噪声
  double nis;                          // 最近 NIS 值
  int nis_fail_count;                  // NIS 失败计数
  bool converged;                      // 是否收敛
  std::string active_param_set;        // 当前激活的参数集名称
};

class ParamTuner
{
public:
  using ParamUpdateCallback = std::function<void(const FilterParamSet &)>;

  static ParamTuner & instance();

  // 从 YAML 加载参数集
  void load_from_yaml(const std::string & config_path);

  // 获取当前应使用的参数集（基于角速度自动选择）
  const FilterParamSet & select(double angular_velocity) const;

  // 获取所有参数集（只读）
  const std::vector<FilterParamSet> & param_sets() const { return param_sets_; }

  // 运行时更新某个参数集（由 GUI 触发）
  void update_param_set(int index, const FilterParamSet & ps);

  // 获取/设置最新 EKF 状态（线程安全）
  void push_ekf_state(const EKFStateData & state);
  EKFStateData latest_ekf_state() const;

  // 当前激活的参数集索引
  int active_index() const { return active_index_.load(); }
  std::string active_name() const;

  // 设置参数更新回调（当 GUI 修改参数时通知 Target）
  void set_update_callback(ParamUpdateCallback cb);

  // 序列化参数集列表为 JSON
  std::string param_sets_to_json() const;
  std::string ekf_state_to_json() const;

  // 从 JSON 更新参数集
  void update_from_json(const std::string & json);

private:
  ParamTuner() = default;

  mutable std::mutex mutex_;
  std::vector<FilterParamSet> param_sets_;
  mutable std::atomic<int> active_index_{0};
  EKFStateData latest_ekf_state_;
  mutable std::mutex state_mutex_;
  ParamUpdateCallback update_callback_;
};

}  // namespace debug
