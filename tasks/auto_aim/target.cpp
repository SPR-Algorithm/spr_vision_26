#include "target.hpp"

#include <chrono>
#include <numeric>

#include "debug/param_tuner.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
namespace
{
constexpr double OUTPOST_ARMOR_Z_STEP_M = 0.1;  // 前哨站三块装甲板 Z 向每两块差 100mm
}

Target::Target(
  const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
  Eigen::VectorXd P0_dig)
: name(armor.name),
  armor_type(armor.type),
  jumped(false),
  last_id(0),
  update_count_(0),
  armor_num_(armor_num),
  t_(t),
  is_switch_(false),
  is_converged_(false),
  switch_count_(0)
{
  auto r = radius;
  priority = armor.priority;
  const Eigen::VectorXd & xyz = armor.xyz_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;

  // 旋转中心的坐标
  auto center_x = xyz[0] + r * std::cos(ypr[0]);
  auto center_y = xyz[1] + r * std::sin(ypr[0]);
  auto center_z = xyz[2];

  // x vx y vy z vz a w r l h
  // a: angle, w: angular velocity, l: r2 - r1, h: z2 - z1
  // 前哨站改用 13 维: x vx y vy z0 vz a w r l h z1 z2
  Eigen::VectorXd x0;
  Eigen::MatrixXd P0;
  if (armor_num == 3 && P0_dig.size() == 13) {
    x0.resize(13);
    x0 << center_x, 0, center_y, 0, center_z, 0, ypr[0], 0, r, 0, 0, center_z + OUTPOST_ARMOR_Z_STEP_M,
      center_z - OUTPOST_ARMOR_Z_STEP_M;
    P0 = P0_dig.asDiagonal();
  } else {
    x0.resize(11);
    x0 << center_x, 0, center_y, 0, center_z, 0, ypr[0], 0, r, 0, 0;
    P0 = P0_dig.asDiagonal();
  }

  // 防止夹角求和出现异常值
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);  //初始化滤波器（预测量、预测量协方差）
}

Target::Target(double x, double vyaw, double radius, double h) : armor_num_(4)
{
  Eigen::VectorXd x0{{x, 0, 0, 0, 0, 0, 0, vyaw, radius, 0, h}};
  Eigen::VectorXd P0_dig{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  // 防止夹角求和出现异常值
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);  //初始化滤波器（预测量、预测量协方差）
}

void Target::predict(std::chrono::steady_clock::time_point t)
{
  auto dt = tools::delta_time(t, t_);
  predict(dt);
  t_ = t;
}

void Target::predict(double dt)
{
  int state_dim = ekf_.x.size();
  double v1, v2;
  auto a = dt * dt * dt * dt / 4;
  auto b = dt * dt * dt / 2;
  auto c = dt * dt;

  if (state_dim == 13) {
    // clang-format off
    Eigen::MatrixXd F(13, 13);
    F << 1, dt,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  1, dt,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  1, dt,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  1, dt,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1;
    // clang-format on

    v1 = 10;
    v2 = 0.1;
    // clang-format off
    Eigen::MatrixXd Q(13, 13);
    Q << a * v1, b * v1,      0,      0,      0,      0,      0,      0, 0, 0, 0,    0,    0,
         b * v1, c * v1,      0,      0,      0,      0,      0,      0, 0, 0, 0,    0,    0,
              0,      0, a * v1, b * v1,      0,      0,      0,      0, 0, 0, 0,    0,    0,
              0,      0, b * v1, c * v1,      0,      0,      0,      0, 0, 0, 0,    0,    0,
              0,      0,      0,      0, a * v1, b * v1,      0,      0, 0, 0, 0,    0,    0,
              0,      0,      0,      0, b * v1, c * v1,      0,      0, 0, 0, 0,    0,    0,
              0,      0,      0,      0,      0,      0, a * v2, b * v2, 0, 0, 0,    0,    0,
              0,      0,      0,      0,      0,      0, b * v2, c * v2, 0, 0, 0,    0,    0,
              0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0,    0,    0,
              0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0,    0,    0,
              0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0,    0,    0,
              0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0, 5e-4,    0,
              0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0,    0, 5e-4;
    // clang-format on

    auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
      Eigen::VectorXd x_prior = F * x;
      x_prior[6] = tools::limit_rad(x_prior[6]);
      return x_prior;
    };

    if (this->convergened() && this->name == ArmorName::outpost && std::abs(this->ekf_.x[7]) > 2) {
      this->ekf_.x[7] = this->ekf_.x[7] > 0 ? 2.51 : -2.51;
    }

    ekf_.predict(F, Q, f);
    return;
  }

  // 状态转移矩阵
  // clang-format off
  Eigen::MatrixXd F{
    {1, dt,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    {0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  1, dt,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  1, dt,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  1, dt,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1}
  };
  // clang-format on

  if (name == ArmorName::outpost) {
    v1 = 10;
    v2 = 0.1;
  } else {
    // 使用 ParamTuner 根据角速度动态选择 v1/v2
    double angular_velocity = ekf_.x[7];  // 当前估计角速度
    const auto & param = debug::ParamTuner::instance().select(angular_velocity);
    v1 = param.v1;
    v2 = param.v2;
  }
  // clang-format off
  Eigen::MatrixXd Q{
    {a * v1, b * v1,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {b * v1, c * v1,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0, a * v1, b * v1,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0, b * v1, c * v1,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0, a * v1, b * v1,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0, b * v1, c * v1,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0, a * v2, b * v2, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0, b * v2, c * v2, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0}
  };
  // clang-format on

  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = F * x;
    x_prior[6] = tools::limit_rad(x_prior[6]);
    return x_prior;
  };

  if (this->convergened() && this->name == ArmorName::outpost && std::abs(this->ekf_.x[7]) > 2) {
    this->ekf_.x[7] = this->ekf_.x[7] > 0 ? 2.51 : -2.51;
  }

  ekf_.predict(F, Q, f);
}

void Target::update(const Armor & armor)
{
  // 装甲板匹配
  int id;
  const std::vector<Eigen::Vector4d> & xyza_list = armor_xyza_list();

  if (name == ArmorName::outpost && armor_num_ == 3 && ekf_.x.size() >= 13) {
    double z0_var = ekf_.P(4, 4);
    double z1_var = ekf_.P(11, 11);
    double z2_var = ekf_.P(12, 12);
    bool z_converged = (z0_var < 0.01) && (z1_var < 0.01) && (z2_var < 0.01) &&
                       (std::abs(ekf_.x[11] - ekf_.x[4]) > 0.03 ||
                        std::abs(ekf_.x[12] - ekf_.x[4]) > 0.03);

    double z_obs = armor.xyz_in_world[2];
    int height_matched_id = match_armor_id(z_obs);
    double min_error = 1e10;
    int final_id = height_matched_id;

    for (int offset = -1; offset <= 1; offset++) {
      int check_id = (height_matched_id + offset + armor_num_) % armor_num_;
      const auto & xyza = xyza_list[check_id];
      double angle_error = std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza[3]));
      double height_error = std::abs(z_obs - xyza[2]) * (z_converged ? 2.0 : 0.5);
      double total_error = angle_error + height_error;

      if (total_error < min_error) {
        min_error = total_error;
        final_id = check_id;
      }
    }

    id = final_id;
  } else {
    auto min_angle_error = 1e10;
    std::vector<std::pair<Eigen::Vector4d, int>> xyza_i_list;
    for (int i = 0; i < armor_num_; i++) {
      xyza_i_list.push_back({xyza_list[i], i});
    }

    std::sort(
      xyza_i_list.begin(), xyza_i_list.end(),
      [](const std::pair<Eigen::Vector4d, int> & a, const std::pair<Eigen::Vector4d, int> & b) {
        Eigen::Vector3d ypd1 = tools::xyz2ypd(a.first.head(3));
        Eigen::Vector3d ypd2 = tools::xyz2ypd(b.first.head(3));
        return ypd1[2] < ypd2[2];
      });

    // 取前3个distance最小的装甲板
    for (int i = 0; i < 3; i++) {
      const auto & xyza = xyza_i_list[i].first;
      Eigen::Vector3d ypd = tools::xyz2ypd(xyza.head(3));
      auto angle_error = std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza[3])) +
                         std::abs(tools::limit_rad(armor.ypd_in_world[0] - ypd[0]));

      if (std::abs(angle_error) < std::abs(min_angle_error)) {
        id = xyza_i_list[i].second;
        min_angle_error = angle_error;
      }
    }
  }

  if (id != 0) jumped = true;

  if (id != last_id) {
    is_switch_ = true;
  } else {
    is_switch_ = false;
  }

  if (is_switch_) switch_count_++;

  last_id = id;
  update_count_++;

  update_ypda(armor, id);
}

int Target::match_armor_id(double z_obs) const
{
  double z0 = ekf_.x[4];
  double z1 = ekf_.x[11];
  double z2 = ekf_.x[12];
  double diff0 = std::abs(z_obs - z0);
  double diff1 = std::abs(z_obs - z1);
  double diff2 = std::abs(z_obs - z2);

  if (diff0 <= diff1 && diff0 <= diff2) return 0;
  if (diff1 <= diff0 && diff1 <= diff2) return 1;
  return 2;
}

void Target::update_ypda(const Armor & armor, int id)
{
  //观测jacobi
  Eigen::MatrixXd H = h_jacobian(ekf_.x, id);
  auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);
  double r_yaw_scale = 1.0;
  if (name == ArmorName::outpost && armor_num_ == 3 && ekf_.x.size() >= 13) {
    double yaw_resid = std::abs(delta_angle);
    r_yaw_scale += std::min(yaw_resid / 0.15, 3.0);
  }

  // 使用 ParamTuner 获取动态 R 参数
  const auto & param = debug::ParamTuner::instance().select(ekf_.x[7]);
  double R_yaw = param.R_yaw;
  double R_pitch = param.R_pitch;
  double R_distance = param.R_distance;

  Eigen::VectorXd R_dig{
    {R_yaw * r_yaw_scale, R_pitch * r_yaw_scale, log(std::abs(delta_angle) + 1) + 1,
     log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + R_distance}};
  Eigen::MatrixXd R = R_dig.asDiagonal();

  // 定义非线性转换函数h: x -> z
  auto h = [&](const Eigen::VectorXd & x) -> Eigen::Vector4d {
    Eigen::VectorXd xyz = h_armor_xyz(x, id);
    Eigen::VectorXd ypd = tools::xyz2ypd(xyz);
    auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
    return {ypd[0], ypd[1], ypd[2], angle};
  };

  // 防止夹角求差出现异常值
  auto z_subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;
    c[0] = tools::limit_rad(c[0]);
    c[1] = tools::limit_rad(c[1]);
    c[3] = tools::limit_rad(c[3]);
    return c;
  };

  const Eigen::VectorXd & ypd = armor.ypd_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;
  Eigen::VectorXd z{{ypd[0], ypd[1], ypd[2], ypr[0]}};  //获得观测量

  ekf_.update(z, H, R, h, z_subtract);

  // 推送 EKF 状态到调试面板
  {
    debug::EKFStateData state;
    state.timestamp = std::chrono::duration<double>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
    state.x = ekf_.x[0];
    state.y = ekf_.x[2];
    state.z = ekf_.x[4];
    state.vx = ekf_.x[1];
    state.vy = ekf_.x[3];
    state.vz = ekf_.x[5];
    state.yaw = ekf_.x[6];
    state.w = ekf_.x[7];
    state.r = ekf_.x[8];
    state.l = (ekf_.x.size() > 9) ? ekf_.x[9] : 0;
    state.h = (ekf_.x.size() > 10) ? ekf_.x[10] : 0;
    state.P_dim = std::min(static_cast<int>(ekf_.P.rows()), 13);
    for (int i = 0; i < state.P_dim; ++i) state.P_diag[i] = ekf_.P(i, i);
    state.v1 = param.v1;
    state.v2 = param.v2;
    state.R_yaw = R_yaw;
    state.R_pitch = R_pitch;
    state.R_distance = R_distance;
    state.nis = ekf_.last_nis;
    state.nis_fail_count = ekf_.recent_nis_failures.size();
    state.converged = is_converged_;
    state.active_param_set = param.name;
    debug::ParamTuner::instance().push_ekf_state(state);
  }
}

Eigen::VectorXd Target::ekf_x() const { return ekf_.x; }

const tools::ExtendedKalmanFilter & Target::ekf() const { return ekf_; }

std::vector<Eigen::Vector4d> Target::armor_xyza_list() const
{
  std::vector<Eigen::Vector4d> _armor_xyza_list;

  for (int i = 0; i < armor_num_; i++) {
    auto angle = tools::limit_rad(ekf_.x[6] + i * 2 * CV_PI / armor_num_);
    Eigen::Vector3d xyz = h_armor_xyz(ekf_.x, i);
    _armor_xyza_list.push_back({xyz[0], xyz[1], xyz[2], angle});
  }
  return _armor_xyza_list;
}

bool Target::diverged() const
{
  auto r_ok = ekf_.x[8] > 0 && ekf_.x[8] < 0.7;
  auto l_ok = ekf_.x[8] + ekf_.x[9] > 0.07 && ekf_.x[8] + ekf_.x[9] < 0.9;

  if (r_ok && l_ok) return false;

  tools::logger()->debug("[Target] r={:.3f}, l={:.3f}", ekf_.x[8], ekf_.x[9]);
  return true;
}

bool Target::convergened()
{
  if (this->name != ArmorName::outpost && update_count_ > 3 && !this->diverged()) {
    is_converged_ = true;
  }

  //前哨站特殊判断
  if (this->name == ArmorName::outpost && update_count_ > 50 && !this->diverged()) {
    is_converged_ = true;
  }

  return is_converged_;
}

// 计算出装甲板中心的坐标（考虑长短轴）
Eigen::Vector3d Target::h_armor_xyz(const Eigen::VectorXd & x, int id) const
{
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  if (name == ArmorName::outpost && armor_num_ == 3 && x.size() >= 13) {
    auto r = x[8];
    auto armor_x = x[0] - r * std::cos(angle);
    auto armor_y = x[2] - r * std::sin(angle);
    double armor_z = (id == 0) ? x[4] : (id == 1 ? x[11] : x[12]);
    return {armor_x, armor_y, armor_z};
  }
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  auto r = (use_l_h) ? x[8] + x[9] : x[8];
  auto armor_x = x[0] - r * std::cos(angle);
  auto armor_y = x[2] - r * std::sin(angle);
  double armor_z;
  if (name == ArmorName::outpost && armor_num_ == 3) {
    armor_z = x[4] + id * OUTPOST_ARMOR_Z_STEP_M;
  } else {
    armor_z = (use_l_h) ? x[4] + x[10] : x[4];
  }

  return {armor_x, armor_y, armor_z};
}

Eigen::MatrixXd Target::h_jacobian(const Eigen::VectorXd & x, int id) const
{
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  if (name == ArmorName::outpost && armor_num_ == 3 && x.size() >= 13) {
    auto r = x[8];
    auto dx_da = r * std::sin(angle);
    auto dy_da = -r * std::cos(angle);
    auto dx_dr = -std::cos(angle);
    auto dy_dr = -std::sin(angle);
    double dz_dz0 = (id == 0) ? 1.0 : 0.0;
    double dz_dz1 = (id == 1) ? 1.0 : 0.0;
    double dz_dz2 = (id == 2) ? 1.0 : 0.0;

    // clang-format off
    Eigen::MatrixXd H_armor_xyza{
      {1, 0, 0, 0,     0, 0, dx_da, 0, dx_dr, 0, 0,     0,     0},
      {0, 0, 1, 0,     0, 0, dy_da, 0, dy_dr, 0, 0,     0,     0},
      {0, 0, 0, 0, dz_dz0, 0,     0, 0,     0, 0, 0, dz_dz1, dz_dz2},
      {0, 0, 0, 0,     0, 0,     1, 0,     0, 0, 0,     0,     0}
    };
    // clang-format on

    Eigen::VectorXd armor_xyz = h_armor_xyz(x, id);
    Eigen::MatrixXd H_armor_ypd = tools::xyz2ypd_jacobian(armor_xyz);
    // clang-format off
    Eigen::MatrixXd H_armor_ypda{
      {H_armor_ypd(0, 0), H_armor_ypd(0, 1), H_armor_ypd(0, 2), 0},
      {H_armor_ypd(1, 0), H_armor_ypd(1, 1), H_armor_ypd(1, 2), 0},
      {H_armor_ypd(2, 0), H_armor_ypd(2, 1), H_armor_ypd(2, 2), 0},
      {                0,                 0,                 0, 1}
    };
    // clang-format on

    return H_armor_ypda * H_armor_xyza;
  }
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  auto r = (use_l_h) ? x[8] + x[9] : x[8];
  auto dx_da = r * std::sin(angle);
  auto dy_da = -r * std::cos(angle);

  auto dx_dr = -std::cos(angle);
  auto dy_dr = -std::sin(angle);
  auto dx_dl = (use_l_h) ? -std::cos(angle) : 0.0;
  auto dy_dl = (use_l_h) ? -std::sin(angle) : 0.0;

  auto dz_dh = (use_l_h) ? 1.0 : 0.0;

  // clang-format off
  Eigen::MatrixXd H_armor_xyza{
    {1, 0, 0, 0, 0, 0, dx_da, 0, dx_dr, dx_dl,     0},
    {0, 0, 1, 0, 0, 0, dy_da, 0, dy_dr, dy_dl,     0},
    {0, 0, 0, 0, 1, 0,     0, 0,     0,     0, dz_dh},
    {0, 0, 0, 0, 0, 0,     1, 0,     0,     0,     0}
  };
  // clang-format on

  Eigen::VectorXd armor_xyz = h_armor_xyz(x, id);
  Eigen::MatrixXd H_armor_ypd = tools::xyz2ypd_jacobian(armor_xyz);
  // clang-format off
  Eigen::MatrixXd H_armor_ypda{
    {H_armor_ypd(0, 0), H_armor_ypd(0, 1), H_armor_ypd(0, 2), 0},
    {H_armor_ypd(1, 0), H_armor_ypd(1, 1), H_armor_ypd(1, 2), 0},
    {H_armor_ypd(2, 0), H_armor_ypd(2, 1), H_armor_ypd(2, 2), 0},
    {                0,                 0,                 0, 1}
  };
  // clang-format on

  return H_armor_ypda * H_armor_xyza;
}

bool Target::checkinit() { return isinit; }

}  // namespace auto_aim
