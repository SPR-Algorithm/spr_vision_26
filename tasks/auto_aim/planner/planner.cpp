#include "planner.hpp"

#include <vector>

#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"
#include "tools/yaml.hpp"

using namespace std::chrono_literals;

namespace auto_aim {
Planner::Planner(const std::string &config_path) {
  auto yaml = tools::load(config_path);
  yaw_offset_ = tools::read<double>(yaml, "yaw_offset") / 57.3;
  pitch_offset_ = tools::read<double>(yaml, "pitch_offset") / 57.3;
  fire_thresh_ = tools::read<double>(yaml, "fire_thresh");
  decision_speed_ = tools::read<double>(yaml, "decision_speed");
  high_speed_delay_time_ = tools::read<double>(yaml, "high_speed_delay_time");
  low_speed_delay_time_ = tools::read<double>(yaml, "low_speed_delay_time");

  setup_yaw_solver(config_path);
  setup_pitch_solver(config_path);
}

Plan Planner::plan(Target target, double bullet_speed) {
  // 0. Check bullet speed
  if (bullet_speed < 10 || bullet_speed > 25) {
    bullet_speed = 22;
  }

  locked_aim_id_ = -1;
  if (target.name == ArmorName::outpost && target.armor_xyza_list().size() == 3) {
    locked_aim_id_ = choose_outpost_aim_id(target);
  }

  // 1. Predict fly_time
  auto aim_xyza = choose_aim_xyza(target);
  Eigen::Vector3d xyz = aim_xyza.head<3>();
  auto min_dist = xyz.head<2>().norm();
  auto bullet_traj = tools::Trajectory(bullet_speed, min_dist, xyz.z());
  target.predict(bullet_traj.fly_time);

  // 2. Get trajectory
  double yaw0;
  Trajectory traj;
  try {
    yaw0 = aim(target, bullet_speed)(0);
    traj = get_trajectory(target, yaw0, bullet_speed);
  } catch (const std::exception &e) {
    tools::logger()->warn("Unsolvable target {:.2f}", bullet_speed);
    return {false};
  }

  // 3. Solve yaw
  Eigen::VectorXd x0(2);
  x0 << traj(0, 0), traj(1, 0);
  tiny_set_x0(yaw_solver_, x0);

  yaw_solver_->work->Xref = traj.block(0, 0, 2, HORIZON);
  tiny_solve(yaw_solver_);

  // 4. Solve pitch
  x0 << traj(2, 0), traj(3, 0);
  tiny_set_x0(pitch_solver_, x0);

  pitch_solver_->work->Xref = traj.block(2, 0, 2, HORIZON);
  tiny_solve(pitch_solver_);

  Plan plan;
  plan.control = true;

  plan.target_yaw = tools::limit_rad(traj(0, HALF_HORIZON) + yaw0);
  plan.target_pitch = traj(2, HALF_HORIZON);

  plan.yaw = tools::limit_rad(yaw_solver_->work->x(0, HALF_HORIZON) + yaw0);
  plan.yaw_vel = yaw_solver_->work->x(1, HALF_HORIZON);
  plan.yaw_acc = yaw_solver_->work->u(0, HALF_HORIZON);

  plan.pitch = pitch_solver_->work->x(0, HALF_HORIZON);
  plan.pitch_vel = pitch_solver_->work->x(1, HALF_HORIZON);
  plan.pitch_acc = pitch_solver_->work->u(0, HALF_HORIZON);

  auto shoot_offset_ = 2;
  plan.fire =
      std::hypot(traj(0, HALF_HORIZON + shoot_offset_) -
                     yaw_solver_->work->x(0, HALF_HORIZON + shoot_offset_),
                 traj(2, HALF_HORIZON + shoot_offset_) -
                     pitch_solver_->work->x(0, HALF_HORIZON + shoot_offset_)) <
      fire_thresh_;
  return plan;
}

Plan Planner::plan(std::optional<Target> target, double bullet_speed) {
  if (!target.has_value())
    return {false};

  double delay_time = std::abs(target->ekf_x()[7]) > decision_speed_
                          ? high_speed_delay_time_
                          : low_speed_delay_time_;

  auto future = std::chrono::steady_clock::now() +
                std::chrono::microseconds(int(delay_time * 1e6));

  target->predict(future);

  return plan(*target, bullet_speed);
}

void Planner::setup_yaw_solver(const std::string &config_path) {
  auto yaml = tools::load(config_path);
  auto max_yaw_acc = tools::read<double>(yaml, "max_yaw_acc");
  auto Q_yaw = tools::read<std::vector<double>>(yaml, "Q_yaw");
  auto R_yaw = tools::read<std::vector<double>>(yaml, "R_yaw");

  Eigen::MatrixXd A{{1, DT}, {0, 1}};
  Eigen::MatrixXd B{{0}, {DT}};
  Eigen::VectorXd f{{0, 0}};
  Eigen::Matrix<double, 2, 1> Q(Q_yaw.data());
  Eigen::Matrix<double, 1, 1> R(R_yaw.data());
  tiny_setup(&yaw_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1,
             HORIZON, 0);

  Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
  Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
  Eigen::MatrixXd u_min =
      Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_yaw_acc);
  Eigen::MatrixXd u_max =
      Eigen::MatrixXd::Constant(1, HORIZON - 1, max_yaw_acc);
  tiny_set_bound_constraints(yaw_solver_, x_min, x_max, u_min, u_max);

  yaw_solver_->settings->max_iter = 10;
}

void Planner::setup_pitch_solver(const std::string &config_path) {
  auto yaml = tools::load(config_path);
  auto max_pitch_acc = tools::read<double>(yaml, "max_pitch_acc");
  auto Q_pitch = tools::read<std::vector<double>>(yaml, "Q_pitch");
  auto R_pitch = tools::read<std::vector<double>>(yaml, "R_pitch");

  Eigen::MatrixXd A{{1, DT}, {0, 1}};
  Eigen::MatrixXd B{{0}, {DT}};
  Eigen::VectorXd f{{0, 0}};
  Eigen::Matrix<double, 2, 1> Q(Q_pitch.data());
  Eigen::Matrix<double, 1, 1> R(R_pitch.data());
  tiny_setup(&pitch_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1,
             HORIZON, 0);

  Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
  Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
  Eigen::MatrixXd u_min =
      Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_pitch_acc);
  Eigen::MatrixXd u_max =
      Eigen::MatrixXd::Constant(1, HORIZON - 1, max_pitch_acc);
  tiny_set_bound_constraints(pitch_solver_, x_min, x_max, u_min, u_max);

  pitch_solver_->settings->max_iter = 10;
}

int Planner::choose_outpost_aim_id(const Target &target) const
{
  const auto armor_xyza_list = target.armor_xyza_list();
  const auto armor_num = static_cast<int>(armor_xyza_list.size());
  const Eigen::VectorXd ekf_x = target.ekf_x();
  const auto center_yaw = std::atan2(ekf_x[2], ekf_x[0]);

  // 正对云台的一块板（|delta_angle| 最小）即当前最易看见的那层
  int facing_id = 0;
  auto min_abs_delta = 1e10;
  for (int i = 0; i < armor_num; i++) {
    const auto delta_angle =
      tools::limit_rad(armor_xyza_list[i][3] - center_yaw);
    const auto abs_delta = std::abs(delta_angle);
    if (abs_delta < min_abs_delta) {
      min_abs_delta = abs_delta;
      facing_id = i;
    }
  }

  if (!target.jumped) {
    return facing_id;
  }

  constexpr double coming_angle = 70 / 57.3;
  constexpr double leaving_angle = 30 / 57.3;
  for (int i = 0; i < armor_num; i++) {
    const auto delta_angle =
      tools::limit_rad(armor_xyza_list[i][3] - center_yaw);
    if (std::abs(delta_angle) > coming_angle) {
      continue;
    }
    if (ekf_x[7] > 0 && delta_angle < leaving_angle) {
      return i;
    }
    if (ekf_x[7] < 0 && delta_angle > -leaving_angle) {
      return i;
    }
  }

  return facing_id;
}

Eigen::Vector4d Planner::choose_min_dist_aim_xyza(const Target &target) const
{
  const auto armor_xyza_list = target.armor_xyza_list();
  if (armor_xyza_list.empty()) {
    throw std::runtime_error("Empty armor_xyza_list");
  }

  auto min_dist = 1e10;
  Eigen::Vector4d best = armor_xyza_list[0];
  for (const auto &xyza : armor_xyza_list) {
    const auto dist = xyza.head<2>().norm();
    if (dist < min_dist) {
      min_dist = dist;
      best = xyza;
    }
  }
  return best;
}

Eigen::Vector4d Planner::choose_aim_xyza(const Target &target) const
{
  const auto armor_xyza_list = target.armor_xyza_list();
  if (armor_xyza_list.empty()) {
    throw std::runtime_error("Empty armor_xyza_list");
  }

  if (target.name == ArmorName::outpost && armor_xyza_list.size() == 3) {
    const int id = (locked_aim_id_ >= 0) ? locked_aim_id_ : choose_outpost_aim_id(target);
    return armor_xyza_list[id];
  }

  return choose_min_dist_aim_xyza(target);
}

int Planner::reprojection_aim_id(const Target &target) const
{
  const auto armor_xyza_list = target.armor_xyza_list();
  if (armor_xyza_list.empty()) {
    throw std::runtime_error("Empty armor_xyza_list");
  }

  if (target.name == ArmorName::outpost && armor_xyza_list.size() == 3) {
    return choose_outpost_aim_id(target);
  }

  auto min_dist = 1e10;
  int best_id = 0;
  for (int i = 0; i < static_cast<int>(armor_xyza_list.size()); i++) {
    const auto dist = armor_xyza_list[i].head<2>().norm();
    if (dist < min_dist) {
      min_dist = dist;
      best_id = i;
    }
  }
  return best_id;
}

Eigen::Vector4d Planner::reprojection_aim_xyza(const Target &target) const
{
  const auto armor_xyza_list = target.armor_xyza_list();
  if (armor_xyza_list.empty()) {
    throw std::runtime_error("Empty armor_xyza_list");
  }

  return armor_xyza_list[reprojection_aim_id(target)];
}

Eigen::Matrix<double, 2, 1> Planner::aim(const Target &target,
                                         double bullet_speed) {
  const auto aim_xyza = choose_aim_xyza(target);
  debug_xyza = aim_xyza;

  const Eigen::Vector3d xyz = aim_xyza.head<3>();
  const auto min_dist = xyz.head<2>().norm();
  const auto azim = std::atan2(xyz.y(), xyz.x());
  auto bullet_traj = tools::Trajectory(bullet_speed, min_dist, xyz.z());
  if (bullet_traj.unsolvable)
    throw std::runtime_error("Unsolvable bullet trajectory!");

  return {tools::limit_rad(azim + yaw_offset_),
          -bullet_traj.pitch - pitch_offset_};
}

Trajectory Planner::get_trajectory(Target &target, double yaw0,
                                   double bullet_speed) {
  Trajectory traj;

  target.predict(-DT * (HALF_HORIZON + 1));
  auto yaw_pitch_last = aim(target, bullet_speed);

  target.predict(DT); // [0] = -HALF_HORIZON * DT -> [HHALF_HORIZON] = 0
  auto yaw_pitch = aim(target, bullet_speed);

  for (int i = 0; i < HORIZON; i++) {
    target.predict(DT);
    auto yaw_pitch_next = aim(target, bullet_speed);

    auto yaw_vel =
        tools::limit_rad(yaw_pitch_next(0) - yaw_pitch_last(0)) / (2 * DT);
    auto pitch_vel = (yaw_pitch_next(1) - yaw_pitch_last(1)) / (2 * DT);

    traj.col(i) << tools::limit_rad(yaw_pitch(0) - yaw0), yaw_vel, yaw_pitch(1),
        pitch_vel;

    yaw_pitch_last = yaw_pitch;
    yaw_pitch = yaw_pitch_next;
  }

  return traj;
}

} // namespace auto_aim