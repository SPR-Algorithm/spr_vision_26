#include "phase_motion_estimator.hpp"

#include <Eigen/QR>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace auto_buff
{
namespace
{
constexpr double PI = 3.14159265358979323846;
constexpr double SLOT_ANGLE_RAD = 2.0 * PI / 5.0;
constexpr double SMALL_RATE_RAD_S = PI / 3.0;
constexpr double DIRECTION_INTERVAL_S = 0.12;
constexpr double MAX_NORMAL_STEP_RAD = 0.52;

double seconds(FrameTimestamp end, FrameTimestamp begin)
{
  return std::chrono::duration<double>(end - begin).count();
}

int sign(double value) { return value > 0.0 ? 1 : (value < 0.0 ? -1 : 0); }

double median(std::vector<double> values)
{
  const auto middle = values.begin() + values.size() / 2;
  std::nth_element(values.begin(), middle, values.end());
  return *middle;
}

struct LineFit
{
  double rate = 0.0;
  double rms = std::numeric_limits<double>::infinity();
};

template<typename Samples>
LineFit fit_line(const Samples & samples)
{
  if (samples.size() < 2) return {};
  double sum_w = 0.0, sum_t = 0.0, sum_y = 0.0;
  for (const auto & sample : samples) {
    const double t = seconds(sample.timestamp, samples.front().timestamp);
    sum_w += sample.weight;
    sum_t += sample.weight * t;
    sum_y += sample.weight * sample.phase_rad;
  }
  const double mean_t = sum_t / sum_w;
  const double mean_y = sum_y / sum_w;
  double covariance = 0.0, variance = 0.0;
  for (const auto & sample : samples) {
    const double t = seconds(sample.timestamp, samples.front().timestamp) - mean_t;
    covariance += sample.weight * t * (sample.phase_rad - mean_y);
    variance += sample.weight * t * t;
  }
  if (variance < 1e-9) return {};
  LineFit result;
  result.rate = covariance / variance;
  double squared_error = 0.0;
  for (const auto & sample : samples) {
    const double t = seconds(sample.timestamp, samples.front().timestamp) - mean_t;
    const double residual = sample.phase_rad - mean_y - result.rate * t;
    squared_error += sample.weight * residual * residual;
  }
  result.rms = std::sqrt(squared_error / sum_w);
  return result;
}

struct HarmonicFit
{
  Eigen::Vector4d coefficients = Eigen::Vector4d::Zero();
  double frequency = 0.0;
  double rms = std::numeric_limits<double>::infinity();
  double robust_cost = std::numeric_limits<double>::infinity();
};

double fitted_phase(const HarmonicFit & fit, double t)
{
  return fit.coefficients[0] * std::cos(fit.frequency * t) +
         fit.coefficients[1] * std::sin(fit.frequency * t) +
         fit.coefficients[2] * t + fit.coefficients[3];
}

// At fixed frequency the other four phase coefficients form a linear problem. IRLS
// prevents a few otherwise valid PnP frames from dominating the fit.
HarmonicFit fit_frequency(
  const Eigen::VectorXd & times, const Eigen::VectorXd & phases,
  const Eigen::VectorXd & base_weights, double frequency, double noise_std)
{
  Eigen::MatrixXd design(times.size(), 4);
  for (Eigen::Index row = 0; row < times.size(); ++row) {
    design.row(row) << std::cos(frequency * times[row]), std::sin(frequency * times[row]),
      times[row], 1.0;
  }
  Eigen::VectorXd weights = base_weights;
  HarmonicFit fit;
  fit.frequency = frequency;
  for (int iteration = 0; iteration < 5; ++iteration) {
    const Eigen::VectorXd roots = weights.array().sqrt();
    const Eigen::MatrixXd weighted_design = roots.asDiagonal() * design;
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(weighted_design);
    qr.setThreshold(1e-8);
    if (qr.rank() != 4) return fit;
    fit.coefficients = qr.solve(roots.asDiagonal() * phases);
    if (!fit.coefficients.allFinite()) return HarmonicFit{};
    const Eigen::VectorXd residuals = phases - design * fit.coefficients;
    std::vector<double> absolute_residuals;
    absolute_residuals.reserve(static_cast<std::size_t>(residuals.size()));
    for (double residual : residuals) absolute_residuals.push_back(std::abs(residual));
    const double huber = std::max(1.5 * noise_std, 2.0 * 1.4826 * median(absolute_residuals));
    for (Eigen::Index row = 0; row < times.size(); ++row) {
      weights[row] = base_weights[row] *
        std::min(1.0, huber / std::max(1e-12, std::abs(residuals[row])));
    }
  }
  const Eigen::VectorXd residuals = phases - design * fit.coefficients;
  fit.rms = std::sqrt(
    (base_weights.array() * residuals.array().square()).sum() / base_weights.sum());
  const double huber = 2.0 * noise_std;
  fit.robust_cost = 0.0;
  for (Eigen::Index row = 0; row < times.size(); ++row) {
    const double residual = std::abs(residuals[row]);
    fit.robust_cost += base_weights[row] *
      (residual <= huber ? residual * residual : huber * (2.0 * residual - huber));
  }
  return fit;
}

double harmonic_phase(const MotionState & state, FrameTimestamp timestamp)
{
  const double t = seconds(timestamp, state.model_epoch);
  const auto & p = state.harmonic_phase;
  return p.cos_amplitude_rad * std::cos(p.angular_frequency_rad_s * t) +
         p.sin_amplitude_rad * std::sin(p.angular_frequency_rad_s * t) +
         p.linear_rate_rad_s * t + p.phase_offset_rad;
}

double harmonic_rate(const MotionState & state, FrameTimestamp timestamp)
{
  const double t = seconds(timestamp, state.model_epoch);
  const auto & p = state.harmonic_phase;
  return p.angular_frequency_rad_s *
           (-p.cos_amplitude_rad * std::sin(p.angular_frequency_rad_s * t) +
            p.sin_amplitude_rad * std::cos(p.angular_frequency_rad_s * t)) +
         p.linear_rate_rad_s;
}
}  // namespace

PhaseMotionEstimator::PhaseMotionEstimator(PhaseMotionEstimatorConfig config)
: config_(std::move(config))
{
  const std::array<double, 13> positive_values = {
    config_.max_observation_gap_s, config_.small_window_s, config_.small_min_span_s,
    config_.small_rate_tolerance_rad_s, config_.big_window_s, config_.big_min_span_s,
    config_.big_refit_interval_s, config_.big_min_frequency_rad_s,
    config_.big_max_frequency_rad_s, config_.max_angular_speed_rad_s,
    config_.phase_noise_std_rad, config_.max_fit_rms_rad,
    static_cast<double>(config_.max_history_samples)};
  if (
    !std::all_of(positive_values.begin(), positive_values.end(), [](double value) {
      return std::isfinite(value) && value > 0.0;
    }) || config_.small_min_span_s >= config_.small_window_s ||
    config_.big_min_span_s >= config_.big_window_s ||
    config_.big_min_frequency_rad_s >= config_.big_max_frequency_rad_s ||
    config_.max_history_samples < 80 || config_.max_angular_speed_rad_s <= SMALL_RATE_RAD_S) {
    throw std::invalid_argument("auto_buff: invalid phase motion estimator configuration");
  }
}

std::optional<RuneTarget> PhaseMotionEstimator::update(
  const RunePose & pose, const ReconstructedBlade & target_blade, FrameTimestamp timestamp)
{
  const auto & observation = target_blade.observation;
  const int slot = static_cast<int>(target_blade.slot);
  const Eigen::Vector3d displacement = target_blade.armor_center_world_m - pose.position_world_m;
  if (
    !observation.valid() || !observation.state ||
    (observation.state->kind != RuneKind::SMALL && observation.state->kind != RuneKind::BIG) ||
    slot < 1 || slot > 5 || pose.source == PoseSource::UNKNOWN ||
    !pose.position_world_m.allFinite() || !pose.plane_normal_world.allFinite() ||
    !pose.phase_zero_direction_world.allFinite() || !pose.orientation_world.coeffs().allFinite() ||
    !std::isfinite(pose.orientation_world.squaredNorm()) ||
    pose.orientation_world.squaredNorm() < 1e-12 || !displacement.allFinite() ||
    !std::isfinite(pose.plane_normal_world.norm()) ||
    pose.plane_normal_world.norm() < 1e-6 || !std::isfinite(pose.blade_radius_m) ||
    pose.blade_radius_m <= 1e-6 || !std::isfinite(pose.reprojection_error_px) ||
    pose.reprojection_error_px < 0.0 || !std::isfinite(pose.confidence) ||
    pose.confidence <= 0.0F || pose.confidence > 1.0F ||
    !std::isfinite(observation.detection_confidence) ||
    observation.detection_confidence <= 0.0F || observation.detection_confidence > 1.0F) {
    restart("INVALID_GEOMETRY");
    return std::nullopt;
  }

  const RuneKind kind = observation.state->kind;
  const double dt_s = history_.empty() ? 0.0 : seconds(timestamp, history_.back().timestamp);
  const char * restarted = nullptr;
  if (kind_ && *kind_ != kind) {
    restarted = "KIND_CHANGED";
  } else if (!history_.empty() && dt_s <= 0.0) {
    restarted = "TIMESTAMP_RESET";
  } else if (!history_.empty() &&
             (dt_s > config_.max_observation_gap_s ||
              config_.max_angular_speed_rad_s * dt_s + 2.0 * config_.phase_noise_std_rad >=
                SLOT_ANGLE_RAD / 2.0)) {
    // Below half a slot is required to distinguish a blade switch from physical rotation.
    restarted = "GAP_RESET";
  }
  if (restarted) restart(restarted);

  Eigen::Vector3d normal = pose.plane_normal_world.normalized();
  if (!history_.empty()) {
    if (normal.dot(last_normal_world_) < 0.0) normal = -normal;
    if (normal.dot(last_normal_world_) < std::cos(MAX_NORMAL_STEP_RAD)) {
      // Reject isolated planar-PnP branch flips without moving the valid basis/history.
      // A persistent discrepancy eventually exceeds the accepted-observation gap above.
      diagnostics_.status = "POSE_JUMP";
      diagnostics_.observed_phase_rad.reset();
      return std::nullopt;
    }
  }
  if (history_.empty()) {
    kind_ = kind;
    // S4's +Z rotates with its selected blade. Choose an independent world direction once,
    // then project that same direction onto each subsequent observed plane.
    reference_zero_world_ = Eigen::Vector3d::UnitZ();
    if (std::abs(reference_zero_world_.dot(normal)) > 0.90) {
      reference_zero_world_ = Eigen::Vector3d::UnitY();
    }
  }
  Eigen::Vector3d zero = reference_zero_world_ - normal * normal.dot(reference_zero_world_);
  const Eigen::Vector3d radial = displacement - normal * normal.dot(displacement);
  if (
    zero.norm() < 0.1 || radial.norm() < 1e-6 ||
    std::abs(radial.norm() - pose.blade_radius_m) > 0.35 * pose.blade_radius_m ||
    std::abs(normal.dot(displacement)) > 0.15 * pose.blade_radius_m) {
    restart("INVALID_GEOMETRY");
    return std::nullopt;
  }
  zero.normalize();
  const Eigen::Vector3d unit_radial = radial.normalized();
  const double wrapped = std::atan2(normal.dot(zero.cross(unit_radial)), zero.dot(unit_radial));
  double phase = wrapped;
  if (!history_.empty()) {
    const double actual_dt = seconds(timestamp, history_.back().timestamp);
    double predicted_rate = kalman_direction_ * SMALL_RATE_RAD_S;
    if (kind == RuneKind::BIG) {
      predicted_rate = harmonic_model_ ? harmonic_rate(*harmonic_model_, timestamp) : 0.0;
    }
    const double raw_delta = std::remainder(wrapped - last_wrapped_phase_rad_, 2.0 * PI);
    const double prediction = predicted_rate * actual_dt;
    const int switch_steps = static_cast<int>(std::lround((raw_delta - prediction) / SLOT_ANGLE_RAD));
    const double delta = raw_delta - switch_steps * SLOT_ANGLE_RAD;
    if (
      std::abs(delta) > config_.max_angular_speed_rad_s * actual_dt +
        3.0 * config_.phase_noise_std_rad) {
      restart("PHASE_DISCONTINUITY");
      return std::nullopt;
    }
    if (switch_steps != 0) {
      const double shift = switch_steps * SLOT_ANGLE_RAD;
      for (auto & sample : history_) sample.phase_rad += shift;
      kalman_phase_rad_ += shift;
      if (harmonic_model_) {
        harmonic_model_->harmonic_phase.phase_offset_rad += shift;
        harmonic_model_->continuous_phase_rad += shift;
      }
      ++diagnostics_.blade_switches;
      diagnostics_.cumulative_switch_offset_rad += shift;
    }
    phase = history_.back().phase_rad + delta;
  }

  last_normal_world_ = normal;
  last_wrapped_phase_rad_ = wrapped;
  const double weight = std::clamp(
    static_cast<double>(std::min(pose.confidence, observation.detection_confidence)), 0.05, 1.0);
  history_.push_back({timestamp, phase, weight});
  const double window_s = kind == RuneKind::SMALL ? config_.small_window_s : config_.big_window_s;
  while (history_.size() > config_.max_history_samples ||
         (history_.size() > 1 && seconds(timestamp, history_.front().timestamp) > window_s)) {
    history_.pop_front();
  }
  for (auto it = history_.rbegin(); it != history_.rend(); ++it) {
    if (seconds(timestamp, it->timestamp) < DIRECTION_INTERVAL_S) continue;
    const double change = phase - it->phase_rad;
    if (std::abs(change) >= 2.0 * config_.phase_noise_std_rad) {
      direction_votes_.push_back(sign(change));
      if (direction_votes_.size() > 25) direction_votes_.pop_front();
    }
    break;
  }
  diagnostics_.samples = history_.size();
  diagnostics_.span_s = seconds(timestamp, history_.front().timestamp);
  diagnostics_.observed_phase_rad = phase;
  diagnostics_.phase_zero_direction_world = zero;
  diagnostics_.plane_normal_world = normal;
  diagnostics_.direction_votes = 0;
  for (int vote : direction_votes_) diagnostics_.direction_votes += vote;
  diagnostics_.status = "COLLECTING";
  const auto motion = kind == RuneKind::SMALL ?
    estimate_small(timestamp, history_.size() == 1 ? 0.0 : dt_s) : estimate_big(timestamp);
  if (!motion) {
    if (restarted) diagnostics_.status = restarted;
    return std::nullopt;
  }

  RuneTarget target;
  target.observation = observation;
  target.pose = pose;
  target.pose.plane_normal_world = normal;
  target.pose.phase_zero_direction_world = zero;
  Eigen::Matrix3d basis;
  basis.col(0) = normal;
  basis.col(1) = zero.cross(normal);
  basis.col(2) = zero;
  target.pose.orientation_world = Eigen::Quaterniond(basis).normalized();
  target.motion = *motion;
  target.slot = target_blade.slot;
  target.quality = pose.source == PoseSource::PNP ? TrackingQuality::DEGRADED : observation.quality;
  diagnostics_.status = "CONVERGED";
  return target;
}

int PhaseMotionEstimator::voted_direction() const
{
  if (direction_votes_.size() < 8) return 0;
  int sum = 0;
  for (int vote : direction_votes_) sum += vote;
  if (std::abs(sum) < 0.60 * direction_votes_.size()) return 0;
  return sign(sum);
}

std::optional<MotionState> PhaseMotionEstimator::estimate_small(FrameTimestamp timestamp, double dt_s)
{
  const int direction = voted_direction();
  if (direction == 0) {
    kalman_direction_ = 0;
    diagnostics_.status = "DIRECTION_UNCERTAIN";
    return std::nullopt;
  }
  const double measurement = history_.back().phase_rad;
  const double measurement_variance = config_.phase_noise_std_rad * config_.phase_noise_std_rad;
  if (kalman_direction_ != direction) {
    kalman_phase_rad_ = measurement;
    kalman_variance_ = measurement_variance;
    kalman_direction_ = direction;
  } else {
    kalman_phase_rad_ += direction * SMALL_RATE_RAD_S * dt_s;
    kalman_variance_ += 0.0016 * dt_s;
    const double gain = kalman_variance_ / (kalman_variance_ + measurement_variance);
    kalman_phase_rad_ += gain * (measurement - kalman_phase_rad_);
    kalman_variance_ *= 1.0 - gain;
  }
  if (history_.size() < 12 || diagnostics_.span_s < config_.small_min_span_s) return std::nullopt;
  const auto fit = fit_line(history_);
  diagnostics_.fit_rms_rad = fit.rms;
  if (
    !std::isfinite(fit.rate) || !std::isfinite(fit.rms) || fit.rms > config_.max_fit_rms_rad ||
    sign(fit.rate) != direction ||
    std::abs(fit.rate - direction * SMALL_RATE_RAD_S) > config_.small_rate_tolerance_rad_s ||
    std::abs(history_.back().phase_rad - history_.front().phase_rad) < 0.30 ||
    std::abs(kalman_phase_rad_ - measurement) > 3.0 * config_.max_fit_rms_rad) {
    diagnostics_.status = "SMALL_RATE_INCONSISTENT";
    return std::nullopt;
  }
  MotionState motion;
  motion.continuous_phase_rad = kalman_phase_rad_;
  motion.angular_velocity_rad_s = direction * SMALL_RATE_RAD_S;
  motion.direction = direction > 0 ? RotationDirection::COUNTER_CLOCKWISE : RotationDirection::CLOCKWISE;
  motion.reference_timestamp = timestamp;
  motion.model_epoch = timestamp;
  motion.model = MotionModel::CONSTANT_VELOCITY;
  motion.confidence = static_cast<float>(std::clamp(1.0 - fit.rms / config_.max_fit_rms_rad, 0.0, 1.0));
  motion.converged = true;
  return motion;
}

std::optional<MotionState> PhaseMotionEstimator::estimate_big(FrameTimestamp timestamp)
{
  if (history_.size() < 80 || diagnostics_.span_s < config_.big_min_span_s) return std::nullopt;
  const int direction = voted_direction();
  if (direction == 0) {
    diagnostics_.status = "DIRECTION_UNCERTAIN";
    harmonic_model_.reset();
    return std::nullopt;
  }
  if (!last_fit_timestamp_ || seconds(timestamp, *last_fit_timestamp_) >= config_.big_refit_interval_s) {
    last_fit_timestamp_ = timestamp;
    harmonic_model_.reset();
    // Sample decimation bounds QR/IRLS work; every retained sample uses its actual timestamp.
    const auto count = static_cast<Eigen::Index>(std::min<std::size_t>(history_.size(), 240));
    Eigen::VectorXd times(count), phases(count), weights(count);
    for (Eigen::Index i = 0; i < count; ++i) {
      const auto index = static_cast<std::size_t>(i) * (history_.size() - 1) /
        static_cast<std::size_t>(count - 1);
      times[i] = seconds(history_[index].timestamp, history_.front().timestamp);
      phases[i] = history_[index].phase_rad;
      weights[i] = history_[index].weight;
    }
    HarmonicFit best;
    constexpr int FREQUENCY_STEPS = 60;
    for (int step = 0; step <= FREQUENCY_STEPS; ++step) {
      const double frequency = config_.big_min_frequency_rad_s +
        (config_.big_max_frequency_rad_s - config_.big_min_frequency_rad_s) * step / FREQUENCY_STEPS;
      const auto fit = fit_frequency(times, phases, weights, frequency, config_.phase_noise_std_rad);
      if (fit.robust_cost < best.robust_cost) best = fit;
    }
    // Variable-projection LM: each trial frequency re-solves the four linear coefficients
    // with IRLS. Damping keeps the frequency update inside the configured search range.
    double damping = 1e-3;
    for (int iteration = 0; iteration < 12 && std::isfinite(best.robust_cost); ++iteration) {
      constexpr double EPSILON = 1e-3;
      const double lower_frequency = std::max(config_.big_min_frequency_rad_s, best.frequency - EPSILON);
      const double upper_frequency = std::min(config_.big_max_frequency_rad_s, best.frequency + EPSILON);
      const auto lower = fit_frequency(times, phases, weights, lower_frequency, config_.phase_noise_std_rad);
      const auto upper = fit_frequency(times, phases, weights, upper_frequency, config_.phase_noise_std_rad);
      const double half_width = (upper_frequency - lower_frequency) * 0.5;
      if (half_width < 1e-9 || !std::isfinite(lower.robust_cost) || !std::isfinite(upper.robust_cost)) break;
      double gradient = 0.0, curvature = 0.0;
      for (Eigen::Index row = 0; row < count; ++row) {
        const double residual = phases[row] - fitted_phase(best, times[row]);
        const double derivative = (fitted_phase(upper, times[row]) -
          fitted_phase(lower, times[row])) / (2.0 * half_width);
        const double robust_weight = weights[row] * std::min(1.0,
          2.0 * config_.phase_noise_std_rad / std::max(1e-12, std::abs(residual)));
        gradient += robust_weight * derivative * residual;
        curvature += robust_weight * derivative * derivative;
      }
      const double step = std::clamp(gradient / (curvature + damping), -0.10, 0.10);
      const double frequency = std::clamp(best.frequency + step,
        config_.big_min_frequency_rad_s, config_.big_max_frequency_rad_s);
      const auto candidate = fit_frequency(times, phases, weights, frequency, config_.phase_noise_std_rad);
      if (candidate.robust_cost < best.robust_cost) {
        best = candidate;
        damping *= 0.3;
        if (std::abs(step) < 1e-5) break;
      } else {
        damping *= 10.0;
      }
    }
    diagnostics_.fit_rms_rad = best.rms;
    const double rate_amplitude = best.frequency * std::hypot(best.coefficients[0], best.coefficients[1]);
    const double mean_rate = best.coefficients[2];
    // Static/constant-speed data and less than one modulation period cannot establish a
    // large-rune model. Rate bounds apply across a full cycle, including future extrema.
    if (
      !best.coefficients.allFinite() || !std::isfinite(best.rms) || best.rms > config_.max_fit_rms_rad ||
      best.frequency * diagnostics_.span_s < 2.0 * PI ||
      sign(mean_rate) != direction || rate_amplitude < 0.15 ||
      std::abs(mean_rate) < rate_amplitude ||
      std::abs(mean_rate) + rate_amplitude > config_.max_angular_speed_rad_s ||
      std::abs(phases[count - 1] - phases[0]) < 2.0) {
      diagnostics_.status = "HARMONIC_UNCONVERGED";
      return std::nullopt;
    }
    MotionState motion;
    motion.model = MotionModel::HARMONIC_PHASE;
    motion.model_epoch = history_.front().timestamp;
    motion.harmonic_phase = {best.coefficients[0], best.coefficients[1], best.coefficients[2],
      best.coefficients[3], best.frequency};
    motion.direction = direction > 0 ? RotationDirection::COUNTER_CLOCKWISE : RotationDirection::CLOCKWISE;
    motion.confidence = static_cast<float>(std::clamp(1.0 - best.rms / config_.max_fit_rms_rad, 0.0, 1.0));
    motion.converged = true;
    harmonic_model_ = motion;
  }
  if (!harmonic_model_) {
    diagnostics_.status = "HARMONIC_UNCONVERGED";
    return std::nullopt;
  }
  MotionState motion = *harmonic_model_;
  motion.reference_timestamp = timestamp;
  motion.continuous_phase_rad = harmonic_phase(motion, timestamp);
  motion.angular_velocity_rad_s = harmonic_rate(motion, timestamp);
  if (
    !std::isfinite(motion.continuous_phase_rad) || !std::isfinite(motion.angular_velocity_rad_s) ||
    sign(motion.angular_velocity_rad_s) != direction ||
    std::abs(motion.angular_velocity_rad_s) > config_.max_angular_speed_rad_s ||
    std::abs(motion.continuous_phase_rad - history_.back().phase_rad) > 3.0 * config_.max_fit_rms_rad) {
    harmonic_model_.reset();
    diagnostics_.status = "HARMONIC_INNOVATION";
    return std::nullopt;
  }
  return motion;
}

void PhaseMotionEstimator::restart(const char * reason)
{
  const std::size_t resets = diagnostics_.resets + 1;
  const std::size_t switches = diagnostics_.blade_switches;
  reset();
  diagnostics_.resets = resets;
  diagnostics_.blade_switches = switches;
  diagnostics_.status = reason;
}

void PhaseMotionEstimator::reset()
{
  history_.clear();
  direction_votes_.clear();
  kind_.reset();
  harmonic_model_.reset();
  last_fit_timestamp_.reset();
  reference_zero_world_ = Eigen::Vector3d::UnitZ();
  last_normal_world_ = Eigen::Vector3d::UnitX();
  last_wrapped_phase_rad_ = 0.0;
  kalman_phase_rad_ = 0.0;
  kalman_variance_ = 1.0;
  kalman_direction_ = 0;
  diagnostics_ = {};
}
}  // namespace auto_buff
