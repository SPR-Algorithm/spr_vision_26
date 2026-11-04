#ifndef AUTO_BUFF__PHASE_MOTION_ESTIMATOR_HPP
#define AUTO_BUFF__PHASE_MOTION_ESTIMATOR_HPP

#include <cstddef>
#include <deque>
#include <limits>
#include <optional>

#include "rune_types.hpp"

namespace auto_buff
{
// Engineering gates in seconds/radians, not a claim of measured vehicle accuracy.
struct PhaseMotionEstimatorConfig
{
  double max_observation_gap_s = 0.20;
  double small_window_s = 1.5;
  double small_min_span_s = 0.70;
  double small_rate_tolerance_rad_s = 0.30;
  double big_window_s = 16.0;
  double big_min_span_s = 4.5;
  double big_refit_interval_s = 0.15;
  double big_min_frequency_rad_s = 0.5;
  double big_max_frequency_rad_s = 3.5;
  double max_angular_speed_rad_s = 3.0;
  double phase_noise_std_rad = 0.035;
  double max_fit_rms_rad = 0.07;
  std::size_t max_history_samples = 2048;
};

struct PhaseMotionDiagnostics
{
  const char * status = "EMPTY";
  std::size_t samples = 0;
  std::size_t blade_switches = 0;
  std::size_t resets = 0;
  double span_s = 0.0;
  double fit_rms_rad = std::numeric_limits<double>::infinity();
  int direction_votes = 0;
  // Subtract this from an observed/predicted phase to compare different selected blades.
  // It restarts at zero with the phase history; use resets to delimit validation segments.
  double cumulative_switch_offset_rad = 0.0;
  std::optional<double> observed_phase_rad;
  Eigen::Vector3d phase_zero_direction_world = Eigen::Vector3d::UnitZ();
  Eigen::Vector3d plane_normal_world = Eigen::Vector3d::UnitX();
};

// S5 owns a world-referenced phase basis, phase continuity and motion estimation. Only a
// converged model produces a target. Frame-local S4 slot numbers do not identify switches.
class PhaseMotionEstimator
{
public:
  explicit PhaseMotionEstimator(PhaseMotionEstimatorConfig config = {});

  std::optional<RuneTarget> update(
    const RunePose & pose, const ReconstructedBlade & target_blade, FrameTimestamp timestamp);

  void reset();
  const PhaseMotionDiagnostics & diagnostics() const { return diagnostics_; }

private:
  struct PhaseSample
  {
    FrameTimestamp timestamp;
    double phase_rad;
    double weight;
  };

  PhaseMotionEstimatorConfig config_;
  PhaseMotionDiagnostics diagnostics_;
  std::deque<PhaseSample> history_;
  std::deque<int> direction_votes_;
  std::optional<RuneKind> kind_;
  Eigen::Vector3d reference_zero_world_ = Eigen::Vector3d::UnitZ();
  Eigen::Vector3d last_normal_world_ = Eigen::Vector3d::UnitX();
  double last_wrapped_phase_rad_ = 0.0;
  double kalman_phase_rad_ = 0.0;
  double kalman_variance_ = 1.0;
  int kalman_direction_ = 0;
  std::optional<MotionState> harmonic_model_;
  std::optional<FrameTimestamp> last_fit_timestamp_;

  void restart(const char * reason);
  int voted_direction() const;
  std::optional<MotionState> estimate_small(FrameTimestamp timestamp, double dt_s);
  std::optional<MotionState> estimate_big(FrameTimestamp timestamp);
};
}  // namespace auto_buff

#endif  // AUTO_BUFF__PHASE_MOTION_ESTIMATOR_HPP
