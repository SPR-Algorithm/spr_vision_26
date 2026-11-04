#ifndef AUTO_BUFF__RUNE_RECONSTRUCTOR_HPP
#define AUTO_BUFF__RUNE_RECONSTRUCTOR_HPP

#include <string>

#include "rune_pose_tools.hpp"
#include "rune_types.hpp"

namespace auto_buff
{
class RuneReconstructor
{
public:
  explicit RuneReconstructor(const std::string & camera_config);

  // Joint PnP requires 2..5 same-kind observations and returns only distinct accepted slots.
  // Slot ONE and local +Z refer to the highest-confidence inactive accepted blade, or the
  // highest-confidence accepted blade if all are active; this reference is frame-local.
  // Usable armour boundaries optionally refine the PnP prior with per-blade distance fields;
  // missing/rejected contour evidence preserves PNP. Success preserves world-space armour
  // centres; source alone does not authorize firing. An absent result keeps firing disabled.
  // The optional previous pose initializes planar candidates and rejects >20 degree plane
  // jumps. The caller owns timestamps/resets: clear it on invalidation, mode change or a
  // stale gap; a short missing-observation interval may retain the last accepted pose.
  std::optional<RuneReconstruction> reconstruct(
    const std::vector<RefinedObservation> & blades, const Eigen::Quaterniond & imu_q,
    const std::optional<RunePose> & previous_pose = std::nullopt) const;

private:
  rune_pose_tools::CameraCalibration calibration_;
};
}  // namespace auto_buff

#endif  // AUTO_BUFF__RUNE_RECONSTRUCTOR_HPP
