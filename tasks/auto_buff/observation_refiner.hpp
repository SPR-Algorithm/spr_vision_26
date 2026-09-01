#ifndef AUTO_BUFF__OBSERVATION_REFINER_HPP
#define AUTO_BUFF__OBSERVATION_REFINER_HPP

#include <string>

#include <opencv2/core.hpp>

#include "rune_types.hpp"

namespace auto_buff
{
struct ObservationRefinerConfig
{
  float roi_margin_ratio = 0.20F;
  int roi_margin_px = 12;
  float min_point_distance_px = 4.0F;
  float min_armor_area_px2 = 100.0F;
  float min_armor_edge_px = 4.0F;
  float r_boundary_tolerance_px = 2.0F;
  int gaussian_kernel_size = 5;
  int binary_threshold = 40;
  int morphology_kernel_size = 3;
  float min_contour_area_px2 = 12.0F;
  float armor_area_relative_error = 0.80F;
  int light_arm_line_samples = 80;
  float good_confidence = 0.70F;
  float good_geometry_score = 0.50F;
  float good_contour_support = 0.45F;
};

class ObservationRefiner
{
public:
  explicit ObservationRefiner(ObservationRefinerConfig config = {});
  explicit ObservationRefiner(const std::string & config_path);

  RefinedObservation refine(
    const cv::Mat & image, const BuffObservation & observation, const RuneState & state) const;

  const ObservationRefinerConfig & config() const { return config_; }

private:
  ObservationRefinerConfig config_;

  static ObservationRefinerConfig load_config(const std::string & config_path);
  static void validate_config(const ObservationRefinerConfig & config);
};
}  // namespace auto_buff

#endif  // AUTO_BUFF__OBSERVATION_REFINER_HPP
