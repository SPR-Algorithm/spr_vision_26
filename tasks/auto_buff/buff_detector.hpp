#ifndef AUTO_BUFF__TRACK_HPP
#define AUTO_BUFF__TRACK_HPP

#include <optional>
#include <vector>

#include "buff_type.hpp"
#include "yolo11_buff.hpp"
namespace auto_buff {
// 职责(对齐 RP-26 NNDetector): 图像 -> 标准五点观测(BuffObservation)。
// 不再产出 PowerRune / 不再做 get_r_center 聚合、target 跟踪与状态机,
// 这些属于旧 buff_solver/buff_aimer/buff_target 的遗留职责, 由上层处理。
class Buff_Detector {
public:
  Buff_Detector(const std::string &config);

  // 返回本帧检测到的"全部符片"观测(不做 front() 截断)。
  // 直接从 YOLO 多候选结果构造(NMS 后按 quality 降序)。
  // 每个 BuffObservation.points 为标准顺序 [上,右,下,左,R]。
  // 无检测时返回空 vector。
  std::vector<BuffObservation> detect_observations(cv::Mat &bgr_img);

  // 单符片便捷接口: 取 detect_observations() 中置信度最高的一片,
  // 供现有单目标链路使用(不破坏既有调用方)。
  std::optional<BuffObservation> detect_observation(cv::Mat &bgr_img);

  // YOLO 模型结果(模型序 [top,left,R,right,bottom]) -> 标准五点观测
  static std::optional<BuffObservation>
  to_observation(const YOLO11_BUFF::Object &detection);

  // 遗留 FanBlade(五点) -> 标准五点观测
  static std::optional<BuffObservation> to_observation(const FanBlade &blade);

private:
  YOLO11_BUFF MODE_;
};
} // namespace auto_buff
#endif // DETECTOR_HPP
