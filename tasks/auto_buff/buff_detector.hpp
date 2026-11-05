#ifndef AUTO_BUFF__BUFF_DETECTOR_HPP
#define AUTO_BUFF__BUFF_DETECTOR_HPP

#include <optional>
#include <string>
#include <vector>

#include "rune_types.hpp"
#include "yolo11_buff.hpp"
namespace auto_buff {
// 图像 -> 标准五点观测(BuffObservation), 跟踪与状态估计由后续流水线处理。
class Buff_Detector {
public:
  Buff_Detector(const std::string &config);

  // 返回本帧检测到的"全部符片"观测(不做 front() 截断)。
  // 直接从 YOLO 多候选结果构造(NMS 后按 quality 降序)。
  // 每个 BuffObservation.points 为标准顺序 [上,右,下,左,R]。
  // 无检测时返回空 vector。
  std::vector<BuffObservation> detect_observations(cv::Mat &bgr_img);

  // YOLO 模型结果(模型序 [top,left,R,right,bottom]) -> 标准五点观测
  static std::optional<BuffObservation>
  to_observation(const YOLO11_BUFF::Object &detection);

  const BuffInferenceDiagnostics & inference_diagnostics() const
  {
    return MODE_.inference_diagnostics();
  }

private:
  YOLO11_BUFF MODE_;
};
} // namespace auto_buff
#endif // AUTO_BUFF__BUFF_DETECTOR_HPP
