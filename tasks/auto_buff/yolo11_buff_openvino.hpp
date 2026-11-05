#ifndef AUTO_BUFF__YOLO11_BUFF_OPENVINO_HPP
#define AUTO_BUFF__YOLO11_BUFF_OPENVINO_HPP

#include "yolo11_buff_backend.hpp"

namespace auto_buff
{
// OpenVINO preprocessing and inference; model decoding is shared in YOLO11_BUFF.
class Yolo11BuffOpenVINO final : public Yolo11BuffBackend
{
public:
  Yolo11BuffOpenVINO(const std::string & model_path, const std::string & device);
  ~Yolo11BuffOpenVINO() override;
  const BuffInferenceInfo & info() const override;
  const float * infer(
    const cv::Mat & bgr, const BuffLetterbox & transform, BuffInferenceTimings & timings) override;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace auto_buff

#endif
