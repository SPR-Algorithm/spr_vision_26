#ifndef AUTO_BUFF__YOLO11_BUFF_TENSORRT_HPP
#define AUTO_BUFF__YOLO11_BUFF_TENSORRT_HPP

#include <memory>
#include <string>

#include "yolo11_buff_backend.hpp"

namespace auto_buff
{
class Yolo11BuffTensorRT final : public Yolo11BuffBackend
{
public:
  Yolo11BuffTensorRT(const std::string & engine_path, int cuda_device);
  ~Yolo11BuffTensorRT() override;
  Yolo11BuffTensorRT(const Yolo11BuffTensorRT &) = delete;
  Yolo11BuffTensorRT & operator=(const Yolo11BuffTensorRT &) = delete;

  const BuffInferenceInfo & info() const override;
  const float * infer(
    const cv::Mat & bgr, const BuffLetterbox & transform, BuffInferenceTimings & timings) override;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace auto_buff

#endif
