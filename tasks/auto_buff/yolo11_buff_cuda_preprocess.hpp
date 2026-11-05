#ifndef AUTO_BUFF__YOLO11_BUFF_CUDA_PREPROCESS_HPP
#define AUTO_BUFF__YOLO11_BUFF_CUDA_PREPROCESS_HPP

#include <cstddef>
#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>

#include "yolo11_buff_backend.hpp"

namespace auto_buff
{
// Private TensorRT implementation helpers. These symbols are built only with CUDA enabled.
void buff_cuda_check(cudaError_t result, const char * operation);

class BuffCudaPreprocessor
{
public:
  BuffCudaPreprocessor() = default;
  ~BuffCudaPreprocessor() noexcept;
  BuffCudaPreprocessor(const BuffCudaPreprocessor &) = delete;
  BuffCudaPreprocessor & operator=(const BuffCudaPreprocessor &) = delete;

  // The caller keeps bgr alive and unchanged until this stream completes. Row stride is
  // preserved, including for non-contiguous cv::Mat ROIs. Calls on an instance are serial.
  void upload(const cv::Mat & bgr, cudaStream_t stream);

  // Writes RGB float NCHW into the caller-owned tensor. Padding is exactly 114/255;
  // image samples follow OpenCV's half-pixel and 11-bit INTER_LINEAR u8 rounding before
  // normalization. This preserves the pixels consumed by the OpenVINO reference path.
  void launch(float * output, const BuffLetterbox & transform, cudaStream_t stream) const;

private:
  unsigned char * image_ = nullptr;
  std::size_t pitch_ = 0;
  std::size_t capacity_row_bytes_ = 0;
  int capacity_height_ = 0;
  int width_ = 0;
  int height_ = 0;
};
}  // namespace auto_buff

#endif
