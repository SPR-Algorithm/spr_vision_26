#ifndef AUTO_BUFF__YOLO11_BUFF_TYPES_HPP
#define AUTO_BUFF__YOLO11_BUFF_TYPES_HPP

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace auto_buff
{
struct BuffInferenceInfo
{
  std::string backend;
  std::string model_path;
  std::string device;
  std::string precision;
  std::string model_sha256;
  std::vector<std::int64_t> input_shape;
  std::vector<std::int64_t> output_shape;
};

struct BuffInferenceTimings
{
  // CUDA event durations for TensorRT; unavailable stages remain absent for OpenVINO.
  std::optional<double> upload_ms;
  std::optional<double> preprocess_ms;
  std::optional<double> inference_ms;
  std::optional<double> download_ms;
  double postprocess_ms = 0.0;
  double draw_ms = 0.0;
  double total_ms = 0.0;  // Host wall time, including normalization/sync/postprocess, excluding draw.
};

struct BuffInferenceDiagnostics
{
  BuffInferenceInfo info;
  BuffInferenceTimings timings;
};
}  // namespace auto_buff

#endif
