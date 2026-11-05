#include "yolo11_buff_openvino.hpp"
#include "yolo11_buff_model_hash.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>

namespace auto_buff
{
namespace
{
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point start)
{
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

}  // namespace

class Yolo11BuffOpenVINO::Impl
{
public:
  Impl(const std::string & path, const std::string & device)
  {
    auto model = core_.read_model(path);
    if (model->inputs().size() != 1 || model->outputs().size() != 1) {
      throw std::runtime_error("[YOLO11_BUFF] Expected one input and one output");
    }
    const auto input = model->input().get_shape();
    if (input.size() != 4 || input[0] != 1 || input[1] != 3 || input[2] == 0 || input[3] == 0) {
      throw std::runtime_error("[YOLO11_BUFF] Expected static [1,3,H,W] input");
    }
    info_.backend = "openvino";
    info_.model_path = path;
    info_.model_sha256 = yolo11_buff_model_sha256(path);
    info_.device = device;
    // This is the model's I/O type, not a claim about the plugin's internal execution precision.
    info_.precision = "FP32 I/O (plugin-selected compute)";
    info_.input_shape.assign(input.begin(), input.end());
    auto ppp = ov::preprocess::PrePostProcessor(model);
    ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC")
      .set_color_format(ov::preprocess::ColorFormat::BGR);
    ppp.input().preprocess().convert_element_type(ov::element::f32)
      .convert_color(ov::preprocess::ColorFormat::RGB).scale(255.0F);
    ppp.input().model().set_layout("NCHW");
    // Decoder always reads FP32, including IRs whose internal constants are compressed FP16.
    ppp.output().tensor().set_element_type(ov::element::f32);
    model = ppp.build();
    compiled_ = core_.compile_model(
      model, device, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
    request_ = compiled_.create_infer_request();
    const auto output = compiled_.output().get_shape();
    info_.output_shape.assign(output.begin(), output.end());
  }

  const BuffInferenceInfo & info() const { return info_; }

  const float * infer(
    const cv::Mat & bgr, const BuffLetterbox & t, BuffInferenceTimings & timings)
  {
    auto start = Clock::now();
    cv::Mat resized;
    if (t.resized_width != bgr.cols || t.resized_height != bgr.rows) {
      cv::resize(bgr, resized, {t.resized_width, t.resized_height});
    } else {
      resized = bgr;
    }
    cv::Mat padded;
    const int right = t.width - t.resized_width - t.pad_x;
    const int bottom = t.height - t.resized_height - t.pad_y;
    if (t.pad_x || t.pad_y || right || bottom) {
      cv::copyMakeBorder(resized, padded, t.pad_y, bottom, t.pad_x, right,
                        cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    } else {
      padded = resized;
    }
    if (!padded.isContinuous()) padded = padded.clone();
    timings.preprocess_ms = milliseconds(start);  // CPU resize/padding; PPP is inside infer.
    request_.set_input_tensor(ov::Tensor(
      ov::element::u8, {1, size_t(t.height), size_t(t.width), 3}, padded.data));
    start = Clock::now();
    request_.infer();
    timings.inference_ms = milliseconds(start);
    return request_.get_output_tensor().data<float>();
  }

private:
  BuffInferenceInfo info_;
  ov::Core core_;
  ov::CompiledModel compiled_;
  ov::InferRequest request_;
};
Yolo11BuffOpenVINO::Yolo11BuffOpenVINO(const std::string & path, const std::string & device)
: impl_(std::make_unique<Impl>(path, device)) {}

Yolo11BuffOpenVINO::~Yolo11BuffOpenVINO() = default;

const BuffInferenceInfo & Yolo11BuffOpenVINO::info() const { return impl_->info(); }

const float * Yolo11BuffOpenVINO::infer(
  const cv::Mat & bgr, const BuffLetterbox & transform, BuffInferenceTimings & timings)
{
  return impl_->infer(bgr, transform, timings);
}
}  // namespace auto_buff
