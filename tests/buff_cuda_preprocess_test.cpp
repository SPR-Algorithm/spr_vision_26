#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "tasks/auto_buff/yolo11_buff_cuda_preprocess.hpp"

namespace
{
using auto_buff::buff_cuda_check;

struct Case
{
  std::string name;
  cv::Mat image;
  cv::Size target;
};

class CudaFixture
{
public:
  explicit CudaFixture(int device)
  {
    buff_cuda_check(cudaSetDevice(device), "select preprocessing test device");
    buff_cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "create test stream");
  }
  ~CudaFixture() noexcept
  {
    if (stream_) cudaStreamSynchronize(stream_);
    preprocessor_.reset();
    if (output_) cudaFree(output_);
    if (host_output_) cudaFreeHost(host_output_);
    if (stream_) cudaStreamDestroy(stream_);
  }

  const float * preprocess(const cv::Mat & source, const auto_buff::BuffLetterbox & transform)
  {
    const auto count = static_cast<std::size_t>(transform.width) * transform.height * 3;
    if (count > capacity_) {
      // All previous work completed before buffers can be replaced.
      if (output_) { buff_cuda_check(cudaFree(output_), "free test tensor"); output_ = nullptr; }
      if (host_output_) { buff_cuda_check(cudaFreeHost(host_output_), "free test pinned tensor"); host_output_ = nullptr; }
      buff_cuda_check(cudaMalloc(reinterpret_cast<void **>(&output_), count * sizeof(float)),
                      "allocate test tensor");
      buff_cuda_check(cudaMallocHost(reinterpret_cast<void **>(&host_output_), count * sizeof(float)),
                      "allocate test pinned tensor");
      capacity_ = count;
    }
    if (!preprocessor_) preprocessor_ = std::make_unique<auto_buff::BuffCudaPreprocessor>();
    try {
      preprocessor_->upload(source, stream_);
      preprocessor_->launch(output_, transform, stream_);
      buff_cuda_check(cudaMemcpyAsync(
        host_output_, output_, count * sizeof(float), cudaMemcpyDeviceToHost, stream_), "copy test tensor");
      buff_cuda_check(cudaStreamSynchronize(stream_), "complete test preprocessing");
    } catch (...) {
      cudaStreamSynchronize(stream_);
      throw;
    }
    return host_output_;
  }

private:
  cudaStream_t stream_ = nullptr;
  float * output_ = nullptr;
  float * host_output_ = nullptr;
  std::size_t capacity_ = 0;
  std::unique_ptr<auto_buff::BuffCudaPreprocessor> preprocessor_;
};

void compare(CudaFixture & fixture, const Case & item)
{
  if (item.image.empty() || item.image.type() != CV_8UC3) {
    throw std::runtime_error("invalid real-frame test case: " + item.name);
  }
  const auto before = item.image.clone();
  const auto transform = auto_buff::make_buff_letterbox(
    item.image.cols, item.image.rows, item.target.width, item.target.height);
  cv::Mat resized, expected;
  if (item.image.size() != cv::Size(transform.resized_width, transform.resized_height)) {
    cv::resize(item.image, resized, {transform.resized_width, transform.resized_height}, 0, 0, cv::INTER_LINEAR);
  } else {
    resized = item.image;
  }
  cv::copyMakeBorder(
    resized, expected, transform.pad_y, transform.height - transform.resized_height - transform.pad_y,
    transform.pad_x, transform.width - transform.resized_width - transform.pad_x,
    cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
  const float * actual = fixture.preprocess(item.image, transform);
  const auto plane = static_cast<std::size_t>(transform.width) * transform.height;
  double max_error_lsb = 0.0;
  std::size_t different = 0;
  for (int y = 0; y < transform.height; ++y) {
    const auto * row = expected.ptr<cv::Vec3b>(y);
    for (int x = 0; x < transform.width; ++x) {
      const bool padding = x < transform.pad_x || y < transform.pad_y ||
        x >= transform.pad_x + transform.resized_width || y >= transform.pad_y + transform.resized_height;
      const auto offset = static_cast<std::size_t>(y) * transform.width + x;
      for (std::size_t channel = 0; channel < 3; ++channel) {
        const float value = actual[offset + channel * plane];
        if (!std::isfinite(value) || value < 0.0F || value > 1.0F) {
          throw std::runtime_error(item.name + ": nonfinite/out-of-range GPU sample");
        }
        const float reference = static_cast<float>(row[x][2 - channel]) / 255.0F;
        const double error = std::abs(static_cast<double>(value - reference)) * 255.0;
        max_error_lsb = std::max(max_error_lsb, error);
        different += error > 1e-4;
        if (padding && error > 1e-4) {
          throw std::runtime_error(item.name + ": exact RGB/channel/padding comparison failed");
        }
      }
    }
  }
  // A one-LSB resize difference was enough to change detected keypoints and reject six
  // planar PnP frames in a real capture. Require matching u8 samples, rather than treating
  // those differences as harmless. Allow only floating-point normalization roundoff.
  if (max_error_lsb > 1e-4) {
    throw std::runtime_error(item.name + ": CPU/CUDA u8 resize samples differ: " + std::to_string(max_error_lsb));
  }
  if (cv::norm(item.image, before, cv::NORM_INF) != 0) {
    throw std::runtime_error(item.name + ": preprocessing modified the source image");
  }
  std::cout << item.name << ": " << item.image.cols << 'x' << item.image.rows
            << " step=" << item.image.step[0] << " -> " << transform.width << 'x' << transform.height
            << ", padding=" << transform.pad_x << ',' << transform.pad_y
            << ", max_error_lsb=" << max_error_lsb << ", different_samples=" << different << '\n';
}

std::vector<Case> cases_from(const cv::Mat & frame)
{
  if (frame.cols < 16 || frame.rows < 16) throw std::runtime_error("real frame is too small for ROI tests");
  std::vector<Case> result;
  result.push_back({"full_frame", frame, {640, 480}});
  result.push_back({"native_size", frame, frame.size()});
  result.push_back({"half_size_area_fast_path", frame, {frame.cols / 2, frame.rows / 2}});
  const cv::Rect odd_crop(1, 1, frame.cols - (frame.cols % 2 == 0 ? 3 : 4),
                           frame.rows - (frame.rows % 2 == 0 ? 3 : 4));
  const cv::Mat roi = frame(odd_crop);
  if (roi.isContinuous()) throw std::runtime_error("ROI test must exercise non-contiguous input");
  result.push_back({"strided_odd_roi", roi, {640, 480}});
  result.push_back({"odd_output", roi, {321, 239}});
  result.push_back({"wide_crop_vertical_padding", frame(cv::Rect(0, 0, frame.cols, frame.rows / 2)), {640, 480}});
  result.push_back({"tall_crop_horizontal_padding", frame(cv::Rect(1, 0, frame.cols / 2, frame.rows)), {640, 480}});
  result.push_back({"small_roi_upscale", frame(cv::Rect(2, 3, std::min(127, frame.cols - 2),
                            std::min(93, frame.rows - 3))), {640, 480}});
  result.push_back({"single_real_pixel", frame(cv::Rect(2, 3, 1, 1)), {640, 480}});
  result.push_back({"single_real_row", frame(cv::Rect(0, 2, frame.cols, 1)), {640, 480}});
  result.push_back({"single_real_column", frame(cv::Rect(2, 0, 1, frame.rows)), {640, 480}});
  cv::Mat gray, gray_bgr, bgra, bgra_bgr;
  cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  cv::cvtColor(gray, gray_bgr, cv::COLOR_GRAY2BGR);
  cv::cvtColor(frame, bgra, cv::COLOR_BGR2BGRA);
  cv::cvtColor(bgra, bgra_bgr, cv::COLOR_BGRA2BGR);
  if (cv::norm(frame, bgra_bgr, cv::NORM_INF) != 0) {
    throw std::runtime_error("BGRA normalization did not preserve the real BGR frame");
  }
  result.push_back({"real_gray_normalized_to_bgr", gray_bgr, {640, 480}});
  result.push_back({"real_bgra_normalized_to_bgr", bgra_bgr, {640, 480}});
  return result;
}
}  // namespace

int main(int argc, char ** argv)
{
  try {
    std::string video_path;
    int device = 0;
    int requested_frames = 1;
    for (int index = 1; index < argc; ++index) {
      const std::string argument = argv[index];
      if (argument.rfind("--video=", 0) == 0) video_path = argument.substr(8);
      else if (argument == "--video" && index + 1 < argc) video_path = argv[++index];
      else if (argument.rfind("--cuda-device=", 0) == 0) device = std::stoi(argument.substr(14));
      else if (argument.rfind("--frames=", 0) == 0) requested_frames = std::stoi(argument.substr(9));
      else if (argument == "--help" || argument == "-h") {
        std::cout << "buff_cuda_preprocess_test --video=REAL_CAPTURE.avi [--cuda-device=0] [--frames=1]\n";
        return 0;
      } else if (!argument.empty() && argument[0] != '-' && video_path.empty()) video_path = argument;
      else throw std::invalid_argument("unknown preprocessing test argument: " + argument);
    }
    if (video_path.empty()) throw std::invalid_argument("provide --video= with a recorded real camera video");
    if (requested_frames <= 0) throw std::invalid_argument("--frames must be positive");
    cv::VideoCapture video(video_path);
    cv::Mat frame;
    if (!video.isOpened() || !video.read(frame) || frame.empty() || frame.type() != CV_8UC3) {
      throw std::runtime_error("cannot read a BGR real frame from " + video_path);
    }
    CudaFixture fixture(device);
    const auto cases = cases_from(frame);
    for (const auto & item : cases) compare(fixture, item);
    for (int index = 1; index < requested_frames; ++index) {
      if (!video.read(frame) || frame.empty()) throw std::runtime_error("video ended before --frames was reached");
      compare(fixture, {"source_frame_" + std::to_string(index), frame, {640, 480}});
    }
    std::cout << "PASS: " << cases.size() + requested_frames - 1
              << " real-frame preprocessing cases from " << requested_frames << " source frames; source="
              << video_path << '\n';
    return 0;
  } catch (const std::exception & error) {
    std::cerr << "buff_cuda_preprocess_test: " << error.what() << '\n';
    return 1;
  }
}
