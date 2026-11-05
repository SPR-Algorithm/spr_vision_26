#include "yolo11_buff_cuda_preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace auto_buff
{
namespace
{
__device__ float resize_channel(
  const unsigned char * source, std::size_t pitch, int source_width, int source_height,
  double scale_x, double scale_y, int x, int y, int channel)
{
  // Match OpenCV 4.x INTER_LINEAR's u8 path: double coordinates cast to float,
  // 11-bit rounded coefficients, integer horizontal sums, then its staged vertical
  // shifts. A single final floating-point rounding differs by 1 LSB and can perturb
  // the planar PnP branch even when the detector's point displacement is subpixel.
  // Explicit operations prevent CUDA FMA contraction from changing a coefficient.
  const float sx = __double2float_rn(__dadd_rn(__dmul_rn(x + 0.5, scale_x), -0.5));
  const float sy = __double2float_rn(__dadd_rn(__dmul_rn(y + 0.5, scale_y), -0.5));
  int x0 = static_cast<int>(floorf(sx));
  const int original_y0 = static_cast<int>(floorf(sy));
  float dx = __fsub_rn(sx, static_cast<float>(x0));
  const float dy = __fsub_rn(sy, static_cast<float>(original_y0));
  if (x0 < 0) { x0 = 0; dx = 0.0F; }
  if (x0 >= source_width - 1) { x0 = source_width - 1; dx = 0.0F; }
  const int x1 = min(x0 + 1, source_width - 1);
  // OpenCV clamps the two source row indices independently while preserving beta.
  const int y0 = min(max(original_y0, 0), source_height - 1);
  const int y1 = min(max(original_y0 + 1, 0), source_height - 1);
  constexpr float coefficient_scale = 2048.0F;
  const int a0 = __float2int_rn(__fmul_rn(__fsub_rn(1.0F, dx), coefficient_scale));
  const int a1 = __float2int_rn(__fmul_rn(dx, coefficient_scale));
  const int b0 = __float2int_rn(__fmul_rn(__fsub_rn(1.0F, dy), coefficient_scale));
  const int b1 = __float2int_rn(__fmul_rn(dy, coefficient_scale));
  const auto * row0 = source + static_cast<std::size_t>(y0) * pitch;
  const auto * row1 = source + static_cast<std::size_t>(y1) * pitch;
  const int top = row0[3 * x0 + channel] * a0 + row0[3 * x1 + channel] * a1;
  const int bottom = row1[3 * x0 + channel] * a0 + row1[3 * x1 + channel] * a1;
  const int sample = (((b0 * (top >> 4)) >> 16) + ((b1 * (bottom >> 4)) >> 16) + 2) >> 2;
  return static_cast<float>(sample) / 255.0F;
}

__global__ void preprocess_kernel(
  const unsigned char * source, std::size_t pitch, int source_width, int source_height,
  float * output, int width, int height, int resized_width, int resized_height,
  int pad_x, int pad_y, double scale_x, double scale_y)
{
  const int plane = width * height;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= plane) return;
  const int x = index % width - pad_x;
  const int y = index / width - pad_y;
  if (x < 0 || y < 0 || x >= resized_width || y >= resized_height) {
    constexpr float padding = 114.0F / 255.0F;
    output[index] = padding;
    output[index + plane] = padding;
    output[index + 2 * plane] = padding;
    return;
  }
  // Source is BGR interleaved; TensorRT input is RGB planar.
  output[index] = resize_channel(
    source, pitch, source_width, source_height, scale_x, scale_y, x, y, 2);
  output[index + plane] = resize_channel(
    source, pitch, source_width, source_height, scale_x, scale_y, x, y, 1);
  output[index + 2 * plane] = resize_channel(
    source, pitch, source_width, source_height, scale_x, scale_y, x, y, 0);
}
}  // namespace

void buff_cuda_check(cudaError_t result, const char * operation)
{
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string("auto_buff CUDA ") + operation + ": " +
                             cudaGetErrorString(result));
  }
}

BuffCudaPreprocessor::~BuffCudaPreprocessor() noexcept
{
  if (image_) {
    const auto result = cudaFree(image_);
    if (result != cudaSuccess) {
      std::fprintf(stderr, "[auto_buff CUDA cleanup] free raw image: %s\n", cudaGetErrorString(result));
    }
  }
}

void BuffCudaPreprocessor::upload(const cv::Mat & bgr, cudaStream_t stream)
{
  if (bgr.empty() || bgr.type() != CV_8UC3) {
    throw std::invalid_argument("auto_buff CUDA upload requires a nonempty CV_8UC3 image");
  }
  const auto row_bytes = static_cast<std::size_t>(bgr.cols) * 3;
  if (bgr.step[0] < row_bytes) throw std::invalid_argument("auto_buff CUDA invalid image stride");
  if (row_bytes > capacity_row_bytes_ || bgr.rows > capacity_height_) {
    unsigned char * replacement = nullptr;
    std::size_t replacement_pitch = 0;
    const auto capacity_row_bytes = std::max(row_bytes, capacity_row_bytes_);
    const int capacity_height = std::max(bgr.rows, capacity_height_);
    buff_cuda_check(cudaMallocPitch(
      reinterpret_cast<void **>(&replacement), &replacement_pitch,
      capacity_row_bytes, capacity_height), "allocate pitched image");
    if (image_) {
      const auto result = cudaFree(image_);
      if (result != cudaSuccess) {
        const auto cleanup = cudaFree(replacement);
        if (cleanup != cudaSuccess) {
          std::fprintf(stderr, "[auto_buff CUDA cleanup] free replacement: %s\n", cudaGetErrorString(cleanup));
        }
        buff_cuda_check(result, "release replaced image");
      }
    }
    image_ = replacement;
    pitch_ = replacement_pitch;
    capacity_row_bytes_ = capacity_row_bytes;
    capacity_height_ = capacity_height;
  }
  buff_cuda_check(cudaMemcpy2DAsync(
    image_, pitch_, bgr.data, bgr.step[0], row_bytes, bgr.rows,
    cudaMemcpyHostToDevice, stream), "upload strided image");
  width_ = bgr.cols;
  height_ = bgr.rows;
}

void BuffCudaPreprocessor::launch(
  float * output, const BuffLetterbox & t, cudaStream_t stream) const
{
  const auto pixels = static_cast<std::int64_t>(t.width) * t.height;
  if (
    !image_ || !output || width_ <= 0 || height_ <= 0 || t.width <= 0 || t.height <= 0 ||
    pixels > std::numeric_limits<int>::max() / 3 || t.resized_width <= 0 ||
    t.resized_height <= 0 || t.resized_width > t.width || t.resized_height > t.height ||
    t.pad_x != (t.width - t.resized_width) / 2 ||
    t.pad_y != (t.height - t.resized_height) / 2) {
    throw std::invalid_argument("auto_buff CUDA invalid letterbox dimensions or output");
  }
  // cv::resize derives inverse scales from the requested integer destination size.
  const double scale_x = 1.0 / (static_cast<double>(t.resized_width) / width_);
  const double scale_y = 1.0 / (static_cast<double>(t.resized_height) / height_);
  constexpr int threads = 256;
  const int blocks = static_cast<int>((pixels + threads - 1) / threads);
  preprocess_kernel<<<blocks, threads, 0, stream>>>(
    image_, pitch_, width_, height_, output, t.width, t.height,
    t.resized_width, t.resized_height, t.pad_x, t.pad_y, scale_x, scale_y);
  buff_cuda_check(cudaGetLastError(), "launch letterbox kernel");
}
}  // namespace auto_buff
