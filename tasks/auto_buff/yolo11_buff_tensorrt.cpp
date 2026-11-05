#include "yolo11_buff_tensorrt.hpp"

#include <NvInfer.h>
#include <NvInferVersion.h>
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "yolo11_buff_cuda_preprocess.hpp"

namespace auto_buff
{
namespace
{
class TensorRTLogger final : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char * message) noexcept override
  {
    if (severity <= Severity::kWARNING) {
      std::fprintf(stderr, "[auto_buff TensorRT] %s\n", message);
    }
  }
};

class DeviceGuard
{
public:
  explicit DeviceGuard(int device)
  {
    buff_cuda_check(cudaGetDevice(&previous_), "query current device");
    if (previous_ != device) buff_cuda_check(cudaSetDevice(device), "select device");
  }
  ~DeviceGuard() noexcept
  {
    if (previous_ >= 0) {
      const auto result = cudaSetDevice(previous_);
      if (result != cudaSuccess) {
        std::fprintf(stderr, "[auto_buff CUDA cleanup] restore device: %s\n", cudaGetErrorString(result));
      }
    }
  }
  DeviceGuard(const DeviceGuard &) = delete;
  DeviceGuard & operator=(const DeviceGuard &) = delete;

private:
  int previous_ = -1;
};

void cleanup_cuda(cudaError_t result, const char * operation) noexcept
{
  if (result != cudaSuccess) {
    std::fprintf(stderr, "[auto_buff CUDA cleanup] %s: %s\n", operation, cudaGetErrorString(result));
  }
}

struct CudaState
{
  explicit CudaState(int device) : device(device) {}
  ~CudaState() noexcept
  {
    int previous = -1;
    cleanup_cuda(cudaGetDevice(&previous), "query device");
    cleanup_cuda(cudaSetDevice(device), "select owning device");
    if (stream) cleanup_cuda(cudaStreamSynchronize(stream), "complete pending inference");
    preprocessor.reset();
    if (input) cleanup_cuda(cudaFree(input), "free input");
    if (output) cleanup_cuda(cudaFree(output), "free output");
    if (host_output) cleanup_cuda(cudaFreeHost(host_output), "free pinned output");
    for (auto event : events) if (event) cleanup_cuda(cudaEventDestroy(event), "destroy event");
    if (stream) cleanup_cuda(cudaStreamDestroy(stream), "destroy stream");
    if (previous >= 0 && previous != device) cleanup_cuda(cudaSetDevice(previous), "restore device");
  }

  void initialize(std::size_t input_elements, std::size_t output_elements)
  {
    // initialize() is called only after this object has an owner. Partial failures therefore
    // run its destructor and release every already-created CUDA object.
    buff_cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    for (auto & event : events) buff_cuda_check(cudaEventCreate(&event), "create timing event");
    buff_cuda_check(cudaMalloc(reinterpret_cast<void **>(&input), input_elements * sizeof(float)),
                    "allocate input tensor");
    buff_cuda_check(cudaMalloc(reinterpret_cast<void **>(&output), output_elements * sizeof(float)),
                    "allocate output tensor");
    buff_cuda_check(cudaMallocHost(reinterpret_cast<void **>(&host_output), output_elements * sizeof(float)),
                    "allocate pinned output");
    preprocessor = std::make_unique<BuffCudaPreprocessor>();
  }

  int device;
  cudaStream_t stream = nullptr;
  std::array<cudaEvent_t, 5> events{};
  float * input = nullptr;
  float * output = nullptr;
  float * host_output = nullptr;
  std::unique_ptr<BuffCudaPreprocessor> preprocessor;
};

std::vector<char> read_engine(const std::string & path)
{
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error("auto_buff cannot open TensorRT engine: " + path);
  const auto end = input.tellg();
  if (end <= 0 || static_cast<std::uintmax_t>(end) > std::numeric_limits<std::size_t>::max()) {
    throw std::runtime_error("auto_buff TensorRT engine has invalid size: " + path);
  }
  std::vector<char> data(static_cast<std::size_t>(end));
  input.seekg(0);
  if (!input.read(data.data(), static_cast<std::streamsize>(data.size()))) {
    throw std::runtime_error("auto_buff cannot read TensorRT engine: " + path);
  }
  return data;
}

std::vector<std::int64_t> shape_of(const nvinfer1::Dims & dims)
{
  if (dims.nbDims <= 0) throw std::runtime_error("auto_buff TensorRT tensor rank is unresolved");
  std::vector<std::int64_t> shape;
  for (int index = 0; index < dims.nbDims; ++index) {
    if (dims.d[index] <= 0) throw std::runtime_error("auto_buff TensorRT requires static positive shapes");
    shape.push_back(dims.d[index]);
  }
  return shape;
}

std::size_t elements(const std::vector<std::int64_t> & shape)
{
  std::size_t count = 1;
  for (const auto dimension : shape) {
    if (dimension <= 0 || static_cast<std::uint64_t>(dimension) >
        std::numeric_limits<std::size_t>::max() / sizeof(float) / count) {
      throw std::runtime_error("auto_buff TensorRT tensor size overflows");
    }
    count *= static_cast<std::size_t>(dimension);
  }
  return count;
}

void verify_contiguous(
  const nvinfer1::IExecutionContext & context, const char * name,
  const std::vector<std::int64_t> & shape)
{
  const auto strides = context.getTensorStrides(name);
  if (strides.nbDims != static_cast<int>(shape.size())) {
    throw std::runtime_error("auto_buff TensorRT tensor has unresolved strides");
  }
  std::int64_t expected = 1;
  for (int index = strides.nbDims - 1; index >= 0; --index) {
    if (strides.d[index] != expected) {
      throw std::runtime_error("auto_buff TensorRT requires contiguous linear tensors");
    }
    if (shape[index] > std::numeric_limits<std::int64_t>::max() / expected) {
      throw std::runtime_error("auto_buff TensorRT stride size overflows");
    }
    expected *= shape[index];
  }
}

void read_metadata(
  const std::string & engine_path, const cudaDeviceProp & device, BuffInferenceInfo & info)
{
  const auto path = std::filesystem::path(engine_path + ".json");
  info.precision = "unknown (FP32 I/O)";
  if (!std::filesystem::exists(path)) return;
  try {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open metadata");
    const auto metadata = nlohmann::json::parse(input);
    if (!metadata.is_object() || !metadata.at("schema_version").is_number_integer() ||
        metadata.at("schema_version").get<int>() != 1) {
      throw std::runtime_error("unsupported metadata schema");
    }
    const auto precision = metadata.at("precision").get<std::string>();
    if (precision != "fp32" && precision != "fp16") {
      throw std::runtime_error("metadata precision must be fp32 or fp16");
    }
    const auto hash = metadata.at("model_sha256").get<std::string>();
    if (hash.size() != 64 || !std::all_of(hash.begin(), hash.end(), [](unsigned char c) {
          return std::isxdigit(c) != 0;
        })) {
      throw std::runtime_error("metadata ONNX model_sha256 is invalid");
    }
    const auto version = metadata.at("trt_version").get<std::string>();
    const auto current = std::to_string(NV_TENSORRT_MAJOR) + "." +
                         std::to_string(NV_TENSORRT_MINOR) + "." +
                         std::to_string(NV_TENSORRT_PATCH);
    if (version != current && version.rfind(current + ".", 0) != 0) {
      throw std::runtime_error("metadata TensorRT version differs from the runtime build");
    }
    const auto & gpu = metadata.at("gpu");
    const auto capability = std::to_string(device.major) + "." + std::to_string(device.minor);
    if (gpu.at("name").get<std::string>() != device.name ||
        gpu.at("compute_capability").get<std::string>() != capability) {
      throw std::runtime_error("metadata GPU name or compute capability differs from selected device");
    }
    for (const auto * key : {"input_shape", "output_shape"}) {
      if (metadata.contains(key) &&
          metadata.at(key).get<std::vector<std::int64_t>>() !=
            (std::string(key) == "input_shape" ? info.input_shape : info.output_shape)) {
        throw std::runtime_error(std::string("metadata ") + key + " differs from engine");
      }
    }
    // This is the builder's declared precision and ONNX identity, not proof of each layer's
    // arithmetic type. Python validation verifies the manifest's engine/ONNX file hashes.
    info.precision = precision;
    info.model_sha256 = hash;
  } catch (const std::exception & error) {
    throw std::runtime_error("auto_buff invalid TensorRT metadata " + path.string() + ": " + error.what());
  }
}

double event_ms(cudaEvent_t begin, cudaEvent_t end)
{
  float elapsed = 0.0F;
  buff_cuda_check(cudaEventElapsedTime(&elapsed, begin, end), "read timing event");
  if (!std::isfinite(elapsed) || elapsed < 0.0F) {
    throw std::runtime_error("auto_buff CUDA returned invalid timing");
  }
  return elapsed;
}
}  // namespace

class Yolo11BuffTensorRT::Impl
{
public:
  explicit Impl(int cuda_device) : cuda_device_(cuda_device) {}

  void initialize(const std::string & engine_path)
  {
    int device_count = 0;
    buff_cuda_check(cudaGetDeviceCount(&device_count), "enumerate devices");
    if (cuda_device_ < 0 || cuda_device_ >= device_count) {
      throw std::invalid_argument("auto_buff cuda_device is outside the available device range");
    }
    DeviceGuard guard(cuda_device_);
    cudaDeviceProp device{};
    buff_cuda_check(cudaGetDeviceProperties(&device, cuda_device_), "query device properties");
    const auto data = read_engine(engine_path);
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) throw std::runtime_error("auto_buff cannot create TensorRT runtime");
    engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size()));
    if (!engine_) throw std::runtime_error("auto_buff cannot deserialize TensorRT engine: " + engine_path);
    if (engine_->getNbIOTensors() != 2) {
      throw std::runtime_error("auto_buff TensorRT engine must have exactly one input and one output");
    }
    for (int index = 0; index < engine_->getNbIOTensors(); ++index) {
      const char * name = engine_->getIOTensorName(index);
      if (!name || engine_->getTensorDataType(name) != nvinfer1::DataType::kFLOAT ||
          engine_->getTensorLocation(name) != nvinfer1::TensorLocation::kDEVICE ||
          engine_->getTensorFormat(name) != nvinfer1::TensorFormat::kLINEAR) {
        throw std::runtime_error("auto_buff TensorRT requires device FP32 linear I/O tensors");
      }
      if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
        if (!input_name_.empty()) throw std::runtime_error("auto_buff TensorRT has multiple inputs");
        input_name_ = name;
      } else if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
        if (!output_name_.empty()) throw std::runtime_error("auto_buff TensorRT has multiple outputs");
        output_name_ = name;
      } else {
        throw std::runtime_error("auto_buff TensorRT has an invalid I/O tensor mode");
      }
    }
    if (input_name_.empty() || output_name_.empty()) {
      throw std::runtime_error("auto_buff TensorRT input or output tensor is missing");
    }
    info_.backend = "tensorrt";
    info_.model_path = engine_path;
    info_.device = "CUDA:" + std::to_string(cuda_device_) + " (" + device.name + ")";
    info_.input_shape = shape_of(engine_->getTensorShape(input_name_.c_str()));
    info_.output_shape = shape_of(engine_->getTensorShape(output_name_.c_str()));
    const auto & in = info_.input_shape;
    const auto & out = info_.output_shape;
    if (in.size() != 4 || in[0] != 1 || in[1] != 3 ||
        in[2] > std::numeric_limits<int>::max() || in[3] > std::numeric_limits<int>::max() ||
        in[2] * in[3] > std::numeric_limits<int>::max() / 3) {
      throw std::runtime_error("auto_buff TensorRT input must be static [1,3,H,W]");
    }
    if (out.size() != 3 || out[0] != 1 || !((out[1] == 18) != (out[2] == 18))) {
      throw std::runtime_error("auto_buff TensorRT output must be [1,18,A] or [1,A,18]");
    }
    output_elements_ = elements(out);
    const auto input_elements = elements(in);
    context_.reset(engine_->createExecutionContext());
    if (!context_) throw std::runtime_error("auto_buff cannot create TensorRT execution context");
    verify_contiguous(*context_, input_name_.c_str(), in);
    verify_contiguous(*context_, output_name_.c_str(), out);
    read_metadata(engine_path, device, info_);
    cuda_ = std::make_unique<CudaState>(cuda_device_);
    cuda_->initialize(input_elements, output_elements_);
    if (!context_->setTensorAddress(input_name_.c_str(), cuda_->input) ||
        !context_->setTensorAddress(output_name_.c_str(), cuda_->output)) {
      throw std::runtime_error("auto_buff cannot bind TensorRT input/output tensor addresses");
    }
  }

  ~Impl() noexcept
  {
    if (!cuda_ && !context_ && !engine_ && !runtime_) return;
    int previous = -1;
    cleanup_cuda(cudaGetDevice(&previous), "query device before TensorRT cleanup");
    cleanup_cuda(cudaSetDevice(cuda_device_), "select TensorRT device for cleanup");
    cuda_.reset();
    context_.reset();
    engine_.reset();
    runtime_.reset();
    if (previous >= 0 && previous != cuda_device_) cleanup_cuda(cudaSetDevice(previous), "restore device");
  }

  const float * infer(const cv::Mat & bgr, const BuffLetterbox & transform, BuffInferenceTimings & timings)
  {
    if (bgr.empty() || bgr.type() != CV_8UC3 ||
        transform.width != info_.input_shape[3] || transform.height != info_.input_shape[2]) {
      throw std::invalid_argument("auto_buff TensorRT received an invalid image or input dimensions");
    }
    const auto expected = make_buff_letterbox(
      bgr.cols, bgr.rows, transform.width, transform.height);
    if (transform.resized_width != expected.resized_width ||
        transform.resized_height != expected.resized_height || transform.pad_x != expected.pad_x ||
        transform.pad_y != expected.pad_y || !std::isfinite(transform.scale) ||
        transform.scale != expected.scale) {
      throw std::invalid_argument("auto_buff TensorRT letterbox metadata does not match the image");
    }
    DeviceGuard guard(cuda_device_);
    try {
      auto & state = *cuda_;
      buff_cuda_check(cudaEventRecord(state.events[0], state.stream), "record upload start");
      state.preprocessor->upload(bgr, state.stream);
      buff_cuda_check(cudaEventRecord(state.events[1], state.stream), "record upload end");
      state.preprocessor->launch(state.input, transform, state.stream);
      buff_cuda_check(cudaEventRecord(state.events[2], state.stream), "record preprocess end");
      if (!context_->enqueueV3(state.stream)) throw std::runtime_error("auto_buff TensorRT enqueueV3 failed");
      buff_cuda_check(cudaEventRecord(state.events[3], state.stream), "record inference end");
      buff_cuda_check(cudaMemcpyAsync(
        state.host_output, state.output, output_elements_ * sizeof(float),
        cudaMemcpyDeviceToHost, state.stream), "download inference output");
      buff_cuda_check(cudaEventRecord(state.events[4], state.stream), "record download end");
      buff_cuda_check(cudaStreamSynchronize(state.stream), "complete inference");
      timings.upload_ms = event_ms(state.events[0], state.events[1]);
      timings.preprocess_ms = event_ms(state.events[1], state.events[2]);
      timings.inference_ms = event_ms(state.events[2], state.events[3]);
      timings.download_ms = event_ms(state.events[3], state.events[4]);
      return state.host_output;
    } catch (...) {
      // Complete or fail pending transfers before the caller can release this cv::Mat.
      cleanup_cuda(cudaStreamSynchronize(cuda_->stream), "drain failed inference");
      throw;
    }
  }

  BuffInferenceInfo info_;

private:
  // Destruction is explicitly ordered; the logger always outlives all TensorRT objects.
  TensorRTLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  std::unique_ptr<CudaState> cuda_;
  int cuda_device_;
  std::string input_name_;
  std::string output_name_;
  std::size_t output_elements_ = 0;
};

Yolo11BuffTensorRT::Yolo11BuffTensorRT(const std::string & engine_path, int cuda_device)
: impl_(std::make_unique<Impl>(cuda_device))
{
  impl_->initialize(engine_path);
}

Yolo11BuffTensorRT::~Yolo11BuffTensorRT() = default;
const BuffInferenceInfo & Yolo11BuffTensorRT::info() const { return impl_->info_; }
const float * Yolo11BuffTensorRT::infer(
  const cv::Mat & bgr, const BuffLetterbox & transform, BuffInferenceTimings & timings)
{
  return impl_->infer(bgr, transform, timings);
}
}  // namespace auto_buff
