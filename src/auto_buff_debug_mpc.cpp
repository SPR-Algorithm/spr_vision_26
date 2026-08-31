#include <fmt/format.h>

#include <nlohmann/json.hpp>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/buff_processor.hpp"
#include "tools/exiter.hpp"
#include "tools/plotter.hpp"

namespace
{
const std::string keys =
  "{help h usage ? | | 输出命令行参数说明}"
  "{@config-path   | | yaml配置文件路径 }";

auto_buff::BuffActivation activation_from_detector(int cls)
{
  switch (cls) {
    case 0:
      return auto_buff::BuffActivation::INACTIVE;
    case 1:
      return auto_buff::BuffActivation::SMALL_ACTIVATED;
    case 2:
      return auto_buff::BuffActivation::BIG_ACTIVATED;
    default:
      return auto_buff::BuffActivation::INVALID;
  }
}

auto_buff::BuffInput make_input(
  const cv::Mat & img, std::chrono::steady_clock::time_point timestamp, const Eigen::Quaterniond & q,
  io::GimbalState state, io::GimbalMode mode, const std::optional<auto_buff::PowerRune> & detection)
{
  auto_buff::BuffInput input;
  input.img = img;
  input.timestamp = timestamp;
  input.imu_q = q;
  input.gimbal_state = state;
  input.gimbal_mode = mode;
  if (detection.has_value() && !detection->fanblades.empty() &&
      detection->fanblades.front().points.size() == input.points.size()) {
    const auto & blade = detection->fanblades.front();
    std::copy_n(blade.points.begin(), input.points.size(), input.points.begin());
    input.activation = activation_from_detector(blade.cls);
    input.confidence = 1.0F;
  }
  return input;
}
}  // namespace

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  const auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  tools::Plotter plotter;
  tools::Exiter exiter;
  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);
  auto_buff::Buff_Detector detector(config_path);
  auto_buff::BuffProcessor processor(config_path, "configs/auto_buff.yaml");

  while (!exiter.exit()) {
    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp;
    camera.read(img, timestamp);
    const auto q = gimbal.q(timestamp);
    const auto detection = detector.detect(img);
    const auto output = processor.process(
      make_input(img, timestamp, q, gimbal.state(), gimbal.mode(), detection));
    gimbal.send(output);

    nlohmann::json data;
    data["mode"] = output.mode;
    data["yaw"] = output.yaw * 57.3F;
    data["pitch"] = output.pitch * 57.3F;
    plotter.plot(data);
    cv::resize(img, img, {}, 0.5, 0.5);
    cv::imshow("result", img);
    if (cv::waitKey(1) == 'q') break;
  }
  return 0;
}
