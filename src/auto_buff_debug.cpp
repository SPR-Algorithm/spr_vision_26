#include <fmt/format.h>

#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/buff_processor.hpp"
#include "tools/exiter.hpp"
#include "tools/plotter.hpp"

namespace
{
const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明 }"
  "{config-path c  | configs/sentry.yaml    | yaml配置文件的路径}"
  "{start-index s  | 0                      | 视频起始帧下标    }"
  "{end-index e    | 0                      | 视频结束帧下标    }"
  "{@input-path    |                        | avi和txt文件的路径}";

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
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  const auto input_path = cli.get<std::string>(0);
  const auto config_path = cli.get<std::string>("config-path");
  const auto start_index = cli.get<int>("start-index");
  const auto end_index = cli.get<int>("end-index");

  tools::Plotter plotter;
  tools::Exiter exiter;
  io::Gimbal gimbal(config_path);
  auto_buff::Buff_Detector detector(config_path);
  auto_buff::BuffProcessor processor(config_path, "configs/auto_buff.yaml");

  cv::VideoCapture video(fmt::format("{}.avi", input_path));
  std::ifstream text(fmt::format("{}.txt", input_path));
  cv::Mat img;
  const auto t0 = std::chrono::steady_clock::now();

  video.set(cv::CAP_PROP_POS_FRAMES, start_index);
  for (int i = 0; i < start_index; ++i) {
    double t, w, x, y, z;
    text >> t >> w >> x >> y >> z;
  }

  for (int frame_count = start_index; !exiter.exit(); ++frame_count) {
    if (end_index > 0 && frame_count > end_index) break;
    video.read(img);
    if (img.empty()) break;

    double t, w, x, y, z;
    text >> t >> w >> x >> y >> z;
    const auto timestamp = t0 + std::chrono::microseconds(static_cast<int>(t * 1e6));
    const Eigen::Quaterniond q(w, x, y, z);
    const auto detection = detector.detect(img);
    const auto output = processor.process(
      make_input(img, timestamp, q, gimbal.state(), gimbal.mode(), detection));
    gimbal.send(output);

    nlohmann::json data;
    data["mode"] = output.mode;
    data["yaw"] = output.yaw * 57.3F;
    data["pitch"] = output.pitch * 57.3F;
    plotter.plot(data);
    cv::imshow("result", img);

    const int key = cv::waitKey(1);
    if (key == 'q') break;
  }
  return 0;
}
