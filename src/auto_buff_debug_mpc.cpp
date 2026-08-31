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
    const auto observation = detector.detect_observation(img);
    const auto output = processor.process(auto_buff::make_buff_input(
      img, timestamp, q, gimbal.state(), gimbal.mode(), observation));
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
