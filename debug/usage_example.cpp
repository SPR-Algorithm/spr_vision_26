// 调试事件总线接入示例；不是独立可执行文件。
// 完整集成可参考 src/auto_aim_debug_mpc.cpp。
#include "debug/debug_bus.hpp"
#include "debug/param_tuner.hpp"
#include "debug/web_debugger.hpp"  // ReprojectionData 的完整定义
#include "tasks/auto_aim/armor.hpp"

#include <chrono>
#include <list>
#include <string>
#include <vector>

// 初始化一次；在 YAML 的 debug_bus.sinks.web 下启用 Web sink。
void init_debug(const std::string & config_path)
{
  debug::ParamTuner::instance().load_from_yaml(config_path);
  debug::DebugBus::instance().load_config(YAML::LoadFile(config_path));
}

// 每帧在 camera.read 返回后记录 proc_start，处理结束后调用此函数。
// capture_time 使用 camera.read 返回的采集时间，必须与 steady_clock 同一时钟。
void post_debug(
  const cv::Mat & frame, const std::list<auto_aim::Armor> & armors,
  const std::vector<debug::ReprojectionData> & reprojections,
  std::chrono::steady_clock::time_point proc_start,
  std::chrono::steady_clock::time_point capture_time)
{
  const double proc_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - proc_start).count();
  debug::DebugBus::instance().post({frame, &armors, reprojections, proc_ms, capture_time});
}

// 程序退出时停止并等待服务线程。
void shutdown_debug() { debug::DebugBus::instance().shutdown(); }
