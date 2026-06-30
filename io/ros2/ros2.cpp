#include "ros2.hpp"

namespace io
{
ROS2::ROS2()
{
  context_ = std::make_shared<rclcpp::Context>();
  rclcpp::InitOptions init_options;
  context_->init(0, nullptr, init_options);

  rclcpp::NodeOptions node_options;
  node_options.context(context_);
  node_ = std::make_shared<rclcpp::Node>("spr_vision_nav_interface", node_options);

  publish2nav_ = std::make_shared<Publish2Nav>(node_);
  subscribe2nav_ = std::make_shared<Subscribe2Nav>(node_);

  rclcpp::ExecutorOptions executor_options;
  executor_options.context = context_;
  executor_ = std::make_unique<rclcpp::executors::SingleThreadedExecutor>(executor_options);
  executor_->add_node(node_);

  spin_thread_ = std::make_unique<std::thread>([this]() { executor_->spin(); });
}

ROS2::~ROS2()
{
  if (executor_) {
    executor_->cancel();
  }

  if (spin_thread_ && spin_thread_->joinable()) {
    spin_thread_->join();
  }

  if (executor_ && node_) {
    executor_->remove_node(node_);
  }

  subscribe2nav_.reset();
  publish2nav_.reset();
  node_.reset();

  if (context_ && context_->is_valid()) {
    context_->shutdown("io::ROS2 destroyed");
  }
}

void ROS2::publish(const Eigen::Vector4d & target_pos) { publish2nav_->send_data(target_pos); }

std::vector<int8_t> ROS2::subscribe_enemy_status()
{
  return subscribe2nav_->subscribe_enemy_status();
}

std::vector<int8_t> ROS2::subscribe_autoaim_target()
{
  return subscribe2nav_->subscribe_autoaim_target();
}

}  // namespace io
