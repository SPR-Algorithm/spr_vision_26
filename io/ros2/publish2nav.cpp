#include "publish2nav.hpp"

#include <Eigen/Dense>
#include <memory>
#include <utility>

#include "tools/logger.hpp"

namespace io
{

Publish2Nav::Publish2Nav(rclcpp::Node::SharedPtr node) : node_(std::move(node))
{
  publisher_ = node_->create_publisher<std_msgs::msg::String>("auto_aim_target_pos", 10);

  RCLCPP_INFO(node_->get_logger(), "auto_aim_target_pos publisher initialized.");
}

Publish2Nav::~Publish2Nav()
{
  RCLCPP_INFO(node_->get_logger(), "auto_aim_target_pos publisher shutting down.");
}

void Publish2Nav::send_data(const Eigen::Vector4d & target_pos)
{
  // 创建消息
  auto message = std::make_shared<std_msgs::msg::String>();

  // 将 Eigen::Vector3d 数据转换为字符串并存储在消息中
  message->data = std::to_string(target_pos[0]) + "," + std::to_string(target_pos[1]) + "," +
                  std::to_string(target_pos[2]) + "," + std::to_string(target_pos[3]);

  // 发布消息
  publisher_->publish(*message);

  // RCLCPP_INFO(
  //   node_->get_logger(), "auto_aim_target_pos publisher sent message: '%s'",
  //   message->data.c_str());
}

}  // namespace io
