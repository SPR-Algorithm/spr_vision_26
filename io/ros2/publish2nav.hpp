#ifndef IO__PBLISH2NAV_HPP
#define IO__PBLISH2NAV_HPP

#include <Eigen/Dense>  // For Eigen::Vector3d
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

namespace io
{
class Publish2Nav
{
public:
  explicit Publish2Nav(rclcpp::Node::SharedPtr node);

  ~Publish2Nav();

  void send_data(const Eigen::Vector4d & data);

private:
  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
};

}  // namespace io

#endif  // Publish2Nav_HPP_
