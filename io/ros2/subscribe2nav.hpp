#ifndef IO__SUBSCRIBE2NAV_HPP
#define IO__SUBSCRIBE2NAV_HPP

#include <chrono>
#include <mutex>
#include <optional>
#include <rclcpp/rclcpp.hpp>
#include <vector>

#include "spr_msgs/msg/autoaim_target_msg.hpp"
#include "spr_msgs/msg/enemy_status_msg.hpp"

namespace io
{
class Subscribe2Nav
{
public:
  explicit Subscribe2Nav(rclcpp::Node::SharedPtr node);

  ~Subscribe2Nav();

  std::vector<int8_t> subscribe_enemy_status();
  std::vector<int8_t> subscribe_autoaim_target();

private:
  void enemy_status_callback(const spr_msgs::msg::EnemyStatusMsg::SharedPtr msg);
  void autoaim_target_callback(const spr_msgs::msg::AutoaimTargetMsg::SharedPtr msg);

  bool is_cache_fresh(const std::optional<std::chrono::steady_clock::time_point> & updated_at) const;

  static constexpr std::chrono::milliseconds kCacheTimeout{1500};

  rclcpp::Node::SharedPtr node_;

  rclcpp::Subscription<spr_msgs::msg::EnemyStatusMsg>::SharedPtr enemy_status_subscription_;
  rclcpp::Subscription<spr_msgs::msg::AutoaimTargetMsg>::SharedPtr autoaim_target_subscription_;

  mutable std::mutex cache_mutex_;
  std::vector<int8_t> invincible_enemy_ids_;
  std::vector<int8_t> autoaim_target_ids_;
  std::optional<std::chrono::steady_clock::time_point> enemy_status_updated_at_;
  std::optional<std::chrono::steady_clock::time_point> autoaim_target_updated_at_;
};
}  // namespace io

#endif  // IO__SUBSCRIBE2NAV_HPP
