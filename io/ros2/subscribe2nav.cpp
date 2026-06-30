#include "subscribe2nav.hpp"

#include <chrono>
#include <functional>
#include <mutex>
#include <utility>
#include <vector>

namespace io
{

Subscribe2Nav::Subscribe2Nav(rclcpp::Node::SharedPtr node) : node_(std::move(node))
{
  enemy_status_subscription_ = node_->create_subscription<spr_msgs::msg::EnemyStatusMsg>(
    "enemy_status", 10,
    std::bind(&Subscribe2Nav::enemy_status_callback, this, std::placeholders::_1));

  autoaim_target_subscription_ = node_->create_subscription<spr_msgs::msg::AutoaimTargetMsg>(
    "autoaim_target", 10,
    std::bind(&Subscribe2Nav::autoaim_target_callback, this, std::placeholders::_1));

  RCLCPP_INFO(node_->get_logger(), "nav subscriptions initialized.");
}

Subscribe2Nav::~Subscribe2Nav()
{
  RCLCPP_INFO(node_->get_logger(), "nav subscriptions shutting down.");
}

void Subscribe2Nav::enemy_status_callback(const spr_msgs::msg::EnemyStatusMsg::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(cache_mutex_);
  invincible_enemy_ids_ = msg->invincible_enemy_ids;
  enemy_status_updated_at_ = std::chrono::steady_clock::now();
}

void Subscribe2Nav::autoaim_target_callback(const spr_msgs::msg::AutoaimTargetMsg::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(cache_mutex_);
  autoaim_target_ids_ = msg->target_ids;
  autoaim_target_updated_at_ = std::chrono::steady_clock::now();
}

bool Subscribe2Nav::is_cache_fresh(
  const std::optional<std::chrono::steady_clock::time_point> & updated_at) const
{
  return updated_at.has_value() && std::chrono::steady_clock::now() - *updated_at <= kCacheTimeout;
}

std::vector<int8_t> Subscribe2Nav::subscribe_enemy_status()
{
  std::lock_guard<std::mutex> lock(cache_mutex_);

  if (!is_cache_fresh(enemy_status_updated_at_)) {
    return {};
  }

  return invincible_enemy_ids_;
}

std::vector<int8_t> Subscribe2Nav::subscribe_autoaim_target()
{
  std::lock_guard<std::mutex> lock(cache_mutex_);

  if (!is_cache_fresh(autoaim_target_updated_at_)) {
    return {};
  }

  return autoaim_target_ids_;
}

}  // namespace io