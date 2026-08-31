#include "buff_detector.hpp"

#include "tools/logger.hpp"

namespace auto_buff {
// RuneDetectionModel kpt 索引 -> OBJECT_POINTS 的 上/右/下/左 顺序
// 若实测标注顺序即 上右下左, 填 {0,1,2,3}; 否则按实测填写, 例如 {0,3,2,1}
static const int KPT_MAP[4] = {0, 3, 2, 1};

// 将模型输出的 5 关键点重排到 [上, 右, 下, 左, R标中心]
static std::vector<cv::Point2f>
order_keypoints(const std::vector<cv::Point2f> &kpt) {
  return {kpt[KPT_MAP[0]], kpt[KPT_MAP[1]], kpt[KPT_MAP[2]], kpt[KPT_MAP[3]],
          kpt[2]};
}

static BuffActivation activation_from_class(int cls)
{
  switch (cls) {
    case 0:
      return BuffActivation::INACTIVE;
    case 1:
      return BuffActivation::SMALL_ACTIVATED;
    case 2:
      return BuffActivation::BIG_ACTIVATED;
    default:
      return BuffActivation::INVALID;
  }
}

Buff_Detector::Buff_Detector(const std::string &config)
    : status_(LOSE), lose_(0), MODE_(config) {}

void Buff_Detector::handle_img(const cv::Mat &bgr_img, cv::Mat &dilated_img) {
  // 彩色图转灰度图
  cv::Mat gray_img;
  cv::cvtColor(bgr_img, gray_img, cv::COLOR_BGR2GRAY); // 彩色图转灰度图
  // cv::imshow("gray", gray_img);  // 调试用

  // 进行二值化           :把高于100变成255，低于100变成0
  cv::Mat binary_img;
  cv::threshold(gray_img, binary_img, 100, 255, cv::THRESH_BINARY);
  // cv::imshow("binary", binary_img);  // 调试用

  // 膨胀
  cv::Mat kernel =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)); // 使用矩形核
  cv::dilate(binary_img, dilated_img, kernel, cv::Point(-1, -1), 1);
  // cv::imshow("Dilated Image", dilated_img);  // 调试用
}

cv::Point2f Buff_Detector::get_r_center(std::vector<FanBlade> &fanblades,
                                        cv::Mat &bgr_img) {
  /// error

  if (fanblades.empty()) {
    tools::logger()->debug("[Buff_Detector] 无法计算r_center!");
    return {0, 0};
  }

  /// 算出大概位置

  cv::Point2f r_center_t = {0, 0};
  for (auto &fanblade : fanblades) {
    // RuneDetectionModel: points[4] 即 R 标旋转中心, 直接作为 r_center 初值
    r_center_t += fanblade.points[4];
  }
  r_center_t /= float(fanblades.size());

  /// 处理图片,mask选出大概范围

  cv::Mat dilated_img;
  handle_img(bgr_img, dilated_img);
  // radius 覆盖 R 标中心(r_center_t)到装甲板最远角点, 留 20% 裕量, 避免 mask
  // 截断轮廓
  double radius = 0.0;
  for (int i = 0; i < 4; ++i)
    radius = std::max(radius, cv::norm(fanblades[0].points[i] - r_center_t));
  radius *= 1.2;
  cv::Mat mask = cv::Mat::zeros(dilated_img.size(), CV_8U); // mask
  circle(mask, r_center_t, radius, cv::Scalar(255), -1);
  bitwise_and(dilated_img, mask, dilated_img); // 将遮罩应用于二值化图像
  tools::draw_point(bgr_img, r_center_t, {255, 255, 0}, 5); // 调试用
  // cv::imshow("Dilated Image", dilated_img);                // 调试用

  /// 获取轮廓点,矩阵框筛选  TODO

  std::vector<std::vector<cv::Point>> contours;
  auto r_center = r_center_t;
  cv::findContours(dilated_img, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_NONE); // external找外部区域
  double ratio_1 = INF;
  for (auto &it : contours) {
    auto rotated_rect = cv::minAreaRect(it);
    double ratio = rotated_rect.size.height > rotated_rect.size.width
                       ? rotated_rect.size.height / rotated_rect.size.width
                       : rotated_rect.size.width / rotated_rect.size.height;
    ratio += cv::norm(rotated_rect.center - r_center_t) / (radius / 3);
    if (ratio < ratio_1) {
      ratio_1 = ratio;
      r_center = rotated_rect.center;
    }
  }
  return r_center;
};

void Buff_Detector::handle_lose() {
  lose_++;
  if (lose_ >= LOSE_MAX) {
    status_ = LOSE;
    last_powerrune_ = std::nullopt;
  }
  status_ = TEM_LOSE;
}

std::optional<PowerRune> Buff_Detector::detect_24(cv::Mat &bgr_img) {
  /// onnx 模型检测

  std::vector<YOLO11_BUFF::Object> results =
      MODE_.get_multicandidateboxes(bgr_img);

  /// 处理未获得的情况

  if (results.empty()) {
    handle_lose();
    return std::nullopt;
  }

  /// results转扇叶FanBlade

  std::vector<FanBlade> fanblades;
  for (auto &result : results) {
    // 重排关键点顺序(上右下左) + 扇叶中心用 4 角点均值
    std::vector<cv::Point2f> ordered = order_keypoints(result.kpt);
    cv::Point2f center =
        (ordered[0] + ordered[1] + ordered[2] + ordered[3]) * 0.25f;
    fanblades.emplace_back(FanBlade(ordered, center, _light, result.label, result.prob));
  }

  /// 生成PowerRune
  auto r_center = get_r_center(fanblades, bgr_img);
  PowerRune powerrune(fanblades, r_center, last_powerrune_);

  /// handle error
  if (powerrune.is_unsolve()) {
    handle_lose();
    return std::nullopt;
  }

  status_ = TRACK;
  lose_ = 0;
  std::optional<PowerRune> P;
  P.emplace(powerrune);
  last_powerrune_ = P;
  return P;
}

std::optional<PowerRune> Buff_Detector::detect(cv::Mat &bgr_img) {
  // 方案A: 主流程使用多扇叶检测, 完整追踪扇叶切换(与 detect_24 同义)
  return detect_24(bgr_img);
}

std::optional<BuffObservation> Buff_Detector::detect_observation(cv::Mat & bgr_img)
{
  const auto power_rune = detect(bgr_img);
  if (!power_rune.has_value() || power_rune->fanblades.empty()) return std::nullopt;
  return to_observation(power_rune->fanblades.front());
}

std::optional<BuffObservation> Buff_Detector::to_observation(const FanBlade & blade)
{
  if (blade.points.size() != 5) return std::nullopt;
  BuffObservation observation;
  std::copy_n(blade.points.begin(), observation.points.size(), observation.points.begin());
  observation.activation = activation_from_class(blade.cls);
  observation.confidence = blade.confidence;
  return observation;
}

std::optional<PowerRune> Buff_Detector::detect_debug(cv::Mat &bgr_img,
                                                     cv::Point2f v) {
  /// onnx 模型检测

  std::vector<YOLO11_BUFF::Object> results =
      MODE_.get_multicandidateboxes(bgr_img);

  /// 处理未获得的情况

  if (results.empty())
    return std::nullopt;

  /// results转扇叶FanBlade

  std::vector<FanBlade> fanblades_t;
  for (auto &result : results) {
    std::vector<cv::Point2f> ordered = order_keypoints(result.kpt);
    cv::Point2f center =
        (ordered[0] + ordered[1] + ordered[2] + ordered[3]) * 0.25f;
    fanblades_t.emplace_back(FanBlade(ordered, center, _light, result.label, result.prob));
  }

  /// 计算r_center,筛选fanblade
  auto r_center = get_r_center(fanblades_t, bgr_img);
  std::vector<FanBlade> fanblades;
  for (auto &fanblade : fanblades_t) {
    if (cv::norm((fanblade.center - r_center) - v) < 10 ||
        results.size() == 1) {
      fanblades.emplace_back(fanblade);
      break;
    }
  }
  if (fanblades.empty())
    return std::nullopt;
  PowerRune powerrune(fanblades, r_center, std::nullopt);

  std::optional<PowerRune> P;
  P.emplace(powerrune);
  return P;
}

} // namespace auto_buff
