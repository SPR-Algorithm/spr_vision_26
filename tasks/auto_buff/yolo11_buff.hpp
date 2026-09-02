#ifndef AUTO_BUFF__YOLO11_BUFF_HPP
#define AUTO_BUFF__YOLO11_BUFF_HPP
#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include "tools/logger.hpp"

namespace auto_buff {
// RuneDetectionModel 类别定义 (与 buffer-0624.xml / model-0624.onnx 一致)
const std::vector<std::string> class_names = {"inactive", "small_activated",
                                              "big_activated"};

class YOLO11_BUFF {
public:
  struct Object {
    cv::Point2f center; // 目标中心(有效关键点均值, 与 RP-26 一致)
    int label = -1;     // 类别: 0-未激活 1-小符已激活 2-大符已激活
    float prob = 0.f;   // 类别置信度
    float quality =
        0.f; // conf * mean(kpt_conf 有效点), NMS 排序键(与 RP-26 一致)
    std::vector<cv::Point2f> kpt; // 5个关键点(4角点 + R标中心)
    std::vector<float> kpt_conf;  // 5个关键点置信度
  };

  explicit YOLO11_BUFF(const std::string &config);

  // 使用NMS，用来获取多个目标 (返回图中所有符片, 按 quality 降序)
  std::vector<Object> get_multicandidateboxes(cv::Mat &image);

  // 寻找置信度最高的目标
  std::vector<Object> get_onecandidatebox(cv::Mat &image);

  // 只读访问生效的后处理阈值(便于测试/日志核对)
  float conf_threshold() const { return conf_threshold_; }
  float kpt_conf_threshold() const { return kpt_conf_threshold_; }
  float nms_distance() const { return nms_dist_threshold_; }
  int min_valid_keypoints() const { return min_valid_kpts_; }
  bool output_is_nca() const { return output_is_nca_; }

private:
  ov::Core core; // 创建OpenVINO Runtime Core对象
  std::shared_ptr<ov::Model> model;
  ov::CompiledModel compiled_model;
  ov::InferRequest infer_request;

  // 模型输入/输出规格 (从模型自动解析)
  int input_height_ = 480;
  int input_width_ = 640;
  int output_channels_ = 18; // 3 类别 + 5 关键点 * 3 (x,y,conf)
  int num_anchors_ = 6300;   // 80*60 + 40*30 + 20*15
  int num_classes_ = 3;
  int num_keypoints_ = 5;
  int kpt_dim_ = 3;           // x, y, confidence
  bool output_is_nca_ = true; // 输出布局 [N,C,A] (否则 [N,A,C]), 与 RP-26 一致

  // 后处理阈值 (默认与 RP-26 detect.json 一致: conf 0.65 / kconf 0.5 / nms 30 /
  // min_valid 5)
  float conf_threshold_ = 0.65f;
  float kpt_conf_threshold_ = 0.5f;
  float nms_dist_threshold_ = 30.0f;
  int min_valid_kpts_ = 5;

  // 统一输入为 8-bit BGR (支持 1/3/4 通道), 与 RP-26 normalize_input 一致
  static cv::Mat normalize_input(const cv::Mat &source);

  // letterbox: 等比例缩放 + 灰边填充 (4:3 输入零padding)
  void letterbox(const cv::Mat &src, cv::Mat &dst, float &scale, int &pad_w,
                 int &pad_h) const;

  // 后处理: 从输出张量解码检测结果 (自动支持 [N,C,A]/[N,A,C])
  std::vector<Object> postprocess(float scale, int pad_w, int pad_h, int orig_w,
                                  int orig_h);

  // 按中心点距离做 NMS (先按 quality 降序, 与 RP-26 center_distance_nms 一致)
  std::vector<Object> nms(std::vector<Object> &objects);

  // 可视化绘制
  void draw(cv::Mat &image, const std::vector<Object> &objects);

  // 将image保存为"../result/$${programName}.jpg"
  void save(const std::string &programName, const cv::Mat &image);
};
} // namespace auto_buff
#endif