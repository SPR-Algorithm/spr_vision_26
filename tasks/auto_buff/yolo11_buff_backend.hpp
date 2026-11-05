#ifndef AUTO_BUFF__YOLO11_BUFF_BACKEND_HPP
#define AUTO_BUFF__YOLO11_BUFF_BACKEND_HPP

#include <memory>
#include <opencv2/core.hpp>

#include "yolo11_buff_types.hpp"

namespace auto_buff
{
struct BuffLetterbox
{
  int width = 0;
  int height = 0;
  int resized_width = 0;
  int resized_height = 0;
  int pad_x = 0;
  int pad_y = 0;
  float scale = 1.0F;
};

BuffLetterbox make_buff_letterbox(int source_width, int source_height, int width, int height);

// Each instance is synchronous and single-threaded. Returned storage survives until next infer.
class Yolo11BuffBackend
{
public:
  virtual ~Yolo11BuffBackend() = default;
  virtual const BuffInferenceInfo & info() const = 0;
  virtual const float * infer(
    const cv::Mat & bgr, const BuffLetterbox & transform, BuffInferenceTimings & timings) = 0;
};

}  // namespace auto_buff

#endif
