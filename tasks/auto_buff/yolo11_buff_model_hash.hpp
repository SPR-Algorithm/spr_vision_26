#ifndef AUTO_BUFF__YOLO11_BUFF_MODEL_HASH_HPP
#define AUTO_BUFF__YOLO11_BUFF_MODEL_HASH_HPP

#include <string>

namespace auto_buff
{
// Private model identity: ONNX/engine hashes its bytes; an XML IR hashes XML bytes
// followed by its sibling .bin weights. Missing/unreadable files are errors.
std::string yolo11_buff_model_sha256(const std::string & model_path);
}  // namespace auto_buff

#endif
