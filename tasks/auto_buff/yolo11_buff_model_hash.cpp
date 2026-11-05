#include "yolo11_buff_model_hash.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>

#include <openssl/evp.h>

namespace auto_buff
{
namespace
{
void append_file(EVP_MD_CTX * context, const std::filesystem::path & path)
{
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("[YOLO11_BUFF] Cannot open model for SHA256: " + path.string());
  }
  std::array<char, 64 * 1024> buffer;
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const auto count = input.gcount();
    if (count > 0 && EVP_DigestUpdate(context, buffer.data(), static_cast<std::size_t>(count)) != 1) {
      throw std::runtime_error("[YOLO11_BUFF] SHA256 update failed: " + path.string());
    }
  }
  if (input.bad() || !input.eof()) {
    throw std::runtime_error("[YOLO11_BUFF] Cannot read model for SHA256: " + path.string());
  }
}
}  // namespace

std::string yolo11_buff_model_sha256(const std::string & model_path)
{
  const std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
    EVP_MD_CTX_new(), &EVP_MD_CTX_free);
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) {
    throw std::runtime_error("[YOLO11_BUFF] Cannot initialize SHA256");
  }
  const std::filesystem::path path(model_path);
  append_file(context.get(), path);
  auto extension = path.extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(),
                 [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
  if (extension == ".xml") {
    auto weights = path;
    weights.replace_extension(".bin");
    append_file(context.get(), weights);
  }
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest;
  unsigned int size = 0;
  if (EVP_DigestFinal_ex(context.get(), digest.data(), &size) != 1 || size != 32) {
    throw std::runtime_error("[YOLO11_BUFF] Cannot finalize SHA256");
  }
  constexpr char hexadecimal[] = "0123456789abcdef";
  std::string result(size * 2, '0');
  for (unsigned int index = 0; index < size; ++index) {
    result[index * 2] = hexadecimal[digest[index] >> 4];
    result[index * 2 + 1] = hexadecimal[digest[index] & 0x0f];
  }
  return result;
}
}  // namespace auto_buff
