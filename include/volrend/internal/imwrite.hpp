#pragma once

#include <string>

namespace volrend {
namespace internal {

// Write a u8, 4 channel PNG file
bool write_png_file(const std::string &filename, uint8_t *ptr, int width,
                    int height);

}  // namespace internal
}  // namespace volrend
