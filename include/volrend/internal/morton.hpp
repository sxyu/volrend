#include <cstdint>
#include "volrend/common.hpp"

namespace volrend {
namespace internal {
namespace {
// 3D Morton code helpers
VOLREND_COMMON_FUNCTION uint32_t _expand_bits(uint32_t v) {
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v << 8)) & 0x0300F00F;
    v = (v | (v << 4)) & 0x030C30C3;
    v = (v | (v << 2)) & 0x09249249;
    return v;
}

VOLREND_COMMON_FUNCTION uint32_t _unexpand_bits(uint32_t v) {
    v &= 0x49249249;
    v = (v | (v >> 2)) & 0xc30c30c3;
    v = (v | (v >> 4)) & 0xf00f00f;
    v = (v | (v >> 8)) & 0xff0000ff;
    v = (v | (v >> 16)) & 0x0000ffff;
    return v;
}

// 3D Morton code (interleave)
VOLREND_COMMON_FUNCTION uint32_t morton_code_3(uint32_t x, uint32_t y,
                                               uint32_t z) {
    uint32_t xx = _expand_bits(x);
    uint32_t yy = _expand_bits(y);
    uint32_t zz = _expand_bits(z);
    return (xx << 2) + (yy << 1) + zz;
}

// Invert 3D Morton code (deinterleave)
VOLREND_COMMON_FUNCTION void inv_morton_code_3(uint32_t code, uint32_t* x,
                                               uint32_t* y, uint32_t* z) {
    *x = _unexpand_bits(code >> 2);
    *y = _unexpand_bits(code >> 1);
    *z = _unexpand_bits(code);
}
}  // namespace
}  // namespace internal
}  // namespace volrend
