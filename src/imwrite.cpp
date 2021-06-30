#include "volrend/common.hpp"
#include "volrend/internal/imwrite.hpp"
#include <cstdint>
#include <vector>

#ifdef VOLREND_PNG
#include <png.h>
#include <zlib.h>
#endif

namespace volrend {
namespace internal {

bool write_png_file(const std::string &filename, uint8_t *ptr, int width,
                    int height) {
#ifdef VOLREND_PNG
    FILE *fp = fopen(filename.c_str(), "wb");
    if (!fp) {
        fprintf(stderr, "PNG destination could not be opened\n");
        return false;
    }

    png_structp png =
        png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fprintf(stderr, "PNG write failed\n");
        return false;
    }
    png_set_compression_level(png, 0);
    png_set_compression_strategy(png, Z_HUFFMAN_ONLY);
    png_set_filter_heuristics(png, PNG_FILTER_NONE, 0, 0, 0);

    png_infop info = png_create_info_struct(png);
    if (!info) {
        fprintf(stderr, "PNG write failed\n");
        return false;
    }

    if (setjmp(png_jmpbuf(png))) {
        fprintf(stderr, "PNG write failed\n");
        return false;
    }

    png_init_io(png, fp);

    // Output is 8bit depth, RGBA format.
    png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_RGBA,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    // To remove the alpha channel for PNG_COLOR_TYPE_RGB format,
    // Use png_set_filler().
    // png_set_filler(png, 0, PNG_FILLER_AFTER);

    if (!ptr) {
        fprintf(stderr, "PNG write failed\n");
        return false;
    }

    std::vector<uint8_t *> row_ptrs(height);
    for (int i = 0; i < height; ++i) {
        row_ptrs[i] = ptr + i * width * 4;
    }

    png_write_image(png, row_ptrs.data());
    png_write_end(png, NULL);

    fclose(fp);

    png_destroy_write_struct(&png, &info);
    return true;
#else
    fprintf(stderr,
            "WARNING: Not writing image because volrend was not built with "
            "libpng\n");
    return false;
#endif
}

}  // namespace internal
}  // namespace volrend
