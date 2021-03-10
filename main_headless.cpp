#include <cstdlib>
#include <cstdio>
#include <string>
#include <png.h>

#include "volrend/n3tree.hpp"

#include "volrend/cuda/common.cuh"
#include "volrend/cuda/renderer_kernel.hpp"

void write_png_file(const char *filename, uint8_t *ptr, int width, int height) {
    int y;

    FILE *fp = fopen(filename, "wb");
    if (!fp) abort();

    png_structp png =
        png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) abort();

    png_infop info = png_create_info_struct(png);
    if (!info) abort();

    // if (setjmp(png_jmpbuf(png))) abort();

    png_init_io(png, fp);

    // Output is 8bit depth, RGBA format.
    png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_RGBA,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    // To remove the alpha channel for PNG_COLOR_TYPE_RGB format,
    // Use png_set_filler().
    // png_set_filler(png, 0, PNG_FILLER_AFTER);

    if (!ptr) abort();

    std::vector<uint8_t *> row_ptrs(height);
    for (int i = 0; i < height; ++i) {
        row_ptrs[i] = ptr + i * width * 4;
    }

    png_write_image(png, row_ptrs.data());
    png_write_end(png, NULL);

    fclose(fp);

    png_destroy_write_struct(&png, &info);
}

int main(int argc, char *argv[]) {
    using namespace volrend;
    if (argc <= 1) {
        fprintf(stderr, "expect argument: npz file\n");
        return 1;
    }
    const int device_id = (argc > 2) ? atoi(argv[2]) : -1;

    N3Tree tree(argv[1]);
    int width = 800, height = 800;
    if (tree.use_ndc) {
        width = 1008;
        height = 756;
    }
    Camera camera;
    camera.width = width;
    camera.height = height;
    camera.focal = 1111;

    camera.center = glm::vec3(0.0, 2.737260103225708, 2.959291696548462);
    camera.v_back = glm::vec3(0.0, 0.6790305972099304, 0.7341098785400391);
    // c2w = torch.tensor([
    //             [ -0.9999999403953552, 0.0, 0.0, 0.0 ],
    //             [ 0.0, -0.7341099977493286,
    //             0.6790305972099304, 2.737260103225708 ], [ 0.0,
    //             0.6790306568145752, 0.7341098785400391, 2.959291696548462 ],
    //             [ 0.0, 0.0, 0.0, 1.0 ],
    //         ], device=device)

    camera._update();
    RenderOptions options;

    cudaArray_t array;
    cudaStream_t stream;

    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);

    std::vector<uint8_t> buf(4 * width * height);

    cuda(MallocArray(&array, &channelDesc, width, height));
    cuda(StreamCreateWithFlags(&stream, cudaStreamDefault));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    launch_renderer(tree, camera, options, array, stream);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << milliseconds << "ms = " << 1000.f / milliseconds << " fps \n";

    cuda(Memcpy2DFromArrayAsync(buf.data(), 4 * width, array, 0, 0, 4 * width,
                                height, cudaMemcpyDeviceToHost, stream));

    write_png_file("a.png", buf.data(), width, height);
    cuda(FreeArray(array));
}
