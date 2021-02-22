#include "volrend/common.hpp"

// CUDA backend only enabled when VOLREND_USE_CUDA=ON
#ifdef VOLREND_CUDA
#include "volrend/renderer.hpp"

#include <ctime>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <array>

#include "volrend/cuda/common.cuh"
#include "volrend/cuda/renderer_kernel.hpp"

namespace volrend {

struct VolumeRenderer::Impl {
    Impl(Camera& camera, RenderOptions& options)
        : camera(camera), options(options), buf_index(0) {
        cuda(StreamCreateWithFlags(&stream, cudaStreamDefault));

        glCreateRenderbuffers(2, rb.data());
        glCreateFramebuffers(2, fb.data());

        // Attach rbo to fbo
        for (int index = 0; index < 2; index++) {
            glNamedFramebufferRenderbuffer(fb[index], GL_COLOR_ATTACHMENT0,
                                           GL_RENDERBUFFER, rb[index]);
        }
    }

    ~Impl() {
        // Unregister CUDA resources
        for (int index = 0; index < 2; index++) {
            if (cgr[index] != nullptr)
                cuda(GraphicsUnregisterResource(cgr[index]));
        }
        glDeleteRenderbuffers(2, rb.data());
        glDeleteFramebuffers(2, fb.data());
    }

    void render() {
        GLfloat clear_color[] = {1.f, 1.f, 1.f, 1.f};
        glClearNamedFramebufferfv(fb[buf_index], GL_COLOR, 0, clear_color);

        if (tree) {
            camera._update();
            cuda(GraphicsMapResources(1, &cgr[buf_index], stream));
            launch_renderer(*tree, camera, options, ca[buf_index], stream);
            cuda(GraphicsUnmapResources(1, &cgr[buf_index], stream));

            glBlitNamedFramebuffer(fb[buf_index], 0, 0, 0, camera.width,
                                   camera.height, 0, camera.height,
                                   camera.width, 0, GL_COLOR_BUFFER_BIT,
                                   GL_NEAREST);
        }
        buf_index ^= 1;
    }

    void resize(const int width, const int height) {
        if (camera.width == width && camera.height == height) return;
        // save new size
        camera.width = width;
        camera.height = height;

        // resize color buffer
        for (int index = 0; index < 2; index++) {
            // unregister resource
            if (cgr[index] != nullptr)
                cuda(GraphicsUnregisterResource(cgr[index]));

            // resize rbo
            glNamedRenderbufferStorage(rb[index], GL_RGBA8, width, height);

            // register rbo
            cuda(GraphicsGLRegisterImage(
                &cgr[index], rb[index], GL_RENDERBUFFER,
                cudaGraphicsRegisterFlagsSurfaceLoadStore |
                    cudaGraphicsRegisterFlagsWriteDiscard));
        }

        // map graphics resources
        cuda(GraphicsMapResources(2, cgr.data(), 0));

        // get CUDA Array refernces
        for (int index = 0; index < 2; index++) {
            cuda(GraphicsSubResourceGetMappedArray(&ca[index], cgr[index], 0,
                                                   0));
        }

        // unmap graphics resources
        cuda(GraphicsUnmapResources(2, cgr.data(), 0));
    }

    const N3Tree* tree = nullptr;

   private:
    Camera& camera;
    RenderOptions& options;
    int buf_index;

    // GL buffers
    std::array<GLuint, 2> fb, rb;

    // CUDA resources
    std::array<cudaGraphicsResource_t, 2> cgr = {0};
    std::array<cudaArray_t, 2> ca;
    cudaStream_t stream;
};

VolumeRenderer::VolumeRenderer(int device_id)
    : impl_(std::make_unique<Impl>(camera, options)) {
    GLint gl_device_id;
    GLuint gl_device_count;

    cuda(GLGetDevices(&gl_device_count, &gl_device_id, 1, cudaGLDeviceListAll));

    if (device_id == -1) device_id = gl_device_id;
    cuda(SetDevice(device_id));

    // Show info
    struct cudaDeviceProp props;
    cuda(GetDeviceProperties(&props, gl_device_id));
    printf("OpenGL : %-24s (%d)\n", props.name, props.multiProcessorCount);

    cuda(GetDeviceProperties(&props, device_id));
    printf("CUDA   : %-24s (%d)\n", props.name, props.multiProcessorCount);
}

VolumeRenderer::~VolumeRenderer() {}

void VolumeRenderer::render() { impl_->render(); }
void VolumeRenderer::set(const N3Tree& tree) { impl_->tree = &tree; }
void VolumeRenderer::clear() { impl_->tree = nullptr; }

void VolumeRenderer::resize(int width, int height) {
    impl_->resize(width, height);
}
const char* VolumeRenderer::get_backend() { return "CUDA"; }

}  // namespace volrend
#endif
