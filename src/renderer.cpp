#include "volrend/renderer.hpp"

#include <ctime>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <array>

#include "volrend/cuda/common.cuh"
#include "volrend/internal/renderer_kernel.hpp"

namespace volrend {

struct CUDAVolumeRenderer::Impl {
    Impl(Camera& camera) : camera(camera), buf_index(0) {
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

    void render(const N3Tree& tree, float step_size, int max_n_steps) {
        camera._update();
        tree.precompute_step(step_size);
        cuda(GraphicsMapResources(1, &cgr[buf_index], stream));
        launch_renderer(tree, camera, ca[buf_index], step_size, max_n_steps,
                        stream);
        cuda(GraphicsUnmapResources(1, &cgr[buf_index], stream));
    }

    void swap() {
        glBlitNamedFramebuffer(fb[buf_index], 0, 0, 0, camera.width,
                               camera.height, 0, camera.height, camera.width, 0,
                               GL_COLOR_BUFFER_BIT, GL_NEAREST);
        buf_index ^= 1;
    }

    void clear(float r, float g, float b, float a) {
        GLfloat clear_color[] = {r, g, b, a};
        glClearNamedFramebufferfv(fb[buf_index], GL_COLOR, 0, clear_color);
    }

    void resize(const int width, const int height) {
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

   private:
    Camera& camera;
    int buf_index;

    // GL buffers
    std::array<GLuint, 2> fb, rb;

    // CUDA resources
    std::array<cudaGraphicsResource_t, 2> cgr = {0};
    std::array<cudaArray_t, 2> ca;
    cudaStream_t stream;
};

CUDAVolumeRenderer::~CUDAVolumeRenderer() {}
CUDAVolumeRenderer::CUDAVolumeRenderer()
    : impl_(std::make_unique<Impl>(camera)) {}

void CUDAVolumeRenderer::render(const N3Tree& tree) {
    impl_->render(tree, step_size, max_n_steps);
}
void CUDAVolumeRenderer::swap() { impl_->swap(); }
void CUDAVolumeRenderer::clear(float r, float g, float b, float a) {
    impl_->clear(r, g, b, a);
}

void CUDAVolumeRenderer::resize(int width, int height) {
    impl_->resize(width, height);
}

}  // namespace volrend
