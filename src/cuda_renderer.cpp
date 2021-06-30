#include "volrend/common.hpp"

// CUDA backend only enabled when VOLREND_USE_CUDA=ON
#ifdef VOLREND_CUDA
#include "volrend/renderer.hpp"
#include "volrend/mesh.hpp"

#include <ctime>
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <array>

#include "volrend/cuda/common.cuh"
#include "volrend/cuda/renderer_kernel.hpp"
#include "volrend/internal/imwrite.hpp"

namespace volrend {

// Starting CUDA/OpenGL interop code from
// https://gist.github.com/allanmac/4ff11985c3562830989f

struct VolumeRenderer::Impl {
    Impl(Camera& camera, RenderOptions& options, std::vector<Mesh>& meshes)
        : camera(camera), options(options), meshes(meshes), buf_index(0) {
        probe_ = Mesh::Cube(glm::vec3(0.0));
        probe_.name = "_probe_cube";
        probe_.visible = false;
        probe_.scale = 0.05f;
        // Make face colors
        for (int i = 0; i < 3; ++i) {
            int off = i * 12 * 9;
            for (int j = 0; j < 12; ++j) {
                int soff = off + 9 * j + 3;
                probe_.vert[soff + 2 - i] = 1.f;
            }
        }
        probe_.unlit = true;
        probe_.update();
        wire_.face_size = 2;
        wire_.unlit = true;
    }

    ~Impl() {
        // Unregister CUDA resources
        for (int index = 0; index < cgr.size(); index++) {
            if (cgr[index] != nullptr)
                cuda(GraphicsUnregisterResource(cgr[index]));
        }
        glDeleteRenderbuffers(2, rb.data());
        glDeleteRenderbuffers(2, depth_rb.data());
        glDeleteRenderbuffers(2, depth_buf_rb.data());
        glDeleteFramebuffers(2, fb.data());
        cuda(StreamDestroy(stream));
    }

    void start() {
        if (started_) return;
        cuda(StreamCreateWithFlags(&stream, cudaStreamDefault));

        glCreateRenderbuffers(2, rb.data());
        // Depth buffer cannot be read in CUDA,
        // have to write fake depth buffer manually..
        glCreateRenderbuffers(2, depth_rb.data());
        glCreateRenderbuffers(2, depth_buf_rb.data());
        glCreateFramebuffers(2, fb.data());

        // Attach rbo to fbo
        for (int index = 0; index < 2; index++) {
            glNamedFramebufferRenderbuffer(fb[index], GL_COLOR_ATTACHMENT0,
                                           GL_RENDERBUFFER, rb[index]);
            glNamedFramebufferRenderbuffer(fb[index], GL_COLOR_ATTACHMENT1,
                                           GL_RENDERBUFFER, depth_rb[index]);
            glNamedFramebufferRenderbuffer(fb[index], GL_DEPTH_ATTACHMENT,
                                           GL_RENDERBUFFER,
                                           depth_buf_rb[index]);
            const GLenum attach_buffers[]{GL_COLOR_ATTACHMENT0,
                                          GL_COLOR_ATTACHMENT1};
            glNamedFramebufferDrawBuffers(fb[index], 2, attach_buffers);
        }
        started_ = true;
    }

    void render() {
        start();
        GLfloat clear_color[] = {options.background_brightness,
                                 options.background_brightness,
                                 options.background_brightness, 1.f};
        GLfloat depth_inf = 1e9, zero = 0;
        glClearDepth(1.f);
        glClearNamedFramebufferfv(fb[buf_index], GL_COLOR, 0, clear_color);
        glClearNamedFramebufferfv(fb[buf_index], GL_COLOR, 1, &depth_inf);
        glClearNamedFramebufferfv(fb[buf_index], GL_DEPTH, 0, &depth_inf);

        probe_.visible = options.enable_probe;
        for (int i = 0; i < 3; ++i) probe_.translation[i] = options.probe[i];

        camera._update();

        if (options.show_grid) {
            maybe_gen_wire(options.grid_max_depth);
        }

        glDepthMask(GL_TRUE);
        glBindFramebuffer(GL_FRAMEBUFFER, fb[buf_index]);
        for (const Mesh& mesh : meshes) {
            mesh.draw(camera.w2c, camera.K);
        }
        probe_.draw(camera.w2c, camera.K);
        if (options.show_grid) {
            wire_.draw(camera.w2c, camera.K);
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        if (tree != nullptr) {
            cuda(GraphicsMapResources(2, &cgr[buf_index * 2], stream));
            launch_renderer(*tree, camera, options, ca[buf_index * 2],
                            ca[buf_index * 2 + 1], stream);
            cuda(GraphicsUnmapResources(2, &cgr[buf_index * 2], stream));
        }

        glNamedFramebufferReadBuffer(fb[buf_index], GL_COLOR_ATTACHMENT0);
        glBlitNamedFramebuffer(fb[buf_index], 0, 0, 0, camera.width,
                               camera.height, 0, camera.height, camera.width, 0,
                               GL_COLOR_BUFFER_BIT, GL_NEAREST);
        buf_index ^= 1;
    }

    void resize(const int width, const int height) {
        if (camera.width == width && camera.height == height) return;
        start();
        // save new size
        camera.width = width;
        camera.height = height;

        // unregister resource
        for (int index = 0; index < cgr.size(); index++) {
            if (cgr[index] != nullptr)
                cuda(GraphicsUnregisterResource(cgr[index]));
        }

        // resize color buffer
        for (int index = 0; index < 2; index++) {
            // resize rbo
            glNamedRenderbufferStorage(rb[index], GL_RGBA8, width, height);
            glNamedRenderbufferStorage(depth_rb[index], GL_R32F, width, height);
            glNamedRenderbufferStorage(depth_buf_rb[index],
                                       GL_DEPTH_COMPONENT32F, width, height);
            const GLenum attach_buffers[]{GL_COLOR_ATTACHMENT0,
                                          GL_COLOR_ATTACHMENT1};
            glNamedFramebufferDrawBuffers(fb[index], 2, attach_buffers);

            // register rbo
            cuda(GraphicsGLRegisterImage(
                &cgr[index * 2], rb[index], GL_RENDERBUFFER,
                cudaGraphicsRegisterFlagsSurfaceLoadStore |
                    cudaGraphicsRegisterFlagsWriteDiscard));
            cuda(GraphicsGLRegisterImage(
                &cgr[index * 2 + 1], depth_rb[index], GL_RENDERBUFFER,
                cudaGraphicsRegisterFlagsSurfaceLoadStore |
                    cudaGraphicsRegisterFlagsWriteDiscard));
        }

        cuda(GraphicsMapResources(cgr.size(), cgr.data(), 0));
        for (int index = 0; index < cgr.size(); index++) {
            cuda(GraphicsSubResourceGetMappedArray(&ca[index], cgr[index], 0,
                                                   0));
        }
        cuda(GraphicsUnmapResources(cgr.size(), cgr.data(), 0));
    }

    void set(N3Tree& tree) {
        start();
        this->tree = &tree;
        wire_.vert.clear();
        wire_.faces.clear();
        options.basis_minmax[0] = 0;
        options.basis_minmax[1] = std::max(tree.data_format.basis_dim - 1, 0);
        probe_.scale = 0.02f / tree.scale[0];
        last_wire_depth_ = -1;
    }

    void maybe_gen_wire(int depth) {
        if (last_wire_depth_ != depth) {
            wire_.vert = tree->gen_wireframe(depth);
            wire_.update();
            last_wire_depth_ = depth;
        }
    }

    const N3Tree* tree = nullptr;

   private:
    Camera& camera;
    RenderOptions& options;
    int buf_index;

    // GL buffers
    std::array<GLuint, 2> fb, rb, depth_rb, depth_buf_rb;

    // CUDA resources
    std::array<cudaGraphicsResource_t, 4> cgr = {{0}};
    std::array<cudaArray_t, 4> ca;

    Mesh probe_, wire_;
    // The depth level of the octree wireframe; -1 = not yet generated
    int last_wire_depth_ = -1;

    std::vector<Mesh>& meshes;
    cudaStream_t stream;
    bool started_ = false;
};

VolumeRenderer::VolumeRenderer()
    : impl_(std::make_unique<Impl>(camera, options, meshes)) {}

VolumeRenderer::~VolumeRenderer() {}

void VolumeRenderer::render() { impl_->render(); }
void VolumeRenderer::set(N3Tree& tree) { impl_->set(tree); }
void VolumeRenderer::clear() { impl_->tree = nullptr; }

void VolumeRenderer::resize(int width, int height) {
    impl_->resize(width, height);
}
const char* VolumeRenderer::get_backend() { return "CUDA"; }

}  // namespace volrend
#endif
