#include "volrend/common.hpp"

#include "volrend/renderer.hpp"
#include "volrend/mesh.hpp"
#include <glm/gtc/type_ptr.hpp>

#include <fstream>
#include <cstdio>
#include <cstdint>
#include <string>

#ifdef __EMSCRIPTEN__
// WebGL
#include <GLES3/gl3.h>
#else
#include <GL/glew.h>
#endif

#include "volrend/internal/glutil.hpp"
#include "volrend/internal/plenoctree.shader"
#include "volrend/internal/fxaa.shader"

namespace volrend {

namespace {

const float quad_verts[] = {
    -1.f, -1.f, 0.5f, 1.f, -1.f, 0.5f, -1.f, 1.f, 0.5f, 1.f, 1.f, 0.5f,
};

}  // namespace

struct VolumeRenderer::Impl {
    Impl(Camera& camera, RenderOptions& options, std::vector<Mesh>& meshes,
         int& time, int max_tries = 4)
        : camera(camera), options(options), time(time), meshes(meshes) {
        wire_.face_size = 2;
        wire_.unlit = true;
    }

    ~Impl() {
        glDeleteFramebuffers(1, &fb);
        glDeleteTextures(1, &tex_tree_data);
        glDeleteTextures(1, &tex_tree_child);
        glDeleteTextures(1, &tex_mesh_color);
        glDeleteTextures(1, &tex_mesh_depth);
        glDeleteTextures(1, &tex_mesh_depth_buf);
    }

    void start() {
        if (started_) return;
        glGetIntegerv(GL_MAX_TEXTURE_SIZE, &tex_max_size);

        glGenTextures(1, &tex_tree_data);
        glGenTextures(1, &tex_tree_child);

        glGenTextures(1, &tex_mesh_color);
        glGenTextures(1, &tex_mesh_depth);
        glGenTextures(1, &tex_mesh_depth_buf);
        glGenFramebuffers(1, &fb);

        // Put some dummy information to suppress browser warnings
        glBindTexture(GL_TEXTURE_2D, tex_tree_data);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, 1, 1, 0, GL_RED, GL_HALF_FLOAT,
                     nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glBindTexture(GL_TEXTURE_2D, tex_tree_child);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32I, 1, 1, 0, GL_RED_INTEGER, GL_INT,
                     nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glBindTexture(GL_TEXTURE_2D, 0);

        resize(800, 800);

        glBindFramebuffer(GL_FRAMEBUFFER, fb);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, tex_mesh_color, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,
                               GL_TEXTURE_2D, tex_mesh_depth, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                               GL_TEXTURE_2D, tex_mesh_depth_buf, 0);
        const GLenum attach_buffers[]{GL_COLOR_ATTACHMENT0,
                                      GL_COLOR_ATTACHMENT1};
        glDrawBuffers(2, attach_buffers);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) !=
            GL_FRAMEBUFFER_COMPLETE) {
            fprintf(stderr, "Framebuffer not complete\n");
            std::exit(1);
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        quad_init();
        shader_init();
        started_ = true;
    }

    void render() {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        if (!started_) return;

        camera._update();
        if (options.show_grid) {
            maybe_gen_wire(options.grid_max_depth);
        }

        GLfloat clear_color[] = {options.background_brightness,
                                 options.background_brightness,
                                 options.background_brightness, 1.f};
        GLfloat depth_inf = 1e9, zero = 0;

        glBindFramebuffer(GL_FRAMEBUFFER, fb);
        glDepthMask(GL_TRUE);

#ifdef __EMSCRIPTEN__
        // GLES 3
        glClearDepthf(1.f);
#else
        glClearDepth(1.f);
#endif
        glClearBufferfv(GL_COLOR, 0, clear_color);
        glClearBufferfv(GL_COLOR, 1, &depth_inf);
        glClearBufferfv(GL_DEPTH, 0, &depth_inf);

        for (const Mesh& mesh : meshes) {
            mesh.draw(camera.w2c, camera.K, false, time);
        }
        if (options.show_grid) {
            wire_.draw(camera.w2c, camera.K, false);
        }

        plenoctree_program.use();
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // FIXME reduce uniform transfers?
        glUniformMatrix4x3fv(plenoctree_program["cam.transform"], 1, GL_FALSE,
                             glm::value_ptr(camera.transform));
        glUniform2f(plenoctree_program["cam.focal"], camera.fx, camera.fy);
        glUniform2f(plenoctree_program["cam.reso"], (float)camera.width, (float)camera.height);
        glUniform1f(plenoctree_program["opt.step_size"], options.step_size);
        glUniform1f(plenoctree_program["opt.backgrond_brightness"], options.background_brightness);
        glUniform1f(plenoctree_program["opt.stop_thresh"], options.stop_thresh);
        glUniform1f(plenoctree_program["opt.sigma_thresh"], options.sigma_thresh);
        glUniform1fv(plenoctree_program["opt.render_bbox"], 6, options.render_bbox);
        glUniform1iv(plenoctree_program["opt.basis_minmax"], 2, options.basis_minmax);
        glUniform3fv(plenoctree_program["opt.rot_dirs"], 1, options.rot_dirs);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex_tree_child);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, tex_tree_data);

        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, tex_mesh_depth);

        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, tex_mesh_color);

        glBindVertexArray(vao_quad);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, (GLsizei)4);
        glBindVertexArray(0);
    }

    void set(N3Tree& tree) {
        start();
        if (tree.capacity > 0) {
            this->tree = &tree;
            upload_data();
            upload_child_links();
            upload_tree_spec();
        }
        options.basis_minmax[0] = 0;
        options.basis_minmax[1] = std::max(tree.data_format.basis_dim - 1, 0);
    }

    void maybe_gen_wire(int depth) {
        if (last_wire_depth_ != depth) {
            wire_.vert = tree->gen_wireframe(depth);
            wire_.update();
            last_wire_depth_ = depth;
        }
    }

    void clear() { this->tree = nullptr; }

    void resize(const int width, const int height) {
        if (camera.width == width && camera.height == height) return;
        if (width <= 0 || height <= 0) return;
        camera.width = width;
        camera.height = height;

        // Re-allocate memory for textures used in mesh-volume compositing
        // process
        glBindTexture(GL_TEXTURE_2D, tex_mesh_color);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA,
                     GL_UNSIGNED_BYTE, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        glBindTexture(GL_TEXTURE_2D, tex_mesh_depth);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED,
                     GL_FLOAT, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        glBindTexture(GL_TEXTURE_2D, tex_mesh_depth_buf);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, width, height, 0,
                     GL_DEPTH_COMPONENT, GL_FLOAT, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        glViewport(0, 0, width, height);
    }

   private:
    void auto_size_2d(size_t size, size_t* width, size_t* height,
                      int base_dim = 1) {
        // Will find H*W such that H*W >= size and W % base_dim == 0
        if (size == 0) {
            *width = *height = 0;
            return;
        }
        *width = std::sqrt(size);
        if (*width % base_dim) {
            *width += base_dim - (*width) % base_dim;
        }
        *height = (size - 1) / *width + 1;
        if (*height > tex_max_size || *width > tex_max_size) {
            throw std::runtime_error(
                "Octree data exceeds your OpenGL driver's 2D texture limit.\n"
                "Please try the CUDA renderer or another device.");
        }
    }

    void upload_data() {
        const GLint data_size =
            tree->capacity * tree->N * tree->N * tree->N * tree->data_dim_pad / 4;
        size_t width, height;

        auto_size_2d(data_size, &width, &height, tree->data_dim_pad / 4);
        const size_t pad = width * height - data_size;

        plenoctree_program.use();
        glUniform1i(plenoctree_program["tree_data_stride"], width);

        tree->data_.data_holder.resize((data_size + pad) * sizeof(half) * 4);
        glBindTexture(GL_TEXTURE_2D, tex_tree_data);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F,
                     width, height, 0, GL_RGBA,
                     GL_HALF_FLOAT, (void*)tree->data_.data<half>());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        glBindTexture(GL_TEXTURE_2D, 0);
    }

    void upload_child_links() {
        plenoctree_program.use();
        const size_t child_size =
            size_t(tree->capacity) * tree->N * tree->N * tree->N;
        size_t width, height;
        auto_size_2d(child_size, &width, &height);

        const size_t pad = width * height - child_size;
        tree->child_.data_holder.resize((child_size + pad) * sizeof(int32_t));
        glUniform1i(plenoctree_program["tree_child_stride"], width);

        glBindTexture(GL_TEXTURE_2D, tex_tree_child);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32I, width, height, 0,
                     GL_RED_INTEGER, GL_INT, tree->child_.data<int32_t>());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        glBindTexture(GL_TEXTURE_2D, 0);
    }

    void upload_tree_spec() {
        plenoctree_program.use();
        glUniform1i(plenoctree_program["tree.N"], tree->N);
        glUniform1i(plenoctree_program["tree.data_dim"], tree->data_dim);
        glUniform1i(plenoctree_program["tree.data_dim_rgba"], tree->data_dim_pad / 4);
        glUniform1i(plenoctree_program["tree.format"], (int)tree->data_format.format);
        glUniform1i(plenoctree_program["tree.basis_dim"],
                    tree->data_format.format == DataFormat::RGBA
                        ? 1
                        : tree->data_format.basis_dim);
        glUniform3f(plenoctree_program["tree.center"],
                    tree->offset[0], tree->offset[1], tree->offset[2]);
        glUniform3f(plenoctree_program["tree.scale"], tree->scale[0],
                    tree->scale[1], tree->scale[2]);
        if (tree->use_ndc) {
            glUniform1f(plenoctree_program["tree.ndc_width"],
                        tree->ndc_width);
            glUniform1f(plenoctree_program["tree.ndc_height"],
                        tree->ndc_height);
            glUniform1f(plenoctree_program["tree.ndc_focal"],
                        tree->ndc_focal);
        } else {
            glUniform1f(plenoctree_program["tree.ndc_width"], -1.f);
        }
    }

    void shader_init() {
        plenoctree_program = GLShader(PLENOCTREE_SHADER_SRC, "PLENOCTREE");
        fxaa_program = GLShader(FXAA_SHADER_SRC, "FXAA");

        plenoctree_program.use();
        plenoctree_program.set_texture_uniforms(
                {"tree_child_tex", "tree_data_tex", "mesh_depth_tex", "mesh_color_tex"});
        glUniform1i(plenoctree_program["tree_data_stride"], 0);
    }

    void quad_init() {
        GLuint vbo;
        glGenBuffers(1, &vbo);
        glGenVertexArrays(1, &vao_quad);
        glBindVertexArray(vao_quad);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof quad_verts, (GLvoid*)quad_verts,
                     GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float),
                              (GLvoid*)0);
        glEnableVertexAttribArray(0);
        glBindVertexArray(0);
    }

    Camera& camera;
    RenderOptions& options;

    int& time;

    N3Tree* tree;

    GLShader plenoctree_program, fxaa_program;
    GLuint tex_tree_data = -1, tex_tree_child;
    GLuint vao_quad;
    GLint tex_max_size;

    Mesh wire_;
    // The depth level of the octree wireframe; -1 = not yet generated
    int last_wire_depth_ = -1;

    std::vector<Mesh>& meshes;

    GLuint fb, tex_mesh_color, tex_mesh_depth, tex_mesh_depth_buf;

    std::string shader_fname = "shaders/rt.frag";

    bool started_ = false;
};

VolumeRenderer::VolumeRenderer()
    : impl_(std::make_unique<Impl>(camera, options, meshes, time)) {}

VolumeRenderer::~VolumeRenderer() {}

void VolumeRenderer::render() { impl_->render(); }

void VolumeRenderer::set(N3Tree& tree) { impl_->set(tree); }
void VolumeRenderer::clear() { impl_->clear(); }

void VolumeRenderer::resize(int width, int height) {
    impl_->resize(width, height);
}
const char* VolumeRenderer::get_backend() { return "Shader"; }

}  // namespace volrend
