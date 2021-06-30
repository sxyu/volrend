#include "volrend/common.hpp"

// Shader backend only enabled when build with VOLREND_USE_CUDA=OFF
#ifndef VOLREND_CUDA
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

#include "volrend/internal/rt_frag.inl"
#include "volrend/internal/shader.hpp"

namespace volrend {

namespace {

const char* PASSTHRU_VERT_SHADER_SRC =
    R"glsl(
in vec3 aPos;

void main()
{
    gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);
}
)glsl";

const float quad_verts[] = {
    -1.f, -1.f, 0.5f, 1.f, -1.f, 0.5f, -1.f, 1.f, 0.5f, 1.f, 1.f, 0.5f,
};

struct _RenderUniforms {
    GLint cam_transform, cam_focal, cam_reso;
    GLint opt_step_size, opt_backgrond_brightness, opt_stop_thresh,
        opt_sigma_thresh, opt_render_bbox, opt_basis_minmax, opt_rot_dirs;
    GLint tree_data_tex, tree_child_tex;  //, tree_extra_tex;
    GLint mesh_depth_tex, mesh_color_tex;
};

}  // namespace

struct VolumeRenderer::Impl {
    Impl(Camera& camera, RenderOptions& options, std::vector<Mesh>& meshes,
         int max_tries = 4)
        : camera(camera), options(options), meshes(meshes) {
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
        glDeleteProgram(program);
        glDeleteFramebuffers(1, &fb);
        glDeleteTextures(1, &tex_tree_data);
        glDeleteTextures(1, &tex_tree_child);
        glDeleteTextures(1, &tex_tree_extra);
        glDeleteTextures(1, &tex_mesh_color);
        glDeleteTextures(1, &tex_mesh_depth);
        glDeleteTextures(1, &tex_mesh_depth_buf);
    }

    void start() {
        if (started_) return;
        glGetIntegerv(GL_MAX_TEXTURE_SIZE, &tex_max_size);
        // int tex_3d_max_size;
        // glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &tex_3d_max_size);

        glGenTextures(1, &tex_tree_data);
        glGenTextures(1, &tex_tree_child);
        glGenTextures(1, &tex_tree_extra);

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

        Mesh::use_shader();
        for (const Mesh& mesh : meshes) {
            mesh.draw(camera.w2c, camera.K, false);
        }
        probe_.draw(camera.w2c, camera.K);
        if (options.show_grid) {
            wire_.draw(camera.w2c, camera.K, false);
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glUseProgram(program);

        // FIXME reduce uniform transfers?
        glUniformMatrix4x3fv(u.cam_transform, 1, GL_FALSE,
                             glm::value_ptr(camera.transform));
        glUniform2f(u.cam_focal, camera.fx, camera.fy);
        glUniform2f(u.cam_reso, (float)camera.width, (float)camera.height);
        glUniform1f(u.opt_step_size, options.step_size);
        glUniform1f(u.opt_backgrond_brightness, options.background_brightness);
        glUniform1f(u.opt_stop_thresh, options.stop_thresh);
        glUniform1f(u.opt_sigma_thresh, options.sigma_thresh);
        glUniform1fv(u.opt_render_bbox, 6, options.render_bbox);
        glUniform1iv(u.opt_basis_minmax, 2, options.basis_minmax);
        glUniform3fv(u.opt_rot_dirs, 1, options.rot_dirs);

        // FIXME Probably can be done only once
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex_tree_child);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, tex_tree_data);

        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, tex_mesh_depth);

        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, tex_mesh_color);

        // glActiveTexture(GL_TEXTURE4);
        // glBindTexture(GL_TEXTURE_2D, tex_tree_extra);
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
    void auto_size_2d(size_t size, size_t& width, size_t& height,
                      int base_dim = 1) {
        if (size == 0) {
            width = height = 0;
            return;
        }
        width = std::sqrt(size);
        if (width % base_dim) {
            width += base_dim - width % base_dim;
        }
        height = (size - 1) / width + 1;
        if (height > tex_max_size || width > tex_max_size) {
            throw std::runtime_error(
                "Octree data exceeds your OpenGL driver's 2D texture limit.\n"
                "Please try the CUDA renderer or another device.");
        }
    }

    void upload_data() {
        const GLint data_size =
            tree->capacity * tree->N * tree->N * tree->N * tree->data_dim;
        size_t width, height;
        auto_size_2d(data_size, width, height, tree->data_dim);
        const size_t pad = width * height - data_size;

        glUseProgram(program);
        glUniform1i(glGetUniformLocation(program, "tree_data_dim"), width);

#ifdef __EMSCRIPTEN__
        tree->data_.data_holder.resize((data_size + pad) * sizeof(half));
        glBindTexture(GL_TEXTURE_2D, tex_tree_data);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, width, height, 0, GL_RED,
                     GL_HALF_FLOAT, tree->data_.data<half>());
#else
        // FIXME: there seems to be some weird bug in the NVIDIA OpenGL
        // implementation where GL_HALF_FLOAT is sometimes ignored, and we have
        // to use float32 for uploads
        std::vector<float> tmp(data_size + pad);
        std::copy(tree->data_.data<half>(),
                  tree->data_.data<half>() + data_size, tmp.begin());
        glBindTexture(GL_TEXTURE_2D, tex_tree_data);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, width, height, 0, GL_RED,
                     GL_FLOAT, (void*)tmp.data());
#endif
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        // Maybe upload extra data
        const size_t extra_sz = tree->extra_.data_holder.size() / sizeof(float);
        if (extra_sz) {
            glBindTexture(GL_TEXTURE_2D, tex_tree_extra);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F,
                         extra_sz / tree->data_format.basis_dim,
                         tree->data_format.basis_dim, 0, GL_RED, GL_FLOAT,
                         tree->extra_.data<float>());
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        }

        glBindTexture(GL_TEXTURE_2D, 0);
    }

    void upload_child_links() {
        const size_t child_size =
            size_t(tree->capacity) * tree->N * tree->N * tree->N;
        size_t width, height;
        auto_size_2d(child_size, width, height);

        const size_t pad = width * height - child_size;
        tree->child_.data_holder.resize((child_size + pad) * sizeof(int32_t));
        glUniform1i(glGetUniformLocation(program, "tree_child_dim"), width);

        glBindTexture(GL_TEXTURE_2D, tex_tree_child);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32I, width, height, 0,
                     GL_RED_INTEGER, GL_INT, tree->child_.data<int32_t>());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        glBindTexture(GL_TEXTURE_2D, 0);
    }

    void upload_tree_spec() {
        glUniform1i(glGetUniformLocation(program, "tree.N"), tree->N);
        glUniform1i(glGetUniformLocation(program, "tree.data_dim"),
                    tree->data_dim);
        glUniform1i(glGetUniformLocation(program, "tree.format"),
                    (int)tree->data_format.format);
        glUniform1i(glGetUniformLocation(program, "tree.basis_dim"),
                    tree->data_format.format == DataFormat::RGBA
                        ? 1
                        : tree->data_format.basis_dim);
        glUniform3f(glGetUniformLocation(program, "tree.center"),
                    tree->offset[0], tree->offset[1], tree->offset[2]);
        glUniform3f(glGetUniformLocation(program, "tree.scale"), tree->scale[0],
                    tree->scale[1], tree->scale[2]);
        if (tree->use_ndc) {
            glUniform1f(glGetUniformLocation(program, "tree.ndc_width"),
                        tree->ndc_width);
            glUniform1f(glGetUniformLocation(program, "tree.ndc_height"),
                        tree->ndc_height);
            glUniform1f(glGetUniformLocation(program, "tree.ndc_focal"),
                        tree->ndc_focal);
        } else {
            glUniform1f(glGetUniformLocation(program, "tree.ndc_width"), -1.f);
        }
    }

    void shader_init() {
        program = create_shader_program(PASSTHRU_VERT_SHADER_SRC, RT_FRAG_SRC);

        u.cam_transform = glGetUniformLocation(program, "cam.transform");
        u.cam_focal = glGetUniformLocation(program, "cam.focal");
        u.cam_reso = glGetUniformLocation(program, "cam.reso");
        u.opt_step_size = glGetUniformLocation(program, "opt.step_size");
        u.opt_backgrond_brightness =
            glGetUniformLocation(program, "opt.background_brightness");
        u.opt_stop_thresh = glGetUniformLocation(program, "opt.stop_thresh");
        u.opt_sigma_thresh = glGetUniformLocation(program, "opt.sigma_thresh");
        u.opt_render_bbox = glGetUniformLocation(program, "opt.render_bbox");
        u.opt_basis_minmax = glGetUniformLocation(program, "opt.basis_minmax");
        u.opt_rot_dirs = glGetUniformLocation(program, "opt.rot_dirs");
        u.tree_data_tex = glGetUniformLocation(program, "tree_data_tex");
        u.tree_child_tex = glGetUniformLocation(program, "tree_child_tex");
        u.mesh_depth_tex = glGetUniformLocation(program, "mesh_depth_tex");
        u.mesh_color_tex = glGetUniformLocation(program, "mesh_color_tex");
        // u.tree_extra_tex = glGetUniformLocation(program, "tree_extra_tex");
        glUniform1i(u.tree_child_tex, 0);
        glUniform1i(u.tree_data_tex, 1);
        glUniform1i(u.mesh_depth_tex, 2);
        glUniform1i(u.mesh_color_tex, 3);
        glUniform1i(glGetUniformLocation(program, "tree_data_dim"), 0);
        // glUniform1i(u.tree_extra_tex, 4);
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

    N3Tree* tree;

    GLuint program = -1;
    GLuint tex_tree_data = -1, tex_tree_child, tex_tree_extra;
    GLuint vao_quad;
    GLint tex_max_size;

    Mesh probe_, wire_;
    // The depth level of the octree wireframe; -1 = not yet generated
    int last_wire_depth_ = -1;

    std::vector<Mesh>& meshes;

    GLuint fb, tex_mesh_color, tex_mesh_depth, tex_mesh_depth_buf;

    std::string shader_fname = "shaders/rt.frag";

    _RenderUniforms u;
    bool started_ = false;
};

VolumeRenderer::VolumeRenderer()
    : impl_(std::make_unique<Impl>(camera, options, meshes)) {}

VolumeRenderer::~VolumeRenderer() {}

void VolumeRenderer::render() { impl_->render(); }

void VolumeRenderer::set(N3Tree& tree) { impl_->set(tree); }
void VolumeRenderer::clear() { impl_->clear(); }

void VolumeRenderer::resize(int width, int height) {
    impl_->resize(width, height);
}
const char* VolumeRenderer::get_backend() { return "Shader"; }

}  // namespace volrend

#endif
