#include "volrend/common.hpp"

// Compute-shader backend only enabled when build with VOLREND_USE_CUDA=OFF
#ifndef VOLREND_CUDA
#include "volrend/renderer.hpp"
#include <glm/gtc/type_ptr.hpp>

#include <fstream>
#include <cstdint>
#include <string>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

namespace volrend {

const float axes_verts[] = {
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
};

namespace {
void check_compile_errors(GLuint shader, const std::string& type) {
    GLint success;
    GLchar infoLog[1024];
    if (type != "PROGRAM") {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 1024, NULL, infoLog);
            std::cout
                << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n"
                << infoLog
                << "\n -- ---------------------------------------------------  "
                << std::endl;
        }
    } else {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shader, 1024, NULL, infoLog);
            std::cout
                << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n"
                << infoLog
                << "\n -- --------------------------------------------------- "
                   "-- "
                << std::endl;
        }
    }
}
}  // namespace

struct VolumeRenderer::Impl {
    Impl(Camera& camera, RenderOptions& options, int max_tries = 4)
        : camera(camera), options(options), buf_index(0) {
        std::string shader_fname = "shaders/render.comp";
        int i;
        for (i = 0; i < max_tries; ++i) {
            if (std::ifstream(shader_fname)) break;
            shader_fname = "../" + shader_fname;
        }
        if (i == max_tries) {
            throw std::runtime_error(
                "Could not find the compute shader! "
                "Please launch pogram in project directory or a subdirectory.");
        }
        glCreateFramebuffers(2, fb.data());
        glGenTextures(2, fb_tex.data());
        glGenBuffers(1, &ssb_tree_data);
        glGenBuffers(1, &ssb_tree_child);
        resize(0, 0);

        std::ifstream shader_ifs(shader_fname);
        shader_ifs.seekg(0, std::ios::end);
        size_t size = shader_ifs.tellg();
        std::string shader_source(size, ' ');
        shader_ifs.seekg(0);
        shader_ifs.read(&shader_source[0], size);
        const char* ptr = shader_source.c_str();

        GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
        glShaderSource(shader, 1, &ptr, NULL);
        glCompileShader(shader);
        check_compile_errors(shader, "COMPUTE");

        program = glCreateProgram();
        glAttachShader(program, shader);
        glLinkProgram(program);
        check_compile_errors(program, "PROGRAM");
        glDeleteShader(shader);
        glUseProgram(program);
    }

    ~Impl() {
        glDeleteTextures(2, fb_tex.data());
        glDeleteFramebuffers(2, fb.data());
        glDeleteProgram(program);
    }

    void render() {
        GLfloat clear_color[] = {1.f, 1.f, 1.f, 1.f};
        glClearNamedFramebufferfv(fb[buf_index], GL_COLOR, 0, clear_color);

        camera._update();
        if (tree == nullptr) return;
        // FIXME reduce uniform transfers?
        glUniformMatrix4x3fv(glGetUniformLocation(program, "cam.transform"), 1,
                             GL_FALSE, glm::value_ptr(camera.transform));
        glm::vec2 focal_norm(camera.focal / (camera.width * 0.5f),
                             camera.focal / (camera.height * 0.5f));
        glUniform2f(glGetUniformLocation(program, "cam.focal"), focal_norm.x,
                    focal_norm.y);
        glUniform1f(glGetUniformLocation(program, "opt.step_size"),
                    options.step_size);
        glUniform1f(glGetUniformLocation(program, "opt.background_brightness"),
                    options.background_brightness);
        glUniform1f(glGetUniformLocation(program, "opt.stop_thresh"),
                    options.stop_thresh);
        glUniform1f(glGetUniformLocation(program, "opt.sigma_thresh"),
                    options.sigma_thresh);
        glUniform1i(glGetUniformLocation(program, "opt.show_miss"),
                    options.show_miss);

        // Run compute shader on image texture
        glBindImageTexture(0, fb_tex[buf_index], 0, GL_FALSE, 0, GL_WRITE_ONLY,
                           GL_RGBA32F);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssb_tree_data);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssb_tree_child);

        glDispatchCompute((GLuint)camera.width, (GLuint)camera.height, 1);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, 0);

        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        glBindImageTexture(0, 0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

        glBlitNamedFramebuffer(fb[buf_index], 0, 0, 0, camera.width,
                               camera.height, 0, camera.height, camera.width, 0,
                               GL_COLOR_BUFFER_BIT, GL_NEAREST);
        buf_index ^= 1;
    }

    void set(const N3Tree& tree) {
        this->tree = &tree;
        upload_data();
        upload_child_links();
        upload_tree_spec();
    }

    void clear() {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssb_tree_data);
        glBufferData(GL_SHADER_STORAGE_BUFFER, 0, nullptr, GL_STATIC_READ);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssb_tree_child);
        glBufferData(GL_SHADER_STORAGE_BUFFER, 0, nullptr, GL_STATIC_READ);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        this->tree = nullptr;
    }

    void resize(const int width, const int height) {
        if (camera.width == width && camera.height == height) return;
        if (width > 0) {
            camera.width = width;
            camera.height = height;
        }

        glActiveTexture(GL_TEXTURE0);
        for (int i = 0; i < 2; ++i) {
            glBindFramebuffer(GL_FRAMEBUFFER, fb[i]);
            glBindTexture(GL_TEXTURE_2D, fb_tex[i]);

            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, camera.width,
                         camera.height, 0, GL_RGBA, GL_FLOAT, 0);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_2D, fb[i], 0);
        }
        glBindTexture(GL_TEXTURE_2D, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

   private:
    void upload_data() {
        const size_t data_size = size_t(tree->capacity) * tree->N * tree->N *
                                 tree->N * tree->data_dim * sizeof(float);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssb_tree_data);
        const float* data_ptr = tree->data_ptr();
        glBufferData(GL_SHADER_STORAGE_BUFFER, data_size, data_ptr,
                     GL_STATIC_READ);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }
    void upload_child_links() {
        const size_t child_size = size_t(tree->capacity) * tree->N * tree->N *
                                  tree->N * sizeof(int32_t);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssb_tree_child);
        glBufferData(GL_SHADER_STORAGE_BUFFER, child_size,
                     tree->child_.data<int32_t>(), GL_STATIC_READ);

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }
    void upload_tree_spec() {
        glUniform1i(glGetUniformLocation(program, "tree.N"), tree->N);
        glUniform1i(glGetUniformLocation(program, "tree.data_dim"),
                    tree->data_dim);
        glUniform1i(glGetUniformLocation(program, "tree.sh_order"),
                    tree->sh_order);
        glUniform1i(glGetUniformLocation(program, "tree.n_coe"),
                    (tree->sh_order + 1) * (tree->sh_order + 1));
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

    Camera& camera;
    RenderOptions& options;
    int buf_index;

    const N3Tree* tree;

    GLuint program = -1;
    std::array<GLuint, 2> fb, fb_tex;
    GLuint ssb_tree_data, ssb_tree_child;
};

VolumeRenderer::VolumeRenderer(int device_id)
    : impl_(std::make_unique<Impl>(camera, options)) {
    const GLubyte* vendor = glGetString(GL_VENDOR);  // Returns the vendor
    const GLubyte* renderer =
        glGetString(GL_RENDERER);  // Returns a hint to the model
    printf("OpenGL : %s %s\n", vendor, renderer);
    int work_grp_cnt[3];

    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &work_grp_cnt[0]);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &work_grp_cnt[1]);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, &work_grp_cnt[2]);

    printf("Max global (total) work group counts x:%i y:%i z:%i\n",
           work_grp_cnt[0], work_grp_cnt[1], work_grp_cnt[2]);

    int work_grp_size[3];

    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &work_grp_size[0]);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &work_grp_size[1]);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &work_grp_size[2]);

    printf("Max local (in one shader) work group sizes x:%i y:%i z:%i\n",
           work_grp_size[0], work_grp_size[1], work_grp_size[2]);

    int work_grp_inv;
    glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &work_grp_inv);
    printf("max local work group invocations %i\n", work_grp_inv);
}

VolumeRenderer::~VolumeRenderer() {}

void VolumeRenderer::render() { impl_->render(); }

void VolumeRenderer::set(const N3Tree& tree) { impl_->set(tree); }
void VolumeRenderer::clear() { impl_->clear(); }

void VolumeRenderer::resize(int width, int height) {
    impl_->resize(width, height);
}
const char* VolumeRenderer::get_backend() { return "CS"; }

}  // namespace volrend

#endif
