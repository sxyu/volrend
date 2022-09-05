#include "volrend/internal/glutil.hpp"
#include <iostream>

namespace volrend {

void check_compile_errors(GLuint shader, const std::string& type, std::string debug_escription) {
    GLint success;
    GLchar infoLog[1024];
    if (type != "PROGRAM") {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 1024, NULL, infoLog);
            printf(
                "ERROR::SHADER_COMPILATION_ERROR of type: %s on shader %s\n%s\n"
                "-- --------------------------------------------------- --\n",
                type.c_str(), debug_escription.c_str(), infoLog);
            fflush(stdout);
        }
    } else {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shader, 1024, NULL, infoLog);
            printf(
                "ERROR::PROGRAM_LINKING_ERROR of type: %s on shader %s\n%s\n"
                "-- --------------------------------------------------- --\n",
                type.c_str(), debug_escription.c_str(), infoLog);
            fflush(stdout);
        }
    }
}

// Currently supports vertex/fragment shaders
GLuint create_shader_program(const std::string& shader_src,
                             const std::string& debug_description,
                             const std::string& src_prepend,
                             bool use_geometry_shader) {
    // Auto-prepend the version
#ifdef __EMSCRIPTEN__
    const std::string version_str = "#version 300 es\n";
#else
    const std::string version_str = "#version 330\n";
#endif
    std::string vert_shader_src = version_str + "#define VERTEX\n" + src_prepend + "\n" + shader_src;
    std::string frag_shader_src = version_str + "#define FRAGMENT\n" + src_prepend + "\n" + shader_src;
    // Dummy vertex shader
    GLuint vert_shader = glCreateShader(GL_VERTEX_SHADER);
    const GLchar* vert_shader_src_ptr = vert_shader_src.c_str();
    glShaderSource(vert_shader, 1, &vert_shader_src_ptr, NULL);
    glCompileShader(vert_shader);
    check_compile_errors(vert_shader, "VERTEX", debug_description);

    // Fragment shader
    GLuint frag_shader = glCreateShader(GL_FRAGMENT_SHADER);
    const GLchar* frag_shader_src_ptr = frag_shader_src.c_str();
    glShaderSource(frag_shader, 1, &frag_shader_src_ptr, NULL);
    glCompileShader(frag_shader);
    check_compile_errors(frag_shader, "FRAGMENT", debug_description);

    // Geometry shader (optional)
    GLuint geom_shader;
    if (use_geometry_shader) {
#ifdef __EMSCRIPTEN__
        puts("Geometry shader is not supported in WebGL");
#else
        std::string geom_shader_src = version_str + "#define GEOMETRY\n" + shader_src;
        geom_shader = glCreateShader(GL_GEOMETRY_SHADER);
        const GLchar* geom_shader_src_ptr = geom_shader_src.c_str();
        glShaderSource(geom_shader, 1, &geom_shader_src_ptr, NULL);
        glCompileShader(geom_shader);
        check_compile_errors(geom_shader, "GEOMETRY", debug_description);
#endif
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, vert_shader);
    glAttachShader(program, frag_shader);
    if (use_geometry_shader) {
        glAttachShader(program, geom_shader);
    }

    glLinkProgram(program);
    check_compile_errors(program, "PROGRAM", debug_description);

    glDeleteShader(vert_shader);
    glDeleteShader(frag_shader);
    if (use_geometry_shader) {
        glDeleteShader(geom_shader);
    }
    return program;
}

GLShader::GLShader() : m_program(-1) {}

GLShader::GLShader(const std::string &shader_src, const std::string &debug_description,
                   const std::string &src_prepend, bool use_geometry_shader)
{
    m_uniforms.reserve(15);
    m_program = create_shader_program(shader_src,
                                      debug_description,
                                      src_prepend,
                                      use_geometry_shader);
    }

GLShader::GLShader(GLShader&& other) : m_program(other.m_program) {
    other.m_program = -1;
}
GLShader& GLShader::operator= (GLShader&& other) {
    if (m_program != -1) {
        glDeleteProgram(m_program);
    }
    m_program = other.m_program;
    other.m_program = -1;
    return *this;
}

GLShader::~GLShader() {
    if (m_program != -1) {
        glDeleteProgram(m_program);
    }
}

void GLShader::use() const {
    if (m_program == -1) {
        puts("Trying to use NULL shader program");
    } else {
        glUseProgram(m_program);
    }
}

void GLShader::set_texture_uniforms(const std::initializer_list<std::string>& tex_names) const {
    use();
    int i = 0;
    for (const std::string& tex_name : tex_names) {
        glUniform1i((*this)[tex_name], i++);
    }
}

GLint GLShader::operator[](const std::string& name) const {
    auto it = m_uniforms.find(name);
    GLint unif;
    if (it == m_uniforms.end()) {
        unif = glGetUniformLocation(m_program, name.c_str());
        m_uniforms[name] = unif;
    } else {
        unif = it->second;
    }
    return unif;
}

void GLImage2D::generate(int width, int height, void* data) {
    if (m_buffer_type == GLImage2D::NONE) return;
    else if (m_buffer_type == GLImage2D::RENDER_BUFFER) {
        if (m_id == (GLuint)-1) {
            glGenRenderbuffers(1, &m_id);
        }
        glBindRenderbuffer(GL_RENDERBUFFER, m_id);
        glRenderbufferStorage(GL_RENDERBUFFER, m_internalformat,
                width,
                height);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);
    } else {
        if (m_id == (GLuint)-1) {
            glGenTextures(1, &m_id);
        }
        glBindTexture(GL_TEXTURE_2D, m_id);
        glTexImage2D(GL_TEXTURE_2D, 0, m_internalformat,
                width, height, 0, m_format, m_type, data);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, m_interp_type);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, m_interp_type);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, m_clamp_type);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, m_clamp_type);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, m_clamp_type);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
}

void GLImage2D::free() {
    if (m_buffer_type == GLImage2D::NONE || m_id == (GLuint)-1) return;
    else if (m_buffer_type == GLImage2D::RENDER_BUFFER) {
        glDeleteRenderbuffers(1, &m_id);
    } else {
        glDeleteTextures(1, &m_id);
    }
    m_buffer_type = GLImage2D::NONE;
}

void GLImage2D::bind_unit(int unit) {
    if (m_buffer_type == GLImage2D::TEXTURE) {
        glActiveTexture(GL_TEXTURE0 + unit);
        glBindTexture(GL_TEXTURE_2D, m_id);
    }
}

GLFramebuffer::GLFramebuffer() : m_fbo(0) { }
GLFramebuffer::GLFramebuffer(int width,
                             int height,
                             std::vector<std::pair<GLint, GLImage2D>>&& attachments)
    : m_width(width), m_height(height) {
    glGenFramebuffers(1, &m_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);

    size_t color_attachment_cnt = 0;
    m_attachments.reserve(attachments.size());
    for (auto& att_pair : attachments) {
        GLint attachment = std::move(att_pair.first);
        GLImage2D buf = att_pair.second;
        buf.generate(m_width, m_height);

        if (buf.m_buffer_type == GLImage2D::RENDER_BUFFER) {
            glFramebufferRenderbuffer(
                    GL_FRAMEBUFFER,
                    attachment,
                    GL_RENDERBUFFER,
                    buf.m_id);
        } else if (buf.m_buffer_type == GLImage2D::TEXTURE) {
            glFramebufferTexture2D(
                    GL_FRAMEBUFFER,
                    attachment, GL_TEXTURE_2D, buf.m_id, 0);
        }

        m_attachments[att_pair.first] = std::move(buf);
        if (att_pair.first != GL_DEPTH_ATTACHMENT) {
            ++color_attachment_cnt;
        }
    }
    std::vector<GLenum> draw_buffers(color_attachment_cnt);
    color_attachment_cnt = 0;
    for (const auto& att_pair : attachments) {
        if (att_pair.first != GL_DEPTH_ATTACHMENT) {
            draw_buffers[color_attachment_cnt++] = att_pair.first;
        }
    }
    glDrawBuffers(draw_buffers.size(), draw_buffers.data());
    check();
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

GLFramebuffer::GLFramebuffer(GLFramebuffer&& other) : m_fbo(other.m_fbo) {
    m_attachments = std::move(other.m_attachments);
    m_width = other.m_width;
    m_height = other.m_height;
    other.m_fbo = 0;
}
GLFramebuffer& GLFramebuffer::operator= (GLFramebuffer&& other) {
    for (auto& att_buf : m_attachments) att_buf.second.free();
    free_fbo();
    m_fbo = other.m_fbo;
    m_attachments = std::move(other.m_attachments);
    m_width = other.m_width;
    m_height = other.m_height;
    other.m_fbo = 0;
    return *this;
}

GLFramebuffer GLFramebuffer::Screen(int width, int height) {
    GLFramebuffer buf;
    buf.m_fbo = 0;
    buf.m_width = width;
    buf.m_height = height;
    return buf;
}

GLFramebuffer::~GLFramebuffer() {
    for (auto& att_buf : m_attachments) att_buf.second.free();
    free_fbo();
}

void GLFramebuffer::free_fbo() {
    if (m_fbo) {
        glDeleteFramebuffers(1, &m_fbo);
    }
    m_fbo = 0;
}

void GLFramebuffer::bind() const {
    glViewport(0, 0, m_width, m_height);
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
}

bool GLFramebuffer::check() const {
    if (m_fbo) {
        GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if(status != GL_FRAMEBUFFER_COMPLETE) {
            puts("FBO construction Failed");
            switch (status) {
                case GL_FRAMEBUFFER_UNDEFINED:
                    puts("GL_FRAMEBUFFER_UNDEFINED");
                    return false;
                case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
                    puts("GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT");
                    return false;
                case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
                    puts("GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT");
                    return false;
                case GL_FRAMEBUFFER_UNSUPPORTED:
                    puts("FRAMEBUFFER_UNSUPPORTED");
                    return false;
                case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
                    puts("GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE");
                    return false;
            }
        }
        return true;
    } else {
        puts("Framebuffer is null");
        return false;
    }
}

void GLFramebuffer::set_draw_buffers(
        const std::vector<GLenum>& draw_buffers) const {
    glBindFramebuffer(GL_READ_FRAMEBUFFER, m_fbo);
    glDrawBuffers(draw_buffers.size(), draw_buffers.data());
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
}

void GLFramebuffer::blit_to(const GLFramebuffer& other, GLint read_attachment) const {
    glBindFramebuffer(GL_READ_FRAMEBUFFER, m_fbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, other.m_fbo);
    glReadBuffer(read_attachment);
    glBlitFramebuffer(0, 0, m_width, m_height, 0, 0,
            m_width, m_height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
}

void GLFramebuffer::resize(int width, int height) {
    m_width = width;
    m_height = height;

    // Re-init FBOs and RBOs
    if (m_fbo) {
        glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
        for (auto& att_buf : m_attachments) {
            att_buf.second.generate(m_width, m_height);
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
}

}  // namespace volrend
