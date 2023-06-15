// Basic personal OpenGL helpers
#include <string>
#include <cstdio>
#include <vector>
#include <unordered_map>

#ifdef __EMSCRIPTEN__
// WebGL
#include <GLES3/gl3.h>
#else
#include <GL/glew.h>
#endif

namespace volrend {

namespace util {

// Draw a full screen quad, initializing geometry lazily
void draw_fs_quad();

} // namespace util

// Shader; each shader program has a UNIQUE instance
// that is move-only
class GLShader {
public:
    GLShader();
    GLShader(GLShader&&);
    GLShader& operator= (GLShader&&);

    // No copying allowed
    GLShader(const GLShader&) =delete;
    GLShader& operator= (const GLShader& ) =delete;

    // Combined vertex/fragment shader, and geometry shader if use_geometry_shader=true.
    // shader_src should be base64-encoded combined shader source
    //
    // For vertex shader use
    // #if defined(VERTEX) ... #endif
    // for fragment shader
    // #if defined(FRAGMENT) ... #endif
    // for geometry shader
    // #if defined(GEOMETRY) ... #endif
    // see current shaders in shaders/ for examples
    explicit GLShader(const std::string& shader_src,
                      const std::string& debug_description = "",
                      const std::string& src_prepend = "",
                      bool use_geometry_shader = false);

    ~GLShader();

    // Use the shader program
    void use() const;

    inline explicit operator bool() const { return m_program != -1; }

    // Set texture uniforms with given names to texture unit IDs 0, ... 1
    void set_texture_uniforms(
            const std::initializer_list<std::string>& tex_names) const;

    GLint operator[](const std::string& name) const;

    GLuint m_program;
private:
    mutable std::unordered_map<std::string, GLint> m_uniforms;
};

// Texture/holder used by GLFramebuffer
// note that this is not move-only for convenience during FBO construction.
// We must use generate/free to manage the GPU memory.
struct GLImage2D {
    void generate(int width, int height, void* data = nullptr);
    void free();

    // Bind to a texture unit (texture only), starting from 0 (NOT enum)
    void bind_unit(int unit);

    enum Image2DType {
        NONE,
        RENDER_BUFFER,
        TEXTURE
    } m_buffer_type = NONE;

    GLint m_internalformat = GL_RGBA16F;

    // Only for texture
    GLint m_format = GL_RGBA;
    GLint m_type = GL_HALF_FLOAT;
    GLint m_interp_type = GL_NEAREST;
    GLint m_clamp_type = GL_CLAMP_TO_EDGE;

    // Texture/renderbuffer ID
    GLuint m_id = GLuint(-1);

    inline operator uint32_t() { return m_id; }
};

// Framebuffer utility; each framebuffer has a UNIQUE instance
// that is move-only.
// Special construction syntax:
// list all desired attached textures in an initializer list with 
// the attachment points (e.g. GL_COLOR_ATTACHMENT0, GL_DEPTH_ATTACHMENT)
//
// GLFramebuffer(
//      width,
//      height,
//      { 
//          {ATTACHMENT, {RENDER_BUFFER/TEXTURE,
//                  internalformat[, format, type, interp_type, clamp_type]}},
//           ...
//      }
// );
class GLFramebuffer {
public:
    GLFramebuffer();
    GLFramebuffer(int width,
            int height,
            std::vector<std::pair<GLint, GLImage2D>>&& attachments);
    GLFramebuffer(GLFramebuffer&&);
    GLFramebuffer& operator= (GLFramebuffer&&);

    // No copying allowed
    GLFramebuffer(const GLFramebuffer&) =delete;
    GLFramebuffer& operator= (const GLFramebuffer& ) =delete;

    ~GLFramebuffer();
    // Manually free framebuffer (does not free the textures)
    void free_fbo();

    // Create a framebuffer object representing the screen (0)
    static GLFramebuffer Screen(int width, int height);

    // Bind the framebuffer and set the glViewport
    void bind() const;
    bool check() const;

    void set_draw_buffers(
            const std::vector<GLenum>& draw_buffers) const;

    // Basic blit (blits full screen, from given attachment to all destination draw buffers)
    void blit_to(const GLFramebuffer& other, GLint read_attachment = GL_COLOR_ATTACHMENT0) const;

    void resize(int width, int height);

    inline GLImage2D& operator[](GLuint attachment) {
        return m_attachments[attachment];
    }
    inline const GLImage2D& operator[](GLuint attachment) const {
        return m_attachments.at(attachment);
    }

    int m_width, m_height;

    std::unordered_map<GLint, GLImage2D> m_attachments;

    GLuint m_fbo;
};

}  // namespace volrend
