#include <string>
#include <cstdio>

#ifdef __EMSCRIPTEN__
// WebGL
#include <GLES3/gl3.h>
#else
#include <GL/glew.h>
#endif

namespace {

void check_compile_errors(GLuint shader, const std::string& type) {
    GLint success;
    GLchar infoLog[1024];
    if (type != "PROGRAM") {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 1024, NULL, infoLog);
            printf(
                "ERROR::SHADER_COMPILATION_ERROR of type: %s\n%s\n"
                "-- --------------------------------------------------- --\n",
                type.c_str(), infoLog);
            fflush(stdout);
        }
    } else {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shader, 1024, NULL, infoLog);
            printf(
                "ERROR::PROGRAM_LINKING_ERROR of type: %s\n%s\n"
                "-- --------------------------------------------------- --\n",
                type.c_str(), infoLog);
            fflush(stdout);
        }
    }
}

GLuint create_shader_program(std::string vert_shader_src,
                             std::string frag_shader_src) {
    // Auto-prepend the version
#ifdef __EMSCRIPTEN__
    const std::string version_str = "#version 300 es\n";
#else
    const std::string version_str = "#version 330\n";
#endif
    vert_shader_src = version_str + vert_shader_src;
    frag_shader_src = version_str + frag_shader_src;
    // Dummy vertex shader
    GLuint vert_shader = glCreateShader(GL_VERTEX_SHADER);
    const GLchar* vert_shader_src_ptr = vert_shader_src.c_str();
    glShaderSource(vert_shader, 1, &vert_shader_src_ptr, NULL);
    glCompileShader(vert_shader);
    check_compile_errors(vert_shader, "VERTEX");

    // Fragment shader
    GLuint frag_shader = glCreateShader(GL_FRAGMENT_SHADER);
    const GLchar* frag_shader_src_ptr = frag_shader_src.c_str();
    glShaderSource(frag_shader, 1, &frag_shader_src_ptr, NULL);
    glCompileShader(frag_shader);
    check_compile_errors(frag_shader, "FRAGMENT");

    GLuint program = glCreateProgram();
    glAttachShader(program, vert_shader);
    glAttachShader(program, frag_shader);
    glBindAttribLocation(program, 0, "aPos");
    glBindAttribLocation(program, 1, "aColor");
    glBindAttribLocation(program, 2, "aNormal");
    glBindAttribLocation(program, 0, "FragColor");
    glBindAttribLocation(program, 1, "Depth");
    glLinkProgram(program);
    check_compile_errors(program, "PROGRAM");

    glDeleteShader(vert_shader);
    glDeleteShader(frag_shader);

    glUseProgram(program);
    return program;
}

}  // namespace
