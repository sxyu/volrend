#include <string>
#include <iostream>

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

GLuint create_shader_program(const char* vert_shader_src,
                             const char* frag_shader_src) {
    // Dummy vertex shader
    GLuint vert_shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vert_shader, 1, &vert_shader_src, NULL);
    glCompileShader(vert_shader);
    check_compile_errors(vert_shader, "VERTEX");

    // Fragment shader
    GLuint frag_shader = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(frag_shader, 1, &frag_shader_src, NULL);
    glCompileShader(frag_shader);
    check_compile_errors(frag_shader, "FRAGMENT");

    GLuint program = glCreateProgram();
    glAttachShader(program, vert_shader);
    glAttachShader(program, frag_shader);
    glLinkProgram(program);
    check_compile_errors(program, "PROGRAM");

    glDeleteShader(vert_shader);
    glDeleteShader(frag_shader);

    glUseProgram(program);
    return program;
}

}  // namespace
