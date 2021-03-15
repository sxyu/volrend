#include "volrend/mesh.hpp"

#include "volrend/common.hpp"

#ifdef VOLREND_CUDA

#ifdef __EMSCRIPTEN__
// WebGL
#include <GLES3/gl3.h>
#else
#include <GL/glew.h>
#endif

#include <numeric>
#include <sstream>
#include <fstream>

#include <glm/gtc/type_ptr.hpp>
#include "volrend/internal/shader.hpp"

namespace {
const int VERT_SZ = 9;

GLenum get_gl_ele_type(int face_size) {
    switch (face_size) {
        case 2:
            return GL_LINES;
        case 3:
            return GL_TRIANGLES;
        default:
            throw std::invalid_argument("Unsupported mesh face size");
    }
}

template <class scalar_t>
void _cross3(const scalar_t* a, const scalar_t* b, scalar_t* out) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

template <typename scalar_t>
static scalar_t _norm(scalar_t* __restrict__ dir) {
    return sqrtf(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
}

template <typename scalar_t>
static void _normalize(scalar_t* __restrict__ dir) {
    scalar_t norm = _norm(dir);
    if (norm > 1e-6) {
        dir[0] /= norm;
        dir[1] /= norm;
        dir[2] /= norm;
    }
}

void estimate_normals(std::vector<float>& verts,
                      const std::vector<unsigned int>& faces) {
    const int n_faces =
        faces.size() ? faces.size() / 3 : verts.size() / VERT_SZ / 3;
    float a[3], b[3], cross[3], off[3];
    for (int i = 0; i < n_faces; ++i) {
        if (faces.size()) {
            off[0] = faces[3 * i] * VERT_SZ;
            off[1] = faces[3 * i + 1] * VERT_SZ;
            off[2] = faces[3 * i + 2] * VERT_SZ;
        } else {
            off[0] = i * VERT_SZ * 3;
            off[1] = off[0] + VERT_SZ;
            off[2] = off[1] + VERT_SZ;
        }

        for (int j = 0; j < 3; ++j) {
            a[j] = verts[off[1] + j] - verts[off[0] + j];
            b[j] = verts[off[2] + j] - verts[off[0] + j];
        }
        _cross3(a, b, cross);
        for (int j = 0; j < 3; ++j) {
            float* ptr = &verts[off[j] + 6];
            for (int k = 0; k < 3; ++k) {
                ptr[k] += cross[k];
            }
        }
    }
    for (int i = 0; i < verts.size() / VERT_SZ; ++i) {
        _normalize(&verts[i * VERT_SZ + 6]);
    }
}

const char* VERT_SHADER_SRC =
    R"glsl(#version 300 es
uniform mat4x4 K;
uniform mat4x4 MV;
uniform mat4x4 M;

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec3 aNormal;

out lowp vec3 VertColor;
out highp vec4 FragPos;
out highp vec3 Normal;

void main()
{
    FragPos = MV * vec4(aPos.x, aPos.y, aPos.z, 1.0);
    gl_Position = K * FragPos;
    VertColor = aColor;
    Normal = normalize(mat3x3(M) * aNormal);
}
)glsl";

const char* FRAG_SHADER_SRC =
    R"glsl(#version 300 es
precision highp float;
in lowp vec3 VertColor;
in vec4 FragPos;
in vec3 Normal;

uniform bool unlit;
uniform vec3 camPos;

layout(location = 0) out lowp vec4 FragColor;
layout(location = 1) out float Depth;

void main()
{
    if (unlit) {
        FragColor = vec4(VertColor, 1);
    } else {
        // FIXME make these uniforms, whatever for now
        float ambient = 0.3;
        float specularStrength = 0.6;
        float diffuseStrength = 0.7;
        float diffuse2Strength = 0.2;
        vec3 lightDir = normalize(vec3(0.5, 0.2, 1));
        vec3 lightDir2 = normalize(vec3(-0.5, -1.0, -0.5));

        float diffuse = diffuseStrength * max(dot(lightDir, Normal), 0.0);
        float diffuse2 = diffuse2Strength * max(dot(lightDir2, Normal), 0.0);

        vec3 viewDir = normalize(camPos - vec3(FragPos));
        vec3 reflectDir = reflect(-lightDir, Normal);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
        float specular = specularStrength * spec;

        FragColor = (ambient + diffuse + diffuse2 + specular) * vec4(VertColor, 1);
    }

    Depth = length(FragPos.xyz);
}
)glsl";

unsigned int program = -1;
unsigned int u_K, u_MV, u_M, u_cam_pos, u_unlit;

}  // namespace

namespace volrend {

Mesh::Mesh(int n_verts, int n_faces, int face_size, bool unlit)
    : vert(n_verts * 9),
      faces(n_faces * face_size),
      rotation(0),
      translation(0),
      face_size(face_size),
      unlit(unlit) {
    if (program == -1) {
        program = create_shader_program(VERT_SHADER_SRC, FRAG_SHADER_SRC);
        u_MV = glGetUniformLocation(program, "MV");
        u_M = glGetUniformLocation(program, "M");
        u_K = glGetUniformLocation(program, "K");
        u_cam_pos = glGetUniformLocation(program, "camPos");
        u_unlit = glGetUniformLocation(program, "unlit");
    }
}

void Mesh::update() {
    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_);
    glGenBuffers(1, &ebo_);

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, vert.size() * sizeof(vert[0]), vert.data(),
                 GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, VERT_SZ * sizeof(float),
                          (void*)0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, VERT_SZ * sizeof(float),
                          (void*)(3 * sizeof(float)));
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, VERT_SZ * sizeof(float),
                          (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.size() * sizeof(faces[0]),
                 faces.data(), GL_STATIC_DRAW);
    glBindVertexArray(0);
}

void Mesh::draw(const glm::mat4x4& V, const glm::mat4x4& K) const {
    if (!visible) return;
    float norm = glm::length(rotation);
    glm::mat4 transform;
    if (norm < 1e-3) {
        transform = glm::mat4(1.0);
    } else {
        glm::quat rot = glm::angleAxis(norm, rotation / norm);
        transform = glm::mat4_cast(rot);
    }
    transform *= scale;
    glm::vec3 cam_pos = -glm::transpose(glm::mat3x3(V)) * glm::vec3(V[3]);

    transform[3] = glm::vec4(translation, 1);
    glm::mat4x4 MV = V * transform;
    glUniformMatrix4fv(u_MV, 1, GL_FALSE, glm::value_ptr(MV));
    glUniformMatrix4fv(u_M, 1, GL_FALSE, glm::value_ptr(transform));
    glUniformMatrix4fv(u_K, 1, GL_FALSE, glm::value_ptr(K));
    glUniform3fv(u_cam_pos, 1, glm::value_ptr(cam_pos));
    glUniform1i(u_unlit, unlit);
    glBindVertexArray(vao_);
    if (faces.empty()) {
        glDrawArrays(get_gl_ele_type(face_size), 0, vert.size() / face_size);
    } else {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
        glDrawElements(get_gl_ele_type(face_size), faces.size(),
                       GL_UNSIGNED_INT, (void*)0);
    }
    glBindVertexArray(0);
}

Mesh Mesh::Cube(glm::vec3 color) {
    Mesh m(36, 0, 3);
    // clang-format off
    m.vert = {
        -0.5, -0.5, -0.5,  color.x, color.y, color.z,  0.0f, 0.0f, -1.0f,
         0.5, 0.5, -0.5,   color.x, color.y, color.z,  0.0f, 0.0f, -1.0f,
         0.5, -0.5, -0.5,  color.x, color.y, color.z,  0.0f, 0.0f, -1.0f,
         0.5, 0.5, -0.5,   color.x, color.y, color.z,  0.0f, 0.0f, -1.0f,
        -0.5, -0.5, -0.5,  color.x, color.y, color.z,  0.0f, 0.0f, -1.0f,
        -0.5, 0.5, -0.5,   color.x, color.y, color.z,  0.0f, 0.0f, -1.0f,

        -0.5, -0.5, 0.5,  color.x, color.y, color.z,  0.0f, 0.0f, 1.0f,
         0.5, 0.5, 0.5,   color.x, color.y, color.z,  0.0f, 0.0f, 1.0f,
         0.5, -0.5, 0.5,  color.x, color.y, color.z,  0.0f, 0.0f, 1.0f,
         0.5, 0.5, 0.5,   color.x, color.y, color.z,  0.0f, 0.0f, 1.0f,
        -0.5, -0.5, 0.5,  color.x, color.y, color.z,  0.0f, 0.0f, 1.0f,
        -0.5, 0.5, 0.5,   color.x, color.y, color.z,  0.0f, 0.0f, 1.0f,

        -0.5, -0.5, -0.5,  color.x, color.y, color.z,  0.0f, -1.0f, 0.0f,
         0.5, -0.5, 0.5,   color.x, color.y, color.z,  0.0f, -1.0f, 0.0f,
         0.5, -0.5, -0.5,  color.x, color.y, color.z,  0.0f, -1.0f, 0.0f,
         0.5, -0.5, 0.5,   color.x, color.y, color.z,  0.0f, -1.0f, 0.0f,
        -0.5, -0.5, -0.5,  color.x, color.y, color.z,  0.0f, -1.0f, 0.0f,
        -0.5, -0.5, 0.5,   color.x, color.y, color.z,  0.0f, -1.0f, 0.0f,

        -0.5, 0.5, -0.5,  color.x, color.y, color.z,  0.0f, 1.0f, 0.0f,
         0.5, 0.5, 0.5,   color.x, color.y, color.z,  0.0f, 1.0f, 0.0f,
         0.5, 0.5, -0.5,  color.x, color.y, color.z,  0.0f, 1.0f, 0.0f,
         0.5, 0.5, 0.5,   color.x, color.y, color.z,  0.0f, 1.0f, 0.0f,
        -0.5, 0.5, -0.5,  color.x, color.y, color.z,  0.0f, 1.0f, 0.0f,
        -0.5, 0.5, 0.5,   color.x, color.y, color.z,  0.0f, 1.0f, 0.0f,

        -0.5, -0.5, -0.5,  color.x, color.y, color.z,  -1.0f, 0.0f, 0.0f,
        -0.5, 0.5, 0.5,    color.x, color.y, color.z,  -1.0f, 0.0f, 0.0f,
        -0.5, -0.5, 0.5,   color.x, color.y, color.z,  -1.0f, 0.0f, 0.0f,
        -0.5, 0.5, 0.5,    color.x, color.y, color.z,  -1.0f, 0.0f, 0.0f,
        -0.5, -0.5, -0.5,  color.x, color.y, color.z,  -1.0f, 0.0f, 0.0f,
        -0.5, 0.5, -0.5,   color.x, color.y, color.z,  -1.0f, 0.0f, 0.0f,

        0.5, -0.5, -0.5,  color.x, color.y, color.z,  1.0f, 0.0f, 0.0f,
        0.5, 0.5, 0.5,    color.x, color.y, color.z,  1.0f, 0.0f, 0.0f,
        0.5, -0.5, 0.5,   color.x, color.y, color.z,  1.0f, 0.0f, 0.0f,
        0.5, 0.5, 0.5,    color.x, color.y, color.z,  1.0f, 0.0f, 0.0f,
        0.5, -0.5, -0.5,  color.x, color.y, color.z,  1.0f, 0.0f, 0.0f,
        0.5, 0.5, -0.5,   color.x, color.y, color.z,  1.0f, 0.0f, 0.0f,
    };
    // clang-format on

    m.name = "Cube";
    return m;
}

Mesh Mesh::Sphere(int rings, int sectors, glm::vec3 color) {
    Mesh m(rings * sectors, (rings - 1) * sectors * 2, 3);
    const float R = M_PI / (float)(rings - 1);
    const float S = 2 * M_PI / (float)sectors;
    float* vptr = m.vert.data();
    for (int r = 0; r < rings; r++) {
        for (int s = 0; s < sectors; s++) {
            const float z = sin(-0.5f * M_PI + r * R);
            const float x = cos(s * S) * sin(r * R);
            const float y = sin(s * S) * sin(r * R);
            vptr[0] = x;
            vptr[1] = y;
            vptr[2] = z;
            // Color
            vptr[3] = color[0];
            vptr[4] = color[1];
            vptr[5] = color[2];
            // Normal
            vptr[6] = x;
            vptr[7] = y;
            vptr[8] = z;
            vptr += VERT_SZ;
        }
    }
    unsigned int* ptr = m.faces.data();
    for (int r = 0; r < rings - 1; r++) {
        const int nx_r = r + 1;
        for (int s = 0; s < sectors; s++) {
            const int nx_s = (s + 1) % sectors;
            ptr[0] = r * sectors + nx_s;
            ptr[1] = r * sectors + s;
            ptr[2] = nx_r * sectors + s;
            ptr[3] = nx_r * sectors + s;
            ptr[4] = nx_r * sectors + nx_s;
            ptr[5] = r * sectors + nx_s;
            ptr += 6;
        }
    }
    m.name = "Sphere";
    return m;
}

void Mesh::load_basic_obj(const std::string& path) {
    std::ifstream ifs(path);
    std::string line;
    std::vector<float> tmp_pos, tmp_rgb;
    std::vector<unsigned int> tmp_faces;
    size_t attrs_per_vert = 0;
    while (std::getline(ifs, line)) {
        if (line.size() < 2 || line[1] != ' ') continue;
        if (line[0] == 'v') {
            std::stringstream ss(line.substr(2));
            double tmp;
            size_t cnt = 0;
            while (ss >> tmp) {
                if (cnt >= 3) {
                    tmp_rgb.push_back(tmp);
                } else {
                    tmp_pos.push_back(tmp);
                }
                ++cnt;
            }
            if (attrs_per_vert) {
            } else {
                attrs_per_vert = cnt;
            }
        } else if (line[0] == 'f') {
            size_t i = 2, j = 0;
            while (i < line.size() && std::isspace(i)) ++i;
            std::string num;
            for (; i < line.size(); ++i) {
                if (line[i] == '/') {
                    ++j;
                } else if (std::isspace(line[i])) {
                    tmp_faces.push_back((unsigned int)std::atoll(num.c_str()) -
                                        1);
                    j = 0;
                    num.clear();
                } else if (j == 0) {
                    num.push_back(line[i]);
                }
            }
            if (num.size()) {
                tmp_faces.push_back((unsigned int)std::atoll(num.c_str()) - 1);
            }
        }
    }
    if (tmp_pos.size() % 3 != 0) {
        std::cerr << "BAD or unsupported OBJ\n";
        return;
    }
    size_t n_verts = tmp_pos.size() / 3;
    vert.resize(n_verts * VERT_SZ);

    for (int i = 0; i < n_verts; ++i) {
        const auto* vert_in_ptr = &tmp_pos[i * 3];
        auto* vert_ptr = &vert[i * VERT_SZ];
        for (int j = 0; j < 3; ++j) {
            vert_ptr[j] = vert_in_ptr[j];
        }
    }
    if (tmp_rgb.size()) {
        for (int i = 0; i < n_verts; ++i) {
            const auto* color_in_ptr = &tmp_rgb[i * 3];
            auto* color_ptr = &vert[i * VERT_SZ + 3];
            for (int j = 0; j < 3; ++j) {
                color_ptr[j] = color_in_ptr[j];
            }
        }
    } else {
        for (int i = 0; i < n_verts; ++i) {
            auto* color_ptr = &vert[i * VERT_SZ + 3];
            for (int j = 0; j < 3; ++j) {
                color_ptr[j] = 0.8f;
            }
        }
    }
    if (tmp_faces.size()) {
        std::swap(faces, tmp_faces);
    } else {
        faces.clear();
    }

    // FIXME load normals from obj if available
    estimate_normals(vert, faces);
    face_size = 3;

    name = path;
}

void Mesh::auto_faces() {
    faces.resize(vert.size());
    std::iota(faces.begin(), faces.end(), 0);
}

}  // namespace volrend

#endif
