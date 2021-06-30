#include "volrend/mesh.hpp"

#include "volrend/common.hpp"

#ifdef __EMSCRIPTEN__
// WebGL
#include <GLES3/gl3.h>
#else
#include <GL/glew.h>
#endif

#include <numeric>
#include <sstream>
#include <fstream>
#include <cstdint>
#include <cstdio>
#include <map>
#include <cnpy.h>
#include "half.hpp"

#include <glm/gtc/type_ptr.hpp>
#include <tiny_obj_loader.h>
#include "volrend/internal/shader.hpp"

namespace {
const int VERT_SZ = 9;

GLenum get_gl_ele_type(int face_size) {
    switch (face_size) {
        case 1:
            return GL_POINTS;
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
scalar_t _norm(scalar_t* dir) {
    return sqrtf(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
}

template <typename scalar_t>
void _normalize(scalar_t* dir) {
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
    for (int i = 0; i < verts.size() / VERT_SZ; ++i) {
        for (int j = 0; j < 3; ++j) verts[i * VERT_SZ + 6 + j] = 0.f;
    }
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
    R"glsl(
uniform mat4x4 K;
uniform mat4x4 MV;
uniform mat4x4 M;

in vec3 aPos;
in vec3 aColor;
in vec3 aNormal;

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
    R"glsl(
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

// Split a string by '__'
std::vector<std::string> split_by_2underscore(const std::string& s) {
    std::vector<std::string> r;
    size_t j = 0;
    for (size_t i = 1; i < s.size(); ++i) {
        if (s[i] == '_' && s[i - 1] == '_') {
            if (i - 1 - j > 0) {
                r.push_back(s.substr(j, i - 1 - j));
            }
            j = i + 1;
        }
    }
    if (j < s.size()) {
        r.push_back(s.substr(j));
    }
    return r;
}

// Get int with default val from a NpyArray map
int map_get_int(const std::map<std::string, cnpy::NpyArray>& m,
                const std::string& key, const int defval, std::ostream& errs) {
    const auto it = m.find(key);
    if (it == m.end()) {
        return defval;
    } else {
        if (it->second.word_size == 1) {
            return *it->second.data<int8_t>();
        } else if (it->second.word_size == 2) {
            return *it->second.data<int16_t>();
        } else if (it->second.word_size == 4) {
            return *it->second.data<int32_t>();
        } else if (it->second.word_size == 8) {
            return (int)*it->second.data<int64_t>();
        }
        errs << "Invalid word size for int " << it->second.word_size << "\n";
        return 0;
    }
}

float map_get_float(const std::map<std::string, cnpy::NpyArray>& m,
                    const std::string& key, const float defval,
                    std::ostream& errs) {
    const auto it = m.find(key);
    if (it == m.end()) {
        return defval;
    } else {
        if (it->second.word_size == 2) {
            return *it->second.data<half>();
        } else if (it->second.word_size == 4) {
            return *it->second.data<float>();
        } else if (it->second.word_size == 8) {
            return (float)*it->second.data<double>();
        }
        errs << "Invalid word size for float " << it->second.word_size << "\n";
        return 0;
    }
}

glm::vec3 map_get_vec3(const std::map<std::string, cnpy::NpyArray>& m,
                       const std::string& key, const glm::vec3& defval,
                       std::ostream& errs) {
    const auto it = m.find(key);
    if (it == m.end()) {
        return defval;
    } else {
        glm::vec3 r;
        auto assn_ptr = [&](auto* ptr) {};
#define _ASSN_PTR_V3(dtype)                          \
    do {                                             \
        const dtype* ptr = it->second.data<dtype>(); \
        r[0] = (float)ptr[0];                        \
        r[1] = (float)ptr[1];                        \
        r[2] = (float)ptr[2];                        \
    } while (0)

        if (it->second.shape.size() != 1 || it->second.shape[0] != 3) {
            errs << "Invalid shape for float3, must be (3,)";
        }

        if (it->second.word_size == 2) {
            _ASSN_PTR_V3(half);
        } else if (it->second.word_size == 4) {
            _ASSN_PTR_V3(float);
        } else if (it->second.word_size == 8) {
            _ASSN_PTR_V3(double);
        } else {
            errs << "Invalid word size for float " << it->second.word_size
                 << "\n";
        }
#undef _ASSN_PTR_V3
        return r;
    }
}

std::vector<float> map_get_floatarr(
    const std::map<std::string, cnpy::NpyArray>& m, const std::string& key,
    std::ostream& errs) {
    const auto it = m.find(key);
    std::vector<float> result;
    if (it == m.end()) {
        return result;
    }

#define _ASSN_PTR_ARR(dtype)                                       \
    do {                                                           \
        const dtype* ptr = it->second.data<dtype>();               \
        std::copy(ptr, ptr + it->second.num_vals, result.begin()); \
    } while (0)

    result.resize(it->second.num_vals);
    if (it->second.word_size == 2) {
        _ASSN_PTR_ARR(half);
    } else if (it->second.word_size == 4) {
        _ASSN_PTR_ARR(float);
    } else if (it->second.word_size == 8) {
        _ASSN_PTR_ARR(double);
    } else {
        errs << "Invalid word size for float " << it->second.word_size << "\n";
    }
#undef _ASSN_PTR_ARR
    return result;
}

std::vector<int> map_get_intarr(const std::map<std::string, cnpy::NpyArray>& m,
                                const std::string& key, std::ostream& errs) {
    const auto it = m.find(key);
    std::vector<int> result;
    if (it == m.end()) {
        return result;
    }

#define _ASSN_PTR_ARR(dtype)                                       \
    do {                                                           \
        const dtype* ptr = it->second.data<dtype>();               \
        std::copy(ptr, ptr + it->second.num_vals, result.begin()); \
    } while (0)

    result.resize(it->second.num_vals);
    if (it->second.word_size == 1) {
        _ASSN_PTR_ARR(int8_t);
    } else if (it->second.word_size == 2) {
        _ASSN_PTR_ARR(int16_t);
    } else if (it->second.word_size == 4) {
        _ASSN_PTR_ARR(int32_t);
    } else if (it->second.word_size == 8) {
        _ASSN_PTR_ARR(int64_t);
    } else {
        errs << "Invalid word size for int " << it->second.word_size << "\n";
    }
#undef _ASSN_PTR_ARR
    return result;
}
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

void Mesh::use_shader() { glUseProgram(program); }

void Mesh::draw(const glm::mat4x4& V, glm::mat4x4 K, bool y_up) const {
    if (!visible) return;
    float norm = glm::length(rotation);
    if (norm < 1e-3) {
        transform_ = glm::mat4(1.0);
    } else {
        glm::quat rot = glm::angleAxis(norm, rotation / norm);
        transform_ = glm::mat4_cast(rot);
    }
    transform_ *= scale;
    glm::vec3 cam_pos = -glm::transpose(glm::mat3x3(V)) * glm::vec3(V[3]);
    if (!y_up) {
        K[1][1] *= -1.0;
    }

    transform_[3] = glm::vec4(translation, 1);
    glm::mat4x4 MV = V * transform_;
    glUniformMatrix4fv(u_MV, 1, GL_FALSE, glm::value_ptr(MV));
    glUniformMatrix4fv(u_M, 1, GL_FALSE, glm::value_ptr(transform_));
    glUniformMatrix4fv(u_K, 1, GL_FALSE, glm::value_ptr(K));
    glUniform3fv(u_cam_pos, 1, glm::value_ptr(cam_pos));
    glUniform1i(u_unlit, unlit);
    glBindVertexArray(vao_);
    if (faces.empty()) {
        glDrawArrays(get_gl_ele_type(face_size), 0, vert.size() / VERT_SZ);
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

Mesh Mesh::Lattice(int reso, glm::vec3 color) {
    Mesh m(reso * reso * reso, 0, 1);
    float* vptr = m.vert.data();
    for (int i = 0; i < reso; i++) {
        const float x = (i + 0.5f) / reso;
        for (int j = 0; j < reso; j++) {
            const float y = (j + 0.5f) / reso;
            for (int k = 0; k < reso; k++) {
                const float z = (k + 0.5f) / reso;
                vptr[0] = x;
                vptr[1] = y;
                vptr[2] = z;
                // Color
                vptr[3] = color[0];
                vptr[4] = color[1];
                vptr[5] = color[2];
                // Normal
                vptr[6] = 1;
                vptr[7] = 0;
                vptr[8] = 0;
                vptr += VERT_SZ;
            }
        }
    }
    m.name = "Lattice";
    m.unlit = true;
    return m;
}

Mesh Mesh::CameraFrustum(float focal_length, float image_width,
                         float image_height, float z, glm::vec3 color) {
    float invf = 1.f / focal_length;
    float halfw = image_width * 0.5f, halfh = image_height * 0.5f;
    Mesh m(5, 8, 2);
    // clang-format off
    m.vert = {
        0.f, 0.f, 0.f, color[0], color[1], color[2],
            0.f, 0.f, 1.f,
        z * -halfw * invf, z * -halfh * invf, z, color[0], color[1], color[2],
            0.f, 0.f, 1.f,
        z * -halfw * invf, z * halfh * invf, z, color[0], color[1], color[2],
            0.f, 0.f, 1.f,
        z * halfw * invf, z * halfh * invf, z, color[0], color[1], color[2],
            0.f, 0.f, 1.f,
        z * halfw * invf, z * -halfh * invf, z, color[0], color[1], color[2],
            0.f, 0.f, 1.f,
    };
    m.faces = {
        0, 1,
        0, 2,
        0, 3,
        0, 4,
        1, 2,
        2, 3,
        3, 4,
        4, 1,
    };
    // clang-format on
    m.name = "CameraFrustum";
    m.unlit = true;
    return m;
}

Mesh Mesh::Line(glm::vec3 a, glm::vec3 b, glm::vec3 color) {
    volrend::Mesh m(2, 1, 2);
    // clang-format off
    m.vert = {
        a[0], a[1], a[2], color[0], color[1], color[2], 0.f, 0.f, 1.f,
        b[0], b[1], b[2], color[0], color[1], color[2], 0.f, 0.f, 1.f,
    };
    // clang-format on
    m.faces = {0, 1};
    m.name = "Line";
    m.unlit = true;
    return m;
}

Mesh Mesh::Lines(std::vector<float> points, glm::vec3 color) {
    if (points.size() % 3 != 0) {
        printf("Lines: Number of elements in points must be divisible by 3\n");
    }
    const int n_points = (int)points.size() / 3;
    volrend::Mesh m(n_points, n_points - 1, 2);
    float* vptr = m.vert.data();
    float* pptr = points.data();
    for (int i = 0; i < n_points; ++i) {
        vptr[0] = pptr[0];
        vptr[1] = pptr[1];
        vptr[2] = pptr[2];
        vptr[3] = color[0];
        vptr[4] = color[1];
        vptr[5] = color[2];
        vptr[6] = 0.f;
        vptr[7] = 0.f;
        vptr[8] = 1.f;
        vptr += VERT_SZ;
        pptr += 3;
    }
    unsigned int* fptr = m.faces.data();
    for (int i = 0; i < n_points - 1; ++i) {
        fptr[0] = i;
        fptr[1] = i + 1;
        fptr += 2;
    }
    m.name = "Lines";
    m.unlit = true;
    return m;
}

Mesh Mesh::Points(std::vector<float> points, glm::vec3 color) {
    if (points.size() % 3 != 0) {
        printf("Points: Number of elements in points must be divisible by 3\n");
    }
    const int n_points = (int)points.size() / 3;
    volrend::Mesh m(n_points, 0, 1);
    float* vptr = m.vert.data();
    float* pptr = points.data();
    for (int i = 0; i < n_points; ++i) {
        vptr[0] = pptr[0];
        vptr[1] = pptr[1];
        vptr[2] = pptr[2];
        vptr[3] = color[0];
        vptr[4] = color[1];
        vptr[5] = color[2];
        vptr[6] = 0.f;
        vptr[7] = 0.f;
        vptr[8] = 1.f;
        vptr += VERT_SZ;
        pptr += 3;
    }
    m.name = "Points";
    m.unlit = true;
    return m;
}

void Mesh::auto_faces() {
    faces.resize(vert.size() / VERT_SZ);
    std::iota(faces.begin(), faces.end(), 0);
}

void Mesh::repeat(int n) {
    if (n < 1) return;
    const size_t vert_size = vert.size();
    const size_t n_verts = vert_size / VERT_SZ;
    const size_t faces_size = faces.size();
    vert.resize(vert.size() * n);
    faces.resize(faces.size() * n);
    for (int i = 1; i < n; ++i) {
        std::copy(vert.begin(), vert.begin() + vert_size,
                  vert.begin() + i * vert_size);
        std::copy(faces.begin(), faces.begin() + faces_size,
                  faces.begin() + i * faces_size);
        auto* fptr = &faces[i * faces_size];
        for (size_t j = 0; j < faces_size; ++j) {
            fptr[j] += i * n_verts;
        }
    }
}

void Mesh::apply_transform(glm::vec3 r, glm::vec3 t, int start, int end) {
    glm::mat4 transform;
    float norm = glm::length(r);
    if (norm < 1e-3) {
        transform = glm::mat4(1.0);
    } else {
        glm::quat rot = glm::angleAxis(norm, r / norm);
        transform = glm::mat4_cast(rot);
    }
    transform[3] = glm::vec4(t, 1);
    apply_transform(transform, start, end);
}

void Mesh::apply_transform(glm::mat4 transform, int start, int end) {
    if (end == -1) {
        end = vert.size() / VERT_SZ;
    }
    auto* ptr = &vert[start * VERT_SZ];
    for (int i = start; i < end; ++i) {
        glm::vec4 v(ptr[0], ptr[1], ptr[2], 1.0);
        v = transform * v;
        ptr[0] = v[0];
        ptr[1] = v[1];
        ptr[2] = v[2];
        ptr += VERT_SZ;
    }
}

Mesh _load_basic_obj(const std::string& path_or_string, bool from_string) {
    Mesh mesh;
    tinyobj::ObjReaderConfig reader_config;
    if (!from_string) {
        reader_config.mtl_search_path = "./";  // Path to material files
    }
    reader_config.triangulate = true;
    reader_config.vertex_color = true;
    tinyobj::ObjReader reader;

    bool read_success;
    if (from_string)
        read_success =
            reader.ParseFromString(path_or_string, "", reader_config);
    else
        read_success = reader.ParseFromFile(path_or_string, reader_config);
    if (!read_success) {
        if (!reader.Error().empty()) {
            printf("ERROR Failed to load OBJ: %s", reader.Error().c_str());
        }
        return mesh;
    }
    // if (!reader.Warning().empty()) {
    //     printf("TinyObjReader: %s\n", reader.Warning().c_str());
    // }

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();

    const size_t n_verts = attrib.vertices.size() / 3;
    mesh.vert.resize(VERT_SZ * n_verts);
    for (size_t i = 0; i < n_verts; i++) {
        auto* ptr = &mesh.vert[i * VERT_SZ];
        for (int j = 0; j < 3; ++j) {
            ptr[j] = attrib.vertices[i * 3 + j];
        }
    }
    for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);
            if (fv == 3) {
                for (size_t v = 0; v < fv; v++) {
                    tinyobj::index_t idx =
                        shapes[s].mesh.indices[index_offset + v];
                    mesh.faces.push_back(idx.vertex_index);
                }
            }
            index_offset += fv;
        }
    }

    if (attrib.colors.size() / 3 >= n_verts) {
        for (int i = 0; i < n_verts; ++i) {
            auto* color_ptr = &mesh.vert[i * VERT_SZ + 3];
            for (int j = 0; j < 3; ++j) {
                color_ptr[j] = attrib.colors[3 * i + j];
            }
        }
    }
    if (attrib.normals.size() / 3 >= n_verts) {
        for (size_t i = 0; i < n_verts; i++) {
            auto* normal_ptr = &mesh.vert[i * VERT_SZ + 6];
            for (int j = 0; j < 3; ++j) {
                normal_ptr[j] = attrib.normals[i * 3 + j];
            }
        }
    } else {
        estimate_normals(mesh.vert, mesh.faces);
    }

    mesh.face_size = 3;
    if (from_string) {
        mesh.name = "OBJ";
    } else {
        mesh.name = path_or_string;
    }
    return mesh;
}

Mesh Mesh::load_basic_obj(const std::string& path) {
    return _load_basic_obj(path, false);
}

Mesh Mesh::load_mem_basic_obj(const std::string& str) {
    return _load_basic_obj(str, true);
}

namespace {
std::vector<Mesh> _load_npz(const cnpy::npz_t& npz, bool default_visible) {
    printf("INFO: Loading drawlist npz\n");
    std::map<std::string,
             std::pair<std::string /*type*/,
                       std::map<std::string, cnpy::NpyArray> /*fields*/>>
        mesh_parse_map;

    for (const std::pair<std::string, cnpy::NpyArray>& kv : npz) {
        const std::string& fullname = kv.first;
        std::vector<std::string> spl = split_by_2underscore(fullname);
        if (spl.size() == 1) {
            // Mesh type
            std::string meshtype(kv.second.data_holder.begin(),
                                 kv.second.data_holder.end());
            for (size_t i = 4; i < meshtype.size(); i += 4)
                meshtype[i / 4] = std::tolower(meshtype[i]);
            meshtype.resize(meshtype.size() / 4);
            mesh_parse_map[spl[0]].first = meshtype;
        } else if (spl.size() == 2) {
            // Field
            mesh_parse_map[spl[0]].second[spl[1]] = kv.second;
        } else
            printf(
                "Mesh load_npz warning: invalid field '%s"
                "', must be of the form <name>=mesh_type or "
                "<name>__<field>=val\n",
                fullname.c_str());
        continue;
    }

    std::vector<Mesh> meshes;
    std::stringstream errs;
    const glm::vec3 DEFAULT_COLOR{1.f, 0.5f, 0.2f};
    for (const auto& kv : mesh_parse_map) {
        const std::string& mesh_name = kv.first;
        const std::string& mesh_type = kv.second.first;
        const std::map<std::string, cnpy::NpyArray>& fields = kv.second.second;

        volrend::Mesh me;
        glm::vec3 color = map_get_vec3(fields, "color", DEFAULT_COLOR, errs);
        if (mesh_type == "cube") {
            me = volrend::Mesh::Cube(color);
        } else if (mesh_type == "sphere") {
            auto rings = map_get_int(fields, "rings", 15, errs);
            auto sectors = map_get_int(fields, "sectors", 30, errs);
            me = volrend::Mesh::Sphere(rings, sectors, color);
        } else if (mesh_type == "line") {
            auto a = map_get_vec3(fields, "a", glm::vec3(0.f, 0.f, 0.f), errs);
            auto b = map_get_vec3(fields, "b", glm::vec3(0.f, 0.f, 1.f), errs);
            me = volrend::Mesh::Line(a, b, color);
        } else if (mesh_type == "camerafrustum") {
            auto focal_length =
                map_get_float(fields, "focal_length", 1111.0f, errs);
            auto image_width =
                map_get_float(fields, "image_width", 800.0f, errs);
            auto image_height =
                map_get_float(fields, "image_height", 800.0f, errs);
            auto z = map_get_float(fields, "z", -0.3f, errs);
            me = volrend::Mesh::CameraFrustum(focal_length, image_width,
                                              image_height, z, color);
            if (fields.count("t")) {
                auto t = map_get_floatarr(fields, "t", errs);
                auto r = map_get_floatarr(fields, "r", errs);
                if (r.size() != t.size() || r.size() % 3) {
                    errs << "camerafrustums r, t have different sizes or "
                            "not "
                            "multiple of 3\n";
                }
                const size_t n_verts = me.vert.size() / VERT_SZ;
                const size_t n_reps = t.size() / 3;
                me.repeat(n_reps);
                for (int i = 0; i < n_reps; ++i) {
                    const int j = i * 3;
                    glm::vec3 ri{r[j], r[j + 1], r[j + 2]};
                    glm::vec3 ti{t[j], t[j + 1], t[j + 2]};
                    me.apply_transform(ri, ti, n_verts * i, n_verts * (i + 1));
                }
                bool connect = map_get_int(fields, "connect", 0, errs) != 0;
                if (connect) {
                    // Connect camera centers in a trajectory
                    const size_t start_idx = me.faces.size();
                    me.faces.resize(start_idx + (n_reps - 1) * 2);
                    for (int i = 0; i < n_reps - 1; ++i) {
                        me.faces[start_idx + i * 2] = n_verts * i;
                        me.faces[start_idx + i * 2 + 1] = n_verts * (i + 1);
                    }
                }
            }
        } else if (mesh_type == "lines") {
            // Lines
            auto data = map_get_floatarr(fields, "points", errs);
            me = volrend::Mesh::Lines(data, color);
            if (fields.count("segs")) {
                // By default, the points are connected in a single line
                // i -> i+1 etc
                // specify this to connect every consecutive pair of indices
                // 0a 0b 1a 1b 2a 2b ...
                auto lines = map_get_intarr(fields, "segs", errs);
                me.faces.resize(lines.size());
                std::copy(lines.begin(), lines.end(), me.faces.begin());
            }
        } else if (mesh_type == "points") {
            // Point cloud
            auto data = map_get_floatarr(fields, "points", errs);
            me = volrend::Mesh::Points(data, color);
        } else if (mesh_type == "mesh") {
            // Most generic mesh
            auto data = map_get_floatarr(fields, "points", errs);
            me = volrend::Mesh::Points(data, color);
            // Face_size = 1: points  2: lines  3: triangles
            me.face_size = map_get_int(fields, "face_size", 3, errs);
            if (me.face_size <= 0 || me.face_size > 3) {
                me.face_size = 3;
                errs << "Mesh face size must be one of 1,2,3\n";
            }
            if (fields.count("faces")) {
                auto faces = map_get_intarr(fields, "faces", errs);
                if (faces.size() % me.face_size) {
                    errs << "Faces must have face_size=" << me.face_size
                         << " elements\n";
                }
                me.faces.resize(faces.size());
                std::copy(faces.begin(), faces.end(), me.faces.begin());
            }
            if (me.face_size == 3) {
                estimate_normals(me.vert, me.faces);
            }
        } else {
            errs << "Mesh '" << mesh_name << "' has unsupported type '"
                 << mesh_type << "'\n";
            continue;
        }
        if (fields.count("vert_color")) {
            // Support manual vertex colors
            auto vert_color = map_get_floatarr(fields, "vert_color", errs);
            if (vert_color.size() * VERT_SZ != me.vert.size() * 3) {
                errs << "Mesh " << mesh_name
                     << " vert_color has invalid size\n";
                continue;
            }
            const float* in_ptr = vert_color.data();
            float* out_ptr = me.vert.data() + 3;
            for (int i = 0; i < vert_color.size(); i += 3) {
                for (int j = 0; j < 3; ++j) {
                    out_ptr[j] = in_ptr[j];
                }
                in_ptr += 3;
                out_ptr += VERT_SZ;
            }
        }
        me.name = mesh_name;
        me.scale = map_get_float(fields, "scale", 1.0f, errs);
        me.translation =
            map_get_vec3(fields, "translation", glm::vec3{0.f, 0.f, 0.f}, errs);
        me.rotation =
            map_get_vec3(fields, "rotation", glm::vec3{0.f, 0.f, 0.f}, errs);
        me.visible = map_get_int(fields, "visible", default_visible, errs) != 0;
        me.unlit = map_get_int(fields, "unlit", 0, errs) != 0;
        me.update();
        meshes.push_back(std::move(me));
    }
    std::string errstr = errs.str();
    if (errstr.size()) {
        printf("Mesh load_npz encountered errors while parsing:\n%s",
               errstr.c_str());
    }

    return meshes;
}
}  // namespace

std::vector<Mesh> Mesh::open_drawlist(const std::string& path,
                                      bool default_visible) {
    auto npz = cnpy::npz_load(path);
    return _load_npz(npz, default_visible);
}

std::vector<Mesh> Mesh::open_drawlist_mem(const char* data, uint64_t size,
                                          bool default_visible) {
    auto npz = cnpy::npz_load_mem(data, size);
    return _load_npz(npz, default_visible);
}

}  // namespace volrend
