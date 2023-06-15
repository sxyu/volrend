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

#include "volrend/internal/glutil.hpp"
#include "volrend/internal/basic_mesh.shader"
#include "volrend/internal/basic_image.shader"

namespace volrend {
namespace {

GLShader g_mesh_program;
GLShader g_image_program;

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

// Mesh normal estimation
void estimate_normals(std::vector<float>& verts,
                      const std::vector<unsigned int>& faces,
                      int mesh_vert_size) {
    const int n_faces =
        faces.size() ? faces.size() / 3 : verts.size() / mesh_vert_size / 3;
    float a[3], b[3], cross[3];
    unsigned off[3];
    for (int i = 0; i < verts.size() / mesh_vert_size; ++i) {
        for (int j = 0; j < 3; ++j) verts[i * mesh_vert_size + 6 + j] = 0.f;
    }
    for (int i = 0; i < n_faces; ++i) {
        if (faces.size()) {
            off[0] = faces[3 * i] * mesh_vert_size;
            off[1] = faces[3 * i + 1] * mesh_vert_size;
            off[2] = faces[3 * i + 2] * mesh_vert_size;
        } else {
            off[0] = i * mesh_vert_size * 3;
            off[1] = off[0] + mesh_vert_size;
            off[2] = off[1] + mesh_vert_size;
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
    for (int i = 0; i < verts.size() / mesh_vert_size; ++i) {
        _normalize(&verts[i * mesh_vert_size + 6]);
    }
}
}  // namespace

Mesh::Mesh(int n_verts, int n_faces, int face_size, bool unlit, bool is_image)
    : 
      vert_size(is_image ? 5 : 9),
      vert(n_verts * vert_size),
      faces(n_faces * face_size),
      model_rotation(0),
      model_translation(0),
      face_size(face_size),
      unlit(unlit) {
}

void Mesh::update() {
    if (is_image) {
        if (!g_image_program) {
            g_image_program = GLShader(BASIC_IMAGE_SHADER_SRC, "BASIC_IMAGE");
        }
        vert_size = 5;
    } else {
        if (!g_mesh_program) {
            g_mesh_program = GLShader(BASIC_MESH_SHADER_SRC, "BASIC_MESH");
        }
        vert_size = 9;
    }

    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_);
    glGenBuffers(1, &ebo_);

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, vert.size() * sizeof(vert[0]), vert.data(),
                 GL_STATIC_DRAW);
    if (is_image) {
        // Position
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vert_size * sizeof(float),
                              (void*)0);
        // UV
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, vert_size * sizeof(float),
                              (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
    } else {
        // Position
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vert_size * sizeof(float),
                              (void*)0);
        // Color
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, vert_size * sizeof(float),
                              (void*)(3 * sizeof(float)));
        // Normals
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, vert_size * sizeof(float),
                              (void*)(6 * sizeof(float)));
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glEnableVertexAttribArray(2);
    }

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.size() * sizeof(faces[0]),
                 faces.data(), GL_STATIC_DRAW);
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void Mesh::draw(const glm::mat4x4& V, glm::mat4x4 K, bool y_up,
                int time) const {
    if (!visible) return;
    if (this->time != -1 && time != this->time) return;

    float norm = glm::length(model_rotation);
    if (norm < 1e-3) {
        transform_ = glm::mat4(1.0);
    } else {
        glm::quat rot = glm::angleAxis(norm, model_rotation / norm);
        transform_ = glm::mat4_cast(rot);
    }
    transform_ *= model_scale;
    transform_[3] = glm::vec4(model_translation, 1);
    glm::vec3 cam_pos = -glm::transpose(glm::mat3x3(V)) * glm::vec3(V[3]);
    if (!y_up) {
        K[1][1] *= -1.0;
    }
    if (!g_mesh_program) {
        fprintf(stderr, "mesh program not initialized\n");
        return;
    }

    if (is_image) {
        g_image_program.use();
        glUniform1i(g_image_program["tex"], 0);
    } else {
        g_mesh_program.use();
    }
    if (~texture_) {
        // Bind any textures required
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture_);
    }
    glm::mat4x4 MV = V * transform_;
    glUniformMatrix4fv(g_mesh_program["MV"], 1, GL_FALSE, glm::value_ptr(MV));
    glUniformMatrix4fv(g_mesh_program["M"], 1, GL_FALSE, glm::value_ptr(transform_));
    glUniformMatrix4fv(g_mesh_program["K"], 1, GL_FALSE, glm::value_ptr(K));
    glUniform3fv(g_mesh_program["cam_pos"], 1, glm::value_ptr(cam_pos));
    glUniform1f(g_mesh_program["point_size"], point_size);
    glUniform1i(g_mesh_program["unlit"], unlit);
    glBindVertexArray(vao_);
    if (faces.empty()) {
        glDrawArrays(get_gl_ele_type(face_size), 0, vert.size() / vert_size);
    } else {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
        glDrawElements(get_gl_ele_type(face_size), faces.size(),
                       GL_UNSIGNED_INT, (void*)0);
    }
    if (~texture_) {
        glBindTexture(GL_TEXTURE_2D, 0);
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
            vptr += m.vert_size;
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
                vptr += m.vert_size;
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
    Mesh m(2, 1, 2);
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
        throw std::runtime_error("Lines: Number of elements in points must be divisible by 3\n");
    }
    const int n_points = (int)points.size() / 3;
    Mesh m(n_points, n_points - 1, 2);
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
        vptr += m.vert_size;
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

Mesh Mesh::Image(
                float focal_length,
                float image_width,
                float image_height,
                float z,
                const std::vector<float>& r,
                const std::vector<float>& t,
                const uint8_t* data) {
    if (r.size() != t.size()) {
        throw std::runtime_error("Image: r must be same size as t\n");
    }
    if (r.size() != 3) {
        throw std::runtime_error("Image: r size must be divisible by 3\n");
    }
    int n_images = static_cast<int>(r.size() / 3);
    Mesh m(4,
           2,
           3,
           true,
           true);
    size_t vert_base = 0;
    float halfw = image_width * 0.5f, halfh = image_height * 0.5f;
    float invf = 1.f / focal_length;
    // Vertex generation
    m.vert[vert_base] = z * -halfw * invf;
    m.vert[vert_base + 1] = z * -halfh * invf;
    m.vert[vert_base + 2] = z;
    m.vert[vert_base + 3] = 0.f;
    m.vert[vert_base + 4] = 0.f;
    vert_base += m.vert_size;

    m.vert[vert_base] = z * -halfw * invf;
    m.vert[vert_base + 1] = z * halfh * invf;
    m.vert[vert_base + 2] = z;
    m.vert[vert_base + 3] = 0.f;
    m.vert[vert_base + 4] = 1.f;
    vert_base += m.vert_size;

    m.vert[vert_base] = z * halfw * invf;
    m.vert[vert_base + 1] = z * halfh * invf;
    m.vert[vert_base + 2] = z;
    m.vert[vert_base + 3] = 1.f;
    m.vert[vert_base + 4] = 1.f;
    vert_base += m.vert_size;

    m.vert[vert_base] = z * halfw * invf;
    m.vert[vert_base + 1] = z * -halfh * invf;
    m.vert[vert_base + 2] = z;
    m.vert[vert_base + 3] = 1.f;
    m.vert[vert_base + 4] = 0.f;

    glm::vec3 ri{r[0], r[1], r[2]};
    glm::vec3 ti{t[0], t[1], t[2]};
    m.apply_transform(ri, ti, 0, 4);

    // Face generation
    // triangle 1
    m.faces[0] = 0;
    m.faces[1] = 1;
    m.faces[2] = 2;
    // triangle 2 (could use strip instead)
    m.faces[3] = 0;
    m.faces[4] = 2;
    m.faces[5] = 3;

    // Texture
    m.set_image_texture(image_width, image_height, data);

    m.name = "Image";
    return m;
}

Mesh Mesh::Points(std::vector<float> points, glm::vec3 color) {
    if (points.size() % 3 != 0) {
        printf("Points: Number of elements in points must be divisible by 3\n");
    }
    const int n_points = (int)points.size() / 3;
    Mesh m(n_points, 0, 1);
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
        vptr += m.vert_size;
        pptr += 3;
    }

    m.name = "Points";
    m.unlit = true;
    return m;
}

void Mesh::auto_faces() {
    faces.resize(vert.size() / vert_size);
    std::iota(faces.begin(), faces.end(), 0);
}

void Mesh::repeat(int n) {
    if (n < 1) return;
    const size_t vert_size = vert.size();
    const size_t n_verts = vert_size / vert_size;
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
        end = vert.size() / vert_size;
    }
    auto* ptr = &vert[start * vert_size];
    for (int i = start; i < end; ++i) {
        glm::vec4 v(ptr[0], ptr[1], ptr[2], 1.0);
        v = transform * v;
        ptr[0] = v[0];
        ptr[1] = v[1];
        ptr[2] = v[2];
        ptr += vert_size;
    }
}

void Mesh::set_image_texture(int width, int height, const uint8_t* data) {
    is_image = true;
    if (texture_ == (unsigned)-1) {
        glGenTextures(1, &texture_);
    }
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, //8UI,
            width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
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
    mesh.vert_size = 9;
    mesh.vert.resize(mesh.vert_size * n_verts);
    for (size_t i = 0; i < n_verts; i++) {
        auto* ptr = &mesh.vert[i * mesh.vert_size];
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
            auto* color_ptr = &mesh.vert[i * mesh.vert_size + 3];
            for (int j = 0; j < 3; ++j) {
                color_ptr[j] = attrib.colors[3 * i + j];
            }
        }
    }
    if (attrib.normals.size() / 3 >= n_verts) {
        for (size_t i = 0; i < n_verts; i++) {
            auto* normal_ptr = &mesh.vert[i * mesh.vert_size + 6];
            for (int j = 0; j < 3; ++j) {
                normal_ptr[j] = attrib.normals[i * 3 + j];
            }
        }
    } else {
        estimate_normals(mesh.vert, mesh.faces, mesh.vert_size);
    }

    mesh.face_size = 3;
    if (from_string) {
        mesh.name = "OBJ";
    } else {
        mesh.name = path_or_string;
    }
    return mesh;
}

void Mesh::estimate_normals() {
    volrend::estimate_normals(vert, faces, vert_size);
}

Mesh Mesh::load_basic_obj(const std::string& path) {
    return _load_basic_obj(path, false);
}

Mesh Mesh::load_mem_basic_obj(const std::string& str) {
    return _load_basic_obj(str, true);
}

namespace {
}  // namespace
}  // namespace volrend
