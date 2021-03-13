#pragma once

#include <vector>
#include <string>
#include "glm/mat4x4.hpp"

namespace volrend {

struct Mesh {
    explicit Mesh(int n_verts = 0, int n_faces = 0, int face_size = 3,
                  bool unshaded = false);

    // Upload to GPU
    void update();

    // Draw the mesh (unshaded)
    void draw(const glm::mat4x4& V, const glm::mat4x4& K) const;

    void load_basic_obj(const std::string& path);

    // Vertex positions
    std::vector<float> vert;
    // Triangle indices
    std::vector<unsigned int> faces;

    // Model transform, rotatin is axis-angle
    glm::vec3 rotation, translation;
    float scale = 1.f;

    int face_size;
    bool unlit = true;

    std::string name = "Mesh";

    // * Preset meshes
    // Unit cube centered at 0
    static Mesh Cube(glm::vec3 color = glm::vec3(1.0f, 0.5f, 0.2f));
    // Unit UV sphere centered at 0
    static Mesh Sphere(int rings = 30, int sectors = 30,
                       glm::vec3 color = glm::vec3(1.0f, 0.5f, 0.2f));

   private:
    unsigned int vao_, vbo_, ebo_;
};

}  // namespace volrend
