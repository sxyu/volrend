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

    // Use mesh shader program (for shader render purposes only)
    static void use_shader();

    // Draw the mesh
    void draw(const glm::mat4x4& V, glm::mat4x4 K, bool y_up = true) const;

    // Load a basic OBJ file (triangles & optionally vertex colors)
    void load_basic_obj(const std::string& path);

    // Create faces by grouping consecutive vertices
    void auto_faces();

    // Vertex positions
    std::vector<float> vert;
    // Triangle indices
    std::vector<unsigned int> faces;

    // Model transform, rotation is axis-angle
    glm::vec3 rotation, translation;
    float scale = 1.f;

    // Computed transform
    mutable glm::mat4 transform_;

    int face_size;
    bool visible = true;
    bool unlit = false;

    std::string name = "Mesh";

    // * Preset meshes
    // Unit cube centered at 0
    static Mesh Cube(glm::vec3 color = glm::vec3(1.0f, 0.5f, 0.2f));
    // Unit UV sphere centered at 0
    static Mesh Sphere(int rings = 30, int sectors = 30,
                       glm::vec3 color = glm::vec3(1.0f, 0.5f, 0.2f));
    // Point lattice
    static Mesh Lattice(int reso = 8,
                        glm::vec3 color = glm::vec3(0.5f, 0.5f, 0.5f));

   private:
    unsigned int vao_, vbo_, ebo_;
};

}  // namespace volrend
