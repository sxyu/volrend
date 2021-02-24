#include "volrend/n3tree.hpp"

#include <cassert>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <thread>
#include <atomic>

#include "glm/geometric.hpp"
#ifdef VOLREND_OPENEXR
#include <OpenEXR/ImfRgbaFile.h>
#else
#include "ilm/half.h"
#endif

namespace volrend {
namespace {
// Extract mean pose & other info from poses_bounds.npy
template <typename npy_scalar_t>
void unpack_llff_poses_bounds(cnpy::NpyArray& poses_bounds, float& width,
                              float& height, float& focal, glm::vec3& up,
                              glm::vec3& backward, glm::vec3& cen) {
    const npy_scalar_t* ptr = poses_bounds.data<npy_scalar_t>();
    height = ptr[4];
    width = ptr[9];
    focal = ptr[14];
    cen = glm::vec3(0);
    backward = glm::vec3(0);
    glm::vec3 right(0);

    // Random spaghetti emulating NeRF LLFF data loader
    const size_t BLOCK_SZ = 17;
    float bd_min = 1e9;
    for (size_t offs = 0; offs < poses_bounds.num_vals; offs += BLOCK_SZ) {
        for (size_t r = 0; r < 3; ++r) {
            const npy_scalar_t* row_ptr = &ptr[offs + 5 * r];
            right[r] += row_ptr[1];
            up[r] -= row_ptr[0];
            backward[r] += row_ptr[2];
            cen[r] += row_ptr[3];
        }
        bd_min =
            std::min(bd_min, (float)std::min(ptr[offs + 15], ptr[offs + 16]));
    }

    size_t total_cams = poses_bounds.num_vals / BLOCK_SZ;
    cen = cen / (total_cams * bd_min * 0.75f);
    backward = glm::normalize(backward);
    right = glm::normalize(glm::cross(up, backward));
    up = glm::normalize(glm::cross(backward, right));
}
}  // namespace

N3Tree::N3Tree() {}
N3Tree::N3Tree(const std::string& path) : npz_path_(path) { open(path); }
N3Tree::~N3Tree() {
#ifdef VOLREND_CUDA
    free_cuda();
#endif
}

void N3Tree::open(const std::string& path) {
    data_loaded_ = false;
#ifdef VOLREND_CUDA
    cuda_loaded_ = false;
#endif
    npz_path_ = path;
    assert(path.size() > 3 && path.substr(path.size() - 4) == ".npz");

    data_path_ = path.substr(0, path.size() - 4) + "_data.exr";
    poses_bounds_path_ = path.substr(0, path.size() - 4) + "_poses_bounds.npy";

    auto npz = cnpy::npz_load(path);
    data_dim = (int)*npz["data_dim"].data<int64_t>();
    switch (data_dim) {
        case 4 * 3 + 1:
            sh_order = 1;
            break;
        case 9 * 3 + 1:
            sh_order = 2;
            break;
        case 16 * 3 + 1:
            sh_order = 3;
            break;
        case 25 * 3 + 1:
            sh_order = 4;
            break;
        default:
            assert(data_dim == 4);
            sh_order = -1;
            break;
    }
    if (~sh_order) {
        std::cout << "INFO: Spherical harmonics order " << sh_order << "\n";
    } else {
        std::cout << "INFO: Spherical harmonics disabled\n";
    }

    n_internal = (int)*npz["n_internal"].data<int64_t>();
    if (npz.count("invradius3")) {
        const float* scale_data = npz["invradius3"].data<float>();
        for (int i = 0; i < 3; ++i) scale[i] = scale_data[i];
        std::cout << "INFO: New style scale " << scale[0] << " " << scale[1]
                  << " " << scale[2] << "\n";
    } else {
        scale[0] = scale[1] = scale[2] =
            (float)*npz["invradius"].data<double>();
    }
    {
        const float* offset_data = npz["offset"].data<float>();
        for (int i = 0; i < 3; ++i) offset[i] = offset_data[i];
    }

    auto child_node = npz["child"];
    std::swap(child_, npz["child"]);
    N = child_node.shape[1];
    N2_ = N * N;
    N3_ = N * N * N;

    if (npz.count("data")) {
        auto data_node = npz["data"];
        data_.clear();
        capacity = data_node.shape[0];
        if (capacity != n_internal) {
            std::cerr << "WARNING: N3Tree capacity != n_internal, "
                      << "call shrink_to_fit() before saving to save space\n";
        }
        if (data_node.word_size == 2) {
            std::cout << "INFO: Found data stored in half precision\n";
            const half* ptr = data_node.data<half>();
            data_ = std::vector<float>(ptr, ptr + data_node.num_vals);
        } else {
            std::cout << "INFO: Found data stored in single precision\n";
            // Avoid copy
            std::swap(data_cnpy_, npz["data"]);
        }
    } else {
        load_data();
    }

    use_ndc = bool(std::ifstream(poses_bounds_path_));
    if (use_ndc) {
        std::cout << "INFO: Found poses_bounds.npy for NDC: "
                  << poses_bounds_path_ << "\n";
        cnpy::NpyArray poses_bounds = cnpy::npy_load(poses_bounds_path_);

        if (poses_bounds.word_size == 4) {
            const float* ptr = poses_bounds.data<float>();
            unpack_llff_poses_bounds<float>(poses_bounds, ndc_width, ndc_height,
                                            ndc_focal, ndc_avg_up, ndc_avg_back,
                                            ndc_avg_cen);
        } else {
            assert(poses_bounds.word_size == 8);
            unpack_llff_poses_bounds<double>(poses_bounds, ndc_width,
                                             ndc_height, ndc_focal, ndc_avg_up,
                                             ndc_avg_back, ndc_avg_cen);
        }
    }
    last_sigma_thresh_ = -1.f;
#ifdef VOLREND_CUDA
    load_cuda();
#endif
}

void N3Tree::load_data() {
#ifdef VOLREND_OPENEXR
    std::cout << "INFO: Loading with OpenEXR (legacy)\n";
    Imf::RgbaInputFile file(data_path_.c_str());
    Imath::Box2i dw = file.dataWindow();
    int width = dw.max.x - dw.min.x + 1;
    int height = dw.max.y - dw.min.y + 1;
    capacity = height / N;
    assert(capacity >= n_internal);

    std::vector<Imf::Rgba> tmp(height * width);
    file.setFrameBuffer(&tmp[0] - dw.min.x - dw.min.y * width, 1, width);
    file.readPixels(dw.min.y, dw.max.y);
    half* loaded = reinterpret_cast<half*>(tmp.data());
    // FIXME get rid of this copy (low priority)
    data_ = std::vector<float>(loaded, loaded + height * width * data_dim);
    data_loaded_ = true;
#else
    throw std::runtime_error(
        "Volrend was not built with OpenEXR, "
        "legacy format is not available");
#endif
}

int32_t N3Tree::get_child(int nd, int i, int j, int k) {
    return child_.data<int32_t>()[pack_index(nd, i, j, k)];
}

::volrend::Rgba N3Tree::get_data(int nd, int i, int j, int k) {
    assert(data_loaded_);  // Call load_data()
    auto base_idx = pack_index(nd, i, j, k) * data_dim;
    float r = data_[base_idx];
    float g = data_[base_idx + 1];
    float b = data_[base_idx + 2];
    float a = data_[base_idx + 3];
    return {r, g, b, a};
}

bool N3Tree::is_data_loaded() { return data_loaded_; }
#ifdef VOLREND_CUDA
bool N3Tree::is_cuda_loaded() { return cuda_loaded_; }
#endif

int N3Tree::pack_index(int nd, int i, int j, int k) {
    assert(i < N && j < N && k < N && i >= 0 && j >= 0 && k >= 0);
    return nd * N3_ + i * N2_ + j * N + k;
}

std::tuple<int, int, int, int> N3Tree::unpack_index(int packed) {
    int k = packed % N;
    packed /= N;
    int j = packed % N;
    packed /= N;
    int i = packed % N;
    packed /= N;
    return {packed, i, j, k};
}

float* N3Tree::data_ptr() {
    return data_.empty() ? data_cnpy_.data<float>() : data_.data();
}
const float* N3Tree::data_ptr() const {
    return data_.empty() ? data_cnpy_.data<float>() : data_.data();
}
}  // namespace volrend
