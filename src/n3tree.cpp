#include "volrend/n3tree.hpp"
#include "volrend/data_format.hpp"

#include <cassert>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <thread>
#include <atomic>

#include "glm/geometric.hpp"

#ifndef VOLREND_CUDA
#include "half.hpp"
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

void DataFormat::parse(const std::string& str) {
    size_t nonalph_idx = -1;
    for (size_t i = 0; i < str.size(); ++i) {
        if (!std::isalpha(str[i])) {
            nonalph_idx = i;
            break;
        }
    }
    if (~nonalph_idx) {
        basis_dim = std::atoi(str.c_str() + nonalph_idx);
        const std::string tmp = str.substr(0, nonalph_idx);
        if (tmp == "ASG")
            format = ASG;
        else if (tmp == "SG")
            format = SG;
        else if (tmp == "SH")
            format = SH;
        else
            format = RGBA;
    } else {
        basis_dim = -1;
        format = RGBA;
    }
}

std::string DataFormat::to_string() const {
    std::string out;
    switch (format) {
        case ASG:
            out = "ASG";
            break;
        case SG:
            out = "SG";
            break;
        case SH:
            out = "SH";
            break;
        case RGBA:
            out = "RGBA";
            break;
        default:
            out = "UNKNOWN";
            break;
    }
    if (~basis_dim) out.append(std::to_string(basis_dim));
    return out;
}

N3Tree::N3Tree() {}
N3Tree::N3Tree(const std::string& path) { open(path); }
N3Tree::~N3Tree() {
#ifdef VOLREND_CUDA
    free_cuda();
#endif
}

void N3Tree::open(const std::string& path) {
    child_.data_holder.clear();
    child_.data_holder.shrink_to_fit();
    data_.data_holder.clear();
    data_.data_holder.shrink_to_fit();

    data_loaded_ = false;
#ifdef VOLREND_CUDA
    cuda_loaded_ = false;
#endif
    npz_path_ = path;
    assert(path.size() > 3 && path.substr(path.size() - 4) == ".npz");

    poses_bounds_path_ = path.substr(0, path.size() - 4) + "_poses_bounds.npy";

    cnpy::npz_t npz = cnpy::npz_load(path);
    load_npz(npz);

    use_ndc = bool(std::ifstream(poses_bounds_path_));
    if (use_ndc) {
        std::cerr << "INFO: Found poses_bounds.npy for NDC: "
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
    data_loaded_ = true;
}

void N3Tree::open_mem(const char* data, uint64_t size) {
    data_loaded_ = false;
#ifdef VOLREND_CUDA
    cuda_loaded_ = false;
#endif
    child_.data_holder.clear();
    child_.data_holder.shrink_to_fit();
    data_.data_holder.clear();
    data_.data_holder.shrink_to_fit();

    npz_path_ = "";
    cnpy::npz_t npz = cnpy::npz_load_mem(data, size);
    load_npz(npz);

    last_sigma_thresh_ = -1.f;
#ifdef VOLREND_CUDA
    load_cuda();
#endif
    data_loaded_ = true;
}

void N3Tree::load_npz(cnpy::npz_t& npz) {
    data_dim = (int)*npz["data_dim"].data<int64_t>();
    if (npz.count("data_format")) {
        auto& df_node = npz["data_format"];
        std::string data_format_str =
            std::string(df_node.data_holder.begin(), df_node.data_holder.end());
        // Unicode to ASCII
        for (size_t i = 4; i < data_format_str.size(); i += 4) {
            data_format_str[i / 4] = data_format_str[i];
        }
        data_format_str.resize(data_format_str.size() / 4);
        data_format.parse(data_format_str);
    } else {
        // Old style auto-infer SH dims
        if (data_dim == 4) {
            data_format.format = DataFormat::RGBA;
            std::cerr << "INFO: Legacy file with no format specifier; "
                         "spherical basis disabled\n";
        } else {
            data_format.format = DataFormat::SH;
            data_format.basis_dim = (data_dim - 1) / 3;
            std::cerr << "INFO: Legacy file with no format specifier; "
                         "autodetect spherical harmonics order\n";
        }
    }
    std::cerr << "INFO: Data format " << data_format.to_string() << "\n";

    n_internal = (int)*npz["n_internal"].data<int64_t>();
    if (npz.count("invradius3")) {
        const float* scale_data = npz["invradius3"].data<float>();
        for (int i = 0; i < 3; ++i) scale[i] = scale_data[i];
    } else {
        scale[0] = scale[1] = scale[2] =
            (float)*npz["invradius"].data<double>();
    }
    std::cerr << "INFO: Scale " << scale[0] << " " << scale[1] << " "
              << scale[2] << "\n";
    {
        const float* offset_data = npz["offset"].data<float>();
        for (int i = 0; i < 3; ++i) offset[i] = offset_data[i];
    }

    auto child_node = npz["child"];
    std::swap(child_, npz["child"]);
    N = child_node.shape[1];
    N2_ = N * N;
    N3_ = N * N * N;

    auto& data_node = npz["data"];
    capacity = data_node.shape[0];
    if (capacity != n_internal) {
        std::cerr << "WARNING: N3Tree capacity != n_internal, "
                  << "call shrink_to_fit() before saving to save space\n";
    }
    if (npz["data"].word_size != 2) {
        throw std::runtime_error("data must be stored in half precision");
    }

    if (npz.count("extra_data")) {
        std::swap(extra_, npz["extra_data"]);
    } else {
        extra_.data_holder.clear();
    }
    std::swap(data_, data_node);
}

namespace {
void _push_wireframe_bb(const float bb[6], std::vector<float>& verts_out) {
#define PUSH_VERT(i, j, k)              \
    verts_out.push_back(bb[i * 3]);     \
    verts_out.push_back(bb[j * 3 + 1]); \
    verts_out.push_back(bb[k * 3 + 2]); \
    verts_out.push_back(0);             \
    verts_out.push_back(0);             \
    verts_out.push_back(0);             \
    verts_out.push_back(0);             \
    verts_out.push_back(0);             \
    verts_out.push_back(1)
    // clang-format off
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            PUSH_VERT(0, i, j);
            PUSH_VERT(1, i, j);
            PUSH_VERT(i, 0, j);
            PUSH_VERT(i, 1, j);
            PUSH_VERT(i, j, 0);
            PUSH_VERT(i, j, 1);
        }
    }
    // clang-format on
#undef PUSH_VERT
}
void _gen_wireframe_impl(const N3Tree& tree, size_t nodeid, size_t xi,
                         size_t yi, size_t zi, int depth, size_t gridsz,
                         int max_depth, std::vector<float>& verts_out) {
    const int32_t* child =
        tree.child_.data<int32_t>() + nodeid * tree.N * tree.N * tree.N;
    int cnt = 0;
    // Use integer coords to avoid precision issues
    for (size_t i = xi * tree.N; i < (xi + 1) * tree.N; ++i) {
        for (size_t j = yi * tree.N; j < (yi + 1) * tree.N; ++j) {
            for (size_t k = zi * tree.N; k < (zi + 1) * tree.N; ++k) {
                if (child[cnt] == 0 || depth >= max_depth) {
                    // Add this cube
                    const float bb[6] = {
                        ((float)i / gridsz - tree.offset[0]) / tree.scale[0],
                        ((float)j / gridsz - tree.offset[1]) / tree.scale[1],
                        ((float)k / gridsz - tree.offset[2]) / tree.scale[2],
                        ((float)(i + 1) / gridsz - tree.offset[0]) /
                            tree.scale[0],
                        ((float)(j + 1) / gridsz - tree.offset[1]) /
                            tree.scale[1],
                        ((float)(k + 1) / gridsz - tree.offset[2]) /
                            tree.scale[2]};
                    _push_wireframe_bb(bb, verts_out);
                } else {
                    _gen_wireframe_impl(tree, nodeid + child[cnt], i, j, k,
                                        depth + 1, gridsz * tree.N, max_depth,
                                        verts_out);
                }
                ++cnt;
            }
        }
    }
}  // namespace
}  // namespace

std::vector<float> N3Tree::gen_wireframe(int max_depth) const {
    std::vector<float> verts;
    if (!data_loaded_) {
        std::cerr << "ERROR: Please load data before gen_wireframe!\n";
        return verts;
    }
    _gen_wireframe_impl(*this, 0, 0, 0, 0,
                        /*depth*/ 0, N, max_depth, verts);
    return verts;
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
    return std::tuple<int, int, int, int> {packed, i, j, k};
}

}  // namespace volrend
