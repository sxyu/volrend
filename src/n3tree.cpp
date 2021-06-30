#include "volrend/n3tree.hpp"
#include "volrend/data_format.hpp"
#include "volrend/internal/morton.hpp"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <thread>
#include <atomic>

#include "glm/geometric.hpp"

#include "half.hpp"

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
    clear_cpu_memory();

    data_loaded_ = false;
#ifdef VOLREND_CUDA
    cuda_loaded_ = false;
#endif
    npz_path_ = path;
    assert(path.size() > 3 && path.substr(path.size() - 4) == ".npz");

    poses_bounds_path_ = path.substr(0, path.size() - 4) + "_poses_bounds.npy";

    if (!std::ifstream(path)) {
        printf("Can't load because file does not exist: %s\n", path.c_str());
        return;
    }

    cnpy::npz_t npz = cnpy::npz_load(path);
    load_npz(npz);

    use_ndc = bool(std::ifstream(poses_bounds_path_));
    if (use_ndc) {
        fprintf(stderr, "INFO: Found poses_bounds.npy for NDC: %s\n",
                poses_bounds_path_.c_str());
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
    clear_cpu_memory();

    npz_path_ = "";
    cnpy::npz_t npz = cnpy::npz_load_mem(data, size);
    load_npz(npz);

    last_sigma_thresh_ = -1.f;
#ifdef VOLREND_CUDA
    load_cuda();
#endif
    data_loaded_ = true;
}

// namespace {
// int _calc_tree_maxdepth(const N3Tree& tree, size_t nodeid, size_t xi, size_t
// yi,
//                         size_t zi) {
//     const int32_t* child =
//         tree.child_.data<int32_t>() + nodeid * tree.N * tree.N * tree.N;
//     int maxdep = 0, cnt = 0;
//     // Use integer coords to avoid precision issues
//     for (size_t i = xi * tree.N; i < (xi + 1) * tree.N; ++i) {
//         for (size_t j = yi * tree.N; j < (yi + 1) * tree.N; ++j) {
//             for (size_t k = zi * tree.N; k < (zi + 1) * tree.N; ++k) {
//                 if (child[cnt] != 0) {
//                     int subdep =
//                         _calc_tree_maxdepth(tree, nodeid + child[cnt], i, j,
//                         k);
//                     maxdep = std::max(subdep + 1, maxdep);
//                 }
//                 ++cnt;
//             }
//         }
//     }
//     return maxdep;
// }
//
// // Populate the occupancy + voxel size grid
// void _calc_occu_lut(N3Tree& tree, size_t nodeid, uint32_t xi, uint32_t yi,
//                     uint32_t zi, int depth) {
//     const int32_t* child =
//         tree.child_.data<int32_t>() + nodeid * tree.N * tree.N * tree.N;
//     int cnt = 0;
//     uint8_t depth_diff = tree.max_depth - depth;
//     uint32_t scale = 1 << depth_diff;
//     uint32_t scale3 = scale * scale * scale;
//     // Use integer coords to avoid precision issues
//     for (uint32_t i = xi * tree.N; i < (xi + 1) * tree.N; ++i) {
//         for (uint32_t j = yi * tree.N; j < (yi + 1) * tree.N; ++j) {
//             for (uint32_t k = zi * tree.N; k < (zi + 1) * tree.N; ++k) {
//                 if (child[cnt] == 0) {
//                     uint32_t start = internal::morton_code_3(
//                         i * scale, j * scale, k * scale);
//                     uint32_t end = start + scale3;
//                     std::fill(tree.occu_lut_.begin() + start,
//                               tree.occu_lut_.begin() + end, depth_diff);
//                 } else {
//                     _calc_occu_lut(tree, nodeid + child[cnt], i, j, k,
//                                    depth + 1);
//                 }
//                 ++cnt;
//             }
//         }
//     }
// }
// }  // namespace

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
            fprintf(stderr,
                    "INFO: Legacy file with no format specifier; "
                    "spherical basis disabled\n");
        } else {
            data_format.format = DataFormat::SH;
            data_format.basis_dim = (data_dim - 1) / 3;
            fprintf(stderr,
                    "INFO: Legacy file with no format specifier; "
                    "autodetect spherical harmonics order\n");
        }
    }
    fprintf(stderr, "INFO: Data format %s\n", data_format.to_string().c_str());

    if (npz.count("invradius3")) {
        const float* scale_data = npz["invradius3"].data<float>();
        for (int i = 0; i < 3; ++i) scale[i] = scale_data[i];
    } else {
        scale[0] = scale[1] = scale[2] =
            (float)*npz["invradius"].data<double>();
    }
    printf("INFO: Scale %f %f %f", scale[0], scale[1], scale[2]);
    {
        const float* offset_data = npz["offset"].data<float>();
        for (int i = 0; i < 3; ++i) offset[i] = offset_data[i];
    }

    auto child_node = npz["child"];
    std::swap(child_, npz["child"]);
    N = child_node.shape[1];
    if (N != 2) {
        fprintf(stderr, "WARNING: N != 2 probably doesn't work.\n");
    }
    N2_ = N * N;
    N3_ = N * N * N;

    if (npz.count("quant_colors")) {
        fprintf(stderr, "INFO: Decoding quantized colors\n");
        auto& quant_colors_node = npz["quant_colors"];
        if (quant_colors_node.word_size != 2) {
            throw std::runtime_error(
                "codebook must be stored in half precision");
        }
        auto& quant_map_node = npz["quant_map"];
        capacity = quant_map_node.shape[1];
        int n_basis = quant_map_node.shape[0];
        if (quant_colors_node.shape[0] != n_basis) {
            throw std::runtime_error(
                "codebook and map basis numbers does not match");
        }
        int n_basis_retain =
            npz.count("data_retained") ? npz["data_retained"].shape[0] : 0;
        n_basis += n_basis_retain;

        data_.data_holder.clear();
        data_.reinit({(size_t)capacity, (size_t)N, (size_t)N, (size_t)N,
                      (size_t)data_dim},
                     2, false);

        // Decode quantized
        auto& sigma_node = npz["sigma"];
        half* data_ptr = data_.data<half>();
        const half* sigma_ptr = sigma_node.data<half>();
        const uint16_t* quant_map_ptr = quant_map_node.data<uint16_t>();
        const half* quant_colors_ptr = quant_colors_node.data<half>();

        const size_t n_child = (size_t)capacity * N * N * N;
        for (size_t i = 0; i < n_child; ++i) {
            int off = i * data_dim;
            for (int j = 0; j < n_basis - n_basis_retain; ++j) {
                int boff = off + j + n_basis_retain;
                int id = quant_map_ptr[j * n_child + i];
                const half* colors_ptr =
                    quant_colors_ptr + j * 65536 * 3 + id * 3;
                for (int k = 0; k < 3; ++k) {
                    data_ptr[boff] = colors_ptr[k];
                    boff += n_basis;
                }
            }

            data_ptr[off + data_dim - 1] = sigma_ptr[i];
        }
        if (n_basis_retain) {
            auto& retain_node = npz["data_retained"];
            const half* retain_ptr = retain_node.data<half>();
            for (size_t i = 0; i < n_child; ++i) {
                int off = i * data_dim;
                for (int j = 0; j < n_basis_retain; ++j) {
                    int boff = off + j;
                    const half* colors_ptr =
                        retain_ptr + j * n_child * 3 + i * 3;
                    for (int k = 0; k < 3; ++k) {
                        data_ptr[boff] = colors_ptr[k];
                        boff += n_basis;
                    }
                }
            }
        }
    } else {
        auto& data_node = npz["data"];
        capacity = data_node.shape[0];
        if (data_node.word_size != 2) {
            throw std::runtime_error("data must be stored in half precision");
        }
        std::swap(data_, data_node);
    }

    if (npz.count("extra_data")) {
        std::swap(extra_, npz["extra_data"]);
    } else {
        extra_.data_holder.clear();
    }

    // max_depth = _calc_tree_maxdepth(*this, 0, 0, 0, 0);
    // resolution = 1 << (max_depth + 1);
    // resolution3_ = resolution * resolution * resolution;
    //
    // occu_lut_.resize(resolution3_);
    // _calc_occu_lut(*this, 0, 0, 0, 0, 0);
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
        fprintf(stderr, "ERROR: Please load data before gen_wireframe!\n");
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

void N3Tree::clear_cpu_memory() {
    // Keep child in order to generate grids
    // child_.data_holder.clear();
    // child_.data_holder.shrink_to_fit();
    data_.data_holder.clear();
    data_.data_holder.shrink_to_fit();
}

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
    return std::tuple<int, int, int, int>{packed, i, j, k};
}

}  // namespace volrend
