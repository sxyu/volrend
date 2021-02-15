#include "volrend/n3tree.hpp"

#include <cassert>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <thread>
#include <atomic>
#ifdef VOLREND_OPENEXR
#include <OpenEXR/ImfRgbaFile.h>
#else
#include "ilm/half.h"
#endif

namespace volrend {

void precomp_kernel(float* data, const int32_t* __restrict__ child,
                    float step_sz, float sigma_thresh, const int N,
                    const int data_dim) {}

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
    scale = (float)*npz["invradius"].data<double>();
    float* offset_data = npz["offset"].data<float>();
    for (int i = 0; i < 3; ++i) offset[i] = offset_data[i];

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
            ndc_height = ptr[4];
            ndc_width = ptr[9];
            ndc_focal = ptr[14];
        } else {
            assert(poses_bounds.word_size == 8);
            const double* ptr = poses_bounds.data<double>();
            ndc_height = ptr[4];
            ndc_width = ptr[9];
            ndc_focal = ptr[14];
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

#ifndef VOLREND_CUDA
bool N3Tree::precompute_step(float sigma_thresh) const {
    if (last_sigma_thresh_ == sigma_thresh) {
        return false;
    }
    last_sigma_thresh_ = sigma_thresh;
    const size_t data_count = capacity * N3_;
    const float* data_ptr =
        data_.empty() ? data_cnpy_.data<float>() : data_.data();
    data_proc_.resize(data_count * data_dim);
    float* data_out_ptr = data_proc_.data();
    const int32_t* child_ptr = child_.data<int32_t>();
    float step_sz = 1.f / scale;

    std::atomic<size_t> counter(0);
    auto worker = [&]() {
        while (true) {
            size_t i = counter++;
            if (i >= data_count) {
                break;
            }
            const float* rgba = data_ptr + i * data_dim;
            float* rgba_out = data_out_ptr + i * data_dim;
            if (child_ptr[i]) continue;

            for (int i = 0; i < data_dim; ++i) {
                if (isnan(rgba[i]))
                    rgba_out[i] = 0.f;
                else {
                    rgba_out[i] = std::min(std::max(rgba[i], -1e9f), 1e9f);
                }
            }

            const int alpha_idx = data_dim - 1;
            if (rgba[alpha_idx] < sigma_thresh)
                rgba_out[alpha_idx] = 0.f;
            else
                rgba_out[alpha_idx] = rgba[alpha_idx] * step_sz;
        }
    };
    std::vector<std::thread> thds;
    for (int i = 0; i < std::thread::hardware_concurrency(); ++i) {
        thds.emplace_back(worker);
    }
    for (auto& thd : thds) thd.join();
    return true;
}
#endif

}  // namespace volrend
