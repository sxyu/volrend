#include "volrend/n3tree.hpp"

#include <limits>
#include <cstdio>
#include <cassert>
#include "volrend/cuda/n3tree_query.cuh"

namespace volrend {
namespace device {
namespace {
__global__ void precomp_kernel(float* data,
                               const int32_t* __restrict__ child,
                               float step_sz,
                               float sigma_thresh,
                               const int N,
                               const int data_dim) {
    CUDA_GET_THREAD_ID(tid, N);
    float* rgba = data + tid * data_dim;

    // Nonleaf (TODO don't even need to store these, waste of 1/7)
    if (child[tid]) return;

    for (int i = 0; i < data_dim; ++i) {
        if (isnan(rgba[i]))
            rgba[i] = 0.f;
        rgba[i] = min(max(rgba[i], -1e9f), 1e9f);
    }

    // for (int i = 0; i < 3; ++i) {
        // rgba[i] = 1.f / (1.f + expf(-rgba[i]));
    // }

    const int alpha_idx = data_dim - 1;
    if (rgba[alpha_idx] < sigma_thresh)
        rgba[alpha_idx] = 0.f;
    rgba[alpha_idx] *= step_sz;
}
}  // namespace
}  // namespace device

void N3Tree::load_cuda() {
    if (device.data != nullptr) cuda(Free(device.data));
    if (device.child != nullptr) cuda(Free(device.child));
    const size_t data_sz = capacity * N3_ * data_dim * sizeof(float);
    const size_t child_sz = capacity * N3_ * sizeof(int32_t);
    cuda(Malloc((void**)&device.data, data_sz));
    cuda(Malloc((void**)&device.child, child_sz));
    if (device.offset == nullptr) {
        cuda(Malloc((void**)&device.offset, 3 * sizeof(float)));
    }
    cuda(MemcpyAsync(device.child, child_.data<int32_t>(),  child_sz,
                cudaMemcpyHostToDevice));
    const float* data_ptr = data_.empty() ? data_cnpy_.data<float>() : data_.data();
    cuda(MemcpyAsync(device.data, data_ptr, data_sz,
                cudaMemcpyHostToDevice));
    cuda(MemcpyAsync(device.offset, offset.data(), 3 * sizeof(float),
                cudaMemcpyHostToDevice));
    cuda_loaded_ = true;
    precompute_step(0.f);
}

void N3Tree::free_cuda() {
    if (device.data != nullptr) cuda(Free(device.data));
    if (device.child != nullptr) cuda(Free(device.child));
    if (device.offset != nullptr) cuda(Free(device.offset));
}

bool N3Tree::precompute_step(float sigma_thresh) const {
    if (last_sigma_thresh_ == sigma_thresh) {
        return false;
    }
    last_sigma_thresh_ = sigma_thresh;
    const size_t data_count = capacity * N3_;
    const float* data_ptr = data_.empty() ? data_cnpy_.data<float>() : data_.data();
    cuda(MemcpyAsync(device.data, data_ptr, data_count * data_dim * sizeof(float),
                cudaMemcpyHostToDevice));

    const int N_CUDA_THREADS = 512;

    device::precomp_kernel<<<N_BLOCKS_NEEDED(data_count, N_CUDA_THREADS), N_CUDA_THREADS>>>
        (
            device.data,
            device.child,
            1.f / scale,
            sigma_thresh,
            data_count,
            data_dim
        );
    return true;
}
}  // namespace volrend
