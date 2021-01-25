#include "volrend/n3tree.hpp"

#include <limits>
#include <cstdio>
#include <cassert>
#include "volrend/cuda/n3tree_query.cuh"

namespace volrend {
namespace device {
namespace {
__global__ void query_kernel(const float* data,
                                      const int32_t* __restrict__ child,
                                      const float* __restrict__ indices,
                                      const float* __restrict__ offset,
                                      float scale,
                                      float* __restrict__ result,
                                      const int N,
                                      const int Q) {
    CUDA_GET_THREAD_ID(tid, Q);
    const float* xyz = indices + 3 * tid;
    float* out = result + 4 * tid;
    float q[3] = {xyz[0], xyz[1], xyz[2]};
    transform_coord<float>(q, offset, scale);
    float _cube_sz;
    query_single_from_root(data, child, q, out, &_cube_sz, N);
}

__global__ void precomp_kernel(float* data,
                               const int32_t* __restrict__ child,
                               float step_sz,
                               float sigma_thresh,
                               const int N) {
    CUDA_GET_THREAD_ID(tid, N);
    float* rgba = data + tid * 4;

    // Nonleaf
    if (child[tid]) return;

    for (int i = 0; i < 3; ++i) {
        rgba[i] = 1.f / (1.f + expf(-rgba[i]));
    }
    rgba[3] = fmaxf(rgba[3] * step_sz, 0.f);
    if (rgba[3] < sigma_thresh)
        rgba[3] = 0.f;
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
    cuda(MemcpyAsync(device.data, data_.data(), data_sz,
                cudaMemcpyHostToDevice));
    cuda(MemcpyAsync(device.offset, offset.data(), 3 * sizeof(float),
                cudaMemcpyHostToDevice));
    cuda_loaded_ = true;
    last_sigma_thresh_ = -1.f;
    precompute_step(0.f);
}

void N3Tree::free_cuda() {
    if (device.data != nullptr) cuda(Free(device.data));
    if (device.child != nullptr) cuda(Free(device.child));
    if (device.offset != nullptr) cuda(Free(device.offset));
}

std::vector<float> N3Tree::operator[](const std::vector<float>& indices) const {
    float *d_result, *d_indices;

    assert(indices.size() % 3 == 0);
    int Q = indices.size() / 3;
    cuda(Malloc((void**)&d_result, data_dim * Q * sizeof(float)));
    cuda(Malloc((void**)&d_indices, indices.size() * sizeof(float)));
    cuda(MemcpyAsync(d_indices, indices.data(), indices.size() * sizeof(float),
                cudaMemcpyHostToDevice));
    device::query_kernel<<<N_BLOCKS_NEEDED(Q), N_CUDA_THREADS>>>(
         device.data,
         device.child,
         d_indices,
         device.offset,
         scale,
         d_result,
         N, Q);

    std::vector<float> out(data_dim * Q);
    cuda(MemcpyAsync(out.data(), d_result,
                data_dim * Q * sizeof(float), cudaMemcpyDeviceToHost));
    cuda(Free(d_result));
    cuda(Free(d_indices));
    return out;
}

void N3Tree::precompute_step(float sigma_thresh) const {
    if (last_sigma_thresh_ == sigma_thresh) {
        return;
    }
    last_sigma_thresh_ = sigma_thresh;
    const size_t data_count = capacity * N3_;
    cuda(MemcpyAsync(device.data, data_.data(), data_count * data_dim * sizeof(float),
                cudaMemcpyHostToDevice));

    device::precomp_kernel<<<N_BLOCKS_NEEDED(data_count), N_CUDA_THREADS>>>
        (
            device.data,
            device.child,
            1.f / scale,
            sigma_thresh,
            data_count
        );
}
}  // namespace volrend
