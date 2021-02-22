#include "volrend/n3tree.hpp"

#include <limits>
#include <cstdio>
#include <cassert>
#include "volrend/cuda/n3tree_query.cuh"

namespace volrend {
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
    if (device.scale == nullptr) {
        cuda(Malloc((void**)&device.scale, 3 * sizeof(float)));
    }
    cuda(MemcpyAsync(device.child, child_.data<int32_t>(),  child_sz,
                cudaMemcpyHostToDevice));
    const float* data_ptr = this->data_ptr();
    cuda(MemcpyAsync(device.data, data_ptr, data_sz,
                cudaMemcpyHostToDevice));
    cuda(MemcpyAsync(device.offset, offset.data(), 3 * sizeof(float),
                cudaMemcpyHostToDevice));
    cuda(MemcpyAsync(device.scale, scale.data(), 3 * sizeof(float),
                cudaMemcpyHostToDevice));
    cuda_loaded_ = true;
}

void N3Tree::free_cuda() {
    if (device.data != nullptr) cuda(Free(device.data));
    if (device.child != nullptr) cuda(Free(device.child));
    if (device.offset != nullptr) cuda(Free(device.offset));
    if (device.scale != nullptr) cuda(Free(device.scale));
}
}  // namespace volrend
