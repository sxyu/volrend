#pragma once
#ifdef VOLREND_CUDA

#include "volrend/common.hpp"
#include <cuda_runtime.h>

#define CUDA_GET_THREAD_ID(tid, Q) const int tid = blockIdx.x * blockDim.x + threadIdx.x; \
                      if (tid >= Q) return
#define N_CUDA_THREADS 1024
#define N_BLOCKS_NEEDED(Q) ((Q - 1) / N_CUDA_THREADS + 1)

template <typename scalar_t>
__device__ __inline__ bool outside_grid(const scalar_t* __restrict__ q) {
    for (int i = 0; i < 3; ++i) {
        if (q[i] < 0.0 || q[i] >= 1.0 - 1e-10)
            return true;
    }
    return false;
}

template <typename scalar_t>
__device__ __inline__ void transform_coord(scalar_t* __restrict__ q,
                                           const scalar_t* __restrict__ offset,
                                           scalar_t scale) {
    for (int i = 0; i < 3; ++i) {
        q[i] = offset[i] + scale * q[i];
    }
}

namespace volrend {

// Beware that NVCC doesn't work with C files and __VA_ARGS__
cudaError_t cuda_assert(const cudaError_t code, const char* const file,
                        const int line, const bool abort);

}  // namespace volrend

#define cuda(...) cuda_assert((cuda##__VA_ARGS__), __FILE__, __LINE__, true);

#else
#define cuda(...)
#endif
