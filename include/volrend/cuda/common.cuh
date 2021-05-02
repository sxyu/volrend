#pragma once
#include "volrend/common.hpp"

#ifdef VOLREND_CUDA

#include <cuda_runtime.h>

#define CUDA_GET_THREAD_ID(tid, Q) const int tid = blockIdx.x * blockDim.x + threadIdx.x; \
                      if (tid >= Q) return
#define N_BLOCKS_NEEDED(Q, N_CUDA_THREADS) ((Q - 1) / N_CUDA_THREADS + 1)

template<typename scalar_t>
__host__ __device__ __inline__ static scalar_t _norm(
        scalar_t* __restrict__ dir) {
    return sqrtf(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
}

template<typename scalar_t>
__host__ __device__ __inline__ static void _normalize(
        scalar_t* __restrict__ dir) {
    scalar_t invnorm = 1.f / _norm(dir);
    dir[0] *= invnorm; dir[1] *= invnorm; dir[2] *= invnorm;
}

template<typename scalar_t>
__host__ __device__ __inline__ static void _mv3(
        const scalar_t* __restrict__ m,
        const scalar_t* __restrict__ v,
        scalar_t* __restrict__ out) {
    out[0] = m[0] * v[0] + m[3] * v[1] + m[6] * v[2];
    out[1] = m[1] * v[0] + m[4] * v[1] + m[7] * v[2];
    out[2] = m[2] * v[0] + m[5] * v[1] + m[8] * v[2];
}

template<typename scalar_t>
__host__ __device__ __inline__ static void _copy3(
        const scalar_t* __restrict__ v,
        scalar_t* __restrict__ out) {
    out[0] = v[0]; out[1] = v[1]; out[2] = v[2];
}

template<typename scalar_t>
__host__ __device__ __inline__ static scalar_t _dot3(
        const scalar_t* __restrict__ u,
        const scalar_t* __restrict__ v) {
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

template<typename scalar_t>
__host__ __device__ __inline__ static
void _cross3(const scalar_t* a, const scalar_t* b, scalar_t* out) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

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
