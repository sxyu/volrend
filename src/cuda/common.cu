#include "volrend/cuda/common.cuh"

#include <stdlib.h>
#include <stdio.h>

namespace volrend {

cudaError_t cuda_assert(const cudaError_t code, const char* const file,
                        const int line, const bool abort) {
    if (code != cudaSuccess) {
        fprintf(stderr, "cuda_assert: %s %s %d\n", cudaGetErrorString(code),
                file, line);

        if (abort) {
            cudaDeviceReset();
            exit(code);
        }
    }

    return code;
}

}  // namespace volrend
