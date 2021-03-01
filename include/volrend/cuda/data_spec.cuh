#pragma once

#include "volrend/n3tree.hpp"
#include "volrend/camera.hpp"

// Max global basis 
#define VOLREND_GLOBAL_BASIS_MAX 25

namespace volrend {
namespace {

struct CameraSpec {
    int width;
    int height;
    float focal;
    float* transform;
    static CameraSpec load(const Camera& camera) {
        CameraSpec spec;
        spec.width = camera.width;
        spec.height = camera.height;
        spec.focal = camera.focal;
        spec.transform = camera.device.transform;
        return spec;
    }
};
struct TreeSpec {
    __half* data;
    int32_t* child;
    float* offset;
    float* scale;
    float* extra;
    int N;
    int data_dim;
    DataFormat data_format;
    float ndc_width;
    float ndc_height;
    float ndc_focal;

    static TreeSpec load(const N3Tree& tree) {
        TreeSpec spec;
        spec.data = tree.device.data;
        spec.child = tree.device.child;
        spec.offset = tree.device.offset;
        spec.scale = tree.device.scale;
        spec.extra = tree.device.extra;
        spec.N = tree.N;
        spec.data_dim = tree.data_dim;
        spec.data_format = tree.data_format;
        spec.ndc_width = tree.use_ndc ? tree.ndc_width : -1,
        spec.ndc_height = tree.ndc_height;
        spec.ndc_focal = tree.ndc_focal;
        return spec;
    }
};

}  // namespace
}  // namespace volrend
