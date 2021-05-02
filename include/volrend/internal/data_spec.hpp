#pragma once

#include "volrend/common.hpp"
#include "volrend/n3tree.hpp"
#include "volrend/camera.hpp"

namespace volrend {
namespace internal {
namespace {

struct CameraSpec {
    const int width;
    const int height;
    const float fx, fy;
    const float* VOLREND_RESTRICT transform;
    CameraSpec(const Camera& camera)
        : width(camera.width),
          height(camera.height),
          fx(camera.fx),
          fy(camera.fy),
          transform(camera.device.transform) {}
};
struct TreeSpec {
    const __half* VOLREND_RESTRICT const data;
    const int32_t* VOLREND_RESTRICT const child;
    const float* VOLREND_RESTRICT const offset;
    const float* VOLREND_RESTRICT const scale;
    const float* VOLREND_RESTRICT const extra;
    const int N;
    const int N3;
    const int data_dim;
    const DataFormat data_format;
    const float ndc_width;
    const float ndc_height;
    const float ndc_focal;

    TreeSpec(const N3Tree& tree, bool cpu = false)
        : data(cpu ? tree.data_.data<__half>() : tree.device.data),
          child(cpu ? tree.child_.data<int32_t>() : tree.device.child),
          offset(cpu ? tree.offset.data() : tree.device.offset),
          scale(cpu ? tree.scale.data() : tree.device.scale),
          extra(cpu ? tree.extra_.data<float>() : tree.device.extra),
          N(tree.N),
          N3(tree.N * tree.N * tree.N),
          data_dim(tree.data_dim),
          data_format(tree.data_format),
          ndc_width(tree.use_ndc ? tree.ndc_width : -1),
          ndc_height(tree.ndc_height),
          ndc_focal(tree.ndc_focal) {}
};

}  // namespace
}  // namespace internal
}  // namespace volrend
