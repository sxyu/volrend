#include "volrend/n3tree.hpp"

#include <iostream>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <OpenEXR/ImfRgbaFile.h>

namespace volrend {

N3Tree::N3Tree(const std::string& path) : npz_path_(path) { open(path); }
N3Tree::~N3Tree() { free_cuda(); }

void N3Tree::open(const std::string& path) {
    data_loaded_ = false;
    cuda_loaded_ = false;
    npz_path_ = path;
    assert(path.size() > 3 && path.substr(path.size() - 4) == ".npz");

    data_path_ = path.substr(0, path.size() - 4) + "_data.exr";

    auto npz = cnpy::npz_load(path);
    data_dim = (int)*npz["data_dim"].data<int64_t>();
    assert(data_dim == 4);
    n_internal = (int)*npz["data_dim"].data<int64_t>();
    scale = (float)*npz["invradius"].data<double>();
    float* offset_data = npz["offset"].data<float>();
    for (int i = 0; i < 3; ++i) offset[i] = offset_data[i];

    auto child_node = npz["child"];
    std::swap(child_, npz["child"]);
    N = child_node.shape[1];
    N2_ = N * N;
    N3_ = N * N * N;
    load_data();
    load_cuda();
}

void N3Tree::load_data() {
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
    data_ = std::vector<float>(loaded, loaded + height * width * data_dim);
    data_loaded_ = true;
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
bool N3Tree::is_cuda_loaded() { return cuda_loaded_; }

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

}  // namespace volrend
