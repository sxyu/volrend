# N3-Tree Volume Rendering

This is a real-time volume renderer built using CUDA + OpenGL interop.
Here, we use a N^3 tree to model volumetric emission and absorption at RGB frequencies
(by N^3 tree we mean octree but with branching factor N, the standard case is N=2).

Similar to NeRF, each voxel cell contains `(r,g,b,sigma)`, and
alpha composition is applied via the rule `1 - exp(-length * sigma)`.
A variant of the DDA ray tracing algorithm is used to compute the length within each grid cell;
performance improvement may be possible..

## Build
Build using CMake
```sh
mkdir build && cd build
cmake ..
make -j12
```

### Dependencies
- C++14
- OpenGL
    - any dependencies of GLFW
- CUDA Toolkit (tested on CUDA version 11.2)
- OpenEXR (I installed from source <https://github.com/AcademySoftwareFoundation/openexr>)
