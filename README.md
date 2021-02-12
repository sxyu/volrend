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

On Ubuntu, to get the dependencies, try
`sudo apt-get install libgl1-mesa-dev libxi-dev libxinerama-dev libxcursor-dev libxrandr-dev libgl1-mesa-dev libglu1-mesa-dev`

## Run
```sh
./volrend <name>.npz
```
For LLFF scenes, we also expect
```sh
<name>_poses_bounds.npy
```
In the same directory. This may be copied directly from the scene's `poses_bounds.npy` in the LLFF dataset.

Note: In older octrees, an OpenEXR file was used to store the image data, this was
```sh
<name>_data.exr
```
In the same direcotry.

### Dependencies
- C++14
- OpenGL
    - any dependencies of GLFW
- CUDA Toolkit (tested on CUDA version 11.2)
- OpenEXR *Now optional* 
(I installed from source <https://github.com/AcademySoftwareFoundation/openexr>)
