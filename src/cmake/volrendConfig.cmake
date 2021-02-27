include(CMakeFindDependencyMacro)

find_dependency(Threads)
find_dependency(glfw3)
set(OpenGL_GL_PREFERENCE GLVND)
find_dependency(OpenGL)
find_dependency(ZLIB)

include("${CMAKE_CURRENT_LIST_DIR}/volrendTargets.cmake")
