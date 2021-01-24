cmake_minimum_required( VERSION 3.3 )

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
if (POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)
if (POLICY CMP0069)
    cmake_policy(SET CMP0069 NEW)
endif (POLICY CMP0069)
if (POLICY CMP0072)
    cmake_policy(SET CMP0072 NEW)
endif (POLICY CMP0072)

project( volrend )
set(VOLREND_VERSION_MAJOR 0)
set(VOLREND_VERSION_MINOR 0)
set(VOLREND_VERSION_PATCH 1)
set(VOLREND_VERSION ${VOLREND_VERSION_MAJOR}.${VOLREND_VERSION_MINOR}.${VOLREND_VERSION_PATCH})

option( VOLREND_USE_SYSTEM_GLFW
    "Use system glfw rather than the included glfw submodule if available" ON )
option( VOLREND_BUILD_INSTALL "Build the install target" ON )
option( VOLREND_BUILD_PYTHON "Build Python bindings" OFF )
option( VOLREND_USE_FFAST_MATH "Use -ffast-math" OFF )

set( CMAKE_CXX_STACK_SIZE "10000000" )
set( CMAKE_CXX_STANDARD 14 )
set( CMAKE_CXX_STANDARD_REQUIRED ON)
set( CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" )

# CUDA
enable_language(CUDA)
message(STATUS "CUDA enabled")
set( CMAKE_CUDA_STANDARD 14 )
set( CMAKE_CUDA_STANDARD_REQUIRED ON)
set( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -g -Xcudafe \"--display_error_number --diag_suppress=3057 --diag_suppress=3058 --diag_suppress=3059 --diag_suppress=3060\" -lineinfo")

set( INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include" )
set( SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src" )
set( VENDOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty" )
set( IMGUI_DIR "${VENDOR_DIR}/imgui_mod" )
set( GLFW_DIR "${VENDOR_DIR}/glfw" )
set( GLEW_DIR "${VENDOR_DIR}/glew" )
set( GLM_DIR "${VENDOR_DIR}/glm" )
set( CNPY_DIR "${VENDOR_DIR}/cnpy" )
set( ZLIB_DIR "${VENDOR_DIR}/zlib" )

set( PROJ_LIB_NAME "volrend" )

if ( CMAKE_COMPILER_IS_GNUCXX OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang") )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -Wno-deprecated-declarations -O3 -funroll-loops -g" )
    if( VOLREND_USE_FFAST_MATH )
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math" )
    endif()
elseif( MSVC )
    if( VOLREND_USE_FFAST_MATH )
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:fast" )
    endif()
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT /GLT /Ox")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /MTd")
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler=\"/MT\"" )
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /MT /GLT")
endif ( )

# Git submodule auto update
# https://cliutils.gitlab.io/modern-cmake/chapters/projects/submodule.html
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    # Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Running git submodule update --init --recursive")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
        message(STATUS "Submodule update done")
    endif()
endif()

if(NOT EXISTS "${GLFW_DIR}/CMakeLists.txt")
   message(FATAL_ERROR "A submodule as not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

set (_VOLREND_IMGUI_ "")
file(GLOB IMGUI_SOURCES ${IMGUI_DIR}/volrend/imgui/*.cpp)
set(IMGUI_HEADERS
        ${IMGUI_DIR}/volrend/imgui/imconfig.h
        ${IMGUI_DIR}/volrend/imgui/imgui.h
        ${IMGUI_DIR}/volrend/imgui/imgui_impl_glfw.h
        ${IMGUI_DIR}/volrend/imgui/imgui_impl_opengl3.h
        ${IMGUI_DIR}/volrend/imgui/imgui_stdlib.h
        ${IMGUI_DIR}/volrend/imgui/imstb_rectpack.h
        ${IMGUI_DIR}/volrend/imgui/imstb_textedit.h
        ${IMGUI_DIR}/volrend/imgui/imstb_truetype.h
    )

file(GLOB VOLREND_SOURCES ${SRC_DIR}/*.cpp)
file(GLOB VOLREND_SOURCES_CUDA ${SRC_DIR}/cuda/*.cu)
set(VOLREND_SOURCES ${VOLREND_SOURCES} ${VOLREND_SOURCES_CUDA})
file(GLOB VOLREND_PUBLIC_HEADERS ${INCLUDE_DIR}/volrend/*.hpp)
file(GLOB VOLREND_PRIVATE_HEADERS ${INCLUDE_DIR}/volrend/internal/*.hpp)

# create volrend.hpp
set ( COMMON_HPP_PATH "${PROJECT_BINARY_DIR}/include/volrend/common.hpp" )
configure_file( common.hpp.in "${COMMON_HPP_PATH}" )
set ( VOLREND_PUBLIC_HEADERS ${VOLREND_PUBLIC_HEADERS} "${COMMON_HPP_PATH}")

set ( VOLREND_HEADERS ${VOLREND_PUBLIC_HEADERS} ${VOLREND_PRIVATE_HEADERS} )

set( VOLREND_VENDOR_SOURCES
    ${GLEW_DIR}/src/glew.c
    ${CNPY_DIR}/cnpy.cpp
)

source_group( "Header Files" FILES ${VOLREND_HEADERS} )
source_group( "Source Files" FILES ${VOLREND_SOURCES} )

set(
  DEPENDENCIES
)

find_package(ZLIB QUIET)
if (ZLIB_FOUND)
    message( STATUS "Using system zlib" )
    set( DEPENDENCIES ${DEPENDENCIES} ZLIB::ZLIB )
else()
    message( STATUS "Using included zlib." )
    add_subdirectory(${ZLIB_DIR})
    include_directories(${ZLIB_DIR})
    set( DEPENDENCIES ${DEPENDENCIES} zlibstatic )
endif()

# require OpenEXR
find_package (OpenEXR REQUIRED)

add_definitions(-DGLEW_STATIC -DTINYEXR_USE_MINIZ=0)

include_directories(
    ${CNPY_DIR}
    ${GLEW_DIR}/include
    ${GLM_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${VENDOR_DIR}/misc
)

find_package(OpenGL REQUIRED)
set( DEPENDENCIES ${DEPENDENCIES}
      OpenGL::GL
    )

set ( WILL_USE_SYSTEM_GLFW ${VOLREND_USE_SYSTEM_GLFW} )

find_package(PkgConfig)
if ( NOT PkgConfig_FOUND )
    set ( WILL_USE_SYSTEM_GLFW OFF )
else()
    pkg_check_modules(GLFW glfw3)
    if ( NOT GLFW_FOUND )
        set ( WILL_USE_SYSTEM_GLFW OFF )
    endif ()
endif ()

if ( NOT WILL_USE_SYSTEM_GLFW )
    message ( STATUS "Using included glfw3 (in 3rdparty/)" )
    # Build GLFW
    SET(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "GLFW example" FORCE)
    SET(GLFW_BUILD_TESTS OFF CACHE BOOL "GLFW tests" FORCE)
    SET(GLFW_BUILD_DOCS OFF CACHE BOOL "GLFW docs" FORCE)
    SET(GLFW_INSTALL ${VOLREND_BUILD_INSTALL} CACHE BOOL "GLFW install" FORCE)
    add_subdirectory(${GLFW_DIR})
    set( DEPENDENCIES ${DEPENDENCIES} glfw )
    include_directories( "${GLFW_DIR}/include" )
else()
    message ( STATUS "Using system glfw3" )
    set( DEPENDENCIES ${DEPENDENCIES} glfw3 )
endif()

find_package(Threads REQUIRED)

add_library( ${PROJ_LIB_NAME} STATIC ${VOLREND_SOURCES}
    ${IMGUI_SOURCES} ${VOLREND_VENDOR_SOURCES} )
set_target_properties( ${PROJ_LIB_NAME}
                           PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(
    ${PROJ_LIB_NAME} PUBLIC
    $<BUILD_INTERFACE:${INCLUDE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_include_directories(
    ${PROJ_LIB_NAME} PUBLIC
    $<BUILD_INTERFACE:${IMGUI_DIR}>
)
target_link_libraries( ${PROJ_LIB_NAME} PUBLIC
    ${DEPENDENCIES}
    ${CMAKE_THREAD_LIBS_INIT}
    ${CMAKE_DL_LIBS}
    OpenEXR::OpenEXR
)

if (VOLREND_BUILD_INSTALL)
    if ( NOT WILL_USE_SYSTEM_GLFW )
        # Ugly hack
        export(TARGETS glfw
               NAMESPACE
               FILE "${CMAKE_CURRENT_BINARY_DIR}/3rdparty/glfwConfig.cmake")
    endif()
    set_target_properties( ${PROJ_LIB_NAME} PROPERTIES
        PUBLIC_HEADER "${VOLREND_PUBLIC_HEADERS}"
    )

    include(GNUInstallDirs)
    install(TARGETS ${PROJ_LIB_NAME}
            EXPORT ${PROJ_LIB_NAME}Targets
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/volrend
    )
    export(TARGETS ${PROJ_LIB_NAME}
           NAMESPACE volrend::
           FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJ_LIB_NAME}Config.cmake")
    install(EXPORT
                ${PROJ_LIB_NAME}Targets
                DESTINATION "${CMAKE_INSTALL_DATADIR}/volrend/cmake"
                NAMESPACE volrend::
            )
    install(FILES "${SRC_DIR}/cmake/volrendConfig.cmake"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/volrend/cmake")
    install(FILES ${IMGUI_HEADERS}
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/volrend/imgui")
endif(VOLREND_BUILD_INSTALL)

add_executable( volrend_exe main.cpp )
target_link_libraries( volrend_exe ${PROJ_LIB_NAME} )
set_target_properties( volrend_exe PROPERTIES OUTPUT_NAME "volrend" )

if(WIN32)
    add_definitions(-DNOMINMAX -D_USE_MATH_DEFINES)
    if (MSVC AND VOLREND_BUILD_EXAMPLE)
        set_property(TARGET example APPEND PROPERTY LINK_FLAGS "/DEBUG /LTCG" )
    endif ( MSVC AND VOLREND_BUILD_EXAMPLE )
endif()
