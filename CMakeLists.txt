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
set( CMAKE_CXX_STANDARD 17 )
set( CMAKE_CXX_STANDARD_REQUIRED ON)
set( CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" )

set( INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include" )
set( SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src" )
set( VENDOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty" )
set( IMGUI_DIR "${VENDOR_DIR}/imgui" )
set( GLFW_DIR "${VENDOR_DIR}/glfw" )
set( GLEW_DIR "${VENDOR_DIR}/glew" )
set( GLM_DIR "${VENDOR_DIR}/glm" )
set( CNPY_DIR "${VENDOR_DIR}/cnpy" )
set( ZLIB_DIR "${VENDOR_DIR}/zlib" )

set( PROJ_LIB_NAME "volrend" )

if ( CMAKE_COMPILER_IS_GNUCXX OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang") )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -Wno-deprecated-declarations -O3 -funroll-loops" )
    if (NOT EMSCRIPTEN)
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g" )
    endif()

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

file(GLOB IMGUI_SOURCES ${IMGUI_DIR}/*.cpp)
set(IMGUI_HEADERS
    ${IMGUI_DIR}/imconfig.h
    ${IMGUI_DIR}/imgui.h
    ${IMGUI_DIR}/imgui_impl_glfw.h
    ${IMGUI_DIR}/imgui_impl_opengl3.h
    ${IMGUI_DIR}/imgui_stdlib.h
    ${IMGUI_DIR}/imstb_rectpack.h
    ${IMGUI_DIR}/imstb_textedit.h
    ${IMGUI_DIR}/imstb_truetype.h
    )

file(GLOB VOLREND_SOURCES ${SRC_DIR}/*.cpp)
file(GLOB VOLREND_PUBLIC_HEADERS ${INCLUDE_DIR}/volrend/*.hpp)
file(GLOB VOLREND_PRIVATE_HEADERS ${INCLUDE_DIR}/volrend/internal/*.hpp)

set(VOLREND_HEADERS ${VOLREND_PUBLIC_HEADERS} ${VOLREND_PRIVATE_HEADERS})

set( VOLREND_VENDOR_SOURCES
    ${CNPY_DIR}/cnpy.cpp
    )
if (NOT EMSCRIPTEN)
    set( VOLREND_VENDOR_SOURCES
        ${VOLREND_VENDOR_SOURCES}
        ${GLEW_DIR}/src/glew.c
        ${VENDOR_DIR}/misc/ImGuizmo.cpp
        ${IMGUI_SOURCES}
    )
endif()

set( VOLREND_VENDOR_SOURCES
    ${VOLREND_VENDOR_SOURCES}
    ${VENDOR_DIR}/misc/tiny_obj_loader.cpp
)

source_group( "Header Files" FILES ${VOLREND_HEADERS} )
source_group( "Source Files" FILES ${VOLREND_SOURCES} )

set(
    DEPENDENCIES
    )
if (NOT EMSCRIPTEN AND NOT WIN32 AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    set(
        DEPENDENCIES
        ${DEPENDENCIES}
        stdc++fs
        )
endif()

if (NOT EMSCRIPTEN)
    find_package(ZLIB QUIET)
endif()
if (ZLIB_FOUND)
    message( STATUS "Using system zlib" )
    set( DEPENDENCIES ${DEPENDENCIES} ZLIB::ZLIB )
else()
    set( SKIP_INSTALL_ALL OFF )
    set( SKIP_INSTALL_LIBRARIES OFF )
    message( STATUS "Using included zlib. ${ZLIB_DIR}" )
    add_subdirectory(${ZLIB_DIR})
    include_directories(${ZLIB_DIR})
    set( DEPENDENCIES ${DEPENDENCIES} zlibstatic )
endif()

set (_VOLREND_PNG_ "// ")
if (NOT EMSCRIPTEN)
    find_package(PNG QUIET)
    if (PNG_FOUND)
        set (_VOLREND_PNG_ "")
        include_directories(${PNG_INCLUDE_DIR})
        set( DEPENDENCIES ${DEPENDENCIES} ${PNG_LIBRARY} )
    else(PNG_FOUND)
        message( STATUS "libpng not found! Headless renderer won't write images out" )
    endif(PNG_FOUND)
endif()

add_definitions(-DGLEW_STATIC -DTINYEXR_USE_MINIZ=0)

include_directories(
    ${CNPY_DIR}
    ${GLEW_DIR}/include
    ${GLM_DIR}
    ${VENDOR_DIR}/misc
    )

if (NOT EMSCRIPTEN)
    find_package(OpenGL REQUIRED)
    set( DEPENDENCIES ${DEPENDENCIES} OpenGL::GL)

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
        set( DEPENDENCIES ${DEPENDENCIES} glfw )
    endif()

    find_package(Threads REQUIRED)
endif(NOT EMSCRIPTEN)

# create volrend.hpp
set ( COMMON_HPP_PATH "${PROJECT_BINARY_DIR}/include/volrend/common.hpp" )
configure_file( common.hpp.in "${COMMON_HPP_PATH}" )
set ( VOLREND_PUBLIC_HEADERS ${VOLREND_PUBLIC_HEADERS} "${COMMON_HPP_PATH}")

# Add main library
add_library( ${PROJ_LIB_NAME} STATIC ${VOLREND_SOURCES} ${VOLREND_VENDOR_SOURCES} )
if (WIN32)
    set_target_properties( ${PROJ_LIB_NAME} PROPERTIES OUTPUT_NAME "libvolrend" )
endif()

# Inline shader
macro(VOLREND_ADD_SHADER shader_name shader_file shader_inlined_file)
    add_custom_command(
        OUTPUT ${shader_inlined_file}
        COMMAND ${CMAKE_COMMAND}
        -DSHADER_FILE=${shader_file}
        -DSHADER_NAME=${shader_name}_SHADER
        -DOUTPUT=${shader_inlined_file}
        -P ${CMAKE_MODULE_PATH}/shader_inl.cmake
        DEPENDS ${shader_file}
        COMMENT "Inlining ${shader_name} shader"
        VERBATIM )
    add_custom_target(shader_inline_${shader_name}_SHADER ALL DEPENDS ${shader_inlined_file})
    add_dependencies( ${PROJ_LIB_NAME} shader_inline_${shader_name}_SHADER )
endmacro()

# Add shaders
VOLREND_ADD_SHADER(PLENOCTREE
                    "${PROJECT_SOURCE_DIR}/shaders/plenoctree.glsl"
                    "${PROJECT_BINARY_DIR}/include/volrend/internal/plenoctree.shader")
VOLREND_ADD_SHADER(FXAA
                    "${PROJECT_SOURCE_DIR}/shaders/fxaa.glsl"
                    "${PROJECT_BINARY_DIR}/include/volrend/internal/fxaa.shader")
VOLREND_ADD_SHADER(BASIC_MESH
                    "${PROJECT_SOURCE_DIR}/shaders/basic_mesh.glsl"
                    "${PROJECT_BINARY_DIR}/include/volrend/internal/basic_mesh.shader")
VOLREND_ADD_SHADER(BASIC_IMAGE
                    "${PROJECT_SOURCE_DIR}/shaders/basic_image.glsl"
                    "${PROJECT_BINARY_DIR}/include/volrend/internal/basic_image.shader")

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
    )

if (EMSCRIPTEN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++14")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -sUSE_WEBGL2=1 -sUSE_GLFW=3 -sFULL_ES3=1 -sWASM=1 -sALLOW_MEMORY_GROWTH=1 -sFETCH=1 -sINITIAL_MEMORY=50MB --bind -s EXPORT_NAME=\"'Volrend'\"")

    add_executable( volrend_web "web/main_web.cpp" )
    target_link_libraries( volrend_web
        ${PROJ_LIB_NAME} ${DEPENDENCIES} )

    # Not sure why this is needed in emscripten
    include_directories(
        ${PROJECT_BINARY_DIR}/3rdparty/zlib
        )

    set(INIT_JS_SRC "${PROJECT_SOURCE_DIR}/web/js/init.js" )
    set(MAIN_JS ${PROJECT_BINARY_DIR}/main.js )

    # Concat JS files automatically
    add_custom_command(
        OUTPUT ${MAIN_JS}
        COMMAND ${CMAKE_COMMAND} -DFILELIST=${INIT_JS_SRC},$<TARGET_FILE:volrend_web>
        -DOUTPUT=${MAIN_JS}
        -P ${CMAKE_MODULE_PATH}/concat.cmake
        DEPENDS ${INIT_JS_SRC} volrend_web
        COMMENT "Concatenating JS sources to main.js"
        VERBATIM )

    find_package(Python REQUIRED COMPONENTS Interpreter)

    add_custom_command(
        OUTPUT "${PROJECT_BINARY_DIR}/build/index.html"
        COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/web/inline.py "${PROJECT_BINARY_DIR}/build/index.html" "${PROJECT_SOURCE_DIR}/web/index.html" "${MAIN_JS}" "${PROJECT_BINARY_DIR}/volrend_web.wasm"
        DEPENDS "${PROJECT_SOURCE_DIR}/web/index.html" ${MAIN_JS}
    )
    add_custom_target(js_config ALL DEPENDS "${PROJECT_BINARY_DIR}/build/index.html" "${MAIN_JS}")
    add_custom_target(serve
        COMMAND ${Python_EXECUTABLE} -m http.server --directory "${PROJECT_BINARY_DIR}/build/"
        DEPENDS js_config)

else (EMSCRIPTEN)
    if (VOLREND_BUILD_INSTALL)
        if ( NOT WILL_USE_SYSTEM_GLFW )
            # Ugly hack
            export(TARGETS glfw
                NAMESPACE
                FILE "${CMAKE_CURRENT_BINARY_DIR}/3rdparty/glfwConfig.cmake")
        endif()
        if (NOT ZLIB_FOUND)
            export(TARGETS zlibstatic
                FILE "${CMAKE_CURRENT_BINARY_DIR}/zlibConfig.cmake"
                )
        endif()

        set_target_properties( ${PROJ_LIB_NAME} PROPERTIES
            PUBLIC_HEADER "${VOLREND_PUBLIC_HEADERS}"
            )

        set( INSTALL_TARGETS ${PROJ_LIB_NAME})
        if (NOT ZLIB_FOUND)
            set (INSTALL_TARGETS ${INSTALL_TARGETS} zlibstatic)
        endif()


        include(GNUInstallDirs)
        install(TARGETS ${INSTALL_TARGETS}
            EXPORT ${PROJ_LIB_NAME}Targets
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
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

    macro(VOLREND_ADD_EXECUTABLE target_name output_name main_file)
        add_executable( ${target_name} ${main_file} )
        target_link_libraries( ${target_name} ${PROJ_LIB_NAME} )
        set_target_properties( ${target_name} PROPERTIES OUTPUT_NAME ${output_name} )
    endmacro()

    VOLREND_ADD_EXECUTABLE(volrend_exe volrend main.cpp)

    if(WIN32)
        add_definitions(-DNOMINMAX -D_USE_MATH_DEFINES)
    endif()
endif (EMSCRIPTEN)
