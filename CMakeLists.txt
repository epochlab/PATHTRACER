cmake_minimum_required(VERSION 3.21)
project(Pathtracer LANGUAGES C CXX OBJCXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Single-config generators default to unoptimized if CMAKE_BUILD_TYPE is unset: measured 1.1s Release vs 47s+ (5M-triangle glTF load).
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(OpenGL REQUIRED)
find_package(glfw3 3.5 REQUIRED)
find_package(GLEW REQUIRED)
# Homebrew's glew config links OpenGL.framework by absolute path; a foreign SDK in -F shadows this build's own system frameworks.
set_property(TARGET GLEW::GLEW PROPERTY INTERFACE_LINK_LIBRARIES OpenGL::GL)
find_package(glm REQUIRED)
find_package(Imath REQUIRED)
find_package(OpenEXR REQUIRED)
find_package(OpenColorIO REQUIRED)
find_package(embree 4 CONFIG REQUIRED)
# render_beauty's PNG encode only. Not a new dependency: OpenEXR already links zlib, so naming it makes the direct use explicit.
find_package(ZLIB REQUIRED)

# Static-analysis gate on every build: -Wall -Wextra -Werror, clang-tidy per compile (non-fatal), cppcheck as its own target.
find_program(CLANG_TIDY_EXE NAMES clang-tidy)
if(CLANG_TIDY_EXE)
    # Homebrew's clang-tidy does not share Xcode clang's default sysroot, so without this it finds no standard library headers.
    execute_process(COMMAND xcrun --show-sdk-path
                     OUTPUT_VARIABLE MACOS_SDK_PATH
                     OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(CLANG_TIDY_COMMAND ${CLANG_TIDY_EXE}
        --extra-arg=-isysroot --extra-arg=${MACOS_SDK_PATH})
endif()

# Every first-party target's compile flags in one place. NATIVE/IPO/TIDY are opt-in.
function(pathtracer_target_defaults target)
    cmake_parse_arguments(ARG "NATIVE;IPO;TIDY" "" "" ${ARGN})
    target_compile_options(${target} PRIVATE -Wall -Wextra -Werror)
    if(ARG_NATIVE)
        target_compile_options(${target} PRIVATE -march=native)
    endif()
    if(ARG_IPO AND IPO_SUPPORTED)
        set_target_properties(${target} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
    if(ARG_TIDY AND CLANG_TIDY_EXE)
        set_target_properties(${target} PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_COMMAND}")
    endif()
endfunction()

include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED)

# Build provenance for the startup spec block, at build time not configure time: the SHA keys benchmark-log records.
find_package(Git QUIET)
set(PATHTRACER_GENERATED_DIR ${CMAKE_BINARY_DIR}/generated)
add_custom_target(pathtracer_git_sha ALL
    COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=${CMAKE_SOURCE_DIR} -DOUTPUT=${PATHTRACER_GENERATED_DIR}/pathtracer_git_sha.h
            -DGIT_EXECUTABLE=${GIT_EXECUTABLE} -P ${CMAKE_SOURCE_DIR}/cmake/GitSha.cmake
    BYPRODUCTS ${PATHTRACER_GENERATED_DIR}/pathtracer_git_sha.h
    COMMENT "Stamping git SHA")
# Compile-time half of BuildInfo, $<CONFIG> for multi-config correctness. Hardcodes -march=native: a target that differs must not use it.
set(PATHTRACER_PROVENANCE_DEFINITIONS PATHTRACER_BUILD_TYPE="$<CONFIG>" PATHTRACER_MARCH="native" PATHTRACER_IPO=$<BOOL:${IPO_SUPPORTED}>)

find_program(CPPCHECK_EXE NAMES cppcheck)
if(CPPCHECK_EXE)
    add_custom_target(cppcheck
        COMMAND ${CPPCHECK_EXE} --enable=warning,style,performance --std=c++20
                --suppress=missingInclude
                --suppressions-list=${CMAKE_SOURCE_DIR}/.cppcheck-suppressions
                -I ${CMAKE_SOURCE_DIR}/include
                ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tools
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "cppcheck over src/ and tools/")
endif()

# The comment budget: one line per comment, 140 columns, over every first-party source.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_custom_target(comment_lint
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/comment_lint.py
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "comment budget over C++, GLSL, CMake, Python and the tool configs")
endif()

# Opt-in, default off: the pathtracer parses external glTF/JSON/EXR, where memory-safety tooling earns its keep.
option(PATHTRACER_SANITIZE "Build with AddressSanitizer + UndefinedBehaviorSanitizer" OFF)
if(PATHTRACER_SANITIZE)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()

set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/third_party/imgui)
if(NOT EXISTS ${IMGUI_DIR}/imgui.cpp)
    message(FATAL_ERROR "third_party/imgui is empty, run: git submodule update --init")
endif()

set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp)

set(CGLTF_DIR ${CMAKE_SOURCE_DIR}/third_party/cgltf)
set(MIKKTSPACE_DIR ${CMAKE_SOURCE_DIR}/third_party/mikktspace)
if(NOT EXISTS ${MIKKTSPACE_DIR}/mikktspace.c)
    message(FATAL_ERROR "third_party/mikktspace is empty, run: git submodule update --init")
endif()
set(JSON_DIR ${CMAKE_SOURCE_DIR}/third_party)

# Vendored, not ours to fix. OBJECT libraries since CXX_CLANG_TIDY is target-level only; kept separate so headless targets skip imgui.
add_library(vendored_cgltf OBJECT src/scene/cgltf_impl.cpp)
target_include_directories(vendored_cgltf PRIVATE ${CGLTF_DIR})
target_compile_options(vendored_cgltf PRIVATE -w)

add_library(vendored_mikktspace OBJECT ${MIKKTSPACE_DIR}/mikktspace.c)
target_compile_options(vendored_mikktspace PRIVATE -w)

add_library(vendored_imgui OBJECT ${IMGUI_SOURCES})
target_include_directories(vendored_imgui PRIVATE ${IMGUI_DIR} ${IMGUI_DIR}/backends)
target_link_libraries(vendored_imgui PRIVATE glfw)
target_compile_options(vendored_imgui PRIVATE -w)

# Colorimetry, shared by pathtracer_core and metal_fit. No -march=native or IPO: albedo_table's committed output must reproduce exactly.
add_library(pathtracer_cie OBJECT src/scene/cie.cpp src/scene/cone_space.cpp)
target_include_directories(pathtracer_cie PUBLIC include)
pathtracer_target_defaults(pathtracer_cie)
target_link_libraries(pathtracer_cie PRIVATE glm::glm)

# The renderer proper: every source that includes no GL, GLFW or imgui header, which is what lets tools/ and the C ABI link it.
add_library(pathtracer_core STATIC
    src/config/profile_config.cpp
    src/config/scene_config.cpp
    src/debug/aov.cpp
    src/debug/aov_filters.cpp
    src/debug/bench_log.cpp
    src/debug/frame_stats.cpp
    src/debug/host_info.cpp
    src/debug/memory_tracker.cpp
    src/debug/perf_dashboard.cpp
    src/debug/power_spectrum.cpp
    src/debug/scale_space.cpp
    src/gfx/hdr_image.cpp
    src/gfx/ocio_cpu_transform.cpp
    src/gfx/viewport.cpp
    src/scene/bsdf.cpp
    src/scene/camera.cpp
    src/scene/embree_accel.cpp
    src/scene/environment_map.cpp
    src/scene/gbuffer_shading.cpp
    src/scene/gltf_loader.cpp
    src/scene/lens.cpp
    src/scene/light.cpp
    src/scene/material_binding.cpp
    src/scene/path_trace_driver.cpp
    src/scene/path_tracer.cpp
    src/scene/gbuffer.cpp
    src/scene/ray_types.cpp
    src/scene/sampler.cpp
    src/scene/shading_scene.cpp
    src/scene/thread_pool.cpp
    src/api/headless_renderer.cpp
    $<TARGET_OBJECTS:pathtracer_cie>
    $<TARGET_OBJECTS:vendored_cgltf>
    $<TARGET_OBJECTS:vendored_mikktspace>)
# include/ and the json headers are PUBLIC because every consumer includes them; cgltf, MikkTSpace and the SHA header are internal.
target_include_directories(pathtracer_core PUBLIC include ${JSON_DIR})
target_include_directories(pathtracer_core PRIVATE ${CGLTF_DIR} ${MIKKTSPACE_DIR} ${PATHTRACER_GENERATED_DIR})
add_dependencies(pathtracer_core pathtracer_git_sha)
target_compile_definitions(pathtracer_core PRIVATE ${PATHTRACER_PROVENANCE_DEFINITIONS})
pathtracer_target_defaults(pathtracer_core NATIVE IPO TIDY)
target_link_libraries(pathtracer_core PUBLIC glm::glm embree OpenEXR::OpenEXR Imath::Imath
                       OpenColorIO::OpenColorIO ZLIB::ZLIB)

# The C ABI a foreign runtime binds to via ctypes. Shared, not static: it is dlopen'd, not linked.
add_library(pathtracer_c SHARED src/api/pathtracer_c.cpp)
target_compile_definitions(pathtracer_c PRIVATE ASSET_ROOT_DIR="${CMAKE_SOURCE_DIR}/assets")
pathtracer_target_defaults(pathtracer_c NATIVE IPO TIDY)
target_link_libraries(pathtracer_c PRIVATE pathtracer_core)

# The GUI application: main.cpp plus exactly the sources that touch GL/GLFW/imgui. Everything else it needs comes from pathtracer_core.
add_executable(pathtracer
    src/main.cpp
    src/platform/window.cpp
    src/platform/display_link.mm
    src/gfx/gl_debug.cpp
    src/gfx/shader_program.cpp
    src/gfx/texture.cpp
    src/gfx/post_process_pass.cpp
    src/gfx/ocio_display_transform.cpp
    src/scene/debug_camera_controller.cpp
    src/debug/gpu_timer.cpp
    src/debug/histogram.cpp
    src/debug/hud_overlay.cpp
    src/debug/system_info.cpp
    src/debug/spec_report.cpp
    $<TARGET_OBJECTS:vendored_imgui>)
target_include_directories(pathtracer PRIVATE ${IMGUI_DIR} ${IMGUI_DIR}/backends)
target_compile_definitions(pathtracer PRIVATE
    GL_SILENCE_DEPRECATION
    ASSET_ROOT_DIR="${CMAKE_SOURCE_DIR}/assets")
# IPO is a free codegen speedup for the path tracer's hot loop (ray traversal, BSDF eval), no behaviour or quality change.
pathtracer_target_defaults(pathtracer NATIVE IPO TIDY)
target_link_libraries(pathtracer PRIVATE
    pathtracer_core glfw GLEW::GLEW OpenGL::GL
    "-framework AppKit" "-framework QuartzCore")
set_source_files_properties(src/platform/display_link.mm PROPERTIES COMPILE_OPTIONS -fobjc-arc)

# Asset/codegen CLIs, deliberately not linking pathtracer_core.
add_executable(test_pattern tools/test_pattern.cpp)
target_compile_definitions(test_pattern PRIVATE
    ASSET_ROOT_DIR="${CMAKE_SOURCE_DIR}/assets")
pathtracer_target_defaults(test_pattern)
target_link_libraries(test_pattern PRIVATE OpenEXR::OpenEXR Imath::Imath)

add_executable(downsample tools/downsample.cpp)
pathtracer_target_defaults(downsample)
target_link_libraries(downsample PRIVATE OpenEXR::OpenEXR Imath::Imath)

# Offline generator for blue_noise_mask.inc: it produces a committed artifact rather than checking one, so it stays out of ctest.
add_executable(bluenoise_mask tools/bluenoise_mask.cpp)
pathtracer_target_defaults(bluenoise_mask NATIVE IPO)
target_link_libraries(bluenoise_mask PRIVATE pathtracer_core)

# ColorChecker spectral data to linear Rec.709, shared by its texture generator and colour_validate's colour-science oracle.
add_library(colorchecker_core OBJECT tools/colorchecker.cpp)
target_include_directories(colorchecker_core PUBLIC include tools)
pathtracer_target_defaults(colorchecker_core)
target_link_libraries(colorchecker_core PRIVATE glm::glm)

# Offline generator for assets/textures/macbeth.exr, a regenerable (gitignored) asset, so it stays out of ctest.
add_executable(colorchecker_texture tools/colorchecker_texture.cpp $<TARGET_OBJECTS:colorchecker_core>)
target_compile_definitions(colorchecker_texture PRIVATE
    ASSET_ROOT_DIR="${CMAKE_SOURCE_DIR}/assets"
    TOOLS_DATA_DIR="${CMAKE_SOURCE_DIR}/tools/data")
pathtracer_target_defaults(colorchecker_texture)
target_link_libraries(colorchecker_texture PRIVATE pathtracer_core)

# Offline generator for albedo_table.inc. Built without -march=native or IPO: its output is committed source and must reproduce exactly.
add_executable(albedo_table tools/albedo_table.cpp)
target_include_directories(albedo_table PRIVATE include)
pathtracer_target_defaults(albedo_table)
target_link_libraries(albedo_table PRIVATE glm::glm)

# Measured conductor (lambda, n, k) to a material JSON's reflectivity/edgeTint. No -march=native or IPO, for albedo_table's reason.
add_library(metal_fit_core OBJECT tools/metal_fit.cpp)
target_include_directories(metal_fit_core PUBLIC include tools)
pathtracer_target_defaults(metal_fit_core)
target_link_libraries(metal_fit_core PRIVATE glm::glm)

add_executable(metal_fit tools/metal_fit_main.cpp $<TARGET_OBJECTS:metal_fit_core> $<TARGET_OBJECTS:pathtracer_cie>)
target_include_directories(metal_fit PRIVATE include tools)
pathtracer_target_defaults(metal_fit)
target_link_libraries(metal_fit PRIVATE glm::glm)

# Shared validator harness (check.h, stats.h), depending on nothing but glm so the light validators need not link Embree or OpenEXR.
add_library(pathtracer_check OBJECT tools/check.cpp)
target_include_directories(pathtracer_check PUBLIC tools)
pathtracer_target_defaults(pathtracer_check NATIVE)
target_link_libraries(pathtracer_check PRIVATE glm::glm)

# Guards the harness's own invariants (NaN rejection, name-derived seeds): a broken assert macro would report success.
add_executable(check_selftest tools/check_selftest.cpp $<TARGET_OBJECTS:pathtracer_check>)
target_include_directories(check_selftest PRIVATE tools)
pathtracer_target_defaults(check_selftest NATIVE)
target_link_libraries(check_selftest PRIVATE glm::glm)

# Every validator links pathtracer_core and pathtracer_check; what each asserts is documented at the top of its own tools/*.cpp.
foreach(validator embree_validate bsdf_validate sampler_validate nee_validate gbuffer_validate
                  integrator_validate driver_validate io_validate display_validate filter_validate
                  camera_validate)
    add_executable(${validator} tools/${validator}.cpp $<TARGET_OBJECTS:pathtracer_check>)
    target_include_directories(${validator} PRIVATE tools)
    pathtracer_target_defaults(${validator} NATIVE IPO)
    target_link_libraries(${validator} PRIVATE pathtracer_core)
endforeach()

# integrator_validate and io_validate resolve real assets/ files against named instances -- the materialOverrides path.
target_compile_definitions(integrator_validate PRIVATE ASSET_ROOT_DIR="${CMAKE_SOURCE_DIR}/assets")
target_compile_definitions(io_validate PRIVATE ASSET_ROOT_DIR="${CMAKE_SOURCE_DIR}/assets")

# CIE 1931 module and the measured-metal fit, checked end to end against the Fresnel the renderer shades chrome.json with.
add_executable(colour_validate tools/colour_validate.cpp
               $<TARGET_OBJECTS:metal_fit_core> $<TARGET_OBJECTS:colorchecker_core> $<TARGET_OBJECTS:pathtracer_check>)
target_include_directories(colour_validate PRIVATE tools)
target_compile_definitions(colour_validate PRIVATE ASSET_ROOT_DIR="${CMAKE_SOURCE_DIR}/assets"
                           TOOLS_DATA_DIR="${CMAKE_SOURCE_DIR}/tools/data")
pathtracer_target_defaults(colour_validate NATIVE IPO)
target_link_libraries(colour_validate PRIVATE pathtracer_core)

# The headless render path through HeadlessRenderer: producer dispatch, per-AOV channel counts, agreement with renderPathTraced.
add_executable(api_validate tools/api_validate.cpp $<TARGET_OBJECTS:pathtracer_check>)
target_include_directories(api_validate PRIVATE tools)
target_compile_definitions(api_validate PRIVATE ASSET_ROOT_DIR="${CMAKE_SOURCE_DIR}/assets")
pathtracer_target_defaults(api_validate NATIVE IPO)
target_link_libraries(api_validate PRIVATE pathtracer_core)

# Headless before/after comparison renders. Not a correctness gate: its output is a picture, and a full render outlasts any validator.
add_executable(render_beauty tools/render_beauty.cpp $<TARGET_OBJECTS:pathtracer_check>)
target_include_directories(render_beauty PRIVATE tools)
target_compile_definitions(render_beauty PRIVATE ASSET_ROOT_DIR="${CMAKE_SOURCE_DIR}/assets")
pathtracer_target_defaults(render_beauty NATIVE IPO)
target_link_libraries(render_beauty PRIVATE pathtracer_core)

# Timing harness, not a correctness gate, so it stays out of the add_test loop. Same flags as the validators, timing the shipped codegen.
add_executable(gbuffer_bench tools/gbuffer_bench.cpp)
target_compile_definitions(gbuffer_bench PRIVATE ASSET_ROOT_DIR="${CMAKE_SOURCE_DIR}/assets")
pathtracer_target_defaults(gbuffer_bench NATIVE IPO)
target_link_libraries(gbuffer_bench PRIVATE pathtracer_core)

# Benchmark-log analysis and randomized interleaved A/B runner. Not a gate: its verdict is a confidence interval for a human.
add_executable(bench_compare tools/bench_compare.cpp)
target_include_directories(bench_compare PRIVATE include tools ${JSON_DIR})
pathtracer_target_defaults(bench_compare)


# All correctness tools run under `ctest --test-dir build`; previously buildable but unrun unless someone invoked them by hand.
enable_testing()

# Concurrent checks would each size a pool to hardware_concurrency(), so each declares PROCESSORS for ctest's job pool.
include(ProcessorCount)
ProcessorCount(PATHTRACER_HOST_PROCESSORS)
if(PATHTRACER_HOST_PROCESSORS EQUAL 0)
    set(PATHTRACER_HOST_PROCESSORS 1)
endif()
math(EXPR PATHTRACER_TEST_THREADS "${PATHTRACER_HOST_PROCESSORS} / 4")
if(PATHTRACER_TEST_THREADS LESS 1)
    set(PATHTRACER_TEST_THREADS 1)
endif()

include(${CMAKE_SOURCE_DIR}/cmake/PathtracerChecks.cmake)

# Image-level gates needing the real scene, glTF and material stack. Small and short: a gate that takes minutes stops being run.
add_test(NAME image.determinism
         COMMAND render_beauty --scene scenes/cornell.json --width 48 --height 36 --passes 8
                 --assert-deterministic)
set_tests_properties(image.determinism PROPERTIES LABELS "image;fast;exact" TIMEOUT 300)

# A G-buffer AOV runs no path-traced passes; reducing over that empty set segfaulted until the reporting learned to say so.
add_test(NAME image.gbuffer_aov
         COMMAND render_beauty --scene scenes/cornell.json --width 48 --height 36 --passes 4
                 --aov depth --out gbuffer_aov.png)
set_tests_properties(image.gbuffer_aov PROPERTIES LABELS "image;fast;exact" TIMEOUT 300)

# The same lane through a fisheye whose image circle sits inside the gate: every G-buffer AOV renders under either projection.
add_test(NAME image.gbuffer_aov_fisheye
         COMMAND render_beauty --scene scenes/cornell.json --width 48 --height 36 --passes 4
                 --aov depth --fisheye 4 --out gbuffer_aov_fisheye.png)
set_tests_properties(image.gbuffer_aov_fisheye PROPERTIES LABELS "image;fast;exact" TIMEOUT 300)

add_test(NAME image.convergence
         COMMAND render_beauty --scene scenes/cornell.json --width 48 --height 36 --passes 64
                 --assert-converged)
set_tests_properties(image.convergence PROPERTIES LABELS "image;slow;statistical" TIMEOUT 900
                     PROCESSORS ${PATHTRACER_TEST_THREADS})

# The comment budget is a gate, not a report, so it runs with the rest; its own lexer is tested separately.
if(Python3_Interpreter_FOUND)
    add_test(NAME style.comment_budget
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/comment_lint.py
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(style.comment_budget PROPERTIES LABELS "style;fast;exact" TIMEOUT 120)

    add_test(NAME style.comment_lint_selftest
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/test_comment_lint.py
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(style.comment_lint_selftest PROPERTIES LABELS "style;fast;exact" TIMEOUT 120)
endif()

# Migrated to tools/check.h: one ctest entry per check, discovered from the binary itself.
pathtracer_discover_checks(check_selftest)
pathtracer_discover_checks(sampler_validate)
pathtracer_discover_checks(embree_validate)
pathtracer_discover_checks(gbuffer_validate)
pathtracer_discover_checks(nee_validate)
pathtracer_discover_checks(integrator_validate)
pathtracer_discover_checks(bsdf_validate)
pathtracer_discover_checks(driver_validate)
pathtracer_discover_checks(io_validate)
pathtracer_discover_checks(display_validate)
pathtracer_discover_checks(colour_validate)
pathtracer_discover_checks(api_validate)
pathtracer_discover_checks(filter_validate)
pathtracer_discover_checks(camera_validate)

