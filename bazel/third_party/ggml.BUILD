# ggml (ggml-org/ggml 456172ec, version 0.24.0), the revision of the host package the oracle
# fixtures were generated with: an external test oracle and benchmark reference only, never
# linked into zerv (docs/specs/hermetic-build.md). Sources and definitions as upstream
# src/CMakeLists.txt; backends registered statically (no GGML_BACKEND_DL), no OpenMP (ggml's
# own thread pool). The CPU backend is the `haswell` variant of GGML_CPU_ALL_VARIANTS (SSE4.2,
# AVX, F16C, FMA, AVX2, BMI2): the variant the host package selects on this machine (Zen 2).
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_shared_library.bzl", "cc_shared_library")

package(default_visibility = ["//visibility:public"])

genrule(
    name = "ggml_version_h",
    outs = ["src/ggml-version.h"],
    cmd = "printf '#pragma once\\n#define GGML_VERSION \"0.24.0\"\\n#define GGML_COMMIT \"456172ec\"\\n' > $@",
)

COPTS = ["-O3", "-DNDEBUG", "-fno-fast-math"]
C11 = ["-std=gnu11"]
CXX17 = ["-std=gnu++17"]

cc_library(
    name = "ggml_base",
    srcs = [
        "src/ggml.c",
        "src/ggml.cpp",
        "src/ggml-alloc.c",
        "src/ggml-backend.cpp",
        "src/ggml-backend-meta.cpp",
        "src/ggml-opt.cpp",
        "src/ggml-threading.cpp",
        "src/ggml-quants.c",
        "src/gguf.cpp",
        ":ggml_version_h",
    ] + glob([
        "src/*.h",
        "src/ggml-cpu/*.h",
    ]),
    hdrs = glob(["include/*.h"]),
    conlyopts = C11,
    copts = COPTS,
    cxxopts = CXX17,
    includes = [
        "include",
        "src",
    ],
    linkopts = [
        "-lm",
        "-lpthread",
    ],
)

# The CPU backend (haswell variant).
X86_DEFINES = [
    "GGML_SSE42",
    "GGML_AVX",
    "GGML_F16C",
    "GGML_FMA",
    "GGML_AVX2",
    "GGML_BMI2",
    "GGML_USE_LLAMAFILE",
]

cc_library(
    name = "ggml_cpu",
    srcs = glob(
        [
            "src/ggml-cpu/*.c",
            "src/ggml-cpu/*.cpp",
            "src/ggml-cpu/*.h",
            "src/ggml-cpu/amx/*.cpp",
            "src/ggml-cpu/amx/*.h",
            "src/ggml-cpu/llamafile/*.cpp",
            "src/ggml-cpu/llamafile/*.h",
            "src/ggml-cpu/arch/x86/quants.c",
            "src/ggml-cpu/arch/x86/repack.cpp",
        ],
    ),
    conlyopts = C11,
    copts = COPTS + [
        "-msse4.2",
        "-mavx",
        "-mf16c",
        "-mfma",
        "-mavx2",
        "-mbmi2",
    ],
    cxxopts = CXX17,
    includes = ["src/ggml-cpu"],
    local_defines = X86_DEFINES + ["GGML_BACKEND_BUILD"],
    deps = [":ggml_base"],
)

# Backend registry with the statically registered backends: `ggml` the CPU backend only,
# `ggml_vulkan_all` also the Vulkan backend (programs that use it need a Vulkan loader at run
# time).
cc_library(
    name = "ggml",
    srcs = [
        "src/ggml-backend-dl.cpp",
        "src/ggml-backend-reg.cpp",
    ] + glob(["src/*.h"]),
    copts = COPTS,
    cxxopts = CXX17,
    local_defines = ["GGML_USE_CPU"],
    deps = [
        ":ggml_base",
        ":ggml_cpu",
    ],
)

cc_library(
    name = "ggml_vulkan_all",
    srcs = [
        "src/ggml-backend-dl.cpp",
        "src/ggml-backend-reg.cpp",
    ] + glob(["src/*.h"]),
    copts = COPTS,
    cxxopts = CXX17,
    local_defines = [
        "GGML_USE_CPU",
        "GGML_USE_VULKAN",
    ],
    deps = [
        ":ggml_base",
        ":ggml_cpu",
        ":ggml_vulkan",
    ],
)

# The Vulkan backend. Its shaders: vulkan-shaders-gen runs the source-built glslc for every
# variant of each vulkan-shaders/*.comp (one action per file) and embeds the SPIR-V in a .cpp,
# as upstream src/ggml-vulkan/CMakeLists.txt. All seven optional GLSL extensions are supported
# by the pinned glslc (checked with vulkan-shaders/feature-tests on 2026-09-26).
VK_FEATURES = [
    "GGML_VULKAN_COOPMAT_GLSLC_SUPPORT",
    "GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT",
    "GGML_VULKAN_COOPMAT2_DECODE_VECTOR_GLSLC_SUPPORT",
    "GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT",
    "GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT",
    "GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT",
    "GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT",
]

cc_binary(
    name = "vulkan_shaders_gen",
    srcs = ["src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp"],
    copts = ["-std=gnu++17"],
    linkopts = ["-lpthread"],
    local_defines = VK_FEATURES,
)

VK_DIR = "src/ggml-vulkan/vulkan-shaders/"

VK_INCLUDES = glob([VK_DIR + "*.glsl"])

genrule(
    name = "ggml_vulkan_shaders_hpp",
    outs = ["ggml-vulkan-shaders.hpp"],
    cmd = "$(execpath :vulkan_shaders_gen) --output-dir $$(mktemp -d) --target-hpp $@",
    tools = [":vulkan_shaders_gen"],
)

[
    genrule(
        name = "vk_" + comp[len(VK_DIR):-len(".comp")],
        srcs = [comp] + VK_INCLUDES,
        outs = ["vk/" + comp[len(VK_DIR):] + ".cpp"],
        cmd = "$(execpath :vulkan_shaders_gen) --glslc $(execpath @shaderc//:glslc) --source $(location %s) --output-dir $$(mktemp -d) --target-hpp ggml-vulkan-shaders.hpp --target-cpp $@ && rm -f $@.d" % comp,
        tools = [
            ":vulkan_shaders_gen",
            "@shaderc//:glslc",
        ],
    )
    for comp in glob([VK_DIR + "*.comp"])
]

cc_library(
    name = "ggml_vulkan",
    srcs = [
        "src/ggml-vulkan/ggml-vulkan.cpp",
        ":ggml_vulkan_shaders_hpp",
    ] + ["vk/" + comp[len(VK_DIR):] + ".cpp" for comp in glob([VK_DIR + "*.comp"])] + glob(["src/*.h"]),
    copts = COPTS + ["-Wno-deprecated-declarations"],
    cxxopts = CXX17,
    includes = ["."],
    local_defines = VK_FEATURES + ["GGML_BACKEND_BUILD"],
    # Linked against the link stub libvulkan.so.1 (zerv's //src/gpu:vulkan): the dynamic linker
    # loads the real loader at run time.
    additional_linker_inputs = ["@@//src/gpu:vulkan"],
    linkopts = ["$(location @@//src/gpu:vulkan)"],
    deps = [
        ":ggml_base",
        ":ggml_cpu",
        "@spirv_headers//:spirv_cpp_headers",
        "@vulkan_headers",
    ],
)

# For Python oracles (ctypes).
cc_shared_library(
    name = "ggml_base_so",
    shared_lib_name = "libggml-base.so",
    deps = [":ggml_base"],
)
