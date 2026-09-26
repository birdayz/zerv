# ggml (ggml-org/ggml 456172ec, version 0.24.0), the revision of the host package the oracle
# fixtures were generated with: an external test oracle and benchmark reference only, never
# linked into zerv (docs/specs/hermetic-build.md). Sources and definitions as upstream
# src/CMakeLists.txt; backends registered statically (no GGML_BACKEND_DL), no OpenMP (ggml's
# own thread pool). The CPU backend is the `haswell` variant of GGML_CPU_ALL_VARIANTS (SSE4.2,
# AVX, F16C, FMA, AVX2, BMI2): the variant the host package selects on this machine (Zen 2).
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

# Backend registry with the statically registered backends.
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

# For Python oracles (ctypes).
cc_shared_library(
    name = "ggml_base_so",
    shared_lib_name = "libggml-base.so",
    deps = [":ggml_base"],
)
