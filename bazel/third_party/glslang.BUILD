# glslang (KhronosGroup/glslang vulkan-sdk-1.4.357.0, 168d452a), the GLSL front end of glslc.
# Sources and defines follow upstream BUILD.gn (glslang_sources_common with enable_opt, no
# HLSL: zerv's shaders are GLSL).
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_python//python:py_binary.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "build_info",
    srcs = ["build_info.py"],
)

genrule(
    name = "build_info_h",
    srcs = [
        "CHANGES.md",
        "build_info.h.tmpl",
    ],
    outs = ["include/glslang/build_info.h"],
    cmd = "$(location :build_info) $$(dirname $(location CHANGES.md)) -i $(location build_info.h.tmpl) -o $@",
    tools = [":build_info"],
)

COPTS = [
    "-Wno-conversion",
    "-Wno-extra-semi",
    "-Wno-ignored-qualifiers",
    "-Wno-implicit-fallthrough",
    "-Wno-inconsistent-missing-override",
    "-Wno-missing-field-initializers",
    "-Wno-newline-eof",
    "-Wno-sign-compare",
    "-Wno-suggest-destructor-override",
    "-Wno-suggest-override",
    "-Wno-unused-variable",
]

cc_library(
    name = "glslang",
    srcs = glob(
        [
            "SPIRV/*.cpp",
            "SPIRV/*.h",
            "glslang/GenericCodeGen/*.cpp",
            "glslang/MachineIndependent/**/*.cpp",
            "glslang/MachineIndependent/**/*.h",
            "glslang/OSDependent/Unix/*.cpp",
        ],
    ) + [
        "glslang/HLSL/hlslParseHelper.h",
        "glslang/HLSL/hlslParseables.h",
        "glslang/HLSL/hlslScanContext.h",
        "glslang/HLSL/hlslTokens.h",
    ],
    hdrs = glob([
        "SPIRV/*.h",
        "SPIRV/spirv.hpp11",
        "glslang/Include/*.h",
        "glslang/OSDependent/*.h",
        "glslang/Public/*.h",
        "glslang/MachineIndependent/*.h",
    ]) + [":build_info_h"],
    copts = COPTS,
    defines = [
        "ENABLE_SPIRV=1",
        "ENABLE_OPT=1",
    ],
    includes = [
        ".",
        "include",
    ],
    local_defines = ["GLSLANG_OSINCLUDE_UNIX"],
    deps = [
        "@spirv_tools//:spirv_tools_opt",
        "@spirv_tools//:spirv_tools",
    ],
)

cc_library(
    name = "default_resource_limits",
    srcs = ["glslang/ResourceLimits/ResourceLimits.cpp"],
    hdrs = ["glslang/Public/ResourceLimits.h"],
    copts = COPTS,
    includes = ["."],
    deps = [":glslang"],
)
