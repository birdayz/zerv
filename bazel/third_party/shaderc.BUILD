# shaderc v2026.3 (google/shaderc): libshaderc_util, libshaderc and glslc, as upstream
# CMakeLists.txt; built against glslang and SPIRV-Tools from this module.
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "shaderc_util",
    srcs = [
        "libshaderc_util/src/args.cc",
        "libshaderc_util/src/compiler.cc",
        "libshaderc_util/src/file_finder.cc",
        "libshaderc_util/src/io_shaderc.cc",
        "libshaderc_util/src/message.cc",
        "libshaderc_util/src/resources.cc",
        "libshaderc_util/src/shader_stage.cc",
        "libshaderc_util/src/spirv_tools_wrapper.cc",
        "libshaderc_util/src/version_profile.cc",
    ],
    hdrs = glob(["libshaderc_util/include/libshaderc_util/*"]),
    includes = ["libshaderc_util/include"],
    deps = [
        "@glslang",
        "@spirv_tools",
        "@spirv_tools//:spirv_tools_opt",
    ],
)

cc_library(
    name = "shaderc",
    srcs = [
        "libshaderc/src/shaderc.cc",
        "libshaderc/src/shaderc_private.h",
    ],
    hdrs = glob(["libshaderc/include/shaderc/*.h*"]),
    includes = ["libshaderc/include"],
    local_defines = ["SHADERC_IMPLEMENTATION"],
    deps = [
        ":shaderc_util",
        "@glslang",
        "@glslang//:default_resource_limits",
        "@spirv_headers//:spirv_cpp_headers",
        "@spirv_tools",
    ],
)

# What `glslc --version` prints (utils/update_build_version.py derives it from git; the
# archives carry no git metadata, so the pinned revisions are written here).
genrule(
    name = "build_version_inc",
    outs = ["build-version.inc"],
    cmd = "printf '%s\\n' '\"shaderc v2026.3\\n\"' '\"spirv-tools vulkan-sdk-1.4.357.0 9a49b0883b9b635689a85b5647dbfcb223268151\\n\"' '\"glslang vulkan-sdk-1.4.357.0 168d452a4f460d24b588fed08477a81c44ee27a1\\n\"' > $@",
)

cc_library(
    name = "build_version",
    hdrs = [":build_version_inc"],
    includes = ["."],
)

cc_binary(
    name = "glslc",
    srcs = glob(
        [
            "glslc/src/*.cc",
            "glslc/src/*.h",
        ],
        exclude = ["glslc/src/*_test.cc"],
    ),
    includes = ["glslc/src"],
    deps = [
        ":build_version",
        ":shaderc",
        ":shaderc_util",
        "@glslang",
        "@spirv_tools",
    ],
)
