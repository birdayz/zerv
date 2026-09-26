# Vulkan-Loader vulkan-sdk-1.4.357.0 (the version of the host's vulkan-icd-loader): the loader
# of the test-only GPU runtime (docs/specs/hermetic-build.md, phase 4). Sources and definitions
# as upstream loader/CMakeLists.txt for Linux, without assembly (unknown physical-device and
# device functions are unsupported; zerv uses none) and without X11/Wayland WSI. Every
# compiled-in search directory points nowhere: the loader finds only the ICD it is given
# (VK_DRIVER_FILES) and never a host driver, layer or settings file.
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_shared_library.bzl", "cc_shared_library")

package(default_visibility = ["//visibility:public"])

NOWHERE = "/nonexistent/zerv-hermetic-vulkan"

cc_library(
    name = "loader",
    srcs = [
        "loader/allocation.c",
        "loader/cJSON.c",
        "loader/debug_utils.c",
        "loader/extension_manual.c",
        "loader/gpa_helper.c",
        "loader/loader.c",
        "loader/loader_environment.c",
        "loader/loader_json.c",
        "loader/loader_linux.c",
        "loader/log.c",
        "loader/settings.c",
        "loader/terminator.c",
        "loader/trampoline.c",
        "loader/unknown_function_handling.c",
        "loader/wsi.c",
    ] + glob([
        "loader/*.h",
        "loader/generated/*.h",
    ]),
    textual_hdrs = ["loader/generated/vk_loader_extensions.c"],
    copts = [
        "-std=gnu99",
        "-O2",
        "-fvisibility=hidden",
    ],
    includes = [
        "loader",
        "loader/generated",
    ],
    linkopts = [
        "-ldl",
        "-lpthread",
    ],
    local_defines = [
        "_GNU_SOURCE",
        "HAVE_ALLOCA_H",
        "HAVE_REALPATH",
        "LOADER_ENABLE_LINUX_SORT",
        "VK_ENABLE_BETA_EXTENSIONS",
        'FALLBACK_CONFIG_DIRS=\\"%s\\"' % NOWHERE,
        'FALLBACK_DATA_DIRS=\\"%s\\"' % NOWHERE,
        'SYSCONFDIR=\\"%s\\"' % NOWHERE,
    ],
    deps = ["@vulkan_headers"],
    alwayslink = True,
)

cc_shared_library(
    name = "libvulkan",
    shared_lib_name = "libvulkan.so.1",
    deps = [":loader"],
)
