# Vulkan-Tools vulkan-sdk-1.4.357.0: vulkaninfo only (docs/specs/hermetic-build.md; a
# development tool that benchmark manifests use to record the device and driver). Sources and
# definitions as upstream vulkaninfo/CMakeLists.txt for Linux, without window-system support
# (no X11/Wayland libraries): it reports the display-less devices. It loads the Vulkan loader
# with dlopen at run time (VK_NO_PROTOTYPES): the host's, or the test-only GPU runtime's.
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "vulkaninfo_headers",
    hdrs = [
        "vulkaninfo/generated/vulkaninfo.hpp",
        "vulkaninfo/outputprinter.h",
        "vulkaninfo/vulkaninfo.h",
        "vulkaninfo/vulkaninfo_functions.h",
    ],
    includes = [
        "vulkaninfo",
        "vulkaninfo/generated",
    ],
)

cc_binary(
    name = "vulkaninfo",
    srcs = ["vulkaninfo/vulkaninfo.cpp"],
    copts = [
        "-std=c++17",
        "-O2",
    ],
    linkopts = ["-ldl"],
    local_defines = [
        "VK_ENABLE_BETA_EXTENSIONS",
        "VK_NO_PROTOTYPES",
        "VK_USE_PLATFORM_DISPLAY_KHR",
    ],
    deps = [
        ":vulkaninfo_headers",
        "@vulkan_headers",
    ],
)
