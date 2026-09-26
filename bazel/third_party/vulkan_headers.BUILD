# Vulkan-Headers vulkan-sdk-1.4.357.0 (KhronosGroup/Vulkan-Headers): the C headers and the API
# registry (vk.xml), from which //src/gpu:vulkan generates its link stub.
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

exports_files(["registry/vk.xml"])

cc_library(
    name = "vulkan_headers",
    hdrs = glob([
        "include/vulkan/*.h",
        "include/vulkan/*.hpp",
        "include/vk_video/*.h",
    ]),
    includes = ["include"],
)
