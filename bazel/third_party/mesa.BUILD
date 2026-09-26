# Mesa 26.2.3 (the version of the host's vulkan-radeon), RADV only: the user-mode Vulkan
# driver of the GPU tests and GPU oracle captures (docs/specs/hermetic-build.md, phase 4; test
# only, zerv ships nothing of it). Mesa's own meson build runs in one hermetic action
# (//bazel:meson.bzl) with every other driver, platform and optional dependency disabled.
load("@@//bazel:meson.bzl", "meson_project")

package(default_visibility = ["//visibility:public"])

meson_project(
    name = "radv",
    options = [
        "platforms=",
        "gallium-drivers=",
        "vulkan-drivers=amd",
        "vulkan-layers=",
        "tools=",
        "video-codecs=",
        "llvm=disabled",
        "shared-llvm=disabled",
        "amd-use-llvm=false",
        "opengl=false",
        "gles1=disabled",
        "gles2=disabled",
        "glx=disabled",
        "egl=disabled",
        "gbm=disabled",
        "glvnd=disabled",
        "expat=disabled",
        "xmlconfig=disabled",
        # No disk shader cache (it needs zlib/zstd): no state between test runs; RADV's
        # pipeline binaries (VK_KHR_pipeline_binary) do not depend on it.
        "shader-cache=disabled",
        "zlib=disabled",
        "zstd=disabled",
        "valgrind=disabled",
        "libunwind=disabled",
        "lmsensors=disabled",
        "display-info=disabled",
        "spirv-tools=disabled",
        "microsoft-clc=disabled",
        "intel-rt=disabled",
        "android-libbacktrace=disabled",
        "perfetto=false",
        "build-tests=false",
        "html-docs=disabled",
        "allow-fallback-for=libdrm",
        # libdrm from its subproject (below), never looked up on the host, linked statically:
        # the driver loads no libdrm of the host.
        "force_fallback_for=libdrm",
        "libdrm:default_library=static",

        # Compiled-in install paths (libdrm's amdgpu.ids, ...) point nowhere on the host; the
        # runtime passes the device-name table by AMDGPU_ASIC_ID_TABLE_PATHS.
        "prefix=/nonexistent/zerv-hermetic-vulkan",
        # The driver stays mapped until the process exits: unloaded by the loader at
        # vkDestroyInstance, its statically linked C++ runtime's exit-time destructors crashed
        # the process after the last test (the host's GCC build links the shared libstdc++).
        # No debug sections either: those of the statically linked C++ runtime hold absolute
        # paths of the build (output base, scratch directory), which made the driver and its
        # build ID differ between two otherwise identical builds; the code sections did not.
        # (meson's strip option applies only at install.)
        "cpp_link_args=['-Wl,-z,nodelete', '-Wl,--strip-debug']",
        "c_link_args=['-Wl,-z,nodelete', '-Wl,--strip-debug']",
    ],
    outs = [
        "src/amd/vulkan/libvulkan_radeon.so",
        "src/amd/vulkan/radeon_icd.x86_64.json",
    ],
    root = "meson.build",
    srcs = glob(["**"], exclude = ["subprojects/**"]) + glob(["subprojects/*.wrap"]),
    subproject_srcs = ["@libdrm//:all"],
    subprojects = {"@libdrm//:meson.build": "libdrm-2.4.133"},
    targets = [
        "src/amd/vulkan/libvulkan_radeon.so",
        "src/amd/vulkan/radeon_icd.x86_64.json",
    ],
    tools = {
        "@glslang//:glslangValidator": "glslangValidator",
        "@@//bazel:nm_unavailable": "nm",
    },
)
