# llama.cpp (ggml-org/llama.cpp b29c606e, version 0.4.1): libllama as the host package built it,
# against the separate ggml of this module (@ggml), an external test oracle only
# (docs/specs/hermetic-build.md). Sources as upstream src/CMakeLists.txt.
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_shared_library.bzl", "cc_shared_library")

package(default_visibility = ["//visibility:public"])

genrule(
    name = "llama_version_h",
    outs = ["src/llama-version.h"],
    cmd = "printf '#pragma once\\n#define LLAMA_VERSION \"0.4.1\"\\n#define LLAMA_COMMIT \"b29c606e\"\\n' > $@",
)

# `llama` with ggml's CPU backend registry; `llama_vulkan` with the registry that also has the
# Vulkan backend (@ggml//:ggml_vulkan_all), for the GPU oracles (the model oracle, the
# batch capture). The same sources; the registry decides which backends exist.
[
    cc_library(
        name = name,
        srcs = glob([
            "src/*.cpp",
            "src/*.h",
            "src/models/*.cpp",
            "src/models/*.h",
        ]) + [":llama_version_h"],
        hdrs = glob(["include/*.h"]),
        copts = [
            "-O3",
            "-DNDEBUG",
            "-std=gnu++17",
        ],
        includes = [
            "include",
            "src",
        ],
        deps = [registry],
    )
    for name, registry in [
        ("llama", "@ggml"),
        ("llama_vulkan", "@ggml//:ggml_vulkan_all"),
    ]
]

cc_shared_library(
    name = "llama_so",
    shared_lib_name = "libllama.so.0.4.1",
    deps = [":llama"],
)

# llama-server with the Vulkan backend: the serving competitor built in the graph
# (docs/specs/hermetic-build.md, phase 5), as upstream's CMake (common/, tools/mtmd/,
# tools/server/, vendor/) with its Linux defaults (LLAMA_SUBPROCESS on, so MTMD_VIDEO on)
# except: no OpenSSL (HTTPS model downloads; the benchmark serves a local file) and no
# embedded web UI (npm; upstream's own no-asset build). Release flags as upstream
# (-O3 -DNDEBUG). The build info names the pinned commit and upstream's build number of it.
SERVER_COPTS = [
    "-O3",
    "-DNDEBUG",
    "-std=gnu++17",
]

genrule(
    name = "build_info_cpp",
    srcs = ["common/build-info.cpp.in"],
    outs = ["common/build-info.cpp"],
    cmd = "sed -e 's/@LLAMA_BUILD_NUMBER@/10964/' -e 's/@LLAMA_BUILD_COMMIT@/b29c606e28/' " +
          "-e 's/@BUILD_COMPILER@/Clang (Zig, hermetic_cc_toolchain)/' -e 's/@BUILD_TARGET@/x86_64-linux-gnu/' $< > $@",
)

cc_library(
    name = "common_base",
    srcs = [":build_info_cpp"],
    hdrs = ["common/build-info.h"],
    copts = SERVER_COPTS,
    includes = ["common"],
)

cc_library(
    name = "vendor_headers",
    hdrs = glob([
        "vendor/nlohmann/*.hpp",
        "vendor/sheredom/*.h",
        "vendor/stb/*.h",
        "vendor/miniaudio/*.h",
    ]),
    includes = ["vendor"],
)

cc_library(
    name = "vendor_hash",
    srcs = [
        "vendor/hash/hash.cpp",
        "vendor/hash/sha256/sha256.c",
        "vendor/hash/sha256/sha256.h",
        "vendor/hash/xxhash/xxhash.c",
    ],
    hdrs = [
        "vendor/hash/hash.h",
        "vendor/hash/rotate-bits/rotate-bits.h",
        "vendor/hash/xxhash/xxhash.h",
    ],
    copts = [
        "-O3",
        "-DNDEBUG",
        "-w",
    ],
    cxxopts = ["-std=gnu++17"],
    # vendor/hash: its sources include "rotate-bits/..." (upstream's private include directory).
    includes = [
        "vendor",
        "vendor/hash",
    ],
)

cc_library(
    name = "httplib",
    srcs = ["vendor/cpp-httplib/httplib.cpp"],
    hdrs = ["vendor/cpp-httplib/httplib.h"],
    copts = SERVER_COPTS + ["-w"],
    includes = ["vendor"],
    local_defines = [
        "CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH=1048576",
        "CPPHTTPLIB_LISTEN_BACKLOG=512",
        "CPPHTTPLIB_REQUEST_URI_MAX_LENGTH=32768",
        "CPPHTTPLIB_TCP_NODELAY=1",
    ],
    linkopts = ["-lpthread"],
)

cc_library(
    name = "common",
    srcs = glob([
        "common/*.cpp",
        "common/*.h",
        "common/*.hpp",
        "common/jinja/*.cpp",
        "common/jinja/*.h",
        "common/parsers/*.cpp",
        "common/parsers/*.h",
    ]) + ["src/llama-ext.h"],
    copts = SERVER_COPTS,
    defines = ["LLAMA_SUBPROCESS"],
    includes = ["common"],
    deps = [
        ":common_base",
        ":httplib",
        ":llama_vulkan",
        ":vendor_headers",
    ],
)

cc_library(
    name = "mtmd",
    srcs = glob([
        "tools/mtmd/*.cpp",
        "tools/mtmd/*.h",
        "tools/mtmd/models/*.cpp",
        "tools/mtmd/models/*.h",
        "tools/mtmd/debug/*.h",
    ], exclude = [
        "tools/mtmd/mtmd-cli.cpp",
        "tools/mtmd/deprecation-warning.cpp",
    ]) + ["src/llama-ext.h"],
    copts = SERVER_COPTS + ["-Wno-cast-qual"],
    includes = ["tools/mtmd"],
    local_defines = ["MTMD_VIDEO"],
    deps = [
        ":llama_vulkan",
        ":vendor_hash",
        ":vendor_headers",
    ],
)

# The UI asset table without assets (scripts/ui-assets.cmake when no UI is provisioned).
genrule(
    name = "ui_sources",
    srcs = [
        "tools/ui/ui.h.in",
        "tools/ui/ui.cpp.in",
    ],
    outs = [
        "ui/ui.h",
        "ui/ui.cpp",
    ],
    cmd = """
sed -e 's|#cmakedefine LLAMA_UI_HAS_ASSETS 1|/* #undef LLAMA_UI_HAS_ASSETS */|' -e 's/@N_ASSETS@/0/g' $(location tools/ui/ui.h.in) > $(location ui/ui.h)
sed -e 's/@N_ASSETS@/0/g' -e 's/@ASSET_ARRAYS@//' -e 's/@ASSET_TABLE@//' -e 's/@USE_GZIP@/false/' $(location tools/ui/ui.cpp.in) > $(location ui/ui.cpp)
""",
)

cc_library(
    name = "llama_ui",
    srcs = ["ui/ui.cpp"],
    hdrs = ["ui/ui.h"],
    copts = SERVER_COPTS,
    includes = ["ui"],
)

cc_binary(
    name = "llama-server",
    srcs = glob([
        "tools/server/*.cpp",
        "tools/server/*.h",
    ]),
    copts = SERVER_COPTS,
    deps = [
        ":common",
        ":httplib",
        ":llama_ui",
        ":mtmd",
    ],
)
