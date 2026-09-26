# llama.cpp (ggml-org/llama.cpp b29c606e, version 0.4.1): libllama as the host package built it,
# against the separate ggml of this module (@ggml), an external test oracle only
# (docs/specs/hermetic-build.md). Sources as upstream src/CMakeLists.txt.
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_shared_library.bzl", "cc_shared_library")

package(default_visibility = ["//visibility:public"])

genrule(
    name = "llama_version_h",
    outs = ["src/llama-version.h"],
    cmd = "printf '#pragma once\\n#define LLAMA_VERSION \"0.4.1\"\\n#define LLAMA_COMMIT \"b29c606e\"\\n' > $@",
)

cc_library(
    name = "llama",
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
    deps = ["@ggml"],
)

cc_shared_library(
    name = "llama_so",
    shared_lib_name = "libllama.so.0.4.1",
    deps = [":llama"],
)
