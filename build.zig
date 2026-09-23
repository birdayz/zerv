const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const core = b.addModule("zerv", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/root.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "zerv", .module = core }},
        }),
    });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run native component and independent golden tests");
    test_step.dependOn(&run_tests.step);
    b.default_step = test_step;

    const gpu_tests = b.addTest(.{
        .use_llvm = true,
        .use_lld = true, // System GCC16 crt .sframe relocations require LLD on Zig0.16.
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/gpu.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "zerv", .module = core }},
        }),
    });
    // The system Vulkan loader/ICD needs the OS libc startup/TLS contract.
    gpu_tests.root_module.link_libc = true;
    gpu_tests.root_module.linkSystemLibrary("vulkan", .{ .use_pkg_config = .no });
    const run_gpu_tests = b.addRunArtifact(gpu_tests);
    b.step("gpu-test", "Run explicit real-device Vulkan tests (requires system driver)")
        .dependOn(&run_gpu_tests.step);

    const gpu_workload = b.createModule(.{
        .root_source_file = b.path("tests/gpu_workload.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "zerv", .module = core }},
    });
    const gpu_benchmark = b.addExecutable(.{
        .name = "zerv-gpu-driver-bench",
        .use_llvm = true,
        .use_lld = true,
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/gpu_driver.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{ .{ .name = "zerv", .module = core }, .{ .name = "gpu_workload", .module = gpu_workload } },
        }),
    });
    gpu_benchmark.root_module.linkSystemLibrary("vulkan", .{ .use_pkg_config = .no });
    const install_gpu_benchmark = b.addInstallArtifact(gpu_benchmark, .{});
    b.step("gpu-driver-bench-build", "Build the matched native Vulkan driver diagnostic benchmark")
        .dependOn(&install_gpu_benchmark.step);

    const matvec_workload = b.createModule(.{
        .root_source_file = b.path("tests/matvec_workload.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "zerv", .module = core }},
    });
    const matvec_benchmark = b.addExecutable(.{
        .name = "zerv-gpu-matvec-bench",
        .use_llvm = true,
        .use_lld = true,
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/gpu_matvec.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{ .{ .name = "zerv", .module = core }, .{ .name = "matvec_workload", .module = matvec_workload } },
        }),
    });
    matvec_benchmark.root_module.linkSystemLibrary("vulkan", .{ .use_pkg_config = .no });
    const install_matvec_benchmark = b.addInstallArtifact(matvec_benchmark, .{});
    b.step("gpu-matvec-bench-build", "Build native resident packed-weight GPU matvec benchmark")
        .dependOn(&install_matvec_benchmark.step);

    const server = b.addExecutable(.{
        .name = "zerv",
        .use_llvm = true,
        .use_lld = true,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{.{ .name = "zerv", .module = core }},
        }),
    });
    server.root_module.linkSystemLibrary("vulkan", .{ .use_pkg_config = .no });
    b.step("server", "Build the native OpenAI Chat Completions v1 server")
        .dependOn(&b.addInstallArtifact(server, .{}).step);

    const model_capture = b.addExecutable(.{
        .name = "zerv-model-capture",
        .use_llvm = true,
        .use_lld = true,
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/model_capture.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{.{ .name = "zerv", .module = core }},
        }),
    });
    model_capture.root_module.linkSystemLibrary("vulkan", .{ .use_pkg_config = .no });
    b.step("model-capture-build", "Build the native model intermediate-capture verification tool")
        .dependOn(&b.addInstallArtifact(model_capture, .{}).step);

    const gemm_benchmark = b.addExecutable(.{
        .name = "zerv-gemm-bench",
        .use_llvm = true,
        .use_lld = true,
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/gemm.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{ .{ .name = "zerv", .module = core }, .{ .name = "matvec_workload", .module = matvec_workload } },
        }),
    });
    gemm_benchmark.root_module.linkSystemLibrary("vulkan", .{ .use_pkg_config = .no });
    b.step("gemm-bench-build", "Build the prefill GEMM component benchmark")
        .dependOn(&b.addInstallArtifact(gemm_benchmark, .{}).step);

    const model_profile = b.addExecutable(.{
        .name = "zerv-model-profile",
        .use_llvm = true,
        .use_lld = true,
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/profile_model.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{ .{ .name = "zerv", .module = core }, .{ .name = "matvec_workload", .module = matvec_workload } },
        }),
    });
    model_profile.root_module.linkSystemLibrary("vulkan", .{ .use_pkg_config = .no });
    b.step("model-profile-build", "Build the per-phase model GPU-time profiler")
        .dependOn(&b.addInstallArtifact(model_profile, .{}).step);

    const prefix_check = b.addExecutable(.{
        .name = "zerv-prefix-check",
        .use_llvm = true,
        .use_lld = true,
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/prefix_check.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{.{ .name = "zerv", .module = core }},
        }),
    });
    prefix_check.root_module.linkSystemLibrary("vulkan", .{ .use_pkg_config = .no });
    b.step("prefix-check-build", "Build the real-model prefix-cache exactness check")
        .dependOn(&b.addInstallArtifact(prefix_check, .{}).step);

    const kernel_chain = b.addExecutable(.{
        .name = "zerv-kernel-chain",
        .use_llvm = true,
        .use_lld = true,
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/kernel_chain.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{ .{ .name = "zerv", .module = core }, .{ .name = "matvec_workload", .module = matvec_workload } },
        }),
    });
    kernel_chain.root_module.linkSystemLibrary("vulkan", .{ .use_pkg_config = .no });
    b.step("kernel-chain-build", "Build the dependent-kernel chain microbenchmark (research)")
        .dependOn(&b.addInstallArtifact(kernel_chain, .{}).step);

    const coopmat_probe = b.addExecutable(.{
        .name = "zerv-coopmat-probe",
        .use_llvm = true,
        .use_lld = true,
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/coopmat_probe.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{ .{ .name = "zerv", .module = core }, .{ .name = "matvec_workload", .module = matvec_workload } },
        }),
    });
    coopmat_probe.root_module.linkSystemLibrary("vulkan", .{ .use_pkg_config = .no });
    b.step("coopmat-probe-build", "Build the cooperative-matrix numerics probe (research)")
        .dependOn(&b.addInstallArtifact(coopmat_probe, .{}).step);

    const inspect = b.addExecutable(.{
        .name = "zerv-inspect",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/inspect.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "zerv", .module = core }},
        }),
    });
    const install_inspect = b.addInstallArtifact(inspect, .{});
    b.step("inspect-build", "Build native GGUF artifact inspection")
        .dependOn(&install_inspect.step);

    const tokenizer_benchmark = b.addExecutable(.{
        .name = "zerv-tokenizer-bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/tokenizer.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "zerv", .module = core }},
        }),
    });
    const install_tokenizer_benchmark = b.addInstallArtifact(tokenizer_benchmark, .{});
    b.step("tokenizer-bench-build", "Build the full native tokenizer checker/benchmark")
        .dependOn(&install_tokenizer_benchmark.step);

    const split_benchmark = b.addExecutable(.{
        .name = "zerv-split-bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/split.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "zerv", .module = core }},
        }),
    });
    const install_split_benchmark = b.addInstallArtifact(split_benchmark, .{});
    b.step("split-bench-build", "Build the native Qwen text-split benchmark")
        .dependOn(&install_split_benchmark.step);

    const nfc_benchmark = b.addExecutable(.{
        .name = "zerv-nfc-bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/nfc.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "zerv", .module = core }},
        }),
    });
    const install_nfc_benchmark = b.addInstallArtifact(nfc_benchmark, .{});
    b.step("nfc-bench-build", "Build the native Unicode-9 NFC benchmark")
        .dependOn(&install_nfc_benchmark.step);

    const chat_benchmark = b.addExecutable(.{
        .name = "zerv-chat-bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/chat.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "zerv", .module = core }},
        }),
    });
    const install_chat_benchmark = b.addInstallArtifact(chat_benchmark, .{});
    b.step("chat-bench-build", "Build the native text chat-template benchmark")
        .dependOn(&install_chat_benchmark.step);

    const gguf_benchmark = b.addExecutable(.{
        .name = "zerv-gguf-bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/gguf.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "zerv", .module = core }},
        }),
    });
    const install_gguf_benchmark = b.addInstallArtifact(gguf_benchmark, .{});
    b.step("gguf-bench-build", "Build the native GGUF parse/index benchmark")
        .dependOn(&install_gguf_benchmark.step);

    const model_quant_benchmark = b.addExecutable(.{
        .name = "zerv-model-quant-bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/model_quant.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "zerv", .module = core }},
        }),
    });
    const install_model_quant_benchmark = b.addInstallArtifact(model_quant_benchmark, .{});
    b.step("model-quant-bench-build", "Build actual-model quant decoder benchmark")
        .dependOn(&install_model_quant_benchmark.step);

    const benchmark = b.addExecutable(.{
        .name = "zerv-quant-bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/quant.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "zerv", .module = core }},
        }),
    });
    const install_benchmark = b.addInstallArtifact(benchmark, .{});
    b.step("bench-build", "Build the standalone quant component benchmark")
        .dependOn(&install_benchmark.step);
}
