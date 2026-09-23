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
