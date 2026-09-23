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
