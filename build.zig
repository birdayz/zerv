const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    // One module per src/ package (`packages`), and the `zerv` umbrella over all of them.
    const pkgs = addPackages(b, target, optimize);
    const core = pkgs.umbrella;
    b.modules.put(b.allocator, "zerv", core) catch @panic("OOM");
    // One test binary per file: the Zig 0.16 test runner runs a binary's tests one at a time
    // (ziglang/zig#15953), the build runner runs binaries in parallel
    // (docs/development.md, "Tests in parallel").
    const test_step = b.step("test", "Run native component and independent golden tests");
    // Test support compiled in ReleaseFast whatever the test mode: the golden fingerprints'
    // SHA-256 (tests/support/fast_sha256.zig).
    const support = b.addObject(.{
        .name = "zerv-test-support",
        .root_module = b.createModule(.{ .root_source_file = b.path("tests/support/fast_sha256.zig"), .target = target, .optimize = .ReleaseFast }),
    });
    // Optimized test binaries without debug info by default: DWARF doubles their LLVM time
    // (an empty ReleaseFast test: 14.4 s -> 6.4 s). Failures still print; Debug keeps traces.
    const symbols = b.option(bool, "test-symbols", "Keep debug info in optimized test binaries (stack traces; slower builds)") orelse false;
    // A fixed test seed makes a test run a cached result of its binary: an unchanged test is
    // not run again (the build runner's own `--seed` is random per invocation and would be
    // passed to every test binary). The CPU unit tests read no files at run time.
    const seed = b.option(u32, "test-seed", "Seed passed to the unit test binaries (default 0; results are cached per seed)") orelse 0;
    const opts: TestOptions = .{ .target = target, .support = support, .symbols = symbols, .seed = seed };
    addUnitTests(b, test_step, optimize, pkgs, opts);
    // The same binaries installed to zig-out/tests (tools/test_profile.py runs them alone).
    addTestInstall(b, b.step("test-install", "Install the unit test binaries to zig-out/tests (for tools/test_profile.py)"), optimize, pkgs, opts);
    b.default_step = test_step;

    // Every required check of AGENTS.md in one parallel build graph: zig fmt, the unit tests
    // in Debug and ReleaseFast, and the Python tests.
    const check_step = b.step("check", "Run all required checks (fmt, Debug and ReleaseFast tests, Python tests) in parallel");
    const fmt = b.addFmt(.{ .paths = &.{ "build.zig", "src", "bench", "tools", "tests" }, .check = true });
    check_step.dependOn(&fmt.step);
    for ([_]std.builtin.OptimizeMode{ .Debug, .ReleaseFast }) |mode| {
        addUnitTests(b, check_step, mode, if (mode == optimize) pkgs else addPackages(b, target, mode), opts);
    }
    const python = b.addSystemCommand(&.{ "python3", "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py" });
    python.has_side_effects = true;
    check_step.dependOn(&python.step);

    const gpu_tests = b.addTest(.{
        .use_llvm = true,
        .use_lld = true, // System GCC16 crt .sframe relocations require LLD on Zig0.16.
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/gpu.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{ .{ .name = "zerv", .module = core }, .{ .name = "gpu", .module = pkgs.get("gpu") }, .{ .name = "matvec", .module = pkgs.get("matvec") } },
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
        .imports = &.{ .{ .name = "gpu", .module = pkgs.get("gpu") }, .{ .name = "matvec", .module = pkgs.get("matvec") } },
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

    const decode_f16_benchmark = b.addExecutable(.{
        .name = "zerv-decode-f16-bench",
        .use_llvm = true,
        .use_lld = true,
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/decode_f16.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{ .{ .name = "zerv", .module = core }, .{ .name = "matvec_workload", .module = matvec_workload } },
        }),
    });
    decode_f16_benchmark.root_module.linkSystemLibrary("vulkan", .{ .use_pkg_config = .no });
    b.step("decode-f16-bench-build", "Build the f16 batched-decode projection benchmark (block 18e)")
        .dependOn(&b.addInstallArtifact(decode_f16_benchmark, .{}).step);

    const matvec_rows_benchmark = b.addExecutable(.{
        .name = "zerv-matvec-rows-bench",
        .use_llvm = true,
        .use_lld = true,
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/matvec_rows.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{ .{ .name = "zerv", .module = core }, .{ .name = "matvec_workload", .module = matvec_workload } },
        }),
    });
    matvec_rows_benchmark.root_module.linkSystemLibrary("vulkan", .{ .use_pkg_config = .no });
    b.step("matvec-rows-bench-build", "Build the multi-row decode projection benchmark (speculative verification)")
        .dependOn(&b.addInstallArtifact(matvec_rows_benchmark, .{}).step);

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

    const kv_quality = b.addExecutable(.{
        .name = "zerv-kv-quality",
        .use_llvm = true,
        .use_lld = true,
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/kv_quality.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{.{ .name = "zerv", .module = core }},
        }),
    });
    kv_quality.root_module.linkSystemLibrary("vulkan", .{ .use_pkg_config = .no });
    b.step("kv-quality-build", "Build the long-context KV precision logits tool (block 17c)")
        .dependOn(&b.addInstallArtifact(kv_quality, .{}).step);

    const spec_check = b.addExecutable(.{
        .name = "zerv-spec-check",
        .use_llvm = true,
        .use_lld = true,
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/spec_check.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{.{ .name = "zerv", .module = core }},
        }),
    });
    spec_check.root_module.linkSystemLibrary("vulkan", .{ .use_pkg_config = .no });
    b.step("spec-check-build", "Build the real-model speculative verification exactness check")
        .dependOn(&b.addInstallArtifact(spec_check, .{}).step);

    const batch_check = b.addExecutable(.{
        .name = "zerv-batch-check",
        .use_llvm = true,
        .use_lld = true,
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/batch_check.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{ .{ .name = "zerv", .module = core }, .{ .name = "matvec_workload", .module = matvec_workload } },
        }),
    });
    batch_check.root_module.linkSystemLibrary("vulkan", .{ .use_pkg_config = .no });
    b.step("batch-check-build", "Build the real-model batched decode exactness check (block 18b.2)")
        .dependOn(&b.addInstallArtifact(batch_check, .{}).step);

    const mtp_check = b.addExecutable(.{
        .name = "zerv-mtp-check",
        .use_llvm = true,
        .use_lld = true,
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/mtp_check.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{.{ .name = "zerv", .module = core }},
        }),
    });
    mtp_check.root_module.linkSystemLibrary("vulkan", .{ .use_pkg_config = .no });
    b.step("mtp-check-build", "Build the MTP draft-layer capture tool (gate 2 input for the FP64 reference)")
        .dependOn(&b.addInstallArtifact(mtp_check, .{}).step);

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

    const sampler_benchmark = b.addExecutable(.{
        .name = "zerv-sampler-bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/sampler.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "zerv", .module = core }},
        }),
    });
    b.step("sampler-bench-build", "Build the token sampler component benchmark")
        .dependOn(&b.addInstallArtifact(sampler_benchmark, .{}).step);

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

/// The src/ packages, one module each (`src/NAME/root.zig`), with the packages they import;
/// listed in dependency order (a package's dependencies come before it). Directional: no
/// cycles, as docs/architecture.md requires. The `zerv` umbrella (`src/root.zig`) re-exports
/// them all for executables that use several.
const Package = struct { name: []const u8, deps: []const []const u8 };
const packages = [_]Package{
    .{ .name = "quant", .deps = &.{} },
    .{ .name = "gpu", .deps = &.{} },
    .{ .name = "artifact", .deps = &.{} },
    .{ .name = "text", .deps = &.{} },
    .{ .name = "chat", .deps = &.{} },
    .{ .name = "matvec", .deps = &.{"gpu"} },
    .{ .name = "tokenizer", .deps = &.{ "text", "artifact" } },
    .{ .name = "model", .deps = &.{ "artifact", "gpu", "matvec" } },
    .{ .name = "session", .deps = &.{"chat"} },
    .{ .name = "serve", .deps = &.{ "chat", "session", "model", "tokenizer" } },
};

const Packages = struct {
    modules: [packages.len]*std.Build.Module,
    umbrella: *std.Build.Module,

    fn get(self: *const Packages, name: []const u8) *std.Build.Module {
        for (packages, 0..) |p, i| if (std.mem.eql(u8, p.name, name)) return self.modules[i];
        std.debug.panic("build.zig: unknown package '{s}'", .{name});
    }
};

fn addPackages(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) Packages {
    var result: Packages = undefined;
    for (packages, 0..) |p, i| {
        const m = b.createModule(.{ .root_source_file = b.path(b.fmt("src/{s}/root.zig", .{p.name})), .target = target, .optimize = optimize });
        for (p.deps) |d| {
            const at = for (packages[0..i], 0..) |q, j| {
                if (std.mem.eql(u8, q.name, d)) break j;
            } else std.debug.panic("build.zig: package '{s}' imports '{s}', which is not listed before it", .{ p.name, d });
            m.addImport(d, result.modules[at]);
        }
        result.modules[i] = m;
    }
    result.umbrella = b.createModule(.{ .root_source_file = b.path("src/root.zig"), .target = target, .optimize = optimize });
    for (packages, result.modules) |p, m| result.umbrella.addImport(p.name, m);
    return result;
}

/// Unit test files (CPU only), each its own binary, with the packages it imports (only
/// those: an edit rebuilds and reruns the tests of its dependents). `tests/test_build_lists.py`
/// checks that every test file is here or in the GPU aggregate (`tests/gpu.zig`).
const UnitTest = struct { name: []const u8, deps: []const []const u8 };
const unit_tests = [_]UnitTest{
    .{ .name = "quant", .deps = &.{"quant"} },
    .{ .name = "q4_1", .deps = &.{"quant"} },
    .{ .name = "q5_k", .deps = &.{"quant"} },
    .{ .name = "q6_k", .deps = &.{"quant"} },
    .{ .name = "gguf", .deps = &.{"artifact"} },
    .{ .name = "nfc", .deps = &.{"text"} },
    .{ .name = "chat", .deps = &.{"chat"} },
    .{ .name = "tokenizer_split", .deps = &.{"tokenizer"} },
    .{ .name = "tokenizer", .deps = &.{ "artifact", "tokenizer" } },
    .{ .name = "gpu_abi", .deps = &.{"gpu"} },
    .{ .name = "matvec", .deps = &.{ "gpu", "matvec" } },
    .{ .name = "model", .deps = &.{ "artifact", "matvec", "model" } },
    .{ .name = "session", .deps = &.{"session"} },
    .{ .name = "sampler", .deps = &.{"session"} },
    .{ .name = "sampler_nucleus", .deps = &.{"session"} },
    .{ .name = "prefix", .deps = &.{"session"} },
    .{ .name = "serve", .deps = &.{ "chat", "serve" } },
    .{ .name = "batcher", .deps = &.{"serve"} },
    .{ .name = "tools", .deps = &.{ "chat", "serve", "session" } },
};

const TestOptions = struct { target: std.Build.ResolvedTarget, support: *std.Build.Step.Compile, symbols: bool, seed: u32 };

fn unitTest(b: *std.Build, u: UnitTest, optimize: std.builtin.OptimizeMode, pkgs: Packages, opts: TestOptions) *std.Build.Step.Compile {
    const root = b.createModule(.{
        .root_source_file = b.path(b.fmt("tests/{s}.zig", .{u.name})),
        .target = opts.target,
        .optimize = optimize,
        .strip = optimize != .Debug and !opts.symbols,
    });
    for (u.deps) |d| root.addImport(d, pkgs.get(d));
    const t = b.addTest(.{ .name = b.fmt("test-{s}-{s}", .{ u.name, @tagName(optimize) }), .root_module = root });
    t.root_module.addObject(opts.support);
    return t;
}

fn addUnitTests(b: *std.Build, step: *std.Build.Step, optimize: std.builtin.OptimizeMode, pkgs: Packages, opts: TestOptions) void {
    for (unit_tests) |u| {
        const run = b.addRunArtifact(unitTest(b, u, optimize, pkgs, opts));
        var replaced = false;
        for (run.argv.items) |*arg| switch (arg.*) {
            .bytes => |bytes| if (std.mem.startsWith(u8, bytes, "--seed=")) {
                arg.* = .{ .bytes = b.fmt("--seed=0x{x}", .{opts.seed}) };
                replaced = true;
            },
            else => {},
        };
        if (!replaced) @panic("test runner arguments changed: no --seed to fix (build.zig addUnitTests)");
        step.dependOn(&run.step);
    }
}

fn addTestInstall(b: *std.Build, step: *std.Build.Step, optimize: std.builtin.OptimizeMode, pkgs: Packages, opts: TestOptions) void {
    for (unit_tests) |u| {
        const install = b.addInstallArtifact(unitTest(b, u, optimize, pkgs, opts), .{ .dest_dir = .{ .override = .{ .custom = "tests" } } });
        step.dependOn(&install.step);
    }
}
