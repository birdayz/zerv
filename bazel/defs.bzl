"""zerv's macros: unit tests in both required modes, Vulkan-linked executables."""

load("@rules_zig//zig:defs.bzl", "zig_binary", "zig_configure_test", "zig_test")

def zerv_test(name, deps = [], srcs = [], embeds = [], data = [], sha = False, **kwargs):
    """A unit test file `tests/<name>.zig`, as `name` (Debug) and `name_release_fast`.

    Args:
      name: the test file's stem.
      deps: the src/ packages it imports (only those: an edit re-runs its dependents).
      srcs: helper files it imports by path (tests/*.zig).
      embeds: files it embeds (`@embedFile`), relative to tests/.
      data: files it opens at run time (repository-relative paths from the runfiles root).
      sha: it fingerprints goldens with `fast_sha256.zig` (the ReleaseFast SHA-256
        library, //tests/support).
      **kwargs: passed to both tests (size, tags, ...).
    """
    all_srcs = list(srcs)
    all_deps = list(deps)
    if sha:
        all_srcs.append("fast_sha256.zig")
        all_deps.append("//tests/support:fast_sha256")
    zig_test(
        name = name,
        main = name + ".zig",
        srcs = all_srcs,
        extra_srcs = embeds,
        data = data,
        deps = all_deps,
        **kwargs
    )

    # ReleaseFast without debug info: DWARF doubles its LLVM time
    # (docs/bench/2026-09-26-test-parallelism.md).
    release_kwargs = dict(kwargs)
    release_kwargs["tags"] = kwargs.get("tags", []) + ["release_fast"]
    zig_configure_test(
        name = name + "_release_fast",
        actual = name,
        mode = "release_fast",
        zigopt = ["-mcpu=native", "-fstrip"],
        **release_kwargs
    )

_LOADER = "@vulkan_loader//:libvulkan.so"

def zerv_vk_binary(name, main, deps = [], **kwargs):
    """An executable that links the system Vulkan loader (and libc: Zig's bundled glibc).

    Args:
      name: target name (the executable's name).
      main: its root source file.
      deps: Zig modules it imports.
      **kwargs: passed to zig_binary.
    """
    zig_binary(
        name = name,
        main = main,
        deps = deps,
        # The loader is a link input (extra_srcs) and named on the link line (data enables
        # $(location)).
        extra_srcs = [_LOADER],
        data = [_LOADER],
        linkopts = ["$(location %s)" % _LOADER],
        zigopts = ["-lc"],
        **kwargs
    )

def _zig_fmt_test_impl(ctx):
    zig = ctx.toolchains["@rules_zig//zig:toolchain_type"].zigtoolchaininfo.zig_exe.file
    script = ctx.actions.declare_file(ctx.label.name + ".sh")
    ctx.actions.write(
        output = script,
        content = "#!/usr/bin/env bash\nset -euo pipefail\nexec {zig} fmt --check {files}\n".format(
            zig = zig.short_path,
            files = " ".join([f.short_path for f in ctx.files.srcs]),
        ),
        is_executable = True,
    )
    return [DefaultInfo(executable = script, runfiles = ctx.runfiles(files = ctx.files.srcs + [zig]))]

zig_fmt_test = rule(
    implementation = _zig_fmt_test_impl,
    doc = "`zig fmt --check` over the given Zig sources, with the registered (hermetic) Zig.",
    attrs = {"srcs": attr.label_list(allow_files = [".zig"], doc = "Zig sources.")},
    toolchains = ["@rules_zig//zig:toolchain_type"],
    test = True,
)

def _zig_exe_impl(ctx):
    info = ctx.toolchains["@rules_zig//zig:toolchain_type"].zigtoolchaininfo
    out = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.symlink(output = out, target_file = info.zig_exe.file, is_executable = True)
    return [DefaultInfo(
        executable = out,
        files = depset([out]),
        runfiles = ctx.runfiles(files = [info.zig_exe.file, info.zig_lib.file]),
    )]

zig_exe = rule(
    implementation = _zig_exe_impl,
    doc = """The registered Zig executable itself (a symlink to the toolchain's).

    For benchmark provenance (its version and hash, tools/zerv_build.py) and ad-hoc use:
    `bazel run //bazel:zig -- version`.""",
    toolchains = ["@rules_zig//zig:toolchain_type"],
    executable = True,
)
