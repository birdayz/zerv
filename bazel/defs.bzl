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

VULKAN = "//src/gpu:vulkan"

def zerv_vk_binary(name, main, deps = [], **kwargs):
    """An executable that uses Vulkan (and libc: Zig's bundled glibc).

    It links the stub libvulkan.so.1 (//src/gpu:vulkan, hermetic); at run time the dynamic
    linker loads the real loader by that name.

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
        # The stub is a link input (extra_srcs) named on the link line (data enables
        # $(location)). No run path points at it: tests/test_build_lists.py checks the linked
        # executables for RPATH/RUNPATH.
        extra_srcs = [VULKAN],
        data = [VULKAN],
        linkopts = ["$(location %s)" % VULKAN],
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

def _env_test_impl(ctx):
    out = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.symlink(output = out, target_file = ctx.executable.actual, is_executable = True)
    runfiles = ctx.runfiles(files = [out]).merge(ctx.attr.actual[DefaultInfo].default_runfiles)
    return [DefaultInfo(executable = out, runfiles = runfiles), RunEnvironmentInfo(environment = ctx.attr.env)]

env_test = rule(
    implementation = _env_test_impl,
    doc = """Runs the test `actual` with its runfiles and the environment `env`.

    rules_zig's zig_configure_test (a test in another mode) forwards no RunEnvironmentInfo, so
    the `env` of the test it configures does not reach the configured one; this sets it.""",
    attrs = {
        "actual": attr.label(executable = True, cfg = "target", mandatory = True, doc = "The test executable."),
        "env": attr.string_dict(doc = "Environment of the test (runfiles-relative paths as in `env` of native tests)."),
    },
    test = True,
)

CcRulesInfo = provider(doc = "C/C++ rule targets in the transitive dependencies.", fields = ["labels"])

def _cc_rules_aspect_impl(target, ctx):
    own = [str(target.label)] if ctx.rule.kind.startswith("cc_") else []
    deps = []
    for attr in ("deps", "srcs", "extra_srcs", "data", "main", "cdeps", "actual"):
        value = getattr(ctx.rule.attr, attr, None)
        if value == None:
            continue
        for dep in (value if type(value) == "list" else [value]):
            if type(dep) == "Target" and CcRulesInfo in dep:
                deps.append(dep[CcRulesInfo].labels)
    return [CcRulesInfo(labels = depset(own, transitive = deps))]

_cc_rules_aspect = aspect(
    implementation = _cc_rules_aspect_impl,
    attr_aspects = ["deps", "srcs", "extra_srcs", "data", "main", "cdeps", "actual"],
)

def _no_cc_test_impl(ctx):
    found = sorted(depset(transitive = [t[CcRulesInfo].labels for t in ctx.attr.targets]).to_list())
    script = ctx.actions.declare_file(ctx.label.name + ".sh")
    if found:
        content = "#!/bin/sh\necho 'C/C++ in what zerv ships (AGENTS.md: no C++ dependencies):'\n" + "".join(["echo '  {}'\n".format(l) for l in found]) + "exit 1\n"
    else:
        content = "#!/bin/sh\necho 'no C/C++ rule in the dependencies of {}'\n".format(", ".join([str(t.label) for t in ctx.attr.targets]))
    ctx.actions.write(script, content, is_executable = True)
    return [DefaultInfo(executable = script)]

no_cc_test = rule(
    implementation = _no_cc_test_impl,
    doc = "Fails if a C/C++ rule (cc_*) is among the targets' transitive dependencies.",
    attrs = {"targets": attr.label_list(aspects = [_cc_rules_aspect], mandatory = True)},
    test = True,
)
