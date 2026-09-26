"""A meson project built inside one hermetic Bazel action (bazel/meson_build.py).

For third-party projects whose configuration logic is too large to translate into BUILD files
without risking a differently configured result (Mesa, docs/specs/hermetic-build.md): meson
and ninja from the Bazel registry, the Bazel C/C++ toolchain, the pinned Python, subprojects
and tools from the graph. One action: the project is pinned and rebuilt only when it changes.
"""

load("@rules_cc//cc:find_cc_toolchain.bzl", "find_cc_toolchain", "use_cc_toolchain")

def _resources(_os, _inputs):
    return {"cpu": 12, "memory": 8192}

def _meson_project_impl(ctx):
    cc = find_cc_toolchain(ctx)
    outs = {path: ctx.actions.declare_file(ctx.label.name + "/" + path.rsplit("/", 1)[-1]) for path in ctx.attr.outs}
    args = ctx.actions.args()
    args.add("--source", ctx.file.root.dirname)
    args.add("--meson", ctx.file._meson_py)
    args.add("--ninja", ctx.executable._ninja)
    args.add("--cc", cc.compiler_executable)
    args.add("--ar", cc.ar_executable)
    for target, name in ctx.attr.subprojects.items():
        args.add("--subproject", "%s=%s" % (name, target[DefaultInfo].files.to_list()[0].dirname))
    for target, name in ctx.attr.tools.items():
        args.add("--tool", "%s=%s" % (name, target[DefaultInfo].files_to_run.executable.path))
    args.add_all(ctx.attr.options, before_each = "--option")
    args.add_all(ctx.attr.targets, before_each = "--target")
    for path, out in outs.items():
        args.add("--output", "%s=%s" % (path, out.path))
    args.add("--jobs", "12")
    ctx.actions.run(
        executable = ctx.executable._driver,
        arguments = [args],
        inputs = depset(
            ctx.files.srcs + ctx.files._meson_runtime + [ctx.file._meson_py] +
            [f for t in ctx.attr.subprojects for f in t[DefaultInfo].files.to_list()] +
            [f for t in ctx.attr.subproject_srcs for f in t[DefaultInfo].files.to_list()],
            transitive = [cc.all_files],
        ),
        tools = [ctx.attr._ninja[DefaultInfo].files_to_run] + [t[DefaultInfo].files_to_run for t in ctx.attr.tools],
        outputs = outs.values(),
        mnemonic = "MesonBuild",
        progress_message = "Building %{label} with meson",
        resource_set = _resources,
    )
    return [DefaultInfo(files = depset(outs.values()))]

meson_project = rule(
    implementation = _meson_project_impl,
    attrs = {
        "root": attr.label(allow_single_file = True, mandatory = True, doc = "The project's top-level meson.build."),
        "srcs": attr.label_list(allow_files = True, doc = "Every file of the project."),
        "subprojects": attr.label_keyed_string_dict(allow_files = True, doc = "A subproject's meson.build -> its directory name under subprojects/."),
        "subproject_srcs": attr.label_list(allow_files = True, doc = "Every file of the subprojects."),
        "tools": attr.label_keyed_string_dict(cfg = "exec", doc = "Executable -> the name it has on PATH."),
        "options": attr.string_list(doc = "meson -D options."),
        "targets": attr.string_list(doc = "ninja targets."),
        "outs": attr.string_list(mandatory = True, doc = "Build-directory-relative files to return."),
        "_driver": attr.label(default = "//bazel:meson_build", executable = True, cfg = "exec"),
        "_meson_py": attr.label(default = "@meson//:meson.py", allow_single_file = True),
        "_meson_runtime": attr.label(default = "@meson//:runtime"),
        "_ninja": attr.label(default = "@ninja//:ninja", executable = True, cfg = "exec"),
    },
    toolchains = use_cc_toolchain(),
    fragments = ["cpp"],
)
