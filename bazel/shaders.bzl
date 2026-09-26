"""SPIR-V modules as Bazel actions (docs/development.md, "Shaders").

The modules are generated and committed (as ../fdb-go does with generated code): the build
embeds the committed files, and `spirv_modules` makes Bazel regenerate them from the GLSL
sources with the pinned shader tools and test that the committed files are exactly that
output. `bazel run //PKG:update_shaders` writes the regenerated set back.
"""

load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_test.bzl", "py_test")

_JOBS = 8

def _resources(_os, _inputs):
    return {"cpu": _JOBS, "memory": 1024}

def _spirv_tree_impl(ctx):
    out = ctx.actions.declare_directory(ctx.label.name)
    args = ctx.actions.args()
    args.add("--output-dir", out.path)
    args.add("--glslc", ctx.file._glslc)
    args.add("--spirv-val", ctx.file._spirv_val)
    args.add("--jobs", str(_JOBS))
    args.add("--quiet")
    ctx.actions.run(
        executable = ctx.executable.compiler,
        arguments = [args],
        inputs = [ctx.file._glslc, ctx.file._spirv_val] + ctx.files._tool_libs,
        outputs = [out],
        mnemonic = "SpirvCompile",
        progress_message = "Compiling the SPIR-V modules of %{label}",
        resource_set = _resources,
    )
    return [DefaultInfo(files = depset([out]))]

spirv_tree = rule(
    implementation = _spirv_tree_impl,
    doc = """All modules (and manifest.json) a compile script (tools/compile_*.py) produces, as a
    directory. The script carries its GLSL sources as data and checks the tools' sha256 pins.""",
    attrs = {
        "compiler": attr.label(executable = True, cfg = "exec", mandatory = True, doc = "The compile script (py_binary)."),
        "_glslc": attr.label(default = "@shader_tools//:glslc", allow_single_file = True),
        "_spirv_val": attr.label(default = "@shader_tools//:spirv-val", allow_single_file = True),
        "_tool_libs": attr.label(default = "@shader_tool_libs//:libs"),
    },
)

def spirv_modules(name, compiler, committed_dir, committed):
    """`name`: the regenerated modules; `name_test`: they equal the committed ones;
    `update_shaders`: writes them back into the source tree (`bazel run`).

    Args:
      name: the generated directory's target name.
      compiler: the compile script (py_binary).
      committed_dir: the committed modules' directory, relative to the package.
      committed: every committed file in it (a glob).
    """
    spirv_tree(name = name, compiler = compiler)
    where = native.package_name() + "/" + committed_dir
    py_test(
        name = name + "_test",
        srcs = ["//tools:check_generated.py"],
        main = "//tools:check_generated.py",
        args = ["$(rootpath :%s)" % name, where, "--hint", "'bazel run //%s:update_shaders'" % native.package_name()],
        data = [":" + name] + committed,
        size = "small",
        tags = ["shaders"],
    )
    py_binary(
        name = "update_shaders",
        srcs = ["//tools:check_generated.py"],
        main = "//tools:check_generated.py",
        args = ["$(rootpath :%s)" % name, where, "--update"],
        data = [":" + name],
    )
