"""The native gemm_f16x machine code as Bazel actions (docs/development.md, "Native kernels").

The kernel's pipeline binary is committed (src/model/native/). Its code part is a pure
function of the generator (bench/isa_lab/gen_f16x.py) and the assembler: Bazel regenerates the
assembly, assembles it with the toolchain's Zig (`zig clang`, the same object bytes as the
host's clang 22) and splices it into the committed binary, and a test checks that the
committed assembly and binary are exactly that. The rest of the binary (driver config, keys)
and the bitwise sweep come from the GPU (tools/build_native_gemm.py), which Bazel does not run.
"""

def _native_code_impl(ctx):
    zig = ctx.toolchains["@rules_zig//zig:toolchain_type"].zigtoolchaininfo
    name = ctx.attr.kernel
    asm = ctx.actions.declare_file(ctx.label.name + "/" + name + ".s")
    code = ctx.actions.declare_file(ctx.label.name + "/code.bin")
    binary = ctx.actions.declare_file(ctx.label.name + "/" + name + ".bin")
    clang = ctx.actions.declare_file(ctx.label.name + "_tools/clang")
    ctx.actions.write(clang, '#!/bin/sh\nexec "{}" clang "$@"\n'.format(zig.zig_exe.file.path), is_executable = True)
    ctx.actions.run(
        executable = ctx.executable.generator,
        arguments = [asm.path],
        outputs = [asm],
        mnemonic = "IsaGenerate",
        progress_message = "Generating %{output}",
    )

    # isa_tool.py runs `clang` from PATH: the wrapper, then the system directories for the
    # Python launcher's shell.
    ctx.actions.run(
        executable = ctx.executable.isa_tool,
        arguments = ["asm", asm.path, code.path],
        inputs = [asm, clang, zig.zig_exe.file, zig.zig_lib.file],
        outputs = [code],
        env = {"PATH": clang.dirname + ":/usr/bin:/bin"},
        mnemonic = "IsaAssemble",
        progress_message = "Assembling %{input}",
    )
    ctx.actions.run(
        executable = ctx.executable.isa_tool,
        arguments = ["splice", ctx.file.template.path, code.path, binary.path],
        inputs = [ctx.file.template, code],
        outputs = [binary],
        mnemonic = "IsaSplice",
        progress_message = "Splicing %{output}",
    )
    return [DefaultInfo(files = depset([asm, binary]))]

native_code = rule(
    implementation = _native_code_impl,
    doc = "KERNEL.s and KERNEL.bin (the template's pipeline binary with the regenerated code).",
    attrs = {
        "kernel": attr.string(mandatory = True, doc = "Kernel name (file stem)."),
        "template": attr.label(allow_single_file = True, mandatory = True, doc = "The committed pipeline binary."),
        "generator": attr.label(executable = True, cfg = "exec", mandatory = True),
        "isa_tool": attr.label(executable = True, cfg = "exec", mandatory = True),
    },
    toolchains = ["@rules_zig//zig:toolchain_type"],
)
