#!/usr/bin/env python3
"""Build an instrumented research copy; leave production tokenizer/build untouched."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import zerv_build  # noqa: E402  (tools/zerv_build.py: Bazel builds and provenance)
PROFILE = r'''
const std = @import("std");
pub var io: std.Io = undefined;
pub var enabled = false;
pub const Stage = enum { total, validation, added, nfc, split, bpe };
pub var ns: [6]u64 = @splat(0);
pub var calls: [6]u64 = @splat(0);
pub var lengths: [6]u64 = @splat(0);
pub fn begin() std.Io.Timestamp {
    return if (enabled) std.Io.Clock.awake.now(io) else .{ .nanoseconds = 0 };
}
pub fn end(stage: Stage, start: std.Io.Timestamp) void {
    if (!enabled) return;
    const elapsed = start.durationTo(std.Io.Clock.awake.now(io)).nanoseconds;
    ns[@intFromEnum(stage)] += @intCast(elapsed);
    calls[@intFromEnum(stage)] += 1;
}
pub fn length(n: usize) void {
    if (enabled) lengths[if (n == 1) 0 else if (n <= 4) 1 else if (n <= 8) 2 else if (n <= 16) 3 else if (n <= 32) 4 else 5] += 1;
}
pub fn reset() void { ns = @splat(0); calls = @splat(0); lengths = @splat(0); }
'''


def replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"instrumentation anchor changed: {old}")
    return text.replace(old, new)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True, help="fresh path below ignored third_party")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--cpu", type=int, default=2)
    a = p.parse_args()
    dest = a.output.resolve()
    if not dest.is_relative_to(ROOT / "third_party"):
        p.error("instrumented research copy must be in third_party")
    dest.mkdir(parents=True, exist_ok=False)
    # A workspace of its own (third_party/ is outside this one's packages, .bazelignore).
    for name in ("src", "bench", "tests", "tools"):
        shutil.copytree(ROOT / name, dest / name, ignore=shutil.ignore_patterns("__pycache__"))
    for path in zerv_build.build_files():
        (dest / path.relative_to(ROOT)).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest / path.relative_to(ROOT))
    bpe = dest / "src/tokenizer/bpe.zig"
    source = bpe.read_text()
    original_sha = hashlib.sha256(source.encode()).hexdigest()
    source = source.replace('const std = @import("std");', 'const std = @import("std");\npub const profile = @import("profile.zig");', 1)
    source = replace(source, '        if (input.len > limits.max_input_bytes)', '        const total_start = profile.begin();\n        defer profile.end(.total, total_start);\n        const validation_start = profile.begin();\n        if (input.len > limits.max_input_bytes)')
    source = replace(source, '        var workspace: Workspace', '        profile.end(.validation, validation_start);\n        var workspace: Workspace')
    source = replace(source, '            const special = self.nextAdded(input, cursor);', '            const added_start = profile.begin();\n            const special = self.nextAdded(input, cursor);\n            profile.end(.added, added_start);')
    source = replace(source, '                const ordinary = input[cursor..end];', '                const nfc_start = profile.begin();\n                const ordinary = input[cursor..end];')
    source = replace(source, '                while (it.next()) |text| try workspace.mergePiece(self, text, &output, limits.max_tokens);', '''                profile.end(.nfc, nfc_start);
                while (true) {
                    const split_start = profile.begin();
                    const next = it.next();
                    profile.end(.split, split_start);
                    const text = next orelse break;
                    try workspace.mergePiece(self, text, &output, limits.max_tokens);
                }''')
    source = replace(source, '        if (text.len == 0) return;', '        const bpe_start = profile.begin();\n        defer profile.end(.bpe, bpe_start);\n        profile.length(text.len);\n        if (text.len == 0) return;')
    bpe.write_text(source)
    (dest / "src/tokenizer/profile.zig").write_text(PROFILE)
    path = dest / "bench/tokenizer.zig"
    source = path.read_text().replace('    const allocator = std.heap.smp_allocator;', '    const profile = zerv.tokenizer.bpe.profile;\n    profile.io = init.io;\n    const allocator = std.heap.smp_allocator;', 1)
    source = replace(source, '            var check: [32]u8 = undefined;', '            profile.reset();\n            profile.enabled = comptime std.mem.eql(u8, operation, "encode");\n            var check: [32]u8 = undefined;')
    source = replace(source, '        }\n    }\n    try stdout.interface.flush();', '''            if (comptime std.mem.eql(u8, operation, "encode")) {
                profile.enabled = false;
                std.debug.print("PROFILE {s} ns={any} calls={any} lengths={any}\\n", .{case.name, profile.ns, profile.calls, profile.lengths});
            }
        }
    }
    try stdout.interface.flush();''')
    path.write_text(source)
    build = zerv_build.build_command("zerv-tokenizer-bench")
    subprocess.run(build, cwd=dest, check=True)
    bench = zerv_build.path("zerv-tokenizer-bench", root=dest)
    commands = [build, ["taskset", "-c", str(a.cpu), str(bench), str(a.model.resolve()), str(ROOT / "tests/fixtures/tokenizer/manifest.json"), "--bench"]]
    # The copy's Bazel server is not needed after its one build.
    subprocess.run([zerv_build.BAZEL, "shutdown"], cwd=dest, check=True)
    for i, command in enumerate(commands[1:], 1):
        result = subprocess.run(command, cwd=dest, capture_output=True, text=True)
        (dest / f"{i}.stdout").write_text(result.stdout)
        (dest / f"{i}.stderr").write_text(result.stderr)
        if result.returncode:
            print(result.stderr)
        result.check_returncode()
    (dest / "manifest.json").write_text(json.dumps(dict(commands=commands, original_bpe_sha256=original_sha,
        stages=["total", "validation", "added", "nfc", "split", "bpe"], length_bins=["1", "2-4", "5-8", "9-16", "17-32", ">32"],
        caveat="Instrumented stage-clock overhead, especially per split/piece; not a speed comparison. Total includes all clock probes; stage times include closing timer cost. Full checker still runs."), indent=2) + "\n")
    print((dest / "1.stderr").read_text())


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    main()
