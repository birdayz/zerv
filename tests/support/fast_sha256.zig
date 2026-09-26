//! SHA-256 for the golden fingerprints of the Debug unit tests, compiled in ReleaseFast
//! and linked as an object (build.zig, `test_support`). The fingerprints hash hundreds of
//! MB of decoded output; std's SHA-256 runs at ~134 MB/s in Debug and ~1.75 GB/s optimized
//! on the test machine (docs/bench/2026-09-26-test-parallelism.md). The code under test
//! stays in the test's own mode; only this hashing is optimized. Test support only.
const std = @import("std");
const Sha256 = std.crypto.hash.sha2.Sha256;

export fn zerv_test_sha256_new() ?*anyopaque {
    const h = std.heap.page_allocator.create(Sha256) catch return null;
    h.* = .init(.{});
    return h;
}

export fn zerv_test_sha256_update(state: *anyopaque, bytes: [*]const u8, len: usize) void {
    const h: *Sha256 = @ptrCast(@alignCast(state));
    h.update(bytes[0..len]);
}

/// Writes the digest and frees the state.
export fn zerv_test_sha256_final(state: *anyopaque, out: *[32]u8) void {
    const h: *Sha256 = @ptrCast(@alignCast(state));
    h.final(out);
    std.heap.page_allocator.destroy(h);
}
