//! Drop-in `Sha256` for large golden fingerprints: the hashing runs in the ReleaseFast
//! support object (tests/support/fast_sha256.zig); the same digests as std's.
const std = @import("std");

extern fn zerv_test_sha256_new() ?*anyopaque;
extern fn zerv_test_sha256_update(state: *anyopaque, bytes: [*]const u8, len: usize) void;
extern fn zerv_test_sha256_final(state: *anyopaque, out: *[32]u8) void;

pub const Sha256 = struct {
    state: *anyopaque,

    pub const digest_length = 32;
    /// std's `Sha256.Options` has no fields either.
    pub const Options = struct {};

    pub fn init(_: Options) Sha256 {
        return .{ .state = zerv_test_sha256_new() orelse @panic("out of memory") };
    }
    pub fn update(self: *Sha256, bytes: []const u8) void {
        zerv_test_sha256_update(self.state, bytes.ptr, bytes.len);
    }
    /// Once per `init`.
    pub fn final(self: *Sha256, out: *[32]u8) void {
        zerv_test_sha256_final(self.state, out);
    }
    pub fn hash(bytes: []const u8, out: *[32]u8, options: Options) void {
        var h = init(options);
        h.update(bytes);
        h.final(out);
    }
};

test "same digests as std" {
    const Std = std.crypto.hash.sha2.Sha256;
    var bytes: [5000]u8 = undefined;
    for (&bytes, 0..) |*b, i| b.* = @truncate(i *% 131 +% 7);
    for ([_]usize{ 0, 1, 55, 56, 63, 64, 65, 1000, 5000 }) |n| {
        var want: [32]u8 = undefined;
        Std.hash(bytes[0..n], &want, .{});
        var got: [32]u8 = undefined;
        // Streamed in uneven pieces.
        var h = Sha256.init(.{});
        var at: usize = 0;
        var step: usize = 1;
        while (at < n) : (step += 17) {
            const k = @min(step, n - at);
            h.update(bytes[at..][0..k]);
            at += k;
        }
        h.final(&got);
        try std.testing.expectEqualSlices(u8, &want, &got);
    }
}
