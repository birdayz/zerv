//! Unicode-16 properties used by text splitting, derived from normative UCD.
const std = @import("std");
const data = @embedFile("data/classes16.bin");
pub const unicode_version = "16.0.0";
pub const Properties = packed struct(u8) {
    letter: bool = false,
    mark: bool = false,
    number: bool = false,
    space: bool = false,
    reserved: u4 = 0,
};
/// SHA-256 of the embedded table, hex, computed at run time (at comptime it cost seconds of
/// every compile that referenced it).
pub fn tableSha256() [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(data, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

fn read(at: usize) u32 {
    return std.mem.readInt(u32, data[at..][0..4], .little);
}

fn lookup(cp: u21) Properties {
    var low: usize = 0;
    var high: usize = read(4);
    while (low < high) {
        const mid = low + (high - low) / 2;
        if (read(8 + mid * 8) <= cp) low = mid + 1 else high = mid;
    }
    // The first range starts at zero; the final non-scalar range has no flags.
    return @bitCast(@as(u8, @intCast(read(8 + (low - 1) * 8 + 4))));
}

const ascii = blk: {
    @setEvalBranchQuota(20_000);
    var result: [128]Properties = undefined;
    for (&result, 0..) |*value, cp| value.* = lookup(@intCast(cp));
    break :blk result;
};

/// Non-scalars have no properties. UTF-8 validation is the iterator's boundary.
pub fn properties(cp: u21) Properties {
    return if (cp < 128) ascii[cp] else lookup(cp);
}
