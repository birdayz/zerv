//! Unicode-9 NFC for compatibility with the official Qwen tokenizer backend.
const std = @import("std");
pub const unicode_version = "9.0.0";
pub const Error = error{ InvalidUtf8, InsufficientScratch, InsufficientOutput, Overflow };
const data = @embedFile("data/nfc9.bin");
/// SHA-256 of the embedded table, hex. Computed at run time: at comptime it cost ~25 s of
/// every compile that referenced it (docs/development.md, "Tests in parallel").
pub fn tableSha256() [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(data, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}
const decomp_count = read(u32, 4);
const ccc_count = read(u32, 8);
const composition_count = read(u32, 12);
const ccc_start = 16 + decomp_count * 24;
const composition_start = ccc_start + ccc_count * 8;

fn read(comptime T: type, offset: usize) T {
    return std.mem.readInt(T, data[offset..][0..@sizeOf(T)], .little);
}

fn lookup(comptime T: type, key: T, start: usize, stride: usize, count: usize) ?usize {
    var low: usize = 0;
    var high = count;
    while (low < high) {
        const mid = low + (high - low) / 2;
        const at = start + mid * stride;
        const candidate = read(T, at);
        if (candidate == key) return at;
        if (candidate < key) low = mid + 1 else high = mid;
    }
    return null;
}

fn combining(cp: u21) u8 {
    if (cp < 0x300) return 0;
    const at = lookup(u32, cp, ccc_start, 8, ccc_count) orelse return 0;
    return @intCast(read(u32, at + 4));
}

fn expand(cp: u21, out: *[4]u21) usize {
    if (cp >= 0xac00 and cp < 0xac00 + 11172) {
        const index = cp - 0xac00;
        out[0] = 0x1100 + index / 588;
        out[1] = 0x1161 + index % 588 / 28;
        if (index % 28 == 0) return 2;
        out[2] = 0x11a7 + index % 28;
        return 3;
    }
    if (cp >= 0xc0) {
        if (lookup(u32, cp, 16, 24, decomp_count)) |at| {
            const count = read(u32, at + 4);
            for (0..count) |i| out[i] = @intCast(read(u32, at + 8 + i * 4));
            return count;
        }
    }
    out[0] = cp;
    return 1;
}

fn compose(a: u21, b: u21) ?u21 {
    if (a >= 0x1100 and a < 0x1100 + 19 and b >= 0x1161 and b < 0x1161 + 21)
        return 0xac00 + (a - 0x1100) * 588 + (b - 0x1161) * 28;
    if (a >= 0xac00 and a < 0xac00 + 11172 and (a - 0xac00) % 28 == 0 and b > 0x11a7 and b < 0x11a7 + 28)
        return a + b - 0x11a7;
    const key = (@as(u64, a) << 21) | b;
    const at = lookup(u64, key, composition_start, 12, composition_count) orelse return null;
    return @intCast(read(u32, at + 8));
}

fn ascii(input: []const u8) bool {
    for (input) |byte| if (byte >= 128) return false;
    return true;
}

/// Required u21 elements, not bytes. ASCII uses no scratch.
pub fn scratchSize(input: []const u8) Error!usize {
    if (ascii(input)) return 0;
    const view = std.unicode.Utf8View.init(input) catch return error.InvalidUtf8;
    var it = view.iterator();
    var count: usize = 0;
    var expanded: [4]u21 = undefined;
    while (it.nextCodepoint()) |cp| {
        count = std.math.add(usize, count, expand(cp, &expanded)) catch return error.Overflow;
    }
    return std.math.mul(usize, count, 2) catch error.Overflow;
}

// Stable counting sort has linear work even for adversarial combining-mark runs.
fn orderRun(input: []const u21, output: []u21) void {
    var previous: u8 = 0;
    var ordered = true;
    for (input) |cp| {
        const cls = combining(cp);
        if (cls < previous) ordered = false;
        previous = cls;
    }
    if (ordered) return; // output already contains the unchanged decomposition
    var positions: [256]usize = @splat(0);
    for (input) |cp| positions[combining(cp)] += 1;
    var total: usize = 0;
    for (&positions) |*position| {
        const count = position.*;
        position.* = total;
        total += count;
    }
    for (input) |cp| {
        const cls = combining(cp);
        output[positions[cls]] = cp;
        positions[cls] += 1;
    }
}

/// Input/output/scratch must be disjoint. Errors preserve output, not scratch.
pub fn normalize(input: []const u8, output: []u8, scratch: []u21) Error!usize {
    const required = try scratchSize(input);
    if (required == 0) {
        if (output.len < input.len) return error.InsufficientOutput;
        @memcpy(output[0..input.len], input);
        return input.len;
    }
    if (scratch.len < required) return error.InsufficientScratch;
    const count = required / 2;
    const decomposed = scratch[0..count];
    const ordered = scratch[count..required];
    var it = (std.unicode.Utf8View.initUnchecked(input)).iterator();
    var used: usize = 0;
    var expanded: [4]u21 = undefined;
    while (it.nextCodepoint()) |cp| {
        const n = expand(cp, &expanded);
        @memcpy(decomposed[used..][0..n], expanded[0..n]);
        used += n;
    }
    @memcpy(ordered, decomposed);
    var start: usize = 0;
    for (decomposed, 0..) |cp, i| {
        if (combining(cp) == 0) {
            orderRun(decomposed[start..i], ordered[start..i]);
            start = i + 1;
        }
    }
    orderRun(decomposed[start..], ordered[start..]);
    used = 0;
    var starter: ?usize = null;
    var previous: u8 = 0;
    for (ordered) |cp| {
        const cls = combining(cp);
        if (starter) |at| {
            if (previous == 0 or previous < cls) {
                if (compose(ordered[at], cp)) |composite| {
                    ordered[at] = composite;
                    continue;
                }
            }
        }
        if (cls == 0) starter = used;
        ordered[used] = cp;
        used += 1;
        previous = cls;
    }
    var bytes: usize = 0;
    for (ordered[0..used]) |cp| {
        const length = std.unicode.utf8CodepointSequenceLength(cp) catch unreachable;
        bytes = std.math.add(usize, bytes, length) catch return error.Overflow;
    }
    if (bytes > output.len) return error.InsufficientOutput;
    var at: usize = 0;
    for (ordered[0..used]) |cp| at += std.unicode.utf8Encode(cp, output[at..]) catch unreachable;
    return at;
}
