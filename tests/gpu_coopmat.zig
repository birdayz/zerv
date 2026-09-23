//! Cooperative-matrix device path (explicit hardware test; part of `zig build gpu-test`).
//! Checks that the extension and features are enabled, that f16 x f16 -> f32
//! multiply-add uses the expected layouts, and that a stride-0 column-major load
//! broadcasts a per-row scale. The inputs make every result exact in f32 and are
//! non-negative, where the hardware accumulation is exact.
const std = @import("std");
const gpu = @import("zerv").gpu;
const t = std.testing;

const shader align(4) = @embedFile("fixtures/gpu/coopmat.spv").*;

fn halfBits(value: f32) u16 {
    return @bitCast(@as(f16, @floatCast(value)));
}

test "cooperative matrix: enabled on request, f16 x f16 -> f32 layouts and stride-0 row broadcast" {
    {
        var plain = try gpu.Device.open(.{ .max_allocated_bytes = 1024 * 1024 });
        defer plain.deinit() catch @panic("device still has children");
        try t.expect(!plain.cooperative_matrix);
    }
    var device = try gpu.Device.open(.{ .max_allocated_bytes = 1024 * 1024, .cooperative_matrix = true });
    defer device.deinit() catch @panic("device still has children");
    try t.expect(device.cooperative_matrix);

    var a = try gpu.Buffer.init(&device, 16 * 16 * 2, .host);
    defer a.deinit() catch @panic("a");
    var b = try gpu.Buffer.init(&device, 16 * 16 * 2, .host);
    defer b.deinit() catch @panic("b");
    var s = try gpu.Buffer.init(&device, 16 * 4, .host);
    defer s.deinit() catch @panic("s");
    var product = try gpu.Buffer.init(&device, 16 * 16 * 4, .host);
    defer product.deinit() catch @panic("product");
    var scaled = try gpu.Buffer.init(&device, 16 * 16 * 4, .host);
    defer scaled.deinit() catch @panic("scaled");

    // A: small integers (like quantized weights), row-major. B: quarter-steps, column-major.
    // All values are non-negative: RDNA3's WMMA is exact on these, but not in general
    // (negative terms can be off by an ulp; see docs/research/coopmat-prefill.md).
    var a_values: [16][16]f32 = undefined;
    var b_values: [16][16]f32 = undefined; // b_values[k][j]
    const a_half: []align(1) u16 = @ptrCast(try a.mapped());
    const b_half: []align(1) u16 = @ptrCast(try b.mapped());
    const scales: []align(1) f32 = @ptrCast(try s.mapped());
    for (0..16) |i| for (0..16) |k| {
        a_values[i][k] = @floatFromInt((i * 7 + k * 3) % 16);
        a_half[i * 16 + k] = halfBits(a_values[i][k]);
    };
    for (0..16) |k| for (0..16) |j| {
        b_values[k][j] = @as(f32, @floatFromInt((k * 5 + j * 11) % 17)) * 0.25;
        b_half[j * 16 + k] = halfBits(b_values[k][j]); // column-major: column j contiguous
    };
    for (0..16) |i| scales[i] = @as(f32, @floatFromInt(i + 1)) * 0.5;
    @memset(try product.mapped(), 0xff);
    @memset(try scaled.mapped(), 0xff);

    var kernel = try gpu.Kernel.init(&device, &shader, &.{ &a, &b, &s, &product, &scaled }, 0);
    defer kernel.deinit() catch @panic("kernel");
    var commands = try gpu.Commands.init(&device);
    defer commands.deinit() catch @panic("commands");
    try commands.begin();
    try commands.dispatch(&kernel, &.{}, .{ 1, 1, 1 });
    try commands.barrier(.compute, .host);
    try commands.end();
    try commands.run(10 * std.time.ns_per_s);

    const got: []align(1) const f32 = @ptrCast(try product.mapped());
    const got_scaled: []align(1) const f32 = @ptrCast(try scaled.mapped());
    for (0..16) |i| for (0..16) |j| {
        var expected: f64 = 1;
        for (0..16) |k| expected += @as(f64, a_values[i][k]) * b_values[k][j];
        try t.expectEqual(@as(f32, @floatCast(expected)), got[i * 16 + j]);
        try t.expectEqual(@as(f32, @floatCast(expected * scales[i])), got_scaled[i * 16 + j]);
    };
}
