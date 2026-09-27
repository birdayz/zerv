//! Prefix checkpoint store (docs/specs/concurrent.md, "18d.4 design"): lookup, points,
//! least-recently-used replacement.
const std = @import("std");
const session = @import("session");
const Store = session.checkpoint.Store;
const t = std.testing;

const B = 9; // the boundary token

test "checkpoint store: points are the system-prompt end and the generation-prompt start, once" {
    var s = try Store.init(t.allocator, 4, 64, 8, B);
    defer s.deinit(t.allocator);
    // [B sys sys] [B user] [B asst-header]
    const prompt = [_]u32{ B, 1, 2, B, 3, B, 4 };
    var out: [session.checkpoint.max_points]u32 = undefined;
    try t.expectEqualSlices(u32, &.{ 3, 5 }, s.points(&prompt, 0, &out));
    // Already held prefixes are not taken again; points at or before the start neither.
    try s.fill(s.claim(), prompt[0..3], &.{ 10, 11 });
    try t.expectEqualSlices(u32, &.{5}, s.points(&prompt, 0, &out));
    try t.expectEqualSlices(u32, &.{5}, s.points(&prompt, 3, &out));
    try t.expectEqualSlices(u32, &.{}, s.points(&prompt, 5, &out));
    // One boundary after the start: one point. None: none.
    try t.expectEqualSlices(u32, &.{2}, s.points(&.{ B, 1, B, 2 }, 0, &out));
    try t.expectEqualSlices(u32, &.{}, s.points(&.{ B, 1, 2 }, 0, &out));
}

test "checkpoint store: lookup takes the longest prefix that leaves a token; LRU replacement" {
    var s = try Store.init(t.allocator, 2, 64, 8, B);
    defer s.deinit(t.allocator);
    const a = [_]u32{ B, 1, 2, B, 3, B, 4 };
    try t.expectEqual(@as(?usize, null), s.lookup(&a));
    const e0 = s.claim();
    try s.fill(e0, a[0..3], &.{1});
    const e1 = s.claim();
    try t.expect(e1 != e0);
    try s.fill(e1, a[0..5], &.{ 1, 2 });
    try t.expectEqual(@as(?usize, e1), s.lookup(&a));
    // The whole prompt as a prefix would leave nothing to process.
    try t.expectEqual(@as(?usize, e0), s.lookup(a[0..5]));
    // A different continuation matches only the common part.
    try t.expectEqual(@as(?usize, e0), s.lookup(&.{ B, 1, 2, B, 7, 7 }));
    try t.expectEqual(@as(?usize, null), s.lookup(&.{ B, 1, 3, B }));
    // Full: a leaf goes before its parent, even when the parent is older (e0 is a prefix
    // of e1).
    s.touch(e1);
    try t.expectEqual(e1, s.claim());
    try t.expectEqual(@as(?usize, e1), s.oldest());
    try t.expect(s.isParent(e0) and !s.isParent(e1));
    try t.expectEqualSlices(u32, &.{ 1, 2 }, s.entryPages(e1));
    s.drop(e1);
    try t.expectEqual(@as(u32, 1), s.live());
    try t.expectEqual(e1, s.claim());
    try t.expectError(error.InvalidOptions, s.fill(e1, &.{}, &.{}));
}

test "checkpoint store: the shared system prompt survives churn of per-conversation leaves" {
    var s = try Store.init(t.allocator, 3, 64, 8, B);
    defer s.deinit(t.allocator);
    const sys = [_]u32{ B, 1, 2 };
    try s.fill(s.claim(), &sys, &.{1});
    // Many conversations add their own turn checkpoints (leaves under `sys`), never touching it.
    for (0..20) |c| {
        const turn = [_]u32{ B, 1, 2, B, @intCast(100 + c) };
        const i = s.claim();
        if (s.entries[i].len != 0) s.drop(i);
        try s.fill(i, &turn, &.{ 1, 2 });
    }
    try t.expect(s.has(&sys));
    try t.expectEqual(@as(u32, 3), s.live());
}
