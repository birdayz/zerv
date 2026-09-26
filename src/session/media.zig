//! Media spans: prompt rows whose model inputs are not their tokens' embeddings (an
//! image's encoder outputs). docs/specs/prefix-cache.md, "Media spans".
const std = @import("std");

pub const Media = struct {
    /// First prompt row and row count. The rows carry a placeholder token id.
    start: u32,
    len: u32,
    /// Identity of the rows' inputs, chosen by the producer: equal ids mean equal inputs.
    id: [32]u8,
    /// RoPE positions the span advances (1..len); plain rows advance one each.
    positions: u32,
    /// Opaque producer handle for the backend (e.g. an index into the request's encoded
    /// images). Not part of the identity.
    handle: u32 = 0,

    pub fn end(self: Media) u32 {
        return self.start + self.len;
    }

    /// Same inputs at the same rows.
    pub fn same(a: Media, b: Media) bool {
        return a.start == b.start and a.len == b.len and a.positions == b.positions and std.mem.eql(u8, &a.id, &b.id);
    }
};

/// Spans are sorted, disjoint, non-empty, inside `rows`, and advance 1..len positions.
pub fn validate(spans: []const Media, rows: usize) error{InvalidMedia}!void {
    var next: u64 = 0;
    for (spans) |s| {
        if (s.len == 0 or s.positions == 0 or s.positions > s.len) return error.InvalidMedia;
        if (s.start < next or @as(u64, s.start) + s.len > rows) return error.InvalidMedia;
        next = @as(u64, s.start) + s.len;
    }
}

/// The spans of `spans` inside rows [from, to), which must not cut a span.
pub fn within(spans: []const Media, from: u32, to: u32) []const Media {
    var lo: usize = 0;
    while (lo < spans.len and spans[lo].end() <= from) lo += 1;
    var hi = lo;
    while (hi < spans.len and spans[hi].start < to) hi += 1;
    if (lo < hi) std.debug.assert(spans[lo].start >= from and spans[hi - 1].end() <= to);
    return spans[lo..hi];
}

/// Whether row `p` lies strictly inside a span (after its first row).
pub fn inside(spans: []const Media, p: u32) bool {
    for (spans) |s| {
        if (p > s.start and p < s.end()) return true;
        if (s.start >= p) break;
    }
    return false;
}
