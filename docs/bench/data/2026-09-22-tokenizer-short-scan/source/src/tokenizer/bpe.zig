//! Owned byte-BPE tables with the Qwen NFC/split profile and raw-byte decoding.
const std = @import("std");
const Allocator = std.mem.Allocator;
const nfc = @import("../text/nfc.zig");
const split = @import("qwen_split.zig");
pub const Kind = enum(u8) { missing = 0, normal = 1, added = 2, unused = 3 };
pub const Entry = struct { id: u32, kind: Kind, bytes: []const u8 };
pub const Merge = struct { left: u32, right: u32, result: u32 };
pub const InitLimits = struct {
    max_vocabulary: usize = 1048576,
    max_merges: usize = 2097152,
    max_piece_bytes: usize = 16384,
    max_total_bytes: usize = 64 * 1024 * 1024,
    max_added: usize = 1024,
};
pub const EncodeLimits = struct {
    max_input_bytes: usize = 1024 * 1024,
    max_normalized_bytes: usize = 4 * 1024 * 1024,
    max_tokens: usize = 1048576,
};
pub const Error = Allocator.Error || error{
    InvalidVocabulary,
    InvalidMerge,
    LimitExceeded,
    Overflow,
    InvalidTokenId,
    InvalidUtf8,
    InputTooLarge,
    NormalizedTooLarge,
    TooManyTokens,
};
const Piece = struct { kind: Kind = .missing, bytes: []const u8 = &.{} };
const Rule = struct { rank: u32, result: u32 };
const none = std.math.maxInt(u32);
fn pair(a: u32, b: u32) u64 {
    return (@as(u64, a) << 32) | b;
}

pub const Tokenizer = struct {
    allocator: Allocator,
    pieces: []Piece = &.{},
    storage: []u8 = &.{},
    added: []u32 = &.{},
    byte_ids: [256]u32 = @splat(none),
    added_first: [256]bool = @splat(false),
    rules: std.AutoHashMapUnmanaged(u64, Rule) = .empty,

    pub fn init(allocator: Allocator, entries: []const Entry, merges: []const Merge, limits: InitLimits) Error!Tokenizer {
        const id_limit = @min(limits.max_vocabulary, std.math.maxInt(u32) / 4);
        if (entries.len > id_limit or merges.len > @min(limits.max_merges, std.math.maxInt(u32) / 4)) return error.LimitExceeded;
        var size: usize = 0;
        var bytes: usize = 0;
        var additions: usize = 0;
        for (entries) |entry| {
            if (entry.id >= id_limit or entry.bytes.len > limits.max_piece_bytes) return error.LimitExceeded;
            size = @max(size, @as(usize, entry.id) + 1);
            bytes = std.math.add(usize, bytes, entry.bytes.len) catch return error.Overflow;
            if (bytes > limits.max_total_bytes) return error.LimitExceeded;
            switch (entry.kind) {
                .normal => if (entry.bytes.len == 0) return error.InvalidVocabulary,
                .added => {
                    if (entry.bytes.len == 0 or !std.unicode.utf8ValidateSlice(entry.bytes)) return error.InvalidVocabulary;
                    additions += 1;
                    if (additions > limits.max_added) return error.LimitExceeded;
                },
                .unused => if (entry.bytes.len != 0) return error.InvalidVocabulary,
                .missing => return error.InvalidVocabulary,
            }
        }
        var self: Tokenizer = .{ .allocator = allocator };
        errdefer self.deinit();
        self.pieces = try allocator.alloc(Piece, size);
        @memset(self.pieces, .{});
        self.storage = try allocator.alloc(u8, bytes);
        self.added = try allocator.alloc(u32, additions);
        var added_names: std.StringHashMapUnmanaged(void) = .empty;
        defer added_names.deinit(allocator);
        var offset: usize = 0;
        var added_at: usize = 0;
        for (entries) |entry| {
            if (self.pieces[entry.id].kind != .missing) return error.InvalidVocabulary;
            const dest = self.storage[offset..][0..entry.bytes.len];
            @memcpy(dest, entry.bytes);
            offset += dest.len;
            self.pieces[entry.id] = .{ .kind = entry.kind, .bytes = dest };
            if (entry.kind == .normal and dest.len == 1) {
                if (self.byte_ids[dest[0]] != none) return error.InvalidVocabulary;
                self.byte_ids[dest[0]] = entry.id;
            }
            if (entry.kind == .added) {
                const found = try added_names.getOrPut(allocator, dest);
                if (found.found_existing) return error.InvalidVocabulary;
                self.added_first[dest[0]] = true;
                self.added[added_at] = entry.id;
                added_at += 1;
            }
        }
        for (self.byte_ids) |id| if (id == none) return error.InvalidVocabulary;
        try self.rules.ensureTotalCapacity(allocator, @intCast(merges.len));
        for (merges, 0..) |merge, rank| {
            for ([_]u32{ merge.left, merge.right, merge.result }) |id| {
                if (id >= self.pieces.len or self.pieces[id].kind != .normal) return error.InvalidMerge;
            }
            const a = self.pieces[merge.left].bytes;
            const b = self.pieces[merge.right].bytes;
            const result = self.pieces[merge.result].bytes;
            const length = std.math.add(usize, a.len, b.len) catch return error.Overflow;
            if (length != result.len or !std.mem.eql(u8, a, result[0..a.len]) or !std.mem.eql(u8, b, result[a.len..])) return error.InvalidMerge;
            const slot = self.rules.getOrPutAssumeCapacity(pair(merge.left, merge.right));
            if (slot.found_existing) return error.InvalidMerge;
            slot.value_ptr.* = .{ .rank = @intCast(rank), .result = merge.result };
        }
        return self;
    }

    pub fn deinit(self: *Tokenizer) void {
        self.rules.deinit(self.allocator);
        self.allocator.free(self.added);
        self.allocator.free(self.storage);
        self.allocator.free(self.pieces);
        self.* = undefined;
    }

    pub fn piece(self: *const Tokenizer, id: u32) Error![]const u8 {
        if (id >= self.pieces.len or self.pieces[id].kind == .missing) return error.InvalidTokenId;
        return self.pieces[id].bytes;
    }

    /// Raw bytes, not per-piece UTF-8 replacement. Invalid IDs write nothing.
    pub fn decode(self: *const Tokenizer, ids: []const u32, writer: *std.Io.Writer) !void {
        for (ids) |id| _ = try self.piece(id);
        for (ids) |id| try writer.writeAll(self.pieces[id].bytes);
    }

    const Match = struct { start: usize, end: usize, id: u32 };
    fn nextAdded(self: *const Tokenizer, input: []const u8, from: usize) ?Match {
        for (from..input.len) |at| {
            if (!self.added_first[input[at]]) continue;
            var best: ?Match = null;
            for (self.added) |id| {
                const text = self.pieces[id].bytes;
                if (std.mem.startsWith(u8, input[at..], text) and (best == null or text.len > best.?.end - at))
                    best = .{ .start = at, .end = at + text.len, .id = id };
            }
            if (best != null) return best;
        }
        return null;
    }

    pub fn encode(self: *const Tokenizer, allocator: Allocator, input: []const u8, limits: EncodeLimits) Error![]u32 {
        if (input.len > limits.max_input_bytes) return error.InputTooLarge;
        if (!std.unicode.utf8ValidateSlice(input)) return error.InvalidUtf8;
        var workspace: Workspace = .{ .allocator = allocator };
        defer workspace.deinit();
        var output: std.ArrayList(u32) = .empty;
        errdefer output.deinit(allocator);
        var cursor: usize = 0;
        var remaining = limits.max_normalized_bytes;
        while (cursor < input.len) {
            const special = self.nextAdded(input, cursor);
            const end = if (special) |m| m.start else input.len;
            if (end > cursor) {
                const ordinary = input[cursor..end];
                const required = nfc.scratchSize(ordinary) catch return error.Overflow;
                try workspace.scratch.resize(allocator, required);
                const upper = if (required == 0) ordinary.len else std.math.mul(usize, required, 2) catch return error.Overflow;
                try workspace.normalized.resize(allocator, @min(upper, remaining));
                const length = nfc.normalize(ordinary, workspace.normalized.items, workspace.scratch.items) catch |err| switch (err) {
                    error.InsufficientOutput => return error.NormalizedTooLarge,
                    error.Overflow => return error.Overflow,
                    else => unreachable, // UTF-8 was validated and scratch was sized above.
                };
                remaining -= length;
                var it = split.Iterator.init(workspace.normalized.items[0..length], .{ .max_input_bytes = limits.max_normalized_bytes }) catch unreachable;
                while (it.next()) |text| try workspace.mergePiece(self, text, &output, limits.max_tokens);
            }
            if (special) |m| {
                const len = m.end - m.start;
                if (len > remaining) return error.NormalizedTooLarge;
                remaining -= len;
                try append(allocator, &output, m.id, limits.max_tokens);
                cursor = m.end;
            } else break;
        }
        return output.toOwnedSlice(allocator);
    }
};

fn append(allocator: Allocator, output: *std.ArrayList(u32), id: u32, limit: usize) Error!void {
    if (output.items.len >= limit) return error.TooManyTokens;
    try output.append(allocator, id);
}
const Symbol = struct { id: u32, prev: u32, next: u32, live: bool = true };
const Edge = struct { left: u32, right: u32, left_id: u32, right_id: u32, rank: u32, result: u32 };
fn order(_: void, a: Edge, b: Edge) std.math.Order {
    return if (a.rank == b.rank) std.math.order(a.left, b.left) else std.math.order(a.rank, b.rank);
}
const Queue = std.PriorityQueue(Edge, void, order);
const Workspace = struct {
    allocator: Allocator,
    scratch: std.ArrayList(u21) = .empty,
    normalized: std.ArrayList(u8) = .empty,
    symbols: std.ArrayList(Symbol) = .empty,
    queue: Queue = .empty,

    fn deinit(self: *Workspace) void {
        self.queue.deinit(self.allocator);
        self.symbols.deinit(self.allocator);
        self.normalized.deinit(self.allocator);
        self.scratch.deinit(self.allocator);
    }
    fn addEdge(self: *Workspace, tokenizer: *const Tokenizer, left: u32) Error!void {
        const a = self.symbols.items[left];
        if (a.next == none) return;
        const b = self.symbols.items[a.next];
        if (tokenizer.rules.get(pair(a.id, b.id))) |rule| {
            try self.queue.push(self.allocator, .{ .left = left, .right = a.next, .left_id = a.id, .right_id = b.id, .rank = rule.rank, .result = rule.result });
        }
    }
    // A bounded rank scan avoids heap/link overhead for short regex pieces.
    // Cached rules change only at the two new adjacencies after a merge.
    fn mergeShort(self: *Workspace, tokenizer: *const Tokenizer, text: []const u8, output: *std.ArrayList(u32), limit: usize) Error!void {
        if (text.len == 1) return append(self.allocator, output, tokenizer.byte_ids[text[0]], limit);
        var ids: [16]u32 = undefined;
        var rules: [15]Rule = undefined;
        const absent: Rule = .{ .rank = none, .result = none };
        var n = text.len;
        for (text, 0..) |byte, i| ids[i] = tokenizer.byte_ids[byte];
        for (0..n - 1) |i| rules[i] = tokenizer.rules.get(pair(ids[i], ids[i + 1])) orelse absent;
        while (n > 1) {
            var rank: u32 = none;
            var at: usize = 0;
            for (rules[0 .. n - 1], 0..) |rule, i| {
                // Strict less-than retains the leftmost equal-rank candidate.
                if (rule.rank < rank) {
                    rank = rule.rank;
                    at = i;
                }
            }
            if (rank == none) break;
            ids[at] = rules[at].result;
            n -= 1;
            var j = at + 1;
            while (j < n) : (j += 1) {
                ids[j] = ids[j + 1];
                if (j + 1 < n) rules[j] = rules[j + 1];
            }
            if (at > 0) rules[at - 1] = tokenizer.rules.get(pair(ids[at - 1], ids[at])) orelse absent;
            if (at + 1 < n) rules[at] = tokenizer.rules.get(pair(ids[at], ids[at + 1])) orelse absent;
            // Rescan from the start: newly eligible ranks may be lower.
        }
        for (ids[0..n]) |id| try append(self.allocator, output, id, limit);
    }

    fn mergePiece(self: *Workspace, tokenizer: *const Tokenizer, text: []const u8, output: *std.ArrayList(u32), limit: usize) Error!void {
        if (text.len == 0) return;
        if (text.len <= 16) return self.mergeShort(tokenizer, text, output, limit);
        if (text.len >= none) return error.LimitExceeded;
        self.queue.items.len = 0;
        const capacity = std.math.mul(usize, text.len, 3) catch return error.Overflow;
        try self.queue.ensureTotalCapacity(self.allocator, capacity);
        try self.symbols.resize(self.allocator, text.len);
        const symbols = self.symbols.items;
        for (text, 0..) |byte, i| symbols[i] = .{ .id = tokenizer.byte_ids[byte], .prev = if (i == 0) none else @intCast(i - 1), .next = if (i + 1 == text.len) none else @intCast(i + 1) };
        for (0..text.len - 1) |i| try self.addEdge(tokenizer, @intCast(i));
        while (self.queue.pop()) |edge| {
            const left = &symbols[edge.left];
            const right = &symbols[edge.right];
            if (!left.live or !right.live or left.next != edge.right or left.id != edge.left_id or right.id != edge.right_id) continue;
            left.id = edge.result;
            left.next = right.next;
            right.live = false;
            if (left.next != none) symbols[left.next].prev = edge.left;
            if (left.prev != none) try self.addEdge(tokenizer, left.prev);
            try self.addEdge(tokenizer, edge.left);
        }
        var at: u32 = 0;
        while (at != none) {
            try append(self.allocator, output, symbols[at].id, limit);
            at = symbols[at].next;
        }
    }
};
