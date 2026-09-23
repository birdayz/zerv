//! Streaming text assembly for generated token bytes: UTF-8 completion with
//! replacement of invalid sequences, stop-string holdback, and the reasoning/content
//! split. Allocation-free. Semantics follow llama-server's output pipeline for this
//! model (docs/research/output-parsing.md).
const std = @import("std");
pub const tools = @import("tools.zig");

pub const Event = tools.Event;

pub const replacement = "\u{fffd}";

/// Accepts arbitrary byte chunks; yields only complete, valid UTF-8. Invalid
/// sequences become U+FFFD per maximal invalid prefix (WHATWG decoder behavior).
pub const Utf8 = struct {
    pending: [4]u8 = undefined,
    pending_len: u8 = 0,

    /// Writes decoded text for `bytes` into `out` (needs bytes.len*3 + 12 capacity).
    pub fn push(self: *Utf8, bytes: []const u8, out: []u8) []const u8 {
        var n: usize = 0;
        var i: usize = 0;
        while (i < bytes.len or self.pending_len > 0) {
            // Fill the pending sequence from the input one byte at a time.
            if (self.pending_len == 0) {
                const b = bytes[i];
                if (b < 0x80) {
                    out[n] = b;
                    n += 1;
                    i += 1;
                    continue;
                }
                const len = std.unicode.utf8ByteSequenceLength(b) catch {
                    @memcpy(out[n..][0..3], replacement);
                    n += 3;
                    i += 1;
                    continue;
                };
                self.pending[0] = b;
                self.pending_len = 1;
                i += 1;
                _ = len;
            }
            const need = std.unicode.utf8ByteSequenceLength(self.pending[0]) catch unreachable;
            while (self.pending_len < need and i < bytes.len) {
                const b = bytes[i];
                if (b & 0xc0 != 0x80 or !validPrefix(self.pending[0..self.pending_len], b)) break;
                self.pending[self.pending_len] = b;
                self.pending_len += 1;
                i += 1;
            }
            if (self.pending_len == need) {
                @memcpy(out[n..][0..need], self.pending[0..need]);
                n += need;
                self.pending_len = 0;
            } else if (i < bytes.len) {
                // Next byte cannot continue this sequence: the prefix is invalid.
                @memcpy(out[n..][0..3], replacement);
                n += 3;
                self.pending_len = 0;
            } else break; // wait for more bytes
        }
        return out[0..n];
    }

    /// End of stream: an unfinished (valid-prefix) sequence is dropped, as the
    /// reference parser drops an incomplete UTF-8 tail. Invalid bytes earlier in
    /// the stream were already replaced by `push`.
    pub fn finish(self: *Utf8) void {
        self.pending_len = 0;
    }
};

/// Second byte ranges per RFC 3629 (no overlongs, surrogates or > U+10FFFF).
fn validPrefix(prefix: []const u8, next: u8) bool {
    if (prefix.len != 1) return true;
    return switch (prefix[0]) {
        0xe0 => next >= 0xa0,
        0xed => next <= 0x9f,
        0xf0 => next >= 0x90,
        0xf4 => next <= 0x8f,
        0xc0, 0xc1, 0xf5...0xff => false,
        else => true,
    };
}

pub const max_stops = 4;
pub const max_stop_bytes = 64;

/// Detects the earliest occurrence of any stop string across chunk boundaries and
/// holds back bytes that could still begin one. Text before a match is released;
/// the match and anything after it are dropped.
pub const Stops = struct {
    stops: []const []const u8,
    held: [max_stop_bytes]u8 = undefined,
    held_len: usize = 0,
    stopped: bool = false,

    pub fn init(stops: []const []const u8) error{InvalidStop}!Stops {
        if (stops.len > max_stops) return error.InvalidStop;
        for (stops) |s| if (s.len == 0 or s.len > max_stop_bytes) return error.InvalidStop;
        return .{ .stops = stops };
    }

    /// `out` needs held + text capacity. Returns releasable text.
    pub fn push(self: *Stops, text: []const u8, out: []u8) []const u8 {
        if (self.stopped) return out[0..0];
        @memcpy(out[0..self.held_len], self.held[0..self.held_len]);
        @memcpy(out[self.held_len..][0..text.len], text);
        const all = out[0 .. self.held_len + text.len];
        var cut: ?usize = null;
        for (self.stops) |s| if (std.mem.indexOf(u8, all, s)) |at| {
            if (cut == null or at < cut.?) cut = at;
        };
        if (cut) |at| {
            self.stopped = true;
            self.held_len = 0;
            return all[0..at];
        }
        // Keep the longest suffix that is a proper prefix of some stop string.
        var keep: usize = 0;
        for (self.stops) |s| {
            var k = @min(s.len - 1, all.len);
            while (k > keep) : (k -= 1) {
                if (std.mem.eql(u8, all[all.len - k ..], s[0..k])) {
                    keep = k;
                    break;
                }
            }
        }
        @memcpy(self.held[0..keep], all[all.len - keep ..]);
        self.held_len = keep;
        return all[0 .. all.len - keep];
    }

    /// End of stream: release held bytes (no stop matched).
    pub fn finish(self: *Stops, out: []u8) []const u8 {
        const n = self.held_len;
        @memcpy(out[0..n], self.held[0..n]);
        self.held_len = 0;
        return out[0..n];
    }
};

/// Splits the raw output text (after stop strings) into reasoning, content and tool
/// calls, as the reference Qwen3-Coder parser does:
/// reasoning = text after leading isspace bytes up to the first `</think>`
/// (consumed) or `<tool_call>` (kept, starts content), trailing whitespace kept;
/// content = the rest after leading isspace bytes. Without thinking, everything
/// after leading isspace bytes is content. In tool mode (`calls` set), content ends
/// at the first `<tool_call>`, and everything from there goes to the call parser
/// (docs/specs/tool-calling.md). A suffix that could still begin a delimiter is held
/// back; at the end of the stream it is dropped.
pub const Splitter = struct {
    state: State,
    held: [max_delimiter - 1]u8 = undefined,
    held_len: usize = 0,
    calls: ?*tools.Calls,

    const State = enum { lead_reasoning, reasoning, lead_content, content, calls };
    const think_end = "</think>";
    const tool_call = tools.call_start;
    const delimiters = [_][]const u8{ think_end, tool_call };
    const max_delimiter = tool_call.len;
    /// Extra `out` capacity `push` needs beyond the input length.
    pub const max_held = max_delimiter - 1;

    pub fn init(thinking: bool, calls: ?*tools.Calls) Splitter {
        return .{ .state = if (thinking) .lead_reasoning else .lead_content, .calls = calls };
    }

    /// Emits the releasable parts of `text` to `sink.emit(Event)`.
    /// `out` needs `text.len + max_held` bytes.
    pub fn push(self: *Splitter, text: []const u8, out: []u8, sink: anytype) !void {
        @memcpy(out[0..self.held_len], self.held[0..self.held_len]);
        @memcpy(out[self.held_len..][0..text.len], text);
        var rest: []const u8 = out[0 .. self.held_len + text.len];
        self.held_len = 0;
        while (rest.len > 0) switch (self.state) {
            .lead_reasoning, .lead_content => {
                var i: usize = 0;
                while (i < rest.len and isSpace(rest[i])) i += 1;
                rest = rest[i..];
                if (rest.len > 0) self.state = if (self.state == .lead_reasoning) .reasoning else .content;
            },
            .content => {
                if (self.calls == null) return sink.emit(.{ .content = rest });
                if (std.mem.indexOf(u8, rest, tool_call)) |at| {
                    if (at > 0) try sink.emit(.{ .content = rest[0..at] });
                    rest = rest[at..];
                    self.state = .calls;
                    continue;
                }
                return self.hold(rest, &.{tool_call}, .content, sink);
            },
            .calls => return self.calls.?.push(rest, sink),
            .reasoning => {
                const none = std.math.maxInt(usize);
                const end_at = std.mem.indexOf(u8, rest, think_end) orelse none;
                const call_at = std.mem.indexOf(u8, rest, tool_call) orelse none;
                if (end_at != none or call_at != none) {
                    const at = @min(end_at, call_at);
                    if (at > 0) try sink.emit(.{ .reasoning = rest[0..at] });
                    // `</think>` is consumed; `<tool_call>` begins the content.
                    rest = rest[if (end_at < call_at) at + think_end.len else at..];
                    self.state = .lead_content;
                    continue;
                }
                return self.hold(rest, &delimiters, .reasoning, sink);
            },
        };
    }

    /// Emits `rest` except its longest suffix that is a proper prefix of a delimiter,
    /// which is held for the next push.
    fn hold(self: *Splitter, rest: []const u8, candidates: []const []const u8, comptime kind: std.meta.Tag(Event), sink: anytype) !void {
        var keep: usize = 0;
        for (candidates) |d| {
            var k = @min(d.len - 1, rest.len);
            while (k > keep) : (k -= 1) {
                if (std.mem.eql(u8, rest[rest.len - k ..], d[0..k])) {
                    keep = k;
                    break;
                }
            }
        }
        if (rest.len > keep) try sink.emit(@unionInit(Event, @tagName(kind), rest[0 .. rest.len - keep]));
        @memcpy(self.held[0..keep], rest[rest.len - keep ..]);
        self.held_len = keep;
    }

    /// End of stream: a held partial delimiter is dropped; an unfinished tool call is
    /// reported as far as it got.
    pub fn finish(self: *Splitter, sink: anytype) !void {
        self.held_len = 0;
        if (self.state == .calls) try self.calls.?.finish(sink);
    }

    /// The call parser requires generation to stop.
    pub fn done(self: *const Splitter) bool {
        return self.state == .calls and self.calls.?.done;
    }

    /// See `tools.Calls.afterCall`.
    pub fn afterCall(self: *const Splitter) ?[]const u8 {
        return if (self.state == .calls) self.calls.?.afterCall() else null;
    }
};

/// C `isspace` in the "C" locale, as the reference parser's `space` rule uses.
fn isSpace(c: u8) bool {
    return c == ' ' or (c >= '\t' and c <= '\r');
}
