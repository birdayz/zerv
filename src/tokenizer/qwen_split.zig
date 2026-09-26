//! Exact Qwen text regex profile; borrowed UTF-8 slices, no allocation or BPE.
const std = @import("std");
const classes = @import("classes.zig");
pub const properties = classes.properties;
pub const Properties = classes.Properties;
pub const unicode_version = classes.unicode_version;
pub const tableSha256 = classes.tableSha256;
pub const Limits = struct { max_input_bytes: usize = 1024 * 1024 };
pub const Error = error{ InvalidUtf8, InputTooLarge };

const Scalar = struct {
    cp: u21,
    end: usize,
    flags: Properties,

    fn word(self: Scalar) bool {
        return self.flags.letter or self.flags.mark;
    }
    fn other(self: Scalar) bool {
        return !self.word() and !self.flags.number and !self.flags.space;
    }
    fn newline(self: Scalar) bool {
        return self.cp == '\r' or self.cp == '\n';
    }
};

fn fold(cp: u21) u21 {
    if (cp >= 'A' and cp <= 'Z') return cp + ('a' - 'A');
    return if (cp == 0x17f) 's' else cp;
}

pub const Iterator = struct {
    input: []const u8,
    cursor: usize = 0,

    /// Input must remain alive and unchanged. Validation precedes iteration.
    pub fn init(input: []const u8, limits: Limits) Error!Iterator {
        if (input.len > limits.max_input_bytes) return error.InputTooLarge;
        if (!std.unicode.utf8ValidateSlice(input)) return error.InvalidUtf8;
        return .{ .input = input };
    }

    fn scalar(self: *const Iterator, at: usize) Scalar {
        const len = std.unicode.utf8ByteSequenceLength(self.input[at]) catch unreachable;
        const cp = std.unicode.utf8Decode(self.input[at..][0..len]) catch unreachable;
        return .{ .cp = cp, .end = at + len, .flags = properties(cp) };
    }

    fn wordEnd(self: *const Iterator, from: usize) usize {
        var at = from;
        while (at < self.input.len) {
            const c = self.scalar(at);
            if (!c.word()) break;
            at = c.end;
        }
        return at;
    }

    fn pieceEnd(self: *const Iterator, start: usize) usize {
        const first = self.scalar(start);
        // Alternatives are intentionally ordered as in the pinned regex.
        if (first.cp == '\'' and first.end < self.input.len) {
            const second = self.scalar(first.end);
            const letter = fold(second.cp);
            switch (letter) {
                's', 't', 'm', 'd' => return second.end,
                'r', 'v', 'l' => if (second.end < self.input.len) {
                    const third = self.scalar(second.end);
                    if (fold(third.cp) == @as(u21, if (letter == 'l') 'l' else 'e')) return third.end;
                },
                else => {},
            }
        }
        if (first.word()) return self.wordEnd(first.end);
        if (!first.newline() and !first.flags.letter and !first.flags.number and first.end < self.input.len) {
            const second = self.scalar(first.end);
            if (second.word()) return self.wordEnd(second.end);
        }
        if (first.flags.number) return first.end;

        var punctuation: ?usize = if (first.other()) start else null;
        if (first.cp == ' ' and first.end < self.input.len and self.scalar(first.end).other()) punctuation = first.end;
        if (punctuation) |from| {
            var at = from;
            while (at < self.input.len) {
                const c = self.scalar(at);
                if (!c.other()) break;
                at = c.end;
            }
            while (at < self.input.len) {
                const c = self.scalar(at);
                if (!c.newline()) break;
                at = c.end;
            }
            return at;
        }

        std.debug.assert(first.flags.space);
        var at = start;
        var last_start = start;
        var last_newline: ?usize = null;
        while (at < self.input.len) {
            const c = self.scalar(at);
            if (!c.flags.space) break;
            last_start = at;
            at = c.end;
            if (c.newline()) last_newline = at;
        }
        if (last_newline) |end| return end;
        // Negative nonspace lookahead: leave one whitespace before nonspace.
        if (at != self.input.len and last_start != start) return last_start;
        return at;
    }

    /// Returns contiguous nonempty input slices, then null on every later call.
    pub fn next(self: *Iterator) ?[]const u8 {
        if (self.cursor == self.input.len) return null;
        const start = self.cursor;
        self.cursor = self.pieceEnd(start);
        std.debug.assert(self.cursor > start and self.cursor <= self.input.len);
        return self.input[start..self.cursor];
    }
};
