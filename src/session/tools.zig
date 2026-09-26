//! Incremental parser for Qwen3-Coder-style tool calls in generated text
//! (docs/specs/tool-calling.md):
//!
//!   "<tool_call>\n<function=" NAME ">\n" ("<parameter=" KEY ">\n" VALUE "\n</parameter>\n")*
//!   "</function>\n</tool_call>", then SPACE_RULE whitespace, then another call or the end.
//!
//! Emits call headers and OpenAI `arguments` fragments assembled as llama.cpp does
//! (compact JSON in generation order). Byte-at-a-time state machine with bounded
//! buffers; allocation-free. This is a parser, not a constraint: names are not checked.
const std = @import("std");
const pyjson = @import("chat").pyjson;

/// JSON value kinds a parameter schema admits (llama.cpp `value_types`).
pub const Types = packed struct(u8) {
    string: bool = false,
    integer: bool = false,
    number: bool = false,
    boolean: bool = false,
    null: bool = false,
    array: bool = false,
    object: bool = false,
    _: u1 = 0,

    pub const all: Types = .{ .string = true, .integer = true, .number = true, .boolean = true, .null = true, .array = true, .object = true };
    pub const none: Types = .{};
    pub fn onlyString(t: Types) bool {
        return @as(u8, @bitCast(t)) == @as(u8, @bitCast(Types{ .string = true }));
    }
    pub fn @"or"(a: Types, b: Types) Types {
        return @bitCast(@as(u8, @bitCast(a)) | @as(u8, @bitCast(b)));
    }
    pub fn @"and"(a: Types, b: Types) Types {
        return @bitCast(@as(u8, @bitCast(a)) & @as(u8, @bitCast(b)));
    }
};
pub const Param = struct { name: []const u8, types: Types };
pub const Tool = struct { name: []const u8, params: []const Param };
pub const Mode = struct { tools: []const Tool, parallel: bool = true };

pub const max_name = 256;
/// Largest non-string parameter value (buffered until its close).
pub const max_value = 64 * 1024;

pub const Event = union(enum) {
    reasoning: []const u8,
    content: []const u8,
    /// NAME is complete. The call's arguments start with `{`, which this event
    /// carries implicitly (the SSE header sends it, as llama-server does); the
    /// `call_arguments` fragments for this index continue after it.
    call_begin: struct { index: u32, name: []const u8 },
    call_arguments: struct { index: u32, text: []const u8 },
};

pub const call_start = "<tool_call>";
const open_literal = "<tool_call>\n<function=";
const param_literal = "<parameter=";
const close_literal = "</function>\n</tool_call>";
const close_function_len = "</function>\n".len;
const terminator = "\n</parameter>\n";

pub const Calls = struct {
    mode: Mode,
    /// Caller-owned buffer of at least `max_value` bytes for non-string values.
    value_buf: []u8,
    state: State = .open,
    pos: usize = 0, // progress in the current literal
    closing: bool = false, // in .arg_or_close: matching close_literal, not param_literal
    name: [max_name]u8 = undefined,
    name_len: usize = 0,
    key: [max_name]u8 = undefined,
    key_len: usize = 0,
    tool: ?*const Tool = null,
    types: Types = .all,
    string_value: bool = false,
    value_len: usize = 0,
    term: usize = 0, // bytes of `terminator` matched and held back
    args: u32 = 0, // arguments emitted in the current call
    /// Calls whose NAME is complete (each was reported with `call_begin`).
    count: u32 = 0,
    space: [24]u8 = undefined,
    space_len: usize = 0,
    /// Generation must stop: malformed call, or post-call text the grammar forbids.
    done: bool = false,
    /// A call was malformed (counted by the server).
    failed: bool = false,
    out: [4096]u8 = undefined, // pending `call_arguments` bytes of call `count - 1`
    out_len: usize = 0,

    const State = enum { open, name, name_newline, arg_or_close, key, key_newline, value, post_call };

    pub fn init(mode: Mode, value_buf: []u8) Calls {
        std.debug.assert(value_buf.len >= max_value);
        return .{ .mode = mode, .value_buf = value_buf };
    }

    /// The whitespace since the last `</tool_call>` while the next token must be
    /// constrained (spec: "Constrained decoding"); null otherwise.
    pub fn afterCall(self: *const Calls) ?[]const u8 {
        return if (self.state == .post_call and !self.done) self.space[0..self.space_len] else null;
    }

    /// Feeds generated text, starting with the `<tool_call>` that opened call mode.
    pub fn push(self: *Calls, text: []const u8, sink: anytype) !void {
        for (text, 0..) |c, i| {
            if (self.done) break;
            if (!try self.byte(c, sink)) {
                // Malformed before NAME was complete: nothing of this call was reported,
                // so its raw text (and the rest of the chunk) becomes content.
                try sink.emit(.{ .content = open_literal[0..if (self.state == .open) self.pos else open_literal.len] });
                if (self.state == .name and self.name_len > 0) try sink.emit(.{ .content = self.name[0..self.name_len] });
                try sink.emit(.{ .content = text[i..] });
                self.done = true;
                self.failed = true;
                return;
            }
        }
        try self.flush(sink);
    }

    /// End of generation (EOS, length or stop string). A call with a complete NAME is
    /// reported as far as it got: an open value is closed (string: `"`), no `}`.
    /// A call without a complete NAME is dropped, as llama-server does.
    pub fn finish(self: *Calls, sink: anytype) !void {
        if (!self.done and self.state == .value) try self.closePartial(sink);
        self.done = true;
        try self.flush(sink);
    }

    /// Returns false for a malformed call whose NAME is not complete yet.
    fn byte(self: *Calls, c: u8, sink: anytype) !bool {
        switch (self.state) {
            .open => {
                if (c != open_literal[self.pos]) return false;
                self.pos += 1;
                if (self.pos == open_literal.len) {
                    self.state = .name;
                    self.name_len = 0;
                }
            },
            .name => if (c == '>') {
                const name = std.mem.trim(u8, self.name[0..self.name_len], " \t\r\n");
                if (name.len == 0) return false;
                self.tool = null;
                for (self.mode.tools) |*tool| {
                    if (std.mem.eql(u8, tool.name, name)) self.tool = tool;
                }
                try self.flush(sink);
                try sink.emit(.{ .call_begin = .{ .index = self.count, .name = name } });
                self.count += 1;
                self.args = 0; // the leading `{` is part of `call_begin`
                self.state = .name_newline;
            } else {
                if (c == '\n' or self.name_len == max_name) return false;
                self.name[self.name_len] = c;
                self.name_len += 1;
            },
            .name_newline => if (c == '\n') {
                self.state = .arg_or_close;
                self.pos = 0;
            } else try self.fail(sink),
            .arg_or_close => {
                if (self.pos == 1) self.closing = c == '/';
                const literal: []const u8 = if (self.closing) close_literal else param_literal;
                if (c != literal[self.pos]) {
                    try self.fail(sink);
                    return true;
                }
                self.pos += 1;
                if (self.closing and self.pos == close_function_len) try self.emitArgs(sink, "}");
                if (self.pos == literal.len) {
                    if (self.closing) {
                        self.state = .post_call;
                        self.space_len = 0;
                    } else {
                        self.state = .key;
                        self.key_len = 0;
                    }
                    self.pos = 0;
                    self.closing = false;
                }
            },
            .key => if (c == '>') {
                const key = std.mem.trim(u8, self.key[0..self.key_len], " \t\r\n");
                if (key.len == 0) {
                    try self.fail(sink);
                    return true;
                }
                self.types = .all;
                if (self.tool) |tool| for (tool.params) |param| {
                    if (std.mem.eql(u8, param.name, key)) self.types = param.types;
                };
                self.string_value = self.types.onlyString();
                if (self.args > 0) try self.emitArgs(sink, ",");
                self.args += 1;
                try self.emitArgs(sink, "\"");
                try self.emitEscaped(sink, key);
                try self.emitArgs(sink, if (self.string_value) "\":\"" else "\":");
                self.state = .key_newline;
            } else {
                if (c == '\n' or self.key_len == max_name) try self.fail(sink) else {
                    self.key[self.key_len] = c;
                    self.key_len += 1;
                }
            },
            .key_newline => if (c == '\n') {
                self.state = .value;
                self.value_len = 0;
                self.term = 0;
            } else try self.fail(sink),
            .value => try self.valueByte(c, sink),
            .post_call => if (c == '<' and self.mode.parallel) {
                self.state = .open;
                self.pos = 1;
            } else if (spaceRule(self.space[0..self.space_len], c)) {
                self.space[self.space_len] = c;
                self.space_len += 1;
            } else {
                // Only whitespace, another call or the end may follow a call.
                self.done = true;
            },
        }
        return true;
    }

    fn valueByte(self: *Calls, c: u8, sink: anytype) !void {
        while (true) {
            if (c == terminator[self.term]) {
                self.term += 1;
                if (self.term == terminator.len) {
                    self.term = 0;
                    try self.closeValue(sink);
                    self.state = .arg_or_close;
                    self.pos = 0;
                }
                return;
            }
            if (self.term == 0) return self.valueText(sink, &.{c});
            // Mismatch after a partial match: the held bytes are value text, and `c`
            // is examined again (only the terminator's leading '\n' can restart it).
            const held = self.term;
            self.term = 0;
            try self.valueText(sink, terminator[0..held]);
            if (self.done) return;
        }
    }

    fn valueText(self: *Calls, sink: anytype, bytes: []const u8) !void {
        if (self.string_value) return self.emitEscaped(sink, bytes);
        if (self.value_len + bytes.len > max_value) return self.fail(sink);
        @memcpy(self.value_buf[self.value_len..][0..bytes.len], bytes);
        self.value_len += bytes.len;
    }

    fn closeValue(self: *Calls, sink: anytype) !void {
        if (self.string_value) return self.emitArgs(sink, "\"");
        const raw = self.value_buf[0..self.value_len];
        const trimmed = std.mem.trim(u8, raw, " \t\r\n");
        if (jsonOfTypes(trimmed, self.types)) return self.emitArgs(sink, trimmed);
        try self.emitArgs(sink, "\"");
        try self.emitEscaped(sink, raw);
        try self.emitArgs(sink, "\"");
    }

    /// The value so far, as llama-server reports an interrupted one: a held partial
    /// `\n</parameter>\n` is dropped (lenient `until`), then the value is closed.
    fn closePartial(self: *Calls, sink: anytype) !void {
        self.term = 0;
        try self.closeValue(sink);
    }

    /// Malformed after NAME: close the call as incomplete and stop.
    fn fail(self: *Calls, sink: anytype) !void {
        if (self.state == .value) try self.closePartial(sink);
        self.done = true;
        self.failed = true;
    }

    fn emitArgs(self: *Calls, sink: anytype, bytes: []const u8) !void {
        var rest = bytes;
        while (rest.len > 0) {
            if (self.out_len == self.out.len) try self.flush(sink);
            const n = @min(rest.len, self.out.len - self.out_len);
            @memcpy(self.out[self.out_len..][0..n], rest[0..n]);
            self.out_len += n;
            rest = rest[n..];
        }
    }

    fn emitEscaped(self: *Calls, sink: anytype, bytes: []const u8) !void {
        var start: usize = 0;
        for (bytes, 0..) |c, i| {
            if (c >= 0x20 and c != '"' and c != '\\') continue;
            try self.emitArgs(sink, bytes[start..i]);
            var tmp: [8]u8 = undefined;
            var w: std.Io.Writer = .fixed(&tmp);
            pyjson.writeEscaped(&w, bytes[i..][0..1]) catch unreachable;
            try self.emitArgs(sink, w.buffered());
            start = i + 1;
        }
        try self.emitArgs(sink, bytes[start..]);
    }

    fn flush(self: *Calls, sink: anytype) !void {
        if (self.out_len == 0) return;
        try sink.emit(.{ .call_arguments = .{ .index = self.count - 1, .text = self.out[0..self.out_len] } });
        self.out_len = 0;
    }
};

/// `SPACE_RULE` (`"" | " " | "\n"{1,2} [ \t]{0,20}`): may `c` extend `prefix`?
/// Every valid prefix is itself a complete match.
pub fn spaceRule(prefix: []const u8, c: u8) bool {
    if (prefix.len == 0) return c == ' ' or c == '\n';
    if (prefix[0] == ' ') return false;
    var newlines: usize = 0;
    while (newlines < prefix.len and prefix[newlines] == '\n') newlines += 1;
    const tail = prefix.len - newlines;
    if (c == '\n') return tail == 0 and newlines < 2;
    if (c == ' ' or c == '\t') return tail < 20;
    return false;
}

/// Whether `prefix ++ text` still satisfies `SPACE_RULE`.
pub fn spaceRuleExtends(prefix: []const u8, text: []const u8) bool {
    var buf: [24]u8 = undefined;
    if (prefix.len + text.len > buf.len) return false;
    @memcpy(buf[0..prefix.len], prefix);
    var n = prefix.len;
    for (text) |c| {
        if (!spaceRule(buf[0..n], c)) return false;
        buf[n] = c;
        n += 1;
    }
    return true;
}

/// Whether `text` (already trimmed) is exactly one JSON value of a declared non-string
/// kind. A JSON string literal never qualifies: llama.cpp tries only the non-string
/// alternatives before the raw-string one.
pub fn jsonOfTypes(text: []const u8, types: Types) bool {
    if (text.len == 0) return false;
    const ok_kind = switch (text[0]) {
        '{' => types.object,
        '[' => types.array,
        't', 'f' => types.boolean,
        'n' => types.null,
        '-', '0'...'9' => types.number or (types.integer and std.mem.indexOfAny(u8, text, ".eE") == null),
        else => false,
    };
    if (!ok_kind) return false;
    var stack: [1024]u8 = undefined;
    var fba: std.heap.FixedBufferAllocator = .init(&stack);
    return std.json.validate(fba.allocator(), text) catch false;
}
