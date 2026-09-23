//! Python `json.dumps(x, ensure_ascii=False)` output for parsed JSON values: the
//! `tojson` filter of the Hugging Face chat-template environment the official Qwen3.8
//! template targets (docs/specs/tool-calling.md). Values must come from std.json with
//! `parse_numbers = false`, so every number is its source literal: integer literals are
//! printed as Python ints, other literals as Python float `repr`. Allocation-free.
const std = @import("std");

/// Nesting deeper than this is rejected before writing (the writer recurses).
pub const max_depth = 64;

pub const Error = std.Io.Writer.Error || error{ TooDeep, InvalidNumber };

/// Returns error.TooDeep or error.InvalidNumber without writing anything.
pub fn validate(value: std.json.Value) error{ TooDeep, InvalidNumber }!void {
    return check(value, 0);
}

fn check(value: std.json.Value, depth: usize) error{ TooDeep, InvalidNumber }!void {
    if (depth > max_depth) return error.TooDeep;
    switch (value) {
        .number_string => |s| _ = try classify(s),
        .integer, .float => {}, // only produced with parse_numbers = true; printed directly
        .array => |items| for (items.items) |item| try check(item, depth + 1),
        .object => |object| for (object.values()) |item| try check(item, depth + 1),
        else => {},
    }
}

/// Writes `value` as Python's json.dumps(ensure_ascii=False) would.
pub fn write(w: *std.Io.Writer, value: std.json.Value) Error!void {
    try validate(value);
    try writeValue(w, value);
}

fn writeValue(w: *std.Io.Writer, value: std.json.Value) Error!void {
    switch (value) {
        .null => try w.writeAll("null"),
        .bool => |b| try w.writeAll(if (b) "true" else "false"),
        .integer => |i| try w.print("{d}", .{i}),
        .float => |f| try writeFloat(w, f),
        .number_string => |s| try writeNumber(w, s),
        .string => |s| try writeString(w, s),
        .array => |items| {
            try w.writeByte('[');
            for (items.items, 0..) |item, i| {
                if (i > 0) try w.writeAll(", ");
                try writeValue(w, item);
            }
            try w.writeByte(']');
        },
        .object => |object| {
            try w.writeByte('{');
            var it = object.iterator();
            var first = true;
            while (it.next()) |entry| {
                if (!first) try w.writeAll(", ");
                first = false;
                try writeString(w, entry.key_ptr.*);
                try w.writeAll(": ");
                try writeValue(w, entry.value_ptr.*);
            }
            try w.writeByte('}');
        },
    }
}

/// JSON string with Python's escapes: \" \\ \n \r \t \b \f, other C0 controls as
/// \u00xx (lowercase); everything else (DEL, U+2028, non-ASCII) raw. The same escapes
/// as nlohmann::json::dump() without ensure_ascii.
pub fn writeString(w: *std.Io.Writer, s: []const u8) std.Io.Writer.Error!void {
    try w.writeByte('"');
    try writeEscaped(w, s);
    try w.writeByte('"');
}

/// The inside of a JSON string (no quotes).
pub fn writeEscaped(w: *std.Io.Writer, s: []const u8) std.Io.Writer.Error!void {
    var start: usize = 0;
    for (s, 0..) |c, i| {
        const escape: ?[]const u8 = switch (c) {
            '"' => "\\\"",
            '\\' => "\\\\",
            '\n' => "\\n",
            '\r' => "\\r",
            '\t' => "\\t",
            0x08 => "\\b",
            0x0c => "\\f",
            else => null,
        };
        if (escape == null and c >= 0x20) continue;
        try w.writeAll(s[start..i]);
        if (escape) |e| try w.writeAll(e) else try w.print("\\u{x:0>4}", .{c});
        start = i + 1;
    }
    try w.writeAll(s[start..]);
}

const Kind = enum { integer, float };

/// JSON number literal grammar: -?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?
fn classify(s: []const u8) error{InvalidNumber}!Kind {
    var i: usize = 0;
    if (i < s.len and s[i] == '-') i += 1;
    if (i >= s.len or !std.ascii.isDigit(s[i])) return error.InvalidNumber;
    if (s[i] == '0') i += 1 else while (i < s.len and std.ascii.isDigit(s[i])) i += 1;
    var kind: Kind = .integer;
    if (i < s.len and s[i] == '.') {
        kind = .float;
        i += 1;
        const begin = i;
        while (i < s.len and std.ascii.isDigit(s[i])) i += 1;
        if (i == begin) return error.InvalidNumber;
    }
    if (i < s.len and (s[i] == 'e' or s[i] == 'E')) {
        kind = .float;
        i += 1;
        if (i < s.len and (s[i] == '+' or s[i] == '-')) i += 1;
        const begin = i;
        while (i < s.len and std.ascii.isDigit(s[i])) i += 1;
        if (i == begin) return error.InvalidNumber;
    }
    if (i != s.len) return error.InvalidNumber;
    return kind;
}

/// A JSON number literal as Python prints the value json.loads makes of it.
pub fn writeNumber(w: *std.Io.Writer, literal: []const u8) Error!void {
    switch (try classify(literal)) {
        // JSON forbids leading zeros, so the digits are canonical except for "-0".
        .integer => try w.writeAll(if (std.mem.eql(u8, literal, "-0")) "0" else literal),
        .float => try writeFloat(w, std.fmt.parseFloat(f64, literal) catch return error.InvalidNumber),
    }
}

/// Python `float.__repr__`: shortest round-trip digits; exponent form when the decimal
/// point position is < -3 or > 16 (as in CPython's format_float_short, mode 'r').
pub fn writeFloat(w: *std.Io.Writer, x: f64) std.Io.Writer.Error!void {
    if (std.math.isNan(x)) return w.writeAll("NaN");
    if (std.math.isInf(x)) return w.writeAll(if (x < 0) "-Infinity" else "Infinity");
    if (x == 0) return w.writeAll(if (std.math.signbit(x)) "-0.0" else "0.0");
    var buf: [64]u8 = undefined;
    const sci = std.fmt.bufPrint(&buf, "{e}", .{@abs(x)}) catch unreachable;
    // `sci` is d[.ddd]e[-]X with the shortest round-trip digits.
    const e_at = std.mem.indexOfScalar(u8, sci, 'e').?;
    var digits_buf: [32]u8 = undefined;
    var n: usize = 0;
    for (sci[0..e_at]) |c| if (c != '.') {
        digits_buf[n] = c;
        n += 1;
    };
    while (n > 1 and digits_buf[n - 1] == '0') n -= 1;
    const digits = digits_buf[0..n];
    const exponent = std.fmt.parseInt(i32, sci[e_at + 1 ..], 10) catch unreachable;
    const decpt = exponent + 1; // value = 0.DIGITS × 10^decpt
    if (x < 0) try w.writeByte('-');
    if (decpt <= -4 or decpt > 16) {
        try w.writeByte(digits[0]);
        if (n > 1) {
            try w.writeByte('.');
            try w.writeAll(digits[1..]);
        }
        const magnitude: u32 = @abs(exponent);
        try w.print("e{c}{d:0>2}", .{ @as(u8, if (exponent < 0) '-' else '+'), magnitude });
    } else if (decpt <= 0) {
        try w.writeAll("0.");
        try w.splatByteAll('0', @intCast(-decpt));
        try w.writeAll(digits);
    } else {
        const point: usize = @intCast(decpt);
        if (point >= n) {
            try w.writeAll(digits);
            try w.splatByteAll('0', point - n);
            try w.writeAll(".0");
        } else {
            try w.writeAll(digits[0..point]);
            try w.writeByte('.');
            try w.writeAll(digits[point..]);
        }
    }
}
