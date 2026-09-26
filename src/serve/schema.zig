//! Tool parameter schemas → the value kinds the output parser types values by.
//! Follows llama.cpp's `common_chat_schema_builder::build_node` + `value_types`
//! (docs/research/tool-calling.md) for the schema of each top-level property.
//! Values must be parsed with `parse_numbers = false`. Driver-free, allocation-free.
const std = @import("std");
const tools = @import("session").tools;

pub const Types = tools.Types;
pub const Error = error{InvalidSchema};

const max_depth = 64;
const max_refs = 32;

/// The properties of a tool's `parameters` when its root schema is an object schema
/// (llama.cpp `foreach_parameter`); null when it has no known properties. A missing,
/// null or empty `parameters` means `{"type": "object", "properties": {}}`.
pub fn properties(parameters: ?std.json.Value) Error!?std.json.ObjectMap {
    const root = parameters orelse return null;
    if (root == .null) return null;
    if (root != .object) return error.InvalidSchema;
    const o = root.object;
    if (o.count() == 0) return null;
    if (o.get("$ref") != null or o.get("oneOf") != null or o.get("anyOf") != null or o.get("const") != null or o.get("enum") != null) return null;
    const kind = o.get("type") orelse std.json.Value.null;
    if (kind == .array) return null;
    const name = if (kind == .string) kind.string else if (kind == .null) "" else return error.InvalidSchema;
    const has_properties = hasProperties(o);
    const object = if (name.len == 0) has_properties else std.mem.eql(u8, name, "object") and (has_properties or o.get("allOf") == null);
    if (!object) return null;
    const props = o.get("properties") orelse return null;
    if (props != .object) return error.InvalidSchema;
    return props.object;
}

/// Value kinds admitted by `schema`, resolving `$ref`s against `root` (the tool's
/// `parameters`).
pub fn valueTypes(root: std.json.Value, schema: std.json.Value) Error!Types {
    var refs: Refs = .{};
    return typesOf(root, schema, 0, &refs);
}

const Refs = struct {
    stack: [max_refs][]const u8 = undefined,
    len: usize = 0,
    fn contains(self: *const Refs, ref: []const u8) bool {
        for (self.stack[0..self.len]) |r| if (std.mem.eql(u8, r, ref)) return true;
        return false;
    }
};

fn hasProperties(o: std.json.ObjectMap) bool {
    if (o.get("properties") != null) return true;
    const additional = o.get("additionalProperties") orelse return false;
    return !(additional == .bool and additional.bool);
}

fn typesOf(root: std.json.Value, schema: std.json.Value, depth: usize, refs: *Refs) Error!Types {
    if (depth > max_depth or schema != .object) return error.InvalidSchema;
    const o = schema.object;
    if (o.get("$ref")) |ref| {
        if (ref != .string or !std.mem.startsWith(u8, ref.string, "#/")) return error.InvalidSchema;
        // A cycle contributes no type, as in llama.cpp.
        if (refs.contains(ref.string)) return .none;
        if (refs.len == max_refs) return error.InvalidSchema;
        const target = try resolve(root, ref.string);
        refs.stack[refs.len] = ref.string;
        refs.len += 1;
        defer refs.len -= 1;
        return typesOf(root, target, depth + 1, refs);
    }
    if (o.get("oneOf") orelse o.get("anyOf")) |alternatives| {
        const list = try nonEmpty(alternatives);
        var union_: Types = .none;
        for (list) |alt| union_ = union_.@"or"(try typesOf(root, alt, depth + 1, refs));
        return union_;
    }
    const kind = o.get("type") orelse std.json.Value.null;
    if (kind == .array) {
        const list = try nonEmpty(kind);
        var union_: Types = .none;
        for (list) |t| union_ = union_.@"or"(try typesWith(root, o, t, depth + 1, refs));
        return union_;
    }
    return typesWith(root, o, kind, depth, refs);
}

/// `build_node` after `$ref`, alternatives and type arrays, with `type` = `kind`.
fn typesWith(root: std.json.Value, o: std.json.ObjectMap, kind: std.json.Value, depth: usize, refs: *Refs) Error!Types {
    if (o.get("const")) |value| return jsonType(value);
    if (o.get("enum")) |values| {
        var union_: Types = .none;
        for (try nonEmpty(values)) |value| union_ = union_.@"or"(try jsonType(value));
        return union_;
    }
    const name = switch (kind) {
        .null => "",
        .string => |s| s,
        else => return error.InvalidSchema,
    };
    const has_properties = hasProperties(o);
    if (name.len == 0) {
        if (has_properties) return .{ .object = true };
        if (o.get("allOf")) |all| return intersection(root, all, depth, refs);
        if (o.get("items") != null or o.get("prefixItems") != null) return .{ .array = true };
        if (o.get("pattern") != null or o.get("minLength") != null or o.get("maxLength") != null or try knownFormat(o)) return .{ .string = true };
        return .all;
    }
    if (std.mem.eql(u8, name, "object")) {
        if (!has_properties) if (o.get("allOf")) |all| return intersection(root, all, depth, refs);
        return .{ .object = true };
    }
    if (std.mem.eql(u8, name, "string")) {
        if (o.get("allOf")) |all| return intersection(root, all, depth, refs);
        return .{ .string = true };
    }
    if (std.mem.eql(u8, name, "array")) return .{ .array = true };
    if (std.mem.eql(u8, name, "integer")) return .{ .integer = true };
    if (std.mem.eql(u8, name, "number")) return .{ .number = true, .integer = true };
    if (std.mem.eql(u8, name, "boolean")) return .{ .boolean = true };
    if (std.mem.eql(u8, name, "null")) return .{ .null = true };
    return error.InvalidSchema;
}

fn intersection(root: std.json.Value, all: std.json.Value, depth: usize, refs: *Refs) Error!Types {
    var result: Types = .all;
    for (try nonEmpty(all)) |child| result = result.@"and"(try typesOf(root, child, depth + 1, refs));
    return result;
}

fn nonEmpty(v: std.json.Value) Error![]const std.json.Value {
    if (v != .array or v.array.items.len == 0) return error.InvalidSchema;
    return v.array.items;
}

fn knownFormat(o: std.json.ObjectMap) Error!bool {
    const format = o.get("format") orelse return false;
    if (format != .string) return error.InvalidSchema;
    const f = format.string;
    if (std.mem.eql(u8, f, "date") or std.mem.eql(u8, f, "time") or std.mem.eql(u8, f, "date-time") or std.mem.eql(u8, f, "uuid")) return true;
    return f.len == 5 and std.mem.startsWith(u8, f, "uuid") and f[4] >= '1' and f[4] <= '5';
}

fn jsonType(value: std.json.Value) Error!Types {
    return switch (value) {
        .null => .{ .null = true },
        .bool => .{ .boolean = true },
        .integer => .{ .integer = true },
        .float => .{ .number = true },
        .number_string => |s| if (std.mem.indexOfAny(u8, s, ".eE") == null) .{ .integer = true } else .{ .number = true },
        .string => .{ .string = true },
        .array => .{ .array = true },
        .object => .{ .object = true },
    };
}

/// JSON pointer `#/a/b/0` (no `~` escapes, as in llama.cpp).
fn resolve(root: std.json.Value, ref: []const u8) Error!std.json.Value {
    var target = root;
    var it = std.mem.splitScalar(u8, ref[2..], '/');
    while (it.next()) |token| switch (target) {
        .object => |o| target = o.get(token) orelse return error.InvalidSchema,
        .array => |a| {
            const i = std.fmt.parseInt(usize, token, 10) catch return error.InvalidSchema;
            if (i >= a.items.len) return error.InvalidSchema;
            target = a.items[i];
        },
        else => return error.InvalidSchema,
    };
    return target;
}
