//! OpenAI Chat Completions v1 request validation and response encoding. Driver-free.
//! Unsupported features are rejected with OpenAI-style errors, never silently ignored.
const std = @import("std");
const chat = @import("../chat/qwen38.zig");
const pyjson = @import("../chat/pyjson.zig");
const sampler = @import("../session/sampler.zig");
const text = @import("../session/text.zig");
const tools = @import("../session/tools.zig");
const schema = @import("schema.zig");

pub const Limits = struct {
    max_body_bytes: usize = 4 * 1024 * 1024,
    max_messages: usize = 1024,
    max_tools: usize = 128,
    max_tool_calls: usize = 128,
};

/// Defaults applied when a field is absent (served model's recommended values).
pub const Defaults = struct {
    temperature: f32 = 1.0,
    top_k: u32 = 20,
    top_p: f32 = 0.95,
    min_p: f32 = 0.0,
};

pub const ChatRequest = struct {
    model: []const u8,
    messages: []chat.Message,
    template: chat.Options,
    stream: bool = false,
    include_usage: bool = false,
    max_tokens: ?u32 = null,
    params: sampler.Params,
    seed_given: bool = false,
    stops: []const []const u8 = &.{},
    /// Parameter kinds per tool for the output parser (template.tools has the text).
    tools: []const tools.Tool = &.{},
    tool_choice: enum { auto, none } = .auto,
    parallel_tool_calls: bool = true,

    /// Tool-call parsing applies (docs/specs/tool-calling.md).
    pub fn toolMode(self: *const ChatRequest) ?tools.Mode {
        if (self.tools.len == 0 or self.tool_choice == .none) return null;
        return .{ .tools = self.tools, .parallel = self.parallel_tool_calls };
    }
};

pub const ApiError = struct {
    status: std.http.Status = .bad_request,
    message: []const u8,
    kind: []const u8 = "invalid_request_error",
    param: ?[]const u8 = null,
    code: ?[]const u8 = null,
};

pub const Parsed = union(enum) { ok: ChatRequest, err: ApiError };

fn fail(message: []const u8, param: ?[]const u8) Parsed {
    return .{ .err = .{ .message = message, .param = param } };
}
fn unsupported(message: []const u8, param: []const u8) Parsed {
    return .{ .err = .{ .message = message, .param = param, .code = "unsupported_parameter" } };
}

// The body is parsed with `parse_numbers = false` (numbers stay literals, so tool
// JSON can be re-rendered exactly as Python would); these read the literals.
fn number(v: std.json.Value) ?f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        .number_string => |s| std.fmt.parseFloat(f64, s) catch null,
        else => null,
    };
}

fn integer(v: std.json.Value) ?i64 {
    return switch (v) {
        .integer => |i| i,
        .number_string => |s| std.fmt.parseInt(i64, s, 10) catch null,
        else => null,
    };
}

/// `arena` owns every returned slice. `served` lists accepted model ids.
pub fn parseChat(arena: std.mem.Allocator, body: []const u8, served: []const []const u8, defaults: Defaults, limits: Limits) std.mem.Allocator.Error!Parsed {
    if (body.len > limits.max_body_bytes) return .{ .err = .{ .status = .payload_too_large, .message = "request body too large" } };
    const root = std.json.parseFromSliceLeaky(std.json.Value, arena, body, .{ .duplicate_field_behavior = .@"error", .parse_numbers = false }) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return fail("request body is not valid JSON", null),
    };
    if (root != .object) return fail("request body must be a JSON object", null);
    const o = root.object;
    var request: ChatRequest = .{ .model = "", .messages = &.{}, .template = .{}, .params = .{} };

    const model = o.get("model") orelse return fail("'model' is required", "model");
    if (model != .string) return fail("'model' must be a string", "model");
    for (served) |id| {
        if (std.mem.eql(u8, id, model.string)) request.model = id;
    }
    if (request.model.len == 0) return .{ .err = .{ .status = .not_found, .message = "the requested model is not served here", .param = "model", .code = "model_not_found" } };

    // Explicitly unsupported capabilities.
    if (o.get("n")) |v| if (integer(v) != 1 and v != .null) return unsupported("only n=1 is supported", "n");
    if (o.get("logprobs")) |v| if (!(v == .bool and !v.bool) and v != .null) return unsupported("logprobs are not supported", "logprobs");
    if (o.get("top_logprobs")) |v| if (v != .null) return unsupported("top_logprobs are not supported", "top_logprobs");
    if (o.get("functions")) |v| if (v != .null) return unsupported("legacy 'functions' are not supported; use 'tools'", "functions");
    if (o.get("function_call")) |v| if (v != .null) return unsupported("legacy 'function_call' is not supported; use 'tool_choice'", "function_call");
    if (o.get("tool_choice")) |v| switch (v) {
        .null => {},
        .string => |s| {
            if (std.mem.eql(u8, s, "none")) request.tool_choice = .none //
            else if (std.mem.eql(u8, s, "required")) return unsupported("tool_choice 'required' is not supported (needs constrained decoding)", "tool_choice") //
            else if (!std.mem.eql(u8, s, "auto")) return fail("'tool_choice' must be 'none' or 'auto'", "tool_choice");
        },
        .object => return unsupported("forcing a named function is not supported (needs constrained decoding)", "tool_choice"),
        else => return fail("'tool_choice' must be a string", "tool_choice"),
    };
    if (o.get("parallel_tool_calls")) |v| switch (v) {
        .null => {},
        .bool => |b| request.parallel_tool_calls = b,
        else => return fail("'parallel_tool_calls' must be a boolean", "parallel_tool_calls"),
    };
    if (o.get("tools")) |v| if (v != .null) {
        if (v != .array) return fail("'tools' must be an array", "tools");
        if (v.array.items.len > limits.max_tools) return fail("too many tools", "tools");
        const rendered = try arena.alloc([]const u8, v.array.items.len);
        const parsed = try arena.alloc(tools.Tool, v.array.items.len);
        for (v.array.items, rendered, parsed) |tool, *r, *parser| {
            if (try parseTool(arena, tool, r, parser)) |message| return fail(message, "tools");
        }
        request.template.tools = rendered;
        request.tools = parsed;
    };
    if (o.get("logit_bias")) |v| if (!(v == .null or (v == .object and v.object.count() == 0))) return unsupported("logit_bias is not supported", "logit_bias");
    if (o.get("response_format")) |v| if (v != .null) {
        const kind = if (v == .object) v.object.get("type") else null;
        if (kind == null or kind.? != .string or !std.mem.eql(u8, kind.?.string, "text")) return unsupported("only response_format type 'text' is supported", "response_format");
    };
    if (o.get("modalities")) |v| if (v != .null) return unsupported("only text output is supported", "modalities");
    if (o.get("audio")) |v| if (v != .null) return unsupported("audio output is not supported", "audio");
    if (o.get("prediction")) |v| if (v != .null) return unsupported("predicted outputs are not supported", "prediction");

    // Messages.
    const messages = o.get("messages") orelse return fail("'messages' is required", "messages");
    if (messages != .array or messages.array.items.len == 0) return fail("'messages' must be a non-empty array", "messages");
    if (messages.array.items.len > limits.max_messages) return fail("too many messages", "messages");
    const out = try arena.alloc(chat.Message, messages.array.items.len);
    for (messages.array.items, out) |m, *dest| {
        if (m != .object) return fail("each message must be an object", "messages");
        const role = m.object.get("role") orelse return fail("message 'role' is required", "messages");
        if (role != .string) return fail("message 'role' must be a string", "messages");
        dest.* = .{ .role = .user, .content = "" };
        if (std.mem.eql(u8, role.string, "system")) dest.role = .system //
        else if (std.mem.eql(u8, role.string, "user")) dest.role = .user //
        else if (std.mem.eql(u8, role.string, "assistant")) dest.role = .assistant //
        else if (std.mem.eql(u8, role.string, "developer")) dest.role = .developer //
        else if (std.mem.eql(u8, role.string, "tool")) dest.role = .tool //
        else if (std.mem.eql(u8, role.string, "function")) return unsupported("legacy 'function' messages are not supported; use 'tool'", "messages") //
        else return fail("unknown message role", "messages");
        if (m.object.get("tool_calls")) |v| if (v != .null) {
            if (dest.role != .assistant) return fail("'tool_calls' is only valid on assistant messages", "messages");
            if (v != .array) return fail("'tool_calls' must be an array", "messages");
            if (v.array.items.len > limits.max_tool_calls) return fail("too many tool calls in a message", "messages");
            const calls = try arena.alloc(chat.ToolCall, v.array.items.len);
            for (v.array.items, calls) |call, *out_call| {
                if (try parseToolCall(arena, call, out_call)) |message| return fail(message, "messages");
            }
            dest.tool_calls = calls;
        };
        if (m.object.get("tool_call_id")) |v| if (v != .null and v != .string) return fail("'tool_call_id' must be a string", "messages");
        if (m.object.get("function_call")) |v| if (v != .null) return unsupported("legacy 'function_call' is not supported; use 'tool_calls'", "messages");
        if (m.object.get("audio")) |v| if (v != .null) return unsupported("audio messages are not supported", "messages");
        const content = m.object.get("content") orelse std.json.Value.null;
        switch (content) {
            .null => if (dest.role != .assistant and dest.role != .tool) return fail("message 'content' is required", "messages"),
            .string => |s| dest.content = s,
            .array => |parts| {
                var joined: std.ArrayList(u8) = .empty;
                for (parts.items) |part| {
                    if (part != .object) return fail("content parts must be objects", "messages");
                    const kind = part.object.get("type") orelse return fail("content part 'type' is required", "messages");
                    if (kind != .string) return fail("content part 'type' must be a string", "messages");
                    if (!std.mem.eql(u8, kind.string, "text")) return unsupported("only text content parts are supported", "messages");
                    const t = part.object.get("text") orelse return fail("text part requires 'text'", "messages");
                    if (t != .string) return fail("text part 'text' must be a string", "messages");
                    try joined.appendSlice(arena, t.string);
                }
                dest.content = joined.items;
            },
            else => return fail("message 'content' must be a string, array or null", "messages"),
        }
        if (m.object.get("reasoning_content")) |r| switch (r) {
            .null => {},
            .string => |s| {
                if (dest.role != .assistant) return fail("'reasoning_content' is only valid on assistant messages", "messages");
                dest.reasoning_content = s;
            },
            else => return fail("'reasoning_content' must be a string", "messages"),
        };
    }
    request.messages = out;

    // Template controls: OpenAI `reasoning_effort` and llama-server `chat_template_kwargs`.
    if (o.get("reasoning_effort")) |v| if (v != .null) {
        if (v != .string) return fail("'reasoning_effort' must be a string", "reasoning_effort");
        request.template.reasoning_effort = effort(v.string) orelse return fail("'reasoning_effort' must be low, medium, high or xhigh", "reasoning_effort");
    };
    if (o.get("chat_template_kwargs")) |kw| if (kw != .null) {
        if (kw != .object) return fail("'chat_template_kwargs' must be an object", "chat_template_kwargs");
        var it = kw.object.iterator();
        while (it.next()) |entry| {
            const key = entry.key_ptr.*;
            const v = entry.value_ptr.*;
            if (std.mem.eql(u8, key, "enable_thinking") or std.mem.eql(u8, key, "preserve_thinking")) {
                if (v != .bool) return fail("template flags must be booleans", "chat_template_kwargs");
                if (key[0] == 'e') request.template.enable_thinking = v.bool else request.template.preserve_thinking = v.bool;
            } else if (std.mem.eql(u8, key, "reasoning_effort")) {
                if (v != .string) return fail("'reasoning_effort' must be a string", "chat_template_kwargs");
                request.template.reasoning_effort = effort(v.string) orelse return fail("'reasoning_effort' must be low, medium, high or xhigh", "chat_template_kwargs");
            } else return unsupported("unsupported chat_template_kwargs key", "chat_template_kwargs");
        }
    };

    if (o.get("stream")) |v| switch (v) {
        .null => {},
        .bool => |b| request.stream = b,
        else => return fail("'stream' must be a boolean", "stream"),
    };
    if (o.get("stream_options")) |v| if (v != .null) {
        if (v != .object) return fail("'stream_options' must be an object", "stream_options");
        if (v.object.get("include_usage")) |u| switch (u) {
            .null => {},
            .bool => |b| request.include_usage = b,
            else => return fail("'include_usage' must be a boolean", "stream_options"),
        };
    };

    for ([_][]const u8{ "max_completion_tokens", "max_tokens" }) |key| if (o.get(key)) |v| if (v != .null) {
        const i = integer(v) orelse return fail("maximum token count must be a positive integer", key);
        if (i < 1 or i > std.math.maxInt(u32)) return fail("maximum token count must be a positive integer", key);
        const n: u32 = @intCast(i);
        request.max_tokens = if (request.max_tokens) |m| @min(m, n) else n;
    };

    var p: sampler.Params = .{ .temperature = defaults.temperature, .top_k = defaults.top_k, .top_p = defaults.top_p, .min_p = defaults.min_p };
    const floats = .{ .{ "temperature", "temperature", 0.0, 2.0 }, .{ "top_p", "top_p", 0.0, 1.0 }, .{ "min_p", "min_p", 0.0, 1.0 }, .{ "presence_penalty", "presence_penalty", -2.0, 2.0 }, .{ "frequency_penalty", "frequency_penalty", -2.0, 2.0 }, .{ "repetition_penalty", "repeat_penalty", 0.0, 10.0 } };
    inline for (floats) |f| {
        if (o.get(f[1])) |v| if (v != .null) {
            const x = number(v) orelse return fail("sampling parameters must be numbers", f[1]);
            if (!(x >= f[2] and x <= f[3])) return fail("sampling parameter out of range", f[1]);
            @field(p, f[0]) = @floatCast(x);
        };
    }
    if (p.top_p == 0) return fail("'top_p' must be greater than 0", "top_p");
    if (p.repetition_penalty == 0) return fail("'repeat_penalty' must be greater than 0", "repeat_penalty");
    if (o.get("top_k")) |v| if (v != .null) {
        const k = integer(v) orelse return fail("'top_k' must be a non-negative integer", "top_k");
        if (k < 0 or k > 1 << 20) return fail("'top_k' must be a non-negative integer", "top_k");
        p.top_k = @intCast(k);
    };
    if (o.get("seed")) |v| if (v != .null) {
        p.seed = @bitCast(integer(v) orelse return fail("'seed' must be an integer", "seed"));
        request.seed_given = true;
    };
    p.validate() catch return fail("invalid sampling parameters", null);
    request.params = p;

    if (o.get("stop")) |v| switch (v) {
        .null => {},
        .string => |s| {
            const list = try arena.alloc([]const u8, 1);
            list[0] = s;
            request.stops = list;
        },
        .array => |items| {
            const list = try arena.alloc([]const u8, items.items.len);
            for (items.items, list) |item, *slot| {
                if (item != .string) return fail("'stop' entries must be strings", "stop");
                slot.* = item.string;
            }
            request.stops = list;
        },
        else => return fail("'stop' must be a string or array of strings", "stop"),
    };
    _ = text.Stops.init(request.stops) catch return fail("'stop' allows at most 4 non-empty strings of at most 64 bytes", "stop");
    return .{ .ok = request };
}

/// One `tools[]` entry: its template text (the llama.cpp-normalized object as Python
/// `json.dumps`) and its parameter kinds. Returns an error message for invalid input.
fn parseTool(arena: std.mem.Allocator, tool: std.json.Value, rendered: *[]const u8, parsed: *tools.Tool) std.mem.Allocator.Error!?[]const u8 {
    if (tool != .object) return "each tool must be an object";
    const kind = tool.object.get("type") orelse return "tool 'type' is required";
    if (kind != .string or !std.mem.eql(u8, kind.string, "function")) return "only tools of type 'function' are supported";
    const function = tool.object.get("function") orelse return "tool 'function' is required";
    if (function != .object) return "tool 'function' must be an object";
    const name = function.object.get("name") orelse return "function 'name' is required";
    if (name != .string or name.string.len == 0 or name.string.len > tools.max_name) return "function 'name' must be a non-empty string of at most 256 bytes";
    const description = function.object.get("description") orelse std.json.Value{ .string = "" };
    if (description != .string) return "function 'description' must be a string";
    var parameters = function.object.get("parameters") orelse std.json.Value.null;
    if (parameters != .null and parameters != .object) return "function 'parameters' must be an object";
    if (parameters == .null) parameters = .{ .object = .empty };

    var inner: std.json.ObjectMap = .empty;
    try inner.put(arena, "name", name);
    try inner.put(arena, "description", description);
    try inner.put(arena, "parameters", parameters);
    var normalized: std.json.ObjectMap = .empty;
    try normalized.put(arena, "type", kind);
    try normalized.put(arena, "function", .{ .object = inner });
    var out: std.Io.Writer.Allocating = .init(arena);
    pyjson.write(&out.writer, .{ .object = normalized }) catch |e| return switch (e) {
        error.TooDeep => "tool definition is nested too deeply",
        error.InvalidNumber => "tool definition contains an invalid number",
        error.WriteFailed => error.OutOfMemory,
    };
    rendered.* = out.written();

    const props = schema.properties(parameters) catch return "invalid tool 'parameters' schema";
    const params = try arena.alloc(tools.Param, if (props) |p| p.count() else 0);
    if (props) |p| {
        var it = p.iterator();
        var i: usize = 0;
        while (it.next()) |entry| : (i += 1) {
            params[i] = .{ .name = entry.key_ptr.*, .types = schema.valueTypes(parameters, entry.value_ptr.*) catch return "invalid tool 'parameters' schema" };
        }
    }
    parsed.* = .{ .name = name.string, .params = params };
    return null;
}

/// One assistant `tool_calls[]` entry for the template: arguments as a JSON-object
/// string ("" = none) or object; string values raw, others as Python `json.dumps`.
fn parseToolCall(arena: std.mem.Allocator, call: std.json.Value, out: *chat.ToolCall) std.mem.Allocator.Error!?[]const u8 {
    if (call != .object) return "each tool call must be an object";
    const kind = call.object.get("type") orelse return "tool call 'type' is required";
    if (kind != .string or !std.mem.eql(u8, kind.string, "function")) return "only tool calls of type 'function' are supported";
    if (call.object.get("id")) |id| if (id != .string and id != .null) return "tool call 'id' must be a string";
    const function = call.object.get("function") orelse return "tool call 'function' is required";
    if (function != .object) return "tool call 'function' must be an object";
    const name = function.object.get("name") orelse return "tool call function 'name' is required";
    if (name != .string or name.string.len == 0) return "tool call function 'name' must be a non-empty string";
    const raw = function.object.get("arguments") orelse return "tool call function 'arguments' is required";
    const arguments: std.json.Value = switch (raw) {
        .string => |s| if (s.len == 0) {
            out.* = .{ .name = name.string };
            return null;
        } else std.json.parseFromSliceLeaky(std.json.Value, arena, s, .{ .duplicate_field_behavior = .@"error", .parse_numbers = false }) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return "tool call 'arguments' is not valid JSON",
        },
        .object => raw,
        else => return "tool call 'arguments' must be a JSON string or an object",
    };
    if (arguments != .object) return "tool call 'arguments' must be a JSON object";
    const list = try arena.alloc(chat.Argument, arguments.object.count());
    var it = arguments.object.iterator();
    var i: usize = 0;
    while (it.next()) |entry| : (i += 1) {
        const value = entry.value_ptr.*;
        if (value == .string) {
            list[i] = .{ .name = entry.key_ptr.*, .value = value.string };
            continue;
        }
        var text_out: std.Io.Writer.Allocating = .init(arena);
        pyjson.write(&text_out.writer, value) catch |e| return switch (e) {
            error.TooDeep => "tool call 'arguments' are nested too deeply",
            error.InvalidNumber => "tool call 'arguments' contain an invalid number",
            error.WriteFailed => error.OutOfMemory,
        };
        list[i] = .{ .name = entry.key_ptr.*, .value = text_out.written() };
    }
    out.* = .{ .name = name.string, .arguments = list };
    return null;
}

fn effort(s: []const u8) ?@FieldType(chat.Options, "reasoning_effort") {
    if (std.mem.eql(u8, s, "low")) return .low;
    if (std.mem.eql(u8, s, "medium")) return .medium;
    // The template's maximum effort; OpenAI's "high" maps to it.
    if (std.mem.eql(u8, s, "high") or std.mem.eql(u8, s, "xhigh")) return .xhigh;
    return null;
}

const EmptyObject = struct {};
const no_choices = [0]EmptyObject{};

pub const Usage = struct { prompt_tokens: u32, completion_tokens: u32, cached_tokens: u32 = 0 };
pub const Meta = struct { id: []const u8, created: i64, model: []const u8 };

pub fn writeError(w: *std.Io.Writer, e: ApiError) std.Io.Writer.Error!void {
    try std.json.Stringify.value(.{ .@"error" = .{ .message = e.message, .type = e.kind, .param = e.param, .code = e.code } }, .{}, w);
}

pub const CallOut = struct { id: []const u8, type: []const u8 = "function", function: struct { name: []const u8, arguments: []const u8 } };

/// `call_` + the 24 hex digits of the completion id + the call index (2+ hex digits).
pub fn callId(buf: *[40]u8, meta: Meta, index: u32) []const u8 {
    const suffix = meta.id[std.mem.indexOfScalar(u8, meta.id, '-').? + 1 ..];
    return std.fmt.bufPrint(buf, "call_{s}{x:0>2}", .{ suffix, index }) catch unreachable;
}

pub fn writeCompletion(w: *std.Io.Writer, meta: Meta, reasoning: ?[]const u8, content: []const u8, calls: ?[]const CallOut, finish: []const u8, usage: Usage) std.Io.Writer.Error!void {
    const message = .{ .role = "assistant", .content = content, .reasoning_content = reasoning, .tool_calls = calls };
    try std.json.Stringify.value(.{
        .id = meta.id,
        .object = "chat.completion",
        .created = meta.created,
        .model = meta.model,
        .choices = .{.{ .index = 0, .message = message, .logprobs = null, .finish_reason = finish }},
        .usage = .{ .prompt_tokens = usage.prompt_tokens, .completion_tokens = usage.completion_tokens, .total_tokens = usage.prompt_tokens + usage.completion_tokens, .prompt_tokens_details = .{ .cached_tokens = usage.cached_tokens } },
    }, .{ .emit_null_optional_fields = false }, w);
}

pub const Delta = union(enum) {
    role,
    reasoning: []const u8,
    content: []const u8,
    finish: []const u8,
    call_begin: struct { index: u32, id: []const u8, name: []const u8 },
    call_arguments: struct { index: u32, text: []const u8 },
};

/// One SSE event line `data: {...}\n\n`.
pub fn writeChunk(w: *std.Io.Writer, meta: Meta, delta: Delta) std.Io.Writer.Error!void {
    try w.writeAll("data: ");
    const head = .{ .id = meta.id, .object = "chat.completion.chunk", .created = meta.created, .model = meta.model };
    switch (delta) {
        .role => try std.json.Stringify.value(.{ .id = head.id, .object = head.object, .created = head.created, .model = head.model, .choices = .{.{ .index = 0, .delta = .{ .role = "assistant", .content = "" }, .logprobs = null, .finish_reason = null }} }, .{}, w),
        .reasoning => |s| try std.json.Stringify.value(.{ .id = head.id, .object = head.object, .created = head.created, .model = head.model, .choices = .{.{ .index = 0, .delta = .{ .reasoning_content = s }, .logprobs = null, .finish_reason = null }} }, .{}, w),
        .content => |s| try std.json.Stringify.value(.{ .id = head.id, .object = head.object, .created = head.created, .model = head.model, .choices = .{.{ .index = 0, .delta = .{ .content = s }, .logprobs = null, .finish_reason = null }} }, .{}, w),
        .finish => |f| try std.json.Stringify.value(.{ .id = head.id, .object = head.object, .created = head.created, .model = head.model, .choices = .{.{ .index = 0, .delta = EmptyObject{}, .logprobs = null, .finish_reason = f }} }, .{}, w),
        .call_begin => |c| {
            const call = .{ .index = c.index, .id = c.id, .type = "function", .function = .{ .name = c.name, .arguments = "{" } };
            try std.json.Stringify.value(.{ .id = head.id, .object = head.object, .created = head.created, .model = head.model, .choices = .{.{ .index = 0, .delta = .{ .tool_calls = .{call} }, .logprobs = null, .finish_reason = null }} }, .{}, w);
        },
        .call_arguments => |c| {
            const call = .{ .index = c.index, .function = .{ .arguments = c.text } };
            try std.json.Stringify.value(.{ .id = head.id, .object = head.object, .created = head.created, .model = head.model, .choices = .{.{ .index = 0, .delta = .{ .tool_calls = .{call} }, .logprobs = null, .finish_reason = null }} }, .{}, w);
        },
    }
    try w.writeAll("\n\n");
}

pub fn writeUsageChunk(w: *std.Io.Writer, meta: Meta, usage: Usage) std.Io.Writer.Error!void {
    try w.writeAll("data: ");
    try std.json.Stringify.value(.{ .id = meta.id, .object = "chat.completion.chunk", .created = meta.created, .model = meta.model, .choices = no_choices, .usage = .{ .prompt_tokens = usage.prompt_tokens, .completion_tokens = usage.completion_tokens, .total_tokens = usage.prompt_tokens + usage.completion_tokens, .prompt_tokens_details = .{ .cached_tokens = usage.cached_tokens } } }, .{}, w);
    try w.writeAll("\n\n");
}

pub fn writeModels(w: *std.Io.Writer, ids: []const []const u8, created: i64) std.Io.Writer.Error!void {
    try w.writeAll("{\"object\":\"list\",\"data\":[");
    for (ids, 0..) |id, i| {
        if (i > 0) try w.writeByte(',');
        try std.json.Stringify.value(.{ .id = id, .object = "model", .created = created, .owned_by = "zerv" }, .{}, w);
    }
    try w.writeAll("]}");
}
