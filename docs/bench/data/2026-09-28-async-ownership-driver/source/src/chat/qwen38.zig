//! Official Qwen3.8 prompt rendering: text messages, tools, tool calls and tool
//! responses (docs/specs/chat-template.md, docs/specs/tool-calling.md). No tokenizer or
//! model execution. JSON values arrive pre-rendered (see pyjson.zig).
const std = @import("std");

pub const Role = enum { system, user, assistant, developer, tool };
/// One `<parameter=name>` of a historical call. `value` is already rendered: the raw
/// string for string values, Python `json.dumps` text otherwise.
pub const Argument = struct { name: []const u8, value: []const u8 };
pub const ToolCall = struct { name: []const u8, arguments: []const Argument = &.{} };
pub const Message = struct {
    role: Role,
    content: []const u8,
    reasoning_content: []const u8 = "",
    /// Assistant messages only.
    tool_calls: []const ToolCall = &.{},
};
pub const Options = struct {
    /// Tool definitions, each already rendered as the template's `tool|tojson`.
    tools: []const []const u8 = &.{},
    enable_thinking: bool = true,
    reasoning_effort: enum { xhigh, medium, low } = .xhigh,
    preserve_thinking: bool = true,
    add_generation_prompt: bool = true,
    max_messages: usize = 1024,
    max_content_bytes: usize = 1024 * 1024,
};
pub const Error = error{ EmptyMessages, MissingUserQuery, MisplacedSystem, UnsupportedRole, InvalidUtf8, LimitExceeded } || std.Io.Writer.Error;

const xhigh = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
const low = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";
const end = "<|im_end|>\n";

/// Caller owns inputs/writer. Validation errors never write; writer errors may.
pub fn render(messages: []const Message, options: Options, writer: *std.Io.Writer) Error!void {
    if (messages.len == 0) return error.EmptyMessages;
    if (messages.len > options.max_messages) return error.LimitExceeded;
    var remaining = options.max_content_bytes;
    for (options.tools) |tool| try budget(&remaining, tool);
    var last_user: ?usize = null;
    for (messages, 0..) |message, index| {
        if (message.role == .developer) return error.UnsupportedRole;
        if (message.role == .system and index != 0) return error.MisplacedSystem;
        for ([_][]const u8{ message.content, message.reasoning_content }) |text| try budget(&remaining, text);
        if (message.tool_calls.len > 0 and message.role != .assistant) return error.UnsupportedRole;
        for (message.tool_calls) |call| {
            try budget(&remaining, call.name);
            for (call.arguments) |argument| {
                try budget(&remaining, argument.name);
                try budget(&remaining, argument.value);
            }
        }
        if (message.role == .user) {
            const content = strip(message.content);
            if (!(std.mem.startsWith(u8, content, "<tool_response>") and std.mem.endsWith(u8, content, "</tool_response>"))) last_user = index;
        }
    }
    const cutoff = last_user orelse return error.MissingUserQuery;
    const instructions = if (!options.enable_thinking) "" else switch (options.reasoning_effort) {
        .xhigh => xhigh,
        .medium => "",
        .low => low,
    };
    const system = if (messages[0].role == .system) strip(messages[0].content) else "";
    if (options.tools.len > 0) {
        try writer.writeAll("<|im_start|>system\n");
        if (instructions.len != 0) {
            try writer.writeAll(instructions);
            try writer.writeAll("\n\n");
        }
        try writer.writeAll("# Tools\n\nYou have access to the following functions:\n\n<tools>");
        for (options.tools) |tool| {
            try writer.writeByte('\n');
            try writer.writeAll(tool);
        }
        try writer.writeAll("\n</tools>");
        try writer.writeAll(tool_instructions);
        if (system.len != 0) {
            try writer.writeAll("\n\n");
            try writer.writeAll(system);
        }
        try writer.writeAll(end);
    } else if (system.len != 0 or instructions.len != 0) {
        try writer.writeAll("<|im_start|>system\n");
        try writer.writeAll(instructions);
        if (instructions.len != 0 and system.len != 0) try writer.writeAll("\n\n");
        try writer.writeAll(system);
        try writer.writeAll(end);
    }
    for (messages, 0..) |message, index| {
        const content = strip(message.content);
        switch (message.role) {
            .system => continue,
            .developer => unreachable, // rejected before any output
            .user => try writer.writeAll("<|im_start|>user\n"),
            .assistant => {
                try writer.writeAll("<|im_start|>assistant\n");
                if (options.preserve_thinking or index > cutoff) {
                    try writer.writeAll("<think>\n");
                    try writer.writeAll(strip(message.reasoning_content));
                    try writer.writeAll("\n</think>\n\n");
                }
                try writer.writeAll(content);
                for (message.tool_calls, 0..) |call, i| {
                    try writer.writeAll(if (i > 0) "\n<tool_call>\n<function=" else if (content.len > 0) "\n\n<tool_call>\n<function=" else "<tool_call>\n<function=");
                    try writer.writeAll(call.name);
                    try writer.writeAll(">\n");
                    for (call.arguments) |argument| {
                        try writer.writeAll("<parameter=");
                        try writer.writeAll(argument.name);
                        try writer.writeAll(">\n");
                        try writer.writeAll(argument.value);
                        try writer.writeAll("\n</parameter>\n");
                    }
                    try writer.writeAll("</function>\n</tool_call>");
                }
                try writer.writeAll(end);
                continue;
            },
            .tool => {
                // A run of tool messages shares one user turn. As in the template, a tool
                // message at index 0 opens none (`loop.previtem` is undefined there).
                if (index > 0 and messages[index - 1].role != .tool) try writer.writeAll("<|im_start|>user");
                try writer.writeAll("\n<tool_response>\n");
                try writer.writeAll(content);
                try writer.writeAll("\n</tool_response>");
                if (index + 1 == messages.len or messages[index + 1].role != .tool) try writer.writeAll(end);
                continue;
            },
        }
        try writer.writeAll(content);
        try writer.writeAll(end);
    }
    if (options.add_generation_prompt) {
        try writer.writeAll("<|im_start|>assistant\n<think>\n");
        if (!options.enable_thinking) try writer.writeAll("\n</think>\n\n");
    }
}

fn budget(remaining: *usize, text: []const u8) Error!void {
    if (text.len > remaining.*) return error.LimitExceeded;
    remaining.* -= text.len;
    if (!std.unicode.utf8ValidateSlice(text)) return error.InvalidUtf8;
}

const tool_instructions = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

// Python/Jinja str.strip's fixed Unicode whitespace set (not ASCII-only trim).
fn whitespace(cp: u21) bool {
    return switch (cp) {
        0x09...0x0d, 0x1c...0x20, 0x85, 0xa0, 0x1680, 0x2000...0x200a, 0x2028...0x2029, 0x202f, 0x205f, 0x3000 => true,
        else => false,
    };
}

/// Input has already passed UTF-8 validation. Return a borrowed edge-stripped view.
fn strip(text: []const u8) []const u8 {
    var begin: usize = 0;
    var finish = text.len;
    while (begin < finish) {
        const len = std.unicode.utf8ByteSequenceLength(text[begin]) catch unreachable;
        const cp = std.unicode.utf8Decode(text[begin..][0..len]) catch unreachable;
        if (!whitespace(cp)) break;
        begin += len;
    }
    while (finish > begin) {
        var start = finish - 1;
        while (text[start] & 0xc0 == 0x80) start -= 1;
        const cp = std.unicode.utf8Decode(text[start..finish]) catch unreachable;
        if (!whitespace(cp)) break;
        finish = start;
    }
    return text[begin..finish];
}
