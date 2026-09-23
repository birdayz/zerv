//! Strict Qwen3.8 artifact adapter. Generic byte-BPE tables own their data afterwards.
const std = @import("std");
const gguf = @import("../artifact/gguf.zig");
const bpe = @import("bpe.zig");
pub const Error = bpe.Error || gguf.ParseError || error{UnsupportedProfile};
const additions = [_][]const u8{
    "<|endoftext|>",  "<|im_start|>",          "<|im_end|>",      "<|object_ref_start|>", "<|object_ref_end|>",
    "<|box_start|>",  "<|box_end|>",           "<|quad_start|>",  "<|quad_end|>",         "<|vision_start|>",
    "<|vision_end|>", "<|vision_pad|>",        "<|image_pad|>",   "<|video_pad|>",        "<tool_call>",
    "</tool_call>",   "<|fim_prefix|>",        "<|fim_middle|>",  "<|fim_suffix|>",       "<|fim_pad|>",
    "<|repo_name|>",  "<|file_sep|>",          "<tool_response>", "</tool_response>",     "<think>",
    "</think>",       "<|audio_start|>",       "<|audio_end|>",   "<tts_pad>",            "<tts_text_bos>",
    "<tts_text_eod>", "<tts_text_bos_single>", "<|audio_pad|>",
};

fn direct(cp: u21) bool {
    return (cp >= 33 and cp <= 126) or (cp >= 161 and cp <= 172) or (cp >= 174 and cp <= 255);
}
const escaped = blk: {
    var bytes: [68]u8 = undefined;
    var at: usize = 0;
    for (0..256) |cp| if (!direct(@intCast(cp))) {
        bytes[at] = @intCast(cp);
        at += 1;
    };
    break :blk bytes;
};
fn byte(cp: u21) Error!u8 {
    if (direct(cp)) return @intCast(cp);
    if (cp >= 256 and cp < 256 + escaped.len) return escaped[cp - 256];
    return error.InvalidVocabulary;
}
fn value(container: *const gguf.Container, key: []const u8) Error!gguf.Value {
    return container.findMetadata(key) orelse error.UnsupportedProfile;
}

pub fn fromGGUF(allocator: std.mem.Allocator, container: *const gguf.Container, limits: bpe.InitLimits) Error!bpe.Tokenizer {
    if (!std.mem.eql(u8, try (try value(container, "tokenizer.ggml.model")).string(), "gpt2") or
        !std.mem.eql(u8, try (try value(container, "tokenizer.ggml.pre")).string(), "qwen35")) return error.UnsupportedProfile;
    const vocab = try (try value(container, "tokenizer.ggml.tokens")).array();
    const types = try (try value(container, "tokenizer.ggml.token_type")).array();
    const merges = try (try value(container, "tokenizer.ggml.merges")).array();
    if (vocab.kind != .string or types.kind != .int32 or merges.kind != .string or
        vocab.count != 248320 or types.count != vocab.count or merges.count != 247587) return error.UnsupportedProfile;
    if (vocab.count > limits.max_vocabulary or merges.count > limits.max_merges) return error.LimitExceeded;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const temp = arena.allocator();
    const entries = try temp.alloc(bpe.Entry, vocab.count);
    const rules = try temp.alloc(bpe.Merge, merges.count);
    const pool = try temp.alloc(u8, vocab.encoded.len);
    var names: std.StringHashMapUnmanaged(u32) = .empty;
    try names.ensureTotalCapacity(temp, @intCast(vocab.count));
    var vit = vocab.iterator();
    var tit = types.iterator();
    var used: usize = 0;
    for (entries, 0..) |*entry, id| {
        const text = try (try vit.next()).?.string();
        const kind = try (try tit.next()).?.scalar(i32);
        const slot = names.getOrPutAssumeCapacity(text);
        if (slot.found_existing) return error.InvalidVocabulary;
        slot.value_ptr.* = @intCast(id);
        const start = used;
        if (id < 248044) {
            if (kind != 1) return error.UnsupportedProfile;
            var it = (std.unicode.Utf8View.init(text) catch return error.InvalidVocabulary).iterator();
            while (it.nextCodepoint()) |cp| {
                pool[used] = try byte(cp);
                used += 1;
            }
            entry.* = .{ .id = @intCast(id), .kind = .normal, .bytes = pool[start..used] };
        } else if (id < 248077) {
            const expected_type: i32 = if (id == 248058 or id == 248059 or (id >= 248066 and id <= 248069)) 4 else 3;
            if (kind != expected_type or !std.mem.eql(u8, text, additions[id - 248044])) return error.UnsupportedProfile;
            entry.* = .{ .id = @intCast(id), .kind = if (expected_type == 3) .control else .added, .bytes = text };
        } else {
            var buffer: [32]u8 = undefined;
            const expected = std.fmt.bufPrint(&buffer, "[PAD{d}]", .{id}) catch unreachable;
            if (kind != 5 or !std.mem.eql(u8, text, expected)) return error.UnsupportedProfile;
            entry.* = .{ .id = @intCast(id), .kind = .unused, .bytes = &.{} };
        }
    }
    var mit = merges.iterator();
    var joined: std.ArrayList(u8) = .empty;
    for (rules) |*rule| {
        const text = try (try mit.next()).?.string();
        const space = std.mem.indexOfScalar(u8, text, ' ') orelse return error.InvalidMerge;
        if (std.mem.indexOfScalar(u8, text[space + 1 ..], ' ') != null) return error.InvalidMerge;
        const left = names.get(text[0..space]) orelse return error.InvalidMerge;
        const right = names.get(text[space + 1 ..]) orelse return error.InvalidMerge;
        try joined.resize(temp, text.len - 1);
        @memcpy(joined.items[0..space], text[0..space]);
        @memcpy(joined.items[space..], text[space + 1 ..]);
        const result = names.get(joined.items) orelse return error.InvalidMerge;
        rule.* = .{ .left = left, .right = right, .result = result };
    }
    return bpe.Tokenizer.init(allocator, entries, rules, limits);
}
