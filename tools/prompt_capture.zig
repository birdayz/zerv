//! Developer-only capture of production chat parsing/rendering/tokenization, no GPU.
const std = @import("std");
const zerv = @import("zerv");
pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 3) return error.Usage;
    var model = try zerv.artifact.MappedFile.open(init.io, args[1], 64 << 30);
    defer model.deinit();
    var gguf = try zerv.artifact.gguf.Container.parse(a, model.bytes, .{});
    defer gguf.deinit();
    var tokenizer = try zerv.tokenizer.fromGGUF(a, &gguf, .{});
    defer tokenizer.deinit();
    var fixture = try zerv.artifact.MappedFile.open(init.io, args[2], 16 << 20);
    defer fixture.deinit();
    const parsed = try std.json.parseFromSlice([]std.json.Value, a, fixture.bytes, .{});
    defer parsed.deinit();
    var buffer: [8192]u8 = undefined;
    var output: std.Io.File.Writer = .init(.stdout(), init.io, &buffer);
    for (parsed.value, 0..) |body, index| {
        const encoded = try std.json.Stringify.valueAlloc(a, body, .{});
        const request = switch (try zerv.serve.api.parseChat(a, encoded, &.{"qwen3.8-27b"}, .{}, .{})) {
            .ok => |r| r,
            .err => return error.InvalidRequest,
        };
        var prompt: std.Io.Writer.Allocating = .init(a);
        try zerv.chat.qwen38.render(request.messages, request.template, &prompt.writer);
        const ids = try tokenizer.encode(a, prompt.written(), .{});
        try std.json.Stringify.value(.{ .index = index, .prompt = prompt.written(), .tokens = ids }, .{}, &output.interface);
        try output.interface.writeByte('\n');
    }
    try output.interface.flush();
}
