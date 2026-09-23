//! File storage only. No GGUF/model interpretation and no eager tensor read/copy.
const std = @import("std");
const builtin = @import("builtin");

pub const MappedFile = struct {
    bytes: []align(std.heap.page_size_min) const u8,

    /// The underlying file must remain immutable/untruncated until deinit.
    pub fn open(io: std.Io, path: []const u8, max_bytes: u64) !MappedFile {
        if (builtin.os.tag != .linux) return error.UnsupportedMappingPlatform;
        const file = try std.Io.Dir.cwd().openFile(io, path, .{});
        defer file.close(io);
        const info = try file.stat(io);
        if (info.kind != .file) return error.NotRegularFile;
        if (info.size == 0) return error.EmptyFile;
        if (info.size > max_bytes or info.size > std.math.maxInt(isize)) return error.FileTooLarge;
        const length = std.math.cast(usize, info.size) orelse return error.FileTooLarge;
        const mapping = try std.posix.mmap(null, length, .{ .READ = true }, .{ .TYPE = .PRIVATE }, file.handle, 0);
        return .{ .bytes = mapping };
    }

    pub fn deinit(self: *MappedFile) void {
        std.posix.munmap(self.bytes);
        self.* = undefined;
    }
};
