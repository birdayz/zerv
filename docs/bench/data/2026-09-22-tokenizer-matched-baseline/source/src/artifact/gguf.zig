//! Bounded zero-copy GGUF v3 container parsing. No I/O, kernels or model semantics.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const ParseError = error{
    Truncated,
    InvalidMagic,
    UnsupportedVersion,
    UnsupportedValueType,
    UnsupportedNestedArray,
    UnsupportedTensorType,
    InvalidKey,
    InvalidString,
    InvalidBool,
    InvalidRank,
    InvalidDimension,
    InvalidAlignment,
    InvalidTensorOffset,
    DuplicateKey,
    DuplicateTensor,
    Overflow,
    LimitExceeded,
    MissingPayload,
    WrongValueType,
};
pub const Error = ParseError || Allocator.Error;

pub const Limits = struct {
    max_tensors: usize = 16384,
    max_metadata: usize = 65536,
    max_header_bytes: usize = 64 * 1024 * 1024,
    max_string_bytes: usize = 16 * 1024 * 1024,
    max_array_elements: usize = 16777216,
    max_alignment: usize = 1024 * 1024,
};

pub const ValueType = enum(u32) {
    uint8 = 0,
    int8 = 1,
    uint16 = 2,
    int16 = 3,
    uint32 = 4,
    int32 = 5,
    float32 = 6,
    boolean = 7,
    string = 8,
    array = 9,
    uint64 = 10,
    int64 = 11,
    float64 = 12,

    fn width(kind: ValueType) ?usize {
        return switch (kind) {
            .uint8, .int8, .boolean => 1,
            .uint16, .int16 => 2,
            .uint32, .int32, .float32 => 4,
            .uint64, .int64, .float64 => 8,
            .string, .array => null,
        };
    }
};

pub const TensorType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q8_0 = 8,
    q5_k = 13,
    q6_k = 14,
    bf16 = 30,

    pub fn blockElements(kind: TensorType) usize {
        return switch (kind) {
            .f32, .f16, .bf16 => 1,
            .q4_0, .q4_1, .q8_0 => 32,
            .q5_k, .q6_k => 256,
        };
    }

    pub fn blockBytes(kind: TensorType) usize {
        return switch (kind) {
            .f32 => 4,
            .f16, .bf16 => 2,
            .q4_0 => 18,
            .q4_1 => 20,
            .q8_0 => 34,
            .q5_k => 176,
            .q6_k => 210,
        };
    }
};

/// Borrowed validated wire value, excluding its outer type tag.
pub const Value = struct {
    kind: ValueType,
    encoded: []const u8,

    pub fn scalar(value: Value, comptime T: type) ParseError!T {
        const expected: ValueType = switch (T) {
            u8 => .uint8,
            i8 => .int8,
            u16 => .uint16,
            i16 => .int16,
            u32 => .uint32,
            i32 => .int32,
            u64 => .uint64,
            i64 => .int64,
            f32 => .float32,
            f64 => .float64,
            bool => .boolean,
            else => @compileError("unsupported metadata scalar type"),
        };
        if (value.kind != expected) return error.WrongValueType;
        var reader = Cursor.init(value.encoded, value.encoded.len);
        if (T == bool) {
            const raw = try reader.integer(u8);
            if (raw > 1) return error.InvalidBool;
            return raw != 0;
        }
        if (T == f32) return @bitCast(try reader.integer(u32));
        if (T == f64) return @bitCast(try reader.integer(u64));
        return reader.integer(T);
    }

    pub fn string(value: Value) ParseError![]const u8 {
        if (value.kind != .string) return error.WrongValueType;
        var reader = Cursor.init(value.encoded, value.encoded.len);
        return reader.string(std.math.maxInt(usize));
    }

    pub fn array(value: Value) ParseError!Array {
        if (value.kind != .array) return error.WrongValueType;
        var reader = Cursor.init(value.encoded, value.encoded.len);
        const kind = try reader.valueType();
        if (kind == .array) return error.UnsupportedNestedArray;
        const count = try reader.count(std.math.maxInt(usize));
        return .{ .kind = kind, .count = count, .encoded = value.encoded[reader.pos..] };
    }
};

pub const Array = struct {
    kind: ValueType,
    count: usize,
    encoded: []const u8,

    pub fn iterator(array: Array) Iterator {
        return .{ .kind = array.kind, .remaining = array.count, .reader = Cursor.init(array.encoded, array.encoded.len) };
    }

    pub const Iterator = struct {
        kind: ValueType,
        remaining: usize,
        reader: Cursor,

        pub fn next(it: *Iterator) ParseError!?Value {
            if (it.remaining == 0) return null;
            const result = try it.reader.value(it.kind, .{ .max_string_bytes = std.math.maxInt(usize) });
            it.remaining -= 1;
            return result;
        }
    };
};

pub const Metadata = struct { name: []const u8, value: Value };
pub const Tensor = struct {
    name: []const u8,
    kind: TensorType,
    rank: u32,
    dims: [4]u64,
    offset: usize,
    size: usize,
    data: []const u8,
};

/// Input bytes (including the backing file) must remain immutable and alive.
/// Owns index allocations only; deinit before releasing the input storage.
pub const Container = struct {
    allocator: Allocator,
    version: u32 = 3,
    alignment: usize = 32,
    data_offset: usize = 0,
    metadata: []Metadata = &.{},
    tensors: []Tensor = &.{},
    metadata_index: std.StringHashMapUnmanaged(usize) = .empty,
    tensor_index: std.StringHashMapUnmanaged(usize) = .empty,

    pub fn deinit(self: *Container) void {
        self.metadata_index.deinit(self.allocator);
        self.tensor_index.deinit(self.allocator);
        self.allocator.free(self.metadata);
        self.allocator.free(self.tensors);
        self.* = undefined;
    }

    pub fn findMetadata(self: *const Container, name: []const u8) ?Value {
        return self.metadata[self.metadata_index.get(name) orelse return null].value;
    }

    pub fn findTensor(self: *const Container, name: []const u8) ?*const Tensor {
        return &self.tensors[self.tensor_index.get(name) orelse return null];
    }

    pub fn parse(allocator: Allocator, data: []const u8, limits: Limits) Error!Container {
        var reader = Cursor.init(data, limits.max_header_bytes);
        if (!std.mem.eql(u8, try reader.take(4), "GGUF")) return error.InvalidMagic;
        if (try reader.integer(u32) != 3) return error.UnsupportedVersion;
        const tensor_count = try reader.count(limits.max_tensors);
        const metadata_count = try reader.count(limits.max_metadata);
        // Capacity rounds up to a power of two in u32. Leave headroom for
        // load-factor expansion even when a caller relaxes the normal limits.
        const max_index_entries = std.math.maxInt(u32) / 4;
        if (tensor_count > max_index_entries or metadata_count > max_index_entries) return error.LimitExceeded;
        var result = Container{ .allocator = allocator };
        errdefer result.deinit();
        result.metadata = try allocator.alloc(Metadata, metadata_count);
        result.tensors = try allocator.alloc(Tensor, tensor_count);
        try result.metadata_index.ensureTotalCapacity(allocator, @intCast(metadata_count));
        try result.tensor_index.ensureTotalCapacity(allocator, @intCast(tensor_count));

        for (result.metadata, 0..) |*entry, index| {
            const name = try reader.string(@min(limits.max_string_bytes, 65535));
            if (!validKey(name)) return error.InvalidKey;
            const slot = try result.metadata_index.getOrPut(allocator, name);
            if (slot.found_existing) return error.DuplicateKey;
            slot.value_ptr.* = index;
            const kind = try reader.valueType();
            entry.* = .{ .name = name, .value = try reader.value(kind, limits) };
        }
        if (result.findMetadata("general.alignment")) |value| {
            if (value.kind != .uint32) return error.InvalidAlignment;
            result.alignment = try value.scalar(u32);
        }
        if (result.alignment == 0 or !std.math.isPowerOfTwo(result.alignment)) return error.InvalidAlignment;
        if (result.alignment > limits.max_alignment) return error.LimitExceeded;

        var next_offset: usize = 0;
        for (result.tensors, 0..) |*tensor, index| {
            const name = try reader.string(@min(limits.max_string_bytes, 63));
            if (name.len == 0 or std.mem.indexOfScalar(u8, name, 0) != null) return error.InvalidString;
            const slot = try result.tensor_index.getOrPut(allocator, name);
            if (slot.found_existing) return error.DuplicateTensor;
            slot.value_ptr.* = index;
            const rank = try reader.integer(u32);
            if (rank == 0 or rank > 4) return error.InvalidRank;
            var dims: [4]u64 = @splat(1);
            var elements: usize = 1;
            for (dims[0..rank]) |*dimension| {
                dimension.* = try reader.integer(u64);
                if (dimension.* == 0 or dimension.* > std.math.maxInt(i64)) return error.InvalidDimension;
                elements = try multiply(elements, std.math.cast(usize, dimension.*) orelse return error.Overflow);
            }
            if (elements > std.math.maxInt(i64)) return error.Overflow;
            const kind = std.enums.fromInt(TensorType, try reader.integer(u32)) orelse return error.UnsupportedTensorType;
            const block = kind.blockElements();
            if (dims[0] % block != 0) return error.InvalidDimension;
            const size = try multiply(elements / block, kind.blockBytes());
            const offset = std.math.cast(usize, try reader.integer(u64)) orelse return error.Overflow;
            if (offset != next_offset) return error.InvalidTensorOffset;
            next_offset = try add(offset, try alignSize(size, result.alignment));
            tensor.* = .{ .name = name, .kind = kind, .rank = rank, .dims = dims, .offset = offset, .size = size, .data = &.{} };
        }
        result.data_offset = if (tensor_count == 0) reader.pos else try alignSize(reader.pos, result.alignment);
        if (result.data_offset > limits.max_header_bytes) return error.LimitExceeded;
        const required = try add(result.data_offset, next_offset);
        if (required > data.len) return error.MissingPayload;
        for (result.tensors) |*tensor| {
            const begin = result.data_offset + tensor.offset; // bounded by required above
            tensor.data = data[begin..][0..tensor.size];
        }
        return result;
    }
};

fn validKey(name: []const u8) bool {
    if (name.len == 0) return false;
    var segment_empty = true;
    for (name) |byte| {
        if (byte == '.') {
            if (segment_empty) return false;
            segment_empty = true;
        } else {
            if (!(byte >= 'a' and byte <= 'z') and !(byte >= '0' and byte <= '9') and byte != '_') return false;
            segment_empty = false;
        }
    }
    return !segment_empty;
}

fn add(a: usize, b: usize) ParseError!usize {
    return std.math.add(usize, a, b) catch error.Overflow;
}
fn multiply(a: usize, b: usize) ParseError!usize {
    return std.math.mul(usize, a, b) catch error.Overflow;
}
fn alignSize(size: usize, alignment: usize) ParseError!usize {
    return (try add(size, alignment - 1)) & ~(alignment - 1);
}

const Cursor = struct {
    data: []const u8,
    pos: usize = 0,
    limit: usize,

    fn init(data: []const u8, limit: usize) Cursor {
        return .{ .data = data, .limit = @min(data.len, limit) };
    }
    fn take(self: *Cursor, length: usize) ParseError![]const u8 {
        if (length > self.data.len - self.pos) return error.Truncated;
        if (length > self.limit - self.pos) return error.LimitExceeded;
        const result = self.data[self.pos..][0..length];
        self.pos += length;
        return result;
    }
    fn integer(self: *Cursor, comptime T: type) ParseError!T {
        return std.mem.readInt(T, (try self.take(@sizeOf(T)))[0..@sizeOf(T)], .little);
    }
    fn count(self: *Cursor, maximum: usize) ParseError!usize {
        const n = try self.integer(u64);
        if (n > maximum) return error.LimitExceeded;
        return @intCast(n);
    }
    fn string(self: *Cursor, maximum: usize) ParseError![]const u8 {
        const bytes = try self.take(try self.count(maximum));
        if (!std.unicode.utf8ValidateSlice(bytes)) return error.InvalidString;
        return bytes;
    }
    fn valueType(self: *Cursor) ParseError!ValueType {
        return std.enums.fromInt(ValueType, try self.integer(u32)) orelse error.UnsupportedValueType;
    }
    fn value(self: *Cursor, kind: ValueType, limits: Limits) ParseError!Value {
        const start = self.pos;
        switch (kind) {
            .string => _ = try self.string(limits.max_string_bytes),
            .array => {
                const element = try self.valueType();
                if (element == .array) return error.UnsupportedNestedArray;
                const count_value = try self.count(limits.max_array_elements);
                if (element == .string) {
                    for (0..count_value) |_| _ = try self.string(limits.max_string_bytes);
                } else {
                    const bytes = try self.take(try multiply(count_value, element.width().?));
                    if (element == .boolean) for (bytes) |byte| {
                        if (byte > 1) return error.InvalidBool;
                    };
                }
            },
            else => {
                const bytes = try self.take(kind.width().?);
                if (kind == .boolean and bytes[0] > 1) return error.InvalidBool;
            },
        }
        return .{ .kind = kind, .encoded = self.data[start..self.pos] };
    }
};
