//! Bounded, externally serialized raw Vulkan infrastructure; no model math.
pub const Device = @import("device.zig").Device;
pub const Options = @import("device.zig").Options;
pub const Error = @import("device.zig").Error;
pub const Location = @import("device.zig").Location;
pub const Buffer = @import("buffer.zig").Buffer;
pub const Kernel = @import("kernel.zig").Kernel;
pub const Commands = @import("commands.zig").Commands;
pub const Scope = @import("commands.zig").Scope;

/// Driver-free ABI/policy testing; not a model/backend interface.
pub const testing = struct {
    pub const abi = @import("vk.zig");
    pub const memoryType = @import("device.zig").memoryType;
    pub const validRange = @import("buffer.zig").validRange;
    pub const validateCode = @import("kernel.zig").validateCode;
};
