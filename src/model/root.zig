//! Native Qwen3.8 (qwen35) forward pass: configuration, layout and resident runtime.
pub const config = @import("config.zig");
pub const layout = @import("layout.zig");
pub const gemm = @import("gemm.zig");
pub const attention = @import("attention.zig");
pub const Model = @import("runtime.zig").Model;
pub const Options = @import("runtime.zig").Options;
pub const snapshot_bytes = @import("runtime.zig").snapshot_bytes;
pub const vram_headroom = @import("runtime.zig").vram_headroom;
pub const Capture = @import("runtime.zig").Capture;
pub const Error = @import("runtime.zig").Error;
pub const Hooks = @import("runtime.zig").Hooks;
pub const Probe = @import("runtime.zig").Probe;
pub const Phase = @import("runtime.zig").Phase;
pub const Plan = @import("runtime.zig").Plan;
pub const max_plans = @import("runtime.zig").max_plans;
pub const Chunk = @import("runtime.zig").Chunk;
pub const makePlans = @import("runtime.zig").makePlans;
pub const chunkFor = @import("runtime.zig").chunkFor;
/// f16-mode producer kernels (model.comp with F16OUT; block 16b) and their modules.
pub const F16Producer = @import("runtime.zig").HKernel;
pub const f16ProducerModule = @import("runtime.zig").hModule;
