const std = @import("std");
const builtin = @import("builtin");
const vk = @import("vk.zig");

pub const Error = error{ UnsupportedTarget, InvalidLimit, EnumerationLimit, NoDevice, InvalidDriverProperties, DriverError, HostOutOfMemory, DeviceOutOfMemory, DeviceLost, Timeout, ResourceLimit, ResourceInUse, InvalidState, InvalidRange, UnsupportedMemoryType, NotHostVisible, WrongDevice, InvalidShader, InvalidLayout, InvalidDispatch, UnsupportedFeature };
pub const Location = enum { device, host };
pub const Options = struct {
    device_index: ?u32 = null,
    max_allocated_bytes: u64,
    /// Enable `VK_KHR_cooperative_matrix` with f16 shader arithmetic, 16-bit storage buffer
    /// access and the Vulkan memory model (`VK_KHR_vulkan_memory_model`, which coopmat
    /// shaders declare); `open` fails with `UnsupportedFeature` if any is missing.
    cooperative_matrix: bool = false,
    /// Enable `VK_EXT_subgroup_size_control` so kernels may require a subgroup size
    /// (`Kernel.Options.subgroup_size`); `open` fails with `UnsupportedFeature` if the
    /// extension, the feature, or required sizes for compute are unavailable. Also enables
    /// `computeFullSubgroups` when supported (`Device.full_subgroups`).
    subgroup_size_control: bool = false,
    /// Enable `storageBuffer16BitAccess` alone (16-bit loads/stores and conversions, no
    /// f16 arithmetic; e.g. an f16 KV cache). Implied by `cooperative_matrix`. `open` fails
    /// with `UnsupportedFeature` if it is unavailable.
    storage16: bool = false,
    /// Enable `VK_KHR_pipeline_binary` (with its dependency chain at Vulkan 1.1) when the
    /// device supports it, so kernels may be created from our own machine code
    /// (`Kernel.Options.binary`). Unsupported is not an error: `Device.pipeline_key` stays
    /// null and such kernels use their SPIR-V (docs/specs/gpu-driver.md, "Pipeline binaries").
    pipeline_binaries: bool = false,
    /// Enable `VK_EXT_external_memory_host` so host buffers may use memory the process
    /// allocated (`Buffer.initImported`); `open` fails with `UnsupportedFeature` if the
    /// extension or its entry point is missing.
    host_import: bool = false,
};
/// `VK_KHR_pipeline_binary` and the extensions it depends on at Vulkan 1.1
/// (maintenance5 -> dynamic_rendering -> depth_stencil_resolve -> create_renderpass2).
const pipeline_binary_extensions = [_][*:0]const u8{ "VK_KHR_pipeline_binary", "VK_KHR_maintenance5", "VK_KHR_dynamic_rendering", "VK_KHR_depth_stencil_resolve", "VK_KHR_create_renderpass2" };
const cooperative_matrix_extensions = [_][]const u8{ "VK_KHR_cooperative_matrix", "VK_KHR_vulkan_memory_model" };

/// Subgroup capabilities (VkPhysicalDeviceSubgroupProperties).
pub const Subgroup = struct {
    size: u32 = 0,
    stages: u32 = 0,
    operations: u32 = 0,
    /// Compute shaders may use basic + ballot subgroup operations and a subgroup never
    /// spans more than `max_size` invocations.
    /// Compute shaders may use subgroup arithmetic (reductions) with subgroups of at least
    /// `min_size` invocations.
    pub fn computeArithmeticFrom(self: Subgroup, min_size: u32) bool {
        return self.size >= min_size and self.stages & vk.VK_SHADER_STAGE_COMPUTE_BIT != 0 and self.operations & vk.VK_SUBGROUP_FEATURE_ARITHMETIC_BIT != 0;
    }
    pub fn computeBallotWithin(self: Subgroup, max_size: u32) bool {
        const needed = vk.VK_SUBGROUP_FEATURE_BASIC_BIT | vk.VK_SUBGROUP_FEATURE_BALLOT_BIT;
        return self.size >= 1 and self.size <= max_size and std.math.isPowerOfTwo(self.size) and
            self.stages & vk.VK_SHADER_STAGE_COMPUTE_BIT != 0 and self.operations & needed == needed;
    }
};

/// Stable-address owner. Externally serialize this device and all its children.
pub const Device = struct {
    instance: vk.VkInstance = null,
    physical: vk.VkPhysicalDevice = null,
    handle: vk.VkDevice = null,
    queue: vk.VkQueue = null,
    family: u32 = 0,
    properties: vk.VkPhysicalDeviceProperties = .{},
    /// Vulkan 1.1 subgroup properties of the selected device (queried at open).
    subgroup: Subgroup = .{},
    /// Cooperative matrices (and f16 arithmetic/storage) are enabled on this device.
    cooperative_matrix: bool = false,
    /// 16-bit storage buffer access is enabled (`Options.storage16` or cooperative matrices).
    storage16: bool = false,
    /// Subgroup sizes a kernel may require (both 0 unless subgroup size control is enabled).
    subgroup_sizes: struct { min: u32 = 0, max: u32 = 0 } = .{},
    /// Kernels may require full subgroups (`Kernel.Options.full_subgroups`): the
    /// `computeFullSubgroups` feature is supported and enabled.
    full_subgroups: bool = false,
    memory: vk.VkPhysicalDeviceMemoryProperties = .{},
    /// `VK_EXT_memory_budget` is supported: `heapBudget` can report memory in use by
    /// other processes (queried with `memoryBudget`).
    memory_budget: bool = false,
    /// The driver's global pipeline-binary key (`vkGetPipelineKeyKHR` without create info)
    /// when pipeline binaries are enabled; null otherwise.
    pipeline_key: ?[32]u8 = null,
    create_pipeline_binaries: ?vk.PFN_vkCreatePipelineBinariesKHR = null,
    /// `VK_EXT_external_memory_host` enabled (`Options.host_import`).
    host_pointer_properties: ?vk.PFN_vkGetMemoryHostPointerPropertiesEXT = null,
    host_import_alignment: u64 = 0,
    destroy_pipeline_binary: ?vk.PFN_vkDestroyPipelineBinaryKHR = null,
    budget: u64,
    allocated_bytes: u64 = 0,
    buffers: u32 = 0,
    kernels: u32 = 0,
    commands: u32 = 0,
    pending: u32 = 0,
    last_result: i32 = 0,
    lost: bool = false,

    /// The exact usage checked for external-buffer import support.
    pub const buffer_usage = vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT | vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    pub fn open(options: Options) Error!Device {
        if (builtin.os.tag != .linux or builtin.cpu.arch != .x86_64) return error.UnsupportedTarget;
        if (options.max_allocated_bytes == 0) return error.InvalidLimit;
        var self: Device = .{ .budget = options.max_allocated_bytes };
        const app: vk.VkApplicationInfo = .{ .pApplicationName = "zerv", .apiVersion = vk.VK_API_VERSION_1_1 };
        const info: vk.VkInstanceCreateInfo = .{ .pApplicationInfo = &app };
        try self.check(vk.vkCreateInstance(&info, null, &self.instance));
        errdefer vk.vkDestroyInstance(self.instance, null);
        var devices: [16]vk.VkPhysicalDevice = undefined;
        var count: u32 = devices.len;
        const result = vk.vkEnumeratePhysicalDevices(self.instance, &count, &devices);
        if (result == vk.VK_INCOMPLETE) return error.EnumerationLimit;
        try self.check(result);
        if (count > devices.len) return error.InvalidDriverProperties;
        for (devices[0..count], 0..) |physical, index| {
            if (options.device_index) |wanted| if (wanted != index) continue;
            var properties: vk.VkPhysicalDeviceProperties = .{};
            vk.vkGetPhysicalDeviceProperties(physical, &properties);
            if (properties.apiVersion < vk.VK_API_VERSION_1_1) continue;
            if (options.device_index == null and properties.deviceType != vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) continue;
            var queue_count: u32 = 0;
            vk.vkGetPhysicalDeviceQueueFamilyProperties(physical, &queue_count, null);
            if (queue_count > 32) return error.EnumerationLimit;
            var families: [32]vk.VkQueueFamilyProperties = undefined;
            vk.vkGetPhysicalDeviceQueueFamilyProperties(physical, &queue_count, &families);
            if (queue_count > families.len) return error.InvalidDriverProperties;
            var selected: ?u32 = null;
            for (families[0..queue_count], 0..) |family, i| {
                if (family.queueCount == 0 or family.queueFlags & vk.VK_QUEUE_COMPUTE_BIT == 0) continue;
                if (selected == null or family.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT == 0) selected = @intCast(i);
            }
            self.family = selected orelse continue;
            self.physical = physical;
            self.properties = properties;
            break;
        }
        if (self.physical == null) return error.NoDevice;
        var subgroup: vk.VkPhysicalDeviceSubgroupProperties = .{};
        var properties2: vk.VkPhysicalDeviceProperties2 = .{ .pNext = &subgroup };
        vk.vkGetPhysicalDeviceProperties2(self.physical, &properties2);
        self.subgroup = .{ .size = subgroup.subgroupSize, .stages = subgroup.supportedStages, .operations = subgroup.supportedOperations };
        vk.vkGetPhysicalDeviceMemoryProperties(self.physical, &self.memory);
        if (self.memory.memoryTypeCount > 32 or self.memory.memoryHeapCount > 16) return error.InvalidDriverProperties;
        self.memory_budget = try self.hasExtension("VK_EXT_memory_budget");
        const priority: f32 = 1;
        const queue_info: vk.VkDeviceQueueCreateInfo = .{ .queueFamilyIndex = self.family, .queueCount = 1, .pQueuePriorities = &priority };
        var device_info: vk.VkDeviceCreateInfo = .{ .queueCreateInfoCount = 1, .pQueueCreateInfos = &queue_info };
        // Enabled-feature chain (only the required members set), used with cooperative matrices.
        var memory_model: vk.VkPhysicalDeviceVulkanMemoryModelFeatures = .{ .vulkanMemoryModel = vk.VK_TRUE };
        var storage16: vk.VkPhysicalDevice16BitStorageFeatures = .{ .pNext = &memory_model, .storageBuffer16BitAccess = vk.VK_TRUE };
        var float16: vk.VkPhysicalDeviceShaderFloat16Int8Features = .{ .pNext = &storage16, .shaderFloat16 = vk.VK_TRUE };
        var coopmat: vk.VkPhysicalDeviceCooperativeMatrixFeaturesKHR = .{ .pNext = &float16, .cooperativeMatrix = vk.VK_TRUE };
        var extensions: [4 + pipeline_binary_extensions.len][*c]const u8 = undefined;
        var extension_count: u32 = 0;
        var sizes: vk.VkPhysicalDeviceSubgroupSizeControlFeatures = .{ .subgroupSizeControl = vk.VK_TRUE };
        // 16-bit storage alone (a chain of its own; coopmat's chain includes it).
        var storage16_only: vk.VkPhysicalDevice16BitStorageFeatures = .{ .storageBuffer16BitAccess = vk.VK_TRUE };
        if (options.cooperative_matrix) {
            try self.requireCooperativeMatrix();
            device_info.pNext = &coopmat;
            extensions[0] = "VK_KHR_cooperative_matrix";
            extensions[1] = "VK_KHR_vulkan_memory_model";
            extension_count = 2;
        } else if (options.storage16) {
            try self.requireStorage16();
            device_info.pNext = &storage16_only;
        }
        if (options.subgroup_size_control) {
            sizes.computeFullSubgroups = if (try self.requireSubgroupSizeControl()) vk.VK_TRUE else 0;
            sizes.pNext = @constCast(device_info.pNext);
            device_info.pNext = &sizes;
            extensions[extension_count] = "VK_EXT_subgroup_size_control";
            extension_count += 1;
        }
        var binaries: vk.VkPhysicalDevicePipelineBinaryFeaturesKHR = .{ .pipelineBinaries = vk.VK_TRUE };
        const use_binaries = options.pipeline_binaries and try self.supportsPipelineBinaries();
        if (use_binaries) {
            binaries.pNext = @constCast(device_info.pNext);
            device_info.pNext = &binaries;
            for (pipeline_binary_extensions) |name_z| {
                extensions[extension_count] = name_z;
                extension_count += 1;
            }
        }
        if (options.host_import) {
            if (!try self.hasExtension("VK_EXT_external_memory_host")) return error.UnsupportedFeature;
            var host_props: vk.VkPhysicalDeviceExternalMemoryHostPropertiesEXT = .{};
            var host_properties2: vk.VkPhysicalDeviceProperties2 = .{ .pNext = &host_props };
            vk.vkGetPhysicalDeviceProperties2(self.physical, &host_properties2);
            const alignment = host_props.minImportedHostPointerAlignment;
            if (alignment == 0 or !std.math.isPowerOfTwo(alignment)) return error.InvalidDriverProperties;
            self.host_import_alignment = alignment;
            const external_info: vk.VkPhysicalDeviceExternalBufferInfo = .{ .usage = buffer_usage, .handleType = vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT };
            var external_props: vk.VkExternalBufferProperties = .{};
            vk.vkGetPhysicalDeviceExternalBufferProperties(self.physical, &external_info, &external_props);
            const features = external_props.externalMemoryProperties.externalMemoryFeatures;
            if (features & vk.VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT == 0 or features & vk.VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT != 0) return error.UnsupportedFeature;
            extensions[extension_count] = "VK_EXT_external_memory_host";
            extension_count += 1;
        }
        device_info.enabledExtensionCount = extension_count;
        device_info.ppEnabledExtensionNames = if (extension_count == 0) null else &extensions;
        try self.check(vk.vkCreateDevice(self.physical, &device_info, null, &self.handle));
        errdefer vk.vkDestroyDevice(self.handle, null);
        self.cooperative_matrix = options.cooperative_matrix;
        self.storage16 = options.cooperative_matrix or options.storage16;
        self.full_subgroups = sizes.computeFullSubgroups == vk.VK_TRUE and options.subgroup_size_control;
        vk.vkGetDeviceQueue(self.handle, self.family, 0, &self.queue);
        if (use_binaries) try self.loadPipelineBinaries();
        if (options.host_import) {
            const f = vk.vkGetDeviceProcAddr(self.handle, "vkGetMemoryHostPointerPropertiesEXT") orelse return error.UnsupportedFeature;
            self.host_pointer_properties = @ptrCast(f);
        }
        return self;
    }

    /// The pipeline-binary extensions and the `pipelineBinaries` feature are supported.
    fn supportsPipelineBinaries(self: *Device) Error!bool {
        for (pipeline_binary_extensions) |wanted| {
            if (!try self.hasExtension(std.mem.span(wanted))) return false;
        }
        var binaries: vk.VkPhysicalDevicePipelineBinaryFeaturesKHR = .{};
        var features: vk.VkPhysicalDeviceFeatures2 = .{ .pNext = &binaries };
        vk.vkGetPhysicalDeviceFeatures2(self.physical, &features);
        return binaries.pipelineBinaries == vk.VK_TRUE;
    }

    /// Fetches the extension's entry points and the global key. A missing entry point or a
    /// key that is not 32 bytes leaves `pipeline_key` null (kernels then use their SPIR-V).
    fn loadPipelineBinaries(self: *Device) Error!void {
        const create = vk.vkGetDeviceProcAddr(self.handle, "vkCreatePipelineBinariesKHR");
        const destroy = vk.vkGetDeviceProcAddr(self.handle, "vkDestroyPipelineBinaryKHR");
        const get_key = vk.vkGetDeviceProcAddr(self.handle, "vkGetPipelineKeyKHR");
        if (create == null or destroy == null or get_key == null) return;
        const getKey: vk.PFN_vkGetPipelineKeyKHR = @ptrCast(get_key.?);
        var key: vk.VkPipelineBinaryKeyKHR = .{};
        try self.check(getKey(self.handle, null, &key));
        if (key.keySize != 32) return;
        self.create_pipeline_binaries = @ptrCast(create.?);
        self.destroy_pipeline_binary = @ptrCast(destroy.?);
        self.pipeline_key = key.key[0..32].*;
    }

    /// The physical device lists device extension `wanted`.
    fn hasExtension(self: *Device, wanted: []const u8) Error!bool {
        var available: [512]vk.VkExtensionProperties = undefined;
        var count: u32 = available.len;
        const result = vk.vkEnumerateDeviceExtensionProperties(self.physical, null, &count, &available);
        if (result == vk.VK_INCOMPLETE) return error.EnumerationLimit;
        try self.check(result);
        if (count > available.len) return error.InvalidDriverProperties;
        for (available[0..count]) |extension| {
            if (std.mem.eql(u8, std.mem.sliceTo(&extension.extensionName, 0), wanted)) return true;
        }
        return false;
    }

    pub const Budget = struct {
        /// Heap of the memory type `.device` buffers use.
        heap: u32,
        size: u64,
        /// What this process may use (the driver's estimate, including other processes).
        budget: u64,
        /// What this process uses now.
        usage: u64,
        /// Still available to this process: `budget - usage` (0 if over budget).
        pub fn free(self: Budget) u64 {
            return self.budget -| self.usage;
        }
    };

    /// Current budget of the device-local heap (`VK_EXT_memory_budget`); null when the
    /// extension is unsupported.
    pub fn memoryBudget(self: *Device) Error!?Budget {
        try self.ready();
        if (!self.memory_budget) return null;
        const heap = self.memory.memoryTypes[try memoryType(&self.memory, std.math.maxInt(u32), .device)].heapIndex;
        var budget: vk.VkPhysicalDeviceMemoryBudgetPropertiesEXT = .{};
        var properties: vk.VkPhysicalDeviceMemoryProperties2 = .{ .pNext = &budget };
        vk.vkGetPhysicalDeviceMemoryProperties2(self.physical, &properties);
        return .{ .heap = heap, .size = self.memory.memoryHeaps[heap].size, .budget = budget.heapBudget[heap], .usage = budget.heapUsage[heap] };
    }

    /// The extension and the subgroupSizeControl feature are supported, and compute
    /// shaders can require a size; records the supported range. Returns whether
    /// `computeFullSubgroups` is supported.
    fn requireSubgroupSizeControl(self: *Device) Error!bool {
        if (!try self.hasExtension("VK_EXT_subgroup_size_control")) return error.UnsupportedFeature;
        var features: vk.VkPhysicalDeviceSubgroupSizeControlFeatures = .{};
        var features2: vk.VkPhysicalDeviceFeatures2 = .{ .pNext = &features };
        vk.vkGetPhysicalDeviceFeatures2(self.physical, &features2);
        if (features.subgroupSizeControl != vk.VK_TRUE) return error.UnsupportedFeature;
        var properties: vk.VkPhysicalDeviceSubgroupSizeControlProperties = .{};
        var properties2: vk.VkPhysicalDeviceProperties2 = .{ .pNext = &properties };
        vk.vkGetPhysicalDeviceProperties2(self.physical, &properties2);
        if (properties.requiredSubgroupSizeStages & vk.VK_SHADER_STAGE_COMPUTE_BIT == 0) return error.UnsupportedFeature;
        if (properties.minSubgroupSize == 0 or !std.math.isPowerOfTwo(properties.minSubgroupSize) or properties.maxSubgroupSize < properties.minSubgroupSize or properties.maxSubgroupSize > 128)
            return error.InvalidDriverProperties;
        self.subgroup_sizes = .{ .min = properties.minSubgroupSize, .max = properties.maxSubgroupSize };
        return features.computeFullSubgroups == vk.VK_TRUE;
    }

    /// The extensions are listed and the cooperative-matrix, shaderFloat16,
    /// storageBuffer16BitAccess and vulkanMemoryModel features are supported.
    fn requireCooperativeMatrix(self: *Device) Error!void {
        for (cooperative_matrix_extensions) |wanted| {
            if (!try self.hasExtension(wanted)) return error.UnsupportedFeature;
        }
        var memory_model: vk.VkPhysicalDeviceVulkanMemoryModelFeatures = .{};
        var storage16: vk.VkPhysicalDevice16BitStorageFeatures = .{ .pNext = &memory_model };
        var float16: vk.VkPhysicalDeviceShaderFloat16Int8Features = .{ .pNext = &storage16 };
        var coopmat: vk.VkPhysicalDeviceCooperativeMatrixFeaturesKHR = .{ .pNext = &float16 };
        var features: vk.VkPhysicalDeviceFeatures2 = .{ .pNext = &coopmat };
        vk.vkGetPhysicalDeviceFeatures2(self.physical, &features);
        if (coopmat.cooperativeMatrix != vk.VK_TRUE or float16.shaderFloat16 != vk.VK_TRUE or storage16.storageBuffer16BitAccess != vk.VK_TRUE or memory_model.vulkanMemoryModel != vk.VK_TRUE)
            return error.UnsupportedFeature;
    }

    /// The storageBuffer16BitAccess feature (Vulkan 1.1 core) is supported.
    fn requireStorage16(self: *Device) Error!void {
        var storage16: vk.VkPhysicalDevice16BitStorageFeatures = .{};
        var features: vk.VkPhysicalDeviceFeatures2 = .{ .pNext = &storage16 };
        vk.vkGetPhysicalDeviceFeatures2(self.physical, &features);
        if (storage16.storageBuffer16BitAccess != vk.VK_TRUE) return error.UnsupportedFeature;
    }

    pub fn name(self: *const Device) []const u8 {
        return std.mem.sliceTo(&self.properties.deviceName, 0);
    }

    pub fn deinit(self: *Device) Error!void {
        if (self.handle == null) return error.InvalidState;
        if (self.buffers != 0 or self.kernels != 0 or self.commands != 0 or self.pending != 0) return error.ResourceInUse;
        vk.vkDestroyDevice(self.handle, null);
        vk.vkDestroyInstance(self.instance, null);
        self.handle = null;
        self.instance = null;
    }

    pub fn ready(self: *const Device) Error!void {
        if (self.handle == null) return error.InvalidState;
        if (self.lost) return error.DeviceLost;
    }

    pub fn check(self: *Device, result: vk.VkResult) Error!void {
        if (result == vk.VK_SUCCESS) return;
        self.last_result = result;
        if (result == vk.VK_ERROR_DEVICE_LOST) self.lost = true;
        return switch (result) {
            vk.VK_ERROR_OUT_OF_HOST_MEMORY => error.HostOutOfMemory,
            vk.VK_ERROR_OUT_OF_DEVICE_MEMORY => error.DeviceOutOfMemory,
            vk.VK_ERROR_DEVICE_LOST => error.DeviceLost,
            vk.VK_TIMEOUT => error.Timeout,
            else => error.DriverError,
        };
    }
};

pub fn memoryType(properties: *const vk.VkPhysicalDeviceMemoryProperties, allowed: u32, location: Location) Error!u32 {
    if (properties.memoryTypeCount > 32 or properties.memoryHeapCount > 16) return error.InvalidDriverProperties;
    const required: u32 = if (location == .host) vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT else vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    var selected: ?u32 = null;
    for (properties.memoryTypes[0..properties.memoryTypeCount], 0..) |item, i| {
        if (item.heapIndex >= properties.memoryHeapCount) return error.InvalidDriverProperties;
        const base_flags: u32 = vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | vk.VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        // No protected/lazy/extension memory features are enabled on this device.
        if (item.propertyFlags & ~base_flags != 0) continue;
        if (allowed & (@as(u32, 1) << @intCast(i)) == 0 or item.propertyFlags & required != required) continue;
        if (selected == null or (location == .host and item.propertyFlags & vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT == 0)) selected = @intCast(i);
    }
    return selected orelse error.UnsupportedMemoryType;
}
