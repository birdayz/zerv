//! Raw system API declarations, generated from Khronos Vulkan-Headers
//! 01393c3df0e5285b54ee6527466513f9e614be94; see tools/generate_vulkan_bindings.py.
//! No inference code, C import, or third_party build dependency.
const std = @import("std");

pub const VK_ACCESS_HOST_READ_BIT = 8192;
pub const VK_ACCESS_HOST_WRITE_BIT = 16384;
pub const VK_ACCESS_SHADER_READ_BIT = 32;
pub const VK_ACCESS_SHADER_WRITE_BIT = 64;
pub const VK_ACCESS_TRANSFER_READ_BIT = 2048;
pub const VK_ACCESS_TRANSFER_WRITE_BIT = 4096;
pub const VK_API_VERSION_1_1 = 4198400;
pub const VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 32;
pub const VK_BUFFER_USAGE_TRANSFER_DST_BIT = 2;
pub const VK_BUFFER_USAGE_TRANSFER_SRC_BIT = 1;
pub const VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0;
pub const VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7;
pub const VK_ERROR_DEVICE_LOST = -4;
pub const VK_ERROR_OUT_OF_DEVICE_MEMORY = -2;
pub const VK_ERROR_OUT_OF_HOST_MEMORY = -1;
pub const VK_INCOMPLETE = 5;
pub const VK_MAX_EXTENSION_NAME_SIZE = 256;
pub const VK_MAX_MEMORY_HEAPS = 16;
pub const VK_MAX_MEMORY_TYPES = 32;
pub const VK_MAX_PHYSICAL_DEVICE_NAME_SIZE = 256;
pub const VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 1;
pub const VK_MEMORY_PROPERTY_HOST_CACHED_BIT = 8;
pub const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 4;
pub const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 2;
pub const VK_NOT_READY = 1;
pub const VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU = 2;
pub const VK_PIPELINE_BIND_POINT_COMPUTE = 1;
pub const VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT = 2;
pub const VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT = 2048;
pub const VK_PIPELINE_STAGE_HOST_BIT = 16384;
pub const VK_PIPELINE_STAGE_TRANSFER_BIT = 4096;
pub const VK_QUEUE_COMPUTE_BIT = 2;
pub const VK_QUEUE_FAMILY_IGNORED = 4294967295;
pub const VK_QUEUE_GRAPHICS_BIT = 1;
pub const VK_SHADER_STAGE_COMPUTE_BIT = 32;
pub const VK_SHARING_MODE_EXCLUSIVE = 0;
pub const VK_STRUCTURE_TYPE_APPLICATION_INFO = 0;
pub const VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO = 12;
pub const VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER = 44;
pub const VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = 40;
pub const VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 42;
pub const VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO = 41;
pub const VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = 39;
pub const VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO = 29;
pub const VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET = 36;
pub const VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO = 33;
pub const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO = 34;
pub const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32;
pub const VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3;
pub const VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 2;
pub const VK_STRUCTURE_TYPE_FENCE_CREATE_INFO = 8;
pub const VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER = 45;
pub const VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1;
pub const VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5;
pub const VK_STRUCTURE_TYPE_MEMORY_BARRIER = 46;
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES = 1000083000;
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR = 1000506000;
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 = 1000059000;
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT = 1000237000;
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2 = 1000059006;
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 = 1000059001;
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES = 1000082000;
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES = 1000094000;
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES = 1000225002;
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES = 1000225000;
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES = 1000211000;
pub const VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO = 30;
pub const VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO = 18;
pub const VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO = 1000225001;
pub const VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO = 16;
pub const VK_STRUCTURE_TYPE_SUBMIT_INFO = 4;
pub const VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET = 35;
pub const VK_SUBGROUP_FEATURE_BALLOT_BIT = 8;
pub const VK_SUBGROUP_FEATURE_BASIC_BIT = 1;
pub const VK_SUCCESS = 0;
pub const VK_TIMEOUT = 2;
pub const VK_TRUE = 1;
pub const VK_UUID_SIZE = 16;
pub const VkAllocationCallbacks = opaque {};
pub const VkAccessFlags = VkFlags;
pub const VkApplicationInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
    pNext: ?*const anyopaque = null,
    pApplicationName: [*c]const u8 = null,
    applicationVersion: u32 = std.mem.zeroes(u32),
    pEngineName: [*c]const u8 = null,
    engineVersion: u32 = std.mem.zeroes(u32),
    apiVersion: u32 = std.mem.zeroes(u32),
};
pub const VkBool32 = u32;
pub const VkBuffer = ?*opaque {};
pub const VkBufferCopy = extern struct {
    srcOffset: VkDeviceSize = std.mem.zeroes(VkDeviceSize),
    dstOffset: VkDeviceSize = std.mem.zeroes(VkDeviceSize),
    size: VkDeviceSize = std.mem.zeroes(VkDeviceSize),
};
pub const VkBufferCreateFlags = VkFlags;
pub const VkBufferCreateInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkBufferCreateFlags = std.mem.zeroes(VkBufferCreateFlags),
    size: VkDeviceSize = std.mem.zeroes(VkDeviceSize),
    usage: VkBufferUsageFlags = std.mem.zeroes(VkBufferUsageFlags),
    sharingMode: VkSharingMode = std.mem.zeroes(VkSharingMode),
    queueFamilyIndexCount: u32 = std.mem.zeroes(u32),
    pQueueFamilyIndices: [*c]const u32 = null,
};
pub const VkBufferMemoryBarrier = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
    pNext: ?*const anyopaque = null,
    srcAccessMask: VkAccessFlags = std.mem.zeroes(VkAccessFlags),
    dstAccessMask: VkAccessFlags = std.mem.zeroes(VkAccessFlags),
    srcQueueFamilyIndex: u32 = std.mem.zeroes(u32),
    dstQueueFamilyIndex: u32 = std.mem.zeroes(u32),
    buffer: VkBuffer = null,
    offset: VkDeviceSize = std.mem.zeroes(VkDeviceSize),
    size: VkDeviceSize = std.mem.zeroes(VkDeviceSize),
};
pub const VkBufferUsageFlags = VkFlags;
pub const VkBufferView = ?*opaque {};
pub const VkCommandBuffer = ?*opaque {};
pub const VkCommandBufferAllocateInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    pNext: ?*const anyopaque = null,
    commandPool: VkCommandPool = null,
    level: VkCommandBufferLevel = std.mem.zeroes(VkCommandBufferLevel),
    commandBufferCount: u32 = std.mem.zeroes(u32),
};
pub const VkCommandBufferBeginInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkCommandBufferUsageFlags = std.mem.zeroes(VkCommandBufferUsageFlags),
    pInheritanceInfo: [*c]const VkCommandBufferInheritanceInfo = null,
};
pub const VkCommandBufferInheritanceInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO,
    pNext: ?*const anyopaque = null,
    renderPass: VkRenderPass = null,
    subpass: u32 = std.mem.zeroes(u32),
    framebuffer: VkFramebuffer = null,
    occlusionQueryEnable: VkBool32 = std.mem.zeroes(VkBool32),
    queryFlags: VkQueryControlFlags = std.mem.zeroes(VkQueryControlFlags),
    pipelineStatistics: VkQueryPipelineStatisticFlags = std.mem.zeroes(VkQueryPipelineStatisticFlags),
};
pub const VkCommandBufferLevel = i32;
pub const VkCommandBufferUsageFlags = VkFlags;
pub const VkCommandPool = ?*opaque {};
pub const VkCommandPoolCreateFlags = VkFlags;
pub const VkCommandPoolCreateInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkCommandPoolCreateFlags = std.mem.zeroes(VkCommandPoolCreateFlags),
    queueFamilyIndex: u32 = std.mem.zeroes(u32),
};
pub const VkCommandPoolResetFlags = VkFlags;
pub const VkComputePipelineCreateInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkPipelineCreateFlags = std.mem.zeroes(VkPipelineCreateFlags),
    stage: VkPipelineShaderStageCreateInfo = std.mem.zeroes(VkPipelineShaderStageCreateInfo),
    layout: VkPipelineLayout = null,
    basePipelineHandle: VkPipeline = null,
    basePipelineIndex: i32 = std.mem.zeroes(i32),
};
pub const VkCopyDescriptorSet = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET,
    pNext: ?*const anyopaque = null,
    srcSet: VkDescriptorSet = null,
    srcBinding: u32 = std.mem.zeroes(u32),
    srcArrayElement: u32 = std.mem.zeroes(u32),
    dstSet: VkDescriptorSet = null,
    dstBinding: u32 = std.mem.zeroes(u32),
    dstArrayElement: u32 = std.mem.zeroes(u32),
    descriptorCount: u32 = std.mem.zeroes(u32),
};
pub const VkDependencyFlags = VkFlags;
pub const VkDescriptorBufferInfo = extern struct {
    buffer: VkBuffer = null,
    offset: VkDeviceSize = std.mem.zeroes(VkDeviceSize),
    range: VkDeviceSize = std.mem.zeroes(VkDeviceSize),
};
pub const VkDescriptorImageInfo = extern struct {
    sampler: VkSampler = null,
    imageView: VkImageView = null,
    imageLayout: VkImageLayout = std.mem.zeroes(VkImageLayout),
};
pub const VkDescriptorPool = ?*opaque {};
pub const VkDescriptorPoolCreateFlags = VkFlags;
pub const VkDescriptorPoolCreateInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkDescriptorPoolCreateFlags = std.mem.zeroes(VkDescriptorPoolCreateFlags),
    maxSets: u32 = std.mem.zeroes(u32),
    poolSizeCount: u32 = std.mem.zeroes(u32),
    pPoolSizes: [*c]const VkDescriptorPoolSize = null,
};
pub const VkDescriptorPoolSize = extern struct {
    type: VkDescriptorType = std.mem.zeroes(VkDescriptorType),
    descriptorCount: u32 = std.mem.zeroes(u32),
};
pub const VkDescriptorSet = ?*opaque {};
pub const VkDescriptorSetAllocateInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    pNext: ?*const anyopaque = null,
    descriptorPool: VkDescriptorPool = null,
    descriptorSetCount: u32 = std.mem.zeroes(u32),
    pSetLayouts: [*c]const VkDescriptorSetLayout = null,
};
pub const VkDescriptorSetLayout = ?*opaque {};
pub const VkDescriptorSetLayoutBinding = extern struct {
    binding: u32 = std.mem.zeroes(u32),
    descriptorType: VkDescriptorType = std.mem.zeroes(VkDescriptorType),
    descriptorCount: u32 = std.mem.zeroes(u32),
    stageFlags: VkShaderStageFlags = std.mem.zeroes(VkShaderStageFlags),
    pImmutableSamplers: [*c]const VkSampler = null,
};
pub const VkDescriptorSetLayoutCreateFlags = VkFlags;
pub const VkDescriptorSetLayoutCreateInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkDescriptorSetLayoutCreateFlags = std.mem.zeroes(VkDescriptorSetLayoutCreateFlags),
    bindingCount: u32 = std.mem.zeroes(u32),
    pBindings: [*c]const VkDescriptorSetLayoutBinding = null,
};
pub const VkDescriptorType = i32;
pub const VkDevice = ?*opaque {};
pub const VkDeviceCreateFlags = VkFlags;
pub const VkDeviceCreateInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkDeviceCreateFlags = std.mem.zeroes(VkDeviceCreateFlags),
    queueCreateInfoCount: u32 = std.mem.zeroes(u32),
    pQueueCreateInfos: [*c]const VkDeviceQueueCreateInfo = null,
    enabledLayerCount: u32 = std.mem.zeroes(u32),
    ppEnabledLayerNames: [*c]const [*c]const u8 = null,
    enabledExtensionCount: u32 = std.mem.zeroes(u32),
    ppEnabledExtensionNames: [*c]const [*c]const u8 = null,
    pEnabledFeatures: [*c]const VkPhysicalDeviceFeatures = null,
};
pub const VkDeviceMemory = ?*opaque {};
pub const VkDeviceQueueCreateFlags = VkFlags;
pub const VkDeviceQueueCreateInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkDeviceQueueCreateFlags = std.mem.zeroes(VkDeviceQueueCreateFlags),
    queueFamilyIndex: u32 = std.mem.zeroes(u32),
    queueCount: u32 = std.mem.zeroes(u32),
    pQueuePriorities: [*c]const f32 = null,
};
pub const VkDeviceSize = u64;
pub const VkExtensionProperties = extern struct {
    extensionName: [VK_MAX_EXTENSION_NAME_SIZE]u8 = std.mem.zeroes([VK_MAX_EXTENSION_NAME_SIZE]u8),
    specVersion: u32 = std.mem.zeroes(u32),
};
pub const VkExtent3D = extern struct {
    width: u32 = std.mem.zeroes(u32),
    height: u32 = std.mem.zeroes(u32),
    depth: u32 = std.mem.zeroes(u32),
};
pub const VkFence = ?*opaque {};
pub const VkFenceCreateFlags = VkFlags;
pub const VkFenceCreateInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkFenceCreateFlags = std.mem.zeroes(VkFenceCreateFlags),
};
pub const VkFlags = u32;
pub const VkFramebuffer = ?*opaque {};
pub const VkImage = ?*opaque {};
pub const VkImageAspectFlags = VkFlags;
pub const VkImageLayout = i32;
pub const VkImageMemoryBarrier = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
    pNext: ?*const anyopaque = null,
    srcAccessMask: VkAccessFlags = std.mem.zeroes(VkAccessFlags),
    dstAccessMask: VkAccessFlags = std.mem.zeroes(VkAccessFlags),
    oldLayout: VkImageLayout = std.mem.zeroes(VkImageLayout),
    newLayout: VkImageLayout = std.mem.zeroes(VkImageLayout),
    srcQueueFamilyIndex: u32 = std.mem.zeroes(u32),
    dstQueueFamilyIndex: u32 = std.mem.zeroes(u32),
    image: VkImage = null,
    subresourceRange: VkImageSubresourceRange = std.mem.zeroes(VkImageSubresourceRange),
};
pub const VkImageSubresourceRange = extern struct {
    aspectMask: VkImageAspectFlags = std.mem.zeroes(VkImageAspectFlags),
    baseMipLevel: u32 = std.mem.zeroes(u32),
    levelCount: u32 = std.mem.zeroes(u32),
    baseArrayLayer: u32 = std.mem.zeroes(u32),
    layerCount: u32 = std.mem.zeroes(u32),
};
pub const VkImageView = ?*opaque {};
pub const VkInstance = ?*opaque {};
pub const VkInstanceCreateFlags = VkFlags;
pub const VkInstanceCreateInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkInstanceCreateFlags = std.mem.zeroes(VkInstanceCreateFlags),
    pApplicationInfo: [*c]const VkApplicationInfo = null,
    enabledLayerCount: u32 = std.mem.zeroes(u32),
    ppEnabledLayerNames: [*c]const [*c]const u8 = null,
    enabledExtensionCount: u32 = std.mem.zeroes(u32),
    ppEnabledExtensionNames: [*c]const [*c]const u8 = null,
};
pub const VkMemoryAllocateInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    pNext: ?*const anyopaque = null,
    allocationSize: VkDeviceSize = std.mem.zeroes(VkDeviceSize),
    memoryTypeIndex: u32 = std.mem.zeroes(u32),
};
pub const VkMemoryBarrier = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
    pNext: ?*const anyopaque = null,
    srcAccessMask: VkAccessFlags = std.mem.zeroes(VkAccessFlags),
    dstAccessMask: VkAccessFlags = std.mem.zeroes(VkAccessFlags),
};
pub const VkMemoryHeap = extern struct {
    size: VkDeviceSize = std.mem.zeroes(VkDeviceSize),
    flags: VkMemoryHeapFlags = std.mem.zeroes(VkMemoryHeapFlags),
};
pub const VkMemoryHeapFlags = VkFlags;
pub const VkMemoryMapFlags = VkFlags;
pub const VkMemoryPropertyFlags = VkFlags;
pub const VkMemoryRequirements = extern struct {
    size: VkDeviceSize = std.mem.zeroes(VkDeviceSize),
    alignment: VkDeviceSize = std.mem.zeroes(VkDeviceSize),
    memoryTypeBits: u32 = std.mem.zeroes(u32),
};
pub const VkMemoryType = extern struct {
    propertyFlags: VkMemoryPropertyFlags = std.mem.zeroes(VkMemoryPropertyFlags),
    heapIndex: u32 = std.mem.zeroes(u32),
};
pub const VkPhysicalDevice = ?*opaque {};
pub const VkPhysicalDevice16BitStorageFeatures = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
    pNext: ?*anyopaque = null,
    storageBuffer16BitAccess: VkBool32 = std.mem.zeroes(VkBool32),
    uniformAndStorageBuffer16BitAccess: VkBool32 = std.mem.zeroes(VkBool32),
    storagePushConstant16: VkBool32 = std.mem.zeroes(VkBool32),
    storageInputOutput16: VkBool32 = std.mem.zeroes(VkBool32),
};
pub const VkPhysicalDeviceCooperativeMatrixFeaturesKHR = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
    pNext: ?*anyopaque = null,
    cooperativeMatrix: VkBool32 = std.mem.zeroes(VkBool32),
    cooperativeMatrixRobustBufferAccess: VkBool32 = std.mem.zeroes(VkBool32),
};
pub const VkPhysicalDeviceFeatures = extern struct {
    robustBufferAccess: VkBool32 = std.mem.zeroes(VkBool32),
    fullDrawIndexUint32: VkBool32 = std.mem.zeroes(VkBool32),
    imageCubeArray: VkBool32 = std.mem.zeroes(VkBool32),
    independentBlend: VkBool32 = std.mem.zeroes(VkBool32),
    geometryShader: VkBool32 = std.mem.zeroes(VkBool32),
    tessellationShader: VkBool32 = std.mem.zeroes(VkBool32),
    sampleRateShading: VkBool32 = std.mem.zeroes(VkBool32),
    dualSrcBlend: VkBool32 = std.mem.zeroes(VkBool32),
    logicOp: VkBool32 = std.mem.zeroes(VkBool32),
    multiDrawIndirect: VkBool32 = std.mem.zeroes(VkBool32),
    drawIndirectFirstInstance: VkBool32 = std.mem.zeroes(VkBool32),
    depthClamp: VkBool32 = std.mem.zeroes(VkBool32),
    depthBiasClamp: VkBool32 = std.mem.zeroes(VkBool32),
    fillModeNonSolid: VkBool32 = std.mem.zeroes(VkBool32),
    depthBounds: VkBool32 = std.mem.zeroes(VkBool32),
    wideLines: VkBool32 = std.mem.zeroes(VkBool32),
    largePoints: VkBool32 = std.mem.zeroes(VkBool32),
    alphaToOne: VkBool32 = std.mem.zeroes(VkBool32),
    multiViewport: VkBool32 = std.mem.zeroes(VkBool32),
    samplerAnisotropy: VkBool32 = std.mem.zeroes(VkBool32),
    textureCompressionETC2: VkBool32 = std.mem.zeroes(VkBool32),
    textureCompressionASTC_LDR: VkBool32 = std.mem.zeroes(VkBool32),
    textureCompressionBC: VkBool32 = std.mem.zeroes(VkBool32),
    occlusionQueryPrecise: VkBool32 = std.mem.zeroes(VkBool32),
    pipelineStatisticsQuery: VkBool32 = std.mem.zeroes(VkBool32),
    vertexPipelineStoresAndAtomics: VkBool32 = std.mem.zeroes(VkBool32),
    fragmentStoresAndAtomics: VkBool32 = std.mem.zeroes(VkBool32),
    shaderTessellationAndGeometryPointSize: VkBool32 = std.mem.zeroes(VkBool32),
    shaderImageGatherExtended: VkBool32 = std.mem.zeroes(VkBool32),
    shaderStorageImageExtendedFormats: VkBool32 = std.mem.zeroes(VkBool32),
    shaderStorageImageMultisample: VkBool32 = std.mem.zeroes(VkBool32),
    shaderStorageImageReadWithoutFormat: VkBool32 = std.mem.zeroes(VkBool32),
    shaderStorageImageWriteWithoutFormat: VkBool32 = std.mem.zeroes(VkBool32),
    shaderUniformBufferArrayDynamicIndexing: VkBool32 = std.mem.zeroes(VkBool32),
    shaderSampledImageArrayDynamicIndexing: VkBool32 = std.mem.zeroes(VkBool32),
    shaderStorageBufferArrayDynamicIndexing: VkBool32 = std.mem.zeroes(VkBool32),
    shaderStorageImageArrayDynamicIndexing: VkBool32 = std.mem.zeroes(VkBool32),
    shaderClipDistance: VkBool32 = std.mem.zeroes(VkBool32),
    shaderCullDistance: VkBool32 = std.mem.zeroes(VkBool32),
    shaderFloat64: VkBool32 = std.mem.zeroes(VkBool32),
    shaderInt64: VkBool32 = std.mem.zeroes(VkBool32),
    shaderInt16: VkBool32 = std.mem.zeroes(VkBool32),
    shaderResourceResidency: VkBool32 = std.mem.zeroes(VkBool32),
    shaderResourceMinLod: VkBool32 = std.mem.zeroes(VkBool32),
    sparseBinding: VkBool32 = std.mem.zeroes(VkBool32),
    sparseResidencyBuffer: VkBool32 = std.mem.zeroes(VkBool32),
    sparseResidencyImage2D: VkBool32 = std.mem.zeroes(VkBool32),
    sparseResidencyImage3D: VkBool32 = std.mem.zeroes(VkBool32),
    sparseResidency2Samples: VkBool32 = std.mem.zeroes(VkBool32),
    sparseResidency4Samples: VkBool32 = std.mem.zeroes(VkBool32),
    sparseResidency8Samples: VkBool32 = std.mem.zeroes(VkBool32),
    sparseResidency16Samples: VkBool32 = std.mem.zeroes(VkBool32),
    sparseResidencyAliased: VkBool32 = std.mem.zeroes(VkBool32),
    variableMultisampleRate: VkBool32 = std.mem.zeroes(VkBool32),
    inheritedQueries: VkBool32 = std.mem.zeroes(VkBool32),
};
pub const VkPhysicalDeviceFeatures2 = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
    pNext: ?*anyopaque = null,
    features: VkPhysicalDeviceFeatures = std.mem.zeroes(VkPhysicalDeviceFeatures),
};
pub const VkPhysicalDeviceLimits = extern struct {
    maxImageDimension1D: u32 = std.mem.zeroes(u32),
    maxImageDimension2D: u32 = std.mem.zeroes(u32),
    maxImageDimension3D: u32 = std.mem.zeroes(u32),
    maxImageDimensionCube: u32 = std.mem.zeroes(u32),
    maxImageArrayLayers: u32 = std.mem.zeroes(u32),
    maxTexelBufferElements: u32 = std.mem.zeroes(u32),
    maxUniformBufferRange: u32 = std.mem.zeroes(u32),
    maxStorageBufferRange: u32 = std.mem.zeroes(u32),
    maxPushConstantsSize: u32 = std.mem.zeroes(u32),
    maxMemoryAllocationCount: u32 = std.mem.zeroes(u32),
    maxSamplerAllocationCount: u32 = std.mem.zeroes(u32),
    bufferImageGranularity: VkDeviceSize = std.mem.zeroes(VkDeviceSize),
    sparseAddressSpaceSize: VkDeviceSize = std.mem.zeroes(VkDeviceSize),
    maxBoundDescriptorSets: u32 = std.mem.zeroes(u32),
    maxPerStageDescriptorSamplers: u32 = std.mem.zeroes(u32),
    maxPerStageDescriptorUniformBuffers: u32 = std.mem.zeroes(u32),
    maxPerStageDescriptorStorageBuffers: u32 = std.mem.zeroes(u32),
    maxPerStageDescriptorSampledImages: u32 = std.mem.zeroes(u32),
    maxPerStageDescriptorStorageImages: u32 = std.mem.zeroes(u32),
    maxPerStageDescriptorInputAttachments: u32 = std.mem.zeroes(u32),
    maxPerStageResources: u32 = std.mem.zeroes(u32),
    maxDescriptorSetSamplers: u32 = std.mem.zeroes(u32),
    maxDescriptorSetUniformBuffers: u32 = std.mem.zeroes(u32),
    maxDescriptorSetUniformBuffersDynamic: u32 = std.mem.zeroes(u32),
    maxDescriptorSetStorageBuffers: u32 = std.mem.zeroes(u32),
    maxDescriptorSetStorageBuffersDynamic: u32 = std.mem.zeroes(u32),
    maxDescriptorSetSampledImages: u32 = std.mem.zeroes(u32),
    maxDescriptorSetStorageImages: u32 = std.mem.zeroes(u32),
    maxDescriptorSetInputAttachments: u32 = std.mem.zeroes(u32),
    maxVertexInputAttributes: u32 = std.mem.zeroes(u32),
    maxVertexInputBindings: u32 = std.mem.zeroes(u32),
    maxVertexInputAttributeOffset: u32 = std.mem.zeroes(u32),
    maxVertexInputBindingStride: u32 = std.mem.zeroes(u32),
    maxVertexOutputComponents: u32 = std.mem.zeroes(u32),
    maxTessellationGenerationLevel: u32 = std.mem.zeroes(u32),
    maxTessellationPatchSize: u32 = std.mem.zeroes(u32),
    maxTessellationControlPerVertexInputComponents: u32 = std.mem.zeroes(u32),
    maxTessellationControlPerVertexOutputComponents: u32 = std.mem.zeroes(u32),
    maxTessellationControlPerPatchOutputComponents: u32 = std.mem.zeroes(u32),
    maxTessellationControlTotalOutputComponents: u32 = std.mem.zeroes(u32),
    maxTessellationEvaluationInputComponents: u32 = std.mem.zeroes(u32),
    maxTessellationEvaluationOutputComponents: u32 = std.mem.zeroes(u32),
    maxGeometryShaderInvocations: u32 = std.mem.zeroes(u32),
    maxGeometryInputComponents: u32 = std.mem.zeroes(u32),
    maxGeometryOutputComponents: u32 = std.mem.zeroes(u32),
    maxGeometryOutputVertices: u32 = std.mem.zeroes(u32),
    maxGeometryTotalOutputComponents: u32 = std.mem.zeroes(u32),
    maxFragmentInputComponents: u32 = std.mem.zeroes(u32),
    maxFragmentOutputAttachments: u32 = std.mem.zeroes(u32),
    maxFragmentDualSrcAttachments: u32 = std.mem.zeroes(u32),
    maxFragmentCombinedOutputResources: u32 = std.mem.zeroes(u32),
    maxComputeSharedMemorySize: u32 = std.mem.zeroes(u32),
    maxComputeWorkGroupCount: [3]u32 = std.mem.zeroes([3]u32),
    maxComputeWorkGroupInvocations: u32 = std.mem.zeroes(u32),
    maxComputeWorkGroupSize: [3]u32 = std.mem.zeroes([3]u32),
    subPixelPrecisionBits: u32 = std.mem.zeroes(u32),
    subTexelPrecisionBits: u32 = std.mem.zeroes(u32),
    mipmapPrecisionBits: u32 = std.mem.zeroes(u32),
    maxDrawIndexedIndexValue: u32 = std.mem.zeroes(u32),
    maxDrawIndirectCount: u32 = std.mem.zeroes(u32),
    maxSamplerLodBias: f32 = std.mem.zeroes(f32),
    maxSamplerAnisotropy: f32 = std.mem.zeroes(f32),
    maxViewports: u32 = std.mem.zeroes(u32),
    maxViewportDimensions: [2]u32 = std.mem.zeroes([2]u32),
    viewportBoundsRange: [2]f32 = std.mem.zeroes([2]f32),
    viewportSubPixelBits: u32 = std.mem.zeroes(u32),
    minMemoryMapAlignment: usize = std.mem.zeroes(usize),
    minTexelBufferOffsetAlignment: VkDeviceSize = std.mem.zeroes(VkDeviceSize),
    minUniformBufferOffsetAlignment: VkDeviceSize = std.mem.zeroes(VkDeviceSize),
    minStorageBufferOffsetAlignment: VkDeviceSize = std.mem.zeroes(VkDeviceSize),
    minTexelOffset: i32 = std.mem.zeroes(i32),
    maxTexelOffset: u32 = std.mem.zeroes(u32),
    minTexelGatherOffset: i32 = std.mem.zeroes(i32),
    maxTexelGatherOffset: u32 = std.mem.zeroes(u32),
    minInterpolationOffset: f32 = std.mem.zeroes(f32),
    maxInterpolationOffset: f32 = std.mem.zeroes(f32),
    subPixelInterpolationOffsetBits: u32 = std.mem.zeroes(u32),
    maxFramebufferWidth: u32 = std.mem.zeroes(u32),
    maxFramebufferHeight: u32 = std.mem.zeroes(u32),
    maxFramebufferLayers: u32 = std.mem.zeroes(u32),
    framebufferColorSampleCounts: VkSampleCountFlags = std.mem.zeroes(VkSampleCountFlags),
    framebufferDepthSampleCounts: VkSampleCountFlags = std.mem.zeroes(VkSampleCountFlags),
    framebufferStencilSampleCounts: VkSampleCountFlags = std.mem.zeroes(VkSampleCountFlags),
    framebufferNoAttachmentsSampleCounts: VkSampleCountFlags = std.mem.zeroes(VkSampleCountFlags),
    maxColorAttachments: u32 = std.mem.zeroes(u32),
    sampledImageColorSampleCounts: VkSampleCountFlags = std.mem.zeroes(VkSampleCountFlags),
    sampledImageIntegerSampleCounts: VkSampleCountFlags = std.mem.zeroes(VkSampleCountFlags),
    sampledImageDepthSampleCounts: VkSampleCountFlags = std.mem.zeroes(VkSampleCountFlags),
    sampledImageStencilSampleCounts: VkSampleCountFlags = std.mem.zeroes(VkSampleCountFlags),
    storageImageSampleCounts: VkSampleCountFlags = std.mem.zeroes(VkSampleCountFlags),
    maxSampleMaskWords: u32 = std.mem.zeroes(u32),
    timestampComputeAndGraphics: VkBool32 = std.mem.zeroes(VkBool32),
    timestampPeriod: f32 = std.mem.zeroes(f32),
    maxClipDistances: u32 = std.mem.zeroes(u32),
    maxCullDistances: u32 = std.mem.zeroes(u32),
    maxCombinedClipAndCullDistances: u32 = std.mem.zeroes(u32),
    discreteQueuePriorities: u32 = std.mem.zeroes(u32),
    pointSizeRange: [2]f32 = std.mem.zeroes([2]f32),
    lineWidthRange: [2]f32 = std.mem.zeroes([2]f32),
    pointSizeGranularity: f32 = std.mem.zeroes(f32),
    lineWidthGranularity: f32 = std.mem.zeroes(f32),
    strictLines: VkBool32 = std.mem.zeroes(VkBool32),
    standardSampleLocations: VkBool32 = std.mem.zeroes(VkBool32),
    optimalBufferCopyOffsetAlignment: VkDeviceSize = std.mem.zeroes(VkDeviceSize),
    optimalBufferCopyRowPitchAlignment: VkDeviceSize = std.mem.zeroes(VkDeviceSize),
    nonCoherentAtomSize: VkDeviceSize = std.mem.zeroes(VkDeviceSize),
};
pub const VkPhysicalDeviceMemoryBudgetPropertiesEXT = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT,
    pNext: ?*anyopaque = null,
    heapBudget: [VK_MAX_MEMORY_HEAPS]VkDeviceSize = std.mem.zeroes([VK_MAX_MEMORY_HEAPS]VkDeviceSize),
    heapUsage: [VK_MAX_MEMORY_HEAPS]VkDeviceSize = std.mem.zeroes([VK_MAX_MEMORY_HEAPS]VkDeviceSize),
};
pub const VkPhysicalDeviceMemoryProperties = extern struct {
    memoryTypeCount: u32 = std.mem.zeroes(u32),
    memoryTypes: [VK_MAX_MEMORY_TYPES]VkMemoryType = std.mem.zeroes([VK_MAX_MEMORY_TYPES]VkMemoryType),
    memoryHeapCount: u32 = std.mem.zeroes(u32),
    memoryHeaps: [VK_MAX_MEMORY_HEAPS]VkMemoryHeap = std.mem.zeroes([VK_MAX_MEMORY_HEAPS]VkMemoryHeap),
};
pub const VkPhysicalDeviceMemoryProperties2 = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,
    pNext: ?*anyopaque = null,
    memoryProperties: VkPhysicalDeviceMemoryProperties = std.mem.zeroes(VkPhysicalDeviceMemoryProperties),
};
pub const VkPhysicalDeviceProperties = extern struct {
    apiVersion: u32 = std.mem.zeroes(u32),
    driverVersion: u32 = std.mem.zeroes(u32),
    vendorID: u32 = std.mem.zeroes(u32),
    deviceID: u32 = std.mem.zeroes(u32),
    deviceType: VkPhysicalDeviceType = std.mem.zeroes(VkPhysicalDeviceType),
    deviceName: [VK_MAX_PHYSICAL_DEVICE_NAME_SIZE]u8 = std.mem.zeroes([VK_MAX_PHYSICAL_DEVICE_NAME_SIZE]u8),
    pipelineCacheUUID: [VK_UUID_SIZE]u8 = std.mem.zeroes([VK_UUID_SIZE]u8),
    limits: VkPhysicalDeviceLimits = std.mem.zeroes(VkPhysicalDeviceLimits),
    sparseProperties: VkPhysicalDeviceSparseProperties = std.mem.zeroes(VkPhysicalDeviceSparseProperties),
};
pub const VkPhysicalDeviceProperties2 = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
    pNext: ?*anyopaque = null,
    properties: VkPhysicalDeviceProperties = std.mem.zeroes(VkPhysicalDeviceProperties),
};
pub const VkPhysicalDeviceShaderFloat16Int8Features = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
    pNext: ?*anyopaque = null,
    shaderFloat16: VkBool32 = std.mem.zeroes(VkBool32),
    shaderInt8: VkBool32 = std.mem.zeroes(VkBool32),
};
pub const VkPhysicalDeviceSparseProperties = extern struct {
    residencyStandard2DBlockShape: VkBool32 = std.mem.zeroes(VkBool32),
    residencyStandard2DMultisampleBlockShape: VkBool32 = std.mem.zeroes(VkBool32),
    residencyStandard3DBlockShape: VkBool32 = std.mem.zeroes(VkBool32),
    residencyAlignedMipSize: VkBool32 = std.mem.zeroes(VkBool32),
    residencyNonResidentStrict: VkBool32 = std.mem.zeroes(VkBool32),
};
pub const VkPhysicalDeviceSubgroupProperties = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
    pNext: ?*anyopaque = null,
    subgroupSize: u32 = std.mem.zeroes(u32),
    supportedStages: VkShaderStageFlags = std.mem.zeroes(VkShaderStageFlags),
    supportedOperations: VkSubgroupFeatureFlags = std.mem.zeroes(VkSubgroupFeatureFlags),
    quadOperationsInAllStages: VkBool32 = std.mem.zeroes(VkBool32),
};
pub const VkPhysicalDeviceSubgroupSizeControlFeatures = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES,
    pNext: ?*anyopaque = null,
    subgroupSizeControl: VkBool32 = std.mem.zeroes(VkBool32),
    computeFullSubgroups: VkBool32 = std.mem.zeroes(VkBool32),
};
pub const VkPhysicalDeviceSubgroupSizeControlProperties = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES,
    pNext: ?*anyopaque = null,
    minSubgroupSize: u32 = std.mem.zeroes(u32),
    maxSubgroupSize: u32 = std.mem.zeroes(u32),
    maxComputeWorkgroupSubgroups: u32 = std.mem.zeroes(u32),
    requiredSubgroupSizeStages: VkShaderStageFlags = std.mem.zeroes(VkShaderStageFlags),
};
pub const VkPhysicalDeviceType = i32;
pub const VkPhysicalDeviceVulkanMemoryModelFeatures = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES,
    pNext: ?*anyopaque = null,
    vulkanMemoryModel: VkBool32 = std.mem.zeroes(VkBool32),
    vulkanMemoryModelDeviceScope: VkBool32 = std.mem.zeroes(VkBool32),
    vulkanMemoryModelAvailabilityVisibilityChains: VkBool32 = std.mem.zeroes(VkBool32),
};
pub const VkPipeline = ?*opaque {};
pub const VkPipelineBindPoint = i32;
pub const VkPipelineCache = ?*opaque {};
pub const VkPipelineCreateFlags = VkFlags;
pub const VkPipelineLayout = ?*opaque {};
pub const VkPipelineLayoutCreateFlags = VkFlags;
pub const VkPipelineLayoutCreateInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkPipelineLayoutCreateFlags = std.mem.zeroes(VkPipelineLayoutCreateFlags),
    setLayoutCount: u32 = std.mem.zeroes(u32),
    pSetLayouts: [*c]const VkDescriptorSetLayout = null,
    pushConstantRangeCount: u32 = std.mem.zeroes(u32),
    pPushConstantRanges: [*c]const VkPushConstantRange = null,
};
pub const VkPipelineShaderStageCreateFlags = VkFlags;
pub const VkPipelineShaderStageCreateInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkPipelineShaderStageCreateFlags = std.mem.zeroes(VkPipelineShaderStageCreateFlags),
    stage: VkShaderStageFlagBits = std.mem.zeroes(VkShaderStageFlagBits),
    module: VkShaderModule = null,
    pName: [*c]const u8 = null,
    pSpecializationInfo: [*c]const VkSpecializationInfo = null,
};
pub const VkPipelineShaderStageRequiredSubgroupSizeCreateInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    requiredSubgroupSize: u32 = std.mem.zeroes(u32),
};
pub const VkPipelineStageFlags = VkFlags;
pub const VkPushConstantRange = extern struct {
    stageFlags: VkShaderStageFlags = std.mem.zeroes(VkShaderStageFlags),
    offset: u32 = std.mem.zeroes(u32),
    size: u32 = std.mem.zeroes(u32),
};
pub const VkQueryControlFlags = VkFlags;
pub const VkQueryPipelineStatisticFlags = VkFlags;
pub const VkQueue = ?*opaque {};
pub const VkQueueFamilyProperties = extern struct {
    queueFlags: VkQueueFlags = std.mem.zeroes(VkQueueFlags),
    queueCount: u32 = std.mem.zeroes(u32),
    timestampValidBits: u32 = std.mem.zeroes(u32),
    minImageTransferGranularity: VkExtent3D = std.mem.zeroes(VkExtent3D),
};
pub const VkQueueFlags = VkFlags;
pub const VkRenderPass = ?*opaque {};
pub const VkResult = i32;
pub const VkSampleCountFlags = VkFlags;
pub const VkSampler = ?*opaque {};
pub const VkSemaphore = ?*opaque {};
pub const VkShaderModule = ?*opaque {};
pub const VkShaderModuleCreateFlags = VkFlags;
pub const VkShaderModuleCreateInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkShaderModuleCreateFlags = std.mem.zeroes(VkShaderModuleCreateFlags),
    codeSize: usize = std.mem.zeroes(usize),
    pCode: [*c]const u32 = null,
};
pub const VkShaderStageFlagBits = i32;
pub const VkShaderStageFlags = VkFlags;
pub const VkSharingMode = i32;
pub const VkSpecializationInfo = extern struct {
    mapEntryCount: u32 = std.mem.zeroes(u32),
    pMapEntries: [*c]const VkSpecializationMapEntry = null,
    dataSize: usize = std.mem.zeroes(usize),
    pData: ?*const anyopaque = null,
};
pub const VkSpecializationMapEntry = extern struct {
    constantID: u32 = std.mem.zeroes(u32),
    offset: u32 = std.mem.zeroes(u32),
    size: usize = std.mem.zeroes(usize),
};
pub const VkStructureType = i32;
pub const VkSubgroupFeatureFlags = VkFlags;
pub const VkSubmitInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
    pNext: ?*const anyopaque = null,
    waitSemaphoreCount: u32 = std.mem.zeroes(u32),
    pWaitSemaphores: [*c]const VkSemaphore = null,
    pWaitDstStageMask: [*c]const VkPipelineStageFlags = null,
    commandBufferCount: u32 = std.mem.zeroes(u32),
    pCommandBuffers: [*c]const VkCommandBuffer = null,
    signalSemaphoreCount: u32 = std.mem.zeroes(u32),
    pSignalSemaphores: [*c]const VkSemaphore = null,
};
pub const VkWriteDescriptorSet = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
    pNext: ?*const anyopaque = null,
    dstSet: VkDescriptorSet = null,
    dstBinding: u32 = std.mem.zeroes(u32),
    dstArrayElement: u32 = std.mem.zeroes(u32),
    descriptorCount: u32 = std.mem.zeroes(u32),
    descriptorType: VkDescriptorType = std.mem.zeroes(VkDescriptorType),
    pImageInfo: [*c]const VkDescriptorImageInfo = null,
    pBufferInfo: [*c]const VkDescriptorBufferInfo = null,
    pTexelBufferView: [*c]const VkBufferView = null,
};
pub extern fn vkCreateInstance(pCreateInfo: [*c]const VkInstanceCreateInfo, pAllocator: ?*const VkAllocationCallbacks, pInstance: [*c]VkInstance) callconv(.c) VkResult;
pub extern fn vkDestroyInstance(instance: VkInstance, pAllocator: ?*const VkAllocationCallbacks) callconv(.c) void;
pub extern fn vkEnumeratePhysicalDevices(instance: VkInstance, pPhysicalDeviceCount: [*c]u32, pPhysicalDevices: [*c]VkPhysicalDevice) callconv(.c) VkResult;
pub extern fn vkGetPhysicalDeviceProperties(physicalDevice: VkPhysicalDevice, pProperties: [*c]VkPhysicalDeviceProperties) callconv(.c) void;
pub extern fn vkGetPhysicalDeviceProperties2(physicalDevice: VkPhysicalDevice, pProperties: [*c]VkPhysicalDeviceProperties2) callconv(.c) void;
pub extern fn vkGetPhysicalDeviceMemoryProperties(physicalDevice: VkPhysicalDevice, pMemoryProperties: [*c]VkPhysicalDeviceMemoryProperties) callconv(.c) void;
pub extern fn vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice: VkPhysicalDevice, pQueueFamilyPropertyCount: [*c]u32, pQueueFamilyProperties: [*c]VkQueueFamilyProperties) callconv(.c) void;
pub extern fn vkCreateDevice(physicalDevice: VkPhysicalDevice, pCreateInfo: [*c]const VkDeviceCreateInfo, pAllocator: ?*const VkAllocationCallbacks, pDevice: [*c]VkDevice) callconv(.c) VkResult;
pub extern fn vkDestroyDevice(device: VkDevice, pAllocator: ?*const VkAllocationCallbacks) callconv(.c) void;
pub extern fn vkGetDeviceQueue(device: VkDevice, queueFamilyIndex: u32, queueIndex: u32, pQueue: [*c]VkQueue) callconv(.c) void;
pub extern fn vkDeviceWaitIdle(device: VkDevice) callconv(.c) VkResult;
pub extern fn vkCreateBuffer(device: VkDevice, pCreateInfo: [*c]const VkBufferCreateInfo, pAllocator: ?*const VkAllocationCallbacks, pBuffer: [*c]VkBuffer) callconv(.c) VkResult;
pub extern fn vkDestroyBuffer(device: VkDevice, buffer: VkBuffer, pAllocator: ?*const VkAllocationCallbacks) callconv(.c) void;
pub extern fn vkGetBufferMemoryRequirements(device: VkDevice, buffer: VkBuffer, pMemoryRequirements: [*c]VkMemoryRequirements) callconv(.c) void;
pub extern fn vkAllocateMemory(device: VkDevice, pAllocateInfo: [*c]const VkMemoryAllocateInfo, pAllocator: ?*const VkAllocationCallbacks, pMemory: [*c]VkDeviceMemory) callconv(.c) VkResult;
pub extern fn vkFreeMemory(device: VkDevice, memory: VkDeviceMemory, pAllocator: ?*const VkAllocationCallbacks) callconv(.c) void;
pub extern fn vkBindBufferMemory(device: VkDevice, buffer: VkBuffer, memory: VkDeviceMemory, memoryOffset: VkDeviceSize) callconv(.c) VkResult;
pub extern fn vkMapMemory(device: VkDevice, memory: VkDeviceMemory, offset: VkDeviceSize, size: VkDeviceSize, flags: VkMemoryMapFlags, ppData: ?*?*anyopaque) callconv(.c) VkResult;
pub extern fn vkUnmapMemory(device: VkDevice, memory: VkDeviceMemory) callconv(.c) void;
pub extern fn vkCreateCommandPool(device: VkDevice, pCreateInfo: [*c]const VkCommandPoolCreateInfo, pAllocator: ?*const VkAllocationCallbacks, pCommandPool: [*c]VkCommandPool) callconv(.c) VkResult;
pub extern fn vkDestroyCommandPool(device: VkDevice, commandPool: VkCommandPool, pAllocator: ?*const VkAllocationCallbacks) callconv(.c) void;
pub extern fn vkResetCommandPool(device: VkDevice, commandPool: VkCommandPool, flags: VkCommandPoolResetFlags) callconv(.c) VkResult;
pub extern fn vkAllocateCommandBuffers(device: VkDevice, pAllocateInfo: [*c]const VkCommandBufferAllocateInfo, pCommandBuffers: [*c]VkCommandBuffer) callconv(.c) VkResult;
pub extern fn vkBeginCommandBuffer(commandBuffer: VkCommandBuffer, pBeginInfo: [*c]const VkCommandBufferBeginInfo) callconv(.c) VkResult;
pub extern fn vkEndCommandBuffer(commandBuffer: VkCommandBuffer) callconv(.c) VkResult;
pub extern fn vkCreateFence(device: VkDevice, pCreateInfo: [*c]const VkFenceCreateInfo, pAllocator: ?*const VkAllocationCallbacks, pFence: [*c]VkFence) callconv(.c) VkResult;
pub extern fn vkDestroyFence(device: VkDevice, fence: VkFence, pAllocator: ?*const VkAllocationCallbacks) callconv(.c) void;
pub extern fn vkResetFences(device: VkDevice, fenceCount: u32, pFences: [*c]const VkFence) callconv(.c) VkResult;
pub extern fn vkQueueSubmit(queue: VkQueue, submitCount: u32, pSubmits: [*c]const VkSubmitInfo, fence: VkFence) callconv(.c) VkResult;
pub extern fn vkWaitForFences(device: VkDevice, fenceCount: u32, pFences: [*c]const VkFence, waitAll: VkBool32, timeout: u64) callconv(.c) VkResult;
pub extern fn vkCmdCopyBuffer(commandBuffer: VkCommandBuffer, srcBuffer: VkBuffer, dstBuffer: VkBuffer, regionCount: u32, pRegions: [*c]const VkBufferCopy) callconv(.c) void;
pub extern fn vkCmdPipelineBarrier(commandBuffer: VkCommandBuffer, srcStageMask: VkPipelineStageFlags, dstStageMask: VkPipelineStageFlags, dependencyFlags: VkDependencyFlags, memoryBarrierCount: u32, pMemoryBarriers: [*c]const VkMemoryBarrier, bufferMemoryBarrierCount: u32, pBufferMemoryBarriers: [*c]const VkBufferMemoryBarrier, imageMemoryBarrierCount: u32, pImageMemoryBarriers: [*c]const VkImageMemoryBarrier) callconv(.c) void;
pub extern fn vkCreateShaderModule(device: VkDevice, pCreateInfo: [*c]const VkShaderModuleCreateInfo, pAllocator: ?*const VkAllocationCallbacks, pShaderModule: [*c]VkShaderModule) callconv(.c) VkResult;
pub extern fn vkDestroyShaderModule(device: VkDevice, shaderModule: VkShaderModule, pAllocator: ?*const VkAllocationCallbacks) callconv(.c) void;
pub extern fn vkCreateDescriptorSetLayout(device: VkDevice, pCreateInfo: [*c]const VkDescriptorSetLayoutCreateInfo, pAllocator: ?*const VkAllocationCallbacks, pSetLayout: [*c]VkDescriptorSetLayout) callconv(.c) VkResult;
pub extern fn vkDestroyDescriptorSetLayout(device: VkDevice, descriptorSetLayout: VkDescriptorSetLayout, pAllocator: ?*const VkAllocationCallbacks) callconv(.c) void;
pub extern fn vkCreatePipelineLayout(device: VkDevice, pCreateInfo: [*c]const VkPipelineLayoutCreateInfo, pAllocator: ?*const VkAllocationCallbacks, pPipelineLayout: [*c]VkPipelineLayout) callconv(.c) VkResult;
pub extern fn vkDestroyPipelineLayout(device: VkDevice, pipelineLayout: VkPipelineLayout, pAllocator: ?*const VkAllocationCallbacks) callconv(.c) void;
pub extern fn vkCreateComputePipelines(device: VkDevice, pipelineCache: VkPipelineCache, createInfoCount: u32, pCreateInfos: [*c]const VkComputePipelineCreateInfo, pAllocator: ?*const VkAllocationCallbacks, pPipelines: [*c]VkPipeline) callconv(.c) VkResult;
pub extern fn vkDestroyPipeline(device: VkDevice, pipeline: VkPipeline, pAllocator: ?*const VkAllocationCallbacks) callconv(.c) void;
pub extern fn vkCreateDescriptorPool(device: VkDevice, pCreateInfo: [*c]const VkDescriptorPoolCreateInfo, pAllocator: ?*const VkAllocationCallbacks, pDescriptorPool: [*c]VkDescriptorPool) callconv(.c) VkResult;
pub extern fn vkDestroyDescriptorPool(device: VkDevice, descriptorPool: VkDescriptorPool, pAllocator: ?*const VkAllocationCallbacks) callconv(.c) void;
pub extern fn vkAllocateDescriptorSets(device: VkDevice, pAllocateInfo: [*c]const VkDescriptorSetAllocateInfo, pDescriptorSets: [*c]VkDescriptorSet) callconv(.c) VkResult;
pub extern fn vkUpdateDescriptorSets(device: VkDevice, descriptorWriteCount: u32, pDescriptorWrites: [*c]const VkWriteDescriptorSet, descriptorCopyCount: u32, pDescriptorCopies: [*c]const VkCopyDescriptorSet) callconv(.c) void;
pub extern fn vkCmdBindPipeline(commandBuffer: VkCommandBuffer, pipelineBindPoint: VkPipelineBindPoint, pipeline: VkPipeline) callconv(.c) void;
pub extern fn vkCmdBindDescriptorSets(commandBuffer: VkCommandBuffer, pipelineBindPoint: VkPipelineBindPoint, layout: VkPipelineLayout, firstSet: u32, descriptorSetCount: u32, pDescriptorSets: [*c]const VkDescriptorSet, dynamicOffsetCount: u32, pDynamicOffsets: [*c]const u32) callconv(.c) void;
pub extern fn vkCmdPushConstants(commandBuffer: VkCommandBuffer, layout: VkPipelineLayout, stageFlags: VkShaderStageFlags, offset: u32, size: u32, pValues: ?*const anyopaque) callconv(.c) void;
pub extern fn vkCmdDispatch(commandBuffer: VkCommandBuffer, groupCountX: u32, groupCountY: u32, groupCountZ: u32) callconv(.c) void;
pub extern fn vkEnumerateDeviceExtensionProperties(physicalDevice: VkPhysicalDevice, pLayerName: [*c]const u8, pPropertyCount: [*c]u32, pProperties: [*c]VkExtensionProperties) callconv(.c) VkResult;
pub extern fn vkGetPhysicalDeviceFeatures2(physicalDevice: VkPhysicalDevice, pFeatures: [*c]VkPhysicalDeviceFeatures2) callconv(.c) void;
pub extern fn vkGetPhysicalDeviceMemoryProperties2(physicalDevice: VkPhysicalDevice, pMemoryProperties: [*c]VkPhysicalDeviceMemoryProperties2) callconv(.c) void;
