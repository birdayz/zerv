// Lab (research tool, not runtime): run a compute kernel from our own machine code
// through VK_KHR_pipeline_binary on RADV (docs/research/native-isa-via-vulkan.md).
//
//   dump SPV OUT          create gemm_f16x from SPIR-V with CAPTURE_DATA, write the RADV
//                         pipeline binary to OUT.bin and its key to OUT.key, and the
//                         driver's global pipeline key to OUT.global
//   race SPV BIN [key=value ...]
//                         run gemm_f16x from SPV (reference) and from the binary BIN on the
//                         same random Q4_0 GEMM, compare all outputs bitwise, then time both
//                         interleaved (GPU timestamps), print JSON lines. Keys (defaults):
//                         m=17408 k=5120 rows=512 (plan rows, grid y = rows/256) n=rows
//                         (io row count; rows >= n read row n-1) seed=1 reps=101 (0: check
//                         only) abase=0 (A byte offset; 2 makes even blocks misaligned)
//                         xbase=0 (X offset in halves, multiple of 8) sub=0 (per mille of
//                         blocks with f16-subnormal scales) batch=1 (dispatches per timed side
//                         and rep, separated by barriers; the time is per dispatch) dump=FILE
//                         (write the candidate's Y: after the check run if reps=0, else after
//                         the last timed dispatch) iodev=0 (1: io in device-local memory; zerv
//                         keeps it host-visible)
// Interface of gemm_f16x (src/model/gemm_f16x.comp): bindings 0 A (Q4_0 rows, bytes),
// 1 Y (f32), 2 io (io[2] = row count), 3 X (f16, halves); push constants gemm.Push
// (15 u32); workgroup 256, required subgroup size 32, full subgroups; grid (M/128, rows/256).
// Build: bazel build //bench/isa_lab:pipeline_binary_lab (tools/zerv_build.py builds it for the scripts)
#include <vulkan/vulkan_core.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s:%d %s -> %d\n", __FILE__, __LINE__, #x, r_); exit(1); } } while (0)

static VkInstance inst; static VkPhysicalDevice phys; static VkDevice dev; static VkQueue queue; static uint32_t family;
static PFN_vkCreatePipelineBinariesKHR pCreatePipelineBinaries;
static PFN_vkGetPipelineBinaryDataKHR pGetPipelineBinaryData;
static PFN_vkGetPipelineKeyKHR pGetPipelineKey;
static PFN_vkDestroyPipelineBinaryKHR pDestroyPipelineBinary;
static PFN_vkReleaseCapturedPipelineDataKHR pReleaseCaptured;

typedef struct { uint32_t m, k, rows, n, abase, xbase, sub, batch, iodev; uint64_t seed; int reps; const char *dump; } Cfg;
typedef struct { uint32_t a_base, a_rs, a_cs, a_bs, a_group, x_base, x_rs, x_bs, y_base, y_rs, y_bs, m, k, flags, k_chunk; } Push;

static void *read_file(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb"); if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); *size = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    void *d = malloc(*size + 4); if (fread(d, 1, *size, f) != *size) { perror(path); exit(1); } fclose(f); return d;
}
static void write_file(const char *path, const void *d, size_t n) {
    FILE *f = fopen(path, "wb"); if (!f || fwrite(d, 1, n, f) != n) { perror(path); exit(1); } fclose(f);
}

static void open_device(void) {
    VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_3 };
    VkInstanceCreateInfo ii = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app };
    CHECK(vkCreateInstance(&ii, NULL, &inst));
    uint32_t n = 1; vkEnumeratePhysicalDevices(inst, &n, &phys);
    uint32_t qn = 16; VkQueueFamilyProperties q[16]; vkGetPhysicalDeviceQueueFamilyProperties(phys, &qn, q);
    family = UINT32_MAX;
    for (uint32_t i = 0; i < qn; i++) if (q[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { if (family == UINT32_MAX || !(q[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) family = i; }
    float prio = 1;
    VkDeviceQueueCreateInfo qi = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = family, .queueCount = 1, .pQueuePriorities = &prio };
    VkPhysicalDevicePipelineBinaryFeaturesKHR pb = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_BINARY_FEATURES_KHR, .pipelineBinaries = VK_TRUE };
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR, .pNext = &pb, .cooperativeMatrix = VK_TRUE };
    VkPhysicalDeviceVulkan13Features v13 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, .pNext = &cm, .subgroupSizeControl = VK_TRUE, .computeFullSubgroups = VK_TRUE, .maintenance4 = VK_TRUE };
    VkPhysicalDeviceVulkan12Features v12 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .pNext = &v13, .shaderFloat16 = VK_TRUE, .vulkanMemoryModel = VK_TRUE };
    VkPhysicalDeviceVulkan11Features v11 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, .pNext = &v12, .storageBuffer16BitAccess = VK_TRUE };
    VkPhysicalDeviceMaintenance5FeaturesKHR m5 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR, .pNext = &v11, .maintenance5 = VK_TRUE };
    const char *ext[] = { "VK_KHR_cooperative_matrix", "VK_KHR_pipeline_binary", "VK_KHR_maintenance5" };
    VkDeviceCreateInfo di = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &m5, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi, .enabledExtensionCount = 3, .ppEnabledExtensionNames = ext };
    CHECK(vkCreateDevice(phys, &di, NULL, &dev));
    vkGetDeviceQueue(dev, family, 0, &queue);
    pCreatePipelineBinaries = (PFN_vkCreatePipelineBinariesKHR)vkGetDeviceProcAddr(dev, "vkCreatePipelineBinariesKHR");
    pGetPipelineBinaryData = (PFN_vkGetPipelineBinaryDataKHR)vkGetDeviceProcAddr(dev, "vkGetPipelineBinaryDataKHR");
    pGetPipelineKey = (PFN_vkGetPipelineKeyKHR)vkGetDeviceProcAddr(dev, "vkGetPipelineKeyKHR");
    pDestroyPipelineBinary = (PFN_vkDestroyPipelineBinaryKHR)vkGetDeviceProcAddr(dev, "vkDestroyPipelineBinaryKHR");
    pReleaseCaptured = (PFN_vkReleaseCapturedPipelineDataKHR)vkGetDeviceProcAddr(dev, "vkReleaseCapturedPipelineDataKHR");
    if (!pCreatePipelineBinaries || !pGetPipelineBinaryData || !pGetPipelineKey) { fprintf(stderr, "no VK_KHR_pipeline_binary\n"); exit(1); }
}

static VkDescriptorSetLayout set_layout; static VkPipelineLayout pipe_layout;
static void make_layout(void) {
    VkDescriptorSetLayoutBinding b[4];
    for (int i = 0; i < 4; i++) b[i] = (VkDescriptorSetLayoutBinding){ .binding = (uint32_t)i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
    VkDescriptorSetLayoutCreateInfo si = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 4, .pBindings = b };
    CHECK(vkCreateDescriptorSetLayout(dev, &si, NULL, &set_layout));
    VkPushConstantRange pr = { .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = sizeof(Push) };
    VkPipelineLayoutCreateInfo li = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &set_layout, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr };
    CHECK(vkCreatePipelineLayout(dev, &li, NULL, &pipe_layout));
}

// The stage exactly as zerv creates gemm_f16x (gpu.Kernel with subgroup_size 32, full subgroups).
static VkPipeline pipeline_from_spirv(const char *spv_path, int capture) {
    size_t n; void *code = read_file(spv_path, &n);
    VkShaderModuleCreateInfo mi = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = n, .pCode = code };
    VkShaderModule mod; CHECK(vkCreateShaderModule(dev, &mi, NULL, &mod));
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo req = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO, .requiredSubgroupSize = 32 };
    VkPipelineCreateFlags2CreateInfoKHR f2 = { .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO_KHR, .flags = VK_PIPELINE_CREATE_2_CAPTURE_DATA_BIT_KHR };
    VkComputePipelineCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .pNext = capture ? &f2 : NULL,
        .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .pNext = &req, .flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = mod, .pName = "main" },
        .layout = pipe_layout, .basePipelineIndex = -1 };
    VkPipeline p; CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, NULL, &p));
    vkDestroyShaderModule(dev, mod, NULL); free(code);
    return p;
}

static VkPipeline pipeline_from_binary(const char *bin_path) {
    size_t n; void *data = read_file(bin_path, &n);
    char key_path[4096]; snprintf(key_path, sizeof key_path, "%.*s.key", (int)(strlen(bin_path) - 4), bin_path);
    size_t kn; void *keyd = read_file(key_path, &kn);
    VkPipelineBinaryKeyKHR key = { .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR, .keySize = (uint32_t)kn };
    memcpy(key.key, keyd, kn);
    VkPipelineBinaryDataKHR bd = { .dataSize = n, .pData = data };
    VkPipelineBinaryKeysAndDataKHR kd = { .binaryCount = 1, .pPipelineBinaryKeys = &key, .pPipelineBinaryData = &bd };
    VkPipelineBinaryCreateInfoKHR bci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR, .pKeysAndDataInfo = &kd };
    VkPipelineBinaryKHR bin; VkPipelineBinaryHandlesInfoKHR hi = { .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_HANDLES_INFO_KHR, .pipelineBinaryCount = 1, .pPipelineBinaries = &bin };
    CHECK(pCreatePipelineBinaries(dev, &bci, NULL, &hi));
    VkPipelineBinaryInfoKHR bi = { .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_INFO_KHR, .binaryCount = 1, .pPipelineBinaries = &bin };
    // With a binary, the stage's module is ignored by the spec (pipeline binaries replace
    // compilation); RADV still reads pName/stage for bookkeeping.
    VkComputePipelineCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .pNext = &bi,
        .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .pName = "main" },
        .layout = pipe_layout, .basePipelineIndex = -1 };
    VkPipeline p; CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, NULL, &p));
    pDestroyPipelineBinary(dev, bin, NULL); free(data); free(keyd);
    return p;
}

static void cmd_dump(const char *spv, const char *out) {
    VkPipeline p = pipeline_from_spirv(spv, 1);
    VkPipelineBinaryCreateInfoKHR bci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR, .pipeline = p };
    VkPipelineBinaryHandlesInfoKHR hi = { .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_HANDLES_INFO_KHR };
    CHECK(pCreatePipelineBinaries(dev, &bci, NULL, &hi));
    if (hi.pipelineBinaryCount != 1) { fprintf(stderr, "expected 1 binary, got %u\n", hi.pipelineBinaryCount); exit(1); }
    VkPipelineBinaryKHR bin; hi.pPipelineBinaries = &bin;
    CHECK(pCreatePipelineBinaries(dev, &bci, NULL, &hi));
    VkPipelineBinaryDataInfoKHR di = { .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_DATA_INFO_KHR, .pipelineBinary = bin };
    VkPipelineBinaryKeyKHR key = { .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR };
    size_t n = 0; CHECK(pGetPipelineBinaryData(dev, &di, &key, &n, NULL));
    void *data = malloc(n); CHECK(pGetPipelineBinaryData(dev, &di, &key, &n, data));
    char path[4096];
    snprintf(path, sizeof path, "%s.bin", out); write_file(path, data, n);
    snprintf(path, sizeof path, "%s.key", out); write_file(path, key.key, key.keySize);
    VkPipelineBinaryKeyKHR global = { .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR };
    CHECK(pGetPipelineKey(dev, NULL, &global));
    snprintf(path, sizeof path, "%s.global", out); write_file(path, global.key, global.keySize);
    printf("{\"binary_bytes\":%zu,\"key_bytes\":%u,\"global_key_bytes\":%u}\n", n, key.keySize, global.keySize);
    VkReleaseCapturedPipelineDataInfoKHR rel = { .sType = VK_STRUCTURE_TYPE_RELEASE_CAPTURED_PIPELINE_DATA_INFO_KHR, .pipeline = p };
    pReleaseCaptured(dev, &rel, NULL);
    pDestroyPipelineBinary(dev, bin, NULL); vkDestroyPipeline(dev, p, NULL); free(data);
}

typedef struct { VkBuffer b; VkDeviceMemory m; void *map; VkDeviceSize size; } Buf;
static Buf make_buf(VkDeviceSize size, int host) {
    Buf r = { .size = size };
    VkBufferCreateInfo bi = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT };
    CHECK(vkCreateBuffer(dev, &bi, NULL, &r.b));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, r.b, &mr);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    VkMemoryPropertyFlags want = host ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    uint32_t t = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) if ((mr.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) { t = i; break; }
    VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size, .memoryTypeIndex = t };
    CHECK(vkAllocateMemory(dev, &ai, NULL, &r.m)); CHECK(vkBindBufferMemory(dev, r.b, r.m, 0));
    if (host) CHECK(vkMapMemory(dev, r.m, 0, VK_WHOLE_SIZE, 0, &r.map));
    return r;
}

static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)(rng >> 32); }
static uint16_t f16_of(float f) { // round to nearest even, normal range only (|f| in [2^-14, 65504])
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u; int32_t e = (int32_t)((x >> 23) & 255) - 127 + 15; uint32_t mant = x & 0x7fffffu;
    if (e <= 0) return (uint16_t)sign;
    uint32_t h = sign | ((uint32_t)e << 10) | (mant >> 13); uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h++;
    return (uint16_t)h;
}

static int cmp_double(const void *p, const void *q) { double a_ = *(const double *)p, b_ = *(const double *)q; return (a_ > b_) - (a_ < b_); }

static void cmd_race(const char *spv, const char *bin, Cfg c) {
    const uint32_t M = c.m, K = c.k, ROWS = c.rows; int reps = c.reps;
    const uint32_t row_bytes = K / 32 * 18;
    rng = 0x9e3779b97f4a7c15ull ^ (c.seed * 0xd1b54a32d192ed03ull); if (!rng) rng = 1;
    // Host staging (filled here) and device-local buffers (what the kernels read), as in
    // the runtime: weights and activations in VRAM.
    Buf ah = make_buf((VkDeviceSize)M * row_bytes + 64, 1), xh = make_buf(((VkDeviceSize)ROWS * K + c.xbase) * 2 + 64, 1), yh = make_buf((VkDeviceSize)2 * ROWS * M * 4, 1), io = make_buf(256, 1);
    Buf a = make_buf(ah.size, 0), x = make_buf(xh.size, 0), y = make_buf(yh.size, 0), iod = make_buf(256, 0);
    uint8_t *ab = ah.map;
    for (uint64_t i = 0; i < ah.size; i++) ab[i] = (uint8_t)rnd(); // bytes outside the rows too
    for (uint64_t blk = 0; blk < (uint64_t)M * (K / 32); blk++) {
        uint8_t *bp = ab + c.abase + blk * 18;
        float d = ((float)(rnd() % 2000) / 1000.0f - 1.0f) * 0.02f; if (fabsf(d) < 1e-3f) d = 1e-3f;
        uint16_t h = f16_of(d);
        if (rnd() % 1000 < c.sub) h = (uint16_t)((rnd() & 0x8000u) | (1u + rnd() % 0x3ffu)); // f16 subnormal scale
        memcpy(bp, &h, 2);
        for (int j = 0; j < 16; j++) bp[2 + j] = (uint8_t)rnd();
    }
    uint16_t *xb = xh.map;
    for (uint64_t i = 0; i < (uint64_t)ROWS * K + c.xbase; i++) { float v = ((float)(rnd() % 20001) / 10000.0f - 1.0f); if (fabsf(v) < 1e-3f) v = 0.5f; xb[i] = f16_of(v); }
    ((uint32_t *)io.map)[2] = c.n;
    size_t sl = strlen(spv); // a reference ending in .bin is another pipeline binary
    VkPipeline pa = (sl > 4 && !strcmp(spv + sl - 4, ".bin")) ? pipeline_from_binary(spv) : pipeline_from_spirv(spv, 0), pb = pipeline_from_binary(bin);
    VkDescriptorPoolSize ps = { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 4 };
    VkDescriptorPoolCreateInfo dpi = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps };
    VkDescriptorPool pool; CHECK(vkCreateDescriptorPool(dev, &dpi, NULL, &pool));
    VkDescriptorSetAllocateInfo sai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = pool, .descriptorSetCount = 1, .pSetLayouts = &set_layout };
    VkDescriptorSet set; CHECK(vkAllocateDescriptorSets(dev, &sai, &set));
    VkDescriptorBufferInfo bi[4] = { { a.b, 0, VK_WHOLE_SIZE }, { y.b, 0, VK_WHOLE_SIZE }, { c.iodev ? iod.b : io.b, 0, VK_WHOLE_SIZE }, { x.b, 0, VK_WHOLE_SIZE } };
    VkWriteDescriptorSet w[4];
    for (int i = 0; i < 4; i++) w[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = (uint32_t)i, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[i] };
    vkUpdateDescriptorSets(dev, 4, w, 0, NULL);
    VkCommandPoolCreateInfo cpi = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = family };
    VkCommandPool cp; CHECK(vkCreateCommandPool(dev, &cpi, NULL, &cp));
    VkCommandBufferAllocateInfo cai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
    VkCommandBuffer cb; CHECK(vkAllocateCommandBuffers(dev, &cai, &cb));
    VkQueryPoolCreateInfo qpi = { .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 4 };
    VkQueryPool qp; CHECK(vkCreateQueryPool(dev, &qpi, NULL, &qp));
    VkFenceCreateInfo fi = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO }; VkFence fence; CHECK(vkCreateFence(dev, &fi, NULL, &fence));
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(phys, &props);
    Push pa_push = { .a_base = c.abase, .a_rs = row_bytes, .a_group = 1, .x_base = c.xbase, .x_rs = K, .y_base = 0, .y_rs = M, .m = M, .k = K };
    Push pb_push = pa_push; pb_push.y_base = ROWS * M; // B writes the second half of Y
    // One submission: both pipelines once (A into Y[0], B into Y[1]), then check bitwise.
    // Timing: per rep, A then B (then B then A on odd reps), each bracketed by timestamps.
    static double ta[4096], tb[4096]; if (reps > 4096) reps = 4096;
    memset(yh.map, 0xff, yh.size);
    { // upload A, X and the Y sentinel
        CHECK(vkResetCommandBuffer(cb, 0));
        VkCommandBufferBeginInfo cbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
        CHECK(vkBeginCommandBuffer(cb, &cbi));
        VkBufferCopy ca = { .size = ah.size }, cx = { .size = xh.size }, cy = { .size = yh.size };
        vkCmdCopyBuffer(cb, ah.b, a.b, 1, &ca); vkCmdCopyBuffer(cb, xh.b, x.b, 1, &cx); vkCmdCopyBuffer(cb, yh.b, y.b, 1, &cy);
        VkBufferCopy cio = { .size = 256 }; vkCmdCopyBuffer(cb, io.b, iod.b, 1, &cio);
        VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
        CHECK(vkEndCommandBuffer(cb));
        VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb };
        CHECK(vkQueueSubmit(queue, 1, &si, fence)); CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 60000000000ull)); CHECK(vkResetFences(dev, 1, &fence));
    }
    for (int rep = -1; rep < reps; rep++) {
        CHECK(vkResetCommandBuffer(cb, 0));
        VkCommandBufferBeginInfo cbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
        CHECK(vkBeginCommandBuffer(cb, &cbi));
        vkCmdResetQueryPool(cb, qp, 0, 4);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout, 0, 1, &set, 0, NULL);
        int b_first = rep >= 0 && (rep & 1);
        for (int s = 0; s < 2; s++) {
            int use_b = s ^ b_first;
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, use_b ? pb : pa);
            vkCmdPushConstants(cb, pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push), use_b ? &pb_push : &pa_push);
            VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, (uint32_t)(use_b * 2));
            for (uint32_t d = 0; d < (rep >= 0 ? c.batch : 1u); d++) {
                if (d) vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
                vkCmdDispatch(cb, M / 128, ROWS / 256, 1);
            }
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, (uint32_t)(use_b * 2 + 1));
        }
        if (rep == -1) { // read Y back for the bitwise check
            VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT };
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
            VkBufferCopy cy = { .size = yh.size }; vkCmdCopyBuffer(cb, y.b, yh.b, 1, &cy);
            VkMemoryBarrier hb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT };
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0, NULL, 0, NULL);
        }
        CHECK(vkEndCommandBuffer(cb));
        VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb };
        CHECK(vkQueueSubmit(queue, 1, &si, fence)); CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 60000000000ull)); CHECK(vkResetFences(dev, 1, &fence));
        uint64_t t[4]; CHECK(vkGetQueryPoolResults(dev, qp, 0, 4, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        if (rep >= 0) { ta[rep] = (double)(t[1] - t[0]) * props.limits.timestampPeriod * 1e-6 / c.batch; tb[rep] = (double)(t[3] - t[2]) * props.limits.timestampPeriod * 1e-6 / c.batch; }
        if (rep == -1) {
            const uint32_t *yy = yh.map; size_t diff = 0, first = SIZE_MAX, nan = 0;
            for (size_t i = 0; i < (size_t)ROWS * M; i++) {
                if (yy[i] != yy[(size_t)ROWS * M + i]) { diff++; if (first == SIZE_MAX) first = i; }
                if ((yy[i] & 0x7f800000u) == 0x7f800000u) nan++;
            }
            if (c.dump && reps <= 0) write_file(c.dump, (const uint32_t *)yh.map + (size_t)ROWS * M, (size_t)ROWS * M * 4);
            size_t zero = 0; for (size_t i = 0; i < (size_t)ROWS * M; i++) zero += (yy[i] & 0x7fffffffu) == 0;
            printf("{\"check\":\"bitwise\",\"m\":%u,\"k\":%u,\"rows\":%u,\"n\":%u,\"seed\":%llu,\"abase\":%u,\"xbase\":%u,\"sub\":%u,\"values\":%zu,\"different\":%zu,\"first_different\":%lld,\"nonfinite_reference\":%zu,\"zero_reference\":%zu}\n",
                   M, K, ROWS, c.n, (unsigned long long)c.seed, c.abase, c.xbase, c.sub, (size_t)ROWS * M, diff, first == SIZE_MAX ? -1LL : (long long)first, nan, zero);
            fflush(stdout);
        }
    }
    if (reps <= 0) return;
    if (c.dump) { // Y of the candidate's last timed dispatch (sustained state)
        CHECK(vkResetCommandBuffer(cb, 0));
        VkCommandBufferBeginInfo cbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
        CHECK(vkBeginCommandBuffer(cb, &cbi));
        VkBufferCopy cy = { .size = yh.size }; vkCmdCopyBuffer(cb, y.b, yh.b, 1, &cy);
        CHECK(vkEndCommandBuffer(cb));
        VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb };
        CHECK(vkQueueSubmit(queue, 1, &si, fence)); CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 60000000000ull)); CHECK(vkResetFences(dev, 1, &fence));
        write_file(c.dump, (const uint32_t *)yh.map + (size_t)ROWS * M, (size_t)ROWS * M * 4);
    }
    qsort(ta, (size_t)reps, sizeof(double), cmp_double); qsort(tb, (size_t)reps, sizeof(double), cmp_double);
    double flops = 2.0 * M * K * ROWS;
    printf("{\"timing\":\"interleaved\",\"m\":%u,\"k\":%u,\"rows\":%u,\"reps\":%d,\"batch\":%u,\"spirv_ms_median\":%.4f,\"binary_ms_median\":%.4f,\"spirv_tflops\":%.2f,\"binary_tflops\":%.2f,\"ratio\":%.4f}\n",
           M, K, ROWS, reps, c.batch, ta[reps / 2], tb[reps / 2], flops / (ta[reps / 2] * 1e-3) / 1e12, flops / (tb[reps / 2] * 1e-3) / 1e12, ta[reps / 2] / tb[reps / 2]);
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: dump SPV OUT | race SPV BIN [key=value ...]\n"); return 2; }
    Cfg c = { .m = 17408, .k = 5120, .rows = 512, .n = 0, .seed = 1, .reps = 101, .batch = 1 };
    for (int i = 4; i < argc; i++) {
        const char *eq = strchr(argv[i], '='); if (!eq) { fprintf(stderr, "bad argument %s\n", argv[i]); return 2; }
        if (!strncmp(argv[i], "dump=", 5)) { c.dump = eq + 1; continue; }
        unsigned long long v = strtoull(eq + 1, NULL, 0); size_t kl = (size_t)(eq - argv[i]);
#define KEY(name) (kl == strlen(name) && !strncmp(argv[i], name, kl))
        if (KEY("m")) c.m = (uint32_t)v; else if (KEY("k")) c.k = (uint32_t)v; else if (KEY("rows")) c.rows = (uint32_t)v;
        else if (KEY("n")) c.n = (uint32_t)v; else if (KEY("seed")) c.seed = v; else if (KEY("reps")) c.reps = (int)v;
        else if (KEY("abase")) c.abase = (uint32_t)v; else if (KEY("xbase")) c.xbase = (uint32_t)v; else if (KEY("sub")) c.sub = (uint32_t)v; else if (KEY("batch")) c.batch = v ? (uint32_t)v : 1u; else if (KEY("iodev")) c.iodev = (uint32_t)v;
        else { fprintf(stderr, "unknown key %s\n", argv[i]); return 2; }
#undef KEY
    }
    if (!c.n) c.n = c.rows;
    if (c.m % 128 || c.k % 64 || !c.rows || c.rows % 256 || c.n > c.rows || c.xbase % 8 || c.abase > 60) { fprintf(stderr, "invalid shape\n"); return 2; }
    open_device(); make_layout();
    if (!strcmp(argv[1], "dump")) cmd_dump(argv[2], argv[3]);
    else if (!strcmp(argv[1], "race")) cmd_race(argv[2], argv[3], c);
    else return 2;
    return 0;
}
