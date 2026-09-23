/* Independent raw Vulkan driver oracle/benchmark. Never linked into zerv. */
#define _POSIX_C_SOURCE 200809L
#include <vulkan/vulkan_core.h>
#include <openssl/sha.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void fail(const char *what) { fprintf(stderr, "%s\n", what); exit(2); }
#define CHECK(call) do { VkResult r_ = (call); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s: VkResult %d\n", #call, r_); exit(2); } } while (0)
static uint64_t now(void) { struct timespec t; if (clock_gettime(CLOCK_MONOTONIC, &t)) fail("clock"); return (uint64_t)t.tv_sec*1000000000+t.tv_nsec; }
static void digest(const void *p, size_t n, char hex[65]) { unsigned char h[32]; if (!SHA256(p,n,h)) fail("sha"); for (int i=0;i<32;++i) snprintf(hex+2*i,3,"%02x",h[i]); }
static uint32_t input_word(size_t i) {
    if (i == 0) return 0;
    if (i == 1) return UINT32_MAX;
    if (i == 2) return UINT32_C(0x80000000);
    return ((uint32_t)i * UINT32_C(0x9e3779b9)) ^ UINT32_C(0xa5a5a5a5);
}
static VkInstance instance;
static VkPhysicalDevice physical;
static VkPhysicalDeviceProperties properties;
static VkPhysicalDeviceMemoryProperties memory;
static VkDevice device;
static VkQueue queue;
static uint32_t family;
static void init(void) {
    VkApplicationInfo app = {.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,.pApplicationName="zerv-external-driver-oracle",.apiVersion=VK_API_VERSION_1_1};
    VkInstanceCreateInfo info = {.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,.pApplicationInfo=&app};
    CHECK(vkCreateInstance(&info,NULL,&instance));
    VkPhysicalDevice list[16]; uint32_t count=16;
    CHECK(vkEnumeratePhysicalDevices(instance,&count,list));
    for (uint32_t i=0;i<count;++i) {
        VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(list[i],&p);
        if (p.apiVersion < VK_API_VERSION_1_1 || p.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) continue;
        uint32_t n=0; vkGetPhysicalDeviceQueueFamilyProperties(list[i],&n,NULL);
        if (n>32) fail("queue bound");
        VkQueueFamilyProperties qs[32]; vkGetPhysicalDeviceQueueFamilyProperties(list[i],&n,qs);
        uint32_t selected=UINT32_MAX;
        for (uint32_t q=0;q<n;++q) if (qs[q].queueCount && (qs[q].queueFlags&VK_QUEUE_COMPUTE_BIT)) {
            if (selected==UINT32_MAX || !(qs[q].queueFlags&VK_QUEUE_GRAPHICS_BIT)) selected=q;
        }
        if (selected==UINT32_MAX) continue;
        physical=list[i]; properties=p; family=selected; break;
    }
    if (!physical) fail("no suitable discrete device");
    vkGetPhysicalDeviceMemoryProperties(physical,&memory);
    float priority=1;
    VkDeviceQueueCreateInfo qi={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,.queueFamilyIndex=family,.queueCount=1,.pQueuePriorities=&priority};
    VkDeviceCreateInfo di={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,.queueCreateInfoCount=1,.pQueueCreateInfos=&qi};
    CHECK(vkCreateDevice(physical,&di,NULL,&device)); vkGetDeviceQueue(device,family,0,&queue);
    fprintf(stderr,"device=%s vendor=%x id=%x api=%u driver=%u family=%u\n",properties.deviceName,properties.vendorID,properties.deviceID,properties.apiVersion,properties.driverVersion,family);
}
struct buffer { VkBuffer handle; VkDeviceMemory allocation; void *mapped; VkDeviceSize size, allocation_size; uint32_t memory_type; };
static struct buffer buffer_new(size_t n, int host) {
    struct buffer b={.size=n};
    VkBufferCreateInfo bi={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=n,.usage=VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,.sharingMode=VK_SHARING_MODE_EXCLUSIVE};
    CHECK(vkCreateBuffer(device,&bi,NULL,&b.handle));
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(device,b.handle,&req);
    uint32_t required=host ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    uint32_t index=UINT32_MAX;
    const uint32_t base_flags=VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT|VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT|VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    for (uint32_t i=0;i<memory.memoryTypeCount;++i) if (!(memory.memoryTypes[i].propertyFlags&~base_flags) && (req.memoryTypeBits&(1u<<i)) && (memory.memoryTypes[i].propertyFlags&required)==required) {
        if (index==UINT32_MAX || (host && !(memory.memoryTypes[i].propertyFlags&VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))) index=i;
    }
    if (index==UINT32_MAX) fail("no compatible memory");
    b.memory_type=index; b.allocation_size=req.size;
    VkMemoryAllocateInfo ai={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.allocationSize=req.size,.memoryTypeIndex=index};
    CHECK(vkAllocateMemory(device,&ai,NULL,&b.allocation)); CHECK(vkBindBufferMemory(device,b.handle,b.allocation,0));
    if (host) CHECK(vkMapMemory(device,b.allocation,0,n,0,&b.mapped));
    return b;
}
static void buffer_free(struct buffer b) { if (b.mapped) vkUnmapMemory(device,b.allocation); vkDestroyBuffer(device,b.handle,NULL); vkFreeMemory(device,b.allocation,NULL); }
static void barrier(VkCommandBuffer cmd, VkPipelineStageFlags src, VkAccessFlags read, VkPipelineStageFlags dst, VkAccessFlags write) {
    VkMemoryBarrier b={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=read,.dstAccessMask=write};
    vkCmdPipelineBarrier(cmd,src,dst,0,1,&b,0,NULL,0,NULL);
}
#define TRANSFER VK_PIPELINE_STAGE_TRANSFER_BIT
#define COMPUTE VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
#define HOST VK_PIPELINE_STAGE_HOST_BIT
#define TR (VK_ACCESS_TRANSFER_READ_BIT|VK_ACCESS_TRANSFER_WRITE_BIT)
#define CR (VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT)
#define HR (VK_ACCESS_HOST_READ_BIT|VK_ACCESS_HOST_WRITE_BIT)
static void begin(VkCommandBuffer c) { VkCommandBufferBeginInfo b={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; CHECK(vkBeginCommandBuffer(c,&b)); }
static void copy(VkCommandBuffer c, struct buffer from, struct buffer to) { VkBufferCopy r={.size=from.size}; vkCmdCopyBuffer(c,from.handle,to.handle,1,&r); }
static void submit(VkCommandBuffer c, VkFence fence) {
    CHECK(vkResetFences(device,1,&fence));
    VkSubmitInfo s={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&c};
    CHECK(vkQueueSubmit(queue,1,&s,fence)); CHECK(vkWaitForFences(device,1,&fence,VK_TRUE,UINT64_C(10000000000)));
}
static void report_check(const char *kind, size_t n, struct buffer in, struct buffer out) {
    char ih[65],oh[65];digest(in.mapped,in.size,ih);digest(out.mapped,out.size,oh);
    printf("{\"kind\":\"%s\",\"count\":%zu,\"bytes\":%llu,\"input_sha256\":\"%s\",\"output_sha256\":\"%s\"}\n",kind,n,(unsigned long long)in.size,ih,oh);
}
static void timed(const char *kind, size_t n, VkCommandBuffer cmd, VkFence fence, size_t iterations) {
    for (int i=0;i<3;++i) submit(cmd,fence);
    for (int trial=0;trial<7;++trial) {
        uint64_t start=now(); for (size_t i=0;i<iterations;++i) submit(cmd,fence); uint64_t ns=now()-start;
        printf("{\"kind\":\"timing\",\"workload\":\"%s\",\"count\":%zu,\"trial\":%d,\"iterations\":%zu,\"elapsed_ns\":%llu}\n",kind,n,trial,iterations,(unsigned long long)ns);
    }
}
static void workload(size_t n, int affine, const uint32_t *code, size_t code_bytes, int bench) {
    size_t bytes=affine ? (n+64)*4 : n;
    struct buffer input=buffer_new(bytes,1), readback=buffer_new(bytes,1), a=buffer_new(bytes,0), b=buffer_new(bytes,0);
    fprintf(stderr,"allocation kind=%s count=%zu bytes=%zu allocation_sizes=%llu,%llu,%llu,%llu memory_types=%u,%u,%u,%u\n",affine ? "affine" : "roundtrip",n,bytes,(unsigned long long)input.allocation_size,(unsigned long long)readback.allocation_size,(unsigned long long)a.allocation_size,(unsigned long long)b.allocation_size,input.memory_type,readback.memory_type,a.memory_type,b.memory_type);
    uint32_t *src=input.mapped,*dst=readback.mapped;
    for (size_t i=0;i<bytes/4;++i) { src[i]=input_word(i); dst[i]=UINT32_C(0xcdcdcdcd); }
    VkCommandPool pool; VkCommandPoolCreateInfo pi={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,.queueFamilyIndex=family}; CHECK(vkCreateCommandPool(device,&pi,NULL,&pool));
    VkCommandBuffer cmds[4]; VkCommandBufferAllocateInfo ca={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=4}; CHECK(vkAllocateCommandBuffers(device,&ca,cmds));
    VkFence fence; VkFenceCreateInfo fi={.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; CHECK(vkCreateFence(device,&fi,NULL,&fence));
    VkShaderModule shader=VK_NULL_HANDLE; VkDescriptorSetLayout set_layout=VK_NULL_HANDLE;
    VkPipelineLayout layout=VK_NULL_HANDLE; VkPipeline pipeline=VK_NULL_HANDLE; VkDescriptorPool descriptors=VK_NULL_HANDLE;
    if (affine) {
        VkShaderModuleCreateInfo sm={.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,.codeSize=code_bytes,.pCode=code}; CHECK(vkCreateShaderModule(device,&sm,NULL,&shader));
        VkDescriptorSetLayoutBinding bindings[2]; memset(bindings,0,sizeof(bindings));
        for (int i=0;i<2;++i) { bindings[i].binding=i; bindings[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bindings[i].descriptorCount=1; bindings[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT; }
        VkDescriptorSetLayoutCreateInfo sl={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,.bindingCount=2,.pBindings=bindings}; CHECK(vkCreateDescriptorSetLayout(device,&sl,NULL,&set_layout));
        VkPushConstantRange push={.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,.size=4};
        VkPipelineLayoutCreateInfo pl={.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,.setLayoutCount=1,.pSetLayouts=&set_layout,.pushConstantRangeCount=1,.pPushConstantRanges=&push}; CHECK(vkCreatePipelineLayout(device,&pl,NULL,&layout));
        VkComputePipelineCreateInfo cp={.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,.stage={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_COMPUTE_BIT,.module=shader,.pName="main"},.layout=layout,.basePipelineIndex=-1}; CHECK(vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&cp,NULL,&pipeline));
        VkDescriptorPoolSize ps={.type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=2};
        VkDescriptorPoolCreateInfo dp={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=1,.poolSizeCount=1,.pPoolSizes=&ps}; CHECK(vkCreateDescriptorPool(device,&dp,NULL,&descriptors));
        VkDescriptorSet set; VkDescriptorSetAllocateInfo da={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,.descriptorPool=descriptors,.descriptorSetCount=1,.pSetLayouts=&set_layout}; CHECK(vkAllocateDescriptorSets(device,&da,&set));
        VkDescriptorBufferInfo db[2]={{.buffer=a.handle,.range=bytes},{.buffer=b.handle,.range=bytes}}; VkWriteDescriptorSet writes[2]; memset(writes,0,sizeof(writes));
        for (int i=0;i<2;++i) writes[i]=(VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=set,.dstBinding=i,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&db[i]};
        vkUpdateDescriptorSets(device,2,writes,0,NULL);
        begin(cmds[0]); copy(cmds[0],input,a); copy(cmds[0],readback,b); barrier(cmds[0],TRANSFER,TR,COMPUTE,CR); CHECK(vkEndCommandBuffer(cmds[0])); submit(cmds[0],fence);
        begin(cmds[1]); barrier(cmds[1],TRANSFER,TR,COMPUTE,CR); barrier(cmds[1],COMPUTE,CR,COMPUTE,CR);
        vkCmdBindPipeline(cmds[1],VK_PIPELINE_BIND_POINT_COMPUTE,pipeline); vkCmdBindDescriptorSets(cmds[1],VK_PIPELINE_BIND_POINT_COMPUTE,layout,0,1,&set,0,NULL);
        uint32_t count=(uint32_t)n; vkCmdPushConstants(cmds[1],layout,VK_SHADER_STAGE_COMPUTE_BIT,0,4,&count); vkCmdDispatch(cmds[1],(count+63)/64,1,1); barrier(cmds[1],COMPUTE,CR,TRANSFER,TR); CHECK(vkEndCommandBuffer(cmds[1]));
        begin(cmds[2]); barrier(cmds[2],COMPUTE,CR,TRANSFER,TR); copy(cmds[2],b,readback); barrier(cmds[2],TRANSFER,TR,HOST,HR); CHECK(vkEndCommandBuffer(cmds[2]));
        submit(cmds[1],fence);
        if (bench) timed("affine",n,cmds[1],fence,n>=1048576 ? 100 : 1000);
        submit(cmds[2],fence);
        for (size_t i=0;i<n+64;++i) { uint32_t expected=i<n ? (src[i]*UINT32_C(1664525)+UINT32_C(1013904223))^(uint32_t)i : UINT32_C(0xcdcdcdcd); if (dst[i]!=expected) fail("affine/sentinel mismatch"); }
        report_check("affine",n,input,readback);
    } else {
        begin(cmds[3]); barrier(cmds[3],TRANSFER,TR,TRANSFER,TR); copy(cmds[3],input,a); barrier(cmds[3],TRANSFER,TR,TRANSFER,TR); copy(cmds[3],a,readback); barrier(cmds[3],TRANSFER,TR,HOST,HR); CHECK(vkEndCommandBuffer(cmds[3]));
        submit(cmds[3],fence);
        if (bench) timed("roundtrip",n,cmds[3],fence,n>=67108864 ? 10 : (n>=1048576 ? 100 : 1000));
        if (memcmp(input.mapped,readback.mapped,bytes)) fail("roundtrip mismatch");
        report_check("roundtrip",n,input,readback);
    }
    vkDestroyFence(device,fence,NULL); vkDestroyCommandPool(device,pool,NULL);
    if (affine) { vkDestroyDescriptorPool(device,descriptors,NULL); vkDestroyPipeline(device,pipeline,NULL); vkDestroyPipelineLayout(device,layout,NULL); vkDestroyDescriptorSetLayout(device,set_layout,NULL); vkDestroyShaderModule(device,shader,NULL); }
    buffer_free(b); buffer_free(a); buffer_free(readback); buffer_free(input);
}
int main(int argc,char **argv) {
    if (argc!=2 && (argc!=3 || strcmp(argv[2],"--bench"))) fail("usage: vulkan-driver SHADER [--bench]");
    FILE *f=fopen(argv[1],"rb"); if (!f || fseek(f,0,SEEK_END)) fail("shader open"); long size=ftell(f); if (size<20 || size%4 || size>1048576 || fseek(f,0,SEEK_SET)) fail("shader size");
    uint32_t *code=malloc((size_t)size); if (!code || fread(code,1,(size_t)size,f)!=(size_t)size) fail("shader read"); fclose(f);
    if (code[0]!=UINT32_C(0x07230203)) fail("shader magic");
    init();
    const size_t cases[]={1,63,64,65,5120,65537,1048576};
    for (size_t i=0;i<sizeof(cases)/sizeof(cases[0]);++i) if (argc==2 || cases[i]==65 || cases[i]==5120 || cases[i]==1048576) workload(cases[i],1,code,(size_t)size,argc==3);
    const size_t transfers[]={256,1048576,67108864};
    for (size_t i=0;i<3;++i) workload(transfers[i],0,code,(size_t)size,argc==3);
    vkDestroyDevice(device,NULL); vkDestroyInstance(instance,NULL); free(code);
    return fflush(stdout) ? 3 : 0;
}
