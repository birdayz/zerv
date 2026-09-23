"""Scoped raw Vulkan ABI inventory, not an inference implementation."""
import xml.etree.ElementTree as ET

COMMANDS = """vkCreateInstance vkDestroyInstance vkEnumeratePhysicalDevices
vkGetPhysicalDeviceProperties vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceQueueFamilyProperties
vkCreateDevice vkDestroyDevice vkGetDeviceQueue vkDeviceWaitIdle
vkCreateBuffer vkDestroyBuffer vkGetBufferMemoryRequirements vkAllocateMemory vkFreeMemory
vkBindBufferMemory vkMapMemory vkUnmapMemory
vkCreateCommandPool vkDestroyCommandPool vkResetCommandPool vkAllocateCommandBuffers
vkBeginCommandBuffer vkEndCommandBuffer vkCreateFence vkDestroyFence vkResetFences
vkQueueSubmit vkWaitForFences vkCmdCopyBuffer vkCmdPipelineBarrier
vkCreateShaderModule vkDestroyShaderModule vkCreateDescriptorSetLayout vkDestroyDescriptorSetLayout
vkCreatePipelineLayout vkDestroyPipelineLayout vkCreateComputePipelines vkDestroyPipeline
vkCreateDescriptorPool vkDestroyDescriptorPool vkAllocateDescriptorSets vkUpdateDescriptorSets
vkCmdBindPipeline vkCmdBindDescriptorSets vkCmdPushConstants vkCmdDispatch""".split()
CONSTANTS = """VK_SUCCESS VK_NOT_READY VK_TIMEOUT VK_INCOMPLETE VK_ERROR_OUT_OF_HOST_MEMORY
VK_ERROR_OUT_OF_DEVICE_MEMORY VK_ERROR_DEVICE_LOST VK_API_VERSION_1_1
VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU VK_QUEUE_COMPUTE_BIT VK_QUEUE_GRAPHICS_BIT
VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT VK_MEMORY_PROPERTY_HOST_COHERENT_BIT VK_MEMORY_PROPERTY_HOST_CACHED_BIT
VK_BUFFER_USAGE_TRANSFER_SRC_BIT VK_BUFFER_USAGE_TRANSFER_DST_BIT VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
VK_SHARING_MODE_EXCLUSIVE VK_COMMAND_BUFFER_LEVEL_PRIMARY VK_SHADER_STAGE_COMPUTE_BIT
VK_DESCRIPTOR_TYPE_STORAGE_BUFFER VK_PIPELINE_BIND_POINT_COMPUTE
VK_PIPELINE_STAGE_TRANSFER_BIT VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT VK_PIPELINE_STAGE_HOST_BIT
VK_ACCESS_TRANSFER_READ_BIT VK_ACCESS_TRANSFER_WRITE_BIT VK_ACCESS_SHADER_READ_BIT VK_ACCESS_SHADER_WRITE_BIT
VK_ACCESS_HOST_READ_BIT VK_ACCESS_HOST_WRITE_BIT VK_QUEUE_FAMILY_IGNORED
VK_MAX_PHYSICAL_DEVICE_NAME_SIZE VK_UUID_SIZE VK_MAX_MEMORY_TYPES VK_MAX_MEMORY_HEAPS""".split()
OPAQUE = {"VkAllocationCallbacks"}


def inventory(xml):
    root = ET.parse(xml).getroot()
    # The registry also contains Vulkan SC alternatives, including duplicate members.
    for parent in root.iter():
        for child in list(parent):
            if child.get("api") and "vulkan" not in child.get("api").split(","):
                parent.remove(child)
    types = {t.get("name", t.findtext("name")): t for t in root.findall("types/type")}
    commands = {c.findtext("proto/name"): c for c in root.findall("commands/command") if c.find("proto") is not None}
    found = {}
    constants = set(CONSTANTS)

    def visit(name):
        if name in found or name in OPAQUE or name in {"void", "char", "uint8_t", "uint32_t", "int32_t", "uint64_t", "size_t", "float"} or name not in types:
            return
        node = types[name]
        if node.get("category") not in ("struct", "handle", "enum", "basetype", "bitmask"):
            raise ValueError("unsupported ABI type " + name)
        found[name] = node
        if node.get("category") != "handle":
            for child in node.findall(".//type"):
                visit(child.text)
        for field in node.findall("member"):
            if field.get("values"):
                constants.add(field.get("values"))
            constants.update(e.text for e in field.findall("enum"))

    for name in COMMANDS:
        for t in commands[name].findall(".//type"):
            visit(t.text)
    return found, {name: commands[name] for name in COMMANDS}, sorted(constants)
