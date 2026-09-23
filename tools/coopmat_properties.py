#!/usr/bin/env python3
"""Research probe (development only): list the cooperative-matrix configurations the
Vulkan driver reports (vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR), which
vulkaninfo does not print. It uses ctypes against the system loader and is not a
runtime dependency. Enum and struct values come from the pinned registry
(third_party/vulkan/1.4.354/registry/vk.xml; VK_KHR_cooperative_matrix is extension 507)."""
import argparse
import ctypes as C
import json
from pathlib import Path

COMPONENT = {0: "f16", 1: "f32", 2: "f64", 3: "s8", 4: "s16", 5: "s32", 6: "s64", 7: "u8", 8: "u16", 9: "u32", 10: "u64", 1000141000: "bf16"}
SCOPE = {1: "device", 2: "workgroup", 3: "subgroup", 5: "queue_family"}
ST_APPLICATION_INFO, ST_INSTANCE_CREATE_INFO = 0, 1
ST_COOPMAT_PROPERTIES = 1000000000 + (507 - 1) * 1000 + 1


class AppInfo(C.Structure):
    _fields_ = [("sType", C.c_int), ("pNext", C.c_void_p), ("pApplicationName", C.c_char_p), ("applicationVersion", C.c_uint32),
                ("pEngineName", C.c_char_p), ("engineVersion", C.c_uint32), ("apiVersion", C.c_uint32)]


class InstanceInfo(C.Structure):
    _fields_ = [("sType", C.c_int), ("pNext", C.c_void_p), ("flags", C.c_uint32), ("pApplicationInfo", C.POINTER(AppInfo)),
                ("enabledLayerCount", C.c_uint32), ("ppEnabledLayerNames", C.c_void_p),
                ("enabledExtensionCount", C.c_uint32), ("ppEnabledExtensionNames", C.c_void_p)]


class CoopmatProperties(C.Structure):
    _fields_ = [("sType", C.c_int), ("pNext", C.c_void_p), ("MSize", C.c_uint32), ("NSize", C.c_uint32), ("KSize", C.c_uint32),
                ("AType", C.c_int), ("BType", C.c_int), ("CType", C.c_int), ("ResultType", C.c_int),
                ("saturatingAccumulation", C.c_uint32), ("scope", C.c_int)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    vk = C.CDLL("libvulkan.so.1")
    app = AppInfo(ST_APPLICATION_INFO, None, b"zerv-coopmat-probe", 0, None, 0, (1 << 22) | (3 << 12))
    info = InstanceInfo(ST_INSTANCE_CREATE_INFO, None, 0, C.pointer(app), 0, None, 0, None)
    instance = C.c_void_p()
    if vk.vkCreateInstance(C.byref(info), None, C.byref(instance)) != 0: raise SystemExit("vkCreateInstance failed")
    count = C.c_uint32()
    vk.vkEnumeratePhysicalDevices(instance, C.byref(count), None)
    devices = (C.c_void_p * count.value)()
    vk.vkEnumeratePhysicalDevices(instance, C.byref(count), devices)
    vk.vkGetInstanceProcAddr.restype = C.c_void_p
    fn_ptr = vk.vkGetInstanceProcAddr(instance, b"vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR")
    get = C.CFUNCTYPE(C.c_int, C.c_void_p, C.POINTER(C.c_uint32), C.POINTER(CoopmatProperties))(fn_ptr)
    report = []
    for device in devices:
        props = (C.c_uint8 * 1024)()
        vk.vkGetPhysicalDeviceProperties(C.c_void_p(device), props)
        name = bytes(props[20:20 + 256]).split(b"\0")[0].decode()  # deviceName follows 5 u32 fields
        n = C.c_uint32()
        get(device, C.byref(n), None)
        entries = (CoopmatProperties * n.value)()
        for e in entries: e.sType = ST_COOPMAT_PROPERTIES
        get(device, C.byref(n), entries)
        rows = [dict(M=e.MSize, N=e.NSize, K=e.KSize, A=COMPONENT.get(e.AType, e.AType), B=COMPONENT.get(e.BType, e.BType),
                     C=COMPONENT.get(e.CType, e.CType), Result=COMPONENT.get(e.ResultType, e.ResultType),
                     saturating=bool(e.saturatingAccumulation), scope=SCOPE.get(e.scope, e.scope)) for e in entries]
        report.append(dict(device=name, configurations=rows))
        print(name)
        for r in rows: print("  ", r)
    vk.vkDestroyInstance(instance, None)
    if a.output: a.output.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
