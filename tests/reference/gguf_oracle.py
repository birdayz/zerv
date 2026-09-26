#!/usr/bin/env python3
"""External GGUF writer/reader oracle. Never imported by the native engine/build."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import struct
import sys

WIDTHS = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
SCALARS = {0: ("u8", C.c_uint8, 241), 1: ("i8", C.c_int8, -97),
           2: ("u16", C.c_uint16, 61234), 3: ("i16", C.c_int16, -23456),
           4: ("u32", C.c_uint32, 3456789012), 5: ("i32", C.c_int32, -123456789),
           6: ("f32", C.c_float, -0.125), 7: ("bool", C.c_bool, True),
           10: ("u64", C.c_uint64, 12345678901234567890),
           11: ("i64", C.c_int64, -1234567890123456789), 12: ("f64", C.c_double, 1.125)}


def portable(path):
    """A path for a manifest: relative to Bazel's execution root for built files (the output
    base differs per machine), else as given."""
    return str(path).split("/execroot/_main/", 1)[-1]


def library(path=None):
    """The ggml oracle library: PATH, else the source-built one (@ggml//:ggml_base_so,
    docs/specs/hermetic-build.md). Returns (path, Bazel identity or None)."""
    if path is not None: return Path(path), None
    sys.path.insert(0, str(Path(__file__).absolute().parents[2]/"tools"))
    import zerv_build
    return zerv_build.oracle("libggml-base.so")


class InitParams(C.Structure):
    _fields_ = [("no_alloc", C.c_bool), ("ctx", C.c_void_p)]


class TensorInitParams(C.Structure):
    _fields_ = [("mem_size", C.c_size_t), ("mem_buffer", C.c_void_p), ("no_alloc", C.c_bool)]


class Oracle:
    def __init__(self, path):
        if sys.byteorder != "little" or C.sizeof(C.c_float) != 4 or C.sizeof(C.c_double) != 8:
            raise RuntimeError("oracle requires little-endian IEEE floats")
        self.path = Path(path).resolve(strict=True)
        self.lib = C.CDLL(str(self.path))
        self.identity = {"path": portable(self.path), "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest()}
        for name in ("ggml_version", "ggml_commit"):
            self.identity[name] = self.bind(name, C.c_char_p, [])().decode()
        self.load = self.bind("gguf_init_from_file", C.c_void_p, [C.c_char_p, InitParams])
        self.free = self.bind("gguf_free", None, [C.c_void_p])

    def bind(self, name, result, args):
        fn = getattr(self.lib, name)
        fn.restype, fn.argtypes = result, args
        return fn

    def get(self, name, ctx, *indices, result=C.c_int64):
        return self.bind("gguf_get_" + name, result, [C.c_void_p] + [C.c_int64] * len(indices))(ctx, *indices)

    def canonical_value(self, ctx, index, kind):
        if kind == 8:
            value = self.get("val_str", ctx, index, result=C.c_char_p)
            return struct.pack("<Q", len(value)) + value
        if kind == 9:
            element = self.get("arr_type", ctx, index, result=C.c_int)
            count = self.get("arr_n", ctx, index, result=C.c_size_t)
            prefix = struct.pack("<IQ", element, count)
            if element == 8:
                values = []
                fn = self.bind("gguf_get_arr_str", C.c_char_p, [C.c_void_p, C.c_int64, C.c_size_t])
                for position in range(count):
                    value = fn(ctx, index, position)
                    values.append(struct.pack("<Q", len(value)) + value)
                return prefix + b"".join(values)
            if element not in WIDTHS:
                raise RuntimeError("unsupported oracle array")
            data = self.get("arr_data", ctx, index, result=C.c_void_p)
            return prefix + C.string_at(data, count * WIDTHS[element])
        data = self.get("val_data", ctx, index, result=C.c_void_p)
        return C.string_at(data, WIDTHS[kind])

    def inspect(self, path, samples=True):
        path = Path(path)
        ctx = self.load(bytes(path), InitParams(True, None))
        if not ctx:
            raise RuntimeError("reference rejected GGUF: " + str(path))
        try:
            result = {"version": self.get("version", ctx, result=C.c_uint32),
                      "alignment": self.get("alignment", ctx, result=C.c_size_t),
                      "data_offset": self.get("data_offset", ctx, result=C.c_size_t),
                      "file_size": path.stat().st_size, "metadata": [], "tensors": []}
            for index in range(self.get("n_kv", ctx)):
                name = self.get("key", ctx, index, result=C.c_char_p).decode()
                kind = self.get("kv_type", ctx, index, result=C.c_int)
                encoded = self.canonical_value(ctx, index, kind)
                result["metadata"].append({"name": name, "type": kind,
                                           "value_sha256": hashlib.sha256(encoded).hexdigest()})
            with path.open("rb") as source:
                for index in range(self.get("n_tensors", ctx)):
                    name = self.get("tensor_name", ctx, index, result=C.c_char_p).decode()
                    dims = self.get("tensor_ne", ctx, index, result=C.POINTER(C.c_int64))
                    tensor = {"name": name, "type": self.get("tensor_type", ctx, index, result=C.c_int),
                              "dims": list(dims[:4]), "offset": self.get("tensor_offset", ctx, index, result=C.c_size_t),
                              "size": self.get("tensor_size", ctx, index, result=C.c_size_t)}
                    if samples:
                        digest = hashlib.sha256()
                        size = tensor["size"]
                        for start in (0, size // 2, max(0, size - 64)):
                            count = min(64, size - start)
                            source.seek(result["data_offset"] + tensor["offset"] + start)
                            data = source.read(count)
                            if len(data) != count:
                                raise RuntimeError("truncated tensor payload: " + name)
                            digest.update(data)
                        tensor["sample_sha256"] = digest.hexdigest()
                    result["tensors"].append(tensor)
            return result
        finally:
            self.free(ctx)

    def fixture(self, path, alignment=None, tensors=True):
        if alignment is None:
            ctx = self.bind("gguf_init_empty", C.c_void_p, [])()
        else:
            # The metadata setter does not change the writer's internal alignment.
            key = b"general.alignment"
            header = b"GGUF" + struct.pack("<IQQQ", 3, 0, 1, len(key)) + key + struct.pack("<II", 4, alignment)
            initial = C.create_string_buffer(header)
            ctx = self.bind("gguf_init_from_buffer", C.c_void_p, [C.c_void_p, C.c_size_t, InitParams])(
                initial, len(header), InitParams(True, None))
        arena = self.bind("ggml_init", C.c_void_p, [TensorInitParams])(TensorInitParams(256 * 1024, None, True))
        if not ctx or not arena:
            if ctx:
                self.free(ctx)
            if arena:
                self.bind("ggml_free", None, [C.c_void_p])(arena)
            raise RuntimeError("failed to create oracle fixture contexts")
        buffers = []
        try:
            for kind, (suffix, ctype, value) in SCALARS.items():
                self.bind("gguf_set_val_" + suffix, None, [C.c_void_p, C.c_char_p, ctype])(ctx, ("test." + suffix).encode(), value)
                # Numeric arrays: preserve raw element bits, no narrowing through JSON.
                data = (ctype * 3)(value, 0, value)
                self.bind("gguf_set_arr_data", None, [C.c_void_p, C.c_char_p, C.c_int, C.c_void_p, C.c_size_t])(
                    ctx, ("test.array_" + suffix).encode(), kind, data, 3)
            self.bind("gguf_set_val_str", None, [C.c_void_p, C.c_char_p, C.c_char_p])(ctx, b"test.text", "\"héllo\"\n世界".encode())
            array = (C.c_char_p * 3)(b"", "é".encode(), b"a\nb")
            self.bind("gguf_set_arr_str", None, [C.c_void_p, C.c_char_p, C.POINTER(C.c_char_p), C.c_size_t])(ctx, b"test.strings", array, 3)
            self.bind("gguf_set_arr_str", None, [C.c_void_p, C.c_char_p, C.POINTER(C.c_char_p), C.c_size_t])(ctx, b"test.empty", array, 0)
            if tensors:
                block_size = self.bind("ggml_blck_size", C.c_int64, [C.c_int])
                type_size = self.bind("ggml_type_size", C.c_size_t, [C.c_int])
                for kind in (0, 1, 2, 3, 8, 13, 14, 30):
                    shape = [max(3, block_size(kind)), 2]
                    if kind == 1:
                        shape += [2, 2]
                    tensor = self.bind("ggml_new_tensor", C.c_void_p, [C.c_void_p, C.c_int, C.c_int, C.POINTER(C.c_int64)])(
                        arena, kind, len(shape), (C.c_int64 * len(shape))(*shape))
                    name = f"tensor.type_{kind}".encode()
                    self.bind("ggml_set_name", C.c_void_p, [C.c_void_p, C.c_char_p])(tensor, name)
                    self.bind("gguf_add_tensor", None, [C.c_void_p, C.c_void_p])(ctx, tensor)
                    count = 1
                    for extent in shape:
                        count *= extent
                    size = count // block_size(kind) * type_size(kind)
                    data = C.create_string_buffer(bytes((kind * 11 + i * 17) % 256 for i in range(size)))
                    buffers.append(data)
                    self.bind("gguf_set_tensor_data", None, [C.c_void_p, C.c_char_p, C.c_void_p])(ctx, name, data)
            ok = self.bind("gguf_write_to_file", C.c_bool, [C.c_void_p, C.c_char_p, C.c_bool])(ctx, bytes(path), False)
            if not ok:
                raise RuntimeError("reference failed to write fixture")
        finally:
            self.free(ctx)
            self.bind("ggml_free", None, [C.c_void_p])(arena)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, help="default: the source-built ggml (@ggml//:ggml_base_so)")
    sub = parser.add_subparsers(dest="action", required=True)
    fixture = sub.add_parser("fixtures")
    fixture.add_argument("--output", type=Path, required=True)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("model", type=Path)
    inspect.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path, built = library(args.library)
    oracle = Oracle(path)
    provenance = {"oracle": dict(oracle.identity, build=built), "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if args.action == "fixtures":
        args.output.mkdir(parents=True, exist_ok=False)
        cases = []
        for name, alignment, tensors in [("default", None, True), ("aligned64", 64, True), ("metadata", None, False)]:
            path = args.output / (name + ".gguf")
            oracle.fixture(path, alignment, tensors)
            cases.append(dict(oracle.inspect(path), path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        (args.output / "manifest.json").write_text(json.dumps(dict(provenance, cases=cases), indent=2) + "\n")
        print("Generated/reopened", len(cases), "reference fixtures")
    else:
        result = oracle.inspect(args.model)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as output:
            json.dump(dict(provenance, container=result), output, indent=2)
            output.write("\n")
        print("Reference read", len(result["metadata"]), "metadata fields and", len(result["tensors"]), "tensor descriptors/payload samples")


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    main()
