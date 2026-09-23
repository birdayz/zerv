import hashlib
import json
from pathlib import Path
import struct
import unittest

ROOT = Path(__file__).resolve().parents[1]


def sha(data): return hashlib.sha256(data).hexdigest()


class ModelArtifactsTests(unittest.TestCase):
    def test_owned_model_shader_manifest_and_capabilities(self):
        directory = ROOT/"src/model/shaders"
        manifest = json.loads((directory/"manifest.json").read_text())
        self.assertEqual(manifest["source_sha256"], sha((ROOT/"src/model/model.comp").read_bytes()))
        self.assertEqual(manifest["gemm_source_sha256"], sha((ROOT/"src/model/gemm.comp").read_bytes()))
        self.assertEqual(manifest["gemm_f16_source_sha256"], sha((ROOT/"src/model/gemm_f16.comp").read_bytes()))
        self.assertEqual(manifest["gemm_f16x_source_sha256"], sha((ROOT/"src/model/gemm_f16x.comp").read_bytes()))
        self.assertEqual(set(manifest["modules"]), {"embed", "norm", "qkprep", "conv", "delta", "swiglu", "zero", "reduce",
                                                    "embed_b", "qk_b", "softmax", "gate", "conv_b", "delta_b",
                                                    "attn_scores", "attn_pv", "attn_combine", "gnorm_b"} |
                         {f"gemm_{f}" for f in ("f32_k", "f32_m", "q4_0", "q4_1", "q5_k", "q6_k")} |
                         {f"gemm_{f}_w" for f in ("q4_0", "q4_1", "q5_k", "q6_k")} |
                         {f"gemm_f16_{f}" for f in ("q4_0", "q4_1", "q5_k")} | {"gemm_f16x_q4_0", "norm_h", "swiglu_h", "gate_h"})
        for name, record in manifest["modules"].items():
            raw = (directory/(name+".spv")).read_bytes()
            self.assertEqual(sha(raw), record["sha256"])
            words = struct.unpack("<"+"I"*(len(raw)//4), raw)
            self.assertEqual(words[0], 0x07230203)
            pos, capabilities = 5, []
            while pos < len(words):
                count, opcode = words[pos] >> 16, words[pos] & 65535
                self.assertGreater(count, 0)
                if opcode == 17: capabilities.append(words[pos+1])
                pos += count
            # Shader only; the scalar-X GEMMs also use GroupNonUniform (61) + Ballot (64) for
            # subgroupBroadcastFirst (host requires compute ballot, subgroup size <= 64).
            # gemm_f16_*: Float16, GroupNonUniform (gl_SubgroupID), VulkanMemoryModel and
            # CooperativeMatrixKHR (block 14; the host enables them only in the f16 mode).
            # gemm_f16x_*: the same (block 16b). The *_h producers: StorageBuffer16BitAccess
            # (4433) for their f16 copy (f16 mode only).
            expected = ([1, 9, 61, 5345, 6022] if name.startswith(("gemm_f16_", "gemm_f16x_")) else [1, 61, 64] if name.startswith("gemm_")
                        else [1, 4433] if name.endswith("_h") else [1])
            self.assertEqual(sorted(capabilities), expected, name)

    def test_oracle_fixture_provenance_and_self_consistency(self):
        for path, names in (("tests/fixtures/model/qwen38-oracle.json", ["short-nothink", "long-think"]),
                            ("tests/fixtures/model/qwen38-oracle-long.json", ["long-prefill"])):
            with self.subTest(fixture=path):
                self.check_oracle_fixture(json.loads((ROOT/path).read_text()), names)

    def check_oracle_fixture(self, fixture, names):
        for name, digest in fixture["sources"].items():
            self.assertEqual(sha((ROOT/"tests/reference"/name).read_bytes()), digest, name)
        self.assertEqual(fixture["model_sha256"], "ede16c7b36e578ca87a8c70e011e4b4633a32c831c0ce76d0f474582384e671d")
        self.assertEqual([c["name"] for c in fixture["cases"]], names)
        for case in fixture["cases"]:
            self.assertEqual(sha(case["prompt"].encode()), case["prompt_sha256"])
            self.assertEqual(len(case["tokens"]), case["n_prompt"] + case["generate"])
            self.assertEqual(len(case["positions"]), len(case["tokens"]))
            self.assertEqual(case["argmax_disagreements"], [])
            for i, position in enumerate(case["positions"]):
                self.assertEqual(position["position"], i)
                if i + 1 < len(case["tokens"]) and i + 1 >= case["n_prompt"]:
                    # The recorded continuation is llama's greedy choice.
                    self.assertEqual(case["tokens"][i+1], position["llama_argmax"])
            for base, record in case["llama_vs_fp64"].items():
                self.assertLess(record["normalized_l2"], 1e-3, base)


if __name__ == "__main__": unittest.main()
