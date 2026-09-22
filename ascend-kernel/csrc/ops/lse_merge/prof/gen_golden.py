"""Generate bit-exact golden anchors for the "v1 装机版" ascend-kernel lse_merge.

Protective read-only work: this script only reads the installed operator and
writes golden tensors under profiles/lse-merge-pipe-20260918/golden/. It does
not modify operator sources, rebuild, or touch site-packages.

Case matrix and input construction are reused verbatim from the existing test
harness (lse_merge_cases.py + test_lse_merge_precision.py seed derivation) so
the golden is directly comparable with the precision suite.

Run on an idle NPU, e.g.:
    cd /tmp && ASCEND_RT_VISIBLE_DEVICES=<card> \
        /opt/miniconda3/envs/hust/bin/python \
        /vllm-workspace/ops/kernels/ascend-kernel/csrc/ops/lse_merge/prof/gen_golden.py
"""
import hashlib
import json
import os
import sys
import zlib

# import order matters: importing ascend_kernel registers torch.ops.npu.lse_merge
import ascend_kernel  # noqa: F401
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_TEST_DIR = os.path.join(os.path.dirname(_HERE), "test")
sys.path.insert(0, _TEST_DIR)
from lse_merge_cases import MODES, SHAPES, make_inputs

GOLDEN_DIR = "/vllm-workspace/profiles/lse-merge-pipe-20260918/golden"


def _sha256_of_tensor(t):
    """sha256 of the raw bytes of a CPU contiguous uint8 view of t."""
    u8 = t.cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(u8).hexdigest()


def main():
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    manifest = {}
    failures = []

    for mode in MODES:
        o1_dtype, _, out_code, padded, _, _, _ = MODES[mode]
        for cat, (T, H, D) in SHAPES:
            key = f"{mode}__{cat}"
            try:
                seed = zlib.crc32(f"{cat}/{mode}".encode()) % 2**31
                o1, o2, l1c, l2c, _, _ = make_inputs(
                    T, H, D, o1_dtype, padded, seed=seed
                )
                out = torch.ops.npu.lse_merge(o1, o2, l1c, l2c, out_code)
                torch.npu.synchronize()
                out_cpu = out.cpu()

                fname = f"{key}.pt"
                fpath = os.path.join(GOLDEN_DIR, fname)
                torch.save(out_cpu, fpath)

                digest = _sha256_of_tensor(out_cpu)
                manifest[key] = {
                    "dtype": str(out_cpu.dtype).replace("torch.", ""),
                    "shape": list(out_cpu.shape),
                    "sha256": digest,
                    "file": fname,
                }
                print(f"[OK] {key}: dtype={manifest[key]['dtype']} "
                      f"shape={manifest[key]['shape']} sha256={digest}")
            except Exception as exc:  # noqa: BLE001
                failures.append({"key": key, "error": repr(exc)})
                print(f"[FAIL] {key}: {exc!r}")

    manifest_path = os.path.join(GOLDEN_DIR, "MANIFEST.json")
    payload = {
        "generator": "gen_golden.py",
        "operator": "torch.ops.npu.lse_merge",
        "kernel_baseline": "v1-ce8bf74f (libascend_kernel.so md5 ce8bf74faabaa622aafdd5da15d8d262)",
        "sha256_source": "sha256(out.cpu().contiguous().view(torch.uint8).numpy().tobytes())",
        "total_cases": len(MODES) * len(SHAPES),
        "ok_cases": len(manifest),
        "failures": failures,
        "cases": manifest,
    }
    with open(manifest_path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(f"manifest -> {manifest_path} "
          f"({len(manifest)}/{len(MODES) * len(SHAPES)} cases, "
          f"{len(failures)} failures)")
    print("GOLDEN_DONE")


if __name__ == "__main__":
    main()
