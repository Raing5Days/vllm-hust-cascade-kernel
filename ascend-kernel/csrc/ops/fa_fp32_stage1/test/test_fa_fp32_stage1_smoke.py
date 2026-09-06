# Copyright (c) 2026 Huawei Technologies Co., Ltd
# BSD 3-Clause License.
#
# Smoke test for torch.ops.npu.fa_fp32_stage1 (plan-20260903 M-B step 2):
#  S1 port equivalence: bit-compare against the example binary dump
#     (/tmp/mb2_example_dump[.lse], same example data inputs).
#  S3 fp64 reference: O maxAbs <= 1e-5, LSE <= 1e-3.
#  S4 stability: 10 repeated runs bit-identical.
# Usage: python test_fa_fp32_stage1_smoke.py <device_id>

import sys
import os
import numpy as np
import torch
import torch_npu
import ml_dtypes

DATA = "/vllm-workspace/cascade-merge-op/catlass-example-data"
DUMP_O = "/tmp/mb2_example_dump"
DUMP_LSE = "/tmp/mb2_example_dump.lse"


def load_example_inputs():
    q_ntokens = int(np.fromfile(f"{DATA}/q_ntokens.bin", dtype=np.int32)[0])
    q_seqlens = np.fromfile(f"{DATA}/q_seqlen.bin", dtype=np.int64)
    kv_seqlens = np.fromfile(f"{DATA}/kv_seqlen.bin", dtype=np.int64)
    batch = len(q_seqlens)
    num_tokens = q_ntokens
    num_heads, kv_heads, embed = 40, 8, 128
    max_kv = int(kv_seqlens.max())
    cols = (max_kv + 127) // 128
    bt = np.fromfile(f"{DATA}/block_table.bin", dtype=np.int32).reshape(batch, cols)
    q = np.fromfile(f"{DATA}/q.bin", dtype=ml_dtypes.bfloat16)
    kv_per_block = 128
    num_blocks = int(bt.max()) + 1
    kv_shape = (num_blocks, kv_per_block, kv_heads, embed)
    k = np.fromfile(f"{DATA}/k.bin", dtype=ml_dtypes.bfloat16)[: np.prod(kv_shape)].reshape(kv_shape)
    v = np.fromfile(f"{DATA}/v.bin", dtype=ml_dtypes.bfloat16)[: np.prod(kv_shape)].reshape(kv_shape)
    q = q[: num_tokens * num_heads * embed].reshape(num_tokens, num_heads, embed)
    return q, k, v, bt, q_seqlens, kv_seqlens


def np_bf16_to_torch(x):
    """ml_dtypes.bfloat16 ndarray -> torch bf16 tensor (bit view)."""
    u16 = np.ascontiguousarray(np.asarray(x)).view(np.uint16)
    return torch.from_numpy(u16).view(torch.bfloat16)


def to_npu(q, k, v, bt, qs, kvs):
    dev = "npu"
    return (
        np_bf16_to_torch(q).to(dev),
        np_bf16_to_torch(k).to(dev),
        np_bf16_to_torch(v).to(dev),
        torch.from_numpy(np.asarray(bt)).to(dev),
        torch.from_numpy(np.asarray(qs)).to(dev),
        torch.from_numpy(np.asarray(kvs)).to(dev),
    )


def main():
    dev_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    torch.npu.set_device(dev_id)
    import ascend_kernel  # registers torch.ops.npu.*

    q, k, v, bt, qs, kvs = load_example_inputs()
    T, H, D = q.shape
    args = to_npu(q, k, v, bt, qs, kvs)

    out, lse = torch.ops.npu.fa_fp32_stage1(*args)
    torch.npu.synchronize()
    assert out.shape == (T, H, D) and out.dtype == torch.float32, (out.shape, out.dtype)
    assert lse.shape == (T * H * 8,) and lse.dtype == torch.float32

    out_np = out.cpu().numpy()
    lse_np = lse.cpu().numpy().reshape(T * H, 8)[:, 0].reshape(T, H)

    # S1: bit equality vs the example binary dump
    ref_o = np.fromfile(DUMP_O, dtype=np.float32).reshape(T, H, D)
    ref_lse = np.fromfile(DUMP_LSE, dtype=np.float32).reshape(T * H, 8)[:, 0].reshape(T, H)
    o_bits = np.array_equal(out_np.view(np.uint32), ref_o.view(np.uint32))
    lse_bits = np.array_equal(lse_np.view(np.uint32), ref_lse.view(np.uint32))
    print(f"S1 bit-equality vs example binary: O={o_bits} LSE={lse_bits}")
    if not o_bits:
        diff = np.abs(out_np.astype(np.float64) - ref_o.astype(np.float64))
        print(f"  O maxAbsDiff vs example dump = {diff.max():.3e}")
    if not lse_bits:
        diff = np.abs(lse_np.astype(np.float64) - ref_lse.astype(np.float64))
        print(f"  LSE maxAbsDiff vs example dump = {diff.max():.3e}")

    # S3: fp64 semantic reference
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fa_fp32_ref import fa_fp64_ref

    o_ref, lse_ref = fa_fp64_ref(q, k, v, bt, qs, kvs, D, H, k.shape[2])
    o_abs = np.abs(out_np.astype(np.float64) - o_ref)
    lse_abs = np.abs(lse_np.astype(np.float64) - lse_ref)
    print(f"S3 vs fp64: O maxAbs={o_abs.max():.3e} (<=1e-5), LSE maxAbs={lse_abs.max():.3e} (<=1e-3)")
    o_rel = o_abs / np.maximum(np.abs(o_ref), 1e-2)
    print(f"   O rel(|ref|>=1e-2 domain) max={o_rel.max():.3e}")

    # S4: stability 10 runs bit-identical
    first = None
    stable = True
    for i in range(10):
        o_i, l_i = torch.ops.npu.fa_fp32_stage1(*args)
        torch.npu.synchronize()
        b = (o_i.cpu().numpy().tobytes(), l_i.cpu().numpy().tobytes())
        if first is None:
            first = b
        elif b != first:
            stable = False
            print(f"  run {i} differs")
    print(f"S4 stability 10/10: {'PASS' if stable else 'FAIL'}")

    ok = o_bits and lse_bits and o_abs.max() <= 1e-5 and lse_abs.max() <= 1e-3 and stable
    print("SMOKE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
