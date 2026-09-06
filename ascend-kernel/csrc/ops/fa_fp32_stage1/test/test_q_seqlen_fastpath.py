# Copyright (c) 2026 Huawei Technologies Co., Ltd
# BSD 3-Clause License.
#
# plan-20260903 M-C C0-3 regression gate, fast-path legs:
#  F1: q_seqlen_value=1 output is BIT-IDENTICAL to the default (D2H) path
#      across the same shapes/seeds the precision suite uses.
#  F2: fast path vs the fp64 semantic reference (tie-band gate) on a subset.
#  F3: q_seqlen_value=2 with T != 2*B must be rejected (TORCH_CHECK).
#  F4: 10x bit-level stability on the fast path.
# Usage: python test_q_seqlen_fastpath.py <device_id>

import sys
import os
import numpy as np
import torch
import torch_npu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fa_fp32_ref import gen_inputs, to_bf16_bytes, fa_fp64_ref_tieband

SHAPES = [
    # (name, batch, kv_seqlens, H, KVH, block_cols_extra)
    ("main_14b_b64",      64, [8192] * 64, 40, 8, 0),
    ("probe_shape_b4",    4,  [4096] * 4,  40, 8, 0),
    ("b16_h32",           16, [4096] * 16, 32, 8, 0),
    ("min_b1_kv128",      1,  [128],       40, 8, 0),
    ("kv_448_tail",       3,  [448] * 3,    40, 8, 0),
    ("b64_kv128",         64, [128] * 64,   40, 8, 0),
    ("padded_block_cols", 4,  [4096] * 4,   40, 8, 3),
    ("varlen_kv",         6,  [4095, 2048, 8192, 512, 128, 4096], 40, 8, 0),
]


def np_bf16_to_torch(x):
    u16 = np.ascontiguousarray(np.asarray(x)).view(np.uint16)
    return torch.from_numpy(u16).view(torch.bfloat16)


def to_npu(q16, k16, v16, bt, qs, kvs, dev):
    return (
        np_bf16_to_torch(q16).to(dev),
        np_bf16_to_torch(k16).to(dev),
        np_bf16_to_torch(v16).to(dev),
        torch.from_numpy(bt).to(dev),
        torch.from_numpy(np.asarray(qs, dtype=np.int64)).to(dev),
        torch.from_numpy(np.asarray(kvs, dtype=np.int64)).to(dev),
    )


def main():
    dev_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    torch.npu.set_device(dev_id)
    import ascend_kernel

    dev = "npu"
    ok_all = True

    for name, batch, kv_seqlens, H, KVH, cols_extra in SHAPES:
        q, k, v, bt, qs, kvs, T = gen_inputs(
            batch, kv_seqlens, H, KVH, 128, seed=0, q_len=1,
            block_cols=None if cols_extra == 0
            else (max((kv + 127) // 128 for kv in kv_seqlens) + cols_extra),
            share_blocks=True)
        q16, k16, v16 = to_bf16_bytes(q), to_bf16_bytes(k), to_bf16_bytes(v)
        args = to_npu(q16, k16, v16, bt, qs, kvs, dev)

        out_d, lse_d = torch.ops.npu.fa_fp32_stage1(*args)
        out_f, lse_f = torch.ops.npu.fa_fp32_stage1(*args, 1)  # fast path
        torch.npu.synchronize()

        o_d, o_f = out_d.cpu().numpy(), out_f.cpu().numpy()
        l_d = lse_d.cpu().numpy().reshape(-1, 8)[:, 0]
        l_f = lse_f.cpu().numpy().reshape(-1, 8)[:, 0]
        bit_ok = bool(np.array_equal(o_d.view(np.uint32), o_f.view(np.uint32))
                      and np.array_equal(l_d.view(np.uint32), l_f.view(np.uint32)))

        # fp64 reference on every shape (cheap enough at these sizes)
        o_ref, lse_ref, A = fa_fp64_ref_tieband(
            q16.astype(np.float64), k16.astype(np.float64), v16.astype(np.float64),
            bt, qs, kvs, 128, H, KVH)
        o_abs = np.abs(o_f.astype(np.float64) - o_ref)
        lse_abs = np.abs(l_f.astype(np.float64).reshape(T, H) - lse_ref)
        within = o_abs <= (A + 3e-6)
        real_fail = int((~within).sum())
        ref_ok = bool(within.all()) and float(lse_abs.max()) <= 1e-3

        ok = bit_ok and ref_ok
        ok_all = ok_all and ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name:18s}: fast-vs-default bit={bit_ok} "
              f"O maxAbs={float(o_abs.max()):.3e} LSE maxAbs={float(lse_abs.max()):.3e} "
              f"real-fail={real_fail}")

    # F3: negative - q_seqlen_value=2 but T != 2*B
    q, k, v, bt, qs, kvs, T = gen_inputs(4, [4096] * 4, 40, 8, 128, seed=0, share_blocks=True)
    q16, k16, v16 = to_bf16_bytes(q), to_bf16_bytes(k), to_bf16_bytes(v)
    args = to_npu(q16, k16, v16, bt, qs, kvs, dev)
    try:
        torch.ops.npu.fa_fp32_stage1(*args, 2)
        torch.npu.synchronize()
        print("[FAIL] negative q_seqlen_value=2 (T=4,B=4): accepted (should reject)")
        ok_all = False
    except RuntimeError as e:
        msg = str(e)
        tag = "PASS" if "q_seqlen_value" in msg else "WARN-unexpected-error"
        print(f"[{tag}] negative q_seqlen_value=2: rejected ({msg.splitlines()[0][:80]})")
        ok_all = ok_all and (tag == "PASS")

    # F4: 10x bit-level stability on the fast path (main 14B shape)
    q, k, v, bt, qs, kvs, T = gen_inputs(64, [8192] * 64, 40, 8, 128, seed=3, share_blocks=True)
    q16, k16, v16 = to_bf16_bytes(q), to_bf16_bytes(k), to_bf16_bytes(v)
    args = to_npu(q16, k16, v16, bt, qs, kvs, dev)
    first = None
    stable = True
    for i in range(10):
        o_i, l_i = torch.ops.npu.fa_fp32_stage1(*args, 1)
        torch.npu.synchronize()
        b = (o_i.cpu().numpy().tobytes(), l_i.cpu().numpy().tobytes())
        if first is None:
            first = b
        elif b != first:
            stable = False
            print(f"  run {i} differs")
    print(f"[{'PASS' if stable else 'FAIL'}] F4 fast-path stability 10/10")
    ok_all = ok_all and stable

    print("FASTPATH SUITE:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
