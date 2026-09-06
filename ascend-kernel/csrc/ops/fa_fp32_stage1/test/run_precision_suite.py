# Copyright (c) 2026 Huawei Technologies Co., Ltd
# BSD 3-Clause License.
#
# Precision suite for fa_fp32_stage1 (plan-20260903 M-B step 2, S3):
# 15 shapes x 2 seeds = 30 cases vs the fp64 semantic reference, plus
# op_host TORCH_CHECK boundary cases and the padded-LSE stride invariant.
# Usage: python run_precision_suite.py <device_id> [--quick]

import sys
import os
import time
import numpy as np
import torch
import torch_npu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fa_fp32_ref import gen_inputs, to_bf16_bytes, fa_fp64_ref_tieband

CASES = [
    # (name, batch, kv_seqlens, H, KVH, q_len, block_cols_extra)
    ("main_14b_b64",      64, [8192] * 64, 40, 8, 1, 0),
    ("probe_shape_b4",    4,  [4096] * 4,  40, 8, 1, 0),
    ("b16_h32",           16, [4096] * 16, 32, 8, 1, 0),
    ("b8_h8_kvh1",        8,  [2048] * 8,  8,  1, 1, 0),
    ("b32_kv8192",        32, [8192] * 32, 40, 8, 1, 0),
    ("min_b1_kv128",      1,  [128],       40, 8, 1, 0),
    ("kv_448_tail",       3,  [448] * 3,    40, 8, 1, 0),
    ("varlen_3req",       3,  [129, 640, 1024], 40, 8, 1, 0),
    ("qlen2_b4",          4,  [2048] * 4,   40, 8, 2, 0),
    ("b7_kv4096",         7,  [4096] * 7,   40, 8, 1, 0),
    ("b2_kv16384",        2,  [16384] * 2,  40, 8, 1, 0),
    ("b64_kv128",         64, [128] * 64,   40, 8, 1, 0),
    ("h2_kvh1",           4,  [512] * 4,    2,  1, 1, 0),
    ("padded_block_cols", 4,  [4096] * 4,   40, 8, 1, 3),
    ("varlen_6req",       6,  [4095, 2048, 8192, 512, 128, 4096], 40, 8, 1, 0),
]

SEEDS = [0, 1]


def np_bf16_to_torch(x):
    u16 = np.ascontiguousarray(np.asarray(x)).view(np.uint16)
    return torch.from_numpy(u16).view(torch.bfloat16)


def run_case(name, batch, kv_seqlens, H, KVH, q_len, cols_extra, seed, dev, fast=False):
    q, k, v, bt, qs, kvs, T = gen_inputs(batch, kv_seqlens, H, KVH, 128, seed=seed,
                                         q_len=q_len, block_cols=None if cols_extra == 0
                                         else (max((kv + 127) // 128 for kv in kv_seqlens) + cols_extra),
                                         share_blocks=True)
    q16, k16, v16 = to_bf16_bytes(q), to_bf16_bytes(k), to_bf16_bytes(v)

    args = (
        np_bf16_to_torch(q16).to(dev),
        np_bf16_to_torch(k16).to(dev),
        np_bf16_to_torch(v16).to(dev),
        torch.from_numpy(bt).to(dev),
        torch.from_numpy(qs).to(dev),
        torch.from_numpy(kvs).to(dev),
    )
    # --fast: uniform-q host hint (plan-20260903 M-C C0) - q_seqlen_value=q_len
    # (all requests share that q seqlen by construction in gen_inputs); skips
    # the op-host D2H pulls. Note the fast path degrades the
    # ceil(maxKv/128)<=cols host check to a caller contract, so the "cols too
    # small" negative case only applies to the default mode.
    out, lse = torch.ops.npu.fa_fp32_stage1(*args, q_len) if fast \
        else torch.ops.npu.fa_fp32_stage1(*args)
    torch.npu.synchronize()
    out_np = out.cpu().numpy()
    lse_np = lse.cpu().numpy()

    # padded-LSE stride invariant: all 8 copies per (t,h) row identical
    lse_rows = lse_np.reshape(-1, 8)
    stride_ok = bool(np.array_equal(lse_rows, lse_rows[:, :1].repeat(8, axis=1)))
    lse_compact = lse_rows[:, 0].reshape(T, H)

    o_ref, lse_ref, A = fa_fp64_ref_tieband(q16.astype(np.float64), k16.astype(np.float64),
                                            v16.astype(np.float64), bt, qs, kvs, 128, H, KVH)
    o_abs = np.abs(out_np.astype(np.float64) - o_ref)
    lse_abs = np.abs(lse_compact.astype(np.float64) - lse_ref)
    o_max, lse_max = float(o_abs.max()), float(lse_abs.max())
    # Tie-aware gate: |O - O_ref| <= A (bf16 near-tie rounding ambiguity, numerator
    # space, already scaled by gl) + fp32 accumulation floor. Elements beyond the
    # raw 1e-5 but within A are flip-attributed; beyond A+floor is a real failure.
    floor = 3e-6
    within = o_abs <= (A + floor)
    n_flip = int(((o_abs > 1e-5) & within).sum())
    n_real = int((~within).sum())
    ok = bool(within.all()) and lse_max <= 1e-3 and stride_ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name:18s} seed{seed}: O maxAbs={o_max:.3e} "
          f"(bound {float((A + floor).max()):.1e}) LSE maxAbs={lse_max:.3e} "
          f"stride8={'ok' if stride_ok else 'BAD'} flip-attributed={n_flip} real-fail={n_real}")
    return ok


def run_negative_checks(dev):
    """op_host TORCH_CHECK boundary cases (expect clean rejections)."""
    import ml_dtypes
    negative = []
    q, k, v, bt, qs, kvs, T = gen_inputs(4, [4096] * 4, 40, 8, 128, seed=0, share_blocks=True)

    def npu_args(q16, k16, v16, bt, qs, kvs):
        return (
            np_bf16_to_torch(to_bf16_bytes(q16)).to(dev),
            np_bf16_to_torch(to_bf16_bytes(k16)).to(dev),
            np_bf16_to_torch(to_bf16_bytes(v16)).to(dev),
            torch.from_numpy(bt).to(dev),
            torch.from_numpy(np.asarray(qs, dtype=np.int64)).to(dev),
            torch.from_numpy(np.asarray(kvs, dtype=np.int64)).to(dev),
        )

    # 1. T != sum(q_seqlens): drop one token row
    args = npu_args(q[:-1], k, v, bt, qs, kvs)
    # q[:-1] has 3 tokens but seqlens sum to 4 -> reject
    negative.append(("T mismatch", args))
    # 2. blockSize != 128 is hard to fabricate cheaply via gen (key shape fixed);
    #    emulate with kv seqlen exceeding block_table cols
    bt_small = bt[:, :1]  # cols=1 < ceil(4096/128)
    args = npu_args(q, k, v, bt_small, qs, kvs)
    negative.append(("cols too small", args))
    # 3. kv_seqlen = 0
    args = npu_args(q, k, v, bt, [1, 1, 1, 1], [4096, 4096, 4096, 0])
    negative.append(("kv seqlen 0", args))

    ok_all = True
    for name, args in negative:
        try:
            torch.ops.npu.fa_fp32_stage1(*args)
            torch.npu.synchronize()
            print(f"[FAIL] negative '{name}': accepted (should reject)")
            ok_all = False
        except RuntimeError as e:
            msg = str(e)
            if "fa_fp32_stage1" in msg:
                print(f"[PASS] negative '{name}': rejected ({msg.splitlines()[0][:80]})")
            else:
                print(f"[WARN] negative '{name}': rejected with unexpected error: {msg.splitlines()[0][:80]}")
    return ok_all


def main():
    dev_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    quick = "--quick" in sys.argv
    fast = "--fast" in sys.argv
    torch.npu.set_device(dev_id)
    import ascend_kernel

    cases = CASES if not quick else CASES[:4]
    n_pass = 0
    n_total = 0
    t0 = time.time()
    for name, batch, kv_seqlens, H, KVH, q_len, cols_extra in cases:
        for seed in SEEDS:
            n_total += 1
            if run_case(name, batch, kv_seqlens, H, KVH, q_len, cols_extra, seed, "npu", fast=fast):
                n_pass += 1
    print(f"\nprecision cases ({'fast path' if fast else 'default path'}): {n_pass}/{n_total} PASS ({time.time() - t0:.0f}s)")
    neg_ok = True if fast else run_negative_checks("npu")
    ok = (n_pass == n_total) and neg_ok
    print("SUITE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
