# Copyright (c) 2026 Huawei Technologies Co., Ltd
# BSD 3-Clause License.
#
# LSE flattened-form regression (plan-20260903-lse-flatten-fix P2-1: the
# immunization cases for the staging-bank intra-task race).
#
# Defect being fenced (fixed 2026-09-04, lib md5 f654c2ce): with sub-block rows
# > 32 (LSE_CHUNK_ROWS), chunk 1's staging Brcb overwrote the shared bank while
# chunk 0's async MTE3 copies were still in flight -> timing-dependent LSE
# scramble in the B=1 flattened form (kernel views one request q_len=T).
# Boundary law: sub-block rows = qsB * (qnTile/2 int-div) <= 32 <=> was correct.
# These cases pin the >32-row domain (2+ chunks) and the exact boundary, across
# kv tails, in BOTH call modes (default D2H + fast q_seqlen_value), plus the
# q_len=1 per-request (S1 anchor domain) and B>1 uniform q_len=2 control forms.
#
# Usage: python test_lse_flatten_regression.py <device_id>
import sys
import os
import numpy as np
import torch
import torch_npu
import ascend_kernel  # noqa: F401  (registers torch.ops.npu.fa_fp32_stage1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fa_fp32_ref import gen_inputs, to_bf16_bytes, fa_fp64_ref_tieband

LSE_THRESH = 1e-5     # plan P2-1 gate for the flattened form (floor 9.5e-7)
O_FLOOR = 3e-6        # fp32 accumulation floor of the tie-aware O gate

# (name, T, H, KVH, kv, q_seqlen_value)  -- batch=1, q_len=T (flattened)
FLAT_CASES = [
    ("flat_T64_kv128",   64, 40, 8, 128, 64),    # small kv, tie-band visible
    ("flat_T64_kv4096",  64, 40, 8, 4096, 64),   # 14B main e2e stage-1 shape
    ("flat_T64_kv4356",  64, 40, 8, 4356, 64),   # non-integral kv tail
    ("flat_T8_3b",        8, 16, 2, 4096, 8),    # 3B shape (qnTile=8)
    ("flat_T16_bnd",     16, 40, 8, 4096, 16),   # sub-block rows == 32 (was correct)
    ("flat_T17_bnd",     17, 40, 8, 4096, 17),   # sub-block rows == 34 (was scrambled)
    ("flat_T48",         48, 40, 8, 4096, 48),
    ("flat_T63_odd",     63, 40, 8, 4096, 63),   # odd qsB + qnB=1 tail tiles
]

# (name, batch, kv, H, KVH, q_len)  -- control forms
CTL_CASES = [
    ("ctl_q1_b64",    64, 4096, 40, 8, 1),   # S1 anchor domain (must stay bit-stable)
    ("ctl_qlen2_b4",   4, 4096, 40, 8, 2),   # B>1 uniform q_len=2
    ("ctl_qlen2_3b",   4, 4096, 16, 2, 2),
]


def np_bf16_to_torch(x):
    u16 = np.ascontiguousarray(np.asarray(x)).view(np.uint16)
    return torch.from_numpy(u16).view(torch.bfloat16)


def run_one(name, batch, kv, H, KVH, q_len, seed, dev, fast):
    q, k, v, bt, qs, kvs, T = gen_inputs(batch, [kv] * batch, H, KVH, 128,
                                         seed=seed, q_len=q_len, share_blocks=True)
    args = (
        np_bf16_to_torch(to_bf16_bytes(q)).to(dev),
        np_bf16_to_torch(to_bf16_bytes(k)).to(dev),
        np_bf16_to_torch(to_bf16_bytes(v)).to(dev),
        torch.from_numpy(bt).to(dev),
        torch.from_numpy(qs).to(dev),
        torch.from_numpy(kvs).to(dev),
    )
    out, lse = torch.ops.npu.fa_fp32_stage1(*args, q_len) if fast \
        else torch.ops.npu.fa_fp32_stage1(*args)
    torch.npu.synchronize()
    lse_rows = lse.cpu().numpy().reshape(-1, 8)
    stride_ok = bool(np.array_equal(lse_rows, lse_rows[:, :1].repeat(8, axis=1)))
    lse_flat = lse_rows[:, 0].astype(np.float64)
    o_np = out.cpu().numpy().astype(np.float64)

    o_ref, lse_ref, A = fa_fp64_ref_tieband(
        to_bf16_bytes(q).astype(np.float64), to_bf16_bytes(k).astype(np.float64),
        to_bf16_bytes(v).astype(np.float64), bt, qs, kvs, 128, H, KVH)
    lse_err = float(np.abs(lse_flat - lse_ref.reshape(-1)).max())
    o_abs = np.abs(o_np - o_ref)
    within = o_abs <= (A + O_FLOOR)
    ok = bool(lse_err <= LSE_THRESH and within.all() and stride_ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:18s} mode={'fast' if fast else 'dflt'} "
          f"lse={lse_err:.3e} O maxAbs={float(o_abs.max()):.3e} "
          f"(bound {float((A + O_FLOOR).max()):.1e}) stride8={'ok' if stride_ok else 'BAD'}")
    return ok


def main():
    dev = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    torch.npu.set_device(dev)
    print(f"fa_fp32_stage1 LSE flattened-form regression (device {dev})")
    all_ok = True
    print("flattened B=1 cases (both call modes):")
    for name, T, H, KVH, kv, qlen in FLAT_CASES:
        for fast in (False, True):
            all_ok = run_one(name, 1, kv, H, KVH, qlen, 0, dev, fast) and all_ok
    print("control forms:")
    for name, batch, kv, H, KVH, qlen in CTL_CASES:
        all_ok = run_one(name, batch, kv, H, KVH, qlen, 0, dev, False) and all_ok
    print("RESULT:", "ALL PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
