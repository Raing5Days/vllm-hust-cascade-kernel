"""S1 anchor for the F2 norm stage kernel: wall-clock medians of

  * the production oracle chain member  torch_npu.npu_add_rms_norm  (= the
    CANN `AddRmsNormBias` kernel the profile measured at 79.70us device time),
  * the three add_rms_norm_stats modes (the exposed part of each candidate F2
    fusion topology), and
  * aclnnMatmul (torch.matmul) at the F2 successor-GEMM shapes.

and the pre-registered gate-(2) arithmetic of design.md 4:

    saved_us_per_pair = t_norm_device - t_exposed_device
    projection_prefill(%) = saved_us_per_pair * 96 / 261710 * 100

with t_norm_device = 79.70us (tables/step_ops_prefill.csv: 7651.3us / 96 calls,
add_rms_norm_bias M=2048 K=5120 bf16) and the 261710us prefill-pass device
denominator of REPORT.md 6 (same profile). t_exposed_device is this run's
per-mode wall median minus the dispatch constant c, with c calibrated from the
oracle on the same shape in the same process:

    c = t_wall(oracle) - 79.70us

The formula is an UPPER bound (fusion-side GEMM work and rstd reduction are
accounted as free; no launch/dispatch headroom is credited).

Usage (NPU box, scratch cwd):
    PYTHONPATH=<repo>/python/ascend_kernel python -u run_s1_anchor.py --out <json>
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch_npu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ascend_kernel  # noqa: F401
from add_rms_norm_stats_cases import DEV, make_inputs

HERE = os.path.dirname(os.path.abspath(__file__))
NORM_DEVICE_US = 79.70          # production profile, M=2048 K=5120 bf16
PREFILL_PASS_DEVICE_US = 261710.0   # production profile, 2048-token chunk
N_NORM_CALLS = 96              # 48 layers x 2 norms, production profile
S1_SHAPES = {"prefill-main": (2048, 5120), "decode": (32, 5120)}
WARMUP, ITERS, ROUNDS = 20, 100, 3


def bench(fn):
    for _ in range(WARMUP):
        fn()
    torch.npu.synchronize()
    meds = []
    for _ in range(ROUNDS):
        ts = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            fn()
            torch.npu.synchronize()
            ts.append((time.perf_counter() - t0) * 1e6)
        meds.append(statistics.median(ts))
    return meds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "add_rms_norm_stats_s1_anchor.json"))
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    args = ap.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    res = {
        "op": "add_rms_norm_stats",
        "dtype": args.dtype,
        "date": "2026-09-12",
        "env": "910B2 / CANN 9.1.0 / torch_npu 2.13.0rc1",
        "shapes": {},
    }
    for name, (m, k) in S1_SHAPES.items():
        x1, x2, gamma, beta = make_inputs(m, k, dtype, 1234, has_beta=False)
        b = {}
        b["oracle_npu_add_rms_norm"] = bench(
            lambda a=x1, c=x2, d=gamma: torch_npu.npu_add_rms_norm(a, c, d, 1e-6))
        for mode in (0, 1, 2):
            b[f"mode{mode}"] = bench(
                lambda mode=mode, a=x1, c=x2, d=gamma, e=beta:
                torch.ops.npu.add_rms_norm_stats(a, c, d, e, 1e-6, mode))
        # successor GEMM shapes of the F2 hotspot: qkv N=7168 (K=5120), o_proj/gate_up K=5120
        w_qkv = torch.randn(k, 7168, device=DEV, dtype=dtype)
        w_o = torch.randn(k, k, device=DEV, dtype=dtype)
        b["matmul_N7168"] = bench(lambda a=x1, ww=w_qkv: torch.matmul(a, ww))
        b["matmul_N5120"] = bench(lambda a=x1, ww=w_o: torch.matmul(a, ww))
        res["shapes"][name] = {kk: {"medians_us": [round(v, 2) for v in vv],
                                    "best_us": round(min(vv), 2)} for kk, vv in b.items()}

    # pre-registered gate-(2) arithmetic (prefill main shape only)
    pre = res["shapes"]["prefill-main"]
    c_dispatch = pre["oracle_npu_add_rms_norm"]["best_us"] - NORM_DEVICE_US
    res["dispatch_calibration_us"] = round(c_dispatch, 2)
    proj = {}
    for mode, label in ((0, "A_successor_prologue"), (1, "B_apply"), (2, "B_stats_only")):
        t_wall = pre[f"mode{mode}"]["best_us"]
        t_dev = t_wall - c_dispatch
        saved = NORM_DEVICE_US - t_dev
        proj[label] = {
            "t_exposed_wall_us": t_wall,
            "t_exposed_device_us": round(t_dev, 2),
            "saved_us_per_pair": round(saved, 2),
            "projection_prefill_pct": round(saved * N_NORM_CALLS / PREFILL_PASS_DEVICE_US * 100.0, 3),
        }
    res["gate2_projection"] = proj
    res["formula"] = ("projection_pct = (79.70 - t_exposed_device) * 96 / 261710 * 100  "
                      "[upper bound: GEMM-side fusion work accounted as free]")
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
