#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""D3 norm-stage IO attribution probe (F2 follow-up).

Lives with the operator so the attribution is reproducible from the repo:
    python run_io_profile.py --phase w --out /tmp/wall.json
    for t in oracle mode0 mode1 mode2 ref_add ref_cast; do
        python run_io_profile.py --phase p --target $t --prof-dir /tmp/prof
    done
    python parse_kernel_pipe_csv.py --prof-dir /tmp/prof --out /tmp/prof.json

Question: the F2 report claims the norm stage runs at ~565GB/s read bandwidth at
(M=2048, K=5120) bf16, derived as bytes/(wall - c) with c taken from the oracle.
That subtraction assumes a fixed host overhead c AND that the oracle's device
time at this shape is exactly 79.70us (the production profiler number). This
probe measures both terms instead of assuming them:

  phase p   torch_npu.profiler capture per target -> kernel_details.csv with the
            per-pipe breakdown (aiv_mte2 / aiv_vec / aiv_mte3 / aiv_scalar
            ratios). This is host-overhead free device time + the pipe occupancy
            that says WHAT the kernel spends its cycles on.
  phase w   wall-clock, two ways: synced after every call (the S1 method) and
            pipelined (no sync in the loop, so per-iteration = max(host,device)).
            Sweeping M gives intercept (fixed cost) and slope (per-row cost).

Each phase-target runs in its own process: the profiler's in-process analyse is
known to segfault on this box after a successful stop (pitfall 7), so the data
must survive a crash of one capture.

The ARMNS_STATS / ARMNS_SUM switches (op_host) are read per call, so an A/B can
be measured inside one session by setting os.environ between phases.
"""
import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
CASES = HERE
sys.path.insert(0, CASES)
sys.path.insert(0, REPO + "/python/ascend_kernel")

import ascend_kernel  # noqa: F401  registers torch.ops.npu.*

M_MAIN, K_MAIN = 2048, 5120          # F2 prefill main shape
M_DECODE = 32
SWEEP_M = [8, 64, 256, 512, 1024, 2048]
WARMUP, ITERS, ROUNDS = 20, 100, 3
PROF_ITERS = 20


def bytes_of(m, k, dtype):
    return m * k * 2  # bf16/fp16 are both 2B


def make(m, k, dtype, seed=1234):
    from add_rms_norm_stats_cases import make_inputs
    return make_inputs(m, k, dtype, seed, has_beta=False)


def targets(m, k, dtype):
    """name -> (fn, read_bytes, write_bytes, note)"""
    import torch
    import torch_npu
    x1, x2, gamma, beta = make(m, k, dtype)
    ones = m * k * 2
    y = torch.empty_like(x1)
    y32 = torch.empty(m, k, dtype=torch.float32, device=x1.device)
    T = {}
    T["oracle"] = (lambda: torch_npu.npu_add_rms_norm(x1, x2, gamma, 1e-6),
                   2 * ones, 2 * ones, "production chain member (y, rstd, residual)")
    for mode in (0, 1, 2):
        rd = ones if mode == 1 else 2 * ones
        wr = ones if mode in (0, 1) else 0
        T[f"mode{mode}"] = (
            lambda mode=mode: torch.ops.npu.add_rms_norm_stats(x1, x2, gamma, beta, 1e-6, mode),
            rd, wr, "our kernel")
    # CANN elementwise references with the same byte count as mode0 (62.9MB)
    T["ref_add"] = (lambda: torch.add(x1, x2, out=y), 2 * ones, ones, "aclnnAdd 2R+1W")
    T["ref_cast"] = (lambda: y32.copy_(x1), ones, 2 * ones, "bf16->fp32 cast 1R+2W")
    T["ref_copy"] = (lambda: y.copy_(x1), ones, ones, "pure copy 1R+1W")
    return T


def bench_synced(fn, iters=ITERS, warmup=WARMUP, rounds=ROUNDS):
    import torch
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    meds = []
    for _ in range(rounds):
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            torch.npu.synchronize()
            ts.append((time.perf_counter() - t0) * 1e6)
        meds.append(statistics.median(ts))
    return meds


def bench_pipelined(fn, iters=ITERS, warmup=WARMUP, rounds=ROUNDS):
    """No sync inside the loop: the host runs ahead and the per-iteration cost is
    max(host per-call cost, device time)."""
    import torch
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    out = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
        out.append((time.perf_counter() - t0) * 1e6 / iters)
    return out


def phase_w(args):
    import torch
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    res = {"op": "d3_norm_io_probe", "phase": "wall", "dtype": args.dtype,
           "env": "910B2 / CANN 9.1.0 / torch_npu 2.13.0rc1", "sweep": {}}
    for m in SWEEP_M:
        tg = targets(m, K_MAIN, dtype)
        row = {}
        for name, (fn, rd, wr, note) in tg.items():
            syn = bench_synced(fn)
            pip = bench_pipelined(fn)
            row[name] = {"synced_medians_us": [round(v, 2) for v in syn],
                         "synced_best_us": round(min(syn), 2),
                         "pipelined_medians_us": [round(v, 3) for v in pip],
                         "pipelined_best_us": round(min(pip), 3),
                         "read_mb": round(rd / 1e6, 2), "write_mb": round(wr / 1e6, 2),
                         "note": note}
        res["sweep"][str(m)] = row
        print(f"  M={m}: " + "  ".join(
            f"{n}={row[n]['synced_best_us']:.1f}/{row[n]['pipelined_best_us']:.1f}"
            for n in ("oracle", "mode0", "mode1", "mode2", "ref_add")), flush=True)
    # decode shape for the same targets
    tg = targets(M_DECODE, K_MAIN, dtype)
    row = {}
    for name, (fn, rd, wr, note) in tg.items():
        row[name] = {"synced_best_us": round(min(bench_synced(fn)), 2),
                     "pipelined_best_us": round(min(bench_pipelined(fn)), 3),
                     "read_mb": round(rd / 1e6, 2), "write_mb": round(wr / 1e6, 2)}
    res["decode_shape"] = row
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print("written", args.out, flush=True)
    return 0


def phase_p(args):
    """One profiler capture per target, so the analyser can never confuse the
    three modes (they share a kernel name)."""
    import torch
    from torch_npu.profiler import (
        AiCMetrics,
        ExportType,
        ProfilerActivity,
        ProfilerLevel,
        _ExperimentalConfig,
        profile,
        tensorboard_trace_handler,
    )
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    m = args.m
    tg = targets(m, K_MAIN, dtype)
    fn = tg[args.target][0]
    out = os.path.join(args.prof_dir, f"{args.target}_{args.dtype}_M{m}")
    os.makedirs(out, exist_ok=True)
    for _ in range(WARMUP):
        fn()
    torch.npu.synchronize()
    cfg = _ExperimentalConfig(export_type=ExportType.Text,
                              profiler_level=ProfilerLevel.Level1,
                              msprof_tx=False, aic_metrics=AiCMetrics.PipeUtilization,
                              l2_cache=False, op_attr=False, data_simplification=True,
                              record_op_args=False, gc_detect_threshold=None)
    p = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
                with_stack=False, with_modules=False, experimental_config=cfg,
                on_trace_ready=tensorboard_trace_handler(out, worker_name=args.target))
    p.start()
    for _ in range(PROF_ITERS):
        fn()
    torch.npu.synchronize()
    p.stop()
    print(f"PROF_STOP_OK {args.target}", flush=True)
    time.sleep(1.0)
    sys.stdout.flush()
    os._exit(0)  # skip the in-process analyse that segfaults on this box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["w", "p"])
    ap.add_argument("--target", default="mode0")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--m", type=int, default=M_MAIN)
    ap.add_argument("--prof-dir", default="/tmp/f2bw/prof")
    ap.add_argument("--out", default="/tmp/f2bw/wall.json")
    args = ap.parse_args()
    return phase_w(args) if args.phase == "w" else phase_p(args)


if __name__ == "__main__":
    sys.exit(main())
