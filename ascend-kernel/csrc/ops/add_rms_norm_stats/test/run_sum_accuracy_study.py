#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU study: how much of the mode-1 `y` defect does the pairwise row sum fix?

The defect (F2 raw f2_mode1_y_defect_evidence.md) is located in the row sum of
squares: our rstd is good to ~3.3e-6 relative where the golden reaches ~6.1e-8,
and that difference moves round_dtype(x*rstd) across a dtype rounding boundary
for ~3e-4 of the elements. Where the sum's error comes from is an accumulation
-depth question, which is a pure-float32 question and can therefore be answered
on CPU, without the NPU, before deciding what to implement.

Models, all in fp32 over the same bf16-rounded input x = round_bf16(N(0, 0.5)):

  seq        x^2 accumulated left to right, 5120 steps        (error grows with n)
  partial80  x^2 summed in 80 chunks of 64, then the 80 chunk partials combined
             left to right - an idealised model of the Level-2 ReduceSum shape
             that the kernel used before this change
  pairwise   x^2 folded pairwise down to 64 lanes, then those 64 combined left
             to right - exactly the shape the kernel computes now

Reference is fp64. Reported: relative error of the row sum, relative error of
rstd, and the element-level cost that actually matters - how often the
application of rstd to x crosses a bf16 rounding boundary in a way the golden
would not, per million row*K elements.

CPU only: no NPU, no torch_npu. Run with -q from the test directory or pytest.
"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROWS = 512
K = 5120
EPS = 1e-6
SEED = 20260912


def round_bf16(v):
    return v.to(torch.bfloat16).to(torch.float32)


def sum_seq(t):
    """Left-to-right fp32 accumulation (torch.sum on CPU is pairwise, so the
    accumulation is written out explicitly)."""
    acc = torch.zeros(t.shape[0], dtype=torch.float32)
    for i in range(t.shape[1]):
        acc = acc + t[:, i]
    return acc


def sum_partial80(sq):
    """80 chunks of 64, chunk sums combined left to right."""
    chunks = sq.view(sq.shape[0], 80, 64).sum(dim=2)  # chunk-internal: torch tree
    acc = torch.zeros_like(chunks[:, 0])
    for i in range(chunks.shape[1]):
        acc = acc + chunks[:, i]
    return acc


def sum_pairwise(sq):
    """Pairwise (dichotomy) fold to 64 lanes + a left-to-right combine of 64."""
    w = sq.clone()
    n = w.shape[1]
    p2 = 1
    while p2 * 2 <= n:
        p2 *= 2
    if n > p2:
        w[:, : n - p2] = w[:, : n - p2] + w[:, p2:n]  # fold the tail
        w = w[:, :p2]
    while p2 > 64:
        p2 //= 2
        w[:, :p2] = w[:, :p2] + w[:, p2 : 2 * p2]
    acc = torch.zeros(w.shape[0], dtype=torch.float32)
    for i in range(64):
        acc = acc + w[:, i]
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    ap.add_argument("--out", default=os.path.join(HERE, "sum_accuracy_study.json"))
    args = ap.parse_args()

    g = torch.Generator().manual_seed(SEED)
    rows = max(1, args.rows)
    x = round_bf16(torch.randn(rows, K, generator=g, dtype=torch.float32) * 0.5)
    sq = x * x  # fp32 squares of the rounded values
    exact = (x.double() * x.double()).sum(dim=1)

    res = {"study": "row sum of squares accumulation depth", "rows": rows, "K": K,
           "eps": EPS, "seed": SEED, "dtype": "bf16-rounded x, fp32 arithmetic",
           "methods": {}}
    maxima = {}
    for name, fn in (("seq", sum_seq), ("partial80", sum_partial80),
                     ("pairwise", sum_pairwise)):
        t0 = time.perf_counter()
        s = fn(sq)
        dt = time.perf_counter() - t0
        rel_sum = ((s.double() - exact) / exact).abs()
        rstd = 1.0 / torch.sqrt((s / K) + EPS)
        rstd64 = 1.0 / torch.sqrt((exact.float() / K) + EPS)
        rel_rstd = ((rstd.double() - rstd64.double()) / rstd64.double()).abs()
        # element-level consequence: does applying rstd move round_dtype(x*rstd)?
        mid = round_bf16(x * rstd.unsqueeze(1))
        mid64 = round_bf16(x * rstd64.double().float().unsqueeze(1))
        flips = (mid != mid64).sum().item()
        res["methods"][name] = {
            "sum_rel_err_max": float(rel_sum.max()),
            "sum_rel_err_median": float(rel_sum.median()),
            "sum_rel_err_median_eps": float(rel_sum.median() / 2**-24),
            "rstd_rel_err_max": float(rel_rstd.max()),
            "rstd_rel_err_p95": float(rel_rstd.quantile(0.95)),
            "rstd_rel_err_median": float(rel_rstd.median()),
            "mid_flips": flips,
            "mid_flips_per_million": flips / (rows * K) * 1e6,
            "cpu_seconds": round(dt, 2),
        }
        maxima[name] = float(rel_rstd.max())
    # worst-case ratio is the meaningful one here: the median of a good method is
    # exactly 0, and it is the worst rows that decide a tier violation.
    base = maxima["seq"]
    res["rstd_max_error_reduction_vs_seq"] = {
        k: (round(base / v, 2) if v > 0 else None) for k, v in maxima.items()}
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["methods"], indent=1))
    print("worst-case rstd error reduction vs the plain sequential sum:",
          json.dumps(res["rstd_max_error_reduction_vs_seq"]))
    print("written", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
