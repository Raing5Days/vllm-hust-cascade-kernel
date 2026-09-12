"""Is the declared tier attainable at all on `y`? (F2 gate-① 诊断脚本, 2026-09-12)

The mode-1 `y` output of this op differs from *both* goldens (the CPU fp64
reference and the CANN `npu_add_rms_norm` production op) by exactly one output
ulps on 1-20 elements out of 5-10 million, which trips the elementwise CANN tier
(violations must be 0). To tell "our kernel is defective" apart from "no
implementation can satisfy this tier on this output", the same tier is applied to
the incumbent itself:

    cann_vs_ref   CANN's y  vs the fp64 reference   <- attainable floor for any
                                                       implementation, including
                                                       the one this op is meant
                                                       to be indistinguishable
                                                       from
    ours_vs_ref   this op's y vs the fp64 reference
    ours_vs_cann  this op's y vs CANN's y

If `cann_vs_ref` violates the tier on the same shapes, the tier cannot be met on
`y` by construction (the golden is itself 1 ulp off the exact reference), and the
headline gate-① shortfall has to be read as a limit of the criterion rather than
as evidence of a wrong kernel. Either way the criterion is reported as it is -
this script only characterises it, it does not move it.

Usage (NPU box, from a scratch cwd, under the shared lock):
    PYTHONPATH=<repo>/python/ascend_kernel python -u run_golden_selfcheck.py --out <json>
"""
import argparse
import json
import os
import sys
import zlib

import torch
import torch_npu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ascend_kernel  # noqa: F401
from add_rms_norm_stats_cases import SHAPES, make_inputs
from add_rms_norm_stats_ref import cpu_ref, metrics, tier

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = 1e-6
DTYPES = [torch.bfloat16, torch.float16]


def dump_violations(dtype, name, shape, atol, rtol):
    """Print the elements where this op's y leaves the tier: which side is off."""
    m, k = shape
    seed = zlib.crc32(f"golden/{dtype}/{name}".encode()) % 2 ** 31
    x1, x2, gamma, _ = make_inputs(m, k, dtype, seed, has_beta=False)
    can_y, _, _ = torch_npu.npu_add_rms_norm(x1, torch.zeros_like(x1), gamma, EPS)
    _, _, y = torch.ops.npu.add_rms_norm_stats(x1, x2, gamma, None, EPS, 1)
    torch.npu.synchronize()

    a = y.cpu().double().reshape(-1)
    c = can_y.cpu().double().reshape(-1)
    b = cpu_ref(x1.cpu(), x2.cpu(), gamma.cpu(), None, EPS, 1)[2].double().reshape(-1)
    d = (a - b).abs()
    viol = (d > (atol + rtol * b.abs())).nonzero().flatten()
    print(f"\n--- violating elements: {dtype} {name} {m}x{k} ({viol.numel()} total, "
          f"tier atol={atol:.4e} rtol={rtol:.4e}) ---")
    print(f"{'idx':>8} {'row':>6} {'col':>6} {'ours':>14} {'ref(fp64)':>14} {'cann':>14} "
          f"{'|ours-ref|':>11} {'|cann-ref|':>11} {'gamma':>8} {'allowed':>11}")
    out = []
    for i in viol[:10].tolist():
        row, col = divmod(i, k)
        rec = {"idx": i, "row": row, "col": col, "ours": a[i].item(), "ref": b[i].item(),
               "cann": c[i].item(), "abs_ours_ref": d[i].item(),
               "abs_cann_ref": abs(c[i].item() - b[i].item()),
               "gamma": gamma[col].item(), "allowed": (atol + rtol * abs(b[i].item()))}
        out.append(rec)
        print(f"{i:>8} {row:>6} {col:>6} {rec['ours']:>14.6g} {rec['ref']:>14.6g} "
              f"{rec['cann']:>14.6g} {rec['abs_ours_ref']:>11.3e} {rec['abs_cann_ref']:>11.3e} "
              f"{rec['gamma']:>8.4f} {rec['allowed']:>11.3e}")
    return {"dtype": str(dtype).replace("torch.", ""), "case": name, "shape": f"{m}x{k}",
            "n_viol": int(viol.numel()), "elements": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "add_rms_norm_stats_golden_selfcheck.json"))
    args = ap.parse_args()

    rows = []
    print(f"{'dtype':>9} {'case':<13} {'shape':<10} "
          f"{'cann_vs_ref':>22} {'ours_vs_ref':>22} {'ours_vs_cann':>22}")
    for dtype in DTYPES:
        atol, rtol = tier("y", dtype)[:2]
        for name, (m, k) in SHAPES:
            seed = zlib.crc32(f"golden/{dtype}/{name}".encode()) % 2 ** 31
            x1, x2, gamma, _ = make_inputs(m, k, dtype, seed, has_beta=False)
            # CANN fed x2 = 0 so its residual output is the input: same statistics,
            # same y, over the same (rounded) input as this op's mode 1.
            can_y, _, _ = torch_npu.npu_add_rms_norm(x1, torch.zeros_like(x1), gamma, EPS)
            _, _, y = torch.ops.npu.add_rms_norm_stats(x1, x2, gamma, None, EPS, 1)
            torch.npu.synchronize()
            _, _, ref_y = cpu_ref(x1.cpu(), x2.cpu(), gamma.cpu(), None, EPS, 1)

            res = {}
            for tag, (a, b) in {
                "cann_vs_ref": (can_y.cpu().float(), ref_y),
                "ours_vs_ref": (y.cpu(), ref_y),
                "ours_vs_cann": (y.cpu(), can_y.cpu().float()),
            }.items():
                mtr = metrics(a, b, atol, rtol)
                res[tag] = {"n_viol": mtr["n_viol"], "n_el": mtr["n_el"],
                            "max_abs_err": mtr["MaxAbsErr"], "rel_l2": mtr["rel_l2"],
                            "MARE": mtr["MARE"]}
            rows.append({"dtype": str(dtype).replace("torch.", ""), "case": name,
                         "shape": f"{m}x{k}", "atol": atol, "rtol": rtol, "checks": res})
            print(f"{str(dtype).replace('torch.',''):>9} {name:<13} {f'{m}x{k}':<10} "
                  + " ".join(f"viol={res[t]['n_viol']:>3}/{res[t]['n_el']:<8} "
                             f"1ulp={res[t]['max_abs_err']:.3e} rl2={res[t]['rel_l2']:.1e}".ljust(23)
                             for t in ("cann_vs_ref", "ours_vs_ref", "ours_vs_cann")), flush=True)

    n_cann_viol = sum(r["checks"]["cann_vs_ref"]["n_viol"] > 0 for r in rows)
    n_ours_viol = sum(r["checks"]["ours_vs_ref"]["n_viol"] > 0 for r in rows)
    dumps = [dump_violations(dtype, "prefill-main", (2048, 5120), *tier("y", dtype)[:2])
             for dtype in DTYPES]
    summary = {
        "op": "add_rms_norm_stats", "date": "2026-09-12", "output": "y (mode 1)",
        "env": "910B2 / CANN 9.1.0 / torch_npu 2.13.0rc1",
        "tier": "CANN elementwise |d| <= atol + rtol*|ref|, atol=rtol=2^-7 bf16 / 2^-10 fp16",
        "n_cases": len(rows),
        "n_cases_cann_violates_tier_vs_fp64_ref": n_cann_viol,
        "n_cases_this_op_violates_tier_vs_fp64_ref": n_ours_viol,
        "violating_element_dumps": dumps,
        "reading": ("If the incumbent (cann_vs_ref) violates the tier too, the tier is not "
                    "attainable on y by any implementation - the golden is itself ~1 ulp off "
                    "the exact reference - so the gate-(1) shortfall is a criterion limit, "
                    "not a kernel defect. Criterion left untouched either way."),
        "cases": rows,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\ncases where CANN itself violates the tier vs the fp64 reference: "
          f"{n_cann_viol}/{len(rows)}")
    print(f"cases where this op violates the tier vs the fp64 reference:      "
          f"{n_ours_viol}/{len(rows)}")
    print(f"written: {args.out}  (criterion unchanged: violations must be 0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
