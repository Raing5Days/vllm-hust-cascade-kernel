"""Where does this op sit relative to the incumbent, tier by tier? (gate-(1) evidence)

Two tiers are reported for the mode-1 `y` output over the whole shape matrix:

  standard  the ops-precision-standard mixed tolerance (atol+rtol per element,
            matched_ratio >= 0.99) - the acceptance rule of the gate;
  strict    the tier CANN's own test_add_rms_norm_bias.py asserts
            (atol = rtol = 2^-7 bf16 / 2^-10 fp16, all elements required).

and each is applied to three pairings:

    cann_vs_ref   CANN's y          vs the fp64 reference
    ours_vs_ref   this op's y       vs the fp64 reference
    ours_vs_cann  this op's y       vs CANN's y

`cann_vs_ref` is the control that separates "our kernel is wrong" from "the tier
is unattainable": measured 2026-09-12, CANN clears the strict tier on 16/16 cases
while this op misses it by 1-20 elements per case (each off by one output ulp).
So the strict-tier gap is a real, ours-only marginal difference - not a criterion
artifact - and it is worth naming precisely, which the element dump does:

    for every violating element, both goldens and the intermediate rstd are shown;
    our rstd sits ~4e-6 relative from CANN's (fp32 reduction order), and that is
    what moves `round_dtype(x*rstd)` across a dtype rounding boundary.

The verdict for the gate comes from run_precision_report.py (the standard tier);
this script only characterises the distance, and never moves a tier.

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
from add_rms_norm_stats_ref import CANN_TEST_TIER, cpu_ref, metrics, std_spec

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = 1e-6
DTYPES = [torch.bfloat16, torch.float16]


def tiers_for(dtype):
    """(standard spec, strict spec) for the mode-1 `y` output of `dtype`."""
    return std_spec(dtype), CANN_TEST_TIER[dtype]


def pair_metrics(out, ref, dtype):
    """Standard-tier metrics plus the strict-tier violation count, one pass."""
    spec, strict = tiers_for(dtype)
    m = metrics(out, ref, spec["atol"], spec["rtol"],
                strict_atol=strict["atol"], strict_rtol=strict["rtol"])
    m["strict_n_viol"] = m.pop("strict_n_viol")
    m["max_abs_error_limit"] = spec["max_abs_error_limit"]
    return m


def dump_strict_violations(dtype, name, shape):
    """Print the elements where our y leaves the *strict* tier, with both goldens."""
    m, k = shape
    spec, strict = tiers_for(dtype)
    seed = zlib.crc32(f"golden/{dtype}/{name}".encode()) % 2 ** 31
    x1, x2, gamma, _ = make_inputs(m, k, dtype, seed, has_beta=False)
    can_y, can_rstd, _ = torch_npu.npu_add_rms_norm(x1, torch.zeros_like(x1), gamma, EPS)
    _, our_rstd, y = torch.ops.npu.add_rms_norm_stats(x1, x2, gamma, None, EPS, 1)
    torch.npu.synchronize()

    a = y.cpu().double().reshape(-1)
    c = can_y.cpu().double().reshape(-1)
    b = cpu_ref(x1.cpu(), x2.cpu(), gamma.cpu(), None, EPS, 1)[2].double().reshape(-1)
    ur = our_rstd[:m].cpu().double().reshape(-1)
    cr = can_rstd.reshape(-1)[:m].double()
    d = (a - b).abs()
    viol = (d > (strict["atol"] + strict["rtol"] * b.abs())).nonzero().flatten()
    print(f"\n--- strict-tier violations: {dtype} {name} {m}x{k} ({viol.numel()} total, "
          f"strict atol={strict['atol']:.4e} rtol={strict['rtol']:.4e}; "
          f"standard atol={spec['atol']:.4e} rtol={spec['rtol']:.4e}) ---")
    print(f"{'idx':>8} {'row':>6} {'col':>6} {'ours':>14} {'ref(fp64)':>14} {'cann':>14} "
          f"{'|ours-ref|':>11} {'|cann-ref|':>11} {'rstd rel':>10} {'gamma':>8} {'allowed':>11}")
    out = []
    for i in viol[:10].tolist():
        row, col = divmod(i, k)
        rstd_rel = (ur[row].item() - cr[row].item()) / cr[row].item()
        rec = {"idx": i, "row": row, "col": col, "ours": a[i].item(), "ref": b[i].item(),
               "cann": c[i].item(), "abs_ours_ref": d[i].item(),
               "abs_cann_ref": abs(c[i].item() - b[i].item()),
               "rstd_ours": ur[row].item(), "rstd_cann": cr[row].item(),
               "rstd_rel_diff": rstd_rel, "gamma": gamma[col].item(),
               "strict_allowed": strict["atol"] + strict["rtol"] * abs(b[i].item()),
               "standard_allowed": spec["atol"] + spec["rtol"] * abs(b[i].item())}
        out.append(rec)
        print(f"{i:>8} {row:>6} {col:>6} {rec['ours']:>14.6g} {rec['ref']:>14.6g} "
              f"{rec['cann']:>14.6g} {rec['abs_ours_ref']:>11.3e} {rec['abs_cann_ref']:>11.3e} "
              f"{rstd_rel:>10.3e} {rec['gamma']:>8.4f} {rec['strict_allowed']:>11.3e}")
    return {"dtype": str(dtype).replace("torch.", ""), "case": name, "shape": f"{m}x{k}",
            "strict_n_viol": int(viol.numel()), "elements": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "add_rms_norm_stats_golden_selfcheck.json"))
    args = ap.parse_args()

    rows = []
    print(f"{'dtype':>9} {'case':<13} {'shape':<10} " +
          " ".join(f"{t:>28}" for t in ("cann_vs_ref", "ours_vs_ref", "ours_vs_cann")))
    for dtype in DTYPES:
        spec, strict = tiers_for(dtype)
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
                mtr = pair_metrics(a, b, dtype)
                res[tag] = {kk: mtr[kk] for kk in
                            ("matched_ratio", "n_viol", "strict_n_viol", "n_el",
                             "MaxAbsErr", "rel_l2", "MARE")}
                res[tag]["standard_pass"] = bool(
                    mtr["matched_ratio"] >= spec["required_matched_ratio"]
                    and mtr["MaxAbsErr"] <= spec["max_abs_error_limit"])
            rows.append({"dtype": str(dtype).replace("torch.", ""), "case": name,
                         "shape": f"{m}x{k}", "checks": res,
                         "standard_tier": {"atol": spec["atol"], "rtol": spec["rtol"],
                                           "matched_ratio": spec["required_matched_ratio"],
                                           "max_abs_error_limit": spec["max_abs_error_limit"]},
                         "strict_tier": dict(strict)})
            print(f"{str(dtype).replace('torch.',''):>9} {name:<13} {f'{m}x{k}':<10} "
                  + " ".join(f"std {'P' if res[t]['standard_pass'] else 'F'}"
                             f" viol={res[t]['n_viol']:<2} strict={res[t]['strict_n_viol']:<3}"
                             f" 1ulp={res[t]['MaxAbsErr']:.1e}".ljust(28)
                             for t in ("cann_vs_ref", "ours_vs_ref", "ours_vs_cann")),
                  flush=True)

    dumps = [dump_strict_violations(dtype, "prefill-main", (2048, 5120)) for dtype in DTYPES]
    n_cann_strict = sum(r["checks"]["cann_vs_ref"]["strict_n_viol"] > 0 for r in rows)
    n_ours_strict = sum(r["checks"]["ours_vs_ref"]["strict_n_viol"] > 0 for r in rows)
    n_ours_std = sum(not r["checks"]["ours_vs_ref"]["standard_pass"] for r in rows)
    summary = {
        "op": "add_rms_norm_stats", "date": "2026-09-12", "output": "y (mode 1)",
        "env": "910B2 / CANN 9.1.0 / torch_npu 2.13.0rc1",
        "standard_tier": "ops-precision-standard mixed tolerance (matched_ratio >= 0.99)",
        "strict_tier": "CANN test_add_rms_norm_bias.py tier, all elements required",
        "n_cases": len(rows),
        "n_cases_this_op_fails_standard": n_ours_std,
        "n_cases_cann_fails_strict_vs_fp64_ref": n_cann_strict,
        "n_cases_this_op_fails_strict_vs_fp64_ref": n_ours_strict,
        "violating_element_dumps": dumps,
        "reading": (
            "CANN clears the strict tier on every case while this op misses it by 1-20 "
            "elements per case, each off by one output ulp (the dump shows our rstd ~4e-6 "
            "relative from CANN's, which is what moves round_dtype(x*rstd) across a "
            "rounding boundary). The strict-tier gap is therefore a real, ours-only "
            "marginal difference, not an artifact of an unattainable criterion - it is "
            "recorded as a residual numerical-quality gap. The gate verdict comes from "
            "the standard tier (run_precision_report.py); no tier was moved here."),
        "cases": rows,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\n[fail standard tier]  ours vs fp64 ref:            {n_ours_std}/{len(rows)}")
    print(f"[control]             CANN vs fp64 ref, strict:     {n_cann_strict}/{len(rows)}")
    print(f"[ours]                ours vs fp64 ref, strict:     {n_ours_strict}/{len(rows)}")
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
