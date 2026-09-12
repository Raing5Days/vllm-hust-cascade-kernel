"""Precision suite + report (JSON + Markdown) for add_rms_norm_stats.

Runs every (mode x dtype x shape [x beta]) case against the CPU reference
(add_rms_norm_stats_ref.cpu_ref, CANN golden semantics, fp64 accumulation) and,
where the CANN op offers the same semantics (beta = None), cross-checks the
device outputs against torch_npu.npu_add_rms_norm (the production oracle of the
F2 hotspot).

Usage (on the NPU box, from a scratch cwd - not from /vllm-workspace):
    PYTHONPATH=<repo>/python/ascend_kernel python -u run_precision_report.py \
        --out-json add_rms_norm_stats_precision_report.json \
        --out-md add_rms_norm_stats_precision_report.md
"""
import argparse
import json
import os
import sys
import zlib

import torch
import torch_npu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ascend_kernel  # noqa: F401  (registers torch.ops.npu.add_rms_norm_stats)
from add_rms_norm_stats_cases import MODES, SHAPES, make_inputs
from add_rms_norm_stats_ref import (
    CANN_TEST_TIER,
    cpu_ref,
    judge,
    metrics,
    output_dtype,
    std_spec,
    verdict,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DTYPES = [torch.bfloat16, torch.float16]
EPS = 1e-6


def run_case(mode, dtype, name, shape, has_beta, seed):
    m, k = shape
    x1, x2, gamma, beta = make_inputs(m, k, dtype, seed, has_beta)
    x_out, rstd, y = torch.ops.npu.add_rms_norm_stats(x1, x2, gamma, beta, EPS, mode)
    torch.npu.synchronize()
    ref_xo, ref_rstd, ref_y = cpu_ref(x1.cpu(), x2.cpu(),
                                      gamma.cpu() if gamma is not None else None,
                                      beta.cpu() if beta is not None else None, EPS, mode)
    outs = {"x_out": (x_out, ref_xo), "rstd": (rstd, ref_rstd), "y": (y, ref_y)}
    row = {"mode": mode, "mode_name": MODES[mode]["name"], "shape": f"{m}x{k}", "case": name,
           "dtype": str(dtype).replace("torch.", ""), "beta": bool(has_beta), "checks": {}}
    ok_all = True
    for key in MODES[mode]["produces"]:
        out, ref = outs[key]
        if key == "rstd":
            out = out[:m]  # rstd is padded to ceil(M/8) rows
        spec = std_spec(output_dtype(key, dtype))
        # The strict tier is the one CANN's own test_add_rms_norm_bias.py asserts
        # (all elements required); it is reported so the marginal distance to the
        # incumbent stays visible, but the standard's rule is the verdict.
        ct = CANN_TEST_TIER[dtype]
        mtr = metrics(out.cpu(), ref, spec["atol"], spec["rtol"],
                      strict_atol=ct["atol"], strict_rtol=ct["rtol"])
        mtr["cann_tier_viol"] = mtr.pop("strict_n_viol")
        mtr["max_abs_error_limit"] = spec["max_abs_error_limit"]
        ok, why = judge(dtype, key, mtr)
        mtr["pass"] = bool(ok)
        if why:
            mtr["fail_reason"] = why
        row["checks"][key] = mtr
        ok_all = ok_all and ok
    # structural check: the rstd padding rows must be zero
    pad_ok = True
    if rstd.numel() > m:
        pad_ok = bool((rstd[m:].abs() == 0).all().item())
    row["rstd_padding_zero"] = pad_ok
    row["pass"] = bool(ok_all and pad_ok)
    return row


def run_oracle(mode, dtype, name, shape, seed):
    """Cross-check against the CANN production oracle (torch_npu.npu_add_rms_norm).

    Only valid for beta = None; for mode 1 the CANN op is fed x2 = 0 so that its
    residual output is exactly the input x1, i.e. it computes the same
    statistics and the same y over the same (rounded) input.
    """
    m, k = shape
    x1, x2, gamma, _ = make_inputs(m, k, dtype, seed, has_beta=False)
    if mode == 1:
        can_y, can_rstd, can_xo = torch_npu.npu_add_rms_norm(x1, torch.zeros_like(x1), gamma, EPS)
    else:
        can_y, can_rstd, can_xo = torch_npu.npu_add_rms_norm(x1, x2, gamma, EPS)
    x_out, rstd, y = torch.ops.npu.add_rms_norm_stats(x1, x2, gamma, None, EPS, mode)
    torch.npu.synchronize()
    out = {"mode": mode, "shape": f"{m}x{k}", "case": name,
           "dtype": str(dtype).replace("torch.", ""), "checks": {}}
    # Both sides are flattened inside metrics(), so a (M, 1) rstd against a (M,)
    # oracle cannot broadcast into an (M, M) pairing (that defect produced a
    # spurious rel_l2 = 0.64 > MARE = 0.066, impossible for an aligned pair).
    pairs = [("rstd", rstd[:m], can_rstd.reshape(-1)[:m])]
    if mode == 0:
        pairs.append(("x_out", x_out, can_xo))
    if mode == 1:
        pairs.append(("y", y, can_y))
    for key, out_t, ref_t in pairs:
        # vs the CANN op the requirement is *stricter* than the standard: every
        # element within the standard's tier (matched_ratio 1.0), because this
        # comparison exists to show indistinguishability from the incumbent, not
        # to pass a spec.
        spec = std_spec(output_dtype(key, dtype))
        ct = CANN_TEST_TIER[dtype]
        mtr = metrics(out_t.cpu(), ref_t.cpu().float(), spec["atol"], spec["rtol"],
                      strict_atol=ct["atol"], strict_rtol=ct["rtol"])
        mtr["cann_tier_viol"] = mtr.pop("strict_n_viol")
        mtr["require_all_elements"] = True
        mtr["max_abs_error_limit"] = spec["max_abs_error_limit"]
        ok, why = verdict(mtr, spec, require_all_elements=True)
        mtr["pass"] = bool(ok)
        if why:
            mtr["fail_reason"] = why
        out["checks"][key] = mtr
    out["pass"] = all(c["pass"] for c in out["checks"].values())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default=os.path.join(HERE, "add_rms_norm_stats_precision_report.json"))
    ap.add_argument("--out-md", default=os.path.join(HERE, "add_rms_norm_stats_precision_report.md"))
    ap.add_argument("--skip-oracle", action="store_true")
    args = ap.parse_args()

    rows = []
    for mode in sorted(MODES):
        for dtype in DTYPES:
            for name, shape in SHAPES:
                beta_flags = [False, True] if mode == 1 else [False]
                for has_beta in beta_flags:
                    seed = zlib.crc32(f"{mode}/{dtype}/{name}/{has_beta}".encode()) % 2 ** 31
                    rows.append(run_case(mode, dtype, name, shape, has_beta, seed))
                    print(f"[case] mode={mode} {str(dtype).replace('torch.','')} {name} "
                          f"{rows[-1]['shape']} beta={has_beta} -> "
                          f"{'PASS' if rows[-1]['pass'] else 'FAIL'}", flush=True)

    oracle_rows = []
    if not args.skip_oracle:
        for mode in sorted(MODES):
            for dtype in DTYPES:
                for name, shape in SHAPES:
                    seed = zlib.crc32(f"oracle/{mode}/{dtype}/{name}".encode()) % 2 ** 31
                    oracle_rows.append(run_oracle(mode, dtype, name, shape, seed))
                    print(f"[oracle] mode={mode} {str(dtype).replace('torch.','')} {name} -> "
                          f"{'PASS' if oracle_rows[-1]['pass'] else 'FAIL'}", flush=True)

    n_pass = sum(r["pass"] for r in rows)
    report = {
        "op": "add_rms_norm_stats",
        "date": "2026-09-12",
        "env": {
            "device": "910B2 / CANN 9.1.0 / torch_npu 2.13.0rc1",
            "ref": "CPU fp64 accumulation, CANN golden rounding protocol",
            "oracle": "torch_npu.npu_add_rms_norm (beta=None cases)",
        },
        "n_cases": len(rows), "n_pass": n_pass, "n_fail": len(rows) - n_pass,
        "cases": rows,
        "oracle_cases": oracle_rows,
        "oracle_n_pass": sum(r["pass"] for r in oracle_rows),
    }
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=1)

    lines = [
        "# add_rms_norm_stats 精度验证报告（F2 norm 阶段核，2026-09-12）",
        "",
        (
            f"- 用例总数 {len(rows)}，通过 {n_pass}，失败 {len(rows) - n_pass}，"
            f"通过率 {100.0 * n_pass / len(rows):.1f}%"
        ),
        "- 参考实现：CPU fp64 累加 + CANN golden 舍入口径（`add_rms_norm_stats_ref.py`）",
        (
            "- **判据（容差按 ops-precision-standard）**：workspace skill "
            "`.agents/skills/ops-precision-standard/`（浮点计算类）的混合容差——逐元素 "
            "`|out−ref| <= atol + rtol·|ref|`，整体 `matched_ratio >= 0.99` 且 "
            "`max_abs_error <= max(fixed_limit, 32·ULP@1.0)`。档位按**输出** dtype 取表："
            "fp16 atol=rtol=2^-9、bf16 2^-6、rstd（fp32）atol=2^-16/rtol=2^-10；"
            "abs 上限 fp16 0.1 / bf16 1.0 / fp32 1e-2"
        ),
        (
            "- 旁证列（**不参与判定**，留档以便复核）：`matched_ratio` 为标准判据量；"
            "`rel_l2` 为批级相对 L2；`cann_viol/n` 为改用 **CANN 自带用例更严档位**"
            "（atol=rtol=2^-7/2^-10 且要求全元素）时的违例数——用它把『与现役算子的边际距离』"
            "量化留档（见 `add_rms_norm_stats-test-cases.md` §3.1/§3.2）"
        ),
        (
            f"- 生产 oracle 交叉核对（`torch_npu.npu_add_rms_norm`，beta=None）："
            f"{report['oracle_n_pass']}/{len(oracle_rows)} 通过；该路径按**更严**口径要求"
            "逐元素全过（matched_ratio = 1.0），因为它的目的是证明与现役算子不可区分"
        ),
        "- shape/口径见 `add_rms_norm_stats-test-cases.md`",
        "",
        "## 用例明细",
        "",
        "| mode | case | shape | dtype | beta | 输出 | matched | rel_l2 | viol/n | MaxAbsErr | abs上限 | 判定 | CANN严档 viol |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        for key, mtr in r["checks"].items():
            lines.append(f"| {r['mode']} {r['mode_name']} | {r['case']} | {r['shape']} | {r['dtype']} | "
                         f"{'Y' if r['beta'] else 'N'} | {key} | {mtr['matched_ratio']:.8f} | "
                         f"{mtr['rel_l2']:.3e} | {mtr['n_viol']}/{mtr['n_el']} | "
                         f"{mtr['MaxAbsErr']:.3e} | {mtr['max_abs_error_limit']:.3e} | "
                         f"{'PASS' if mtr['pass'] else 'FAIL'} | {mtr['cann_tier_viol']} |")
    lines += [
        "",
        "## 生产 oracle 交叉核对（vs `torch_npu.npu_add_rms_norm`）",
        "",
        "| mode | case | shape | dtype | 输出 | matched | rel_l2 | viol/n | MaxAbsErr | 判定 | CANN严档 viol |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in oracle_rows:
        for key, mtr in r["checks"].items():
            lines.append(f"| {r['mode']} | {r['case']} | {r['shape']} | {r['dtype']} | {key} | "
                         f"{mtr['matched_ratio']:.8f} | {mtr['rel_l2']:.3e} | "
                         f"{mtr['n_viol']}/{mtr['n_el']} | {mtr['MaxAbsErr']:.3e} | "
                         f"{'PASS' if mtr['pass'] else 'FAIL'} | {mtr['cann_tier_viol']} |")
    with open(args.out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nreport: {args.out_json}\nmd: {args.out_md}")
    print(f"cases {n_pass}/{len(rows)} pass; oracle {report['oracle_n_pass']}/{len(oracle_rows)} pass")
    return 0 if n_pass == len(rows) and report["oracle_n_pass"] == len(oracle_rows) else 1


if __name__ == "__main__":
    sys.exit(main())
