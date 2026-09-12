"""Reference implementation and metric helpers for add_rms_norm_stats.

The reference follows the CANN `npu_add_rms_norm_bias` golden semantics
(vllm-ascend-hust tests/e2e/nightly/single_node/ops/singlecard_ops/
test_add_rms_norm_bias.py, kernelType 1 = fp16, 2 = bf16):

    x_out = round_dtype(x1 + x2)                     # residual output
    rms2  = mean_k(fp32(x_out) ** 2)                 # statistics on the ROUNDED value
    rstd  = 1 / sqrt(rms2 + eps)
    y     = round_dtype( round_dtype(fp32(x_out) * rstd) * gamma + beta )

Differences from the golden, both deliberate and both documented in
test-cases.md:
  - the sum of squares is accumulated in fp64 here (the golden uses fp32
    torch.sum): this reference is the "exact" side of the comparison, so the
    tolerances below must absorb the kernel's fp32 reduction order;
  - the gamma multiply is carried out in fp32 for fp16 too (the golden's fp16
    path multiplies in fp16); the fp16 tier absorbs the <= 2^-11 difference of
    that choice.

The acceptance criterion is the workspace ops-precision-standard skill's mixed
tolerance (see STD below), not one invented here.
"""
import torch

# Judgement per output: the workspace's ops-precision-standard skill (read-only
# reference, `.agents/skills/ops-precision-standard/`), float-compute class:
#   references/float_compute.md 2-4 + scripts/mixed_tolerance_check.py
#     elementwise:  |actual - golden| <= atol + rtol * |golden|
#     overall:      matched_ratio >= required_matched_ratio (0.99)
#                   AND max_abs_error <= max(fixed_limit, 32 * ULP_at_one)
# The table below is that skill's table, keyed by the *output* dtype; the numbers
# are copied verbatim, not re-derived (the checker cross-checks them, see
# test_add_rms_norm_stats_ref.py::test_matches_skill_checker_if_available).
# rel_l2 is kept as a batch-level diagnostic on top of the standard's rule: the
# standard is a matched-ratio rule, so rel_l2 is not part of the verdict.
STD = {
    torch.float16: {"atol": 2.0 ** -9, "rtol": 2.0 ** -9,
                    "required_matched_ratio": 0.99, "fixed_limit": 1e-1, "ulp_at_one": 2.0 ** -10},
    torch.bfloat16: {"atol": 2.0 ** -6, "rtol": 2.0 ** -6,
                     "required_matched_ratio": 0.99, "fixed_limit": 1e-0, "ulp_at_one": 2.0 ** -7},
    torch.float32: {"atol": 2.0 ** -16, "rtol": 2.0 ** -10,
                    "required_matched_ratio": 0.99, "fixed_limit": 1e-2, "ulp_at_one": 2.0 ** -23},
}
# Secondary, deliberately stricter reading, kept for the record: the tier CANN's
# own test_add_rms_norm_bias.py asserts (torch.allclose(rtol=atol=2^-7 bf16 /
# 2^-10 fp16)) applied to *every* element. It is not the verdict - it is reported
# so that the marginal gap to the incumbent stays visible (test-cases.md 3.1/3.2).
CANN_TEST_TIER = {
    torch.bfloat16: {"atol": 7.9345703125e-03, "rtol": 7.9345703125e-03},
    torch.float16: {"atol": 1.0986328125e-03, "rtol": 1.0986328125e-03},
}
REL_L2_DIAG = {"x_out": 1.0e-03, "y": 5.0e-03, "rstd": 5.0e-03}  # diagnostic only


def std_spec(out_dtype):
    """The standard's row for one output dtype, with the derived abs-error limit."""
    t = STD[out_dtype]
    return dict(t, max_abs_error_limit=max(t["fixed_limit"], 32.0 * t["ulp_at_one"]))


def output_dtype(name, x1_dtype):
    """`rstd` is fp32 by contract; the other outputs carry the data dtype."""
    return torch.float32 if name == "rstd" else x1_dtype


def tier(name, x1_dtype):
    """(atol, rtol) of the standard's tier for one output."""
    s = std_spec(output_dtype(name, x1_dtype))
    return s["atol"], s["rtol"]


def metrics(out, ref, atol=0.0, rtol=0.0, strict_atol=None, strict_rtol=None):
    """Standard metrics for one output (ref = the higher-precision side).

    Both operands are flattened first and a differing element count is a hard
    error: `rstd` is allocated as (M_padded, 1) while a reference may be (M,) or
    (M, 1), and letting torch broadcast those two silently pairs every row with
    every other row (a defect this helper must not be able to reproduce: the
    inflated numbers then look like a real precision failure - and note it bit
    the *diagnostic* column once, producing more "violations" than elements, so
    every count in this module goes through here).

    `strict_atol`/`strict_rtol` add a second, stricter tier (the CANN test tier)
    in the same single pass; it is reported, never judged.
    """
    a = out.double().reshape(-1)
    b = ref.double().reshape(-1)
    if a.numel() != b.numel():
        raise ValueError(
            f"metrics: element count mismatch {a.numel()} vs {b.numel()} "
            f"(shapes {tuple(out.shape)} vs {tuple(ref.shape)}) - refusing a broadcast compare")
    d = (a - b).abs()
    rel = d / (b.abs() + 1e-7)
    ref_l2 = b.norm()
    n_el = a.numel()
    n_viol = int((d > (atol + rtol * b.abs())).sum().item())
    res = {
        "n_el": n_el,
        "n_viol": n_viol,
        "viol_frac": n_viol / n_el,
        "matched_ratio": 1.0 - n_viol / n_el,
        "MERE": rel.mean().item(),
        "MARE": rel.max().item(),
        "MaxAbsErr": d.max().item(),
        "rel_l2": (d.norm() / ref_l2).item() if ref_l2 > 0 else d.max().item(),
    }
    if strict_atol is not None:
        res["strict_n_viol"] = int((d > (strict_atol + strict_rtol * b.abs())).sum().item())
    return res


def verdict(m, spec, require_all_elements=False):
    """The standard's overall rule; returns (ok, reason).

    `require_all_elements` is used by the production-oracle cross-check, where the
    point is indistinguishability from the incumbent rather than spec compliance.
    """
    ratio_req = 1.0 if require_all_elements else spec["required_matched_ratio"]
    limit = spec["max_abs_error_limit"]
    ratio_ok = m["matched_ratio"] >= ratio_req
    abs_ok = m["MaxAbsErr"] <= limit
    ok = ratio_ok and abs_ok
    reason = "" if ok else (
        f"matched_ratio {m['matched_ratio']:.8f} >= {ratio_req}? "
        f"max_abs_error {m['MaxAbsErr']:.3e} <= {limit:.3e}? "
        f"(tier atol={spec['atol']:.3e}, rtol={spec['rtol']:.3e}, "
        f"max abs err at |ref| ~ {m['MARE']:.2e} relative)")
    return ok, reason


def judge(x1_dtype, name, m):
    """Applies the standard's verdict to one output; returns (ok, reason)."""
    return verdict(m, std_spec(output_dtype(name, x1_dtype)))


def cpu_ref(x1, x2, gamma, beta, eps, mode):
    """Returns (x_out, rstd, y); components not produced by `mode` are None."""
    x1f = x1.double()
    x2f = x2.double()
    if mode == 1:
        xo = x1f  # input already is the rounded residual
    else:
        xo = (x1f + x2f).to(x1.dtype).double()
    rstd = 1.0 / torch.sqrt((xo * xo).mean(dim=-1, keepdim=True) + eps)
    x_out = None
    y = None
    if mode == 0:
        x_out = xo.to(x1.dtype)
    if mode == 1:
        mid = (xo * rstd).to(x1.dtype).double()
        acc = mid * gamma.double().unsqueeze(0)
        if beta is not None:
            acc = acc + beta.double().unsqueeze(0)
        y = acc.to(x1.dtype)
    return x_out, rstd.float(), y
