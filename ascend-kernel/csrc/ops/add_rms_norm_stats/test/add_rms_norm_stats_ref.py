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
    path multiplies in fp16); the fp16 MARE tolerance absorbs the <= 2^-11
    difference of that choice.
"""
import torch

# Judgement per output: rel_l2 (batch-level) plus the CANN official elementwise
# criterion |out - ref| <= atol + rtol * |ref|, with the CANN dtype tier
# (atol = rtol = 2^-7 for bf16, 2^-10 for fp16) rather than one invented here.
# The tier is applied the way the CANN test uses it (torch.allclose(rtol, atol)):
# the rtol term is the whole point of the tier for a tensor whose |y| reaches ~8,
# where a bare 2^-7 absolute bound would be ~7x stricter than one bf16 ulp.
TOL = {
    torch.bfloat16: {"atol": 7.9345703125e-03, "rtol": 7.9345703125e-03,
                     "rel_l2_round": 1.0e-03, "rel_l2_soft": 5.0e-03},
    torch.float16: {"atol": 1.0986328125e-03, "rtol": 1.0986328125e-03,
                    "rel_l2_round": 1.0e-03, "rel_l2_soft": 3.0e-03},
}
# rstd is fp32 computed from the rounded residual; it has no CANN atol entry, so
# it is judged relatively (fp32 reduction order is the only difference source):
# max relative error <= 2^-6, i.e. atol = 0, rtol = 2^-6 in the same form.
RSTD_TOL = {"rtol": 2.0 ** -6, "rel_l2": 5.0e-03}


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


def tier(name, x1_dtype):
    """(atol, rtol, rel_l2_tol) of the declared criterion for one output.

    `rstd` carries its own relative tier (no CANN atol entry); the dtype outputs
    use the CANN official pair, whose rtol term must be kept - see TOL above.
    """
    if name == "rstd":
        t = RSTD_TOL
        return 0.0, t["rtol"], t["rel_l2"]
    t = TOL[x1_dtype]
    return t["atol"], t["rtol"], (t["rel_l2_round"] if name == "x_out" else t["rel_l2_soft"])


def metrics(out, ref, atol=0.0, rtol=0.0):
    """MERE/MARE/MaxAbsErr/rel_l2/violation count for one output (ref = exact side).

    Both operands are flattened first and a differing element count is a hard
    error: `rstd` is allocated as (M_padded, 1) while a reference may be (M,) or
    (M, 1), and letting torch broadcast those two silently pairs every row with
    every other row (a defect this helper must not be able to reproduce: the
    inflated numbers then look like a real precision failure).
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
    n_viol = int((d > (atol + rtol * b.abs())).sum().item())
    return {
        "n_el": a.numel(),
        "n_viol": n_viol,
        "viol_frac": n_viol / a.numel(),
        "MERE": rel.mean().item(),
        "MARE": rel.max().item(),
        "MaxAbsErr": d.max().item(),
        "rel_l2": (d.norm() / ref_l2).item() if ref_l2 > 0 else d.max().item(),
    }


def verdict(m, atol, rtol, rel_tol):
    """Single implementation of the declared criterion (used by both suites)."""
    ok = m["rel_l2"] <= rel_tol and m["n_viol"] == 0
    reason = "" if ok else (
        f"rel_l2 {m['rel_l2']:.3e}<={rel_tol:.1e}? viol {m['n_viol']}/{m['n_el']} "
        f"(criterion |d|<=atol+rtol*|ref|, atol={atol:.3e}, rtol={rtol:.3e}; "
        f"MaxAbsErr {m['MaxAbsErr']:.3e}, MARE {m['MARE']:.3e})?")
    return ok, reason


def judge(x1_dtype, name, m):
    """Applies the declared per-output criterion; returns (ok, reason)."""
    atol, rtol, rel_tol = tier(name, x1_dtype)
    return verdict(m, atol, rtol, rel_tol)
