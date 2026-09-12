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

# Judgement per output: rel_l2 (batch-level), MARE (max relative error, |ref|
# domain) and MaxAbsErr, with tolerances taken from the CANN official test
# (atol=rtol=2^-7 for bf16, 2^-10 for fp16) rather than invented here.
TOL = {
    torch.bfloat16: {"atol": 7.9345703125e-03, "mare": 7.9345703125e-03,
                     "rel_l2_round": 1.0e-03, "rel_l2_soft": 5.0e-03},
    torch.float16: {"atol": 1.0986328125e-03, "mare": 1.0986328125e-03,
                    "rel_l2_round": 1.0e-03, "rel_l2_soft": 3.0e-03},
}
# rstd is fp32 computed from the rounded residual; it has no CANN atol entry, so
# it is judged relatively (fp32 reduction order is the only difference source).
RSTD_TOL = {"mare": 2.0 ** -6, "rel_l2": 5.0e-03}


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


def metrics(out, ref):
    """MERE/MARE/MaxAbsErr/rel_l2 for one output tensor (ref is the exact side)."""
    d = (out.double() - ref.double()).abs()
    rel = d / (ref.double().abs() + 1e-7)
    ref_l2 = ref.double().norm()
    return {
        "MERE": rel.mean().item(),
        "MARE": rel.max().item(),
        "MaxAbsErr": d.max().item(),
        "rel_l2": (d.norm() / ref_l2).item() if ref_l2 > 0 else d.max().item(),
    }


def judge(x1_dtype, name, m):
    """Applies the declared per-output criterion; returns (ok, reason)."""
    t = TOL[x1_dtype]
    if name == "rstd":
        rel_tol = RSTD_TOL["rel_l2"]
        mare_tol = RSTD_TOL["mare"]
    else:
        rel_tol = t["rel_l2_round"] if name == "x_out" else t["rel_l2_soft"]
        mare_tol = t["mare"]
    ok = m["rel_l2"] <= rel_tol and m["MARE"] <= mare_tol and m["MaxAbsErr"] <= t["atol"]
    reason = "" if ok else (
        f"rel_l2 {m['rel_l2']:.3e}<={rel_tol:.1e}? MARE {m['MARE']:.3e}<={mare_tol:.1e}? "
        f"MaxAbsErr {m['MaxAbsErr']:.3e}<={t['atol']:.1e}?")
    return ok, reason
