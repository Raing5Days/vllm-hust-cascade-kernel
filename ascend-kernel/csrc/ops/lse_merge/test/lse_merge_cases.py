"""Shared case matrix for lse_merge precision tests (self-designed, see
lse_merge-test-cases.md - no testcase-gen doc exists for this M2 operator)."""
import torch

DEV = "npu:0"

# (mode, o1 dtype, o2 dtype, out_code, padded_lse, ref_tol MERE, ref_tol MARE, mare ref_floor)
# mare ref_floor: MARE is judged on elements with |ref| >= floor. For fp32-out
# mode the merge output can cancel to ~1e-5 where relative error is unbounded
# for ANY finite-precision implementation (abs err there is at the fp32 noise
# floor, ~50x BELOW the operand-scale noise); those elements are reported via
# MaxAbsErr instead. bf16-out modes keep the full-domain MARE (bf16 relative
# quantization is scale-free), floor 0.
MODES = {
    "A_tier0":        (torch.bfloat16, torch.bfloat16, 0, False, 7.81e-3, 7.81e-2, 0.0),
    "B_fp32out":      (torch.float32, torch.bfloat16, 2, False, 1.22e-4, 1.22e-3, 1e-2),
    "Bp_fp32out_pad": (torch.float32, torch.bfloat16, 2, True,  1.22e-4, 1.22e-3, 1e-2),
    "C_hybrid_bf16":  (torch.float32, torch.bfloat16, 1, False, 7.81e-3, 7.81e-2, 0.0),
    "D_tier0_pad":    (torch.bfloat16, torch.bfloat16, 0, True,  7.81e-3, 7.81e-2, 0.0),
}

SHAPES = [
    ("production", (64, 40, 128)),
    ("0.5B",       (56, 8, 128)),
    ("tiny",       (7, 4, 16)),     # per-core rows not divisible by 8 (M2 latent bug)
    ("odd-rows",   (13, 8, 128)),   # non-multiple-of-32 tail tile
    ("single-row", (1, 40, 128)),
    ("small-tile", (32, 32, 64)),
]


def make_inputs(T, H, D, o1_dtype, padded, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    o1 = torch.randn(T, H, D, generator=g, dtype=torch.float32).to(o1_dtype).to(DEV)
    o2 = torch.randn(T, H, D, generator=g, dtype=torch.float32).to(torch.bfloat16).to(DEV)
    l1 = torch.randn(T, H, generator=g, dtype=torch.float32).to(DEV)
    l2 = torch.randn(T, H, generator=g, dtype=torch.float32).to(DEV)
    l1c, l2c = l1, l2
    if padded:  # catlass FAInferBf16Fp32Out layout: 8 replicated fp32 per (t,h)
        l1c = l1.unsqueeze(-1).expand(T, H, 8).contiguous().reshape(T, H * 8)
        l2c = l2.unsqueeze(-1).expand(T, H, 8).contiguous().reshape(T, H * 8)
    return o1, o2, l1c, l2c, l1, l2


def fp64_ref(o1, o2, l1, l2):
    m = torch.maximum(l1, l2)
    w1 = torch.exp(l1.double() - m.double())
    w2 = torch.exp(l2.double() - m.double())
    return ((o1.double() * w1[..., None] + o2.double() * w2[..., None])
            / (w1 + w2)[..., None])


def boundary_cases():
    """(name, mutator) applied to the production shape; mutator edits o1/o2/l1/l2."""
    T, H, D = 64, 40, 128

    def equal(o1, o2, l1, l2):
        l2.copy_(l1)
    def dlse_p40(o1, o2, l1, l2):   # w2 -> 0, out ~ o1
        l2.copy_(l1 - 40.0)
    def dlse_m40(o1, o2, l1, l2):   # out ~ o2
        l2.copy_(l1 + 40.0)
    def large_mag(o1, o2, l1, l2):
        l1.fill_(30.0); l2.fill_(35.0)
    def o2_zero(o1, o2, l1, l2):
        o2.zero_()
    def o1_zero(o1, o2, l1, l2):
        o1.zero_()
    return [("equal_weights", equal), ("dlse_p40", dlse_p40), ("dlse_m40", dlse_m40),
            ("large_mag_30_35", large_mag), ("o2_zero", o2_zero), ("o1_zero", o1_zero)]
