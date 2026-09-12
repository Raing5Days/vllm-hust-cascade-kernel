"""Case matrix for the add_rms_norm_stats precision suite and S1 anchor.

Shape domain: rows M (per-core split boundaries, all 8-alignment branches) x
K <= 5120 with K % 16 == 0. The production shapes of the F2 hotspot are
(2048, 5120) prefill and (32, 5120) decode (profiles/qwen14b-instruct-hotspot-
20260910); the rest are boundary cases of this kernel's own arithmetic.
"""
import torch

DEV = "npu:0"

# mode -> (uses gamma/beta, produces x_out, produces y)
MODES = {
    0: {"name": "residual_stats", "produces": ("x_out", "rstd")},
    1: {"name": "stats_apply", "produces": ("rstd", "y")},
    2: {"name": "stats_only", "produces": ("rstd",)},
}

SHAPES = [
    ("prefill-main", (2048, 5120)),   # F2 主 shape (profile: M=2048 chunk)
    ("decode", (32, 5120)),           # decode shape (B=32)
    ("single-row", (1, 5120)),        # M < RSTD_GROUP, partial rstd group
    ("odd-rows-77", (77, 128)),       # M % 8 != 0 and M % (cores*8) != 0
    ("k-not-pow2", (255, 5008)),      # K not a power of two (16-aligned)
    ("k-min", (7, 16)),               # minimal K, M < 8
    ("m-1000", (1000, 5120)),         # non-uniform per-core split
    ("k-2048", (128, 2048)),          # mid K
]

# S1 anchor shapes (perf only, no reference needed)
S1_SHAPES = [("prefill-main", (2048, 5120)), ("decode", (32, 5120))]


def make_inputs(m, k, dtype, seed, has_beta, with_gamma_beta=True):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x1 = (torch.randn(m, k, generator=g, dtype=torch.float32) * 0.5).to(dtype).to(DEV)
    x2 = (torch.randn(m, k, generator=g, dtype=torch.float32) * 0.5).to(dtype).to(DEV)
    gamma = beta = None
    if with_gamma_beta:
        gamma = (1.0 + torch.randn(k, generator=g, dtype=torch.float32) * 0.1).to(dtype).to(DEV)
        if has_beta:
            beta = (torch.randn(k, generator=g, dtype=torch.float32) * 0.05).to(dtype).to(DEV)
    return x1, x2, gamma, beta
