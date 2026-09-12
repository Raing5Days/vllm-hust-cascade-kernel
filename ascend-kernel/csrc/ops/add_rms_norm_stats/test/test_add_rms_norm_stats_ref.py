"""CPU-only tests for the reference / judgement helpers (no NPU required).

These are the pure-logic half of the operator's test kit: they assert the
semantics of the tolerance rule itself, so they run in the CPU `pytest -q` gate
(the device suite in test_add_rms_norm_stats_precision.py skips without an NPU).

Regression guard for a defect found on 2026-09-12 in the first precision run:
the operator returns `rstd` shaped (M_padded, 1) while the oracle side was
flattened to (M,), and the metric helper let torch broadcast the two into an
(M, M) pairing. The reported numbers then looked like a real precision failure
(`rel_l2 = 0.64` together with a max relative error of only `0.066`, which is
impossible for an elementwise-aligned pair: `rel_l2 <= MARE` always holds).
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from add_rms_norm_stats_ref import cpu_ref, judge, metrics, tier


def test_metrics_is_zero_for_identical_tensors():
    a = torch.randn(16, 64, dtype=torch.float32)
    m = metrics(a, a.clone(), 8e-3, 8e-3)
    assert m["MaxAbsErr"] == 0.0
    assert m["rel_l2"] == 0.0
    assert m["n_viol"] == 0
    assert m["n_el"] == 16 * 64


def test_metrics_refuses_element_count_mismatch():
    a = torch.randn(8, 1, dtype=torch.float32)   # e.g. a padded rstd of 8 rows
    b = torch.randn(7, dtype=torch.float32)      # vs a 7-row reference
    with pytest.raises(ValueError, match="refusing a broadcast compare"):
        metrics(a, b)


def test_metrics_pairs_rows_of_rstd_shaped_outputs():
    """(M, 1) vs (M,) with equal numel is allowed *and* element i meets element i."""
    a = torch.tensor([[1.0], [2.0], [3.0]])
    b = torch.tensor([1.0, 2.0, 3.0])
    m = metrics(a, b, 8e-3, 8e-3)
    assert m["MaxAbsErr"] == 0.0 and m["rel_l2"] == 0.0
    # the broadcast defect would have produced max |a_i - b_j| = 2.0 here
    assert m["MaxAbsErr"] < 1e-6


def test_cann_tier_keeps_the_rtol_term():
    """1 bf16 ulp at |ref| = 8 (2^-5) is inside the CANN tier, outside the bare atol.

    This is why the criterion is the elementwise CANN pair and not a bare
    absolute bound: for the F2 `y` (values up to ~8) the bare atol is ~7x
    stricter than one representable step, i.e. unsatisfiable by any correct
    kernel, and it is not what the CANN official test asserts.
    """
    ref = torch.full((64,), 8.0)
    out = ref + 2.0 ** -5
    atol, rtol, _ = tier("y", torch.bfloat16)
    m = metrics(out, ref, atol, rtol)
    assert m["MaxAbsErr"] == pytest.approx(2.0 ** -5)
    assert m["MaxAbsErr"] > atol            # the bare-atol reading rejects it ...
    assert m["n_viol"] == 0                 # ... the CANN pair accepts it
    assert judge(torch.bfloat16, "y", m)[0]


def test_rstd_is_judged_relatively():
    """rstd passes on a 2^-9 relative error (under both the 2^-6 and rel_l2 5e-3 gates)."""
    ref = torch.full((32,), 1.5)
    ok_m = metrics(ref * (1 + 2.0 ** -9), ref, 0.0, 2.0 ** -6)
    assert judge(torch.bfloat16, "rstd", ok_m)[0]
    bad_m = metrics(ref * (1 + 2.0 ** -4), ref, 0.0, 2.0 ** -6)
    ok, why = judge(torch.bfloat16, "rstd", bad_m)
    assert not ok and "viol" in why


def test_cpu_ref_semantics_and_shapes():
    x1 = torch.randn(4, 32, dtype=torch.float32).to(torch.bfloat16)
    x2 = torch.randn(4, 32, dtype=torch.float32).to(torch.bfloat16)
    gamma = torch.ones(32, dtype=torch.bfloat16)

    xo, rstd, y = cpu_ref(x1, x2, None, None, 1e-6, 0)
    assert xo.shape == x1.shape and rstd.shape == (4, 1) and y is None
    # x_out is the dtype-rounded residual: the rounding must be the only difference
    assert torch.equal(xo.float(), (x1.float() + x2.float()).to(torch.bfloat16).float())

    _, rstd1, y1 = cpu_ref(x1, x2, gamma, None, 1e-6, 1)
    assert rstd1.shape == (4, 1) and y1.shape == x1.shape
    # mode 1 takes the already-rounded input as the residual
    assert torch.allclose(rstd1, torch.rsqrt((x1.double() ** 2).mean(-1, keepdim=True) + 1e-6)
                          .float(), rtol=1e-6)
