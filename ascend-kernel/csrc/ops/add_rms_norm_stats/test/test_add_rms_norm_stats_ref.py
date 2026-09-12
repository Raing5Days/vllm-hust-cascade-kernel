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

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from add_rms_norm_stats_ref import cpu_ref, judge, metrics, std_spec, tier, verdict


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


def test_standard_tier_is_the_mixed_tolerance_pair():
    """The tier is the ops-precision-standard mixed tolerance, per output dtype.

    fp16 2^-9, bf16 2^-6, and rstd (an fp32 output) 2^-16/2^-10 - with the abs
    error limit max(fixed, 32*ULP@1.0) from the same table.
    """
    assert tier("y", torch.float16) == (2.0 ** -9, 2.0 ** -9)
    assert tier("y", torch.bfloat16) == (2.0 ** -6, 2.0 ** -6)
    assert tier("x_out", torch.bfloat16) == (2.0 ** -6, 2.0 ** -6)
    assert tier("rstd", torch.bfloat16) == (2.0 ** -16, 2.0 ** -10)  # fp32 output
    assert std_spec(torch.bfloat16)["max_abs_error_limit"] == max(1.0, 32 * 2.0 ** -7)
    assert std_spec(torch.float16)["max_abs_error_limit"] == max(0.1, 32 * 2.0 ** -10)
    assert std_spec(torch.float32)["max_abs_error_limit"] == max(1e-2, 32 * 2.0 ** -23)


def test_matched_ratio_rule_is_0_99_not_all_elements():
    """The standard is a two-part rule: ratio >= 0.99 *and* a max-abs-error ceiling.

    One element outside the tier does not fail the case while the max absolute
    error stays under the ceiling - but a gross outlier fails on that ceiling even
    though the ratio is satisfied, which is what keeps the ratio rule from being
    an escape hatch.
    """
    ref = torch.ones(1000)
    spec = std_spec(torch.float32)               # atol=2^-16, rtol=2^-10, limit 1e-2
    out = ref.clone()
    out[0] = ref[0] + 5e-3                       # outside the tier, inside the limit
    m = metrics(out, ref, spec["atol"], spec["rtol"])
    assert m["n_viol"] == 1 and m["matched_ratio"] == pytest.approx(0.999)
    assert verdict(m, spec)[0] is True
    # ... but the strict reading used for the oracle cross-check rejects it
    assert verdict(m, spec, require_all_elements=True)[0] is False
    # a gross outlier trips the max-abs-error ceiling even with the ratio satisfied
    m2 = metrics(ref + 1.0, ref, spec["atol"], spec["rtol"])
    assert m2["matched_ratio"] == 0.0
    ok, why = verdict(m2, spec)
    assert not ok and "max_abs_error" in why


def test_rstd_is_judged_against_the_fp32_tier():
    """rstd is an fp32 output: 1e-6 relative is far inside its tier."""
    ref = torch.full((32,), 1.5)
    ok_m = metrics(ref * (1 + 2.0 ** -20), ref, *tier("rstd", torch.bfloat16))
    assert judge(torch.bfloat16, "rstd", ok_m)[0]
    bad_m = metrics(ref * (1 + 2.0 ** -4), ref, *tier("rstd", torch.bfloat16))
    ok, why = judge(torch.bfloat16, "rstd", bad_m)
    assert not ok and "matched_ratio" in why


def test_matches_skill_checker_if_available():
    """No drift from the skill's own reference implementation of the standard.

    The skill ships scripts/mixed_tolerance_check.py; when that read-only
    reference is present, our metrics/verdict must agree with it on the same
    inputs (tier values, matched_ratio and pass/fail). Skipped where absent.
    """
    import glob
    import importlib.util
    hits = glob.glob("/vllm-workspace/.agents/skills/ops-precision-standard/scripts/"
                     "mixed_tolerance_check.py")
    hits += glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "../../../../../../../.agents/skills/"
                                   "ops-precision-standard/scripts/mixed_tolerance_check.py"))
    if not hits:
        pytest.skip("ops-precision-standard checker not present in this workspace")
    spec_mod = importlib.util.spec_from_file_location("mixed_tolerance_check", hits[0])
    mod = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(mod)

    # tier table drift (via the checker's own lookup where numpy supports the dtype,
    # else its table directly - numpy here has no bfloat16 dtype)
    for name, torch_dtype in (("float16", torch.float16), ("float32", torch.float32),
                              ("bfloat16", torch.bfloat16)):
        try:
            rtol, atol, req, fixed = mod.get_tolerance_by_dtype(np.dtype(name))
        except (ValueError, TypeError):
            rtol, atol, req, fixed = mod._TOLERANCE_TABLE[name]
        ours = std_spec(torch_dtype)
        assert (ours["rtol"], ours["atol"], ours["required_matched_ratio"],
                ours["fixed_limit"]) == (rtol, atol, req, fixed), name
        assert ours["ulp_at_one"] == mod._ULP_TABLE[name], name

    golden = (np.ones(4096, dtype=np.float32) * 2.0).astype(np.float16)
    actual = golden.copy()
    actual[::1024] = (golden[::1024].astype(np.float32) + 0.01).astype(np.float16)
    theirs = mod.check_mixed_tolerance(actual, golden)
    spec = std_spec(torch.float16)
    m = metrics(torch.from_numpy(actual.copy()), torch.from_numpy(golden.copy()),
                spec["atol"], spec["rtol"])
    assert m["matched_ratio"] == pytest.approx(theirs["matched_ratio"], rel=1e-12)
    assert m["MaxAbsErr"] == pytest.approx(theirs["max_abs_error"], rel=1e-12)
    assert verdict(m, spec)[0] == theirs["is_pass"]


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
