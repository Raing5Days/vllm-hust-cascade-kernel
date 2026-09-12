"""pytest wrapper for the add_rms_norm_stats precision suite.

The full sweep (64 reference cases + 48 production-oracle cases, JSON/Markdown
report) lives in run_precision_report.py; this file re-runs a representative
slice as pytest cases so the operator is covered by the repo-wide `pytest -q`
gate, and adds the negative (rejection cases) of test-cases.md 5.

NPU resource discipline: this file executes real device work (including the
M=2048 x K=5120 production shapes), so wrap it in the shared lock when other
tasks are running: `flock /tmp/w3-npu.lock python -m pytest -q <this file>`.

On a machine without an NPU the whole module is skipped.
"""
import os
import sys
import zlib

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch_npu = pytest.importorskip("torch_npu")

try:
    import ascend_kernel  # noqa: F401
except (ImportError, RuntimeError) as exc:  # pragma: no cover - env guard
    pytest.skip(f"ascend_kernel not importable: {exc}", allow_module_level=True)

if not torch.npu.is_available():
    pytest.skip("no NPU device", allow_module_level=True)

from add_rms_norm_stats_cases import SHAPES, make_inputs
from add_rms_norm_stats_ref import cpu_ref, judge, metrics
from run_precision_report import run_case

CASES = [
    (mode, dtype, name, shape, has_beta)
    for mode in (0, 1, 2)
    for dtype in (torch.bfloat16,)          # bf16 covers the production dtype
    for name, shape in SHAPES
    for has_beta in ([False, True] if mode == 1 else [False])
]


@pytest.mark.parametrize("mode,dtype,name,shape,has_beta", CASES)
def test_precision(mode, dtype, name, shape, has_beta):
    seed = zlib.crc32(f"{mode}/{dtype}/{name}/{has_beta}".encode()) % 2 ** 31
    row = run_case(mode, dtype, name, shape, has_beta, seed)
    assert row["pass"], row


def test_oracle_matches_cann_on_prefill_shape():
    """The production oracle cross-check on the F2 main shape (bf16, beta=None)."""
    from run_precision_report import run_oracle
    for mode in (0, 1):
        row = run_oracle(mode, torch.bfloat16, "prefill-main", (2048, 5120), 7)
        assert row["pass"], row


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_rstd_padding_rows_are_zero(mode):
    x1, x2, gamma, _ = make_inputs(77, 128, torch.bfloat16, 11, has_beta=False)
    _, rstd, _ = torch.ops.npu.add_rms_norm_stats(x1, x2, gamma, None, 1e-6, mode)
    torch.npu.synchronize()
    assert rstd.shape[0] == 80  # ceil(77/8)*8
    assert (rstd[77:].abs() == 0).all().item()


def _expect_reject(fn, msg):
    with pytest.raises(RuntimeError, match=msg):
        fn()
        torch.npu.synchronize()


def test_negative_cases():
    m, k = 64, 128
    x1, x2, gamma, _ = make_inputs(m, k, torch.bfloat16, 5, has_beta=False)
    op = torch.ops.npu.add_rms_norm_stats
    _expect_reject(lambda: op(x1[:, :64].t().contiguous().t(), x2, gamma, None, 1e-6, 0),
                   "must be contiguous")
    _expect_reject(lambda: op(x1, x2[:, :64].contiguous(), gamma, None, 1e-6, 0),
                   "same shape")
    _expect_reject(lambda: op(x1, x2.to(torch.float16), gamma, None, 1e-6, 0), "same shape")
    _expect_reject(lambda: op(torch.randn(m, 130, device=x1.device).to(torch.bfloat16), x2,
                              gamma, None, 1e-6, 0), "multiple of 16")
    _expect_reject(lambda: op(torch.randn(m, 6000, device=x1.device).to(torch.bfloat16),
                              torch.randn(m, 6000, device=x1.device).to(torch.bfloat16),
                              gamma, None, 1e-6, 0), "K must be in")
    _expect_reject(lambda: op(x1, x2, gamma, None, 1e-6, 3), "mode must be")
    _expect_reject(lambda: op(x1, x2, gamma, None, 0.0, 0), "eps must be")
    _expect_reject(lambda: op(x1, x2, None, None, 1e-6, 1), "mode 1 needs")
    _expect_reject(lambda: op(x1.to(torch.float32), x2.to(torch.float32), gamma, None, 1e-6, 0),
                   "bf16 or fp16")


def test_reference_metric_helper_is_sane():
    """CPU-only sanity check: the metric helper and judgement agree on a match."""
    a = torch.randn(16, 64, dtype=torch.bfloat16)
    b = (torch.randn(16, 64) * 0.5).to(torch.bfloat16)
    xo, rstd, _ = cpu_ref(a, b, None, None, 1e-6, 0)
    mtr = metrics(xo, xo)
    assert mtr["MaxAbsErr"] == 0.0
    ok, _ = judge(torch.bfloat16, "x_out", mtr)
    assert ok
    assert rstd.shape == (16, 1)
