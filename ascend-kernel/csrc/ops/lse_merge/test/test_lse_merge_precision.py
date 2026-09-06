"""lse_merge precision tests (ascendc-operator-precision-eval skill flow).
MERE/MARE thresholds per the ecosystem standard: bf16-out 7.81e-3 / 7.81e-2,
fp32-out 1.22e-4 / 1.22e-3. Run on an idle NPU (device fixed via DEV)."""
import os
import sys
import zlib

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
import ascend_kernel  # noqa: F401  registers torch.ops.npu.lse_merge
from lse_merge_cases import DEV, MODES, SHAPES, boundary_cases, fp64_ref, make_inputs


def _metrics(out, ref, mare_floor=0.0):
    rel = (out.double() - ref).abs() / (ref.abs() + 1e-7)
    if mare_floor > 0:
        dom = ref.abs() >= mare_floor
        mare = (rel * dom).max().item() if dom.any() else 0.0
        floor_abs = ((out.double() - ref).abs() * ~dom).max().item() if (~dom).any() else 0.0
    else:
        mare = rel.max().item()
        floor_abs = 0.0
    return {"MERE": rel.mean().item(), "MARE": mare,
            "MaxAbsErr": (out.double() - ref).abs().max().item(),
            "MaxAbsErr_below_floor": floor_abs}


@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("shape", SHAPES)
def test_shape(mode, shape):
    cat, (T, H, D) = shape
    o1_dtype, _, out_code, padded, mere_tol, mare_tol, mare_floor = MODES[mode]
    seed = zlib.crc32(f"{cat}/{mode}".encode()) % 2**31
    o1, o2, l1c, l2c, l1, l2 = make_inputs(T, H, D, o1_dtype, padded, seed=seed)
    out = torch.ops.npu.lse_merge(o1, o2, l1c, l2c, out_code)
    torch.npu.synchronize()
    assert out.dtype == (torch.bfloat16 if mode in ("A_tier0", "C_hybrid_bf16", "D_tier0_pad")
                         else torch.float32)
    ref = fp64_ref(o1.cpu(), o2.cpu(), l1.cpu(), l2.cpu()).to(out.dtype).cpu()
    m = _metrics(out.cpu(), ref.double(), mare_floor)
    abs_bound = 1e-5 if out.dtype == torch.float32 else 2e-2  # bf16 ULP(2.0)/2 ~ 7.8e-3
    assert m["MERE"] < mere_tol, f"{mode}/{cat}: MERE={m['MERE']:.3e} >= {mere_tol}"
    assert m["MARE"] < mare_tol, f"{mode}/{cat}: MARE={m['MARE']:.3e} >= {mare_tol}"
    assert m["MaxAbsErr"] < abs_bound, f"{mode}/{cat}: MaxAbsErr={m['MaxAbsErr']:.3e} >= {abs_bound}"


@pytest.mark.parametrize("mode", ["A_tier0", "B_fp32out", "C_hybrid_bf16", "Bp_fp32out_pad"])
@pytest.mark.parametrize("bname,bfn", boundary_cases())
def test_boundary(mode, bname, bfn):
    T, H, D = 64, 40, 128
    o1_dtype, _, out_code, padded, mere_tol, mare_tol, mare_floor = MODES[mode]
    o1, o2, l1c, l2c, l1, l2 = make_inputs(T, H, D, o1_dtype, padded, seed=zlib.crc32(bname.encode()) % 2**31)
    bfn(o1, o2, l1, l2)
    if padded:  # re-pack the mutated compact lse into padded rows
        l1c = l1.unsqueeze(-1).expand(T, H, 8).contiguous().reshape(T, H * 8)
        l2c = l2.unsqueeze(-1).expand(T, H, 8).contiguous().reshape(T, H * 8)
    out = torch.ops.npu.lse_merge(o1, o2, l1c, l2c, out_code)
    torch.npu.synchronize()
    ref = fp64_ref(o1.cpu(), o2.cpu(), l1.cpu(), l2.cpu()).to(out.dtype).cpu()
    m = _metrics(out.cpu(), ref.double(), mare_floor)
    abs_bound = 1e-5 if out.dtype == torch.float32 else 2e-2  # bf16 ULP(2.0)/2 ~ 7.8e-3
    assert m["MERE"] < mere_tol, f"{mode}/{bname}: MERE={m['MERE']:.3e} >= {mere_tol}"
    assert m["MARE"] < mare_tol, f"{mode}/{bname}: MARE={m['MARE']:.3e} >= {mare_tol}"
    assert m["MaxAbsErr"] < abs_bound, f"{mode}/{bname}: MaxAbsErr={m['MaxAbsErr']:.3e} >= {abs_bound}"


def test_out_code_validation():
    o1 = torch.randn(2, 4, 16, dtype=torch.bfloat16, device=DEV)
    o2 = torch.randn_like(o1)
    l1 = torch.randn(2, 4, dtype=torch.float32, device=DEV)
    l2 = torch.randn_like(l1)
    with pytest.raises(Exception):
        torch.ops.npu.lse_merge(o1, o2, l1, l2, 3)   # invalid out_code
    with pytest.raises(Exception):
        torch.ops.npu.lse_merge(o1.float(), o2.float(), l1, l2)  # o2 must be bf16


def test_padded_lse_matches_compact_bitwise():
    """Padded-LSE path must produce the same merged values as compact (stride
    independence), bit-for-bit, on the production shape."""
    T, H, D = 64, 40, 128
    o1 = torch.randn(T, H, D, dtype=torch.float32).to(torch.bfloat16).to(DEV)
    o2 = torch.randn(T, H, D, dtype=torch.bfloat16, device=DEV)
    l1 = torch.randn(T, H, dtype=torch.float32, device=DEV)
    l2 = torch.randn_like(l1)
    l1p = l1.unsqueeze(-1).expand(T, H, 8).contiguous().reshape(T, H * 8)
    l2p = l2.unsqueeze(-1).expand(T, H, 8).contiguous().reshape(T, H * 8)
    a = torch.ops.npu.lse_merge(o1, o2, l1, l2, 2)
    b = torch.ops.npu.lse_merge(o1, o2, l1p, l2p, 2)
    torch.npu.synchronize()
    assert torch.equal(a.cpu(), b.cpu()), "padded vs compact LSE results differ"
