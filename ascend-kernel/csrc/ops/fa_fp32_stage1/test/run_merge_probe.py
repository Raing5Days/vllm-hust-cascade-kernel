# Copyright (c) 2026 Huawei Technologies Co., Ltd
# BSD 3-Clause License.
#
# S5 end-to-end merge probe (plan-20260903 §2.3): the direct Tier-1 numerical claim.
#   stage-1 = fa_fp32_stage1 (real, fp32 out + padded LSE)
#   stage-2 = FIA v2 (real CANN kernel, bf16 out + fp32 compact LSE) on suffix-only paged KV
#   merge   = torch.ops.npu.lse_merge(o1_fp32, o2_bf16, l1_padded, l2_compact, out_code=2)
# Full chain vs fp64 reference over prefix+suffix; residual expectation ~ w2*eps2
# (~1.4e-4 relative at S_P=8192/S_S=512); Tier0 (both-outputs-bf16) arm as control.
# Usage: python run_merge_probe.py <device_id>

import sys
import os
import numpy as np
import ml_dtypes
import torch
import torch_npu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fa_fp32_ref import fa_fp64_ref, to_bf16_bytes

B, S_P, S_S = 64, 8192, 512
H, KVH, D, BLK = 40, 8, 128, 128


def np2t(x):
    x = np.ascontiguousarray(np.asarray(x))
    u16 = x.view(np.uint16)
    return torch.from_numpy(u16).view(torch.bfloat16).to("npu")


def main():
    dev_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    torch.npu.set_device(dev_id)
    import ascend_kernel

    rng = np.random.default_rng(7)
    scale = 1.0 / np.sqrt(D)
    kv_total = S_P + S_S

    # one logical kv sequence shared by all requests (cascade shared-prefix semantics),
    # stored as a single 68-block pool; prefix pool = blocks 0..63, suffix = last 4.
    n_blocks = kv_total // BLK
    kv_pool = rng.standard_normal((n_blocks, BLK, KVH, D)).astype(np.float32)
    v_pool = rng.standard_normal((n_blocks, BLK, KVH, D)).astype(np.float32)
    q_np = rng.standard_normal((B, H, D)).astype(np.float32)
    kv_pool16, v_pool16, q16 = to_bf16_bytes(kv_pool), to_bf16_bytes(v_pool), to_bf16_bytes(q_np)

    bt_full = np.tile(np.arange(n_blocks, dtype=np.int32), (B, 1))
    bt_pre = bt_full[:, : S_P // BLK]

    dev = "npu"
    q_t = np2t(q16)
    # stage-1: prefix only
    o1, l1 = torch.ops.npu.fa_fp32_stage1(
        q_t, np2t(kv_pool16[: S_P // BLK]), np2t(v_pool16[: S_P // BLK]),
        torch.from_numpy(bt_pre).to(dev),
        torch.from_numpy(np.ones(B, dtype=np.int64)).to(dev),
        torch.from_numpy(np.full(B, S_P, dtype=np.int64)).to(dev))
    torch.npu.synchronize()
    o1_np = o1.cpu().numpy().astype(np.float64)
    l1_np = l1.cpu().numpy().reshape(-1, 8)[:, 0].reshape(B, H).astype(np.float64)

    # stage-2: suffix via real FIA v2 (paged, bf16 out, fp32 lse)
    suf_off = S_P // BLK
    pool_k_fia = np2t(kv_pool16[suf_off:].transpose(0, 2, 1, 3))  # [nb, KVH, BLK, D]
    pool_v_fia = np2t(v_pool16[suf_off:].transpose(0, 2, 1, 3))
    bt_suf = torch.from_numpy(np.tile(np.arange(S_S // BLK, dtype=np.int32), (B, 1))).to(dev)
    o2, l2 = torch_npu.npu_fused_infer_attention_score_v2(
        q_t.view(B, H, 1, D), pool_k_fia, pool_v_fia,
        num_query_heads=H, num_key_value_heads=KVH,
        input_layout="BNSD", softmax_scale=scale,
        block_table=bt_suf, block_size=BLK,
        actual_seq_qlen=[1] * B, actual_seq_kvlen=[S_S] * B,
        return_softmax_lse=True)
    torch.npu.synchronize()
    o2_np = o2.float().cpu().numpy().reshape(B, H, D).astype(np.float64)
    l2_np = l2.float().cpu().numpy().reshape(B, H).astype(np.float64)

    # fp64 references
    o_pre_ref, l_pre_ref = fa_fp64_ref(q16.astype(np.float64), kv_pool16[: suf_off].astype(np.float64),
                                       v_pool16[: suf_off].astype(np.float64), bt_pre,
                                       np.ones(B, dtype=np.int64), np.full(B, S_P, dtype=np.int64),
                                       D, H, KVH)
    o_full_ref, l_full_ref = fa_fp64_ref(q16.astype(np.float64), kv_pool16.astype(np.float64),
                                         v_pool16.astype(np.float64), bt_full,
                                         np.ones(B, dtype=np.int64), np.full(B, kv_total, dtype=np.int64),
                                         D, H, KVH)

    def rel(x, ref):
        m = np.abs(ref) >= 1e-2
        return float((np.abs(x - ref)[m] / np.abs(ref)[m]).max()), float(np.abs(x - ref).max())

    # stage sanity
    r1 = rel(o1_np, o_pre_ref)
    r2 = rel(o2_np, o_full_ref - 0)  # placeholder replaced below
    # suffix-only reference (manual: full softmax restricted to suffix rows)
    # = softmax over suffix keys only
    suf_ref, suf_lse_ref = fa_fp64_ref(q16.astype(np.float64),
                                       kv_pool16[suf_off:].astype(np.float64),
                                       v_pool16[suf_off:].astype(np.float64),
                                       bt_suf_cpu := np.tile(np.arange(S_S // BLK, dtype=np.int32), (B, 1)),
                                       np.ones(B, dtype=np.int64), np.full(B, S_S, dtype=np.int64),
                                       D, H, KVH)
    r2 = rel(o2_np, suf_ref)
    r2l = float(np.abs(l2_np - suf_lse_ref).max())
    r1l = float(np.abs(l1_np - l_pre_ref).max())
    print(f"stage-1 sanity: O rel(max,|ref|>=1e-2)={r1[0]:.3e} abs={r1[1]:.3e} LSE abs={r1l:.3e}")
    print(f"stage-2 (FIA bf16) sanity: O rel={r2[0]:.3e} abs={r2[1]:.3e} LSE abs={r2l:.3e}")

    # merge: Tier1 (o1 fp32) and Tier0 control (o1 bf16)
    o1_bf16 = o1.to(torch.bfloat16)
    m_t1 = torch.ops.npu.lse_merge(o1, o2.reshape(B, H, D), l1, l2.reshape(-1), 2)
    m_t0 = torch.ops.npu.lse_merge(o1_bf16, o2.reshape(B, H, D), l1, l2.reshape(-1), 2)
    torch.npu.synchronize()
    m_t1_np = m_t1.cpu().numpy().astype(np.float64)
    m_t0_np = m_t0.cpu().numpy().astype(np.float64)

    t1 = rel(m_t1_np, o_full_ref)
    t0 = rel(m_t0_np, o_full_ref)
    # bf16 deployment form (final cast)
    m_t1_bf = np.asarray(m_t1_np, dtype=ml_dtypes.bfloat16).astype(np.float64)
    t1_bf = rel(m_t1_bf, o_full_ref)
    w2 = S_S / kv_total
    theory = w2 * 2e-3
    # The Tier-1 claim (plan-20260902 §1): the stage-2 bf16 error enters the
    # merged result suppressed by the softmax weight w2, and the stage-1 bf16
    # rounding term (w1*eps1, ~94% of the error weight) is eliminated entirely.
    # Direct evidence: suppression factor = merged_err / stage2_err ~= w2.
    supp_t1 = t1[1] / max(r2[1], 1e-12)
    supp_t0 = t0[1] / max(r2[1], 1e-12)
    print(f"Tier1 merge (fp32 out):  abs={t1[1]:.3e} rel(|ref|>=1e-2)={t1[0]:.3e}")
    print(f"Tier0 merge (o1 bf16):   abs={t0[1]:.3e} rel(|ref|>=1e-2)={t0[0]:.3e}")
    print(f"stage-2 err abs={r2[1]:.3e}; suppression merged/stage2: "
          f"Tier1={supp_t1:.3f} (theory w2={w2:.3f}), Tier0={supp_t0:.3f}")
    print(f"Tier1 -> bf16 cast (deployment form): abs={t1_bf[1]:.3e} "
          f"(final bf16 cast floor ~1 ulp of merged)")
    ok = t1[1] < t0[1] and supp_t1 <= 2 * w2 and r1l < 1e-4 and r2l < 1e-3
    print("MERGE PROBE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
