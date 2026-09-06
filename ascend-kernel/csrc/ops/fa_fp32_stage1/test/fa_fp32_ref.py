# Copyright (c) 2026 Huawei Technologies Co., Ltd
# BSD 3-Clause License.
#
# fp64 semantic reconstruction of the catlass example-23 FA-infer kernel
# (fa_fp32_stage1) arithmetic, plus input generators for the precision suite.
#
# Kernel semantics reproduced here (from block_epilogue_online_softmax_no_mask.hpp
# + block_epilogue_rescale_o_no_split_row.hpp, plan-20260903 M-B):
#   - inputs bf16 -> upcast, S = (q @ k^T) * scale in fp32 (reference: fp64)
#   - online softmax over kv in stack tiles of 512 (4 paged blocks), last tile =
#     kv - 512*k:  hm = max(lm, gm_old); dm = exp(gm_old - hm);
#     ls = exp(S - hm); P = bf16_cast_rint(ls) (quantized for the PV mmad);
#     gl = dm * gl_old + rowsum(ls)  (UNquantized fp32 row sums);
#     O = dm * O_old + P @ V (P quantized, V bf16 upcast)
#   - final O = O / gl; LSE = ln(gl) + hm  (32B padded rows, elem0 = value)
# The fp32 accumulation-order noise (~1e-6) is NOT reproduced (probe口径:
# kernel vs this reference maxAbs ~1.9e-6, dominated by exp/mmad ulp noise).
# Vectorized over (token, head) per request; per-row scalar version kept as
# fa_fp64_ref_rowwise (cross-check).

import numpy as np
import ml_dtypes


def bf16(x):
    """Round-to-nearest-even bf16 quantization, returned as fp64 values."""
    return np.asarray(x, dtype=ml_dtypes.bfloat16).astype(np.float64)


def gen_inputs(batch, kv_seqlens, num_heads, kv_heads, embed, seed=0,
               q_len=1, block_cols=None, share_blocks=False):
    """Build paged-KV inputs. Per-request distinct blocks by default;
    share_blocks=True gives every request the same block row (cascade
    shared-prefix semantics)."""
    rng = np.random.default_rng(seed)
    kv_seqlens = list(kv_seqlens)
    assert len(kv_seqlens) == batch
    block_size = 128
    if block_cols is None:
        block_cols = max((kv + block_size - 1) // block_size for kv in kv_seqlens)
    blocks_per_req = [(kv + block_size - 1) // block_size for kv in kv_seqlens]
    num_blocks = max(blocks_per_req) if share_blocks else sum(blocks_per_req)

    k = rng.standard_normal((num_blocks, block_size, kv_heads, embed)).astype(np.float32)
    v = rng.standard_normal((num_blocks, block_size, kv_heads, embed)).astype(np.float32)

    q = rng.standard_normal((batch * q_len, num_heads, embed)).astype(np.float32)

    q_seqlens = np.full(batch, q_len, dtype=np.int64)
    kv_arr = np.asarray(kv_seqlens, dtype=np.int64)

    block_table = np.zeros((batch, block_cols), dtype=np.int32)
    nxt = 0
    for b, (kv, nb) in enumerate(zip(kv_seqlens, blocks_per_req)):
        if share_blocks:
            block_table[b, :nb] = np.arange(nb, dtype=np.int32)
        else:
            block_table[b, :nb] = np.arange(nxt, nxt + nb, dtype=np.int32)
            nxt += nb
    num_tokens = int(q_seqlens.sum())
    return q, k, v, block_table, q_seqlens, kv_arr, num_tokens


def to_bf16_bytes(x):
    return np.asarray(x, dtype=ml_dtypes.bfloat16)


def _gather_kv(k64, v64, block_table_row, kv):
    """Gather one request's kv from the paged pool -> [kv, KVH, D] each.
    Block i stores logical rows [i*bs, (i+1)*bs) at in-block rows [0, valid)."""
    block_size = k64.shape[1]
    nblk = (kv + block_size - 1) // block_size
    bt = block_table_row[:nblk].astype(np.int64)
    ks, vs = [], []
    for i in range(nblk):
        valid = min(block_size, kv - i * block_size)
        ks.append(k64[bt[i], :valid])
        vs.append(v64[bt[i], :valid])
    return np.concatenate(ks), np.concatenate(vs)


def fa_fp64_ref(q, k, v, block_table, q_seqlens, kv_seqlens, embed, num_heads, kv_heads,
                chunk=512):
    """Vectorized fp64 reference. q: [T,H,D] (bf16 values as fp32/fp64 array),
    k/v: [numBlocks, blockSize, KVH, D]. Returns (O [T,H,D] fp64, lse [T,H] fp64)."""
    scale = 1.0 / np.sqrt(embed)
    q64 = np.asarray(q, dtype=np.float64)
    k64 = np.asarray(k, dtype=np.float64)
    v64 = np.asarray(v, dtype=np.float64)
    T = q64.shape[0]
    group = num_heads // kv_heads

    O = np.zeros((T, num_heads, embed), dtype=np.float64)
    LSE = np.zeros((T, num_heads), dtype=np.float64)

    tok = 0
    for b in range(len(q_seqlens)):
        kv = int(kv_seqlens[b])
        qlen = int(q_seqlens[b])
        kb, vb = _gather_kv(k64, v64, block_table[b], kv)  # [kv, KVH, D]
        # per-head kv: [H, kv, D]
        kh = kb[:, np.arange(num_heads) // group, :]  # [kv, H, D]
        vh = vb[:, np.arange(num_heads) // group, :]
        qb = q64[tok:tok + qlen].transpose(1, 0, 2)  # [H, T_b, D]

        # S = (q @ k^T) * scale, per head: [H, T_b, kv]
        S = np.einsum('htd,khd->htk', qb, kh) * scale

        m = np.full((num_heads, qlen), -np.inf)
        gl = np.zeros((num_heads, qlen))
        acc = np.zeros((num_heads, qlen, embed))
        for s in range(0, kv, chunk):
            e = min(kv, s + chunk)
            sc = S[:, :, s:e]
            m_new = np.maximum(m, sc.max(axis=2))
            dm = np.where(s == 0, 1.0, np.exp(m - m_new))
            ls = np.exp(sc - m_new[:, :, None])   # fp32 exp in kernel
            p = bf16(ls)                          # P quantization point
            gl = dm * gl + ls.sum(axis=2)
            # acc = dm*acc + P @ V  (einsum over chunk)
            acc = dm[:, :, None] * acc + np.einsum('htk,khd->htd', p, vh[s:e])
            m = m_new
        O[tok:tok + qlen] = (acc / gl[:, :, None]).transpose(1, 0, 2)
        LSE[tok:tok + qlen] = (np.log(gl) + m).T
        tok += qlen
    return O, LSE


def _bf16_round_and_neighbors(ls):
    """Vectorized bf16 nearest rounding + neighbors + midpoint distance.
    Returns (p_nearest fp64, g_lo fp64, g_hi fp64, midpoint fp64, ulp fp64)."""
    x32 = np.asarray(ls, dtype=np.float32)
    u = x32.view(np.uint32)
    # round-to-nearest-even bf16: add rounding bias then truncate
    rne_u = (u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))) & np.uint32(0xFFFF0000)
    p = rne_u.view(np.float32).astype(np.float64)
    # the two adjacent bf16 grid values around ls (fp32 -> bf16 grid)
    up_u = (rne_u + np.uint32(1 << 16)).view(np.float32).astype(np.float64)
    dn_u = (rne_u - np.uint32(1 << 16)).view(np.float32).astype(np.float64)
    g_hi = np.where(p <= ls, up_u, p)
    g_lo = np.where(p <= ls, p, dn_u)
    # fix rounding-down-at-boundary: nearest may equal g_hi when ls below grid point
    g_hi = np.where(p < ls, up_u, p)
    g_lo = np.where(p < ls, p, dn_u)
    midpoint = (g_lo + g_hi) / 2.0
    ulp = g_hi - g_lo
    return p, g_lo, g_hi, midpoint, ulp


def fa_fp64_ref_tieband(q, k, v, block_table, q_seqlens, kv_seqlens, embed, num_heads,
                        kv_heads, chunk=512, tie_band_rel=2e-5):
    """fa_fp64_ref + per-element bf16 rounding-ambiguity bound on the numerator.

    The kernel computes S in fp32 (cube mmad) while this reference is exact fp64;
    a P element whose fp64 value sits within the S-noise band of a bf16 rounding
    midpoint may legitimately round either way. Returns (O, LSE, A) with A
    [T,H,D] = upper bound of |num_kernel - num_ref| from such near-tie flips:
    A_j = sum over ambiguous i of ulp(ls_i) * |V_ij|. The honest acceptance gate
    is |O_kernel - O_ref| <= A/gl + fp32_noise_floor.
    tie_band_rel: |ls - midpoint| <= tie_band_rel * ls marks an element ambiguous
    (default 2e-5 ~= fp32 mmad S noise incl. safety factor)."""
    scale = 1.0 / np.sqrt(embed)
    q64 = np.asarray(q, dtype=np.float64)
    k64 = np.asarray(k, dtype=np.float64)
    v64 = np.asarray(v, dtype=np.float64)
    T = q64.shape[0]
    group = num_heads // kv_heads

    O = np.zeros((T, num_heads, embed), dtype=np.float64)
    LSE = np.zeros((T, num_heads), dtype=np.float64)
    A = np.zeros((T, num_heads, embed), dtype=np.float64)

    tok = 0
    for b in range(len(q_seqlens)):
        kv = int(kv_seqlens[b])
        qlen = int(q_seqlens[b])
        kb, vb = _gather_kv(k64, v64, block_table[b], kv)
        kh = kb[:, np.arange(num_heads) // group, :]
        vh = vb[:, np.arange(num_heads) // group, :]
        qb = q64[tok:tok + qlen].transpose(1, 0, 2)

        S = np.einsum('htd,khd->htk', qb, kh) * scale

        m = np.full((num_heads, qlen), -np.inf)
        gl = np.zeros((num_heads, qlen))
        acc = np.zeros((num_heads, qlen, embed))
        amb = np.zeros((num_heads, qlen, embed))
        for s in range(0, kv, chunk):
            e = min(kv, s + chunk)
            sc = S[:, :, s:e]
            m_new = np.maximum(m, sc.max(axis=2))
            dm = np.where(s == 0, 1.0, np.exp(m - m_new))
            ls = np.exp(sc - m_new[:, :, None])
            p, g_lo, g_hi, mid, ulp = _bf16_round_and_neighbors(ls)
            ambiguous = np.abs(ls - mid) <= tie_band_rel * ls
            gl = dm * gl + ls.sum(axis=2)
            acc = dm[:, :, None] * acc + np.einsum('htk,khd->htd', p, vh[s:e])
            # ambiguity bound in numerator space (both rounding directions legitimate)
            amb += dm[:, :, None] * np.einsum('htk,khd->htd',
                                              ambiguous * (g_hi - g_lo), np.abs(vh[s:e]))
            m = m_new
        O[tok:tok + qlen] = (acc / gl[:, :, None]).transpose(1, 0, 2)
        LSE[tok:tok + qlen] = (np.log(gl) + m).T
        A[tok:tok + qlen] = (amb / gl[:, :, None]).transpose(1, 0, 2)
        tok += qlen
    return O, LSE, A


def fa_fp64_ref_rowwise(q, k, v, block_table, q_seqlens, kv_seqlens, embed, num_heads,
                        kv_heads, chunk=512):
    """Scalar per-row reference (slow; cross-check of fa_fp64_ref)."""
    scale = 1.0 / np.sqrt(embed)
    q64 = np.asarray(q, dtype=np.float64)
    k64 = np.asarray(k, dtype=np.float64)
    v64 = np.asarray(v, dtype=np.float64)
    T = q64.shape[0]
    O = np.zeros((T, num_heads, embed), dtype=np.float64)
    LSE = np.zeros((T, num_heads), dtype=np.float64)
    tok = 0
    group = num_heads // kv_heads
    for b in range(len(q_seqlens)):
        kv = int(kv_seqlens[b])
        kb, vb = _gather_kv(k64, v64, block_table[b], kv)
        qlen = int(q_seqlens[b])
        for t_local in range(qlen):
            qrow = q64[tok + t_local]
            for h in range(num_heads):
                kvh = h // group
                kk, vv = kb[:, kvh, :], vb[:, kvh, :]
                m, gl = -np.inf, 0.0
                acc = np.zeros(embed)
                for s in range(0, kv, chunk):
                    e = min(kv, s + chunk)
                    sc = (kk[s:e] @ qrow[h]) * scale
                    m_new = max(m, sc.max())
                    dm = np.exp(m - m_new) if s > 0 else 1.0
                    ls = np.exp(sc - m_new)
                    p = bf16(ls)
                    gl = dm * gl + ls.sum()
                    acc = dm * acc + p @ vv[s:e]
                    m = m_new
                O[tok + t_local, h] = acc / gl
                LSE[tok + t_local, h] = np.log(gl) + m
        tok += qlen
    return O, LSE
