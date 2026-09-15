// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPS_H
#define OPS_H

#include <tuple>

namespace ascend_kernel {

at::Tensor lse_merge(const at::Tensor &o1, const at::Tensor &o2, const at::Tensor &lse1,
                     const at::Tensor &lse2, int64_t out_code = 0);

// Stage-1 fp32-out flash attention (cascade shared-prefix stage; paged, no mask,
// GQA). Returns (out fp32 [T, H, D], lse fp32 [T*H*8] with 32B padded rows).
// q_seqlen_value (default 0 = legacy D2H path): when > 0, every request's q
// seqlen equals this value (decode = 1) - the host skips the seqlen D2H pulls
// and computes the tiling purely from shapes (graph-capture-safe); the sumQ
// check becomes T == batch * q_seqlen_value and the ceil(maxKv/128) <= cols
// check becomes a caller contract (kv seqlens are read by the kernel on
// device).
std::tuple<at::Tensor, at::Tensor> fa_fp32_stage1(const at::Tensor &query, const at::Tensor &key,
                                                  const at::Tensor &value, const at::Tensor &block_table,
                                                  const at::Tensor &actual_q_seqlens,
                                                  const at::Tensor &actual_kv_seqlens,
                                                  int64_t q_seqlen_value = 0);

// F2 fusion "norm stage" (see ops/add_rms_norm_stats/design.md): the part of an
// AddRmsNormBias -> GEMM fusion that stays exposed outside the GEMM, split out so
// each candidate fusion topology can be measured with real silicon numbers.
// mode 0: (x_out, rstd) = residual add + row statistics;
// mode 1: (rstd, y)     = row statistics + norm apply (reads the rounded residual);
// mode 2: (rstd)        = row statistics only.
// Unused outputs come back as 0-element tensors. rstd is padded to ceil(M/8) rows
// (32B MTE3 groups); rows >= M are padding.
std::tuple<at::Tensor, at::Tensor, at::Tensor> add_rms_norm_stats(
    const at::Tensor &x1, const at::Tensor &x2, const c10::optional<at::Tensor> &gamma,
    const c10::optional<at::Tensor> &beta, double eps, int64_t mode);

// D3 Phase-0 benchmark probe (see ops/bw_probe/op_kernel header): the mode-0
// memory pattern with the compute removed stage by stage, to measure the
// achievable ceiling of that pattern. variant 0 = 2 reads + 1 write and zero
// compute (the anchor), 1 = + the apply path, 2 = + the statistics path (must
// reproduce add_rms_norm_stats mode 0's device time, i.e. it is the probe's own
// validity check). Benchmark-only: only `out` is defined for variants 0/1.
// Returns (out, rstd).
std::tuple<at::Tensor, at::Tensor> bw_probe(const at::Tensor &x1, const at::Tensor &x2,
                                            int64_t variant, double eps);

// F3 (SwiGLU into the gate_up GEMM epilogue) Phase B ladder; see
// ops/f3_floor/design.md. Single catlass-backed AIC/AIV kernel with 8 modes
// (0..5, 7, 8; 6 unused):
//   mode 0 floor : gate_up GEMM + "read the gate/up windows, write D = gate"
//                  (the epilogue's exact byte pattern with zero compute)
//   mode 1 swiglu: mode 0 with D = SiLU(gate) * up   (hook only; NOT implemented)
//   mode 2 gemm  : the GEMM alone (AIV only consumes the cross-core flags)
//   mode 3 epi   : the floor epilogue alone (AIC only raises the flags)
//   mode 4 floor_up  : D = up window       (proves the +I window addressing)
//   mode 5 gate_only : read the gate window only, write D
//   mode 7 read_only : read gate+up, write nothing  (read/write attribution)
//   mode 8 write_only: read nothing, write D        (read/write attribution)
// Measurement-only: no e2e consumer, and none of these modes belongs to any
// plugin bundle. `out_d` selects the D landing spot (a fresh tensor vs an
// in-place view with row stride 2I); `workspace` ([M, 2I] bf16) may be passed in
// to reuse/prime it; the C workspace is returned.
std::tuple<at::Tensor, at::Tensor> f3_gateup_epilogue(const at::Tensor &a, const at::Tensor &b,
                                                     const c10::optional<at::Tensor> &workspace,
                                                     const c10::optional<at::Tensor> &out_d,
                                                     int64_t mode);

// fia_grain_floor: MEASUREMENT-ONLY copy of fa_fp32_stage1 (see
// ops/fia_grain_floor/design.md) used to sweep the KV搬运粒度 (blockStackNum,
// source-local define B1_GRAIN_STACK_NUM). Same signature and semantics as
// fa_fp32_stage1: returns (out fp32 [T, H, D], lse fp32 [T*H*8]). Not wired into
// any e2e path and not part of any plugin bundle.
std::tuple<at::Tensor, at::Tensor> fia_grain_floor(const at::Tensor &query, const at::Tensor &key,
                                                   const at::Tensor &value, const at::Tensor &block_table,
                                                   const at::Tensor &actual_q_seqlens,
                                                   const at::Tensor &actual_kv_seqlens,
                                                   int64_t q_seqlen_value = 0);

} // namespace ascend_kernel

#endif // OPS_H
