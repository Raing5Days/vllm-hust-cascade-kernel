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

// add_rms_norm_stats - F2 fusion "norm stage" host side (design.md §1/§5).
//
// Three modes over (M, K) contiguous rows; all outputs are allocated here and
// the unused ones come back as 0-element tensors (so no 20MB allocation is paid
// for a tensor the mode never writes):
//   mode 0: x_out = round_dtype(x1+x2), rstd = rms_stats(x_out)
//   mode 1: rstd = rms_stats(x1),        y     = round_dtype(round_dtype(x1*rstd)*gamma+beta)
//   mode 2: rstd = rms_stats(round_dtype(x1+x2)), no other output
//
// rstd is padded to ceil(M/8) rows: every core writes its rows in 32B (8 fp32)
// groups, which is the minimum legal MTE3 unit on 910B (4B writes hang - see
// fa_fp32_stage1 design.md §3). Rows >= M are padding and are zeroed.
//
// v1 limits (rejected explicitly, not silently clamped): K <= 5120 (one row
// resident in UB) and K % 16 == 0 (32B row copies); bf16/fp16 only.

#include "torch_kernel_helper.h"

#include <tuple>

#include "aclrtlaunch_add_rms_norm_stats_bf16.h"
#include "aclrtlaunch_add_rms_norm_stats_fp16.h"

namespace ascend_kernel {
namespace {
// ascendc runtime export (linked into this plugin via no_workspace_kernel_norm).
extern "C" uint32_t GetCoreNumForMixVectorCore(uint32_t *aiCoreNum, uint32_t *vectorCoreNum);

constexpr uint32_t kMaxK = 5120;      // per-row UB residency budget (kernel MAX_K)
constexpr uint32_t kRstdGroup = 8;    // rstd rows per 32B MTE3 write

// D3 (design.md 9): the two kernel-side accuracy/scheduling decisions of this
// round are fixed here rather than exposed. They were introduced behind
// ARMNS_STATS / ARMNS_SUM switches so each could be measured against the
// F2-frozen path inside one NPU session; the measurements are archived in
// profiles/qwen14b-instruct-hotspot-20260910/f2-kernel/raw/, and the winning
// combination is now the operator's single behaviour:
//
//   batchStats 1  rstd chain once per RSTD_GROUP rows instead of per row
//                 (modes 0/2; -11.9% device time at the prefill main shape)
//   pairSum    1  pairwise fold of the row sum of squares, and a second
//                 Newton step on the vector Rsqrt
//
// pairSum is not optional: it is what takes mode-1 `y` from 5/16 to 0/16
// violations on the CANN strict tier (the first Newton step's residual,
// ~1.5*eps^2, was the dominant term left after the row sum became accurate).
constexpr uint32_t kBatchStats = 1u;
constexpr uint32_t kPairSum = 1u;
}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> add_rms_norm_stats(
    const at::Tensor &x1, const at::Tensor &x2, const c10::optional<at::Tensor> &gamma,
    const c10::optional<at::Tensor> &beta, double eps, int64_t mode)
{
    TORCH_CHECK(x1.dim() == 2, "add_rms_norm_stats: x1 must be 2D (M, K)");
    TORCH_CHECK(x1.is_contiguous(), "add_rms_norm_stats: x1 must be contiguous");
    TORCH_CHECK(mode >= 0 && mode <= 2, "add_rms_norm_stats: mode must be 0, 1 or 2");
    TORCH_CHECK(eps > 0.0, "add_rms_norm_stats: eps must be > 0");
    const bool isBf16 = x1.scalar_type() == at::kBFloat16;
    const bool isFp16 = x1.scalar_type() == at::kHalf;
    TORCH_CHECK(isBf16 || isFp16, "add_rms_norm_stats: x1 must be bf16 or fp16");
    if (mode != 1) {
        TORCH_CHECK(x2.defined() && x2.is_contiguous() && x2.sizes() == x1.sizes() &&
                        x2.scalar_type() == x1.scalar_type(),
                    "add_rms_norm_stats: x2 must be contiguous, same shape and dtype as x1");
    }
    const int64_t mRows = x1.size(0);
    const int64_t kDim = x1.size(1);
    TORCH_CHECK(mRows > 0, "add_rms_norm_stats: empty row count");
    TORCH_CHECK(kDim > 0 && kDim <= static_cast<int64_t>(kMaxK),
                "add_rms_norm_stats: K must be in (0, ", kMaxK, "] (one row resident in UB)");
    TORCH_CHECK(kDim % 16 == 0, "add_rms_norm_stats: K must be a multiple of 16 (32B row copies)");

    const bool hasBeta = mode == 1 && beta.has_value() && beta->defined() && beta->numel() > 0;
    if (mode == 1) {
        TORCH_CHECK(gamma.has_value() && gamma->defined() && gamma->is_contiguous() &&
                        gamma->numel() == kDim && gamma->scalar_type() == x1.scalar_type(),
                    "add_rms_norm_stats: mode 1 needs a contiguous gamma of K elements, same dtype");
        if (hasBeta) {
            TORCH_CHECK(beta->is_contiguous() && beta->numel() == kDim &&
                            beta->scalar_type() == x1.scalar_type(),
                        "add_rms_norm_stats: beta must be contiguous, K elements, same dtype");
        }
    }

    const int64_t mPadded = (mRows + kRstdGroup - 1) / kRstdGroup * kRstdGroup;
    const auto xOut = (mode == 0) ? at::empty_like(x1) : at::empty({0}, x1.options());
    const auto rstd = at::empty({mPadded, 1}, x1.options().dtype(at::kFloat));
    const auto y = (mode == 1) ? at::empty_like(x1) : at::empty({0}, x1.options());

    uint32_t aiCoreNum = 0;
    uint32_t vectorCoreNum = 0;
    const uint32_t coreRet = GetCoreNumForMixVectorCore(&aiCoreNum, &vectorCoreNum);
    TORCH_CHECK(coreRet == 0 && vectorCoreNum > 0, "add_rms_norm_stats: core count query failed");
    const uint32_t rowBlocks = static_cast<uint32_t>((mRows + kRstdGroup - 1) / kRstdGroup);
    const uint32_t blockDim = rowBlocks < vectorCoreNum ? rowBlocks : vectorCoreNum;

    const uint32_t mRowsU = static_cast<uint32_t>(mRows);
    const uint32_t kDimU = static_cast<uint32_t>(kDim);
    const uint32_t modeU = static_cast<uint32_t>(mode);
    const uint32_t hasBetaU = hasBeta ? 1u : 0u;
    const uint32_t batchStatsU = kBatchStats;
    const uint32_t pairSumU = kPairSum;
    const float epsF = static_cast<float>(eps);
    const float kInvF = 1.0f / static_cast<float>(kDim);  // AICore rejects uint32 -> float casts

    // Named lvalues: the launch macro binds its arguments to lvalue references,
    // and a pointer to a tensor the mode never reads is passed as nullptr.
    void *gammaPtr = (mode == 1) ? gamma->data_ptr() : nullptr;
    void *betaPtr = hasBeta ? beta->data_ptr() : nullptr;
    void *x2Ptr = (mode == 1) ? nullptr : x2.data_ptr();  // x2 is ignored in mode 1
    void *xOutPtr = (mode == 0) ? xOut.data_ptr() : nullptr;
    void *yPtr = (mode == 1) ? y.data_ptr() : nullptr;

    if (isBf16) {
        EXEC_KERNEL_CMD(add_rms_norm_stats_bf16, blockDim, x1, x2Ptr, gammaPtr, betaPtr, xOutPtr,
                        rstd, yPtr, mRowsU, kDimU, modeU, epsF, kInvF, hasBetaU, batchStatsU,
                        pairSumU);
    } else {
        EXEC_KERNEL_CMD(add_rms_norm_stats_fp16, blockDim, x1, x2Ptr, gammaPtr, betaPtr, xOutPtr,
                        rstd, yPtr, mRowsU, kDimU, modeU, epsF, kInvF, hasBetaU, batchStatsU,
                        pairSumU);
    }
    return {xOut, rstd, y};
}

}  // namespace ascend_kernel
