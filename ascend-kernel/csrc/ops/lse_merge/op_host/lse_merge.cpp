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

#include "torch_kernel_helper.h"

#include "aclrtlaunch_lse_merge.h"

namespace ascend_kernel {

// LSE-merge for two-stage cascade decode (FlashInfer formula), computed
// per row in fp32. Mixed-precision extension (plan-20260902 M-B, Tier 1):
//  - o1: bf16 (legacy Tier-0) or fp32 (stage-1 custom fp32-out kernel);
//  - o2: bf16 (stage-2 FIA output); lse1/lse2: fp32 with row stride
//    numel/rows (1 = compact FIA layout, 8 = the 32B padded-row layout the
//    catlass FAInferBf16Fp32Out kernel writes - zero-copy pass-through);
//  - out_code: 0 = follow o1 dtype (default, backward compatible),
//    1 = bf16 (integration form: merged result feeds the bf16 output buffer
//    without an extra cast kernel), 2 = fp32 (probe form: Tier-1 residual
//    before the final cast).
at::Tensor lse_merge(const at::Tensor &o1, const at::Tensor &o2, const at::Tensor &lse1,
                     const at::Tensor &lse2, int64_t out_code)
{
    TORCH_CHECK(o1.is_contiguous() && o2.is_contiguous() && lse1.is_contiguous() && lse2.is_contiguous(),
                "lse_merge: all inputs must be contiguous");
    TORCH_CHECK(o1.scalar_type() == at::kBFloat16 || o1.scalar_type() == at::kFloat,
                "lse_merge: o1 must be bf16 or fp32");
    TORCH_CHECK(o2.scalar_type() == at::kBFloat16, "lse_merge: o2 must be bf16");
    TORCH_CHECK(lse1.scalar_type() == at::kFloat && lse2.scalar_type() == at::kFloat,
                "lse_merge: lse1/lse2 must be fp32");
    TORCH_CHECK(out_code >= 0 && out_code <= 2,
                "lse_merge: out_code must be 0 (follow o1), 1 (bf16) or 2 (fp32)");
    at::ScalarType outType = (out_code == 1) ? at::kBFloat16
                             : (out_code == 2) ? at::kFloat
                                               : o1.scalar_type();
    TORCH_CHECK(o1.size(-1) % 16 == 0, "lse_merge: head dim must be a multiple of 16 (MTE 32B alignment)");

    auto out = at::empty(o1.sizes(), o1.options().dtype(outType));
    const uint32_t dim = o1.size(-1);
    const uint32_t totalRows = o1.numel() / dim;
    if (totalRows == 0) {
        return out;
    }
    TORCH_CHECK(lse1.numel() % totalRows == 0 && lse2.numel() % totalRows == 0,
                "lse_merge: lse numel must be a multiple of the row count (row-stride packing)");
    const int64_t stride1 = lse1.numel() / totalRows;
    const int64_t stride2 = lse2.numel() / totalRows;
    TORCH_CHECK(stride1 >= 1 && stride1 <= 64 && stride2 >= 1 && stride2 <= 64,
                "lse_merge: lse row stride must be in [1, 64]");
    // 910B3 AIV count; rows are split evenly across blocks.
    const uint32_t blockDim = totalRows < 8 ? totalRows : 8;
    const uint32_t o1IsF32 = (o1.scalar_type() == at::kFloat) ? 1u : 0u;
    const uint32_t outIsF32 = (outType == at::kFloat) ? 1u : 0u;
    const uint32_t s1 = static_cast<uint32_t>(stride1);
    const uint32_t s2 = static_cast<uint32_t>(stride2);
    EXEC_KERNEL_CMD(lse_merge, blockDim, o1, o2, lse1, lse2, out, totalRows, dim, o1IsF32,
                    outIsF32, s1, s2);
    return out;
}

}  // namespace ascend_kernel
