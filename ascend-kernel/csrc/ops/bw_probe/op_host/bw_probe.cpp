// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// bw_probe host side - D3 Phase-0 access-pattern anchor (kernel file header has
// the variant ladder). Host limits mirror add_rms_norm_stats so that the probe is
// callable on exactly the shapes the production op accepts: K <= 5120, K % 16 == 0.
//
// NOT a production op: only `out` is defined for variant 0/1, and the returned
// rstd is written for variant 2 only. It exists to measure a memory pattern.

#include "torch_kernel_helper.h"

#include <tuple>

#include "aclrtlaunch_bw_probe_bf16.h"
#include "aclrtlaunch_bw_probe_fp16.h"

namespace ascend_kernel {
namespace {
extern "C" uint32_t GetCoreNumForMixVectorCore(uint32_t *aiCoreNum, uint32_t *vectorCoreNum);

constexpr uint32_t kMaxK = 5120;
constexpr uint32_t kRstdGroup = 8;
}  // namespace

std::tuple<at::Tensor, at::Tensor> bw_probe(const at::Tensor &x1, const at::Tensor &x2,
                                            int64_t variant, double eps)
{
    TORCH_CHECK(x1.dim() == 2 && x1.is_contiguous(), "bw_probe: x1 must be 2D contiguous (M, K)");
    TORCH_CHECK(x2.defined() && x2.is_contiguous() && x2.sizes() == x1.sizes() &&
                    x2.scalar_type() == x1.scalar_type(),
                "bw_probe: x2 must be contiguous, same shape and dtype as x1");
    TORCH_CHECK(variant >= 0 && variant <= 2, "bw_probe: variant must be 0 (copy), 1 (apply) or 2 (mode0)");
    const bool isBf16 = x1.scalar_type() == at::kBFloat16;
    const bool isFp16 = x1.scalar_type() == at::kHalf;
    TORCH_CHECK(isBf16 || isFp16, "bw_probe: x1 must be bf16 or fp16");
    const int64_t mRows = x1.size(0);
    const int64_t kDim = x1.size(1);
    TORCH_CHECK(kDim > 0 && kDim <= static_cast<int64_t>(kMaxK),
                "bw_probe: K must be in (0, ", kMaxK, "]");
    TORCH_CHECK(kDim % 16 == 0, "bw_probe: K must be a multiple of 16 (32B row copies)");

    const int64_t mPadded = (mRows + kRstdGroup - 1) / kRstdGroup * kRstdGroup;
    const auto out = at::empty_like(x1);
    const auto rstd = at::empty({mPadded, 1}, x1.options().dtype(at::kFloat));

    uint32_t aiCoreNum = 0;
    uint32_t vectorCoreNum = 0;
    const uint32_t coreRet = GetCoreNumForMixVectorCore(&aiCoreNum, &vectorCoreNum);
    TORCH_CHECK(coreRet == 0 && vectorCoreNum > 0, "bw_probe: core count query failed");
    const uint32_t rowBlocks = static_cast<uint32_t>((mRows + kRstdGroup - 1) / kRstdGroup);
    const uint32_t blockDim = rowBlocks < vectorCoreNum ? rowBlocks : vectorCoreNum;

    const uint32_t mRowsU = static_cast<uint32_t>(mRows);
    const uint32_t kDimU = static_cast<uint32_t>(kDim);
    const uint32_t variantU = static_cast<uint32_t>(variant);
    const float epsF = static_cast<float>(eps);
    const float kInvF = 1.0f / static_cast<float>(kDim);

    if (isBf16) {
        EXEC_KERNEL_CMD(bw_probe_bf16, blockDim, x1, x2, out, rstd, mRowsU, kDimU, variantU, epsF, kInvF);
    } else {
        EXEC_KERNEL_CMD(bw_probe_fp16, blockDim, x1, x2, out, rstd, mRowsU, kDimU, variantU, epsF, kInvF);
    }
    return {out, rstd};
}

}  // namespace ascend_kernel
