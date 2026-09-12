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

// f3_floor - F3 Phase B Step 1/2 host side (see csrc/ops/f3_floor/design.md).
//
//   a         bf16 [M, K] contiguous   (activations; production gate_up GEMM input)
//   b         bf16 [2I, K] contiguous  (nn.Linear weight layout, TP=1)
//   workspace bf16 [M, 2I] optional    (C workspace; pass one in to reuse it)
//   mode      0 = gate_up GEMM + floor-copy epilogue (the fused floor)
//             1 = gate_up GEMM + SiLU(gate)*up epilogue (Step 2)
//             2 = gate_up GEMM only (AIV consumes the flags, moves nothing)
//             3 = floor-copy epilogue only (AIC consumes the flags, computes nothing)
//             4 = floor with D = the UP window (proves the +I window addressing)
//             5 = floor reading the gate window only (113.2 MB vs 169.9 MB: tells
//                 whether the marginal cost scales with the epilogue's bytes or
//                 is a cross-core protocol/scheduling cost)
//   -> (d bf16 [M, I], workspace bf16 [M, 2I])
//
// This is a measurement kernel for the F3 fusion decision; it is not wired into
// any e2e path and is not part of a plugin bundle. Shape contract (all rejected
// explicitly, no silent clamping): bf16, contiguous, K % 16 == 0, M % 16 == 0,
// I % 128 == 0 (the N pairing granularity), K >= 64.

#include "torch_kernel_helper.h"

#include <tuple>

#include "aclrtlaunch_f3_gateup_epilogue.h"

namespace ascend_kernel {
namespace {
// ascendc runtime export (linked into this plugin via no_workspace_kernel_f3).
extern "C" uint32_t GetCoreNumForMixVectorCore(uint32_t *aiCoreNum, uint32_t *vectorCoreNum);

constexpr int64_t kModeFloor = 0;
constexpr int64_t kModeSwiglu = 1;
constexpr int64_t kModeGemmOnly = 2;
constexpr int64_t kModeEpilogueOnly = 3;
constexpr int64_t kModeFloorUp = 4;        // attribution rung: D = up window copy
constexpr int64_t kModeFloorGateOnly = 5;  // attribution rung: gate window only (2/3 traffic)
constexpr int64_t kModeMax = 5;
constexpr int64_t kNTile = 128;  // must match L1TileShape::N in the kernel
}  // namespace

std::tuple<at::Tensor, at::Tensor> f3_gateup_epilogue(const at::Tensor &a, const at::Tensor &b,
                                                      const c10::optional<at::Tensor> &workspace,
                                                      int64_t mode)
{
    TORCH_CHECK(a.dim() == 2 && a.is_contiguous() && a.scalar_type() == at::kBFloat16,
                "f3_gateup_epilogue: a must be a contiguous bf16 [M, K]");
    TORCH_CHECK(b.dim() == 2 && b.is_contiguous() && b.scalar_type() == at::kBFloat16,
                "f3_gateup_epilogue: b must be a contiguous bf16 [2I, K]");
    TORCH_CHECK(mode >= kModeFloor && mode <= kModeMax,
                "f3_gateup_epilogue: mode must be 0..5");
    const int64_t mRows = a.size(0);
    const int64_t kDim = a.size(1);
    TORCH_CHECK(b.size(0) % 2 == 0, "f3_gateup_epilogue: b.size(0) must be even (gate | up)");
    const int64_t iHalf = b.size(0) / 2;
    TORCH_CHECK(b.size(1) == kDim, "f3_gateup_epilogue: a.size(1) must equal b.size(1)");
    TORCH_CHECK(mRows > 0 && kDim > 0, "f3_gateup_epilogue: empty input");
    TORCH_CHECK(kDim % 16 == 0, "f3_gateup_epilogue: K must be a multiple of 16");
    TORCH_CHECK(mRows % 16 == 0, "f3_gateup_epilogue: M must be a multiple of 16");
    TORCH_CHECK(iHalf % kNTile == 0, "f3_gateup_epilogue: I must be a multiple of the N tile");
    TORCH_CHECK(kDim >= 64, "f3_gateup_epilogue: K must be >= 64 (L0 K tile)");

    const at::Tensor wsIn = (workspace.has_value() && workspace->defined()) ? *workspace
                                                                           : at::Tensor();
    at::Tensor ws;
    if (wsIn.defined()) {
        TORCH_CHECK(wsIn.is_contiguous() && wsIn.scalar_type() == at::kBFloat16 &&
                        wsIn.dim() == 2 && wsIn.size(0) == mRows && wsIn.size(1) == 2 * iHalf,
                    "f3_gateup_epilogue: workspace must be contiguous bf16 [M, 2I]");
        ws = wsIn;
    } else {
        ws = at::empty({mRows, 2 * iHalf}, a.options());
    }
    const auto d = at::empty({mRows, iHalf}, a.options());

    uint32_t aiCoreNum = 0;
    uint32_t vectorCoreNum = 0;
    const uint32_t coreRet = GetCoreNumForMixVectorCore(&aiCoreNum, &vectorCoreNum);
    TORCH_CHECK(coreRet == 0 && aiCoreNum > 0, "f3_gateup_epilogue: core count query failed");

    // Named lvalues: the launch macro binds its arguments to lvalue references.
    const uint32_t mU = static_cast<uint32_t>(mRows);
    const uint32_t nHalfU = static_cast<uint32_t>(iHalf);
    const uint32_t kU = static_cast<uint32_t>(kDim);
    const uint32_t modeU = static_cast<uint32_t>(mode);

    EXEC_KERNEL_CMD(f3_gateup_epilogue, aiCoreNum, a, b, ws, d, mU, nHalfU, kU, modeU);
    return {d, ws};
}

}  // namespace ascend_kernel
