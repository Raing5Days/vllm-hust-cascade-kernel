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

// FATilingData: byte-identical port of catlass example 23 kernel_common.hpp.
// Shared by op_host (fill + H2D) and the device kernel (__gm__ reinterpret),
// so the layout can never drift between the two sides. Keep standard layout:
// 10 x uint32 + 5 x uint64 + float = 84 used bytes, sizeof == 88 with tail pad.
//
// Ported for fa_fp32_stage1 (plan-20260903 M-B step 2); one semantic fix vs the
// example: maxNumBlocksPerBatch is the block-table ROW STRIDE (kernel blockBOffset
// stepping), taken from block_table.size(1) by op_host instead of ceil(maxKv/128).

#ifndef FAI_TILING_DATA_HPP
#define FAI_TILING_DATA_HPP

#include <cstdint>

struct FATilingData {
    uint32_t numHeads = 0;
    uint32_t embeddingSize = 0;
    uint32_t numBlocks = 0;
    uint32_t blockSize = 0;
    uint32_t maxKvSeqlen = 0;
    uint32_t kvHeads = 0;
    uint32_t batch = 0;
    uint32_t maxNumBlocksPerBatch = 0;
    uint32_t firstBatchTaskNum = 0;
    uint32_t totalTaskNum = 0;
    uint32_t maskType = 0;
    uint64_t mm1OutSize = 0;
    uint64_t smOnlineOutSize = 0;
    uint64_t mm2OutSize = 0;
    uint64_t UpdateSize = 0;
    uint64_t workSpaceSize = 0;
    float scaleValue = 0.0;
};

#endif  // FAI_TILING_DATA_HPP
