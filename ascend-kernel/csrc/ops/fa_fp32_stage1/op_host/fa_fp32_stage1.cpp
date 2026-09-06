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

// Stage-1 fp32-out flash attention (cascade decode shared-prefix stage) as a
// torch op: port of catlass example 23 host logic (fai.cpp + fai_tiling.cpp,
// probe version) - see design.md for the mapping table. The kernel dataflow is
// untouched; this file only ports allocation, tiling fill and launch.
//
// Differences vs the example host, all deliberate:
//  - allocations via at::empty (torch caching allocator; M-C graph pooling note
//    in design.md §5);
//  - maxNumBlocksPerBatch = block_table.size(1): the kernel steps the block
//    table by ROW STRIDE per batch, so it must be the real stride, not
//    ceil(maxKv/128) (the example's data happened to make the two equal);
//  - mask is passed as nullptr (maskType is fixed 0; every gMask access is
//    behind maskType != 0 branches - the example passed an uninitialised
//    pointer in that case);
//  - launch via the framework ACLRT_LAUNCH_KERNEL path. For mix kernels the
//    framework launcher injects the cross-core sync base automatically
//    (GetAscendCoreSyncAddr -> set_ffts_base_addr in the auto-generated device
//    wrapper), so the kernel entry has no ffts argument and the host does not
//    call rtGetC2cCtrlAddr (the example's manual path is superseded).
//
// plan-20260903 M-C C0 host-side additions (kernel unchanged):
//  - q_seqlen_value host hint (default 0 = legacy behaviour): when > 0, every
//    request's q seqlen equals this value (decode = 1). The host then skips
//    BOTH seqlen D2H pulls (.to(kCPU), two stream syncs per call) and computes
//    the tiling purely from shapes; the sumQ check becomes T == batch * value.
//    The ceil(maxKv/128) <= cols check degrades to a documented caller
//    contract (the kv seqlens are read by the kernel from device memory and
//    the caller - vLLM cascade integration - guarantees block_table cols fit).
//    This is the graph-capture-safe path: for a uniform q_len the tiling is a
//    pure shape function (GetQSBlockTile is a constant 128), so a captured H2D
//    tiling copy replays with identical bytes.
//  - persistent tiling staging (per-device, single caller thread assumed -
//    the vLLM worker model): the previous from_blob(&tiling).clone().to(device)
//    made the H2D source a temporary CPU tensor that dies at op return. Eager
//    mode only survives because the driver syncs the copy; under ACL graph
//    capture the H2D becomes a replayed node that would read freed host
//    memory. The staging buffers are process-persistent RAW allocations
//    (host: aclrtMallocHost pinned; device: aclrtMalloc) with no at::Tensor
//    anywhere in the static registry - a static C++ tensor would be destroyed
//    during global C++ teardown, which on torch_npu runs after interpreter
//    finalization and aborts (PyEval_SaveThread GIL error, A/B-proven against
//    the md5-identical pre-C0 build: clean exit with the old temporary-tensor
//    path, abort with a tensor-holding static registry). Each call wraps the
//    raw buffers in non-owning from_blob views for the H2D copy, so no tensor
//    outlives the call except the raw process-lifetime memory.
//  - graph-capture H2D (C2 probe finding, plan-20260903-mc-integration): the
//    H2D must be a PINNED-host, non-blocking (async) copy. A pageable-source
//    H2D becomes a synchronising rtMemcpy, and ACL capture mode rejects
//    synchronising operations outright (runtime error 107030 "the current
//    capture mode does not support this operation"). With pinned memory +
//    non_blocking=true the copy is a plain async DMA node that graphs can
//    record and replay.
//  - tiling staging is DEDUPLICATED BY CONTENT (per device): one pinned host
//    buffer + one device buffer per distinct FATilingData byte pattern. The
//    tiling is a pure shape function on the uniform-q fast path, so a decode
//    bucket has exactly one content - captured graphs for different buckets
//    each keep their own staging pair instead of overwriting each other (a
//    single shared staging would corrupt every earlier capture the moment a
//    second bucket captures its tiling). Dedup uses an exact 88-byte memcmp,
//    not a hash, so equal contents never duplicate and distinct contents
//    never collide. A fresh memcpy into the staging host buffer happens only
//    on a content miss, which also removes the multi-bucket replay hazard of
//    rewriting host bytes a captured in-flight DMA might still be reading.

#include "torch_kernel_helper.h"

#include "../op_kernel/fai_tiling_data.hpp"

#include "aclrtlaunch_fa_fp32_stage1.h"

#include <cmath>
#include <cstring>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "acl/acl.h"
namespace {
// ascendc runtime export (linked into this plugin via no_workspace_kernel).
extern "C" uint32_t GetCoreNumForMixVectorCore(uint32_t *aiCoreNum, uint32_t *vectorCoreNum);

constexpr uint32_t kWorkspaceBlockSizeDb = 131072;  // WORKSPACE_BLOCK_SIZE_DB, per core per slot (elements)
constexpr uint32_t kPreLaunch = 2;                  // kernel ping-pong depth + 1 == 3 slots

uint32_t GetQNBlockTile(int64_t qSeqlen, uint32_t groupSize)
{
    uint32_t qRowNumCeil = 128;
    uint32_t qNBlockTile = static_cast<uint32_t>(qRowNumCeil / qSeqlen) / 2 * 2;
    qNBlockTile = std::min(qNBlockTile, groupSize);
    qNBlockTile = std::max(qNBlockTile, 1u);
    return qNBlockTile;
}

// Persistent tiling staging, deduplicated by content: one pinned host buffer
// + one device buffer per distinct FATilingData byte pattern, per device.
// Host side is allocated THROUGH the torch pinned allocator (pin_memory()):
// a raw aclrtMallocHost pointer is unknown to torch's CachingHostAllocator,
// and its non_blocking copy path then falls back to a synchronising check
// that ACL capture mode rejects (probe: "Not allow to synchronize captured-
// stream" from process_unregistered_mem_location_type). The pin_memory()
// tensor is deliberately LEAKED (heap-allocated, never destroyed): the
// C++ destructor must never run (global teardown runs after interpreter
// finalization and aborts - see the A/B note above), the CachingHostAllocator
// block must stay alive, and one 88-byte block per distinct bucket content is
// negligible. Device side is a raw aclrtMalloc.
// The uniform-q fast path makes the tiling a pure shape function, so each
// decode bucket maps to exactly one content; graphs captured for different
// buckets therefore never overwrite each other's staging (a single shared
// staging would corrupt every earlier capture once a second bucket captured).
// Thread-safety: the registry map is mutex-guarded, but the staging buffers
// themselves are reused across calls - concurrent op invocations on the same
// device would race on the staging bytes. The vLLM worker model is
// single-threaded per device; do not call this op concurrently on one device.
struct TilingStaging
{
    at::Tensor *leakedHost = nullptr;  // deliberate leak, see struct comment
    void *device = nullptr;            // aclrtMalloc'd, sizeof(FATilingData)
    uint8_t content[sizeof(FATilingData)] = {};  // last bytes copied host->device
    bool initialized = false;
};

TilingStaging &GetTilingStaging(const c10::Device &device, const FATilingData &tiling)
{
    static std::mutex mu;
    static std::unordered_map<int64_t, std::vector<TilingStaging *>> registry;
    const auto *bytes = reinterpret_cast<const uint8_t *>(&tiling);
    const int64_t devIdx = static_cast<int64_t>(device.index());
    TilingStaging *entry = nullptr;
    {
        std::lock_guard<std::mutex> lock(mu);
        auto &slots = registry[devIdx];
        for (TilingStaging *s : slots) {
            if (s->initialized &&
                std::memcmp(s->content, bytes, sizeof(FATilingData)) == 0) {
                entry = s;
                break;
            }
        }
        if (entry == nullptr) {
            entry = new TilingStaging;
            entry->leakedHost = new at::Tensor(
                at::empty({static_cast<int64_t>(sizeof(FATilingData))},
                          at::TensorOptions().dtype(at::kByte))
                    .pin_memory());
            const aclError devRet = aclrtMalloc(&entry->device, sizeof(FATilingData),
                                                ACL_MEM_MALLOC_NORMAL_ONLY);
            TORCH_CHECK(devRet == ACL_SUCCESS,
                        "fa_fp32_stage1: aclrtMalloc for tiling staging failed, code=",
                        static_cast<int>(devRet));
            std::memcpy(entry->leakedHost->data_ptr(), bytes, sizeof(FATilingData));
            std::memcpy(entry->content, bytes, sizeof(FATilingData));
            entry->initialized = true;
            slots.push_back(entry);
        }
    }
    return *entry;
}
}  // namespace

namespace ascend_kernel {

std::tuple<at::Tensor, at::Tensor> fa_fp32_stage1(const at::Tensor &query, const at::Tensor &key,
                                                  const at::Tensor &value, const at::Tensor &block_table,
                                                  const at::Tensor &actual_q_seqlens,
                                                  const at::Tensor &actual_kv_seqlens,
                                                  int64_t q_seqlen_value)
{
    TORCH_CHECK(query.is_contiguous() && key.is_contiguous() && value.is_contiguous() &&
                    block_table.is_contiguous() && actual_q_seqlens.is_contiguous() &&
                    actual_kv_seqlens.is_contiguous(),
                "fa_fp32_stage1: all inputs must be contiguous");
    TORCH_CHECK(query.scalar_type() == at::kBFloat16, "fa_fp32_stage1: query must be bf16");
    TORCH_CHECK(key.scalar_type() == at::kBFloat16 && value.scalar_type() == at::kBFloat16,
                "fa_fp32_stage1: key/value must be bf16");
    TORCH_CHECK(block_table.scalar_type() == at::kInt, "fa_fp32_stage1: block_table must be int32");
    TORCH_CHECK(actual_q_seqlens.scalar_type() == at::kLong && actual_kv_seqlens.scalar_type() == at::kLong,
                "fa_fp32_stage1: seqlens must be int64");
    TORCH_CHECK(query.dim() == 3, "fa_fp32_stage1: query must be (T, H, D)");
    TORCH_CHECK(key.dim() == 4 && value.dim() == 4, "fa_fp32_stage1: key/value must be (numBlocks, blockSize, KVH, D)");
    TORCH_CHECK(key.sizes() == value.sizes(), "fa_fp32_stage1: key/value shapes must match");
    TORCH_CHECK(q_seqlen_value >= 0, "fa_fp32_stage1: q_seqlen_value must be >= 0 (0 = read seqlens from device)");

    const int64_t numTokens = query.size(0);
    const int64_t numHeads = query.size(1);
    const int64_t embed = query.size(2);
    const int64_t kvHeads = key.size(2);
    const int64_t blockSize = key.size(1);
    const int64_t batch = actual_q_seqlens.numel();
    TORCH_CHECK(actual_kv_seqlens.numel() == batch, "fa_fp32_stage1: seqlen tensors must have batch entries");
    TORCH_CHECK(embed == 128, "fa_fp32_stage1: only head dim 128 is supported (L1TileShape::K constraint)");
    TORCH_CHECK(blockSize == 128, "fa_fp32_stage1: only paged blockSize 128 is supported");
    TORCH_CHECK(kvHeads > 0 && numHeads % kvHeads == 0, "fa_fp32_stage1: numHeads must be a multiple of kvHeads");
    if (batch == 0 || numTokens == 0) {
        auto emptyOut = at::empty({numTokens, numHeads, embed}, query.options().dtype(at::kFloat));
        auto emptyLse = at::empty({0}, query.options().dtype(at::kFloat));
        return {emptyOut, emptyLse};
    }

    // Per-batch q seqlens on host: fast path (q_seqlen_value > 0) needs no D2H
    // at all; the generic path pulls both seqlen tensors (one D2H each).
    std::vector<int64_t> qSeqHost;
    int64_t maxKv = 0;
    if (q_seqlen_value > 0) {
        TORCH_CHECK(numTokens == batch * q_seqlen_value,
                    "fa_fp32_stage1: with q_seqlen_value=", q_seqlen_value,
                    ", query.size(0) must equal batch * q_seqlen_value");
    } else {
        auto qCpu = actual_q_seqlens.to(at::kCPU);
        auto kvCpu = actual_kv_seqlens.to(at::kCPU);
        const int64_t *qSeq = qCpu.data_ptr<int64_t>();
        const int64_t *kvSeq = kvCpu.data_ptr<int64_t>();
        int64_t sumQ = 0;
        for (int64_t b = 0; b < batch; b++) {
            TORCH_CHECK(qSeq[b] >= 1 && kvSeq[b] >= 1, "fa_fp32_stage1: seqlens must be >= 1");
            sumQ += qSeq[b];
            maxKv = std::max(maxKv, kvSeq[b]);
            qSeqHost.push_back(qSeq[b]);
        }
        TORCH_CHECK(sumQ == numTokens, "fa_fp32_stage1: query.size(0) must equal sum(actual_q_seqlens)");
    }

    const int64_t maxNumBlocksPerBatch = block_table.size(1);
    if (q_seqlen_value == 0) {
        TORCH_CHECK((maxKv + blockSize - 1) / blockSize <= maxNumBlocksPerBatch,
                    "fa_fp32_stage1: block_table cols too small for max kv seqlen");
    } else {
        // Fast path: kv seqlens stay on device (the kernel reads them there);
        // the caller (vLLM cascade integration) guarantees block_table cols
        // cover the per-batch kv seqlens. Not host-checkable without a D2H.
    }

    // blockDim = number of AI cores (mix kernel: one task loop per AIC, AIVs paired).
    uint32_t aiCoreNum = 0;
    uint32_t vectorCoreNum = 0;
    uint32_t coreRet = GetCoreNumForMixVectorCore(&aiCoreNum, &vectorCoreNum);
    TORCH_CHECK(coreRet == 0 && aiCoreNum > 0, "fa_fp32_stage1: GetCoreNumForMixVectorCore failed");

    // Tiling: verbatim port of fai_tiling.cpp (see design.md §4).
    FATilingData tiling;
    tiling.batch = static_cast<uint32_t>(batch);
    tiling.numHeads = static_cast<uint32_t>(numHeads);
    tiling.kvHeads = static_cast<uint32_t>(kvHeads);
    tiling.embeddingSize = static_cast<uint32_t>(embed);
    tiling.numBlocks = static_cast<uint32_t>(key.size(0));
    tiling.blockSize = static_cast<uint32_t>(blockSize);
    tiling.maxKvSeqlen = static_cast<uint32_t>(
        q_seqlen_value > 0 ? maxNumBlocksPerBatch * blockSize : maxKv);  // kernel reads the per-batch kv seqlens from device memory (maxKvSeqlen unused on device)
    tiling.maxNumBlocksPerBatch = static_cast<uint32_t>(maxNumBlocksPerBatch);
    tiling.maskType = 0;  // NO_MASK (shared-prefix stage)
    tiling.scaleValue = static_cast<float>(1.0 / std::sqrt(1.0 * embed));

    const uint32_t groupSize = static_cast<uint32_t>(numHeads / kvHeads);
    uint32_t totalTaskNum = 0;
    for (int64_t b = 0; b < batch; b++) {
        const uint32_t curQSeqlen = static_cast<uint32_t>(
            q_seqlen_value > 0 ? q_seqlen_value : qSeqHost[b]);
        uint32_t curQNBlockTile = GetQNBlockTile(curQSeqlen, groupSize);
        uint32_t qNBlockNumPerGroup = (groupSize + curQNBlockTile - 1) / curQNBlockTile;
        uint32_t curQNBlockNum = qNBlockNumPerGroup * static_cast<uint32_t>(kvHeads);
        uint32_t curQSBlockTile = 128;  // GetQSBlockTile
        uint32_t curQSBlockNum = (curQSeqlen + curQSBlockTile - 1) / curQSBlockTile;
        uint32_t curTaskNum = curQNBlockNum * curQSBlockNum;
        if (b == 0) {
            tiling.firstBatchTaskNum = curTaskNum;
        }
        totalTaskNum += curTaskNum;
    }
    tiling.totalTaskNum = totalTaskNum;

    const uint64_t mm1OutSize = static_cast<uint64_t>(aiCoreNum) * kWorkspaceBlockSizeDb * 4 * 3;
    const uint64_t smOnlineOutSize = static_cast<uint64_t>(aiCoreNum) * kWorkspaceBlockSizeDb * 2 * 3;
    const uint64_t mm2OutSize = static_cast<uint64_t>(aiCoreNum) * kWorkspaceBlockSizeDb * 4 * 3;
    const uint64_t updateSize = static_cast<uint64_t>(aiCoreNum) * kWorkspaceBlockSizeDb * 4 * 3;
    tiling.mm1OutSize = mm1OutSize;
    tiling.smOnlineOutSize = smOnlineOutSize;
    tiling.mm2OutSize = mm2OutSize;
    tiling.UpdateSize = updateSize;
    tiling.workSpaceSize = mm1OutSize + smOnlineOutSize + mm2OutSize + updateSize;

    // Outputs + workspace (all device tensors; see design.md §5 for the
    // M-C graph-pooling note).
    auto out = at::empty({numTokens, numHeads, embed}, query.options().dtype(at::kFloat));
    auto lse = at::empty({numTokens * numHeads * 8}, query.options().dtype(at::kFloat));
    auto sWs = at::empty({static_cast<int64_t>(mm1OutSize)}, query.options().dtype(at::kByte));
    auto pWs = at::empty({static_cast<int64_t>(smOnlineOutSize)}, query.options().dtype(at::kByte));
    auto oTempWs = at::empty({static_cast<int64_t>(mm2OutSize)}, query.options().dtype(at::kByte));
    auto oUpdateWs = at::empty({static_cast<int64_t>(updateSize)}, query.options().dtype(at::kByte));

    // Persistent tiling staging, deduplicated by content: pick (or create)
    // the staging pair matching these exact tiling bytes, then copy pinned
    // host -> device via non-owning from_blob views (the views die at op
    // return; only the deliberately-leaked pinned block and the raw device
    // allocation persist). The copy is non-blocking (torch-allocator pinned
    // source): recordable/replayable inside ACL graph capture, where a
    // pageable-source or foreign-pinned H2D ends in a rejected synchronising
    // memcpy (probe: runtime 107027/107030). On a content hit no host bytes
    // are rewritten, so a captured graph replaying its H2D never races a
    // later call's memcpy.
    auto &staging = GetTilingStaging(query.device(), tiling);
    auto hostView = at::from_blob(staging.leakedHost->data_ptr(),
                                  {static_cast<int64_t>(sizeof(FATilingData))}, at::kByte);
    auto devView = at::from_blob(staging.device, {static_cast<int64_t>(sizeof(FATilingData))},
                                 at::TensorOptions().dtype(at::kByte).device(query.device()));
    devView.copy_(*staging.leakedHost, /*non_blocking=*/true);

    // mask is never dereferenced with maskType == 0 (fixed here); a named lvalue
    // because the launch macro binds args to lvalue references.
    void *maskPtr = nullptr;
    EXEC_KERNEL_CMD(fa_fp32_stage1, aiCoreNum, query, key, value, maskPtr,
                    block_table, out, actual_q_seqlens, actual_kv_seqlens, sWs, pWs, oTempWs, oUpdateWs,
                    devView, lse);
    return {out, lse};
}

}  // namespace ascend_kernel
