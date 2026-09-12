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

#include <torch/extension.h>
#include <torch/library.h>

#include "ops.h"

namespace {
TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("lse_merge(Tensor o1, Tensor o2, Tensor lse1, Tensor lse2, int out_code=0) -> Tensor");
    m.def("fa_fp32_stage1(Tensor query, Tensor key, Tensor value, Tensor block_table, "
          "Tensor actual_q_seqlens, Tensor actual_kv_seqlens, int q_seqlen_value=0) -> (Tensor, Tensor)");
    m.def("add_rms_norm_stats(Tensor x1, Tensor x2, Tensor? gamma, Tensor? beta, float eps, "
          "int mode) -> (Tensor, Tensor, Tensor)");
    m.def("bw_probe(Tensor x1, Tensor x2, int variant, float eps) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("lse_merge", TORCH_FN(ascend_kernel::lse_merge));
    m.impl("fa_fp32_stage1", TORCH_FN(ascend_kernel::fa_fp32_stage1));
    m.impl("add_rms_norm_stats", TORCH_FN(ascend_kernel::add_rms_norm_stats));
    m.impl("bw_probe", TORCH_FN(ascend_kernel::bw_probe));
}

}  // namespace
