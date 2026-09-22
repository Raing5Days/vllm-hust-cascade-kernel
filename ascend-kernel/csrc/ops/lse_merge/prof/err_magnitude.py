# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""判定 bit 门失败的性质：舍入差 vs 计算错。

读 golden 的 .pt 与当前实现输出逐元素比较，报 ULP 量级的最大差。
若最大差只有几个 ULP ⇒ 是"乘法重结合"的舍入差（bit 门确实过不去，但数值无害）；
若差值远大于 ULP ⇒ 是逻辑/索引错，必须修。
"""

import os
import sys
import zlib

import ascend_kernel  # noqa: F401
import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test"))
from lse_merge_cases import MODES, SHAPES, make_inputs

GOLDEN = "/vllm-workspace/profiles/lse-merge-pipe-20260918/golden"


def ulp_diff(a, b):
    """同 dtype 两个张量的整数表示差（ULP 计数）。"""
    ai = a.cpu().contiguous().view(torch.int32).to(torch.int64)
    bi = b.cpu().contiguous().view(torch.int32).to(torch.int64)
    return (ai - bi).abs()


def main():
    worst = 0
    for mode, (o1_dtype, _o2d, out_code, padded, *_r) in MODES.items():
        for cat, (T, H, D) in SHAPES:
            key = f"{mode}__{cat}"
            gpath = os.path.join(GOLDEN, f"{key}.pt")
            if not os.path.exists(gpath):
                continue
            seed = zlib.crc32(f"{cat}/{mode}".encode()) % 2**31
            o1, o2, l1c, l2c, _l1, _l2 = make_inputs(T, H, D, o1_dtype, padded, seed=seed)
            out = torch.ops.npu.lse_merge(o1, o2, l1c, l2c, out_code)
            torch.npu.synchronize()
            gold = torch.load(gpath, weights_only=True)
            if out.dtype != gold.dtype or out.shape != gold.shape:
                print(f"{key}: SHAPE/DTYPE 不符！")
                continue
            u = ulp_diff(out, gold)
            mx, nz = int(u.max()), int((u > 0).sum())
            worst = max(worst, mx)
            kind = "相同" if mx == 0 else ("舍入级(<=8 ULP)" if mx <= 8 else "**远大于舍入，疑似逻辑错**")
            print(f"{key:34s} 元素数={u.numel():7d} 不同元素={nz:7d} 最大 ULP 差={mx:6d}  {kind}")
    print(f"\n全用例最大 ULP 差 = {worst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
