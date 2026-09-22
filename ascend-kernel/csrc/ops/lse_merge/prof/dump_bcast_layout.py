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

"""反解"第 r 行实际用了哪个权重"，用来定位行缩放的行错位。

构造：o1 = 1, o2 = 0, lse1 = 0
  ⇒ w1[r] = 1, w2[r] = exp(lse2[r]), s[r] = 1 + exp(lse2[r])
  ⇒ out[r, :] 应为常数 1/(1+exp(lse2[r]))

取 lse2[r] = -(r+1)*0.5（负值 ⇒ exp 很小 ⇒ out 接近 1，反解数值稳定），
则反解 applied[r] = ln(out[r,0]/(1-out[r,0])) = **-lse2[r]**（注意这个负号：
out = 1/(1+exp(lse2)) ⇒ out/(1-out) = 1/exp(lse2) = exp(-lse2)）。
applied[r] 等于 -lse2[σ(r)] 就说明第 r 行拿到的是第 σ(r) 行的权重。

注意：lse2 用负值是为了反解稳定；用 0..31 会让 out ~1e-14 落到 fp32 输出的
量化噪声里，反解必然"失败"（这是本脚本前两版踩过的坑）。
"""

import sys

import ascend_kernel  # noqa: F401
import torch
import torch_npu  # noqa: F401

T, H, D = 4, 8, 128
rows = T * H


def main():
    lse2_vals = [-(r + 1) * 0.1 for r in range(rows)]   # 限幅：避免 out→1 时反解抵消
    lse2 = torch.tensor(lse2_vals, dtype=torch.float32, device="npu").reshape(T, H)
    o1 = torch.ones(T, H, D, dtype=torch.float32, device="npu")
    o2 = torch.zeros(T, H, D, dtype=torch.bfloat16, device="npu")
    l1 = torch.zeros(T, H, dtype=torch.float32, device="npu")

    out = torch.ops.npu.lse_merge(o1, o2, l1, lse2, 2)
    torch.npu.synchronize()
    got = out.cpu().reshape(rows, D).double()

    # 每行应当是全常数
    spread = (got.max(dim=1).values - got.min(dim=1).values).abs()
    bad_const = (spread > 1e-9).nonzero().flatten().tolist()

    print(f"非常数的行: {bad_const if bad_const else '无'}")

    # 反解实际生效的 lse2
    n_ok = 0
    for r in range(rows):
        v = got[r, 0].item()
        if 0.0 < v < 1.0:
            applied = torch.log(torch.tensor(v / (1.0 - v))).item()
        else:
            applied = float("nan")
        ok = abs(applied + lse2_vals[r]) < 2e-3
        n_ok += ok
        if not ok:
            print(f"  r={r:3d} 期望 applied={-lse2_vals[r]:8.4f}  实际 applied={applied:8.4f}  BAD")
    print(f"行权重映射正确: {n_ok}/{rows}")
    return 0 if (n_ok == rows and not bad_const) else 1


if __name__ == "__main__":
    sys.exit(main())
