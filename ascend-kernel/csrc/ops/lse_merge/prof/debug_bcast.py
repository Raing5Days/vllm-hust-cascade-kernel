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

"""块广播路径的定点定位：用可解析输入把中间量逼出来。

用例设计（o1 全 1、o2 全 0）：
  out[r, :] = (1*w1[r] + 0*w2[r]) * sInv[r] = w1[r] / s[r]
  - 令 lse1 = 0  ⇒ m = 0, w1 = 1
  - 令 lse2 = L   ⇒ w2 = exp(L), s = 1 + exp(L)
  ⇒ out[r, :] 应为常数 1/(1+exp(L[r]))
用不同的 L[r] 就能看出"哪一行拿到了谁的权重"。
"""

import json
import sys

import ascend_kernel  # noqa: F401
import torch
import torch_npu  # noqa: F401

T, H, D = 4, 8, 128   # rows = 32
rows = T * H


def run(lse2_vals, out_code=2):
    o1 = torch.ones(T, H, D, dtype=torch.float32, device="npu")
    o2 = torch.zeros(T, H, D, dtype=torch.bfloat16, device="npu")
    l1 = torch.zeros(T, H, dtype=torch.float32, device="npu")
    l2 = torch.tensor(lse2_vals, dtype=torch.float32, device="npu").reshape(T, H)
    out = torch.ops.npu.lse_merge(o1, o2, l1, l2, out_code)
    torch.npu.synchronize()
    return out.cpu()


def main():
    # 每行的 L 取不同整数，便于反查行映射
    lse2 = [float(i) for i in range(rows)]
    exp_l = torch.exp(torch.tensor(lse2, dtype=torch.float64))
    expect = (1.0 / (1.0 + exp_l))

    out = run(lse2)
    got = out.reshape(rows, D).double()

    # 每行应当是全常数
    row_const = got.max(dim=1).values - got.min(dim=1).values
    print("每行是否常数（max-min，应全为 0）:")
    bad = (row_const.abs() > 1e-9).nonzero().flatten().tolist()
    print(f"  非常数的行: {bad[:10]}{' ...' if len(bad) > 10 else ''}  共 {len(bad)}/{rows}")

    # 行映射检查：第 r 行的值应当等于 expect[r]
    got_row0 = got[:, 0]
    diff = (got_row0 - expect).abs()
    print("行映射检查（第 r 行 vs 期望 1/(1+exp(L[r]))) :")
    print(f"  最大绝对差 {diff.max().item():.3e}  超差行数 {(diff > 1e-6).sum().item()}/{rows}")

    # 前 6 行逐行打印，看是不是错位
    print("  逐行对照（前 8 行）:")
    print(f"    {'r':>3} {'got[r,0]':>14} {'expect[r]':>14} {'got[r,1]':>14}")
    for r in range(min(8, rows)):
        print(f"    {r:>3} {got_row0[r].item():>14.8f} {expect[r].item():>14.8f} {got[r,1].item():>14.8f}")

    res = {"non_constant_rows": bad, "max_abs_diff": diff.max().item()}
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
