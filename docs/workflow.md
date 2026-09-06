# 工作流程(日常工作循环)

> 新算子的立项/验收标准见 [multi-operator-dev.md](multi-operator-dev.md) §1;
> 本文只讲"一个改动的生命周期怎么转"。

## Loop A — 单算子开发循环(算子库内)

```text
design.md 定稿 → 实现(op_host/op_kernel)→ ./build.sh → 精度套件
   ↑______________________________________↓ 不达标回到实现
   达标 → 性能锚点(README §6 加行)→ 文档三件套更新 → commit/PR
```

- 每轮改动必跑:`./build.sh` 后先跑该算子 smoke(S1 锚点),再跑精度套件;
  动了共享构建面才需要全量回归(四元组纪律,AGENTS.md 硬约束 4)。
- commit:conventional 风格 `feat(<op>): ...` / `fix(<op>): ...`,一个逻辑
  变更一个 commit;push 走 SSH over 443(工作区 `docs/git-remote-and-network.md`)。

## Loop B — 集成循环(算子库 → 插件库)

算子改动只有进到插件库的消费面才产生 e2e 价值。每次算子库合入 main 后:

```bash
# ① 算子库:版本与产物
vi  ascend-kernel/config.ini          # 语义变更必 bump 版本(单处维护)
./build.sh                            # 出新 wheel 到 ascend-kernel/output/

# ② 插件库:钉版与验证
cd /vllm-workspace/vllm-ascend-split-batch-hust
#    pyproject.toml 的 [kernels] extra 改成新版本号
pip install ".[kernels]" --find-links /vllm-workspace/cascade-merge-op/ascend-kernel/output
python -c "import ascend_kernel, torch; assert hasattr(torch.ops.npu, 'fa_fp32_stage1')"
pytest -q && ruff check .             # CPU 门槛
#    NPU 冒烟:两段式 Tier1(bf16/fp32 各一轮)+ default-off 对比

# ③ 插件库 commit + push,commit message 引用算子库的 commit hash
```

约定:

- **wheel 版本由算子库 bump,插件库只跟随改 extra 钉版**,两边 commit 互相
  引用 hash,保证配对可追溯(README 兼容矩阵同步更新);
- 纯性能优化(接口/语义不变)可 bump patch 位;接口/tiling 语义变化必须
  minor 及以上,并通知插件侧跑全量冒烟。

## Loop C — 宿主环境变化(CANN / torch_npu 升级)

1. 算子库:四元组更新(README §4 表)→ 全量重编 → 按序回归
   (S1 锚点 → 精度套件 → 图捕获冒烟);
2. 插件库:按其 `docs/release.md` §4 宿主核对清单过一遍 monkeypatch 面;
3. 两仓各自提交"环境迁移"记录,锚点数字更新进 README §6。

## 快速对照:谁改什么

| 改动 | 算子库 | 插件库 |
|---|---|---|
| 新算子 | 目录 + 三件套 + config.ini bump + build | `kernels` extra bump + 冒烟 |
| 算子 bugfix | fix commit + 回归 | 仅在语义变化时跟随 |
| 共享构建面 | 最小 diff + owner 复审 + 全量回归 | 冒烟确认 |
| CANN/torch 升级 | 四元组 + 全量回归 | release.md §4 核对 |
