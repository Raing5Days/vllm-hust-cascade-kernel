# AGENTS.md — 算子库(cascade kernel)

## 命名约定(两仓关系)

| 简称 | 内容 | 本地路径 | 远端 |
|---|---|---|---|
| **插件库** | vllm-ascend-split-batch-hust(唯一对外交付面:bundle/manifest、vllm-hust-ext 受管) | `/vllm-workspace/vllm-ascend-split-batch-hust` | `vLLM-HUST/vllm-ascend-split-batch-hust` |
| **算子库**(本仓库) | cascade CCE 算子工程 + S1 锚点数据(wheel 工厂) | `/vllm-workspace/ops/kernels` | `Raing5Days/vllm-hust-cascade-kernel` |

**逻辑关系:插件库 ⊃ 算子库**。插件库通过 `kernels` extra 钉版消费本仓库
的 wheel,单向依赖;本仓库对 vllm-hust-ext、宿主、插件机制零感知。
团队沟通只用"插件库/算子库"两个简称。

## 硬约束

1. **零上层依赖**:禁止 import vllm / vllm-ascend / 插件模块;对上层只暴露
   `torch.ops.npu.*`(`import ascend_kernel` 即注册)。
2. **算子即目录**:`csrc/ops/<op>/` 自包含(op_host + op_kernel + test +
   design.md);文档三件套(README / design.md / test-cases.md)随代码走。
3. **共享文件高门槛**:`csrc/register.cpp`、`csrc/utils/*`、`CMakeLists.txt`、
   `build.sh`、`setup.py`、`config.ini`(版本号单处维护)改动需最小 diff +
   其他算子 owner 复审;build 开关一律源内 define(CATLASS_ARCH 教训)。
4. **兼容四元组**(平台 tag / PyABI / torch_npu / CANN)任一变化 = 全量重编,
   按 S1 锚点 → 精度套件 → 图捕获冒烟顺序回归(README §4);CANN/torch 的
   so 严禁打进 wheel;`build/`、`output/` 等产物不入库。
5. **NPU 是共享资源**:单卡可见;大 bench 先招呼,micro-bench 分组进程
   (同进程 FIA 危害,README §4/§5),不留占卡僵尸任务。

## 常用命令

```bash
cd /vllm-workspace/ops/kernels/ascend-kernel
./build.sh                                                    # 全量重编 + 出 wheel 到 output/
python csrc/ops/fa_fp32_stage1/test/test_fa_fp32_stage1_smoke.py    # S1 锚点
python csrc/ops/fa_fp32_stage1/test/run_precision_suite.py          # 精度 30/30
# git:SSH over 443,见 /vllm-workspace/docs/git-remote-and-network.md
```

## 知识库

- 根 README:使用方法、实测锚点、§4-§5 环境配对与红线
- [docs/multi-operator-dev.md](docs/multi-operator-dev.md):多人协作规则
- [docs/workflow.md](docs/workflow.md):**日常工作流**(单算子循环 + 集成到插件库循环)
- 消费方纪律:插件库 `vllm-ascend-split-batch-hust` 的 AGENTS.md 与 docs/
