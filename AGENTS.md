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

## 构建与环境的已知陷阱（2026-09-19 实测，全部踩过）

这几条的共同症状是**构建"成功"但 device 侧代码没变**——测量随之全错。改 kernel 前先读本节。

1. **`build.sh` 必须在仓库根目录执行。** 它内部用 `CURRENT_DIR=$(pwd)` 推 `PROJECT_ROOT`；
   从 `/tmp` 之类目录调用会让 cmake 落在错误目录，**device 侧静默不重编**，只有 host 侧与
   wheel 被更新。判据：`build/…/auto_gen_kernel_<op>.cpp.o` 的 mtime 没变。
2. **本工程有两个 `build` 目录，语义不同。** 真产物在 `$REPO/build`（`build.sh` 用）；
   `$REPO/csrc/build` 是 `csrc/CMakeLists.txt` 被当根工程执行时留下的**残骸**。
   用后者判断新鲜度会误判"没重编"。判据只用 `$REPO/build`。
3. **给本工程加自己的编译宏，不要走 `-D`/`CXXFLAGS`。** 构建链会把环境 `CXXFLAGS`
   吸收成 device 侧的 `-D` 并**长期缓存**（实测所有 kernel 的 device 编译行里都曾带着
   同一个项目的宏），它会**压过源码里的 `#ifndef` 默认值**，导致开关静默失效。
   正确做法：源内 define（同硬约束 3 的 CATLASS_ARCH 教训），用**独有宏名**、且**不写
   `#ifndef` 守卫**（守卫是让缓存宏生效的帮凶）。
4. **不要手删 ExternalProject 的 `CMakeCache.txt` 来"强制重配置"。** 会留下 `.o` 与产物
   互相污染的中间态，`ld.lld` 报 `unknown file type` 且构建可能仍返回成功。
   一律走 `build.sh` 全量构建（它自己 `rm -rf build`），慢但可靠。
5. **改了 kernel 后，先验"确实重编了"再测量。** 可靠判据（本仓库脚本已内置）：
   device 目标文件 mtime ≥ 源码 mtime。不满足就硬失败，不要拿旧二进制继续跑。

自动化护栏见 `ascend-kernel/csrc/ops/lse_merge/prof/build_probe.sh`（含上述校验）。

### 测量前先自检仪器（2026-09-19 追加，血的教训）

上面 1–4 条的共同表象是**"不同输入产出同一个二进制"**，而这类错误**不会自己报警**：
构建返回 0、脚本继续跑、数字看着很稳。2026-09-19 因此产出过一批假读数，
其中一条还被当成正面结果汇报出去（`Brcb` 行广播"32/32"），事后完全不复现。

**硬性规则**：

1. **先自检仪器，再读它的数。** 跑
   `bash csrc/ops/lse_merge/prof/selftest.sh`——它断言
   "探针 0 与探针 3 必须产出不同 md5"、"生产配置可复现"、"生产路径与 bit 锚点全等"。
   自检不过，任何测量都不作数。
2. **同一个 md5 出现两次 = 警报，不是"稳定"。** 两次不同配置构建出同一二进制，
   几乎必然是配置没进去（见第 1/2/4 条），必须停下来查，不能接着往下测。
3. **只信产物，不信退出码。** 断言要落在产物上（`.so` md5、device 目标 mtime、
   与锚点的 bit 比较），不是"命令返回 0"。
4. **正面结果必须独立重跑复现后才可采信。** 单次通过不算通过；
   本仓库已经出现过"结果依赖 UB 缓冲地址、换一次就变"的情形。
5. **不要用带外通道传编译期配置。** 环境变量会被构建链吸收并缓存（第 3 条），
   一律源内 define；也不要假设 `bash <abs>/build.sh` 等价于在仓库里执行（第 1 条）。


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
