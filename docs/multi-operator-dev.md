# 多人算子协作开发指南

> 适用范围:本工程(`csrc/ops/<op>/` 模式)及后续新增的 CCE 算子。
> 既有约定(单一事实源分工)见根 README 头部;本文补协作规则。

## 0. 三条总原则

1. **算子即目录**:一个算子 = `csrc/ops/<op>/` 下自包含的
   `op_host/` + `op_kernel/` + `test/` + `design.md`。算子之间不共享源文件,
   冲突面天然最小化。
2. **文档三件套随代码走**:使用方法进 README、内部设计进 `design.md`、
   判据与容差进 `test/*-test-cases.md`——PR 不带三件套不收。
3. **kernel 包零上层依赖**:任何算子禁止 import vllm / vllm-ascend / 插件
   模块;对上层只暴露 `torch.ops.npu.*`(见 README §4 依赖方向)。

## 1. 新增算子标准流程

1. **立项与设计**(写码前):先查 `knowledge/flashinfer-kb/01-可迁移性总表.md` 的可迁移性
   条目;`design.md` 必须含:接口签名、tiling 与内存规划、**对照参考**
   (flashinfer CUDA 原实现或论文公式)、与相邻算子的融合评估(dvm 机会,
   避免"为算子而算子")。
2. **实现**:按 op_host/op_kernel 结构落码。build 开关一律**源内 define**
   (教训:CATLASS_ARCH 若做目标级 `-D` 会漏进同目标其他算子的编译单元,
   见 `kernel_fa_fp32_stage1.cpp` 头注)。
3. **用例**:`test-cases.md` 写明判据与数值容差;精度套件 ≥30 例;有历史
   缺陷域的(如 LSE staging 竞态)必须带回归脚本。
4. **验收门槛(DoD)**:
   - [ ] 精度:对照参考通过,容差在 test-cases.md 声明;
   - [ ] 性能:profiling 数据 + 与标杆(CANN aclnn/官方)对比留存;
   - [ ] 稳定性:同 shape 三次运行无 aicore/NaN;任何 FIA 搭档计时前先对拍
         正确性(**存活≠正确**,README §5);
   - [ ] README 更新:§1 一览表、§2 使用方法、§6 实测锚点;
   - [ ] 共享文件改动符合 §3 纪律。
5. **e2e 义务在消费方**:算子只报单算子与标杆对比;e2e TPOT 变化由接入
   插件的 PR 报告(AGENTS.md:禁止只报单算子加速比)。

## 2. 分支与提交

- 分支:`feat/<op>-<topic>` / `fix/<op>-<topic>`;一个逻辑变更一个 commit
  (conventional 风格)。
- 远端配置走共享开发机统一方案:**SSH over 443**,五步见
  `/vllm-workspace/docs/git-remote-and-network.md`;gh-proxy 只留给 pip git 依赖。

## 3. 共享文件的修改纪律(高审查门槛)

以下文件影响所有人的构建与注册面,改动必须最小 diff 并留痕
(复审分级见 §5,2026-09-22 裁定):

- `csrc/register.cpp`(torch.ops 注册面)
- `csrc/utils/*`(公共 helper)
- `CMakeLists.txt` / `build.sh` / `python/ascend_kernel/setup.py`
- `config.ini`(**版本号单处维护**:新算子合入或既有算子语义变更 → bump)

禁止为单个算子特化共享文件;算子私有开关进自己的目录。

## 4. 版本、wheel 与环境配对

- wheel 是本工程唯一交付物,版本以 `config.ini` 为准;
- **兼容四元组**(平台 tag / PyABI / torch_npu / CANN,README §4)任一变化 =
  全量重编,并按 README §4 顺序回归所有算子:S1 锚点 → 精度套件 → 图捕获冒烟;
- 通知所有消费方(插件仓库)重装新 wheel 并跑各自回归;
- `output/`、`build/` 是本地产物,不入库。

## 5. 所有权与复审分级(2026-09-22 用户裁定)

- **每算子一名 owner**(下表维护)。**owner = 责任接口人**(问责、消费方通知、对外口径),
  **不是代码复审人**——本工程 CCE 层开发与验证在本工作区闭环,团队内无第二人具备
  CCE 代码级复审能力,"待 owner 复审"曾因此成为死项,自本裁定起废除该要求。
- **收口判据 = 机械化验证包**(由执行流程完成并留痕,owner 不参与):
  md5 逐字节比对(纯编译期改动)/ 负向反证(守卫类改动须证"真会拦")/
  lib 身份核查(显式 `PYTHONPATH=仓库构建`,防测成装机版)/ 算子回归套件 / 实机冒烟。
  五层按改动性质裁剪,裁剪理由随证据留痕。
- **仅三类情况升级为人工确认**(找 owner 或项目负责人):
  ① `register.cpp`/构建链/`config.ini` 的**语义**变更(纯版本号 bump 除外);
  ② **对外交付** wheel(消费方重装、出仓);
  ③ 机械化验证覆盖不到的**新语义**(新算子、既有算子语义变更)。

| 算子/共享面 | Owner(责任接口人) |
|---|---|
| `fa_fp32_stage1` | 吴天宇 |
| `lse_merge` | 吴天宇 |
| `register.cpp` / 构建链 | 吴天宇 |

- **NPU 是共享资源**(当前仅 1 卡可见):大 bench / 长占用前在群里打招呼;
  micro-bench 一律分组进程(同进程 FIA 危害见 README §4/§5);
  跑挂了立即清理进程,不留占卡的僵尸任务。
- 跨算子事实(如 FIA v2 的形态互斥、TND 计时探针须逐请求传 `actual_seq_kvlen`,
  传累计值会越界读→静默 NaN/aicore 异常;计时前先对拍)写进各自的 design.md 并
  互相引用,不口头传播。
