# 新实验 Codex 任务提示词模板

复制本文件内容到新的 Codex 任务，填写所有 `<...>` 占位符。未决定的项目必须明确写“待用户确认”，不要删除字段或让 Codex自行假设。

---

## 任务标题

```text
[tgv_ptycho_sim] exp<XXX> - <实验简短名称>
```

标题必须以 `[tgv_ptycho_sim]` 开头。项目治理或调试任务可使用：

```text
[tgv_ptycho_sim] 项目治理 - <主题>
[tgv_ptycho_sim] 调试 - <主题>
```

## 开始前必须执行

1. 首先读取项目根目录 `AGENTS.md`。
2. 再读取本任务依赖的 `docs/experiment_design/` 实验记录和 `docs/theory_notes/` 理论文档。
3. 执行 `git status -sb`，识别 staged、unstaged、untracked 和 deleted 文件。
4. 保留用户现有修改，不回退、不覆盖、不删除来源不明的文件。
5. 不覆盖任何历史 run；每次实际运行创建新的 timestamped run。
6. 不擅自修改项目级 HDF5、坐标、单位或目录规范。如确需修改，先列为待确认并说明兼容影响。
7. 不把实验流程、实验 HDF5 字段、逐张图片或实验指标重复写入 README。
8. 完成后更新对应 `docs/experiment_design/exp<XXX>_<name>.md`。

## 实验编号

```text
exp<XXX>
```

- 所属 Phase：`Phase <N>`
- 实验状态：`Planned / Running / Passed / Failed / Inconclusive`
- 是否为新研究问题：`是 / 否`
- 若不是新问题，说明为何继续原实验编号：`<说明>`

## 背景

<说明该实验位于 TGV ptychography pipeline 的哪个位置，以及现有方法为什么不足。不要复制整个 README。>

## 研究问题

<用一个可回答的问题描述本实验。例如：“在已知 B 和 integer-pixel scan 条件下，能否从 intensity stack 稳定恢复由薄相位 A 产生的未知 probe？”>

## 实验假设

1. <假设 1>
2. <假设 2>
3. <阶段性简化及适用边界>

明确说明哪些假设不代表真实 3D TGV 或真实实验条件。

## 所依赖的已有实验

- 依赖实验：`exp<XXX>`
- 依赖 run：`<可选，写完整路径>`
- 依赖实验文档：`docs/experiment_design/<file>.md`
- 依赖理论/数据文档：`docs/theory_notes/<file>.md`
- 复用公共模块：`src/tgv_ptycho/<module>.py`

## 基线配置

- 基线 config：`configs/experiments/<file>.yaml`
- 基线 Git commit：`<commit 或“不固定”>`
- Python/环境：`tgv_ptycho_sim, Python 3.11`
- 随机种子策略：`<固定种子 / 多种子统计>`
- 预期 run name：`exp<XXX>_<name>`

## 允许修改的范围

- `<允许修改的文件或目录 1>`
- `<允许修改的文件或目录 2>`
- 对公共模块的允许改动：`<说明>`

## 禁止修改的内容

- 用户已有 unstaged/untracked 文件。
- 历史 `runs/`、notebooks、个人报告和原始数据。
- 与本实验无关的算法、配置和测试。
- 项目级 HDF5、坐标、单位和文档职责规范，除非本任务明确授权。
- `<其他禁止项>`

## 输入

- 数据来源：`simulation / experimental / existing HDF5 / other`
- 输入 datasets：`<HDF5 path、数组 shape、dtype>`
- 坐标顺序与单位：`<例如 scan_positions=(x,y), m>`
- instrument metadata：`<wavelength、dx、distance、pixel size 等>`
- 前处理/标定状态：`<说明；仿真可写不适用>`

## 参数和对照组

### 主实验参数

| 参数 | 基线值 | 单位 | 来源/理由 |
|---|---:|---|---|
| `<parameter>` | `<value>` | `<SI unit>` | `<reason>` |

### 对照组

| 组别 | 变量 | 取值 | 其余条件 |
|---|---|---|---|
| Baseline | `<变量>` | `<值>` | 固定 |
| Control 1 | `<变量>` | `<值>` | 与 baseline 相同 |

## 评价指标

- 数据一致性：`<例如 relative amplitude loss>`
- reconstruction accuracy：`<例如 complex relative error>`
- amplitude 指标：`<例如 amplitude RMSE>`
- phase 指标：`<例如 wrapped phase RMSE>`
- TGV 指标：`<例如 D_waist absolute/relative error>`
- 稳定性：`<种子、噪声、stage error 统计>`
- 失败判据：`<NaN、发散、误差阈值等>`

说明是否需要 global-phase alignment。若使用 truth alignment，必须标注为 simulation evaluation only，保留 raw reconstruction。

## HDF5 输出要求

- 输出文件名：`outputs/<name>.h5`
- 必须包含：
  - `/entry/data/I_stack`
  - `/entry/data/scan_positions`
  - `/entry/instrument/...`
  - `/entry/sample/...`
  - `/entry/config_yaml`
  - `/entry/metadata/...`
  - `/entry/metrics/...`
- 仿真 truth：`<列出本实验自然产生的字段>`
- reconstruction：`<列出本实验实际产生的字段>`
- calibration/preprocessing：`<真实实验时列出；仿真通常不适用>`
- 新增字段：`<字段、shape、dtype、单位、语义>`

不得伪造不适用的 group。详细结构遵循 `docs/theory_notes/data_format.md`。

## 图片输出要求

| 文件名 | 物理量 | 坐标/单位 | colorbar | 用途 |
|---|---|---|---|---|
| `<name>.png` | `<quantity>` | `<axes>` | `<label>` | `<human check>` |

要求：

- 标题、坐标轴和 colorbar 不遮挡数据。
- 明确区分 amplitude、phase、intensity 和 error。
- PNG 仅供人工查看，数值结果必须进入 HDF5/metrics。

## 验收标准

1. `<可量化标准 1>`
2. `<可量化标准 2>`
3. HDF5 与外部 config/metadata/metrics 语义一致。
4. figures 可读且与保存 datasets 对应。
5. 不覆盖历史 run，不使用 truth 泄漏帮助 reconstruction。

## 测试要求

- 新增/修改公共接口的 unit tests：`<列表>`
- shape、dtype、单位或异常输入测试：`<列表>`
- HDF5 layout 测试：`<需要/不需要及原因>`
- 回归实验：`<exp001/exp010/其他>`
- 运行命令：

  ```powershell
  python -m pytest -q
  python -m ruff check <本次修改的 Python 路径>
  ```

若项目级 Ruff 仍存在既有问题，区分本次新增问题与历史问题，不在无关任务中顺手修复。

## 文档要求

- 创建或更新：`docs/experiment_design/exp<XXX>_<name>.md`
- 通用算法变化时更新：`docs/theory_notes/<file>.md`
- HDF5 通用结构变化时先更新：`docs/theory_notes/data_format.md`
- README：只在需要新增公共入口时修改，不写实验细节。

## Git 操作是否授权

选择且只选择一项：

- `[ ]` 只允许本地修改，保持 unstaged，不 commit、不 push、不更新 PR。
- `[ ]` 允许 stage 和本地 commit，但不 push。
- `[ ]` 允许创建分支、commit、push 和 Draft PR。
- `[ ]` 其他：`<明确范围>`

发布时必须列出 included/excluded 文件。未勾选或表述不清时，默认为只允许本地修改。

## 完成交付清单

- [ ] 实验设计文档已创建/更新
- [ ] YAML config 已创建/更新
- [ ] 公共模块已实现或复用
- [ ] 运行脚本已完成
- [ ] tests 已通过或失败已记录
- [ ] 新 timestamped run 已生成
- [ ] HDF5 tree 已检查
- [ ] metrics 已检查
- [ ] figures 已人工检查
- [ ] 实际结果和限制已写回实验文档
- [ ] 未覆盖历史 run 或用户修改
- [ ] Git 状态和发布行为已在最终回复说明

## 最终回复必须包含

1. 修改文件。
2. 实际命令与测试结果。
3. 新 run 路径。
4. 关键指标。
5. HDF5 新增/变化字段。
6. 已知限制。
7. Git staged/commit/push/PR 状态。
8. 下一步应继续当前实验还是新开实验。

---
