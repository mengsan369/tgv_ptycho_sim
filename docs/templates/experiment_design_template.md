# expXXX：实验名称

```text
Status: Planned
Phase: Phase N
Owner: <name or team>
Created: YYYY-MM-DD
Last updated: YYYY-MM-DD
```

状态建议使用：`Planned`、`Running`、`Passed`、`Failed`、`Inconclusive`、`Deprecated`、`Replaced`。

## 1. 实验编号与名称

- 实验编号：`expXXX`
- 实验名称：`<name>`
- 所属 pipeline：`A-generated probe + B scan / 3D multi-slice / shared infrastructure`
- 对应 Codex 任务：`[tgv_ptycho_sim] expXXX - <name>`

## 2. 所属 Phase

- Phase：`Phase N`
- 路线图位置：`docs/theory_notes/roadmap.md`
- 该 Phase 在实验前的状态：`<未实现/部分实现/已验证>`

## 3. 目的与假设

### 目的

<本实验要回答的单一研究问题。>

### 假设

1. <假设 1>
2. <假设 2>

### 适用边界

<说明理想化条件、简化模型和不能外推的结论。>

## 4. 理论依据

<简述必要公式和物理依据，详细推导链接到 `docs/theory_notes/`，不要在多个实验文档重复整篇理论。>

```text
<核心公式>
```

- 理论文档：`docs/theory_notes/<file>.md`
- 参考文献：`<DOI / citation>`

## 5. 依赖实验

| 依赖实验 | 使用内容 | 文档 | 基线 run |
|---|---|---|---|
| `expXXX` | `<传播/数据/算法>` | `docs/experiment_design/<file>.md` | `<run path or N/A>` |

## 6. 启动脚本

- 脚本：`scripts/<script>.py`
- 工作目录：项目根目录
- 命令：

  ```powershell
  python scripts/<script>.py --config configs/experiments/expXXX_<name>.yaml
  ```

- VS Code debug 配置：`<name or N/A>`

## 7. 配置文件

- 主配置：`configs/experiments/expXXX_<name>.yaml`
- 复用配置：`<paths or N/A>`
- 随机种子：`<value/strategy>`
- config 版本或 Git commit：`<value>`

## 8. 数据流

```text
<input>
-> <forward/preprocessing>
-> <measurement>
-> <reconstruction/inverse>
-> <metrics and figures>
```

明确：

- 输入/输出 shape：`<shape>`
- dtype：`<dtype>`
- 坐标顺序：`<axis and scan order>`
- 单位：`<SI units>`

## 9. 关键参数

| 参数 | 值 | 单位 | 作用 | 来源 |
|---|---:|---|---|---|
| `<parameter>` | `<value>` | `<unit>` | `<meaning>` | `<config/default>` |

## 10. 对照组与扫描条件

| 组别 | 改变量 | 取值 | 固定条件 | 目的 |
|---|---|---|---|---|
| Baseline | `<variable>` | `<value>` | `<conditions>` | `<purpose>` |

## 11. 输出 run 结构

```text
runs/expXXX_<name>_YYYYMMDD_HHMMSS/
├── config.yaml
├── metadata.json
├── metrics.json
├── figures/
│   └── <figure>.png
└── outputs/
    └── <output>.h5
```

实际 run 路径：`<运行后填写；未运行写 N/A>`

## 12. HDF5 结构

只记录本实验实际使用的字段。通用规则链接 `docs/theory_notes/data_format.md`。

```text
/entry/data/...
/entry/instrument/...
/entry/sample/...
/entry/truth/...              # simulation only
/entry/calibration/...        # experimental when applicable
/entry/preprocessing/...      # experimental when applicable
/entry/reconstruction/...
/entry/config_yaml
/entry/metadata/...
/entry/metrics/...
```

| Dataset | Shape | dtype | 单位 | 语义/来源 |
|---|---|---|---|---|
| `/entry/...` | `<shape>` | `<dtype>` | `<unit>` | `<meaning>` |

truth-aided alignment 或其他 simulation-only 结果必须明确标记，并与 raw reconstruction 分开。

## 13. 图片及物理含义

| 图片 | 显示量 | 横纵坐标 | colorbar | 对应 HDF5 dataset | 判读目的 |
|---|---|---|---|---|---|
| `<file>.png` | `<amplitude/phase/intensity/error>` | `<axes, units>` | `<label, units>` | `/entry/...` | `<purpose>` |

记录所有输出图片，不使用图片作为后续数值计算来源。

## 14. Metrics

| Metric | 定义 | 计算区域/对齐 | 单位 | 验收阈值 |
|---|---|---|---|---|
| `<name>` | `<formula>` | `<mask/alignment>` | `<unit>` | `<threshold>` |

说明 global phase、phase wrapping、illumination mask 和 normalization 的处理。

## 15. 验收标准

- [ ] `<定量标准 1>`
- [ ] `<定量标准 2>`
- [ ] HDF5 与外部 config/metadata/metrics 一致
- [ ] figures 可读且对应正确物理量
- [ ] tests 通过或已记录失败原因
- [ ] 未使用 truth 泄漏帮助 reconstruction

## 16. 实际结果

### 运行信息

- 运行日期：`<date>`
- Git commit：`<commit>`
- run 路径：`<path>`
- Python/environment：`<version>`
- 运行命令：`<command>`

### 数值结果

| Metric | 结果 | 是否通过 |
|---|---:|---|
| `<name>` | `<value>` | `Yes/No` |

### 观察

<描述 figures、loss、异常和对照差异，不夸大结论。>

## 17. 已知限制

- <数值模型限制>
- <采样/边界限制>
- <噪声与标定限制>
- <不能外推到真实 TGV 的部分>

区分阶段性限制、未实现功能和待用户确认的设计。

## 18. 失败、废弃或替代记录

- 是否失败/废弃：`No / Failed / Inconclusive / Deprecated`
- 原因：`<reason>`
- 保留 run：`<path>`
- 替代实验：`expXXX / N/A`
- 对后续实验的影响：`<impact>`

不得删除失败实验的编号、文档或已有 run。

## 19. 结论

<用与验收标准直接对应的语言回答研究问题。明确哪些结论只在当前条件下成立。>

## 20. 下一步实验

- 继续当前实验的小修正：`<items or N/A>`
- 建议新实验编号：`expXXX`
- 新研究问题：`<question>`
- 开启新任务前需要确认：`<decisions>`
