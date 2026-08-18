# tgv_ptycho_sim 仓库级协作规范

本文件是本仓库内所有 Codex 任务的默认规则。执行任务前先阅读本文件，再读取与任务相关的 `docs/experiment_design/` 和 `docs/theory_notes/` 文档。除非用户在当前任务中明确覆盖，以下规则持续有效。

## 1. 项目定位

本项目用于研究基于 ptychography / PIE 的 TGV（Through Glass Via，玻璃通孔）内部腰径测量。目标量为孔径随深度的变化

```text
D(z)
D_waist = min_z D(z)
```

项目包含两条共享底层框架的 pipeline：

1. 样品 A 调制入射场并在 B 平面形成未知 probe；可移动编码样品 B 通过 overlapping scan 提供 ptychographic redundancy；由 detector intensity 恢复 B-plane probe 和 B，再将 probe backpropagate 到 A 附近用于 TGV 参数估计。
2. 将 TGV 样品 A 表示为 `n(x, y, z)` 或多个薄层 transmission，使用 3D multi-slice forward model 逐层调制和传播。

两条 pipeline 共用 `optics`、`forward`、`recon`、`inverse`、`io` 和数据格式。不得拆成两个互不兼容的项目，也不得为单个实验复制一套平行基础设施。

## 2. 当前阶段

以下是已由当前代码、配置和测试确认的状态：

- `exp001`：Phase 0 propagation sanity check，已经实现并运行验证。实验记录见 `docs/experiment_design/exp001_propagation_sanity.md`。
- `exp010`：Phase 1 known-probe、object-only ePIE，已经实现并运行验证。probe 固定，只恢复随机样品 B。实验记录见 `docs/experiment_design/exp010_epie_known_probe.md`。
- 当前测试基线为 Python 3.11 环境下 `12 passed`。
- Phase 2 及后续阶段仍需作为独立实验任务逐项实现和验证。配置占位、函数签名或 TODO 不等于阶段完成。

必须保持以下表述边界：

- 2D effective phase model 只用于早期 sensitivity / observability 验证，不等价于真实 3D TGV 腰径模型。
- 当前 ePIE 是理想仿真条件下的 baseline，不是 production-grade reconstruction engine。
- 当前 integer-pixel shift、periodic `np.roll` boundary、无噪声配置、同 shape sampling 等均为阶段性限制，不得提升为项目永久设计。
- VS Code 中的 `MPLBACKEND=Agg` 和当前非交互式 PNG 写入方式是 Windows 调试/绘图兼容措施，不是所有未来可视化必须遵守的算法架构。

路线图状态以 `docs/theory_notes/roadmap.md` 为准。只有具备实验文档、可运行入口、测试和实际结果时，才能把某一 Phase 标记为已实现。

## 3. 目录职责

- `configs/`：YAML 参数。实验配置放 `configs/experiments/`，可复用的 optics、scan、sample B 配置放对应子目录。
- `src/tgv_ptycho/`：可复用实现。公共传播、对象生成、forward、reconstruction、inverse、IO、preprocess、calibration 和 viz 逻辑放这里。
- `scripts/`：可执行入口。脚本负责读取 config、编排公共模块、创建 run 和保存结果，不承载可复用算法主体。
- `tests/`：自动化验证。新增或修改公共接口时必须补充与风险匹配的测试。
- `docs/theory_notes/`：通用理论、算法依据、路线图和跨实验数据格式。
- `docs/experiment_design/`：实验专属记录。一个实验编号对应一个 Markdown 文件。
- `notebooks/`：探索、交互分析和个人调试。notebook 不作为唯一实现，也不自动纳入 Git 提交。
- `data/`：长期数据区。`raw` 保存不可手动覆盖的实验原始数据；`simulated` 保存可复现仿真数据；`processed` 保存派生数据；`calibration` 保存标定数据；`external` 保存外部数据。
- `runs/`：每次执行产生的独立运行产物。不能作为源代码、长期配置或手工编辑数据的目录。
- `reports/`：周报、汇报材料和导出图表。个人周报和草稿不得自动提交。

实现位置遵循以下规则：可复用实现放 `src/`，可执行入口放 `scripts/`，参数放 YAML，实验说明放 `docs/experiment_design/`，通用理论与数据格式放 `docs/theory_notes/`。

## 4. 实验编号和任务规则

- 实验使用 `exp001`、`exp010`、`exp020` 等稳定编号。编号一旦被正式文档和 run 使用，不因重构随意改名。
- 一个独立研究问题对应一个实验编号。
- 一个独立实验原则上对应一个 Codex 任务。
- 同一实验的小修正、调参、补图、错误修复和文档补充可以继续原任务。
- 新假设、新 pipeline 阶段、新数据来源、不同 forward model 或新的验收目标应新开任务。
- 所有本项目 Codex 任务标题必须以 `[tgv_ptycho_sim]` 开头。

建议标题：

```text
[tgv_ptycho_sim] exp020 - A 薄相位 probe recovery
[tgv_ptycho_sim] exp040 - 3D TGV multi-slice forward
[tgv_ptycho_sim] 项目治理 - HDF5 数据规范
[tgv_ptycho_sim] 调试 - Windows 环境与绘图后端
```

新实验任务优先使用 `docs/templates/experiment_task_prompt.md`。

## 5. 实验标准流程

独立实验原则上按以下顺序完成：

1. 在 `docs/experiment_design/` 编写或初始化实验设计文档。
2. 创建对应 YAML 配置，记录默认参数和随机种子。
3. 实现或复用 `src/tgv_ptycho/` 公共模块。
4. 编写或扩展 `scripts/` 运行入口。
5. 补充测试。
6. 实际运行实验。
7. 创建独立 timestamped run，不覆盖历史结果。
8. 检查 HDF5、外部 `metrics.json` 和所有 figures。
9. 用实际结果更新对应实验记录。
10. 汇报结果、限制、Git 状态和下一步。

如果实验失败，也必须保存可诊断信息并更新实验记录，不得只留下聊天结论。

## 6. runs 管理

每个 run 的建议结构为：

```text
config.yaml
metadata.json
metrics.json
figures/
outputs/*.h5
```

规则：

- 每次执行创建新的 `runs/<run_name>_YYYYMMDD_HHMMSS/`。
- 不覆盖或删除已有 run。需要复现时也创建新 run。
- 同秒重名时追加 `_01`、`_02` 等后缀；当前 `make_run_dir()` 已实现该行为。
- `config.yaml`、`metadata.json` 和 `metrics.json` 面向人工审阅；语义相同的数据应同时进入 HDF5。
- PNG 只用于人工查看，不作为后续计算的主要数据源。
- HDF5 是机器读取、批处理和复现的主要数据源。
- 大型 runs、HDF5、生成 PNG、缓存和大数据不得提交 Git。

## 7. HDF5 约定

内部主格式是 CXI / NeXus-inspired HDF5。`/entry` 下采用并列结构：

```text
/entry/config_yaml
/entry/data
/entry/instrument
/entry/sample
/entry/truth
/entry/calibration
/entry/preprocessing
/entry/reconstruction
/entry/metadata
/entry/metrics
```

规则：

- `config_yaml`、`metadata` 和 `metrics` 必须并列，不得把 config 或 metrics 塞进 metadata。
- `truth`、`calibration`、`preprocessing` 和 `reconstruction` 按数据来源与处理阶段出现，不得为了结构整齐写入伪造或空洞结果。
- 仿真数据可以包含 `truth`；真实实验数据不得包含 `truth`。
- 真实实验数据应记录实际执行的 `calibration` 和 `preprocessing`。尚未执行的步骤不得伪装成已完成。
- 仿真 truth 只能用于评估、误差计算和画图，不能泄漏进真实实验 reconstruction 算法。
- truth 字段按实验自然产生的内容增加，不要求所有脚本硬写完全相同的 truth。
- reconstruction 结果由对应重建脚本写入。纯 forward 实验可以没有 reconstruction group。
- 任何使用 truth 做 global-phase alignment 的结果必须用清晰名称标明“simulation evaluation only”，原始 reconstruction 结果必须单独保留。
- 仿真和真实数据都应尽量统一 `I_stack`、`scan_positions`、instrument metadata 和 reconstruction result 的语义。

详细字段、simulation/experimental 差异和 CXI / NeXus 关系见 `docs/theory_notes/data_format.md`。修改项目级 HDF5 结构前必须先更新该文档并评估已有实验兼容性。

## 8. 数值和代码规范

- Python 3.11，src-layout，import package 为 `tgv_ptycho`。
- 物理量优先使用 SI 单位：m、rad 等。展示层可转换为 `um`、`nm`，但必须明确标签。
- 明确数组 shape、axis 顺序、坐标顺序和单位。当前二维场约定通常为 `(ny, nx)`，scan position 列顺序为 `(x, y)`，单位 m。
- `dx` 为 tuple 时必须在接口文档中明确其顺序；不得依靠猜测混用 `(dx, dy)` 与 `(dy, dx)`。
- complex field、amplitude `abs(U)`、phase `angle(U)` 和 intensity `abs(U) ** 2` 必须区分命名和保存。
- 使用 type hints 和核心函数 docstring。
- 优先函数式实现；只有明确减少复杂度时才引入 class。
- 参数优先来自 config，不在算法函数或脚本中散落硬编码实验参数。
- 新增或修改公共接口必须有测试；共享传播和 IO 变更需回归已有实验。
- 不使用仿真 truth 帮助真实实验重建。
- 对未实现内容使用 TODO 或 `NotImplementedError`，不得用占位结果冒充完成。
- 不擅自修改用户现有代码、notebook、草稿或 runs；遇到相关变更时与其共存。

## 9. 验证命令

当前仓库与本机已核对的环境：

```powershell
conda activate tgv_ptycho_sim
pip install -e .
python -m pytest -q
```

当前解释器为 Python 3.11；本机 VS Code 应选择：

```text
D:\anaconda3\envs\tgv_ptycho_sim\python.exe
```

其他机器应选择名为 `tgv_ptycho_sim` 的 Conda 环境解释器，不得把上述绝对路径写进跨机器配置。

pytest 已在 `pyproject.toml` 中设置：

```text
--basetemp=.pytest_tmp
```

这是为了避免当前 Windows 环境系统临时目录权限导致 fixture 失败，并让临时测试文件落在项目内可控路径；`.pytest_tmp/` 已被 Git 忽略。

Ruff 命令：

```powershell
python -m ruff check <本次修改的 Python 文件或目录>
```

项目级诊断命令为：

```powershell
python -m ruff check src scripts tests
```

截至本文件创建时，项目级 Ruff 仍报告既有 scaffold 和用户修改中的 14 个问题，因此它是诊断命令，尚不是“全仓必须全绿”的已验证门槛。不得在无关任务中顺手修改这些文件；是否建立全仓 Ruff clean baseline 见“待用户确认”。

已验证实验入口：

```powershell
python scripts/run_exp001_forward.py --config configs/experiments/exp001_propagation_sanity.yaml
python scripts/run_exp010_recon.py --config configs/experiments/exp010_epie_known_probe.yaml
```

`.vscode/launch.json` 已提供 exp001、exp010 和当前 Python 文件的 debug 配置。其中 `MPLBACKEND=Agg` 是当前 Windows/debugpy 绘图兼容设置，不应自动推广为所有环境的永久要求。

## 10. 文档职责

- `README.md` 只保留项目定位、环境、公共运行入口、总体目录和通用数据管理原则。
- README 不重复具体实验流程、实验 HDF5 字段、逐张图片含义或实验指标。
- 每个实验在 `docs/experiment_design/` 下建立一个 Markdown 文件；一个实验对应一个文件。
- 实验专属流程、脚本、配置、HDF5 字段、图片、指标、实际结果和限制写入实验文档。
- 通用理论、算法说明、路线图和通用数据格式写入 `docs/theory_notes/`。
- 仓库级执行规范写入 `AGENTS.md`；更详细的项目管理方法写入 `docs/project_management.md`。
- 已有详细文档时使用链接引用，避免多处复制导致不一致。

## 11. Git 规则

- 修改前先运行 `git status -sb`，识别 staged、unstaged、untracked 和 deleted 文件。
- 保留用户所有未提交修改，不覆盖、不删除、不回退。
- 不提交 runs、HDF5、生成 PNG、缓存和大数据；使用 `.gitkeep` 保留必要空目录。
- notebooks、个人周报、实验草稿和个人记录不得自动纳入提交。
- 只有用户明确要求时才执行 `git add`、commit、push、PR 或 merge。
- 用户要求“只在本地修改”时，修改必须保持 unstaged，不得提交或上传。
- 发布前必须明确列出计划提交和排除的文件；混合工作区必须使用显式路径暂存，不使用未经确认的 `git add -A`。
- 不使用 destructive Git 命令处理用户变更。
- Git 操作与科学结果是两件事：不能因为 run 已验证就默认获得发布授权。

## 12. 任务完成时的汇报格式

后续任务完成后至少汇报：

1. 创建或修改的文件。
2. 实际运行命令。
3. 测试和 lint 结果，包括未通过项。
4. 新 run 路径；未运行实验时明确写“无新 run”。
5. 关键指标；不适用时说明原因。
6. HDF5 新增或变化的字段；无变化时明确说明。
7. 已知限制和阶段性假设。
8. 是否改变 Git staged/commit/push/PR 状态。
9. 下一步建议，区分继续当前实验还是新开实验。

## 13. 待用户确认

以下事项当前没有形成长期决定，后续不得自行假定：

- 是否以及何时把全仓 `ruff check src scripts tests` 建立为强制全绿门槛。
- subpixel scan shift 的具体实现、finite object support 和 periodic boundary 的替代方案。
- detector pixel integration、sample/detector sampling remap 和不同 pixel size 的标准处理方式。
- HDF5 schema version、compression/chunking 策略，以及正式 CXI / NeXus 导出兼容级别。
- Phase 2 unknown-probe reconstruction 的更新策略、约束、验收指标和是否采用 ePIE/rPIE。
- 真实实验数据的 mandatory calibration 字段及缺失标定时的失败策略。

遇到上述问题时，应在实验或治理任务中显式提出并记录决策，不得根据 exp001/exp010 的临时选择推断。
