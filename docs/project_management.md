# tgv_ptycho_sim 项目管理与实验治理

本文档说明如何组织项目阶段、Codex 任务、实验编号、配置、运行结果和文档。仓库级硬性规则见根目录 `AGENTS.md`；本文提供更详细的执行方法，不替代实验文档或理论文档。

## 1. 管理目标

项目治理要同时满足四个目标：

1. 研究问题可以沿路线图逐步推进，不把未验证假设包装成完成结果。
2. 同一实验可以从 config、代码版本和 run 复现。
3. 公共模块保持共享，不为每个实验复制一套传播、IO 或 reconstruction 代码。
4. 多任务和混合 Git 工作区中不覆盖用户修改、个人记录或历史结果。

## 2. 项目阶段路线图

路线图的理论定义见 `docs/theory_notes/roadmap.md`。当前治理状态如下：

| Phase | 实验主题 | 当前状态 | 主要证据 |
|---|---|---|---|
| Phase 0 | propagation sanity check | 已实现并验证 | `exp001` config、脚本、测试、run、实验记录 |
| Phase 1 | known-probe object-only ePIE | 已实现并验证 | `exp010` config、脚本、测试、run、实验记录 |
| Phase 2 | A thin phase object 生成未知 probe，恢复并 backpropagate | 未独立验证 | 仅有路线图/占位配置或可复用接口 |
| Phase 3 | TGV-like 2D effective phase | 未独立验证 | 仅作为简化模型方向，不代表 3D 腰径 |
| Phase 4 | 3D TGV multi-slice forward | 未独立验证 | 已有初版模块/占位配置，仍需 reference 验证 |
| Phase 5 | waist observability 与 parametric fitting | 未实现或未验证 | 路线图与占位接口 |
| Phase 6 | tilted A 与 multi-angle | 未实现 | 路线图 |
| Phase 7 | noise、stage error、calibration、experimental data | 接口预留，流程未实现 | `preprocess/`、`calibration/` scaffold |

“有文件”不等于“阶段完成”。阶段完成至少需要：明确实验问题、配置、可运行脚本、测试、实际 run、HDF5/metrics/figures 检查和实验记录。

## 3. 总控任务与实验任务

### 3.1 总控任务

总控任务处理跨实验、仓库级问题，例如：

- HDF5 数据规范；
- 坐标、单位和命名约定；
- 项目目录与文档职责；
- 测试、lint、环境和 Git 治理；
- 多个实验共同使用的公共 API 迁移。

总控任务标题示例：

```text
[tgv_ptycho_sim] 项目治理 - HDF5 数据规范
[tgv_ptycho_sim] 调试 - Windows 环境与绘图后端
```

总控任务原则上不产生新的科学实验结论。若治理修改会改变数值结果，必须回归相关实验，并在实验文档中记录影响。

### 3.2 实验任务

实验任务回答一个独立研究问题，例如“已知 probe 时能否恢复随机 B”或“A 的薄相位调制是否能通过恢复 probe 反推”。它必须有稳定实验编号，并使用 `docs/templates/experiment_task_prompt.md` 定义输入、对照、指标和验收标准。

实验任务标题示例：

```text
[tgv_ptycho_sim] exp020 - A 薄相位 probe recovery
```

实验任务的结果写入对应 `docs/experiment_design/expXXX_*.md`，而不是堆进 README。

## 4. 什么时候继续原任务

满足以下情况时，通常继续原任务和原实验编号：

- 修复同一实验的实现 bug；
- 调整同一研究问题下的参数；
- 增加同一结果的必要 figure 或 metric；
- 补充同一 HDF5 的自然 truth/reconstruction 字段；
- 修正文档与代码不一致；
- 在不改变核心假设的情况下增加噪声等级或重复种子。

继续原任务时，仍需创建新的 timestamped run。不得覆盖旧 run 来“更新结果”。

## 5. 什么时候新开任务

出现以下任一情况时，应新开 Codex 任务；多数情况下也应新建实验编号：

- 提出新的科学假设；
- 进入新的 roadmap Phase；
- 从 known probe 改为 unknown probe；
- 从 thin 2D model 改为 3D multi-slice model；
- 使用新的数据来源或真实实验数据；
- 更换核心 forward/reconstruction 模型；
- 新增不同的主要验收目标；
- 项目级 HDF5、坐标或 API 规则需要变更。

如果只是项目治理而不是科学实验，使用“项目治理”或“调试”标题，不占用实验编号。

## 6. 实验编号与资产对应关系

一个正式实验应形成以下映射：

```text
expXXX research question
├── docs/experiment_design/expXXX_<name>.md
├── configs/experiments/expXXX_<name>.yaml
├── scripts/<entrypoint>.py
├── src/tgv_ptycho/<shared modules>
├── tests/<relevant tests>.py
└── runs/expXXX_<name>_<timestamp>/
```

当前映射：

| 实验 | 配置 | 脚本 | 实验文档 | 典型 run |
|---|---|---|---|---|
| exp001 | `configs/experiments/exp001_propagation_sanity.yaml` | `scripts/run_exp001_forward.py` | `docs/experiment_design/exp001_propagation_sanity.md` | `runs/exp001_propagation_sanity_<timestamp>/` |
| exp010 | `configs/experiments/exp010_epie_known_probe.yaml` | `scripts/run_exp010_recon.py` | `docs/experiment_design/exp010_epie_known_probe.md` | `runs/exp010_epie_known_probe_<timestamp>/` |

`exp020` 及后续 YAML 当前可能只是 scaffold。只有完成独立任务和验证后，才能补齐上表并将其视为正式实验。

## 7. 配置管理

- 实验参数写入 YAML，不散落在脚本或公共函数中。
- config 应包含 run name、关键物理参数、算法参数、随机种子和输出开关。
- 公共配置片段可放入 `configs/optics/`、`configs/scan/` 和 `configs/sample_b/`，但当前 config loader 不保证自动继承或合并。是否引入配置组合机制属于待确认设计。
- run 中保存的是本次实际使用的完整 `config.yaml`，不是只保存源 config 路径。
- 修改默认 config 后不得覆盖旧 run；复现实验时保留原 config 副本。

## 8. 公共模块修改与回归

修改 `src/tgv_ptycho/` 公共模块前先判断影响面：

| 模块 | 可能受影响的资产 |
|---|---|
| `optics/` | 所有 forward、backpropagation、multi-slice 和 reconstruction |
| `forward/scan.py`、`sampling.py` | scan position、pixel shift、所有 ptychography 数据 |
| `forward/scheme_probe_B.py` | exp010 及后续方案一实验 |
| `recon/` | exp010、Phase 2 及后续 reconstruction |
| `io/save_load.py` | 所有 HDF5、真实数据接口和已有 reader |
| `inverse/metrics.py` | 仿真评价、报告和参数拟合 |
| `viz/` | 人工判读，不应改变机器可读数值结果 |

公共模块变更流程：

1. 搜索调用方和测试。
2. 说明兼容策略，避免无意改变函数签名或单位。
3. 增加或更新测试。
4. 回归至少一个受影响的既有实验；如数值结果改变，创建新 run。
5. 更新通用理论/数据文档和受影响实验记录。
6. 不删除旧 HDF5 字段或改变语义而不记录 migration。

## 9. 多任务并行与冲突避免

每个任务开始时执行 `git status -sb` 并记录：当前 branch、staged、unstaged、untracked 和 deleted 文件。

并行任务遵循：

- 尽量拆分不重叠的文件所有权，例如一个任务负责 `recon/`，另一个负责独立文档。
- 同一个公共模块只由一个活跃任务修改，除非事先协调。
- 不把用户 notebook、个人报告或未跟踪草稿纳入自动修改范围。
- 不清理或恢复无法识别的变化；先确认来源。
- 每个任务使用自己的 timestamped run，避免输出目录冲突。
- 同一实验文档发生并行修改时，以最新实际 config、代码和 run 为依据人工合并，不整文件覆盖。
- Git 发布范围使用显式文件路径；混合工作区不使用未经确认的 `git add -A`。

## 10. 失败实验如何记录

失败是实验结果的一部分，不应删除痕迹。实验文档至少记录：

- 状态：`Failed` 或 `Inconclusive`；
- 使用的 config、Git commit 和 run 路径；
- 失败发生在哪个阶段；
- 可观察症状和关键 metrics；
- 已排除的原因；
- 是否产生有效的 partial output；
- 下一次尝试需要改变的假设或参数。

如果脚本在创建 run 后失败，不要删除该 run。可在 metadata 或实验记录中标注 incomplete；是否为 run 增加机器可读 status 字段目前待用户确认。

## 11. 废弃与替代实验

实验不因失败或被替代而删除编号、配置、文档或历史 run。

建议在实验文档顶部记录状态：

```text
Status: Deprecated
Replaced by: expXXX
Reason: ...
```

替代实验使用新编号，并在新旧文档中互相链接。旧脚本若仍用于复现可保留；若不再安全运行，应在文档中明确，而不是静默修改其历史语义。

## 12. 复现实验

复现实验不是覆盖原结果，而是创建新 run 并比较：

1. 从原 run 读取 `config.yaml` 和 HDF5 metadata。
2. 记录原 Git commit、Python/依赖环境和随机种子。
3. 在当前代码上执行相同入口，生成新 timestamped run。
4. 比较 machine-readable metrics 和关键 HDF5 datasets。
5. 在实验文档中记录“复现自哪个 run”、差异和结论。

如果无法 checkout 原 commit，不得声称 bitwise reproduction，只能说明是 config-level reproduction。

## 13. 决策和细节记录位置

| 内容 | 记录位置 |
|---|---|
| 仓库级长期规则 | `AGENTS.md` |
| 任务拆分、实验治理和协作方法 | `docs/project_management.md` |
| 项目定位、环境、公共入口、通用数据管理 | `README.md` |
| 路线图与 Phase 定义 | `docs/theory_notes/roadmap.md` |
| HDF5 通用规范 | `docs/theory_notes/data_format.md` |
| 通用算法与物理理论 | `docs/theory_notes/` 中对应文件 |
| 单个实验流程、参数、HDF5、图片、metrics、结果、限制 | `docs/experiment_design/expXXX_*.md` |
| 探索代码与临时分析 | `notebooks/`，不得作为唯一正式记录 |
| 周报和汇报 | `reports/` |

README 是入口，不是实验日志；AGENTS 是规则，不是结果报告；theory notes 不记录某一次 run 的偶然数值；experiment design 不重复整个项目的数据规范。

## 14. 文档维护顺序

发生变化时按责任边界更新：

- 新实验：创建实验文档，再创建 config/脚本，完成后回填实际结果。
- 算法推导变化：更新 theory note，并链接到受影响实验。
- HDF5 结构变化：先更新 `data_format.md`，再修改 writer/reader 和实验文档。
- 项目规则变化：更新 `AGENTS.md` 和必要的模板；README 只增加入口或通用信息。
- 路线图状态变化：需要可验证证据，再更新 `roadmap.md`。

已有文档能表达信息时使用链接，不复制同一字段列表到多个位置。

## 15. Git 与发布

- 默认只做本地修改，不自动 stage、commit、push 或更新 PR。
- Git 发布必须由用户明确授权。
- 发布前列出 included/excluded 文件，尤其要排除 notebooks、个人记录、runs 和大数据。
- 用户要求“只在本地修改”时，文件保持 unstaged；即使已有 Draft PR，也不得自动更新。
- 远程 PR 只反映已推送 commit，不代表本地 unstaged 修改已经发布。

## 16. 当前待确认事项

以下问题不能由单个实验默认决定：

- 全仓 Ruff clean baseline 的建立时间与兼容修复范围。
- 配置继承/组合机制是否引入，以及如何保存展开后的 config。
- HDF5 schema version、compression、chunking 和 migration 策略。
- incomplete/failed run 是否需要统一机器可读状态字段。
- subpixel shift、finite support 和 detector sampling 的标准实现。
- 真实实验 mandatory calibration/preprocessing 字段。

这些问题应使用 `[tgv_ptycho_sim] 项目治理 - ...` 任务单独决策，并同步更新 `AGENTS.md` 或相应 theory note。
