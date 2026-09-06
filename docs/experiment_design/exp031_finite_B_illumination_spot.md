# exp031：有限非周期大 B 覆盖与 A 面光斑尺寸敏感性

## 0. 当前状态、最新版本与阅读说明

- 当前实验状态：`Passed / 2D preregistered numerical problem only / Change 06 completed`。
- 当前文档版本：`exp031-v9`。
- 当前 change 编号：`Change 06`。
- 最新正式 run：`runs/exp031_finite_B_illumination_spot_20260830_211803/`，run state completed，artifact audit passed，scientific status `passed_2D_preregistered_numerical_problem`。
- 最新失败或诊断 run：最近科学负结果为Change 05 formal `runs/exp031_finite_B_illumination_spot_20260830_204113/`；最近preflight为`runs/exp031_finite_B_illumination_spot_preflight_20260830_211717/`并通过。
- 最新测试状态：exp031定向`18 passed in 1.06 s`；全量`340 passed, 12 failed in 180.91 s`，12项仍均为用户已有exp040 frozen-config hash mismatch，未新增失败类型。
- 最新 Ruff 状态：全部本次Python文件`All checks passed`。
- 当前最可信结论：全部预注册primary gates已通过：50 um `1e-7` formal residual edge为`0.152039%`，64到128 um padding FI变化`0.01526%`，FOV/support最大变化`0.13595%`，finite-B fully covered且zero wrap。secondary known-B一轮重建完成，100/400 um/plane的recovered sensitivity排序与truth一致，但aligned probe error仍为`37.71%/10.50%/26.74%`，只支持排序保持，不支持高精度恢复。
- 当前仍未解决的问题：`1e-8` control edge为`0.200231%`，略高于0.2%且未呈单调下降；它不是第23节formal hard gate，但提示single-realization edge-ring metric有cell-pattern波动。真实beam/B/detector标定、3D迁移和多轮reconstruction收敛均未解决。
- 当前 blocker：无exp031预注册二维primary blocker；Passed不解除真实3D、真实噪声、标定与production reconstruction限制。
- 下一步动作：exp031可在v9封口。若要提高edge证据强度或重建精度，必须分别开新的exp031 Change/独立reconstruction实验；B family转exp041，共享接口迁移另开exp04x回归。
- 阅读顺序：1--12为冻结预注册；13--23是Change 01--06预注册与历史结果；第24节是最新正式结果并修正第22节的formal residual-edge失败状态。推荐阅读0、1--12、20--24；当前科学结论以第24节为准，第22节的padding平台证据继续有效，第20/22节的失败路线保留为历史记录。

| 版本 | 日期时间 | 对应章节 | 改动主题 | 状态 | 关键结论 |
|---|---|---|---|---|---|
| exp031-v0 | 2026-08-25 12:06 +08:00 | 1--12 | 初始预注册 | Frozen | 固定 finite nonperiodic B、Gaussian reference-plus-residual、双归一化和 gates |
| exp031-v1 | 2026-08-25 12:06 +08:00 | 13 | Change 01 预注册 | Superseded by result | 只实现预注册公共能力、测试、preflight、正式 run 与 gated secondary reconstruction |
| exp031-v2 | 2026-08-25 12:53 +08:00 | 14--15 | Change 01 结果；Change 02 预注册 | In progress | 共享实现和 preflight 通过；单 worker 修复在修改前登记 |
| exp031-v3 | 2026-08-25 12:59 +08:00 | 16--17 | Change 02 结果；Change 03 预注册 | In progress | 单worker未修复；预注册复用已计算coverage/plan以消除后置峰值 |
| exp031-v4 | 2026-08-25 13:09 +08:00 | 18--19 | Change 03 结果；Change 04 预注册 | In progress | gates已计算为False；预注册fresh只读绘图子进程完成artifact闭环 |
| exp031-v5 | 2026-08-25 13:40 +08:00 | 20 | Change 04 结果 | Complete negative | artifact闭环通过；residual-edge与padding gates失败，secondary按规则跳过 |
| exp031-v6 | 2026-08-30 20:35 +08:00 | 21 | Change 05 预注册 | In progress | 固定50 um residual support与32/64/128 um padding收敛方案；尚无结果 |
| exp031-v7 | 2026-08-30 20:58 +08:00 | 22 | Change 05 结果 | Complete negative | padding已收敛；50 um `1e-6` residual edge仍为0.363%，primary保持False |
| exp031-v8 | 2026-08-30 21:13 +08:00 | 23 | Change 06 预注册 | In progress | 固定50 um `1e-7/1e-8` support与88 um envelope comparison ROI；尚无结果 |
| exp031-v9 | 2026-08-30 21:34 +08:00 | 24 | Change 06 结果 | Passed 2D only | primary全过、secondary完成并保序；重建误差仍大且control edge非单调 |

## 1. 研究问题

本实验分别回答：第一，同一有限、非周期 physical sample-B canvas 能否在 exp030 的所有 `(x,y)` integer scan positions 为不同 Gaussian active windows 提供完整 patch，且 extraction/scatter 严格配对、无 wrap、无 silent crop；第二，在 2D projected-phase、轴对称、居中、zero-tilt 条件下，A 面 50/100/200/400 um 的 1/e2 intensity diameter 如何分别改变 TGV 区域照明、B-plane probe、detector waist sensitivity、每入射光子 Poisson Fisher information、动态范围和 matched known-B probe recovery。

两个归因不能混合：finite/aperiodic B 是边界模型变化；spot diameter 是 illumination-envelope 变化；fixed-total-power 另含中心照度下降；formal/strict FOV 与 open padding 只量化数值误差。

### 本节判断

问题是 Phase 3 projected-model 的实验可实施性与尺度敏感性，不是多孔问题，也不构成真实 3D 可测性证明。

### 下一步建议

继续当前 exp031；先建立共享 scan-window contract，再运行 illumination matrix。

## 2. 实验边界

- 只使用单个、轴对称、居中、无倾斜 TGV 的 2D projected-phase working model。
- Gaussian waist 位于 effective-A plane，中心与 TGV 重合；无额外 curvature、astigmatism、plate-edge clipping 或像差。
- 光斑内和 TGV 周围均为连续玻璃；玻璃 reference background 保留。远处玻璃只可在 incident-energy/FOV convergence 通过后忽略。
- B 是单一有限非周期 phase-cell realization；不重抽、不优化、不周期铺砖。B-family、频谱和制造筛选属于 exp041。
- 主结果是 forward observability、FI、detector diagnostics、coverage 和 numerical validity。known-B/probe-only 是 gated secondary endpoint；不更新 B、不做 blind recovery。
- noise baseline 为 none。FI 是 independent Poisson、shot-noise-only optimistic bound，不含 read/dark/gain/full-well/saturation/stage/model mismatch。

### 本节判断

任何 Passed 只表示上述二维数值问题通过，不能迁移为 scalar multislice、Maxwell 或真实 detector 精度结论。

### 下一步建议

继续当前 exp031；beam decenter/tilt/waist-position/aberration 若需研究必须新开独立实验。

## 3. 依赖和已有证据

- exp030 权威 run：`runs/exp030_TGV_2d_effective_phase_20260810_121124/`；继承几何、532 nm、z_AB/z_BC=1 mm、49-frame scan、0.25 um sampling/detector pixel、2 um B feature、0.8 rad phase range、0.48828125 nm waist finite-difference step与 no-noise baseline。
- exp030 已给出解析 plane-wave reference + compact `(T-1)` Fresnel--Hankel A→B 路径；旧 hard-truncated full-field 传播不是可接受 reference。
- exp040 R5/R6 理论说明 finite support、transparent exterior、reference-plus-residual open padding和 support sensitivity 的证据边界。
- exp041 只提供 B-design 讨论，不授权 exp031 选择新 family 或结果后重抽 B。
- exp042 说明 measurement operator、adjoint、truth-fixed-point 和 raw/simulation-only evaluation 必须分离；其 3D development 结果不迁移成 exp031 科学结论。
- 数据结构遵循 `docs/theory_notes/data_format.md` 的 `/entry` 并列布局，不建立项目级 compression/chunking 标准。

### 本节判断

现有证据足以冻结 exp031 的 2D forward 与边界语义，但没有 exp031 spot/FI/coverage 结果。

### 下一步建议

继续当前 exp031；未来把新 scan-window API 迁移到 exp04x 必须另开回归任务。

## 4. 物理模型

光斑统一定义

$$A(r)=A_0\exp(-r^2/w^2),\qquad I(r)=|A_0|^2\exp(-2r^2/w^2),$$

配置名 `spot_diameter_1e2_intensity_m=2w`。fixed-center 取 `A0=1`；fixed-total-power 取

$$|A_0|^2=\frac{2P_0}{\pi w^2},\qquad P_0=1.$$

A→B 写成

$$U_A=G_A+G_A(T_{TGV}-1),$$

其中 analytic Gaussian reference 用 paraxial complex-q 传播，compact radial perturbation 用与 exp030 同一 Fresnel--Hankel convention。plane wave 以 `G_A=1` 为渐近 reference。

B→detector 对每个 physical patch `B_s=1+M_s` 写成

$$P_BB_s=G_B+[\delta P_B+P_BM_s].$$

`G_B` 单独解析传播；方括号内 residual 居中零填充到 open grid 后用 band-limited、alias-controlled ASM 传播，再裁到固定 96 um detector ROI。不得把 nonzero Gaussian reference 截断成矩形 aperture。

### 本节判断

该分解保留 TGV 周围玻璃产生的 coherent reference，并把 active physical window 与 numerical padding 分开。

### 下一步建议

继续当前 exp031；若 paraxial Gaussian reference control 或 compact perturbation convergence 不通过，追加 exp031 新 Change，不静默换模型。

## 5. 参数与对照组

| 变量 | 预注册值 | 语义 |
|---|---:|---|
| spot diameter | 50, 100, 200, 400 um | A plane 1/e2 intensity diameter |
| plane wave | amplitude 1 | exp030 legacy/asymptotic、只进入 fixed-center |
| formal omitted incident power | <=1e-3 | square active-window analytic target |
| strict controls | 50 and 400 um, <=1e-4 | full-scan FOV controls |
| B phase range / cell | 0.8 rad / 2 um | 与 exp030 相同 |
| B/scan seeds | 20260731 / 20260732 | 所有 cases 共用 |
| B guard | 1 physical feature cell per side | 加在 continuous geometry lower bound 后再 cell/parity 对齐 |
| detector ROI | 384x384 at 0.25 um | 固定 96 um，不随 spot 变化 |
| finite difference | 0.48828125 nm | matched minus/baseline/plus |
| open padding guard | 16 um per side | numerical only；100 um 另做 32 um control |
| reconstruction | 1 fixed epoch | 100 um、400 um、plane；仅 primary gates 通过后 |

periodic negative control 使用同一中心 phase-cell realization，只在 100 um fixed-center 上将 central legacy tile 周期延拓；正式模型保持 enlarged aperiodic fully covered。

### 本节判断

矩阵同时隔离 envelope、power normalization、B boundary 和 numerical FOV/padding；plane wave 不进入 fixed-total 定量排序。

### 下一步建议

继续当前 exp031；任何新增 spot、seed、B family 或 reconstruction schedule 都必须作为 exp031 新 Change 或转交 exp041。

## 6. 数值方法

Gaussian square active window 的 half-width `a` 由

$$\operatorname{erf}(\sqrt{2}a/w)^2\ge 1-\epsilon$$

解析求得，再向上对齐到整数 pixel、2 um cell 和 centered parity。plane-wave active window 固定为 exp030 的 96 um。master B physical extent 根据最大 strict active window、实际 scan min/max、一个 cell guard及同样 alignment 自动计算。

公共 `ScanWindowPlan` 接收 large/window `(ny,nx)`、positions `(x,y)` m 和 scalar/`(dy,dx)`；只允许 integer shifts。它预计算 slices、left/right/top/bottom margins、coverage/visited/unvisited maps，并提供 extraction 与 scatter-add。odd/even shape 的 half-pixel center offset必须显式报告。

runner 逐 illumination/FOV case、逐 frame处理 open arrays；只保留 detector ROI 和需要持久化的 case-level fields，禁止同时常驻全部 spot x waist x scan x open-grid intensities。传播使用 complex64 工作数组以约束内存，metrics/reductions 使用 float64；该 precision 选择进入 metadata。

### 本节判断

0.25 um sampling使最大 spot 成为高内存 case，必须先做 preflight；若超过配置的内存/runtime budget，正式 run 前追加 Change 调整，而不是临时降采样。

### 下一步建议

继续当前 exp031，先生成 preflight run并审计实际 shape、峰值内存、runtime与 HDF5 估计。

## 7. 指标与 denominator

- B coverage：physical active size、scan min/max/span、continuous required size `L_window + span`、guarded/aligned actual size、每侧 margin、minimum margin、fully-inside、wrap count、coverage/visited/unvisited、cell determinism、energy-weighted coded fraction。margin 以 pixel 与 m 同时保存。
- illumination：center intensity；`D_waist/2` 与 `D_top/2` 处相对强度；top aperture power/total；active captured/total；edge-ring energy/total；B-plane second-moment diameter；在共同 96 um ROI 对 plane-wave normalized envelope 的 relative L2。
- probe sensitivity：`||dP/dD||_2`；去 gauge 后 `D0||dP/dD||/||P||`；每入射光子场幅只作为明确标注的 normalization，不替代 absolute metric。
- detector sensitivity：每 frame `||dI/dD||_2`、stack absolute norm、`D0||dI/dD||/||I||`、min/median/max；fixed-total ratio to 100 um；Gaussian/plane ratio只在 fixed-center定义。
- Poisson：每 frame先定义一入射光子 pixel mean `mu=I*pixel_area/P_incident`，`dmu=dI*pixel_area/P_incident`；总入射光子在 frames 间等分，因此 `FI_per_incident_photon=mean_frame sum_pixel(dmu^2/max(mu,epsilon))`。`CRLB_per_sqrt_incident_photon=1/sqrt(FI_per_incident_photon)`，单位 m sqrt(photon)。
- detector diagnostics：ROI captured energy/incident energy、max/median、p99.9/p1、center/tail、低于 `1e-4*max` 的 pixel fraction、TGV-vs-glass-background differential L2。
- relative change denominator统一为 comparison/reference 的绝对值，零值用 config epsilon保护。small/moderate/large分别为 `<5%`、`[5%,20%)`、`>=20%`。

### 本节判断

absolute、normalized、per-photon 三类量不会互相替代；fixed-center 的总功率变化必须单独报告。

### 下一步建议

继续当前 exp031；没有 photon budget/full well 前只报告条件性指标，不下饱和或实测精度结论。

## 8. Hard gates

1. 所有正式 B patches fully inside。
2. minimum physical coverage margin `>0`。
3. periodic-wrap count `=0`。
4. extraction/scatter dot-test relative error `<=1e-12`。
5. 同 seed/config phase cells、scan plan和小型重复 forward一致。
6. 50/400 um formal vs strict 的 normalized detector sensitivity 与 FI relative change均 `<5%`。
7. source residual edge-ring energy fraction `<=2e-3`，100 um padding control主要 sensitivity/FI变化 `<5%`。
8. 所有数组和 metrics finite，无 NaN/Inf；intensity nonnegative。
9. HDF5、metrics.json、config.yaml、metadata.json语义一致。

5%只表示 numerical convergence，不是工程检测阈值。若任一 primary gate失败，实验记录为 Failed/Inconclusive、保留诊断 run，并跳过 secondary reconstruction。

### 本节判断

这些 gate足以阻止 wrap、截断或 padding artifact被误称为 spot sensitivity。

### 下一步建议

继续当前 exp031；失败时按单一原因追加 Change，不放宽 gate制造通过。

## 9. HDF5 和 figures 计划

HDF5 顶层保持 `/entry/config_yaml,data,instrument,sample,truth,metadata,metrics`；只有实际执行 secondary 时增加 `/entry/reconstruction`。必需字段包括用户指定的 illumination、sample-B cells、spot diameters、primary `P_B_true`、coverage/illumination/probe/detector/FI/numerical metrics。case stacks以 fixed-center canonical arrays加branch scale/provenance保存，避免复制可由同一 field严格缩放的 fixed-total arrays；master B优先保存 phase-cell map、origin、extent与 rasterization metadata，不保存冗余多分辨率 full canvas。

预注册 figures：`B_master_and_corner_patches.png`、`B_scan_coverage_map.png`、`illumination_profiles_with_TGV_overlay.png`、`spot_power_and_uniformity.png`、`probe_sensitivity_vs_spot_size.png`、`detector_sensitivity_vs_spot_size.png`、`poisson_information_vs_spot_size.png`、`detector_dynamic_range_vs_spot_size.png`、`fov_convergence.png`；若执行 secondary再生成 `known_B_recovery_vs_spot_size.png`。

### 本节判断

PNG只用于人工审阅；全部曲线数据和标量进入 HDF5/JSON。

### 下一步建议

继续当前 exp031；不修改项目级 compression/chunking规范。

## 10. 初始实现计划

1. 新建 `forward/scan_windows.py` 和相应 dot/coverage tests。
2. 小范围兼容扩展 `objects/sample_b.py`，增加 physical cell generation/rasterization，不改变旧 API默认行为。
3. 新建 `forward/exp031.py` 统一 Gaussian reference、radial compact perturbation、open residual forward和 streaming case计算。
4. 新建 `inverse/exp031.py` 放 spot/FI/detector metrics与 gated known-B control。
5. runner只负责编排、timestamped run、HDF5/JSON/figures和审计。
6. 先 targeted tests与Ruff，再 preflight-only diagnostic run；通过预算后正式 run；最后全量 pytest确认不新增既有失败。

### 本节判断

共享能力集中在 src，不复制 exp040 private R5/R6 helper，也不改冻结 exp040/exp042文件。

### 下一步建议

继续当前 exp031；未来 exp04x migration point仅记录接口，不在本任务切换历史 pipeline。

## 11. 初始风险

- 400 um spot在0.25 um sampling下需要数千像素的 active/open grid，FFT峰值内存和runtime可能成为 blocker。
- formal incident-energy containment不自动保证 residual或 detector ROI收敛，必须分别看 edge/padding/FOV controls。
- same master B在strict扩窗后新增真实编码区域，formal/strict差异同时包含遗漏Gaussian tail而非随机重抽；这是预期的物理FOV control。
- plane wave没有有限 total power，不能与fixed-total Gaussian作FI或absolute power直接排名。
- 一轮 secondary known-B control只回答固定schedule下的排序保留，不证明optimizer收敛或production reconstruction能力。

### 本节判断

最大风险是计算规模，而不是公式尚未定义；preflight必须先于正式大矩阵运行。

### 下一步建议

继续当前 exp031；若预算不满足，在 exp031 新 Change中只调整明确的数值实现/预算变量，不改物理矩阵。

## 12. 初始下一步与状态规则

只有文档、config、公共实现、runner、tests、timestamped run、HDF5/JSON/figures审计和结果追加记录齐全，才能把 exp031标为已实现。Passed仍只代表二维预注册问题。失败也必须保留run、诊断指标和append-only结果章节。

### 本节判断

`exp031-v0`至此冻结；第0节以外不再改写本初始主体。

### 下一步建议

继续当前 exp031，执行Change 01。

## 13. Change 01 预注册：共享有限 B、Gaussian/open forward、指标与完整实验闭环

### 触发原因

exp030仍使用同shape periodic B；现有公共模块没有large-canvas patch plan、paired scatter、physical cells、Gaussian analytic reference或exp031指标闭环。

### 当前证据

强制文档已完整读取。修改前Git为混合工作区，staged为空；已有exp030/exp040/shared文件修改、deleted notebook与untracked内容全部保留。修改前全量pytest为`322 passed, 12 failed`，失败均是exp040 frozen config hash mismatch。

### 当前问题或失败

尚无exp031实现、config、run或artifact；152 um只是问题陈述中的建议数量级，不是代码结果。

### 为什么需要改变

没有共享finite-window与reference-plus-residual能力就无法排除periodic wrap和hard aperture，也无法对spot-size影响作独立归因。

### 本次只允许改变的变量

只新增exp031文件和小范围兼容扩展`sample_b.py`；按第5--8节冻结的case、normalization、FOV、padding、metrics和gates实现。若preflight超预算，本Change结果只记录诊断，不在同Change静默改config。

### 保持不变的变量

TGV几何、wavelength、z_AB/z_BC、scan positions、detector ROI/pixel、B phase range/cell/seed、finite-difference step、no-noise、single centered axisymmetric projected model均不变；不改exp040/exp042历史文件、run或结论。

### 计划修改的文件

创建本任务列出的doc/theory/config/runner/scan-window/tests，并新增`src/tgv_ptycho/forward/exp031.py`、`src/tgv_ptycho/inverse/exp031.py`；只对`objects/sample_b.py`做向后兼容扩展。

### 预期机制

larger spot使TGV aperture内illumination更接近plane wave；fixed-total同时降低中心照度。coherent glass reference可能提高中心背景与dynamic range并贡献shot noise。aperiodic B消除wrap但会相对legacy periodic model改变detector编码。

### 评价指标与 denominator

完全沿用第7节，不在看到结果后更换denominator。branch amplitude严格按解析power缩放，canonical fixed-center arrays只计算一次。

### numerical gate

完全沿用第8节；scan-window dot test独立要求`<=1e-12`。

### scientific reporting rule

按illumination uniformity、normalized detector sensitivity、absolute signal、per-photon FI、dynamic range和known-B error分别给small/moderate/large标签；分别归因envelope、fixed-power irradiance、B boundary和FOV/padding，禁止总平均。

### 可能失败的方式

preflight memory/runtime超限；Gaussian analytic/compact identity测试失败；strict FOV或padding变化>=5%；residual edge过大；B margin非正；FFT nonfinite；secondary过慢或不保序。

### 本节判断

Change 01已在代码修改前登记，允许开始实现；尚无结果。

### 下一步建议

继续当前 exp031；先实现并运行targeted tests，再创建独立preflight diagnostic run。

## 14. Change 01 结果：共享实现完成，正式运行暴露 Windows native worker/cache blocker

### 实际修改

创建 finite scan-window plan、paired scatter-add、automatic B-size planner、physical phase-cell map/rasterization、analytic Gaussian reference、compact radial perturbation、open reference-plus-residual propagation、spot/FI/dynamic-range metrics、matched known-B secondary、runner、config和测试。`sample_b.py`只增加新API，旧`make_random_phase_object()`默认行为保留。实现过程中将scan符号对齐到历史`shift_field_integer_pixels`：正样品位移对应从未移动master canvas的负向源坐标取patch。

### 与预注册是否一致

物理case、TGV、scan、sampling、B realization、normalization、denominator和gate均与Change 01一致。为避免Windows BLAS硬崩，complex Hankel contraction和所有新norm/inner-product使用显式real/imag或sum/sqrt；这是数值等价实现。共同B-plane envelope比较域在首次shape失败后显式取所有formal case的共同`352x352`区域，不改变固定`384x384` detector ROI。

### 实际运行命令

```powershell
D:\anaconda3\envs\tgv_ptycho_sim\python.exe -m pytest -q tests/test_scan_windows.py tests/test_sample_b_physical_cells.py tests/test_exp031_finite_B_illumination_spot.py
D:\anaconda3\envs\tgv_ptycho_sim\python.exe -m ruff check scripts/run_exp031_finite_B_illumination_spot.py src/tgv_ptycho/recon/exp031.py src/tgv_ptycho/forward/exp031.py src/tgv_ptycho/forward/scan_windows.py src/tgv_ptycho/inverse/exp031.py src/tgv_ptycho/objects/sample_b.py tests/test_scan_windows.py tests/test_sample_b_physical_cells.py tests/test_exp031_finite_B_illumination_spot.py
D:\anaconda3\envs\tgv_ptycho_sim\python.exe scripts/run_exp031_finite_B_illumination_spot.py --config configs/experiments/exp031_finite_B_illumination_spot.yaml --preflight-only
D:\anaconda3\envs\tgv_ptycho_sim\python.exe scripts/run_exp031_finite_B_illumination_spot.py --config configs/experiments/exp031_finite_B_illumination_spot.yaml
```

### tests 和 Ruff 结果

定向测试最终为`15 passed in 0.37 s`；修改范围Ruff为`All checks passed`。Change 01开始前全量基线`322 passed, 12 failed`，12项全是既有exp040 frozen-config hash mismatch，尚未在Change 01后重复全量测试。

### run 路径

- 失败preflight：`runs/exp031_finite_B_illumination_spot_preflight_20260825_123731/`，HDF5递归writer不支持dict list，已保留。
- 通过preflight：`runs/exp031_finite_B_illumination_spot_preflight_20260825_123800/`。
- 失败formal：`..._123855/`（Windows BLAS norm硬终止）、`..._124124/`（共同probe crop shape错误）、`..._124343/`（后置large adjoint额外内存）、`..._124712/`与`..._125026/`（全部forward完成后native worker/cache阶段硬终止）。

### HDF5/JSON/figures 审计

通过preflight具有config、metadata、metrics和preflight HDF5。失败formal的HDF5达到约323 MB并含全部formal detector stacks，但run state未完成、metrics/figures未封口，因此明确不作为正式科学结果，不覆盖也不删除。

### 关键指标

preflight确认scan span x/y均49 um；96 um active window连续无guard下限145 um；双侧2 um guard连续下限149 um；cell/parity对齐推荐152 um。strict 400 um控制决定master B为`3472x3472`、868 um；最大open grid`3388x3388`；估算峰值1.412 GiB、runtime 1984.6 s、HDF5 0.551 GiB，均通过4 GB/7200 s预算。

### 与上一版本的对比

相对v0/v1，152 um从问题陈述数量级变成planner结果；共享代码与preflight已完成，但“可以正式运行并审计”被Windows native终止反对。

### 结果支持或反对什么机制

结果支持finite-B geometry与streaming矩阵在注册内存预算内；多次run均完成最大forward，反对“单个open array超过4 GB”解释。终止总发生在多线程大FFT完成后的后处理，结合单独adjoint test`7.91e-16`正常，当前最强诊断是native worker/cache生命周期，而非物理模型失败。

### 是否出现新问题

出现Windows多worker native终止；Python异常handler无法捕获。尚不能用未封口HDF5报告spot科学结论。

### 已知限制

preflight runtime是规划估计而非实测；失败HDF5未通过语义和figure审计；secondary未执行。

### 是否改变实验状态

exp031保持`In progress / blocked pending Change 02`，不能标记Implemented或Passed。

### 本节判断

Change 01完成共享实现和preflight，但没有完成正式artifact闭环；失败路线必须保留，不能从partial HDF5提取选择性结论。

### 下一步建议

属于`exp031 内的新 change`：只限制FFT worker为1，重新preflight并运行；若仍native终止，再预注册独立的process-per-case或显式cache隔离，不改变科学矩阵。

## 15. Change 02 预注册：单 FFT worker 的 Windows 内存生命周期控制

### 触发原因

Change 01的多个formal run已完成全部forward但在大FFT后的metrics阶段无Python traceback终止；显式BLAS归约和前移adjoint test后仍复现。

### 当前证据

`fft_workers=-1`的小grid功能测试正常；最大case本身也能完成。独立actual-scan adjoint dot test为`7.91e-16`。失败点随大FFT后续分配出现，符合多worker工作缓存持久化而非算子错误。

### 当前问题或失败

没有通过artifact audit的正式run；多worker路径使run state停在`running`且不能捕获异常。

### 为什么需要改变

必须限制native并行工作集生命周期，才能完成注册的后处理、gate和artifact审计。worker数只影响执行资源和时序，不改变离散传播算子。

### 本次只允许改变的变量

仅将`open_boundary.fft_workers`从`-1`改为`1`，并相应更新文档版本/config状态；允许增加worker provenance和实测runtime记录。

### 保持不变的变量

所有物理参数、dx、active/open shape、padding guard、B cells/seed、scan、normalization、finite-difference、metrics、denominator、gates、reconstruction schedule和HDF5字段不变。

### 计划修改的文件

`configs/experiments/exp031_finite_B_illumination_spot.yaml`与本实验记录；runner仅在必要时补充worker provenance，不改算子。

### 预期机制

单worker消除多线程FFT保留缓存叠加和native线程生命周期不确定性，以较长runtime换取可预测峰值内存和Python控制流返回。

### 评价指标

preflight预算、formal全部hard gates、artifact audit、实测总runtime；科学指标与Change 01完全相同。

### denominator

不变：relative change以comparison/reference绝对值；FI以每帧incident power归一到每入射光子；frame间取mean。

### numerical gate

全部沿用第8节；worker变化前后离散结果在可获得的small repeat中应exact或只含float32舍入，formal仍要求registered FOV/padding gate<5%。

### scientific reporting rule

worker数不得解释成物理效应；只报告为Windows数值执行兼容措施。

### 可能失败的方式

单worker实测runtime超过7200 s；native终止仍复现；primary scientific gate失败；artifact语义不一致。

### 本节判断

证据足以允许单一数值执行变量变化；尚未授权调dx、缩小case、改变padding或放宽阈值。

### 下一步建议

继续当前exp031：修改worker后创建新preflight和新formal run；若正式完成则追加Change 02结果，否则保留失败run并另行预注册。

## 16. Change 02 结果：单 worker 未消除 post-forward native termination

### 实际修改

只把`open_boundary.fft_workers`从`-1`改为`1`，同步config中的文档版本和change provenance；没有修改物理、shape、padding、metric或gate。

### 与预注册是否一致

完全一致；没有调低分辨率、缩小case或放宽阈值。

### 实际运行命令

```powershell
D:\anaconda3\envs\tgv_ptycho_sim\python.exe scripts/run_exp031_finite_B_illumination_spot.py --config configs/experiments/exp031_finite_B_illumination_spot.yaml --preflight-only
D:\anaconda3\envs\tgv_ptycho_sim\python.exe scripts/run_exp031_finite_B_illumination_spot.py --config configs/experiments/exp031_finite_B_illumination_spot.yaml
```

### tests 和 Ruff 结果

本Change只改YAML worker值；Change 01的`15 passed`和修改范围Ruff green仍是最近验证，正式修复后必须重跑。

### run 路径

通过preflight：`runs/exp031_finite_B_illumination_spot_preflight_20260825_125608/`。失败formal：`runs/exp031_finite_B_illumination_spot_20260825_125610/`。

### HDF5/JSON/figures 审计

formal再次写完约323 MB forward HDF5并打印`periodic-B negative control completed`和`convergence ratios completed`，随后无Python traceback终止；未完成metrics/figures/audit，不能作为正式结果。

### 关键指标

preflight geometry/resource数值与Change 01一致。单worker没有把post-forward路径带回Python控制流，因此没有新的可审计科学指标。

### 与上一版本的对比

失败位置比早期run更精确：所有forward、periodic control和convergence ratio均完成；worker从all变1没有改变终止行为。

### 结果支持或反对什么机制

直接反对“多worker本身是必要原因”。当前更支持大FFT后任意额外full-master分配触发系统终止；runner在gates后还会重新创建coverage plan及int32/int16 map，尽管formal 400 case已计算相同coverage。

### 是否出现新问题

发现runner后处理有可避免的重复large-plan分配和reconstruction/plot plan重建。

### 已知限制

没有OS级峰值working-set trace，内存机制仍是诊断假设；只能通过预注册的allocation-removal对照检验。

### 是否改变实验状态

Change 02为negative，exp031仍In progress。

### 本节判断

单worker不是充分修复，不能把Change 02标记Passed；应保留worker=1以减少并发变量，同时测试计划复用机制。

### 下一步建议

属于`exp031 内的新 change`：复用case内已有coverage和slice plan，避免forward后再分配full-master coverage；若仍失败则新开Change考虑process-per-case checkpoint。

## 17. Change 03 预注册：复用 streaming case 的 coverage 与 slice plan

### 触发原因

Change 02在convergence ratios完成后终止；代码审计显示后续重新调用`make_scan_window_plan(master_shape, ...)`会分配`3472x3472` int32 coverage，再转int16，且plot/reconstruction还会重复plan。

### 当前证据

每个formal `_simulate_case`内部已经成功生成并使用实际plan；400 um formal plan正是最终coverage figure/HDF5所需数据。单独96/152 um adjoint test已在大FFT前通过`7.91e-16`。

### 当前问题或失败

后处理重复分配约48 MB coverage加约24 MB cast，并可能叠加native FFT保留内存；这是无科学必要的峰值。

### 为什么需要改变

复用已计算plan可保持bitwise相同coverage/slices，同时消除后置分配；也更符合逐case streaming原则。

### 本次只允许改变的变量

只改变对象生命周期和plan复用：400 formal时保存int16 coverage；代表case只保存不含full coverage的slice metadata；后处理、plot和secondary不得重新构建full-master coverage。保留`fft_workers=1`。

### 保持不变的变量

全部物理、case、shape、B/scan seed、propagation、normalization、metrics、denominator、gate和reconstruction schedule不变。

### 计划修改的文件

只修改runner和config provenance；测试按需增加plan-reuse/artifact覆盖。

### 预期机制

将必要的24 MB int16 coverage在400 formal结果仍存活时生成，随后释放48 MB int32 plan；后处理不再触发新的large allocation。代表case patch getter复用轻量slice plan。

### 评价指标

formal是否通过原终止点、artifact audit、所有既定scientific/numerical指标；结果必须与同算子partial forward语义一致。

### denominator

不变。

### numerical gate

不变；coverage map必须与formal plan一致，all-inside/margin/wrap/adjoint阈值不变。

### scientific reporting rule

内存生命周期修复不作为光斑或B物理结论。

### 可能失败的方式

native cache本身仍耗尽内存；secondary重新分配large open buffers失败；HDF5 metrics重写产生新峰值；primary gate科学上失败。

### 本节判断

Change 03只去除冗余分配且不改变数值算子，适合作为下一单因子诊断。

### 下一步建议

继续当前exp031：实现plan复用并创建新formal run；若primary gate失败则按结果记录且不强行执行secondary。

## 18. Change 03 结果：plan 复用通过 gates，绘图触发剩余 native blocker

### 实际修改

在400 um formal case存留int16 coverage；代表case保留去除full map的slice plan；post-forward不再创建large coverage plan；绘图前显式释放master phase raster、radial cache和control/common-probe state，并把coverage PNG输入降采样。

### 与预注册是否一致

一致；只改变对象生命周期和可视化采样，不改变HDF5 full coverage、forward或metric。

### 实际运行命令

```powershell
D:\anaconda3\envs\tgv_ptycho_sim\python.exe scripts/run_exp031_finite_B_illumination_spot.py --config configs/experiments/exp031_finite_B_illumination_spot.yaml --preflight-only
D:\anaconda3\envs\tgv_ptycho_sim\python.exe scripts/run_exp031_finite_B_illumination_spot.py --config configs/experiments/exp031_finite_B_illumination_spot.yaml
```

### tests 和 Ruff 结果

runner Ruff通过；完整tests等待最终实现后统一运行。

### run 路径

通过preflight：`runs/exp031_finite_B_illumination_spot_preflight_20260825_130155/`。失败formal：`runs/exp031_finite_B_illumination_spot_20260825_130156/`和`runs/exp031_finite_B_illumination_spot_20260825_130559/`。

### HDF5/JSON/figures 审计

forward HDF5完成；runner打印`primary numerical gates passed=False`和`released forward-only working state`，随后在首次figure完成前native退出。正式metrics/HDF5 audit仍未封口。

### 关键指标

首次得到明确gate aggregate：primary为False，因此按预注册正确跳过secondary reconstruction。具体失败gate必须在完成metrics落盘后读取，不能从stdout猜测。

### 与上一版本的对比

相对Change 02，已越过原终止位置并完成hard-gate计算；反复full-plan allocation确为一个峰值触发因素，但不是最终artifact blocker。

### 结果支持或反对什么机制

支持plan复用降低后处理峰值；剩余终止严格位于Matplotlib阶段，支持将绘图从科学进程隔离。

### 是否出现新问题

Matplotlib在经历最大FFT的同一Windows进程中不可靠，即使已释放Python数组。

### 已知限制

primary失败的具体gate未落盘；Change 03 run仍不能作为正式科学结果。

### 是否改变实验状态

实验仍In progress，但secondary应不会在当前数值结果上执行。

### 本节判断

科学计算和gate已经完成；剩余工作是可靠保存和可视化，不允许借artifact修复改变False gate。

### 下一步建议

属于`exp031 内的新 change`：将figure生成隔离到fresh只读子进程，父进程先保存并关闭HDF5，再完成audit。

## 19. Change 04 预注册：数值封口后使用 fresh 只读绘图子进程

### 触发原因

Change 03只在首次Matplotlib渲染时native终止；同一进程此前已完成全部science与gates。

### 当前证据

stdout顺序为periodic control完成、convergence完成、primary=False、forward-only state释放，之后无Python traceback退出。

### 当前问题或失败

figure生成阻止metrics/HDF5 audit闭环，并使run state停在running。

### 为什么需要改变

绘图是只读artifact步骤，天然可在fresh process中从已封口HDF5/JSON重建；隔离可避免继承FFT/native allocator状态。

### 本次只允许改变的变量

只改变runner artifact顺序与process boundary：父进程先写metrics和full HDF5、关闭文件；子进程只读config/metrics/HDF5生成预注册PNG；父进程检查返回码和文件后audit。PNG coverage可从HDF5按stride读取，定量full map保持不变。

### 保持不变的变量

全部science、worker=1、case、shape、B、scan、forward、metrics、denominator、gates和secondary gating不变。

### 计划修改的文件

runner、config provenance和相应runner/artifact test。

### 预期机制

fresh Python/Matplotlib进程没有大FFT历史工作集；父进程不在HDF5打开状态绘图，artifact失败可作为普通subprocess返回码捕获。

### 评价指标

九张必需PNG存在且可读；HDF5/JSON/config/metadata语义audit通过；run state正常completed；primary gate结果保持False时不得出现reconstruction group。

### denominator

不变。

### numerical gate

不变；figure只可视化已保存定量数据，不参与gate或选择。

### scientific reporting rule

若primary为False，正式状态应是completed-but-failed/inconclusive二维数值实验，而非实现失败；仍需回答各指标但不执行secondary。

### 可能失败的方式

父进程在保存大coverage时终止；子进程无法重建corner patches；PNG缺失；audit JSON循环不一致。

### 本节判断

process-isolated plotting是最小artifact修复，不污染预注册science。

### 下一步建议

继续当前exp031：实现并运行；完成后读取具体failed gate，追加Change 04结果，不为得到Passed而改阈值。

## 20. Change 04 结果：artifact 闭环完成，二维预注册数值实验为负结果

### 实际修改

父进程在关闭HDF5前写完全部定量结果和full-resolution coverage，关闭后调用fresh只读绘图子进程。由于本机Matplotlib在tiny affine matrix路径仍触发Windows BLAS DLL硬崩，绘图子进程改用Pillow直接读取HDF5/metrics生成同名heatmap、line chart和control chart；数值结果不经过PNG。audit的tuple/list比较改为两侧canonical JSON后比较。最后只改config中的状态和文档版本，不改运行参数。

### 与预注册是否一致

science、worker、case、shape、B、scan、propagation、normalization、denominator、gate和secondary gating全部不变。绘图只读、在数值封口后执行；coverage PNG降采样而HDF5保存完整`3472x3472 int16` map。primary为False后没有执行known-B reconstruction，也没有创建空`/entry/reconstruction`。

### 实际运行命令

```powershell
D:\anaconda3\envs\tgv_ptycho_sim\python.exe -m pytest -q tests/test_scan_windows.py tests/test_sample_b_physical_cells.py tests/test_exp031_finite_B_illumination_spot.py
D:\anaconda3\envs\tgv_ptycho_sim\python.exe -m ruff check scripts/run_exp031_finite_B_illumination_spot.py src/tgv_ptycho/forward/scan_windows.py src/tgv_ptycho/forward/exp031.py src/tgv_ptycho/inverse/exp031.py src/tgv_ptycho/objects/sample_b.py src/tgv_ptycho/recon/exp031.py tests/test_scan_windows.py tests/test_sample_b_physical_cells.py tests/test_exp031_finite_B_illumination_spot.py
D:\anaconda3\envs\tgv_ptycho_sim\python.exe scripts/run_exp031_finite_B_illumination_spot.py --config configs/experiments/exp031_finite_B_illumination_spot.yaml
D:\anaconda3\envs\tgv_ptycho_sim\python.exe -m pytest -q
```

### tests 和 Ruff 结果

exp031定向为`17 passed in 1.06 s`，包括rectangular/tuple-dx/sign/parity/OOB、scan adjoint、physical-cell multi-resolution、Gaussian定义与双normalization、large-waist、reference-plus-residual、compact propagation、open propagation adjoint及formal HDF5 audit。修改范围Ruff为`All checks passed`。全量为`339 passed, 12 failed in 176.03 s`；相对修改前`322 passed, 12 failed`正好多17个通过项，12个失败仍全部是用户已有exp040 frozen-config hash mismatch，未新增失败类型。

### run 路径

最新正式run为`runs/exp031_finite_B_illumination_spot_20260825_133554/`。`run_state.json`为completed，科学状态`failed_or_inconclusive`。最近通过preflight为`..._preflight_20260825_130155/`；Change 01--04所有失败/诊断run均保留且未覆盖。

### HDF5/JSON/figures 审计

HDF5为372,350,376 bytes；`I_stack`为`(49,384,384) float32`，phase cells为`(434,434)`，coverage为`(3472,3472) int16`。全部用户要求路径存在；numeric finite、intensity nonnegative、external/HDF5 config/metadata/metrics canonical semantics、九张PNG均通过。PNG均为RGB且可读：B master/corners为`1500x360`，其余为`850x760`或`1000x700`。视觉审阅确认phase realization、scan coverage、profiles、spot metrics和FOV/padding趋势可辨。没有`/entry/reconstruction`，与secondary未执行一致。

### Geometry 与 hard gates

- scan span x/y均49 um。96 um active window无guard连续最小B为`96+49=145 um`；每侧2 um guard后连续下限149 um；8-pixel cell和center parity对齐后planner给`608 px=152 um`。
- 通用公式为`L_B,axis >= L_window,axis + (p_max-p_min)_axis + 2g_axis`，随后向上对齐physical cell lattice和center parity；open padding不进入physical B公式。
- 最大strict 400 um control决定本run master B为`3472 px=868 um`；formal 400 patch最小margin 60.5 um，strict 400最小margin 3.5 um。所有patch fully inside，formal wrap count 0。
- scan-window dot test为`7.91e-16 <= 1e-12`；seed/config repeat差为0；全部finite/nonnegative；artifact audit通过。
- formal/strict FOV最大注册变化：50 um为2.8437%，400 um为0.3934%，均小于5%。
- 失败gate 1：50 um formal full B residual edge maximum为2.7302%，大于0.2%；100/200/400 um分别为0.19917%、0.04526%、0.01634%。
- 失败gate 2：100 um将padding guard由16增加到32 um时，normalized detector sensitivity变化2.4681%，但FI变化9.1731%，最大值超过5%。

### illumination 与趋近平面波

| A-plane 1/e2 intensity diameter | top-radius relative intensity | top-radius nonuniformity | common-88-um B-plane envelope difference vs plane | 工程分类 |
|---:|---:|---:|---:|---|
| 100 um | 0.6065 | 39.35% | 28.17% | illumination与envelope均large |
| 200 um | 0.8825 | 11.75% | 7.455% | moderate |
| 400 um | 0.9692 | 3.077% | 1.883% | small |

所以spot变宽在几何照明上明确趋近plane wave；这不保证所有detector/FI指标同时趋近。formal active captured incident power均约99.9%。

### fixed-total-power 结果

以100 um为branch内reference：

| spot | normalized detector sensitivity | raw detector derivative change | detector absolute signal | FI / photon (m^-2) | FI change | max/median | 分类摘要 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 100 um | 1.6482 | reference | 0.49431 | 1.3487e9 | reference | 5.298 | reference |
| 200 um | 1.0258 (-37.76%) | -67.36% | 0.28724 (-41.89%) | 3.5369e8 | -73.78% | 3.805 (-28.18%) | sensitivity/signal/FI/dynamic-range均large |
| 400 um | 0.80896 (-50.92%) | -90.81% | 0.098248 (-80.12%) | 8.8270e7 | -93.46% | 5.458 (+3.01%) | sensitivity/signal/FI large；max/median small |

中心照度相对100 um按`1/w^2`为100%/25%/6.25%。raw derivative在fixed-center反而随spot由100到200/400增加30.58%/47.05%，而fixed-total下降67.36%/90.81%，因此固定功率的单位面积光子下降是absolute signal/raw sensitivity的主要限制。per-photon FI在两个branch几乎相同，说明其spot趋势不是纯幅度缩放，而是spatial distribution、B coding和固定detector ROI共同造成。

### fixed-center-intensity 结果

| spot | normalized detector sensitivity | absolute signal | FI / photon (m^-2) | CRLB m/sqrt(photon) | max/median |
|---:|---:|---:|---:|---:|---:|
| 100 um | 1.6482 | 1.9412e-9 | 1.3487e9 | 2.7229e-5 | 5.298 |
| 200 um | 1.0258 | 4.5119e-9 | 3.5369e8 | 5.3173e-5 | 3.805 |
| 400 um | 0.80896 | 6.1731e-9 | 8.8270e7 | 1.0644e-4 | 5.458 |
| plane legacy ROI | 0.87276 | 6.3183e-9 | 6.8358e8 | 3.8248e-5 | 3.578 |

相对100 um，200/400 um的absolute signal增加132.4%/218.0%（large），但normalized sensitivity下降37.76%/50.92%且FI下降73.78%/93.46%（均large）；dynamic max/median分别变化-28.18%（large）和+3.01%（small）。相对plane，100/200/400 normalized detector sensitivity变化+88.85%/+17.54%/-7.31%（large/moderate/moderate），而FI变化+97.30%/-48.26%/-87.09%（均large）。不同指标明显不一致，不能合并成“spot越大越好”。

### 玻璃reference、detector与Poisson解释

光斑内TGV周围连续玻璃通过analytic`G_B/G_C`保留为coherent reference，B-only glass background从未删除。TGV differential intensity相对glass+B background L2在100/200/400 um为45.59%/39.02%/34.27%；max/median为5.30/3.81/5.46，p99.9/p1为33.23/15.42/37.39。reference使TGV扰动形成线性相干交叉项，同时其强背景进入Poisson`mu`分母并贡献shot noise；本run未做“删除周围玻璃”对照，不能进一步把这些数值因果拆开。没有photon budget、exposure、gain或full-well，因此不宣称一定可测或饱和。

formal/strict FOV使50/400 um sensitivity/FI变化分别不超过2.84%/0.393%，支持“Gaussian远尾及其远处玻璃对这些FOV指标可忽略到5%以内”。但100 um padding FI仍变化9.17%，50 um residual edge也失败，因此不能把这一局部FOV结论提升为整个B-to-detector数值模型已收敛。

### periodic-B negative control

同一中心realization下，legacy periodic与enlarged aperiodic 100 um fixed-center detector stack relative L2为2.1311%，mean frame relative L2为1.9918%，按预注册规则为small但非零。legacy control发生49次periodic wrap语义；正式模型为0次。该数值只量化exp030 periodic假设变化，不是B-family优劣。

### secondary reconstruction

primary gates未通过，所以matched known-B/probe-only secondary按预注册跳过；没有truth-driven optimizer选择、没有raw reconstruction，也没有`known_B_recovery_vs_spot_size.png`。因此recovered-probe error和spot ordering目前未知，不能用forward truth ordering代替。

### 与上一版本的对比

相对Change 03，Change 04完成了HDF5/JSON/PNG/audit/run-state闭环，并把具体失败gate从stdout aggregate解析成可审计数值。它反对任何把当前exp031标为Passed的表述，但支持把实现状态标为完成、科学状态标为negative/inconclusive。

### 结果支持或反对什么机制

支持：large finite nonperiodic B可无wrap覆盖所有scan；spot越宽在TGV几何照明上趋近plane；fixed-power中心光子密度下降强烈压低absolute signal/raw sensitivity；periodic假设变化在本100 um detector stack上是small；formal Gaussian tail FOV controls通过。反对或未支持：当前16 um padding足够、50 um active residual edge足够、所有detector指标随plane-like illumination单调改善、known-B recovery保序或真实3D waist可测。

### 是否出现新问题

scientific blocker是100 um padding FI和50 um residual edge。Windows BLAS/Matplotlib兼容通过显式归约与Pillow只读绘图解决，但这些只是本机artifact路径，不是未来算法架构要求。

### 已知限制

2D projected-phase、轴对称居中、zero tilt、Gaussian waist在effective-A、paraxial analytic reference、integer shifts、ideal no-noise与Poisson-only bound；plane wave仍是96 um legacy/asymptotic ROI reference，没有有限total-power；未研究beam decenter/tilt/curvature/aberration、真实B exterior、detector calibration或3D multiple scattering。

### 是否改变实验状态

exp031代码与artifact实现完整，正式run完成；科学状态为`Failed/Inconclusive`，不是`Passed`。这不表示真实3D TGV不可测，只表示当前预注册二维数值矩阵未同时满足全部convergence gates。

### 本节判断

问题1--3得到确定几何答案；问题4--10得到可审计二维指标及明确归因，但padding/residual failures限制科学置信；问题11的共享代码可以迁移，问题12的二维科学数值不能迁移；secondary问题因正确gating保持未知。

### 下一步建议

- 属于`exp031 内的新 change`：若继续，Change 05只研究50 um residual support定义/active FOV与100 um 32/更大padding序列，保持物理矩阵和5% gate，不直接放宽阈值。
- 属于`exp041`：研究B频谱、family、制造性和信息量；不得用本run重抽B。
- 属于`exp04x`：迁移large canvas、scan plan、extract/scatter、physical cells、Gaussian/reference-plus-residual和metrics接口，另做3D回归。
- 属于`新开独立实验`：beam decenter/tilt/waist-position/aberration。
- 属于`等待真实 beam/B/detector 标定`：实际photon budget、gain、full well、read/dark noise、B exterior和plate/beam参数。

可迁移为exp04x共享代码的是finite large-B canvas、integer window plan、paired adjoint scatter、coverage/margins、physical phase cells和multi-resolution rasterization、Gaussian reference/open residual、spot/edge/FI/dynamic-range metrics。不能迁移的是本run的2D sensitivity/FI数值、spot排序、paraxial Gaussian充分性、plane-ROI比较、shot-noise CRLB、未执行的recovery结论或任何真实3D waist可测性判断。

## 21. Change 05 预注册：50 um residual support与32/64/128 um open-padding收敛

### 触发原因

Change 04完成artifact闭环，但primary numerical gates因两个明确数值问题失败：50 um formal source residual edge energy为`2.7302%`，超过`0.2%`；100 um将open guard由16增加到32 um时，FI变化`9.1731%`，超过`5%`。用户要求联系上一版继续下一版改动并形成完整实验记录。

### 当前证据

旧run中50 um的`1e-3`与`1e-4` incident-power support对应residual edge分别为`2.7302%`和`1.0575%`，说明仅使用原strict FOV仍不足以满足residual gate；100/200/400 um formal edge分别为`0.19917%/0.04526%/0.01634%`。formal/strict FOV sensitivity/FI最大变化50 um为`2.8437%`、400 um为`0.3934%`，均已通过。100 um normalized detector sensitivity的16到32 um变化为`2.4681%`，但FI为`9.1731%`，说明FI是当前padding限制量。

### 当前问题或失败

当前代码把Gaussian incident-energy FOV同时当作B-modulated residual source support；对50 um而言，随机phase B使低强度尾部仍形成非零`P_B(B_s-1)`，因此`1e-3/1e-4` incident containment并不自动保证residual-edge gate。单一16/32 um padding pair也不能判断更大guard下是否出现平台。

### 为什么需要改变

必须分别收敛physical residual support和numerical open padding，才能排除source truncation、FFT circular wrap或shape-dependent ASM bandlimit对FI的污染，并决定是否允许执行secondary known-B reconstruction。

### 本次只允许改变的变量

1. 50 um主case的physical active/residual support改为遗漏incident power不超过`1e-6`，并增加`1e-7`更严格support control；旧`1e-3`与`1e-4`结果仅作为Change 04历史证据，不回写或重解释。
2. 所有正式case的open guard由16 um改为64 um，保持同一正式数值算子；100 um另计算32 um与128 um controls，形成`32/64/128 um`序列。
3. padding主gate只比较正式64 um与更严格128 um；32 um只用于显示由旧区间向平台的趋势，不作为通过依据。
4. 增加相应case provenance、逐级relative-change metrics、preflight资源估计、测试和HDF5/JSON审计字段。

### 保持不变的变量

TGV几何、wavelength、z_AB/z_BC、dx、detector ROI/pixel、50/100/200/400 um spot矩阵、Gaussian定义、双normalization、scan positions、同一master B cells/seed、B phase range/feature、finite-difference step、no-noise、reference-plus-residual恒等式、bandlimit/alias-control、FI denominator、5% convergence gate、0.2% residual-edge gate和secondary schedule均不变。不得重抽/优化B，不改exp040/exp042历史文件或run。

### 计划修改的文件

`configs/experiments/exp031_finite_B_illumination_spot.yaml`、`scripts/run_exp031_finite_B_illumination_spot.py`、按需小范围修改`src/tgv_ptycho/forward/exp031.py`、`tests/test_exp031_finite_B_illumination_spot.py`，以及本实验记录。若共享helper无需改变则不制造无关改动。

### 预期机制

50 um support从`1e-4`继续扩大到`1e-6/1e-7`应使B-modulated Gaussian tail在source边缘显著衰减；64到128 um open guard应降低periodic image和shape-dependent transfer变化。若FI仍不收敛，则问题不是简单padding不足，需要下一Change检查ASM alias-control定义或更独立的propagation benchmark。

### 评价指标

保存50 um `1e-6`主case与`1e-7`control的active shape/size、captured power、residual-edge min/median/max、normalized detector sensitivity和FI relative change；保存100 um 32/64/128 um各级open shape、edge、normalized detector sensitivity、FI及相邻级变化。继续审计全部原geometry、illumination、probe/detector/FI/dynamic-range、periodic-B和artifact字段。

### denominator

relative change继续使用更严格/更大support或padding结果相对当前正式baseline的绝对值，即`|control-formal|/|formal|`；FI继续按每帧incident power归一到每入射光子并跨frame取mean。不得以control值或两个值平均作分母。

### numerical gate

- 50 um `1e-6`主support residual-edge maximum必须`<=2e-3`，且其normalized detector sensitivity/FI相对`1e-7`control的最大变化必须`<5%`。
- 100 um正式64 um相对128 um padding的normalized detector sensitivity/FI最大变化必须`<5%`；32 um级只报告。
- 原all-inside、positive margin、zero wrap、adjoint`<=1e-12`、repeat exact、finite/nonnegative、artifact semantics等gate全部保留。
- 不允许因结果接近阈值而放宽阈值或选择未预注册padding级。

### scientific reporting rule

本Change只决定数值收敛是否恢复，不把support/padding变化解释为新的光斑物理效应。与Change 04 spot/B指标的变化必须明确标为numerical-model correction；只有全部primary gates通过才执行secondary。若通过，新的正式run取代第20节作为最新二维数值结果来源；若失败，保留negative run并逐项报告。

### 可能失败的方式

`1e-6`仍不能把50 um residual edge降到0.2%；64到128 um FI仍变化5%以上；更大open shape超过4 GB/7200 s preflight预算；扩大support改变B margin；Windows native FFT/绘图问题复现；primary通过后secondary出现新的数值或artifact失败。

### 本节判断

Change 05已在修改config、代码或启动新run前登记。它只响应Change 04已观测的两个numerical failures，并保留全部物理归因与阈值，因此可以开始实现；当前尚不能声称收敛或执行reconstruction。

### 下一步建议

属于`继续当前 exp031`：实现上述support/padding sequence，先跑targeted tests和preflight，再创建新的timestamped formal run；若preflight超预算或任一gate仍失败，作为Change 05结果如实记录，后续再决定是否需要`exp031 内的新 change`研究ASM alias-control。

## 22. Change 05 结果：padding已收敛，50 um formal residual support仍未通过

### 实际修改

在config中将所有正式case的open guard由16 um改为64 um，并给100 um增加32和128 um controls；50 um formal/support-control的遗漏incident-power目标分别改为`1e-6/1e-7`。runner现在验证padding sequence严格递增、唯一、包含formal guard且最大值与control一致，保存每级open shape、两种branch的normalized sensitivity/FI、相邻级relative change和明确gate comparison。hard gates新增显式residual-support convergence字段。新增case-matrix测试，未修改共享传播算子、B生成、scan-window API或阈值。

### 与预注册是否一致

support、padding、gate、B、scan、detector、normalization和denominator修改均按第21节执行，没有看到结果后放宽阈值或把`1e-7` control临时提升为formal。一个非预期的派生变化是common probe-envelope比较shape由旧`352x352`（88 um）自动变为plane-wave限制的`384x384`（96 um），因为50 um formal active window已从352扩至504；该metric不参与hard gate，但其Change 04/05数值不能直接作物理对比，必须在后续Change显式冻结comparison ROI。

### 实际运行命令

```powershell
D:\anaconda3\envs\tgv_ptycho_sim\python.exe -m pytest -q tests/test_scan_windows.py tests/test_sample_b_physical_cells.py tests/test_exp031_finite_B_illumination_spot.py
D:\anaconda3\envs\tgv_ptycho_sim\python.exe -m ruff check scripts/run_exp031_finite_B_illumination_spot.py src/tgv_ptycho/forward/scan_windows.py src/tgv_ptycho/forward/exp031.py src/tgv_ptycho/inverse/exp031.py src/tgv_ptycho/objects/sample_b.py src/tgv_ptycho/recon/exp031.py tests/test_scan_windows.py tests/test_sample_b_physical_cells.py tests/test_exp031_finite_B_illumination_spot.py
D:\anaconda3\envs\tgv_ptycho_sim\python.exe scripts/run_exp031_finite_B_illumination_spot.py --config configs/experiments/exp031_finite_B_illumination_spot.yaml --preflight-only
D:\anaconda3\envs\tgv_ptycho_sim\python.exe scripts/run_exp031_finite_B_illumination_spot.py --config configs/experiments/exp031_finite_B_illumination_spot.yaml
D:\anaconda3\envs\tgv_ptycho_sim\python.exe -m pytest -q
```

### tests 和 Ruff 结果

exp031定向为`18 passed in 5.68 s`，比Change 04增加一个support/padding case-matrix测试。修改范围Ruff为`All checks passed`。全量为`340 passed, 12 failed in 196.24 s`；12项全部仍是用户已有exp040 R10--R14b frozen-config SHA256 mismatch，与Change 04相同类型，本Change未修改这些文件且未新增失败。

### run 路径

通过preflight：`runs/exp031_finite_B_illumination_spot_preflight_20260830_204042/`。新formal：`runs/exp031_finite_B_illumination_spot_20260830_204113/`。formal `run_state.json`为completed，scientific status为`failed_or_inconclusive`；两个目录均为新timestamp且未覆盖历史run。

### HDF5/JSON/figures 审计

formal HDF5为`372,470,368 bytes`。原CXI/NeXus-inspired并列布局和必需路径继续存在；新增/变化主要位于`/entry/metrics/B_coverage/geometry/case_descriptors`中的support basis/target和`/entry/metrics/numerical_convergence/padding_control/{sequence,adjacent_changes,gate_comparison}`、`hard_gates/{residual_support_max_relative_change,residual_support_convergence_passed}`。config/metadata/metrics canonical一致、全部numeric finite、`I_stack`非负、必需路径齐全。primary失败后没有`/entry/reconstruction`。

九张预注册PNG全部存在、可读且artifact audit passed：B master/corners为`1500x360`，其余为`850x760`或`1000x700`。人工复核B realization/corners、FI、detector sensitivity和FOV/padding图，曲线、单位、spot定义与control index可辨；因secondary未执行，正确地没有`known_B_recovery_vs_spot_size.png`。

### 关键数值与hard gates

- preflight：master B仍为`3472x3472=868 um`；最大open grid为`3773x3773`；估计峰值`1.638 GiB`、runtime`2845.37 s`、HDF5`0.551 GiB`，均通过4 GB/7200 s预算。
- 50 um active support：`1e-6` formal为`504x504=126 um`，residual-edge maximum`0.363091% > 0.2%`，失败；`1e-7` control为`552x552=138 um`，edge`0.152039%`，通过。二者normalized sensitivity/FI最大变化仅`0.083933% < 5%`。
- 100 um padding：32/64/128 um open shapes为`960/1225/1728`。32到64 um的normalized sensitivity/FI变化分别`0.14479%/0.63010%`；正式gate 64到128 um分别`0.001454%/0.015256%`，通过。
- 400 um formal/strict FOV最大变化`0.025337%`；全局formal/strict/support最大变化`0.083933%`，通过。
- 所有B patches fully inside，minimum margin仍`3.5 um`，formal wrap为0；adjoint error`7.91e-16`，repeat difference 0，finite/nonnegative和artifact semantics均通过。
- 唯一False gate是`residual_edge_energy_passed`；因此primary aggregate为False，secondary按预注册跳过。

### 与Change 04的对比

padding结论被实质修正：旧16到32 um FI变化为`9.1731%`，新32到64 um已降到`0.6301%`，64到128 um进一步降到`0.01526%`，说明64 um guard已处于注册5%平台。50 um residual edge从`2.7302%`降到`0.36309%`，方向正确但formal仍未过0.2%；`1e-7` control首次给出通过证据。

扩大padding相对Change 04使fixed-total per-photon FI在50/100/200/400 um分别变化`-9.42%/-9.75%/-2.68%/-1.50%`；前两项属moderate numerical correction，后两项small。normalized detector sensitivity分别变化`-0.12%/-2.61%/-1.49%/-0.96%`，均small；absolute signal除50 um因support改变为`-2.03%`外均小于0.2%。所以Change 04的spot定性排序保留，但其50/100 um FI绝对数值已由本节修正。

### 更新后的spot与periodic-B指标

fixed-total相对100 um，200/400 um的normalized detector sensitivity为`-37.05%/-50.09%`，absolute signal为`-41.78%/-80.09%`，FI/photon为`-71.72%/-92.86%`；结论仍是large。fixed-center的200/400 um absolute signal为`+132.88%/+218.64%`，而normalized sensitivity与FI变化同上，继续证明absolute signal与per-photon/normalized指标不能合并。

periodic-vs-aperiodic 100 um detector-stack relative L2由`2.1311%`变为`2.0532%`，mean-frame为`1.9118%`，仍是small且非零；正式wrap为0、legacy语义wrap为49，B边界归因未改变。

### 结果支持或反对什么机制

支持：旧padding失败确由guard不足或shape-dependent transfer尚未平台造成，64/128 um已给出清晰平台；50 um B-modulated residual需要比incident-power`1e-6`更大的physical support，且`1e-7`已同时满足edge和relative-change控制。反对：`1e-6`足以作为50 um formal support。未支持：known-B recovery排序、blind recovery、真实detector精度或3D TGV可测性。

### 是否出现新问题

出现一个metric provenance问题：自动common probe-envelope比较域随50 um active support从88 um变成96 um，使该envelope metric跨Change不可直接比较。它不影响detector ROI、padding/residual gates或上述FI结论，但后续必须固定物理comparison ROI，不能继续依赖case集合的最小shape。

### 已知限制

继续受2D projected-phase、轴对称居中、zero tilt、paraxial Gaussian、integer scan、ideal no-noise和Poisson-only bound限制。50 um正式case仍未数值封口；secondary未执行；本Change没有验证`1e-8`support control，也没有独立改变ASM bandlimit/alias-control。source config在结果记录后只更新状态/version，正式run内部保存的执行config仍为`exp031-v6/Change 05`并与其HDF5/metadata一致。

### 是否改变实验状态

Change 05实现、测试、preflight、formal run和artifact记录均完成，但exp031科学状态仍为`Failed/Inconclusive`，不是Passed。相对Change 04，padding blocker已消除；剩余blocker收缩为50 um formal residual support单项。

### 本节判断

Change 05达到“定位并收敛padding”的目标，也证明`1e-7` support可满足50 um edge gate；但预注册formal是`1e-6`，其0.363%仍失败，所以不能执行secondary或把control结果冒充正式通过。Change 04的padding不收敛判断被本节修正，50/100 um FI绝对值也应采用本run；finite-B尺寸公式与periodic差异结论保持稳定。

### 下一步建议

- 属于`exp031 内的新 change`：Change 06预注册50 um formal/control为`1e-7/1e-8`，保持64/128 um padding gate，并在config中冻结probe-envelope comparison ROI为Change 04的88 um或另一个事前指定固定物理域；不得结果后选择。
- 属于`exp041`：继续研究B family、频谱、制造性和信息量，不因本Change重抽B。
- 属于`exp04x`：另开回归任务迁移finite canvas、scan-window adjoint、Gaussian/open residual和convergence metrics；本二维FI数值不迁移。
- 属于`新开独立实验`：beam decenter、tilt、waist-position和像差。
- 属于`等待真实 beam/B/detector 标定`：实际photon budget、gain/full-well、read/dark noise、B exterior与beam参数。

## 23. Change 06 预注册：提升50 um formal support并冻结probe-envelope比较域

### 触发原因

Change 05证明100 um padding已收敛，但50 um预注册formal `1e-6` support的residual edge仍为`0.363091% > 0.2%`；同run的`1e-7` control为`0.152039%`且与formal的sensitivity/FI最大变化仅`0.083933%`。用户授权继续下一步。

### 当前证据

50 um residual edge随support扩大由Change 04的`1e-3: 2.7302%`、`1e-4: 1.0575%`下降到Change 05的`1e-6: 0.36309%`、`1e-7: 0.15204%`。64到128 um padding FI变化仅`0.01526%`，padding不再是blocker。Change 05还发现common probe-envelope比较shape由自动最小case决定，随50 um support扩大从352变为384，造成跨Change metric domain漂移。

### 当前问题或失败

`1e-7`仅作为Change 05 control，不能事后冒充formal；尚无比它更严格的support control。probe-envelope比较域也尚未进入config，虽不参与hard gate，但其自动变化妨碍跨版本审计。primary为False使known-B secondary仍未执行。

### 为什么需要改变

需要在结果前明确把已有通过证据的`1e-7`提升为formal，并用`1e-8`独立验证edge与主要sensitivity/FI已进入平台。同时固定一个共同物理comparison ROI，避免数值support选择改变illumination-envelope报告口径。

### 本次只允许改变的变量

1. 50 um formal/support-control遗漏incident-power目标由`1e-6/1e-7`改为`1e-7/1e-8`。
2. 在config中新增`probe_envelope_comparison_shape: [352,352]`，对应0.25 um sampling下固定`88x88 um`；runner必须验证该shape能装入全部formal/reference probe，只在此固定域计算Gaussian-relative-to-plane envelope L2。
3. 增加comparison-shape provenance和相应unit/HDF5 audit；若primary通过，按既有固定schedule执行secondary reconstruction并保存原始结果及simulation-evaluation-only error。

### 保持不变的变量

所有TGV/optics/scan/B/spot/normalization/finite-difference/noise参数不变；formal open guard保持64 um，100 um padding sequence保持32/64/128 um且gate仍为64到128；detector ROI保持`384x384=96 um`，不得与88 um probe-envelope报告域混淆。0.2% edge、5% convergence、adjoint、coverage、finite/nonnegative和artifact gates不变。reconstruction仍为100/400 um与plane、known-B/probe-only、1 fixed epoch、measurement-derived initialization、truth不参与优化或checkpoint选择。

### 计划修改的文件

`configs/experiments/exp031_finite_B_illumination_spot.yaml`、`scripts/run_exp031_finite_B_illumination_spot.py`、`tests/test_exp031_finite_B_illumination_spot.py`和本实验记录；共享forward/recon接口只有发现必要bug时才允许小范围修改并如实记录。

### 预期机制

`1e-7` formal预期复现Change 05 control的edge通过，`1e-8`更严格control应进一步降低edge且只小幅改变normalized detector sensitivity/FI。固定352 shape应恢复Change 04的88 um envelope metric domain，但不会改变任何传播、detector intensity、FI或gate。若全部primary gates通过，secondary将首次检验matched known-B recovery是否保持spot ordering。

### 评价指标

继续保存全部既有coverage、illumination、absolute/normalized/photon-normalized sensitivity、FI、detector diagnostics、periodic-B和convergence指标；重点报告`1e-7/1e-8` active shape/size、captured power、edge min/median/max、relative changes，固定comparison shape/physical size，以及secondary每case raw recovery、measurement-only schedule、simulation-evaluation-only probe error和spot ordering。

### denominator

support relative change仍为`|1e-8 control - 1e-7 formal| / |1e-7 formal|`。probe-envelope L2在固定352x352域内分别单位L2归一化后比较。FI和其他denominator完全沿用第7节与Change 05，不因reconstruction执行而改变。

### numerical gate

- 50 um `1e-7` formal residual-edge maximum必须`<=2e-3`，且相对`1e-8` control的normalized detector sensitivity/FI最大变化必须`<5%`。
- 100 um 64到128 um padding和400 um formal/strict FOV仍必须`<5%`。
- comparison shape必须exact为352x352、可由所有formal/reference fields无padding裁出；HDF5/JSON/config/metadata必须一致。
- 原all-inside、positive margin、zero wrap、adjoint`<=1e-12`、repeat exact、finite/nonnegative与artifact gates全部保留。
- 只有上述primary全部通过才执行secondary；secondary失败不得反向改变primary gate或optimizer schedule。

### scientific reporting rule

`1e-7/1e-8`只属于physical residual-support numerical convergence，不是新的光斑物理条件。88 um只固定envelope报告口径，不改变96 um detector ROI。secondary truth只用于simulation-evaluation-only error，不得选择optimizer、step、case或checkpoint。即使primary和secondary通过，也只允许称为预注册二维projected-phase数值问题通过，不得外推真实3D可测性。

### 可能失败的方式

`1e-7`在新run因同一master patch/frame sequence仍超过edge gate；`1e-8`造成B margin或preflight预算问题；support/FI变化超过5%；固定352 crop实现方向/中心偏移错误；primary通过后reconstruction出现adjoint、内存、artifact或ordering问题；新增known-B figure/HDF5字段不完整。

### 本节判断

Change 06已在config/code修改和新run之前登记，变量由Change 05直接证据确定且没有放宽门槛。当前只能预期`1e-7`可能通过，不能提前宣称primary或reconstruction成功。

### 下一步建议

属于`继续当前 exp031`：按本节实现、测试、preflight并创建全新formal run；无论primary或secondary结果正负，均追加Change 06结果并更新Section 0。若primary仍失败，下一步应基于具体失败项另开exp031 Change；若primary通过但secondary失败，优先诊断known-B reconstruction算子而非修改forward物理矩阵。

## 24. Change 06 结果：二维primary通过并首次完成known-B secondary

### 实际修改

50 um formal/support-control omitted-power targets按预注册改为`1e-7/1e-8`，对应active shapes `552x552`与`592x592`。config新增`probe_envelope_comparison_shape: [352,352]`；runner新增固定comparison-shape读取、positive/fit/center-parity验证，并将shape、`88x88 um`物理大小及`fixed_config_centered_B_plane` provenance写入geometry和illumination metrics。测试更新为同时验证support、padding和固定reporting ROI。共享传播、B/scan、detector、gate阈值和reconstruction schedule未改。

### 与预注册是否一致

实际support pair、88 um comparison域、64/128 um padding gate和secondary触发均与第23节一致。primary通过后runner自动执行既有100/400 um/plane known-B一轮fixed schedule；没有用truth选择optimizer、case或checkpoint。预期之外的是`1e-8` control edge为`0.200231%`，略高于`1e-7` formal的`0.152039%`，没有随support单调下降；第23节hard gate明确要求formal edge与formal/control sensitivity/FI变化，因此run仍按预注册Passed，但该反例必须作为限制保留。

### 实际运行命令

```powershell
D:\anaconda3\envs\tgv_ptycho_sim\python.exe -m pytest -q tests/test_scan_windows.py tests/test_sample_b_physical_cells.py tests/test_exp031_finite_B_illumination_spot.py
D:\anaconda3\envs\tgv_ptycho_sim\python.exe -m ruff check scripts/run_exp031_finite_B_illumination_spot.py src/tgv_ptycho/forward/scan_windows.py src/tgv_ptycho/forward/exp031.py src/tgv_ptycho/inverse/exp031.py src/tgv_ptycho/objects/sample_b.py src/tgv_ptycho/recon/exp031.py tests/test_scan_windows.py tests/test_sample_b_physical_cells.py tests/test_exp031_finite_B_illumination_spot.py
D:\anaconda3\envs\tgv_ptycho_sim\python.exe scripts/run_exp031_finite_B_illumination_spot.py --config configs/experiments/exp031_finite_B_illumination_spot.yaml --preflight-only
D:\anaconda3\envs\tgv_ptycho_sim\python.exe scripts/run_exp031_finite_B_illumination_spot.py --config configs/experiments/exp031_finite_B_illumination_spot.yaml
D:\anaconda3\envs\tgv_ptycho_sim\python.exe -m pytest -q
```

### tests 和 Ruff 结果

exp031定向为`18 passed in 1.06 s`，其中exp031 runner/physics/HDF5文件单独为`9 passed in 0.92 s`。最终修改范围Ruff为`All checks passed`。全量为`340 passed, 12 failed in 180.91 s`；12项仍全部是用户已有exp040 R10--R14b frozen-config SHA256 mismatch，未新增exp031或reconstruction失败。

### run 路径

通过preflight：`runs/exp031_finite_B_illumination_spot_preflight_20260830_211717/`。最新formal：`runs/exp031_finite_B_illumination_spot_20260830_211803/`。formal `run_state.json`为completed，scientific status为`passed_2D_preregistered_numerical_problem`；所有Change 01--05失败/诊断run继续保留。

### HDF5/JSON/figures 审计

formal HDF5为`575,070,040 bytes`。config/metadata/metrics canonical一致、全部numeric finite、`I_stack`非负、required paths齐全、artifact audit passed。新增/确认字段包括`/entry/metrics/B_coverage/geometry/probe_envelope_comparison_*`、`/entry/metrics/illumination/probe_envelope_comparison_*`和实际`/entry/reconstruction/{gaussian_100um_formal,gaussian_400um_formal,plane_wave_legacy_roi}/{minus,baseline,plus}`。每个reconstruction variant保存raw `P_B_rec_raw`、measurement-only metadata/loss和分离的`simulation_evaluation_only`，没有只保存truth-aligned结果。

十张PNG全部存在可读：原九张加`known_B_recovery_vs_spot_size.png`；B图为`1500x360`，coverage为`850x760`，其余`1000x700`。人工审阅known-B图确认三case error曲线与metrics一致；固定one-epoch与simulation-evaluation-only标签可见。PNG仍只供人工检查，定量值均在HDF5/JSON。

### Primary geometry与numerical gates

- master B保持`3472x3472=868 um`；scan span x/y为49 um；96 um legacy active的145/149/152 um几何答案不变。所有patch fully inside，minimum margin`3.5 um`，formal wrap 0。
- 50 um `1e-7` formal active为`552x552=138 um`，residual-edge maximum`0.152039% <=0.2%`；`1e-8` control为`592x592=148 um`，formal/control normalized sensitivity与FI最大变化`0.135953% <5%`。
- formal cases的最大residual edge为100 um的`0.199170%`，仍通过；64到128 um padding最大变化`0.015256%`；formal/strict/support最大变化`0.135953%`。
- adjoint error`7.91e-16`、repeat difference 0、B determinism/finite/nonnegative/artifact semantics全部通过；primary aggregate为True。
- fixed probe-envelope comparison精确为`352x352=88x88 um`，与detector ROI `384x384=96x96 um`分开。100/200/400 um相对plane envelope差恢复为Change 04同域的`28.168%/7.455%/1.883%`。

### 更新后的spot与periodic-B结果

fixed-total相对100 um，200/400 um normalized detector sensitivity为`-37.05%/-50.09%`，absolute signal为`-41.78%/-80.09%`，FI/photon为`-71.72%/-92.86%`，均维持large结论。fixed-center的200/400 um absolute signal为`+132.88%/+218.64%`，而normalized sensitivity/FI下降同上。50 um fixed-total相对100 um的normalized sensitivity、absolute signal、FI分别为`+34.99%/+3.80%/+120.08%`，但max/median为`56.10`，说明高FI与动态范围压力并存。

periodic-vs-aperiodic 100 um detector-stack relative L2为`2.0532%`，mean-frame`1.9118%`，仍属small；formal wrap 0、legacy语义wrap 49。Change 06未改变B边界结论。

### Secondary known-B reconstruction

secondary实际执行，truth未进入optimizer。固定one-epoch结果为：

| case | measurement-only loss | raw probe error | aligned probe error（simulation only） | truth normalized sensitivity | recovered normalized sensitivity | relative bias |
|---|---:|---:|---:|---:|---:|---:|
| Gaussian 100 um | 0.08589 | 39.74% | 37.71% | 1.6676 | 0.7948 | -52.34% |
| Gaussian 400 um | 0.07633 | 16.21% | 10.50% | 0.8241 | 0.2345 | -71.55% |
| plane legacy ROI | 0.05923 | 27.00% | 26.74% | 0.8478 | 0.6168 | -27.25% |

truth与recovered sensitivity ascending order均为`400 um < plane < 100 um`，所以预注册的ordering-preserved endpoint为True。但误差和sensitivity bias很大；一轮schedule只证明matched operator下排序没有翻转，不证明probe已高精度收敛，也不能用truth误差选择更多iteration。

### 与Change 05的对比

formal residual edge由`1e-6` support的`0.363091%`降为`1e-7` support的`0.152039%`，唯一primary blocker解除。100/200/400 spot、padding与periodic-B数值保持Change 05结果；50 um formal相对Change 05的sensitivity/FI变化落在预注册support-control的0.136%以内。固定88 um comparison域消除了Change 05的96 um派生域漂移，并精确恢复Change 04的同域envelope指标。

### 结果支持或反对什么机制

支持：finite nonperiodic B在所有注册active windows下完整覆盖；64 um open guard已收敛；50 um需要约138 um physical residual support才能在该formal realization/frame set上通过edge gate；更宽spot在几何illumination上趋近plane；fixed-power photon-density下降仍显著压低absolute signal；matched known-B one-epoch保持truth sensitivity排序。反对或未支持：扩大support会使single-realization edge-ring严格单调下降、one-epoch可高精度恢复probe、二维排序可直接迁移真实3D或盲B重建。

### 是否出现新问题

`1e-8` control edge为`0.200231%`，比formal略高且只比阈值高`0.000231`个百分点。最可信解释是扩大window后edge ring落在不同非周期phase cells上，局部B contrast使single-realization edge fraction有小幅波动；这不改变formal gate与主要sensitivity/FI平台，但说明edge metric不应被解释为严格单调函数。reconstruction还显示排序保留与绝对准确度可以明显分离。

### 已知限制

Passed仅针对2D projected-phase、轴对称居中、zero tilt、paraxial Gaussian、integer scan、ideal/no-noise与shot-noise-only FI。secondary只有一个固定epoch，raw/aligned error仍大；没有blind B、真实noise/quantization、stage error、beam aberration或3D multiple scattering。control edge非单调意味着若未来要求更强support证据，应事前要求formal和control都过edge或采用能量加权/多ring诊断，不能在本Change回改gate。source config在结果后只更新状态/version；正式run内执行config为`exp031-v8/Change 06`且与HDF5/metadata一致。

### 是否改变实验状态

exp031从Failed/Inconclusive改为`Passed / 2D preregistered numerical problem only`。这表示预注册primary gates、artifact与secondary execution均完成，不表示真实3D TGV已可测，也不表示one-epoch reconstruction达到工程精度。

### 本节判断

Change 06完成了剩余formal support收敛、固定metric domain和gated secondary闭环。有限非周期B、Gaussian spot矩阵、padding/FOV与known-B ordering现在有同一正式run支撑。最重要的保留条件是：control edge存在轻微非单调，reconstruction仅保序但误差较大，所有结论仍限于二维理想模型。

### 下一步建议

- 属于`暂停/封口当前 exp031`：v9已回答原二维预注册问题；不建议继续为降低单一control edge或truth error在同实验中结果后调参。
- 属于`exp031 内的新 change`：只有需要更强edge robustness时，事前注册formal/control双edge gate、多ring或energy-weighted support诊断。
- 属于`exp041`：研究B频谱、family、制造性及edge metric对phase-cell realization的统计敏感性。
- 属于`exp04x`：另开回归任务迁移large canvas、scan-window adjoint、fixed comparison ROI、Gaussian/open residual和convergence metrics；不得迁移本二维FI/ordering数值。
- 属于`新开独立实验`：系统研究known-B reconstruction iteration schedule、measurement-only stopping与accuracy/order tradeoff；另行研究beam decenter/tilt/aberration。
- 属于`等待真实 beam/B/detector 标定`：在声称真实可测或饱和前获得photon budget、gain/full-well、read/dark、B exterior和beam参数。
