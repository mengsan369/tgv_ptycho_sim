# 路线图

## Phase 0: propagation sanity check

验证 field generation、angular spectrum propagation、plotting、HDF5 saving 和基础 test coverage。

## Phase 1: standard ePIE with known probe and random B

使用 known probe 和 synthetic random B object，验证 scan overlap、diffraction data generation 和 ePIE update 的基础流程。

## Phase 2: A thin phase object -> recover P_B -> backpropagate to A

用简化 thin phase 样品 A 生成未知 B-plane probe。通过 B 扫描恢复 `P_B`，再 backpropagate 到 A 附近。

状态：已由 `exp020_A_thin_phase_probe_recovery` 在理想二维无噪声条件下实现并运行验证。当前实现使用纯相位 A/B、已知 A 空白参考区、整数像素抖动扫描和每轮 A 平面投影；这只是可辨识性 baseline，不表示弱先验、含噪或真实 3D TGV 已解决。详见 `docs/experiment_design/exp020_A_thin_phase_probe_recovery.md`。

## Phase 3: TGV-like 2D effective phase model

使用 effective 2D phase model 做早期 observability 测试。该模型只用于验证 probe sensitivity，不等价于真实三维腰径模型。

状态：已由 `exp030_TGV_2d_effective_phase` 在单孔、轴对称、无噪声 projected-phase 条件下完成模型验证、采样/有限差分收敛、probe 与 detector sensitivity、local Jacobian 和 matched blind Stage D 检查。正式通过 run 为 `runs/exp030_TGV_2d_effective_phase_20260810_121124/`。该结论仍不表示真实 3D TGV 腰径已可测；多孔阵列、multislice、noise、tilt 和 parametric fitting 需要后续独立实验。

`exp031_finite_B_illumination_spot` 随后在独立预注册的二维数值问题中关闭了 finite nonperiodic B、Gaussian spot、support/FOV/open-padding 和 known-B one-epoch sensitivity-ordering controls，正式 run 为 `runs/exp031_finite_B_illumination_spot_20260830_211803/`，状态为 `Passed / 2D preregistered numerical problem only`。其 recovered probe error 仍较大，且所有结果都不能迁移为 3D、真实噪声或 production reconstruction 结论。

## Phase 4: 3D TGV multi-slice forward model

生成轴对称 refractive-index volume，并用 multi-slice propagation 逐层传播。

状态：`exp040_TGV_3d_multislice_forward` 已完成从 R0 baseline 到 R14B 的分阶段 forward、数值边界和
reference-validation 诊断。自 2026-08-17 起工作状态为 `Frozen / Paused`，整体科学状态仍为
`Inconclusive`。最新 R14B formal 状态为 `Failed / r14_no_scalable_scipy_solver`；当前保持
`reference_validated=false`、`full_tgv_reference_authorized=false`，不得把同模型 self-consistency
提升为真实三维电磁物理准确性。恢复更高级物理验证前须重新定义研究问题、reference 身份、solver 路线和
验收门槛。详见
`docs/experiment_design/exp040_TGV_3d_multislice_forward.md`。

Phase 4 与 Phase 5 之间的 measurement/reconstruction 衔接状态：

- `exp041_information_rich_sample_b_design`：`Discussion draft / Not started`。保留为 sample-B/scan 设计问题；当前 fixed-parameter exp053 已通过，因此不进行无目标的 B-family sweep。以后应由 nuisance identifiability 或 reconstruction bottleneck 给出明确的信息方向后再启动。
- `exp042_TGV_3d_multislice_probe_reconstruction`：Phase 5 已触发并完成一次定向恢复。authoritative run `runs/exp042_TGV_3d_multislice_probe_reconstruction_dev_20260831_183139/` 的 spectrally damped GN-CG raw control 已 complete/validated，并成为 exp053 的 authoritative source。它只覆盖单一、无噪声、matched、known-B case；最低谱端问题仍按第 23 节挂起，不需要因 exp053 已通过而继续无目标优化。
- `exp043_TGV_3d_multislice_blind_probe_B_reconstruction`：在上述 authoritative matched-q4 data/operator 上完成最小 alternating spectrally damped block GN-CG 扩展，唯一 formal run 为 `runs/exp043_TGV_3d_multislice_blind_probe_B_reconstruction_20260903_153944/`。source/operator、P-only、B-only、blind measurement fit 和两初始化 prediction repeatability 均闭合，但 representative gauge-aware active-B/exit errors 超过冻结 gates，故已冻结为 `Failed / measurement_consistent_but_component_recovery_non_identifiable`。该结果不否定 exp042 known-B control，也不构成所有 blind 算法均不可能的证明；它禁止在当前 measurement design 上只凭低 residual 宣称 component recovery。
- `exp044_TGV_3d_large_canvas_B_localized_illumination`：已完成 E43 hash bridge 与 causal C0/C1/C2/C3 measurement-design formal，authoritative run 为 `runs/exp044_TGV_3d_large_canvas_B_localized_illumination_formal_20260903_212213/`。forward 与 E43/C0--C3 known-B controls均闭合，但 C3 blind maximum detector residual `0.0476316` 和两初始化 prediction difference `0.0192078` 超过冻结 gates，故状态为 `Failed / blind_measurement_reconstruction_not_closed`。post-freeze coverage-weighted B/exit errors相对exp043均恶化，不能作为measurement-closed identifiability结论。exp044冻结停止；beam/scan/B或optimizer后续另开实验，不回改exp043。blind reconstructed probe waist fitting只有在新的raw probe通过注册measurement/component合同后才能另开exp05x。

## Phase 5: waist observability and parametric fitting

从 recovered probe 或 simulated probe signature 中拟合 `D(z)` 或低维 TGV shape parameters，估计 `D_waist`。

当前状态：`2D projected true- and reconstructed-probe baselines complete / Passed; minimal scalar true-probe nuisance formal complete / Failed; reconstructed nuisance stage not authorized`。`exp050` 已在 exp030 continuous axisymmetric Fresnel--Hankel projected working model 内，使用 authoritative matched raw `P_B_true` 完成固定其他参数的单参数 oracle fit，并按冻结 gate 判定为 `Passed`。formal run 为 `runs/exp050_TGV_2d_projected_true_probe_waist_fit_20260902_192303/`；`33.2975--33.3025 um` 仅为 5 nm profile-grid cell，不是物理 uncertainty、resolution 或 detection limit。`exp052` 已在同一 exp030 detector data/operator 下完成 known-B probe-only 与 genuine blind-ePIE joint-reconstruction 两条 raw `P_B_rec` 通道的唯一 combined formal；两臂 reconstruction、fit 与 artifacts 均为 `Passed`，authoritative run 为 `runs/exp052_TGV_2d_projected_reconstructed_probe_waist_fit_20260902_235549/`。两臂 `33.2995--33.3005 um` 只是一格 1 nm 注册数值单元，不能解释为物理 uncertainty 或真实三维精度。`exp051` 已完成 selected exp040 scalar working model 下、matched raw `P_B_true` 输入的
q8 plateau-interval 单参数 oracle baseline，并按预注册 gate 判定为 `Passed`。正式结果 run 为
`runs/exp051_TGV_3d_multislice_true_probe_waist_fit_q8_plateau_interval_20260824_161440/`。该结论只说明固定 q8
operator 的区间值 self-consistency；`reference_validated=false`、`full_tgv_reference_authorized=false`，不构成
真实三维电磁准确性、真实计量精度、resolution 或 detection-limit 结论。任意三维 `D(z)` 恢复和 nuisance fitting
仍不在首次 Phase 5 baseline 内。`exp053` 已进一步用 exp042 authoritative raw `P_B_rec` 完成同一 fixed-q8 单参数 reconstructed-probe baseline。两条平行证据链当前状态为：

| 实验 | forward model | 输入 | 问题 | 状态 |
|---|---|---|---|---|
| `exp050` | exp030 2D projected working model | authoritative raw `P_B_true` | 单参数 `D_waist` oracle fitting | `Passed / formal complete and artifacts validated` |
| `exp051` | exp040 scalar multislice working model | `P_B_true` | 单参数 `D_waist` oracle fitting | `Passed` |
| `exp052` | exp030 2D projected working model | known-B raw `P_B_rec` 与 genuine blind-ePIE raw `P_B_rec` | 双通道 reconstruction 后的单参数腰径拟合 | `Passed / combined formal complete and artifacts validated` |
| `exp053` | exp040 scalar multislice working model | exp042 GN-CG raw `P_B_rec` | reconstruction 后的单参数腰径拟合 | `Passed` |
| `exp054` | 2D projected | 先 true、后 matched rec | nuisance / multi-parameter identifiability | `Reserved / Not started` |
| `exp055` | exp040 scalar multislice working model | authoritative raw `P_B_true`；raw `P_B_rec` 未启动 | `D_waist + tied D_surface + z_waist` 最小 geometry nuisance identifiability | `Failed / multistart_search_path_inconsistent; formal complete and artifacts validated` |

依赖链为 `exp050 -> exp052 -> exp054`（projected diagnostic）和 `exp051 -> exp053 -> exp055`（scalar working-model primary）。编号不表示六项必须串行完成。projected 与 multislice 的 truth/data/reconstruction 必须各自 matched；把一种 forward 产生的数据交给另一种模型拟合属于以后单独预注册的 model-mismatch 问题。

`exp055` 的 true-probe formal run 为 `runs/exp055_TGV_3d_multislice_true_probe_nuisance_identifiability_20260901_193257/`。在预注册的225-point coarse与729-point local screen内，三档 equivalence set只有 nominal一个 nuisance node，q8 `D_waist` 投影仍为约 `0.17064 nm` 的单一窄区间；但四条冻结六邻域 multistart paths全部停在不同 non-qualifying local minima，因此按预注册顺序为 `Failed`。这不是 nuisance equivalence non-uniqueness，也不是实际计量 resolution；它说明最小局部搜索合同对初始化不稳健。true-probe stage未闭合，故不启动 raw `P_B_rec` stage，不返回 exp042，也不以本证据触发Gaussian illumination、large aperiodic B或新forward operator。当前应停止本 exp055方向；若以后研究全局profile唯一但局部搜索易陷的算法问题，应另行预注册独立 inverse-method experiment。`exp050` 与 `exp052` 已在各自二维边界内闭合；`exp054` 仍保留为 projected nuisance diagnostic。direct detector-intensity fitting仍须按新的证据和范围另行编号。

## Phase 6: tilted A and multi-angle simulation

扩展到 tilted sample、non-circular via、multi-angle acquisition 和更复杂的 TGV geometry。

## Phase 7: noise, stage error, camera calibration, experimental data

加入 camera noise、stage position error、detector calibration、preprocessing pipeline 和 experimental data reader。
