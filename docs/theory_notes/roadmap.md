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

## Phase 5: waist observability and parametric fitting

从 recovered probe 或 simulated probe signature 中拟合 `D(z)` 或低维 TGV shape parameters，估计 `D_waist`。

当前状态：`Single-parameter baseline complete; nuisance stage not started`。`exp051` 已完成 selected exp040 scalar working model 下、matched raw `P_B_true` 输入的
q8 plateau-interval 单参数 oracle baseline，并按预注册 gate 判定为 `Passed`。正式结果 run 为
`runs/exp051_TGV_3d_multislice_true_probe_waist_fit_q8_plateau_interval_20260824_161440/`。该结论只说明固定 q8
operator 的区间值 self-consistency；`reference_validated=false`、`full_tgv_reference_authorized=false`，不构成
真实三维电磁准确性、真实计量精度、resolution 或 detection-limit 结论。任意三维 `D(z)` 恢复和 nuisance fitting
仍不在首次 Phase 5 baseline 内。`exp053` 已进一步用 exp042 authoritative raw `P_B_rec` 完成同一 fixed-q8 单参数 reconstructed-probe baseline。两条平行证据链当前状态为：

| 实验 | forward model | 输入 | 问题 | 状态 |
|---|---|---|---|---|
| `exp050` | 2D projected | `P_B_true` | 单参数 `D_waist` oracle fitting | `Reserved / Not started` |
| `exp051` | exp040 scalar multislice working model | `P_B_true` | 单参数 `D_waist` oracle fitting | `Passed` |
| `exp052` | 2D projected | matched `P_B_rec` | reconstruction 后的单参数腰径拟合 | `Reserved / Not started` |
| `exp053` | exp040 scalar multislice working model | exp042 GN-CG raw `P_B_rec` | reconstruction 后的单参数腰径拟合 | `Passed` |
| `exp054` | 2D projected | 先 true、后 matched rec | nuisance / multi-parameter identifiability | `Reserved / Not started` |
| `exp055` | exp040 scalar multislice working model | 先 true、后 matched rec | nuisance / multi-parameter identifiability | `Reserved / Not started; next scientific priority` |

依赖链为 `exp050 -> exp052 -> exp054`（projected diagnostic）和 `exp051 -> exp053 -> exp055`（scalar working-model primary）。编号不表示六项必须串行完成。projected 与 multislice 的 truth/data/reconstruction 必须各自 matched；把一种 forward 产生的数据交给另一种模型拟合属于以后单独预注册的 model-mismatch 问题。

当前最有意义的下一科学任务是另行预注册 `exp055` 的最小 nuisance-identifiability gate：先用 `P_B_true` 判断 `D_waist` 与少量 geometry/calibration nuisance 的耦合，再在前一问题闭合后原样换为 exp042 authoritative raw `P_B_rec`。fixed-q8 operator 非光滑，不能只依赖 smooth Jacobian；方法、参数范围、profile/equivalence-set、multi-start contract 和 thresholds 必须在 formal 前冻结，truth 只允许 post-freeze simulation evaluation。`exp050/052/054` 继续保留为 projected diagnostic，不因 primary 链完成而自动启动。direct detector-intensity fitting 从 `exp056` 以后再根据 nuisance 证据决定编号与范围。

## Phase 6: tilted A and multi-angle simulation

扩展到 tilted sample、non-circular via、multi-angle acquisition 和更复杂的 TGV geometry。

## Phase 7: noise, stage error, camera calibration, experimental data

加入 camera noise、stage position error、detector calibration、preprocessing pipeline 和 experimental data reader。
