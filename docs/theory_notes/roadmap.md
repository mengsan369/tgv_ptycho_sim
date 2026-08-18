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

## Phase 4: 3D TGV multi-slice forward model

生成轴对称 refractive-index volume，并用 multi-slice propagation 逐层传播。

状态：`exp040_TGV_3d_multislice_forward` 已完成从 R0 baseline 到 R14B 的分阶段 forward、数值边界和
reference-validation 诊断。自 2026-08-17 起工作状态为 `Frozen / Paused`，整体科学状态仍为
`Inconclusive`。最新 R14B formal 状态为 `Failed / r14_no_scalable_scipy_solver`；当前保持
`reference_validated=false`、`full_tgv_reference_authorized=false`，不得把同模型 self-consistency
提升为真实三维电磁物理准确性。近期优先启动 exp050 复原研究，恢复更高级物理验证前须重新定义研究问题、
reference 身份、solver 路线和验收门槛。详见
`docs/experiment_design/exp040_TGV_3d_multislice_forward.md`。

## Phase 5: waist observability and parametric fitting

从 recovered probe 或 simulated probe signature 中拟合 `D(z)` 或低维 TGV shape parameters，估计 `D_waist`。

## Phase 6: tilted A and multi-angle simulation

扩展到 tilted sample、non-circular via、multi-angle acquisition 和更复杂的 TGV geometry。

## Phase 7: noise, stage error, camera calibration, experimental data

加入 camera noise、stage position error、detector calibration、preprocessing pipeline 和 experimental data reader。
