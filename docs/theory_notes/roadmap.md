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

## Phase 4: 3D TGV multi-slice forward model

生成轴对称 refractive-index volume，并用 multi-slice propagation 逐层传播。

## Phase 5: waist observability and parametric fitting

从 recovered probe 或 simulated probe signature 中拟合 `D(z)` 或低维 TGV shape parameters，估计 `D_waist`。

## Phase 6: tilted A and multi-angle simulation

扩展到 tilted sample、non-circular via、multi-angle acquisition 和更复杂的 TGV geometry。

## Phase 7: noise, stage error, camera calibration, experimental data

加入 camera noise、stage position error、detector calibration、preprocessing pipeline 和 experimental data reader。
