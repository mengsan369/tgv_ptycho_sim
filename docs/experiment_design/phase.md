# phase

项目大概八个phase

## 2026-09-01 实时阶段状态

- Phase 0--3 已完成各自限定条件下的 propagation、基础 ePIE、二维强先验 blind baseline、projected-phase observability，以及 exp031 finite nonperiodic B / illumination-spot 数值控制；这些结论不等价于真实三维腰径计量。
- Phase 4 的 `exp040` 为 `Inconclusive / Frozen / Paused`，保持 `reference_validated=false`、
  `full_tgv_reference_authorized=false`。
- `exp042` 已按 exp053 的定向反馈完成 spectrally damped GN-CG control；authoritative run `20260831_183139` complete/validated，其 raw `P_B_rec` 已被 exp053 原样使用。当前无需继续无目标优化 exp042。
- Phase 5 的 scalar working-model 单参数链已经闭合：`exp051` true-probe oracle 与 `exp053` reconstructed-probe baseline 均正式 `Passed`。exp053 authoritative run 为 `20260901_122210`，但结论只覆盖固定其他参数、单一无噪声 known-B matched case。
- projected diagnostic 的 `exp050/052/054` 仍为 `Reserved / Not started`；`exp055` 为下一科学优先级，但 nuisance parameter set、方法、bounds、gates、YAML、实现和 run 均尚未预注册。
- direct detector-intensity fitting、model mismatch、robustness 和真实数据从 `exp056` 以后根据 Phase 5 前述证据再决定，当前不预先扩张。

- Phase 0: propagation sanity check

    先验证 plane wave、角谱传播、plot、HDF5 保存和基础测试。

- Phase 1: standard ePIE with known probe and random B

    用已知 probe + 随机 B 样品，验证标准 ePIE 流程。

- Phase 2: A thin phase object -> recover P_B -> backpropagate to A

    用简化薄相位样品 A 生成未知 probe，恢复 P_B 后反传回 A 附近。

- Phase 3: TGV-like 2D effective phase model

    做 TGV 的 2D effective phase 近似模型，先看 probe sensitivity 和可观测性。

- Phase 4: 3D TGV multi-slice forward model

    建立 3D 折射率体 n(x,y,z)，用 multi-slice 做真实一些的 forward simulation。

- Phase 5: waist observability and parametric fitting

    从 simulated / recovered probe signature 里拟合 D(z) 或 TGV 参数，估计 D_waist。

- Phase 6: tilted A and multi-angle simulation

    扩展到倾斜样品、非圆孔、多角度采集。

- Phase 7: noise, stage error, camera calibration, experimental data

    加入噪声、位移台误差、相机标定、真实数据 preprocessing 和 experimental data reader。
