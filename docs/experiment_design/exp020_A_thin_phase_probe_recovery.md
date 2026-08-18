# exp020：A 生成 probe，盲 ePIE 恢复 B/probe 并反传到 A

## 实验定位与结论

`exp020_A_thin_phase_probe_recovery` 对应 Phase 2。研究问题是：在 A 调制平面波、B 提供 overlapping ptychographic scan、相机只记录 intensity 的情况下，能否先恢复 B 与 B 平面 probe，再反向传播得到二维 A transmission。

本次理想仿真给出的答案是“在明确强先验下可以”。最终 run 从 initial frozen data-fidelity loss `3.120259e-1` 收敛到 `7.222912e-16`；不使用 `A_true` 的空白参考区校正后，A 有效区 wrapped phase RMSE 为 `6.987997e-16 rad`。该结果接近机器精度，是 noiseless inverse-crime baseline，不代表真实 3D TGV 腰径反演已经解决。

通用算法和歧义处理见 `docs/theory_notes/epie_blind_probe_A.md`。

强先验是算法提前知道：

- A、B 都是纯相位样品，振幅固定为 1；
- A 的圆形作用区域已知，外围是 transmission=1 的空白参考区；
- probe 总能量可由相机帧总能量确定；
- 波长、传播距离、采样和扫描位置完全准确；
- 无噪声，正演与反演使用完全相同的模型。

## 运行入口

配置：

```text
configs/experiments/exp020_A_thin_phase_probe_recovery.yaml
```

入口：

```powershell
python scripts/run_full_pipeline.py --config configs/experiments/exp020_A_thin_phase_probe_recovery.yaml
```

最终审阅 run：

```text
runs/exp020_A_thin_phase_probe_recovery_20260712_135337/
```

该 run 的 `metadata.json` 记录 Git commit `7b85c5a2c7559244625554168103e1cc9332a1ed`。工作区有未提交修改，因此 commit 只标识基线提交，完整实际参数和新代码状态仍需结合 run 内 `config.yaml` 与当前工作区审阅。

## 配置与随机种子

- shape：`64 x 64`，axis 顺序 `(ny, nx)`；
- sampling：`dx = 2 um`；
- wavelength：`532 nm`；
- `z_AB = 1 mm`，`z_BC = 1 mm`；
- A：圆形有效区半径 `40 um`，平滑随机纯相位，RMS `0.5 rad`，相关长度 `6 um`，seed `20260721`；
- B：3 pixel feature 的随机纯相位，范围 `[-0.8, 0.8] rad`，seed `20260722`；
- scan：`9 x 9`，step `10 um`，每轴最多 `1 pixel` 整数抖动，seed `20260723`；
- reconstruction：180 轮，`beta_probe=0.08`，`beta_object=0.5`，shuffle seed `20260724`；
- B 初始微弱随机相位 seed `20260725`；
- detector noise：无。

所有 seed 均在最终 run 的 `config.yaml` 和 HDF5 `/entry/config_yaml` 中保存。

## 算法流程

1. 生成平面波 `U_inc` 和带空白参考区的二维纯相位 `A_true`。
2. 用 angular spectrum 将 `A_true * U_inc` 从 A 传播到 B，得到未知 `P_B_true`。
3. 对随机纯相位 B 执行带整数像素抖动的 overlapping scan，并传播到 detector，形成 81 帧 `I_stack`。
4. 从相机平均 amplitude 零相位回传初始化 probe；B 从接近单位 transmission 的微弱随机相位初始化。
5. 执行 blind ePIE，同时更新 probe 与 B。
6. 每轮把 probe 回传到 A，用已知空白参考区去除相位平面和 amplitude scale，投影为纯相位 A，再传播回 B。
7. 用每帧测得的平均总能量约束 probe L2 norm；B amplitude 固定为 1。
8. 最终回传 `P_B_rec`，保存 raw A、参考校正 A 和纯相位 A。
9. truth 只用于 metrics、error figure 和 `simulation_evaluation_only` 对齐结果。

## 评价指标与实际结果

最终 run `..._135337`：

| 指标 | 数值 | 用途 |
|---|---:|---|
| initial frozen data-fidelity loss | `3.120259e-1` | 初始化与测量不一致程度 |
| final frozen data-fidelity loss | `7.222912e-16` | 最终 detector amplitude 一致性 |
| first sequential loss | `9.087106e-2` | 第一轮顺序更新平均 loss |
| last sequential loss | `5.438905e-16` | 第 180 轮顺序更新平均 loss |
| A active phase RMSE，reference correction | `6.987997e-16 rad` | 不使用 A truth 的主结论指标 |
| A full-field complex relative error，reference correction | `3.890714e-16` | 不使用 A truth 做对齐 |
| P_B complex relative error | `3.950796e-16` | simulation evaluation only |
| B illuminated complex relative error | `1.441070e-15` | simulation evaluation only |
| probe L2 norm target/final | `64 / 64` | 测量能量约束检查 |
| illuminated pixel fraction | `1.0` | 当前 probe 覆盖全场 |

“simulation evaluation only”指标使用 truth 去除盲重建的离散线性相位和复增益歧义。它们不是实际算法可用输出。A 的主指标使用已知空白参考区校正，不读取 `A_true`。

## Run 目录与图片

最终 run 包含：

```text
config.yaml
metadata.json
metrics.json
figures/
  A_true_amp_phase.png
  A_truth_reconstruction_error.png
  P_B_true_amp_phase.png
  P_B_rec_raw_amp_phase.png
  P_B_truth_reconstruction_error_eval_only.png
  B_truth_reconstruction_error_eval_only.png
  detector_frames.png
  scan_positions.png
  loss_curve.png
outputs/
  exp020_full_pipeline.h5
```

人工检查确认：标题、坐标轴和 colorbar 未遮挡数据；amplitude、phase、intensity 和 error 均有独立标签；纯相位 amplitude 使用稳定色标；loss 单调下降并在约 135 轮达到浮点平台；scan 图可见整数像素抖动；detector montage 显示第 0/40/80 帧的 `log10 intensity`。

## HDF5 主要内容

未修改项目级 HDF5 主结构。最终文件使用现有 `/entry` 并列布局：

```text
/entry/data/I_stack                         (81, 64, 64), float64
/entry/data/scan_positions                  (81, 2), float64, (x,y), m
/entry/instrument/{wavelength,dx,z_AB,z_BC,detector_pixel_size,medium_index}
/entry/sample/sample_A_parameters/...
/entry/sample/sample_A_support_mask_known_prior
/entry/sample/sample_A_reference_mask_known_prior
/entry/sample/sample_B_parameters/...
/entry/truth/{incident_field_true,A_true,A_phase_true,P_B_true,B_true}
/entry/reconstruction/{P_B_init,P_B_rec,B_init,B_rec}
/entry/reconstruction/{field_after_A_rec,A_rec_raw,A_rec_reference_corrected,A_rec_phase_only}
/entry/reconstruction/{loss_curve,initial_data_fidelity_loss,final_data_fidelity_loss}
/entry/reconstruction/{illumination_map,illuminated_mask,reference_correction,settings}
/entry/reconstruction/simulation_evaluation_only/...
/entry/config_yaml
/entry/metadata/...
/entry/metrics/...
```

raw reconstruction 始终单独保留。truth-aided 数组只在名称明确的 `simulation_evaluation_only` group 中。PNG 不是计算输入，所有数值结果均在 HDF5/metrics 中。

## 测试、lint 与回归

实际执行：

```powershell
D:\anaconda3\envs\tgv_ptycho_sim\python.exe -m pytest -q
D:\anaconda3\envs\tgv_ptycho_sim\python.exe -m ruff check <本次修改文件>
D:\anaconda3\envs\tgv_ptycho_sim\python.exe -m ruff check src scripts tests
D:\anaconda3\envs\tgv_ptycho_sim\python.exe scripts/run_exp001_forward.py --config configs/experiments/exp001_propagation_sanity.yaml
D:\anaconda3\envs\tgv_ptycho_sim\python.exe scripts/run_exp010_recon.py --config configs/experiments/exp010_epie_known_probe.yaml
```

结果：

- full pytest：`20 passed`；
- 本次修改文件 Ruff：全绿；
- 项目级 Ruff：13 个既有问题，位于 `scripts/run_exp001_forward.py`、`calibration/stage.py`、用户已有修改的 `optics/angular_spectrum.py`、`recon/losses.py` 和 `recon/rpie.py`；本任务未顺手修改；
- exp001 回归成功，新 run：`runs/exp001_propagation_sanity_20260712_135016/`；
- exp010 回归成功，新 run：`runs/exp010_epie_known_probe_20260712_135025/`，final loss `4.226706e-3`，aligned B relative error `4.538628e-2`。

新增测试覆盖 A generator 的 shape/dtype/seed/纯相位/空白区、整数像素 jitter 的轴与单位、probe 初始化能量、blind affine phase/scale 评估、A 参考回传、probe norm/constraint 和 exp020 HDF5 raw/evaluation-only 分离。

## 诊断 run

开发过程中未删除或覆盖任何 run：

- `..._134316`、`..._134339`、`..._134357`、`..._134448`：在结果写入前退出的空诊断 run。原因是本机 Conda NumPy 的 `np.linalg.norm/lstsq/vdot` 路径触发 Windows `0xc06d007f`；实现改为显式 elementwise sum 和二维闭式拟合。
- `..._134635`：数值成功的首次验证 run，之后发现纯相位 amplitude 图的恒定色标展示不佳。
- `..._134920`：修正恒定 amplitude 色标和 Git commit metadata 后的完整 run，后续人工检查发现 raw probe 长标题被裁切。
- `..._135337`：缩短 raw probe 标题、且 run 内 config 已标记验证完成的最终审阅 run。

## 限制与下一步边界

- 结果依赖 A/B 纯相位、A 空白参考区已知和每轮强投影；必须另做 ablation 才能判断去掉某项先验后的可辨识性。
- 当前无噪声、无 calibration error，forward/inverse 完全同模，属于 inverse crime。
- integer shift、periodic B、同 shape/同 pixel size 仍是阶段性限制。
- 这个二维 A phase 不是 3D TGV `D(z)`，不能据此宣称已测得腰径。
- 继续当前 exp020 可做迭代数缩减、约束 ablation 和多 seed 稳健性；加入噪声/位置误差属于新的 robustness 实验；3D TGV multi-slice 与腰径参数反演应继续使用后续独立实验编号。
