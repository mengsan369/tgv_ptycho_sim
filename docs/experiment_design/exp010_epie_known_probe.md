# exp010：Known-Probe ePIE Reconstruction

## 实验定位

`exp010_epie_known_probe` 是项目的 Phase 1 实验，用来验证最小 ptychography 数据闭环：使用已知 probe 和随机样品 B 生成 overlapping scan intensity stack，再通过 ePIE 恢复 B 的 complex transmission。

这个实验暂时不引入 TGV 样品 A。B 平面的 probe 直接定义为 Gaussian field，并在整个 reconstruction 中保持不变。这样可以先单独验证 forward model、扫描位置、detector amplitude constraint 和 object update，减少 Phase 2 同时恢复 probe 与 B 时的调试变量。

## 启动方式

实验配置文件：

```text
configs/experiments/exp010_epie_known_probe.yaml
```

运行脚本：

```text
scripts/run_recon.py
```

在项目根目录执行：

```bash
python scripts/run_recon.py --config configs/experiments/exp010_epie_known_probe.yaml
```

VS Code 中也可以使用 `.vscode/launch.json` 里的 `run_recon exp010 known probe ePIE` 配置启动调试。

## 实验流程

默认配置执行以下步骤：

1. 在 B 平面生成已知 Gaussian probe `P_B_true`。
2. 生成随机 amplitude-phase 样品 `B_true`。
3. 生成 `9 x 9` grid scan，共 81 个扫描位置，相邻位置保持较高 overlap。
4. 对每个位置移动 B，形成 exit wave：

   ```text
   psi_j = P_B_true * shifted(B_true, r_j)
   ```

5. 使用 angular spectrum method 传播到 detector，记录：

   ```text
   I_j = |propagate(psi_j, z_BC)|^2
   ```

6. 将所有 detector intensity 组成 `I_stack`。
7. 使用固定的 `P_B_true` 和均匀的初始 B，调用 `epie_reconstruct()`。
8. 每个扫描位置执行 detector amplitude replacement、反向传播和 B update；probe 不更新。
9. 保存 raw reconstruction、loss curve、illumination map、仿真误差指标、PNG 和 HDF5。

默认配置使用 `96 x 96` 数组、`2 um` sampling、`532 nm` wavelength、`10 mm` 的 `z_BC`、80 次 ePIE 迭代和随机扫描顺序。

更详细的更新公式、global phase 和算法选择依据见：

```text
docs/theory_notes/epie_known_probe.md
```

## Run 目录输出

每次运行都会创建一个带时间戳的新目录：

```text
runs/exp010_epie_known_probe_YYYYMMDD_HHMMSS/
├── config.yaml
├── metadata.json
├── metrics.json
├── figures/
│   ├── known_probe_amp_phase.png
│   ├── detector_frames.png
│   ├── scan_positions.png
│   ├── loss_curve.png
│   └── B_truth_reconstruction_error.png
└── outputs/
    └── epie_known_probe.h5
```

这些文件的用途如下：

- `config.yaml`：本次实际使用的完整配置副本和随机种子。
- `metadata.json`：实验名称、Phase、创建时间、Git commit、算法、扫描坐标约定和当前模型限制。
- `metrics.json`：initial/final data-fidelity loss、loss reduction、B reconstruction error、amplitude RMSE、wrapped phase RMSE、probe error 和 illumination coverage。
- `figures/`：供人判断 forward data、扫描设计和 reconstruction 是否合理。
- `outputs/epie_known_probe.h5`：包含测量数据、真值、重建结果、配置、元数据和指标的主文件。

## 图片说明

### `known_probe_amp_phase.png`

显示 B 平面 known probe：

- 左图：`|P_B_true|`；
- 右图：`angle(P_B_true)`。

它对应 HDF5 中：

```text
/entry/truth/P_B_true
/entry/reconstruction/P_B_rec
```

在 known-probe 实验中两者应该完全相同，`probe_relative_error` 应为 0。

### `detector_frames.png`

显示 `I_stack` 中第 0、40、80 帧的 detector intensity。为了同时观察强弱 diffraction feature，图片使用 `log10 intensity` color scale。

它们对应 HDF5 中：

```text
/entry/data/I_stack[0]
/entry/data/I_stack[40]
/entry/data/I_stack[80]
```

### `scan_positions.png`

显示 81 个 B scan position。横纵坐标分别为 `x` 和 `y`，单位为 `um`；点的颜色表示 acquisition order。原始数值保存在：

```text
/entry/data/scan_positions
```

HDF5 中坐标列顺序固定为 `(x, y)`，单位为 m。

### `loss_curve.png`

显示每轮 ePIE sequential data-fidelity loss，纵轴为 `log10(loss)`。它对应：

```text
/entry/reconstruction/loss_curve
```

程序还会在迭代前后冻结 probe 和 B，对全部扫描位置重新计算一次 loss，分别保存为：

```text
/entry/reconstruction/initial_data_fidelity_loss
/entry/reconstruction/final_data_fidelity_loss
```

这两个数值更适合用于报告迭代前后的总体改善。

### `B_truth_reconstruction_error.png`

这是一张两行三列的 comparison figure：

- 第一行：B truth amplitude、B reconstructed amplitude、amplitude error；
- 第二行：B truth phase、B reconstructed phase、wrapped phase error。

误差图只显示 `illuminated_mask` 内的区域。灰色部分表示 probe illumination 太弱，不应纳入当前 reconstruction accuracy 判断。

为了消除 intensity-only reconstruction 无法确定的 constant global phase，comparison figure 使用仿真 truth 对 B reconstruction 做了单一 global-phase alignment。这个操作只用于仿真评估，不属于 ePIE 原始输出。

## HDF5 数据内容

`outputs/epie_known_probe.h5` 的主要结构为：

```text
/entry
├── data
│   ├── I_stack                         # shape: (81, ny, nx)
│   └── scan_positions                  # shape: (81, 2), (x, y), unit m
├── instrument
│   ├── wavelength                      # m
│   ├── dx                              # B-plane sampling, m
│   ├── z_AB                            # 0 m; probe is directly defined
│   ├── z_BC                            # m
│   ├── detector_pixel_size             # m
│   └── medium_index
├── sample
│   ├── sample_A_type
│   ├── sample_B_type
│   ├── sample_B_parameters/...
│   └── tgv_parameters
├── truth
│   ├── P_B_true
│   └── B_true
├── reconstruction
│   ├── P_B_rec                         # fixed known probe
│   ├── B_init
│   ├── B_rec                           # raw ePIE output
│   ├── B_rec_aligned_to_truth          # simulation evaluation only
│   ├── loss_curve
│   ├── initial_data_fidelity_loss
│   ├── final_data_fidelity_loss
│   ├── illumination_map
│   ├── illuminated_mask
│   └── settings/...
├── config_yaml
├── metadata/...
└── metrics/...
```

### 关于 truth 和 detector field

`I_stack` 是模拟 CMOS 实际记录的测量量。当前 exp010 的 truth group 保存重建评估所需的 `P_B_true` 和 `B_true`，没有额外保存 81 帧 detector complex field stack。

detector complex field 可由 `P_B_true`、`B_true`、`scan_positions` 和 instrument metadata 重新生成；当前不重复保存可以减少 HDF5 体积。如果后续需要专门分析 detector phase，可以在对应实验脚本中增加 `U_detector_true`，不需要改变统一 HDF5 主结构。

### 关于 `B_rec_aligned_to_truth`

`B_rec` 是算法原始输出。`B_rec_aligned_to_truth` 使用了仿真真值，只为消除不可观测的 constant global phase并计算误差。真实实验没有 truth group，因此不能生成这个 dataset。

## 判读重点

- `probe_relative_error` 是否为 0；
- frozen initial/final data-fidelity loss 是否明显下降；
- loss curve 是否总体收敛；
- truth 与 reconstruction 的 amplitude/phase structure 是否一致；
- illuminated region 内的 complex relative error、amplitude RMSE 和 wrapped phase RMSE 是否合理；
- HDF5 中 `I_stack` 帧数是否与 `scan_positions` 数量一致。
