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
scripts/run_exp010_recon.py
```

在项目根目录执行：

```bash
python scripts/run_exp010_recon.py --config configs/experiments/exp010_epie_known_probe.yaml
```

VS Code 中也可以使用 `.vscode/launch.json` 里的 `run_exp010_recon exp010 known probe ePIE` 配置启动调试。

## 实验流程

默认配置执行以下步骤：

1. 在 B 平面生成已知 Gaussian probe `P_B_true`。注意Gaussian 只是 Phase 1 中用于代替“由 A 形成的有限尺寸 probe”的简单模型。Phase 2 开始，P_B 应由 A 的 transmission 或 multi-slice forward model生成，不一定还是 Gaussian。
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

docs/theory_notes/epie_known_probe.md

## 计算窗口、位移与 detector sampling

### 计算窗口

真实的横向光场定义在连续平面上，理论范围可以无限延伸；数值计算只能保存其中一块有限区域。这个有限区域称为 computational window。

当前 exp010 使用：

```text
shape = 96 x 96
dx = 2 um
window width = 96 x 2 um = 192 um
```

因此当前 B 平面的数值窗口约为 `192 x 192 um`。`dx` 是相邻 sample 的物理间隔，`shape` 是 sample 数量，两者共同确定计算窗口的物理范围。

在同一个薄样品平面上做逐点 transmission modulation 时，入射场、样品 transmission 和调制后光场必须定义在同一个计算窗口：

```text
U_before_sample.shape
sample_transmission.shape
U_after_sample.shape
```

三者都等于 computational window 的 shape。exp010 中的样品是 B，所以：

```text
U_before_sample = P_B_true
sample_transmission = shifted(B_true, r_j)
U_after_sample = psi_j

psi_j = P_B_true * shifted(B_true, r_j)
psi_j.shape = U_after_sample.shape = (96, 96)
```

也就是说，`U_after_sample` 的数组尺寸就是当前计算窗口的数组尺寸。

计算窗口不要求等于 B 的真实物理尺寸。若随机编码 active area 小于计算窗口，应将它嵌入完整的 transmission 数组，并定义外围背景：透明且未调制的背景可归一化为 `1`；exit wave 外围补 `0` 则表示该处没有光或光场已小到可以忽略。“乘 1”和“补 0”具有不同物理含义。

更一般的 patch-based ptychography 可以保存较大的 `B_global`，每个扫描位置从中提取局部 `B_patch` 与固定 probe 相乘，再把 exit wave 放入足够大的 propagation window。局部 patch 边界应位于 probe 已经足够弱的位置，避免人为截断光场。

### 位移定义

当前 computational window 固定在 laboratory frame 和 optical axis 上，Gaussian probe 与 detector 中心固定；扫描时移动的是 B 在这个固定窗口中的内容。scan position 的列顺序是 `(x, y)`，单位为 m。

```text
scan_position = (0, 0)
```

表示 B 不发生数组位移。按当前坐标约定，B 的数值原点、Gaussian probe 中心、detector 中心和 optical axis 在横向上重合。真实实验中的 `(0,0)` 需要通过 stage zero、probe center 和 detector center 标定，并不是设备天然保证的绝对位置。

当前 `step=12 um`、`dx=2 um`，所以一步对应：

```text
pixel shift = 12 um / 2 um = 6 pixels
```

正 `x` 位置将 B 数组沿正 x 方向移动，正 `y` 位置将 B 数组沿正 y 方向移动。目前只支持整数 pixel shift。

当前位移使用 `np.roll()`，因此 B 数组具有 periodic boundary condition：从 B 数值窗口右边移出的内容会从同一窗口左边进入，等价于假设随机 B tile 在横向上无限周期重复。

```text
... | B tile | B tile | B tile | ...
```

这里的“移出”是移出 B 平面的数值计算窗口，不是移出 detector C 的感光区域。这个周期假设只用于 Phase 1 验证传播、overlap、位移和 ePIE update，不代表有限尺寸 B 的真实边缘。

### 传播后的 `U_z` 与相机 pixel

理想传播后的复光场 `U_z(x,y)` 定义在整个 detector 平面上，衍射通常会使它扩展到很大范围。数值程序仍然只能计算其中一块有限 propagation window；真实相机又只接收其有限 sensor area 内的光。

当前 angular spectrum implementation 保持输入和输出 shape、sampling 不变：

```text
U_after_sample.shape = (96, 96)
U_z.shape = (96, 96)

dx_B = 2 um
dx_z = 2 um
```

当前又设置 `detector_pixel_size_m=2 um`，所以 exp010 直接把每个 `U_z` sample 当作一个理想 detector sample：

```text
I_stack[j] = |U_z|^2
```

因此当前数值 detector 也是 `96 x 96`，窗口约为 `192 x 192 um`。这是 Phase 1 的简化，并不表示一般情况下 `U_z.shape` 必须等于真实相机 shape，或 `dx_z` 必须等于真实 detector pixel pitch。

真实 CMOS 不直接测量复数 `U_z`，也不获得其 phase。第 `(m,n)` 个 pixel 记录的是该 pixel 有限面积内 intensity 的积分，实际 counts 还与曝光时间、响应度、gain 和噪声有关：

```text
I_camera[m,n] proportional to
    integral_over_pixel(m,n)(|U_z(x,y)|^2 dx dy)
```

相机不是把整个 sensor area 积分成一个数，而是每个 pixel 分别积分。sensor area 外的传播光场不会被相机记录；所有 pixel 的值相加才对应相机接收到的总能量近似。

更真实的 detector model 应先在足够细、足够大的 propagation grid 上计算 `U_z`，再按照真实 sensor size 和 pixel pitch 执行：

```text
crop to sensor area
-> pixel-area integration / binning
-> responsivity and exposure
-> noise, saturation and quantization
```

此时可以有：

```text
U_z.shape != camera.shape
dx_z != detector pixel pitch
```

当前 HDF5 已记录 `detector_pixel_size`，但 exp010 还没有实现独立 propagation grid 到真实 camera grid 的映射。

当前数值空间可概括为：

```text
B plane, z = 0
192 x 192 um computational window
P_B_true * shifted(B_true) -> U_after_sample

             propagate z_BC = 10 mm

C plane, z = 10 mm
192 x 192 um ideal detector window
U_z -> |U_z|^2
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
