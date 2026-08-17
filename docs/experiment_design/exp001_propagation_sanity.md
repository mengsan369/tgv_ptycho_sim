# exp001：Angular Spectrum Propagation Sanity Check

## 实验定位

`exp001_propagation_sanity` 是项目的 Phase 0 实验，用来验证最基础的光场生成、二维样品调制、angular spectrum propagation、可视化和 HDF5 保存流程。

这个实验不包含 ptychographic scan，也不执行 ePIE reconstruction。它的作用是先确认底层传播模型和数据管理流程能够正常工作，为后续实验提供可靠基线。

## 启动方式

实验配置文件：

```text
configs/experiments/exp001_propagation_sanity.yaml
```

运行脚本：

```text
scripts/run_forward.py
```

在项目根目录执行：

```bash
python scripts/run_forward.py --config configs/experiments/exp001_propagation_sanity.yaml
```

VS Code 中也可以使用 `.vscode/launch.json` 里的 `run_forward exp001` 配置启动调试。

## 实验流程

默认配置执行以下步骤：

1. 在样品平面生成 plane wave `U0`。
2. 生成二维 thin phase disk，并作为样品 A 的 complex transmission。
3. 得到样品后的复光场：

   ```text
   U_after_sample = U0 * A_true
   ```

4. 使用 angular spectrum method 传播到 detector 平面：

   ```text
   U_detector = propagate(U_after_sample, z)
   ```

5. 计算 detector intensity：

   ```text
   I_detector = |U_detector|^2
   ```

6. 保存配置、元数据、数值指标、PNG 图片和 HDF5 文件。

当前配置使用 `256 x 256` 数组、`2 um` sampling、`532 nm` wavelength 和 `20 mm` propagation distance。所有核心物理量在程序和 HDF5 中使用 SI 单位。

## Run 目录输出

每次运行都会创建一个带时间戳的新目录：

```text
runs/exp001_propagation_sanity_YYYYMMDD_HHMMSS/
├── config.yaml
├── metadata.json
├── metrics.json
├── figures/
│   ├── intensity.png
│   └── propagated_field_amp_phase.png
└── outputs/
    └── propagation_sanity.h5
```

这些文件的用途如下：

- `config.yaml`：本次实际使用的完整配置副本。
- `metadata.json`：run name、创建时间、Git commit、shape、波长、sampling、传播距离和样品类型等运行身份信息。
- `metrics.json`：输入/输出 energy、energy ratio，以及 intensity 的最小值、最大值和平均值。
- `figures/`：供人快速检查传播结果。
- `outputs/propagation_sanity.h5`：供程序读取、复现和后续处理的主数据文件。

## 图片说明

### `intensity.png`

这张图显示 detector 平面的 intensity：

```text
I_detector = |U_detector|^2
```

横纵坐标为 `x`、`y`，单位为 `um`；colorbar 表示 intensity，单位为 arbitrary units。它对应 HDF5 中：

```text
/entry/data/I_stack[0]
/entry/truth/I_detector_true
```

### `propagated_field_amp_phase.png`

这是一张左右排列的 detector complex field 图：

- 左图：`|U_detector|`，即 amplitude；
- 右图：`angle(U_detector)`，即 wrapped phase，单位为 rad。

它对应 HDF5 中：

```text
/entry/truth/U_detector_true
```

## HDF5 数据内容

`outputs/propagation_sanity.h5` 的主要结构为：

```text
/entry
├── data
│   ├── I_stack                         # shape: (1, ny, nx)
│   └── scan_positions                  # shape: (1, 2), value: (0, 0)
├── instrument
│   ├── wavelength                      # m
│   ├── dx                              # m
│   ├── z_AB                            # 0 m
│   ├── z_BC                            # propagation distance, m
│   └── detector_pixel_size             # m
├── sample
│   ├── sample_A_type
│   ├── sample_B_type                   # none
│   └── tgv_parameters
├── truth
│   ├── incident_probe_true             # U0
│   ├── A_true                          # sample transmission
│   ├── U_after_sample_true
│   ├── P_B_true                        # current compatibility alias
│   ├── U_detector_true
│   └── I_detector_true
├── config_yaml                         # complete YAML text
├── metadata/...
└── metrics/...
```

`I_stack` 只有一帧，因为 exp001 没有扫描。`scan_positions` 仍然保留，是为了让该文件与后续 ptychography 数据使用同一套基础 layout。

这个实验不产生 `/entry/reconstruction`，因为没有执行 phase retrieval 或 inverse reconstruction。

## 判读重点

- 输出数组 shape 是否与输入一致；
- intensity 是否有限且非负；
- input/output energy ratio 是否接近角谱传播的预期；
- amplitude、phase 和 diffraction pattern 是否具有合理的空间结构；
- JSON 与 HDF5 中的 metadata、metrics 是否一致。
