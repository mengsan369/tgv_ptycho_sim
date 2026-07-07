# 内部 HDF5 数据格式

本项目内部优先使用 HDF5 保存仿真和未来真实实验 ptychography 数据。格式设计参考 CXI / NeXus ptychography 的思想，但当前阶段不要求完全符合 `CXI`、`NXcxi` 或 `NXptycho` 标准。

## 共同核心结构

仿真数据和真实实验数据都应该包含以下核心信息：

```text
/entry/data/I_stack
/entry/data/scan_positions
/entry/instrument/wavelength
/entry/instrument/dx
/entry/instrument/z_AB
/entry/instrument/z_BC
/entry/instrument/detector_pixel_size
/entry/sample/sample_A_type
/entry/sample/sample_B_type
/entry/sample/tgv_parameters
/entry/reconstruction/P_B_rec
/entry/reconstruction/B_rec
/entry/reconstruction/loss_curve
/entry/config_yaml
/entry/metadata/git_commit
/entry/metadata/created_at
/entry/metrics
```

其中：

- `I_stack` 是 detector intensity stack，shape 通常为 `(num_positions, ny, nx)`。
- `scan_positions` 使用 SI 单位，列顺序约定为 `(x, y)`，单位为 m。
- `instrument` 存放 wavelength、sampling、propagation distance、detector pixel size 等信息。
- `sample` 存放样品 A / B 的类型和 TGV 参数。
- `reconstruction` 存放恢复结果和 loss curve。
- `config_yaml` 存放本次运行使用的完整 YAML 配置，和外部 run 目录中的 `config.yaml` 对应。
- `metadata` 存放 git commit、created time、run name 等运行元信息，和外部 run 目录中的 `metadata.json` 对应。
- `metrics` 存放运行指标，和外部 run 目录中的 `metrics.json` 对应。

## 仿真数据

仿真数据可以额外包含 ground truth：

```text
/entry/truth/P_B_true
/entry/truth/B_true
/entry/truth/A_true
/entry/truth/n_volume
```

不是每个仿真都必须包含全部 truth 字段。例如 2D thin phase 仿真可能只有 `A_true` 和 `P_B_true`；multi-slice TGV forward model 可能包含 `n_volume`。

## 真实实验数据

真实实验数据不应该包含 `/entry/truth`，因为真实样品和 probe 没有 ground truth。真实数据应该包含 calibration 和 preprocessing 信息：

```text
/entry/calibration/camera
/entry/calibration/stage
/entry/calibration/geometry
/entry/calibration/baseline
/entry/preprocessing/dark_flat
/entry/preprocessing/normalize
/entry/preprocessing/roi
/entry/preprocessing/bad_pixels
```

这些 group 用于记录：

- camera pixel size、gain、offset、dark frame、flat field、bad pixel mask；
- stage log、scan coordinate convention、position refinement；
- detector distance、sample-plane sampling、propagation geometry；
- dark subtraction、flat-field correction、ROI crop、normalization 等 preprocessing 步骤。

## 与 CXI / NeXus 的关系

当前内部格式是“CXI / NeXus-inspired layout”，目标是先保证关键字段完整、层级清晰、可被本项目稳定读写。后续可以增加导出接口：

```python
export_to_cxi(...)
export_to_nexus_ptycho(...)
```

在正式兼容标准前，项目内部代码应优先写入上述 HDF5 layout，避免为主要数据路径使用零散的 `.npy`、`.npz` 或临时二进制文件。
