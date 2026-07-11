# tgv_ptycho_sim

这是一个用于 **TGV（Through Glass Via，玻璃通孔）腰径测量** 的 Python 仿真项目骨架，核心方向是基于 ptychography / PIE 的相位恢复与波传播建模。

目标测量量是孔径随深度的变化：

```text
D(z)
D_waist = min_z D(z)
```

普通二维显微图像通常只能看到入口或出口轮廓，无法直接给出内部最小孔径。因此本项目先从可控仿真开始，逐步验证 probe 形成、ptychographic redundancy、multi-slice propagation、相位恢复以及后续 waist fitting 的可观测性。

## 两条 Pipeline 的关系

本项目不是两个割裂工程，而是一套共享的底层 optics、forward model、reconstruction、IO 和 data format 框架。

1. **样品 A 生成未知 probe，扫描样品 B 提供 ptychographic redundancy**
   - 平行光经过待测 TGV 样品 A。
   - A 在 B 平面形成未知 probe。
   - 可移动编码样品 B 横向扫描，并保持相邻扫描位置有 overlap。
   - CMOS 只记录每个扫描位置的 intensity。
   - 后续用 ePIE / PIE 恢复 B 平面的 probe 和 B 的 complex transmission。
   - 恢复出的 probe 可以 backpropagate 回 A 附近，再结合 TGV forward model 做参数拟合。

2. **Multi-slice 3D TGV forward model**
   - 将 TGV 样品 A 表示为 `n(x, y, z)`，或沿 z 切成多层 `O_m(x, y)`。
   - 光场在层间传播，并逐层被薄层 transmission 调制。
   - 这个模型既可以作为 pipeline 1 中 A 生成 probe 的更真实 forward model，也为后续 3D reconstruction 打基础。

## 环境创建

当前仓库存有 `environment.yml`，创建 conda 环境，需要在终端中运行：

```bash
conda env create -f environment.yml
conda activate tgv_ptycho_sim
pip install -e .
```

pip install -e .是为了：

让在 conda /python 环境的 site-packages 里生成一个软链接（.egg-link 文件）；
系统识别你的项目为一个标准 Python 库；
可以在任意脚本、jupyter、测试文件里直接 import 你的包名，不用管相对路径、不用加 sys.path；
关键是本地改源码，不用重新安装，下一次运行代码直接读取修改后的内容

如果以后修改了 `environment.yml`，可以更新环境：

```bash
conda env update -f environment.yml --prune
conda activate tgv_ptycho_sim
pip install -e .
```

然后可以运行下面的代码，跑一遍tests里面的脚本做最简单的验证，

```bash
pytest
```

## 第一个 Sanity Check

运行最小传播仿真：

```bash
python scripts/run_forward.py --config configs/experiments/exp001_propagation_sanity.yaml
```

这个脚本会：

- 生成 plane wave；
- 经过一个简单 thin phase disk；
- 使用 angular spectrum method 传播一段距离；
- 保存 intensity 图和 complex field 的 amplitude / phase 图；
- 保存 `metadata.json`、`metrics.json` 和内部 HDF5 文件。

输出会写入 `runs/exp001.../`，每个 run 目录建议保持：

```text
config.yaml（实际用的配置）
metadata.json（运行元信息，保存了这次运行的身份信息）
metrics.json（存了一些结果指标）
figures/（存一些可视化结果）
outputs/（机器可以读取的数据，主要是.h5格式）
```

## 第二个实验：已知 probe 的 ePIE

运行 Phase 1 的最小 ptychography reconstruction：

```bash
python scripts/run_recon.py --config configs/experiments/exp010_epie_known_probe.yaml
```

这个实验直接定义 B 平面的 known Gaussian probe，用随机 amplitude-phase 样品 B 做 `9 x 9` overlapping grid scan，生成 `I_stack`，再调用 `epie_reconstruct()` 只恢复 B。probe 在整个 reconstruction 中固定，不参与更新。

输出写入 `runs/exp010_epie_known_probe_时间戳/`，包括：

- `metrics.json`：initial/final loss、B complex relative error、amplitude RMSE、wrapped phase RMSE、照明覆盖率等；
- `figures/known_probe_amp_phase.png`：known probe 的 amplitude 和 phase；
- `figures/detector_frames.png`：代表性的 detector intensity frame，使用 log scale；
- `figures/scan_positions.png`：B 的扫描坐标和 acquisition order；
- `figures/loss_curve.png`：每轮 ePIE 的 relative amplitude loss；
- `figures/B_truth_reconstruction_error.png`：B truth、reconstruction 与 illuminated region error；
- `outputs/epie_known_probe.h5`：测量数据、真值、重建结果、配置、元数据和指标。

算法公式、选择依据、global-phase 对齐和当前限制见 `docs/theory_notes/epie_known_probe.md`。

## 真实实验数据预留接口

项目已经预留真实实验数据处理模块：

```text
src/tgv_ptycho/preprocess/
  dark_flat.py
  normalize.py
  roi.py
  bad_pixels.py

src/tgv_ptycho/calibration/
  camera.py
  stage.py
  geometry.py
  baseline.py
```

当前这些模块只提供函数签名、type hints、docstring 和 TODO，暂不实现真实数据处理逻辑。后续真实数据处理流程大致会是：

```text
raw detector frames
-> dark / flat correction
-> bad pixel correction
-> ROI crop
-> normalization
-> stage / camera / geometry calibration
-> unified HDF5
-> reconstruction
```

## 统一 HDF5 数据原则

仿真数据和真实实验数据最终都应该整理成统一 HDF5 结构，核心字段一致：

- `/entry/data/I_stack`
- `/entry/data/scan_positions`
- `/entry/instrument/...`
- `/entry/reconstruction/...`
- `/entry/config_yaml`
- `/entry/metadata/...`
- `/entry/metrics/...`

区别是：

- 仿真数据可以有 `/entry/truth/...`，例如 `P_B_true`、`B_true`、`A_true` 或 `n_volume`。
- 真实实验数据没有 `/entry/truth`。
- 真实实验数据应包含 `/entry/calibration/...` 和 `/entry/preprocessing/...`。

内部格式说明见：

```text
docs/theory_notes/data_format.md
```

## 数据管理

- `data/raw/` 用于未来实验原始数据，不允许手动覆盖。
- `data/simulated/` 用于可复现的仿真数据。
- `data/processed/` 用于预处理后的派生数据。
- `runs/` 用于每一次运行输出。
- 大文件不要提交到 Git。
- `.gitignore` 已忽略 `runs/*`、`data/raw/*`、`data/simulated/*`、`data/processed/*`，但保留 `.gitkeep`。

## 当前 TODO

- known-probe ePIE 已有可运行的 Phase 1 baseline，但仍不是生产级 reconstruction engine；blind probe update 还需在 Phase 2 验证。
- Multi-slice propagation 已有初版函数，但仍需要和已知 reference 做物理验证。
- TGV 2D model 只是 effective thin phase phantom，不代表真实 waist geometry。
- 真实实验数据的 preprocessing / calibration 目前只预留接口。
- Subpixel scan shift、detector calibration、stage error、CXI / NeXus export 仍待后续加入。

## 其余文件及说明

- `configs`:各种配置，便于重复实验
- `data`：长期数据储存，数据仓库
  data/
├── raw/          # 真实实验原始 CMOS 图像，不要覆盖
├── simulated/   # 大规模仿真生成的数据集
├── processed/   # 预处理后的数据
├── calibration/ # 暗场、平场、相机/位移台标定数据
└── external/    # 下载的公开数据、论文数据、第三方样例数据
- `docs`:存一些文档，当用的知识和笔记
  docs/
├── theory_notes/       # 理论笔记：PIE、角谱法、OCT、HDF5
└── experiment_design/  # 实验设计
- `notebooks`:存的是ipynb文件，方便直接调试和可视化
- `reports`：记录整理可以汇报的文档和图片
- `runs`：每次仿真或重建的输出区，每跑一次实验，就保存一次完整结果，避免结果互相覆盖
- `scripts`：这是可以直接运行的脚本，把src里的内容组织起来，执行某个任务
- `src`：真正计算的核心功能
- `tests`：测试目录，检查代码的低级错误

## HDF5 与 run 目录的关系

`runs/` 目录里通常会同时保存面向人阅读的文件和面向程序读取的文件：

```text
config.yaml
metadata.json
metrics.json
figures/
outputs/*.h5
```

其中 `figures/` 里的 PNG 图片主要用于快速查看结果，不适合作为后续计算的主数据来源。`outputs/*.h5` 是更适合机器读取和长期复现的主数据文件。

从当前版本开始，`run_forward.py` 生成的 HDF5 文件会保存：

- `/entry/data/I_stack`：强度数据；
- `/entry/data/scan_positions`：扫描位置；
- `/entry/instrument/...`：波长、采样间隔、传播距离、探测器像素尺寸等；
- `/entry/sample/...`：样品类型和样品参数；
- `/entry/truth/...`：仿真真值，例如 `incident_probe_true`、`A_true`、`U_after_sample_true`、`U_detector_true`；
- `/entry/config_yaml`：本次运行使用的完整 YAML 配置，对应外部的 `config.yaml`；
- `/entry/metadata/...`：运行元数据，对应外部的 `metadata.json`；
- `/entry/metrics/...`：运行指标，对应外部的 `metrics.json`。

因此，对于当前的 sanity check，HDF5 已经包含除 PNG 可视化图片之外的大部分关键信息，包括 config、运行元数据和 metrics。`metadata.json`、`metrics.json` 仍然保留，是为了方便人直接打开查看；真正做后续批处理或重建时，优先读取 HDF5。

## `.gitignore` 和 `.gitkeep` 说明

这个项目目前还没有初始化 Git 仓库，但已经提前放好了 `.gitignore` 和一些 `.gitkeep` 文件，方便以后直接执行 `git init`。

`.gitignore` 是告诉 Git 哪些文件不要纳入版本管理。比如：

- `runs/*`：忽略每次运行产生的大量结果文件。
- `data/raw/*`、`data/simulated/*`、`data/processed/*`：忽略原始数据、仿真大数据和处理后数据。
- `*.h5`、`*.hdf5`、`*.npy`、`*.npz`、`*.tif`：忽略常见大体积科学数据文件。
- `__pycache__/`、`.pytest_cache/`、`.pytest_tmp/`、`.ruff_cache/`：忽略 Python 和测试工具产生的缓存。
- `.venv/`、`env/`、`build/`、`dist/`、`*.egg-info/`：忽略本地环境、构建产物和安装元数据。

`.gitkeep` 不是 Git 的特殊语法，只是一个约定俗成的空文件。Git 默认不跟踪空目录，所以如果希望保留某个目录结构，例如：

```text
runs/
data/raw/
data/simulated/
data/processed/
data/calibration/
```

就会在目录里放一个 `.gitkeep`。这样 Git 可以记录这个目录本身，但不会记录目录里后续生成的大数据文件。

`.gitignore` 里类似下面的写法：

```text
runs/*
!runs/.gitkeep
```

意思是：忽略 `runs/` 里面的所有运行结果，但不要忽略 `runs/.gitkeep`，从而让空的 `runs/` 目录可以被 Git 保留下来。

当前建议是：代码、配置、文档、测试提交到 Git；大型数据、运行结果、缓存文件不提交。每次实验的完整机器可读结果优先保存在 run 目录中的 HDF5 文件里，PNG 图像主要用于快速查看。
