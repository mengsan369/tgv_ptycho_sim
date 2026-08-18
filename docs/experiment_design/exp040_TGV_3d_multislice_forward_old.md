# exp040：3D TGV multi-slice forward model

```text
Status: Inconclusive
Phase: Phase 4
Owner: tgv_ptycho_sim team
Created: 2026-08-10
Last updated: 2026-08-10
```

本文档的第 1--15 节是执行前预注册；所有阈值、对照组、状态逻辑和主配置均在
查看 exp040 结果之前固定。第 16 节以后已使用正式 run 的实际结果更新。由于
convergence 与 detector visibility gate 未通过，当前仍不得据此将 Phase 4 标记为
已实现，也不得放宽预注册阈值。

## 1. 实验编号与名称

- 实验编号：`exp040`
- 实验名称：`3D TGV multi-slice forward model`
- 所属 pipeline：`3D multi-slice / A-generated probe + B scan / shared infrastructure`
- 对应 Codex 任务：`[tgv_ptycho_sim] exp040 - 3D TGV multi-slice forward model`

本实验只验证三维 sample A forward model 及其到 B、detector 的完整正向链路，
不执行 reconstruction，也不从 intensity 反演腰径。

## 2. 所属 Phase

- Phase：`Phase 4`
- 路线图位置：`docs/theory_notes/roadmap.md`
- 当前状态：`公共 forward baseline 已实现并运行；预注册验证结果 Inconclusive`

只有实验文档、理论文档、配置、公共实现、运行入口、测试、正式 timestamped
run 和实际结果全部完成并通过预注册标准后，才能更新路线图状态。

## 3. 目的与假设

### 目的

回答一个问题：在预注册的理想标量条件下，使用真实 slice width、voxel-center
binary TGV volume 和 centered symmetric split-step，能否得到通过几何、代数、
轴向、横向及 FOV 数值验证的 sample A 出射场，并使预注册腰径扰动在 detector
intensity 中稳定高于数值离散 floor？

### 假设

1. 分段线性的轴对称空气孔可由 `(z, y, x)` 顺序的 voxel-center binary
   refractive-index volume 表示。
2. 对每层使用真实宽度的 centered symmetric split-step，在当前 sampling 下能随
   `dz` 加密收敛，并在 zero-contrast 情况回到相同厚度和出射平面的参考介质传播。
3. 固定物理 FOV 加密 lateral sampling、固定 sampling 扩大 FOV 后，
   `U_A_exit`、`P_B` 和 `I_stack` 的最终 refinement 相对变化均不高于 5%。
4. `D_waist ± 2 um` 的 detector stack 变化至少为 detector 数值离散 floor 的
   3 倍。该扰动仅用于无噪声数值可见性检查，不是检测限。

### 适用边界

当前模型是 scalar、monochromatic、unidirectional、normal-incidence、无吸收的
理想 baseline。它不含 Fresnel interface reflection、backward wave、multiple
reflection/scattering、vector polarization、surface roughness、tilt、偏心、非圆孔
或多个 TGV。玻璃与空气的大折射率差可能超出 thin phase screen 与单向标量近似的
高精度适用区间。因此，即使 exp040 通过，也只能说明当前 multi-slice 数值模型经过
所列控制，不能说明真实三维电磁场或真实硬件中的腰径已经可测。

## 4. 理论依据

真空波长为 `lambda_0`，参考介质折射率为 `n_ref`：

$$
k_0=\frac{2\pi}{\lambda_0},
\qquad
T_j(x,y)=\exp\!\left[
i k_0\left(n_j(x,y)-n_\mathrm{ref}\right)w_j
\right],
$$

其中 $w_j$ 是第 $j$ 层的真实宽度，必须满足

$$
\sum_j w_j=L.
$$

第 $j$ 层位于其 slice center。预注册算子是 centered symmetric split-step：

$$
U_{j+1}
=
\mathcal P_{w_j/2}^{(n_\mathrm{ref})}
\left\{
T_j
\mathcal P_{w_j/2}^{(n_\mathrm{ref})}[U_j]
\right\}.
$$

因此输入位于 sample A entrance boundary，返回的 `U_A_exit` 位于 sample A exit
boundary。单层控制严格使用同一公式；zero-contrast 时所有 $T_j=1$，输出应等于
$\mathcal P_L^{(n_\mathrm{ref})}[U_0]$，不得通过 truth global-phase alignment
掩盖传播距离或物理平面错误。angular-spectrum kernel 保留
$\exp(i k_{z,\mathrm{ref}}z)$ 的参考介质相位。

关闭内部传播时，relative phase-screen product 为

$$
U_{\mathrm{product}}
=U_0\prod_jT_j,
$$

并与使用同一 voxel-center binary masks 和真实 $w_j$ 的离散 projected-phase
product 比较。该代数比较与连续 projected-phase 极限的 sampling convergence 是
两个不同问题。

完整 forward 为

$$
P_B=\mathcal H_{z_{AB}}[U_{A,\mathrm{exit}}],
$$

$$
\Psi_q=\mathcal H_{z_{BC}}[P_B B_q],
\qquad
I_q=|\Psi_q|^2.
$$

`z_AB` 从 sample A exit boundary 开始，不重复计算 sample thickness。scan position
列顺序为 `(x, y)`，单位为 m；二维数组顺序为 `(ny, nx)`，volume 顺序为
`(nz, ny, nx)`；若 `dx` 为 tuple，其顺序为 `(dy, dx)`。

- 理论文档：`docs/theory_notes/tgv_multislice_forward.md`
- projected-phase 对照：`docs/theory_notes/tgv_effective_phase_2d.md`
- 通用数据约定：`docs/theory_notes/data_format.md`
- 参考文献：`N/A；本实验使用上述项目理论文档中登记的标量 split-step/ASM 公式`

## 5. 依赖实验

| 依赖实验 | 使用内容 | 文档 | 基线 run |
|---|---|---|---|
| `exp001` | angular-spectrum propagation 与 forward/HDF5 基线 | `docs/experiment_design/exp001_propagation_sanity.md` | `N/A（不复制历史 run）` |
| `exp020` | A-exit 到 B、B scan 到 detector 的共享语义 | `docs/experiment_design/exp020_A_thin_phase_probe_recovery.md` | `N/A（只复用公共接口）` |
| `exp030` | 公共 `D(z)`、projected-phase product 与 sensitivity 指标 | `docs/experiment_design/exp030_TGV_2d_effective_phase.md` | `N/A（exp040 独立运行）` |

exp030 的二维结果不能作为 exp040 的通过结论；它只提供 projected-phase 极限对照。

## 6. 启动脚本

- 脚本：`scripts/run_exp040_multislice_forward.py`
- 工作目录：项目根目录
- 命令：

  ```powershell
  python scripts/run_exp040_multislice_forward.py --config configs/experiments/exp040_TGV_3d_multislice_forward.yaml
  ```

- VS Code debug 配置：`N/A（尚未创建）`
- 当前运行状态：`已运行；正式判定 Inconclusive`

## 7. 配置文件

- 主配置：`configs/experiments/exp040_TGV_3d_multislice_forward.yaml`
- 复用配置：`N/A；公共算法与数据约定从 src/docs 复用`
- 随机种子：sample B `20260840`；scan jitter `20260841`；plane wave 无随机过程，
  seed 为 `null`
- config 对应基准 Git commit：`23d702b5d430b15f594462a9d9973569e66bff24`

配置中的 `experiment.status=Planned` 是冻结在启动输入中的预注册状态，不是运行
结果。正式 metadata 分别保存 `config_status_at_launch=Planned` 和
`experiment_status=Inconclusive`，避免混淆输入登记状态与结果判定。

主配置同时保存对照组、comparison mapping、阈值与状态逻辑，运行时不得静默替换。

## 8. 数据流

```text
plane-wave incident field
-> shared D(z) and voxel-center binary n(z,y,x)
-> centered symmetric split-step through sample A
-> U_A_exit at the physical exit boundary
-> external-medium propagation over z_AB
-> P_B and canonical physical sample B realization
-> 5 x 5 overlapping integer-pixel scan
-> external-medium propagation over z_BC
-> grid-sampled detector I_stack
-> controls, convergence, visibility metrics, figures and HDF5
```

明确约定：

- baseline `n_volume`：`(100, 128, 128)`, `float64`, axis `(z,y,x)`；
- baseline complex fields：`(128,128)`, `complex128`, axis `(y,x)`；
- baseline `I_stack`：`(25,128,128)`, `float64`, axis `(scan,y,x)`；
- scan positions：`(25,2)`, `float64`，列 `(x,y)`，单位 m；
- length 使用 m，phase 使用 rad，intensity 使用 arbitrary units；
- detector 是 grid-sampled intensity，不执行 detector pixel area integration；
- 不保存所有 slice fields。

不同 lateral grids 的 convergence 只通过配置中预注册的 centered bilinear
interpolation 映射到公共 sample centers。该映射只用于数值比较，不属于 detector
forward model，也不建立项目级 detector sampling-remap 标准。FOV cases 固定
`dx=0.5 um`，仅裁出共同中心 `64 um x 64 um` ROI 比较。

sample B 先在 `96 um x 96 um`、`dx=0.25 um`、`384 x 384` canonical grid 上用
seed `20260840` 生成。所有 case 只从这个 realization 做 centered crop 和对齐的
point sampling；不为每个 grid 重新抽取随机对象。其 physical feature size 固定为
`2 um`。

## 9. 关键参数

| 参数 | 值 | 单位 | 作用 | 来源 |
|---|---:|---|---|---|
| vacuum wavelength | `532e-9` | m | $k_0$ 和传播 kernel | 主配置 |
| internal `n_ref` | `1.5` | 1 | sample A split-step 参考介质 | 预注册 |
| external medium index | `1.0` | 1 | A-exit 到 B、B 到 detector | 预注册 |
| baseline shape | `128 x 128` | pixel | baseline lateral grid | 预注册 |
| baseline `dx` | `0.5e-6` | m | baseline lateral sampling | 预注册 |
| thickness | `100e-6` | m | sample A 物理厚度 | 预注册 |
| target `dz` | `1e-6` | m | baseline slice target width | 预注册 |
| `D_top` | `30e-6` | m | entrance diameter | 预注册 |
| `D_waist` | `20e-6` | m | baseline waist diameter | 预注册 |
| `D_bottom` | `30e-6` | m | exit diameter | 预注册 |
| `z_waist` | `50e-6` | m | waist depth from A entrance | 预注册 |
| `n_glass` / `n_air` | `1.5 / 1.0` | 1 | binary volume index values | 预注册 |
| `z_AB` | `0.5e-3` | m | A exit to B propagation | 预注册 |
| `z_BC` | `1e-3` | m | B to detector propagation | 预注册 |
| detector pixel metadata | `0.5e-6` | m | baseline grid-sampling pitch | 预注册 |
| sample B phase range | `0.8` | rad | phase-only encoding strength | 预注册 |
| sample B feature size | `2e-6` | m | grid-independent physical feature size | 预注册 |
| scan | `5 x 5` | position | overlapping scan | 预注册 |
| scan step | `4e-6` | m | regular-grid spacing | 预注册 |
| jitter | max `1` at quantum `1e-6` | pixel, m | 在所有收敛网格上均为整数像素的 seeded perturbation | 预注册 |
| waist perturbation | `±2e-6` | m | forward-only visibility | 预注册 |

ASM `bandlimit=true`。internal 与 external reference indices、sample material indices
必须分别写入 metadata/HDF5，不得用一个含糊的 `medium_index` 代替。

## 10. 对照组与扫描条件

| 组别 | 改变量 | 取值 | 固定条件 | 目的 |
|---|---|---|---|---|
| Baseline | model/grid | `128²`, `dx=0.5 um`, `dz=1 um` | 主几何、B、scan、传播距离 | 生成主数据 |
| Geometry | volume/slices | baseline | 无传播 | shape、dtype、`D(z)`、centers、widths、厚度 |
| Zero contrast | `n_volume` | `n_ref` everywhere | 同 width 和 exit plane | 对比 $P_L(U_0)$ |
| Single slice | one width | 单层受控小问题 | 同 symmetric operator | 验证算子顺序 |
| No internal propagation | propagation switch | off | 同 binary slices/widths | 对比 phase product |
| Projected product | discrete product | exp030-compatible | 同 voxel centers/widths | 代数一致性 |
| Axial | `dz` | `2, 1, 0.5 um` | `128²`, `dx=0.5 um` | 最终 `1 vs 0.5 um` convergence |
| Lateral | shape/`dx` | `64²@1`, `128²@0.5`, `256²@0.25 um` | FOV=`64 um`, target `dz=1 um` | 最终 `0.5 vs 0.25 um` convergence |
| FOV | shape | `128²,160²,192²` | `dx=0.5 um` | periodic wrap/FOV convergence |
| End-to-end | output plane | A exit/B/detector | 同一物理 B 与 scan | 检查 forward 稳定性 |
| Waist visibility | `D_waist` | `18,20,22 um` | incident/B/scan/detector/seeds 相同 | signal 与 floor 比较 |
| Determinism/failure | repeat/config errors | identical/invalid | 同环境 | repeatability 与 finite |

所有 convergence cases 都使用同一 incident field、同一个 canonical sample B
realization、相同物理 scan positions 和相同 detector 定义。FOV convergence 的共同
中心 ROI 是 `64 um x 64 um`，即 baseline 的 `128 x 128` 区域。

## 11. 输出 run 结构

```text
runs/exp040_TGV_3d_multislice_forward_YYYYMMDD_HHMMSS/
├── config.yaml
├── metadata.json
├── metrics.json
├── figures/
│   ├── tgv_geometry_and_index_slices.png
│   ├── exit_field_multislice.png
│   ├── projected_limit_comparison.png
│   ├── dz_convergence.png
│   ├── lateral_fov_convergence.png
│   ├── B_plane_probe.png
│   ├── detector_intensity_baseline.png
│   └── detector_visibility.png
└── outputs/
    └── exp040_multislice_forward.h5
```

正式 run 路径：
`runs/exp040_TGV_3d_multislice_forward_20260810_154908/`

## 12. HDF5 结构

通用规则见 `docs/theory_notes/data_format.md`。本实验是 simulation，可以保存
truth；它是 forward-only 实验，不创建 `/entry/reconstruction`。没有实际 calibration
或 preprocessing，也不创建空 group。

```text
/entry/data/I_stack
/entry/data/scan_positions
/entry/instrument/...
/entry/sample/...
/entry/truth/...
/entry/truth/parameter_sweep/...
/entry/config_yaml
/entry/metadata/...
/entry/metrics/...
```

| Dataset | Shape | dtype | 单位 | 语义/来源 |
|---|---|---|---|---|
| `/entry/data/I_stack` | `(25,128,128)` | `float64` | a.u. | baseline grid-sampled detector intensity |
| `/entry/data/scan_positions` | `(25,2)` | `float64` | m | columns `(x,y)` |
| `/entry/instrument/wavelength` | scalar | `float64` | m | vacuum wavelength |
| `/entry/instrument/dx` | scalar | `float64` | m | baseline lateral sampling |
| `/entry/instrument/z_AB` | scalar | `float64` | m | A exit to B distance |
| `/entry/instrument/z_BC` | scalar | `float64` | m | B to detector distance |
| `/entry/instrument/detector_pixel_size` | scalar | `float64` | m | baseline grid pitch metadata |
| `/entry/instrument/internal_reference_index` | scalar | `float64` | 1 | sample A split-step reference index |
| `/entry/instrument/external_medium_index` | scalar | `float64` | 1 | external propagation index |
| `/entry/sample/sample_A_type` | scalar string | UTF-8 | 1 | axisymmetric air-filled TGV volume |
| `/entry/sample/tgv_parameters/...` | nested scalars | numeric/string | SI | geometry, materials, voxelization |
| `/entry/sample/sample_B_type` | scalar string | UTF-8 | 1 | canonical random phase object |
| `/entry/sample/sample_B_parameters/...` | nested scalars | numeric/string | SI/rad | physical feature, seed and mapping |
| `/entry/truth/n_volume` | `(100,128,128)` | `float64` | 1 | baseline refractive-index volume `(z,y,x)` |
| `/entry/truth/z_m` | `(100,)` | `float64` | m | baseline slice centers |
| `/entry/truth/slice_thickness_m` | `(100,)` | `float64` | m | true slice widths |
| `/entry/truth/diameter_z_m` | `(100,)` | `float64` | m | shared $D(z)$ at slice centers |
| `/entry/truth/incident_field_true` | `(128,128)` | `complex128` | field a.u. | entrance-boundary plane wave |
| `/entry/truth/U_A_exit_true` | `(128,128)` | `complex128` | field a.u. | baseline field at A exit boundary |
| `/entry/truth/P_B_true` | `(128,128)` | `complex128` | field a.u. | baseline B-plane probe |
| `/entry/truth/B_true` | `(128,128)` | `complex128` | transmission | baseline crop/sample of canonical B |
| `/entry/truth/parameter_sweep/d_waist_m` | `(3,)` | `float64` | m | `[18,20,22] um` |
| `/entry/truth/parameter_sweep/U_A_exit_true` | `(3,128,128)` | `complex128` | field a.u. | minus/baseline/plus A-exit fields |
| `/entry/truth/parameter_sweep/P_B_true` | `(3,128,128)` | `complex128` | field a.u. | minus/baseline/plus B probes |
| `/entry/truth/parameter_sweep/I_stack_true` | `(3,25,128,128)` | `float64` | a.u. | minus/baseline/plus detector stacks |
| `/entry/config_yaml` | scalar string | UTF-8 | 1 | executed complete config |
| `/entry/metadata/...` | nested | mixed | mixed | run identity, planes, axes, limitations |
| `/entry/metrics/...` | nested | `float64/bool/string` | mixed | controls, convergence, visibility and status |

`/entry/data/I_stack` 必须与 parameter sweep 的 baseline case 数值一致。所有
truth 只用于 simulation evaluation；本实验没有 reconstruction，因而不存在 truth
泄漏或 truth-aided alignment。大体积 convergence volumes 和所有 slice fields 不写入
HDF5；相应数值以 metrics 保存。

## 13. 图片及物理含义

| 图片 | 显示量 | 横纵坐标 | colorbar | 对应 HDF5 dataset | 判读目的 |
|---|---|---|---|---|---|
| `tgv_geometry_and_index_slices.png` | $D(z)$、central x-z index、代表性 x-y index slices | `z/x/y`, um | refractive index, 1 | `n_volume`, `z_m`, `diameter_z_m` | 检查腰部、边界 staircase 和材料值 |
| `exit_field_multislice.png` | `|U_A_exit|`, wrapped phase, intensity | `x,y`, um | amplitude / rad / a.u. | `U_A_exit_true` | 检查 A-exit field |
| `projected_limit_comparison.png` | no-propagation product、projected product、complex difference | `x,y`, um | phase rad / abs error | metrics/control source and `U_A_exit_true` context | 代数极限控制 |
| `dz_convergence.png` | `U_A_exit/P_B/I_stack` relative changes vs dz | `dz`, um | N/A | `/entry/metrics/convergence/axial/...` | 检查最终 axial refinement |
| `lateral_fov_convergence.png` | lateral 与 FOV 三个输出域的 relative changes | `dx` 或 FOV, um | N/A | `/entry/metrics/convergence/...` | 区分 sampling 与 wrap/FOV floor |
| `B_plane_probe.png` | baseline `P_B` amplitude、phase、intensity | `x,y`, um | amplitude / rad / a.u. | `P_B_true` | 检查 sample A 到 B 传播 |
| `detector_intensity_baseline.png` | 代表性 baseline scan frames | `x,y`, um | intensity, a.u. | `data/I_stack` | 检查 detector pattern 与 scan 差异 |
| `detector_visibility.png` | ± waist difference maps、per-frame relative changes、floor | `x,y` um / frame index | relative intensity / ratio | parameter sweep and visibility metrics | 判断 detector signal/floor gate |

相位图使用 wrapped phase，并明确单位 rad；复场 convergence 使用未对齐的 complex
field。所有图片需有标题、坐标标签和 colorbar；图片只供人工检查，后续计算读取
HDF5/metrics。

## 14. Metrics

定义同 shape 或已映射到公共网格的相对 L2：

$$
\varepsilon_X
=\frac{\|X_\mathrm{test}-X_\mathrm{ref}\|_2}
{\max(\|X_\mathrm{ref}\|_2,\epsilon_\mathrm{float64})}.
$$

| Metric | 定义 | 计算区域/对齐 | 单位 | 验收阈值 |
|---|---|---|---|---|
| `slice_width_sum_abs_error_m` | $|\sum_jw_j-L|$ | 全部 layers | m | `<= max(1e-15,16 eps L)` |
| `geometry_values_valid` | shape/dtype/index/centers/widths/$D(z)$ | full volume | bool | `True` |
| `zero_contrast_relative_l2` | multislice vs $P_L(U_0)$ | full field，无 phase alignment | 1 | `<=1e-12` |
| `single_slice_relative_l2` | solver vs registered symmetric formula | full field，无 alignment | 1 | `<=1e-12` |
| `no_propagation_product_relative_l2` | solver-off vs explicit $\prod T_j$ | full field | 1 | `<=1e-12` |
| `projected_phase_product_relative_l2` | discrete product vs exp030-compatible discrete product | full field | 1 | `<=1e-12` |
| `determinism_relative_l2` | identical-config repeated output | all saved numerical outputs | 1 | `<=1e-14` |
| `all_outputs_finite` | no NaN/Inf, intensity nonnegative | all cases | bool | `True` |
| `dz_*_relative_l2` | final `1 vs 0.5 um` change | `U_A_exit/P_B/I_stack` | 1 | each `<=0.05` |
| `lateral_*_relative_l2` | final `0.5 vs 0.25 um` change | mapped common grid | 1 | each `<=0.05` |
| `fov_*_relative_l2` | final `160 vs 192` shape change | common center 64 um ROI | 1 | each `<=0.05` |
| `*_waist_minus_signal` | $\|X(D_-)-X(D_0)\|/\|X(D_0)\|$ | same physical grid | 1 | report |
| `*_waist_plus_signal` | $\|X(D_+)-X(D_0)\|/\|X(D_0)\|$ | same physical grid | 1 | report |
| `*_discretization_floor` | max of final dz/lateral/FOV changes | corresponding output | 1 | report |
| `U_A_exit_signal_to_floor_min` | min(± signal)/A-exit floor | baseline domain | 1 | report only |
| `P_B_signal_to_floor_min` | min(± signal)/probe floor | baseline domain | 1 | report only |
| `detector_signal_to_floor_min` | min(± signal)/detector floor | whole stack | 1 | `>=3` visibility gate |
| `detector_per_frame_relative_change_*` | each frame ± relative L2 | per scan frame | 1 | report min/median/max |

detector floor 明确定义为

$$
F_I=\max(\varepsilon_{I,dz},\varepsilon_{I,lateral},\varepsilon_{I,FOV}),
$$

visibility gate 为

$$
R_I=
\frac{\min(\varepsilon_I^{(-)},\varepsilon_I^{(+)})}
{\max(F_I,\epsilon_\mathrm{float64})}\ge3.
$$

`U_A_exit` 和 `P_B` 使用相同形式计算 signal/floor，但只报告，不作为 visibility
gate。不得以 global-phase、affine phase 或 complex-gain truth alignment 改善任何硬
控制或 convergence metric。intensity 不涉及 phase wrapping；phase 图仅作显示。

## 15. 验收标准

- [x] volume shape、dtype、index values、`D(z)`、slice centers/widths 正确，且
  $|\sum_jw_j-L|\le\max(10^{-15}\,\mathrm m,16\epsilon L)$
- [x] zero-contrast、single-slice、no-propagation 和 projected-product relative L2
  均不高于 `1e-12`
- [x] 相同配置 determinism relative L2 不高于 `1e-14`
- [x] 所有 case 无 NaN/Inf，intensity 有限且非负
- [ ] final dz、lateral、FOV refinement 对 `U_A_exit`、`P_B`、`I_stack` 的
  relative L2 各不高于 `5%`
- [ ] detector `min(± waist signal) / max(dz,lateral,FOV detector floor) >= 3`
- [x] HDF5 与外部 config/metadata/metrics 一致
- [x] 八张 figures 可读且对应正确物理量
- [x] 新增测试和全仓回归通过，或已记录失败原因
- [x] 不创建 reconstruction，不使用 truth 泄漏帮助 inverse

状态逻辑在运行前固定如下：

- 任一硬控制（geometry、algebra、zero contrast、single slice、projected product、
  determinism、finite、输出语义/HDF5）失败：`Failed`；
- 硬控制通过，但任一 `U_A_exit/P_B/I_stack` convergence 超过 5%，或 detector
  signal/floor 小于 3：`Inconclusive`；
- 所有硬控制、全部 convergence 和 detector visibility gate 均通过：`Passed`。

不能因 detector 图样“看起来合理”而越过 A-exit convergence gate，也不能在看见
结果后放宽阈值。

## 16. 实际结果

### 运行信息

- 运行日期：`2026-08-10`
- Git commit：`23d702b5d430b15f594462a9d9973569e66bff24`；本任务修改保持 unstaged
- run 路径：`runs/exp040_TGV_3d_multislice_forward_20260810_154908/`
- Python/environment：`tgv_ptycho_sim`，Python `3.11.15`，Windows
- 运行命令：

  ```powershell
  python scripts/run_exp040_multislice_forward.py --config configs/experiments/exp040_TGV_3d_multislice_forward.yaml
  ```

- 测试命令与结果：`python -m pytest -q`，`102 passed in 5.62s`
- 修改范围 Ruff：`All checks passed`
- 项目级 Ruff：`python -m ruff check src scripts tests` 仍报告 11 个既有问题：
  `scripts/run_exp001_forward.py` 的 8 个 `E402`，以及
  `calibration/stage.py`、`recon/losses.py`、`recon/rpie.py` 各 1 个 `E501`；
  它们不在 exp040 修改范围内，本任务未改动。

### 数值结果

| Metric | 结果 | 是否通过 |
|---|---:|---|
| geometry/hard controls | thickness error `0 m`；diameter error `0 m`；材料、shape、dtype 正确 | Yes |
| zero contrast | `2.350435e-13` | Yes |
| single slice | `0` | Yes |
| no-propagation product | `1.355924e-14` | Yes |
| projected-phase discrete product | `8.335707e-14` | Yes |
| discrete projected vs analytic diagnostic | `4.411785e-1` | Report only |
| dz convergence (`U_A_exit/P_B/I_stack`) | `0.343362 / 0.343362 / 0.372246` | No |
| lateral convergence (`U_A_exit/P_B/I_stack`) | `0.111835 / 0.112331 / 0.159475` | No |
| FOV convergence (`U_A_exit/P_B/I_stack`) | `0.005373 / 0.368378 / 0.786966` | No |
| detector waist signal (`minus / plus`) | `0.534377 / 0.452668` | Report only |
| detector floor；signal/floor | `0.786966`；`0.575206`（阈值 `3`） | No |
| determinism/finite/nonnegative | max relative L2 `0`；全部 finite 且 intensity 非负 | Yes |

### 观察

- Geometry、zero contrast、single-slice、no-propagation、projected discrete product、
  determinism 和 finite controls 全部通过。代数控制未使用 global-phase alignment。
- `projected_discrete_to_analytic_relative_l2=0.441178` 是离散 midpoint/binary-volume
  与解析 projected model 的报告项，不是 phase-screen 乘积 identity gate；其较大数值
  与当前 axial/lateral 离散尚未收敛一致，不能解释为代数实现通过后的物理精度证明。
- 三组 convergence 均至少有一个输出超过预注册 `5%` 阈值。尤其 FOV refinement
  虽使 `U_A_exit` 变化降到 `0.54%`，但 `P_B` 和 detector stack 仍分别变化
  `36.84%` 与 `78.70%`，说明外部传播、FFT 周期边界和有限 FOV 的影响仍很强。
- 腰径 `±2 um` 的 detector 信号低于数值离散 floor，signal/floor 仅 `0.575`；因此
  不能声称该扰动在当前仿真离散下数值可分辨，更不能外推到含噪真实实验。
- HDF5 `/entry` 顶层严格为 `config_yaml,data,instrument,metadata,metrics,sample,truth`；
  没有 `reconstruction`、`calibration` 或 `preprocessing`。baseline 与 sweep baseline
  的 `I_stack/U_A_exit/P_B` 逐元素完全一致，slice widths 总和精确为 `100 um`，
  全部数值 dataset finite。metadata 明确记录 internal/external index、axis、plane、
  确定性 plane-wave seed 语义和输入/结果两种 status。
- 八张预注册 PNG 已逐张人工检查。convergence 图在 log 轴上省略 exact-zero reference
  点，避免零值把纵轴拉至机器下溢尺度；所有图的标题、物理坐标、单位和 colorbar 可读。

## 17. 已知限制

- centered symmetric split-step 仍是 scalar、unidirectional phase-screen 近似，不是
  Maxwell solver；高玻璃—空气折射率差下的定量物理误差未由本实验消除。
- voxel-center binary mask 会产生 lateral staircase；本实验通过 lateral convergence
  量化其影响，但不引入面积平均或长期 supersampling 标准。
- ASM/FFT 使用周期性边界；FOV cases 只检查共同中心 ROI 的稳定性，不能证明任何
  有限 padding 已彻底消除 wrap-around。
- sample B 使用当前 periodic integer-pixel shift；不实现 subpixel interpolation、
  finite support 或 position refinement。
- detector 是无噪声 grid-sampled intensity。`detector_pixel_size=0.5 um` 是 baseline
  sampling metadata；本实验不定义 detector pixel integration 或跨 sampling remap 的
  长期项目标准。
- 不含 noise、dynamic range、quantization、background、stage error、tilt、surface
  roughness、材料吸收或 calibration。
- 不保存所有 slice fields，以限制内存和 HDF5 体积；失败诊断依赖显式 metrics 和
  受控 figures。
- `D_waist ±2 um` 是预注册无噪声数值扰动，不是检测限、精度声明或唯一可辨识性证明。

## 18. 失败、废弃或替代记录

- 是否失败/废弃：`未废弃；正式结果为 Inconclusive`
- 原因：硬控制通过，但 axial/lateral/FOV convergence 和 detector visibility gate
  未全部通过；按预注册状态逻辑不能判为 `Passed`。
- 正式保留 run：`runs/exp040_TGV_3d_multislice_forward_20260810_154908/`
- 中间诊断 run：`..._151317/` 与 `..._152516/` 按 runs 规则保留，但不作为最终验收
  产物；前者暴露 convergence 图 exact-zero 尺度和 metadata 审计缺口，后者用于验证
  图形修正，最终 run 已包含全部修复和扩展自检。
- 替代实验：`N/A；应继续当前 exp040 的离散化与 FOV 诊断`
- 对后续实验的影响：在 exp040 达到 `Passed` 前，不得以本实验支持 Phase 5 的腰径
  定量反演结论；`Failed` 或 `Inconclusive` 也必须保留诊断 run 并更新本节。

## 19. 结论

exp040 已建立可运行、可测试、可写出 HDF5 的 centered symmetric split-step 3D TGV
multi-slice forward baseline，并验证几何、算子顺序、参考相位、A-exit 到 B/detector
数据流及代数 controls。正式状态为 `Inconclusive`：预注册网格下尚未达到 axial、
lateral 和外部 FOV 收敛，detector 腰径扰动也未稳定高于离散化 floor。因此不更新
`roadmap.md` 的 Phase 4 状态，不声称当前模型已收敛、真实 TGV 腰径已可测或 Phase 5
可开始。

## 20. 下一步实验

- 继续当前 `exp040`：先做不改变研究问题的诊断 refinement，例如加入
  `dz=0.25 um`、更细 lateral sampling，并扩大/显式 padding 外部传播 FOV；继续使用
  同一 canonical B、物理 scan 和预注册指标，定位 axial、voxelization 与 periodic-wrap
  floor。任何新阈值或模型变更必须先写入文档，不能用看后调参改写本次结论。
- 暂不新开 Phase 5。只有 exp040 在可接受离散下通过 convergence 与 visibility gate，
  才建议使用新实验编号 `exp050` 研究不泄漏 truth 的 `D(z)`/`D_waist` parametric inverse、
  noise floor 与 identifiability。
- Phase 5 开始前仍需单独确认参数化、损失函数、噪声模型、mandatory calibration、
  detector sampling 与验收指标；本实验不自行决定这些项目级标准。

---

## exp040 追加诊断 R1：运行前预注册（2026-08-10）

### R1.1 Append-only 与状态边界

本节是在 R0 正式结果之后追加的运行前预注册。上方第 1--20 节、R0 配置、正式 run、
checkbox、metrics、`Inconclusive` 结论和路线图状态均保持原样，不因 R1 结果回改。
R1 执行后必须再次在文件末尾用新的 `---` 追加执行记录，不修改本预注册段落。

- 诊断编号：`exp040-R1`
- 研究问题：进一步缩小 axial/lateral discretization，并把 internal-FOV 与固定 A-exit
  后的 external AB/BC padding 分开，以定位 voxelization 和 periodic-wrap floor。
- 理论文档：`docs/theory_notes/exp040_refinement_and_external_padding.md`
- 新配置：`configs/experiments/exp040_TGV_3d_multislice_refinement.yaml`
- 启动入口：继续使用 `scripts/run_exp040_multislice_forward.py`
- R0 `metrics.experiment_status`：必须保持 `Inconclusive`
- R1 结果字段：独立写入 `metrics.diagnostics_r1`
- 当前状态：`Pre-registered；尚未实现或运行 R1`

### R1.2 不变条件

R1 不改变以下物理量、seeds 或既有阈值：

- vacuum wavelength `532 nm`；internal $n_\mathrm{ref}=1.5$；external $n=1.0$；
- sample thickness `100 um`，`D_top/D_waist/D_bottom=30/20/30 um`，
  `z_waist=50 um`，binary voxel-center mask；
- plane-wave amplitude、normal incidence、centered symmetric split-step、ASM kernel；
- `z_AB=0.5 mm`，`z_BC=1.0 mm`；
- sample B seed `20260840`、phase range `0.8 rad`、physical cell `2 um`；
- scan seed `20260841`、`5x5` positions、step `4 um`、jitter quantum `1 um`；
- periodic integer-pixel B shifts、noiseless grid-sampled detector；
- algebra `1e-12`、determinism `1e-14`、convergence `5%`、detector signal/floor `3`。

R1 不引入 reconstruction、noise、subpixel shift、detector integration、truth alignment、
临时 apodization 或为了通过结果而选择的新阈值。

### R1.3 预注册 refinement cases

| 组别 | 保留的 R0 cases | R1 新 cases | final acceptance pair | common domain |
|---|---|---|---|---|
| axial | `2,1,0.5 um` | `0.25 um` | `0.5 -> 0.25 um` | `128² @ 0.5 um` |
| lateral | `1,0.5,0.25 um` | `0.125 um` (`512²`) | `0.25 -> 0.125 um` | `64 um` FOV；映射到 `256² @ 0.25 um` |
| full-chain FOV | `64,80,96 um` | `112,128 um` | `112 -> 128 um` | 中心 `128² @ 0.5 um` ROI |

每个 final pair 继续分别计算未对齐的 `U_A_exit/P_B/I_stack` relative L2。旧 cases 和
旧 metrics 必须保持数值兼容；R1 只在 `diagnostics_r1` 下保存新增 series、pair 和状态。

### R1.4 同一 canonical B 的登记

R1 细网格 base canonical 为 `768² @ 0.125 um`、FOV `96 um`。它使用同一 seed 和
`16 px = 2 um` feature，coarse random map 仍为原来的 `48²`，因此是 R0 physical B 的
细网格表示，不是新 realization。

为支持 `128 um` working FOV，先对该 `768²` base 做 centered periodic extension：每侧
增加 `128 px = 16 um = 8 phase cells`，得到 `1024² @ 0.125 um`。禁止直接以相同 seed
生成 `1024²` random object。自动化 gate：

- fine base 映射回 R0 `384² @ 0.25 um` 的 max complex difference `<=1e-12`；
- working canonical 的中心 `768²` 与 base 逐元素完全一致；
- 所有 transmissions finite、complex128，且 $|B|=1$ 到浮点容差；
- 所有 R1 cases 只 crop/sample/periodically extend 此 working canonical，不重抽 RNG。

### R1.5 External-only padding 算子

External diagnostic 固定 R0 baseline 的 `128² @ 0.5 um` A-exit，不重跑或改变 sample A。
先用同一 internal reference medium 和 phase convention 计算 homogeneous exit
$U_{A,ref}$，定义 $\delta U_A=U_A-U_{A,ref}$。对每个 target shape：

1. 只将 $\delta U_A$ 中心 zero-pad；
2. 在完整 target grid 上加回同一 homogeneous full-field reference；
3. 在完整 target grid 上传播到 B；
4. 从同一 working canonical B 取得完整 target FOV，应用相同 physical scan；
5. 传播到 detector，最后 crop 相同中心 `128²` ROI。

禁止直接 zero-pad 非零 plane-wave full field，禁止 zero-pad B，禁止把 B 外部假设成
0 或 1。padding shapes 固定为：

```text
128², 160², 192², 224², 256² @ 0.5 um
```

即 external FOV `64,80,96,112,128 um`；acceptance pair 为 `224² -> 256²`。另报告
原 `128²` grid 外侧固定 `4 um` ring 中 $\delta U_A$ 的 energy fraction；它不设置
看后阈值，也不通过 window 人为降低。

### R1.6 Metrics、floor 与 R1 状态

`metrics.diagnostics_r1` 至少包含：

```text
version
methods
canonical_b_validation
refined_convergence/{axial,lateral,fov}
external_padding
refined_floor
visibility_report
all_finite
all_intensity_nonnegative
status
```

预注册 gates：

- refined axial `0.5 -> 0.25 um`：三个输出各 `<=0.05`；
- refined lateral `0.25 -> 0.125 um`：三个输出各 `<=0.05`；
- refined full-chain FOV `112 -> 128 um`：三个输出各 `<=0.05`；
- external padding `112 -> 128 um`：`P_B/I_stack <=0.05`；
- padded A-exit center invariance：`<=1e-12`；
- 全部 R1 arrays finite，全部 R1 intensity 非负；
- R0 同一 waist `±2 um` detector signal / refined detector floor：`>=3`。

其中 refined detector floor 是 refined axial/lateral/full-chain FOV final-pair
`I_stack` error 的最大值。该 ratio 使用 R0 同一 perturbation signal，只回答它相对于
更新后 discretization floor 的大小，不冒充联合 finest-grid observability。

R1 hard diagnostic 失败时 `status=Failed`；hard checks 通过但任一 refinement、external
padding 或 ratio gate 失败时 `status=Inconclusive`；全部通过才为 `Passed`。无论 R1
结果如何，都不得覆盖 R0 `metrics.experiment_status=Inconclusive`。

### R1.7 产物预注册

HDF5 继续使用原 7 个 `/entry` 顶层 groups，不创建 `reconstruction`、`calibration` 或
`preprocessing`，不保存 R1 大体积 volume/slice fields。新增结果随 metrics 镜像到：

```text
/entry/metrics/diagnostics_r1/...
```

旧八张 figures 和 API 保持不变；R1 另生成：

```text
r1_refined_convergence.png
r1_external_padding_convergence.png
```

两张图只读取已计算 metrics/series，log 轴省略 exact-zero self-reference，并显示原
`5%` threshold。R1 run 必须是新 timestamped directory，不覆盖 R0 或诊断历史。

### R1.8 执行前验收清单

- [ ] optional `diagnostics_r1` 不改变 legacy config 的旧 result/metrics/八张图；
- [ ] 新 config 验证 pair、shape、ROI、center alignment、B extension 和 integer scan；
- [ ] canonical B 的同一 physical realization gates 通过；
- [ ] refined axial/lateral/FOV 与 external-padding metrics 按本节定义实现；
- [ ] HDF5/JSON metrics 一致，旧顶层结构与 baseline/sweep truth 不变；
- [ ] 旧八图和 R1 两图全部可读；
- [ ] 新增/修改范围 tests 与 Ruff 通过；
- [ ] 实际运行前不修改本节参数或阈值；
- [ ] 运行后只在 EOF 新增 `---` 与 R1 执行记录。

---

## exp040 R1 执行记录

### R1.E1 执行身份与产物

- 执行日期：2026-08-10
- 配置：`configs/experiments/exp040_TGV_3d_multislice_refinement.yaml`
- 正式 run：`runs/exp040_TGV_3d_multislice_refinement_20260810_181728`
- run state：`complete`，`artifacts_validated=true`
- R0 `metrics.experiment_status`：`Inconclusive`，未被 R1 覆盖
- R1 `metrics.diagnostics_r1.status`：`Inconclusive`
- 冻结配置 SHA-256：`D987F8216531727E4A6A2F609EE306E9BDDC0D27CB1F92BCE9FB7FD0D2D867DB`
- 外部 metrics SHA-256：`3CFE64D849758A23EB760256B3693A93F4B7DBB86DFC0144CE814AA62AC04FFF`

报告 run 是修正 R1 external-padding 图中 gate 标注后创建的新 timestamped run；没有覆盖
`runs/exp040_TGV_3d_multislice_refinement_20260810_180530`。两次 run 的冻结
`config.yaml` 和 `metrics.json` 哈希分别完全一致，因此图示修正没有改变数值计算或验收
结论。运行脚本的终端短摘要仍打印保留的 R0 convergence/floor；本节所有 R1 数值均读取
`metrics.diagnostics_r1`，二者不得混用。

### R1.E2 预注册 pair 的结果

所有 relative L2 均为未做 global-phase alignment 的原始复场或强度差异；gate 沿用预注册
`0.05`，没有看后改阈值。

| 诊断 final pair | `U_A_exit` | `P_B` | `I_stack` | gate 结果 |
|---|---:|---:|---:|---|
| axial `0.5 -> 0.25 um` | 0.188238 | 0.188238 | 0.207719 | 三项失败 |
| lateral `0.25 -> 0.125 um` | 0.067745 | 0.053777 | 0.075441 | 三项失败 |
| full-chain FOV `112 -> 128 um` | 0.0000705 | 0.276912 | 0.729435 | A-exit 通过，P/I 失败 |
| fixed-A-exit external padding `112 -> 128 um` | 2.55e-16（center invariance） | 0.276348 | 0.729701 | invariance 通过，P/I 失败 |

补充 hard controls 与诊断量：

- fine canonical B 对 R0 physical realization 的实际最大 complex mapping error 为
  `1.57e-16`，小于登记上限 `1e-12`；periodic-center mapping error 为 `0`；
- fixed-A-exit padding 的 center-invariance 最大值为 `2.55e-16`，小于独立 gate
  `1e-12`；
- scattered residual edge-energy fraction 为 `0.017338`；该量未登记独立通过阈值，
  这里只报告，不据此追加看后判定；
- 所有 R1 数组 finite，所有 R1 intensity 非负；
- refined detector floor 为 `0.729435`；同一 R0 waist `+/-2 um` signal 相对该 floor
  的最小 ratio 为 `0.620573 < 3`，visibility gate 失败。

### R1.E3 诊断定位与结论边界

R1 没有达到 axial 或 lateral convergence，因此当前 voxelized multislice 的离散化 floor
仍未被压到预注册 `5%` 以下，不能宣称 axial/voxelization 已收敛，也不能据此验证腰径
observability。

在 refined full-chain FOV pair 中，sample-A exit common-ROI error 已降到
`7.05e-5`，但同一 pair 的 `P_B/I_stack` 仍为 `0.277/0.729`。固定同一 A-exit、只扩大
外部传播网格的独立诊断又得到几乎相同的 `P_B/I_stack` error，并且 A-exit center
invariance 达到机器精度。这支持如下有限定位：当前高 floor 的主要残留发生在 A-exit
之后的外部传播、全平面周期 B 与 detector 路径，而不是 sample-A internal-FOV 截断本身。

该结果不能进一步单独归因于某一个机制。现有 R1 还没有区分 ASM sampling/alias、
`96 um` canonical-B 周期与 `112/128 um` FOV 不相容、periodic `np.roll` wrap、有限传播
窗口及 detector representation 的各自贡献。不得把 external-padding 失败解释为 TGV 的
真实物理不可观测，也不得用未经登记的 window、有限 illumination 或阈值更改重写本次
`Inconclusive` 结论。

### R1.E4 文件与验证

- HDF5 保持原 7 个 `/entry` children：`config_yaml`、`data`、`instrument`、`metadata`、
  `metrics`、`sample`、`truth`；没有伪造 `reconstruction`、`calibration` 或
  `preprocessing`；
- R1 仅新增 `/entry/metrics/diagnostics_r1/...`，外部 JSON 与 HDF5 config、metadata、
  metrics 已用 runner validator 做递归一致性复核；
- 原八张图保留，新增 `r1_refined_convergence.png` 和
  `r1_external_padding_convergence.png`；两图已人工检查，external-padding 下半图显示
  登记的 `1.0e-12 A-exit invariance gate`，不再错误复用 `5%` gate；
- 全量测试：`123 passed in 12.00s`；
- 本次修改范围 Ruff：`All checks passed`；
- Git：所有修改保持 unstaged；未 commit、push 或创建 PR。

### R1.E5 下一步边界

若继续 exp040，下一轮应在再次运行前于本文 EOF 追加独立 R2 预注册。优先拆分
period-commensurate canonical-B/FOV 与 alias-controlled propagation 两类诊断；任何采用
band-limited ASM、有限 aperture 或 Gaussian/super-Gaussian illumination 的方案都属于
新的数值方法或 forward-model 变更，必须先在 `docs/theory_notes/` 记录物理依据、可实现
边界和不能修复的误差类型，再登记参数、comparison pair 与不变阈值。R2 不得覆盖 R0/R1
的 `Inconclusive` 结果。

---

## exp040 R2 预注册：周期相容边界与 transfer-sampling alias

### R2.1 身份、目标与冻结边界

- 理论文档：`docs/theory_notes/exp040_r2_periodic_boundary_and_alias_control.md`
- 计划配置：`configs/experiments/exp040_TGV_3d_multislice_r2_boundary_alias.yaml`
- 启动入口：继续使用 `scripts/run_exp040_multislice_forward.py`
- R2 结果字段：独立写入 `metrics.diagnostics_r2`
- R0 `metrics.experiment_status` 与 R1 `metrics.diagnostics_r1.status`：必须保持
  `Inconclusive`
- 当前状态：`Pre-registered；尚未实现或运行 R2`

R2 只回答两个诊断问题：第一，canonical B 周期为 `96 um` 时，external FOV 改为整数
周期是否降低 periodic-wrap/FOV floor；第二，current ASM 的 sampled transfer alias 是否
material。R2 不改变 TGV geometry、sample-A multislice、plane-wave illumination、
physical scan、periodic integer shift、detector model、canonical B random realization 或
R0/R1 阈值。

### R2.2 Period-commensurate cases

固定 external `dx=0.5 um`，登记：

```text
FOV:    96 um, 192 um, 288 um
shape:  192^2, 384^2, 576^2
period: 1, 2, 3 canonical-B periods
final pair: 192 -> 288 um (384^2 -> 576^2)
common detector ROI: centered 128^2 @ 0.5 um
```

B 必须由同一 `768^2 @ 0.125 um`、`96 um` fine canonical base 先映射出一个
`192^2 @ 0.5 um` 周期，再 centered periodic extension。不得为大 FOV 重新抽 random
object，也不得构造新的 `2304^2` fine detector。physical scan positions 与 R0/R1 完全
一致，并在 `0.5 um` grid 上保持 integer shifts。

A-exit 继续使用 R1 冻结的 homogeneous-reference + zero-padded scattered residual；三个
FOV 的中心 baseline ROI invariance 必须 `<=1e-12`。不能 zero-pad full plane-wave field，
不能在 A-exit 或 detector 临时乘 window。

### R2.3 Alias-controlled propagation

每个 period-commensurate case 用完全相同输入分别运行：

1. `current_evanescent_only_asm`：保持当前仅移除 evanescent components 的 transfer；
2. `matsushima_exact_common_ellipse_same_grid`：按 Matsushima & Shimobaba (2009),
   DOI `10.1364/OE.17.019662` 的 transfer local-frequency Nyquist 条件，使用实际 same-grid
   FFT frequency interval 构造二维 exact common-ellipse mask。

alias-controlled mask 只应用于 external AB/BC propagation，不应用于 sample-A internal
multislice。mask 内 transfer 必须与 current ASM 完全一致，mask 外为零；它保留同一
circular/periodic FFT boundary，不冒充论文示例中的 2x-padded linear convolution，也
不冒充开放边界传播。

### R2.4 冻结 metrics、gates 与状态

`metrics.diagnostics_r2` 至少包含：

```text
version
methods
canonical_b_validation
a_exit_center_invariance
period_aligned.current_asm
period_aligned.alias_controlled
method_difference
alias_masks
determinism
thresholds
outcome_flags
all_finite
all_intensity_nonnegative
hard_checks_pass
status
```

运行前冻结：

- 两种方法分别计算 relative-to-`288 um` series；
- final pair `192 -> 288 um` 的 `P_B/I_stack` convergence gate 继续为 `<=0.05`；
- 同 FOV current-vs-alias method difference 任一 `P_B/I_stack >0.05` 记为 material；
- canonical-B mapping `<=1e-12`；
- padded A-exit center invariance `<=1e-12`；
- largest alias-controlled case 重复运行 determinism `<=1e-14`；
- 全部 arrays finite，全部 intensity 非负。

R2 hard controls 失败时 `status=Failed`；hard controls 通过但 alias-controlled final pair
任一 `P_B/I_stack >0.05` 时 `status=Inconclusive`；alias-controlled 两项均通过时才为
`Passed`。current ASM pass/fail 与 method materiality 只作预注册归因证据，不为了通过而
筛选方法。R2 不把 R0 standard-ASM waist signal 与 alias-controlled floor 混算，不更新
waist observability 结论。

### R2.5 预注册判读表

| current aligned | alias-controlled aligned | largest-FOV method difference | 允许的结论 |
|---|---|---|---|
| pass | pass | non-material | R1 downstream floor 支持由非整数周期/较小 FOV 主导 |
| pass | pass | material | 两方法各自收敛但绝对解 method-dependent，不能二选一 |
| fail | pass | material | 支持 transfer-sampling alias 为主要贡献 |
| fail | fail | 任意 | alias control 未解释主要 floor，保留 periodic/open-boundary/B/detector residual |
| pass | fail | 任意 | 方法冲突，保持 Inconclusive |

这里 R1 与 R2 的 FOV 大小也不同，因此即使 R2 通过，也只能说“整数周期/更大 FOV
组合”消除了 R1 floor；不能把改善全部单独归因于 period commensurability。

### R2.6 产物、资源与执行前清单

预计最大 full intensity stack 为 `25 x 576 x 576 float64`，约 `66 MB`；两方法顺序
执行并立即裁取共同 ROI，不同时保留 full stacks。HDF5 仍保持原 7 个 `/entry` children，
只镜像 compact `/entry/metrics/diagnostics_r2/...`，不保存 full R2 fields。

R2 新图冻结为：

```text
r2_period_aligned_convergence.png
r2_alias_method_difference.png
```

- [ ] 理论文档、公式、方法边界和 DOI 已在运行前冻结；
- [ ] R2 config 验证 period、shape、dx、ROI、scan 和同一 canonical B；
- [ ] shared ASM 新接口默认行为与全部旧实验数值兼容；
- [ ] alias mask 的双椭圆、propagating subset、DC 和 mask-inside transfer controls 通过；
- [ ] R2 JSON/HDF5 validator、两图和 legacy/R1 regression tests 通过；
- [ ] 正式运行前不修改本节 cases、阈值、状态或判读表；
- [ ] 运行后只在 EOF 新增 `---` 与 R2 执行记录。

### R2.7 运行前 method-difference 方向澄清

同 FOV current-vs-alias-controlled relative L2 固定以 alias-controlled 输出作为
denominator：`||Q_current-Q_alias|| / max(||Q_alias||, eps)`，其中
`Q in {P_B, I_stack}`。这只定义非对称 relative L2 的方向，不把 alias-controlled
结果预设为 truth；`5%` materiality threshold 和 R2.5 判读表均不改变。

---

## exp040 R2 执行记录

### R2.E1 执行身份与产物

- 执行日期：2026-08-11
- 配置：`configs/experiments/exp040_TGV_3d_multislice_r2_boundary_alias.yaml`
- 正式 run：
  `runs/exp040_TGV_3d_multislice_r2_boundary_alias_20260811_144331`
- run state：`complete`，`artifacts_validated=true`
- R0 `metrics.experiment_status`：`Inconclusive`，未被 R2 覆盖
- R1 `metrics.diagnostics_r1.status`：`Inconclusive`，未被 R2 覆盖
- R2 `metrics.diagnostics_r2.status`：`Inconclusive`
- R2 interpretation code：`remaining_downstream_floor`
- 冻结配置 SHA-256：
  `9C31B19C9A61DE883629B41DDBCD3A97A609546217F5242A018C0E437AC8DDD5`
- 外部 metrics SHA-256：
  `FACC464A28F57848A8B341B45AF3C2DAA3A684170C650733938BBE289DB44E15`

正式执行命令为：

```powershell
python scripts/run_exp040_multislice_forward.py --config configs/experiments/exp040_TGV_3d_multislice_r2_boundary_alias.yaml
```

### R2.E2 Period-aligned convergence

三组 FOV 均为同一 `96 um` canonical B 的整数周期，固定 `dx=0.5 um`；series 以
`288 um` 为 reference，因此最后一点是 exact-zero self-reference。final acceptance pair
为 `192 -> 288 um`，冻结 gate 为 `0.05`。

| 方法 | 输出 | 96 um 对 288 um | 192 um 对 288 um | final gate |
|---|---|---:|---:|---|
| current ASM | `P_B` | 0.237832 | 0.115572 | 失败 |
| current ASM | `I_stack` | 0.333185 | 0.172736 | 失败 |
| alias-controlled | `P_B` | 0.016503 | 0.007699 | 通过 |
| alias-controlled | `I_stack` | 0.464809 | 0.308996 | 失败 |

因此，把 FOV 改为 canonical-B 整数周期并扩大到 `288 um`，没有使 current ASM 的
`P_B` 或 `I_stack` 达到 `5%` convergence。非整数周期/较小 FOV 不能单独解释 R1
downstream floor。

alias-controlled ASM 使 `P_B` final-pair error 从 current ASM 的 `0.1156` 降到
`0.00770`，但 `I_stack` final-pair error 仍为 `0.3090`，且高于 current ASM 的
`0.1727`。按预注册状态逻辑，alias-controlled 的两个输出没有全部通过，故 R2 必须保持
`Inconclusive`。

### R2.E3 Same-FOV method difference 与 mask

method difference 使用 R2.7 冻结的 alias-controlled denominator：

| FOV | `P_B` current-vs-alias | `I_stack` current-vs-alias |
|---:|---:|---:|
| 96 um | 0.240221 | 0.899119 |
| 192 um | 0.112005 | 0.591163 |
| 288 um | 0.030291 | 0.396474 |

在最大 FOV，`P_B` method difference 已低于 `5%` materiality threshold，但
`I_stack=0.396474` 仍明显 material。transfer-sampling alias 对 detector path 有实质
影响，但 alias control 本身没有给出 detector-FOV-converged 解；不能把其中任一种方法
事后选择为 truth。

exact common-ellipse mask 的 kept-bin fractions 为：

| FOV | AB | BC |
|---:|---:|---:|
| 96 um | 0.033230 | 0.007840 |
| 192 um | 0.124139 | 0.032288 |
| 288 um | 0.263880 | 0.071389 |

BC 因传播距离更长而保留更窄的 sampled-transfer support。mask-inside transfer 与
current ASM 的最大 complex difference 为 `0`，mask 外 nonzero bins 为 `0`，DC 全部
保留。上述 support fractions 只描述数值 mask，不证明被移除的 spectrum 在物理上可以
忽略。

### R2.E4 Hard controls

- R2 base period 对同一 R0/R1 canonical-B realization 的最大 complex mapping error：
  `2.22e-16 <= 1e-12`；
- 三个 centered periodic extensions 的中心周期逐元素不变，unit-modulus 与 finite
  controls 通过；
- padded A-exit center-invariance 最大值：`2.24e-16 <= 1e-12`；
- largest alias-controlled case 重复运行的 `P_B/I_stack` determinism errors 均为 `0`，
  通过 `1e-14` gate；
- 所有 R2 arrays finite，所有 intensity 非负；
- 全部 hard controls 通过，R2 不是由于实现失败而 `Inconclusive`。

### R2.E5 允许的科学定位

预注册判读表对应 `current fail / alias-controlled fail / method difference material`，所以
结果代码为 `remaining_downstream_floor`。R2 支持以下有限结论：

1. R1 的高 detector floor 不能只归因于 `112/128 um` 与 `96 um` B 周期不相容；在
   `192/288 um` 整数周期 pair 上，current ASM 仍未收敛。
2. transfer-function sampling alias 是 material contributor，尤其在 BC-to-detector
   intensity；但 Matsushima same-grid mask 没有让 detector intensity 随 FOV 收敛。
3. `P_B` 在 alias control 下已收敛，而 `I_stack` 没有，进一步把主要未决项定位到
   B-plane multiplication、周期移位后的 exit waves、BC propagation 与 grid-sampled
   detector representation 的组合，而不是单独的 AB probe propagation。

R2 没有实现 linear convolution/open boundary、finite sample B、finite illumination、
detector pixel integration 或 detector oversampling。因此不能把 residual 强行解释为
真实 TGV 不可观测，也不能据此证明 alias-controlled ASM 的绝对物理正确性。若继续，
应先预注册新的 detector-path diagnostic，分别保存/比较 B 后 exit-wave spectrum、BC
传播 sampling 与 detector sampling；任何 finite aperture、Gaussian illumination、
band-limited B 或 pixel integration 都是 forward-model 变化，必须先写理论与阈值。

### R2.E6 文件与验证

- HDF5 仍保持 7 个 `/entry` children：`config_yaml`、`data`、`instrument`、`metadata`、
  `metrics`、`sample`、`truth`；
- 新增 compact `/entry/metrics/diagnostics_r2/...`；没有保存 full `576^2` fields/stacks，
  也没有伪造 `reconstruction`、`calibration` 或 `preprocessing`；
- 外部 config、metadata、metrics 与 HDF5 镜像已用 runner validator 递归复核；
- 原八图、R1 两图和 R2 两图共 12 张均存在；R2 两图已人工检查；
- 全量测试：`146 passed in 14.68s`；
- 本次修改范围 Ruff：`All checks passed`；
- Git：所有修改保持 unstaged；未 commit、push 或创建 PR。

---

## exp040 R3 预注册：detector-path 分层诊断

### R3.1 身份与不变条件

- 理论文档：`docs/theory_notes/exp040_r3_detector_path_diagnostics.md`
- 计划配置：`configs/experiments/exp040_TGV_3d_multislice_r3_detector_path.yaml`
- 运行入口：继续使用 `scripts/run_exp040_multislice_forward.py`
- R3 结果字段：独立写入 `metrics.diagnostics_r3`
- R0/R1/R2 status 和 metrics：必须逐字义保留
- 当前状态：`Pre-registered；尚未实现或运行 R3`

R3 不更换 TGV、illumination、canonical B、physical scan、periodic boundary 或 scalar
one-way propagation。它在 B 后、BC transfer 后和 detector operator 后插入 diagnostic，
并新增一个理想 finite square-pixel area-average branch；该 branch 不替换 baseline。

### R3.2 冻结 grids 与输入

```text
external FOV: 192 um x 192 um (two 96 um canonical-B periods)
native detector pixel: 0.5 um
sampling factors: [1, 2, 4]
dx: [0.5, 0.25, 0.125] um
shapes: [384^2, 768^2, 1536^2]
final sampling pair: factor 2 -> 4
native comparison ROI: centered 128^2 @ 0.5 um
scan positions: same 25 physical positions as R0-R2
```

B 必须由同一 $48\times48$ physical phase-cell realization 构造一个 `96 um` period，再
centered wrap extension 为两个周期。fixed A-exit 继续使用 homogeneous reference +
mapped scattered residual；映射后在 native sample points 上恢复 baseline A-exit 的误差
必须 `<=1e-12`。不得重新运行一个不同 random B，不得用 interpolation 输出冒充新的 TGV
truth。

### R3.3 冻结 propagation 与 spectrum metrics

AB 固定使用 `matsushima_exact_common_ellipse_same_grid`。对每个 factor 计算 native-ROI
`P_B` convergence。每个 scan 构造同一
`E_s=P_B*B_s`，然后 BC 顺序运行 current ASM 与 alias-controlled ASM。

对 B 后 exit wave 报告：

```text
outside_BC_alias_mask_energy_fraction: mean, max
outside_native_detector_nyquist_energy_fraction: mean, max
```

对两种 BC 方法的 full-resolution detector intensity 分别报告 native-detector Nyquist 外
spectrum energy fraction 的 mean/max。factor 1 不可观察自身 Nyquist 外内容，判读使用
factor 4。所有 spectral materiality threshold 冻结为 `0.05`。

### R3.4 冻结 detector operators

每个 full-resolution intensity 生成：

1. `point_sample`；
2. `pixel_box_average`：intensity spectrum 乘
   `sinc(pixel_size*fx)*sinc(pixel_size*fy)` 后在同一 native centers 取样。

两者均裁取同一 `128^2` native ROI。对 BC current/alias 两种方法分别保存 sampling-factor
series 和 factor `2 -> 4` final-pair intensity errors。factor 4 另报告 point-vs-pixel
relative L2，以 pixel branch 为 denominator。

只保存 compact ROI arrays 供当次计算；R3 HDF5 不保存 full `1536^2` fields 或
`25 x 1536^2` stacks。scan 必须流式执行，最大 factor 的 full detector field/intensity
不能跨 scan 累积。

### R3.5 冻结 gates 与状态

- AB `P_B` factor `2 -> 4` convergence：`<=0.05`；
- primary corrected branch = `alias-controlled BC + pixel_box_average`；
- primary detector intensity factor `2 -> 4` convergence：`<=0.05`；
- spectral、BC method 和 point-vs-pixel materiality：`>0.05`；
- A-exit native recovery、canonical-B mapping：`<=1e-12`；
- factor-4 scan `0` primary determinism：`<=1e-14`；
- pixel-box constant/sum/imaginary/negative controls：`<=1e-12` relative scale；
- 全部 arrays finite，最终 detector averages 非负。

hard controls 失败时 R3 `Failed`；hard controls 通过但 AB probe 或 primary detector
convergence 失败时 `Inconclusive`；二者均通过才为 `Passed`。R3 status 不覆盖
`metrics.experiment_status`，也不重新计算 waist observability。

### R3.6 预注册判读

| AB probe | primary pixel branch | 其他证据 | 允许结论 |
|---|---|---|---|
| fail | 任意 | 任意 | upstream sampling 仍混入，detector attribution 不充分 |
| pass | pass | point fail 且 point-vs-pixel material | point-detector model 缺陷得到支持 |
| pass | fail | B-exit/BC alias material | finite pixel response 未解决 B/BC downstream floor |
| pass | fail | alias non-material | 优先检查 boundary/FOV 或更高层物理模型 |

不得在看见结果后改变 primary branch、factor pair、ROI 或 `5%` threshold。

### R3.7 产物与执行前清单

新增 figures 冻结为：

```text
r3_b_exit_and_bc_spectrum.png
r3_detector_sampling_convergence.png
r3_detector_operator_difference.png
```

HDF5 仍保持原 7 个 `/entry` children，只增加 compact
`/entry/metrics/diagnostics_r3/...`。

- [ ] detector pixel integration operator 的 constant、sum、center alignment 测试通过；
- [ ] official/tiny R3 config、同一 B、integer scan 和 streaming contract 验证；
- [ ] legacy/R1/R2 metrics、figures 和 HDF5 contract 回归通过；
- [ ] 正式运行前不修改本节 factors、methods、gates 或判读表；
- [ ] 运行后只在 EOF 追加 `---`、R3 执行记录和模型缺陷建议。

### R3.8 运行前 sampling-origin 澄清

R3.2 的 `point_sample` 必须表示同一组物理 native detector centers。对当前全部为偶数的
shape，若机械沿用 `(N-1)/2` 对称坐标，偶数 oversampling factor 的 native centers 会落在
两个 fine samples 之间，与“aligned fine-grid samples”相矛盾。为消除该实现歧义，R3 在
运行前固定以下索引约定，不改变 FOV、dx、ROI、factor pair 或任何 gate：

```text
native sample indices on factor-q array: o_q + q*k
o_q = floor((q - 1) / 2)
fine-grid physical-origin compensation: ((q - 1) / 2 - o_q) * dx_q
```

因此 `q=1` 无补偿，`q=2/4` 的 array-coordinate origin 分别补偿半个对应 fine pixel；
抽取出的物理 native centers 在三组 factor 中保持一致。A-exit residual 映射、canonical-B
piecewise-constant refinement、physical scan shifts 和 detector readout 必须共同使用该约定，
不得只平移其中一项。运行指标需保存 `native_sample_offset_px` 与
`physical_origin_compensation_m`，并以 A-exit native recovery、canonical-B mapping 和
detector center-alignment controls 验证。该澄清只解决离散索引可实现性，不把 fine-grid
interpolation 当作新的 TGV truth，也不改变 R3.5/R3.6 的预注册判读。

---

## exp040 R3 执行记录：detector-path 分层诊断

### R3.E1 执行身份、append-only 与产物

- 执行日期：2026-08-11；
- 配置：`configs/experiments/exp040_TGV_3d_multislice_r3_detector_path.yaml`；
- 正式 run：
  `runs/exp040_TGV_3d_multislice_r3_detector_path_20260811_153852`；
- 命令：
  `python scripts/run_exp040_multislice_forward.py --config configs/experiments/exp040_TGV_3d_multislice_r3_detector_path.yaml`；
- run state：`complete`，`artifacts_validated=true`；
- R0 `metrics.experiment_status=Inconclusive`、R1 `Inconclusive`、R2
  `Inconclusive` 均保留；R3 独立状态为 `Failed`；
- 冻结 config SHA-256：
  `4B17ADD64B0633540322EA416B4C9E23BB7720A85774A8F48BA7CA95A085B4B6`；
- metrics SHA-256：
  `85BE9131F9797B6837DCFD341834F6876633403B624D1CEB4B080183E2ACF9BE`；
- R3 运行前本文长度为 `59184 bytes`，SHA-256 为
  `2B0B56CDE4F821AA7CAF7FF0BC9A9D74485BE900B0725FF538D7994C520737E8`；
  本执行记录只追加在该前缀之后，未改写 R0--R3 预注册内容。

### R3.E2 fixed-192 um sampling convergence

factor-4 是 reference，表中为预注册 final pair `factor 2 -> 4` 的 native
`128^2 @ 0.5 um` unaligned relative L2：

| BC method | output | final-pair relative L2 | 5% gate |
|---|---|---:|---|
| AB alias-controlled | `P_B` | 0.000418 | 通过 |
| current ASM | point sample | 0.109832 | 失败 |
| current ASM | pixel-box average | 0.068485 | 失败 |
| alias-controlled | point sample | 0.022934 | 通过 |
| alias-controlled | pixel-box average（primary） | 0.022649 | 通过 |

因此 R2 中未收敛的 detector floor 不是 AB probe sampling：R3 的 `P_B` 已远低于 `5%`。
BC alias-controlled 的 point 与 pixel 两个 detector branches 也均达到 sampling convergence；
current ASM 即使加入 pixel average 仍为 `6.85%`，没有达到 gate。

### R3.E3 B-exit spectrum 与 BC propagation attribution

factor-4、25 scans 的 mean/max 结果为：

| metric | mean | max |
|---|---:|---:|
| B-exit field energy outside BC alias mask | 0.118636 | 0.118884 |
| B-exit field energy outside native detector Nyquist | 0.017926 | 0.017965 |
| current-ASM detector-intensity energy outside native Nyquist | 0.014601 | 0.014653 |
| alias-controlled detector-intensity energy outside native Nyquist | `1.68e-31` | `2.03e-31` |

factor-4 current-vs-alias BC method difference 为：

- full detector field relative L2：`0.354198`；
- full detector intensity relative L2：`0.492416`；
- native point detector relative L2：`0.540864`；
- native pixel-average detector relative L2：`0.488078`。

这些值均明显高于预注册 `5%` materiality threshold。结合 B-exit 在 BC mask 外约
`11.86%` 的 field-spectrum energy，R3 支持以下有限定位：B multiplication 把 material
energy 推入 Matsushima BC transfer mask 会排除的频带，current 与 alias-controlled BC
因此给出 materially different detector fields/intensities。该结果说明 R2 的 residual 主要
位于 B 后到 BC propagation 的组合，而不是 AB probe。

### R3.E4 detector operator 与 hard failure

factor-4 point-vs-pixel relative L2 为：

- current ASM：`0.097894`，material；
- alias-controlled BC：`0.004315`，non-material。

因此 finite square-pixel correction 对 current ASM 有明显作用，但对已经 alias-controlled 的
primary branch 只有约 `0.43%` 影响，不能把 R2 detector floor 主要解释为“遗漏 pixel area
average”。

pixel-MTF 的 synthetic/代数 controls 均通过：

- center alignment relative L2：`3.22e-16`；
- constant max error：`0`；
- impulse sum error：`4.44e-16`；
- imaginary leak：synthetic `5.55e-17`，actual `2.25e-16`；
- actual sum relative error：`4.00e-16`；
- selected factor-4/scan-0 primary determinism：`0`。

但实际 filtered intensity 的最大 relative negative value 为 `1.089144e-3`，超过预注册
`1e-12` pixel-operator hard gate；代码没有 clip 该大负值，故
`all_intensity_nonnegative=false`、`pixel_operator_controls.pass=false`，R3 必须保持
`Failed`。不得因 `P_B` 和 primary convergence 已通过而把本次状态事后改成 `Passed` 或
`Inconclusive`。

理论解释与后续物理可实现方案见
`docs/theory_notes/exp040_r3_detector_path_diagnostics.md` 第 9 节：grid nodes 上非负的离散
intensity 不保证其 finite Fourier interpolant 在 nodes 之间非负，因此 sinc-MTF 代数正确、
守恒且 shift-equivariant，并不自动等于 positivity-preserving 的 detector quadrature。

### R3.E5 figures、HDF5 与验证

- 正式 run 共 15 张 PNG：原 8 图、R1 2 图、R2 2 图和 R3 3 图；三张 R3 图均已人工检查；
- 正式 run 的两张 R3 log 图包含 exact-zero reference，纵轴被拉到 machine tiny；这是绘图
  可读性问题，不改变 metrics。代码已改为 log 图省略 exact-zero points；复用同一
  `metrics.json`、没有重算 propagation 的 corrected 图位于
  `reports/exp040_r3_detector_path_20260811_153852_postrun_figures/`；正式 run 未覆盖；
- HDF5 大小约 `29.37 MB`，仍只有 `config_yaml`、`data`、`instrument`、`metadata`、
  `metrics`、`sample`、`truth` 七个 `/entry` children；
- 新增 compact `/entry/metrics/diagnostics_r3/...`，没有把 R3 写入 `truth`，没有保存 full
  `1536^2` fields 或 `25 x 1536^2` stacks，也没有伪造 reconstruction/calibration/
  preprocessing；
- 全量测试：`165 passed in 20.47s`；
- 本次修改范围 Ruff：`All checks passed`；
- Git：修改保持 unstaged；未 commit、push 或创建 PR。

### R3.E6 暂不实施的模型级缺陷与建议

本节只给建议，不在本次结果后切换模型：

1. **下一优先项：positivity-preserving detector quadrature。** 应在新预注册 comparison 中
   使用 detector-specific staggered subpixel nodes 与非负 weights，比较 `q=2/4/8`；不能
   把 R3 even-factor point grid 直接 block-average 后宣称是 pixel midpoint integration。
2. **finite support/open boundary。** 当前 plane wave、periodic B、FFT circular convolution
   仍可能隐藏 wrap 与无限支撑假设；linear-convolution padding、finite illumination 和
   finite B 会改变 forward model，必须单独登记，不能回写 R3。
3. **subvoxel interface。** binary voxel-center sidewall 会产生 grid-dependent staircase 高频；
   可先比较 volume-fraction/level-set interface，再判断 residual 是否仍要求 full-wave。
4. **one-way scalar model 的硬物理边界。** 当前模型无法产生 Fresnel reflection、backward
   wave、sidewall multiple scattering 或 polarization。玻璃--空气法向强度反射约 `4%`，
   该项不能靠继续减小 `dz` 恢复。若数值传播、positive detector quadrature、finite support
   和 detector calibration 都受控后仍有结构化偏差，应比较 bidirectional BPM 或
   Lippmann--Schwinger；只有必要时再承担 vector FEM/FDTD 的更高成本。
5. **真实 detector calibration。** ideal square-pixel sinc MTF 不包含 measured PSF/MTF、
   gaps、finite NA、dynamic range、shot/read noise 或标定误差；后续 filter 必须具有物理可
   实现的 real-space kernel，不能仅因降低 residual 而选择。

本次不建议直接换掉 multi-slice 主模型：R3 已把主要数值差异定位到 B-exit spectrum 与 BC
alias-sensitive propagation，并暴露 detector-MTF positivity 缺陷。先完成 positive detector
quadrature 和 open-boundary/finite-support control，证据仍不闭合时再升级双向或 full-wave
模型，物理与计算成本边界更清楚。

---

## exp040 R4 预注册：positivity-preserving detector quadrature

### R4.1 身份与不变条件

- 理论说明：`docs/theory_notes/exp040_r4_positive_detector_quadrature.md`；
- 计划配置：`configs/experiments/exp040_TGV_3d_multislice_r4_positive_quadrature.yaml`；
- 运行入口：继续使用 `scripts/run_exp040_multislice_forward.py`；
- 新结果只写入 `metrics.diagnostics_r4`；
- R3 formal comparator 固定为
  `runs/exp040_TGV_3d_multislice_r3_detector_path_20260811_153852`，config hash
  `4B17ADD64B0633540322EA416B4C9E23BB7720A85774A8F48BA7CA95A085B4B6`，metrics hash
  `85BE9131F9797B6837DCFD341834F6876633403B624D1CEB4B080183E2ACF9BE`；
- R4 不重跑 R1/R2/R3 diagnostics；保留其正式 run 与结论，不复制历史 figures 冒充本次
  产物。

### R4.2 冻结 nodes、grid 与 quadrature

```text
pixel pitch p: 0.5 um
external FOV: 192 um x 192 um
native full grid: 384 x 384
native ROI: centered 128 x 128
q: [2, 4, 8]
node dx: [0.25, 0.125, 0.0625] um
node shapes: [768^2, 1536^2, 3072^2]
acceptance pair: q=4 -> q=8
weights: uniform 1/q^2, all nonnegative
```

每个 native pixel 内使用
`x_m + ((a+0.5)/q - 0.5)*p` staggered midpoint nodes；even-factor native center 位于 nodes
之间，不使用 R3.8 point-grid origin compensation。detector output 是每个 `q x q` node block
上 `|U|^2` 的等权平均。

固定同一 baseline A-exit homogeneous-reference + bilinear mapped residual、同一 48x48
canonical-B phase cells、同一 25 个 physical scans、同一 periodic B 和 `192 um` FOV。AB/BC
均固定为 alias-controlled ASM；R4 不重复 current ASM。

### R4.3 冻结 metrics、gates 与状态

- `P_B`：complex node field 在每个 native pixel 内等权平均后裁取 `128^2`，报告
  relative-to-q8 与 q4->q8 relative L2；
- detector：保存流式形成的 native `128^2 I_stack`，报告 relative-to-q8 与 q4->q8；
- convergence gate：二者均 `<=0.05`；
- constant/sum/node-geometry normalized errors：`<=1e-12`；
- determinism：q8 scan 0，`<=1e-14`；
- weights finite/nonnegative/sum-one；所有 arrays finite；所有 detector outputs nonnegative；
- 禁止保存 full `3072^2` fields 或 `25 x 3072^2` stacks。

hard controls 失败为 `Failed`；hard controls 通过但 `P_B` 或 detector 任一未通过为
`Inconclusive`；二者均通过才为 `Passed`。不得在看见结果后删除 q8、改变 primary branch、
ROI、factor pair 或阈值。

### R4.4 预注册判读、产物与 R5 条件

- R4 Passed：允许随后新追加 R5 finite-support/open-boundary 预注册；
- `P_B` fail：upstream node sampling 未受控，不进入 boundary attribution；
- `P_B` pass、detector fail：positive quadrature 仍未收敛，不把 residual 推给 boundary；
- hard fail：保持 Failed，不 clip、不降低 q。

R4 新图冻结为：

```text
r4_positive_quadrature_convergence.png
r4_positive_quadrature_controls.png
```

本次 run 只要求原 8 图加 R4 2 图，共 10 图。HDF5 仍保持 7 个 `/entry` children，只增加
compact `/entry/metrics/diagnostics_r4/...`。

- [ ] q=2/4/8 node geometry 与 positive-weight unit tests；
- [ ] official/tiny config、streaming 与 same-B/same-scan tests；
- [ ] runner/HDF5/10-figure contract 和 legacy/R1/R2/R3 regression；
- [ ] 正式运行前锁定本文 prefix；
- [ ] 运行后只在 EOF 追加 R4 执行记录；R5 参数在 R4 之后另行预注册。

### R4.5 运行前澄清：A-exit residual 的外缘半像素约定

R4 tiny test 在尚未生成任何正式结果时暴露了一个离散几何缺口：`q=2/4/8` staggered
nodes 覆盖 baseline 每个像素的完整面积，因此最外侧 node 位于最外层 baseline 中心采样点
之外、但仍在该边缘像素的半像素范围内。原通用 `resample_centered_grid()` 只允许在中心采样点
凸包内插值，会拒绝这些合法的边缘像素内 nodes。

在正式运行前固定如下实现约定，不改变 R4 的 FOV、nodes、传播器、ROI、factor pair、阈值或
判读表：先给 baseline scattered residual 每侧增加一个复制边缘值的 ghost sample，再在物理
staggered node 坐标上做同一 centered bilinear sampling；复制只定义原 baseline 外缘半像素内
的离散插值，随后 residual 仍以零散射外部区域居中 pad 到 `192 um` propagation FOV。
homogeneous reference 仍在完整 node grid 上解析生成。不得把这个数值 ghost sample 解释为
finite sample support 或 open boundary；后两项仍只允许在 R4 结论之后进入独立 R5 预注册。

---

## exp040 R4 执行记录：positive quadrature 已收敛

### R4.E1 正式 run 与锁定身份

```text
run: runs/exp040_TGV_3d_multislice_r4_positive_quadrature_20260811_161412
run_state: complete / artifacts_validated=true
R4 status: Passed
legacy experiment_status: Inconclusive（未被 R4 覆盖）
config SHA256: C9628F9D12663CBCA1FCC0BA3533A14313086E1326076329DB5BB62919631D7E
metrics SHA256: F2093962EFF1C369E45145A6C61E0C600D5B6DB55E8794FE2D4F65893C09467C
HDF5 SHA256: A1F8C649780A8B9A82EDDA5AE80237DD39CE5A61140D82AF984D7BA7E7D402EB
```

正式运行前锁定的前 `70254` bytes SHA256 复核为
`3906CFE39492DEDDACFA7309F36CC142B3164A803D75AAAF81520BC42D64A8FC`，与预运行锁一致。
R4.5 是锁定 prefix 之后、任何正式结果之前追加的边缘半像素实现澄清，没有改变 FOV、q、ROI、
传播器、gate 或判读表。

### R4.E2 预注册收敛结果

| 输出 | q2 相对 q8 | q4 相对 q8 / acceptance | gate | 结果 |
|---|---:|---:|---:|---|
| complex block-mean `P_B` | `2.252326e-5` | `4.455618e-6` | `<=0.05` | 通过 |
| positive pixel `I_stack` | `1.991863e-3` | `3.969993e-4` | `<=0.05` | 通过 |

两个 acceptance 输出都远低于预注册 5% gate。R3 的 `alias pixel q2->q4=2.264873e-2`
不能再解释为正值 detector quadrature 的未收敛；改用实际 staggered midpoint nodes 后，q4 到
q8 的 detector 变化只有约 `0.0397%`。这不使 R3 的 sinc-MTF branch 事后变为通过：R3 仍因
其 finite Fourier interpolant 的负值 hard control 保持 `Failed`。

### R4.E3 hard controls

- canonical-B 同一 phase-cell mapping 最大 complex error：`2.482534e-16 <= 1e-12`；
- node-geometry normalized error：`2.710505e-14 <= 1e-12`；
- constant preservation error：`0`；
- per-frame quadrature sum relative error 最大值：`1.133382e-16 <= 1e-12`；
- q8 scan 0 determinism relative L2：`0 <= 1e-14`；
- weights finite、nonnegative、sum-one；所有 arrays finite；所有 detector outputs nonnegative；
- `hard_checks_pass=true`，故 R4 按预注册逻辑为 `Passed`。

### R4.E4 artifacts 与验证

- 正式 run 有原 8 图与 R4 2 图，共 10 张 PNG；两张 R4 图已人工检查；
- HDF5 大小约 `29.25 MB`，`/entry` 仍只有 `config_yaml`、`data`、`instrument`、
  `metadata`、`metrics`、`sample`、`truth`；
- 只增加 compact `/entry/metrics/diagnostics_r4/...`，没有
  `/entry/truth/diagnostics_r4`，也没有 full `3072^2` field/stack；
- R4 scoped regression：`79 passed`；全量测试：`173 passed in 20.60s`；本次修改范围 Ruff：
  `All checks passed`；
- Git 保持 unstaged；未 commit、push 或创建 PR。

### R4.E5 允许结论与下一步

在固定 canonical B、periodic boundary、alias-controlled AB/BC 和 `192 um` FOV 下，
positivity-preserving staggered detector quadrature 已受 sampling 控制。R4 因此满足 R4.4 的
进入条件，允许从本节之后另行预注册 R5 finite-support/open-boundary comparison；R4 本身不说明
periodic B 或 circular propagation 在物理上正确。

---

## exp040 R5 预注册：finite sample-B support / open boundary

### R5.1 身份、不变量与 R4 provenance

- 理论说明：`docs/theory_notes/exp040_r5_finite_support_open_boundary.md`；
- 计划配置：`configs/experiments/exp040_TGV_3d_multislice_r5_finite_support_open_boundary.yaml`；
- 运行入口：继续使用 `scripts/run_exp040_multislice_forward.py`；
- 新结果只写入 `metrics.diagnostics_r5`；R0--R4 不重跑、不改 status；
- R4 comparator 固定为
  `runs/exp040_TGV_3d_multislice_r4_positive_quadrature_20260811_161412`，config hash
  `C9628F9D12663CBCA1FCC0BA3533A14313086E1326076329DB5BB62919631D7E`，metrics hash
  `F2093962EFF1C369E45145A6C61E0C600D5B6DB55E8794FE2D4F65893C09467C`，status `Passed`；
- 保留同一 TGV、同一 48x48 canonical-B phase cells、同一 25 个 physical scans、同一
  alias-controlled ASM 和 positive midpoint quadrature；不引入 finite illumination 或 full-wave。

### R5.2 冻结 support 与 propagation branches

R4 已给出 `q4->q8 I_stack=3.969993e-4`，因此 R5 固定使用已受控且成本较低的 `q=4`，不再
运行 q8。node `dx=0.125 um`，base FOV 为 `192 um`、shape `1536^2`，比较 ROI 仍为 native
`128^2`。

finite B 冻结为居中的 `96 um x 96 um` 编码方形，内部是同一 48x48 phase cells，外部
transmission 严格为 `1+0j`。scan 平移 `B-1` modulation，窗口外 constant fill 为零；不得用
`np.roll` 把另一侧内容带回。

同一 base probe 下冻结三个比较：

1. periodic B + base circular alias-controlled BC；
2. finite B + base circular alias-controlled BC；
3. finite B + reference-plus-residual open BC padding series。

open branch 将 `P_B=P_0+delta_P_B`，并传播
`H_BC(P_0) + H_BC(delta_P_B + P_B*(B_s-1))`。只对局域 residual 零 pad；homogeneous
plane background 在完整 grid 上生成。padding series 冻结为：

```text
FOV [um]: [192, 288, 384]
node shapes: [1536^2, 2304^2, 3072^2]
acceptance pair: 288 -> 384 um
detector quadrature: q4 positive midpoint average
```

### R5.3 冻结 metrics、gates 与状态

- open `I_stack` relative-to-384 series 与 `288->384 relative L2 <=0.05`；
- base `delta_P_B` 外 `16 um` boundary-ring energy fraction `<=0.05`；
- periodic-circular vs finite-circular = support effect；
- finite-circular vs finite-open-384 = boundary effect；
- periodic-circular vs finite-open-384 = combined effect；
- 三种 effect 均以既有 `>0.05` 标记 material，只用于归因，不事后选 truth；
- support/exterior、probe decomposition、homogeneous-background crop、constant/sum 等 controls
  `<=1e-12`；
- `384 um` scan 0 determinism `<=1e-14`；
- 所有 weights/arrays finite、weights nonnegative/sum-one、detector outputs nonnegative；
- 禁止保存 full node fields 或 full node stacks。

hard controls 失败为 `Failed`；hard controls 通过但 residual containment 或 padding convergence
任一未通过为 `Inconclusive`；二者都通过才为 `Passed`。不得在看见结果后改变 q、support、
exterior transmission、padding series、ROI、acceptance pair、edge ring 或阈值。

### R5.4 冻结判读与产物

| support effect | boundary effect | R5 status | 允许定位 |
|---|---|---|---|
| non-material | non-material | Passed | finite-B/wrap 不是既有 detector floor 主因 |
| material | non-material | Passed | infinite periodic B 是主要 boundary contributor |
| non-material | material | Passed | circular BC wrap 是主要 boundary contributor |
| material | material | Passed | 两者都贡献，不得只修一项后宣称闭合 |
| 任意 | 任意 | 非 Passed | 停止归因，先处理 containment/convergence/hard control |

R5 新图冻结为：

```text
r5_open_boundary_convergence.png
r5_support_boundary_effects.png
r5_detector_comparison.png
```

本次 run 只要求原 8 图加 R5 3 图，共 11 图。HDF5 仍保持 7 个 `/entry` children，只增加
compact `/entry/metrics/diagnostics_r5/...`。

- [ ] finite support/exterior 与 constant-boundary shift tests；
- [ ] residual decomposition、padding geometry、q4 positive detector tests；
- [ ] official/tiny config、streaming、runner/HDF5/11-figure contract；
- [ ] 正式运行前锁定本文 R5 prefix；
- [ ] 运行后只在 EOF 追加 R5 执行记录与模型建议。

### R5.5 运行前差异分母约定

为避免实现时猜测，三个 materiality difference 在任何 R5 结果产生前固定为 unaligned
relative L2，并使用每个比较中物理边界更显式的后一分支作 denominator：support effect 为
`relative_l2(periodic_circular, finite_circular)`，boundary effect 为
`relative_l2(finite_circular, finite_open_384)`，combined effect 为
`relative_l2(periodic_circular, finite_open_384)`。不得在结果后交换分母或做 global-scale/
phase alignment。

### R5.6 运行前 base-equivalence control 与归因边界

R5 的 `192 um` open 分支没有额外 padding；按线性分解，它必须与 finite-B circular-192
分支逐义等价。正式运行前新增 hard control：
`relative_l2(finite_open_192, finite_circular_192) <= 1e-12`。该项只检验实现代数，不改变
任何物理 branch、FOV、ROI 或 materiality gate。

另固定解释边界：本文表格中的短标签 “circular wrap/boundary effect” 实际测量
finite-circular-192 与 finite-open-384 的完整差异，包含局域 residual 的 periodic-image 去除，
也包含已冻结 Matsushima alias-control support 随 padding FOV 的一致变化。R5 可以判断现有
circular finite-FOV detector path 是否足够，但不得把该 difference 的全部数值强行解释为纯
periodic-image wrap 百分比。若该项 material 且需要进一步拆分，应在 R5 后另行预注册
common-passband control，不能事后回写 R5。

### R5.7 正式运行前锁

正式 R5 运行前，本文前 `80271` bytes 的 SHA256 固定为
`28910E760357761D500989AA3C8AE954C6EAFF72B660FF28D08BEFD524CEAEED`；计划配置源文件 SHA256
为 `B6ACDB93C24B81EDBFFA0CE04B02455909EF9293E468AD47E088711AE392CDE1`。之后只允许在 EOF
追加执行记录，不得修改该 prefix 内的 support、padding、分母、control、gate 或判读。

---

## exp040 R5 执行记录：finite B 是 material contributor

### R5.E1 正式 run 与锁复核

```text
run: runs/exp040_TGV_3d_multislice_r5_finite_support_open_boundary_20260811_163555
run_state: complete / artifacts_validated=true
R5 status: Passed
legacy experiment_status: Inconclusive（未被 R5 覆盖）
run config SHA256: A0EC579CDEB7BDB474CC3174A61FE3D3CAC8188329902E79BEF104C0F8C5249B
metrics SHA256: 41AA3522B146EF4063EA65DCFEB707E25E01B17688EBF3036680DE9205A85C28
HDF5 SHA256: 4065E2CB7449F5AA97BF47C9859AC020CC874D76C97BF6D58011DF985715EC30
```

正式运行后复核本文前 `80271` bytes SHA256 仍为
`28910E760357761D500989AA3C8AE954C6EAFF72B660FF28D08BEFD524CEAEED`，与 R5.7 锁定值一致。

### R5.E2 open padding 收敛与 source containment

| open residual FOV | 相对 384 um 的 `I_stack` relative L2 |
|---:|---:|
| 192 um | `0.0326617` |
| 288 um | `0.0148989` |
| 384 um | `0`（self-reference） |

预注册 acceptance pair `288->384 um = 0.0148989 <=0.05`，通过。base
`delta_P_B` 外 `16 um` boundary-ring energy fraction 为 `0.0295802 <=0.05`，source
containment 也通过。因此 R5 的 open residual padding comparison 在当前 5% 标准下已受控。

### R5.E3 support、boundary 与 combined effect

| 注册差异 | relative L2 | 5% materiality | 判读 |
|---|---:|---:|---|
| periodic-circular vs finite-circular | `0.381145` | material | finite-support effect |
| finite-circular vs finite-open-384 | `0.0326617` | non-material | registered boundary effect |
| periodic-circular vs finite-open-384 | `0.387506` | material | combined effect |

按 R5.4 冻结表，正式 interpretation code 为 `finite_support_material`。在“同一 `96 um`
phase-cell realization、外部透明、同一 q4 detector 与同一 physical scan”这一明确条件下，
无限 periodic B 是 detector-path 的 material boundary-model contributor；将 finite B 的 residual
传播从 circular-192 扩到 open-384 只产生 `3.27%` 差异，未达到 5% materiality threshold。

该结论不能改写成“38.1% 就是 R1 floor 的已解释比例”：不同 diagnostic 的 denominator 与
operator 不同。它证明 periodic-B 假设足以显著改变 detector prediction，但不证明预注册的
`96 um` sharp square/transparent exterior 就是真实样品 B。按 R5.6，`3.27%` boundary difference
还包含 padding-dependent alias-control support，不能全部强称为纯 periodic-image wrap。

### R5.E4 hard controls

- canonical phase-cell mapping error：`1.241267e-16 <=1e-12`；
- finite support mapping/exterior errors：均为 `0`；unit-modulus error `2.220446e-16`；
- 最大 scan 后 finite-support margin：`39 um >0`；
- probe decomposition relative L2：`3.114765e-18`；
- open-192 vs finite-circular-192 base equivalence：`3.680114e-16 <=1e-12`；
- homogeneous-background padding consistency 最大 relative L2：`4.440892e-16`；
- constant error：`0`；quadrature sum error 最大 `2.058948e-16`；
- 384 um scan 0 determinism：`0 <=1e-14`；
- weights finite/nonnegative/sum-one，所有 arrays finite，所有 detector outputs nonnegative；
- `hard_checks_pass=true`，结合 convergence/containment 通过，故 R5 为 `Passed`。

### R5.E5 artifacts、测试与 Git

- 正式 run 有原 8 图与 R5 3 图，共 11 张 PNG；三张 R5 图均已人工检查；
- 正式 effects 图的长 legend 在右侧略裁切，但横轴 index、曲线和 gate 可读；后续代码已缩短
  label，未覆盖正式 run，也未改变任何 metrics；
- HDF5 大小约 `29.26 MB`，`/entry` 仍只有 `config_yaml`、`data`、`instrument`、
  `metadata`、`metrics`、`sample`、`truth`；
- 只增加 compact `/entry/metrics/diagnostics_r5/...`，没有
  `/entry/truth/diagnostics_r5`，没有 full node field/stack；
- R5 tiny tests：`11 passed`；最终全量测试：`184 passed in 24.46s`；本次修改范围 Ruff：
  `All checks passed`；
- Git 保持 unstaged；未 commit、push 或创建 PR。

### R5.E6 模型缺陷与建议（本次不继续改模型）

1. **最优先校准真实 sample B support。** R5 的 38.1% support effect 很大，但它依赖
   `96 um` sharp square、透明 exterior 和 phase-only/unit-modulus 的预注册定义。下一步应以真实
   B 的有效编码面积、基底 exterior transmission、边缘过渡和 illumination footprint 替换该理想
   support；若引入实测 B 或新数据源，应新开实验任务，不能把 R5 的 finite square 当作 truth。
2. **若必须纯拆 periodic image 与 alias-mask support，再做 common-passband control。** R5 已说明
   registered open path non-material，但没有把其 3.27% 分解成 wrap 与随 FOV 变化的 Matsushima
   support。只有需要低于 5% 的更细归因时，才另行预注册固定 continuous-frequency passband；
   不建议事后在 R5 中选择对结论更有利的 mask。
3. **随后检查 subvoxel interface，而不是立即放弃 multi-slice。** A 的 voxel-center binary sidewall
   仍会产生 staircase 高频。可预注册 volume-fraction 或 level-set interface，在保持相同几何、
   折射率、scan、B support 和 detector operator 下比较；这仍属于数值离散 refinement。
4. **单向标量模型仍有硬物理边界。** 它不能产生 Fresnel reflection、backward wave、sidewall
   multiple scattering 或 polarization；玻璃/空气法向强度反射量级约 4%，与当前 5% 诊断尺度
   相近。只有 B support、interface、illumination/detector calibration 都受控后仍有结构化偏差，
   才建议依次比较 bidirectional BPM、Lippmann--Schwinger，最后才承担 vector FEM/FDTD 成本。
5. **当前仍不能宣称腰径 forward 问题已通过。** R4/R5 只关闭 detector quadrature 与本次边界
   diagnostic；原 `experiment_status` 和 waist visibility 仍保持 `Inconclusive`。下一项模型升级
   必须继续使用同一 canonical geometry 和预注册指标，不能用新的 forward branch 回写旧结论。

---

## exp040 R6 预注册：sample-B support sensitivity envelope

### R6.1 身份与非 empirical-calibration 声明

- 理论说明：`docs/theory_notes/exp040_r6_sample_b_support_sensitivity.md`；
- 计划配置：`configs/experiments/exp040_TGV_3d_multislice_r6_b_support_sensitivity.yaml`；
- 运行入口：继续使用 `scripts/run_exp040_multislice_forward.py`；
- 新结果只写入 `metrics.diagnostics_r6`；R0--R5 不重跑、不改 status；
- R5 provenance 固定为
  `runs/exp040_TGV_3d_multislice_r5_finite_support_open_boundary_20260811_163555`，run-config hash
  `A0EC579CDEB7BDB474CC3174A61FE3D3CAC8188329902E79BEF104C0F8C5249B`，metrics hash
  `41AA3522B146EF4063EA65DCFEB707E25E01B17688EBF3036680DE9205A85C28`，status `Passed`；
- 当前没有实测 B support 数据，所以 R6 是 virtual sensitivity envelope，不能写成真实 B 已标定。

### R6.2 冻结 assumptions 与 case family

R6 只替换“B 无限周期延拓/scan wrap”这一假设族：每个 finite case 的 `B-1` 具有有限方形
support，外部 transmission 为 `1+0j`，scan 使用 constant-zero modulation shift。TGV A、同一
48x48 canonical-B phase cells、同一 25 scans、q4 positive detector、192 um FOV、
alias-controlled AB/BC 全部固定。

```text
support width [um]: [80, 96, 112]
phase-edge taper width [um]: [0, 4, 8]
case order: width-major full factorial, 9 cases
nominal: 96 um / hard edge (0 um taper)
quadrature: q4, dx=0.125 um
node grid: 1536^2 / 192 um
native ROI: 128^2
```

有限 support 内部由同一 canonical phase-cell realization 居中裁剪/周期延拓；不得重新抽随机 B。
edge taper 采用 separable raised-cosine phase weight，`B=exp(i*w*phi)`，保持 unit modulus；
support 外严格为一。大于 96 um 的部分只是同一 realization 的虚拟延拓，不冒充实测新区域。

### R6.3 冻结比较、分母与 gates

- periodic comparator：同一 192 um periodic B + periodic shift + circular alias-controlled BC；
- finite cases：constant-exterior shift + 同一 circular alias-controlled BC；
- support effect：`relative_l2(periodic_circular, finite_case_circular)`，finite case 为 denominator；
- nominal sensitivity：`relative_l2(finite_case, nominal_96um_hard)`，nominal 为 denominator；
- 禁止 phase/scale/spatial alignment；
- 9 个 support effects 均以既有 `>0.05` 标记 material；
- geometry/exterior/unit-modulus/taper/constant/sum controls `<=1e-12`；
- nominal scan 0 determinism `<=1e-14`；
- 所有 arrays/weights finite，weights nonnegative/sum-one，detector outputs nonnegative；
- 禁止保存 full `1536^2` fields 或 full node stacks。

hard controls 失败为 `Failed`；hard controls 通过且全部 9 cases material 为 `Passed`；任一 case
non-material 为 `Inconclusive`。不得看结果后缩小 envelope、删除 case 或选择误差最小的 case
作为“真实标定值”。

### R6.4 冻结产物与下一步条件

```text
r6_b_support_effect_matrix.png
r6_b_support_nominal_difference.png
r6_b_support_selected_detector.png
```

本次 run 只要求原 8 图加 R6 3 图，共 11 图。HDF5 仍保持 7 个 `/entry` children，只增加
compact `/entry/metrics/diagnostics_r6/...`。

- R6 Passed：允许随后预注册 subvoxel TGV interface comparison；nominal 96 um hard-edge 只作
  固定工作模型，并携带 support-envelope uncertainty；
- R6 Inconclusive：先获取 B 的实际有效面积/边缘信息，再升级 A interface；
- 无论状态如何，R6 都不能替代真实显微/相位/透射标定。

- [ ] support/taper/exterior unit tests；
- [ ] official/tiny config、same-B/same-scan、streaming tests；
- [ ] runner/HDF5/11-figure contract；
- [ ] 正式运行前锁定本文 R6 prefix；
- [ ] 运行后只在 EOF 追加执行记录。

### R6.5 运行前 selected-detector figure 规则

`r6_b_support_selected_detector.png` 固定显示 scan 0 的 periodic comparator、nominal
`96 um/0 um`、以及按已注册 support-effect 数值取得的最小 effect case 和最大 effect case。
min/max 选择只用于展示预注册 envelope 两端，所有 9 个 case metrics 仍完整保存；不得只保留
有利 case 或用该图选择“真实”support。

### R6.6 运行前 nominal provenance reproduction

R6 nominal `96 um/0 um` 与 R5 support-effect branch 的 forward 定义完全相同。正式运行前新增
hard control：nominal support effect 相对 R5 frozen `0.38114505695745043` 的 scalar relative
error `<=1e-12`。该项只防止 comparator 漂移，不改变任何 support/taper case、threshold 或状态
判读。

### R6.7 正式运行前锁

正式 R6 运行前，本文前 `91302` bytes SHA256 固定为
`B7955501A7D70D1164FC8192C961E79D81ADE7A7B14E9E690C3CDD78CF2F8B8B`；计划配置源文件 SHA256
为 `9EA687ED330FB06964D5A59D6F1A8068827C1C9116649756F9D82F61973F1610`。之后只允许在 EOF
追加执行记录，不得修改该 prefix 内的 envelope、taper、分母、gate 或判读。

---

## exp040 R6 执行记录：periodic-B materiality 对 support envelope 稳健

### R6.E1 正式 run 与锁复核

```text
run: runs/exp040_TGV_3d_multislice_r6_b_support_sensitivity_20260811_200120
run_state: complete / artifacts_validated=true
R6 status: Passed
legacy experiment_status: Inconclusive（未被 R6 覆盖）
run config SHA256: 258A146A26C5A419569EAD9C740279EBCE4D48F207F5A64AEFC714D2D0FB67E8
metrics SHA256: 5813692089C374892D961152250225F71CE05843828A5F8D9FBE5CEBA33B987A
HDF5 SHA256: 26AD3384EB8D8F6A5E9FD79A8E9A4AD9BB86E6A6014A121E2763A4ADD31341B0
```

正式运行后复核本文前 `91302` bytes SHA256 仍为
`B7955501A7D70D1164FC8192C961E79D81ADE7A7B14E9E690C3CDD78CF2F8B8B`，与 R6.7 锁定值一致。

### R6.E2 periodic-vs-finite support-effect matrix

数值为 `relative_l2(periodic_circular, finite_case_circular)`，列为 phase-edge taper：

| support width | 0 um hard edge | 4 um taper | 8 um taper |
|---:|---:|---:|---:|
| 80 um | `0.408647` | `0.410881` | `0.408686` |
| 96 um | `0.381145` | `0.385429` | `0.391952` |
| 112 um | `0.343299` | `0.353436` | `0.362793` |

全部 9 cases 均显著超过预注册 `0.05` materiality threshold。envelope minimum 为
`0.343299`（112 um / hard edge），maximum 为 `0.410881`（80 um / 4 um taper），span 为
`0.0675819`。按冻结状态逻辑，R6 `Passed`，interpretation code 为
`periodic_b_materiality_robust_over_support_envelope`。

nominal `96 um/hard-edge` support effect 为 `0.3811450569574505`；相对 R5 frozen 值的误差仅
`1.456431e-16 <=1e-12`，nominal provenance reproduction 通过。

### R6.E3 finite cases 相对 nominal 的敏感性

数值为 `relative_l2(finite_case, nominal_96um_hard)`：

| support width | 0 um hard edge | 4 um taper | 8 um taper |
|---:|---:|---:|---:|
| 80 um | `0.141636` | `0.158785` | `0.171296` |
| 96 um | `0` | `0.061103` | `0.089913` |
| 112 um | `0.149654` | `0.122165` | `0.082699` |

最大 nominal sensitivity 为 `0.171296`。因此 R6 给出两个同时成立、不能混写的结论：

1. **robust qualitative conclusion：** 无限 periodic B 在整个预注册 envelope 中都是 material
   model contributor，R5 的主要定性定位不是单一 `96 um` hard edge 的偶然结果；
2. **unresolved quantitative calibration：** finite-support cases 彼此仍可差 `6%--17%`，仅靠仿真
   不能确定真实 B 的 support width 或 edge taper，也不能把 nominal case 提升为 empirical truth。

### R6.E4 hard controls

- finite support mapping/exterior/taper range/taper endpoint errors：均为 `0`；
- unit-modulus error 最大 `2.220446e-16 <=1e-12`；
- 最大 support 与最大 scan 后最小 FOV margin：`31 um >0`；
- positive quadrature constant error：`0`；sum identity 最大 relative error：
  `2.082429e-16`；
- nominal scan 0 determinism：`0 <=1e-14`；
- 所有 arrays finite、detector outputs nonnegative；
- `hard_checks_pass=true`，全部 9 cases material，故 R6 为 `Passed`。

### R6.E5 artifacts、测试与 Git

- 正式 run 有原 8 图与 R6 3 图，共 11 张 PNG；三张 R6 图均已人工检查；
- HDF5 大小约 `29.26 MB`，`/entry` 仍只有 `config_yaml`、`data`、`instrument`、
  `metadata`、`metrics`、`sample`、`truth`；
- 只增加 compact `/entry/metrics/diagnostics_r6/...`，没有
  `/entry/truth/diagnostics_r6`，没有 full `1536^2` fields/stacks；
- R6 tiny tests：`9 passed`；正式运行前最终全量测试：`193 passed in 36.18s`；修改范围 Ruff：
  `All checks passed`；
- Git 保持 unstaged；未 commit、push 或创建 PR。

### R6.E6 对 subvoxel TGV interface 的进入条件

R6 已满足预注册的 qualitative robustness 条件，因此下一步可以预注册 subvoxel TGV interface
comparison。为保持研究问题不漂移，建议固定 nominal `96 um/hard-edge` finite B、q4 positive
detector、同一 scan 和 R5 open-boundary branch，仅替换 A 的 voxel-center binary interface 为
volume-fraction 或 level-set interface。

但 nominal B 只能作为固定 working model：R6 的 `17.13%` envelope variation 必须作为当前
forward-model uncertainty 明确保留。若下一阶段要求绝对预测精度优于约 5%，仍需要真实 B 的
有效面积/边缘相位或透射数据；不得通过选择某个 R6 case 来降低 subvoxel residual。

---

## exp040 R7 预注册：subvoxel TGV interface comparison

### R7.1 目的与进入依据

- 理论说明：`docs/theory_notes/exp040_r7_subvoxel_tgv_interface.md`；
- 计划配置：`configs/experiments/exp040_TGV_3d_multislice_r7_subvoxel_interface.yaml`；
- 运行入口：继续使用 `scripts/run_exp040_multislice_forward.py`；
- R6 provenance：
  `runs/exp040_TGV_3d_multislice_r6_b_support_sensitivity_20260811_200120`，metrics SHA256
  `5813692089C374892D961152250225F71CE05843828A5F8D9FBE5CEBA33B987A`，status `Passed`；
- legacy `experiment_status=Inconclusive`、R0--R6 metrics 和结论全部保留，不重跑、不覆盖。

R1 的 binary voxel-center branch 在 `dz 0.5 -> 0.25 um` 得到 `I_stack=0.207719`，在
`dx 0.25 -> 0.125 um` 得到 `I_stack=0.075441`，均未通过既有 5% gate。R2--R6 已经把外部
alias、detector sampling/quadrature 和 B boundary 单独诊断。R7 因而只改变 sample-A 曲面与
lateral pixels 相交时的占据表示，检查 staircase/interface 是否为 material contributor；它不是
真实 A/B calibration，也不改变 scalar/unidirectional forward physics。

### R7.2 冻结 interface cases 与数值定义

```text
subpixel factors q: [1, 2, 4, 8]
nodes: q x q staggered midpoint nodes per lateral voxel
weights: uniform, nonnegative, sum one
air fraction: mean of analytic TGV indicator at subnodes
n_eff: n_glass + f_air * (n_air - n_glass)
q1 identity: exact existing voxel-center binary definition
```

同一连续 TGV geometry、materials、slice-center `D(z)` 和 exact slice widths 全部固定。正式 A
grid 为 `256² @ dx=0.25 um`、FOV `64 um`、`dz=0.25 um`（400 slices）。R7 不加 axial
subnodes，不把 z quadrature 与 lateral interface 混在同一次归因中。subvoxel fraction 是解析
indicator 的正权重 cell average，不是把孔壁声明成真实 effective medium，也不加入 Fresnel
reflection、backward wave 或 polarization。

### R7.3 冻结 B、scan 与 detector/open path

- B：同一 48x48 canonical phase cells、`2 um` feature、finite `96 um` hard-edge；外部
  transmission `1+0j`；
- scan：同一 25 个 physical positions，`B-1` 使用 constant-exterior shift；
- detector nodes：q4 positive midpoint；
- final detector prediction：沿用 R5 的 open `384 um` reference branch 与 alias-controlled AB/BC；
- native ROI：centered `128²`；
- 每个 q case 使用同一 A-exit-to-node mapping、B、scan、transfer 和 detector weights。

正式实现允许流式生成 400 个 sample-A slices，禁止为每个 q 保留完整 `400x256x256`
volume。不得保存 full detector-node stacks。

### R7.4 冻结 comparisons、分母与状态

对 `U_A_exit`、`P_B`、`I_stack` 分别保存相对 q8 series；final pair 为 `q4 -> q8`，binary
effect 为 `q1 -> q8`，均以 q8 为 denominator，禁止 global phase、scale 和 spatial alignment。

- final-pair convergence：三个输出分别 `<=0.05`；
- binary materiality：各输出分别以 `>0.05` 标记 material；
- algebra controls `<=1e-12`；determinism `<=1e-14`；
- R6 maximum nominal B variation `0.17129597874704286` 只保存为独立
  `model_uncertainty_context`，不得与 R7 difference 合并、扣除或用作 gate。

hard controls 包括 q1 identity、fraction/index bounds、subnode-count identity、slice-width/geometry、
same input provenance、positive detector quadrature、q8 scan-0 determinism、finite/nonnegative。
解析空气体积与 discrete fraction volume 的差异只报告 q-series，不追加未登记阈值。

R7 hard checks 失败为 `Failed`；hard checks 通过但 q4->q8 任一输出失败为 `Inconclusive`；三项
通过则 `Passed`。Passed 后 binary effect material，解释为 staircase material；Passed 后 binary
effect non-material，解释为当前网格下 non-material。R7 status 不覆盖 legacy experiment status，
也不直接重新计算 waist visibility。

### R7.5 冻结产物与下一步逻辑

```text
r7_interface_fraction_slice.png
r7_interface_convergence.png
r7_interface_selected_detector.png
```

原 8 图保留，R7 新增 3 图，共 11 图。HDF5 仍保持既有 7 个 `/entry` children，只增加 compact
`/entry/metrics/diagnostics_r7/...`，不增加 full volume/node stack 或 diagnostic truth group。

- R7 Passed 且 binary material：用 converged q8 interface 另行预注册 axial/lateral 和 detector
  waist-visibility reevaluation；
- R7 Passed 且 binary non-material：interface 不是当前最大数值 contributor，继续评估 scalar
  interface physics/真实 calibration；
- R7 Inconclusive：先加密 subpixel quadrature；
- R7 Failed：修复 control，禁止解释结果。

- [ ] R7 config 与 prefix hash 锁；
- [ ] fraction/q1 identity/volume/bounds unit tests；
- [ ] tiny runner、HDF5、11-figure contract；
- [ ] 修改范围 Ruff 与全量 pytest；
- [ ] 正式 timestamped run 与 artifacts 人工复核；
- [ ] 运行后只在 EOF 追加执行记录和下一步建议。

### R7.6 正式运行前锁

正式 R7 运行前，本文件前 `101032` bytes 的 SHA256 锁定为
`DB09065045D1636A5206E5653B0DD70D919FBC7D3B76ACD950D4939834458458`；计划配置源文件 SHA256
为 `C2CAAEF1231A6C3C71DC27525BE47F090BDD90518CCFD96D0531AAE0863FA2C3`。锁前专项与全量验证分别为
`10 passed` 和 `204 passed`，修改范围 Ruff 为 `All checks passed`。从此处起不得修改前缀内的 q cases、
界面定义、B/open path、比较分母、阈值或状态逻辑；正式结果及必要的非研究定义实现修复只能追加在 EOF。

---

## exp040 R7 执行记录：subvoxel interface 已收敛，但 detector 端 binary effect 小于 5%

### R7.E1 正式 run 与锁复核

```text
run: runs/exp040_TGV_3d_multislice_r7_subvoxel_interface_20260813_011329
run_state: complete / artifacts_validated=true
R7 status: Passed
legacy experiment_status: Inconclusive（未被 R7 覆盖）
run config SHA256: 5E760D55BFD10E6EFE4BFE68BED0F7E33A90CEE1EDC69E33EF76B5DDE0FF852F
metrics SHA256: F7C26CC9B14778704C4F14B660515A8CD710B750A5C1E5EB0A868969CB4324BE
HDF5 SHA256: 34D10D9FB506BD76FE887041788B803D74A1138E0E434140878686E649258551
```

正式运行后复核：本文件前 `101032` bytes SHA256 仍为
`DB09065045D1636A5206E5653B0DD70D919FBC7D3B76ACD950D4939834458458`，计划配置源文件 SHA256 仍为
`C2CAAEF1231A6C3C71DC27525BE47F090BDD90518CCFD96D0531AAE0863FA2C3`，均与 R7.6 一致。本次未在看见
结果后修改 q cases、分母、阈值、B/open path 或状态逻辑。

### R7.E2 interface q-series 与正式判读

所有数值均为无 phase/scale/spatial alignment 的 `relative_l2(q, q8)`，q8 为分母：

| output | q1 | q2 | q4 | q8 |
|---|---:|---:|---:|---:|
| `U_A_exit` | `0.182189` | `0.0627938` | `0.0198499` | `0` |
| `P_B` | `0.0144609` | `0.00395651` | `0.000498961` | `0` |
| `I_stack` | `0.00824131` | `0.00202584` | `0.000250482` | `0` |

预注册 final pair `q4 -> q8` 的三项均通过 `<=0.05`：A-exit 为 `1.985%`，B-plane probe 为
`0.0499%`，detector stack 为 `0.0250%`，故 R7 `Passed`。这说明 q8 已足以作为当前 scalar phase-screen
中的横向 interface quadrature reference；无需为了得到更有利结果再增加 q 或改变 interface 定义。

binary `q1 -> q8` effect 对 `U_A_exit` 为 `18.22% >5%`，是 material；传播到 `P_B` 后为
`1.446%`，到 `I_stack` 后为 `0.824%`，两者均 non-material。预注册 interpretation code 因而为
`binary_interface_material_for_at_least_one_output`。准确含义是：voxel-center staircase 显著改变 A 出口复场，
但在当前固定 B、scan、q4 detector 与 384 um open path 下，它不是 detector 端 5% 以上 floor 的主因。
不得把 `U_A_exit` 的 18.22% 写成 detector error，也不得因 detector effect 较小而说 binary interface 在所有
平面都可忽略。

### R7.E3 hard controls 与 R6 uncertainty 分离

- q1 对旧 binary indicator 最大绝对误差：`0`；streamed homogeneous 与既有 homogeneous operator 的
  relative L2：`2.41075e-13 <=1e-12`；
- q1/q2/q4/q8 fraction bounds、index bounds、subnode-count identity error 全部为 `0`；
- midpoint-z analytic air-volume reference 的离散相对误差分别为
  `3.03101e-5`、`2.34809e-4`、`2.41847e-5`、`9.93145e-6`；该项按预注册只报告、不设新 gate；
- finite-B mapping/exterior error 均为 `0`，unit-modulus error 为 `2.22045e-16`；
- detector node geometry normalized error `5.42101e-14`，constant error `0`，最大 quadrature-sum relative
  error `1.98481e-16`；q8 scan 0 determinism `0`；
- 所有数组 finite、所有 intensity 非负，`hard_checks_pass=true`；
- R6 maximum nominal B variation `0.17129597874704286` 只保留在
  `model_uncertainty_context`，`combined_with_r7_metrics=false`。它没有与任何 R7 数值相加、相减或合成 gate。

### R7.E4 artifacts、测试、HDF5 与 Git

- 正式 run 有原 8 图加 R7 3 图，共 11 张 PNG；三张 R7 图已人工检查；
- HDF5 约 `29.26 MB`，`/entry` 仍只有 `config_yaml`、`data`、`instrument`、`metadata`、`metrics`、
  `sample`、`truth`；仅新增 compact `/entry/metrics/diagnostics_r7/...`，没有
  `/entry/truth/diagnostics_r7`，没有 full q-volume 或 detector-node stack；
- R7 专项 runner/figure/HDF5 tests：`10 passed`；正式运行前全量：`204 passed in 33.98s`；修改范围 Ruff：
  `All checks passed`；
- Git 保持 unstaged；未 commit、push 或创建 PR。

### R7.E5 对 exp040 主目标的影响与下一步

R7 关闭了一个重要的数值归因问题：在固定的 scalar/unidirectional multi-slice 模型中，横向 subvoxel
interface quadrature 已收敛；binary staircase 虽会显著改变 A-exit 复场，却不足以解释 detector 端原有的
大 floor。因此 `exp040` 仍不能宣称腰径 forward 问题已经通过，legacy `Inconclusive` 保留。

下一步建议在本实验中另行预注册 **R8：使用 q8 interface 重新做 sample-A axial/lateral pair 与 detector
waist-visibility evaluation**。其目的不是换模型，而是把 R1 的 axial/lateral floor 从 binary q1 更新到已收敛
的 q8 representation，确认原 `dz=0.25 um`、lateral sampling 与腰径信号之间的最终关系。R8 必须继续固定
nominal finite B、物理 scan、q4 positive detector 和 384 um open path，并明确携带 R6 的 17.13% B-support
uncertainty；阈值与对比对必须在运行前写入，不能复用本次结果事后挑选。

如果 R8 的 detector axial/lateral floor 仍明显高于 5%，则当前剩余问题已不应继续归咎于 lateral staircase；
优先级依次为：真实 A/B/illumination/detector calibration，以及另行编号的物理模型比较。当前 scalar
unidirectional model 无法表示 Fresnel reflection、backward wave、sidewall multiple scattering 和 polarization；
只有数值采样与实际标定均受控后仍存在结构化偏差，才建议预注册 bidirectional BPM 或
Lippmann--Schwinger comparator。不得为了闭合 exp040 直接把这些更复杂模型当成未验证 truth。
