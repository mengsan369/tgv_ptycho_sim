# exp040：3D TGV multi-slice forward model

```text
Scientific status: Inconclusive
Work status: Frozen / Paused
Frozen date: 2026-08-17
Last completed diagnostic stage: R14B
Latest diagnostic result: Failed
reference_validated: false
full_tgv_reference_authorized: false
Freeze reason: 近期优先启动 exp050 复原研究，之后再考虑更高级的电磁学和 reference validation
Document role: current primary design, modeling, result, and decision record
Reorganized: 2026-08-13
```

本文是 `exp040` 今后的默认阅读、分析和追加维护入口。它把原实验记录中的全部科学主线、
预注册条件、正式结果和证据身份，与原先分散在 `exp040*` theory notes 中的实验专属建模
说明合并为一份自包含文档。

原始时间顺序记录已原样冻结为
`docs/experiment_design/exp040_TGV_3d_multislice_forward_old.md`。迁移没有改变其任何字节：
文件大小为 `107124 bytes`，SHA256 为
`434E9B25837EB78309153855DEDC0922A864964BAF9AC746F9C18FBEFC518646`。原文件中的历史
prefix locks、checkbox、逐次执行记录和措辞仍是审计依据；本次重组不追溯修改任何实验条件、
阈值、状态或结论。

本文件以后继续遵守先预注册、后运行、再追加结果的顺序。文档重组本身不构成新的实验结果。

## 1. 研究问题与当前答案

### 1.1 exp040 要回答的问题

本实验研究轴对称空气填充 TGV sample A 的三维标量 multi-slice forward model：

1. 给定连续几何 $D(z)$、玻璃/空气折射率和入射场，离散 split-step 是否在轴向、横向和
   有限 FOV 上受控；
2. A 出射场传播到 B 后，经同一编码样品 B 的 overlapping scan，能否得到数值稳定的 detector
   intensity；
3. `D_waist ±2 um` 引起的 detector 变化是否稳定高于数值传播、边界、采样和 detector operator
   的误差底；
4. 若不满足，误差主要来自 sample-A voxelization、FFT/ASM、周期边界、B support、detector
   quadrature，还是当前标量单向模型本身没有表示的物理。

本实验只执行 forward simulation 和 simulation-evaluation controls，不执行 reconstruction，也不从
intensity 反演 $D(z)$ 或 $D_\mathrm{waist}$。

### 1.2 当前答案

- R0 建立了可运行、可测试、可写入 HDF5 的三维 multi-slice baseline；几何和代数 hard controls
  通过，但 axial/lateral/FOV convergence 和 detector visibility 未通过，因此总状态为
  `Inconclusive`。
- R1 加密到 `dz=0.25 um`、lateral `dx=0.125 um` 并扩大外部 FOV 后仍未收敛；它把主要高 floor
  定位到 A-exit 之后的 external propagation/B/detector path，同时保留 sample-A axial/lateral
  离散问题。
- R2 表明 `96 um` canonical-B 周期与 `112/128 um` FOV 不相容不是唯一原因；transfer sampling
  alias 是 material contributor，但 same-grid alias control 没有单独关闭 detector floor。
- R3 把 residual 定位到 B multiplication 之后的 BC propagation/detector path；其 sinc-MTF
  pixel branch 出现负强度 hard failure，不能作为物理 detector operator 接受。
- R4 的非负 staggered midpoint quadrature 在 q4 到 q8 已收敛，关闭了 detector quadrature
  sampling 这一数值问题。
- R5 表明无限 periodic B 相对有限 `96 um` 编码 support 是 material contributor；在已登记条件下，
  384 um open residual propagation 已收敛，circular/open difference 小于 5%。
- R6 表明 periodic-vs-finite B 的 materiality 对 `80/96/112 um × 0/4/8 um taper` envelope 稳健；
  但 finite cases 彼此仍可相差最高 `17.13%`，这不是实际 B 已标定。
- R7 的 q4 到 q8 subvoxel interface 已收敛。binary staircase 对 `U_A_exit` 有 `18.22%` 影响，
  但对 `P_B/I_stack` 仅 `1.45%/0.824%`；它不是当前 detector 端 5% 以上 floor 的主因。
- R8 在统一 q8/finite-B/open forward 下得到 detector 数值分量低于 5% 且腰径扰动相对 detector floor
  可见，但 raw `U_A_exit` 的 axial/lateral convergence 未闭合，因此保持 `Inconclusive`；R9 进一步关闭
  axial floor，却仍留下 external propagating passband 内的 lateral discrepancy，状态同样为 `Inconclusive`。
- R10 Stage A 的 lateral reference 诊断通过；但 R10 Stage B、R11 和 R12 的高保真标量 reference
  validation 均按各自预注册 gate 失败。它们的失败、修复审计和未执行 cross-model 的结论均按原记录保留。
- R13 的两个小型 benchmark 为 `Passed`，R14A 的 corrected axial control 也为 `Passed`；这些局部通过只关闭
  各自诊断问题，始终没有授权 full-size TGV reference。
- 最新完成的 R14B formal 为 `Failed / r14_no_scalable_scipy_solver`：两条固定 SciPy solver 路线未找到
  可接受的 scalable candidate，continuous-accuracy carrier 也未通过预注册 gate。因此
  `reference_validated=false`、`full_tgv_reference_authorized=false`。

因此，R4--R7、R10 Stage A、R13、R14A 等个别诊断可以各自为 `Passed`，但不能覆盖 exp040 的整体
科学状态，也不能覆盖最新 R14B 的 `Failed`。exp040 整体 Scientific status 必须保持 `Inconclusive`；
自 `2026-08-17` 起 Work status 为 `Frozen / Paused`。冻结不等于 `Deprecated`，也不删除、否定或追溯改写
已有通过项、失败项和原始证据。

## 2. 连续样品、坐标与物理量

### 2.1 TGV 几何

sample A 是厚度 $L=100\,\mu\mathrm m$ 的玻璃基体，其中有单个轴对称空气孔。连续直径轮廓
$D(z)$ 在 top、waist 和 bottom 之间分段线性：

$$
D(z)=
\begin{cases}
D_\mathrm{top}+
\dfrac{D_\mathrm{waist}-D_\mathrm{top}}{z_\mathrm{waist}}z,
&0\le z\le z_\mathrm{waist},\\[6pt]
D_\mathrm{waist}+
\dfrac{D_\mathrm{bottom}-D_\mathrm{waist}}
{L-z_\mathrm{waist}}(z-z_\mathrm{waist}),
&z_\mathrm{waist}<z\le L.
\end{cases}
$$

本实验固定：

```text
D_top = 30 um
D_waist = 20 um
D_bottom = 30 um
z_waist = 50 um
n_glass = 1.5
n_air = 1.0
TGV center = (0, 0)
```

腰径扰动为 `18/20/22 um`，即 baseline 的 `±2 um`。该扰动只用于理想无噪声 forward visibility，
不是检测限、精度或真实样品公差。

### 2.2 数组、单位与平面

- 二维场轴顺序：`(ny, nx)`；三维体：`(nz, ny, nx)`；
- scan position 列顺序：`(x, y)`，单位 m；
- `dx` 若为 tuple，顺序为 `(dy, dx)`；
- 长度内部使用 m，phase 使用 rad，intensity 使用 arbitrary units；
- input field 位于 A entrance boundary；`U_A_exit` 位于 A exit boundary；
- `z_AB=0.5 mm` 从 A exit 开始，`z_BC=1 mm` 从 B plane 开始，不重复计算 A 厚度。

### 2.3 关键光学与扫描参数

```text
vacuum wavelength = 532 nm
internal reference index = 1.5
external medium index = 1.0
baseline A/detector grid = 128 x 128 @ 0.5 um
baseline A dz = 1 um
canonical B = 96 um x 96 um, 48 x 48 phase cells
B physical feature = 2 um
B phase range = 0.8 rad
B seed = 20260840
scan = 5 x 5, step 4 um, seeded integer jitter
scan seed = 20260841
```

所有 refinement 尽量复用同一 continuous TGV、同一 B phase-cell realization、同一物理 scan、相同传播
距离和 detector ROI。改变 grid 不得重新抽随机 B。

## 3. Forward model 与共同判定规则

本节的通用标量传播与 split-step 推导保留在
`docs/theory_notes/exp040_tgv_multislice_forward.md`；此处记录 `exp040` 实际采用的模型、控制和判定。

### 3.1 Slice-centered phase screen

真空波数 $k_0=2\pi/\lambda_0$。第 $j$ 层中心在 $z_j$，真实层宽为 $w_j$，并满足

$$
\sum_jw_j=L.
$$

相对参考介质 $n_\mathrm{ref}$ 的 phase screen 为

$$
T_j(x,y)=\exp\!\left[
i k_0\bigl(n_j(x,y)-n_\mathrm{ref}\bigr)w_j
\right].
$$

实现采用 centered symmetric split-step：entrance half-step、slice-center screen、相邻中心传播，最后
exit half-step。单层可写为

$$
U_\mathrm{out}=\mathcal P_{w/2}^{(n_\mathrm{ref})}
\left[T\,\mathcal P_{w/2}^{(n_\mathrm{ref})}U_\mathrm{in}\right].
$$

zero contrast 时所有 $T_j=1$，必须回到同一参考介质中的 $\mathcal P_LU_0$。关闭内部传播时，

$$
U_\mathrm{product}=U_0\prod_jT_j,
$$

用于检查 phase-product 代数 identity。所有复场比较禁止用 truth 做 global phase、complex gain、scale
或 spatial alignment。

### 3.2 Binary 与 subvoxel interface

原始 q1 voxelization 在每个 lateral voxel center 判断

$$
(x-x_0)^2+(y-y_0)^2\le [D(z_j)/2]^2,
$$

成立则赋 $n_\mathrm{air}$，否则赋 $n_\mathrm{glass}$。它精确定义了 R0--R6 的 binary baseline，但会在
曲面与 Cartesian grid 相交处产生 staircase。

R7 对 lateral pixel $C_p$ 定义空气面积分数

$$
f_{jp}=\frac{1}{|C_p|}\int_{C_p}
\mathbf 1\!\left[(x-x_0)^2+(y-y_0)^2\le r_j^2\right]dxdy,
$$

并用 q×q staggered midpoint nodes、非负等权 $1/q^2$ 估计。数值 phase-screen index 为

$$
n_{jp}=n_\mathrm{glass}+f_{jp}(n_\mathrm{air}-n_\mathrm{glass}).
$$

这是 indicator 的 cell average，不是 Maxwell effective-medium theory，也不声称孔壁存在真实混合介质。
R7 只改变 lateral interface representation，不加入 axial subnodes、reflection 或 polarization。

### 3.3 A 到 B、B scan 与 detector

外部传播链为

$$
P_B=\mathcal H_{AB}[U_{A,\mathrm{exit}}],
$$

$$
E_{B,s}=P_BB_s,
\qquad
U_{D,s}=\mathcal H_{BC}[E_{B,s}],
$$

$$
I_s=|U_{D,s}|^2.
$$

早期 baseline 使用 periodic B、integer-pixel `np.roll` 与 grid-sampled intensity；R4 以后 detector pixel
由 positive staggered midpoint area quadrature 表示，R5 以后 B 改为有限编码 support、透明 exterior，
scan 平移 `B-1` 并用 constant-zero boundary。

### 3.4 Relative L2、阈值与状态

共同差异定义为

$$
\varepsilon_Q=
\frac{\|Q_\mathrm{test}-Q_\mathrm{ref}\|_2}
{\max(\|Q_\mathrm{ref}\|_2,\epsilon_\mathrm{float64})}.
$$

除各节明确写出的非对称分母外，所有 q/FOV/refinement series 都在运行前登记 reference。共同阈值为：

```text
convergence <= 0.05
material contribution > 0.05
algebra/control error <= 1e-12
determinism <= 1e-14
detector waist signal / numerical floor >= 3
all arrays finite; all physical intensity outputs nonnegative
```

每个 R 阶段有独立状态：hard control 失败为 `Failed`；hard controls 通过但该阶段收敛目标未满足为
`Inconclusive`；全部登记条件满足为 `Passed`。后续阶段不能覆盖前序状态，更不能用新 forward branch
回写旧结果。

## 4. R0：原始 3D multi-slice baseline

### 4.1 设计与控制

R0 baseline 为 `128² @ dx=0.5 um`、`dz=1 um`。对照包括 geometry、zero contrast、single slice、
no-internal-propagation phase product、projected phase product、determinism、finite/nonnegative，以及：

- axial：`dz=[2,1,0.5] um`，final pair `1→0.5 um`；
- lateral fixed FOV 64 um：`64²@1, 128²@0.5, 256²@0.25 um`；
- FOV fixed `dx=0.5 um`：`128²/160²/192²`，共同中心 `128²` ROI；
- waist：`D_waist=[18,20,22] um`。

正式配置为 `configs/experiments/exp040_TGV_3d_multislice_forward.yaml`，运行入口始终为
`scripts/run_exp040_multislice_forward.py`。

### 4.2 正式结果

```text
formal run: runs/exp040_TGV_3d_multislice_forward_20260810_154908
run state: complete / artifacts_validated=true
status: Inconclusive
```

| 项目 | `U_A_exit` | `P_B` | `I_stack` |
|---|---:|---:|---:|
| axial final pair | `0.343362` | `0.343362` | `0.372246` |
| lateral final pair | `0.111835` | `0.112331` | `0.159475` |
| FOV final pair | `0.005373` | `0.368378` | `0.786966` |

hard controls 均通过：zero contrast `2.350435e-13`、single slice `0`、no-propagation product
`1.355924e-14`、projected discrete product `8.335707e-14`、determinism `0`。但三组 convergence
均失败。detector waist minus/plus signal 为 `0.534377/0.452668`，相对 `0.786966` floor 的最小
ratio 只有 `0.575206 <3`。

R0 因此只证明 baseline 实现的几何、平面、算子顺序与代数一致性，不能证明网格收敛或腰径可见。

## 5. R1：轴向、横向与 external padding refinement

### 5.1 为什么需要 reference-plus-residual padding

对应通用数值原理见 `docs/theory_notes/exp040_refinement_and_external_padding.md`。

plane-wave full field 在原窗口边缘不为零。直接 zero-pad full field 等价于乘以未登记的矩形 aperture，
会产生人为衍射。R1 先以同 shape、同入射、同厚度的 homogeneous field 定义

$$
\delta U_A=U_{A,\mathrm{exit}}-U_{A,\mathrm{ref}},
$$

再在 padded grid $p$ 上构造

$$
U_A^{(p)}=U_{A,\mathrm{ref}}^{(p)}+\mathcal E_p[\delta U_A].
$$

只对近似局域的 scattered residual 做零嵌入，homogeneous background 在整个新网格重新生成。该操作
不会把 FFT 变成真正 open-boundary solver；它只是避免对非零平面波背景施加硬孔径，并把周期副本推远。

residual 的边缘有效性用外圈能量占比报告：

$$
\eta_\mathrm{edge}=\frac{\|M_e\delta U_A\|_2^2}
{\max(\|\delta U_A\|_2^2,\epsilon)}.
$$

### 5.2 冻结 refinement

- axial 新增 `dz=0.25 um`，final pair `0.5→0.25 um`；
- lateral 新增 `512²@0.125 um`，固定 64 um FOV，final pair `0.25→0.125 um`；
- full-chain/external FOV 新增 `112/128 um`，final pair `112→128 um`；
- fine canonical B 为 `768²@0.125 um`，仍是同一 `48²` physical phase cells；
- 128 um working B 由 96 um base 的 centered periodic extension 得到，不重新随机抽样。

### 5.3 正式结果与定位

```text
config: configs/experiments/exp040_TGV_3d_multislice_refinement.yaml
formal run: runs/exp040_TGV_3d_multislice_refinement_20260810_181728
R1 status: Inconclusive
config SHA256: D987F8216531727E4A6A2F609EE306E9BDDC0D27CB1F92BCE9FB7FD0D2D867DB
metrics SHA256: 3CFE64D849758A23EB760256B3693A93F4B7DBB86DFC0144CE814AA62AC04FFF
```

| final pair | `U_A_exit` | `P_B` | `I_stack` |
|---|---:|---:|---:|
| axial `0.5→0.25 um` | `0.188238` | `0.188238` | `0.207719` |
| lateral `0.25→0.125 um` | `0.067745` | `0.053777` | `0.075441` |
| full-chain FOV `112→128 um` | `7.05e-5` | `0.276912` | `0.729435` |
| fixed-A-exit external `112→128 um` | center error `2.55e-16` | `0.276348` | `0.729701` |

fine B mapping error `1.57e-16`，residual edge energy `0.017338`。A-exit 在 enlarged FOV 中已经稳定，
但固定同一 A-exit 的 external path 仍给出几乎相同的大 `P_B/I_stack` error。这把主要 FOV residual
定位到 A-exit 之后，却还没有区分 B 周期不相容、ASM alias、circular wrap 与 detector sampling。

## 6. R2：周期相容 FOV 与 ASM transfer alias

### 6.1 Same-grid ASM alias control

对应 sampled-transfer Nyquist 推导见
`docs/theory_notes/exp040_r2_periodic_boundary_and_alias_control.md`。

介质波长为 $\lambda_m=\lambda_0/n$，以 $u,v$ 表示 cycles/m：

$$
H(u,v;z)=\exp\left[i2\pi z
\sqrt{\lambda_m^{-2}-u^2-v^2}\right].
$$

只去除 evanescent components 并不能保证 sampled transfer 不 alias。transfer phase 关于 $u,v$ 的局部
频率为

$$
f_u=\frac{uz}{\sqrt{\lambda_m^{-2}-u^2-v^2}},\qquad
f_v=\frac{vz}{\sqrt{\lambda_m^{-2}-u^2-v^2}}.
$$

R2 依据 Matsushima--Shimobaba 的 sampled-transfer Nyquist 条件，在实际 same-grid FFT 的
$\Delta u=(N_x\Delta x)^{-1}$、$\Delta v=(N_y\Delta y)^{-1}$ 上使用 exact common-ellipse mask：

$$
u_\mathrm{lim}=\frac{\lambda_m^{-1}}
{\sqrt{1+(2\Delta u|z|)^2}},\qquad
v_\mathrm{lim}=\frac{\lambda_m^{-1}}
{\sqrt{1+(2\Delta v|z|)^2}}.
$$

这控制 transfer sampling alias，但仍是 periodic/circular same-grid propagation，不等于论文中的完整
linear-convolution padding，也不自动证明 mask 外真实光场不重要。

### 6.2 冻结比较

- period-commensurate FOV：`96/192/288 um`，对应 B 的 `1/2/3` 个周期；
- current evanescent-only ASM 与 alias-controlled common-ellipse ASM 使用完全相同 A-exit、B、scan、ROI；
- final pair：`192→288 um`；method difference 以 alias-controlled output 为分母；
- current-vs-alias `>0.05` 只标记 material，不把 alias branch 指定为 truth。

### 6.3 正式结果

```text
config: configs/experiments/exp040_TGV_3d_multislice_r2_boundary_alias.yaml
formal run: runs/exp040_TGV_3d_multislice_r2_boundary_alias_20260811_144331
R2 status: Inconclusive
config SHA256: 9C31B19C9A61DE883629B41DDBCD3A97A609546217F5242A018C0E437AC8DDD5
metrics SHA256: FACC464A28F57848A8B341B45AF3C2DAA3A684170C650733938BBE289DB44E15
```

| method / final pair | `P_B` | `I_stack` |
|---|---:|---:|
| current ASM `192→288 um` | `0.115572` | `0.172736` |
| alias-controlled `192→288 um` | `0.007699` | `0.308996` |
| current-vs-alias at 288 um | `0.030291` | `0.396474` |

整数周期没有单独关闭 floor。alias control 使 `P_B` 收敛，但 `I_stack` 仍失败，且两种 BC method 的
detector difference material。主要未决项因此移动到 B multiplication、BC propagation 和 detector
representation 的组合。

## 7. R3：B-exit spectrum、BC propagation 与 detector sampling

### 7.1 为什么 detector point sample 不等于 pixel measurement

对应 detector-path 与像素响应理论见 `docs/theory_notes/exp040_r3_detector_path_diagnostics.md`。

理想 point detector 为

$$
I_{mn}^{\mathrm{point}}=|\psi(x_m,y_n)|^2,
$$

而有限 pixel measurement 是面积积分或平均：

$$
I_{mn}^{\mathrm{pixel}}=
\frac{1}{A_p}\int_{A_{mn}}|\psi(x,y)|^2dxdy.
$$

在频域乘 square-pixel sinc MTF 可以表示一个理想周期插值模型中的 box average，但离散 nodes 上
非负的数据，其 finite Fourier interpolant 在 nodes 之间可能为负。因此“频域守恒、常数保持、虚部很小”
不推出输出逐点非负。physical intensity operator 还必须满足 positivity。

### 7.2 冻结分层诊断

- 固定 192 um periodic FOV，sampling factors `1/2/4`，node dx `0.5/0.25/0.125 um`；
- 比较 B-exit spectrum 在 BC alias mask 与 native detector Nyquist 外的能量；
- 比较 current/alias-controlled BC field 与 intensity；
- detector branches：native point sample 与 periodic sinc-MTF pixel-box average；
- primary branch 为 alias-controlled pixel average；所有 full high-resolution stacks 流式处理，不写 HDF5。

### 7.3 正式结果与 hard failure

```text
config: configs/experiments/exp040_TGV_3d_multislice_r3_detector_path.yaml
formal run: runs/exp040_TGV_3d_multislice_r3_detector_path_20260811_153852
R3 status: Failed
config SHA256: 4B17ADD64B0633540322EA416B4C9E23BB7720A85774A8F48BA7CA95A085B4B6
metrics SHA256: 85BE9131F9797B6837DCFD341834F6876633403B624D1CEB4B080183E2ACF9BE
```

factor `2→4` 的 `P_B=0.000418`；alias-controlled point/pixel 为 `0.022934/0.022649`，均通过；
current point/pixel 为 `0.109832/0.068485`，均失败。factor-4 B-exit 在 BC alias mask 外的平均能量
`0.118636`，current-vs-alias full field/intensity 分别为 `0.354198/0.492416`。

但 sinc-MTF actual filtered intensity 的最大 relative negative scale 为 `1.089144e-3 >1e-12`，故
`all_intensity_nonnegative=false`，R3 必须为 `Failed`。代码没有 clip；后续也不得把 R4 的成功回写成
R3 通过。

## 8. R4：positivity-preserving detector quadrature

### 8.1 Staggered midpoint rule

对应正值 detector quadrature 原理见
`docs/theory_notes/exp040_r4_positive_detector_quadrature.md`。

对每个 detector pixel，在每个方向放置 q 个 cell-centered subpixel nodes：

$$
x_{m,a}=x_m+\left(\frac{a+1/2}{q}-\frac12\right)\Delta x_p,
\qquad a=0,\ldots,q-1,
$$

$y$ 同理。pixel average 为

$$
I_{mn}^{(q)}=\frac{1}{q^2}\sum_{a,b}
|\psi(x_{m,a},y_{n,b})|^2.
$$

所有 weights 为 $1/q^2\ge0$，因此非负 node intensity 的平均必然非负；同时要求 constant preservation、
sum identity、block center geometry 和 q-series convergence。它是 detector pixel integral 的正值数值
quadrature，不依赖 finite Fourier interpolant 的逐点正性。

### 8.2 冻结比较与结果

```text
q = [2, 4, 8]
node dx = [0.25, 0.125, 0.0625] um
FOV = 192 um
reference = q8
acceptance pair = q4 -> q8
formal run: runs/exp040_TGV_3d_multislice_r4_positive_quadrature_20260811_161412
R4 status: Passed
config SHA256: C9628F9D12663CBCA1FCC0BA3533A14313086E1326076329DB5BB62919631D7E
metrics SHA256: F2093962EFF1C369E45145A6C61E0C600D5B6DB55E8794FE2D4F65893C09467C
HDF5 SHA256: A1F8C649780A8B9A82EDDA5AE80237DD39CE5A61140D82AF984D7BA7E7D402EB
```

q4→q8 的 `P_B=4.455618e-6`、`I_stack=3.969993e-4`；node geometry error
`2.710505e-14`、constant error `0`、sum error `1.133382e-16`、determinism `0`，所有输出非负。
因此 q4 足以作为后续 detector quadrature；R4 不证明 periodic B 或 circular BC 正确。

## 9. R5：finite sample B 与 open-boundary residual propagation

### 9.1 有限 B 与 scan boundary

对应有限 support 与 residual open-boundary 推导见
`docs/theory_notes/exp040_r5_finite_support_open_boundary.md`。

R5 将同一 `48×48` phase-cell realization 解释为居中的有限 `96 um×96 um` 编码区：

$$
B_\mathrm{fin}(x,y)=1+M(x,y),
$$

编码区内 $M=B_c-1$，外部 $M=0$，即透明 transmission `1+0j`。scan 只平移 $M$，移出窗口的
部分用零填充；这保持透明 exterior，不从对侧 periodic wrap 内容。

### 9.2 Reference-plus-residual open path

在 B plane 分解

$$
P_B=P_0+\delta P_B,
$$

其中 $P_0$ 是 homogeneous plane-wave background。有限 B 后

$$
E_{B,s}=P_0+R_s,
\qquad
R_s=\delta P_B+P_BM_s.
$$

detector field 写为

$$
U_{D,s}=\mathcal H_{BC}P_0+\mathcal H_{BC}R_s.
$$

homogeneous background 是 FFT 的本征模，可在整个 padded grid 生成；只对局域 residual 增大零 padding，
以压低 circular wrap。这是 open-boundary 的 convergence approximation，不宣称任一有限 padding 等于
无限域 Green-function 解。

### 9.3 冻结比较与结果

```text
detector quadrature = q4, node dx = 0.125 um
base FOV = 192 um / 1536^2
open FOV = [192, 288, 384] um
native ROI = 128^2
finite B = 96 um hard square, transparent exterior
formal run: runs/exp040_TGV_3d_multislice_r5_finite_support_open_boundary_20260811_163555
R5 status: Passed
config SHA256: A0EC579CDEB7BDB474CC3174A61FE3D3CAC8188329902E79BEF104C0F8C5249B
metrics SHA256: 41AA3522B146EF4063EA65DCFEB707E25E01B17688EBF3036680DE9205A85C28
HDF5 SHA256: 4065E2CB7449F5AA97BF47C9859AC020CC874D76C97BF6D58011DF985715EC30
```

- open `288→384 um` detector difference：`0.014899`；
- residual outer-16-um ring energy：`0.029580`；
- periodic vs finite circular support effect：`0.381145`，material；
- finite circular vs finite open-384 boundary effect：`0.032662`，non-material；
- periodic circular vs finite open combined effect：`0.387506`，material。

结论是无限 periodic B 是当前模型的重要 contributor，而登记的 circular/open correction 已低于 5%。
`96 um` hard square 和透明 exterior 只是 working hypothesis，不是真实 B truth。

## 10. R6：sample-B support sensitivity envelope

### 10.1 为什么 sensitivity envelope 不是 calibration

对应 sensitivity 与 calibration 的解释边界见
`docs/theory_notes/exp040_r6_sample_b_support_sensitivity.md`。

没有实测 B 的有效编码面积、边缘过渡与 complex transmission 时，仿真不能识别真实 support。R6 只问：
R5 的 periodic-vs-finite materiality 是否对一个预先登记的合理假设族稳健。

所有 case 使用同一 canonical phase cells。edge taper 只作用于 phase：

$$
B_\mathrm{fin}(x,y)=\exp[iw(x,y)\phi(x,y)],
\qquad 0\le w\le1,
$$

$w$ 为 separable raised-cosine，因而所有 case 仍 unit modulus，support 外为 1。超过 96 um 的区域只是
同一 realization 的虚拟周期扩展，不能冒充实测新区域。

### 10.2 冻结 envelope 与结果

```text
support width = [80, 96, 112] um
phase taper = [0, 4, 8] um
full factorial = 9 cases
nominal = 96 um / hard edge
comparator = 192 um circular alias-controlled, q4
formal run: runs/exp040_TGV_3d_multislice_r6_b_support_sensitivity_20260811_200120
R6 status: Passed
config SHA256: 258A146A26C5A419569EAD9C740279EBCE4D48F207F5A64AEFC714D2D0FB67E8
metrics SHA256: 5813692089C374892D961152250225F71CE05843828A5F8D9FBE5CEBA33B987A
HDF5 SHA256: 26AD3384EB8D8F6A5E9FD79A8E9A4AD9BB86E6A6014A121E2763A4ADD31341B0
```

periodic-vs-finite support effect：

| support | taper 0 um | taper 4 um | taper 8 um |
|---:|---:|---:|---:|
| 80 um | `0.408647` | `0.410881` | `0.408686` |
| 96 um | `0.381145` | `0.385429` | `0.391952` |
| 112 um | `0.343299` | `0.353436` | `0.362793` |

九个 case 全部 material，故 R5 的定性归因对该 envelope 稳健。但 finite cases 相对 nominal 的最大差异
为 `0.171296`。它是 model sensitivity context，不是统计 error bar，不能与后续误差平方相加、相减或
当作 calibration uncertainty 的概率分布。

## 11. R7：subvoxel TGV interface comparison

### 11.1 冻结 interface 与 forward chain

对应 cell-averaged indicator 与 subvoxel interface 理论见
`docs/theory_notes/exp040_r7_subvoxel_tgv_interface.md`。

R7 只替换 sample-A lateral indicator representation：

```text
interface q = [1, 2, 4, 8]
q1 = exact historical voxel-center binary
A grid = 256^2 @ dx=0.25 um, FOV 64 um
dz = 0.25 um, 400 exact-width slices
multislice = streamed centered symmetric split-step
B = nominal finite 96 um hard edge, transparent exterior
scan = same 25 positions, constant shift of B-1
detector = positive q4 midpoint quadrature
external path = alias-controlled 384 um open branch
reference = q8; final pair = q4 -> q8
```

完整 `400×256×256` q-volumes 和 3072² detector-node stacks 不保留；slice 与 scans 流式执行。
R6 的 `0.1712959787` 单独写入 `model_uncertainty_context`，不与 R7 metrics 合并。

### 11.2 正式结果

```text
config: configs/experiments/exp040_TGV_3d_multislice_r7_subvoxel_interface.yaml
formal run: runs/exp040_TGV_3d_multislice_r7_subvoxel_interface_20260813_011329
R7 status: Passed
legacy experiment_status: Inconclusive
run config SHA256: 5E760D55BFD10E6EFE4BFE68BED0F7E33A90CEE1EDC69E33EF76B5DDE0FF852F
metrics SHA256: F7C26CC9B14778704C4F14B660515A8CD710B750A5C1E5EB0A868969CB4324BE
HDF5 SHA256: 34D10D9FB506BD76FE887041788B803D74A1138E0E434140878686E649258551
```

| output | q1 vs q8 | q2 vs q8 | q4 vs q8 |
|---|---:|---:|---:|
| `U_A_exit` | `0.182189` | `0.0627938` | `0.0198499` |
| `P_B` | `0.0144609` | `0.00395651` | `0.000498961` |
| `I_stack` | `0.00824131` | `0.00202584` | `0.000250482` |

q4→q8 三项均通过。q1 binary effect 只在 `U_A_exit` 超过 5%；到 B 和 detector 后均低于 5%。hard
controls 包括 q1 identity `0`、streamed homogeneous `2.41075e-13`、fraction/index/count bounds `0`、
B mapping/exterior `0`、detector sum error `1.98481e-16`、determinism `0`，全部通过。

R7 关闭了 lateral interface quadrature 的收敛问题，但没有重新执行 q8 interface 下的 axial/lateral
和 waist visibility，因此不能改变总体状态。

## 12. 截至 R7 的跨阶段归因（历史快照）

本节保留文档重组时的 R7 阶段归因，不承担 exp040 的现行状态摘要；R8--R14B 的后续证据见第 16--18 节，
冻结后的现行状态见文首、第 1.2 节和第 19 节。

### 12.1 已经受控的数值项

- slice widths、physical planes、split-step order、zero contrast、phase-product identities；
- 同一 physical B realization 在不同 sampling 上的映射；
- same-grid ASM transfer alias 的显式 mask 与其作用范围；
- detector path 的 B-exit/BC/sampling 分层诊断；
- positivity-preserving detector pixel quadrature，q4→q8 已收敛；
- nominal finite B 下的 residual open-boundary FOV convergence；
- periodic B materiality 对预注册 support/taper envelope 的定性稳健性；
- subvoxel lateral interface quadrature，q4→q8 已收敛。

这些结果是“在当前 scalar/unidirectional working model 内的控制”，不是 Maxwell 精度证明。

### 12.2 尚未关闭的数值与标定项

1. **q8 sample-A axial/lateral convergence。** R1 的 axial/lateral floor 来自 q1 binary branch；R7
   只在固定 `dx=dz=0.25 um` 比较 interface q，没有重做 axial/lateral pair。
2. **统一 forward 下的 waist visibility。** 旧 waist signal 与旧 floor 不能同 q8/finite-B/open path 的新
   floor 混算。
3. **真实 sample B。** 需要有效编码面积、边缘过渡、complex transmission、基底 exterior、B 相对
   illumination 的位置；R6 的 17.13% 只说明假设敏感。
4. **真实 sample A。** 需要实际 $D(z)$、厚度、侧壁轮廓、粗糙度、倾斜/偏心和材料 complex index。
5. **illumination/detector/stage。** 需要波长、AB/BC 距离、scan position、pixel pitch、dark/gain、PSF/MTF、
   active area、NA/dynamic range 等实际标定。

公开文献或厂商资料可提供材料折射率和 nominal hardware 参数，不能提供本项目实际 sample B、TGV、
装调和 detector 的 specimen-specific truth。

### 12.3 R7 时点模型绕不过去的物理边界

当前 phase-screen multi-slice 是 scalar、monochromatic、unidirectional。它不产生：

- glass/air interface Fresnel reflection；
- backward wave 与 multiple reflection；
- sidewall multiple scattering；
- vector polarization 与 high-NA coupling；
- material absorption/dispersion、roughness、tilt、noncircularity；
- finite-coherence、noise、background、quantization 与 detector saturation。

玻璃/空气法向强度反射的量级约为

$$
R=\left|\frac{n_\mathrm{glass}-n_\mathrm{air}}
{n_\mathrm{glass}+n_\mathrm{air}}\right|^2\approx0.04,
$$

与 5% 数值诊断尺度相近，但这不能直接推断完整 TGV detector error 为 4%。只有 numerical sampling、
B/A/illumination/detector calibration 都受控后仍存在结构化偏差，才应另行预注册 bidirectional BPM、
Lippmann--Schwinger 或 vector FEM/FDTD comparator；复杂模型不能未经验证就作为 truth。

## 13. 数据、HDF5、图与可复现性

### 13.1 共同 run 结构

每次正式运行都创建独立 timestamped run：

```text
config.yaml
metadata.json
metrics.json
run_state.json
figures/*.png
outputs/*.h5
```

历史 run 不覆盖。R0--R7 的正式 run 和哈希已分别记录在对应章节；更细的历史 prefix locks、测试批次和
中间 run 说明保留在 `_old.md`。

### 13.2 HDF5 语义

所有正式 run 的 `/entry` 顶层保持：

```text
config_yaml
data
instrument
metadata
metrics
sample
truth
```

baseline `/entry/data/I_stack` 为 `(25,128,128)`，scan positions 为 `(25,2)`；simulation truth 保存
baseline `n_volume/z/slice_width/D(z)/incident/U_A_exit/P_B/B` 和 waist sweep。diagnostics R1--R7 只增加
compact `/entry/metrics/diagnostics_rN/...`，不伪造 reconstruction/calibration/preprocessing，不把
diagnostic arrays 冒充 truth，也不保存 full high-resolution fields、volumes 或 node stacks。

### 13.3 图像

R0 的八张图检查几何/index、A-exit、projected product、dz/lateral/FOV convergence、B probe、baseline
detector 和 waist visibility。各 R 阶段只增加其登记图，例如 R5 open convergence/support effect，R6
support matrix，R7 fraction/convergence/detector comparison。PNG 只供人工检查，正式数值从 JSON/HDF5
读取。

截至 R7：专项 runner/figure/HDF5 tests 为 `10 passed`；R7 正式运行前全量测试为
`204 passed in 33.98s`；R7 修改范围 Ruff 为 `All checks passed`。这些是当时的运行记录，不表示未来
修改后无需重新测试。

## 14. 截至 R7 的结论与当时下一步（历史快照）

本节记录 R7 完成后的当时结论和 R8 建议。R8 及其后的实际执行、状态和结论以第 16--18 节为准，
不得把本节的“下一步”误读为冻结后的当前计划。

### 14.1 R7 时点可发表式结论边界

exp040 已建立完整的 3D TGV scalar multi-slice forward baseline，并通过一系列预注册 diagnostics 将
早期混合的 detector floor 分解为传播 sampling、B multiplication、detector quadrature、finite B boundary
和 sample-A interface representation。当前可以说：

- positive detector quadrature、nominal finite-B open path 和 subvoxel lateral interface 在各自冻结条件下
  已数值收敛；
- infinite periodic B 是 material model contributor；
- binary staircase 对 A-exit material，但对登记 detector path 的 effect 小于 5%；
- 当前还没有在统一 q8/finite-B/open forward 上证明 axial/lateral 与 waist visibility。

当前不能说：真实 TGV 腰径已可测、真实 B 已标定、标量单向模型已达到绝对物理精度、Phase 4 已完成，
或 Phase 5 inverse 已获得进入条件。

### 14.2 当时建议的 R8

下一步仍属于 `exp040` 数值 refinement，不换研究问题：预注册 q8 interface 下的 sample-A axial/lateral
pair 和 detector waist-visibility reevaluation。至少固定：

```text
same continuous TGV geometry/materials
q8 lateral interface
nominal finite 96 um hard-edge B, transparent exterior
same 25 physical scans
q4 positive detector quadrature
384 um open residual detector path
unaligned relative L2 and existing 5% / signal-to-floor>=3 gates
R6 17.13% B-support sensitivity kept as separate context
```

必须在运行前登记具体 axial/lateral cases、reference denominator、是否需要共同网格 mapping、需要保存的
metrics 和状态逻辑。不得把旧 q1 waist signal 与新 q8 floor 混合，也不得根据运行结果再选择 support、
passband、q、ROI 或阈值。

若 R8 的 detector axial/lateral floor 仍显著高于 5%，后续优先级是取得真实 calibration，再另开物理模型
comparison；不是继续为得到 `Passed` 而在 exp040 中无边界地增加假设。

## 15. 文档维护规则

- 今后默认读取和修改本文件；`_old.md` 只读冻结，不再追加。
- 新 diagnostic 必须先在本文件末尾写研究问题、改变的假设、冻结 cases、分母、阈值、状态和产物，再运行。
- theory notes 只保存可脱离本次参数、run 和状态成立的物理/数学内容；实验配置、阈值、结果和决策都写在
  本文件。
- 若本文件的重组摘要与正式 metrics/HDF5 或 `_old.md` 的当时证据发生冲突，以正式产物和冻结旧记录为准，
  并在本文件末尾追加勘误；不得静默改写历史数字。
- Git 保持当前 unstaged；本次文档重组未运行仿真、未改变代码、配置、HDF5 或任何 run。

---

## 16. R8 预注册：统一 q8/finite-B/open forward 下的收敛与腰径可见性

### 16.1 研究问题、进入依据与不变边界

本节于 `2026-08-14` 在任何 R8 实现和运行之前追加。R8 仍属于 `exp040` 的数值 refinement，
不更换研究问题，也不回写 R0--R7 的配置、阈值、状态或结论。

R7 已证明 q4→q8 lateral interface quadrature 收敛，但只在固定
`256² @ dx=0.25 um、dz=0.25 um` 上比较 interface factor。R8 要回答：把 interface 固定为 q8，
并在同一 nominal finite-B、positive detector quadrature 和 open detector path 下重做 axial/lateral
final pairs 后，`D_waist=18/22 um` 相对 `20 um` 的 detector signal 是否稳定高于同一 forward 的
numerical floor。

以下物理与数值定义固定不变：

```text
continuous TGV: D_top/D_waist/D_bottom = 30/20/30 um, z_waist = 50 um
materials: n_glass/n_air = 1.5/1.0
wavelength = 532 nm; z_AB/z_BC = 0.5/1.0 mm
illumination = unit-amplitude normal-incidence plane wave
sample-A FOV = 64 um x 64 um
sample-A interface = q8 staggered midpoint air-area fraction
multislice = streamed centered symmetric split-step with exact slice widths
sample B = same canonical 48 x 48 phase cells, finite 96 um hard edge
B exterior = 1+0j; scan shift = constant-zero shift of B-1
scan = same 25 physical positions and seeds
AB/BC transfer = existing alias-controlled same-grid ASM
detector = q4 positive staggered midpoint quadrature, native 128 x 128 ROI
primary detector path = 384 um residual open-boundary branch
all comparisons = unaligned relative L2; no phase/scale/spatial alignment
```

R8 不引入 reflection、backward wave、polarization、noise、有限 illumination、真实 detector MTF、
新的 B support/taper、subpixel stage model 或 reconstruction。`96 um` hard-edge B 仍是 nominal working
hypothesis，不是 empirical truth。

### 16.2 冻结 sample-A cases 与复用关系

R8 只运行以下五个唯一 sample-A cases；共享的 case 不重复计算：

| case id | shape | `dx` | `dz` | `D_waist` | 用途 |
|---|---:|---:|---:|---:|---|
| `axial_coarse` | `256²` | `0.25 um` | `0.50 um` | `20 um` | axial test |
| `common_reference` | `256²` | `0.25 um` | `0.25 um` | `20 um` | axial reference / lateral test |
| `finest_baseline` | `512²` | `0.125 um` | `0.25 um` | `20 um` | lateral/waist reference |
| `waist_minus` | `512²` | `0.125 um` | `0.25 um` | `18 um` | waist signal |
| `waist_plus` | `512²` | `0.125 um` | `0.25 um` | `22 um` | waist signal |

正式 acceptance pairs 冻结为：

```text
axial: axial_coarse -> common_reference, denominator = common_reference
lateral: common_reference -> finest_baseline, denominator = finest_baseline
waist minus/plus: waist case -> finest_baseline, denominator = finest_baseline
```

axial 的 `U_A_exit` 在同一 `256²` 网格直接比较。lateral 的 `U_A_exit` 把 finest field 用既有
centered bilinear complex-field mapping 映射到 `256² @ 0.25 um` 后，与 `common_reference` 比较；
这只用于 numerical comparison，不是 detector model。`P_B` 和 `I_stack` 都已进入同一 q4 detector-node
sampling/native ROI，因此直接比较。禁止只重采样 wrapped phase。

### 16.3 冻结 external/open detector path 与 FOV control

所有五个 cases 的 A-exit 都按 homogeneous-reference + scattered-residual 方式映射到
`192 um @ node dx=0.125 um` 的 AB base grid，再使用同一 finite B 和 25 个 scans。primary output
使用 `384 um @ node dx=0.125 um` 的 BC residual open branch。

为避免把 R5 的旧 sample-A branch 与 R8 q8 waist signal 混算，R8 对 `finest_baseline` 额外计算同一
q8/finite-B/q4-detector 条件下的 `288 um` open branch，并冻结：

```text
open/FOV acceptance pair: 288 um -> 384 um
denominator: 384 um finest_baseline I_stack
comparison ROI: same centered 128 x 128 native detector pixels
```

该额外 case 只更新统一 forward 下的 detector FOV/open numerical component，不重新比较 periodic B、
current ASM、detector q 或 support family。

### 16.4 冻结 metrics、numerical floor 与 visibility

对 axial 和 lateral final pair 分别保存 `U_A_exit/P_B/I_stack` relative L2；对 open/FOV pair保存
`I_stack` relative L2。数值 floor 定义为：

$$
f_{U_A}=\max(\varepsilon_{U_A,\mathrm{axial}},
\varepsilon_{U_A,\mathrm{lateral}}),
$$

$$
f_{P_B}=\max(\varepsilon_{P_B,\mathrm{axial}},
\varepsilon_{P_B,\mathrm{lateral}}),
$$

$$
f_I=\max(\varepsilon_{I,\mathrm{axial}},
\varepsilon_{I,\mathrm{lateral}},
\varepsilon_{I,\mathrm{open}}).
$$

waist minus/plus signal 对三个输出分别以 `finest_baseline` 为分母；detector 另保存逐 scan signal。
visibility ratio 为

$$
R_I^{(\pm)}=\frac{s_I^{(\pm)}}{\max(f_I,\epsilon_{\mathrm{float64}})},
\qquad
R_{I,\min}=\min(R_I^{(-)},R_I^{(+)}).
$$

`U_A_exit/P_B` 的 signal-to-floor 只报告，不控制 R8 status。不得使用旧 q1 waist signal、旧 R1 floor
或从结果中挑选某个 scan 代替 full-stack gate。

### 16.5 阈值、hard controls 与状态逻辑

R8 不新增阈值，全部复用现有 acceptance：

```text
axial/lateral/open convergence <= 0.05
detector full-stack waist signal / numerical floor >= 3
algebra and mapping controls <= 1e-12
determinism <= 1e-14
all arrays finite; all physical intensity outputs nonnegative
```

hard controls 至少包括：exact slice-width sum、q8 fraction/index/count bounds、相同几何/材料/B/scan
provenance、finite-B mapping 与透明 exterior、A-to-node mapping identity、positive detector constant/sum/
node geometry、`finest_baseline` scan 0 determinism，以及所有输出 finite/nonnegative。

状态严格按以下顺序决定：

1. 任一 hard control 失败：R8 `Failed`，interpretation 为
   `unified_forward_attribution_blocked`；
2. hard controls 通过，但 axial/lateral 任一输出或 open detector pair 超过 `0.05`：R8
   `Inconclusive`，interpretation 为 `unified_numerical_floor_not_closed`；
3. convergence 全部通过，但 `R_I,min < 3`：R8 `Inconclusive`，interpretation 为
   `waist_signal_not_above_registered_floor`；
4. convergence、visibility 和 hard controls 全部通过：R8 `Passed`，interpretation 为
   `waist_signal_resolved_within_registered_working_model`。

即使 R8 `Passed`，它也只表示 nominal q8/finite-B/open scalar working model 内的无噪声数值可分辨；
不表示真实 TGV 已达到该检测能力。历史 `metrics.experiment_status=Inconclusive` 与 R0--R7 状态仍保留，
不得静默覆盖。

### 16.6 R6 uncertainty、产物与后续边界

R6 maximum nominal B variation `0.17129597874704286` 继续保存为独立
`model_uncertainty_context`：

```text
combined_with_r8_metrics = false
used_in_r8_gate = false
```

它不是统计 error bar，不能与 R8 floor 或 waist signal 相加、相减、平方合成，也不能选择一个更有利的
B support/taper 重跑 R8。

R8 新增三张图，原 R0 八图继续保留：

```text
r8_unified_convergence.png
r8_waist_visibility.png
r8_selected_detector.png
```

HDF5 仍保持现有七个 `/entry` children，只增加 compact
`/entry/metrics/diagnostics_r8/...`；不增加 diagnostic truth group，不保存 full q8 volumes、full
detector-node fields/stacks。selected slice/scan arrays只用于生成 PNG。

若 R8 的 detector axial/lateral/open floor 仍未关闭，不再把主要误差归咎于 lateral staircase，也不在
`exp040` 中看后增加阈值、support、passband 或更复杂物理。下一优先级是实际 A/B/illumination/detector
calibration；只有这些量与数值采样都受控后仍有结构化偏差，才另行预注册物理模型 comparison。

### 16.7 R8 正式运行前锁定与计算成本

本节在正式 R8 尚未执行、尚未知晓任何 R8 数值结果时追加。第 16.1--16.6 节及其以前的文档前缀锁定为：

```text
document prefix bytes: 40343
document prefix SHA256: 5FB60D8F2EB956B489314EC3F63F98343D7282F6C40DEF444CFE8CCE43BAE2BF
R8 config bytes: 8973
R8 config SHA256: 1C31CEE2DD9960165FD664DFA10B58924AF65AABD6455F781EFBB15D49C1547E
R8 test SHA256: C5973A016228F45E5E8AA2501A5D658D424D007E0792CFD14B199D7C73BEF4DF
```

实现前的配置关系、冻结 outcome table、微型 unified forward、runner、HDF5 和十一图契约均已执行；微型 R8 为
`8 passed in 5.12s`。随后对整个 `tests/` 回归，结果为 `212 passed in 35.21s`。本次修改范围
Ruff 结果为 `All checks passed`。微型测试同时暴露并关闭了一个执行契约缺口：R8 的最细 sample-A
步长必须等于 detector subpixel-node 步长；正式配置中二者原本就均为 `0.125 um`，现在配置校验会显式拒绝
不满足该关系的变体。这没有改变第 16 节冻结的 case、阈值、分母或模型。

同机 R7 正式 run 从 `2026-08-13 01:13:29` 到 `01:16:12`，约 `2 min 43 s`。R8 相对 R7
增加了三个 `512^2 x 400 slices` 的 q8 cases，并把 primary 3072² detector-node open path 扩展为五个
sample-A cases，另有一个 2304² open control。按 subnode、multislice 和 FFT 工作量估计，正式 R8 通常需
`20--60 min`；若当前约 `4.84 GiB` 可用内存触发换页，可能延长至约 `90 min`。运行采用逐 slice、逐 scan
流式执行，不保留 full q8 volumes 或 full detector-node stacks；该计算是闭合同一 unified forward numerical
floor 和 waist visibility 的必要正式步骤，不能由微型测试替代。

正式命令冻结为：

```powershell
python scripts/run_exp040_multislice_forward.py --config configs/experiments/exp040_TGV_3d_multislice_r8_unified_visibility.yaml
```

本锁定之后不得根据运行结果修改第 16 节的 case、support、passband、ROI、分母、阈值或状态逻辑。本次没有向
`docs/theory_notes/` 写入实验配置、实现或结果。

### 16.8 R8 正式运行与产物核对

正式 R8 已按第 16.1--16.7 节的冻结条件执行，没有在看到结果后修改 case、support、passband、ROI、
分母、阈值或状态逻辑：

```text
formal run: runs/exp040_TGV_3d_multislice_r8_unified_visibility_20260814_152034
run created_at: 2026-08-14T07:20:34.604331+00:00
run completed_at: 2026-08-14T07:28:29.004863+00:00
formal elapsed time: 474.40 s (7 min 54.4 s)
run_state: complete; artifacts_validated = true
R8 status: Inconclusive
R8 interpretation: unified_numerical_floor_not_closed
legacy metrics.experiment_status: Inconclusive
source config SHA256: 1C31CEE2DD9960165FD664DFA10B58924AF65AABD6455F781EFBB15D49C1547E
run config SHA256: E78234FE722C37EF7E09A5177124AB257FF6EC4626BF0571752A9816B93C423A
metadata SHA256: 1C873A04C84C7F9D54DD52F4DFA31D0C4D34349F990B584F5633A156EFFC2140
metrics SHA256: DADE29E7625B7FCEB904534B674DBAF31A37C86066A9A1870B22E9F618F3DA17
HDF5 SHA256: 5004947CDF767F3777E4356CCBE67BA75AD27F254D3EE978690B9ECC3AD776E7
```

正式 sampling 与预注册一致：五个 case 的 shape 为 `256²,256²,512²,512²,512²`，slice 数为
`200,400,400,400,400`，interface 为 q8；25 个 scan、q4 detector、1536² AB base、3072²
primary open 与 2304²/3072² open-control grids 均被执行。full volumes 与 full detector-node stacks
均未保留。

HDF5 大小为 `29,285,912 bytes`；`/entry` 仍只有
`config_yaml/data/instrument/metadata/metrics/sample/truth` 七个 children，baseline
`I_stack=(25,128,128)`、`scan_positions=(25,2)`。R8 只出现在
`/entry/metrics/diagnostics_r8`，没有增加 diagnostic truth。十一张 PNG 全部可读且有限；三张 R8 图的哈希为：

```text
r8_unified_convergence.png  10D910198F7F62854B6287EEF4B55D987DD6CC12992315F2691DD08CD70A8A70
r8_waist_visibility.png     9E8BF78794972CB01E6ABDDD96557A0BF343A999F95C5633FCDD189E671E5A70
r8_selected_detector.png    33EB79E44390B13104A554D630521CDB33305511BF1CAC61F56C04360F4CBBC7
```

### 16.9 正式数值结果

收敛 pair 的未对齐 relative L2 为：

| comparison | `U_A_exit` | pass | `P_B` | pass | `I_stack` | pass |
|---|---:|:---:|---:|:---:|---:|:---:|
| axial `dz 0.50 -> 0.25 um` | `0.0709483` | no | `0.00519245` | yes | `0.00169843` | yes |
| lateral `dx 0.25 -> 0.125 um` | `0.208744` | no | `0.0150829` | yes | `0.00803730` | yes |
| open `288 -> 384 um` | n/a | n/a | n/a | n/a | `0.0148213` | yes |

因此冻结的 numerical floors 为：

```text
U_A_exit floor = 0.20874412488300237
P_B floor       = 0.01508293488929651
I_stack floor   = 0.014821323182753709
```

`I_stack` floor 由 open `288 -> 384 um` 的 `1.48213%` 主导；detector axial 与 lateral 分量分别仅为
`0.169843%` 与 `0.803730%`。这三项 detector 分量均低于原有 5% gate。

腰径 signal 与同一 forward floor 的比值为：

| waist case | output | signal relative L2 | signal/floor |
|---|---|---:|---:|
| `18 um` | `U_A_exit` | `0.472305` | `2.26260` |
| `18 um` | `P_B` | `0.0742767` | `4.92455` |
| `18 um` | `I_stack` | `0.0542708` | `3.66167` |
| `22 um` | `U_A_exit` | `0.413554` | `1.98115` |
| `22 um` | `P_B` | `0.0854997` | `5.66864` |
| `22 um` | `I_stack` | `0.0598868` | `4.04058` |

detector gate 使用两侧 full-stack ratio 的最小值 `3.661672031716599`，因此通过预注册的 `>=3` 门槛。
逐 scan detector signal 没有依赖单帧挑选：`18 um` 为 `0.0515338--0.0574551`，`22 um` 为
`0.0568208--0.0625553`。

hard controls 全部通过：q8 fraction/index/count bounds 与 slice-width sum error 为 `0`；finest homogeneous
streamed identity 为 `2.41075e-13`；A-to-node identity 为 `2.58777e-15`；finite-B mapping/exterior error
为 `0`、unit-modulus error 为 `2.22045e-16`；positive detector constant error 为 `0`、sum error 最大为
`1.98557e-16`、node-geometry normalized error 为 `5.42101e-14`；determinism 为 `0`。所有数组 finite，
所有 intensity 非负。R6 的 `0.17129597874704286` 仍以
`combined_with_r8_metrics=false, used_in_r8_gate=false` 单独保存。

### 16.10 结论边界与下一步建议

R8 必须保持 `Inconclusive`，因为冻结状态逻辑要求 axial/lateral/open 对登记的全部输出均收敛，而
`U_A_exit` 的 axial `7.09%` 和 lateral `20.87%` 超过 5%。不能因为 detector 分支结果较好而把 R8
改写为 `Passed`。

同时，R8 给出一个明确但较窄的正面结论：在当前 nominal q8/finite-96-um-B/q4/open scalar working
model 内，detector axial、lateral 和 open 数值分量均低于 5%，`18/22 um` 相对 `20 um` 的 detector
变化约为 `5.43%/5.99%`，并分别达到同一 detector floor 的 `3.66/4.04` 倍。也就是说，本次
`Inconclusive` 不是 detector visibility gate 失败，而是 raw A-exit field 的统一网格收敛没有关闭。

误差从 `U_A_exit` 传播到 `P_B/I_stack` 后显著下降，说明未关闭部分更可能集中在界面附近的高空间频率、
raw discontinuous-field norm、lateral restriction mapping 或其组合；现有结果不能把它唯一归因于其中一项。
它也没有提供跳到 reflection/backward/vector 模型的证据，因为 detector path 已数值收敛且可见。真实样品
可测量性仍不能宣称：R6 的 B-support 假设敏感性 `17.13%` 以及真实 A/B/illumination/detector 标定均未被
R8 消除或统计合成。

若继续 `exp040` 的数值主线，建议另行预注册一个不改写 R8 的 A-exit attribution diagnostic，再运行：

1. 同时保留本次 raw-field relative L2，并增加共同物理 passband 后的 A-exit comparison；
2. 对 lateral pair 比较 conservative cell-average restriction 与当前 centered bilinear restriction，禁止看后选优；
3. 对 axial pair预先固定 `dz=0.25 -> 0.125 um` 的新 reference，并在执行前评估 q8 成本；
4. 只有共同 passband/restriction 与更细 axial reference 仍留下结构化偏差时，才预注册更复杂物理模型 comparator。

以上是后续建议，不属于本次 R8 gate，也未执行。任何新 pair、comparison operator 或阈值必须先追加到本文档；
本轮没有修改 `docs/theory_notes/`。

---

## 17. R9 预注册：A-exit raw/passband、restriction 与 axial attribution

### 17.1 研究问题、R8 provenance 与不变边界

本节于 `2026-08-14` 在任何 R9 实现或数值结果产生前追加。此前文档前缀为 `48649 bytes`，SHA256 为
`ACD39F7B9590E42193439B2826EF4F6106FD204D27D339D25A04BDB5F1053964`。R9 仍属于 `exp040`
数值 refinement，不回写或重新判定 R8。R8 provenance 冻结为：

```text
run: runs/exp040_TGV_3d_multislice_r8_unified_visibility_20260814_152034
metrics SHA256: DADE29E7625B7FCEB904534B674DBAF31A37C86066A9A1870B22E9F618F3DA17
HDF5 SHA256: 5004947CDF767F3777E4356CCBE67BA75AD27F254D3EE978690B9ECC3AD776E7
R8 status: Inconclusive
R8 raw axial U_A_exit: 0.0709483058386522
R8 raw lateral U_A_exit: 0.20874412488300237
```

R9 回答三个问题：

1. R8 的 raw A-exit axial/lateral discrepancy 能否被确定性复现；
2. 在 A→B 外部介质真正可传播的共同物理频带内，`dz=0.25 -> 0.125 um` 与
   `dx=0.25 -> 0.125 um` 是否低于现有 5% convergence gate；
3. lateral 结果是否依赖当前 centered bilinear restriction，还是与 aligned conservative 2×2
   complex cell-average restriction 一致。

以下条件不变：continuous TGV geometry/materials、532 nm、64 µm sample-A FOV、unit normal-incidence
plane wave、q8 staggered midpoint interface、streamed centered symmetric split-step、exact slice widths、
未对齐 complex-field relative L2，以及现有 `5%/1e-12/1e-14` thresholds。R9 不新增 B、scan、BC、
detector 或 reconstruction diagnostic，也不把 R6 B-support uncertainty 混入 A-exit attribution。

### 17.2 冻结 sample-A cases 与比较分母

只运行以下四个唯一 q8 A-exit cases；同一 case 在多个比较中复用：

| case id | shape | `dx` | `dz` | `D_waist` | 用途 |
|---|---:|---:|---:|---:|---|
| `axial_coarse` | `256²` | `0.25 um` | `0.50 um` | `20 um` | 复现 R8 axial test |
| `common_reference` | `256²` | `0.25 um` | `0.25 um` | `20 um` | R8 axial reference / 新 axial test / lateral test |
| `axial_fine_reference` | `256²` | `0.25 um` | `0.125 um` | `20 um` | 新 axial reference |
| `lateral_fine_reference` | `512²` | `0.125 um` | `0.25 um` | `20 um` | R8 lateral 与新 lateral reference |

冻结 pairs 与分母为：

```text
R8 axial reproduction: axial_coarse -> common_reference
new axial refinement: common_reference -> axial_fine_reference
lateral, both restrictions: common_reference -> lateral_fine_reference restricted to 256²
relative L2 denominator: the named reference after the same passband/restriction operation
phase/scale/shift alignment: none
```

R8 axial/lateral raw metrics 必须分别在 `1e-12` absolute tolerance 内复现上述 provenance，否则 R9 hard
check 失败；新 axial raw metric 只新增记录，不替代 R8 的 `0.50 -> 0.25 um` 结论。

### 17.3 冻结共同物理 passband

共同物理 passband 不是看后选择的频率比例，而是 A→B 外部介质 ASM 的 exact propagating disk：

$$
\Omega_{\mathrm{ext}}=\left\{(f_x,f_y):
f_x^2+f_y^2\le\left(\frac{n_{\mathrm{ext}}}{\lambda_0}\right)^2\right\},
$$

其中 `n_ext=1.0`、`lambda0=532 nm`，所以 cutoff 固定为
`1879699.2481203007 cycles/m`。它与各 native grid Nyquist rectangle 取交集；本实验中 fine/coarse FOV
相同，因此 Fourier-bin spacing 相同。mask 包含边界，频率轴使用 `numpy.fft.fftfreq`，投影固定为：

```text
P_ext(U) = ifft2(fft2(U) * mask_ext)
```

对 axial pair，在相同 256² native grid 上分别投影后直接比较。对 lateral pair，先在 coarse/fine 各自
native grid 上使用同一物理 cutoff 投影，再把 projected fine field 用两种冻结 restriction 映射到 coarse
grid；禁止先 decimate 再从结果选择 cutoff。raw 与 passband comparison 均使用 total complex `U_A_exit`，
不改成 amplitude、phase 或 scattered residual。

同时报告 reference retained spectral-energy fraction、difference 在 passband 内/外的 Parseval energy
fraction，以及 `inside + outside = 1` closure。频带外能量比例只用于 attribution，不新增 gate。

对应的通用 Fourier 投影、Parseval 分解与 restriction 理论另记于
`docs/theory_notes/exp040_r9_a_exit_passband_and_restriction.md`；该理论文件不记录本节的 case、阈值或结果。

### 17.4 冻结 lateral restriction comparison

两种方法必须同时保存，不允许看后选优：

1. `centered_bilinear_complex_field`：复用现有 `resample_centered_grid`，在 centered physical pixel centers
   上分别线性插值 real/imag；
2. `aligned_2x2_complex_cell_average`：对 non-overlapping aligned 2×2 fine cells 的 complex field 取
   权重均为 `1/4` 的面积平均，精确保持离散 areal mean。

对当前偶数 shape、相同 FOV、严格 2:1 nested cell-centered grids，bilinear target center 位于对应四个 fine
centers 的几何中心，因此两种 restriction 理论上应在舍入误差内一致。raw 与 passband restricted reference 的
relative L2 disagreement 最大值必须 `<=1e-12`；若失败，它是 mapping/alignment hard failure，不能选择其中
误差较小者继续判定。

### 17.5 冻结 metrics、hard controls 与状态逻辑

R9 保存：

- 四个 case 的 shape/dx/dz/slice count 与 q8 interface controls；
- R8 axial/lateral raw reproduction 及 provenance absolute errors；
- 新 axial raw/passband relative L2；
- lateral bilinear 与 cell-average 的 raw/passband relative L2；
- restriction disagreement、passband mask geometry、retained/reference energy、difference inside/outside energy；
- homogeneous streamed identity、slice sum、fraction/index/count、Parseval closure、finite 与 postprocessing
  determinism controls。

阈值不新增，固定复用：

```text
external-passband convergence <= 0.05
R8 raw reproduction / restriction equivalence / algebra / Parseval closure <= 1e-12
postprocessing determinism <= 1e-14
all fields and metrics finite
```

状态按以下顺序冻结：

1. 任一 hard control 失败：R9 `Failed`，`a_exit_attribution_blocked`；
2. hard controls 通过，但新 axial passband 或任一 lateral restriction passband `>0.05`：R9
   `Inconclusive`，`external_propagating_band_discrepancy_remains`；
3. passband comparisons 全通过，且新 axial 与两个 lateral raw comparisons 也全通过：R9 `Passed`，
   `raw_and_external_passband_a_exit_converged`；
4. passband comparisons 全通过但至少一个 raw comparison 未通过：R9 `Passed`，
   `raw_discrepancy_attributed_outside_external_propagating_gate`。

R9 `Passed` 只表示这项 numerical attribution 被关闭，不改变 legacy
`metrics.experiment_status=Inconclusive`，不证明真实 TGV 可测，也不验证 scalar/unidirectional model。
若 external propagating passband 内仍超过 5%，这只是允许后续预注册更细数值或物理 comparator 的必要条件，
不是直接把差异归因于 Fresnel/backward/vector physics 的充分证据。

### 17.6 冻结产物与运行成本边界

R9 只新增三张图：

```text
r9_a_exit_convergence.png
r9_lateral_restriction.png
r9_difference_spectra.png
```

HDF5 顶层不变，只增加 compact `/entry/metrics/diagnostics_r9/...`；不得把 R9 fields 冒充 truth，也不保存
full q8 volumes。绘图所需的 selected A-exit differences/spectra 只在内存中保留到 PNG 写完。

四个 cases 合计约 `12.583e9` 个 q8 subnode indicator tests，是 R8 sample-A q8 工作量的约 `55.6%`，
但没有 R8 的 3072²/2304² detector FFT stacks。正式运行前必须用微型测试与 R8 同机时长重新给出预计耗时；
若实现意外触发 full-volume retention 或 detector path，禁止启动正式 run。

### 17.7 R9 正式运行前锁定

本节在正式 R9 尚未执行、任何正式 R9 数值结果尚未知晓时追加。第 17.1--17.6 节及其以前的文档前缀锁定为：

```text
document prefix bytes: 56415
document prefix SHA256: 2B13C4D8E853E27E4F1ED432FF3EA2B2EAA774F7C69510769CC23B9D1887CE5A
R9 config bytes: 8137
R9 config SHA256: C3FE05166E17B8F36EB92D24A2F815B657F5574A1FF1D1AE1D17C5C31364494E
R9 theory SHA256: 0A77FEE7003B06FEC734F37D10D710560E260B43236DCDC78FB5CBDF8DDDFEAE
R9 test SHA256: 4DEFBAFB422B974BC82519402C2EC694110FCCB38010A92BBC2EC5C3E3D786E4
forward implementation SHA256: 376AD95916CA0860EB9FEF6E3866E42EED23537EA3FA9D2390488AFF80145ADE
runner SHA256: 04C3C34DF54AA4C7BF78C3ABA8AAC4E7011353A1101349D41C9A0E0C5D3884F7
R9 plot SHA256: 33FF1842019B3327D8007F40DB5329E4F81251B22F3D3CA2A0276D953491311F
```

正式与缩放配置校验、Fourier projector、constant/idempotence、Parseval、conservative restriction、冻结
outcome table、微型 streamed q8 forward、runner、HDF5 和十一图契约均已测试；R9 专项结果为
`10 passed in 7.00s`。缩放 fixture 因物理参数不同而按设计触发 R8 provenance hard gate，不被冒充为正式
科学结果。随后全仓回归为 `222 passed in 44.27s`，本次修改范围 Ruff 为 `All checks passed`。

四个正式 cases 合计约 `12.583e9` 个 q8 subnode tests，约为 R8 sample-A q8 工作量的 `55.6%`，并且没有
R8 的 3072²/2304² detector propagation stacks。结合 R8 同机正式时长 `474.40 s` 与微型 R9，预计本次
正式运行约 `3--6 min`，峰值内存低于 R8。该计算用于判定 discrepancy 是否仍处在外部可传播频带，是考虑
复杂物理 comparator 前必要的数值归因，不能由微型 fixture 或 R8 detector 结果替代。

正式命令冻结为：

```powershell
python scripts/run_exp040_multislice_forward.py --config configs/experiments/exp040_TGV_3d_multislice_r9_a_exit_attribution.yaml
```

本锁定之后不得根据结果修改 passband cutoff、projection order、restriction、pairs、denominators、阈值或
状态逻辑。

### 17.8 首次正式执行异常终止审计（无科学结果）

第 17.7 节锁定后首次启动了冻结的正式命令，创建了：

```text
runs/exp040_TGV_3d_multislice_r9_a_exit_attribution_20260814_181355
```

该进程运行约 `146.9 s` 后以非零状态异常消失；没有进入 Python 的 `Exception` handler，因而没有保存
traceback，`run_state.json` 仍停留在 `status=running`。只存在冻结的 `config.yaml`、`run_state.json` 以及
空的 `figures/`、`outputs/` 目录；没有 `metrics.json`、`metadata.json`、HDF5、PNG 或可复用 A-exit field
checkpoint。系统事件日志中没有发现对应的资源耗尽事件，早期观测的进程峰值 working set 约 `263 MB`，
因此现有证据不能把终止归因为内存耗尽，也不能定位到某一个 R9 case。

本 run 不包含科学结果，不用于 R9 判定，也不改变第 17.1--17.7 节的预注册。为避免无诊断重复执行，重试前
只允许增加非科学性的执行可观测性：逐 case 的开始/完成事件、即时 flush 的控制台输出、独立进度文件、
Python `-u -X faulthandler` 以及 shell exit-code 记录。禁止借此修改 passband、case、restriction、pair、
denominator、threshold、outcome logic 或任何 forward 数值路径；完成诊断补丁和回归后，必须另行记录 retry
实现哈希，再创建新的 timestamped run。

### 17.9 R9 正式重试锁定

本节在重试尚未启动、重试科学结果尚未知晓时追加。第 17.8 节所述补丁只增加可选 runtime callback、四个
case 的开始/完成事件、postprocessing 事件、原子更新的 `run_progress.json` 与即时 flush 控制台输出；该文件
明确标记为 `non_scientific_execution_diagnostic`，不写入 metrics 或 HDF5。R9 配置、理论、绘图代码及第
17.1--17.7 节冻结的所有科学定义均未改变。重试锁定如下：

```text
document prefix bytes: 59922
document prefix SHA256: 67BC146DD38EF3039D4E000A5E1CC85BCBE4CC81D41365FB6D827425671E1014
R9 config bytes: 8137
R9 config SHA256: C3FE05166E17B8F36EB92D24A2F815B657F5574A1FF1D1AE1D17C5C31364494E
R9 theory SHA256: 0A77FEE7003B06FEC734F37D10D710560E260B43236DCDC78FB5CBDF8DDDFEAE
forward implementation SHA256: 7213D575C53B7CE5196EA76E5E69BE1AEAF884776A190FE1E36FC82DBC0C8101
runner SHA256: 4F95B871A7610B5EEA9E626A0EE7B34F55D934378EC79DBA3EB5AB59F93ABEC5
R9 plot SHA256: 33FF1842019B3327D8007F40DB5329E4F81251B22F3D3CA2A0276D953491311F
R9 test SHA256: 19CE9DC6C72A278807B2F584B4F073AAB658D3BD4EADABDE6B7533488607C8FC
```

诊断补丁后的 R9 专项为 `10 passed in 4.07s`，全仓回归为 `222 passed in 36.06s`，修改范围 Ruff 为
`All checks passed`，Python compile check 通过。重试命令仅改变 Python 执行诊断模式，不改变传入配置：

```powershell
python -u -X faulthandler scripts/run_exp040_multislice_forward.py --config configs/experiments/exp040_TGV_3d_multislice_r9_a_exit_attribution.yaml
```

shell 必须另外报告该进程的实际 exit code。若再次出现宿主级终止，以 `run_progress.json` 最后一个完整事件定位
阶段；不得据此改变科学判据或看后选优 restriction。

### 17.10 受限执行环境阻塞与非沙箱重试边界

第 17.9 节冻结命令在默认受限执行环境中创建了：

```text
runs/exp040_TGV_3d_multislice_r9_a_exit_attribution_20260814_182602
```

`run_progress.json` 显示 baseline、legacy core、四个 R9 q8 cases 均依次产生完整的 completed event，最后一个
事件为 `r9_postprocessing_started`（`2026-08-14T10:28:05.859974+00:00`）。随后主 Python 线程长期停在
Windows `LpcReply`，进程 CPU 累计值固定为 `124.484375 s`、working set 约 `202 MiB`；再观察超过 4 分钟
仍无 CPU、内存、进度或文件变化。该进程在确认固定阻塞后被终止，run state 事后标为
`interrupted_by_execution_environment`。

此 run 没有 Python exception、metrics、HDF5 或 PNG，因而仍然没有科学结果。它只把第一次无法定位的异常
缩小到受限执行环境 IPC/进程层，并不能证明 passband postprocessing 算法有数值缺陷。下一次只允许在非沙箱
进程中执行第 17.9 节完全相同的 `python -u -X faulthandler` 命令；配置及实现哈希继续沿用第 17.9 节，
不新增阈值、不修改计算路径，也不因已知四个 fields 曾算完而手工拼接结果。若非沙箱运行仍在同一事件后
阻塞，才将其作为可复现的软件执行缺陷诊断，停止正式计算并保留现场。

### 17.11 原生 `np.vdot` 失败诊断与执行修复预登记

非沙箱命令创建了：

```text
runs/exp040_TGV_3d_multislice_r9_a_exit_attribution_20260814_183349
```

四个 q8 cases 再次全部产生 completed event；随后进程在第一个 comparison 的正交性检查中以 exit code
`-1066598273` 和 Windows fatal exception `0xC06D007F` 退出。`-X faulthandler` 将现场固定到：

```text
src/tgv_ptycho/forward/exp040.py:8601
abs(np.vdot(inside_difference, outside_difference)) / denominator
```

该 run 仍没有 metrics、HDF5 或 PNG，不产生可审阅的科学结果。使用同一 Python 环境独立执行
`np.vdot` 的 `256x256 complex128` 常数数组可最小复现同一 fatal exception；相同数组的显式表达式
`np.sum(np.conjugate(a) * a)` 正常返回 `65536+0j`。因此这是当前 NumPy/OpenBLAS 大数组 `vdot` 原生
dispatch/runtime 缺陷，而不是 R9 field 特异性失败。默认受限环境中的 `LpcReply` 固定等待也发生在同一原生
调用处，只是异常没有正常穿过 sandbox IPC。

在下一次结果未知的正式运行前，预登记唯一允许的执行修复：把这一处复数内积从 `np.vdot(x, y)` 改为数学上
相同的 `np.sum(np.conjugate(x) * y, dtype=np.complex128)`。它仍计算
`|<inside_difference,outside_difference>| / total_energy`，只绕过失效的 BLAS `vdot` dispatch；不修改
passband mask、projection、restriction、pair、denominator、threshold、comparison、Parseval/orthogonality
定义或 outcome logic。必须补充：小数组与 `np.vdot` 等价测试、至少 `256x256 complex128` 的无崩溃/解析值
测试、R9 专项、全仓回归与 Ruff。通过后先追加新的实现/测试哈希，再运行；本轮不需要修改
`docs/theory_notes/`，因为没有新增物理或数学理论。

### 17.12 原生运行库修复后的正式运行锁定

本节在修复后正式结果尚未知晓时追加。执行修复严格按照第 17.11 节实施：新增
`_r9_explicit_complex_inner_product(left,right)`，校验 shape 后返回
`complex(np.sum(np.conjugate(left) * right, dtype=np.complex128))`，且只替换 R9 正交性 control 中唯一一处
`np.vdot`。配置、理论、绘图、runtime progress、scientific metrics schema 与全部冻结判据未修改。新锁定为：

```text
document prefix bytes: 64920
document prefix SHA256: 123C34124E05BFA3EA754567863C415A653C08C5A12CA7D6E5E3185BD27B34D7
R9 config bytes: 8137
R9 config SHA256: C3FE05166E17B8F36EB92D24A2F815B657F5574A1FF1D1AE1D17C5C31364494E
R9 theory SHA256: 0A77FEE7003B06FEC734F37D10D710560E260B43236DCDC78FB5CBDF8DDDFEAE
forward implementation SHA256: BC62111B8729CB731140DF47A687292D31753E7EB9CBD04AFC73ED9DBC3E64FD
runner SHA256: 4F95B871A7610B5EEA9E626A0EE7B34F55D934378EC79DBA3EB5AB59F93ABEC5
R9 plot SHA256: 33FF1842019B3327D8007F40DB5329E4F81251B22F3D3CA2A0276D953491311F
R9 test SHA256: 96285E58C257B22B5497C6625890C19A2101089EA95249E4E8AC2EE84746325C
```

修复后 R9 专项为 `11 passed in 3.98s`，其中包含 small-array `np.vdot` 等价与 `256x256 complex128`
解析值/无崩溃回归；全仓为 `223 passed in 35.83s`，修改范围 Ruff 为 `All checks passed`，Python compile
check 通过。正式命令仍为相同的非沙箱 `python -u -X faulthandler` 运行；shell 必须报告 exit code。除非又有
结果产生前的明确软件故障，不允许继续修改实现或重复运行。

### 17.13 R9 正式运行与产物核验

修复后的正式命令按第 17.12 节锁定版本在非沙箱进程中完成，shell exit code 为 `0`：

```text
formal run: runs/exp040_TGV_3d_multislice_r9_a_exit_attribution_20260814_184026
progress start: 2026-08-14T10:40:26.325498+00:00
artifacts validated: 2026-08-14T10:42:23.646167+00:00
elapsed to validated artifacts: 117.320669 s
run_state: complete; artifacts_validated = true
legacy experiment_status: Inconclusive
R9 status: Inconclusive
R9 interpretation: external_propagating_band_discrepancy_remains
run config SHA256: 7D31DEC1D58F4D661A0B9BB4CFEBEC0F9473FCD51EA0BCE3906BF3FA2837EE4D
metadata SHA256: 9A64D45A5428FC10580BE55DA03654A6EFA99A382D08B675BA32B1BC0054C175
metrics SHA256: 88B362C134657B1BCAA435E1D67B6A20CA233D51DDC9036F312071189D8D27F4
HDF5 SHA256: A6D921E9310F7A4BEE500A181917F041CC0A88BA914949196A3A52CAF9EA161B
```

四个 case 的 shape 为 `256²/256²/256²/512²`，slice 数为 `200/400/800/400`，q8 interface、
`dx=0.25/0.25/0.25/0.125 um` 与 `dz=0.50/0.25/0.125/0.25 um` 均与预注册一致；没有保留 full
volumes，也没有重算 detector path。

HDF5 为 `29,307,408 bytes`，`/entry` 仍只有
`config_yaml/data/instrument/metadata/metrics/sample/truth`；R9 只写入
`/entry/metrics/diagnostics_r9`，没有写入 `/entry/truth/diagnostics_r9`。十一张 PNG 全部可解码并通过目视
检查；三张 R9 图的哈希为：

```text
r9_a_exit_convergence.png  E11F20D8FE2714171888EF780B892B636BB188D7BEF8B8C3992D4917D30D7BD4
r9_difference_spectra.png  A0C51623A50B95C4DC1A086307CEDC4EE0EEB605982CECD7BB05CF1FC2BCC1F0
r9_lateral_restriction.png 64206383D74D0F002183DB657EC9EF89C95C64F2E4BC569AFD103BAA88FFECA0
```

### 17.14 R9 正式数值结果与结论边界

冻结 comparisons 为：

| comparison | raw relative L2 | external-passband relative L2 | raw pass | passband pass |
|---|---:|---:|:---:|:---:|
| R8 axial reproduction `dz 0.50 -> 0.25 um` | `0.0709483058386522` | `0.0488316321221258` | no | yes |
| refined axial `dz 0.25 -> 0.125 um` | `0.013823843891334533` | `0.010159619796014225` | yes | yes |
| lateral centered bilinear `dx 0.25 -> 0.125 um` | `0.20874412488300237` | `0.17089819910643725` | no | no |
| lateral aligned 2x2 cell average `dx 0.25 -> 0.125 um` | `0.20874412488300234` | `0.17089819910643725` | no | no |

R8 raw axial/lateral reproduction absolute errors均为 `0`。bilinear 与 cell-average reference 的 raw 与
passband disagreement 分别只有 `9.692000206129955e-16` 与 `2.710541219844686e-16`，所以不能看后选择其中
一个，也不能把 `17.09%` lateral residual 归因于 restriction。refined axial difference energy 在 external
passband 内/外分别为 `52.1470%/47.8530%`；lateral difference 则为 `67.8389%/32.1611%`。也就是说，
共同物理 passband 去掉了部分 lateral 高频差异，但大多数 difference energy 仍在外部可传播频带内。

全部 hard controls 通过：projection repeat error 与 postprocessing determinism 为 `0`，最大 projection
idempotence error 为 `2.75427e-16`，最大 Parseval closure 为 `3.28825e-16`，最大 inside/outside
orthogonality error 为 `9.34457e-17`，axial-fine homogeneous streamed identity 为 `1.81626e-13`；所有
fields/metrics finite，interface fraction/index/count 与 slice-width errors 为 `0`。air-volume relative error
最大 `9.93145e-6` 继续作为 report-only geometry sampling diagnostic，不进入冻结 gate。

因此 R9 的明确结论是：

1. axial floor 已被新的 `dz=0.125 um` reference 关闭；
2. centered bilinear restriction 不是 R8 lateral floor 的来源，conservative cell average 给出相同结论；
3. lateral floor 不能解释为纯外部不可传播高频，`17.09%` 且具有同心环/界面附近的结构化残差仍留在共同
   external propagating passband；
4. 这项 A-only diagnostic 没有 B、scan、BC 或 detector，因此该残差也不是 canonical-B periodic wrap、
   detector sampling 或 B-support calibration 的结果；
5. 现有证据仍不足以直接宣称 reflection/backward/vector physics 是根因，因为 `dx=0.125 um` 还没有更细
   lateral reference。R8 保持 `Inconclusive`，R9 也按冻结逻辑保持 `Inconclusive`。

本节没有产生新物理理论，故没有修改既有 R9 theory note。

---

## 18. R10 预注册：lateral closure 后的双向标量 Helmholtz comparator

### 18.1 触发条件、研究问题与禁止项

本节在任何 R10 实现、配置或结果产生前追加。R9 的 refined axial 与 hard controls 已通过，而两种冻结
restriction 下的共同 passband lateral residual 均为 `17.0898% > 5%`，因此满足第 17.5 节“只有结构化偏差
仍存在时才允许预注册复杂物理 comparator”的必要条件。满足必要条件不等于已经证明模型物理错误。

R10 只回答：在先关闭更细 lateral scalar numerical floor 后，包含 backward/reflected wave 的双向标量
Helmholtz boundary-value model 是否相对当前单向 multislice 产生可分辨的 A-exit model-form difference。
R10 不重判 R8/R9，不加入 B、scan、BC、detector 或 reconstruction，不用 truth 做 alignment，也不把标量
Helmholtz 冒充完整 vector Maxwell。

### 18.2 Stage A：必须先通过的 lateral numerical gate

固定同一 continuous TGV、`64 um` lateral FOV、q8 interface、unit normal-incidence plane wave、
`dz=0.25 um` 与外部可传播圆盘。仅比较：

```text
current scalar reference: 512²,  dx=0.1250 um
fine scalar reference:    1024², dx=0.0625 um
restriction: aligned 2x2 complex cell average
comparison order: native-grid passband first, then fine-to-coarse restriction
denominator: restricted fine scalar reference
phase/scale/shift alignment: none
```

同时保存 raw 与 passband relative L2；conservative restriction 必须通过 constant/mean/alignment controls，
复用 `1e-12` algebra gate。只有 external-passband relative L2 `<=0.05` 且 hard controls 全通过，才允许进入
Stage B。若 Stage A `>0.05`，R10 状态为 `Inconclusive / scalar_lateral_reference_not_closed`，禁止计算或解释
model-form difference。

单个 `1024² x 400 slices x q8` case 约含 `26.844e9` 个 subnode indicator tests，约为整个 R9 q8 工作量
的 `2.13` 倍；连同 512² control 后约 `33.554e9`，约为 R9 的 `2.67` 倍。正式实现前仍须重新评估内存、
FFT 与 wall time，并保持 streamed slices；当前不启动该高成本计算。

### 18.3 Stage B：冻结的最小复杂物理 comparator

Stage B 使用频域、轴对称、双向标量 Helmholtz total-field/scattered-field 或等价 boundary-value solver：

```text
equation: (nabla^2 + k0^2 n(r,z)^2) u = 0
geometry/material/wavelength: identical to canonical exp040 sample A
incident field: unit on-axis plane wave in the glass reference medium
open boundary: radial and axial PML; no periodic boundary
output plane: 1.0 um into the external air after sample-A exit
comparison field: u_TGV / u_homogeneous from the same model and output plane
multislice comparator: Stage-A fine scalar field propagated to the same output plane,
                       divided by its own frozen homogeneous control
alignment/fitting: none
comparison band: the same n_ext/lambda0 external propagating passband
norm: axisymmetric 2*pi*r-weighted complex-field relative L2
```

在正式实现前必须把 solver、mesh pair、PML pair、radial/axial domain、incident-field injection 与 radial
sampling operator 写入 R10 YAML 并由 validator 锁死。solver 自身的 mesh refinement、PML enlargement、
homogeneous analytic control 与 multislice azimuthal-anisotropy control 均复用 `5%` convergence gate；任一
失败即 `Failed / helmholtz_reference_not_validated`，不得解释跨模型差异。

若 Stage A 与 solver controls 均通过，跨模型 external-passband difference `<=5%` 记为
`bidirectional_scalar_effect_not_resolved_at_registered_gate`；若 `>5%`，记为
`bidirectional_scalar_model_difference_resolved`。后一结论只说明 backward/reflection scalar physics 对当前
A-exit 有可分辨影响，仍不能自动归因于 vector polarization；vector Maxwell 必须另行预注册。R10 当前只完成
protocol 预注册，没有配置、代码或 run。

对应的纯数学/物理边界见
`docs/theory_notes/exp040_r10_bidirectional_scalar_helmholtz_comparator.md`。

### 18.4 R10 protocol 锁定

本节在没有 R10 YAML、实现或 run 时追加。第 18.1--18.3 节与此前全部文档的前缀，以及对应纯理论文件锁定
为：

```text
document prefix bytes: 74952
document prefix SHA256: 8E08895F36A5EB264956E002B24408C7D9550DEC079A97A73F9C12FD135A0807
R10 theory bytes: 4343
R10 theory SHA256: 4D1A98739A33963F91315CF9F70CE02FB962888F167AAA8CC33A264C96A3651F
```

后续在 Stage A 成本与 Stage B solver 可行性完成只读评估前，不得启动 R10；任何 mesh/PML/domain 或
sampling operator 的具体化都必须先追加到本文并重新锁定，不能用结果反向选择。

### 18.5 R10 Stage A 非科学性能预检预注册

本节于 `2026-08-15` 在任何 R10 preflight 配置、实现或计时结果产生前追加。它只判断第 18.2 节
`512²/1024² x 400 slices x q8` streamed workload 在当前机器上是否值得启动，不计算 Stage A 的
raw/passband scientific comparison，也不允许据此改变 `dx`、FOV、q8、`dz`、restriction、denominator 或
5% gate。

预检直接调用与 R9 相同的 `_r7_streamed_tgv_exit` kernel，冻结为：

```text
cases:
  current: shape=512²,  dx=0.1250 um
  fine:    shape=1024², dx=0.0625 um
interface: q8 staggered midpoint area fraction
propagation: centered symmetric split-step, bandlimit=true
benchmark dz: 0.25 um
timed slices per repeat: 16
representative z rule: 16 equal-stratum midpoints spanning the full 100 um TGV
diameters: canonical continuous D(z) evaluated at those z positions
warm-up: one untimed single-slice call per shape
timed repeats: 3 per shape, fixed order 512² then 1024²
retention: final field and one selected fraction only until each repeat is checked
forbidden: full volume, detector path, B, scan, HDF5 truth, PNG, scientific metric
RSS sampling interval: 0.02 s
runtime statistic: median timed seconds / 16 slices
full-case projection: median seconds-per-slice * 400
Stage-A projection: projected 512² + projected 1024²
safety-adjusted projection: 1.5 * Stage-A projection
```

输出只允许保存 `config.yaml`、`metadata.json`、`metrics.json`、`run_state.json`；没有科学 HDF5 或 figure。
每次 repeat 必须保存 elapsed time、seconds/slice、sampled peak RSS、finite/interface controls 和 output
determinism。512² 外推使用已完成 R9 的真实 lateral case 校准：

```text
provenance run:
  runs/exp040_TGV_3d_multislice_r9_a_exit_attribution_20260814_184026
R9 progress SHA256:
  F083A10481EF028D55DD53635D9A503442898753A3852487E7081AB929CDF476
observed 512² x 400 q8 elapsed:
  81.738972 s
```

预检状态在结果未知前冻结：

1. 任一 output/interface 非 finite、fraction/index/count bound error 非零或 repeat determinism
   `>1e-14`：`Failed / preflight_kernel_control_failed`；
2. 512² full-case 外推相对上述实测时间误差 `>25%`：`Inconclusive /
   short_kernel_extrapolation_not_calibrated`；
3. 校准通过，但 `1.5 x` Stage-A 外推 `>900 s`，或任一 sampled peak RSS 大于启动时 available RAM 的
   `50%`：`Inconclusive / stage_a_cost_not_feasible_on_current_host`；
4. 其余情况：`Passed / stage_a_formal_run_feasible`。

以上 `25%/1.5x/900 s/50%` 只属于当前主机性能决策，不进入任何物理或数值收敛结论。即使 preflight
`Passed`，也只能建议执行一次第 18.2 节正式 Stage A；不能提前允许 Stage B。本文当前前缀锁定为：

```text
document prefix bytes: 75595
document prefix SHA256: 50F2E336CC02A07140B41B8087763056088609E1A3BCBC5BA4D9336243715F07
```

### 18.6 R10 Stage A preflight 实现与运行前锁定

本节在正式 preflight 尚未运行、任何 512²/1024² 正式计时结果尚未知晓时追加。实现只新增独立的性能
preflight YAML、runner 和测试，没有修改 exp040 forward kernel、R9/R10 scientific config、theory note 或
HDF5 schema。锁定为：

```text
document prefix bytes: 78525
document prefix SHA256: 1A0741949FD0DAA7504A415A94DDA97038F462B34B94A6D9ED6402A5A7B4BE6B
preflight config bytes: 1989
preflight config SHA256: 2D668B1617BE747E2FBC9F0F8B92E8EE8FD6A75F2C7F5C422CF56930B64F1351
preflight runner SHA256: 97B72ECCD8B2C72BF2BFAA680F19CA48FC1AC2D677E36C67A1C77A54B7780733
preflight test SHA256: A725C483C62D856A0B5ED7FB03969975FF132E3BF36DBB6454774EA17804705E
```

配置精确校验、冻结 outcome table、微型 streamed q8 kernel 和四文件 runner contract 为
`8 passed in 0.62s`，修改范围 Ruff 为 `All checks passed`，Python compile check 通过。正式命令冻结为：

```powershell
python -u -X faulthandler scripts/run_exp040_r10_stage_a_preflight.py --config configs/experiments/exp040_TGV_3d_multislice_r10_stage_a_preflight.yaml
```

运行后不得修改第 18.5 节的 repeats、外推公式或可行性阈值；若进程失败，只能先记录执行缺陷，不能用手工
删减 repeat 冒充正式 preflight。

### 18.7 R10 Stage A preflight 正式结果与建议

正式 preflight 按第 18.5--18.6 节冻结条件完成，shell exit code 为 `0`：

```text
run: runs/exp040_TGV_3d_multislice_r10_stage_a_preflight_20260815_155426
run created_at: 2026-08-15T07:54:26.955372+00:00
run completed_at: 2026-08-15T07:55:28.014265+00:00
total measured preflight elapsed: 60.9927541 s
run_state: complete; artifacts_validated = true
preflight status: Passed
interpretation: stage_a_formal_run_feasible
scientific_result: false
```

在正式 preflight 前的微型 runner test 曾暴露 64-bit Windows process handle 未显式声明
`ctypes argtypes`；该执行层问题在结果未知前修正并重新测试，第 18.6 节哈希对应修正后的版本。正式运行没有
再次修改配置、repeat、外推或 gate。

原始时间结果为：

| case | 16-slice repeats [s] | median [s] | median [s/slice] | projected 400 slices [s] |
|---|---|---:|---:|---:|
| `512², dx=0.125 um` | `3.260446, 3.114255, 3.129128` | `3.129128` | `0.1955705` | `78.228210` |
| `1024², dx=0.0625 um` | `16.362986, 16.754265, 16.754554` | `16.754265` | `1.0471416` | `418.856628` |

512² 已完成 R9 case 的真实时间为 `81.738972 s`，短 kernel 外推相对误差为 `4.29509%`，通过预注册的
`25%` calibration gate。1024² 每层实测约为 512² 的 `5.354×`；这一比例同时包含 q8 interface slice
生成和 FFT propagation，优于只用像素数作 4× 猜测。

两 case 的完整 Stage A 外推合计为：

```text
projected Stage A:              497.084838 s = 8 min 17.1 s
1.5x safety-adjusted Stage A:  745.627256 s = 12 min 25.6 s
registered wall-time maximum:  900 s
```

启动时 available physical memory 为 `4,434,038,784 bytes`；1024² repeats 的最大 sampled RSS 为
`254,861,312 bytes`（约 `243.1 MiB`），只占 available memory 的 `5.74784%`。这是 streamed kernel
测量，不包含尚未实现的 Stage A passband/restriction postprocessing；不过即使为 512²/1024² raw 与 projected
fields 预留数倍同等数组，仍有明显内存余量。正式实现仍不得保留 400-slice volume。

所有六个 timed repeats 均 finite，fraction/index/count bound error 为 `0`，repeat determinism relative L2
为 `0`。正式 run 只包含 `config.yaml/metadata.json/metrics.json/run_state.json`，`figures/` 与
`outputs/` 为空，没有 HDF5、PNG 或 field：

```text
config.yaml SHA256:    FB02AA6BA89E2CCC87F18832DAF44603CC369056CDBD01875350EE5C95786515
metadata.json SHA256:  060BB772793EE6D6D6B6257D1CE7D5F5A2EB3134D9F1A98E645DBE6AF5EF9F51
metrics.json SHA256:   BB4AEE6A9253EABFAEAAF269D382C1FFDC64A64C506B477EDBCDE7F136B0D9D7
run_state.json SHA256: 5808F484431C883F9298B180115CEC583EA9DC6190F978C3F0A8B3F83ABA905E
```

专项测试为 `8 passed in 0.62s`，随后全仓回归为 `231 passed in 40.79s`；修改范围 Ruff 为
`All checks passed`，Python compile check 通过。

本 preflight 的结论只限于执行决策：当前主机有足够时间与内存完成一次第 18.2 节正式 Stage A，因此下一步
建议实现并锁定唯一的 `512² -> 1024²` raw/common-passband/cell-average comparison，然后运行一次。它不是
对 lateral convergence 的证明，也没有允许 Stage B；只有正式 Stage A 的 external-passband relative L2
`<=5%`，才允许进入双向标量 Helmholtz comparator。若 Stage A 仍 `>5%`，应停止继续把 uniform `dx`
减半，转向 interface-aware/adaptive representation，不预注册 `dx=0.03125 um`。

本节没有新增物理或数学理论，因此没有修改 `docs/theory_notes/`。

### 18.8 R10 Stage A 正式执行预注册

本节于 `2026-08-15` 在正式 Stage A 配置、实现和科学结果产生前追加。第 18.7 节的性能预检已判定当前
主机可完成该 workload；本节只具体化第 18.2 节已经冻结的 lateral numerical gate，不改变研究问题、
scalar multislice 模型、canonical TGV 几何、阈值或 R8/R9 结论。

正式执行包含且只包含以下两个 case，顺序固定为 `current_512` 后 `fine_1024`：

```text
current_512: shape=512 x 512,   dx=0.1250 um, dz=0.25 um, 400 slices
fine_1024:   shape=1024 x 1024, dx=0.0625 um, dz=0.25 um, 400 slices
common FOV:  64 x 64 um
interface:   q8 staggered midpoint air-area fraction, uniform nonnegative weights
field:       unaligned total complex U_A_exit
propagator:  centered symmetric split-step, angular-spectrum bandlimit=true
retention:   streamed slices; no 400-slice volume and no detector/B/scan path
```

两 case 使用同一 continuous `D(z)`，并在同一组 `dz=0.25 um` midpoint 上取直径。每个原生 exit field 先用
`make_physical_passband_mask` / `_r9_project_with_controls` 投影到外部介质可传播圆盘
`f <= n_external / lambda0 = 1879699.2481203007 cycles/m`；随后才把 `fine_1024` 的 raw field 与 projected
field 分别通过 `restrict_aligned_cell_average(..., 2)` 限制到 `512²`。不执行 phase、complex scale、spatial
shift 或其他 alignment，也不计算 centered bilinear 备选。raw 与 passband relative L2 均使用对应的
restricted `fine_1024` reference 作为分母，禁止看结果更换 denominator 或 restriction。

正式指标冻结为：

1. `raw_relative_l2` 与 `external_passband_relative_l2`；
2. coarse-grid raw difference 的 external-passband inside/outside energy fraction、Parseval closure 与
   inside/outside orthogonality；
3. 两个 native projection 的 mask、retained energy、repeat、idempotence、constant-preservation controls；
4. q8 fraction/index/count bounds、discrete/continuous air volume（volume error 只报告，不作 hard gate）、
   slice-width sum 与所有 field finite controls；
5. aligned `2x2` restriction 的 shape、constant preservation、area-weighted complex mean conservation 和
   four-subpixel alignment/weight controls；
6. 将 projection、restriction 和 comparison postprocessing 完整重复一次所得 determinism relative L2；
7. 相对 R9 `dx=0.25 -> 0.125 um` lateral cell-average pair 的 raw/passband error ratio
   `e_R9/e_R10` 与表观阶 `log2(e_R9/e_R10)`。后二者严格标记为 report-only empirical diagnostic，不能冒充
   uniform-grid convergence 的数学证明，也不能单独允许 Stage B。

hard controls 沿用已经预注册的阈值：algebra control `<=1e-12`、postprocessing determinism `<=1e-14`、
slice-width sum absolute error `<=max(1e-15 m, 16*eps_float64*thickness)`，并要求所有输出 finite。正式状态表
在结果未知前冻结为：

```text
hard controls fail:
  Failed / stage_a_numerical_controls_failed; stage_b_allowed=false
hard controls pass and external_passband_relative_l2 <= 0.05:
  Passed / scalar_lateral_reference_closed; stage_b_allowed=true
hard controls pass and external_passband_relative_l2 > 0.05:
  Inconclusive / scalar_lateral_reference_not_closed; stage_b_allowed=false
```

`raw_relative_l2 <=0.05` 只作为独立 report flag，不进入上述 Stage B gate。若 passband 未通过，必须停止继续
uniform `dx` 减半，不得临时增加 `dx=0.03125 um`；下一步只能先预注册 interface-aware/adaptive
representation。若通过，才允许按第 18.3 节另行实现 Stage B Helmholtz comparator。

正式 run 必须保存 `config.yaml`、`metadata.json`、`metrics.json`、`run_state.json`、`run_progress.json`，以及
compact `outputs/exp040_r10_stage_a.h5`。HDF5 只保存并列的 config、instrument、sample、metadata 和 metrics，
不保存 slice volume、native complex fields、detector data 或伪造 truth。两张预注册 PNG 为：

```text
figures/r10_stage_a_convergence.png
figures/r10_stage_a_residuals.png
```

第一张固定比较 R9/R10 raw 与 passband error 和 5% gate；第二张固定显示 R10 raw/passband normalized residual
map、raw difference log-power spectrum 和 external-passband mask。runtime progress 至少逐 case 保存
`case_started/case_completed`、`postprocessing_started/postprocessing_completed` 和 `artifacts_validated`。
正式命令只允许执行一次；失败也必须保留该 timestamped run 和失败状态，不得无记录重跑。

本节追加前的主文档前缀锁定为：

```text
document prefix bytes: 83473
document prefix SHA256: 1AA1B724EE7C5BDB7AB6F3001A2A896706B163508479BA22070CA3A46085CE7B
R9 provenance run: runs/exp040_TGV_3d_multislice_r9_a_exit_attribution_20260814_184026
R9 metrics SHA256: 88B362C134657B1BCAA435E1D67B6A20CA233D51DDC9036F312071189D8D27F4
```

本 Stage A 没有引入新的纯物理或纯数学理论，因此不修改 `docs/theory_notes/`。

### 18.9 R10 Stage A 实现与正式运行前锁定

本节在正式 `512² -> 1024²` scientific workload 尚未启动、结果仍未知时追加。实现新增四个独立文件：

```text
configs/experiments/exp040_TGV_3d_multislice_r10_stage_a.yaml
scripts/run_exp040_r10_stage_a.py
src/tgv_ptycho/viz/plot_exp040_r10.py
tests/test_exp040_r10_stage_a.py
```

实现直接复用 `_r7_streamed_tgv_exit`、`_r9_project_with_controls`、`_r9_comparison_metrics`、
`restrict_aligned_cell_average`、`_r9_normalized_error_map`、`_r9_log_difference_spectrum` 和
`make_physical_passband_mask`，没有修改共享 exp040 forward kernel、R9 结果、threshold、theory note 或 HDF5
schema。runner 对 science-controlling YAML 值和源配置 SHA256 双重检查；正式计算逐 case streamed，不保留
`(400, ny, nx)` volume。Windows 大数组路径继续使用显式 `sum(abs(.)**2)` / `sum(conj(a)*b)`，正式 runner
不含 `np.vdot`。

结果未知前完成的验证为：

```text
R10/R9/preflight targeted regression: 28 passed in 4.61 s
full repository regression:             240 passed in 38.44 s
modified-file Ruff:                     All checks passed
Python compile check:                   passed
registered source-config validation:    passed
```

上述测试中的 runner contract 使用 monkeypatched `8x8` map 和临时 pytest run 目录；微型数值测试只使用
`8²/16² x 4 slices`，均不是正式 Stage A scientific workload。正式 `512²/1024² x 400 slices` 的执行次数
在本节追加时仍为 `0`。

实现与前缀锁定为：

```text
document prefix bytes: 88485
document prefix SHA256: 38931AF59D600BA97A1BE48C2674FB9B9D76F0ACC87F4005E65517EAB0199629
formal config bytes: 3164
formal config SHA256: 3585AB185DD2E71A2D3872B310343A5F04F8F3B64B5D57BF9B640105A344D654
formal runner bytes: 32588
formal runner SHA256: 9A515954963705D445B68DC417131EB0A683D0D11FA071842CCD203BF523E7AA
R10 plot bytes: 3953
R10 plot SHA256: E1DA46D5504A1369F25B79A83C538A37FF3BE362AD5F836BEFA90AD3CA647F27
R10 test bytes: 7451
R10 test SHA256: BD8BB2F8E96EBCC6D96C1BB84D00D64FB7924A9BE09B41E18E1D2AEEA4385D90
```

唯一正式命令锁定为：

```powershell
& 'D:\anaconda3\envs\tgv_ptycho_sim\python.exe' -u -X faulthandler scripts/run_exp040_r10_stage_a.py --config configs/experiments/exp040_TGV_3d_multislice_r10_stage_a.yaml
```

本次只允许启动该正式命令一次。若进程或产物验证失败，runner 仍保留 timestamped run、traceback 和
`formal_attempt_retained=true`；不得无记录重跑或在看到结果后修改配置、restriction、denominator、hard
controls、5% gate 或 outcome table。

### 18.10 R10 Stage A 唯一正式执行结果、artifact failure 与非重算修复预注册

第 18.9 节锁定的正式命令只启动了一次，创建：

```text
run: runs/exp040_TGV_3d_multislice_r10_stage_a_20260815_162021
formal run count in workspace: 1
run started: 2026-08-15T08:20:21.762589+00:00
scientific postprocessing completed: 2026-08-15T08:28:01.565021+00:00
scientific execution elapsed: 459.8036773 s
shell exit code: 1
```

两个且仅两个预注册 case 均完成了全部 400 slices：

| case | elapsed [s] | air-volume relative error | fraction/index/count bound errors | width-sum error [m] |
|---|---:|---:|---:|---:|
| `current_512` | `72.2361753` | `3.61956465e-6` | `0 / 0 / 0` | `0` |
| `fine_1024` | `386.7073341` | `9.15173126e-7` | `0 / 0 / 0` | `0` |

冻结 comparison 的实际科学结果为：

```text
raw relative L2:                    0.04574990331167789 = 4.574990331%
external-passband relative L2:      0.023787308028510038 = 2.378730803%
registered convergence gate:        0.05 = 5%
raw pass (report-only):              true
external-passband gate pass:         true
scientific Stage A status:           Passed
interpretation:                      scalar_lateral_reference_closed
stage_b_allowed:                     true
```

raw difference energy 中 `26.3799924%` 位于 external propagating passband 内，`73.6200076%` 位于其外；
Parseval closure error 为 `2.10897e-16`，inside/outside orthogonality error 为 `5.92130e-17`。共同物理频率
spacing 均为 `15625 cycles/m`，两 native masks 均含 `45461` 个频点。512²/1024² projected field retained
energy fraction 分别为 `0.9639665285/0.9718112501`。

全部 hard controls 通过：maximum algebra error `2.47685025e-16 <=1e-12`，restriction complex-mean error
`1.24801107e-16`，constant/alignment error 均为 `0`，postprocessing determinism 为 `0`，所有 field finite。
相对 R9 `0.25 -> 0.125 um` pair，raw/passband error 分别缩小 `4.56272x/7.18443x`，对应表观阶
`2.18989/2.84487`；按预注册它们只支持“当前 pair 明显下降”的经验判断，不是 uniform-grid convergence 的
数学证明。

因此 Stage A 的数值结论已经闭合：在 canonical scalar multislice、固定 `dz=0.25 um`、q8 和 64 um FOV
下，`dx=0.125 -> 0.0625 um` 的共同 external-passband pair 已低于 5%，并且 raw pair 也低于 5%。这允许
进入第 18.3 节的 Stage B 双向标量 Helmholtz comparator，但不等于 Helmholtz、vector Maxwell、真实 sample
标定或整个 exp040 已完成。

#### Artifact failure 的独立状态

科学 postprocessing 和 `metrics.json`、两张 PNG 写入后，compact HDF5 writer 在序列化
`case_controls: [dict, dict]` 时触发：

```text
TypeError: Object dtype dtype('O') has no native HDF5 equivalent
formal run_state: failed_during_execution
latest progress event: artifacts_writing_started
```

只读检查确认 JSON 中唯一的 nested list-of-mappings 是 `case_controls`；原 writer 把它转成 NumPy object
array，h5py 不接受。故这是 artifact serialization contract 的测试覆盖缺口，不是 field、FFT、passband、
restriction 或物理模型失败。原 partial HDF5 和含 traceback 的 `run_state.json` 必须保留，不能把 shell
exit code 改写成 0，也不得重跑正式网格。

现有关键 artifact 锁定为：

```text
config.yaml SHA256:                       5D49274C41DEBE9742D00183FEFD46E6D4F5551F45C59BBEC32A8A468A042892
metadata.json SHA256:                     FEBE74B948FC59BC716DC43745C9BCF25758ECF77C6127BFAFB13A94BD934DC9
metrics.json SHA256:                      3346A2463E7B374EBE850E4CE621A133307F359EF5EBC66D70458E91F0602817
run_progress.json SHA256:                 009533D2B32F20BA12D28834E5D1E02ED8FD56F0EABCA95F4EE8C926C557B9C6
failed run_state.json SHA256:              9A54A9F68464216F78EE1DDC04095DECDEEC6B65D59A36EE94418E02D79265DE
r10_stage_a_convergence.png SHA256:       BD66F6DEA1A7F199F6747695430D23B318D8FAFCA6C12CA7AEFEFF490457E7A0
r10_stage_a_residuals.png SHA256:         BFBB0B03987D83F9503A59C42D7A8EA3B14034A770DFCF506C7DF00580D0FFD1
partial exp040_r10_stage_a.h5 SHA256:       18034DB747102515633D55C86EDE57F7304240D7BBC4B29E3798292A29DF52A8
document prefix bytes:                    91142
document prefix SHA256:                   5511C4FC7500039C0BDE839BF9280D2F13F9AC9551EF98DC43B0C0254BB079B1
```

为完成机器可读归档，预注册一次非科学、非重算 artifact repair：

1. 只接受上述 exact run，并先验证 config/metadata/metrics/failed-state/partial-HDF5 SHA256 与 failure
   signature；任一不符即停止；
2. 不导入或调用 exp040 forward、optics、FFT、restriction、plot 或任何 scientific comparator；只读取已锁定
   的 YAML/JSON；
3. 不覆盖或删除 `outputs/exp040_r10_stage_a.h5`、`run_state.json`、`run_progress.json`、metrics 或 PNG；
4. 只把 HDF5 副本中的 `case_controls` 从有序 list 转为以冻结 id `current_512/fine_1024` 为键的 mapping，
   其余 metrics 值保持 JSON 原值；新文件固定为
   `outputs/exp040_r10_stage_a_repaired.h5`；
5. 新增 `artifact_repair.json`，保存原失败、输入/输出 hash、`scientific_recomputation=false` 和结构/关键指标
   验证结果；repair state 不替代 formal `run_state`；
6. repair 必须验证 HDF5 具有并列 config/data/instrument/sample/metadata/metrics，data 为空、无 truth/
   reconstruction，两个 case controls 完整，且 HDF5 raw/passband/status/stage-B 值与锁定 JSON 完全相同。

该 repair 不改变 Stage A 的科学指标或状态，也不把正式 shell failure 伪装成成功。repair 实现、测试和 hash
必须在执行前继续追加锁定；只允许运行一次 repair。这里没有新增纯物理或纯数学理论，仍不修改
`docs/theory_notes/`。

### 18.11 R10 Stage A artifact repair 实现与执行前锁定

本节在 repaired HDF5 与 `artifact_repair.json` 均不存在、repair 执行次数为 `0` 时追加。repair 实现和测试
只新增：

```text
scripts/repair_exp040_r10_stage_a_artifacts.py
tests/test_exp040_r10_stage_a_artifact_repair.py
```

实现硬编码第 18.10 节的 exact run、六个输入 SHA256、formal failure signature、两个 400-slice completion
events 和五个关键 scientific values。它对 JSON 作深复制，只将 HDF5 表示中的 `case_controls` 有序列表改成
`current_512/fine_1024` keyed groups；随后把整个 normalized metrics 从 HDF5 读回并与内存对象完全相等比较。
源码不导入 `tgv_ptycho.forward`、`tgv_ptycho.optics`，不含 `np.fft`、forward kernel、comparator 或 plot call。

执行前验证为：

```text
repair + Stage A targeted tests: 13 passed in 1.46 s
modified-file Ruff:              All checks passed
Python compile check:            passed
repaired HDF5 exists:            false
artifact_repair.json exists:     false
```

实现锁定为：

```text
document prefix bytes: 96993
document prefix SHA256: 5FF1887796E888021B9B81695767EBFEF19FE84159215602FF47D399214ED91B
repair script bytes: 11650
repair script SHA256: CE654B3F65653D653B7E6C2FAD9E5760BE65EF97D0D9D4B4D2501A574E00AA73
repair test bytes: 5274
repair test SHA256: E65E20C1EFEEDA70951B60F8A0FFB0D4C19E1B5ECF7470E7ECDEB41EED834666
```

唯一 repair 命令锁定为：

```powershell
& 'D:\anaconda3\envs\tgv_ptycho_sim\python.exe' -u scripts/repair_exp040_r10_stage_a_artifacts.py --run-dir runs/exp040_TGV_3d_multislice_r10_stage_a_20260815_162021
```

该命令不是 scientific run，不增加正式 Stage A 执行次数。它只允许运行一次；输出已存在时必须拒绝覆盖。

### 18.12 R10 Stage A artifact repair 结果、最终状态与下一步

第 18.11 节锁定的 repair 命令执行一次并以 exit code `0` 完成，没有 scientific recomputation：

```text
repaired HDF5:
  runs/exp040_TGV_3d_multislice_r10_stage_a_20260815_162021/
  outputs/exp040_r10_stage_a_repaired.h5
bytes: 100000
SHA256: 8044FB2CA0B39E7F72D3FBCA97E98F376C4DDE6B1C284E23B752F132CEA3F16B

repair record:
  runs/exp040_TGV_3d_multislice_r10_stage_a_20260815_162021/artifact_repair.json
bytes: 2255
SHA256: 88C7E5D296DD357119B75EC9C77C2D3771169EB9D4265690AE1813383CC2066F
```

repair record 明确保存：`scientific_recomputation=false`、`forward_or_fft_called=false`、
`plots_recomputed=false`、formal shell exit code 仍为 `1`。修复后的 HDF5 `/entry` 恰含并列的
`config_yaml/data/instrument/sample/metadata/metrics`；`data` 为空，无 `truth/reconstruction`；两个 case-control
groups 完整；全量 normalized metrics HDF5 read-back 与 JSON 完全相等。raw `0.04574990331167789`、passband
`0.023787308028510038`、`Passed` 和 `stage_b_allowed=true` 均精确一致。

原 `config/metadata/metrics/progress/failed-state/partial-HDF5` 的六个 SHA256 在 repair 后全部与第 18.10 节
一致，因此失败证据和正式执行 provenance 没有被覆盖。最终代码验证为：

```text
full repository regression: 244 passed in 36.51 s
all R10 modified-file Ruff:  All checks passed
Python compile check:        passed
```

最终状态必须分三层阅读，不能合并成一个模糊的“成功”或“失败”：

1. **科学 gate**：`Passed / scalar_lateral_reference_closed`；共同 passband 为 `2.37873% <5%`，raw 也为
   `4.57499% <5%`；
2. **唯一 formal wrapper**：两个科学 case 和 postprocessing 完成，但原 compact-HDF5 serialization 失败，
   shell exit `1` 与 failed `run_state` 被保留；
3. **非重算机器归档**：单独 repaired HDF5 已成功、完整验证，并由 `artifact_repair.json` 记录。

这一结果说明 R9 的 `dx=0.25 -> 0.125 um` 大偏差并不是当前 scalar multislice lateral reference 无法跨越的
固定 floor；继续到 `0.125 -> 0.0625 um` 后，冻结 gate 下的 combined lateral discretization difference 已
闭合。它仍没有严格拆开 interface geometry representation 与 lateral propagation sampling 的各自份额，也不
构成连续极限证明；因此不应继续机械增加 uniform resolution，也不应宣称模型已经物理完备。

下一步按第 18.3 节进入 R10 Stage B：在结果产生前具体锁定双向标量 Helmholtz solver、mesh refinement pair、
PML enlargement pair、radial/axial domain、incident-field injection、multislice-to-Helmholtz sampling operator、
homogeneous analytic control 和 azimuthal-anisotropy control。只有 solver controls 先通过，才比较 backward/
reflected-wave physics 是否在共同 external passband 中产生 `>5%` 的可分辨差异。Stage B 不应再增加
uniform `dx=0.03125 um`，也不应提前升级到 vector Maxwell。

本轮没有新增纯数学或纯物理理论，未修改 `docs/theory_notes/`。没有修改 Git staged、commit、push 或 PR
状态。

### 18.13 R10 Stage B 双向标量 Helmholtz comparator 正式预注册

本节在任何 Stage B field、metric 或 scientific run 产生之前追加。此前文档保持不变；追加前文档前缀为
`101990 bytes`，SHA256 为
`2CE808D87B2B054B54D8A084F1A53CA8E699A1A70239D60FB226C2DBD99D2AA1`。对应纯数学与纯物理说明继续追加到
`docs/theory_notes/exp040_r10_bidirectional_scalar_helmholtz_comparator.md`；其本次追加前前缀为 `4343 bytes`，
SHA256 为 `4D1A98739A33963F91315CF9F70CE02FB962888F167AAA8CC33A264C96A3651F`。

#### 研究问题与结论边界

Stage A 已在冻结的共同 external passband 下得到 `2.378730803% <5%`，因此 Stage B 不再机械增加
multislice lateral resolution，而只回答：在同一 canonical TGV 几何和同一输出面上，经自身 mesh/PML controls
验证的**双向标量** Helmholtz 解，与单向 scalar multislice 解是否留下 `>5%` 的结构化共同频带差异。

本 comparator 允许 scalar backward/reflected waves，但仍不含 polarization、vector boundary conditions、
anisotropy、roughness、loss/dispersion 或真实几何标定。即使 cross-model gate 被越过，也只能写成
“bidirectional scalar model difference resolved”，不能宣称已经证明真实误差来自反射，更不能把 Helmholtz
结果称为 vector-Maxwell truth。

#### 连续模型、背景场与 normalization

冻结 scattered-field/TFSF 分解为

```text
(laplacian + k0^2 n_TGV^2) u_s
  = -k0^2 (n_TGV^2 - n_bg^2) u_bg
u_total = u_bg + u_s
```

其中背景平面界面固定为：`z <100 um` 为 glass (`n=1.5`)，`z >=100 um` 为 air (`n=1.0`)；单位振幅
平面波在 glass 中沿 `+z` 正入射。时间约定固定为 `exp(-i omega t)`。标量界面系数固定为
`r=(k_g-k_a)/(k_g+k_a)=0.2`、`t=2k_g/(k_g+k_a)=1.2`。解析 `u_bg` 在 PML 中按复坐标继续，只有
局域散射场 `u_s` 进入径向/轴向 PML；禁止把无限平面背景波直接放入径向 PML。

canonical TGV 固定为 `0 <=z <=100 um` 的 air-filled axisymmetric via：
`D_top=30 um`、`D_waist=20 um`、`D_bottom=30 um`、`z_waist=50 um`，外部为 glass。输出面固定为
air 中 `z=101 um`；若输出面位于两个 cell centers 之间，使用其两侧 cell-center total fields 的固定线性平均，
不得改为看后选择最近 cell。Helmholtz normalized trace 固定为
`v_H=(u_bg+u_s)/u_bg`；必须保存 raw total/scattered controls，且禁止 phase、scale、shift 或 tilt alignment。

multislice comparator 固定重新计算 Stage A 的 `1024^2, dx=0.0625 um, dz=0.25 um, q8` case；其
`z=100 um` exit field 与解析 homogeneous glass control 分别使用 band-limited air ASM 传播 `1 um`，再在
`z=101 um` 相除得到 `v_MS`。共同平面 transmission 常数若显式加入会在该 normalization 中严格抵消，正式
实现固定为不另乘 transmission；不得在结果后增加角度依赖 Fresnel operator。

#### 轴对称 finite-volume、材料与 PML

在 cell-centered `(r,z)` 网格上冻结守恒型二阶 five-point 离散：

```text
d_r[(r_tilde s_z/s_r) d_r u]
+ d_z[(r_tilde s_r/s_z) d_z u]
+ k0^2 n^2 r_tilde s_r s_z u = source.
```

`r=0` 使用自然 zero-flux regularity；PML 最外边界对 `u_s` 使用零 Dirichlet，cell center 到边界的半格距离
必须进入 boundary flux。无 periodic boundary。TGV 每个径向 annular cell 的 disk-intersection area fraction
用半径平方差解析积分；每个 sample axial cell 固定使用 8 个等权 midpoint subnodes；mass term 使用
cell-average `n^2=f_air*n_air^2+(1-f_air)*n_glass^2`。这与 multislice 的 q8 Cartesian `n` screen 是两个
模型各自冻结的材料离散，不能看结果互换。

PML 使用 cubic stretch `s=1+i alpha`。`alpha_max` 由
`exp[-k_ref integral(alpha dl)]=1e-8` 固定求得；radial PML 的 `k_ref=k_air`，lower-z PML 使用
`k_glass`，upper-z PML 使用 `k_air`。物理 core 固定为：

```text
r: 0 ->24 um; radial PML starts at 24 um
z: -2 ->102 um; lower/upper axial PML start at -2/102 um
comparison region: r <=20 um
outer guard: 22 <=r <=24 um
observation plane: z=101 um
```

正式 case 顺序和 mesh 不得更改：

| case | `dr=dz` | PML | `(nr,nz)` | unknowns |
|---|---:|---:|---:|---:|
| `coarse_nominal` | `0.125 um` | `2 um` | `(208,864)` | `179712` |
| `fine_nominal` | `1/12 um` | `2 um` | `(312,1296)` | `404352` |
| `fine_enlarged_pml` | `1/12 um` | `3 um` | `(324,1320)` | `427680` |

#### 线性求解、preflight 与资源 gate

正式 solver 固定为 SciPy SuperLU `splu`、CSC、`permc_spec=COLAMD`、默认 partial pivoting；每次只保留一个
case 的 sparse matrix/LU，提取 compact trace 后立即释放并 `gc.collect()`。禁止因某个 case 困难而改用裸
GMRES、未预注册预条件器、不同 ordering 或降低 residual threshold。

非科学 preflight 只在 `fine_nominal` formal grid 上构造实际 cylindrical-PML homogeneous matrix；用满足
axis regularity 和 outer homogeneous boundary 的固定平滑 manufactured vector `w`，令 `b=A w` 后执行一次
factor/solve，并同时检查解析 background interface continuity、zero-contrast normalization、matrix finite、wall
time 和进程 peak RSS。它不生成 TGV cross-model metric，不能成为科学结论。当前只读资源核对为：总物理内存
约 `13.871 GiB`、preflight 前可用约 `4.253 GiB`；此前纯 sparsity 资源诊断估计 fine LU 的 L/U numeric
lower bound 为约 `1.592 GB`。

preflight go/no-go 在结果前冻结为：

```text
available physical memory before factorization: >=3.0 GiB
factor + solve wall time:                       <=180 s
process peak RSS:                               <=6.0 GiB
relative solve residual:                        <=1e-9
manufactured-vector recovery relative L2:       <=1e-8
background value/derivative continuity error:   <=1e-12
zero-contrast normalization error:              <=1e-12
all matrix/RHS/solution values finite:           required
```

任一项失败，固定写为 `Blocked / formal_grid_preflight_failed`，不执行 Stage B formal scientific run。本机预计
preflight 需要一次约 404k unknown direct factorization、数 GB 峰值内存和数十秒以内；formal run 预计还需
三次顺序 Helmholtz factor/solve，加一次已知约 6–7 分钟的 1024² multislice。正式执行期间每个关键 case
完成后立即写 progress/checkpoint，不重复已完成 case。

#### 冻结 sampling/comparison operator

所有 Helmholtz normalized radial traces 先按固定线性 interpolation 投影到 centered Cartesian
`512^2, dx=0.125 um, FOV=64 um`。`r<=24 um` 使用 trace；`r>24 um` 固定填 homogeneous normalized
value `1`，但只有 outer-guard control 先通过才允许使用该 extension。各 Cartesian field 使用现有
`n_air/lambda0=1879699.2481203007 cycles/m` inclusive FFT mask。

passband projection 后，使用固定 `0.125 um` annular bins（edges `0:0.125:20 um`，160 bins）计算 Cartesian
pixel-center azimuthal mean。multislice 必须在 native 1024² 上先做同一物理 passband，再作 aligned 2x2
complex cell-average restriction 到 512²，之后使用同一 annular mean。径向 relative L2 固定为 `2*pi*r`
加权，不含 phase/scale/shift alignment。

reference pairs 和 denominator 固定为：

```text
mesh:        coarse_nominal vs fine_nominal; denominator=fine_nominal
PML:         fine_nominal vs fine_enlarged_pml; denominator=fine_enlarged_pml
cross-model: v_MS vs fine_enlarged_pml Helmholtz; denominator=fine_enlarged_pml
```

同时保存/report raw 与 passband 值，但科学 gate 使用 passband。multislice Cartesian azimuthal-anisotropy
residual 固定为 `||v_MS_pass - annular_projection(v_MS_pass)||_2 / ||annular_projection(v_MS_pass)||_2`；不得通过
径向平均隐去未通过的 Cartesian grid bias。outer guard 对三个 Helmholtz case 分别用 output-plane
`u_s/u_bg` 的 `2*pi*r` 加权 RMS，报告 `RMS(22–24 um)/RMS(r<=20 um)`。

#### 阈值、outcome 与 artifact

以下阈值在 field 结果前冻结：

```text
sparse solve relative residual:                    <=1e-9
algebra controls:                                  <=1e-12
postprocessing determinism relative L2:            <=1e-14
mesh-convergence passband radial relative L2:      <=5%
PML-enlargement passband radial relative L2:       <=5%
homogeneous analytic continuity/normalization:     <=5%
multislice passband azimuthal-anisotropy residual: <=5%
each Helmholtz outer-guard RMS ratio:               <=5%
cross-model materiality gate:                       5%
```

homogeneous control 中解析恒等式仍按 algebra `1e-12` 检查；上面的 5% 只为与场级 comparator 的统一 hard
上限，不能放宽解析恒等式。若任一 solver、finite、mesh、PML、homogeneous、anisotropy 或 guard control
失败，scientific status 固定为 `Failed / helmholtz_reference_not_validated`，cross-model 数值仅可诊断性报告，
不得解释模型物理。全部 reference controls 通过后：cross-model passband `<=5%` 写为
`Passed / bidirectional_scalar_effect_not_resolved_at_registered_gate`；`>5%` 写为
`Passed / bidirectional_scalar_model_difference_resolved`。

计划新增且只在实现后锁 hash 的文件为：

```text
configs/experiments/exp040_TGV_3d_multislice_r10_stage_b_preflight.yaml
configs/experiments/exp040_TGV_3d_multislice_r10_stage_b.yaml
src/tgv_ptycho/forward/helmholtz_axisymmetric.py
scripts/run_exp040_r10_stage_b_preflight.py
scripts/run_exp040_r10_stage_b.py
src/tgv_ptycho/viz/plot_exp040_r10_stage_b.py
tests/test_helmholtz_axisymmetric.py
tests/test_exp040_r10_stage_b_preflight.py
tests/test_exp040_r10_stage_b.py
```

preflight 与 formal run 各创建独立 timestamped run，不覆盖历史。formal artifact 至少保存 config、metadata、
metrics、progress、state、compact HDF5、三张 PNG；HDF5 保存 compact radial traces 和必要的 512² selected
fields，不保存 sparse matrix、LU、full `(r,z)` volume 或 400-slice Cartesian volume。实现/test/hash 将在 preflight
前继续 append；preflight 通过后再锁正式 source hash，并且只执行一次 formal Stage B。任何失败都保留原 run
与 traceback，禁止看结果修改本节阈值、mesh、PML、denominator 或 conclusion code。

### 18.14 R10 Stage B 实现、执行前澄清与 preflight 锁定

本节在 formal-grid preflight 尚未执行、没有任何 Stage B field 或 metric 时追加。第 18.13 节的研究问题、mesh、
PML、threshold、denominator 和 outcome code 均不改变。

#### Exact-plane background 的执行前澄清

第 18.13 节“输出面位于两个 cell centers 之间时线性平均 total fields”的字面实现会把已知解析平面 carrier
也作低阶插值。在当前 `dr=dz` 下，air carrier 每格相位并不小，这会人为制造与 TGV scattering 无关的
homogeneous interpolation error。由于 scattered-field/TFSF 的唯一数值未知量本来就是 `u_s`，正式实现从结果
产生前固定澄清为：

```text
1. 只对两侧 cell centers 的 u_s 作固定线性平均，得到 exact z=101 um 的 u_s；
2. 在 z=101 um 解析求 u_bg；
3. u_total(z=101 um) = u_bg_exact + u_s_interpolated；
4. v_H = u_total / u_bg_exact。
```

因此第 18.13 节的“total-field linear average”执行语义由本段替代。它没有拟合数据，也不改变 scattering
equation、observation plane 或 normalization；其目的只是利用预注册中已经指定的解析 background，避免把已知
carrier 的低阶插值误差误判为双向散射。所有 Helmholtz case 仍固定报告 bracketing centers 和 interpolation
weight，当前三组网格的注册 weight 均应为 `0.5`，偏差按 algebra gate 检查。

#### 实现与 Windows 环境诊断

新增实现已包含：cylindrical complex stretch、annular/axial-q8 `n^2` integration、cell-centered conservative
matrix、contrast source、SuperLU residual/RSS controls、radial-to-Cartesian/passband/annular operators、四个顺序
checkpoint、compact HDF5 与三张 PNG。正式 multislice generator 每 25 slices 写一次 progress，避免高成本步骤
长期没有可审阅状态。

微型测试曾稳定定位一个非科学环境故障：通过 Python 绝对路径而未执行 `conda activate` 时，进程 `PATH` 中有
base Conda 的 `Library/bin`，却没有当前 `sys.prefix/Library/bin`；real 2x2 SuperLU 正常，而任意 complex 2x2
SuperLU 在 delay-load complex BLAS 时触发 Windows `0xc06d007f`。只在当前进程、且在导入 SciPy 前补入
`sys.prefix/Library/bin` 后，同一 complex 2x2 solve 正常。实现固定保留这一等价于正确激活当前 Conda 环境的
兼容处理；没有改成 real-block system、没有换 solver，也没有放宽阈值。RSS 使用 Windows/psutil 已保存的
process lifetime peak working set，在 factor 前后读取；不启动会干扰当前 SuperLU 的采样线程。

执行前验证为：

```text
Stage-B targeted tests: 11 passed in 1.57 s
full repository regression: 255 passed in 37.84 s
all Stage-B modified-file Ruff: All checks passed
Python compile check: passed
preflight run exists: false
formal Stage-B run exists: false
```

preflight-ready 文件锁定为：

```text
theory note bytes/SHA256:
  7924 / 690BAF00D44556284E601C3C129B5714F86949CDBABAB37CBEE2A8BEF1E9788F
preflight config bytes/SHA256:
  1896 / 5EA2F6C15D2C8930BE7AC7048AFD0350DE7BEF1EABA1381D17EA6EC1DA1F08D9
Helmholtz module bytes/SHA256:
  32367 / B93A6BCCFA73933FB8BAC1061530F5DD3899F4C949F802F0DFCD169BF4AAC440
preflight runner bytes/SHA256:
  18397 / B415A10F2EBE742092144321687C3A6A5E8FEBB9F9573931014A53A00DEDBFE7
Helmholtz tests bytes/SHA256:
  5471 / 172A1D1D8A14897348F38A25BC955E739F00CCB555713A66746A43EDC4F2BAD5
preflight tests bytes/SHA256:
  1562 / 8CF093E8804EB077A61760470DA6696B071D887E7CA8A67B723FEB5A0CD7393A
```

formal runner、plot、test 和 config 已实现，但 config 的三项 preflight provenance 以及 runner 的 source-config
hash 仍明确为 `__LOCK_AFTER_PREFLIGHT__`，所以此刻无法误执行 formal scientific run；它们要在 preflight
通过后只补 provenance/hash，再重新测试和追加最终锁。唯一 preflight 命令固定为：

```powershell
& 'D:\anaconda3\envs\tgv_ptycho_sim\python.exe' -u scripts/run_exp040_r10_stage_b_preflight.py `
  --config configs/experiments/exp040_TGV_3d_multislice_r10_stage_b_preflight.yaml
```

该命令只执行一次 non-scientific formal-grid manufactured solve。只有其全部资源、residual、recovery、algebra 和
finite controls 通过，才允许补锁并运行一次 formal Stage B。

### 18.15 R10 Stage B preflight 结果与唯一 formal run 锁定

第 18.14 节锁定的 non-scientific preflight 命令执行一次并以 exit code `0` 完成：

```text
run:
  runs/exp040_TGV_3d_multislice_r10_stage_b_preflight_20260815_181436
status / interpretation:
  Passed / formal_grid_preflight_passed
formal_stage_b_allowed: true

available physical memory before: 4.071441650390625 GiB  (gate >=3 GiB)
grid:                             nr=312, nz=1296, N=404352
matrix nnz:                       2018544
factor L+U nnz / fill ratio:      66615351 / 33.001683887
factor / solve / combined:        7.5167525 / 0.1522589 / 8.0577743 s
process peak RSS:                 2776797184 bytes = 2.586094 GiB  (gate <=6 GiB)
relative solve residual:          1.286010858985897e-14  (gate <=1e-9)
manufactured recovery L2:         2.0408142194058487e-12 (gate <=1e-8)
maximum algebra error:            0
background value/derivative error: 0 / 0
zero-contrast normalization error: 0
all finite:                       true
```

七个 control-pass flags 全为 `true`。HDF5 `/entry` 恰含
`config_yaml/data/instrument/metadata/metrics`，`data` 为空，JSON/HDF5 均回读为 `Passed` 和
`formal_stage_b_allowed=true`；`run_state.json` 为 `complete` 且 `artifacts_validated=true`。关键 artifact
锁定为：

```text
metrics.json: 2848 bytes
SHA256: 22D6F1D0BBF526B4DD077934963DCDF80E8FC8DA5E8A84E5F5A9EFE87F546D7F

outputs/exp040_r10_stage_b_preflight.h5: 50864 bytes
SHA256: 64EDFB696183675419C114F7AD8B583035CA7E62FCA8B2EA2562CC4AF658F013

run_state.json SHA256:
8F893860D9E657DF550FC922FAD830823D412FB4D08D99EB56C30334F7E682E1
run_progress.json SHA256:
F369380412776C4FF19301A5F9B60570286B83123D389DE9FE0F9C0DECB4332E
```

这只证明 frozen formal-grid direct solve 在当前机器上资源可行并达到 algebraic accuracy；它不是 TGV field
结果，也不预告 mesh/PML/cross-model gate。formal config 的三项 provenance 现已仅用上述 exact run 与 hash
替换占位符，没有改变任何 physics、mesh、operator、threshold 或 denominator。

正式执行前最终验证为：

```text
Stage-B targeted tests: 12 passed in 1.55 s
full repository regression: 256 passed in 37.15 s
all Stage-B modified-file Ruff: All checks passed
Python compile check: passed
formal Stage-B run exists: false
```

追加本节前的文档前缀为 `116669 bytes`，SHA256 为
`224C4BC14FC6B08F4E40395BE4F12DAF705F539101347E3E9FC3766737F1DCBE`。formal scientific implementation
最终锁定为：

```text
formal config:
  5244 bytes
  AF718435D573DA13DD5D01D9AAC13A47B51281D7B2B1745379EC6ADCBD023652
Helmholtz module:
  32367 bytes
  B93A6BCCFA73933FB8BAC1061530F5DD3899F4C949F802F0DFCD169BF4AAC440
formal runner:
  59336 bytes
  0272E1B8CB2F1BFFDE183FAA35552A5C98ACDA7784061CD52BD2A9B41A7A4843
Stage-B plot module:
  5581 bytes
  29738CC8CBE0D63DDCE0E36900123FF2C0B0BA91C92B59255094BA030807D9ED
formal tests:
  6370 bytes
  31EC2C3162C85151CC28024FC834C327EBFEC1C513FC6609439FA6C1392D4CA5
```

唯一 formal scientific command 固定为：

```powershell
& 'D:\anaconda3\envs\tgv_ptycho_sim\python.exe' -u scripts/run_exp040_r10_stage_b.py `
  --config configs/experiments/exp040_TGV_3d_multislice_r10_stage_b.yaml
```

预计三个 Helmholtz solve 共数十秒以内；1024²、400-slice q8 multislice 依据 Stage A 实测约 6–7 分钟，是
主要成本。命令只执行一次；每个 Helmholtz case 和 multislice 完成后立即保存 checkpoint，任何 reference control
失败仍保留 cross metric 作为诊断，但结论固定为 `Failed / helmholtz_reference_not_validated`。

### 18.16 R10 Stage B formal 结果、artifact failure 与 repair 预注册

第 18.15 节锁定的 formal scientific command 只执行一次。三个 Helmholtz solves、400-slice multislice、完整
共同算子和重复 determinism check 均已完成；四个 checkpoint、`metrics.json` 和 `metadata.json` 已在失败前保存：

```text
run:
  runs/exp040_TGV_3d_multislice_r10_stage_b_20260815_181819
forward/postprocessing elapsed: 410.4745739 s
scientific status / interpretation:
  Failed / helmholtz_reference_not_validated
reference_validated: false
```

冻结 reference controls 的实际结果为：

| control | metric | 5%/registered gate | pass |
|---|---:|---:|---:|
| mesh `coarse_nominal -> fine_nominal` | `1.1063644834324253` | `<=0.05` | false |
| PML `2 ->3 um` on fine mesh | `1.0821141299585554e-5` | `<=0.05` | true |
| multislice azimuthal anisotropy | `0.09549961270718764` | `<=0.05` | false |
| maximum outer-guard ratio | `0.7012049647762314` | `<=0.05` | false |
| maximum solver residual | `4.272005221024851e-14` | `<=1e-9` | true |
| maximum algebra error | `1.6264767310758543e-13` | `<=1e-12` | true |
| postprocessing determinism | `0` | `<=1e-14` | true |
| homogeneous field error | `2.6295566127902845e-13` | `<=0.05` | true |

各 Helmholtz case 的 outer-guard ratio 为：

```text
coarse_nominal:     0.7012049647762314
fine_nominal:       0.5901209390505856
fine_enlarged_pml:  0.5901175657151257
```

PML pair 已高度稳定，而 coarse/fine mesh pair 尚未接近共同极限；同时散射场在 `22–24 um` guard 中仍是
`r<=20 um` inner RMS 的 59–70%，说明 `r=24 um` extension/fill 假设不能被验证。multislice 的固定 Cartesian
anisotropy 也为 9.55%。因此失败不是 sparse solve、PML thickness、homogeneous normalization 或 postprocessing
不确定性造成，而是至少 mesh pollution、物理 radial core/support 和 Cartesian anisotropy 三个 reference controls
尚未闭合。

cross-model 仍按预注册保存为诊断：raw/passband radial L2 分别为
`1.1127503579066322/1.1128332252351876`；Cartesian report-only passband L2 为
`0.7608790350410168`。由于 reference 未验证，`111.283%` 不能解释为 backward/reflected-wave physics，也不能
触发更复杂物理 comparator。按预注册，本轮唯一科学结论就是
`Failed / helmholtz_reference_not_validated`。

#### Artifact failure

科学 metrics 保存后，plot wrapper 在第三张 radial figure 读取了
`radial_profiles["multislice_passband"]`，而正式 result 的冻结键是
`radial_profiles["multislice_fine_1024_passband"]`，触发：

```text
KeyError: 'multislice_passband'
formal shell exit code: 1
formal run_state: failed_during_execution
```

故原 HDF5 和三张 PNG 尚未写出。该错误发生在 `postprocessing_completed` 和 metrics/metadata 保存之后；不是
field、mesh/PML metric、passband、annular mean 或 outcome failure。禁止重新执行 formal forward，也禁止把原
shell failure/run_state 改写为成功。

失败证据与科学输入锁定为：

```text
config.yaml:
  B77B7D02B80BF45F3BD917BB821AFC0452E24F38418EAF88FDC4EA62AD5541AA
metadata.json:
  052CE5781219A58906967A85E1AD3221349E0C959570138A827AE9101E208ABA
metrics.json:
  60DAE2C59FB89EA2D201D53C105BCEAF98A6F36D163D9401146058F0F27DB11A
run_progress.json:
  FC8947CD6151C262E2F4A3744E863E662BF49DBEE2180923D3C1954CE999083E
failed run_state.json:
  559B8B51CBF19C1F42221FD917E28446DAEAC37999ABD1C11B37A8DAAFBB0F12
coarse checkpoint:
  899237BB35003FA50AE0C010D3D9A4F047C1CE6BB89ED8838B7A2277060DB1A7
fine-nominal checkpoint:
  F2EE0DBF1DF82342DD7434FB77CE0606015670A3E3531EDC22D1DBF405F32640
fine-enlarged-PML checkpoint:
  3C81CB4D4A5BC9D50ED239BD269B219DB34F312CD01930D4C554FC1843B7B802
multislice checkpoint:
  5D75B300BAE6B3F0A5202A55AE99942C094D1C6267E0C3B68139E00B42824C9D
```

追加本节前文档前缀为 `120286 bytes`，SHA256 为
`8EFEBD8629D8E1ED96F0D2E3FFF65694E1215A0CA7CD831EB133495DA5CD3F76`。为补齐人工/机器可读 artifact，
现在在 repair 结果产生前预注册一次非科学 checkpoint replay：

1. 只接受上述 exact run、九个 exact SHA256、formal failure signature、三个 Helmholtz completion events、
   400-slice completion event 和 `postprocessing_completed` event；任一不符即停止；
2. 不调用 `_solve_helmholtz_case`、`_multislice_reference`、SuperLU、PML/material assembly、multislice propagation
   或任何新 field computation；只从四个 checkpoint 读取已锁定 compact traces/native field；
3. 只重放第 18.13 节固定的 radial-to-Cartesian、external-passband、restriction、annular-mean 后处理，并要求
   重放的全部 mesh/PML/cross metrics 与锁定 `metrics.json` 完全一致；不修改 threshold/outcome；
4. 不修改或覆盖 config、metadata、metrics、progress、failed state 或 checkpoints；不创建缺失的 formal
   `outputs/exp040_r10_stage_b.h5`，而创建独立
   `outputs/exp040_r10_stage_b_repaired.h5`；三张图写入独立 `repaired_figures/`；
5. 仅为 frozen plot contract 提供别名
   `multislice_passband := multislice_fine_1024_passband`，不改变数组值；
6. 新增 `artifact_repair.json`，明确记录 `scientific_recomputation=false`、
   `helmholtz_or_multislice_called=false`、input/output hashes、formal shell failure 保留和 HDF5/PNG read-back；
7. repaired HDF5 保存锁定 metrics、compact native traces、radial profiles 和两张 selected complex passband
   fields；无 `truth/reconstruction`。repair 输出已存在时拒绝覆盖，只允许运行一次。

repair 实现和测试必须在执行前继续 append hash。该 repair 不会把失败 reference 变成通过，也不会产生第二次
formal scientific run。

### 18.17 R10 Stage B artifact repair 实现与执行前锁定

本节在 `artifact_repair.json`、repaired HDF5 和 `repaired_figures/` 均不存在时追加。repair 只新增：

```text
scripts/repair_exp040_r10_stage_b_artifacts.py
tests/test_exp040_r10_stage_b_artifact_repair.py
```

实现首先验证第 18.16 节的 exact run 与九个 SHA256、KeyError signature、三个 Helmholtz completions、400-slice
completion、scientific postprocessing completion 和 progress endpoint。随后把 formal runner 中
`_solve_helmholtz_case`、`_multislice_reference`、`solve_sparse_direct`、
`multislice_propagate_streamed_A` 和 `angular_spectrum_propagate` 全部替换为“调用即报错”的 guard，才允许调用
固定 `_postprocess_once`。重放的 comparisons、projection controls、restriction controls、annular constant 和
anisotropy 必须与锁定 metrics 完全相等。

测试只在 pytest 临时目录写 artifact，不触碰 formal run；它已经验证 full-metrics HDF5 round-trip、两张 selected
complex fields 精确 round-trip 和三张 PNG read-back。执行前状态为：

```text
repair + Stage-B targeted tests: 15 passed in 3.75 s
full repository regression:      259 passed in 40.06 s
repair/Stage-B Ruff:              All checks passed
Python compile check:             passed
artifact_repair.json exists:      false
repaired HDF5 exists:             false
repaired_figures exists:          false
```

追加本节前文档前缀为 `126063 bytes`，SHA256 为
`59571EEEA8744185649AAC2282CD15B83C2B1DAB7D3D0E50B32E7E8F45EE60D4`。repair 实现锁定为：

```text
repair script:
  16465 bytes
  585F870899D88E1B30F3F245963D2C7CFCB313D82F9E495833192F39DAAE861B
repair tests:
  2811 bytes
  93C8CEF8D527134D7D1ABAA9392A9E033C885DE889B821FD021BB070B5E019C2
```

唯一 repair 命令固定为：

```powershell
& 'D:\anaconda3\envs\tgv_ptycho_sim\python.exe' -u `
  scripts/repair_exp040_r10_stage_b_artifacts.py `
  --run-dir runs/exp040_TGV_3d_multislice_r10_stage_b_20260815_181819
```

该命令只允许运行一次 checkpoint-postprocessing replay；不增加 formal scientific execution count，不修改原
failed `run_state.json`，也不改变 `Failed / helmholtz_reference_not_validated`。

### 18.18 R10 Stage B repair 结果、最终结论与下一步

第 18.17 节锁定的 repair 命令执行一次并以 exit code `0` 完成，耗时约 3 秒；没有 Helmholtz solve 或
multislice propagation：

```text
scientific_recomputation: false
helmholtz_or_multislice_called: false
checkpoint_postprocessing_replayed: true
fft_passband_postprocessing_replayed: true
formal shell exit code preserved: 1
formal failed run_state preserved: true
```

replay 的 comparisons、projection controls、restriction controls、annular constant 和 anisotropy 五组 exact
checks 全为 `true`。九个 formal input/checkpoint SHA256 在 repair 前后完全相同；原 failed `run_state.json`
SHA256 仍为 `559B8B51CBF19C1F42221FD917E28446DAEAC37999ABD1C11B37A8DAAFBB0F12`。

repair artifacts 为：

```text
artifact_repair.json:
  3406 bytes
  EF339276824EFFEFD946A1FF4AE0DF3752AF35210F6DE2D2EEFACE435299D90C

outputs/exp040_r10_stage_b_repaired.h5:
  8639104 bytes
  B986E62B0B79F15CD8B2FBD646A7CF844C6A2E1FF7952454BE50A78B85281AF9

repaired_figures/r10_stage_b_reference_controls.png:
  19601 bytes
  D1FEAFD3B29F85E08B859934281B1297164C09C0803E042197412FAE7AF7F569
repaired_figures/r10_stage_b_cross_model.png:
  1026446 bytes
  CF2FDB0463F6C3CB818000C90D2D04ED34AA20978B135E4ED0D8774AE2AE6CA1
repaired_figures/r10_stage_b_radial_profiles.png:
  39316 bytes
  BB7E55F8D10EDB1D6851CBDDD65AC0C1BE1A9CE1C402301C76498145C5A3A2CB
```

repaired HDF5 `/entry` 恰含 `config_yaml/data/instrument/metadata/metrics/sample`；`data` 恰含
`native_helmholtz_traces/radial_profiles/selected_complex_fields`，无 `truth/reconstruction`。全量 metrics 和两张
selected complex fields 精确回读；HDF5 status/interpretation/reference 分别为
`Failed/helmholtz_reference_not_validated/false`，mesh/PML/cross 三个核心值与锁定 JSON 完全相同。三张 PNG
均可读，尺寸分别为 `(460,900)`、`(1256,1580)`、`(928,900)` RGB。

`r10_stage_b_cross_model.png` 左上 panel 的冻结标题使用了 “Validated Helmholtz reference amplitude”；本次
reference 实际**未**验证，该标题只能理解为“候选 Helmholtz reference field”，必须与图一和 HDF5 的
`reference_validated=false` 一起阅读。repair 没有在看结果后重画/改名标题，避免把 presentation 改动冒充原
formal output；后续 plot contract 应改用中性标题。

#### R10 Stage B 的最终分层状态

1. **资源与线性代数 preflight**：通过；fine direct LU 在当前机器可行；
2. **科学 reference validation**：失败；mesh、outer guard、multislice anisotropy 三项未过；
3. **PML/solver/algebra/homogeneous/determinism**：通过，说明这几项不是当前主要 floor；
4. **cross-model diagnostic**：已算出但不可物理解读，不能声称 backward scalar effect 为 111%；
5. **formal wrapper**：科学计算完成后因 plot key bug exit `1`，原失败证据保留；
6. **机器/人工归档**：checkpoint replay repair 成功，repaired HDF5/PNG 完整可审阅。

因此 R10 Stage B 已按预注册完成，但结论是有信息量的 reference failure，而不是双向模型比较成功。图中
fine-nominal 与 enlarged-PML 几乎重合支持“2–3 um PML thickness 不是主导项”；coarse/fine 的巨大径向相位
差和 `dr=0.125 um` 在 glass 中约 2.84 points/wavelength 的事实一致，提示 high-wavenumber pollution；但
`dr=1/12 um` 也只有约 4.26 points/wavelength，仍不能当作已收敛 truth。guard 中强散射说明 `r=24 um` 不是
可安全接 homogeneous fill 的 finite support，且该问题不能只靠加厚 PML 解决——PML 已稳定，必须把**物理
radial core** 外移。multislice 的 9.55% square-grid anisotropy 也必须单独闭合。

#### 建议的下一步（须另行预注册，不在本次改参）

下一步仍应修正数值 reference，而不是升级 vector Maxwell 或用本次 cross difference 解释物理：

1. 先做 radial-core/domain diagnostic：保持共同 comparison `r<=20 um`，预先冻结更大的物理 core/PML-start
   series，并用固定 outer guard 检查散射场是否离开 comparison 区；这与“加厚 PML”不同；
2. 同时避免只靠 uniform brute-force refinement：评估有文献依据的 higher-order FEM、dispersion-controlled
   compact FD 或污染误差可控的 Helmholtz discretization，再用同一 enlarged-domain mesh pair 验证；
3. 对 Cartesian multislice 单独预注册 rotational/annular anisotropy diagnostic，判断 9.55% 来自 q8 circular
   interface 的 square-node imprint、FFT grid，还是 restriction；在其闭合前径向平均不能作为掩盖手段；
4. 只有 enlarged physical core、mesh reference 和 anisotropy 三项均过原 5% gate，才重新运行 cross-model
   comparator；之后若差异仍结构化，才允许预注册 vector/更复杂物理模型。

这些建议不修改本次 R10 的任何结论或 artifact。本节追加前文档前缀为 `128308 bytes`，SHA256 为
`710DF3199F74A1DEE871F51358CBB0680185573E9514419D5518EFBDFF8C4F8D`。本轮没有把实验结果或实现细节写入
`docs/theory_notes/`；theory note 仍只保留第 18.13–18.14 节对应的通用 Helmholtz/PML/pollution 数学与物理。

### 18.19 Post-run plot contract 修复（不改本次 artifact）

为避免后续合法的新实验再次触发第 18.16 节的 presentation-only failure，源码在 repair 完成后作两项非科学
修复：radial plot 改为读取实际 result key `multislice_fine_1024_passband`；候选 Helmholtz panel 的标题从
“Validated”改为中性的“Candidate”。对应 test fixture 也改为使用实际正式键，因此未来 contract mismatch 会在
pytest 中被捕获。

该修改发生在唯一 formal run 和唯一 repair 之后，没有重跑 field/postprocessing，没有覆盖本次 repaired HDF5
或 PNG；第 18.18 节记录的现有 PNG 仍保留原冻结标题与原 hash。追加本节前文档前缀为 `133568 bytes`，
SHA256 为 `7F61BEA3BA813F1DAFF251DFF50AE8B37D588F403418F186742909353E32F523`。修复后的未来源码锁定为：

```text
plot module:
  5591 bytes
  9CF78006AE4F35578884D685DB00CD2FE22CC4D02AD73AD9F304BE132BDBDE17
formal tests:
  6410 bytes
  1A44836FF9B97B0E680395BAB747EF1C06182918473EC8DDC8471E66B7E751B6
Ruff: All checks passed
full repository regression: 259 passed in 39.68 s
```

这只是未来 artifact contract 修复，不改变本次 `Failed / helmholtz_reference_not_validated`、任何 metric、gate、
formal shell failure 或 repair provenance。

---

### 18.20 R11 radial-domain、dispersion 与 Cartesian anisotropy 预注册

本节追加时尚未产生任何 R11 field、metric 或 run。追加前文档前缀为 `134885 bytes`，SHA256 为
`15A77D209760DE7D85AB5A7153936B830C155D6F74871B8C87C62B52763B1B6F`。R11 不改写 R10 的失败结论，
仍使用同一 canonical TGV A、标量正入射、`z=101 um` observation、共同 external passband、`r<=20 um`
comparison 和原 `5%` gate；目的只是分别闭合 R10 混在一起的 physical-domain、Helmholtz pollution 与
Cartesian angular residual。

新增的通用数学与物理依据单独写在
`docs/theory_notes/exp040_r11_domain_dispersion_anisotropy.md`。实验 case、阈值、成本、选择规则和结果只留在
本文档。

#### 18.20.1 文献评估后的固定方法选择

预先评估三条路线：

1. `hp-FEM` 有明确的 `kh/p` 与 `p >= O(log k)` 收敛理论（Melenk--Sauter 2011），但当前环境无
   `scikit-fem/dolfinx/petsc4py`；为轴对称、折射率跳变和 complex-coordinate PML 临时自写高阶弱形式会同时
   改变 basis、quadrature、PML 和 solver，不能在本轮冒充可靠 reference；
2. 13-point/四阶 PML compact FD 有文献依据，但现成公式主要针对 Cartesian 常系数问题；未经推导直接套到
   cylindrical conservative flux 与 discontinuous `n(r,z)` 在物理上和数值上都不充分；
3. R11 因而固定采用 Cocquet--Gander 2024 的五点 shifted-wavenumber asymptotic dispersion correction
   (`ADC5`) 作为唯一正式 pollution-controlled comparator。它保留 R10 的五点守恒 flux、PML 和 direct LU，
   只把 mass/contrast-source 中的局部 `k^2` 同时替换为

```text
k_adc^2(k,h) = 0.5 * [4/h^2 sin^2(kh/2)
                      + 8/h^2 sin^2(kh/(2 sqrt(2)))]
k = k0 * n_cell, h = dr = dz
```

该推广在 discontinuous axisymmetric PML 上不是先验 truth；只有它自己的 enlarged-domain mesh pair 过 gate
才可作为 R11 reference。标准 unshifted five-point pair 同时固定为 report-only comparator，禁止运行后在两者中
选较小者当 reference。

#### 18.20.2 Stage A：固定 radial-core/PML-start series

PML 厚度固定 `2 um`，不再重复 R10 已通过的 `2 -> 3 um` thickness 问题。只移动 radial physical core/PML
start；axial physical core 固定 `[-2,102] um`。ADC5 domain series 固定为：

| case | radial core / PML start | `dr=dz` | `(nr,nz)` | unknowns |
|---|---:|---:|---:|---:|
| `adc_fine_core24` | `24 um` | `1/12 um` | `(312,1296)` | `404352` |
| `adc_fine_core36` | `36 um` | `1/12 um` | `(456,1296)` | `590976` |
| `adc_fine_core48` | `48 um` | `1/12 um` | `(600,1296)` | `777600` |

共同 field comparison 只取 `r<=20 um`。每个 core 的 outer guard 固定为 PML start 前最后 `2 um`，即
`[R_core-2 um,R_core]`，并以该 guard 的 `2*pi*r` weighted scattered RMS 除以 `r<=20 um` inner scattered
RMS。domain gate 在运行前冻结为：

```text
ADC5 core36 -> core48 common-passband radial relative L2 <= 0.05
adc_fine_core48 outer-guard RMS ratio <= 0.05
all solver/algebra/finite controls pass
```

`core24 -> core36` 与全部 raw-field 值只报告，不参与 gate；禁止因其中某个 core 看起来更好而改变 final core。
R11 final physical core 固定为 `48 um`，无论 series 结果如何都不看后缩回较小 domain。

#### 18.20.3 Stage B：同一 enlarged-domain mesh pair

在固定 `48 um` physical core、同一 `2 um` PML 上，冻结两套完全相同的 mesh pair：

| discretization | coarse | fine | denominator |
|---|---|---|---|
| standard five-point（report-only） | `h=0.125 um`, `(400,864)`, `345600` unknowns | `h=1/12 um`, `(600,1296)`, `777600` unknowns | fine |
| ADC5（formal gate） | 同上 | 同上 | fine |

标准 pair 不得替代 ADC5。mesh gate 只使用 ADC5 在 `r<=20 um`、共同 external passband 后的
`2*pi*r` weighted relative L2 `<=0.05`。ADC5 与 standard fine 的差只用于判断 dispersion correction 是否
material，不作为择优规则。所有 case 继续要求 direct-LU relative residual `<=1e-9`；离散代数、PML physical-core
identity、interface volume 和 mapping controls 继续要求 `<=1e-12`。

#### 18.20.4 Stage C：不以径向平均掩盖 Cartesian residual

先从 R10 exact q8 checkpoint 复现 legacy `9.549961270718764%` annular-bin residual；该复现只证明 provenance，
不再把有限 bin 内的径向斜率误叫成纯 anisotropy。新的 formal angular diagnostic 固定在 exact radii 上取
`160` 个 `dr=0.125 um` radial midpoint、每圈 `720` 个等角节点，以固定 cubic interpolation 采样，并报告：

```text
polar angular residual
m=4 and m=8 angular-harmonic energy
45-degree rotation residual
```

插值先用解析径向 manufactured field 验证，最大 angular residual 必须 `<=0.01`。

为拆出 q8 square-node imprint，新增一个不改变连续圆孔几何的 conservative Cartesian cell-average rule：只在
圆周相交 pixel 上对 analytic chord length 做 `64` 阶 Gauss--Legendre 积分。它不是新的物理 effective medium。
在 field 前用固定 top/waist/intermediate diameters 比较 order `64 ->128`；fraction relative L2 与全局圆面积相对
误差均须 `<=1e-5`，否则不得运行该 field。formal multislice cases 固定为：

| case | shape / dx | dz | interface | purpose |
|---|---|---|---|---|
| R10 q8 checkpoint | `1024^2 / 0.0625 um` | `0.25 um` | q8 staggered nodes | legacy anchor |
| `chord512` | `512^2 / 0.125 um` | `0.25 um` | chord-GL64 cell average | FFT/lateral pair |
| `chord1024` | `1024^2 / 0.0625 um` | `0.25 um` | chord-GL64 cell average | corrected reference |

三者 FOV 均为 `64 um`，传播仍是原 centered symmetric split-step，先在各 native grid 投影同一 external
passband，再对 1024 field 作预先固定的 aligned `2x2` conservative cell-average restriction。禁止 output radial
mean 参与下面的 anisotropy gate。固定 attribution 与 gate 为：

```text
q8 imprint diagnostic:
  q8_1024 vs chord1024 native-passband field L2（report/attribution only）

FFT/lateral-grid control:
  chord512 vs restricted chord1024 passband L2 <= 0.05

restriction control:
  chord1024 native 与 restricted 后的 polar angular residual 均 <= 0.05
  restriction-induced absolute increase <= 0.05

final Cartesian anisotropy:
  maximum formal polar angular residual <= 0.05
```

legacy annular residual、m4/m8 与 45-degree residual 全部保存；只有 fixed-radius polar angular residual进入原
5% gate。

#### 18.20.5 Stage D 条件 cross-model 与停止规则

代码必须 fail closed：只有以下三项全为真，才允许调用 cross-model postprocessor：

```text
domain_gate_pass
and adc5_mesh_gate_pass
and cartesian_anisotropy_gate_pass
```

若任一失败，`cross_model_executed=false`，metrics/HDF5 中不得写入一个看似正式的 cross-model 数值；scientific
interpretation 固定为对应的 `domain_not_closed`、`mesh_not_closed`、`anisotropy_not_closed` 或组合 failure。三关
全过时，才比较 `chord1024` 与 `adc_fine_core48`，仍使用 `r<=20 um`、共同 passband、无 phase/scale/shift/tilt
alignment 和原 `5%` materiality gate。只有该合法 cross difference 仍结构化且大于 5%，才允许下一轮另行预注册
vector Maxwell/更复杂物理 comparator；R11 本身不实现 vector model。

#### 18.20.6 成本、preflight 与 checkpoint

按 R10 fine LU 的实测峰值作 `N^1.5` 保守外推，最大 `777600` unknown case 约需 `6.7 GiB` process peak，单次
factor/solve 约 `20 s`；实际 fill 依 domain aspect ratio 变化，因此只作资源估计。新的 chord multislice 预计
`512^2` 约 `1--2 min`、`1024^2` 约 `6--8 min`；formal 总预算约 `8--12 min`，每个 Helmholtz/multislice case
完成后立即写独立 checkpoint。

在高成本 field 前必须先运行独立 preflight，且只做：exact count/内存/磁盘检查、小网格 matrix/source/residual
manufactured controls、ADC dispersion algebra、GL64/128 geometry control、polar manufactured control 和 HDF5
round-trip；不得产生或保存 R11 scientific field。preflight 通过并把 metrics/HDF5 hash 锁入 formal config 后，
才允许唯一一次 formal execution。planned files 为：

```text
configs/experiments/exp040_TGV_3d_multislice_r11_preflight.yaml
configs/experiments/exp040_TGV_3d_multislice_r11.yaml
scripts/run_exp040_r11_preflight.py
scripts/run_exp040_r11.py
src/tgv_ptycho/viz/plot_exp040_r11.py
tests/test_exp040_r11.py
tests/test_exp040_r11_preflight.py
```

公共的 circle-cell integration 与 polar diagnostic 放入现有 `src/` 公共模块并补测试。R10 checkpoints 只读复用，
不覆盖、不修改；R11 每次执行创建新的 timestamped run。本节冻结后才允许实现和 preflight。

### 18.21 R11 实现锁定与 preflight 放行前验证

本节仍未产生任何 R11 scientific field 或 formal run；追加前文档前缀为 `143732 bytes`，SHA256 为
`6240BBA3BEA0FED82E1231B135E740EBC8B8D656C72823463BCB446957495F27`。第 18.20 节冻结的 case、阈值、
模型选择和停止规则均未改变。

实现时专门修正了 conditional comparator 的 fail-closed 顺序：第一次 postprocess 与重复 determinism
postprocess 均固定传入 `hard_controls_prepass=false`，只计算 domain/mesh/anisotropy 指标和代数控制；完成
projection、restriction、finite、algebra 与 determinism 的最终审计后，才执行第三次 postprocess。第三次仅在最终
hard controls 为真时使三个 reference gate 生效，且仍需
`domain_gate_pass && adc5_mesh_gate_pass && cartesian_anisotropy_gate_pass` 才能调用 cross-model。这样不会因
preliminary algebra 看似通过而在最终 determinism 或 projection control 尚未确认时提前生成 cross 数值。

新增测试对 `hard/domain/outer-guard/mesh/legacy/polar/restriction/lateral` 八种独立失败作故障注入，并另行模拟
最终 determinism failure；所有 case 都验证 `_conditional_cross_model` 从未被调用。其余测试覆盖 ADC5 symbol
identity/positivity、GL64 与 GL128 的圆孔 cell-average 和面积、解析径向场的 fixed-radius polar control、两类
checkpoint round-trip、formal/preflight config contract 和三张 figure contract。

实现与放行前文件锁定如下；formal YAML 此时仍含 preflight 占位符，其 hash 只记录当前未放行状态，**不是**
formal execution hash：

```text
preflight config:
  2834 bytes
  2FEAA121E7B6EA4F2B3F3BC0AC3C2843891AC31214FBF2156FD369D072252CF4
formal config before preflight lock:
  6472 bytes
  C0E09B57448AD5830657A0A79EC84E883A7589B8F9EB0E03141F709B740ACD4B
preflight runner:
  26033 bytes
  498808F85CDB40C24E5DB2E62A2D96AF2D74599F96A04F89A5833E468AA3C1B5
formal runner before config-hash lock:
  68307 bytes
  44E88CDC86C67ED283E12DE6B9B28F7803EF9B605DC3FD45A231B573965A7C36
Helmholtz/ADC5/polar module:
  40042 bytes
  549F3EAE3F062A0819D395F54DAEE42B078D62B999A8F4FDF30C3CE72BC9AE82
TGV geometry module:
  11771 bytes
  46A4F172F9356DCA7F102E4FFB03E10A2B34379B4D8D55DF6E08C130614A4A90
objects export:
  673 bytes
  E29F1F4B048F125F0DA66C52AAE3468A6471D041B619499023FEBB9542EBC677
R11 plot module:
  7005 bytes
  B8E5BF96A6FCC2F1F488295E8CC8CF9C038D533F816209C46B9BC8949451089D
formal tests:
  14029 bytes
  69ACD9CA2D1EF3CB19179A7E83B2F2F697778F7A566C86357A39124D6D0C0F64
preflight tests:
  2823 bytes
  15DD9780907721B32C30222725030E9AAFBCB8C5178E7BA7EEE1E30F9D7F9AA3
```

放行前实际验证：相关文件 `compileall` 通过；定向 Ruff 为 `All checks passed`；R11 定向回归为
`19 passed in 3.10 s`；全仓回归为 `278 passed in 42.15 s`。因此允许执行一次第 18.20.6 节定义的
non-scientific preflight。preflight 若任一 hard control 失败，则记录该失败并停止；只有全部通过，才把其 run、
metrics hash 与 HDF5 hash 写入 formal YAML，再锁定唯一 formal execution hash。

### 18.22 R11 preflight artifact failure 与只读结果修复预注册

第 18.21 节锁定的 preflight 已执行一次，run 为
`runs/exp040_TGV_3d_multislice_r11_preflight_20260815_193925`。它在约 `1.02 s` 内完成了全部 control 计算；
外部 `metrics.json` 的所有 11 个 `control_pass` 均为 true，`hard_controls_pass=true`、
`formal_r11_allowed=true`、科学角色为 false。关键值为：

```text
maximum unknowns / estimated peak: 777600 / 6.6679445563 GiB
estimated peak / total physical memory: 0.4806974715
estimated factor+solve time for largest case: 19.9991203389 s
probe relative residual: 3.1143869693e-16
manufactured recovery relative L2: 9.3372895121e-15
ADC5 midpoint identity error: 1.9914115749e-16
GL64 -> GL128 maximum relative L2: 2.9611053270e-11
maximum disk-area relative error: 4.7723082703e-13
polar manufactured maximum angular relative L2: 0.0063548427562
```

但是 wrapper 随后把 `formal_grid_controls` 的 `list[dict]` 直接放入公共 HDF5 writer 的 `instrument`，触发
`TypeError: Object dtype dtype('O') has no native HDF5 equivalent`，shell exit `1`。因此原 `run_state.json` 正确
保留为 `failed_during_execution/formal_r11_allowed=false`，已有 HDF5 只写到
`/entry/instrument/wavelength_m`，不能作为有效 preflight artifact。该失败与 threshold、field、ADC5、geometry、
polar 或资源 gate 无关，但在 artifact contract 修复闭合前 formal R11 仍不放行。

失败证据在任何修复前锁定为：

```text
config.yaml:
  3049 bytes
  FEE2891ACF3A3171DF798041BDEA784E77DF88B97ABF800F9C10F1FB6E609912
metadata.json:
  708 bytes
  90A2EF833D0D48CBD67EBA8FB7D925DF6783A8F28EEA1749D504BC376236B8E7
metrics.json:
  7391 bytes
  D9D2F82D1B808281243A8966614B1E8FF4B9D179B3899C59BCCA105BD3F2BB58
run_progress.json:
  1751 bytes
  1E81B1DA823A299949936011D3FCBD5A19028C9FA36236DFB6842E25ED05AC3B
run_state.json:
  1570 bytes
  77C5EB4CF748655FA3FBD5C36570BD4F437ED242E7D3DAE7C5E0982DA39E872B
partial HDF5:
  4176 bytes
  CDFFD2EE6E2F34D585FEEBF1A15DC1F3D182DEDD03593D664A3DFBCAA04D5EF3
```

本节预先固定以下 artifact-only repair，禁止重新计算 preflight controls，也禁止改写上述六个原文件：

1. 修正 preflight runner 的未来写入契约：把 grid rows 转成以 case id 为键的 nested mapping；不改 config、阈值或
   control 值，本次不重跑该 runner；
2. 新增一次性 repair script，先逐个核对上述 hash、源 preflight config hash，以及 external metrics 的
   `Passed/r11_formal_preflight_passed/all control_pass true`；
3. 只从锁定的 `config.yaml/metadata.json/metrics.json` 生成新的
   `outputs/exp040_r11_preflight_repaired.h5`，原 partial HDF5 和 failed run state 保持不动；
4. repaired HDF5 必须恰含 `/entry/config_yaml,data,instrument,metadata,metrics`，`data` 为空，无
   `truth/reconstruction`，并逐项回读全量 external metrics；另写 `artifact_repair.json` 记录所有输入/输出 hash；
5. 只有 repair script、定向测试、全仓回归和 artifact audit 全部通过，formal YAML 才允许显式锁定 repaired HDF5
   与 repair manifest，而不是把原 failed run 伪装成从未失败。

本节追加前文档前缀为 `146886 bytes`，SHA256 为
`0643F475B3DA1AE1706CD93B3D16378714F4A24280CA90364047E9EDD623921D`。上述 repair 只修复持久化表示，
不会把 preflight control 变成 scientific result，也不会改写第 18.20 节的 formal gate。

### 18.23 R11 preflight HDF5 sequence 编码补充锁定

本节仍在第 18.22 节 repair 执行前。进一步只读审计表明，`metrics.json` 内的 `formal_grid_controls`、
`geometry_controls.cases` 和 `polar_controls.cases` 也都是 `list[dict]`；只把 `instrument` grid rows 改成 mapping
后，公共 writer 会在写 metrics 时再次遇到相同 object dtype。为避免为单个 preflight 修改项目级 HDF5 schema，
本次固定使用 R11-preflight 局部的可逆 adapter：

```text
list/tuple whose children include mappings
-> {
     "__sequence_encoding__": "indexed_mapping_v1",
     "length": N,
     "items": {"000000": child_0, "000001": child_1, ...}
   }
```

纯数值 list、纯字符串 list 和 scalar 保持公共 writer 的原行为；nested mapping 递归应用 adapter。repair validation
必须识别唯一 sentinel、核对连续零填充 index 与 `length`，再完整解码；解码结果必须与锁定的 external
`metrics.json` 和 `metadata.json` 逐项严格相等。该 adapter 只用于 preflight instrument/metrics 的持久化，不改
scientific data、config、threshold、公共 `save_load.py` 或其他实验的 HDF5。原 partial HDF5 仍保留不动。

本节追加前文档前缀为 append-only 的第 18.22 节之后；它只细化已预注册的 artifact repair，不扩大 formal
R11 的执行授权。

### 18.24 R11 repaired-preflight 闭合与唯一 formal execution lock

第 18.22–18.23 节的 artifact-only repair 已执行一次且成功，没有重复任何 preflight control 或 scientific
forward 计算。原 run 的 `run_state.json` 继续保持 `failed_during_execution`，原 partial HDF5 也保持原 hash；
新增文件为：

```text
outputs/exp040_r11_preflight_repaired.h5:
  124664 bytes
  EC62F5DB2EAD550E17E48C802FEA2BDC7E49DA5BECCD2E4FDA067A2ED513226B
artifact_repair.json:
  2340 bytes
  C7376AF4647CE1BA8AA7D30649970DED266370CD4918971CF58C0117D5397E85
```

独立回读确认 repaired HDF5 恰含 `config_yaml/data/instrument/metadata/metrics`，`data` 为空，无
`truth/reconstruction`；indexed sequence 解码后的全量 metrics、metadata 和 instrument 与锁定 external JSON
逐项严格相等。repair 后再次核对原 `config/metadata/metrics/progress/state/partial HDF5`，六个 hash 与第 18.22
节完全相同。

formal config 现显式锁定以下 provenance：R10 metrics/q8/repaired HDF5；preflight source config；preflight
external metrics；原 failed run-state；repaired HDF5 的相对路径和 hash；repair manifest。formal loader 要求
external metrics 的全部 control pass，同时要求原 failure 仍被保留、repair 没有 control/scientific
recomputation、manifest 的全部 validation 为真。最终 formal config 为：

```text
configs/experiments/exp040_TGV_3d_multislice_r11.yaml:
  6964 bytes
  89B531E75749274F6226BE33A424B0ED7398C920C03B298EB665B2117D90772B
```

该 hash 已写死在 formal runner；不存在任何已有 R11 formal run。最终执行前源码与测试锁定为：

```text
formal runner:
  70221 bytes
  D38AC75D253662AE94A81C4CF4BCE19D97BACF8782B6291BA5314077791C2975
preflight runner with future-safe artifact contract:
  29153 bytes
  0120CEFC5C35813C79757BF8511E8C6893FC308F1EC3507202F422B1D94FDE10
preflight repair script:
  11146 bytes
  51CF721A35700D8A30830E6506A8D197A64B6ED73E97F46A324B223A8C978312
Helmholtz/ADC5/polar module:
  40042 bytes
  549F3EAE3F062A0819D395F54DAEE42B078D62B999A8F4FDF30C3CE72BC9AE82
TGV geometry module:
  11771 bytes
  46A4F172F9356DCA7F102E4FFB03E10A2B34379B4D8D55DF6E08C130614A4A90
R11 plot module:
  7005 bytes
  B8E5BF96A6FCC2F1F488295E8CC8CF9C038D533F816209C46B9BC8949451089D
formal tests:
  17480 bytes
  25108923FC27C464A9C5A28EE187476FC44DCC521B58E5031949F8CCA3BE155C
preflight tests:
  3547 bytes
  8D6043AB228E4A92CB6D3D293D06D6DBE920CB9627774CA1D6FC6AFFAEEC0EB6
```

最终定向回归为 `23 passed in 2.92 s`，包括 cross skipped/executed 两条 plot contract 和“全 gate 通过时只调用
一次 comparator”；全仓回归为 `282 passed in 42.59 s`，Ruff 为 `All checks passed`。

formal 最大 system 为 `777600` unknowns；preflight 外推单个最大 direct LU 的 process peak 为
`6.6679445563 GiB`、factor/solve 约 `20.0 s`，整轮仍按第 18.20.6 节估计 `8--12 min`。执行前机器总物理
内存为约 `13.87 GiB`，瞬时 available `3.90 GiB` 仅作 report-only；按预注册不得看后新增 available-memory gate，
但可能因 paging 比外推更慢。磁盘剩余约 `139.18 GiB`。正式命令固定为：

```powershell
& 'D:\anaconda3\envs\tgv_ptycho_sim\python.exe' -u `
  scripts/run_exp040_r11.py `
  --config configs/experiments/exp040_TGV_3d_multislice_r11.yaml
```

只允许执行上述命令一次。每个 Helmholtz 与 multislice case 完成后立即保存 checkpoint；无论 gate 通过或失败，
不得以第二次 formal run 改写结论。

本节追加前文档前缀为 `151747 bytes`，SHA256 为
`B56F7AF0313BEA2A18781099932EB78AB2EF9A30F60C2F0B3646D1ABDF320CD0`。

### 18.25 R11 唯一 formal run：三项 reference closure 均未通过

第 18.24 节锁定的 formal 命令只执行一次，shell exit `0`，run 为
`runs/exp040_TGV_3d_multislice_r11_20260815_195136`。总执行时间 `200.9143444 s`，比资源外推更短；最大实测
process peak 为 `4.8314 GiB`。八个 case 均完成并立即保存 checkpoint，随后 artifact writer 和独立 artifact
validator 均成功。因此本次 `Failed` 是冻结 gate 给出的**科学 reference failure**，不是运行、内存或绘图失败：

```text
status: Failed
interpretation: r11_reference_not_closed__domain_mesh_anisotropy
reference_validated: false
domain_gate_pass: false
adc5_mesh_gate_pass: false
cartesian_anisotropy_gate_pass: false
cross_model_executed: false
cross numeric comparison present: false
```

hard controls 全部通过：最大 direct-LU relative residual 为 `1.2118323033e-13`，最大 algebra error 为
`2.6295566128e-13`，postprocess repeat error 为 `0`；preflight、solver、algebra、determinism、finite controls 均为
true。故本次 floor 不能归因于 solver residual、非有限值、投影不确定性或 postprocess 非确定性。

#### 18.25.1 Physical radial core/domain

固定 `r<=20 um` 的 ADC5 fine domain series 为：

| pair/control | raw radial L2 | common-passband radial L2 | Cartesian report-only L2 |
|---|---:|---:|---:|
| core24 -> core36 | `0.014825%` | `0.279592%` | `40.5860%` |
| core36 -> core48 | `0.0001086%` | `0.0188207%` | `5.13514%` |
| core48 outer guard | - | `7.29624%` | - |

因此 `domain_metric_pass=true`：共同内区的 radial field 对 core36 -> 48 已高度稳定；但固定 outer guard 仍高于
`5%`，使 `outer_guard_pass=false` 和最终 domain gate 失败。该结果区分了“内区径向比较已稳定”与“散射场在
PML start 前尚未充分离开”两件事，不能用前者覆盖后者。core24/36/48 fine guard 分别为
`108.905%/42.293%/7.296%`，随 core 外移显著下降，但 R11 不允许看后把 48 um 宣称为通过。

#### 18.25.2 Enlarged-domain Helmholtz mesh

固定 core48/PML2 的 coarse -> fine 结果为：

| method | common-passband radial L2 | passband Cartesian report-only L2 | gate role |
|---|---:|---:|---|
| ADC5 | `104.6586%` | `76.4628%` | formal，失败 |
| standard five-point | `110.6725%` | `80.6352%` | report-only |

standard-fine 与 ADC5-fine 的 method difference 为 `164.3619%` radial L2。ADC5 相对 standard coarse/fine 只把
mesh mismatch 降低约 6 个百分点，远不足以建立 reference；同时 method difference 很大，说明 correction
material，但两者都未收敛，禁止从二者中选较小者冒充 truth。coarse glass ADC5 effective/physical mass ratio 最低
约 `0.7321`，fine 最低约 `0.8717`；这与当前每波长采样仍过疏、constant-coefficient dispersion correction 在
material jump/cylindrical PML 上无先验 truth 的预注册限制一致。R11 不支持继续把 uniform `h` brute-force 当作
首选：按二维 unknown count 与 direct-LU fill 外推，足以真正细化的下一 pair 会超过当前机器合理内存。

#### 18.25.3 Cartesian interface/FFT/restriction attribution

R10 legacy annular residual 被精确复现为 `9.5499612707%`。新的 fixed-radius polar 结果为：

| field | angular L2 | m4 | m8 | 45-degree residual |
|---|---:|---:|---:|---:|
| q8 native 1024 | `7.59344%` | `2.10589%` | `2.19602%` | `10.6279%` |
| q8 restricted 512 | `7.38418%` | `2.09081%` | `2.18114%` | `10.3367%` |
| chord 512 | `9.15111%` | `2.66466%` | `2.71471%` | `12.8476%` |
| chord native 1024 | `7.60166%` | `2.10945%` | `2.19979%` | `10.6387%` |
| chord restricted 512 | `7.39229%` | `2.09438%` | `2.18486%` | `10.3474%` |

attribution controls 为：

```text
q8 vs chord1024 native-passband field L2: 0.0500181%
chord512 vs restricted chord1024 L2: 2.38618%        (lateral gate pass)
restriction-induced angular increase: 0             (restriction gate pass)
maximum formal angular residual: 9.15111%            (polar gate fail)
manufactured interpolation floor: 0.635484% at 512, 0.0270398% at 1024
```

所以 q8 square-node interface imprint 相对 chord-cell field 只有约 `0.05%`，不是 7--9% angular floor 的主因；
conservative restriction 没有增加该 floor，512 -> 1024 field comparison 也通过原 5% lateral gate。q8 与 chord
在相同 1024 grid 上的 angular residual 几乎相同，进一步排除 interface rule 为主因。剩余结构与 Cartesian
传播表示相关：当前 ASM 把径向 transfer 采样在 square `fftfreq` lattice 上，使用 periodic FFT convolution；R11
只启用 propagating bandlimit，未启用额外 `alias_control`。而 512 与 1024 都保持 `64 um` FOV，故 `dx` 变细只
提高 Nyquist，不改变 `1/FOV` 的 physical frequency-lattice spacing。这支持把剩余项归为 **FFT grid/FOV/
periodic representation family**；本轮尚未把其中的 finite-FOV wrap、transfer sampling 和 square spectral
lattice 三者继续拆开，不能把它们中的某一个单独宣称为唯一原因。

#### 18.25.4 Fail-closed cross-model 结论

三项 gate 全为 false，formal code 正确保存：

```text
conditional_cross_model.executed = false
numeric_comparison_present = false
failed_gates = [domain_gate_pass, adc5_mesh_gate_pass,
                cartesian_anisotropy_gate_pass]
```

因此本轮没有新的 scalar Helmholtz vs multislice 数值，不能解释 backward scattering materiality，也没有达到
预注册 vector/Maxwell comparator 的前置条件。下一轮仍必须闭合数值 reference，不能升级复杂物理来绕过失败。

#### 18.25.5 Artifact audit

run 的 required-file set 精确匹配 config。外部 metrics/metadata 与 HDF5 全量精确回读；HDF5 `/entry` 恰含
`config_yaml/data/instrument/metadata/metrics/sample`，无 `truth/reconstruction`；`data` 恰含
`native_helmholtz_traces/polar_profiles/radial_profiles/selected_complex_fields`。两张 selected complex fields 均为
finite `(512,512) complex128`；六组 native trace、160-point polar profiles 和 13 组 radial profiles 均存在。
三张 PNG 可读，人工检查与 gate 状态一致。主要 artifact 为：

```text
metrics.json:
  34876 bytes
  72B1FAD6ED2AD1D8F00E30CF6F003601FA862CDF7B2646C6EA9EC93CB670621B
outputs/exp040_r11.h5:
  8932896 bytes
  7BA41228BE50821CF23A261549412036D5E275125B040A83B973F8C6257886C0
figures/r11_domain_mesh_controls.png:
  17750 bytes
  DEFC14A70680455497764CA62D4C66064300EC822387A0F035A41D9C786A3407
figures/r11_anisotropy_attribution.png:
  2452719 bytes
  E71C39F33F0DF6FDA4E57F10D7582A2B6A5835680CEFD2233F82B07BBE5DCD8D
figures/r11_conditional_cross_model.png:
  15978 bytes
  43F44DA03031363DA7FC53702055E491AA36FA2CF6090F52B55BCBBBBAE0BEF0
```

checkpoint SHA256 为：

```text
adc_fine_core24: 8545E785FD6D5801090F92AB5F7CDC340CD16A7C580EBF578DB66696F0EC3A1B
adc_fine_core36: 98243ADABC87ECC0F5CEAA29B61D559A4E83BD83D9FF63649B6B491824FEB0BB
adc_fine_core48: 02BFAA2A11EF62E5FA45D717D407F04C8EBB8DBA43B05AB9463318D527629146
adc_coarse_core48: 035B39A7AEDF84E5BF71D3579CE2BE8F1661694FB4435CE282D57552E71EBF6A
standard_coarse_core48: 6137C5243EF238164378DA8FD1261E1D773D360676365949A5BEFEC4A7EFADD7
standard_fine_core48: 52AE9B7937B2E1728CED413039AE67A9BDBA53E2324783E5BB6E10F123061B46
chord512: 1BAC1C293D94E087439765C2B1C3C870600DC5C4257508A0D041105A239F6E2F
chord1024: 83D1B9D69D608708AB8B562A8816896071B630692A55C7C80E52F3AC3024E445
```

### 18.26 R11 后建议：仍留在数值 reference 主线

R11 后最合理的下一步不是直接把 `dx/dr` 再暴力减半，也不是 vector model，而是另行预注册三条相互独立的
closure；以下仅为建议，尚未修改模型或执行新 field：

1. **domain**：在同一 `r<=20 um`、PML2 um 和 outer-guard 定义下，把 physical core/PML-start series 继续
   外移，例如预先固定 `60/72/96 um`，同时保留 full complex Cartesian report；目的只是让 7.296% guard
   是否跨过原 5% 成为可证伪问题，不能用已经很小的 radial-average pair 覆盖 guard；
2. **Helmholtz discretization**：ADC5 已证明不足。优先建立有独立 benchmark 的 higher-order axisymmetric
   weak-form FEM（例如固定 `p=2/3` pair）或经过 cylindrical/PML/material-jump 推导与 manufactured test 的
   dispersion-optimized compact stencil；先在 homogeneous/interface/已知 scattering benchmark 验证，再用于同一
   enlarged domain。不得直接把 Cartesian constant-coefficient 高阶公式贴到 cylindrical discontinuous operator；
3. **Cartesian propagation**：固定 chord-cell interface 和 restriction，做 enlarged-FOV/zero-padding series，
   使 `1/FOV` frequency sampling 与 `dx` refinement 分开；再以 alias-controlled ASM 和 axisymmetric Hankel/
   Fourier-Bessel split-step 作 report comparator。这样才能把 periodic finite-FOV、transfer sampling 与 square FFT
   lattice 继续拆开，而不是再用 radial mean 掩盖 7--9% angular residual。

只有这三项各自仍按原 5% gate 闭合后，才能恢复 conditional scalar cross-model；若届时差异仍结构化且 material，
才有理由预注册 vector/更复杂物理。本轮没有新增纯数学或纯物理理论，因此 `docs/theory_notes/` 不再追加；ADC5、
pollution、PML 和 polar diagnostic 的通用依据仍在既有 R11 theory note，实际结果与建模判断全部留在本文档。

第 18.25 节追加前文档前缀为 `155424 bytes`，SHA256 为
`B6C52E50392463287DC0AD72AD97DCD70CCBEBF59875D868D209071F5DBA562D`。

### 18.27 R12 预注册：core60、Q2/Q3 weak-form FEM 与 enlarged-FOV/QDHT 三路闭合

本节写在任何 R12 TGV field 产生之前。R12 不改变研究问题、canonical 几何、标量物理或原 `5%` gate；它只把
第 18.26 节的三条数值 reference closure 各自变成可证伪实验。固定复用 R11 唯一 formal run
`runs/exp040_TGV_3d_multislice_r11_20260815_195136`，不得重算其中 core48 或 chord512。R12 的三条 closure
保持独立，任何一条失败都不得由另一条的径向平均或较小数值覆盖。

#### 18.27.1 Domain case 与资源停止规则

domain 分支继续使用 R11 的 axisymmetric scattered-field、ADC5、`r<=20 um` comparison、PML `2 um`、outer
guard 宽度 `2 um` 和相同 observation plane。固定复用 `adc_fine_core48`，只新增：

```text
case: adc_fine_core60
physical core/PML start: 60 um
dr = dz = 1/12 um
grid: nr=744, nz=1296, unknowns=964224
formal pair: core48 -> core60
formal guard: 58--60 um
```

domain gate 仍要求共同 physical passband 后的 `core48 -> core60` 径向 relative L2 `<=5%`，且 core60 scattered
outer-guard RMS / inner scattered RMS `<=5%`。full complex Cartesian comparison 继续保存为 report-only，不能代替
径向 gate，也不能反过来被径向结果掩盖。

第 18.26 节举例的 core72/core96 在本轮看结果前即排除：相同 fine spacing 下分别约有 `1,150,848` 和
`1,524,096` unknowns；由 R11 core24/36/48 的 SuperLU fill 外推，core72 已接近本机安全内存边界，core96
会达到或超过约 `13.87 GiB` 的总物理内存。R12 不允许因 core60 结果不理想而临时追加它们；只有以后另行写明
iterative/domain-decomposition solver、资源预算和停止条件，才可执行更大 core。

#### 18.27.2 Higher-order axisymmetric weak-form FEM

候选 reference 固定为连续 tensor-product Lagrange FEM，而不是把 Cartesian 高阶 stencil 贴到 cylindrical/PML
operator。弱式保留 cylindrical Jacobian、complex-coordinate PML、flat-interface analytic background 和
contrast source；element 使用 Gauss--Lobatto nodes，材料与系数固定以 8 点 Gauss--Legendre tensor quadrature
积分。TGV air/glass interface 在 quadrature nodes 上按解析半径判断，不使用 ADC5 effective mass，也不做看后
method selection。

正式 pair 在同一 enlarged core60/PML2 domain 和同一 `0.5 um` element mesh 上固定为：

| case | degree | active unknowns | role |
|---|---:|---:|---|
| `fem_p2_core60` | 2 | `106888` | coarse p-reference |
| `fem_p3_core60` | 3 | `240684` | fixed final Helmholtz candidate |

在正式 TGV FEM 前，非科学 preflight 必须通过两个独立 manufactured weak-form benchmark：一个 homogeneous
mass，一个含未对齐 radial jump 的 discontinuous mass；二者使用同一解析 regular-at-axis Bessel/sine solution，
而 RHS 独立由连续方程给出。阈值在此固定为：最大 linear-solve relative residual `<=1e-10`；homogeneous p3
weighted relative L2 `<=2e-4` 且 `p3/p2<=0.35`；interface p3 weighted relative L2 `<=5e-3` 且
`p3/p2<=0.75`。任一项失败则 formal FEM cases 必须 fail-closed 跳过，不能放宽阈值。

若 benchmark 通过，formal mesh gate 要求 `fem_p2_core60 -> fem_p3_core60` 在原共同 passband、`r<=20 um`
上的 weighted radial relative L2 `<=5%`；同时要求 p3 outer guard `<=5%`。p3 与 `adc_fine_core60` 的差异只作
method-attribution report，不得从两者中看后选择较小者作为 truth。

上述选择由 Melenk--Sauter 的 Helmholtz hp-FEM 结论约束：高频 quasi-optimality 需要同时控制 `kh/p` 并随
wavenumber 提高 polynomial degree，而不是只缩小 uniform h。本轮 p2/p3 是候选 reference 的可证伪起点，
不是预先宣称已达到 asymptotic regime。纯数学说明另记于
`docs/theory_notes/exp040_r12_axisymmetric_fem_qdht.md`。

#### 18.27.3 Cartesian FOV、transfer alias 与 QDHT attribution

Cartesian 分支固定 chord-cell order64、`dz=0.25 um`、centered symmetric split-step、conservative centered crop、
`dx=0.125 um` 和原 physical passband。固定 case 顺序为：

```text
chord_fov64_standard   : reuse R11 chord512, 512^2, FOV 64 um
chord_fov96_standard   : 768^2,  FOV 96 um, propagating bandlimit
chord_fov128_standard  : 1024^2, FOV 128 um, propagating bandlimit
chord_fov128_alias     : 1024^2, FOV 128 um, Matsushima common-ellipse
                         alias control for every internal and post-exit step
qdht_r64_n512          : order-0 QDHT, radial support 64 um, N=512,
                         normalized contrast split-step, report-only
```

所有 Cartesian 比较先取共同中心 `512^2` physical window，再应用同一 passband；因为 `dx` 不变，本轮没有
restriction，也禁止插入 lateral refinement。periodic/FOV closure 要求 `FOV96 -> FOV128 standard` 的中心场
passband relative L2 `<=5%`；最终 anisotropy gate 要求预先固定的 `chord_fov128_alias` polar angular relative L2
`<=5%`。`FOV64 -> 96`、standard -> alias、m4/m8、45-degree residual 均保存用于 attribution，但不允许看后改选
final field。

QDHT 使用 Guizar-Sicairos--Gutiérrez-Vega 的 order-0 quasi-discrete Hankel nodes 与离散 Parseval-compatible
transform；它传播在 radial boundary 消失的 normalized contrast，均匀 background 单独解析保留。preflight
要求 transform involution 和 scaled Parseval relative error 均 `<=1e-10`，zero-distance contrast roundtrip
`<=1e-10`。QDHT 与 Cartesian 的 radial field comparison 只作 report comparator：它可帮助区分 square FFT
lattice，但不得用其零 angular residual 代替 Cartesian full polar gate。

#### 18.27.4 Conditional execution、成本与结论边界

hard controls 继续要求 finite、deterministic、solver residual 和 projection algebra 全部通过。只有
`domain_gate_pass`、`fem_mesh_gate_pass`、`cartesian_reference_gate_pass` 三者均为 true，才执行唯一一次
`fem_p3_core60` 对 `chord_fov128_alias` scalar cross-model comparator；否则 numeric cross-model 必须不存在。
即便 cross-model 差异超过 `5%`，R12 也只允许建议下一轮预注册 vector/更复杂物理，不在本轮启用。

本机约 `13.87 GiB` RAM；结合 R11 peak/fill 外推，core60 direct LU 预计约 `6--7 GiB`，FEM p3 的 sparse
assembly/LU 与 Cartesian/QDHT 均必须逐 case 释放，禁止并行和同时保留矩阵。formal 总成本预估 `8--15 min`，
必须先完成 compile、Ruff、定向/全仓测试和 non-scientific preflight，再锁定 config SHA256；随后只允许执行
一次 formal，并在每个昂贵 case 后立即写 checkpoint。run 失败也作为本轮结果保存，不得第二次运行改写结论。

本节追加前文档为 `165026 bytes`，SHA256 为
`ED462541A7EC96C0EC9F84EAFCF5CF92131E8856A598A855EFB1F6BAA264C66F`。

### 18.28 R12 实现与 non-scientific preflight execution lock

第 18.27 节的预注册已经实现，但尚未产生 formal TGV field。新增的公共实现为：order-0 QDHT 及其 physical
scaling；只传播 normalized contrast 的 radial split-step；带 cylindrical Jacobian、complex PML、Gauss--Lobatto
nodes 和 tensor Gauss quadrature 的连续 Q2/Q3 weak-form FEM；现有 Cartesian streamed multislice 只新增默认
关闭的 `alias_control` 参数，历史调用路径不变。实验 runner 只负责编排、checkpoint、gate 和 artifact。

preflight 的资源 gate 在执行前进一步锁定为：core60 direct-LU peak 的 R11 fill 外推（含固定 `1.15` safety
factor）必须 `<=10.0 GiB`，free disk 必须 `>=20.0 GiB`，最大 formal unknowns 必须保持 `964224`，且
`allow_core72_or_core96=false`。瞬时 available memory 只记录，不作可看后变化的 gate。preflight 不计算 canonical
TGV fields；它只运行小型 manufactured systems、QDHT algebra、formal grid count 和资源外推。

实现后验证为：相关定向测试 `29 passed`；全仓回归 `288 passed in 67.54 s`；本轮修改文件的 Ruff 为
`All checks passed`。preflight source config 已锁定为：

```text
configs/experiments/exp040_TGV_3d_multislice_r12_preflight.yaml
  2338 bytes
  86C560857933727ED9A5574368E12DBE7E67CDA57E1A05AD92EE240E99E33CC3
```

主要实现 hash 为：

```text
preflight runner:
  17110 bytes
  20E4E669D7FADEDBD50C3597BDFA4EDFAB34FB51422169F703EA5FDBA77310C4
formal runner before provenance lock:
  55051 bytes
  715870B0E87B83AE7EF312FC333AB4034494213D39E9B8C052F5BA4C5CBAF737
axisymmetric FEM:
  22889 bytes
  99FFEE9100FBCD3C0C737DCCE58B569DACC9FF2F981DCA2C882F9B2A73F5B703
radial multislice:
  4757 bytes
  8CF5A4FBF52CC97D6E4B012B4B65106522E1E0D36AE841F1A6852C69FD1C8229
QDHT:
  7967 bytes
  CC0D80F3676072DDC3A39C42136C8901F6C4BE637EA23C55A3A1C53CC69FE9D0
R12 plots:
  4451 bytes
  70847BCC555F0A1BE5DC448838B023EF82C5ADEA0B4744196F5266187A060856
```

preflight 只允许执行以下命令一次：

```powershell
& 'D:\anaconda3\envs\tgv_ptycho_sim\python.exe' -u `
  scripts/run_exp040_r12_preflight.py `
  --config configs/experiments/exp040_TGV_3d_multislice_r12_preflight.yaml
```

只有其 external metrics、HDF5 和 artifact audit 均通过，才创建并锁定 formal YAML；preflight 失败不得通过
放宽第 18.27--18.28 节的阈值来修复。本节追加前文档为 `171921 bytes`，SHA256 为
`7DF7BBBFE4CDBCE0651E5451894CC72D4C68F0AEB8CB94F57FE20F2FF4D6D1FA`。

### 18.29 R12 preflight 结果与唯一 formal execution lock

第 18.28 节锁定的 preflight 命令只执行一次，shell exit `0`，run 为
`runs/exp040_TGV_3d_multislice_r12_preflight_20260817_140707`；它是 non-scientific control run，不是 R12
TGV 结果。所有 gate 为 true，`formal_r12_allowed=true`：

```text
maximum manufactured solver relative residual: 5.6210738683e-15
homogeneous p3/p2 weighted-error ratio:          0.02209360119
interface p3/p2 weighted-error ratio:            0.02226085434
QDHT transform involution probe L2:              1.1057986441e-14
QDHT physical roundtrip L2:                      7.9674135063e-15
QDHT scaled Parseval error:                      2.1129580579e-15
predicted core60 peak with 1.15 safety:           7.4402847569 GiB
total physical memory:                           13.8713951111 GiB
available memory at preflight, report-only:      2.2035789490 GiB
free disk:                                       139.1561317444 GiB
```

formal finite-volume/FEM grid counts 与预注册完全相同；资源模型通过 `7.44 GiB <=10 GiB` 和
`139.16 GiB >=20 GiB`。preflight HDF5 恰含 `config_yaml/data/instrument/metadata/metrics`，`data` 为空，未伪造
truth/reconstruction。锁定 artifacts 为：

```text
preflight config.yaml:
  2454 bytes
  7425CD0B2D31F83D04870ADFFA37B47E1322C5EEA0BE8C4619A3A91AE24A404F
preflight metrics.json:
  10159 bytes
  D549820DF8DCDB05C81BE693EB7DFD112FD929BF5433975D99256FB67DDC7D2A
preflight HDF5:
  134752 bytes
  D8A8DD21EE4770AA04DB1B713B89DF9E751E83E1C4E0C5EC33F826B97C0081CF
```

formal YAML 已显式锁定上述 provenance、R11 core48/chord512 hashes、三条 gate、case 顺序和 required artifact
set；最终文件为：

```text
configs/experiments/exp040_TGV_3d_multislice_r12.yaml:
  6087 bytes
  6BE3E84867E65F92AC3D74AE7D3CC02C9C062150CB2C66E0A6D74AD127B54523
formal runner:
  55116 bytes
  3CE21AC9B888181E35824D6353E4D9A45D123E88B8E119E6366815CDC2F6E665
formal contract tests:
  4982 bytes
  3FF4527B81D6AEA94CEA0C0B375697BFE5BF63CFD1B1DA26E970F17F8DD19099
```

formal contract 的 synthetic postprocess、HDF5 和三图写入均已定向验证；最终本轮 Ruff 为 `All checks
passed`，全仓回归为 `291 passed in 58.88 s`。正式命令固定为：

```powershell
& 'D:\anaconda3\envs\tgv_ptycho_sim\python.exe' -u `
  scripts/run_exp040_r12.py `
  --config configs/experiments/exp040_TGV_3d_multislice_r12.yaml
```

该命令只允许执行一次。顺序固定为 core60 ADC5、FEM p2、FEM p3、FOV96 standard、FOV128 standard、
FOV128 alias、QDHT；每个 case 后立即保存 checkpoint。任何科学 gate 失败或运行时资源失败都保留为本轮结果，
不得重复 formal 或临时追加 core72/core96。本节追加前文档为 `174473 bytes`，SHA256 为
`DFF7C46CE98528B810524D220575FC50ED61B60EECD1E972EF01A6A64A9BAB01`。

### 18.30 R12 唯一 formal run：Cartesian closure 通过，domain guard 与 FEM p-pair 未闭合

第 18.29 节锁定的 formal 命令只执行一次，shell exit `0`，run 为
`runs/exp040_TGV_3d_multislice_r12_20260817_141417`。scientific execution 为 `419.448 s`，外层命令墙钟约
`422.9 s`；七个新 case 均按固定顺序完成并立即保存 checkpoint，随后 HDF5、三图和独立 artifact validator
全部成功。因此本次 `Failed` 是预注册 gate 给出的科学 reference failure，不是运行、内存或 artifact failure：

```text
status: Failed
interpretation: r12_reference_not_closed__domain_fem_mesh
reference_validated: false
domain_gate_pass: false
fem_mesh_gate_pass: false
cartesian_reference_gate_pass: true
cross_model_executed: false
numeric cross-model comparison present: false
```

hard controls 全部通过：最大 solver relative residual 为 `1.0600751964e-13`，projection repeat/idempotence 最大
relative error 为 `3.2278702259e-16`，所有 field/profile 均 finite。最大实测 process peak 为 FEM p3 时的
`5.0176 GiB`，低于 preflight 的 `7.44 GiB` 保守外推；资源不是失败原因。

#### 18.30.1 Domain：inner field 已闭合，但原 local-amplitude guard 失败

固定共同 passband、`r<=20 um` 的 core48 -> core60 weighted radial L2 为
`1.0898662847e-6`，即 `0.00010899%`，远低于 `5%`；两者 inner scattered RMS 也分别为
`0.8432583201/0.8432585230`。这说明 PML start 从 48 外移到 60 um 后，真正用于 comparison 的内区 complex
field 已高度不变。

但是预注册的最后 `2 um` local guard 为 `7.399105% >5%`，所以 domain gate 必须保持 false。为理解而做的
post-run report-only sliding-window 回读为：

| radial window (um) | scattered RMS / inner RMS |
|---|---:|
| 46--48 | `7.29624%` |
| 48--50 | `6.15781%` |
| 50--52 | `5.39720%` |
| 52--54 | `4.92762%` |
| 54--56 | `4.87039%` |
| 56--58 | `5.58853%` |
| 58--60 | `7.39911%` |

这张表不能看后选择 52--56 um 来改写 gate；它只说明“PML start 前固定窄窗的场幅必须单调变小”不是稳健的
open-boundary 判据。outgoing scattered field 本来可在非零幅度处进入 PML，窄窗还会受 radial lobe/interference
位置影响。R12 的 domain 结论仍是失败；但后续应把“截断反射有多小”和“PML 起点处场幅是否小”拆开，而不是
继续无条件外移 core。对应的通用理论补充在
`docs/theory_notes/exp040_r12_axisymmetric_fem_qdht.md`。

#### 18.30.2 FEM：小型 manufactured benchmark 通过，但 physical-k regime 严重欠解析

同一 core60、同一 `0.5 um` element mesh 上：

| comparison/control | result |
|---|---:|
| Q2 -> Q3 common-passband radial L2 | `137.9919%` |
| ADC5 -> Q3 method difference, report-only | `97.8223%` |
| Q3 outer guard | `1.08024%` |
| Q2/Q3 solver residual | `3.91e-14 / 4.49e-14` |
| Q2/Q3 active unknowns | `106888 / 240684` |

所以 FEM boundary/solver controls 良好，但 p-pair 完全没有收敛。即使允许一个 post-run、report-only 的最佳全局
complex scale，Q2 对 Q3 的 residual 仍为 `78.95%`，不是单一 global phase/scale 能解释的差别。

原因定位比“高阶 FEM 不适用”更具体：preflight manufactured problem 使用 `kh=1.25`，而 formal glass 中
`k h = 8.85787`，故 `kh/p` 对 Q2/Q3 分别为 `4.42894/2.95262`。preflight 只验证了 weak-form implementation
在容易 regime 的代数与收敛阶，没有证明 formal physical-wavenumber problem 已进入 hp asymptotic regime。
这正是 Melenk--Sauter pollution 条件要求联合控制 `kh/p` 的情形；因此不能把 Q3、ADC5 或二者中较接近某个
预期的一个冒充 truth，也不应直接用 uniform direct-LU brute force 补救。

#### 18.30.3 Cartesian：主要 floor 是 finite FOV/frequency lattice，不是 transfer alias mask

固定 `dx=0.125 um` 后的结果为：

| field/comparison | result |
|---|---:|
| FOV64 polar angular L2 | `9.15111%` |
| FOV96 polar angular L2 | `5.08716%` |
| FOV128 standard polar angular L2 | `1.65649%` |
| FOV128 alias-controlled polar angular L2 | `1.65182%` |
| FOV64 -> 96 center-passband field L2, report-only | `9.07023%` |
| FOV96 -> 128 center-passband field L2 | `4.44271%` |
| FOV128 standard -> alias field L2 | `0.070475%` |
| QDHT -> FOV128 alias radial L2, report-only | `6.82429%` |

因此两个 formal Cartesian 条件同时通过：FOV96 -> 128 小于 `5%`，最终 alias field 的 full polar angular L2
也小于 `5%`。angular floor 从 FOV64 的 9.15% 持续降到 FOV128 的 1.65%，而同一 FOV 上启用 Matsushima
mask 只改变约 0.07%。这把 R11 的 7--9% 主要归因于 finite-FOV / `1/FOV` frequency-lattice / periodic
representation family，而不是 transfer-function sampling alias；没有通过 radial averaging 掩盖 angular result。

QDHT radial comparator 在 `6.82%`，视觉上主要 oscillatory structure 一致，但略高于 5%；它按预注册只是
report-only，不能推翻已通过的 Cartesian gate，也不能宣称 radial Hankel support/interface sampling 已闭合。

#### 18.30.4 Fail-closed cross-model 与 artifact audit

由于 domain 与 FEM mesh 两项失败，formal code 正确保存：

```text
conditional_cross_model.executed = false
numeric_comparison_present = false
failed_gates = [domain_gate_pass, fem_mesh_gate_pass]
```

所以 R12 没有产生新的 scalar FEM-vs-multislice 数值，也不允许 vector/Maxwell comparator。HDF5 `/entry` 恰含
`config_yaml/data/instrument/metadata/metrics/sample`，无 `truth/reconstruction`；`data` 恰含
`polar_means/qdht_native/radial_profiles/selected_complex_fields`。两张 selected complex fields 均为 finite
`(512,512) complex128`，四组 polar means、五组 radial datasets 和 QDHT native trace 均存在。三张 PNG 已
实际打开检查，与 gate 状态一致。主要 artifacts 为：

```text
metrics.json:
  19073 bytes
  50BFDE52D50092D2EBEB97B0BC7DAA7DE442FE0E21EA165461CB084CB973604E
outputs/exp040_r12.h5:
  8638056 bytes
  4FA90ABF2925129DC6DC184BED4540DA162C3213B37FF0755C8FAC6C0593AC98
figures/r12_domain_fem_controls.png:
  52058 bytes
  35AB1E6A015919B9184AAE8B3B03F45949A6FBC7FE54659B26E06F74EB1869C3
figures/r12_cartesian_fov_alias_qdht.png:
  107992 bytes
  5C9644B020AB9F43FEBB222E82E6AF49FEA3F9A56060A464872A368D2F96ED8F
figures/r12_conditional_cross_model.png:
  15040 bytes
  288112F450EB0E84DD30D524F6E157B4D9D8F1DCF685CFE0EDC743AF9F75F896
```

checkpoint SHA256 为：

```text
adc_fine_core60:       42FBA683BF40EBD9087248D57DB50C04F79DF9E71FB5D8843B1A7508DA6A5E30
fem_p2_core60:         54E47CA32D5655DA72C1534185BBD1E0A5EE2970583C7EFC4C1ECAA3A2B55B5F
fem_p3_core60:         9E3BB47273FB1F3C0D7B07ED93E85D833E33B4E0F3D47B269BF1F23ED7EED2D4
chord_fov96_standard:  C33746CAB8F43C19AEDA22284CE5596AF96046B502330FAC07E38934BA2549E5
chord_fov128_standard: 6529F9A243A54C676CA81AA6AFC608B0B455B04AF569C90AEA12E6B948C1EC3B
chord_fov128_alias:    A35D9B8FC58AC2298444FCC67D9D8155001F4FBB24A2191396213654DF0AB528
qdht_r64_n512:         B820A96815A9D3DB3103D32A5B1D011F5DC4B57EECC43088C965B3F07A827FB2
```

### 18.31 R12 后建议：先修正可证伪 benchmark，不继续盲目加 core 或减 h

1. **Domain 优先改为 reflection diagnostic**：另行预注册 analytic outgoing cylindrical/PML benchmark，并在
   多个 radius 上评估 radial flux、Sommerfeld residual 或 incoming/outgoing component；继续保留
   core48 -> 60 内区 complex-field convergence。新判据必须在计算前写明，且只能形成新结论，不能回改 R12
   的 failed guard。若 reflection 很小而内区 pair 仍为 `~1e-6`，则说明 finite-domain reference 已闭合，非零
   guard 只是错误代理量；若 incoming component material，才需要更强/更厚 PML 或更远 core。
2. **FEM 先做 physical-`kh` pollution benchmark**：在小 domain 上固定 formal 的 `kh=8.85787`，预注册
   p2/p3/p4/... dispersion/interface/PML series，先找出满足误差 gate 的 `h,p` 组合，再碰 canonical TGV。
   预计 direct LU 无法承担真正的 p/h pair；应同时评估 shifted-Laplacian/域分解/稀疏迭代预条件，或有误差估计
   的 hp-adaptive FEM。没有 physical-k benchmark 与 solver resource closure 前，不建议直接运行 p4 TGV，
   更不建议把 h 再统一减半。
3. **冻结 Cartesian reference**：当前 `FOV128 + alias-control + chord-cell + dx0.125 um` 已按原 gate 闭合，
   后续 cross-model 可复用该 checkpoint，不再把 lateral dx refinement 当主线。QDHT 的 6.82% 可在主要 FEM
   卡点解决后再用 radial-support/N/interface 三项独立 series 闭合。
4. **cross-model/vector 继续 fail-closed**：只有新 domain-reflection gate 与 physical-k FEM reference 均通过，
   才恢复 scalar comparator；仅当届时差异仍结构化且 material，才预注册 vector/Maxwell 模型。

这一路线没有更换研究模型，也没有用复杂物理解释数值 reference failure；它把当前两个真正卡点收缩为
“PML reflection 的正确观测量”和“physical-k Helmholtz pollution/solver”。本节追加前文档为
`177351 bytes`，SHA256 为
`9D0C7D6B16F27AD3C194AE57993FD9877457041D5A5D9DAB5504488BD20DD1A3`。

### 18.32 R13 预注册：outgoing/PML reflection 与 physical-`kh` pollution benchmark

本轮只执行第 18.31 节要求的可证伪 benchmark，不重跑 R12，不生成新的 Cartesian field，也不运行 full-size
TGV p4/p6 或 scalar cross-model。R12 的 `FOV128 + alias-control + chord-cell + dx=0.125 um` checkpoint 固定
复用并做 SHA256 provenance 检查；它在 R13 中不参与任何新计算。R13 的目的不是把 R12 的 failed gate 改为
passed，而是回答两个新问题：原 radial PML 的真实 incoming reflection 是否足够小，以及 formal
`kh=8.85787` 下何种 `h,p` 才能达到 analytic reference 误差预算。

#### 18.32.1 Domain-reflection diagnostic

预先固定一个一维 annular axisymmetric weak-form FEM。内边界为 `r=20 um`，施加单位归一化的 outgoing
`H0^(1)(kr)` Dirichlet 数据；外侧分别采用 `PML-start=48/60 um`、相同 `2 um` cubic complex-coordinate PML、
`target_one_way_amplitude=1e-8` 与最外侧 homogeneous Dirichlet。两 case 均固定 `h=1/12 um`、p4、12 点
Gauss quadrature；这不是加厚 PML series。共同测量半径固定为 `24/32/40/46 um`，dense comparison 固定为
`21--46 um` 的 1001 点。

每个测量半径由 numerical field 与 derivative 在 `H0^(1)`/`H0^(2)` basis 中做唯一 2x2 decomposition，保存
incoming/outgoing coefficient ratio；同时保存 exact outgoing Hankel impedance residual、
`r Im(conj(u) du/dr)` radial flux、对 analytic outgoing field 的 dense relative L2，以及 core48/core60 的共同
dense field L2。判据在执行前固定为：incoming ratio、impedance residual 和 analytic-field L2 均 `<=0.1%`，
flux relative range `<=0.5%`，core pair `<=0.1%`，solver residual `<=1e-10`。这些阈值比原 cross-model
`5%` gate 至少小一个数量级；即使通过，也只能说明 radial truncation/PML reflection 已闭合，不能证明所有 axial
boundary modes 已闭合。

#### 18.32.2 Physical-`kh` hp-FEM pollution diagnostic

第二条 benchmark 将 formal `0.5 um` element nondimensionalize 为 `h=1`，固定 `k h=8.85787`；计算域固定为
`r,z in [0,4]`，解析场固定为第 7 个 `J0` radial zero 与第 8 个 axial sine mode 的乘积，禁止 global
phase/scale alignment。每个 case 同时运行 homogeneous mass 与一个未对齐的 circular glass--air mass jump
（`r=1.73`、`n_air^2/n_glass^2=4/9`）；manufactured source 使用同一解析场，因此能够分别量化 dispersion 与
non-aligned interface quadrature，而不把某个 numerical case 当 truth。

case 顺序在运行前固定为：`h=1` 的 p2/p3/p4/p6/p8，`h=0.5` 的 p2/p3/p4/p6，以及 `h=0.25` 的
p2/p3/p4。两 family 的 weighted radial relative L2 都必须 `<=1%` 才成为 candidate；`1%` 用来给最终
`5%` scalar cross-model 留出 reference error budget。若多个 case 合格，按投影到 core60 full TGV 后的 active
unknown 数、degree、element-size ratio 的固定字典序选择最小 candidate，不允许看后挑选更符合预期的曲线。

R13 只报告 full-TGV unknown count 与 direct-LU `1,000,000 unknown / 10 GiB` 上限，不执行该 full solve。
即使找到准确 candidate，仍需单独通过 solver resource preflight；若 candidate 超过 direct-LU 上限，下一轮应
预注册 shifted-Laplacian、域分解或其他有收敛记录的 iterative/hp strategy，不能直接 uniform brute-force。

#### 18.32.3 Execution lock、产物与结论边界

正式配置固定为 `configs/experiments/exp040_TGV_3d_multislice_r13.yaml`，preflight 固定为同目录的
`exp040_TGV_3d_multislice_r13_preflight.yaml`。必须先完成定向测试、全仓测试、Ruff 和 non-scientific
preflight；preflight 不得运行上述 formal case。其 artifacts 全部通过且 provenance/hash 锁定后，formal 只允许
运行一次。正式 HDF5 保存 numerical `data`、analytic `truth`、instrument/sample/metrics；无 reconstruction。

R13 status 只由 `domain_reflection_gate_pass` 与 `physical_k_candidate_found` 决定；full-TGV reference、scalar
cross-model 与 vector/Maxwell 均保持 disabled。若任一 gate 失败，保存失败作为结果，不放宽阈值、不追加 case、
不重跑；若两者通过，也只能进入下一轮 solver/full-reference 预注册，不能直接宣称 exp040 完成。

本节追加前文档为 `186637 bytes`，SHA256 为
`C7D10EAB0C62532771E58F9340D8BE29BBDB92C4DA80D349AE2BA4C254988321`。

### 18.33 R13 实现、preflight 结果与唯一 formal execution lock

第 18.32 节的 scientific contract 已实现。公共模块新增 annular radial Qp weak-form FEM、normalized
Hankel H1/H2 field-derivative decomposition、exact outgoing impedance/flux controls，以及以 analytic Bessel-sine
mode 为 truth 的 physical-`kh` homogeneous/interface benchmark；三张固定图分别显示 reflection、hp pollution 与
projected full-TGV unknown count。纯数学与物理推导另记于
`docs/theory_notes/exp040_r13_pml_reflection_and_helmholtz_pollution.md`。

正式 case 执行前，本次相关 Ruff 为 `All checks passed`，新增定向测试为 `5 passed`，全仓回归为
`296 passed in 47.26 s`。一次过短 shell timeout 曾在约 1 s 时终止 pytest，未产生测试结论或 scientific
artifact；随后以正常时限完成上述全仓回归。这不涉及 R13 formal execution count。

non-scientific preflight 只执行一次：

```powershell
& 'D:\anaconda3\envs\tgv_ptycho_sim\python.exe' -u `
  scripts/run_exp040_r13_preflight.py `
  --config configs/experiments/exp040_TGV_3d_multislice_r13_preflight.yaml
```

run 为 `runs/exp040_TGV_3d_multislice_r13_preflight_20260817_150046`，shell exit `0`，status `Passed`，
`formal_r13_allowed=true`。它没有运行任何 formal `kh=8.85787` case；关键 non-scientific controls 为：

```text
H1/H2 coefficient decomposition relative error: 3.2183679066e-15
Hankel derivative identity relative error:        0
low-k p2 weighted relative L2:                    3.8731900196e-4
low-k p4 weighted relative L2:                    1.2411378356e-7
low-k p4/p2 error ratio:                          3.2044331140e-4
maximum formal benchmark active unknowns:         4032
free disk:                                        139.1083 GiB
```

所有 algebra、low-k FEM、resource 与 provenance gates 均为 true。preflight artifacts 为：

```text
config.yaml:                  1813 bytes
  A2EB4615BD271E50BF142ED431D21CF4BC755E4BB73DDDA69C7481F469290D62
metrics.json:                 5493 bytes
  768560E917245846C10C84D555B6EDEEF4731B4A60FECD3E5F71BF9DF810E519
outputs/exp040_r13_preflight.h5: 78848 bytes
  920C52A32D13C840A09DBF5F304033CB0ECC9C2B567A5AA362F0C7E42EAA3FFD
```

preflight 后只向 formal YAML 增加上述 provenance，scientific contract SHA256 仍为
`B37A16F0E6D4F91B498BE4240C2D83A8CA727259A6A38B2B35FEB25EEF51EBB0`。锁定后的正式配置为：

```text
configs/experiments/exp040_TGV_3d_multislice_r13.yaml
  4930 bytes
  FB5BEC5C2A41D303E936AADA1A3A43EB896FF7C71E7DE06B737A825944C908B0
```

最大单 solve 只有 4032 active unknowns；24 个 physical-k family solves 加两个 annular PML solves 预计为分钟内、
远低于 full-TGV 成本。正式命令固定为：

```powershell
& 'D:\anaconda3\envs\tgv_ptycho_sim\python.exe' -u `
  scripts/run_exp040_r13.py `
  --config configs/experiments/exp040_TGV_3d_multislice_r13.yaml
```

该命令只允许执行一次。无论 scientific gate 通过、失败或运行时失败，都保留为 R13 结果；不得第二次执行、看后
改阈值或追加 case。formal 仍明确禁止 Cartesian、full TGV、scalar cross-model 和 vector/Maxwell。

本节追加前文档为 `191094 bytes`，SHA256 为
`F05E02EC27BC861BBC6F4B3D56E10ED2BC858B3355BE93AFE2A2D7375A1BA7AF`。

### 18.34 R13 唯一 formal run：radial PML reflection 与 physical-`kh` candidate 均闭合

第 18.33 节锁定的 formal 命令只执行一次，shell exit `0`，run 为
`runs/exp040_TGV_3d_multislice_r13_20260817_150406`。scientific execution 为 `8.616 s`；所有 case 均按固定
顺序完成，HDF5、三张图和独立 artifact validator 全部通过。最终状态为：

```text
status: Passed
interpretation: r13_benchmarks_closed__iterative_solver_required
benchmark_validated: true
reference_validated: false
full_tgv_reference_authorized: false
domain_reflection_gate_pass: true
physical_k_candidate_found: true
hard_controls_pass: true
direct_lu_unknown_count_gate_report_only: false
```

这里的 `Passed` 只表示第 18.32 节定义的两个小型 benchmark 达标；它不表示 full-size FEM reference 或 exp040
已经完成。最大 solver residual 为 `8.3806e-14`，所有 numerical/truth arrays 均 finite。

#### 18.34.1 Radial PML：R12 outer guard 被证实不是 reflection coefficient

两个固定 annular outgoing cases 的结果几乎相同：

| diagnostic | core48 + PML2 | core60 + PML2 | gate |
|---|---:|---:|---:|
| maximum incoming/outgoing | `0.0274923%` | `0.0274923%` | `<=0.1%` |
| maximum exact-impedance residual | `0.0549710%` | `0.0549710%` | `<=0.1%` |
| dense analytic-field relative L2 | `0.00147288%` | `0.00147287%` | `<=0.1%` |
| radial-flux relative range | `0.00032607%` | `0.00032607%` | `<=0.5%` |

core48 -> core60 的共同 `21--46 um` dense field L2 仅 `9.1213e-11`，即 `9.1213e-9%`。这与 R12
canonical TGV 内区 pair 的 `0.00010899%` 一致地表明：把 radial PML start 从 48 移到 60 um 并不会实质改变
comparison field。R12 的 `7.3991%` local outer-field amplitude 只是 outgoing lobe 在该窗口仍非零，不能解释为
7.4% reflection；因此后续不应继续为了降低该 amplitude 而盲目扩大 core 或加厚 PML。

本 benchmark 直接验证 radial outgoing mode；它没有把所有 axial/material modes 都证明为零反射。后续 full
reference 仍应保留 R12 的 complex-field domain convergence，并可在 solver preflight 中补一个解析 axial
plane-wave PML control，但不得借此回改 R12/R13 已冻结的结论。

#### 18.34.2 Physical-`kh`：首个合格候选是 `h=0.5 h_formal, p=4`

所有误差均是未经 phase/scale alignment 的 analytic-truth weighted relative L2：

| case | `h/h_formal` | p | homogeneous | glass--air interface | eligible | projected full-TGV unknowns |
|---|---:|---:|---:|---:|---|---:|
| h1_p2 | 1 | 2 | `100.000%` | `100.000%` | no | 106,888 |
| h1_p3 | 1 | 3 | `79.4973%` | `61.1025%` | no | 240,684 |
| h1_p4 | 1 | 4 | `36.6027%` | `27.9929%` | no | 428,048 |
| h1_p6 | 1 | 6 | `1.04299%` | `1.08099%` | no | 963,480 |
| h1_p8 | 1 | 8 | `0.0345335%` | `0.0353310%` | yes | 1,713,184 |
| h0p5_p2 | 0.5 | 2 | `1838.03%` | `23.7627%` | no | 428,048 |
| h0p5_p3 | 0.5 | 3 | `7.06586%` | `7.80628%` | no | 963,480 |
| h0p5_p4 | 0.5 | 4 | `0.212098%` | `0.264622%` | yes | 1,713,184 |
| h0p5_p6 | 0.5 | 6 | `0.0019220%` | `0.0021620%` | yes | 3,855,408 |
| h0p25_p2 | 0.25 | 2 | `4.66662%` | `6.96080%` | no | 1,713,184 |
| h0p25_p3 | 0.25 | 3 | `0.223204%` | `0.262127%` | yes | 3,855,408 |
| h0p25_p4 | 0.25 | 4 | `0.0155360%` | `0.0158460%` | yes | 6,854,720 |

正式 spacing 上的 p6 只比 `1%` gate 高约 `0.043/0.081` percentage points，但必须保持不合格；不能看后把阈值
放宽到 1.1% 来选择一个更便宜的 case。合格集合为
`[h1_p8, h0p5_p4, h0p5_p6, h0p25_p3, h0p25_p4]`。按预注册的 projected unknown count、degree、
element ratio 排序，`h1_p8` 与 `h0p5_p4` 同为 1,713,184 unknowns，较低 degree 的 `h0p5_p4` 被唯一选中。

映射回物理量，该 candidate 是 `h=0.25 um, p=4`，formal glass `kh=4.428935`、`kh/p=1.107234`；它在
homogeneous/interface family 上分别为 `0.2121%/0.2646%`。这解释了 R12 `h=0.5 um` p2/p3 的巨大差异：
两者确实处于 pollution regime，而不是 FEM weak form 本身失效。`h0p5_p2` 的 1838% 还显示粗 mesh 可能落入
离散 resonance，误差不必随 uniform h 单调下降；因此只做一次 h-halving 也不是可靠策略。

#### 18.34.3 Resource fail-closed 与 artifact audit

被选 candidate 的 projected full-TGV size 为 `1,713,184` active unknowns，超过预注册 direct-LU 上限
`1,000,000`；而 p4 local coupling 还会增加 matrix/LU fill。因此 R13 正确保持
`full_tgv_reference_authorized=false`，没有组装或求解 canonical p4 TGV，也没有运行 scalar/vector comparator。

正式 HDF5 `/entry` 恰含 `config_yaml/data/instrument/metadata/metrics/sample/truth`，无 `reconstruction`；`data`
恰含 `domain_reflection/physical_k_pollution`，24 个 datasets 全部 finite，selected candidate 两个 family 的
numerical fields 均为 `(129,129) complex128`。三张 PNG 已实际打开检查，曲线、gate 与 candidate coloring 均与
metrics 一致。主要 artifacts 为：

```text
config.yaml:
  5282 bytes
  4186BE7C281FBD9D17EB4ADD040CFAEC5621ECE9BDD405F4C693E8A0009B0A88
metrics.json:
  54408 bytes
  E79206316739EA3535047316E57E0592AA8F1A5FC51E799BD7694904DFE01382
outputs/exp040_r13.h5:
  1790128 bytes
  C993DAAED6A935BD132D7C6B3572E63241E76E6FC02BFF80733CC2909F3607A0
r13_domain_reflection.png:
  126889 bytes
  87E5F07491FEFD15D8CB469DFE499DF72313A76B964F8E991B6C365A02CB1B7F
r13_physical_k_pollution.png:
  142691 bytes
  A0C3E4E3BD8A4EA9DF17D89260556E12E22E33B9A6CD93C1F6FEB49F9C80C42D
r13_candidate_resource.png:
  64778 bytes
  41AAAFC43DBA1CB26A310597B6807350FA4304BE4A358C9AB93ED6572D60C4D2
```

### 18.35 R13 后建议：先闭合 scalable solver，再运行一次 full reference

1. **预注册 R14 solver/resource benchmark**：以 R13 唯一选中的 `h=0.25 um, p=4` 为主，不再搜索 `h,p`；
   在同一 physical-`kh`、glass--air interface 与 PML operator 上做逐级 domain-size scaling，保存 residual history、
   analytic error、iteration count、wall time 与 peak RSS。优先比较 literature-backed shifted-Laplacian
   preconditioned FGMRES/GMRES 和具有 coarse correction 的两级 domain decomposition；仅用小矩阵 direct LU
   作 verification，不把它外推为 full solver。
2. **求解器 gate 必须同时约束准确性与可扩展性**：algebra residual 小不等于 pollution error 小；每一级都要
   保持 analytic-field error `<=1%`，并预先限制 iteration growth 与 projected peak memory。若只能靠接近 full
   direct factorization 的预条件器才能收敛，则 solver gate 失败。`h1_p8` 可以在新预注册中作同 unknown-count、
   不同 stencil 的 report comparator，但不得替换 R13 已选的 p4 主候选。
3. **低成本补齐 axial PML control**：可在 R14 preflight 加入解析向上/向下 plane-wave complex-coordinate PML
   benchmark，阈值沿用本轮预注册前重新书面固定；它只补足 radial-only benchmark 的范围，不重新启用 R12 的
   local-amplitude guard。
4. **条件式 full reference 与 cross-model**：只有 iterative solver 的 accuracy/memory/convergence gates 全部通过，
   才允许一次 canonical core60 `h=0.25 um, p=4` FEM solve；随后复用 R12 已冻结的 Cartesian checkpoint 做 scalar
   cross-model。若该时差异仍超过原 `5%` 且结构化，才讨论 vector/Maxwell。当前没有证据要求更换物理模型。

本轮最重要的变化不是继续“把 dx/h 变细”，而是已经用 analytic benchmark 证明需要的离散区间，并明确下一道
门槛是可扩展 Helmholtz solver。R13 追加前文档为 `194433 bytes`，SHA256 为
`1A6631EC098DE888BE69A28F1D3644059AAA3558D13E3DA89F82E59230316F99`。

### 18.36 R14 预注册：bounded-memory iterative solver scaling 与 axial-PML control

R13 已把 discretization candidate 固定为 physical `h=0.25 um, p=4`；本轮禁止继续搜索 `h,p`、放宽 `1%`
analytic gate，或直接尝试 1,713,184-unknown canonical TGV。当前机器只有 SciPy/SuperLU，没有 PyAMG、
PETSc/Hypre、Pardiso 或 GPU backend。因而 R14 首先回答：仅用当前可复现环境，是否存在不会退化成 full global LU
的 bounded-memory preconditioner；若答案是否定，必须把失败记录下来并转向明确的外部 solver/backend，而不是
继续在小域上调 ILU 参数制造漂亮曲线。

#### 18.36.1 固定 operator、modal families 与 scaling series

solver benchmark 使用 R13 candidate 的 nondimensional equivalent：Q4、`h/h_formal=0.5`、operator
`k=8.85787`、glass--air `n²=4/9 -> 1` radial interface、两侧/径向 cubic complex-coordinate PML，PML thickness
为 1、target one-way amplitude 为 `1e-8`。物理 core extent 按固定顺序为 `4/8/16/32/64`；含 PML 后 active
unknowns 必须恰为 `1880/5688/19448/71544/274040`。最大 case 仍只占 projected full TGV 的约 16%，但已经跨越
约 146 倍 unknown-count range，足以证伪明显的 iteration 或 storage blow-up。

每个 matrix 固定两个 RHS/modal families。primary mode 在 complex-stretched outer domain 上选择最接近 R13
radial/axial modal wavenumbers `5.302909/6.283185` 的 Bessel-zero/sine indices；offset family 将两个 index 均加
一。解析场在 complex coordinates 中满足 outer homogeneous Dirichlet，manufactured source 对同一未对齐
glass--air interface 与 PML coefficients 精确构造。这样 iterative error、FEM pollution 与 boundary stretch 可由
analytic truth 分开；两个 family 用来避免单一模态恰好落在 nodal null 或离散 resonance。禁止 phase/scale
alignment。core4/core8 另运行原 operator direct solve，只检查 iterative discrete solution agreement；direct LU
不得用于其余规模或任何 preconditioner。

#### 18.36.2 两条预注册 solver 路线及明确禁止项

所有 solver 使用 zero initial guess、GMRES `rtol=1e-8`、restart40、最多 400 inner iterations；实际 gate 更严格为
最多 300。共同 complex shifted Laplacian 固定 mass shift `1+0.5i`。

1. `csl_ilu_gmres`：对 shifted operator 做固定 `drop_tol=1e-3`、`fill_factor=4`、`basic,area` drop rule 的
   global incomplete LU。它允许作为候选，是因为 fill 被显式限制；任何自动退化为 complete LU、看后增加 fill 或
   降低 drop tolerance 都禁止。
2. `two_level_ras_csl_gmres`：固定 `64x64` active-node nonoverlapping cores、4-node overlap；每个固定大小
   shifted local block 可 direct-factor，correction 只写回 core。coarse space 在每个 core 上固定一个 normalized
   constant，以 shifted Galerkin operator 做 coarse correction，再作 restricted additive Schwarz。这是最小、可复现
   的 two-level comparator；若 high-frequency coarse space 不足而失败，不能在本轮看后加入 plane waves、GenEO
   vectors 或改变 block size。

尤其禁止以完整 shifted-operator LU 作为“preconditioner”：它即使给出 1--2 次 GMRES，也有与当前卡点相同的
fill/memory complexity，不构成 scalable solver。没有安装的新 package 也不得在 formal 中临时引入。

#### 18.36.3 预注册 gates 与 memory projection

solver candidate 必须对全部五个 scales、两个 modal families 同时满足：true residual `<=1e-8`、无 alignment 的
analytic weighted L2 `<=1%`、GMRES inner iterations `<=300`；core4/core8 对 direct discrete solution 的 L2
还须 `<=1e-6`。core64/core8 的最坏 modal iteration ratio 固定 `<=4`。任一未收敛、breakdown、non-finite 或
缺失 case 都使该 solver 失败，不允许只报告最容易 RHS。

memory 保存原 matrix、preconditioner factors、coarse factor 和 restart basis 的实际 sparse-array bytes；以最大
case bytes/unknown 线性投影到 1,713,184 unknowns，coarse direct fill 单独按实测增长外推，最后乘固定 `1.5`
safety factor。projected peak 必须 `<=10 GiB`，formal 实测 process peak 必须 `<=8 GiB`。至少一条 solver 同时
通过 accuracy、iteration 与 memory gates，`solver_candidate_found` 才为 true；即使通过，本轮也仍禁止 full TGV，
只能授权下一轮一次正式 full-reference solve。

#### 18.36.4 Axial PML 与执行边界

为补足 R13 radial-only benchmark，另固定 air upward 与 glass downward 两个一维 plane-wave weak-form PML case：
physical core `4 um`、PML `2 um`、Q4、`h=1/12 um`、同一 cubic/`1e-8` stretch。field/derivative 的
`exp(+ikz)/exp(-ikz)` decomposition、exact impedance residual 与 dense analytic-field L2 均须 `<=0.1%`。
该 control 只验证 homogeneous axial stretch，不宣称覆盖全部 TGV mixed modes。

必须先完成实现、定向/全仓测试、Ruff 和不含 formal-k scaling fields 的 non-scientific preflight，锁定 provenance
后 formal 只运行一次。无论 gate 失败、内存失败或 solver breakdown，都保存为 R14 结果，不调整 shift、ILU、
block/coarse space、iterations 或 case order。R12 Cartesian checkpoint 仅作 hash provenance，不读取 field；
full TGV、scalar cross-model 与 vector/Maxwell 全部 disabled。

本节追加前文档为 `202172 bytes`，SHA256 为
`0F955AD713DC789A2D22485F4233488901982E698AA1C6F788DE558CB7560CC6`。

### 18.37 R14 实现审计、memory 外推固定与 preflight execution lock

第 18.36 节的 scientific contract 已实现，但尚未运行 formal。公共实现新增 complex-stretched analytic modal
problem、无 alignment 的 axisymmetric nodal error、air/glass axial plane-wave PML、bounded-fill CSL-ILU、
constant-coarse two-level RAS-CSL、restart40 GMRES true-residual history，以及 matrix/factor/Krylov storage
accounting。正式 runner 对每个 solver 的 setup 与两条 modal solve 分别捕获异常；某条路线失败时仍会保留 failure
type/message、另一条路线和已完成 scale，而不是让绘图阶段把科学失败覆盖成 artifact failure。synthetic artifact test
还显式覆盖了 `projected_peak_gib=None` 与缺失 GMRES fields 的情况。

#### 18.37.1 对 18.36.3 coarse-memory 规则的执行前唯一化

第 18.36.3 节已写明 coarse direct fill 要“按实测增长外推”，但原型代码曾暂用固定 `1.5` power；这会把 safety
factor 与 fill-growth exponent 混为一谈，现已在任何 formal field 生成前删除。固定且不再看后改变的算法为：

1. 对 five-scale 实测 `(block_count, coarse_factor_storage_bytes)` 全部取自然对数；
2. 用含 intercept 的 ordinary least-squares 拟合 `log(bytes)=a+b log(block_count)`；
3. 以该拟合预测 full-TGV block count 的 coarse bytes；
4. 同时从 core64 按 block count 做线性延伸，二者取较大者，禁止用 sublinear/偶然 sparse fill 降低预测；
5. non-coarse 最大 case bytes 仍按 unknown count 线性外推，二者相加后才乘第 18.36.3 节已固定的 `1.5`
   safety factor。

该 clarification 没有放宽 `10 GiB` gate，也没有增加 solver candidate；它只是把 formal 前已有文字规则变成唯一可复现
公式。formal config 已增加对应 machine-readable rule，新的 scientific-contract SHA256 固定为
`269AA8FA68EB7795B2A5EB73D3F4A23B5C2C2E90382CE6613EE78253A043E2DB`。

#### 18.37.2 非正式验证与 preflight 锁

纯数学背景另记于 `docs/theory_notes/exp040_r14_shifted_laplacian_domain_decomposition.md`；其中没有 R14 case、
阈值或结果。正式前的相关 Ruff 为 `All checks passed`，定向测试为 `10 passed in 3.47 s`，全仓回归为
`303 passed in 55.44 s`。这些测试仅使用 synthetic contract 或 low-`k` 小矩阵，不生成第 18.36.1 节的 formal
scaling fields。

non-scientific preflight 固定检查：R13 provenance、formal contract hash、SciPy backend 与未安装 backend 状态、
formal shape/unknown count、free disk/RSS、两条 preconditioner 的 deterministic operator repeat、low-`k` direct
agreement/true residual，以及 axial-PML smoke control。low-`k` RAS 只使用 `16x16` core、2-node overlap 来使小矩阵
确实包含多个 subdomains；它不是 formal `64x64/overlap4` 的替代参数，也不参与科学 candidate 选择。preflight
不得读取 R12 field、不得组装 `k=8.85787` formal matrices、不得运行 full TGV 或 Cartesian comparator。

preflight config 已锁定为：

```text
configs/experiments/exp040_TGV_3d_multislice_r14_preflight.yaml
SHA256 BE215853F01FF975B05F0BF5642319AF4C4062BC44A86FAB8B909759387BB3B6
```

只允许执行一次的命令为：

```powershell
& 'D:\anaconda3\envs\tgv_ptycho_sim\python.exe' -u `
  scripts/run_exp040_r14_preflight.py `
  --config configs/experiments/exp040_TGV_3d_multislice_r14_preflight.yaml
```

预计 preflight 为秒级、小于 2 GiB peak-RSS gate；它只决定 formal 是否可启动。若失败，先保存并记录该失败，不运行
formal，也不改变 scientific thresholds。本节追加前文档为 `207691 bytes`，SHA256 为
`B25DE9C7F34EAA4842378210DAD6277B1E9532F47ACD5DA9D5BC73BC7BF15FDE`。

### 18.38 R14 唯一 preflight 结果：solver algebra 通过，但 glass axial derivative floor 阻止 formal

第 18.37 节锁定的 non-scientific preflight 只执行一次，shell exit `0`，run 为
`runs/exp040_TGV_3d_multislice_r14_preflight_20260817_161959`。artifact validator 通过，但 scientific-release
状态按 gate 正确保存为：

```text
status: Failed
formal_r14_allowed: false
provenance_and_contract_pass: true
backend_pass: true
low_k_algebra_pass: true
axial_pml_smoke_pass: false
resource_pass: true
```

因此没有运行第 18.36 节的 formal `k=8.85787` scaling cases，也没有创建任何 R14 formal run。两条 low-`k`
solver 都闭合到 direct discrete solution：

| solver | GMRES inner iterations | true residual | direct agreement L2 | operator repeat L2 |
|---|---:|---:|---:|---:|
| CSL-ILU | 23 | `6.71037e-9` | `1.28404e-8` | `0` |
| two-level RAS-CSL | 30 | `8.39228e-9` | `2.19981e-8` | `0` |

low-`k` analytic-field error 约 `12.409%`，它在执行前已明确是 report-only：该小矩阵只验证 iterative solution 是否
复现同一个 discrete direct solution，不用来替代 R13 已完成的 physical-`kh` accuracy benchmark。当前 SciPy backend、
274,040 maximum formal unknowns、`0.09963 GiB` process peak 与 `139.105 GiB` free disk 均通过。

失败只来自 glass downward axial control：

| case | `kh` | incoming/outgoing | impedance residual | dense field L2 | original gate |
|---|---:|---:|---:|---:|---:|
| air upward | `0.984208` | `0.0274927%` | `0.0549717%` | `0.00146788%` | `0.1%` |
| glass downward | `1.476312` | `0.136524%` | `0.272753%` | `0.0112103%` | `0.1%` |

不能在看见 `0.1365%` 后把 gate 放宽，也不能忽略 preflight 直接启动 formal。现有两点给出一个需独立验证的明确
attribution：air 到 glass 的 wavenumber ratio 恰为 `1.5`，而 incoming、impedance 与 dense-field errors 的
log-ratio orders 分别为 `3.9524/3.9504/5.0140`，与 Q4 的 derivative `O((kh)^4)` 和 field
`O((kh)^5)` floors 高度一致。若是 outer-PML reflection 主导，不应自然产生这一组 Q4 power laws。因此当前最强
假设是 continuum `exp(±ikz)` derivative decomposition 把 interior FEM dispersion/derivative error 读成了 incoming
wave；这还不是最终证明，必须用预注册 mesh/PML separation 检查，而不是继续调 ILU 或盲目运行五级 solver。

主要 artifact 为：

```text
metrics.json:
  8515 bytes
  BC516A9A0428C7696A1E49D6CD175514D41D12E3DB3E97A0B1D53BB7F1366879
outputs/exp040_r14_preflight.h5:
  117216 bytes
  DFC6B68547DD9A2FCEC3DEF1CA8E676626B09DBD653C3C8339BC37E248D58C1B
```

HDF5 `/entry` 为 `config_yaml/data/instrument/metadata/metrics`，`data` 为空，确认没有把 preflight 冒充科学
field。该失败是 R14 initial contract 的永久结果，不重跑、不覆盖。

### 18.39 R14A 预注册：axial Q4 derivative-floor 与 PML attribution

本轮只拆解第 18.38 节唯一失败项，不改变 Helmholtz/TGV 物理模型，不运行 solver scaling。`h=1/12 um` glass
baseline 必须从上述已锁定 metrics hash 读取，禁止重新计算。新 case 顺序固定为：

```text
glass_h16_pml2: Q4, h=1/16 um, core=4 um, PML=2 um
glass_h24_pml2: Q4, h=1/24 um, core=4 um, PML=2 um
glass_h24_pml3: Q4, h=1/24 um, core=4 um, PML=3 um
```

三者均固定 `n=1.5`、downward direction、`lambda=532 nm`、cubic complex stretch、target one-way amplitude
`1e-8`、12-point quadrature、measurement fractions `0.2/0.5/0.8` 与 401-point dense comparison。这里把一维
control 细化到 `h=1/24 um` 的最大 active unknowns 只有约 700，成本与 full-TGV uniform refinement 不同；选择来自
已观测到的 Q4 derivative-order signature，而不是无依据地减小 full-model `h`。

在运行前固定以下判据：

1. 由 locked `h12` 与新 `h16/h24, PML2` 三点，对 incoming ratio、impedance residual、dense-field L2 分别作
   含 intercept 的 `log(error)`--`log(h)` least-squares；前两项 slope 必须 `>=3.0`，field slope 必须 `>=4.0`，
   才支持 Q4 discretization-floor attribution；
2. `h24, PML2` 与 `h24, PML3` 各自仍须通过原 gate：incoming、impedance 与 dense-field L2 均 `<=0.1%`，
   不允许放宽；两者在共同 physical-core dense nodes 上的 raw complex-field L2 必须 `<=0.01%`，禁止 phase/scale
   alignment；
3. 全部 direct-solver residual 必须 `<=1e-10`，所有 arrays/metrics finite；
4. 只有三项同时通过，`corrected_axial_control_eligible=true`。通过也不能回写第 18.38 节，只能授权一个新 formal
   contract 复用 `glass_h24_pml2` checkpoint；失败则转向 discrete-impedance/derivative-recovery verification，
   不再继续减小 `h`。

R14A 只允许在实现、Ruff、定向测试和 artifact synthetic test 通过后运行一次；输出必须含 baseline provenance、
三 case metrics、complex fields、convergence/PML-separation 图和可复用 checkpoint。full TGV、R12/R13、formal
solver scaling、cross-model 与 vector model 全部 disabled。本节追加前文档为 `211462 bytes`，SHA256 为
`43697A2D55D16CBA85077015AEC830820B0E48A17D46A9C828676BD46C39B3C7`。

### 18.40 R14A 实现与唯一 execution lock

第 18.39 节已实现为独立配置/runner/plotter，并加入 config/provenance/order estimator、synthetic HDF5、checkpoint
与 figure tests。测试阶段没有调用三个新 axial cases；只构造 synthetic arrays，并只读验证第 18.38 节 baseline
hash。相关 Ruff 为 `All checks passed`，定向测试为 `7 passed in 2.50 s`，全仓回归为
`305 passed in 56.37 s`。

R14A scientific-contract SHA256 为
`1305D2D4D46E4AE8FB5F974340F3B639928DF876DAA2FC3AB1B9D5C8DE318AE7`；锁定配置为：

```text
configs/experiments/exp040_TGV_3d_multislice_r14a.yaml
SHA256 DE60B9FD75981F3EDD24B4297ABC5CD0AB42253484E96537E7D692597F39607D
```

正式计算只有三个小型 1D Q4 direct solves，预计秒级；只允许执行一次：

```powershell
& 'D:\anaconda3\envs\tgv_ptycho_sim\python.exe' -u `
  scripts/run_exp040_r14a.py `
  --config configs/experiments/exp040_TGV_3d_multislice_r14a.yaml
```

无论结果通过或失败均保留，不改变 order/original/PML-separation gates，不追加 mesh case。若通过，后续新 solver
formal 必须按 hash 复用 `glass_h24_pml2.npz`，不能再次计算该 corrected control。本节追加前文档为
`216661 bytes`，SHA256 为
`B7DF63E299B54A83EACEDC6DCDD65E7C0FBF89573D9DB2227D24FA33619DEBEE`。

### 18.41 R14A 唯一 formal 结果：旧 glass axial failure 被归因于 Q4 derivative floor

第 18.40 节命令只执行一次，shell exit `0`，run 为
`runs/exp040_TGV_3d_multislice_r14a_20260817_163038`，实际 scientific computation 为 `0.1681 s`。最终状态为：

```text
status: Passed
interpretation: q4_derivative_floor_attributed__corrected_control_eligible
corrected_axial_control_eligible: true
q4_order_attribution_pass: true
original_metric_gate_pass: true
pml_separation_pass: true
hard_controls_pass: true
```

锁定 baseline 与两个 PML2 新点给出：

| Q4 h | incoming/outgoing | impedance residual | dense field L2 |
|---:|---:|---:|---:|
| `1/12 um`（复用） | `0.136524%` | `0.272753%` | `0.0112103%` |
| `1/16 um` | `0.0438573%` | `0.0876806%` | `0.00264648%` |
| `1/24 um` | `0.00875826%` | `0.0175151%` | `0.000349018%` |

三点 log--log slopes 分别为 `3.96307/3.96167/5.00479`，与预注册的 Q4 derivative fourth-order、field
fifth-order signature 一致。`h=1/24 um` 下把 PML 从 2 增至 3 um 后，raw complex physical-core field L2 只有
`1.59201e-12`；两者的三项 metric 也在数值精度内相同。由此可排除“继续加厚 axial PML”作为有意义主线：旧
`0.1365%` 不是 material PML reflection，而是以 continuum impedance 分解有限元 derivative 时的离散 floor。

这不是把原阈值看后放宽；`h24` 两个 cases 都通过原 `0.1%` gate。改动只发生在成本极低的 analytic validation
control，不改变 canonical TGV 的 `h=0.25 um, p=4` candidate。全部 direct residual `<=4.392e-15`，三 case
最多 671 active unknowns。HDF5、checkpoint 与图已实际打开检查：HDF5 `/entry/data/axial_attribution` 和
`/entry/truth` 各含三个 cases，共 15 个 data datasets；checkpoint 为 `(401,) complex128` finite field；图中
order lines、原 gate 和 PML2/PML3 separation 与 metrics 一致。

主要 artifacts 为：

```text
metrics.json:
  6636 bytes
  E6984C5094249BCBC5FE500101E17DE0125F6270D512ED9A127819DA52DCE53C
outputs/exp040_r14a.h5:
  164152 bytes
  7D1A8787140C58C3E2160241A6F5904B1038E23A8D9A6CFEFC814C5D048EC91D
checkpoints/glass_h24_pml2.npz:
  17036 bytes
  AA1EE4AE49781F323B4E0D3F87B2C7FA8ABF7E02CCD05E4566D9DA709B4ACBAE
figures/r14a_axial_attribution.png:
  187554 bytes
  B44F9DDE60E2FF38454A9488D583F4B84B1CAED30C3174EAA2CB9C2E85247A3D
```

### 18.42 R14B 预注册：复用 axial controls 的 solver-only formal release

R14 initial preflight 的 solver algebra/backend/resource 均已通过，R14A 又单独关闭了唯一 axial failure；因此下一步
不重跑任一 control，而建立一个 provenance-only release lock。R14B 的 five-scale matrices、两个 modal families、
CSL shift、ILU、RAS block/overlap/coarse space、GMRES、accuracy/iteration/memory gates 与第 18.36--18.37 节完全
相同，禁止搜索新参数。唯一变化是 axial controls 的来源：

- `air_upward` 复用第 18.38 节 initial-preflight metrics；它已通过原三项 `0.1%` gates；该 run 没保存 air field，
  因此 R14B HDF5 只记录 metrics provenance，不伪造 numerical array；
- `glass_downward` 复用本节 `glass_h24_pml2.npz` 的 metrics 与 arrays，并校验 R14A metrics/HDF5/checkpoint 三个
  hashes；不得再次求解该 1D case。

non-scientific release 只读检查：initial preflight 除 axial 外的四个 gates、R14A 全 gate、checkpoint content、R13/R12
provenance、free disk，以及新旧 formal configs 的 `solver_scaling/solvers/memory_projection/thresholds` 完全相等；
它不得 assemble Helmholtz matrix。只有这些检查全部通过才写 `formal_r14_allowed=true`。

随后 R14B formal 只允许执行一次。仍固定 5 scales、2 RHS、2 solvers，最大 274,040 unknowns；预计 5--15 分钟，
必要性是判断 1,713,184-unknown reference 是否存在当前环境可承受的 solver，而不是再减小 `h`。formal status 仍由
reused axial gate、至少一个 solver candidate 和 hard controls 共同决定；无论通过、未收敛、setup failure 或 memory
gate failure 都保存，不重跑、不调 shift/fill/block/coarse space。full TGV、Cartesian、cross-model 与 vector model
继续 disabled。

若两条路线均失败，本轮应得出“当前固定 SciPy bounded-memory routes 不足”，下一步另行预注册具有 plane-wave/
spectral coarse space 的 PETSc/Hypre/DD comparator 或 sweeping method；不能继续在小域上调 ILU 制造好看的曲线。
若至少一条通过，才授权下一轮一次 canonical full-reference solve。本节追加前文档为 `217983 bytes`，SHA256 为
`8347BDCCCC7BDE086DBACAEC6EF2C14DC60291180CB269655B5CDB4D26B24F2E`。

### 18.43 R14B 实现、不变量验证与 provenance-only release lock

R14B 使用与 initial R14 共用的 formal runner；新增逻辑仅在 `axial_pml.reuse_registered_controls=true` 时读取
hash-locked metrics/checkpoint。`air_upward` 没有保存 numerical arrays，HDF5 会诚实地只写其 metrics；
`glass_downward` 从 R14A checkpoint 写 data/truth。runner、HDF5 和 plot 已能接受这种不对称 provenance，且不会
调用 axial solver。

新旧 formal configs 的 `solver_scaling/solvers/memory_projection/thresholds/conditional_execution` 已做 exact
mapping comparison，全部相等。R14B scientific-contract SHA256 固定为
`F4247CC4298E61092363AD7FAD65016820088608D9A5FF42009ACEB50F1D1D37`；与旧 contract 的 hash 不同只因为 axial
section 改成 provenance reuse，solver scientific contract 未改变。相关 Ruff 为 `All checks passed`，定向测试为
`9 passed in 3.71 s`，全仓回归为 `307 passed in 55.89 s`；所有测试只读 checkpoints 或使用 synthetic fields，
没有组装 formal matrix。

release config 锁定为：

```text
configs/experiments/exp040_TGV_3d_multislice_r14b_release.yaml
SHA256 F54AB11A1AA961761E2A10EEA75ABDBA30578281D5FCC7ACB3F1A0073CCB8C42
```

它只允许执行一次：

```powershell
& 'D:\anaconda3\envs\tgv_ptycho_sim\python.exe' -u `
  scripts/run_exp040_r14b_release.py `
  --config configs/experiments/exp040_TGV_3d_multislice_r14b_release.yaml
```

release config 明确 `assemble_formal_matrix=false`、`rerun_axial_control=false`。只有 release artifacts 保存且
`formal_r14_allowed=true` 后，才向 formal config 添加 release hashes、计算最终 source-config hash 和锁定唯一 formal
命令；release 本身不产生科学结果。本节追加前文档为 `222693 bytes`，SHA256 为
`989C64B71ABF9C5FE84B01BC132F5F6A1C65564AD6980DBE7C7BC76B3029854A`。

### 18.44 R14B release 结果与唯一 formal execution lock

第 18.43 节 provenance-only release 只执行一次，shell exit `0`，run 为
`runs/exp040_TGV_3d_multislice_r14b_release_20260817_164058`。状态为 `Passed`、
`formal_r14_allowed=true`；artifact hashes、initial solver preflight、R14A corrected axial、solver-contract invariants、
reused axial gates、R12/R13 provenance 与 resource 七项 gates 均为 true。free disk 为 `139.1054 GiB`，release
HDF5 的 `data` 为空，确认没有 scientific matrix/field computation。

release artifacts 为：

```text
metrics.json:
  4075 bytes
  EC2DDB580ED21C2CC96CFBCF96F76988199C8EFE4689AEDF9152FFFC0141F61A
outputs/exp040_r14_preflight.h5:
  69224 bytes
  ED74F01B815C4BBC4C3064239EEB1DB358BFF05BB8B177C2625CB863197B0D42
```

这些 hashes、R14B scientific-contract hash 与 upstream hashes 已加入 formal provenance；锁定后的正式配置为：

```text
configs/experiments/exp040_TGV_3d_multislice_r14b.yaml
SHA256 178C7E64C0E399F38D41821950C962088FD3E86C07CBA1910D6ACF582B29FE67
```

锁定后相关定向测试为 `5 passed in 2.83 s`，全仓回归为 `307 passed in 55.78 s`。正式运行将 assemble
5 个逐级矩阵，对两条 modal RHS 分别运行两条 solver；最大 case 为 274,040 unknowns，预计约 5--15 分钟，且
可能达到显著 CPU/RAM 使用。这一步不可由更小 benchmark 替代，因为 iteration-growth 与 coarse/global fill 只有跨
146 倍 unknown range 才可判断。

只允许执行一次的命令为：

```powershell
& 'D:\anaconda3\envs\tgv_ptycho_sim\python.exe' -u `
  scripts/run_exp040_r14b.py `
  --config configs/experiments/exp040_TGV_3d_multislice_r14b.yaml
```

若 scientific gate 失败或某条 solver breakdown，仍保留本次 run，不执行第二次、不调参；若进程级 OOM/OS kill
使 Python 无法完成 artifacts，则 `run_state/run_progress` 是本次唯一结果，也不得按原 contract 重跑。formal 仍不
执行 full TGV 或 cross-model。本节追加前文档为 `224566 bytes`，SHA256 为
`CD4A64C5A8781AC1D46636B9F5F938AD4AA0EC506F089CBBF861A20F8AC7A5B2`。

### 18.45 R14B 唯一 formal：当前两条 SciPy solver 路线均未闭合

第 18.44 节命令只执行一次，shell exit `0`，run 为
`runs/exp040_TGV_3d_multislice_r14b_20260817_164415`。五级、两 RHS、两 solver 均按固定顺序完成；总 scientific
execution 为 `402.659 s`，无 setup exception，artifact validator 通过。最终结果为：

```text
status: Failed
interpretation: r14_no_scalable_scipy_solver
axial_pml_gate_pass: true
solver_candidate_found: false
hard_controls_pass: true
reference_validated: false
full_tgv_reference_authorized: false
next_full_reference_preregistration_allowed: false
```

#### 18.45.1 Solver scaling：ILU 内存可承受但 iteration 不可扩展，constant-coarse RAS 更差

| solver | worst iterations core4/8/16/32/64 | core64/core8 | core64 true residual | projected full peak | result |
|---|---|---:|---:|---:|---|
| CSL-ILU | `32/54/113/203/359` | `6.648` | `5.04e-9`（worst converged family） | `7.420 GiB` | iteration/growth failed |
| two-level RAS-CSL | `28/57/120/243/400` | `7.018` | `8.78e-8`（not converged） | `13.557 GiB` | convergence/growth/memory failed |

CSL-ILU 对所有 10 个 solves 都达到 `<=1e-8` true residual，core4/core8 direct agreement 也通过；但 core64 两个
families 为 `359/331` iterations，超过 300 gate，growth 也超过 4。它只能说明这组 bounded-fill ILU 在本机可能
装得下，不能说明随 full 1,713,184 unknowns 可扩展。

RAS 在 core64 两个 families 都用满 400 iterations，true residual 为 `8.78e-8/6.74e-8`。其 memory failure 不是
81-dimensional coarse direct factor 主导：core64 coarse factor 只有 `29,304 bytes`，而 overlapping local factors
为 `968,194,032 bytes`、retained shifted matrix 为 `197,214,404 bytes`。因此以后即使改用更好的 wave-aware
coarse space，也还需改变 local-solve/storage strategy；只加 coarse vectors 未必能把 `13.56 GiB` 降到 gate 内。

两条 solver 的 core4/core8 direct-discrete agreement 均 `<=7.08e-8`，所以小域实现正确；失败来自 domain growth，
不是低级 coding mismatch。最大实际 process peak 为 `2,894,774,272 bytes`（`2.696 GiB`），matrix repeat max error
为 `0`，所有 assembled matrices finite。

#### 18.45.2 Accuracy gate 也独立失败：R13 candidate 没有直接转移到本次 stretched eigenmode

两条 solver 在相同 case/family 上的 analytic errors 一致到约 `1e-9`，说明下列误差不是 iterative residual：

| core | primary analytic L2 | offset analytic L2 |
|---:|---:|---:|
| 4 | `3.08360%` | `3.01300%` |
| 8 | `2.66596%` | `2.63130%` |
| 16 | `2.05912%` | `2.05219%` |
| 32 | `1.82159%` | `1.82502%` |
| 64 | `1.67877%` | `1.68303%` |

所有 scale 都高于预注册 `1%`，不能因随 domain 下降而外推成通过。这与 R13 同 `h,p,k/interface` family 的
`0.2121%/0.2646%` 明显不同，表明 R14 增加的 stretched-eigenmode truth 并非中性的 solver carrier。

post-run 只读审计发现该 truth 由 outer complex Dirichlet boundary 反推 modal beta；以 core64 primary 为例，
`beta_r=5.2990-0.1695i`、`beta_z=6.2583-0.3944i`。虚部使 physical core 内的 analytic amplitude 随 domain
产生很强的指数不均匀性。更直接的现有-array 证据是：core64 offset family 的 center radial trace relative L2 为
`51.88`，而 global weighted L2 仅 `1.683%`；global denominator 被其它高幅区域支配，掩盖了低/中幅区域的巨大
relative error。primary center trace 又落在偶数 axial mode 的解析零面上，虽不参与 gate 的唯一判定，却进一步说明
center trace 不是稳定的单一诊断。

因此 R14B 的合法结论有两层：固定 RHS 下两条 SciPy routes 未通过；同时当前 stretched eigenmode 不能作为唯一的
continuous-accuracy carrier。它不能被提升为“所有 Helmholtz iterative solvers 都不可行”，也不能据此直接引入
vector/Maxwell 来解释数值失败。

#### 18.45.3 Axial reuse、artifact 与图审计

hash-reused air/glass controls 均通过原 `0.1%` gates，`control_source=hash_locked_provenance_reuse`，没有重复
axial solve。HDF5 `/entry` 为 `config_yaml/data/instrument/metadata/metrics/sample/truth`；`data` 含
`axial_pml/solver_scaling`，五个 scales 全部存在，air 仍只有 metrics、未伪造 field。共检查 532 个 floating/complex
datasets，全部 finite。三张 PNG 已实际打开，axial gate、iteration/accuracy curves 与 memory bars 都与 metrics 一致。

主要 artifacts 为：

```text
metrics.json:
  196356 bytes
  43741B86F6124182D1CEBFF4476682D5905E5310DC7064BE8BDE6F594CA8BF65
outputs/exp040_r14b.h5:
  1025904 bytes
  16D638D894310323413AD8DA3DB2F3045221E5EC54647F3862FD741C41857907
figures/r14_axial_pml.png:
  163501 bytes
  7D40021CCDB0C36D557A03153962C1AF8CA43C4702069F8EA58C0360023DC622
figures/r14_solver_convergence.png:
  159287 bytes
  A8B7E43221D7172733C1FCD6890792ECA224C8C7E4AFDBE47431C5AE9E0DDAF9
figures/r14_solver_resource.png:
  83591 bytes
  F466C9D1B77650C4FB0A6AFE1812DC66F85DB849D9EA5310FAA12022EE94B918
```

### 18.46 R14B 后建议：先修正 benchmark carrier，再决定 hp 或外部 solver

本节保留 R14B 完成时形成的科学建议，不是冻结后的 active work plan；当前工作安排以第 19 节的冻结决定为准。
以下原始建议不作追溯改写。

最合理的下一步不是重跑 R14B、把 300 放宽到 400、增加 ILU fill，也不是直接尝试 1.7M full TGV。应另行预注册
一个低成本 Stage C accuracy attribution，再有条件进入 solver Stage D：

1. **有界 manufactured truth**：保持同一 `h=0.5 ratio,p4,k=8.85787`、glass--air interface 与 PML matrix，
   但用 real physical wavenumbers 乘 smooth compact/scaled envelope，使 field 及前两阶导数在 PML start 前为零；
   两个固定 phase families 做单位 weighted-norm normalization。这样不再由 outer complex boundary 决定 beta，也不
   产生随 domain 指数增长的 truth。
2. **先小域 direct accuracy gate**：在 core4/core8 只运行 direct verification，预先同时固定 global weighted L2、
   amplitude-stratified/local error 与 dynamic-range controls。若仍超过原 `1%`，说明 R13 candidate 对这类 operator
   确实不够，应回到有 analytic justification 的 hp/adaptive discretization，而不是换 solver。
3. **accuracy 通过后才重做 RHS-relevant scaling**：固定 normalized bounded RHS 后再比较当前两条 routes；R14B 已经
   证明 constant coarse 与 overlapping direct locals 不是可接受默认方案，不能调参复活。若 iteration/memory 仍失败，
   新 comparator 应预注册 wave-aware/spectral coarse space 加 bounded-memory local solves，或 sweeping/polarized-trace
   method；PETSc/Hypre/外部 backend 需要单独 dependency/resource preflight。
4. **继续 fail-closed**：在 bounded truth、solver scaling 和一次 full scalar reference 依次通过前，不恢复 cross-model，
   不讨论 vector/Maxwell。R12 Cartesian checkpoint 与 R13 candidate 继续保留，不重算。

这条路线不是继续 uniform brute-force refinement；它先修复当前 benchmark 中已经被数据证实的条件数/归一化缺陷，
再让“discretization 不够”和“solver 不可扩展”各自接受可证伪测试。本节追加前文档为 `226698 bytes`，SHA256 为
`FF253F5DD9F002724FD4459EDA1516B88E529D923BED1976DB7317992E472627`。

---

## 19. 冻结决定与恢复条件

自 `2026-08-17` 起，exp040 的工作状态为 `Frozen / Paused`，整体科学状态仍为 `Inconclusive`。冻结原因是近期
优先启动 exp050 复原研究，之后再考虑更高级的电磁学验证和 reference validation。该决定不把 exp040 标记为
`Deprecated`，也不改变 R4--R7 等局部 `Passed`、R14B 最新 `Failed` 或任何历史预注册与正式结果。

冻结期间遵守以下边界：

1. 当前不继续 R15，不新增求解器、forward model 或电磁学验证，也不重跑、覆盖或重解释 R14B。
2. exp050 可以把冻结的 exp040 forward 当作**同模型、自洽复原**的数值基线，用于检验在生成模型与复原模型
   一致时的 inverse/reconstruction 行为。
3. 上述用途不表示 exp040 已通过高保真电磁物理验证；当前仍为
   `reference_validated=false`、`full_tgv_reference_authorized=false`，不能把 self-consistency 当作真实物理准确性。

未来若恢复高级物理验证，必须先重新定义研究问题、reference 身份、solver 路线和验收门槛，再决定是否继续
exp040。若恢复工作改变核心 forward model、物理假设或主要验收目标，应按仓库实验编号规则考虑新建实验编号，
而不是用新模型追溯覆盖 exp040 的冻结证据。
