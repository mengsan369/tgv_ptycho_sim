# exp030：TGV 2D projected-phase probe observability

## 0. 当前状态与文档阅读说明

当前最终状态：`Passed`

最终审阅 run：

```text
runs/exp030_TGV_2d_effective_phase_20260810_121124/
```

当前结论：

- exp030 的单孔、无噪声 projected-phase Stage A--D 已按本实验预设门限通过。
- 修正 band-limited ASM 下 reconstruction residual 与 adjoint operator 的一致性后，blind reconstruction 的 probe error 随迭代继续下降。
- matched baseline、waist-minus 和 waist-plus 在相同 reconstruction 条件下运行 200 iterations 后，恢复得到的 normalized probe sensitivity 与 truth 的相对偏差为 `3.55915e-7`，且腰径变化的排序与 truth 一致。
- 该结果支持理想二维 projected-phase 模型下的局部差分 observability 和 paired blind recovery，不等价于真实三维 TGV 腰径已经能够准确测量。

已知限制：

- 当前结果基于单孔、无噪声、理想 projected-phase 模型和相同 optimizer 设置。
- 当前使用的扫描、边界、采样和样品 B 条件仍属于阶段性仿真假设。
- baseline 运行到 1000 iterations 后，absolute frozen detector-amplitude loss 仍约为真实 detector case separation 的 `4.18` 倍。
- 当前通过的是 matched differential sensitivity gate；它依赖三组 reconstruction 的 common-mode bias 抵消，不表示每个独立 blind solution 的绝对 detector residual 已低于腰径信号。
- 本实验不能代替三维 multislice、高保真电磁传播、噪声、标定误差或真实实验条件下的验证。

文档结构说明：

- 第 1--11 节：exp030 初次构建、预注册条件和第一次完整实验记录。
- 第 12 节：A 到 B 连续径向传播、固定 detector sampling 和首次 Stage D 重跑。
- 第 13 节：operator-consistency 消融、问题定位和约束比较。
- 第 14 节：ePIE residual/adjoint 修正及 optimizer 长程控制实验。
- 第 15 节：200/500/1000 长轨迹、matched cases、checkpoint 和 Stage D 最终完成记录。
- 历史章节保留实验进行到当时的判断，不追溯修改。
- 因此第 1 节中的 `Inconclusive` 是第一次实验记录形成时的历史状态，不是 exp030 的当前最终状态。
- 阅读当前结论时，建议先阅读本节和第 15 节；需要追溯问题如何被定位和修正时，再依次阅读第 12--14 节。

## 1. 先用一句话说明这个实验

exp030 要回答的是：

> 在一个理想、无噪声的二维 projected-phase 模型中，单个 TGV 的腰径 $D_\mathrm{waist}$ 发生小变化后，这个变化是否会稳定地出现在 B 平面复 probe 和最终 detector intensity 中，并且能否与表面孔径、折射率相位尺度等其他参数的影响区分开？

这一步放在完整三维 TGV 仿真之前。原因是：如果在最简单的理想模型里，$D_\mathrm{waist}$ 已经没有独立信息，直接进入昂贵的 3D multislice 和 blind reconstruction 没有意义；反过来，即使 exp030 有灵敏度，也只能说明这个二维近似具有信息，不能证明真实三维 TGV 已经可测。

当前状态：`Inconclusive`。

最终审阅 run：

```text
runs/exp030_TGV_2d_effective_phase_20260804_204748/
```

状态为 `Inconclusive` 的直接原因不是没有信号，而是横向采样 $dx$ 尚未收敛。这个原因会在第 6 节解释。

## 2. 这个实验在整个测量链中的位置

实验中的传播链是：

```text
参数化单个 TGV 的 D(z)
        ↓
沿 z 积分，得到二维 A_effective_true
        ↓
单位平行光通过样品 A
        ↓ 传播 z_AB
得到 B 平面 probe P_B_true
        ↓
probe 照明可移动编码样品 B
        ↓ B 做 7×7 overlapping scan
每个位置传播 z_BC
        ↓
得到 49 帧 detector intensity
        ↓
比较不同 D_waist，并在条件满足时尝试 blind ePIE recovery
```

这里的“样品 A”是 TGV 的二维近似，“样品 B”是用于 ptychography 的已仿真随机相位编码样品。B 不是 TGV 的一部分。

为什么需要样品 B 和扫描？detector 只能测强度，不能直接测复场相位。B 的空间结构和 overlapping scan 会把 probe 中的相位变化转换为多帧强度变化，同时提供 ptychographic redundancy。exp020 已验证过这条理想 recovery 思路；exp030 把 exp020 的平滑随机 A 换成由 TGV 几何生成的 A。

## 3. exp030 使用的模型是什么

### 3.1 三维几何仍然由 $D(z)$ 定义

玻璃中只有一个与传播方向 $z$ 同轴的空气孔。入口、腰部和出口直径分别为

$$
D_\mathrm{top},\qquad D_\mathrm{waist},\qquad D_\mathrm{bottom}.
$$

它们通过上下两段线性轮廓组成 $D(z)$。二维和三维代码共用 `src/tgv_ptycho/objects/tgv_geometry.py` 中的同一个 `diameter_profile()`，因此 projected model 与未来 multislice 不会各自维护一套几何公式。

### 3.2 三维孔怎样变成二维对象

对每个横向位置 $(x,y)$，先计算光线沿 $z$ 方向穿过空气孔的总长度：

$$
\ell_\mathrm{air}(x,y)
=
\int_0^L
\mathbf 1\!\left[r(x,y)\le \frac{D(z)}{2}\right]\,\mathrm dz.
$$

然后相对于纯玻璃参考计算相对光程和相位：

$$
\operatorname{OPD}_\mathrm{rel}(x,y)
=
(n_\mathrm{air}-n_\mathrm{glass})\ell_\mathrm{air}(x,y),
$$

$$
\phi_\mathrm{unwrapped}(x,y)
=
\frac{2\pi}{\lambda}\operatorname{OPD}_\mathrm{rel}(x,y).
$$

最终二维复透过函数是

$$
A_\mathrm{effective}(x,y)
=
\exp\!\left[i\phi_\mathrm{unwrapped}(x,y)\right].
$$

为什么要同时保存 path、OPD、unwrapped phase 和 complex transmission？因为复数相位会按 $2\pi$ 包裹，仅保存 `A_effective_true` 会丢失累计光程。前三个量用于检查物理路径，最后一个量用于波传播。

### 3.3 这个模型故意忽略什么

projected-phase 假设每条光线在样品内部一直沿 $z$ 直线传播，只累计局部相位。它忽略：

- 样品内部横向衍射；
- TGV 侧壁折射；
- 玻璃/空气界面反射；
- 多次反射和多次散射；
- 偏振、粗糙度、倾斜、偏心和非圆孔。

因此 `A_effective_true` 是给定波长、参考玻璃和传播方向后的二维近似，不是严格的三维电磁透过函数。更完整的推导见 `docs/theory_notes/tgv_effective_phase_2d.md`。

## 4. 按实验代码顺序说明每一步

本节严格对应 `scripts/run_exp030_effective_phase.py` 中 `run()` 的执行顺序。

### 4.1 读取配置并创建独立 run

代码首先读取 YAML，提取光学、TGV、灵敏度、采样和重建参数，然后立刻调用 `make_run_dir()` 创建新的 timestamped run。

为什么先创建 run？实验即使中途失败，也需要留下独立诊断目录；同时避免覆盖任何历史结果。

代码还检查：

- shape 必须是正的 `(ny,nx)`；
- $\Delta D_\mathrm{waist}$ 序列必须至少有两级；
- 主步长必须等于序列中最细一级；
- minus/plus 腰径不能违反 $0<D_\mathrm{waist}\le\min(D_\mathrm{top},D_\mathrm{bottom})$。

这些检查的目的，是在昂贵传播前拒绝无效几何或含义不一致的配置。

### 4.2 生成三项所有 case 共用的输入

代码生成：

1. 单位振幅平行入射场 `incident`；
2. 固定 seed 的随机纯相位样品 `sample_b`；
3. 固定 seed 的 $7\times7$ jittered scan positions。

为什么所有腰径 case 必须共用同一 B、scan 和 seeds？这样 detector 差异的唯一来源才是 $D_\mathrm{waist}$。如果每个 case 重新随机 B 或位置，随机差异会混入腰径灵敏度。

当前 scan 必须在 baseline 和 fine 网格上都对应整数像素位移，因为 forward/reconstruction 仍使用 `np.roll`。这是阶段性限制，不是最终硬件模型。

### 4.3 建立 waist-minus、baseline、waist-plus 三个 case

代码设置

$$
D_-=D_0-\Delta D,\qquad
D_0=D_\mathrm{waist},\qquad
D_+=D_0+\Delta D.
$$

对三个 case 分别调用 `_projected_model()`，得到：

| 数组 | shape / dtype | 物理意义 |
|---|---|---|
| `fill_path_length_m` | `(ny,nx)`, float64 | 每条轴向光线穿过空气的长度，m |
| `opd_relative_m` | `(ny,nx)`, float64 | 相对玻璃的 OPD，m |
| `phase_unwrapped_rad` | `(ny,nx)`, float64 | 未包裹相位，rad |
| `A_effective_true` | `(ny,nx)`, complex128 | 用于传播的二维复透过函数 |
| `z_m` | `(nz,)`, float64 | midpoint 层位置，m |
| `diameter_z_m` | `(nz,)`, float64 | 各层的 $D(z)$，m |

为什么用三个 case？两个对称扰动可以构成中心有限差分，比单边差分更能抵消一阶截断偏差；中间 case 同时是所有相对误差的 reference。

### 4.4 对三个 case 执行相同 forward model

对每个 case，代码依次做：

$$
U_A=A_\mathrm{effective}\,U_\mathrm{incident},
$$

$$
P_B=\mathcal H_{z_{AB}}(U_A),
$$

以及对每个 scan position $j$：

$$
I_j
=
\left|
\mathcal H_{z_{BC}}
\left[P_B\,B_j\right]
\right|^2.
$$

$\mathcal H_z$ 表示 angular-spectrum propagation，$B_j$ 表示移动到第 $j$ 个位置的编码样品 B。

这一阶段同时得到两类观测对象：

- `P_B_true`：理想情况下在 B 平面存在的复 probe；
- `I_stack_true`：实验真正可能记录的 detector intensity stack。

必须分别分析它们。复场差异存在，并不自动代表强度探测器仍能看到该差异。

### 4.5 Stage A：先验证 projected model 本身

代码已经生成三个 case，但在解释灵敏度前，先运行 `_model_validation()`。

| 控制检查 | 做什么 | 为什么需要 |
|---|---|---|
| diameter endpoints | 检查 $D(0)$、$D(z_w)$、$D(L)$ | 防止共享几何公式或单位错误 |
| analytic path | midpoint path 与解析空气路径比较 | 区分几何积分误差和后续传播差异 |
| cylinder control | 令三个直径相等，与 thin phase disk 比较 | 验证模型能正确退化到已知简单情况 |
| zero contrast | 令 $n_\mathrm{air}=n_\mathrm{glass}$ | 没有折射率差时必须严格得到 $T=1$ |
| reference region | 检查孔外玻璃区 | 孔外不应凭空出现相位结构 |
| pure phase | 检查 $\lvert A_\mathrm{effective}\rvert=1$ | 无吸收配置不应产生振幅吸收 |

为什么 Stage A 不通过就不能谈灵敏度？如果简单解析控制都不成立，minus/plus 的差异可能只是实现错误。

### 4.6 Stage B：计算 true-probe sensitivity

中心有限差分为

$$
\frac{\partial P_B}{\partial D_\mathrm{waist}}
\approx
\frac{P_B(D_+)-P_B(D_-)}{2\Delta D}.
$$

代码在计算范数前去除 global complex-scale tangent。原因是 ptychographic complex field 存在整体复数尺度自由度：所有像素同时乘同一个相位或复增益，并不是 TGV 几何的空间特征。

归一化 probe sensitivity 是

$$
S_P
=
D_0
\frac{
\left\|
\Pi_\mathrm{gauge}
\left[\partial P_B/\partial D_\mathrm{waist}\right]
\right\|_2
}{\|P_B(D_0)\|_2}.
$$

此处

$$

v=\frac{\partial P_B}{\partial D_{\mathrm{waist}}}.

$$

$$

\Pi_{\mathrm{gauge}}(v)
=
v-
\frac{\langle P_B,v\rangle}
{\langle P_B,P_B\rangle}P_B.
$$

其中第二项就是$v$中与原 $probe\ P_B$ 平行的部分，也就是“整个 probe 统一乘以一个复数”造成的变化。

乘以名义腰径 $D_0$ 后，指标无量纲，便于不同参数尺度之间比较。它表示局部相对响应强度，不是“可检测的最小腰径”。

代码还保存 amplitude、wrapped phase、gauge-aligned complex L2 和 derivative maps，因为单个总范数无法说明差异位于哪里、来自振幅还是相位。

### 4.7 Stage C：计算 detector intensity sensitivity

强度使用同样的中心有限差分：

$$
S_I
=
D_0
\frac{
\left\|
[I(D_+)-I(D_-)]/(2\Delta D)
\right\|_2
}{\|I(D_0)\|_2}.
$$

为什么 intensity 不做 complex gauge alignment？intensity 已经是实数非负测量，整体复相位本来就不存在于数据中。

除了整个 stack 的指标，代码还计算每一帧 sensitivity、最大值、中位数和 baseline frame energy consistency。这样可以判断差异是多数扫描位置都存在，还是只由个别异常帧主导。

### 4.8 检查 $dz$ convergence

代码保持横向网格、B、scan 和所有物理参数不变，只把

$$
dz:1.0\,\mathrm{nm}\rightarrow0.5\,\mathrm{nm}.
$$

为什么做这个检查？`midpoint` 方法沿 $z$ 把空气路径离散成薄层。若 $dz$ 太大，孔壁穿过某一层的位置会被错误归到整层空气或整层玻璃，造成假的腰径响应。注意：这里意味着我们还暂时采用离散的模型来直接对TGV的复透过率进行计算，不是用的积分的解析的解，而是采取离散求和的方法。

### 4.9 检查 $\Delta D$ step convergence

代码在同一网格上依次使用

$$
1000,\ 125,\ 15.625,\ 3.90625,\ 1.953125,\ 0.9765625,\ 0.48828125\ \mathrm{nm}
$$

作为中心差分步长，并观察 $S_P$、$S_I$ 是否趋于稳定。

为什么需要这个检查？

- 步长太大时，算到的是两个相距很远模型之间的平均变化，不是局部导数；
- 步长太小时，浮点误差和栅格边界跳变可能占主导；
- 只有中间出现稳定平台时，finite difference 才可信。

相邻两级指标变化定义为

$$
\varepsilon_\mathrm{step}
=
\frac{\lvert S_\mathrm{coarse}-S_\mathrm{fine}\rvert}
{\lvert S_\mathrm{fine}\rvert}.
$$

### 4.10 检查 $dx$ convergence

代码建立第二套横向网格：

```text
baseline: 384×384, dx=0.25 μm
fine:     768×768, dx=0.125 μm
```

两者都覆盖 $96\,\mu\mathrm m$ FOV。sample B 按整数倍展开，scan positions 保持同一物理坐标。

为什么必须保持相同 FOV？如果同时改变 FOV 和 $dx$，结果变化可能来自边界截断，而不是采样间隔。这里要单独隔离 $dx$ 的影响。

为什么 fine 网格还要重新做完整的 $\Delta D$ step sweep？仅比较某一个步长无法区分“网格没收敛”和“有限差分步长没收敛”。两套网格必须分别先进入自己的步长平台，再比较平台是否一致。

`lateral_supersampling` 只在每个输出像素内部用更多子像素估计空气路径。它能改善圆边界的 fractional coverage，但最终仍只输出 $384^2$ 或 $768^2$ 个 complex samples，因此不能替代减小输出 $dx$。

### 4.11 用解析相位周期解释可能的欠采样

代码根据对称腰形孔的解析 path slope 计算过渡环中最短相位周期：

$$
\Lambda_r
=
\frac{\lambda}
{\lvert n_\mathrm{air}-n_\mathrm{glass}\rvert
\left\lvert\mathrm d\ell_\mathrm{air}/\mathrm dr\right\rvert}.
$$

对应的 Nyquist 条件是

$$
dx\le\frac{\Lambda_r}{2}.
$$

这个诊断不是额外的“可测性判据”，而是当 $dx$ convergence 失败时，用来判断失败是否与 projected phase 中真实存在的高空间频率一致。

### 4.12 构建 local-observability Jacobian

仅证明 $D_\mathrm{waist}$ 改变会让 probe 改变还不够，因为其他参数也可能产生几乎相同的场变化。代码因此额外扰动：

1. `d_waist`；
2. `common_surface_diameter`：同时改变 $D_\mathrm{top}$ 和 $D_\mathrm{bottom}$；
3. `phase_scale`：代表整体折射率差或相位尺度变化。

每一列都是一个 probe derivative。代码先去除 global complex-scale gauge，再乘以参数名义尺度，并把 complex derivative 的 real/imag 部分拼成实 Jacobian。

为什么要乘名义尺度？直径的单位是米，而 `phase_scale` 无量纲；不归一化时，Jacobian 条件数会被单位选择支配。

最后检查：

- singular values 和 numerical rank：是否存在完全不可区分的参数方向；
- condition number：最弱参数组合相对最强组合有多弱；
- normalized column correlation：两个参数是否留下相似空间特征。

这比只看两幅 probe 的 complex L2 更接近“局部可辨识性”问题。

### 4.13 通过 gate 后才允许执行 Stage D

代码先检查 analytic controls，再要求以下三个最大相对变化都小于 $5\%$：

```text
dz convergence
dx convergence
finite-difference-step convergence（其实就是4.9所描述的，检查导数是否收敛）
```

为什么把 Stage D 放在 gate 后？blind ePIE 可能很好地拟合一组数值上错误或混叠的数据。如果 forward sensitivity 尚未收敛，reconstruction 成功只会证明算法能重建离散伪影，不能证明它恢复了腰径信息。

如果 gate 通过，代码才会对 minus、baseline、plus 三个 case 分别运行 constrained blind ePIE。重建初始化只使用 detector data 和固定随机 seed，不读取 truth。truth alignment 只写到 `simulation_evaluation_only`，raw reconstruction 单独保存。

### 4.14 最后保存 figures、JSON 和 HDF5

代码最后才绘图和写文件。这样 metrics、图像和 HDF5 都来自同一批内存结果。

- `config.yaml`：本次实际使用的配置；
- `metadata.json`：run、坐标顺序、模型边界和 truth-use 声明；
- `metrics.json`：便于人工快速阅读的全部标量和小数组；
- `outputs/exp030_effective_phase.h5`：机器读取的场、intensity、truth、metrics；
- `figures/`：人工检查，不作为后续计算输入。

## 5. 本次实际配置

| 参数 | 数值 | 为什么这样设置 |
|---|---:|---|
| wavelength | $532\,\mathrm{nm}$ | 暂时复用 exp020 的理想可见光设置 |
| thickness | $700\,\mu\mathrm m$ | 当前代表性玻璃厚度 |
| $D_\mathrm{top}/D_\mathrm{waist}/D_\mathrm{bottom}$ | $50/33.3/50\,\mu\mathrm m$ | 对称单腰 TGV，开口/腰径约 1.5 |
| $z_\mathrm{waist}$ | $350\,\mu\mathrm m$ | 第一版对称模型 |
| $n_\mathrm{glass}/n_\mathrm{air}$ | $1.5/1.0$ | 无吸收理想材料 |
| baseline grid | $384^2$, $dx=0.25\,\mu\mathrm m$ | $96\,\mu\mathrm m$ FOV，包含孔和玻璃参考区 |
| fine grid | $768^2$, $dx=0.125\,\mu\mathrm m$ | 相同 FOV 下检查 lateral convergence |
| baseline/fine $dz$ | $1.0/0.5\,\mathrm{nm}$ | 检查 midpoint path integration |
| baseline/fine supersampling | 4 / 2 | 两套网格均使用 $62.5\,\mathrm{nm}$ 子像素间隔 |
| $z_{AB}/z_{BC}$ | $1/1\,\mathrm{mm}$ | 暂时复用 exp020 |
| scan | $7\times7$ | 49 帧 overlapping scan |
| noise | none | 先隔离理想数学灵敏度 |

这些参数仍是阶段性基线，不是已经校准的真实仪器参数。

## 6. 实际结果，以及为什么状态仍是 Inconclusive

### 6.1 projected model 的基础控制通过

| 指标 | 结果 | 解读 |
|---|---:|---|
| diameter profile max error | $0$ | 几何端点正确 |
| analytic path max error | $0.513231\,\mathrm{nm}$ | 小于 baseline $dz=1\,\mathrm{nm}$ |
| analytic path RMSE | $0.049472\,\mathrm{nm}$ | midpoint path 与解析解一致 |
| cylinder transmission error | $0$ | 正确退化为 phase disk |
| zero-contrast error | $0$ | 无折射率差时严格 $T=1$ |
| reference-region error | $0$ | 孔外玻璃区严格 $T=1$ |
| pure-phase amplitude error | $2.22\times10^{-16}$ | 只有浮点精度误差 |

所以当前失败不是基础几何或 OPD 公式写错。

### 6.2 $dz$ 和 finite-difference step 已经收敛

| 检查 | 最大相对变化 | 5% gate |
|---|---:|---|
| $dz:1.0\rightarrow0.5\,\mathrm{nm}$ | $0.074118\%$ | 通过 |
| 最后两级 $\Delta D$ | $0.472540\%$ | 通过 |

旧的 $\Delta D=1\,\mu\mathrm m$ 得到的是较大区间差异，不是局部导数。步长缩小到 $0.48828125\,\mathrm{nm}$ 后，两套横向网格内部都进入了稳定平台。

### 6.3 但两套 $dx$ 网格收敛到不同平台

| 指标 | $dx=0.25\,\mu\mathrm m$ | $dx=0.125\,\mu\mathrm m$ | 相对变化 |
|---|---:|---:|---:|
| normalized probe sensitivity | 1222.149378 | 701.630930 | $74.1869\%$ |
| normalized intensity sensitivity | 1338.744952 | 787.010399 | $70.1051\%$ |

这说明“有数值响应”是可复现的，但响应强度仍被输出网格显著改变，不能把某一个数值解释为已经采样收敛的物理灵敏度。

### 6.4 横向失败的原因可以由解析相位周期解释

当前几何的过渡环 path slope 为

$$
\left\lvert\frac{\mathrm d\ell_\mathrm{air}}{\mathrm dr}\right\rvert
=83.832335.
$$

对应 projected-phase 径向周期只有

$$
\Lambda_r=12.692\,\mathrm{nm},
$$

所以仅按 Nyquist 条件就要求

$$
dx\le6.346\,\mathrm{nm}.
$$

当前最细输出 $dx=125\,\mathrm{nm}$，仍比这个要求粗约 $19.697$ 倍。也就是说，700 µm 厚度和 $\lvert\Delta n\rvert=0.5$ 在 projected model 中积累了非常陡的相位坡度。输出 complex field 在进入传播前就已经混叠；增加像素内 supersampling 无法补回这些丢失的空间频率。

因此 $dx$ convergence 失败不是简单代码 bug，也不能通过放宽 5% 阈值解决。

### 6.5 detector 确实保留差异，但仍受同一采样问题影响

baseline 网格中：

- minus/baseline intensity relative L2：0.0196515；
- plus/baseline intensity relative L2：0.0196563；
- 49 帧 sensitivity 中位数：1337.889940；
- 最大值：1354.252119。

因此 B 和 scan 确实把 probe 变化转换成了 detector intensity 变化；但 fine grid 上的归一化 intensity sensitivity 降到 787.010399，所以还不能称为采样稳定的 detector observability。

### 6.6 Jacobian 在当前网格满秩，但结论仍受限

| 指标 | 结果 |
|---|---:|
| singular values | [458844.7462, 9324.4762, 4087.5655] |
| smallest/largest | 0.00890838 |
| condition number | 112.2538 |
| numerical rank | 3 |
| $D_\mathrm{waist}$ 与其他列最大绝对相关 | 0.0861910 |

当前离散网格中，三个参数列没有完全重合；但 Jacobian 没有做 fine-grid convergence。它只能说明“当前离散模型没有显式秩亏”，不能证明真实系统中的 $D_\mathrm{waist}$ 已独立可辨识。

### 6.7 Stage D 按设计没有执行

记录状态为：

```text
status: not_run_sampling_or_step_convergence_gate_failed
executed: false
```

HDF5 中没有 `/entry/reconstruction`，也没有伪造的 recovered probe 或 loss curve。exp030 因此尚未回答 recovery 后是否仍保留腰径排序。

## 7. 建议怎样阅读这个 run

推荐按以下顺序查看，而不是一开始就翻 HDF5：

1. `metrics.json`：先确认 `experiment_status` 和三个 convergence change；
2. `figures/fill_path_radial_profile.png`：确认解析空气路径形状；
3. `figures/opd_and_unwrapped_phase.png`：理解高累计相位从哪里来；
4. `figures/effective_transmission.png`：查看包裹后的二维 transmission；
5. `figures/delta_d_step_convergence.png`：确认两套网格各自进入步长平台；
6. `figures/probe_sensitivity_maps.png`：查看 probe 中的空间差异和高频纹理；
7. `figures/intensity_sensitivity.png`：查看 detector 是否保留差异；
8. `figures/jacobian_correlation.png` 与 `jacobian_singular_values.png`：查看参数退化；
9. 最后才用 HDF5 读取完整数值数组。

## 8. HDF5 如何对应前面的实验步骤

| 实验步骤 | HDF5 group | 主要 shape / dtype |
|---|---|---|
| baseline detector data | `/entry/data` | `I_stack (49,384,384)` float64；positions `(49,2)` float64, m |
| 光学配置 | `/entry/instrument` | wavelength、dx、$z_{AB}$、$z_{BC}$ 等 scalar float64 |
| TGV/B 参数 | `/entry/sample` | 单孔几何、projected model、sample B 参数 |
| baseline projected truth | `/entry/truth` | path/OPD/phase `(384,384)` float64；fields `(384,384)` complex128 |
| 三个 waist cases | `/entry/truth/parameter_sweep` | fields `(3,384,384)`；intensity `(3,49,384,384)` |
| controls 和 sensitivity | `/entry/metrics` | model、probe、intensity、convergence、Jacobian |
| reconstruction | 不存在 | Stage D 未通过 gate，没有空 group |

`truth` 只用于仿真评估和画图，没有被 blind reconstruction 用作先验修正。

## 9. 代码文件各自负责什么

| 文件 | 职责 |
|---|---|
| `configs/experiments/exp030_TGV_2d_effective_phase.yaml` | 所有实验参数和随机 seed |
| `scripts/run_exp030_effective_phase.py` | 按本文件第 4 节的顺序编排实验 |
| `src/tgv_ptycho/objects/tgv_geometry.py` | 公共 $D(z)$、解析空气路径、几何验证 |
| `src/tgv_ptycho/objects/tgv2d.py` | 从三维参数生成二维 projected phase |
| `src/tgv_ptycho/objects/tgv3d.py` | 复用同一几何生成 3D refractive-index volume |
| `src/tgv_ptycho/forward/scheme_probe_B.py` | A→B→detector forward model |
| `src/tgv_ptycho/inverse/observability.py` | gauge alignment、finite difference、Jacobian |
| `src/tgv_ptycho/viz/plot_tgv.py` | exp030 figures |
| `tests/test_tgv_geometry.py` | 几何、解析控制和兼容性测试 |
| `tests/test_exp030_observability.py` | sensitivity、gauge、Jacobian 测试 |
| `tests/test_hdf5_layout.py` | baseline、sweep 和条件 reconstruction layout |

## 10. 复现记录

运行命令：

```powershell
D:\anaconda3\envs\tgv_ptycho_sim\python.exe scripts/run_exp030_effective_phase.py --config configs/experiments/exp030_TGV_2d_effective_phase.yaml
```

最终验证：

- full pytest：`39 passed in 1.28s`；
- exp030 修改范围 Ruff：`All checks passed!`；
- 项目级 Ruff 仍有 13 个与 exp030 无关的既有问题，本实验没有顺手修改。

exp020 文档曾记录不存在的 `scripts/run_full_pipeline.py`；当前真实入口是 `scripts/run_exp020_probe_recovery.py`。exp030 只复用其公共 pipeline，没有重命名历史入口。

## 11. 结论边界与下一步

exp030 当前可以确认：

- 参数化单孔几何和 projected path 计算正确；
- $D_\mathrm{waist}$ 在当前离散模型中会改变 true probe；
- 编码样品 B 和 scan 会把该差异转换成 detector intensity 差异；
- $dz$ 与 finite-difference step 已收敛；
- 当前 $dx$ 没有收敛，因此灵敏度绝对数值和 Jacobian 结论仍不可靠；
- Stage D 未执行，recovery 后是否保留腰径信息仍未知。

继续 exp030 时，优先研究“高分辨率局部 projected transmission + 明确低通或 detector pixel integration + 传播网格重采样”，建立与仪器带宽一致的 forward model；等 $dx$、$dz$ 和 step gate 都通过后，再运行已经实现的 Stage D。

如果研究多个孔及孔间相干作用，应新开 exp031。若进入包含样品内部传播的 3D 模型，应新开 exp040。noise、tilt、stage error 或 parametric fitting 会改变误差模型或验收目标，也应作为后续独立任务。

---

## 12. 2026-08-06：消除 A 平面混叠并固定 detector 后的修正记录

本节只记录本轮修正。前面的章节保留了旧 run 的原始判断，便于追踪为什么要改；本节中的新结果来自：

```text
E:\tgv_ptycho_sim\runs\exp030_TGV_2d_effective_phase_20260806_185142
```

本次完整执行用时约 29 分钟，Stage D 实际执行，没有覆盖任何历史 run。

### 12.1 旧实现为什么不能收敛

旧实现先把只有约 $12.692\,\mathrm{nm}$ 周期的径向复相位条纹采样到 $125$--$250\,\mathrm{nm}$ 的二维数组，再做 FFT 传播。此时高频已经折叠成低频( 也就是发生了采样后的混叠，12.962 nm的周期空间频率约为$f_\mathrm{true}=\frac1{0.012692}\approx78.79\ \mathrm{cycles/\mu m}$ ,而采样频率 $f_s=\frac1{dx}=8\ \mathrm{cycles/\mu m}$ ,发生了混叠,导致这个高频量混到了低频中)；后续再删除 evanescent frequency( 也就是超过光波长对应的截止频率的部分 $f_\mathrm{cutoff}=\frac1{0.532}\approx1.88\ \mathrm{cycles/\mu m}$ )，无法判断某个低频分量原来是真实低还是 alias。因此，旧的归一化 probe sensitivity 约 $701$--$1222$，主要是离散混叠，不应解释为物理灵敏度。

同时，旧 fine-grid detector 是 $768\times768$ 个、间距 $125\,\mathrm{nm}$ 的点采样，却直接和 $384\times384$、间距 $250\,\mathrm{nm}$ 的 baseline 点采样比较。配置中的 `detector_pixel_size_m` 当时只写入 metadata 和绘图，没有进入 detector forward，所以两边并不是同一个物理 detector。

### 12.2 本轮采用的 A 到 B 修正

当前实验限定为单个、轴对称、正入射 TGV，并且原假设已经是标量近轴模型。因此，脚本现在把投影透过函数拆成无限平面波参考和紧支撑扰动：

$$
q(r)=T_{\mathrm{proj}}(r)-1.
$$

只对 $q(r)$ 做连续的零阶 Fresnel--Hankel 积分，平面波参考单独解析传播：

$$
P_B(\rho,z)=A_0 e^{ikz}+A_0 e^{ikz}e^{ik\rho^2/(2z)}\frac{2\pi}{i\lambda_m z}\int_0^{R_{\max}}q(r)e^{ikr^2/(2z)}J_0\!\left(\frac{k\rho r}{z}\right)r\,\mathrm dr,$$

其中：

$$
\lambda_m=\frac{\lambda_0}{n_{\mathrm{medium}}},
\qquad
k=\frac{2\pi}{\lambda_m}.
$$

这样先在高分辨率一维径向积分中处理快速相位，再把已经传播后的平滑径向场采样到二维 B 平面。高频不会先在粗二维 A 网格中折叠成伪低频。无限的 $T=1$ 背景没有被错误截断成有限圆孔（这句话很重要，因为如果没有这个条件，关注区域总是有限的，关注区域内的背景是1，此外数组没有包括的地方直接就被认为赋值成0了这显然不合理，相当于凭空加了个瞳孔导致了衍射，所以算法上要把投影透过函数拆开）。

径向积分采用复合 midpoint rule：中央恒定相位平台用较疏节点，快速变化的 taper annulus 使用 $1\,\mathrm{nm}$ 节点；独立的 fine control 使用 $0.5\,\mathrm{nm}$。（这里很重要，在计算上面那个有关q的路径积分时，为了将混叠降低，采用了很小的网格来计算透过率函数）输出径向表间距为 $62.5\,\mathrm{nm}$，再插值到 $dx=0.25\,\mu\mathrm m$ 和 $0.125\,\mu\mathrm m$ 两个相同 FOV 的网格。

可以总结一下这个实验里有的网格：

---
入射光

 $A_0(x,y)=1$ ,代码会保存一个
$384\times384=147456$个复数的入射场数组。
但在权威 $A\rightarrow B$ 计算中，平面波背景不需要逐像素传播，而是直接解析计算：
$P_{\mathrm{reference}}=A_0e^{ikz}$
所以这里真正进入径向传播公式的只是标量  $A_0=1$ ，没有进行FFT。

---
A平面径向透过率

真正使用的不是二维 A 网格，而是一维半径数组,网格分成两段,总计9674个径向节点

|径向区域|大间隔|节点数量|
|---|---|---|
|$0\sim16.15\,\mu\mathrm m$|$50\,\mathrm{nm}$|324|
|$16.15\sim25.5\,\mu\mathrm m$|$1\,\mathrm{nm}$|9350|

对每一个 $r_j$，代码直接计算：

$$
\ell_{\mathrm{air}}(r_j)
\rightarrow
\phi(r_j)
\rightarrow
T(r_j)=e^{i\phi(r_j)}
\rightarrow
q(r_j)=T(r_j)-1.
$$

---
Fresnel–Hankel 积分

对于每一个 B 平面径向位置 $\rho_l$，计算

$$P_B(\rho_l)=
A_0e^{ikz}
+
C(\rho_l)
\sum_{j=0}^{9673}
q(r_j)
e^{ikr_j^2/(2z)}
J_0\!\left(\frac{k\rho_l r_j}{z}\right)
r_j\Delta r_j$$

B 平面径向坐标使用 $\Delta\rho=62.5\,\mathrm{nm}$,为了覆盖二维网格的最远角点，需要计算到约
$\rho_{\max}=67.875\,\mu\mathrm m$,
因此一共有1087 个 B 平面径向节点.

---
一维 $P_B(\rho)$ 转成二维 $P_B(x,y)$

用上一步得到的径向数值采样插值即可，目前设定为baseline 384×384 nm，dx为250nm；fine conntrol 768×768，dx 为125nm。

### 12.3 固定物理 detector 的修正

fine-grid irradiance 在和 baseline 比较前，先积分回固定的 $250\,\mathrm{nm}$ detector pixel，这里是为了改之前的探测器上的pixel不一样大的问题，必须先控制了变量，把pixel弄一样大了。对于本配置，fine-grid bin factor 为 $2$（意思是传播到探测器平面时，先用fine的数值来算，然后再加和到探测器像素上）：

$$
I^{\mathrm{pixel}}_{mn}
=
\frac{1}{4}
\sum_{a=0}^{1}
\sum_{b=0}^{1}
I^{\mathrm{fine}}_{2m+a,\,2n+b}.
$$

这里保存均值而不是求和，因为它表示 pixel-average irradiance（平均强度）；若要得到 pixel power，还需乘同一个固定 detector pixel area（其实和直接加起来一样，方便保存和节省空间吧）。baseline bin factor 为 $1$，fine bin factor 为 $2$，（也就是基础的250nm的网格和125nm的网格），两边最终均为 `(49, 384, 384)`。B、scan position 和所有 seed 保持相同。

脚本还预先构造一次 B 到 C 的 angular-spectrum transfer，供同一网格上的所有 frame 复用。这只减少重复计算，没有改变原有 propagating-wave cutoff、周期 FFT 和 `np.roll` 边界语义。

### 12.4 新增的自动对照和测试

新增或扩展的测试包括：

- detector block average 对常数场和已知 $2\times2$ block 的结果正确；
- zero contrast 时 Fresnel--Hankel 严格退化为传播后的平面波；
- 平滑 radial Gaussian 扰动与 Fresnel 积分的闭式解一致；
- 原有几何、observability 和 HDF5 layout 回归继续通过。

最终测试结果：

```text
42 passed in 1.53s
```

本轮修改范围 Ruff：

```text
All checks passed!
```

项目级 Ruff 诊断仍有 13 个既有问题，位于 `run_exp001_forward.py`、`calibration/stage.py`、用户已有修改的 `angular_spectrum.py`、`recon/losses.py` 和 `recon/rpie.py`；本轮没有顺手修改。

### 12.5 projected model 和 sampling gate 的新结果

| 检查 | 新结果 | 5% gate | 解释 |
|---|---:|---:|---|
| analytic path max error | $0.513231\,\mathrm{nm}$ | 通过 | 小于 baseline $dz=1\,\mathrm{nm}$ |
| analytic path RMSE | $0.049472\,\mathrm{nm}$ | 通过 | midpoint path 与解析 path 一致 |
| zero contrast transmission error | $0$ | 通过 | 严格 $T=1$ |
| Fresnel plane-wave control | $0$ | 通过 | $q=0$ 时严格为 $A_0e^{ikz}$ |
| reference-region error | $0$ | 通过 | 孔外严格为玻璃参考 |
| pure-phase amplitude error | $2.22\times10^{-16}$ | 通过 | float64 精度量级 |
| raster path/transmission $dz$ change | $0.035870\%$ | 通过 | $1.0\rightarrow0.5\,\mathrm{nm}$ |
| probe output-grid change | $0.000296\%$ | 通过 | $0.25\rightarrow0.125\,\mu\mathrm m$ |
| fixed-detector intensity-grid change | $0.027377\%$ | 通过 | fine intensity 已 bin 回 $250\,\mathrm{nm}$ pixel |
| radial-source change | $0.789735\%$ | 通过 | taper step $1.0\rightarrow0.5\,\mathrm{nm}$ |
| radial-output interpolation change | $0.015619\%$ | 通过 | $125\rightarrow62.5\,\mathrm{nm}$ 表格间距 |
| final $\Delta D$ step change | $0.729462\%$ | 通过 | 最后两级中心差分 |

旧 run 的 $dx$ 变化约为 probe $74.19\%$、intensity $70.11\%$；新结果分别降到约 $0.000296\%$ 和 $0.027377\%$。这说明本轮确实消除了主要的 rasterization alias 和 detector mismatch，而不是放宽阈值。

当前 Fresnel 近轴路径仍有一个可量化限制。

这个限制来自近轴传播时，传播距离本应该是 $R=\sqrt{z^2+r^2}$ ,但Fresnel近似把他展开成 $R
\approx z+\frac{r^2}{2z}$ ,忽略了高阶项

按整个输出角点和孔外缘之间的最坏横向距离估计（也即是r/z的最大情况，也就是近轴近似误差最大的情况）：

$$
\frac{r_{\max}}{z}\approx0.0933,
$$

忽略的四阶路径相位项最坏约为：

$$
\left|\Delta\phi_4\right|\approx0.112\,\mathrm{rad}.
$$

因此当前结果是“采样收敛的近轴 projected-phase 结果”，不是 exact angular-spectrum、更不是 3D 电磁结果。该 $0.112\,\mathrm{rad}$ 估计是后续 exact-control 或 exp040 需要复核的误差上界。

### 12.6 true probe、detector 和 local observability

主中心差分使用：

$$
\Delta D_{\mathrm{waist}}=0.48828125\,\mathrm{nm}.
$$

关键结果为：

| 指标 | 结果 |
|---|---:|
| normalized true-probe sensitivity | $0.84796884$ |
| minus / baseline gauge-aligned complex L2 | $1.245641\times10^{-5}$ |
| plus / baseline gauge-aligned complex L2 | $1.245677\times10^{-5}$ |
| normalized detector-intensity sensitivity | $0.97053171$ |
| minus / baseline intensity relative L2 | $1.423181\times10^{-5}$ |
| plus / baseline intensity relative L2 | $1.427622\times10^{-5}$ |
| median / maximum frame sensitivity | $0.96967267 / 0.99248905$ |
| baseline frame-energy max relative deviation | $2.3872\times10^{-5}$ |

因此，在当前无噪声、完全已知、单孔、projected-phase 模型中，$D_\mathrm{waist}$ 对 true probe 的局部导数是有限、可复现且采样收敛的；固定 sample B 和 scan 后，detector intensity 也保留了该局部差异。这里的 $10^{-5}$ 级 plus/minus 相对差异是 $0.488\,\mathrm{nm}$ 这一有限扰动对应的数据变化，不代表真实含噪 detector 可以测到它。

local Jacobian 结果：

| 指标 | 结果 |
|---|---:|
| singular values | $[5134.014,\ 552.877,\ 303.212]$ |
| smallest / largest | $0.0590594$ |
| condition number | $16.9321$ |
| numerical rank | $3$ |
| $D_\mathrm{waist}$ 与其他列最大绝对相关 | $0.0875763$ |

因此，在这三个归一化列 `D_waist`、共同 surface diameter 和 `phase_scale` 组成的当前局部模型中，没有发现显式秩亏或强列相关。该结论仍不包含 $z_\mathrm{waist}$、tilt、偏心、材料色散、真实 3D 传播或实验误差。

### 12.7 Stage D 已执行，但 recovery 结论没有通过

首先讲清楚一些指标的定义

第一层：是否拟合 detector 数据

Data-fidelity loss：用来验证是否拟合detector数据，定义是 detector 振幅的相对误差：
$$
L=
\frac1{N_{\rm frame}}
\sum_j
\frac{
\left\|
|U^{\rm pred}_{C,j}|-\sqrt{I^{\rm meas}_j}
\right\|_2
}{
\left\|\sqrt{I^{\rm meas}_j}\right\|_2
}.
$$

第二层：单个 baseline 是否重建正确

Baseline aligned probe error：
$$E_P=
\frac{
\left\|
P_{\rm rec}^{\rm aligned}-P_{\rm true}
\right\|_2
}{
\|P_{\rm true}\|_2
}$$

Baseline aligned B error:

$$E_B=
\frac{
\left\|
B_{\rm rec}^{\rm aligned}-B_{\rm true}
\right\|_2
}{
\|B_{\rm true}\|_2
}$$

注意这两者在计算误差时都处理掉了整体相位的偏差，也就是说去除不能唯一确定的 gauge，要注意因为 ePIE 同时恢复 \(P_B\) 和 B。错误可能在二者之间互相补偿：
$$
P_{\rm rec}B_{\rm rec}
\approx
P_{\rm true}B_{\rm true},
$$
即使单独的 probe 或 B 并不正确。

第三层：是否保留腰径敏感度

True normalized probe sensitivity：真实正向模型给出的灵敏度是

$$
S_{\rm true}=
D_0
\frac{
\left\|
\Pi_{\rm gauge}
\left[
\dfrac{P_+^{\rm true}-P_-^{\rm true}}{2\Delta D}
\right]
\right\|_2
}{
\|P_0^{\rm true}\|_2
}.
$$

Recovered normalized probe sensitivity:
用三个重建 probe 做完全相同的计算：
$$
S_{\rm rec}=
D_0
\frac{
\left\|
\Pi_{\rm gauge}
\left[
\dfrac{P_+^{\rm rec}-P_-^{\rm rec}}{2\Delta D}
\right]
\right\|_2
}{
\|P_0^{\rm rec}\|_2
}.
$$

Recovered/true sensitivity relative deviation:
定义为
$$
\varepsilon_S=
\frac{|S_{\rm rec}-S_{\rm true}|}
{|S_{\rm true}|}.
$$

Plus/minus ordering:
分别计算 baseline 到两个扰动 case 的距离：
$$
e_-=
\frac{\|P_- -P_0\|_2}{\|P_0\|_2},
\qquad
e_+=
\frac{\|P_+ -P_0\|_2}{\|P_0\|_2}.
$$

以上就是三个层次的指标，下面正式讲实验结果。

三组 constrained blind ePIE 均执行了 180 iterations。baseline 的 data-fidelity loss 从 $0.349213$ 降到 $0.184847$，另外两组的最终 loss 也约为 $0.185$。loss 确实下降，但停在较高平台；simulation-evaluation-only 指标显示：

| 指标 | 结果 |
|---|---:|
| baseline aligned probe error | $0.315925$ |
| baseline aligned B error | $0.082227$ |
| true normalized probe sensitivity | $0.847969$ |
| recovered normalized probe sensitivity | $116.310980$ |
| recovered / true sensitivity relative deviation | $136.164$，即约 $13616\%$ |
| plus/minus ordering | 与 truth 一致 |

仅有 plus/minus ordering 一致不足以称为 recovery 成功。恢复出的三个 probe 之间差异被 reconstruction error 放大，远大于 true probe 的 $10^{-5}$ 级 case difference。因此本轮应作如下分层判断：

- Stage A--C 数值 forward、sampling sensitivity 和三列 local observability：通过；
- Stage D raw reconstruction：已执行；
- recovery 后定量保留 $D_\mathrm{waist}$ sensitivity：未通过，当前为 Inconclusive / failed quantitative recovery check。

最可能的首要原因不是 detector 已经丢失差异，因为 true detector sensitivity 已收敛；而是 Stage D 仍沿用 coarse A-plane pure-phase/reference projection。新的 authoritative A 到 B forward 来自连续径向 source，粗网格 backpropagated A 是其带限表示，并不严格属于“逐像素单位振幅纯相位”集合。这个 forward/constraint mismatch，加上当前 blind ePIE baseline 本身的有限精度，会形成高于 true case difference 的 reconstruction floor。单纯增加 iteration 数不应作为第一修正动作。

### 12.8 HDF5 新增字段和语义

以下字段是 exp030 专属扩展，没有改变项目级 `/entry` 并列结构：

| 字段 | shape | dtype | 单位 | 语义 |
|---|---:|---|---|---|
| `/entry/truth/effective_forward/radial_source_r_m` | `(9674,)` | float64 | m | authoritative radial quadrature 节点 |
| `/entry/truth/effective_forward/radial_source_weight_m` | `(9674,)` | float64 | m | 每个径向节点的积分权重 |
| `/entry/truth/effective_forward/A_effective_radial_true` | `(9674,)` | complex128 | 1 | baseline 连续径向 projected transmission 样本 |
| `/entry/truth/effective_forward/P_B_radial_r_m` | `(1087,)` | float64 | m | B-plane 径向输出坐标 |
| `/entry/truth/effective_forward/P_B_radial_true` | `(1087,)` | complex128 | a.u. | baseline 连续径向 probe |
| `/entry/truth/parameter_sweep/radial_source_r_m` | `(9674,)` | float64 | m | 三个 waist case 共用径向节点 |
| `/entry/truth/parameter_sweep/A_effective_radial_true` | `(3,9674)` | complex128 | 1 | minus / baseline / plus 径向 transmission |
| `/entry/truth/parameter_sweep/P_B_radial_r_m` | `(1087,)` | float64 | m | 三个 case 共用 B-plane 径向坐标 |
| `/entry/truth/parameter_sweep/P_B_radial_true` | `(3,1087)` | complex128 | a.u. | 三个 case 的连续径向 probe |

原有主要字段仍保持：

- `/entry/data/I_stack`：`(49,384,384)` float64，pixel-average intensity，a.u.；
- `/entry/truth/P_B_true`：`(384,384)` complex128，a.u.；
- `/entry/truth/parameter_sweep/I_stack_true`：`(3,49,384,384)` float64，a.u.；
- `/entry/reconstruction/cases/<case_id>/P_B_rec_raw`：`(384,384)` complex128，a.u.；
- `/entry/reconstruction/cases/<case_id>/loss_curve`：`(180,)` float64，无量纲相对 amplitude loss；
- raw reconstruction 与 `simulation_evaluation_only` alignment 分开保存。

当前通用 HDF5 writer 没有给这些 dataset 写 `units` attribute；上表记录的是字段物理语义。没有生成 calibration 或 preprocessing group。

### 12.9 本 run 中一个已知 metrics 字段问题

检查产物时发现，run `20260806_185142` 中的：

```text
/entry/metrics/intensity_sensitivity/plus_minus_relative_l2_to_baseline
```

以及外部 `metrics.json` 同名字段错误地调用了“两个完整数组之间的 relative error”，所以保存值约为 $0.999997$。它不是：

$$
\frac{\lVert I_+-I_-\rVert_2}{\lVert I_0\rVert_2}.
$$

由同一 run 已保存的中心差分指标可得正确值约为：

$$
2.84620\times10^{-5}.
$$

runner 已在本轮末尾修正为直接计算上述范数比，但按“不覆盖历史 run”的规则，没有回写 `20260806_185142`。除这个报告字段外，A--C gate、derivative、observability、reconstruction 和原始数组不受影响；下一次新 run 会自然写入修正后的值。

### 12.10 当前最好的下一步

下一步最好继续 exp030 做一次 **Stage D operator-consistency ablation**，而不是立即跳到 exp040。理由是：当前 ideal detector data 已经证明保留收敛的 waist signature，但 reconstruction floor 比 true case difference 大约两个到四个数量级；直接加入 3D propagation 只会增加未知因素，不能回答现有失败究竟来自 blind ePIE、A-plane constraint 还是 3D physics。

建议按以下顺序继续：

1. 在同一 exp030 模型中先去掉或替换不一致的 coarse A-plane pure-phase projection，做 probe-only、known-B 或 unconstrained-probe control，测出 reconstruction 自身的 floor；
2. 再实现与连续径向 A 到 B forward 一致的 constraint / adjoint，或者在后续独立任务中直接做参数化 $D_\mathrm{waist}$ fitting；
3. 只有当 ideal Stage D 能定量保留 $0.488\,\mathrm{nm}$ case difference 后，再考虑 noise、tilt 和 calibration error；
4. 单孔 pipeline 稳定后，若研究多孔及孔间相干，新开 exp031；
5. 若研究样品内部横向传播、侧壁折射及 depth ordering，新开 exp040 做 3D multislice，并用它检查本节约 $0.112\,\mathrm{rad}$ 的近轴误差估计。

本轮最终结论不是“真实 TGV 腰径已经可测”，而是：**单孔 projected-phase forward 的理想局部 sensitivity 已经数值收敛，detector 保留该差异；当前 blind recovery 尚未达到分辨这一级差异所需的精度。**

---

## 13. Stage D operator-consistency ablation：问题定位、修正和结果

本节只记录 2026-08-06 继续执行的 Stage D operator-consistency ablation。前面的实验说明和历史结果没有被改写。本轮要回答的问题不是“多跑一些 iteration 会不会偶然变好”，而是把旧 Stage D 中混在一起的几个误差源拆开：

1. B 到 C 的 detector forward 和 frozen data-fidelity loss（意思是把最终的 $P_B$ 和 B 固定，再重新计算全部帧的损失） 是否与仿真数据一致；
2. 已知 B 时，当前 probe update 自身能达到什么 60-iteration screening error（“Screening”是短迭代筛选，例如先运行60轮，而不是最终长期重建）；
3. 已知 probe 时，当前 B update 自身能达到什么 screening error；
4. blind ePIE 在没有 A-plane constraint（一些先验的限制，比如纯相位假设） 时的误差是多少；
5. 旧 coarse A-plane pure-phase constraint（这个是指A样品是纯相位样品的约束，这个限制会在盲ePIE上对$P_B$ 再做一次更新） 是否会主动破坏正确的 probe；
6. 与连续径向 A 到 B forward 成对的 adjoint （指的是foward的共轭转置）是否数值正确；
7. 数学正确的 adjoint 能否直接当作稳定 inverse 或 constraint；
8. 如果只施加与连续径向输出采样一致的 B-plane range constraint，能否降低 blind probe error。（range是值域的意思，这也可以作为一个限制）

最终有效 run 为：

```text
E:\tgv_ptycho_sim\runs\exp030_TGV_2d_effective_phase_20260806_205853
```

该 run 的顶层科学状态为：

```text
experiment_status: Inconclusive
Stage A--C: Passed
Stage D: Inconclusive
```

这里的 `Inconclusive` 不是说 projected-phase forward 失败，而是说 Stage D 尚未达到定量恢复本实验微小 waist perturbation 所需的精度。

### 13.1 先做不需要 reconstruction 的一致性检查

本轮首先直接检查 authoritative truth pair
($(P_B^{\mathrm{true}},B^{\mathrm{true}})$)。把同一 run 的 `P_B_true` 和 `B_true` 送入当前 B 到 C forward，得到：

| 检查 | 结果 |
|---|---:|
| direct detector-intensity relative L2 | $0$ |
| ePIE frozen detector-amplitude loss | $4.16573\times10^{-13}$ |

 direct detector-intensity relative:直接把真值$P_B$ 和 B用传播子传播一下，这里当然为0，因为我们的detector就是用$P_B$ 和 B 真值用传播子传播出来的。

 ePIE frozen detector-amplitude loss则是把固定当前 $P_B$ 和 $B$（这里直接用的真值，所以理论也应该没有），用它们重新计算全部49帧，再对每帧 detector 振幅误差取平均。

基于上面的数据 detector 数据、B 到 C propagation 和 frozen loss 之间没有隐藏的 operator mismatch。旧 Stage D 的约 $0.185$ loss 平台不能归因于 detector forward 生成了另一套数据。

然后把正确的 `P_B_true` 只通过一次旧 constraint：

```text
P_B_true
  -> coarse ASM backpropagation to A
  -> glass-reference correction
  -> pixelwise |A|=1 projection
  -> coarse ASM propagation to B
```
coarse ASM backpropagation to A:用角谱法把 probe 从 B 平面反向传播到 A 平面

glass-reference correction:反传播后的场可能带有整体的相位或振幅，代码利用已知孔外区域拟合并删除这些量，使孔外尽量恢复为 $A= 1$,形式近似为
$$
A_{\rm corrected}(x,y)=
\frac{A_{\rm raw}(x,y)}
{a\,e^{i(\phi_0+k_xx+k_yy)}}.
$$
它的目标是让孔外参考区域整体接近 $A_{\rm corrected}\approx1$,它是个整体的改动

pixelwise |A|=1 projection:无论当前结果是什么，都把它拉到“逐像素单位振幅且孔外为1”的集合中,保持强物理先验。

Coarse ASM propagation to B：把经过强制修改的 A 平面场重新传播到 B

结果为：

| 旧 constraint 直接作用于 truth | 结果 |
|---|---:|
| raw probe relative L2 | $0.320083$ |
| gauge-aligned probe relative L2 | $0.299083$ |
| 使用 true B 后的 detector-amplitude relative L2 | $0.231197$ |

Raw probe relative L2：直接比较旧约束输出和原始 truth：
$$
E_{\rm raw}=
\frac{
\|\widetilde P_B-P_B^{\rm true}\|_2
}{
\|P_B^{\rm true}\|_2
}.
$$

Gauge-aligned probe relative L2:先寻找最合适的全局复数和相位斜坡，对齐后再计算：
$$
E_{\rm aligned}=
\frac{
\|\widetilde P_B^{\rm aligned}-P_B^{\rm true}\|_2
}{
\|P_B^{\rm true}\|_2
}.
$$

使用 true B 后的 detector-amplitude relative L2:把**被旧约束破坏**后的 probe $\widetilde P_B$与完全正确的 $B^{\rm true}$ 相乘，再传播到 detector,然后和真实 detector 振幅比较。

这是一条直接的因果证据：旧 constraint 的可行集合不包含 authoritative continuous-radial truth probe,也就是
$$P_B^{\rm true}=
\text{连续径向 Fresnel–Hankel forward}[T_A(r)].$$
它单次就引入约 $29.9\%$ 的 gauge-aligned probe error，与上一 run 最终约 $31.6\%$ 的 recovered probe error 同量级。旧 Stage D 的首要问题确实是 operator/range mismatch，而不是 detector 已经丢失 waist signature。

原因也是可以推测的，我们在A到B的传播中是用了很精细的网格，但在使用限制时我们又是用了粗糙的角谱，不搭配。

### 13.2 连续径向 forward 的线性部分和加权 adjoint

连续径向 forward 把 A-plane transmission 写成背景与紧支撑扰动之和：

$$
\delta t_j=t_j-1,
\qquad
\mu_j=2\pi r_j\Delta r_j,
\qquad
p_{\mathrm{ref}}=a\exp(ikz).
$$

离散 radial Fresnel--Hankel 算子为：

$$
F_{\ell j}
=
\frac{p_{\mathrm{ref}}}{i\lambda_m z}
\exp\!\left(\frac{ik\rho_\ell^2}{2z}\right)
\exp\!\left(\frac{ikr_j^2}{2z}\right)
J_0\!\left(\frac{k\rho_\ell r_j}{z}\right)
\mu_j.
$$

令 $S$ 表示从径向 B-plane 表格到 Cartesian B-plane pixel 的线性插值，则完整仿射模型是：

$$
p_B
=
p_{\mathrm{ref}}+A\delta t,
\qquad
A=SF.
$$

这里前向的过程很明确，就是把连续的积分离散化，但是后面的验证的作用比较复杂，可以概括为“ePIE 给出一个当前 probe $p_{\mathrm{ePIE}}$ 后，怎样修改 A 平面径向透过率 $t$，使它通过同一个权威 forward 生成更接近 $p_{\mathrm{ePIE}}$ 的 probe？”这其实是ePIE后的优化，要从probe反解A了。

上一节提到，旧方法里

$$p_{\mathrm{ePIE}}
\xrightarrow{\text{粗 ASM 反传}}
A
\xrightarrow{|A|=1}
A'
\xrightarrow{\text{ASM 正传}}
p'.
$$

会造成极大的误差，现在要采取新方法了，新方法把正确的物理 forward
$$
p_{\rm model}(t)=p_{\rm ref}+A(t-1)
$$
嵌入优化，通过 adjoint 计算梯度，寻找能生成当前 $P_B$ 的 $t(r)$,放弃了之前的反传求A。

优化问题首先要定义残差，首先要算B的残差，定义为旧的A生成的probe B和ePIE算出来的probe B，但probe B和A的透过率矩阵结构不同，需要一个算子，将 B 平面的误差反向分配给 A 平面的各个径向节点。这个算子就是 adjoint（要区分一下，不是线性代数里面的古典伴随矩阵，伴随矩阵是由内积关系定义的）：

$$
A^\dagger:
\quad
\text{probe B的残差}
\longrightarrow
\text{A平面梯度}.
$$

下面是实验具体的$A^\dagger$的生成本轮没有把 `np.interp` 反向调用冒充 adjoint，而是为每个 Cartesian pixel 保存相同的左右径向节点和插值权重，并通过 scatter-add 实现严格的 $S^{\mathrm H}$(共轭转置，S是插值部分的矩阵)。采用：

$$
\langle x,y\rangle_A
=
\sum_j\mu_j\overline{x_j}y_j,
$$

（A平面的径向内积，注意有个权重$\mu_j$代表了径向情况下面积不同贡献不同）

以及：

$$
\langle u,v\rangle_B
=
dx^2\sum_q\overline{u_q}v_q,
$$

（B平面的笛卡尔内积，dx平方表示面积，这两个内积公式都是把积分化成离散形式）

总算子的加权 adjoint 为：

$$
A^\dagger
=
W_A^{-1}F^{\mathrm H}S^{\mathrm H}W_B.
$$

（W类似度量矩阵）

综上，这一节里面新型的A平面物理约束可概括为：
一次新型 A 平面物理约束可以概括为：

$$
\boxed{
\begin{aligned}
&\text{ePIE得到当前二维 probe }p\\
&\downarrow\\
&\text{当前径向 source }t\text{ 用 forward模型 生成 }m\\
&\downarrow\\
&\text{计算 B 平面残差 }r=gm-p\\
&\downarrow\\
&\text{用 }A^\dagger\text{ 把残差变成 A 平面梯度}\\
&\downarrow\\
&\text{只更新 source 相位，保持 }|t|=1\\
&\downarrow\\
&\text{用同一个 forward 重新生成 probe}\\
&\downarrow\\
&\text{交还给下一轮 ePIE}
\end{aligned}
}
$$

实际数值检查得到：

| 连续径向 operator 检查 | 结果 |
|---|:---|
| weighted adjoint inner-product relative error | $5.55711\times10^{-15}$ |
| forward reproduction relative L2 | $1.48765\times10^{-15}$ |
| truth fixed-point raw relative L2 | $1.49720\times10^{-15}$ |
| truth fixed-point detector-amplitude relative L2 | $9.86450\times10^{-16}$ |
| truth fixed-point idempotence relative L2 | $2.11500\times10^{-16}$ |
| pure-phase source amplitude max error | $2.22045\times10^{-16}$ |
| estimated weighted operator norm squared | $0.9971193$ |

解释一下这里各个指标的含义：

---

weighted adjoint inner-product relative error：
Adjoint 必须满足：

$$
\langle Ax,y\rangle_B
=
\langle x,A^\dagger y\rangle_A.
$$

所以随机生成了一个A平面的径向向量x和一个B平面的向量y来计算两边差了多少。

---
forward reproduction relative ：因为已经有一个真实的$P_B^{\rm true}$，现在把同一份真实径向透过率 $t^{\rm true}$ 输入新构建的矩阵算子：

$$
P_B^{\rm operator}
=
p_{\rm ref}
+
SF(t^{\rm true}-1).
$$

然后比较：

$$
\varepsilon_{\rm forward}
=
\frac{
\|P_B^{\rm operator}-P_B^{\rm true}\|_2
}{
\|P_B^{\rm true}\|_2
}.
$$

---
truth fixed-point raw relative：对真实的$P_B^{\rm true}$是加一次约束，看看变化多大,理论应该为0.

$$P_B^{\rm true}
\xrightarrow{\text{new constraint}}
P_B^{(1)}.$$

$$
\varepsilon_{\rm fixed}
=
\frac{
\|P_B^{(1)}-P_B^{\rm true}\|_2
}{
\|P_B^{\rm true}\|_2
}.
$$
---
Truth fixed-point detector-amplitude relative：将约束后的 $P_B^{(1)}$ 与真实 B 配对：
$$
U_{C,j}^{(1)}=
\mathcal P_{B\rightarrow C}
\left[P_B^{(1)}B_j^{\rm true}\right].
$$
然后与真实 detector amplitude 比较：
$$
\varepsilon_{\rm det}=
\frac{
\left\|
|U_C^{(1)}|-\sqrt{I_C^{\rm true}}
\right\|_2
}{
\left\|\sqrt{I_C^{\rm true}}\right\|_2
}.
$$
---
Truth fixed-point idempotence relative：Idempotence 意思是幂等性，这个意味着对真实probe B添加多次约束后是否变化。

代码连续施加两次：
$$
P_B^{(1)}=C(P_B^{\rm true}),
$$
$$
P_B^{(2)}=C(P_B^{(1)}),
$$
然后计算
$$
\varepsilon_{\rm idem}=
\frac{
\|P_B^{(2)}-P_B^{(1)}\|_2
}{
\|P_B^{(1)}\|_2
}.
$$

---
Pure-phase source amplitude max error：真实 A 平面径向 source 应满足纯相位条件：
$$
t_j=e^{i\phi_j},
\qquad |t_j|=1.
$$
指标定义为
$$
\varepsilon_{\rm amp}=
\max_j\left||t_j|-1\right|.
$$
---
Estimated weighted operator norm squared
定义为
$$
\|A\|^2=
\max_{x\ne0}
\frac{\|Ax\|_B^2}{\|x\|_A^2}.
$$
它表示 A 平面扰动经过 forward 后，能量最多被放大多少。

所以 continuous radial forward/adjoint pair 已在 float64 精度内通过。后续失败不能再解释为“adjoint 公式写错”。

### 13.3 为什么数学正确的 adjoint 不等于稳定 inverse

本轮实现了 source-side pure-phase projected-gradient control（这一串中source-side指的是直接对source也就是A侧直接优化，pure-phase projected-gradient control可以理解为基于纯相位的投影梯度的约束方法，我们不直接用梯度，而是用只改变相位的梯度）。给定当前 source transmission $t$ 和 ePIE probe $p$，先计算：

$$
m
=
p_{\mathrm{ref}}+A(t-1),
$$

再拟合 global complex-scale gauge：

$$
g
=
\frac{\langle m,p\rangle_B}
{\langle m,m\rangle_B}.
$$

残差和 source gradient 为：

$$
r=gm-p,
$$

$$
\nabla_t\Phi
=
A^\dagger\!\left(\overline g\,r\right).
$$

在 pure-phase manifold（纯相位的解空间） 上使用：

$$
\nabla_\phi\Phi
=
\operatorname{Im}
\left[
\overline t\,\nabla_t\Phi
\right],
$$

并通过 power-iteration step 和 backtracking（这俩都是优化的方法，一个估计步长，一个在步长太过的时候回退） 保证每个内部 projected step 不增加其局部 radial objective（也就是我们定下的优化目标）。最后必须重新执行同一个 forward，而不是在 Cartesian B plane 对 model 和旧 probe 做线性混合。

这个实现通过了 adjoint 和 truth fixed-point 测试（通过这俩测试说明现有的约束不会把真解推走，但不能保证求出真解），但它不是一个良态 inverse。当前 source 有 $9674$ 个 pure-phase unknown，而径向 B 表只有 $1087$ 个 complex samples（9674个未知量对
2174个实观测量，信息量不足）。即使忽略有限视场和插值，source 反演也有很大的 null space（那些经过forward后几乎不产生输出变化的source变化）。于是：

- truth transmission 与 truth probe 的联合状态是机器精度，$t_{\rm true}$和$
p_{\rm true}$这一对儿是这套约束下的不动点；
- 从 $t=1$ 开始，仅靠自由的 9674-point phase update 很难找到正确 source；
- “每个 adjoint step 都让局部 objective 下降”不代表嵌入 ePIE 后的全局 detector loss 或 probe error 会更好（这一点很重要，我们用约束优化了A产生了新的probe B不代表这个probe B对探测器上的结果更好）；
- 不能把 $A^\dagger$ 当成 $A^{-1}$。

这也是本轮保留 `blind_radial_adjoint_constraint` 作为**诊断组**（因为真解一定会被留下），而不把它选为最终 primary constraint 的原因。

### 13.4 增加不做 A inverse 的轴对称径向 B-plane range control

本节提出是一个比“A 平面反演约束”更温和的约束，即不去反演 A 平面的 TGV 透过率，只要求重建出的 B-plane probe 符合轴对称径向场的形式

在具体数学实现上，先设

$p$：ePIE 得到的二维 Cartesian probe；

$q$：一维径向复数表，共1087个节点；

$S$：把径向表 $q(r)$ 线性插值到二维网格的算子。

先寻找最能拟合当前 probe 的径向表：
$$
q^*=\arg\min_q\left(
\|Sq-p\|_2^2+\epsilon\|q\|_2^2
\right),
$$
其解为
$$
q^*=(S^{\mathrm H}S+\epsilon I)^{-1}S^{\mathrm H}p.
$$
然后重新生成轴对称二维 probe：
$$
\Pi_S=Sq^*.
$$

具体实现上：

$$
\Pi_S=
S
\left(
S^{\mathrm H}S+\epsilon I
\right)^{-1}
S^{\mathrm H}.
$$

其中 $S^{\mathrm H}S$ 是三对角矩阵，使用显式 Thomas solver 求解，没有调用可能在目标 Windows 环境卡住的大型 complex BLAS。ridge fraction 为：

$$
\frac{\epsilon}{\max\operatorname{diag}(S^{\mathrm H}S)}
=
10^{-13}.
$$

该 constraint 的 truth 检查为：

| 径向 B-plane range control | 结果 |
|---|---:|
| active radial nodes / total | $1072/1087$ |
| truth fixed-point raw relative L2 | $4.32857\times10^{-12}$ |
| truth fixed-point detector-amplitude relative L2 | $2.91489\times10^{-12}$ |
| idempotence relative L2 | $4.32859\times10^{-12}$ |

再讲一下这些指标的定义：

---
active radial nodes / total：参与插值的点的比例。

---
truth fixed-point raw relative：验证真值加了约束能不能得到正确的结果，这里先是probe B。

---
truth fixed-point detector-amplitude relative：同上面，不过是验证probe B被施加约束后再传播到探测器的振幅

---
idempotence relative：验证是不是不动点，即真值被施加了约束再施加，偏移有多大。

---
这些误差远低于本实验约 $10^{-5}$ 的 true case separation。需要明确：$\Pi_S$ 只约束 B plane 输出属于当前轴对称 radial sampling range；它不是 A-plane inverse，也不证明已经恢复 `D_waist`。

### 13.5 七个 baseline ablation 的设置

这里是七组对照试验（ablation意为消融实验），这七组共享stageA-C的同一份baseline detector data，然后再进入stageD做区分，比较，所有组公用的条件包括

+ 同一套baseline探测器数据；
+ 同一扫描位置；
+ 同一随机扫描顺序和shuffle seed；
+ 相同ePIE参数；
+ 每组先运行60轮，用于screening比较；
+ 除明确说明外，probe都从探测器数据反传播得到的初始化开始，不是真实probe；
+ 除明确说明外，样品B都从同一个随机相位初始化开始；
+ 样品B都施加纯相位振幅范围 $|B|=1$。

所有组都使用同一 baseline detector data、同一 scan、同一 measurement-derived probe initialization、同一 blind-object initialization 和同一 shuffle seed。每组 screening 为 60 iterations。

| Variant | 固定量或额外 constraint | truth 是否进入迭代 | 目的 |
|---|---|---|---|
| `known_b_probe_only` | 固定 `B_true`，只更新 probe | 是，仅 simulation diagnostic | 测 probe update / initialization screening error |
| `known_probe_object_only` | 固定 `P_B_true`，只更新 B | 是，仅 simulation diagnostic | 测 B update screening error |
| `blind_unconstrained_with_energy_norm` | 无 A constraint，保留旧 energy norm | 否 | 隔离 energy norm 影响 |
| `blind_unconstrained` | 无 A constraint、无 energy norm | 否 | 测 blind ePIE baseline |
| `blind_legacy_coarse_a_constraint` | 旧 coarse ASM pure-phase constraint | 否 | 重现不一致 constraint |
| `blind_radial_output_range_constraint` | $\Pi_S$ B-plane radial range | 否 | 测一致的轴对称输出先验 |
| `blind_radial_adjoint_constraint` | full source forward/adjoint projected step | 否 | 测高维 source-side constraint |

其中：

实验3：保留旧的probe energy norm，每轮之后将probe缩放到固定L2范数：
$$
P_B\leftarrow
P_B\frac{\|P_B\|_{\rm target}}{\|P_B\|_2}.
$$

类似于把能量归一化

实验4：完全无多余限制，仍有公共的探测器振幅约束和 $|B|=1$ 等基础限制

实验5：使用旧版coarse ASM A-plane pure-phase constraint，同时包含probe energy norm，这是之前失败的实验。

实验6：无energy norm；使用13.4中的轴对称限制，把每轮的probe投影到轴对称径向的空间中。

实验7：无energy norm；使用13.2建立的连续径向forward/加权adjoint，在A侧维护纯相位source，通过13.3的projected-gradient更新source，再用同一个forward重新生成B-plane probe。

known-B 和 known-probe 结果在 HDF5 中明确标为 `simulation_diagnostic_only`。五个 blind variants 的 truth-input flags 均为 false；truth alignment 只出现在 `simulation_evaluation_only` 下。

### 13.6 energy-norm control 的结论

这一节主要比较实验3和实验4，即是否保留energy norm

旧 probe norm target 来自 detector frame energy：

$$
\lVert P\rVert_{\mathrm{target}}=
359.3781478.
$$

但 authoritative true probe norm 为：

$$
\lVert P_{\mathrm{true}}\rVert_2
=359.5483682.
$$

相对偏差为：

$$
4.73429\times10^{-4}=
0.04734\%.
$$

这是因为我们计算 $\lVert P\rVert_{\mathrm{target}}$ 是从探测器数据估计的，但当前 propagating-wave cutoff 并非严格的离散 unitary map。该 bias 约是真实 probe case separation（也就是由腰径变换带来的探针变化） 的 38 倍，所以它不适合作为 $10^{-5}$ 级定量 sensitivity 的物理恒等式。

不过在当前 60-iteration blind screening 中，有 norm 和无 norm 两组的 aligned probe error 分别为 $0.001735679$ 和 $0.001735669$，相对差约 $5.7\times10^{-6}$。因此 energy norm 在理论上不严格，但它不是本轮 observed recovery floor 的主要来源。

### 13.7 七组实际 reconstruction 结果

下面的 probe 和 B error 都是 `simulation_evaluation_only` gauge-aligned complex relative error。`tail slope` 是最后 20 个 sequential iterations 的线性斜率。

| Variant | initial loss | frozen final loss | tail slope / iter | probe error | probe error / true signal | B error |
|---|---:|---:|---:|---:|---:|---:|
| known-B, probe-only | $0.136585$ | $0.00395051$ | $-8.144\times10^{-7}$ | $0.00988520$ | $793.57$ | $4.83\times10^{-17}$ |
| known-probe, object-only | $0.304856$ | $0.00731366$ | $9.669\times10^{-8}$ | $1.74\times10^{-16}$ | $1.39\times10^{-11}$ | $0.0276375$ |
| blind + energy norm | $0.299644$ | $0.00725216$ | $1.965\times10^{-7}$ | $0.00173568$ | $139.34$ | $0.0276519$ |
| blind unconstrained | $0.299644$ | $0.00725268$ | $1.964\times10^{-7}$ | $0.00173567$ | $139.34$ | $0.0276519$ |
| blind legacy coarse A | $0.349213$ | $0.185937$ | $-3.488\times10^{-5}$ | $0.315649$ | $25339.9$ | $0.0815112$ |
| blind radial B-range | $0.308752$ | $0.00729305$ | $1.786\times10^{-7}$ | $0.000258122$ | $20.7217$ | $0.0276409$ |
| blind radial A-adjoint | $0.359845$ | $0.111586$ | $-1.836\times10^{-3}$ | $0.154877$ | $12433.4$ | $0.0282329$ |

首先确定loss，error大部分都是无量纲的比例量，然后值得注意的是probe error/true signal指的是算法在复原probe B后产生的误差与ΔD带来的probe的变换的比值。B是样品决定的，ΔD不会对B产生变化，所以没有B error/true signal

必须避免把所有这些数值都称为已证明的“算法固有 floor”（这个词的意思就是无论在跑多少轮次都无法逾越的平台）。它们是同条件的 **60-iteration screening residual/error**。known-B 和 radial A-adjoint 的 tail 仍明显为负（也就是最后几轮展现出来的斜率），没有证明已经达到渐近收敛。blind unconstrained、known-probe 和 radial B-range 的 tail 已接近平台，因此它们更接近当前设置下的 practical screening floor。

本表支持以下定量比较：

- 去掉旧 coarse constraint 后，blind probe error 从 $31.56\%$ 降到 $0.1736\%$；
- 旧 constraint 的 probe error 是 blind unconstrained 的约 $181.9$ 倍；
- radial B-range 把 blind unconstrained probe error 再降低约 $6.72$ 倍；
- radial B-range 相比旧 constraint 降低约 $1223$ 倍；
- full radial A-adjoint 虽然每个内部 step 都降低其局部 source objective，但当前嵌入方式产生 $15.49\%$ probe error，不能当成有效 inverse；
- known-probe object-only 的 B error 仍为 $2.76\%$，且 frozen loss 与多个 blind variants 的约 $0.0073$ 平台几乎相同，说明当前 B update、ePIE 优化、周期边界或 object 表示是剩余 plateau 的重要来源；
- known-B probe-only 在 60 iterations 后仍有 $0.9885\%$ probe error，说明简单的 sequential probe update 即使已知 B，也远未达到本实验 $10^{-5}$ signal 所需的精度。

此处还有后续，可以看到实验1和2中已知一个真值来进行ePIE的结果都不太好，后续第14节发现是一个反传的过程推导错误，这个错误甚至会导致probe B和B的真值都不能作为不动点。

### 13.8 为什么没有继续运行 recovered plus/minus sensitivity

当前真实 signal 尺度为：

$$
s_P
=
\frac{1}{2}
\left[
d(P_-,P_0)+d(P_+,P_0)
\right]
=
1.2456589\times10^{-5}.
$$

detector-amplitude signal 尺度为：

$$
s_I
=
9.2478185\times10^{-6}.
$$

当前最佳 blind variant 是 radial B-range，其 baseline aligned probe error 为：

$$
2.5812184\times10^{-4}.
$$

因此：

$$
\frac{\text{screening probe error}}{s_P}
=
20.7217.
$$

该比值仍大于预设 gate =$1$，所以脚本正确记录：

```text
not_run_reconstruction_floor_exceeds_true_signal
```

没有继续运行 recovered minus/plus finite difference。这不是缺失执行，而是避免再次把比真实 case difference 大 20 倍的 reconstruction error 误解释为 recovered waist sensitivity。Stage D 的科学结论仍为 Inconclusive。

### 13.9 loss 图为什么增加 frozen final point

`epie_reconstruct()` 的 `loss_curve` 记录每个 iteration 内 sequential update（这里是说每一帧probe B和B都会更新） 的平均 loss；外部 probe constraint 在 iteration 末尾施加。

`final_data_fidelity_loss` 则是在所有 update 和 constraint 完成后，用冻结的最终 probe/B 对完整 stack 重新评估。

也就是说一轮的loss其实是每帧loss的平均，但每帧的loss会变，所以最终定下probe B和B的结果后，我们就不更新的求一轮loss，这就是`final_data_fidelity_loss`。

二者不应混为同一个量。旧图只画 sequential curve，特别会掩盖：

- legacy 组最后 sequential loss 约为 $0.11929$，但 frozen final loss 为 $0.18594$；
- radial A-adjoint 组最后 sequential loss 约为 $0.00685$，但 frozen final loss 为 $0.11159$。

修正后的 `loss_curves.png` 使用短图例，标题改为 operator-consistency ablation，并把第 61 个点画成 frozen final reevaluation。HDF5 中原始 `loss_curve` 仍保持 `(60,)`，frozen final 继续单独保存在 `final_data_fidelity_loss`，没有改变字段语义。

### 13.10 本轮代码改动

本轮在 `scripts/run_exp030_effective_phase.py` 中增加或修改了：

1. 与 forward 相同权重的 radial-to-Cartesian interpolation plan；
2. 通过 scatter-add 实现的精确 $S^{\mathrm H}$；
3. 连续径向 Fresnel--Hankel linear forward 和 weighted adjoint；
4. operator norm power iteration；
5. global complex-gain fitting、pure-phase projected-gradient 和 backtracking；
6. radial B-plane range projector 及三对角 Thomas solver；
7. known-B、known-probe、blind/no-constraint、legacy、range 和 adjoint 七组 ablation；
8. truth-input metadata 隔离；
9. baseline error / true signal gate；
10. 若未来 gate 通过，则使用 case-to-recovered-baseline 的共同 gauge，再用一次 baseline-to-truth anchor 评价三 case sensitivity，避免三个 case 独立 truth alignment 产生伪差异；
11. 顶层状态改为同时依赖 Stage A--C 和 Stage D；
12. loss 图增加 frozen final point。

`configs/experiments/exp030_TGV_2d_effective_phase.yaml` 新增了 ablation variants、60-iteration screening、range ridge、adjoint power iteration、constraint interval 和 signal gate。`src/tgv_ptycho/viz/plot_tgv.py` 只修改了 Stage D 图的准确标题和标签。

### 13.11 新增测试和 Windows 性能问题

新增测试覆盖：

- $S/S^{\mathrm H}$ Euclidean inner-product identity；
- $A/A^\dagger$ weighted inner-product identity；
- radial adjoint truth fixed point；
- projected step objective 不增加；
- radial B-range truth fixed point 和 idempotence；
- known-B control 确实冻结 B。

最终测试结果：

```text
48 passed in 1.16s
```

本轮修改范围 Ruff：

```text
All checks passed!
```

项目级 Ruff 仍有 13 个既有、与本任务无关的问题：`run_exp001_forward.py` 的 8 个 `E402`，`calibration/stage.py` 的 1 个 `E501`，用户已有修改 `angular_spectrum.py` 的 2 个 `E501`，以及 `recon/losses.py`、`recon/rpie.py` 各 1 个 `E501`。本轮没有顺手修改这些文件。

第一次正式尝试产生目录：

```text
E:\tgv_ptycho_sim\runs\exp030_TGV_2d_effective_phase_20260806_195612
```

它在新增的大数组 complex `np.vdot` 路径上停止消耗 CPU，被终止并保留为失败诊断目录。修正方法是把 large complex dot 和 real-kernel/complex-vector contraction 改为显式实部、虚部分离以及 `sum(conj(a)*b)`，避免目标 Windows 环境的 pathological complex BLAS 路径。实际尺寸 benchmark 为：

| 操作 | 时间 |
|---|---:|
| 构建约 80 MiB Bessel kernel | $0.225\,\mathrm s$ |
| radial forward | $0.021\,\mathrm s$ |
| weighted adjoint | $0.011\,\mathrm s$ |
| 一次 full source constraint | $0.092\,\mathrm s$ |

随后 run `20260806_202941` 完成了核心 ablation，但独立产物审计发现旧 loss 图和顶层 status 语义不够准确。最终 run `20260806_205853` 修复这些展示/状态问题。两个成功 run 的 operator controls、七组 variant metrics、probe metrics 和 intensity metrics 逐项完全相同，说明本轮结果 deterministic。所有历史 run 均未覆盖。

### 13.12 HDF5 新增字段

项目级 `/entry` 并列结构没有改变。baseline data 和 truth 字段保持原语义。新增 experiment-specific reconstruction 结构为：

```text
/entry/reconstruction/operator_consistency_ablation/variant_ids
/entry/reconstruction/operator_consistency_ablation/variants/<variant_id>/cases/baseline/...
/entry/reconstruction/operator_consistency_ablation/selected_sensitivity_check/...
```

主要新增字段如下：

| 字段 | shape | dtype | 单位或语义 |
|---|---:|---|---|
| `.../P_B_init` | `(384,384)` | complex128 | B-plane field，a.u. |
| `.../B_init` | `(384,384)` | complex128 | sample-B transmission，1 |
| `.../P_B_rec_raw` | `(384,384)` | complex128 | blind 或 probe-only raw result，a.u. |
| `.../B_rec_raw` | `(384,384)` | complex128 | blind 或 object-only raw result，1 |
| `.../B_fixed_simulation_diagnostic_only` | `(384,384)` | complex128 | known-B control 的固定 truth B，1 |
| `.../P_B_fixed_simulation_diagnostic_only` | `(384,384)` | complex128 | known-probe control 的固定 truth probe，a.u. |
| `.../loss_curve` | `(60,)` | float64 | sequential relative detector-amplitude loss，1 |
| `.../final_data_fidelity_loss` | scalar | float64 | frozen full-stack detector-amplitude loss，1 |
| `.../simulation_evaluation_only/P_B_rec_aligned_to_truth` | `(384,384)` | complex128 | 仅仿真评价，a.u. |
| `.../radial_constraint/source_transmission_final` | `(9674,)` | complex128 | source radial pure-phase transmission，1 |
| `.../radial_constraint/objective_before` | `(13,)` | float64 | 每次 source projected step 前的 weighted objective |
| `.../radial_constraint/objective_after` | `(13,)` | float64 | 每次 source projected step 后的 weighted objective |
| `.../radial_output_range_constraint/...` | scalar fields | numeric/string | ridge、node count、调用次数和角色说明 |

metrics 新增：

```text
/entry/metrics/stage_status/...
/entry/metrics/reconstruction_check/operator_consistency/...
/entry/metrics/reconstruction_check/operator_consistency_ablation/...
```

最终 HDF5：

```text
E:\tgv_ptycho_sim\runs\exp030_TGV_2d_effective_phase_20260806_205853\outputs\exp030_effective_phase.h5
```

baseline `I_stack` 为 `(49,384,384)` float64，`scan_positions` 为 `(49,2)` float64；七组 baseline probe/B 均为 `(384,384)` complex128。没有生成 calibration 或 preprocessing group。数值数组均 finite，外部 JSON 与 HDF5 metrics/metadata 一致。

当前通用 writer 仍没有写 dataset `units` attributes，也没有项目级 compression/chunking 策略。本节记录了单位和语义；按仓库规则，本轮没有擅自扩展项目级 HDF5 规范。

### 13.13 当前能下和不能下的结论

可以下的结论：

1. Stage A--C 的 projected-phase、true probe sensitivity 和 detector sensitivity 仍数值收敛；
2. B 到 C truth-pair forward 与 detector data 一致；
3. 旧 coarse A-plane constraint 是上一 Stage D 主要失败来源，且会主动破坏 truth；
4. 新 continuous radial forward/weighted adjoint 数学和离散实现正确；
5. 去掉旧 constraint 后 reconstruction 明显改善；
6. radial B-range 是当前最有效的 blind regularizer，但其 screening error 仍是真实 signal 的 20.72 倍；
7. 当前剩余问题主要在 reconstruction/optimization，而不是因为 projected-phase forward 中不存在 waist information。

不能下的结论：

1. 不能说 generic ePIE 在原理上永远无法恢复该 signal；当前只是单 seed、60-iteration screening；
2. 不能把数学正确的 adjoint 称为稳定 inverse；
3. 不能说 radial B-range 已经恢复 A plane 或 `D_waist`；
4. 不能因为 detector truth sensitivity 存在就宣称真实含噪实验可测；
5. 不能把本轮结果外推到 tilt、多孔、noise、真实折射、3D multislice 或实验 calibration；
6. 不能把顶层 `Inconclusive` 解释为 Stage A--C 失败，它专门反映 Stage D 定量恢复尚未通过。

### 13.14 下一步最好的选择

当前不应直接跳到 exp040，也不应立即跑 recovered plus/minus。下一步优先继续 exp030 的同模型 reconstruction controls：

1. 对 `known_probe_object_only` 做 iteration、`beta_object`、finite support、非周期边界和更新规则 ablation，先解释为什么无噪声、known-probe 条件下 B error 仍约为 $2.76\%$；
2. 对 `known_b_probe_only` 跑至少 200、500、1000 iterations，并扫描 `beta_probe` 和 normalization；由于当前 tail 仍下降，$0.9885\%$ 不能称为已验证的渐近 floor；
3. 要求 aligned probe error 低于 $1.2457\times10^{-5}$，最好低于 signal 的 $0.25$ 到 $0.5$ 倍，同时 frozen detector-amplitude loss 低于 $9.2478\times10^{-6}$，并确认 tail 平台和多 seed 不翻转；
4. 保留 radial B-range 作为当前 blind baseline，暂停把 full source adjoint 当硬 constraint；
5. 如果单变量 controls 和更合适的 reconstruction optimizer 仍无法降到 signal 以下，新开后续任务做低维 TGV parametric/model-constrained fitting。该任务必须从未知初值和预先声明的 bounds 出发，不能读取 case truth 修正结果；
6. 单孔 ideal pipeline 达到 signal gate 后，再新开 exp031 研究多孔阵列和孔间相干；
7. exp040 仍用于真正的 3D multislice、内部横向传播和 depth ordering，不应用来掩盖当前 2D reconstruction floor；
8. noise、tilt、calibration error 和实际 detectability 应在 ideal Stage D 通过后新开后续任务。

因此本轮最准确的总结是：**旧 Stage D 的主要 operator mismatch 已经找到并修正；一致的 radial B-range 让 blind probe error 改善约 6.72 倍，但仍高于真实 waist signal 20.72 倍。下一步应先修复/替换当前 ePIE reconstruction optimizer，再考虑参数化拟合、exp031 或 exp040。**

---

## 14. 2026-08-09：ePIE optimizer 的传播伴随修正与长程控制实验

### 14.1 本轮要回答什么

上一节把下一步问题限定为两项：

1. 为什么在无噪声且 known-probe 的条件下，60 轮之后的 B error 仍约为 $2.76\%$；
2. known-B/probe-only 的 $0.9885\%$ probe error 是真实收敛 floor，还是 60 轮过短、步长或 normalization 不合适。

本轮没有改变 projected-phase TGV 模型，也没有把 truth 引入 blind reconstruction。工作重点是检查 B 到 detector 的传播算子和 ePIE update 是否在离散意义下互相一致，再在修正后执行：

- truth-initialized fixed-point control；
- known-probe/object-only 收敛轨迹；
- known-B/probe-only 的 `beta_probe` 单因素筛选；
- normalization 单因素对照；
- 周期边界与 constant-reference 有限边界的配对/不配对对照；
- 一条连续的 1000-iteration known-B trajectory，并在 200、500、1000 轮保存 checkpoint；
- 修正后的原七组 blind/known ablation 60 轮同条件复核。

### 14.2 找到的决定性问题：旧 ePIE 把伴随算子当成逆算子

令 B 到 detector 的离散 angular-spectrum 传播为：

$$
H
=
\mathcal F^{-1}
\operatorname{diag}
\left[
M\exp\left(i k_z z_{BC}\right)
\right]
\mathcal F,
$$

其中 $M$ 是 propagating-wave mask。当前参数为：

$$
\Delta x=0.25\,\mu\mathrm m,
\qquad
\lambda=0.532\,\mu\mathrm m.
$$

此时离散 Fourier grid 中只有：

$$
\frac{102325}{147456}
=
69.3936\%
$$

的频率点位于 propagating mask 内；约 $30.6064\%$ 的采样频率被置零。因此：

$$
H^{\mathrm H}H
=
\mathcal F^{-1}
\operatorname{diag}(M)
\mathcal F
\ne I.
$$

也就是说，负距离传播在这个 band-limited 模型里是 $H^{\mathrm H}$，但不是 $H^{-1}$。

旧 ePIE 使用：

$$
\Delta\psi_{\mathrm{old}}
=
H^{\mathrm H}u_{\mathrm{corrected}}
-
\psi.
$$

把 $u_{\mathrm{pred}}=H\psi$ 加入上式，可以得到：

$$
\Delta\psi_{\mathrm{old}}
=
H^{\mathrm H}
\left(
u_{\mathrm{corrected}}-u_{\mathrm{pred}}
\right)
+
\left(
H^{\mathrm H}H-I
\right)
\psi.
$$

第一项是正确的 detector residual gradient；第二项是不应该存在的 nullspace drift。即使预测振幅已经与测量完全相同，第二项仍会主动修改真值中的高频成分。

修正后的 update 是：

$$
r_{\mathrm{det}}
=
u_{\mathrm{corrected}}-u_{\mathrm{pred}},
$$

$$
\Delta\psi_{\mathrm{new}}
=
H^{\mathrm H}r_{\mathrm{det}}.
$$

因此在 exact-data fixed point 上：

$$
u_{\mathrm{corrected}}
=
u_{\mathrm{pred}}
\quad\Longrightarrow\quad
\Delta\psi_{\mathrm{new}}=0.
$$

这不是单纯调小 `beta`，而是修正 optimizer 使用的线性算子。

### 14.3 truth-initialized one-iteration fixed-point 实证

使用 baseline 的真 probe、真 B 和真 detector stack 作为起点，只允许一个变量更新一轮。truth 只用于这个明确标注的 `simulation_diagnostic_only` control。

| Update | correction | 初始 frozen loss | 一轮后 frozen loss | probe relative change | B relative change |
|---|---|---:|---:|---:|---:|
| object-only | 旧 inverse-difference | $0$ | $7.39216\times10^{-3}$ | $0$ | $2.73765\times10^{-2}$ |
| probe-only | 旧 inverse-difference | $0$ | $2.97255\times10^{-3}$ | $7.75309\times10^{-3}$ | $1.49\times10^{-17}$ |
| object-only | 新 adjoint-residual | $0$ | $2.94929\times10^{-16}$ | $0$ | $1.82727\times10^{-16}$ |
| probe-only | 新 adjoint-residual | $0$ | $2.54607\times10^{-16}$ | $5.65694\times10^{-17}$ | $1.49\times10^{-17}$ |

旧 object-only 一轮产生的 $2.73765\%$ B drift 几乎直接复现了历史 60 轮的 $2.76375\%$ B error。因此原来的 $2.76\%$ 主要不是“迭代不足形成的自然 floor”，而是旧 update 在 exact solution 上也会主动把 B 推离真值。

新 update 在 float64/complex128 精度下保持真值不动，fixed-point gate 通过。

### 14.4 代码修改的具体内容

本轮修改遵循“公共算子放公共模块、实验编排放 runner”的边界：

1. `src/tgv_ptycho/optics/angular_spectrum.py`
   - 增加可复用的 `make_angular_spectrum_transfer()`；
   - 增加 `apply_angular_spectrum_transfer()`；
   - 正向 transfer 只构造一次，伴随明确使用其复共轭；
   - 原 `angular_spectrum_propagate()` 的公共 API 和数值行为保持兼容。
2. `src/tgv_ptycho/forward/integer_shift.py`
   - 新增 forward/reconstruction 共用的 integer shift；
   - 支持历史 `periodic` 和实验 control 所用 `constant` boundary；
   - constant boundary 的 object exterior 为参考透过率 $1+0i$；
   - 对 object increment 使用严格配对的零填充反移伴随。
3. `src/tgv_ptycho/forward/scheme_probe_B.py`
   - 默认仍为历史 periodic forward；
   - 新增明确的 matched constant-reference forward control；
   - 没有改变 exp020/exp030 baseline 的周期数据语义。
4. `src/tgv_ptycho/recon/epie.py`
   - 默认 correction 改为 `adjoint_residual`；
   - 保留 `legacy_inverse_difference`，但只用于显式回归诊断；
   - 增加独立 `update_object`，known-B 不再用 `beta_object=0` 模拟冻结；
   - 增加 200/500/1000 等单轨迹 checkpoint；
   - 增加 paired boundary 选择；
   - 增加可选 rPIE denominator，但本轮 authoritative trajectory 仍使用 ePIE denominator；
   - 缓存 forward/adjoint transfer，避免每帧重复构造频率网格和 transfer。
5. `scripts/run_exp030_effective_phase.py`
   - 增加 bounded optimizer study；
   - `beta` 和 normalization 只做单因素筛选，不做全笛卡尔积；
   - 设置 $6000\,\mathrm s$ optimizer-study runtime budget；
   - beta 和 normalization 的选择只使用 measurement-only frozen loss；
   - truth-aligned error 只保存在 `simulation_evaluation_only`；
   - boundary 分成 matched periodic、故意 mismatch 和 matched constant 三组；
   - 1000 轮只运行一条轨迹并在内部保存 checkpoint。
6. `src/tgv_ptycho/viz/plot_tgv.py`
   - loss 图现在允许不同 case 使用不同迭代长度；
   - 这修复了 60 轮与 1000 轮曲线同时出现时的最终绘图失败。

没有安装 PyTorch、CuPy、JAX 或其他 GPU 依赖。本机有 RTX 3050 Laptop GPU，但当前环境没有 CUDA Python array backend，而且消费级 GPU 的 FP64 吞吐不适合在没有等价测试时直接替代 CPU complex128。缓存 transfer 后实测约为：

$$
1.84\,\mathrm{s/iteration}.
$$

1000 轮核心轨迹实际用时：

$$
1825.38\,\mathrm s
=
30.42\,\mathrm{min}.
$$

因此本轮没有引入 GPU，也没有超过单次两小时运行上限。

### 14.5 known-probe/object-only：旧 $2.76\%$ floor 是否消失

使用与历史 run 相同的 measurement-derived/random object initialization、相同 scan 和 seed，固定真 probe，只更新纯相位 B。

| Iteration | frozen detector-amplitude loss | gauge-aligned B error |
|---:|---:|---:|
| 1 | $3.09499\times10^{-2}$ | $6.64477\times10^{-2}$ |
| 5 | $3.08040\times10^{-3}$ | $1.79073\times10^{-2}$ |
| 20 | $8.40470\times10^{-4}$ | $1.06588\times10^{-2}$ |
| 60 | $3.15924\times10^{-4}$ | $7.82063\times10^{-3}$ |

历史旧式 60 轮 B error 为 $2.76375\times10^{-2}$；修正后为 $7.82063\times10^{-3}$，改善约 $3.53$ 倍。最后 20 轮 loss slope 为：

$$
-2.12062\times10^{-6}\ \mathrm{iteration}^{-1},
$$

仍明显小于零，所以 $0.782\%$ 也不能称为渐近 floor。结论是：旧 $2.76\%$ 的主要原因已经定位并修复；剩余误差来自有限迭代和 object optimization，而不是 exact solution 被错误算子主动破坏。

### 14.6 known-B/probe-only 的 `beta_probe` 筛选

所有 beta 候选均运行 30 轮、固定同一个真 B、不施加 probe normalization。选择标准仅为 frozen detector-amplitude loss。

| `beta_probe` | 30 轮 frozen loss | gauge-aligned probe error |
|---:|---:|---:|
| 0.02 | $9.94585\times10^{-4}$ | $3.46773\times10^{-3}$ |
| 0.08 | $6.22610\times10^{-5}$ | $7.38906\times10^{-4}$ |
| 0.20 | $2.20221\times10^{-5}$ | $3.61105\times10^{-4}$ |
| 0.50 | $9.88165\times10^{-6}$ | $2.07473\times10^{-4}$ |

在已测试区间内，measurement-only loss 选择 `beta_probe=0.5`。由于最优点位于扫描上边界，本表只能说明 $0.5$ 是四个候选中最好，不能声称它是所有步长中的全局最优值。不过后面的 1000 轮轨迹已达到 $10^{-12}$ 级 aligned error，因此不需要为了本轮 fixed-point/long-run 结论再扩大 beta 搜索。

### 14.7 normalization control

固定 `beta_probe=0.5`，每组 30 轮。

| normalization | frozen loss | gauge-aligned probe error | 是否可参与选择 |
|---|---:|---:|---|
| none | $9.88165\times10^{-6}$ | $2.07473\times10^{-4}$ | 是 |
| measurement-energy | $4.73545\times10^{-4}$ | $2.04755\times10^{-4}$ | 是 |
| truth-probe norm | $9.88183\times10^{-6}$ | $2.07477\times10^{-4}$ | 否，仅 simulation diagnostic |

measurement-energy normalization 把 frozen loss 恶化约 48 倍。其原因与上一节一致：带 propagating cutoff 的 $H$ 不是严格酉算子，detector frame energy 不能作为 probe norm 的无偏恒等式。

最终 measurement-only 选择为：

```text
beta_probe = 0.5
normalization = none
boundary = periodic
```

### 14.8 boundary control

本轮没有只在 inverse 端随意更换边界后比较“哪个更好”，而是区分 operator matched 与 mismatched。

| detector data boundary | reconstruction boundary | 角色 | 30 轮 frozen loss | aligned probe error |
|---|---|---|---:|---:|
| periodic | periodic | 当前 exp030 matched baseline | $9.88165\times10^{-6}$ | $2.07473\times10^{-4}$ |
| periodic | constant reference | 故意制造 mismatch | $1.98466\times10^{-1}$ | $1.69422\times10^{-1}$ |
| constant reference | constant reference | matched finite-FOV control | $1.10898\times10^{-5}$ | $2.52531\times10^{-4}$ |

periodic 与 constant-reference 两套 detector stack 的 relative L2 为：

$$
0.290201.
$$

这说明边界会显著改变 forward data，不能只改 reconstruction boundary。两组 matched control 都能正常下降到约 $10^{-5}$ loss；故意 mismatched 组停在约 $0.198$。因此原 $2.76\%$ floor 不是 periodic boundary 本身造成的，而是旧 residual formula；同时，periodic model 仍是阶段性理想假设，不能外推为真实有限样品的永久设计。

### 14.9 known-B 单条 200/500/1000 trajectory

真实 plus/minus probe case separation 为：

$$
s_P
=
1.2456589\times10^{-5}.
$$

真实 detector-amplitude case separation 为：

$$
s_I
=
9.2478185\times10^{-6}.
$$

同一初始化、同一 shuffle sequence、同一 `beta_probe=0.5`、无 normalization 的连续轨迹结果如下。

| Iteration | frozen loss | aligned probe error | probe error / $s_P$ | frozen loss / $s_I$ |
|---:|---:|---:|---:|---:|
| 200 | $5.34469\times10^{-8}$ | $1.53769\times10^{-6}$ | $0.12344$ | $5.7794\times10^{-3}$ |
| 500 | $1.13126\times10^{-10}$ | $4.41160\times10^{-9}$ | $3.5416\times10^{-4}$ | $1.2233\times10^{-5}$ |
| 1000 | $4.52558\times10^{-14}$ | $1.71050\times10^{-12}$ | $1.3732\times10^{-7}$ | $4.8937\times10^{-9}$ |

1000 轮最后 20 轮的 loss slope 为：

$$
-8.15350\times10^{-16}\ \mathrm{iteration}^{-1}.
$$

因此已达到数值平台。最重要的结论是：**修正后的 optimizer 在 known-B 条件下可以把 probe error 压到 true waist signal 以下；历史 $0.9885\%$ 不是模型固有 floor，也不是 ePIE 原理上无法恢复 probe。**

raw probe 与 truth 的直接 relative L2 仍约为 $1.91$，这是 global complex phase/affine phase-ramp gauge 的表现。本表的 aligned error 只位于 `simulation_evaluation_only`，没有用来更新 probe。

### 14.10 修正后 blind 60 轮结果，以及为什么 Stage D 仍未通过

原七组 60-iteration ablation 使用新 adjoint-residual update 重跑。关键结果如下。

| Variant | frozen loss | aligned probe error | aligned B error | probe error / true signal |
|---|---:|---:|---:|---:|
| blind unconstrained | $3.47486\times10^{-4}$ | $6.57318\times10^{-4}$ | $8.37969\times10^{-3}$ | $52.7687$ |
| blind + energy norm | $5.89967\times10^{-4}$ | $6.63266\times10^{-4}$ | $8.59776\times10^{-3}$ | $53.2462$ |
| blind legacy coarse A | $1.83929\times10^{-1}$ | $3.15067\times10^{-1}$ | $1.77448\times10^{-1}$ | $25293.2$ |
| blind radial B-range | $3.33822\times10^{-4}$ | $2.34902\times10^{-5}$ | $8.06900\times10^{-3}$ | $1.88577$ |
| blind radial A-adjoint | $1.11049\times10^{-1}$ | $1.54095\times10^{-1}$ | $1.38277\times10^{-2}$ | $12370.6$ |

与权威历史 run `20260806_205853` 相比：

- radial B-range probe error 从 $2.58122\times10^{-4}$ 降到 $2.34902\times10^{-5}$，改善约 $10.99$ 倍；
- floor/signal ratio 从 $20.72$ 降到 $1.88577$；
- known-probe 60 轮 B error 改善约 $3.53$ 倍；
- known-B 60 轮 probe error 改善约 $25.12$ 倍。

但是 blind radial B-range 的 error 仍是真实 signal 的 $1.88577$ 倍，大于 gate $1$。其最后 20 轮 loss slope 为：

$$
-2.40446\times10^{-6}\ \mathrm{iteration}^{-1},
$$

尚未进入平台。脚本因此继续正确记录：

```text
not_run_reconstruction_floor_exceeds_true_signal
```

没有运行 blind recovered plus/minus finite difference。Stage D 保持 `Inconclusive`。

这里需要区分：

1. known-B control 已证明修正后的 probe optimizer 可以低于 true signal；
2. blind problem 同时需要恢复 B，并存在 probe/object ambiguity；
3. known-B 成功不能替代 blind Stage D gate；
4. 当前 blind 60 轮仍在明显下降，不能称为已验证的 blind 渐近 floor。

### 14.11 Stage A--C 和 observability 没有被本轮修改破坏

正式 run 的 Stage A--C 仍为 `Passed`。

| Metric | 结果 |
|---|---:|
| diameter profile max error | $0$ |
| analytic fill-path max error | $5.13231\times10^{-10}\,\mathrm m$ |
| analytic fill-path RMSE | $4.94716\times10^{-11}\,\mathrm m$ |
| transmission complex relative error | $0$ |
| zero contrast max error | $0$ |
| reference region max $|T-1|$ | $0$ |
| pure-phase amplitude max error | $2.22045\times10^{-16}$ |
| dz convergence change | $0.03587\%$ |
| dx convergence change | $0.02738\%$ |
| radial-source convergence change | $0.78973\%$ |
| radial-output convergence change | $0.01562\%$ |
| $\Delta D$ step convergence change | $0.72946\%$ |

true-probe 和 detector 指标：

| Metric | 结果 |
|---|---:|
| normalized waist probe sensitivity | $0.847969$ |
| minus/plus gauge-aligned probe difference | $1.24564\times10^{-5}$ / $1.24568\times10^{-5}$ |
| normalized intensity sensitivity | $0.970532$ |
| minus/plus intensity relative L2 | $1.42318\times10^{-5}$ / $1.42762\times10^{-5}$ |
| median / maximum frame sensitivity | $0.969673$ / $0.992489$ |

gauge-projected local Jacobian：

| Metric | 结果 |
|---|---:|
| singular values | $[5134.01,\ 552.877,\ 303.212]$ |
| condition number | $16.9321$ |
| smallest/largest ratio | $0.0590594$ |
| numerical rank | $3$ |
| max $|\operatorname{corr}(D_{\mathrm{waist}},\text{other})|$ | $0.0875763$ |

因此顶层 `Inconclusive` 仍只来自 Stage D blind quantitative recovery gate，不来自 projected-phase、sampling convergence、detector sensitivity 或 local Jacobian rank 失败。

### 14.12 run、失败记录和产物审计

第一次正式尝试：

```text
E:\tgv_ptycho_sim\runs\exp030_TGV_2d_effective_phase_20260809_193854
```

它完成了数值计算，但旧 `plot_loss_curves()` 要求所有曲线等长，遇到 60/1000 轮曲线后抛出：

```text
ValueError: All loss curves must have the same iteration count.
```

该目录原样保留为失败诊断 run，没有被覆盖，也不作为权威结果。

修复绘图并新增 ragged-length test 后，成功 run 为：

```text
E:\tgv_ptycho_sim\runs\exp030_TGV_2d_effective_phase_20260809_204330
```

成功 run 总耗时约 $56.7\,\mathrm{min}$。optimizer study 内部耗时约 $41.96\,\mathrm{min}$，没有触发 $6000\,\mathrm s$ budget。

HDF5：

```text
E:\tgv_ptycho_sim\runs\exp030_TGV_2d_effective_phase_20260809_204330\outputs\exp030_effective_phase.h5
```

文件大小为 `751834488` bytes。`/entry` 仍只有：

```text
config_yaml
data
instrument
metadata
metrics
reconstruction
sample
truth
```

没有 calibration 或 preprocessing group。抽查的主要数组均为 finite，外部 JSON 与 HDF5 中的 selected beta 和 checkpoint loss 一致。

新增 optimizer-study 字段位于实验专属 reconstruction 子组，没有改变项目级顶层 schema：

| HDF5 字段 | shape | dtype | 单位/语义 |
|---|---:|---|---|
| `.../optimizer_study/truth_fixed_point/<case>/P_B_after_one_iteration` | `(384,384)` | complex128 | B-plane field，a.u. |
| `.../optimizer_study/truth_fixed_point/<case>/B_after_one_iteration` | `(384,384)` | complex128 | B transmission，1 |
| `.../known_probe_object_only/checkpoints/<iter>/B_rec` | `(384,384)` | complex128 | raw B checkpoint，1 |
| `.../known_b_probe_only/screening/<beta>/P_B_rec_raw` | `(384,384)` | complex128 | raw screened probe，a.u. |
| `.../known_b_probe_only/normalization/<mode>/P_B_rec_raw` | `(384,384)` | complex128 | normalization control probe，a.u. |
| `.../known_b_probe_only/selected_trajectory/checkpoints/{200,500,1000}/P_B_rec` | `(384,384)` | complex128 | raw probe checkpoint，a.u. |
| `.../selected_trajectory/checkpoints/<iter>/B_rec` | `(384,384)` | complex128 | fixed truth-B diagnostic copy，1 |
| `.../checkpoints/<iter>/data_fidelity_loss` | scalar | float64 | frozen relative detector-amplitude loss，1 |
| `.../checkpoints/<iter>/simulation_evaluation_only/P_B_aligned_complex_relative_error` | scalar | float64 | truth-only evaluation，1 |
| `/entry/metrics/.../optimizer_study/...` | scalar/tree | numeric/string | 与外部 `metrics.json` 同语义 |

baseline 和原 truth 字段保持：

- `/entry/data/I_stack`: `(49,384,384)` float64，intensity a.u.；
- `/entry/data/scan_positions`: `(49,2)` float64，m，列为 `(x,y)`；
- `/entry/truth/P_B_true`: `(384,384)` complex128，a.u.；
- `/entry/truth/B_true`: `(384,384)` complex128，1；
- `/entry/truth/parameter_sweep/P_B_true`: `(3,384,384)` complex128；
- `/entry/truth/parameter_sweep/I_stack_true`: `(3,49,384,384)` float64。

所有 13 张 PNG 已生成并人工检查。`loss_curves.png` 现在能同时显示 60 和 1000 轮；`recovered_probe_cases.png`、`probe_sensitivity_maps.png`、`intensity_sensitivity.png`、Jacobian 和 convergence 图均可读。PNG 只用于人工检查，数值结果均已写入 HDF5/metrics。

### 14.13 验证结果

共享重建/传播修改后执行 exp020 回归：

```text
E:\tgv_ptycho_sim\runs\exp020_A_thin_phase_probe_recovery_20260809_193220
```

结果：

```text
final loss = 3.656671e-16
A phase RMSE = 5.263150e-16 rad
```

完整 pytest：

```text
55 passed in 1.70s
```

本轮修改 Python 文件 Ruff：

```text
All checks passed!
```

项目级 Ruff 诊断仍有 11 项既有问题：

- `scripts/run_exp001_forward.py` 的 8 个 `E402`；
- `src/tgv_ptycho/calibration/stage.py` 的 1 个 `E501`；
- `src/tgv_ptycho/recon/losses.py` 的 1 个 `E501`；
- `src/tgv_ptycho/recon/rpie.py` 的 1 个 `E501`。

它们不在本轮修改范围，没有顺手清理。此前 `angular_spectrum.py` 中用户已有的两行长 docstring 已在保留语义的前提下换行，因此项目级既有问题由 13 项变为 11 项。

### 14.14 当前结论和下一步最好的选择

本轮可以明确下结论：

1. 历史 known-probe B error 约 $2.76\%$ 的主因是 ePIE 对 band-limited ASM 使用了错误的 inverse-difference residual；
2. 新 adjoint-residual update 恢复 exact fixed point；
3. known-B 条件下 200 轮已低于 true waist probe signal，1000 轮达到 $10^{-12}$ 级 aligned probe error；
4. 因此 projected-phase forward 并非没有可恢复 waist information，当前 probe optimizer 也不是原理上失效；
5. 修正使 blind radial B-range error 改善约 10.99 倍，离 gate 只剩约 1.886 倍；
6. 但 known-B 成功不能替代 blind recovery，Stage D 仍为 `Inconclusive`；
7. 本轮不能外推到 noise、tilt、多孔、3D multislice 或真实实验 detectability。

下一步最好的选择仍是继续 exp030，而不是立即跳到 exp031 或 exp040：

1. 对 `blind_radial_output_range_constraint` 运行一条 200/500/1000 checkpoint trajectory；当前 60 轮 tail 仍明显下降，而且 floor 只比 signal 高 1.886 倍，这是最有信息量、最直接的下一步；
2. 如果 blind baseline 在某个 checkpoint 低于 signal gate，再对 baseline/minus/plus 三个 case 使用同一 optimizer 设置，执行 recovered sensitivity ordering；
3. 若 blind 长程仍停在 signal 以上，再比较 batch Wirtinger/Adam 或低维 TGV model-constrained fitting，而不是继续无目的增加 ePIE 轮数；
4. 单孔 ideal Stage D 通过后，再新开 exp031 做多孔阵列和孔间相干；
5. exp040 仍用于真正的 3D multislice、内部横向传播与 depth ordering，不能用来回避当前 blind ambiguity；
6. noise、tilt、calibration error 和 parametric fitting 应作为后续独立任务，并保持 truth-free reconstruction 边界。

因此本轮最准确的一句话总结是：**ePIE 的决定性 operator bug 已修复，known controls 已通过 signal floor；blind radial B-range 从高于 signal 20.72 倍改善到 1.886 倍，但尚未过 gate。下一步应继续同一 exp030 做 blind 200/500/1000 长程 checkpoint，而不是马上转向 exp031 或 exp040。**

---

## 15. Blind 200/500/1000 长轨迹、可恢复 checkpoint 与 Stage D 完成（2026-08-10）

### 15.1 这次要解决的最后一个问题

第 14 节已经排除了 forward operator 和 ePIE adjoint 写错这两个决定性问题，但当时只把 blind radial B-range reconstruction 跑到 60 轮。60 轮结果为：

```text
aligned probe error / true waist-probe signal = 1.88576695
```

该比值仍大于门限 1，因此当时不能判断恢复后的 probe 是否真的保留了 `D_waist` 的微小差异。不过 loss 的尾部斜率仍明显为负，这更像“迭代尚未结束”，而不是已经碰到不可辨识 floor。

本轮因此不再更换物理模型，也不再扫描新的 optimizer，而是执行最直接的检验：

1. 对同一个 `blind_radial_output_range_constraint` baseline 连续运行 1000 轮；
2. 在累计 200、500、1000 轮保存 checkpoint；
3. 使用同一主 gate 检查 probe reconstruction floor；
4. 如果通过，采用最早通过点的迭代数，对 baseline、waist minus、waist plus 做严格 matched reconstruction；
5. 用 common-gauge recovered finite difference 检查灵敏度大小和正负扰动排序。

这里“同一条轨迹”很重要。不能把 200 轮结果重新作为初值、重新设置同一个 seed 后称为 200→1000 的连续轨迹，因为 scan shuffle 的随机数生成器已经前进。重新初始化 RNG 会产生另一条 permutation history。

### 15.2 为何先补 checkpoint/resume，而不是直接跑 1000 轮

此前 checkpoint 只存在于 `epie_reconstruct()` 返回值的内存中。若客户端或进程在 500 轮后中断，已经完成的计算仍会全部丢失。本轮在 `src/tgv_ptycho/recon/epie.py` 中加入显式 optimizer state，保存：

- 当前 probe 和 object；
- 已完成的累计迭代数；
- 从第 1 轮开始的完整 loss curve；
- 初始 frozen data-fidelity loss；
- NumPy bit-generator 类型和完整 RNG state；
- 对数据、scan、传播和 optimizer 设置的 problem signature。

恢复时 `num_iters` 仍表示累计终点。例如，从 500 恢复到 1000，只执行迭代 501–1000。恢复路径还必须跳过初始 probe normalization 和初始 radial range projection，否则恢复点会被多投影一次，不再与不中断轨迹一致。

`scripts/run_exp030_effective_phase.py` 进一步实现：

- `--resume-blind-checkpoint <path>`；
- checkpoint 先写临时 HDF5，再原子 rename；
- 已存在的 checkpoint 一律拒绝覆盖；
- 恢复总是在新的 timestamped run 中进行，不修改来源 run；
- runner signature 额外覆盖 constraint 类型、ridge、初始化 seed 和边界设置；
- 每个 checkpoint 写完立即更新 `blind_long_progress.json`。

小网格测试证明：启用 shuffle 时，一次连续运行与“中途落盘、重新加载 RNG state、继续运行”的 probe、object 和完整 loss curve 均逐元素相同。也测试了 signature mismatch、损坏的 loss history 和 checkpoint 覆盖拒绝。

本次正式 run 虽然经历多次客户端断联，但后台进程没有退出，因此 `resume_provenance.resumed=false`。200、500、1000 和 matched-case checkpoint 均实际落盘；若进程退出，已具备从这些文件新开 run 精确续跑的能力。

### 15.3 本轮固定的 reconstruction 设置

本轮没有把 known-B screening 中的最优参数偷换给 blind problem。三条 blind case 保持完全配对：

| 设置 | 值 |
|---|---:|
| variant | `blind_radial_output_range_constraint` |
| `beta_probe` | 0.08 |
| `beta_object` | 0.5 |
| correction | `adjoint_residual` |
| denominator | `epie` |
| B amplitude bounds | `[1, 1]` |
| B boundary | periodic |
| probe normalization | none |
| range ridge fraction | $10^{-13}$ |
| shuffle seed | 20260733 |
| B initialization seed | 20260734 |
| baseline checkpoints | 200, 500, 1000 |

sample B、scan positions、所有初始化策略和随机 seed 在 baseline/minus/plus 间相同。truth 不进入 update；truth 只用于 `simulation_evaluation_only` 的 gate、误差和 common-gauge 评估。

为了避免无意义地重复 2026-08-09 已完成的约 42 分钟 known-B optimizer study，本轮 config 将 `optimizer_study.enabled` 设为 `false`，但保留七组 60 轮 operator-consistency 对照。

### 15.4 baseline 200/500/1000 的实际结果

新 run：

```text
E:\tgv_ptycho_sim\runs\exp030_TGV_2d_effective_phase_20260810_121124
```

baseline long trajectory 耗时约 1976.99 s；三点结果如下：

| Iteration | aligned probe error | probe error / true signal | aligned B error | frozen loss | frozen loss / detector signal | tail slope / iter |
|---:|---:|---:|---:|---:|---:|---:|
| 200 | $4.56649\times10^{-6}$ | 0.366592 | 0.00606901 | $1.26861\times10^{-4}$ | 13.7179 | $-1.71481\times10^{-7}$ |
| 500 | $1.74175\times10^{-6}$ | 0.139825 | 0.00497531 | $6.35328\times10^{-5}$ | 6.87003 | $-3.29499\times10^{-8}$ |
| 1000 | $1.14761\times10^{-6}$ | 0.0921286 | 0.00428018 | $3.86639\times10^{-5}$ | 4.18087 | $-1.16612\times10^{-8}$ |

结论有两层：

1. 200 轮已经满足严格主 gate，即 probe reconstruction floor 小于 true minus/plus probe separation；
2. 200→500→1000 的 probe error、B error 和 frozen loss 均继续下降，说明 200 轮通过不是一次偶然 crossing。

因此 matched sensitivity 使用最早通过点 200，而不是用 1000 轮结果人为降低 baseline error 后再与只跑较少轮的 plus/minus 比较。

### 15.5 matched baseline/minus/plus 恢复结果

三条 case 均从相同策略的 measurement-derived probe init 和相同 B init 出发，并运行 200 轮：

| Case | initial frozen loss | final frozen loss | aligned probe error | aligned B error |
|---|---:|---:|---:|---:|
| baseline | 0.308752 | $1.26861\times10^{-4}$ | $4.56649\times10^{-6}$ | 0.00606901 |
| waist minus | 0.308752 | $1.26861\times10^{-4}$ | $4.56665\times10^{-6}$ | 0.00606902 |
| waist plus | 0.308752 | $1.26860\times10^{-4}$ | $4.56631\times10^{-6}$ | 0.00606901 |

三条 reconstruction floor 几乎相同，没有出现某个几何 case 单独不收敛。评估时先把 recovered minus/plus 对齐到 recovered baseline，再仅使用一次 baseline-to-truth anchor；该 anchor 位于 `simulation_evaluation_only`，不反馈给 reconstruction。

| 灵敏度量 | True | Recovered |
|---|---:|---:|
| normalized probe sensitivity | 0.847968838 | 0.847968536 |
| minus difference | $1.24564105\times10^{-5}$ | $1.24564068\times10^{-5}$ |
| plus difference | $1.24567670\times10^{-5}$ | $1.24567615\times10^{-5}$ |

结果为：

```text
recovered-to-true sensitivity relative deviation = 3.55915e-7
sensitivity ordering matches truth = true
```

因此本轮的配置化 Stage D 判据通过，Stage D 和整个 exp030 的 run status 均为 `Passed`。

### 15.6 仍需保留的一个重要限制：absolute frozen loss 还高于 detector signal

1000 轮时，absolute frozen detector-amplitude loss 仍是 true detector-amplitude case separation 的 4.18 倍。它没有被隐藏，也没有改写成通过项。

这与 recovered differential sensitivity 已通过并不矛盾：

- 主 gate 检查的是 gauge-aligned probe error 是否小于 true probe case separation；
- matched finite difference 使用三条相同 optimizer trajectory，三者共有的大部分 reconstruction bias 会作为 common mode 抵消；
- loss curve 内每轮记录的是顺序更新时的 frame-wise loss，而 `final_data_fidelity_loss` 是冻结 probe/object 后重新遍历完整 stack 的结果，两者不能当作同一个量；
- 当前结果支持“配对、无噪声、同 optimizer 的局部差分可恢复”，不支持“每个 blind solution 的绝对 detector residual 已低于微小几何信号”。

若未来把“absolute frozen loss / detector signal < 1”设为更严格的新验收门限，则仍需继续研究 simultaneous/batch optimizer 或更强的物理参数化；本轮没有通过更改 gate 定义来制造结论。

### 15.7 HDF5 和 checkpoint 产物

主 HDF5：

```text
E:\tgv_ptycho_sim\runs\exp030_TGV_2d_effective_phase_20260810_121124\outputs\exp030_effective_phase.h5
```

文件大小为 528,608,112 bytes。`/entry` 顶层只有：

```text
config_yaml
data
instrument
metadata
metrics
reconstruction
sample
truth
```

没有生成 calibration 或 preprocessing group。新增长轨迹字段位于实验专属 reconstruction 子树，不改变项目级顶层 schema：

| 字段 | shape | dtype | 单位/语义 |
|---|---:|---|---|
| `.../blind_long_study/baseline/P_B_rec_raw` | `(384,384)` | complex128 | B-plane field, a.u. |
| `.../baseline/B_rec_raw` | `(384,384)` | complex128 | dimensionless complex transmission |
| `.../baseline/loss_curve` | `(1000,)` | float64 | sequential relative detector-amplitude loss |
| `.../baseline/checkpoints/{200,500,1000}/P_B_rec` | `(384,384)` | complex128 | raw checkpoint probe, a.u. |
| `.../baseline/checkpoints/{200,500,1000}/B_rec` | `(384,384)` | complex128 | raw checkpoint B, dimensionless |
| `.../baseline/checkpoints/<iter>/data_fidelity_loss` | scalar | float64 | frozen relative detector-amplitude loss |
| `.../cases/<case_id>/P_B_rec_raw` | `(384,384)` | complex128 | matched raw probe, a.u. |
| `.../cases/<case_id>/B_rec_raw` | `(384,384)` | complex128 | matched raw B, dimensionless |
| `.../cases/<case_id>/loss_curve` | `(200,)` | float64 | matched sequential loss |
| `.../cases/<case_id>/simulation_evaluation_only/P_B_rec_common_gauge` | `(384,384)` | complex128 | truth-evaluation-only common-gauge field |
| `.../selected_sensitivity_check/...` | scalar/tree | numeric/string | gate、ordering 和 recovered sensitivity |

外部可恢复 checkpoint 位于：

```text
checkpoints/blind_long/baseline/iter_{0200,0500,1000}.h5
checkpoints/blind_long/waist_minus/iter_0200.h5
checkpoints/blind_long/waist_plus/iter_0200.h5
```

每个文件约 4.74 MB，包含 complex128 probe/object、累计 loss、迭代数、RNG state、problem/runner signatures、constraint diagnostic 和 `simulation_evaluation_only` checkpoint metrics。所有 blind-long 数值 dataset 均已检查为 finite。

13 张 PNG 均已生成。重点人工检查了 `recovered_probe_cases.png` 和 `loss_curves.png`：baseline 1000 与 matched 三 case 图可读，长短 loss 曲线能同时显示，标题、坐标和 colorbar 未遮挡数据。

### 15.8 测试、回归和状态

本轮新增或扩展的测试覆盖：

- shuffled ePIE 连续运行与磁盘恢复逐元素一致；
- callback snapshot 不污染 optimizer；
- no-op resume；
- problem-signature 和 loss-history 拒绝；
- checkpoint HDF5 round trip、SHA256 和拒绝覆盖；
- 小网格 blind-long baseline gate、matched cases 和从 baseline checkpoint 恢复。

完成本轮代码后：

```text
pytest: 63 passed
modified-scope Ruff: All checks passed!
```

项目级 `ruff check src scripts tests` 仍报告 11 项与本轮无关的既有问题：`scripts/run_exp001_forward.py` 的 8 项 `E402`，以及 `stage.py`、`losses.py`、`rpie.py` 各 1 项 `E501`。本轮没有越界清理这些文件。

公共 ePIE API 修改后的 exp020 回归 run：

```text
E:\tgv_ptycho_sim\runs\exp020_A_thin_phase_probe_recovery_20260810_131736
final loss = 3.656671e-16
A phase RMSE = 5.263150e-16 rad
```

未执行 `git add`、commit、push、PR 或 merge；用户原有 staged、unstaged、untracked 和 deleted 状态均未回退或清理。

### 15.9 本轮最终结论和下一步

现在可以把此前的问题定位得更准确：

1. 最初的主要失败来自 band-limited ASM 下错误的 ePIE residual/adjoint 组合；该问题已在第 14 节修复；
2. 修复后 60 轮仍未通过，不是 projected-phase 模型无信息，而是 blind optimizer 尚未迭代到足够低的 probe floor；
3. 同一条 200/500/1000 轨迹证明 floor 稳定下降，200 轮已经低于 true probe signal；
4. matched baseline/minus/plus 在 200 轮恢复出的 normalized sensitivity 与 truth 相差仅 $3.56\times10^{-7}$，且排序正确；
5. 因而 exp030 已完成“理想 projected-phase 单孔 TGV 的 local differential observability 和 paired blind recovery check”；
6. absolute frozen detector loss 仍高于 detector case separation，是后续 optimizer 研究的限制，不影响本轮已定义的 probe-differential gate，但必须在外推时保留。

下一步最好的选择是：

1. **优先新开 exp031 多孔阵列**：单孔 ideal pipeline 已通过，可以检查孔间相干、不同孔腰径的参数相关性，以及多孔时 radial single-hole constraint 如何推广；
2. 若继续 exp030，只建议做同模型下的补充验证，例如新增 seed robustness 或比较 batch optimizer 是否把 absolute frozen loss 降到 detector signal 以下，不再需要重复基本 observability 证明；
3. **新开 exp040** 做真正的 3D multislice，用于内部横向衍射、侧壁传播和 depth ordering；它不能被描述为 exp030 的简单修补；
4. noise、tilt、calibration error 和 parametric fitting 应分别新开后续任务，尤其不能用当前无噪声结果宣称真实实验检测极限。

一句话总结：**exp030 的单孔、无噪声 projected-phase Stage A–D 已按当前门限通过；下一步最有价值的是 exp031 多孔阵列，而 absolute detector residual、3D multislice、noise/tilt 和 parametric fitting 应作为清楚分离的后续问题。**
