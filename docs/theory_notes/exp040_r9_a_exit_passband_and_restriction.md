# exp040 理论：A-exit 共同可传播频带与网格 restriction

本文只记录可脱离具体实验参数成立的 Fourier 投影、误差能量分解和 nested-grid restriction 理论。具体
网格、case、阈值、运行结果和状态判定由 `exp040` 实验设计文档记录。

## 1. Raw field norm 与下游可传播频带 norm

设一个平面上的复场为 $U(x,y)$。离散 raw relative error

$$
\varepsilon_{\mathrm{raw}}
=\frac{\lVert U_1-U_2\rVert_2}{\lVert U_2\rVert_2}
$$

对该网格能够表示的全部频率同等计权。它适合检查离散场本身是否收敛，但并不区分其中哪些横向频率能在
后续均匀介质中传播。

对真空波长 $\lambda_0$、介质折射率 $n$，标量角谱的可传播集合为

$$
\Omega_n=\left\{(f_x,f_y): f_x^2+f_y^2\le(n/\lambda_0)^2\right\}.
$$

超出该圆盘的分量在该介质的标量 Helmholtz 模型中是 evanescent，而不是远距离传播分量。若研究的是从一个
高折射率介质出射到较低折射率外部介质后的下游 observable field，则应明确使用外部介质的
$\Omega_{n_{\mathrm{ext}}}$；使用内部介质 cutoff 会回答不同问题。

## 2. 离散正交 passband 投影

在规则周期网格上，令 $M_\Omega[k,l]\in\{0,1\}$ 是物理频率集合与该网格 Nyquist rectangle 的交集。
离散投影定义为

$$
P_\Omega U
=\mathcal F^{-1}\!\left[M_\Omega\,\mathcal F U\right].
$$

在离散 Fourier 基下，binary real mask 满足 $P_\Omega^2=P_\Omega$，且在一致的 FFT normalization 下是
正交投影。共同 passband error 可写为

$$
\varepsilon_\Omega
=\frac{\lVert P_\Omega U_1-P_\Omega U_2\rVert_2}
{\lVert P_\Omega U_2\rVert_2}.
$$

比较不同 native samplings 时，物理 cutoff、边界包含规则和 frequency-bin convention 必须预先固定。
只写“低通后比较”而不记录 cutoff 不能形成可复现诊断。

## 3. Parseval attribution

令差场 $D=U_1-U_2$。正交投影给出

$$
D=P_\Omega D+(I-P_\Omega)D,
$$

以及 Parseval/Pythagoras 分解

$$
\lVert D\rVert_2^2
=\lVert P_\Omega D\rVert_2^2
+\lVert(I-P_\Omega)D\rVert_2^2.
$$

因此可报告

$$
\eta_{\mathrm{in}}
=\frac{\lVert P_\Omega D\rVert_2^2}{\lVert D\rVert_2^2},
\qquad
\eta_{\mathrm{out}}
=\frac{\lVert(I-P_\Omega)D\rVert_2^2}{\lVert D\rVert_2^2},
$$

并检查 $\eta_{\mathrm{in}}+\eta_{\mathrm{out}}=1$。这能描述 discrepancy 的频谱位置，但不能证明频带外
误差没有近场意义，也不能把频带内误差自动归因于某一种物理缺陷。

## 4. Native-grid projection 应先于 decimation

若 fine field 直接 decimate 到 coarse grid，fine Nyquist 与 coarse Nyquist 之间的分量可能 alias 到 coarse
频带。为了比较一个预先定义的共同物理频带，应先在各自 native grid 上应用相同物理 cutoff 的投影，再执行
restriction：

$$
U_{f,\Omega}=P_{\Omega,f}U_f,
\qquad
U_{c,\Omega}=P_{\Omega,c}U_c,
\qquad
R(U_{f,\Omega})\ \text{vs.}\ U_{c,\Omega}.
$$

“先 restriction 再挑选一个使误差较小的 cutoff”会混入 alias 与看后选择，不能用于预注册收敛结论。

## 5. Conservative cell-average restriction

对严格 nested、cell-centered、每轴 refinement ratio 为整数 $r$ 的二维网格，complex cell-average
restriction 为

$$
(R_{\mathrm{avg}}U_f)_{j,i}
=\frac{1}{r^2}
\sum_{a=0}^{r-1}\sum_{b=0}^{r-1}
(U_f)_{rj+a,\,ri+b}.
$$

所有权重非负且和为一，因此它精确保持常数，并保持离散面积积分：

$$
\Delta x_c^2\sum_{j,i}(R_{\mathrm{avg}}U_f)_{j,i}
=\Delta x_f^2\sum_{m,n}(U_f)_{m,n},
\qquad \Delta x_c=r\Delta x_f.
$$

这里平均的是 complex field，不是 intensity；二者不可互换。

## 6. 与 centered bilinear sampling 的关系

对偶数 shape、相同 FOV、严格 2:1 nested cell-centered grids，每个 coarse center 恰位于对应 2×2 fine
centers 的几何中心。此时 separable bilinear interpolation 的四个权重均为 $1/4$，所以在精确算术下

$$
R_{\mathrm{bilinear}}U_f=R_{\mathrm{avg}}U_f.
$$

这个等价关系不是普遍定理。只要 grid origin、FOV、奇偶 shape、refinement ratio、边界 extension 或 sampling
location 改变，两者就可能不同。实现仍应显式比较它们，而不能根据一个特例把 bilinear interpolation 永久
称为 conservative restriction。

## 7. 解释边界

共同可传播频带内收敛说明：在指定标量传播介质与离散投影定义下，下游可传播频率的 A-exit representation
已受控。它不等于：

- raw near-field 已收敛；
- evanescent 或高空间频率分量没有物理意义；
- glass/air Fresnel transmission、reflection、backward wave 或 polarization 已建模；
- 真实 sample、illumination 或 detector 已标定。

反之，共同频带内仍有差异也不能单独证明必须采用 full-wave 模型；应先排除 axial splitting、lateral
voxelization、boundary、restriction 和其它数值误差，再预注册更完整的物理 comparator。
