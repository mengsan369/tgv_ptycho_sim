# exp040 R11：开放域、Helmholtz 色散与 Cartesian 角向各向异性

本文只记录可独立于某次 run 阅读的数学与物理内容；不记录实验 case、阈值、资源成本或运行结果。

## 1. 物理计算域与 PML 厚度是两个不同控制量

开放域 Helmholtz 问题常在物理区域外接 PML。把 PML 起点从半径 \(R_1\) 外移到 \(R_2\)，改变的是显式保留的
无损物理传播区域；把固定起点后的 PML 从 \(L_1\) 加厚到 \(L_2\)，改变的是吸收层截断误差。两者不能互相
替代。若散射场在 PML 起点前仍较强，PML 仍可能正确吸收外行波，但把 PML 前的场强行替换成 homogeneous fill
会污染随后进行的频带投影或有限视场比较。

因此，开放域数值参考至少需要同时检查：固定内部 comparison region 上的 domain-enlargement convergence、
PML 前物理 guard 中的散射场，以及固定物理域上的 mesh convergence。线性求解残差很小只证明离散方程被解得
准确，不证明离散方程或截断域逼近了连续开放域问题。

## 2. 高频 Helmholtz 的数值色散与污染误差

对常系数二维 Helmholtz 方程，等距网格上的标准五点 Laplacian 对平面波
\(\exp(i k_h(x\cos\theta+y\sin\theta))\) 给出离散色散关系

\[
\frac{4}{h^2}\left[
\sin^2\!\left(\frac{k_hh\cos\theta}{2}\right)+
\sin^2\!\left(\frac{k_hh\sin\theta}{2}\right)
\right]=k^2.
\]

即使局部插值误差不大，\(k_h\ne k\) 产生的相位误差也会随传播距离累积，形成 pollution error。每波长点数
是必要的分辨率描述，但不能代替实际 mesh-pair convergence。

## 3. 五点格式的渐近色散修正

Cocquet 与 Gander 提出的 asymptotic dispersion correction 在原有限差分 stencil 中使用 shifted wavenumber。
对二维等距五点格式，可用轴向与对角传播色散极值的中点定义

\[
k_{\mathrm{adc}}^2(k,h)=\frac12\left[
\frac{4}{h^2}\sin^2\!\left(\frac{kh}{2}\right)
+\frac{8}{h^2}\sin^2\!\left(\frac{kh}{2\sqrt2}\right)
\right].
\]

其小网格展开为

\[
k_{\mathrm{adc}}^2=k^2-\frac{k^4h^2}{16}+O(k^6h^4),
\]

并在 \(h\to0\) 时回到原连续波数。该修正保持五点紧致结构，目标是减小不同传播方向上的最大渐近色散，
并不把二阶格式自动变成严格的高阶真值。

在折射率跳变、轴对称 \(r^{-1}\partial_r\)、有限半径和复坐标 PML 中使用局部
\(k=k_0n(r,z)\) 是一个需要 mesh-pair 实证验证的推广。它必须同时用于离散 mass term 与 contrast-source 的
离散波数差，不能只修正左端而保留不相容的右端。若推广后的 mesh pair 不闭合，不能因为常系数色散理论成立
就把该结果称为已验证 reference。

## 4. 高阶 FEM 的理论位置

高频 Helmholtz 的 \(hp\)-FEM 分析表明，准最优性不仅依赖 \(kh\)，还依赖多项式阶数 \(p\)。在相应正则性和
解算子假设下，典型条件是 \(kh/p\) 足够小，并且 \(p\) 至少随 \(\log k\) 增长。高阶 FEM、CIP-FEM、
Trefftz/plane-wave elements 和 dispersion-minimizing compact differences 都是可行路线，但每一种都必须重新
验证介质跳变、轴条件、PML 弱形式、线性代数误差与共同输出映射；“方法阶数更高”本身不等于污染误差已受控。

## 5. annular-bin residual 不等于纯角向各向异性

把 Cartesian field 在有限宽度 annulus 内替换成常数均值后计算残差，会同时包含两项：真正的角向变化，以及
场在 annulus 宽度内的径向斜率。对快速振荡但严格轴对称的场，后一项也可能很大。因此更直接的角向诊断是在
固定半径 \(r_j\) 上以等角度节点采样

\[
U_{j\ell}=U(r_j,\theta_\ell),\qquad
\bar U_j=\frac1{N_\theta}\sum_\ell U_{j\ell},
\]

并计算

\[
\epsilon_\theta=
\left(
\frac{\sum_j r_j\sum_\ell|U_{j\ell}-\bar U_j|^2}
{N_\theta\sum_jr_j|\bar U_j|^2}
\right)^{1/2}.
\]

角向 Fourier 系数 \(m=4,8,\ldots\) 可进一步识别 square-grid imprint。极坐标插值本身也会产生 floor，故应先用
解析径向场做 manufactured control。annular mean 仍可在角向 gate 已闭合后用于轴对称跨模型比较，但不能拿来
隐藏未闭合的非轴对称残差。

## 6. 圆孔的保守 Cartesian cell average

对半径 \(a\) 的圆与一个 Cartesian pixel，可将交叠面积写成沿 \(x\) 的 chord-length 积分

\[
A_{\mathrm{cell}}=\int_{x_0}^{x_1}
\max\!\left(0,
\min(y_1,\sqrt{a^2-x^2})-
\max(y_0,-\sqrt{a^2-x^2})
\right)\,dx.
\]

只在与圆周相交的 pixel 上进行高阶 Gauss--Legendre 积分，可避免在每个 pixel 内使用固定的
\(q\times q\) square-node count。它仍是 analytic indicator 的数值 cell average，而不是额外的物理
effective-medium 假设。积分阶数必须用更高阶结果和全局面积守恒验证。

## 7. 参考文献

- P.-H. Cocquet and M. J. Gander, “Asymptotic Dispersion Correction in General Finite Difference Schemes for
  Helmholtz Problems,” *SIAM Journal on Scientific Computing* 46(2), 2024,
  https://doi.org/10.1137/22M1531142.
- J. M. Melenk and S. Sauter, “Wavenumber Explicit Convergence Analysis for Galerkin Discretizations of the
  Helmholtz Equation,” *SIAM Journal on Numerical Analysis* 49(3), 1210–1243 (2011),
  https://doi.org/10.1137/090776202.
- I. Babuška and S. A. Sauter, “Is the Pollution Effect of the FEM Avoidable for the Helmholtz Equation Considering
  High Wave Numbers?”, *SIAM Review* 42(3), 451–484 (2000), https://doi.org/10.1137/S0036142994269186.
- I. Singer and E. Turkel, “High-order finite difference methods for the Helmholtz equation,” *Computer Methods in
  Applied Mechanics and Engineering* 163, 343–358 (1998), https://doi.org/10.1016/S0045-7825(98)00023-1.
- H. Dastour and W. Liao, “A generalized optimal fourth-order finite difference scheme for a 2D Helmholtz equation
  with the perfectly matched layer boundary condition,” *Journal of Computational and Applied Mathematics* 394,
  113544 (2021), https://doi.org/10.1016/j.cam.2021.113544.
