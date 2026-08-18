# exp040 R10：双向标量 Helmholtz comparator 的理论边界

本文只记录对应 exp040 R10 的通用数学与物理内容，不记录实验 case、阈值、运行成本或结果。

## 1. 从单向 multislice 到双向 Helmholtz

对单色标量场 $u(\mathbf r)$，各向同性、无源、非磁性介质中的频域标量 Helmholtz 方程为

\[
\left[\nabla^2+k_0^2n^2(\mathbf r)\right]u(\mathbf r)=0,
\qquad k_0=\frac{2\pi}{\lambda_0}.
\]

单向 multislice / BPM 将纵向传播因子化后只保留 forward branch，并通过薄层 transmission 与层间传播近似
折射率变化。它适合以 forward scattering 为主的传播，但不会显式求解由介质突变产生的 backward branch。
直接求解 Helmholtz boundary-value problem 则同时允许 forward 与 backward components，因此可以作为判断
反射/回波是否足以改变 A-exit 场的最小复杂物理 comparator。

该 comparator 仍是标量模型；它没有包含偏振、纵向电场、矢量边界条件或各向异性。它与 vector Maxwell
solver 之间仍有明确的模型层级差异。

## 2. 轴对称形式

当折射率 $n(r,z)$、几何和标量入射场均绕 $z$ 轴对称时，方程可写成

\[
\frac{\partial^2u}{\partial r^2}
+\frac{1}{r}\frac{\partial u}{\partial r}
+\frac{\partial^2u}{\partial z^2}
+k_0^2n^2(r,z)u=0,
\]

并在 $r=0$ 使用 regularity condition $\partial_r u=0$。轴对称降维保留了标量模型中的双向传播与曲面
散射，同时避免构造完整三维 Helmholtz volume。若实际照明、材料或孔形破坏轴对称，则该降维不再成立。

## 3. 入射场、散射场与开放边界

常用写法为 $u=u_{\mathrm{inc}}+u_{\mathrm{scat}}$。入射场应在选定的 homogeneous reference medium 中
满足对应 Helmholtz 方程；散射场在外边界满足 outgoing-wave condition。有限计算域通常使用 PML，通过复坐标
拉伸使外行波在不产生理想反射的条件下衰减。

PML 是数值近似而不是物理吸收层。必须通过 mesh refinement、PML thickness/strength variation 和
homogeneous-medium analytic solution 检查其数值 floor，不能把边界反射误认为 TGV 的 backward scattering。

## 4. 固定 homogeneous normalization

若 comparator 包含共同的平面入口/出口界面，直接比较 total fields 会同时包含与 TGV 无关的 Fresnel
amplitude 和 phase。可在每个模型内部使用固定的 homogeneous control：

\[
v(\mathbf r_\perp)
=\frac{u_{\mathrm{TGV}}(\mathbf r_\perp)}
       {u_{\mathrm{hom}}(\mathbf r_\perp)}.
\]

该比值必须在预先固定的同一 observation plane 计算，且 denominator 不得接近零。它不是从结果拟合的 global
phase/scale alignment，而是由同一模型、同一边界和同一入射场定义的物理 control。原始 total fields 仍应
保留用于检查 normalization 是否掩盖异常。

## 5. 共同外部可传播频带

位于均匀外部介质中的二维 transverse field 只有满足

\[
f_x^2+f_y^2\le\left(\frac{n_{\mathrm{ext}}}{\lambda_0}\right)^2
\]

的频率分量能够作为传播平面波到达远处。跨模型比较可在各自 native representation 上先投影到这一固定物理
频带，再映射到共同 sampling。该投影只能区分外部 propagating 与 evanescent content，不能证明频带内差异
来自 backward wave、vector effect 或任意单一机制。

## 6. 轴对称加权范数与非轴对称残差

轴对称场的二维面积积分满足

\[
\lVert u\rVert_2^2
=2\pi\int_0^{R}|u(r)|^2r\,dr.
\]

因此径向 comparator 应使用 $2\pi r$ 权重。若 Cartesian multislice field 被映射到轴对称 reference，先计算
其 azimuthal mean $\bar u(r)$，并单独报告

\[
\frac{\lVert u-\bar u\rVert_2}{\lVert\bar u\rVert_2}
\]

作为 grid anisotropy diagnostic。若该残差未收敛，径向平均会掩盖 lateral discretization，跨模型归因应被
阻断。

## 7. 可得与不可得的结论

经过自身数值收敛验证的双向标量 Helmholtz 与单向 multislice 若出现稳定差异，可以说明 backward/reflection
scalar physics 在所选输出和频带内具有可分辨影响。它不能单独证明真实实验误差，也不能代替 vector Maxwell、
材料色散/损耗、表面粗糙度、真实三维非轴对称几何或标定不确定性的评估。

## 8. Contrast-source scattered-field 形式

当计算域同时含有无限延伸的平面背景场和局域 TGV 扰动时，直接让 total field 进入径向 PML 会把本来不随
半径衰减的平面波误当成径向外行散射波。更合适的分解是

\[
u=u_{\mathrm{bg}}+u_{\mathrm{s}},\qquad
L_{\mathrm{TGV}}u_{\mathrm{s}}
=-\left(L_{\mathrm{TGV}}-L_{\mathrm{bg}}\right)u_{\mathrm{bg}},
\]

其中 $u_{\mathrm{bg}}$ 是同一平面玻璃—空气界面的解析解，$u_{\mathrm{s}}$ 才满足径向和轴向 outgoing
condition。由于两算子只在 TGV 改变折射率的位置不同，右端是局域的 contrast source。该写法保留 total
field 的物理意义，同时使 PML 只吸收真正的散射场。

对采用 $\exp(-i\omega t)$ 约定、从玻璃侧正入射的标量界面，令 $k_g=k_0n_g$、$k_a=k_0n_a$，则

\[
r=\frac{k_g-k_a}{k_g+k_a},\qquad
t=\frac{2k_g}{k_g+k_a}.
\]

解析背景场在界面处连续，且其法向导数连续。这里的系数属于标量 Helmholtz 方程，不应当被解释成已经选择了
完整 Maxwell TE 或 TM 边界条件。

## 9. 圆柱坐标复拉伸形式

令径向、轴向复坐标拉伸分别为

\[
\widetilde r(r)=\int_0^r s_r(\rho)\,d\rho,
\qquad
\widetilde z(z)=\int^z s_z(\zeta)\,d\zeta.
\]

将轴对称 Helmholtz 方程乘以复坐标 Jacobian 后，可写成适合守恒型离散的形式

\[
\partial_r\!\left(\frac{\widetilde r s_z}{s_r}\partial_r u\right)
+\partial_z\!\left(\frac{\widetilde r s_r}{s_z}\partial_z u\right)
+k_0^2n^2\widetilde r s_rs_z u=0.
\]

在 $r=0$，因 $\widetilde r=0$，零径向通量自然给出轴上 regularity。有限域最外边界上的齐次条件应位于
PML 之后，而不是紧贴物理散射区。PML 厚度、强度和网格必须独立变化，因为有限厚度、离散色散以及 grazing
components 都可能留下非零反射。

## 10. 高频 Helmholtz 的 pollution error

高波数 Helmholtz 离散不仅有局部插值误差，还会累积数值相位误差；仅报告“每波长多少个点”不能自动证明
长距离传播已经收敛。可靠 comparator 至少需要同时检查：

1. 线性方程残差与确定性；
2. 固定物理域上的 mesh-refinement pair；
3. 固定 fine mesh 上的 PML-enlargement pair；
4. 平面背景解的界面连续性与 homogeneous normalization；
5. 散射场在 PML 前 guard 区的残余强度。

若这些 reference controls 未通过，跨模型差异不能归因给 backward/reflected-wave physics。直接稀疏 LU
可以避免迭代预条件器本身引入未标定的收敛行为，但不会消除离散 pollution；对更大问题，shifted-Laplacian、
domain decomposition 或高阶有限元是可研究的替代数值路线，仍需各自的 mesh/PML 验证。

## 11. 参考文献

- W. C. Chew and W. H. Weedon, “A 3D perfectly matched medium from modified Maxwell's equations with stretched
  coordinates,” *Microwave and Optical Technology Letters* 7(13), 599–604 (1994).
- F. L. Teixeira and W. C. Chew, “PML-FDTD in cylindrical and spherical grids,” *IEEE Microwave and Guided Wave
  Letters* 7(9), 285–287 (1997).
- X. Jiang et al., “FEM and CIP-FEM for Helmholtz equation with high wave number and perfectly matched layer
  truncation,” arXiv:2207.04685 (2022), https://arxiv.org/abs/2207.04685.
- M. J. Gander García Ramos and R. Nabben, “A two-level shifted Laplace preconditioner for the Helmholtz equation:
  field-of-values analysis and wavenumber-independent convergence,” arXiv:2006.08750 (2020),
  https://arxiv.org/abs/2006.08750.
