# exp040 R13：圆柱波反射分解与高波数 Helmholtz pollution

本文只记录能够独立于某次实验运行成立的数学与物理关系。R13 的 case、阈值、执行顺序、资源限制与结果记录在
`docs/experiment_design/exp040_TGV_3d_multislice_forward.md`。

## 1. 圆柱 outgoing 与 incoming 波

在均匀介质中，轴对称且与轴向坐标无关的标量 Helmholtz 场满足

```text
(1/r) d/dr (r du/dr) + k²u = 0.
```

其两个线性独立解可以取为 Hankel functions

```text
u(r) = A H0^(1)(kr) + B H0^(2)(kr).
```

在采用 `exp(-i omega t)` 的时间约定时，`H0^(1)` 表示向外传播，`H0^(2)` 表示向内传播。给定同一半径处的
`u` 与 `du/dr`，由

```text
[H0^(1)  H0^(2)] [A] = [u    ]
[dH0^(1) dH0^(2)] [B]   [du/dr]
```

可唯一恢复 outgoing 与 incoming coefficients；唯一性来自 Hankel pair 的非零 Wronskian。导数可用

```text
d H0^(j)(kr) / dr = -k H1^(j)(kr),  j=1,2.
```

因此 `|B/A|` 是比“PML 起点处场幅是否很小”更直接的反射观测量。局部分解需要介质均匀、模式基底正确，并同时
获得可靠的场与导数；若存在多个轴向或角向模式，应在相应 modal basis 中分别分解。

## 2. Outgoing impedance 与 radial flux

纯 outgoing 圆柱波满足精确的局部阻抗关系

```text
du/dr = Z_out(r) u,
Z_out(r) = -k H1^(1)(kr) / H0^(1)(kr).
```

所以 `du/dr - Z_out u` 可作为 exact outgoing-condition residual。远场近似
`Z_out ~ ik - 1/(2r)` 给出通常的 Sommerfeld residual，但在有限半径上精确 Hankel impedance 更合适。

对无耗散均匀介质，径向功率通量与

```text
P_r proportional to r Im(conj(u) du/dr)
```

成正比。纯 outgoing 解的该量为正并且随半径保持常数；incoming component 会改变其方向或守恒关系。不过 flux
主要衡量净功率，等量 incoming/outgoing standing wave 可能具有很小净通量，因此它应与 coefficient decomposition
共同使用，不能单独替代反射率。

## 3. Complex-coordinate PML

令物理径向坐标在 PML 内延拓为

```text
r_tilde(r) = r + i integral sigma(r) dr,
s(r) = dr_tilde/dr = 1 + i sigma(r).
```

将圆柱 Helmholtz weak form 作此坐标变换后，radial stiffness 与 mass coefficients 分别为

```text
A_r = r_tilde / s,
M   = k² r_tilde s.
```

解析 outgoing wave 随之延拓为 `H0^(1)(k r_tilde)`，在理想连续 PML interface 上不产生反射。有限厚度、外侧
截断边界、离散色散和数值积分会产生非零 reflection；因此 PML 需要用 analytic outgoing benchmark 或 interior
domain convergence 验证，而不是要求进入 PML 前场幅已经衰减到零。

Complex-coordinate stretching 与 perfectly matched media 的经典来源包括：W. C. Chew and W. H. Weedon,
*A 3D perfectly matched medium from modified Maxwell's equations with stretched coordinates*, Microwave and Optical
Technology Letters 7, 599--604, 1994。

## 4. `kh/p` 与 Helmholtz pollution

高波数 Helmholtz 离散误差包含局部逼近误差和长距离累积的 phase/pollution error。单独给出“每个 element 的
多项式阶数”或“每波长 element 数”都不充分；`kh/p` 是判断解析能力的重要无量纲量，其中 `k` 是局部物理
wavenumber、`h` 是 element diameter、`p` 是 polynomial degree。

一个可解析验证的轴对称模态为

```text
u(r,z) = C J0(alpha_m r/R) sin(n pi z/L),
beta²  = (alpha_m/R)² + (n pi/L)²,
```

其中 `alpha_m` 为 `J0` 的第 m 个正零点。对 operator `Delta u + k² n²u`，选择 manufactured source

```text
f = (k² n² - beta²)u
```

即可在 homogeneous 或 piecewise-constant `n²` 下保持同一 analytic truth。让 material jump 穿过而非对齐
elements，可以同时检查高波数 approximation 与 interface quadrature；但由于 source 被共同构造，这仍是
verification benchmark，不等价于真实散射问题的全部 interface regularity。

关于 wavenumber-explicit hp-FEM quasi-optimality 与 pollution 条件，见 J. M. Melenk and S. Sauter,
*Wavenumber Explicit Convergence Analysis for Galerkin Discretizations of the Helmholtz Equation*, SIAM Journal on
Numerical Analysis 49(3), 2011, DOI `10.1137/090776202`。
