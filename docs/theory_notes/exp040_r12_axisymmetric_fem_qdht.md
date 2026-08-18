# exp040 R12：axisymmetric hp-FEM 与 quasi-discrete Hankel transform

本文只记录可独立于某次 run 的数学依据；R12 的 case、阈值、资源选择、结果和结论均留在
`docs/experiment_design/exp040_TGV_3d_multislice_forward.md`。

## 1. Axisymmetric scalar Helmholtz weak form

对无方位角依赖的标量场，圆柱坐标 Helmholtz 方程乘以 radial Jacobian 后可写成

```text
∂r(r ∂r u) + ∂z(r ∂z u) + k0² n² r u = f.
```

在 test function 上分部积分后，轴 `r=0` 的 regularity 给出自然零通量；外边界可施加 homogeneous
Dirichlet 或 PML 后的截断条件。complex-coordinate stretch `sr, sz` 把 weak-form 系数变为

```text
Ar = z-stretch * r_tilde / sr
Az = r_tilde * sr / sz
M  = k0² n² r_tilde * sr * sz,
```

因此双线性型保持 complex-symmetric，但一般不是 Hermitian。材料跳变应在 weak-form quadrature 中积分；不能把
只对 Cartesian constant coefficient 推导的 dispersion correction 无证明地移植到该 operator。

高频 Helmholtz 的误差不仅由局部插值阶决定，还存在 pollution error。Melenk 与 Sauter 的 wavenumber-explicit
分析表明，hp-FEM 的 quasi-optimality 需要联合控制 `kh/p`，并让 polynomial degree 至少随 `log(k)` 增长；所以
“固定低阶并无限缩小 h”与“在过粗 mesh 上只提高 p”都不能自动构成 reference。

参考：J. M. Melenk and S. Sauter, *Wavenumber Explicit Convergence Analysis for Galerkin Discretizations of the
Helmholtz Equation*, SIAM Journal on Numerical Analysis 49(3), 2011, DOI `10.1137/090776202`。

## 2. Order-zero quasi-discrete Hankel transform

径向对称二维 Fourier propagation 可由 order-zero Hankel transform 表示。令 `alpha_n` 为 `J0` 的正零点，
在有限 radial support `R` 上采用

```text
r_n  = alpha_n R / alpha_(N+1)
kr_m = alpha_m / R.
```

Guizar-Sicairos--Gutiérrez-Vega 的 QDHT 用 Bessel-zero lattice 构造近似自逆的离散 transform matrix，并以
`J1(alpha_n)` scaling 恢复物理 Hankel pair。该 scaling 使离散 Parseval 关系得到保持，因而适合反复的 field
propagation。传播只需在 Hankel spectrum 上乘

```text
exp(i sqrt(k²-kr²) dz).
```

有限 `R` 的 QDHT basis 隐含 outer radial boundary；对含均匀 plane-wave background 的问题，应传播在边界衰减的
contrast `delta=u/u_bg-1`，而把 background carrier 解析保留，避免强迫常数场在 `R` 处变为零。

参考：M. Guizar-Sicairos and J. C. Gutiérrez-Vega, *Computation of quasi-discrete Hankel transforms of integer order
for propagating optical wave fields*, JOSA A 21(1), 53--58, 2004, DOI `10.1364/JOSAA.21.000053`。

## 3. 与 band-limited ASM 的关系

QDHT 消除了 Cartesian square frequency lattice，但不自动证明 finite radial support 已闭合。Cartesian
band-limited ASM 则仍在 periodic FFT grid 上，只通过限制 sampled transfer bandwidth 控制 transfer-function
alias；它既不会把 circular convolution 变成 open boundary，也不会改变 `1/FOV` frequency spacing。因此
enlarged FOV、alias-controlled transfer 和 Hankel reference 必须分别比较。

参考：K. Matsushima and T. Shimobaba, *Band-limited angular spectrum method for numerical simulation of free-space
propagation in far and near fields*, Optics Express 17, 19662--19673, 2009, DOI `10.1364/OE.17.019662`。

## 4. PML 起点场幅不等于 PML reflection error

PML 的目的，是在有限厚度内无反射地吸收进入层内的 outgoing wave；它并不要求 physical-domain/PML interface
处的 scattered amplitude 已经趋近零。因此“最后一个窄 annulus 的场幅 / 内区场幅”通常不是 reflection
coefficient，也没有必须随 PML start 外移而单调下降的定理。oscillatory outgoing field、多个 axial/radial mode
的 interference，或固定窗口相对 field lobe 的位置都会改变该比值。

更直接的 open-boundary controls 包括：

```text
1. 在 observation region 比较两个不同 truncation/PML-start domain 的 full complex field；
2. 用 analytic outgoing solution 做 PML manufactured benchmark；
3. 由 Im(conj(u) * ∂r u) 等 time-averaged radial flux 检查 outward power；
4. 在适用的 modal/Hankel basis 中分解 incoming 与 outgoing components；
5. 评估 Sommerfeld-type outgoing residual，而不是要求 interface amplitude 为零。
```

这些量分别回答 interior truncation error、PML reflection 和 radiation condition；它们不能由单个 local-amplitude
guard 相互替代。具体窗口、阈值和结论仍属于实验预注册，而不是本理论说明的一部分。
