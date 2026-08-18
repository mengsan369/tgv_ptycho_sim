# exp040 理论：B 后频谱、传播与 detector sampling

本文只记录从样品 B 出射场到 detector measurement 的通用物理与数值关系。具体网格、阈值、
运行结果和实验判定由 `exp040` 实验设计文档记录。

## 1. Detector path 的算子分解

设 B 平面 probe 为 $P_B$，第 $s$ 个扫描位置的 transmission 为 $B_s$。测量链可以分成

$$
E_{B,s}=P_BB_s,
$$

$$
U_{D,s}=\mathcal H_{BC}[E_{B,s}],
$$

$$
I_s=|U_{D,s}|^2,
$$

以及 detector 对连续 irradiance $I_s$ 的像素响应。将这几层分开检查，可以区分：B modulation
生成的高空间频率、传播 transfer 的 alias sensitivity、强度非线性以及 detector pixel sampling。

## 2. Ptychographic redundancy 不等于点探测器

Ptychographic overlap 可以为 inverse problem 提供 redundancy，但不会改变 forward measurement 的
物理像素面积。关于 ptychographic sampling 的讨论可参见：

T. B. Edo *et al.*, “Sampling in x-ray ptychography,” *Physical Review A* **87**, 053850
(2013), [doi:10.1103/PhysRevA.87.053850](https://doi.org/10.1103/PhysRevA.87.053850).

理想点采样写为

$$
I_{mn}^{\mathrm{point}}=|U_D(x_m,y_n)|^2.
$$

边长为 $p$、均匀响应的方形像素测量则是面积平均

$$
I_{mn}^{\mathrm{pixel}}
=\frac{1}{p^2}
\int_{x_m-p/2}^{x_m+p/2}
\int_{y_n-p/2}^{y_n+p/2}
|U_D(x,y)|^2\,dx\,dy.
$$

两者只有在 irradiance 在一个像素内近似常数时才近似相同。

## 3. Square-pixel MTF 与 positivity

对已知的连续、周期且充分带限的 irradiance，方形像素面积平均可在频域写为

$$
M(f_x,f_y)=\operatorname{sinc}(pf_x)\operatorname{sinc}(pf_y),
$$

即对 irradiance 的 Fourier transform 乘 $M$ 后再在像素中心取样。这个恒等式并不意味着：对一组
只在离散 nodes 上非负的强度 samples，其有限 Fourier/trigonometric interpolant 在 nodes 之间也非负。
因此，频域 operator 即使保持常数、总和和实数性，也可能在重采样位置给出负值。

物理 detector operator 除能量与单位一致性外，还应满足 positivity。若一个离散实现产生超过舍入误差的
负像素值，不能靠 clipping 把它重新解释为已验证的像素积分；应改用具有非负权重的实空间 quadrature，
或证明所用连续插值空间本身保持非负。

## 4. 频谱诊断量

对离散场 $E$ 及频率集合 $\Omega$，可以定义集合外能量比例

$$
\eta_{\Omega^c}(E)=
\frac{\sum_{(u,v)\notin\Omega}|\widehat E(u,v)|^2}
{\max\left(\sum_{u,v}|\widehat E(u,v)|^2,\epsilon\right)}.
$$

常用的 $\Omega$ 包括 propagation alias-control mask 和 detector native Nyquist rectangle。
应分别对 B-exit field、detector field 和 detector intensity 报告，因为 multiplication、propagation
和取模平方都会改变频谱。粗网格上不可表示的频率不能因为离散数组中不存在就解释为真实能量为零。

## 5. 分层比较原则

可靠的 detector-path 诊断通常按以下顺序进行：

1. 先确认输入 probe 在不同 computational samplings 上收敛；
2. 检查 B multiplication 后有多少能量进入 propagation-sensitive 或 detector-unresolved band；
3. 在同一个 B-exit field 上比较不同 propagation transfer；
4. 在同一个 detector field 上比较 point sampling 与 pixel integration；
5. 最后检查有限 detector ROI、真实 PSF/MTF、pixel gaps、gain、dark、dynamic range 和噪声。

若 upstream field 尚未收敛，downstream detector 差异不能单独归因于像素模型。

## 6. 模型边界

理想方形 pixel average 不是 detector calibration。真实 detector 还可能包含空间变化的 gain、dark
offset、PSF/MTF、pixel gaps、有限 active area、饱和、量化、shot/read noise 和非线性响应。
这些量需要实际标定或可信的器件数据，不能通过选择一个降低 simulation residual 的任意滤波器代替。

同样，detector-path 数值收敛不验证 sample 内的 Fresnel reflection、backward wave、sidewall
multiple scattering 或 polarization。只有传播采样、边界、像素响应和实验标定均受控后仍存在结构化
偏差，才有依据比较 bidirectional BPM、Lippmann–Schwinger、FEM 或 FDTD 等更完整模型。
