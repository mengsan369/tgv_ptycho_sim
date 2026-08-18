# exp040 理论：周期相容边界与 ASM transfer-sampling alias

本文只记录周期边界与角谱传播采样的通用理论。具体 FOV、阈值、运行结果和实验判定由
`exp040` 实验设计文档记录。

## 1. 两类不同的周期问题

FFT 角谱传播中至少有两类容易混淆的问题：

1. **物理对象的周期不相容**：有限计算 FOV 与周期对象的基本周期不成整数倍，窗口两端的对象值不连续；
2. **传播 transfer 的采样混叠**：即使只保留 propagating waves，离散频率网格也可能不足以采样
   angular-spectrum transfer function 的快速相位变化。

把计算 FOV 选为对象基本周期的整数倍可以消除第一类人为接缝，但不能自动控制第二类混叠，也不能把
circular propagation 变成开放边界传播。

## 2. Angular-spectrum transfer 的局部频率

令 $u,v$ 为 cycles/m，介质波长为 $\lambda_m=\lambda_0/n$。传播距离为 $z$ 时，propagating
angular-spectrum transfer 为

$$
H(u,v;z)=\exp\!\left[i2\pi z
\sqrt{\lambda_m^{-2}-u^2-v^2}\right].
$$

去除 $u^2+v^2>\lambda_m^{-2}$ 的 evanescent components，只限制了传播波圆盘，并未保证 transfer
phase 被离散频率网格充分采样。其沿两轴的局部相位频率为

$$
f_u=\frac{uz}{\sqrt{\lambda_m^{-2}-u^2-v^2}},\qquad
f_v=\frac{vz}{\sqrt{\lambda_m^{-2}-u^2-v^2}}.
$$

若 FFT 的频率间隔为

$$
\Delta u=(N_x\Delta x)^{-1},\qquad
\Delta v=(N_y\Delta y)^{-1},
$$

则 sampled-transfer Nyquist 条件要求

$$
\Delta u^{-1}\ge2|f_u|,\qquad
\Delta v^{-1}\ge2|f_v|.
$$

## 3. Exact common-ellipse mask

上述条件可以写成两个椭圆区域的交集：

$$
\frac{u^2}{u_{\mathrm{lim}}^2}
+\frac{v^2}{\lambda_m^{-2}}\le1,
\qquad
\frac{u^2}{\lambda_m^{-2}}
+\frac{v^2}{v_{\mathrm{lim}}^2}\le1,
$$

其中

$$
u_{\mathrm{lim}}
=\frac{\lambda_m^{-1}}
{\sqrt{1+(2\Delta u|z|)^2}},\qquad
v_{\mathrm{lim}}
=\frac{\lambda_m^{-1}}
{\sqrt{1+(2\Delta v|z|)^2}}.
$$

在交集外把 transfer 设为零，可控制给定 same-grid FFT 上的 transfer-sampling alias。该推导源于：

K. Matsushima and T. Shimobaba, “Band-Limited Angular Spectrum Method for Numerical
Simulation of Free-Space Propagation in Far and Near Fields,” *Optics Express* **17**,
19662–19673 (2009), [doi:10.1364/OE.17.019662](https://doi.org/10.1364/OE.17.019662).

## 4. Same-grid alias control 与 linear convolution 的区别

在原网格上应用 common-ellipse mask，仍然执行 periodic/circular convolution。它能够回答“当前
sampled transfer 是否对结果有显著影响”，但不能：

- 消除传播场在有限 FOV 中的截断；
- 消除对象或扫描的 periodic wrap；
- 取代为 linear convolution 准备的 source padding；
- 证明 mask 外的真实光谱在物理上不重要；
- 恢复反射、后向波、偏振或多重散射。

因此，“band-limited ASM”“开放边界”和“有限支持对象”是三个不同的建模选择，不能用其中一个的
收敛替代另外两个。

## 5. 方法比较与公共通带

比较两种传播方法时，应使用同一输入场、同一物理网格、同一 ROI，并明确 relative-error 的分母。
若两个方法保留的频带不同，方法差异一般同时包含 transfer 相位差和 passband-support 差异。

需要进一步拆分时，可以预先定义 **common-passband control**：只在两种方法共同保留的频率集合上比较
传播结果，再把被各自 mask 排除的频谱能量单独报告。该控制能区分“共同通带内的传播差异”和
“alias mask support 选择”，但不会判定哪个被排除频带更接近真实物理。
