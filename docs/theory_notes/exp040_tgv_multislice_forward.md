# exp040 理论：TGV 三维 multi-slice forward model

本文只记录三维 TGV 标量 multi-slice 正演的通用物理与数学。具体几何参数、离散网格、阈值、运行结果和
实验判定由 `exp040` 实验设计文档记录。

## 1. 坐标、数组与物理平面

令 sample A 的入口面为 $z=0$，出射面为 $z=L$，正传播方向沿 $+z$。二维场使用 $(y,x)$ 轴顺序，
三维折射率体使用 $(z,y,x)$；扫描位置通常写成 $(x_s,y_s)$，不能与数组轴顺序混淆。

真空波长为 $\lambda_0$，真空波数为

$$
k_0=\frac{2\pi}{\lambda_0}.
$$

内部长度统一用 SI 单位，复场、amplitude、phase 与 intensity 应明确区分。A 到 B 的传播距离从 A 的真实
出射边界开始，B 到 detector 的距离从 B 平面开始，不能重复包含 sample A 的厚度。

## 2. 轴对称 TGV 几何

对厚度为 $L$、腰部位于 $z_w$ 的轴对称 TGV，可用分段线性直径轮廓

$$
D(z)=
\begin{cases}
D_{\mathrm{top}}+
\left(D_{\mathrm{waist}}-D_{\mathrm{top}}\right)\dfrac{z}{z_w},
&0\le z\le z_w,\\[8pt]
D_{\mathrm{waist}}+
\left(D_{\mathrm{bottom}}-D_{\mathrm{waist}}\right)
\dfrac{z-z_w}{L-z_w},
&z_w<z\le L.
\end{cases}
$$

若孔轴为 $(x_c,y_c)$，半径为 $r(z)=D(z)/2$，则空气填充孔的连续折射率模型为

$$
n(x,y,z)=
\begin{cases}
n_{\mathrm{void}},
&(x-x_c)^2+(y-y_c)^2\le r^2(z),\\
n_{\mathrm{matrix}},&\text{otherwise}.
\end{cases}
$$

轴对称、圆截面和分段线性轮廓只是几何模型；真实样品还可能包含偏心、倾斜、非圆截面、粗糙侧壁和
制造缺陷。

## 3. 精确 slice widths 与中心采样

给定目标轴向步长，先构造边界

$$
0=b_0<b_1<\cdots<b_{N_z}=L.
$$

第 $j$ 层的真实宽度和中心为

$$
w_j=b_{j+1}-b_j,
\qquad
z_j=\frac{b_j+b_{j+1}}{2}.
$$

必须满足

$$
\sum_{j=0}^{N_z-1}w_j=L.
$$

当 $L$ 不是目标步长的整数倍时，末层应使用真实 remainder width。若只把末层中心裁进样品、却仍赋予
完整目标宽度，会让总传播厚度和相位积累超过 $L$。

材料可以在 slice center $z_j$ 采样。横向 voxel-center binary indicator 简单确定，但曲面会出现
staircase；cell-averaged indicator 可以降低几何量化误差。两者是不同的离散表示，不能混称为同一模型。

## 4. 从标量 Helmholtz 方程到 phase-screen 模型

单色标量场满足

$$
\nabla^2U+k_0^2n^2(x,y,z)U=0.
$$

选定参考折射率 $n_{\mathrm{ref}}$，把正向场的演化近似分为参考均匀介质中的横向传播和相对折射率造成的
局部相位调制：

$$
\partial_z u=\left(\mathcal D+\mathcal V(z)\right)u,
\qquad
\mathcal V(z)=ik_0[n(x,y,z)-n_{\mathrm{ref}}].
$$

第 $j$ 层的薄相位屏为

$$
T_j(x,y)=\exp\!\left{
ik_0[n_j(x,y)-n_{\mathrm{ref}}]w_j
\right}.
$$

该近似保留层间正向衍射与 depth ordering，但不生成界面 Fresnel reflection、backward wave、multiple
reflection 或 vector polarization。

## 5. 参考介质 angular-spectrum propagation

参考介质中传播距离 $s$ 的 angular-spectrum operator 为

$$
\mathcal P_s^{(n_{\mathrm{ref}})}[U]
=\mathcal F^{-1}\!\left{
\mathcal F[U]\exp(ik_zs)
\right},
$$

其中

$$
k_z=\sqrt{(n_{\mathrm{ref}}k_0)^2-k_x^2-k_y^2}.
$$

对于 $k_x^2+k_y^2>(n_{\mathrm{ref}}k_0)^2$ 的 evanescent components，可以保留复平方根表示衰减，
或按所声明的 bandlimit 去除。去除 evanescent waves 不等价于控制 sampled transfer phase alias。
FFT 实现还隐含横向周期边界。

## 6. Centered symmetric split-step

若相位屏位于每层中心，入口场先传播首层半宽：

$$
U_0^-=\mathcal P_{w_0/2}[U_{A,\mathrm{in}}],
\qquad
U_0^+=T_0U_0^-.
$$

相邻中心之间的距离为 $(w_j+w_{j+1})/2$：

$$
U_{j+1}^-=\mathcal P_{(w_j+w_{j+1})/2}[U_j^+],
\qquad
U_{j+1}^+=T_{j+1}U_{j+1}^-.
$$

最后传播末层半宽到出射边界：

$$
U_{A,\mathrm{exit}}=
\mathcal P_{w_{N_z-1}/2}[U_{N_z-1}^+].
$$

所有参考传播区间之和为

$$
\frac{w_0}{2}+
\sum_{j=0}^{N_z-2}\frac{w_j+w_{j+1}}{2}+
\frac{w_{N_z-1}}{2}=L.
$$

当所有 $n_j=n_{\mathrm{ref}}$ 时，$T_j=1$，模型应退化为

$$
U_{A,\mathrm{exit}}=\mathcal P_L[U_{A,\mathrm{in}}].
$$

该 homogeneous identity、单层算子顺序和总宽度守恒是几何/代数控制，不是采样收敛证明。

## 7. Full field、参考载波与 projected-phase 极限

若 $\mathcal P_s$ 使用完整 transfer $\exp(ik_zs)$，multi-slice 返回的是保留参考介质 carrier 的 full field。
对零横向频率平面波，公共载波为

$$
\exp(in_{\mathrm{ref}}k_0L).
$$

展示相对 envelope 时可移除这一个已知载波：

$$
\widetilde U_{A,\mathrm{exit}}=
U_{A,\mathrm{exit}}\exp(-in_{\mathrm{ref}}k_0L).
$$

该操作不移除横向衍射，也不同于用 truth 拟合任意 global phase 或 complex gain。

若关闭层间传播，只保留相位屏乘积，则

$$
U_{\mathrm{rel,out}}=
U_{A,\mathrm{in}}\prod_jT_j,
$$

$$
\prod_jT_j=
\exp\!\left[
ik_0\sum_j(n_j-n_{\mathrm{ref}})w_j
\right].
$$

在轴向与横向离散收敛时，这趋向 projected-phase transmission

$$
T_{\mathrm{proj}}(x,y)=
\exp\!\left[
ik_0\int_0^L(n-n_{\mathrm{ref}})\,dz
\right].
$$

相位屏乘积与相同离散和的指数应在代数上相等；离散和与连续积分的差异才是 quadrature/voxelization
误差。开启层间传播后与 projected phase 的非零差异是模型新增的 depth-dependent diffraction，不应被
当成必须消失的数值误差。

## 8. 从 A 出射面到 detector

外部介质中的 A-to-B 传播为

$$
P_B=\mathcal H_{AB}[U_{A,\mathrm{exit}}].
$$

第 $s$ 个扫描位置的 B transmission 为 $B_s$，则

$$
E_{B,s}=P_BB_s,
$$

$$
U_{D,s}=\mathcal H_{BC}[E_{B,s}],
$$

$$
I_s=|U_{D,s}|^2.
$$

$U_{D,s}$ 是复场，$I_s$ 是非负实 irradiance。真实 detector measurement 通常还要对像素面积与响应核
积分，而不是直接把计算 nodes 上的 $I_s$ 当作像素值。

## 9. 数值收敛与公共比较域

对同一物理平面和同一网格上的量 $Q$，可定义

$$
\varepsilon_Q=
\frac{\|Q_{\mathrm{test}}-Q_{\mathrm{ref}}\|_2}
{\max(\|Q_{\mathrm{ref}}\|_2,\epsilon)}.
$$

不同 sampling 或 FOV 的数组必须先映射到同一物理网格/ROI，再计算误差。复场映射应同时作用于实部和
虚部，不能只重采样 wrapped phase。通常需要分别检查：

- axial slice refinement；
- lateral sampling 与界面 representation；
- FOV/padding 与 periodic wrap；
- A-exit、B-plane probe 和 detector measurement；
- algebra identities、finite/nonnegative 与 determinism controls。

这些项目回答不同问题。代数 identity 通过不能替代 sampling convergence，detector 图样看起来平滑也不能
证明 upstream field 收敛。

## 10. 参数扰动信号与 numerical floor

设关注参数为 $\theta$，扰动为 $\theta_0\pm\Delta\theta$。在保持其余物理输入相同的条件下，可定义输出
变化

$$
s_Q^{(\pm)}=
\frac{\|Q(\theta_0\pm\Delta\theta)-Q(\theta_0)\|_2}
{\max(\|Q(\theta_0)\|_2,\epsilon)}.
$$

numerical floor 应来自与 $Q$ 相对应的离散 refinement，而不是从另一套 forward branch 借用。只有扰动信号
在进一步 refinement 下稳定，并明显高于数值 floor，才能称为“在当前无噪声模型内数值可分辨”。这不等于
含噪检测限、参数唯一可辨识性或已经完成 inverse reconstruction。

## 11. 模型适用边界

标量单色单向 phase-screen multi-slice 可以研究层间衍射、depth ordering 和离散收敛，但不能仅靠减小
$\Delta x$ 或 $\Delta z$ 恢复：

- glass–air interface Fresnel reflection；
- backward wave 与 multiple scattering；
- vector polarization 与 high-NA coupling；
- absorption、dispersion 与 finite coherence；
- roughness、tilt、noncircularity 和制造缺陷；
- detector/stage calibration、noise、background、saturation 与 quantization。

数值收敛证明的是所定义离散模型被稳定求解，不是该模型相对真实 TGV 的绝对物理误差已经足够小。
