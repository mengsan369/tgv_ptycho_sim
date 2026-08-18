# exp040 理论：离散 refinement 与全场外部传播 padding

本文只记录可复用的数值传播原理。具体网格、比较序列、阈值、运行结果和实验判定由
`exp040` 实验设计文档记录。

## 1. 离散误差的分层

三维 multi-slice forward model 中，至少需要区分以下误差来源：

1. 轴向 slice 宽度导致的 split-step 误差；
2. 横向网格对曲面材料界面的 voxelization 误差；
3. sample 内部 FFT 传播的有限 FOV 与周期边界误差；
4. sample 出射面之后各段传播的有限 FOV、circular wrap 与 transfer sampling 误差；
5. detector field 到像素测量值之间的采样或积分误差。

改变一个离散参数时，应尽量固定其余物理对象，并在相同物理区域上比较。否则，一个表面上的
“网格收敛误差”可能同时包含样品重采样、随机对象变化、传播边界变化和 detector operator 变化。

## 2. 为什么不能直接零填充含非零背景的全场

设计算窗口内的复场为 $U(x,y)$。若场在窗口边缘不趋近于零，把它直接中心嵌入更大的零数组，
等价于先乘一个硬矩形孔径 $W$：

$$
U_{\mathrm{direct}}^{(p)}=\mathcal E_p[U]=\mathcal E_p[WU].
$$

这会在孔径边缘制造不属于原物理问题的突变和衍射。平面波、宽光束或任何含非零 carrier/background
的全场都存在这个问题。因此，扩大传播 FOV 不能机械地理解为“把全场四周补零”。

## 3. Reference-plus-residual 分解

若已知一个与真实计算具有相同入射场、参考介质、厚度和相位 convention 的均匀参考场
$U_{\mathrm{ref}}$，可以定义未经拟合的 residual：

$$
\delta U=U-U_{\mathrm{ref}}.
$$

在更大的网格 $p$ 上，重新生成参考背景，只对 residual 做中心零嵌入：

$$
U^{(p)}=U_{\mathrm{ref}}^{(p)}+\mathcal E_p[\delta U].
$$

该构造保留了非零背景，又避免在原窗口边缘人为截断背景。它不需要用 truth 做 global-phase、
complex-gain 或 spatial alignment；参考场必须由 forward model 本身的均匀控制直接给出。

若传播算子为 $\mathcal H_z$，则 enlarged-grid 传播写为

$$
U_z^{(p)}=\mathcal H_z^{(p)}[U^{(p)}].
$$

不同 padding FOV 的输出应裁到同一物理 ROI 后比较，而不是比较形状不同的完整数组。

## 4. Residual 局域性的有效性检查

Reference-plus-residual padding 只有在 $\delta U$ 接近局域时才合理。设 $M_e$ 为原计算窗口的外侧
边缘环，可定义边缘能量比例

$$
\eta_{\mathrm{edge}}
=
\frac{\|M_e\delta U\|_2^2}
{\max(\|\delta U\|_2^2,\epsilon)}.
$$

较大的 $\eta_{\mathrm{edge}}$ 表明原始窗口已截断显著 residual；此时增加外部零 padding 只是传播一个
已经被截断的源。不能通过事后 window 或 apodization 把该比例压低，再把结果解释为原问题的收敛。

## 5. 公共物理对象与公共比较域

离散 refinement 应保持同一连续样品、同一随机实现、同一物理扫描位置、同一传播距离和同一 detector
定义。piecewise-constant 随机对象在不同网格上的表示，应从同一组物理 cells 展开或守恒重采样，
不能用相同随机种子但不同抽样形状重新生成，因为随机数调用数量改变后通常已不是同一对象。

对于共同物理区域上的量 $Q$，可用未对齐 relative $L^2$ 误差

$$
\varepsilon_Q=
\frac{\|Q_{h}-Q_{h/2}\|_2}
{\max(\|Q_{h/2}\|_2,\epsilon)}.
$$

复场、强度和频谱能量应分别报告。除非研究问题本身允许，否则不应通过 global phase、scale、complex
gain 或 spatial alignment 移除 forward-model 差异。

## 6. 方法边界

扩大 FFT 网格只是把周期副本推远，并不把 periodic/circular convolution 自动变成无限域传播。
Reference-plus-residual padding也不能恢复在原窗口中已经丢失的散射场，更不能补回标量单向模型没有包含的
反射、后向波、偏振或多重散射。它解决的是非零背景下的数值嵌入问题，而不是完整开放边界 Maxwell 问题。
