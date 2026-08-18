# exp040 理论：有限 sample support 与 open-boundary residual propagation

本文只记录有限调制样品和开放边界近似的通用建模。具体 support、padding 序列、阈值、运行结果和
实验判定由 `exp040` 实验设计文档记录。

## 1. 周期对象与有限对象是不同物理模型

在 FFT 网格上把 transmission 周期延拓，意味着计算窗口之外存在无限重复的样品。真实有限编码样品更适合
写成

$$
B_{\mathrm{fin}}(x,y)=B_0+M(x,y),
$$

其中 $B_0$ 是外部背景 transmission，$M$ 只在有限区域内非零。透明 exterior 对应 $B_0=1$。

扫描有限样品时，应平移局域 modulation $M$，并对移出计算区域的部分使用 constant-zero boundary；
随后再加回背景 $B_0$。直接对完整 $B_{\mathrm{fin}}$ 做 periodic roll，会把窗口一侧移出的编码内容从
另一侧卷回，代表的是无限周期对象而不是有限样品。

## 2. B 平面的 reference-plus-residual 分解

设 B 平面入射 probe 可分解为

$$
P_B=P_0+\delta P_B,
$$

其中 $P_0$ 是可在整个网格上解析或独立生成的 homogeneous background。若
$B_s=B_0+M_s$，则

$$
E_{B,s}=P_BB_s
=P_0B_0+R_s,
$$

$$
R_s=P_0M_s+\delta P_BB_0+\delta P_BM_s.
$$

对于透明 exterior $B_0=1$，可以合并为

$$
R_s=\delta P_B+P_BM_s.
$$

detector field 为

$$
U_{D,s}=\mathcal H_{BC}[P_0B_0]+\mathcal H_{BC}[R_s].
$$

若 background 是传播算子的已知本征模，可在任意 enlarged grid 上直接生成；只有接近局域的 $R_s$ 需要
零 padding。这样避免对非零背景施加硬孔径。

## 3. 以 padding FOV 逼近开放边界

FFT 传播本质上是 circular convolution。把局域 residual 嵌入逐渐增大的零背景网格，会把其周期副本推远。
在固定中心 ROI 中，若输出随 padding FOV 稳定，可以把它解释为对开放横向边界的数值收敛近似。

必须同时检查 source/residual 在 base grid 边缘的能量。若 residual 在嵌入前已被明显截断，增大 padding
不能恢复丢失信息。可用边缘环能量比例

$$
\eta_{\mathrm{edge}}=
\frac{\|M_eR_s\|_2^2}
{\max(\|R_s\|_2^2,\epsilon)}
$$

诊断其局域性。

## 4. Support effect 与 boundary effect 的分离

为避免把两种变化混为一谈，可构造三类公共输入比较：

1. periodic sample + circular propagation；
2. finite sample + circular propagation；
3. finite sample + enlarged residual propagation。

第一与第二类差异主要反映 object support/scan boundary；第二与第三类差异主要反映 propagation circular
wrap；第一与第三类给出两者的 combined effect。三者必须使用同一 probe、同一有限对象内部 transmission、
同一扫描位置、同一 detector operator 和同一中心 ROI。

## 5. 方法边界

有限 padding 的收敛不等于严格的无限域 Green-function 解，也不代表有限 support 或透明 exterior 已由实验
标定。真实样品还可能有 substrate/background transmission、吸收、边缘 taper、制造缺陷和 illumination
aperture。它们改变的是物理模型，需要由测量或独立先验确定，不能按 residual 大小事后选择。
