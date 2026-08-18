# exp040 理论：positivity-preserving detector quadrature

本文只记录有限 detector pixel 积分的正值数值方法。具体 quadrature 序列、阈值、运行结果和实验判定
由 `exp040` 实验设计文档记录。

## 1. Pixel measurement 是面积积分

设方形 detector pixel 的边长为 $p$，中心为 $(x_m,y_n)$。理想均匀响应像素的面积平均是

$$
I_{mn}=\frac{1}{p^2}
\int_{x_m-p/2}^{x_m+p/2}
\int_{y_n-p/2}^{y_n+p/2}
|U(x,y)|^2\,dx\,dy.
$$

这与仅计算像素中心的 $|U(x_m,y_n)|^2$ 是不同的 forward operator。当 detector-plane irradiance
含有接近或超过 pixel Nyquist 的变化时，两者差异尤其明显。

## 2. Staggered midpoint nodes

在每个方向放置 $q$ 个 cell-centered subpixel nodes：

$$
x_{m,a}=x_m+
\left(\frac{a+1/2}{q}-\frac12\right)p,
\qquad a=0,\ldots,q-1,
$$

$y_{n,b}$ 同理。二维 composite midpoint approximation 为

$$
I_{mn}^{(q)}=\frac{1}{q^2}
\sum_{a=0}^{q-1}\sum_{b=0}^{q-1}
|U(x_{m,a},y_{n,b})|^2.
$$

对于偶数 $q$，原像素中心位于四个中心 nodes 之间，而不是某一个 node 上。这是 midpoint rule 的几何
性质，不应为了与旧 point grid 对齐而平移 node origin。

## 3. Positivity 与守恒性质

所有 quadrature weights 均为

$$
w_{ab}=\frac{1}{q^2}\ge0,
\qquad
\sum_{a,b}w_{ab}=1.
$$

因此，只要 node irradiance finite 且非负，输出像素值按构造非负。常数 irradiance 被精确保留；若完整
node grid 被无重叠地分块为 detector pixels，则像素求和与 node irradiance 求和满足相应的面积尺度关系。

实现至少应检查：

- nodes 的 block center 等于物理 pixel center；
- weights 非负且和为一；
- constant preservation；
- per-frame sum identity；
- 输出 finite 且非负；
- 随 $q$ 增大的 quadrature convergence。

## 4. 与频域 sinc-MTF 方法的关系

对于已知连续、周期、带限的 irradiance，square-pixel box average 可由 sinc MTF 精确表示。但从离散非负
samples 构造的有限 Fourier interpolant 不保证在 nodes 之间非负。正权重实空间 quadrature 绕开了这个
positivity 缺口：先在实际积分 nodes 上评估复场，再形成 $|U|^2$，最后以非负权重求和。

如果传播 grid 和 detector quadrature nodes 的 origin 不兼容，可以用 shifted Fourier evaluation、CZT、
NUFFT 或单独的 staggered detector grid 评估 $U$；不能把错位的 point samples 直接分块后称作物理像素积分。

## 5. 误差与适用边界

Composite midpoint rule 对足够光滑的 integrand 通常具有二阶收敛，但 $|U|^2$ 的高频内容和传播离散误差
会共同影响观测到的 $q$-series。因而 detector quadrature 收敛前，应确认 upstream field sampling 也受控。

正值 quadrature 只实现理想均匀 pixel footprint。它不自动包含 measured PSF/MTF、pixel gaps、空间变化的
gain、dark offset、饱和、量化或噪声；这些仍需要 detector-specific calibration。
