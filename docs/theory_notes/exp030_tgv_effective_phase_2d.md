# TGV 二维 projected-phase 近似

## 1. 本文讨论的对象

本项目当前讨论的是玻璃基体中的**单个** TGV：孔轴与传播方向 $z$ 重合，孔截面为圆形，孔内为空气。需要区分以下三个对象。

1. 三维折射率分布 $n(x,y,z)$ 描述每个空间位置是空气还是玻璃。
2. 完整传播算子描述光场在样品内部被逐层调制，同时发生横向衍射和层间耦合；Phase 4 multislice 属于这一层。
3. 二维 projected-phase 透过函数 $T_{\mathrm{proj}}(x,y)$ 把样品内部传播近似为每条轴向直线上的相位累积；exp030 使用这一层。

第三个对象不是由材料单独决定的严格电磁本征透过函数。它依赖真空波长、参考介质和传播方向，并建立在正入射、标量近轴、弱折射率差以及忽略样品内部横向传播的近似上。

本文使用以下符号：

| 符号 | 含义 | 单位 |
|---|---|---|
| $L$ | 玻璃厚度 | $\mathrm{m}$ |
| $z_w$ | 腰部深度，即代码参数 $\mathtt{z\_waist}$ | $\mathrm{m}$ |
| $D_t,D_w,D_b$ | 入口、腰部、出口直径 | $\mathrm{m}$ |
| $R_t,R_w,R_b$ | 入口、腰部、出口半径，满足 $R_\alpha=D_\alpha/2$ | $\mathrm{m}$ |
| $(x_c,y_c)$ | 孔轴在横向平面的位置 | $\mathrm{m}$ |
| $r(x,y)$ | 点 $(x,y)$ 到孔轴的横向距离 | $\mathrm{m}$ |
| $n_{\mathrm{air}},n_{\mathrm{glass}}$ | 空气和玻璃折射率 | $1$ |
| $\lambda_0$ | 真空波长 | $\mathrm{m}$ |
| $k_0$ | 真空波数，$k_0=2\pi/\lambda_0$ | $\mathrm{rad\,m^{-1}}$ |

横向半径定义为

$$
r(x,y)
=
\sqrt{(x-x_c)^2+(y-y_c)^2}.
$$

## 2. projected phase 从哪里来

### 2.1 从标量 Helmholtz 方程到近轴方程

单色标量场 $E(x,y,z)$ 满足 Helmholtz 方程

$$
\nabla^2 E+k_0^2 n^2(x,y,z)E=0.
$$

选取参考折射率 $n_{\mathrm{ref}}$，并定义参考介质中的波数

$$
k_{\mathrm{ref}}=k_0n_{\mathrm{ref}}.
$$

将快速振荡的载波和缓慢变化的包络分开：

$$
E(x,y,z)=U(x,y,z)\exp(ik_{\mathrm{ref}}z).
$$

代回 Helmholtz 方程，可得

$$
\frac{\partial^2 U}{\partial z^2}
+2ik_{\mathrm{ref}}\frac{\partial U}{\partial z}
+\nabla_\perp^2U
+k_0^2\left[n^2(x,y,z)-n_{\mathrm{ref}}^2\right]U
=0,
$$

其中

$$
\nabla_\perp^2
=
\frac{\partial^2}{\partial x^2}
+\frac{\partial^2}{\partial y^2}
$$

是横向 Laplace 算子。近轴近似忽略包络的二阶轴向导数 $\partial^2U/\partial z^2$。再使用弱折射率差近似

$$
n^2-n_{\mathrm{ref}}^2
=
(n-n_{\mathrm{ref}})(n+n_{\mathrm{ref}})
\approx
2n_{\mathrm{ref}}(n-n_{\mathrm{ref}}),
$$

得到

$$
\frac{\partial U}{\partial z}
=
\frac{i}{2k_{\mathrm{ref}}}\nabla_\perp^2U
+ik_0\left[n(x,y,z)-n_{\mathrm{ref}}\right]U.
$$

方程右侧有两个物理作用：

- 第一项 $i\nabla_\perp^2U/(2k_{\mathrm{ref}})$ 描述样品内部的横向衍射和相邻横向位置之间的耦合；
- 第二项 $ik_0(n-n_{\mathrm{ref}})U$ 描述局部折射率相对于参考介质造成的相位积累。

multislice 会交替处理这两个作用。projected-phase 近似则直接去掉横向 Laplace 项，即令

$$
\nabla_\perp^2U\approx 0.
$$

于是每个固定横向位置 $(x,y)$ 都满足一个互不耦合的一阶常微分方程：

$$
\frac{\partial U(x,y,z)}{\partial z}
=
ik_0\left[n(x,y,z)-n_{\mathrm{ref}}\right]U(x,y,z).
$$

从 $z=0$ 积分到 $z=L$，得到

$$
U(x,y,L)
=
U(x,y,0)
\exp\!\left[
ik_0\int_0^L
\left(n(x,y,z)-n_{\mathrm{ref}}\right)\,\mathrm dz
\right].
$$

因此二维 projected-phase 透过函数定义为

$$
T_{\mathrm{proj}}(x,y)
=
\frac{U(x,y,L)}{U(x,y,0)}
=
\exp\!\left[
ik_0\int_0^L
\left(n(x,y,z)-n_{\mathrm{ref}}\right)\,\mathrm dz
\right].
$$

这一步保留沿 $z$ 方向的局部相位累积，但完全删除样品内部的横向衍射、侧壁造成的角度偏折以及不同横向位置之间的场耦合。

同一形式也可以从几何光学的 eikonal（光程）观点得到。若假设光线始终沿 $z$ 方向直线传播，则它累积的相位为

$$
\Phi(x,y)
=
k_0\int_0^L n(x,y,z)\,\mathrm dz.
$$

减去在参考介质中传播同样厚度所产生的相位

$$
\Phi_{\mathrm{ref}}=k_0n_{\mathrm{ref}}L,
$$

就得到

$$
\Phi(x,y)-\Phi_{\mathrm{ref}}
=
k_0\int_0^L
\left[n(x,y,z)-n_{\mathrm{ref}}\right]\,\mathrm dz.
$$

因此，近轴波动方程在弱折射率差下给出一个推导，而直线光路的光程模型给出另一种解释。exp030 把后一个直线投影形式作为模型定义；这仍然要求忽略空气—玻璃侧壁导致的光线弯折、反射和多次散射，不能因为写成光程积分就认为它是严格三维解。

### 2.2 以玻璃为参考介质

exp030 取

$$
n_{\mathrm{ref}}=n_{\mathrm{glass}}.
$$

定义空气区域的指示函数

$$
\chi_{\mathrm{air}}(x,y,z)
=
\begin{cases}
1, & r(x,y)\le R(z),\\
0, & r(x,y)>R(z).
\end{cases}
$$

则三维折射率分布可写为

$$
n(x,y,z)
=
n_{\mathrm{glass}}
+\left(n_{\mathrm{air}}-n_{\mathrm{glass}}\right)
\chi_{\mathrm{air}}(x,y,z).
$$

空气路径长度定义为

$$
\ell_{\mathrm{air}}(x,y)
=
\int_0^L\chi_{\mathrm{air}}(x,y,z)\,\mathrm dz
=
\int_0^L
\mathbf 1\!\left[r(x,y)\le R(z)\right]\,\mathrm dz.
$$

因此相对玻璃参考的光程差为

$$
\operatorname{OPD}_{\mathrm{rel}}(x,y)
=
\int_0^L
\left[n(x,y,z)-n_{\mathrm{glass}}\right]\,\mathrm dz
=
\left(n_{\mathrm{air}}-n_{\mathrm{glass}}\right)
\ell_{\mathrm{air}}(x,y).
$$

代码采用 $\exp(+i\phi)$ 的相位约定，所以未包裹相位为

$$
\phi_{\mathrm{unwrapped}}(x,y)
=
k_0\operatorname{OPD}_{\mathrm{rel}}(x,y)
=
\frac{2\pi}{\lambda_0}
\left(n_{\mathrm{air}}-n_{\mathrm{glass}}\right)
\ell_{\mathrm{air}}(x,y),
$$

最终得到

$$
A_{\mathrm{effective}}(x,y)
=
T_{\mathrm{proj}}(x,y)
=
\exp\!\left[i\phi_{\mathrm{unwrapped}}(x,y)\right].
$$

在无吸收基线中，折射率均为实数，因此

$$
\left|A_{\mathrm{effective}}(x,y)\right|=1.
$$

由于通常 $n_{\mathrm{air}}<n_{\mathrm{glass}}$，相对光程和未包裹相位在空气路径内为负。即使复数透过函数的相位已经经历多次 $2\pi$ 包裹，也必须单独保留 $\operatorname{OPD}_{\mathrm{rel}}$ 或 $\phi_{\mathrm{unwrapped}}$，否则会丢失累计路径信息。

## 3. 参数化单孔几何

深度范围为

$$
0\le z\le L.
$$

公共分段线性直径轮廓为

$$
D(z)
=
\begin{cases}
D_t+\left(D_w-D_t\right)\dfrac{z}{z_w},
&0\le z\le z_w,\\[8pt]
D_w+\left(D_b-D_w\right)
\dfrac{z-z_w}{L-z_w},
&z_w<z\le L.
\end{cases}
$$

半径轮廓是

$$
R(z)=\frac{D(z)}{2}.
$$

几何必须满足

$$
L>0,\quad D_t>0,\quad D_w>0,\quad D_b>0,
$$

$$
D_w\le \min(D_t,D_b),
\qquad
0<z_w<L.
$$

公共实现位于 **src/tgv_ptycho/objects/tgv_geometry.py**。二维 projected model 和三维折射率 volume 都调用同一个 $\mathtt{diameter\_profile}()$，没有各自复制一套 $D(z)$ 公式。

## 4. 解析空气路径：逐步推导

### 4.1 积分为何等于区间长度

固定一个横向位置后，半径 $r=r(x,y)$ 不再随 $z$ 改变。空气路径

$$
\ell_{\mathrm{air}}(r)
=
\int_0^L\mathbf 1\!\left[r\le R(z)\right]\,\mathrm dz
$$

是在计算集合

$$
\mathcal A_r
=
\left\{z\in[0,L]:r\le R(z)\right\}
$$

沿 $z$ 方向的总长度。因此

$$
\ell_{\mathrm{air}}(r)=\operatorname{meas}(\mathcal A_r).
$$

某一深度若 $r\le R(z)$，该深度属于空气孔，指示函数贡献 $1$；否则属于玻璃，贡献 $0$。

### 4.2 入口到腰部

在 $0\le z\le z_w$ 中，半径由 $R_t$ 线性减小到 $R_w$：

$$
R_{\mathrm{top}}(z)
=
R_t+\left(R_w-R_t\right)\frac{z}{z_w}.
$$

对过渡环内的半径 $R_w<r<R_t$，空气和玻璃的分界深度 $z_{\times,t}$ 满足

$$
R_{\mathrm{top}}(z_{\times,t})=r.
$$

代入轮廓并求解：

$$
r
=
R_t+\left(R_w-R_t\right)\frac{z_{\times,t}}{z_w},
$$

$$
z_{\times,t}
=
z_w\frac{R_t-r}{R_t-R_w}.
$$

孔径在这一段随 $z$ 增大而减小，所以空气区域是 $0\le z\le z_{\times,t}$。入口段贡献为

$$
\ell_{\mathrm{top}}(r)
=
\begin{cases}
z_w,
&0\le r\le R_w,\\[4pt]
z_w\dfrac{R_t-r}{R_t-R_w},
&R_w<r<R_t,\\[8pt]
0,
&r\ge R_t.
\end{cases}
$$

当 $R_t=R_w$ 时不能使用含零分母的过渡区公式。此时入口段是圆柱：

$$
\ell_{\mathrm{top}}(r)
=
z_w\mathbf 1\!\left[r\le R_w\right].
$$

### 4.3 腰部到出口

在 $z_w<z\le L$ 中，半径由 $R_w$ 线性增大到 $R_b$：

$$
R_{\mathrm{bottom}}(z)
=
R_w+\left(R_b-R_w\right)
\frac{z-z_w}{L-z_w}.
$$

对 $R_w<r<R_b$，分界深度满足

$$
R_{\mathrm{bottom}}(z_{\times,b})=r,
$$

所以

$$
z_{\times,b}
=
z_w+\left(L-z_w\right)
\frac{r-R_w}{R_b-R_w}.
$$

孔径在这一段随 $z$ 增大而增大，所以空气区域是 $z_{\times,b}\le z\le L$。出口段贡献为

$$
\ell_{\mathrm{bottom}}(r)
=
\begin{cases}
L-z_w,
&0\le r\le R_w,\\[4pt]
\left(L-z_w\right)\dfrac{R_b-r}{R_b-R_w},
&R_w<r<R_b,\\[8pt]
0,
&r\ge R_b.
\end{cases}
$$

当 $R_b=R_w$ 时，出口段退化为

$$
\ell_{\mathrm{bottom}}(r)
=
\left(L-z_w\right)
\mathbf 1\!\left[r\le R_w\right].
$$

### 4.4 一般情形和两个解析控制组

一般分段线性单孔的解析空气路径为

$$
\ell_{\mathrm{air}}(r)
=
\ell_{\mathrm{top}}(r)
+\ell_{\mathrm{bottom}}(r).
$$

当

$$
R_t=R_b,
\qquad
z_w=\frac{L}{2},
$$

上下两段在过渡环中各贡献一半厚度，得到

$$
\ell_{\mathrm{air}}(r)
=
\begin{cases}
L,
&0\le r\le R_w,\\[4pt]
L\dfrac{R_t-r}{R_t-R_w},
&R_w<r<R_t,\\[8pt]
0,
&r\ge R_t.
\end{cases}
$$

因此二维投影有三个区域：

- $0\le r\le R_w$ 是贯穿整个厚度的中央恒定路径平台；
- $R_w<r<R_t$ 是空气路径随半径线性减小的环形过渡区；
- $r\ge R_t$ 是纯玻璃参考区，其中 $\ell_{\mathrm{air}}=0$ 且 $T_{\mathrm{proj}}=1$。

圆柱控制组满足 $R_t=R_w=R_b=R$，此时

$$
\ell_{\mathrm{air}}(r)
=
\begin{cases}
L,&0\le r\le R,\\
0,&r>R.
\end{cases}
$$

所以 projected phase 退化为普通恒定相位圆盘。连续积分中，单个边界点的集合长度为零，因此在 $r=R_t$ 或 $r=R_b$ 处采用小于号还是小于等于号不会改变解析路径积分。

## 5. 轴向数值积分：代码实际计算了什么

### 5.1 为什么已有解析解还要做数值积分

**src/tgv_ptycho/objects/tgv2d.py** 支持两种方法：

- $\mathtt{analytic}$ 直接调用上一节的解析路径，用来给分段线性单孔提供不含 $z$ 离散误差的对照；
- $\mathtt{midpoint}$ 把厚度离散成薄层并作中点求积，用来检查 $\Delta z$ 收敛，也与未来离散折射率 volume 和 multislice 的层定义对齐。

即使使用 $\mathtt{analytic}$，代码仍会生成中点 $z$ 网格和对应的 $\mathtt{diameter\_z\_m}$ 供保存与绘图；但 $\mathtt{fill\_path\_length\_m}$ 的解析值本身不依赖 $\Delta z$。

### 5.2 非等长末层的中点网格

设用户指定的目标层厚度为 $\Delta z>0$，层数取为

$$
N_z
=
\left\lceil\frac{L}{\Delta z}\right\rceil.
$$

层边界记为

$$
z_j
=
\min(j\Delta z,L),
\qquad
j=0,1,\ldots,N_z.
$$

第 $j$ 层的真实宽度和中点分别为

$$
\Delta z_j=z_{j+1}-z_j,
$$

$$
\bar z_j
=
\frac{z_j+z_{j+1}}{2}
=
z_j+\frac{\Delta z_j}{2}.
$$

如果 $L$ 不是 $\Delta z$ 的整数倍，最后一层会短于其他层。使用每层真实宽度可保证

$$
\sum_{j=0}^{N_z-1}\Delta z_j=L,
$$

而不是把样品厚度错误地扩展到 $N_z\Delta z$。

### 5.3 指示函数的中点求积

对固定横向半径 $r$，定义

$$
h_r(z)
=
\mathbf 1\!\left[r\le R(z)\right].
$$

精确空气路径为

$$
\ell_{\mathrm{air}}(r)
=
\int_0^Lh_r(z)\,\mathrm dz.
$$

中点求积把每一薄层内的材料近似为该层中点处的材料：

$$
\ell_{\mathrm{air}}^{(\Delta z)}(r)
=
\sum_{j=0}^{N_z-1}
\Delta z_j\,h_r(\bar z_j)
=
\sum_{j=0}^{N_z-1}
\Delta z_j
\mathbf 1\!\left[r\le R(\bar z_j)\right].
$$

它的直观含义是：如果第 $j$ 层中点位于空气孔内，就把整层厚度 $\Delta z_j$ 计为空气；否则把整层计为玻璃。

普通中点公式对二阶连续函数常有 $\mathcal O(\Delta z^2)$ 的全局误差，但这里的 $h_r(z)$ 是阶跃函数，不能直接套用该结论。误差只来自孔壁穿过的层。

考虑一个宽度为 $w$ 的 crossing layer（指的是这一层既有空气也有玻璃，必然有一些层存在这样的量化误差）。若孔壁交点使该层中的真实空气长度为 $a$，则精确贡献为 $a$，而中点规则会把贡献近似为 $0$ 或 $w$。因为这个选择在交点经过层中点时切换，所以

$$
\left|\varepsilon_{\mathrm{crossing}}\right|
\le
\frac{w}{2}.
$$

对当前单腰分段线性轮廓，固定 $r$ 时最多有两个孔壁交点。定义最大层宽

$$
\Delta z_{\max}
=
\max_j\Delta z_j,
$$

则有一个实用的保守估计：

$$
\left|
\ell_{\mathrm{air}}^{(\Delta z)}(r)
-\ell_{\mathrm{air}}(r)
\right|
\lesssim
\Delta z_{\max}.
$$

因此这里预期的主要收敛行为是随 $\Delta z$ 近似一阶减小，而不是盲目假定二阶收敛。圆柱内部没有轴向交点；只要所有层宽之和严格等于 $L$，圆柱路径在 $z$ 方向上可被精确积分。

### 5.4 排序和后缀和为何与显式逐层求和等价

最直接的实现会为每个横向位置和每个 $z$ 层计算

$$
B_{jmn}
=
\mathbf 1\!\left[r_{mn}\le R(\bar z_j)\right],
$$

然后求和：

$$
\ell_{mn}^{(\Delta z)}
=
\sum_j\Delta z_jB_{jmn}.
$$

这样会临时占用一个形状为 $(N_z,N_y,N_x)$ 的布尔数组。当前代码没有创建这个三维 volume，而是利用“对于固定层半径，是否为空气只取决于 $r\le R_j$”这一性质作等价加速。

先定义每层中点半径

$$
q_j=R(\bar z_j).
$$

用排列 $\pi$ 将半径从小到大排序：

$$
q_{\pi(0)}\le q_{\pi(1)}
\le\cdots\le q_{\pi(N_z-1)}.
$$

同时按相同顺序排列层宽，并构造后缀和

$$
S_k
=
\sum_{m=k}^{N_z-1}\Delta z_{\pi(m)}.
$$

对某个横向半径 $r$，二分查找第一个满足

$$
q_{\pi(k)}\ge r
$$

的索引 $k$。从这个索引开始的所有层都满足 $r\le q_j$，所以

$$
\ell_{\mathrm{air}}^{(\Delta z)}(r)=S_k.
$$

这与显式指示函数求和完全相同，只是内存从保存 $(N_z,N_y,N_x)$ 布尔体改为保存一维排序数组和二维结果。对满足 $r\le\min_jq_j$ 的中央完整路径，代码还用积分首尾边界之差恢复

$$
S_0=L,
$$

以避免大量浮点层宽累加产生的微小漂移。

## 6. 横向采样与 supersampling：底层含义

### 6.1 输出网格坐标

二维数组顺序固定为

$$
(N_y,N_x).
$$

若 $\mathtt{dx}$ 是 tuple，其顺序固定为

$$
(\Delta y,\Delta x),
$$

不是 $(\Delta x,\Delta y)$。标量 $\mathtt{dx}$ 表示 $\Delta x=\Delta y$。

未平移的居中网格坐标为

$$
x_n
=
\left(n-\frac{N_x-1}{2}\right)\Delta x,
\qquad
n=0,1,\ldots,N_x-1,
$$

$$
y_m
=
\left(m-\frac{N_y-1}{2}\right)\Delta y,
\qquad
m=0,1,\ldots,N_y-1.
$$

因此横向半径样本为

$$
r_{mn}
=
\sqrt{(x_n-x_c)^2+(y_m-y_c)^2}.
$$

当 $N_x$ 或 $N_y$ 为偶数时，坐标原点位于中央像素之间，这是上述居中公式的直接结果，并非额外偏移。

### 6.2 只取像素中心时发生什么

当

$$
\mathtt{lateral\_supersampling}=1
$$

时，代码只在每个输出像素中心计算

$$
\ell_{mn}=\ell_{\mathrm{air}}(r_{mn}).
$$

连续圆边界随后被笛卡尔像素网格表示。对圆柱控制组，$\ell_{\mathrm{air}}(r)$ 在孔边界处跳变，像素中心一旦跨过边界，数值会从 $L$ 直接跳到 $0$。对锥形腰孔，路径在过渡环内连续，但斜率在 $R_w$ 和表面半径处改变，粗网格仍可能把圆形边界和环形斜坡表示成 staircase。

如果改变 $D_w$ 后，只有少量像素中心从边界一侧跳到另一侧，那么得到的“灵敏度”可能主要来自栅格化，而不是稳定的几何响应。

### 6.3 子像素中点平均

设横向 supersampling 因子为正整数 $s$。每个输出像素被划分为 $s\times s$ 个等面积子像素。第 $n$ 个输出像素内，第 $b$ 个子像素中心的 $x$ 坐标为

$$
x_{n,b}
=
x_n+\left(\frac{b+1/2}{s}-\frac{1}{2}\right)\Delta x,
\qquad
b=0,1,\ldots,s-1,
$$

相应地

$$
y_{m,a}
=
y_m+\left(\frac{a+1/2}{s}-\frac{1}{2}\right)\Delta y,
\qquad
a=0,1,\ldots,s-1.
$$

每个子像素中心的半径为

$$
r_{mnab}
=
\sqrt{(x_{n,b}-x_c)^2+(y_{m,a}-y_c)^2}.
$$

当前实现先计算每个子像素的空气路径，再作算术平均：

$$
\bar\ell_{mn}^{(s)}
=
\frac{1}{s^2}
\sum_{a=0}^{s-1}
\sum_{b=0}^{s-1}
\ell_{\mathrm{air}}(r_{mnab}).
$$

这是二维复合中点公式，近似输出像素内的面积平均路径：

$$
\bar\ell_{mn}
=
\frac{1}{\Delta x\Delta y}
\int_{\mathrm{pixel}_{mn}}
\ell_{\mathrm{air}}(x,y)\,\mathrm dx\,\mathrm dy.
$$

所以 supersampling 的直接作用是给边界像素一个 fractional path 或 fractional coverage，而不是强迫整像素只能取“完全空气”或“完全玻璃”。它会减弱 raster staircase，并允许在固定输出网格上更平滑地表示孔径的小变化。

### 6.4 平均路径后指数化不等于平均复透过函数

这一点是理解当前实现的关键。定义

$$
\alpha
=
k_0\left(n_{\mathrm{air}}-n_{\mathrm{glass}}\right)
\times\mathtt{phase\_scale}.
$$

当前代码计算的是

$$
T_{mn}^{\mathrm{code}}
=
\exp\!\left(i\alpha\bar\ell_{mn}^{(s)}\right).
$$

它**不是**先计算每个子像素的复透过函数再平均：

$$
\bar T_{mn}
=
\frac{1}{\Delta x\Delta y}
\int_{\mathrm{pixel}_{mn}}
\exp\!\left[i\alpha\ell_{\mathrm{air}}(x,y)\right]
\,\mathrm dx\,\mathrm dy.
$$

一般情况下

$$
\exp\!\left(i\alpha\,\mathbb E[\ell]\right)
\ne
\mathbb E\!\left[\exp(i\alpha\ell)\right].
$$

为了看清两者的差异，令像素内路径写为

$$
\ell=\bar\ell+\delta\ell,
\qquad
\mathbb E[\delta\ell]=0.
$$

对子像素复透过函数作二阶展开：

$$
\begin{aligned}
\mathbb E\!\left[\exp(i\alpha\ell)\right]
&=
\exp(i\alpha\bar\ell)
\mathbb E\!\left[\exp(i\alpha\delta\ell)\right]\\
&\approx
\exp(i\alpha\bar\ell)
\left[
1-\frac{\alpha^2}{2}
\operatorname{Var}(\ell)
\right].
\end{aligned}
$$

当一个像素内的路径变化很大时，平均复透过函数可能出现小于 $1$ 的有效振幅，这是子像素相位相消造成的；而当前实现始终满足

$$
\left|T_{mn}^{\mathrm{code}}\right|=1.
$$

因此当前 supersampling 应理解为**几何路径的抗混叠近似**，不是探测器 pixel integration，也不是严格的复场面积积分。这个选择与 exp030 的纯相位 projected object 定义一致，但它不会自动补回输出网格无法表示的高空间频率。

### 6.5 supersampling 与减小输出采样间隔的区别

增大 $s$ 会减小内部几何积分步长：

$$
\Delta x_{\mathrm{sub}}=\frac{\Delta x}{s},
\qquad
\Delta y_{\mathrm{sub}}=\frac{\Delta y}{s}.
$$

但最后仍只输出 $N_y\times N_x$ 个复数样本。因此：

- 增大 $s$ 主要检查同一输出像素内的边界面积平均是否收敛；
- 减小输出 $\Delta x$ 并保持物理视场不变，才会增加最终场的空间采样率和可表示带宽；
- 增大 $s$ 不能替代 fine-$\Delta x$ convergence；
- 减小 $\Delta z$ 只改善轴向路径积分，不能修复横向 staircase；
- 扩大横向视场主要减弱周期传播和边界截断影响，也不能替代减小 $\Delta x$。

这就是 exp030 同时要求 $\Delta z$、lateral supersampling 和输出 $\Delta x$ 收敛检查的原因。

### 6.6 视场、玻璃参考区和 Nyquist 带宽

输出网格所代表的像素区域宽度为

$$
\operatorname{FOV}_x=N_x\Delta x,
\qquad
\operatorname{FOV}_y=N_y\Delta y.
$$

令最大表面半径为

$$
R_{\max}=\max(R_t,R_b).
$$

对于可能偏离网格中心的单孔，要使整个孔位于视场内，至少需要

$$
|x_c|+R_{\max}<\frac{\operatorname{FOV}_x}{2},
$$

$$
|y_c|+R_{\max}<\frac{\operatorname{FOV}_y}{2}.
$$

实际计算还应在孔外保留非零玻璃 margin，使参考区中可以直接验证

$$
T_{\mathrm{proj}}=1.
$$

若后续使用 FFT 传播，这个 margin 也有助于区分真实衍射结构与有限视场、周期边界造成的**回卷影响**，这是dft卷积定理采用循环卷积造成的影响，可以理解有两种理解，1、输入被周期延拓，在卷积的过程中本该贡献为0的区域因为出现了周期而产生了贡献；2、输入和卷积核卷积后长度会比原来的输入长，但是输出要和输入一样大，所以卷积后的内容要折回来，故产生回卷。

一般的方法是通过把输入和卷积核变宽补0，也就是zero padding，这样折回来也就没影响，可以得到完整的结果。但是角谱传播的卷积核不是有限的，补0是无法完全消除影响的，只能尽量补然后让我们关注的区域的值收敛就可以了。

最终输出网格能表示的最高横向空间频率由 Nyquist 频率限制：

$$
f_{x,\mathrm{Nyq}}=\frac{1}{2\Delta x},
\qquad
f_{y,\mathrm{Nyq}}=\frac{1}{2\Delta y}.
$$

内部 supersampling 因子 $s$ 不会改变这两个频率。若腰部到表面的过渡宽度

$$
w_r=R_t-R_w
$$

只覆盖很少的输出像素，那么即使子像素面积平均已经收敛，最终的 transmission 和传播场仍可能欠采样。因此必须在固定物理视场下用更小的 $\Delta x,\Delta y$ 单独验证结果。

## 7. 如何区分物理灵敏度和离散误差

当前模型至少有三类彼此不同的离散误差。

### 7.1 轴向求积误差

它来自

$$
\ell_{\mathrm{air}}(r)
\longrightarrow
\ell_{\mathrm{air}}^{(\Delta z)}(r).
$$

对本实验的分段线性几何，可以直接用解析结果作为对照，并比较 $\Delta z$ 与 $\Delta z/2$ 的结果。该检查只隔离轴向积分误差。

### 7.2 像素内几何积分误差

它来自有限的 supersampling 因子：

$$
\bar\ell_{mn}
\longrightarrow
\bar\ell_{mn}^{(s)}.
$$

比较 $s$ 与 $2s$ 可判断边界 fractional path 是否稳定，但不能证明最终传播场的空间采样已经充分。

### 7.3 输出网格和传播带宽误差

它来自有限的 $\Delta x,\Delta y$、有限视场以及后续 FFT 传播的离散带宽。检查时应在保持同一物理视场和同一物理参数的条件下减小输出采样间隔，并把粗细网格结果映射到共同网格后再比较。

设 $Y_h(D_w)$ 表示网格尺度 $h$ 下得到的 transmission、probe 或 detector intensity。物理腰径中心差分为

$$
S_h
=
\frac{Y_h(D_w+\Delta D_w)-Y_h(D_w-\Delta D_w)}
{2\Delta D_w}.
$$

对应的网格离散变化可写为

$$
F_h
=
Y_h(D_w)
-\mathcal R\!\left[Y_{h/2}(D_w)\right],
$$

其中 $\mathcal R$ 表示将细网格结果按明确规则映射到粗网格。只有当 $S_h$ 在网格加密后保持稳定，并且参数扰动造成的差异明显高于 $F_h$ 所代表的 sampling/discretization floor，才能把该差异解释为当前 projected-phase 模型中的腰径信息。

对于标量灵敏度指标 $M_h$（相当于是从 $F_h$ 压成标量的一个人为选定的评价指标），可定义相对收敛变化

$$
\varepsilon_{\mathrm{conv}}
=
\frac{|M_h-M_{h/2}|}
{\max(|M_{h/2}|,\varepsilon_{\mathrm{num}})},
$$

其中 $\varepsilon_{\mathrm{num}}$ 只是防止分母为零的数值保护量。exp030 第一版目标是主要灵敏度指标在加密后的相对变化小于 $5\%$。如果达不到，应记录为采样未收敛或结论不确定，而不能把像素边界跳变解释成亚像素腰径可测性。

## 8. 实现输出与公式的对应关系

$\mathtt{make\_tgv\_projected\_phase}()$ 返回的核心数组均使用 $(N_y,N_x)$ 轴顺序：

| 返回字段 | 数学量 | dtype | 单位 |
|---|---|---|---|
| $\mathtt{fill\_path\_length\_m}$ | $\ell_{\mathrm{air}}(x,y)$ 或其离散近似 | $\mathtt{float64}$ | $\mathrm{m}$ |
| $\mathtt{opd\_relative\_m}$ | $\operatorname{OPD}_{\mathrm{rel}}(x,y)$ | $\mathtt{float64}$ | $\mathrm{m}$ |
| $\mathtt{phase\_unwrapped\_rad}$ | $\phi_{\mathrm{unwrapped}}(x,y)$ | $\mathtt{float64}$ | $\mathrm{rad}$ |
| $\mathtt{A\_effective\_true}$ | $A_{\mathrm{effective}}(x,y)$ | $\mathtt{complex128}$ | $1$ |

另外：

| 返回字段 | 数学量 | dtype | 单位 |
|---|---|---|---|
| $\mathtt{z\_m}$ | 各薄层中点 $\bar z_j$ | $\mathtt{float64}$ | $\mathrm{m}$ |
| $\mathtt{diameter\_z\_m}$ | 中点处直径 $D(\bar z_j)$ | $\mathtt{float64}$ | $\mathrm{m}$ |

实现中的可选相位尺度通过

$$
\phi_{\mathrm{unwrapped}}
=
k_0\operatorname{OPD}_{\mathrm{rel}}
\times\mathtt{phase\_scale}
$$

缩放相位。物理基线使用

$$
\mathtt{phase\_scale}=1.
$$

它可用于局部 Jacobian 中表示整体相位强度或折射率差方向，但不能在不说明含义时当作新的材料定律。

## 9. 适用条件与被忽略的物理

该近似适用于当前理想验证中的：

- 单个轴对称、圆截面、空气填充 TGV；
- 与 $z$ 轴重合的孔轴；
- 正入射单位振幅平行光；
- 标量、单色、近轴传播；
- 已知几何、材料、波长和传播距离；
- 无吸收的玻璃和空气。

它忽略：

- 样品内部横向衍射；
- 侧壁折射和高角度散射；
- Fresnel 界面反射；
- 多次反射和多次散射；
- vector polarization；
- 表面粗糙度；
- 倾斜、偏心、非圆孔和多孔相互作用。

因此 projected-phase 的数值 sensitivity 或局部 Jacobian 满秩，只能说明**在这个理想二维近似内部**存在局部参数信息，不能直接解释为真实三维 TGV 腰径已经可测，也不能替代含噪检测极限分析。

## 10. 与 Phase 4 multislice 的接口边界

Phase 4 应继续复用同一个 $D(z)$ 公共实现生成三维折射率 volume，再用 multislice 传播处理层间横向耦合。exp030 的 $\ell_{\mathrm{air}}$、$\operatorname{OPD}_{\mathrm{rel}}$ 和 $T_{\mathrm{proj}}$ 可以作为 Phase 4 的极限对照。

关闭层间横向传播时，逐层局部相位乘积应逼近

$$
\begin{aligned}
\prod_j
\exp\!\left[
ik_0\left(n_j-n_{\mathrm{ref}}\right)\Delta z_j
\right]
&=
\exp\!\left[
ik_0\sum_j
\left(n_j-n_{\mathrm{ref}}\right)\Delta z_j
\right]\\
&\longrightarrow
T_{\mathrm{proj}}.
\end{aligned}
$$

开启层间传播后，两者的差异用于量化样品内部横向衍射和侧壁几何效应。projected-phase 的参数敏感性不能作为 multislice 或真实硬件结果的先验结论。

当前 **src/tgv_ptycho/forward/multislice_A.py** 仍只是 Phase 4 的起点，不因 exp030 完成二维模型验证而视为 Phase 4 已实现。
