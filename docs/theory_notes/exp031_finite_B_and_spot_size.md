# exp031 理论：有限非周期 B 与 Gaussian 光斑尺度

本文记录可复用模型，不保存某次 run 的结果或阈值决定；实验矩阵、gates和结果以 `docs/experiment_design/exp031_finite_B_illumination_spot.md` 为准。

## 1. 三种不同的横向区域

`active interaction window` 是有显著 probe能量并实际与B调制相乘的物理区域；`detector ROI` 是固定保存/比较的输出物理区域；`open-boundary padding` 是把residual周期副本推远的数值区域。B patterned region必须覆盖前者和所有scan displacement，不需要覆盖纯数值padding。

对x轴，若active window宽度为 `L_window,x`、scan extrema为 `x_min/x_max`，连续几何下限是

$$L_{B,x}^{min}=L_{window,x}+(x_{max}-x_{min}).$$

y轴同理。加入每侧guard `g` 后先得到 `L_min+2g`，再向上对齐到phase-cell lattice和所需center parity；实际推荐值因此是planner输出，不是固定152 um。

### 本节判断

physical B size由active window与实际scan span决定，padding不能混入该公式。

### 下一步建议

exp04x迁移时复用scan-window API，但另做原pipeline回归。

## 2. Gaussian定义、功率与active window

采用

$$A=A_0e^{-r^2/w^2},\quad I=|A_0|^2e^{-2r^2/w^2},\quad D_{1/e^2}=2w.$$

全平面功率为 `P=|A0|^2*pi*w^2/2`。所以fixed-total使用 `A0=sqrt(2P/(pi*w^2))`，fixed-center使用相同A0。对centered square half-width `a`，解析captured fraction为

$$f_{square}=\operatorname{erf}(\sqrt{2}a/w)^2.$$

该式只控制incident Gaussian tail；TGV residual edge和detector ROI仍需单独收敛。

### 本节判断

大spot更均匀与fixed-power中心照度下降是两个不同机制。

### 下一步建议

报告二者，不把fixed-center冒充固定激光功率。

## 3. A到B的reference-plus-compact传播

Gaussian waist在A plane时，paraxial analytic reference可写为

$$G(r,z)=\frac{A_0e^{ikz}}{1+iz/z_R}\exp\left[-\frac{r^2}{w^2(1+iz/z_R)}\right],\qquad z_R=\frac{kw^2}{2}.$$

TGV场拆为 `G + G(T-1)`；只有后项在TGV top radius内compact，使用连续径向Fresnel--Hankel积分。plane wave是`w→infinity`的解析reference control。这样不会把有限二维数组边缘当成矩形光阑。

### 本节判断

只要analytic Gaussian与独立径向control通过，该方法与exp030的plane-reference思想一致。

### 下一步建议

曲率、未知waist plane和aberration应新开beam-calibration实验。

## 4. B到detector的open residual

令 `P_B=G_B+deltaP`、`B_s=1+M_s`，则

$$P_BB_s=G_B+[\delta P+P_BM_s].$$

reference解析传播到detector；residual零填充到open grid后用FFT传播。fixed detector ROI从open-grid中心裁取。active边缘能量检查source是否先被截断，padding/FOV control检查ROI是否受circular wrap影响。

### 本节判断

周围玻璃通过`G_B`形成相干reference；它会改变contrast、dynamic range和Poisson分母，不能删除。

### 下一步建议

真实B exterior/substrate transmission必须等待独立标定，不能由simulation选择。

## 5. Poisson Fisher information口径

把每帧incident power归一为一个入射光子，pixel mean为`mu=I*A_pixel/P_incident`，则

$$F_D^{(1)}=\frac1{N_f}\sum_{s,p}\frac{(\partial_D\mu_{sp})^2}{\max(\mu_{sp},\epsilon)},\qquad
\sigma_D\sqrt{N_{ph}}\ge\frac1{\sqrt{F_D^{(1)}}}.$$

这是conditional、shot-noise-only local bound。任何相机gain、full well、dark/read noise、stage或model mismatch都会改变实际信息。

### 本节判断

per-photon FI消除了纯场幅缩放，但保留spatial envelope、B coding和detector ROI造成的变化。

### 下一步建议

真实detectability应等待beam/B/detector标定并新开Phase 7实验。

## 6. 共享代码与科学结论的迁移边界

可迁移：large canvas、integer slice plan、paired scatter-add、coverage/margin maps、physical cells、多分辨率rasterization、Gaussian config/reference、open residual、spot/FI/dynamic-range metrics。不可迁移：2D projected TGV sensitivity数值、paraxial Gaussian充分性、known-B recovery ordering、shot-noise bound或任何真实3D waist结论。

### 本节判断

代码接口与科学证据必须分开迁移。

### 下一步建议

接口迁移转交未来exp04x任务；B-family优化转交exp041。
