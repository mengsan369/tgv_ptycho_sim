# exp040 理论：subvoxel TGV interface representation

本文只记录曲面界面在 Cartesian grid 上的可复用数值表示。具体 quadrature 序列、阈值、运行结果和
实验判定由 `exp040` 实验设计文档记录。

## 1. Voxel-center binary representation

设 slice center 为 $z_j$，TGV 横截面半径为 $r_j$。最简单的 binary voxelization 在 lateral voxel center
$(x_p,y_p)$ 判断

$$
(x_p-x_0)^2+(y_p-y_0)^2\le r_j^2.
$$

成立时赋孔内材料，否则赋基体材料。该方法确定且便宜，但曲面与 Cartesian cells 相交时会形成 staircase；
细小的半径或网格变化可能让一整圈 cells 突然切换材料。

## 2. Cell-averaged indicator

对 lateral cell $C_p$，更连续的表示是孔内面积分数

$$
f_{jp}=\frac{1}{|C_p|}
\int_{C_p}
\mathbf 1\!\left[(x-x_0)^2+(y-y_0)^2\le r_j^2\right]dx\,dy.
$$

使用 $q\times q$ staggered midpoint nodes 可得

$$
f_{jp}^{(q)}=\frac{1}{q^2}
\sum_{a,b}
\mathbf 1\!\left[(x_{p,a}-x_0)^2+(y_{p,b}-y_0)^2\le r_j^2\right].
$$

因为 weights 非负且和为一，$f_{jp}^{(q)}\in[0,1]$。当 $q=1$ 时，它退化为 voxel-center binary
indicator；随 $q$ 增大，可检查界面面积积分的数值收敛。

## 3. Phase-screen 中的 index representation

在标量 phase-screen 模型中，一种 cell-average representation 为

$$
n_{jp}=n_{\mathrm{matrix}}+
f_{jp}(n_{\mathrm{void}}-n_{\mathrm{matrix}}).
$$

对应层 transmission 为

$$
T_{jp}=\exp\!\left[
i k_0(n_{jp}-n_{\mathrm{ref}})w_j
\right].
$$

这里的 $n_{jp}$ 是对 unresolved indicator 的数值平均，目的是平滑几何积分误差。它不是 Maxwell
effective-medium theory，也不声称真实孔壁存在由两种材料均匀混合而成的介质。

## 4. 横向与轴向误差应分离

横向 subvoxel quadrature 改变的是每个 slice 内的 cell coverage。轴向 midpoint sampling、slice widths
和 split-step error 是另一组误差。若同时提高 lateral interface order 和 axial quadrature，就无法判断
输出变化来自哪一项。通常应先在固定 slice grid 上检查 $q$-convergence，再在固定已收敛 interface rule 下
做 $\Delta z$ refinement。

曲面可以流式逐 slice 生成，无需保存完整高分辨率 index volume。实现应检查 fraction bounds、index
bounds、subnode-count identity、slice-width sum、均匀介质控制和 determinism。

## 5. 解释边界

Subvoxel indicator 可以降低 staircase/voxelization floor，但不能补回：

- glass–air Fresnel reflection；
- backward wave 与 multiple scattering；
- polarization 或 high-NA vector coupling；
- sidewall roughness、tilt、noncircularity；
- 材料 absorption/dispersion。

因此，subvoxel convergence 只说明同一标量单向模型内的界面数值表示受控。若这些数值项和实验标定均受控
后仍存在结构化偏差，才应比较 Fresnel-aware、bidirectional 或 full-wave 模型。
