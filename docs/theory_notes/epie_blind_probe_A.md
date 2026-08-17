# 从盲 ptychography probe 回传到二维样品 A

## 问题定义

Phase 2 在同一二维采样网格上使用三段模型：

```text
U_A+ = A * U_inc
P_B = AngularSpectrum_zAB(U_A+)
psi_j = P_B * shift(B, r_j)
I_j = |AngularSpectrum_zBC(psi_j)|^2
```

相机只记录 `I_j`。未知量是 B 平面的 probe `P_B` 和扫描样品 `B`；得到 `P_B` 后，再计算

```text
U_A+_rec = AngularSpectrum_-zAB(P_B_rec)
A_rec_raw = U_A+_rec / U_inc
```

这仍是二维标量波动光学模型，不是 `n(x,y,z)` 的真实 TGV multi-slice 反演，也不能直接给出 `D(z)` 或腰径。

## 为什么不能直接无约束地同时更新 P_B 和 B

盲 ptychography 至少存在复数尺度、常量相位和线性相位斜坡歧义。规则 raster scan 还可能产生周期性伪解。若不引入物理先验，即使 detector data fidelity 很低，`P_B_rec` 也未必能在固定坐标系下唯一回传为 A。

exp020 用以下可审计先验建立最小闭环：

- A 是纯相位薄样品，圆形有效区外是已知空白参考区，传输应为 1；
- B 是纯相位编码样品，因此每帧总能量在当前单位传播模型下等于 probe 总能量；
- scan 在规则网格上加入带固定 seed 的整数像素抖动，减轻规则 raster 歧义；
- 每轮 ePIE 后把 probe 回传到 A，使用空白区拟合常量和线性相位背景，执行纯相位投影，再传播回 B；
- probe L2 norm 使用相机帧平均总能量确定，不读取仿真真值。

参考区相位平面拟合使用

```text
phi_ref(y,x) ~= c + ky*y + kx*x
```

校正后的 A 为

```text
A_ref = A_raw * exp(-i*phi_ref) / median_ref(|A_raw|)
A_phase_only = exp(i*angle(A_ref))
```

空白参考像素随后明确投影为 `1+0j`。这个约束依赖“已知空白区”实验设计，但不依赖 `A_true`。

## 与 exp010 的关系

两者共享同一 forward model、整数像素 shift、angular spectrum detector propagation、amplitude replacement 和 sequential ePIE update。区别是：

- exp010 固定已知 probe，只更新 B；
- exp020 同时更新 probe 和 B，并在每轮加入 A 平面物理投影；
- exp020 使用纯相位 B 的能量先验来固定 blind scale；
- exp020 最终保存 `A_rec_raw`、参考校正结果和纯相位结果。

## 仿真评估与可用输出的边界

`P_B_rec`、`B_rec`、`A_rec_raw`、`A_rec_reference_corrected` 和 `A_rec_phase_only` 是算法输出。A 的参考区校正不使用 truth，因此可作为当前实验设计下的实际反演结果。

为了量化盲重建固有歧义，仿真还可用 truth 搜索离散线性相位斜坡并拟合复增益。所有这类数组必须放在 `simulation_evaluation_only` 下；它们不能进入真实实验算法，也不能替代 raw reconstruction。

## 当前适用范围

- 无噪声、无坏点、无饱和；
- A/B 都是纯相位二维 transmission；
- A 的空白参考区形状和坐标已知；
- 传播距离、波长、采样和扫描位置精确；
- B shift 为整数像素和 periodic `np.roll`；
- object、probe、detector frame 同 shape，同 pixel size；
- forward 和 inverse 使用完全一致的 angular spectrum 模型。

因此 exp020 验证的是“在这些强先验下，相机 intensity 到二维 A phase 的数值链路可以闭合”。加入噪声、弱化 A 先验、使用 amplitude-phase B、有限 object support、subpixel shift 或真实 3D TGV 都应作为后续独立实验。

## 参考

更新主体沿用 exp010 所述 ePIE：A. M. Maiden and J. M. Rodenburg, *Ultramicroscopy* 109(10), 1256-1262 (2009), DOI `10.1016/j.ultramic.2009.05.012`。本实验增加的 A 平面投影是项目当前二维理想仿真的约束设计，不宣称为通用 production reconstruction engine。
