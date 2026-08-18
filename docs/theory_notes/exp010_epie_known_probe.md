# exp010：已知 probe 的 ePIE

## 实验目的

`exp010_epie_known_probe` 对应项目路线图中的 Phase 1。这个实验暂不引入 TGV 样品 A，而是直接在 B 平面定义一个已知 Gaussian probe `P_B`，让随机 amplitude-phase 样品 B 做二维扫描，生成 detector intensity stack，再只恢复 B 的 complex transmission。

这一阶段验证的是最基本的数据闭环：

```text
known P_B + random B + overlapping scan
-> I_stack
-> known-probe ePIE
-> B_rec + loss_curve + metrics
```

只有这个闭环稳定后，Phase 2 才会把 `P_B` 改成由样品 A 生成的未知 probe，并打开 probe update。

## 为什么选择 known-probe ePIE

本实验采用 Maiden 和 Rodenburg 在 2009 年提出的 ePIE 更新框架，但固定 probe，只更新 object。对当前项目，这是比 blind ePIE 或 rPIE 更合适的起点：

- forward model 是无噪声、已知传播距离的理想仿真，问题条件相对简单；
- 固定 probe 可以单独验证扫描位置、传播方向、幅度替换和 B 更新是否一致；
- ePIE 是 sequential update，代码短、物理含义直接，便于后续逐项调试；
- rPIE 对弱照明区和噪声通常更稳健，但多一个正则化权重，适合在 ePIE 基线通过后再加入。

因此这里的目标不是宣称 ePIE 在所有条件下优于 rPIE，而是建立一个透明、可复现的 Phase 1 baseline。

## Forward model

对第 `j` 个扫描位置 `r_j = (x_j, y_j)`，移动后的样品记为

```text
B_j(r) = B(r - r_j)
```

当前代码使用整数像素 `np.roll` 实现移动。exit wave 和 detector field 为

```text
psi_j(r) = P_B(r) B_j(r)
U_j(q) = A_zBC{psi_j(r)}
I_j(q) = |U_j(q)|^2
```

其中 `A_zBC` 是 angular spectrum propagation。项目在 B 平面和 detector 平面使用相同数组 shape；本阶段暂不加入 detector binning、pixel integration 或 sampling remap。

## ePIE 迭代流程

第 `k` 次迭代、位置 `j` 上的计算步骤如下。

1. 用当前 B 估计形成 exit wave：

   ```text
   psi_j^k = P_B B_j^k
   ```

2. 正向传播到 detector：

   ```text
   U_j^k = A_zBC{psi_j^k}
   ```

3. 保留预测 phase，用实测 amplitude 替换预测 amplitude：

   ```text
   U_j' = sqrt(I_j) exp(i angle(U_j^k))
   ```

4. 反向传播并计算 exit-wave correction：

   ```text
   psi_j' = A_-zBC{U_j'}
   Delta psi_j = psi_j' - psi_j^k
   ```

5. 只更新移动坐标系中的 B：

   ```text
   B_j^(k+1) = B_j^k
     + beta_B P_B* Delta psi_j / (max(|P_B|)^2 + epsilon)
   ```

6. 将更新量反向移动回 B 的全局坐标。

known-probe 模式下始终保持

```text
P_B^(k+1) = P_B^k = P_B_true
```

`epie_reconstruct()` 仍保留 `update_probe=True` 接口，供 Phase 2 的 blind reconstruction 使用，但 `exp010` 明确传入 `update_probe=False`。

## Loss、照明区域和 global phase

每轮记录的 data-fidelity loss 是所有位置 relative amplitude error 的平均值：

```text
L_k = (1 / J) sum_j
      || |U_j^k| - sqrt(I_j) ||_2 / (||sqrt(I_j)||_2 + epsilon)
```

由于 ePIE 在一轮内部顺序更新 B，`loss_curve[k]` 是该轮 sequential update 过程中的平均值。为了让运行前后比较不受扫描顺序影响，程序还会在迭代前后冻结参数，各自对全部位置重新评估一次，保存为 `initial_data_fidelity_loss` 和 `final_data_fidelity_loss`。

Gaussian probe 在数组边缘很弱，因此项目额外保存 illumination map：

```text
W(r) = sum_j shift_back_j(|P_B(r)|^2)
```

默认用 `W >= 0.05 max(W)` 定义 `illuminated_mask`。B 的主要误差指标只在这个区域计算，避免把没有有效约束的边缘混入算法质量判断。

仅使用 intensity 的相位恢复存在不可观测的 constant global phase。仿真评估时，用一个复数单位因子把 `B_rec` 对齐到 `B_true`：

```text
phi_0 = angle(sum(conj(B_rec) B_true))
B_rec_aligned = B_rec exp(i phi_0)
```

这个对齐只用于仿真 metrics 和 comparison figure。算法原始输出仍单独保存在 `B_rec`，真实实验也不会有 `B_rec_aligned_to_truth`。

## 当前配置

默认配置使用：

- array shape：`96 x 96`；
- B-plane sampling：`2 um`；
- wavelength：`532 nm`；
- `z_BC = 10 mm`；
- `9 x 9` grid scan，共 81 帧；
- scan step：`12 um`；
- Gaussian probe 的 `1/e` amplitude radius：`32 um`；
- B 为 4 pixel feature 的随机 amplitude-phase mask；
- 80 次迭代，`beta_object = 0.8`；
- 每轮随机打乱扫描位置，随机种子写入 config。

## 当前限制

- scan shift 只支持整数像素；
- B 使用 periodic boundary，即 `np.roll`；
- object 与 detector frame 使用相同 shape，没有 finite object patch；
- 暂无 detector mask、saturation、missing pixels 和 position refinement；
- 当前 exp010 默认无噪声；
- amplitude bounds 是已知 synthetic transmission 的稳定约束，真实实验需重新评估；
- 传播模型是 angular spectrum near-field model，不是默认 Fraunhofer FFT model。

## 参考文献

1. J. M. Rodenburg and H. M. L. Faulkner, “A phase retrieval algorithm for shifting illumination,” *Applied Physics Letters*, 85(20), 4795-4797 (2004). DOI: [10.1063/1.1823034](https://doi.org/10.1063/1.1823034)
2. A. M. Maiden and J. M. Rodenburg, “An improved ptychographical phase retrieval algorithm for diffractive imaging,” *Ultramicroscopy*, 109(10), 1256-1262 (2009). DOI: [10.1016/j.ultramic.2009.05.012](https://doi.org/10.1016/j.ultramic.2009.05.012)
3. A. M. Maiden, M. J. Humphry, M. C. Sarahan, B. Kraus and J. M. Rodenburg, “Further improvements to the ptychographical iterative engine,” *Optica*, 4(7), 736 (2017). DOI: [10.1364/OPTICA.4.000736](https://doi.org/10.1364/OPTICA.4.000736)
