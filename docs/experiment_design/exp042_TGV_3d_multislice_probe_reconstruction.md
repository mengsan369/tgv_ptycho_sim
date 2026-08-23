# exp042：3D TGV multislice probe reconstruction

```text
Scientific status: Planned / Not run
Work status: Placeholder / Design stage
Results available: false
```

## 0. 实时状态与阅读顺序

本节是本文唯一允许实时更新的状态入口；文首代码块和紧随其后的设计阶段说明是首次建档时的历史快照，
不再代表当前进度。第 1 节以后继续按历史设计与 append-only 实验记录阅读，不回改既有结论。

当前状态：`Stage A/B known-B development baseline complete / Paused pending Phase 5 feedback`。当前可复用输出是
exp040 scalar working model 下的 q4/q4 matched raw `P_B_rec`；最新有效 run 为
`runs/exp042_TGV_3d_multislice_probe_reconstruction_dev_20260822_195139`。这不是 scientific Passed；仍保持
`reference_validated=false`、`full_tgv_reference_authorized=false`，且只覆盖单一、无噪声、matched、known-B case。
按当前 roadmap，scalar `P_B_true` oracle fitting 预留为 `exp051`，本实验 raw `P_B_rec` 的对应下游 fitting 预留为 `exp053`。

建议阅读顺序：

1. 先读本第 0 节获取实时状态；
2. 第 1--15 节是最初固定设计边界；
3. 第 16--22 节是 Implementation iterations 01--07，建立 Stage A/B baseline、optimizer/初始化诊断和 bounded spectral evidence；
4. 第 23 节是第一次挂起记录：暂挂“最低谱端精确收敛”，转向直接消融；
5. 第 24--27 节是 Implementation iterations 08--11，完成初始化幅度与 q4/q1 detector data-model pairing controls；
6. 第 28 节是第二次挂起记录：exp042 整体阶段性挂起，后续由 Phase 5 的参数拟合证据决定是否定向恢复。

两次挂起互不覆盖：第 23 节保留最低谱端问题，第 28 节记录当前项目级暂停与恢复条件。以后发生新的实现、
恢复或路线变化时，在文末追加新节，并同步更新本第 0 节；不得用第 0 节改写历史证据。

本文档只建立 `exp042` 的科学实验设计骨架。当前没有对应的冻结 YAML、公共实现、运行入口、
测试、timestamped run、HDF5 reconstruction、metric、threshold 或 pass/fail 结论。以下所有尚未由
既有实验充分确定的算法、参数和验收条件均标为 `TODO / To preregister`，不得把候选方案理解为
已经选定或验证。

## 1. 实验目的与研究问题

`exp042` 要回答：

> 当 detector intensity 由 `exp040` 的 3D TGV multi-slice forward working model 产生时，
> ptychographic reconstruction 能否从 overlapping scan 数据中稳定恢复 B 平面复 probe
> $P_B$；在 blind 条件下，能否进一步联合恢复 $P_B$ 与编码样品 $B$？

关键场为

$$
P_B=\mathcal H_{AB}[U_{A,\mathrm{exit}}],
$$

第 $s$ 个扫描位置的 B-exit field 为

$$
E_{B,s}=P_B B_s,
$$

理想点采样写法为

$$
I_s=\left|\mathcal H_{BC}[P_BB_s]\right|^2.
$$

实际 `exp040` R4 以后还包含 finite sample-B support、transparent exterior、open
reference-plus-residual propagation、detector-node sampling、positive pixel quadrature 和 native ROI。
因此 exp042 的实际 measurement operator 必须继承所选 exp040 branch 的完整离散定义，不能只按上式的
同网格 point-intensity 简式实现。

本实验的主要研究对象是 field reconstruction / blind ptychographic reconstruction。即使输入数据来自
不同 `D_waist` case，exp042 也只判断各自的 $P_B$ 和 $B$ 是否可恢复，不把 case 间差异解释为腰径估计。

## 2. 在研究路线中的位置

计划链条为

$$
\text{exp040: 3D forward}
\rightarrow
\text{exp041: measurement / B design}
\rightarrow
\text{exp042: probe reconstruction}
\rightarrow
\text{exp05x: waist parameter inference}.
$$

各环节职责为：

- `exp040` 定义 3D TGV scalar multi-slice forward、B-plane truth 和 detector operator，并记录其
  numerical/reference-validation 边界；
- `exp041` 比较或选择具有物理可制造性的 sample B 与 scan/measurement design；
- `exp042` 在指定且冻结的 measurement design 上恢复 $P_B$，并在 blind branch 中恢复 $B$；
- `exp05x` 才使用 simulated 或 reconstructed probe signature 拟合低维 TGV 参数，包括
  $D_\mathrm{waist}$。

仓库目前已有 `configs/experiments/exp050_waist_parametric_fit.yaml` scaffold，但仅有占位 TODO；它不表示
exp05x 已经实现，也没有替 exp042 冻结 reconstructed-probe 接口。若 exp05x 先研究
`simulated P_B` 的参数拟合，该分支可以独立进行；凡以 `reconstructed P_B` 为输入的分支，必须等待
exp042 建立并验证相应输出契约。

## 3. Scope boundary

### 3.1 exp042 做什么

1. 使用 exp040 已定义、且为本实验明确冻结的 3D TGV forward branch 生成
   $U_{A,\mathrm{exit}}^{\mathrm{true}}$、$P_B^{\mathrm{true}}$、$B^{\mathrm{true}}$ 和 detector intensity；
2. 验证从 detector intensity 恢复 $P_B$ 的能力；
3. 按阶段进一步验证 blind reconstruction 中 $P_B$ 与 sample B 的联合恢复；
4. 分析 detector residual、probe error、sample-B error、收敛行为、初始化依赖和必要的
   operator consistency；
5. 用 known-B 与 blind-B 配对诊断，区分问题主要来自 probe recovery，还是来自 blind ambiguity/
   object update；
6. 为后续 exp05x 提供带完整 provenance 和质量信息的 raw reconstructed $P_B$。

### 3.2 exp042 明确不做什么

1. 不拟合 $D_\mathrm{waist}$；
2. 不从 reconstructed $P_B$ 直接输出腰径估计；
3. 不讨论最终腰径测量精度、分辨率、detection limit 或 uncertainty bound；
4. 不把 `18/20/22 um` 等 case 的 reconstruction 差异解释为参数反演结果；
5. 不重新承担 exp040 的 full forward-model reference validation，也不把 matched reconstruction
   解释成 exp040 已获得真实三维电磁准确性；
6. 不重复 exp041 对 sample-B family、编码性能、scan design、动态范围或制造可行性的系统探索；
7. 不以 TGV truth、$D(z)$ 或 $D_\mathrm{waist}$ 作为 reconstruction constraint 或 optimizer 输入；
8. 不把 truth-aligned simulation evaluation field 当作真实实验可获得的输出。

## 4. 与 exp030、exp040、exp041 和 exp05x 的边界

### 4.1 与 exp030

`exp030` 使用 2D projected-phase TGV model，已验证理想条件下的 probe/detector sensitivity 和 matched
blind differential recovery。exp042 可继承其 reconstruction 诊断经验，尤其是
`adjoint_residual`、known-B/known-probe ablation、truth fixed-point 和 gauge-aware evaluation；但不能
假定 exp030 的 reconstruction 可不经修改直接处理 exp040 数据。

exp030 的连续 radial Fresnel--Hankel A-to-B range constraint 只属于其 2D projected model。exp042
核心 reconstruction 把 $P_B$ 直接作为 B-plane unknown，不需要反演 $\mathcal H_{AB}$，也不得把 exp030
的 coarse ASM、radial inverse 或 A-plane pure-phase constraint 移植为 3D TGV constraint。

### 4.2 与 exp040

exp040 负责 forward truth 和 operator provenance；exp042 负责 inverse behavior。exp040 当前为
`Scientific status: Inconclusive`、`Work status: Frozen / Paused`，并明确
`reference_validated=false`、`full_tgv_reference_authorized=false`。因此 exp042 可以做的是：

- 在生成模型与 reconstruction model 一致的条件下检验 self-consistent inverse baseline；
- 将 exp040 尚未关闭的 physical/reference error 明确保留为上游模型限制；
- 不用低 detector residual 反向证明 exp040 物理正确。

若 exp042 需要改变 exp040 的核心 forward model、B support、detector operator 或 propagation branch，
该变化必须先作为新的 forward/governance 决策登记；不得在 reconstruction 内部静默建立另一套数据模型。

### 4.3 与 exp041

exp041 当前只是 `Discussion draft / Not started`，没有冻结候选 B、scan、阈值或 reconstruction protocol。
exp042 首次运行可以使用 exp040 已登记的 nominal finite-B working model，也可以使用以后由 exp041 选定的
B；具体来源是 `TODO / To preregister`。

exp042 只比较必要的 known-B 与 blind-B reconstruction，不在结果出来后重新选择 B family、feature size、
phase range、support/taper 或 scan pattern。若恢复失败提示 measurement design 信息不足，应把系统性的
B/scan redesign 返回 exp041，而不是在 exp042 内无限调换 encoder。

### 4.4 与 exp05x

exp042 的终点是经过质量审计的 reconstructed $P_B$；exp05x 的起点才是 probe signature 与参数化 TGV
forward/library/optimizer 的比较。exp05x 独立负责参数空间、loss、先验、identifiability、confidence/
uncertainty 和 waist accuracy。exp042 不预注册这些参数反演选择。

## 5. Forward truth 的来源与证据身份

### 5.1 允许的 truth 身份

exp042 的 simulation truth 必须由一个明确冻结的 exp040 branch 和 case 产生。候选可以是 exp040
q8 interface、finite `96 um` B、transparent exterior、q4 positive detector quadrature 和 open
reference-plus-residual path 组成的 scalar working model；最终 branch、case、grid 和 run provenance 为
`TODO / To preregister`。

这里的“truth”只表示生成 synthetic data 的数值 ground truth：

```text
simulation truth under the selected exp040 scalar working model
```

它不表示 reference-validated Maxwell truth 或真实器件 truth。所有文档、metadata 和图标题必须保留这一
证据身份。

### 5.2 必须成套保存的输入

同一次 forward generation 至少应向 exp042 提供：

- `I_stack` 和 `scan_positions`，positions 列顺序为 `(x, y)`、单位 m；
- reconstruction 所在物理 B plane 上的 raw $P_B^{\mathrm{true}}$；
- $B^{\mathrm{true}}$、modulation support、transparent exterior 和 shift convention；
- wavelength、external medium index、$z_{BC}$、所有 grid shape/dx/FOV/坐标原点；
- BC transfer、bandlimit/alias-control、padding/crop/restriction 和 FFT normalization 语义；
- detector node geometry、q factor、quadrature weights、pixel size 和 output ROI；
- exp040 source config/run/branch/case identity，以及 `reference_validated=false` 等状态字段。

$P_B^{\mathrm{true}}$ 与 `I_stack` 必须来自同一个完整 operator chain。不得把一个 cropped/native
$P_B$ 与另一个 open-grid detector branch 的 intensity 临时拼接，除非中间 mapping 及其 adjoint 已被
明确实现、测试并写入 provenance。

### 5.3 Truth 使用限制

Truth 只允许用于：

- simulation evaluation；
- operator reproduction、adjoint 和 truth-fixed-point control；
- 画误差图和计算明确标记的 truth-aided metrics。

Truth 不得用于 reconstruction initialization 选择、beta/regularization 筛选、early stopping、candidate
B 选择或实际输出的 gauge correction。若某个 control 必须 truth-initialize，应单列为
`simulation_diagnostic_only`，不能与主 reconstruction 混淆。

## 6. Reconstruction problem formulation

### 6.1 共同未知量和扫描模型

known-B branch 的未知量只有 $P_B$；blind branch 的未知量为 $(P_B,B)$。对 exp040 R5 以后的 finite-B
语义，建议沿用

$$
B=1+M,
$$

其中 $M$ 是有限 modulation，support 外为零。第 $s$ 个位置应由已登记的 shift operator作用于 $M$：

$$
B_s=1+S_sM,
$$

而不是对整个 $B$ 做 periodic `np.roll`。是否存在 taper、非单位 amplitude 或独立 calibration mask 为
`TODO / To preregister`。

### 6.2 实际 measurement operator

若使用 exp040 的 open reference-plus-residual detector path，单帧预测应抽象为完整的

$$
\widehat I_s=\mathcal M_s(P_B,B),
$$

其中 $\mathcal M_s$ 同时包含：

1. finite-B shift 与 transparent exterior；
2. $P_BB_s$ 的乘法；
3. homogeneous reference + localized residual 的 open-grid mapping；
4. alias-controlled $\mathcal H_{BC}$；
5. detector-node complex field；
6. $|U|^2$；
7. q4 positive staggered midpoint pixel average；
8. native detector ROI/crop。

于是 reconstruction 是对指定 data fidelity 的优化：

$$
\min_{P_B}\;\mathcal L(\{\mathcal M_s(P_B,B^{\mathrm{known}})\},\{I_s\})
$$

或

$$
\min_{P_B,B}\;\mathcal L(\{\mathcal M_s(P_B,B)\},\{I_s\})
+\mathcal R_P(P_B)+\mathcal R_B(B).
$$

data fidelity、optimizer、regularizer、constraint 与 stopping rule 均为 `TODO / To preregister`。普通
ePIE/rPIE 的 detector-plane amplitude replacement 只直接对应逐 node intensity；exp040 的 pixel-integrated
intensity 不提供每个 subpixel node 的测得 amplitude。因此不能未经推导把
$\sqrt{I_{mn}}$ 复制到 q4 nodes。候选方向包括显式 differentiable pixel-intensity loss，或另行证明与
quadrature operator 一致的 projection/update；选择留待预注册。

### 6.3 当前 ePIE 的可复用性边界

`src/tgv_ptycho/recon/epie.py` 已提供 known-object/probe-only、blind update、
`adjoint_residual`、periodic/constant integer shift、ePIE/rPIE denominator、checkpoint/resume 和 probe
constraint hooks。这些是可复用候选，不是 exp042 的既定算法。

当前实现仍以同 shape ASM field 和逐点 detector amplitude 为核心，不能自动表示 exp040 的 open padded
branch、grid mapping、q4 pixel quadrature或 finite support 的全部语义。exp042 实现阶段必须先决定是扩展
公共 reconstruction operator，还是使用统一的 loss/gradient 框架；不得通过降级 forward 数据来制造
虚假的 matched reconstruction。

## 7. 候选阶段划分

### Stage A：3D truth 数据接口与 reconstruction pipeline sanity

目标：在不更新未知量前证明 exp040 export 能被 reconstruction operator 原样重放。

候选检查包括：

- 从保存的 $P_B^{\mathrm{true}}$、$B^{\mathrm{true}}$ 和 positions 重算 `I_stack`；
- 检查 shape、axis、dx、FOV、坐标原点、support、ROI 和 units；
- 验证 detector positivity、quadrature constant/sum identity 和 determinism；
- 建立 forward/adjoint dot test、finite-difference gradient check 和 truth fixed point；
- 审计主 reconstruction 不读取 truth-only datasets。

Stage A 的正式 gate 与 tolerance：`TODO / To preregister`。

### Stage B：known-B probe recovery

固定完整且已知的 $B^{\mathrm{true}}$，只恢复 $P_B$。该阶段用于隔离：

- probe initialization 是否合理；
- detector measurement loss 与 probe gradient 是否正确；
- probe error 是否随 measurement-only objective 一致下降；
- residual 是否存在空间频率、scan position 或 ROI 相关结构；
- 迭代长度、步长和约束是否造成 truth fixed-point drift。

known-B 不是最终 blind 结论，而是 blind Stage C 的进入诊断。初始化、optimizer、约束、迭代预算、seed 和
进入条件均为 `TODO / To preregister`。

### Stage C：blind $(P_B+B)$ recovery

在 Stage B 的 operator chain 已通过后，同时更新 $P_B$ 和 $B$。必须与 Stage B 使用同一 measurement
operator、数据、scan 和 detector定义。除非预先登记，不允许只为 blind branch 更换 B support、loss、ROI
或 propagation branch。

Stage C 至少需要：

- raw $P_B^{\mathrm{rec}}$、raw $B^{\mathrm{rec}}$ 和完整 loss/residual history；
- joint gauge-aware simulation evaluation；
- illuminated support/coverage map；
- 多 initialization 或多 seed 的稳定性检查；
- 与 known-B 的 paired comparison；
- 对 scale/ramp/raster pathology 的显式诊断。

blind constraints、normalization来源、B amplitude/phase bounds、fiducial/known exterior 是否可用、成功率定义
和阈值均为 `TODO / To preregister`。

### Stage D：必要的 robustness / matched-case checks

Stage D 只处理为解释 Stage B/C 所必需、且预先登记的 matched controls，例如：

- forward/reconstruction 完全 matched 的主 control；
- 明确标记的 periodic-vs-finite、point-vs-quadrature 或 crop/padding mismatch negative control；
- iteration/checkpoint、初始化和 deterministic repeat；
- 若属于 exp042 已冻结问题，加入有限的 measurement perturbation。

系统性的 B-family/scan redesign 返回 exp041；noise、stage error、camera calibration 和真实数据若成为主要
研究问题，应按路线图使用新的实验任务。Stage D case、单因素范围和停止规则均为
`TODO / To preregister`，不得看结果后扩展成无界调参。

## 8. Truth 与 reconstruction 的比较方式

评价分两层，且 raw output 必须始终保留。

### 8.1 Measurement-only evaluation

不读取 truth，至少候选记录：

- frozen full-stack detector residual；
- per-scan residual 的 median/range/distribution；
- training trajectory 与独立 frozen reevaluation trajectory；
- objective、gradient/update norm、checkpoint 和 stopping reason；
- detector residual image/spectrum，及其对 scan position 的结构；
- 重算数据的 positivity、finite 和 energy/normalization controls。

具体 residual 应使用 intensity、amplitude、Poisson likelihood 或其他形式，取决于正式 measurement model，
为 `TODO / To preregister`。

### 8.2 Simulation evaluation only

使用 truth 的指标必须在名称和 HDF5 中标记 `simulation_evaluation_only`。候选包括：

- gauge-aligned $P_B$ complex relative L2；
- probe amplitude error、wrapped phase error和指定物理 ROI 内 error；
- gauge-aligned B complex/amplitude/phase error，仅在 illuminated support 上报告；
- scan-wise B-exit product $P_BB_s$ 的 gauge-invariant error；
- recovered error 相对已登记 numerical floor 或 true case separation 的比值，仅作为 field-recovery
  诊断，不解释为 waist inference；
- 不同初始化/seed 的 success rate 和 error distribution。

ROI、illumination threshold、denominator、aggregation 和 acceptance threshold 全部为
`TODO / To preregister`。

## 9. Gauge ambiguity 与对齐规则

### 9.1 Known-B

当 B 的 complex transmission、detector gain 和坐标系都绝对已知时，probe 仍至少有 intensity-only 的
global phase ambiguity。若 detector gain/incident energy 未绝对标定，还可能存在 amplitude scale ambiguity。
known-B 主结果保存 raw $P_B^{\mathrm{rec}}$；truth-aided global complex alignment 只用于 simulation
evaluation。

不得默认移除 spatial shift 或 phase ramp。只有当它确实属于所登记 forward 的 gauge、且对齐规则在运行前
固定时，才能在 evaluation copy 中处理。

### 9.2 Blind $P_B+B$

blind factorization 至少允许 reciprocal complex scale，形式为

$$
P_B' = cP_B,\qquad B'=B/c,
$$

并可能存在 affine/linear phase ramp、raster-grid pathology，以及 support/coordinate 未固定时的平移歧义。
phase ramp 对每帧可能只引入 detector intensity 不可见的常量相位，因此必须按 joint probe/B gauge 处理。

对齐必须：

1. 在运行前冻结允许消除的 gauge group、搜索范围、mask 和 objective；
2. 对 $(P_B,B)$ 联合拟合一个保持扫描乘积关系的 gauge，不能分别把 probe 和 B 各自对齐到最好看；
3. 先保存 raw reconstruction，再另存 aligned simulation-evaluation copy；
4. 同时报告对齐参数、是否触及搜索边界和 gauge-invariant $P_BB_s$ error；
5. 不把 truth-derived alignment 写回 optimizer、checkpoint 或 exp05x 的真实数据输入。

是否允许离散 shift、如何处理有限 support/fiducial、ramp 搜索网格和 scale convention 均为
`TODO / To preregister`。

## 10. Operator consistency / adjoint correctness

exp030 已给出两个必须继承的经验：

1. forward 可重放数据不等于 reconstruction update 使用了正确 adjoint；
2. band-limited ASM 中 $H^{\mathrm H}H\ne I$，因此 $H^{\mathrm H}$ 不能被当成 $H^{-1}$。

对 detector residual $r$，线性传播部分的反传必须使用同一 forward transfer 的严格共轭转置：

$$
\Delta E_B=\mathcal H_{BC}^{\mathrm H}r,
$$

而不是把“corrected detector field 的反向传播减去当前 exit field”当成一般公式。exact-data truth 必须是
update fixed point。

exp042 对所选 exp040 branch 至少需要逐项保持以下 forward/adjoint 一致：

1. **BC propagation**：相同 wavelength、medium、distance、FFT normalization、propagating mask、
   bandlimit 和 alias-control transfer；adjoint使用复共轭 transfer；
2. **finite-B shift**：对 $B-1$ 使用 constant-zero shift，并实现与 crop/zero-fill 严格配对的 adjoint；
   不得用 periodic inverse roll 代替；
3. **multiplicative exit wave**：probe/object Jacobian 使用同一 $P_BB_s$ 定义和正确 complex conjugation；
4. **reference-plus-residual open path**：homogeneous background、residual definition、center pad、open FOV、
   crop 与 native ROI 必须一致；由于该 forward 是 affine mapping，gradient 只通过相应线性 residual branch，
   但 forward prediction仍必须包含 reference；
5. **grid mapping/restriction**：任何 interpolation、center crop/pad、block mean 或 physical ROI mapping 都
   必须有真正的 transpose/scatter adjoint，不能把 forward interpolation 反向调用；
6. **detector quadrature**：q4 node geometry 和非负 $1/q^2$ weights 必须进入 forward 与 gradient；pixel
   residual向 nodes 的分配必须是 quadrature operator 的 adjoint；
7. **intensity nonlinearity**：$|U|^2$ 或所选 likelihood 的 Wirtinger/real gradient 必须与 loss 定义一致，
   并通过 finite-difference gradient test；
8. **weighted inner products**：当 B grid、open grid和 detector pixels 的采样面积不同，dot test 必须使用
   与离散物理积分一致的 weights；
9. **support/projection**：transparent exterior、known support、amplitude bounds 或 probe normalization 若
   被采用，必须验证 truth fixed point，且与 objective/optimizer 的顺序明确；
10. **AB/TGV branch**：核心 B-plane reconstruction 不需要 $\mathcal H_{AB}^{-1}$。若以后增加任何
    upstream range constraint，必须作为独立 control，使用 exp040 multi-slice/AB mapping 的一致
    forward/adjoint；不得复用 exp030 的 2D A-plane inverse，并且仍不得拟合 $D_\mathrm{waist}$。

必需的 operator controls 候选为：

- 每个线性子算子和组合算子的 complex inner-product dot test；
- full loss 的 directional finite-difference gradient test；
- truth pair 的 forward reproduction；
- known-B probe-only 与 blind truth-initialized one-step fixed point；
- zero residual 时零 update；
- adjoint dimensions、dtype、finite 和 deterministic checks；
- 故意 mismatch 的 negative control，证明测试确实能检测 periodic/open、point/quadrature 或 transfer
  不一致。

dot-test/gradient tolerance、随机种子和测试 grid 为 `TODO / To preregister`。

## 11. 主要 metrics（候选，尚未冻结）

### 11.1 Reconstruction/data metrics

- initial/final/minimum frozen detector data fidelity；
- per-iteration或 per-checkpoint loss，以及末段 slope；
- per-frame residual 的 median、maximum 和分布；
- detector residual relative to measured signal norm；
- gradient norm、relative update norm、stopping reason 和 runtime；
- 多 seed/initialization success fraction。

### 11.2 Probe metrics

- raw probe norm/energy和任何 measurement-derived normalization target；
- $P_B$ aligned complex relative error（simulation evaluation only）；
- amplitude relative error、wrapped phase RMSE 和 ROI/support-specific error；
- probe error 相对 exp040 registered numerical floor；
- 若运行多个 waist cases，recovered pair separation 对 true pair separation 的 field-level ratio，但不做
  参数拟合。

### 11.3 Sample-B metrics

- illuminated/support pixel fraction 和 coverage map；
- B aligned complex relative error（simulation evaluation only）；
- B amplitude/phase error；
- known transparent exterior violation；
- support leakage 或 boundary discontinuity；
- reciprocal-scale/ramp alignment parameters。

### 11.4 Operator/control metrics

- forward reproduction relative error；
- adjoint inner-product relative error；
- directional gradient relative error；
- truth fixed-point probe/B change；
- zero-residual update norm；
- detector quadrature constant/sum/positivity/node-geometry error；
- deterministic repeat error；
- gauge-invariant $P_BB_s$ error。

所有主 metric、denominator、aggregation、threshold、pass/fail 状态逻辑均为
`TODO / To preregister`。不得直接沿用 exp030 或 exp040 的 `5%`、`>=3` 等 gate，除非另行论证并在
exp042 正式运行前冻结。

## 12. Numerical controls / sanity checks

正式运行前至少考虑以下控制：

1. **shape/units control**：二维场 `(ny,nx)`、position `(x,y)` in m、`dx=(dy,dx)` 的接口检查；
2. **truth replay**：同一 truth pair 重算 full `I_stack`；
3. **zero-residual fixed point**：measurement 已完全匹配时不更新；
4. **adjoint/gradient control**：按第 10 节逐层及组合检查；
5. **detector control**：常量保持、sum identity、nonnegative intensity、q-series provenance；
6. **boundary control**：finite transparent exterior 与 constant-zero shift，无 periodic wrap；
7. **coverage control**：scan overlap、illuminated B support 和未照明区域分离；
8. **gauge invariance control**：施加允许的 reciprocal scale/ramp 后预测 intensity 不变；
9. **known-B isolation**：固定 truth B，只允许 probe 更新；
10. **known-probe isolation（可选诊断）**：固定 truth probe，只允许 B 更新，用于定位 object update；
11. **blind truth-start diagnostic**：只作为 simulation diagnostic，检查主更新是否破坏真解；
12. **initialization control**：truth-free initialization 的来源和 energy normalization 可从 measurement 重建；
13. **determinism/resume**：相同 seed 重复一致，checkpoint/resume 与连续 trajectory 一致；
14. **truth leakage audit**：主算法不读取 `/entry/truth`；
15. **grid/FOV control**：若 reconstruction grid 与 exp040 truth grid 不同，mapping 和误差 floor 必须单独登记。

哪些 control 是 hard gate、哪些只作诊断，以及 tolerance，均为 `TODO / To preregister`。

## 13. TODO / To preregister

正式实现和运行前至少冻结：

1. **exp040 source**：branch、case、source run/config、$D_\mathrm{waist}$ case 集合及证据身份；
2. **grid/operator**：B reconstruction grid、open FOV、padding、crop、ROI、bandlimit、alias control、q factor
   和 quadrature weights；
3. **sample B**：沿用 exp040 nominal B 还是等待 exp041；support、taper、amplitude/phase model、exterior 和
   known-B calibration level；
4. **scan**：position set、overlap、jitter、integer/subpixel shift 和边界；
5. **data model**：noiseless baseline 是否唯一主实验；若有 noise，分布、dose、gain 和 data fidelity；
6. **reconstruction method**：ePIE、rPIE、batch/gradient method 或其他 optimizer，以及选择依据；
7. **loss/update**：pixel-integrated intensity loss、amplitude projection 的可用性、adjoint/gradient 定义；
8. **constraints**：probe normalization、B amplitude/phase/support、regularization、是否使用 fiducial/known
   exterior；
9. **initialization**：probe、B、随机种子和是否运行多 initialization；
10. **schedule**：Stage A--D 的进入/停止规则、iterations、checkpoint、shuffle、beta/learning rate、筛选预算；
11. **gauge evaluation**：joint alignment group、mask、shift/ramp search、scale convention 和 failure flag；
12. **metrics/gates**：主指标、denominator、aggregation、threshold、success rate 和状态逻辑；
13. **robustness**：允许的 matched/mismatch controls，以及哪些变化必须返回 exp041 或新开实验；
14. **artifacts**：config、metadata、HDF5 fields、checkpoint、figure 和 runtime/memory budget；
15. **exp05x handoff**：raw probe field、quality flags、uncertainty representation、允许的 calibration/gauge
   information和拒收条件。

本节未作任何默认选择。

## 14. 与 exp05x 的输出接口

exp042 成功时，面向 exp05x 的最小候选接口应包含：

- raw `P_B_rec`，complex field，明确 plane、shape、axis、dx、FOV、origin、units 和 dtype；
- reconstruction 使用的 `I_stack`、`scan_positions` 和 B model/calibration identity；
- wavelength、medium、$z_{AB}/z_{BC}$ 参考平面和完整 detector/operator provenance；
- raw `B_rec`（blind branch）或 known-B identity；
- measurement-only loss/residual、coverage、convergence、stopping reason 和 quality flags；
- 若执行多 seed/initialization，保留 ensemble、代表解选择规则及失败成员，而不是只导出最好结果；
- simulation 中单独保存 `P_B_true/B_true` 和 aligned evaluation copy，字段名明确包含
  `simulation_evaluation_only`；
- exp040 的 `reference_validated=false`、source branch/status 和相关 numerical/model limitation。

exp05x 默认应消费 raw reconstruction 或由真实 calibration 决定的 gauge-corrected field，不能消费
truth-aligned evaluation copy。如何向 exp05x 表达 probe uncertainty、gauge nuisance、ensemble covariance、
invalid/low-quality rejection 和 exact HDF5 field names 为 `TODO / To preregister`。

建议遵循现有 `/entry` 并列结构：truth 放 `/entry/truth`，raw reconstruction 放
`/entry/reconstruction`，truth-aided对齐放清晰命名的
`/entry/reconstruction/simulation_evaluation_only`，metrics 与 config 分别放 `/entry/metrics` 和
`/entry/config_yaml`。若 exp042 需要改变项目级 HDF5 schema，必须先更新
`docs/theory_notes/data_format.md`；本设计文档本身不修改 schema。

## 15. 当前交付物与状态声明

当前唯一交付物是本文档：

- 无 YAML config；
- 无公共模块、脚本或 notebook 实现；
- 无新增或修改测试；
- 无 timestamped run；
- 无 HDF5、metrics、checkpoint 或 figures；
- 无 preregistered threshold；
- 无 Passed/Failed/Inconclusive 实验结果。

后续只有在完成正式预注册、实现、测试、实际运行、artifact 审计和结果回填后，才能改变本文开头的
`Planned / Not run` 与 `Results available: false`。

## 16. 2026-08-20 17:30：Implementation iteration 01 — Stage A/B known-B baseline

### 本轮目标与明确未做事项

本轮只建立第一次可运行的 development baseline：先闭合 exp040 working-model truth/data 与 exp042
measurement operator 的接口和 Stage A operator-consistency controls，再在 known-B 条件下只更新 B 平面
复 probe `P_B`。本轮状态固定为
`Development baseline / No scientific pass-fail conclusion`，没有改写本文顶部状态。

本轮没有执行 blind `P_B+B`、`D_waist` 拟合或任何腰径输出；没有选择新的 B family、改变 exp040 的
历史 scan、加入 noise/stage error/subpixel shift/真实 detector calibration，也没有运行或修改任何 exp040
Helmholtz/reference-validation pipeline、配置、run、状态或冻结结论。没有修改 exp040、exp041、exp050、
README、roadmap、notebook 或 report。

### 开始时检查的内容

- 首先执行 `git -c safe.directory=E:/tgv_ptycho_sim status -sb`。开始时已有用户的 modified、deleted 和
  untracked 内容，包括修改中的 exp030/theory note、删除的 notebook、未跟踪的 exp041/exp042 文档、
  VS Code 配置、notebook 导出和 report；全部保留，没有覆盖、删除、回退或暂存。
- 完整读取根目录 `AGENTS.md` 和本文第 1--15 节。开始修改前本文为 `28602 bytes`，SHA256 为
  `BE3091E80E01879BA32E459C570E17FFAFD842C49C697B68251A66121789E1EB`，最后章节为第 15 节。
- 对当前 primary `docs/experiment_design/exp040_TGV_3d_multislice_forward.md` 做定向读取：文首状态、第
  1--3 节、R4、R5、R8 全部定义/结果/结论、R9 正式结果、R10 Stage A 最终结果、R10 Stage B 最终状态、
  R11/R12/R13/R14A/R14B 最终状态和结论边界，以及第 19 节冻结决定。没有读取 `_old.md`，也没有顺序
  通读该 primary 的全部执行异常、repair/hash/wrapper 历史。
- 从上述证据确认：R8 的 q4 positive quadrature、finite `96 um` B、transparent exterior、对 `B-1`
  的 constant-zero shift 和 residual open path 是本轮 matched branch 的语义来源；R9 关闭 axial refinement，
  R10 Stage A 关闭 scalar lateral reference，但 R10 Stage B 至 R14B 从未授权 full-TGV reference。
  本轮持续继承 `reference_validated=false`、`full_tgv_reference_authorized=false`。
- 文档读取后只用定向 `rg`/行范围定位：exp040 runner 的实际 imports/R8 路径，
  `src/tgv_ptycho/forward/exp040.py` 的 `center_crop`、`_center_pad`、`_r4_block_mean`、
  `_r5_homogeneous_a_exit`、`_r8_open_context`、`_r8_detector_stack`、`_run_r8_diagnostics`，现有
  `recon/epie.py` 的 adjoint update 边界，`angular_spectrum.py`、`camera.py`、`integer_shift.py`、
  `multislice_A.py`、TGV/sample-B/scan generators、直接 IO writer，以及 R4/R5/R8/ASM/HDF5 直接测试。
- 为创建 run 与 provenance，额外读取了直接依赖的 `io/config.py`、`io/naming.py`、`io/metadata.py`；
  这是唯一的小范围扩展，原因是新 runner 必须创建 timestamped run、保存 config/metadata 并写入既有 HDF5
  contract。得到所需签名后即停止，没有扫描整个 `src`、`tests` 或 `runs`。
- 没有遍历历史 runs 猜测 artifact。既有 R8 formal run 只作为 branch provenance；它没有成套导出 exp042
  所需的 R8 internal open-grid `P_B_true/B_true/I_stack`。因此本轮没有复用数据 artifact，而是用共享 exp040
  building blocks 生成一个缩小、确定性、生成与重建完全 matched 的 development case。

### 上一轮意见

无上一轮 implementation record。首次判断来自本文固定设计边界、上述 exp040 evidence 和实际源码：
主要矛盾是尚无可证明与 q4/finite-B/open chain 匹配的 reconstruction operator；次要矛盾是 grid/坐标契约、
truth provenance 和可重复诊断；收敛速度、更多 case/seed、复杂先验、blind recovery 和 robustness 延后。

### Change 01

#### 改动前：主要矛盾、次要矛盾与必要性判断

- 主要矛盾：缺少包含 open/reference affine mapping、alias-controlled BC、finite-B shift、q4 pixel intensity、
  crop 和严格 adjoint 的统一 probe operator；在它闭合前，任何 recovery 曲线都不可解释。
- 次要矛盾：明确 `(ny,nx)`、`(x,y)` in m、native/open FOV 和 q-node/pixel geometry；建立同链 truth
  provenance 与 deterministic controls；避免 truth 泄漏。
- 延后：优化器调速、blind B update、noise/subpixel/calibration、full-size formal cost、waist inference。
- 优先做本 Change，因为它影响科学正确性和 exact-data fixed point，比先调迭代次数或步长更必要。

#### 技术决策与理由

新增 `MatchedKnownBProbeOperator`。未知量是 native B-plane `P_B`；open grid 由
`P_0_open + E(P_B-P_0_native)` 构造。每帧只对已知 finite modulation `M=B-1` 做 shared
constant-zero integer shift，并计算

```text
P_open = P_0_open + E(P_B - P_0_native)
R_exit = E(P_B - P_0_native) + P_open * S_s(M)
U_D = U_0,D + H_BC(R_exit)
I_pixel = center_crop(q4_mean(|U_D|^2))
```

这与 exp040 R8 的 reference-plus-residual 写法代数一致。BC forward 使用 shared alias-controlled transfer，
field adjoint使用同一 transfer 的复共轭；center pad/crop、detector crop/scatter 和 q4 block average 都有配对
transpose。q4 data fidelity 是显式 pixel-intensity loss，没有把 `sqrt(I_pixel)` 复制到 subpixel nodes。

development truth 使用 canonical TGV 几何、q8 cell-average interface 和 shared streamed scalar multislice；
为了低成本，A/P native grid 为 `96x96 @ 0.5 um`（FOV `48 um`）、`dz=1 um` 共 100 slices，open grid 为
`256x256 @ 0.5 um`（FOV `128 um`）。B 是同 seed `20260840`、`48x48` cells、`2 um` feature 的 finite
`96 um` hard square；scan 保留 `5x5`、`4 um` step、`1 um` integer jitter 与 seed `20260841`；detector
为 q4、pixel `2 um`、centered `32x32` ROI。A-to-native mapping 在本 development case 中有意选为 identity，
没有未配对 interpolation。

#### 创建或修改的文件

- 创建 `configs/experiments/exp042_TGV_3d_multislice_probe_reconstruction.yaml`；
- 创建 `src/tgv_ptycho/recon/exp042.py` 的 matched operator、deterministic fixture、loss/gradient 和 controls；
- 创建 `tests/test_exp042_probe_reconstruction.py` 的 Stage A tests。

没有修改任何 shared forward/optics/shift/IO API 或数值行为；只调用其现有公共实现。

#### 验证命令与结果

首次 Stage A targeted 命令为：

```powershell
python -m pytest -q tests/test_exp042_probe_reconstruction.py `
  -k "config_and_shapes or truth_replay or adjoint_dot_products or full_loss_gradient"
```

结果为 `4 passed in 8.29s`。这次 shell 默认解释器是 Python 3.12，仅作为实现中的快速检查；最终权威验证
在 Change 02 末尾用已激活的 Python 3.11 Conda 环境重跑。

#### 改动后重新评估

Change 01 的主要矛盾已关闭：truth replay、full linear field Jacobian dot test、constant-zero shift adjoint、
physical-area-weighted q4 adjoint、full-loss finite difference 和 zero-residual fixed point 全部通过。没有出现
point/q4 或 inverse/adjoint mismatch。下一主要矛盾转为 measurement-only optimizer、runner 和 artifacts 尚未
闭环；收敛速度/条件化仍是次要项，blind/robustness 继续延后。

### Change 02

#### 继承 Change 01 的评估与改动前判断

继承已通过的 matched operator，不改 forward semantics。主要矛盾是 known-B `P_B` update 和可审计输出尚未
实现；次要矛盾为 truth-free initialization、单调 loss 保障、raw/aligned output 分离和 artifact validator；
更高质量的优化器、更多 initialization 和规模扩展延后。完成最简单的 batch complex gradient descent 比加入
blind/rPIE/大型 autodiff 框架更符合本轮范围。

#### 技术决策与理由

- optimizer 只接收 operator、measured `I_stack`、homogeneous-reference initialization 和固定 settings；API
  不接收 truth。使用 full-batch real-complex gradient 与 Armijo backtracking，60 iterations；sample B 永远不更新。
- optimizer 只按 measurement loss 接受 step 和停止。probe history 仅在优化完成后用于
  `simulation_evaluation_only` error curve；truth-aligned copy 不覆盖 raw `P_B_rec`。
- runner 复用 `make_run_dir()` 和项目 HDF5 writer，写入 config/metadata/metrics/run_state、三个 PNG 和一个
  HDF5；内置 validator 检查 required files、`/entry` layout、shape、finite arrays、raw/aligned 路径与 PNG decode。
- 为规避 Windows 上未激活 Conda 时大数组 BLAS reduction 的 DLL dispatch 异常，dot/norm diagnostics 使用数学
  等价的显式 `sum(conj(a)*b)` reduction；这不改变 forward、gradient、config 或 scientific operator。

#### 创建或修改的文件

- 扩展 `src/tgv_ptycho/recon/exp042.py`：known-B optimizer、measurement residual、post-hoc phase alignment；
- 创建 `scripts/run_exp042_probe_reconstruction.py`；
- 扩展 `tests/test_exp042_probe_reconstruction.py`：8-step recovery trend 和临时目录 artifact integration；
- YAML 保持在看到正式 reconstruction 结果前锁定，没有在 run 后修改 shape、seed、scan、step 或 iteration budget。

#### 验证命令与结果

实现中的默认解释器快速检查：

```text
python -m pytest -q tests/test_exp042_probe_reconstruction.py
6 passed in 9.66s
python -m ruff check ...
failed: default Python 3.12 environment did not install Ruff
```

随后直接调用 `D:\anaconda3\envs\tgv_ptycho_sim\python.exe` 时，pytest 分别在 NumPy `vdot/norm` 和
Matplotlib `linalg.inv` 路径出现 Windows fatal `0xc06d007f`；直接 `np.linalg.inv(I)` 也能复现。原因不是
operator test failure，而是直接启动 env executable 没有注入 Conda `Library\bin` DLL path。使用
`conda run -n tgv_ptycho_sim` 后 `numpy.linalg` control 正常。最终权威命令为：

```powershell
D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim `
  python -m pytest -q tests/test_exp042_probe_reconstruction.py
# 6 passed in 8.06s

D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim `
  python -m ruff check src/tgv_ptycho/recon/exp042.py `
  scripts/run_exp042_probe_reconstruction.py `
  tests/test_exp042_probe_reconstruction.py
# All checks passed!
```

targeted tests 覆盖 config/shape/dtype/units、truth replay、detector controls、三类 adjoint、full gradient、
zero residual、truth one-step fixed point、determinism、8-step known-B loss/residual/probe-error下降和临时 HDF5/
三图 contract。没有修改 shared module，所以未运行 full pytest；避免对未受影响的既有实验做无必要扩大回归。

正式 development command 只执行一次：

```powershell
D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim `
  python scripts/run_exp042_probe_reconstruction.py `
  --config configs/experiments/exp042_TGV_3d_multislice_probe_reconstruction.yaml
```

shell exit `0`，外层墙钟 `23.71 s`，runner 记录 runtime `18.3641 s`。

#### 改动后重新评估

Change 02 的“最小 pipeline 与 artifact 闭环”主要矛盾已关闭：optimizer/runner 可运行，measurement loss、
detector residual 和 simulation-only probe error 都单调下降，artifacts 全部验证。新证据同时表明恢复幅度很小：
60 步均接受 step `1`、无 backtracking，gradient norm 只从 `0.00273701` 到 `0.00270864`；probe 图仍接近
homogeneous initialization。故“是否具备有效的 measurement-only scaling/conditioning”成为下一轮主要矛盾，
而不是继续优化本轮次要画图或扩展 blind scope。本轮没有科学 Passed/Failed 结论。

### Operator-consistency 检查

以下均由正式 run 实际执行并写入 JSON/HDF5：

| control | result |
|---|---:|
| truth replay full `I_stack` relative L2 | `0` |
| full field-Jacobian adjoint relative error | `1.4510142628728665e-15` |
| constant-zero shift adjoint relative error | `5.682234924561534e-16` |
| q4 physical-area-weighted adjoint relative error | `0` |
| full-loss directional gradient relative error | `1.2347820022705336e-9` |
| truth loss / truth gradient norm | `0 / 0` |
| truth-initialized one-step relative change | `0` |
| zero-residual update norm | `0` |
| detector constant / sum error | `0 / 0` |
| detector node-geometry normalized error | `6.776263578034403e-15` |
| minimum detector intensity | `0.04091490300503436` |
| finite-B shifted boundary edge modulation | `0` |
| deterministic repeat relative L2 | `0` |

所有 arrays finite、所有 detector outputs nonnegative。finite B exterior 为 `1+0j`，shift 只作用于 `B-1`；
open mapping包含 centered pad/restriction 与 reference field。本 case 的 A-to-native mapping 是显式 identity，
所以未执行 interpolation adjoint；detector crop/scatter 和 q4 transpose 已由 full gradient 覆盖。没有 amplitude
replacement，也没有 point-detector simplified primary branch。

### Development run 与 artifacts

唯一 workspace timestamped run：

```text
runs/exp042_TGV_3d_multislice_probe_reconstruction_dev_20260820_172734
```

- `config.yaml`：`3592 bytes`；source YAML SHA256
  `08FB573AEB454E0842B0BFC5EC08CA4D5D8750325D862EC2DD70E9D265CC5CFC`；run copy 经过 YAML
  round-trip，SHA256 为 `929544AC6D5918230B05E08AB8F32E498E1108C6E2D477E21CF8E39615C46215`。
- `metadata.json`：`913 bytes`，SHA256
  `DADDCBA6780BE273C5A47662304931096351D597110BF8A453D9DEF0FAF89E7C`；明确记录 branch、known-B、
  probe-only、truth-free optimizer 和两个 false reference flags。
- `metrics.json`：`5631 bytes`，SHA256
  `D4A31A9DC7964BDB3CC67181F6F580D05E38AA987DC4065D0AA750EF2071EBBE`。
- `outputs/exp042_probe_reconstruction.h5`：`2657504 bytes`，SHA256
  `1C9B0DF5C692044BCFF570A2E75694C28E8FC3068018A6E75EF3A96BA22978AA`。
- 三图已实际打开检查：曲线均单调下降；probe update 视觉上很小；final detector residual 保留明显结构。
  PNG 分别为 `97558/66384/34272 bytes`，SHA256 为
  `A804871DF6ABA67670CD5F7EA8852F249ECF47F2D2E09B110D325D83640E8E7A`、
  `EC82544450008C7F2751B1DBB6A35F9A6AC9385A5D38D7137C30EEA10DF88750`、
  `7CA37C5FAE13C896DC7A394541402F68807CD4FB5D393ADFC01B21CE8783DD86`。
- `run_state.json` 为 `complete`、`artifacts_validated=true`。required files、JSON、HDF5、三图均通过 runner
  validator 和本轮只读复核。

HDF5 `/entry` 恰含
`config_yaml/data/instrument/metadata/metrics/reconstruction/sample/truth`。主要自然字段为：

```text
/entry/data/I_stack                                      (25,32,32) float64
/entry/data/scan_positions                               (25,2) float64
/entry/truth/P_B_true                                    (96,96) complex128
/entry/truth/B_true                                      (256,256) complex128
/entry/truth/B_support_true                              (192,192) complex128
/entry/truth/U_A_exit_true                               (96,96) complex128
/entry/truth/z_m, slice_widths_m, D_z_m                  (100,) float64
/entry/reconstruction/P_B_init                           (96,96) complex128
/entry/reconstruction/P_B_rec                            (96,96) complex128
/entry/reconstruction/loss_curve                         (61,) float64
/entry/reconstruction/detector_relative_residual_curve   (61,) float64
/entry/reconstruction/gradient_l2_norm_curve             (61,) float64
/entry/reconstruction/simulation_evaluation_only/
  P_B_rec_global_phase_aligned                           (96,96) complex128
```

raw `P_B_rec` 单独保留；truth-aligned field 只位于 `simulation_evaluation_only`，没有反馈 optimizer。
没有 calibration/preprocessing 空 group，没有改变项目级 HDF5 schema。

### 当前 metrics

measurement-only：

```text
initial loss                         0.03127716305291964
final/minimum loss                   0.03083230643594776
initial detector relative residual   0.356762301025841
final detector relative residual     0.354216091073014
loss nonincreasing                   true
iterations completed                 60
stopping reason                      iteration_budget
accepted steps                       {1.0}; no backtracking
```

simulation evaluation only：

```text
initial/final raw probe relative L2       0.480288973912617 / 0.478961549257642
initial/final phase-aligned relative L2   0.478736914953423 / 0.477422108679947
final unit phase factor                   0.999284617579477 - 0.0378186867704205j
```

这些值只证明方向正确且变化可重复；没有预注册 recovery threshold，没有给出 `Passed` 或 `Failed`，也没有
使用 exp030/exp040 的 5% 或 signal/floor gate。`waist_estimation_performed=false`。

### 失败、限制与未关闭问题

- 三次直接启动 env executable 的 fatal DLL-path 失败已保留在本轮命令记录；它们没有创建 workspace formal
  run，也没有产生或删除科学 evidence。正确的 `conda run` 路径随后完整通过。
- 本 run 是缩小且 matched 的 development fixture，不是 R8 formal 384-um open grid artifact 的重建；native
  `dx=0.5 um`、`dz=1 um`、open FOV `128 um` 均为成本受控选择。没有在本轮建立 open-FOV、axial 或 lateral
  convergence gate。
- final detector residual 仍高，probe error 只小幅下降；当前结果不支持“稳定高质量恢复”的陈述。可能因素包括
  unpreconditioned gradient scale、cropped q4 measurement conditioning 和当前 iteration budget；本轮没有用 truth
  选择更有利的 step/optimizer，也没有看后修改 YAML 或重跑第二个 formal run。
- homogeneous reference initialization 来自已知 forward/operator calibration，不是 truth；真实实验能否获得相同
  reference calibration仍未研究。
- B 是 deterministic nominal simulation truth，不是实测 calibration；没有验证 exp041 的信息丰富性或制造可行性。
- `reference_validated=false`、`full_tgv_reference_authorized=false`；matched inverse 不能证明真实三维电磁准确性、
  真实 TGV probe recovery、腰径精度、分辨率或 detection limit。

### 改动后总体优先级

- **下一轮主要矛盾（最多一个）**：在不读取 truth 的前提下闭合 measurement-only optimizer scaling/
  conditioning。最值得做的是先预注册一个 operator/Hessian-action 或 gradient-scale diagnostic，再据此实现一个
  有明确 adjoint依据的 normalized/preconditioned probe step；预期作用是判断当前缓慢下降来自 step scale 还是
  q4/crop observability，而不是盲目增加 iterations。
- **次要矛盾 1**：增加 per-scan residual distribution/spectrum 和 native-grid coverage/conditioning diagnostics，
  用于定位 residual 的空间/scan structure；它提高诊断力，但不能先于主要 optimizer 闭合无限扩张。
- **次要矛盾 2**：增加一个预注册的 point-vs-q4 或 periodic-vs-finite negative mismatch control，证明 control
  会主动拒绝错误 operator；当前正向 controls 已充分闭合第一次实现，故不阻止本轮 baseline。
- **次要矛盾 3**：在 optimizer 固定后再做一个额外 truth-free initialization/seed repeat，用于判断初始化依赖；
  本轮单 seed 只建立 determinism，不建立 success rate。
- **明确延后**：blind B update、sample-B redesign、noise/stage/subpixel/calibration、full-size R8 reconstruction、
  upstream range constraint、任何 `D_waist` inference 和 full electromagnetic reference。

上述主要工作仍属于 exp042 的同一 known-B Stage B，不需要新实验编号；若问题转为 B/scan redesign，应返回
exp041；若转为真实数据 robustness 或新的 forward physics，应新开实验任务。

### 下一轮快速恢复上下文

- 当前 authoritative appended section：本文第 16 节。
- 本轮完成的 Change：Change 01 matched operator/Stage A controls；Change 02 truth-free known-B optimizer、runner、
  artifacts 和一次 development run。
- 本轮修改文件：
  `configs/experiments/exp042_TGV_3d_multislice_probe_reconstruction.yaml`、
  `src/tgv_ptycho/recon/exp042.py`、`scripts/run_exp042_probe_reconstruction.py`、
  `tests/test_exp042_probe_reconstruction.py`，以及本文末尾本节。
- 当前有效 config：上述 exp042 YAML，source SHA256
  `08FB573AEB454E0842B0BFC5EC08CA4D5D8750325D862EC2DD70E9D265CC5CFC`。
- 最新有效 run：
  `runs/exp042_TGV_3d_multislice_probe_reconstruction_dev_20260820_172734`，complete/validated。
- 已通过命令：Python 3.11 `6 passed in 8.06s`；修改范围 Ruff `All checks passed`；唯一正式 runner exit `0`；
  artifact/HDF5/PNG read-back 均通过。
- 已确认不需要重跑的 controls：只要 operator/config 未变，本节列出的 truth replay、field/shift/q4 adjoint、full
  gradient、zero residual、fixed point、determinism无需机械重跑；exp040 R4/R5/R8/R9/R10A/R13/R14A 既有
  controls 和 R10B--R14B reference failures也不需要重算。
- 当前未关闭的主要矛盾：measurement-only optimizer scaling/conditioning。
- 当前次要矛盾：residual/coverage diagnostics、negative mismatch control、额外 truth-free initialization。
- 下一轮推荐最小改动：只在 exp042 module/config/test/runner 内预注册并加入一个 measurement-derived
  gradient/Hessian scaling diagnostic及对应 normalized/preconditioned known-B step；先保持同一 case/B/scan/operator。
- 下一轮最小读取集合：`AGENTS.md`；本文 Scope/Reconstruction/Gauge/Operator consistency（第 3、6、9、10
  节）与本第 16 节，尤其本恢复块；当前 exp042 YAML/module/test/runner；最新 run 的 `metrics.json` 和必要 HDF5
  reconstruction paths。
- 只有 forward/operator/B/scan 语义改变、最新摘要与 artifact 冲突、出现 regression、artifact 不完整，或新的
  主要矛盾无法由本节定位时，才重新读取 exp040 对应章节。
- 明确不应重新扫描：exp040 全文/`_old.md`、全部历史 runs、整个 `src`/`tests`、notebooks、reports、data、
  exp041/exp05x 详细内容和无关 theory notes。

### Git 状态

本轮只产生本地 unstaged/untracked 修改；没有执行 `git add`、commit、push、PR、merge 或 branch 操作。
exp042 的 config/script/module/test 和本文仍为 untracked；runs 被 Git ignore。开始时的所有用户 modified、deleted、
untracked 内容保持原状态。staged 为空，commit/push/PR 均未改变。

## 17. 2026-08-20 18:12：Implementation iteration 02 — measurement-derived scaling

### 本轮目标与明确未做事项

本轮继承第 16 节唯一未关闭的主要矛盾：在保持同一 deterministic matched development case、known B、scan、
q4/open detector 和 loss 不变的条件下，加入不依赖 simulation truth 的 measurement-derived optimizer scaling，检查它能否
实质改善 known-B probe recovery 的数值收敛，并完整保存诊断与一次新的 timestamped run。

本轮没有恢复 B、没有拟合或输出 `D_waist`、没有改变 sample-B family 或 scan、没有加入 noise/stage error/subpixel
shift/calibration、没有改变 exp040 frozen evidence identity、没有运行 exp040 Helmholtz/reference pipeline，也没有形成 scientific
pass/fail 结论。simulation truth 仍只在 optimizer 完成后用于明确标记的 evaluation。

### 开始时检查的内容

- 执行 `git -c safe.directory=E:/tgv_ptycho_sim status -sb`；识别并保留开始时所有 modified、deleted 和 untracked
  用户内容，staged 为空。
- 重新读取 `AGENTS.md`；读取本文第 3、6、9、10 节固定边界和末尾第 16 节，尤其“改动后总体优先级”“下一轮快速恢复
  上下文”“Git 状态”。
- 在改代码前锁定本文：`51236` bytes，SHA256
  `5B6DE4C784E43B2300BE3BFC6E1FC717545DF1991367B4B25DA9745158967A31`，最后章节号 `16`。
- 定向检查当前 exp042 YAML、`MatchedKnownBProbeOperator`、`loss_and_gradient()`、
  `reconstruct_known_b_probe()`、operator-consistency helpers、runner HDF5/metrics/figure writer 以及对应 exp042 tests。
- 只读取第 16 节最新有效 run
  `runs/exp042_TGV_3d_multislice_probe_reconstruction_dev_20260820_172734` 的 `metrics.json`、
  `metadata.json` 和必要 reconstruction artifact 信息，作为 iteration 01 的 measurement-metric 对照；没有把旧 run 或 truth
  输入本轮 optimizer。
- 没有扩大读取范围：forward/operator/B/scan 语义未改变，第 16 节恢复块信息与实际 exp042 文件一致，因此没有重读
  exp040、没有扫描 runs 全目录、整个 `src`/`tests`、notebooks、reports、data 或其他实验文档。

### 上一轮意见

上一轮 authoritative implementation record 为第 16 节。其未关闭主要矛盾是 measurement-only optimizer
scaling/conditioning；最多三个次要矛盾为 residual/coverage diagnostics、negative mismatch control 和额外 truth-free
initialization；推荐最小改动是加入 measurement-derived gradient/Hessian scaling diagnostic 与 normalized/preconditioned
known-B step，同时保持 case/B/scan/operator 不变。

本轮复核后继续采用该意见。iteration 01 的 60 步全部接受固定步长 `1.0`、无 backtracking，但 loss、detector residual 和
simulation-only probe error 仅缓慢下降，说明先解决尺度比扩展 case、blind update 或 robustness 更必要。

### Change 01

#### 改动前：主要矛盾、次要矛盾与必要性判断

- 主要矛盾：固定 `initial_step=1.0` 没有反映完整 q4 pixel-intensity measurement Jacobian 的局部曲率，阻碍已验证
  正确的 matched inverse pipeline 在受控预算内产生有意义的恢复进展。
- 次要矛盾（最多三个）：缺少 proposed/accepted step、curvature、fallback 和 backtracking 的可审计诊断；剩余 detector
  residual 的结构/coverage 尚未刻画；额外非 truth 初始化的 basin/stability 尚未复核。
- 明确延后：negative mismatch 正式 control、blind `P_B+B`、noise/误差鲁棒性、full-size cases、复杂先验、scan/sample-B
  redesign、任何 waist inference 或物理准确性结论。
- 必要性：operator replay、adjoint、gradient、fixed point 已在第 16 节关闭；继续用固定小步长增加迭代只会混淆尺度问题与
  observability 问题。一个由当前 measurement gradient 和 exact Jacobian 导出的标量曲率可以独立验证，且不扩大科学范围。

#### 技术决策与理由

- 在 `MatchedKnownBProbeOperator` 中加入完整 real intensity Jacobian action。对每个 scan 先用现有
  `linear_detector_field()` 得到 `delta U`，再计算 `2 Re(conj(U) delta U)`，经过同一个 positive q4 midpoint pixel
  average 和 native ROI crop；没有把 pixel amplitude 复制到 q4 nodes。
- 对当前 gradient `g` 计算方向 Gauss--Newton 曲率
  `c = mean(|J g|^2)`，并使用 `alpha_GN = ||g||^2 / c` 作为 Armijo line search 的 proposed step。YAML 预注册
  `curvature_recompute_interval: 1`，即每次 iteration 重算。
- fallback 只在 gradient norm/curvature/proposed step 非有限或非正时使用 YAML 的 `initial_step=1.0`；没有使用 truth、
  probe error、`D(z)`、`D_waist` 或结果驱动 clipping/stopping。
- 保留 `batch_complex_gradient_descent_armijo` 旧算法供固定预算 regression comparison；本轮 authoritative config 显式使用
  `batch_complex_gradient_descent_gn_scaled_armijo`。
- 新增并保存 `proposed_step_curve`、`gauss_newton_directional_curvature_curve`、
  `step_scale_fallback_curve/count` 和 `total_backtracking_steps`。raw `P_B_rec` 与 simulation-only aligned field 仍分开。

#### 创建或修改的文件

- `configs/experiments/exp042_TGV_3d_multislice_probe_reconstruction.yaml`
- `src/tgv_ptycho/recon/exp042.py`
- `scripts/run_exp042_probe_reconstruction.py`
- `tests/test_exp042_probe_reconstruction.py`
- 本文只在原 `51236` bytes 之后追加本第 17 节。

没有修改 shared forward/optics/shift/IO API 或项目级 HDF5 schema。

#### 验证命令与结果

实际执行：

```powershell
D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim python -m pytest -q tests/test_exp042_probe_reconstruction.py
D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim python -m ruff check src/tgv_ptycho/recon/exp042.py scripts/run_exp042_probe_reconstruction.py tests/test_exp042_probe_reconstruction.py
```

- 第一版实现：targeted pytest `7 passed in 13.35s`；修改范围 Ruff `All checks passed!`。
- 将 Jacobian finite-difference 数值加入正式 operator metrics 时，第一次 targeted pytest 在 collection 阶段失败：
  `SyntaxError` at `src/tgv_ptycho/recon/exp042.py:872`。原因是新方向数组插入到了 boundary-control 的括号内部；局部恢复
  原括号并把方向生成移到循环之后，没有运行实验、没有产生失败 run。
- 修复后同一 targeted pytest：`7 passed in 13.81s`；最终修改范围 Ruff：`All checks passed!`。
- 没有运行 full pytest：本轮没有改变 shared forward/optics/shift/IO，风险由 exp042 targeted operator、optimizer、runner、
  HDF5 和 figure tests 覆盖；不为本轮扩大到全仓既有问题。
- 固定 8 步 measurement-only 对照（相同 operator/data/init/Armijo，truth 未输入两种 optimizer）：scaled loss
  `0.03127716305291964 -> 0.00023847114959227932`，unscaled final loss
  `0.031217311437932525`；scaled/unscaled final detector residual 分别为 `0.031151804015628247` 和
  `0.3564207894771509`。首个 proposed/accepted step 均为 `7028.983996600794`，fallback/backtracking 均为 `0`。
- 冻结 source YAML 后记录 SHA256
  `652E9D06D6D75EBBE38D89C7F4E2A1605E80854C976D1941B92E4117E625E524`，之后才启动唯一正式 run。

#### 改动后重新评估

主要矛盾判定为**已关闭**：Jacobian action 与 full loss gradient 均通过 finite difference；scale 只依赖当前 probe、known
B operator 和 measured-data gradient；固定预算对照及正式 60 步 run 都显示数量级改善，且没有 fallback 或 line-search
failure。

新证据没有支持继续调步长。剩余 probe error/residual 属于下一阶段的 initialization stability、有限 iteration budget 与
measurement observability/conditioning 区分问题，不应在本轮通过 truth-driven iteration selection 或更复杂 optimizer 继续扩张。

### Operator-consistency 检查

正式 run 实际记录：

- truth replay relative L2：`0.0`；truth loss、truth gradient norm、truth one-step change、zero-residual update norm 均为 `0.0`。
- full field linear adjoint dot error：`1.4510142628728665e-15`；constant-zero shift adjoint error：
  `5.682234924561534e-16`；q4 physical-weighted quadrature adjoint error：`0.0`。
- full-loss directional gradient relative error：`1.2347820022705336e-09`；新增 full q4 intensity Jacobian directional
  finite-difference relative error：`1.0858519027590559e-09`。
- detector constant/sum errors：`0.0/0.0`；node geometry normalized error：`6.776263578034403e-15`；minimum intensity
  `0.04091490300503436`，全非负。
- finite-B constant-zero boundary edge modulation max：`0.0`；deterministic repeat relative L2：`0.0`；所有数组 finite。
- interpolation/restriction mapping 仍明确为 development native grid 上的 identity；open padding、q4 restriction/crop、BC
  bandlimited adjoint 和 finite-B shift 均由同一 matched operator chain 覆盖。

### Development run 与 artifacts

唯一新 run：

`runs/exp042_TGV_3d_multislice_probe_reconstruction_dev_20260820_181010`

实际命令：

```powershell
D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim python scripts/run_exp042_probe_reconstruction.py --config configs/experiments/exp042_TGV_3d_multislice_probe_reconstruction.yaml
```

- runner exit `0`；`run_state.json` 为 `complete`、`artifacts_validated: true`，runtime `29.3301011 s`，figure count `3`。
- run 内规范化 `config.yaml` SHA256 为
  `C616A4CC1C3415572C6AC5B3373A1CE7C7C72F0A9E6239FE5F96AF0ED7606353`；它与冻结 source YAML 及 HDF5
  `/entry/config_yaml` 逐项语义相等。source/run 的字节哈希差异来自 `save_config()` 的规范化序列化，非参数差异。
- metadata 保持 `reference_validated=false`、`full_tgv_reference_authorized=false`、known-B/probe-only、
  `truth_used_by_optimizer=false` 和 frozen exp040 R8 scalar working-model claim boundary。
- HDF5 `/entry` 仍为 `config_yaml/data/instrument/metadata/metrics/reconstruction/sample/truth` 并列结构；没有空 group、
  没有 schema change。新增 reconstruction 字段为：
  `proposed_step_curve (60,)`、`gauss_newton_directional_curvature_curve (60,)`、
  `step_scale_fallback_curve (60,)`、`step_scale_fallback_count`、`total_backtracking_steps`；metrics 增加对应 summary 与
  `intensity_jacobian_directional_relative_error`。`P_B_rec (96,96)` raw complex field 保留，truth-aligned copy 只位于
  `simulation_evaluation_only`。
- HDF5 SHA256：`07927725079155CF6BD03816A991277B49C031D2388BF355423B70BC5AF2B912`；metrics SHA256：
  `3BA04E09BC74DBC6E2C58F4C598004D2FA760874A9EBB9C5FCEE964D7877AFFF`。
- 三张注册 figure 全部实际读取并目视检查：convergence 三条曲线平滑下降；probe 主结构从 homogeneous initialization 恢复，
  但仍有细粒度差异；中心 scan prediction 与 measured 视觉一致，difference 图仍有弱结构且无非有限/空白图像。

### 当前 metrics

Measurement-only：

- loss：`0.03127716305291964 -> 1.9277964661849684e-05`，60 步 nonincreasing；停止原因为
  `iteration_budget`。
- detector relative residual：`0.3567623010258413 -> 0.008857188126643553`。
- GN directional curvature：`1.065762487334625e-09 -> 1.421920868458713e-15`；proposed/accepted step 范围均为
  `[7028.983996600794, 28632.840215292184]`；fallback count `0`、total backtracking `0`。
- 相比 iteration 01 同为 60 步的 final loss `0.03083230643594776` 和 detector residual
  `0.35421609107301383`，本轮 measurement metrics 明显改善；没有据此设置或倒推 scientific threshold。

Simulation evaluation only：global-phase-aligned probe relative L2
`0.47873691495342324 -> 0.13937197214033825`；raw probe relative L2
`0.48028897391261666 -> 0.13937201182884235`。这些 truth-aided metrics 没有进入 optimizer、step、clipping 或 stopping。

状态保持 `Development baseline / No scientific pass-fail conclusion`，没有写 Passed/Failed。

### 失败、限制与未关闭问题

- 本轮唯一失败是新增正式 Jacobian metric 时的 pytest collection syntax error；已局部修复并由最终 targeted pytest/Ruff
  覆盖，失败过程如实保留在本节，没有失败 run 可保存。
- 60 步由 iteration budget 停止，loss/residual/probe evaluation 仍在下降；本轮没有用 truth 决定继续迭代或停止，因此不能把
  当前 final probe error 解释为 resolution/detection limit。
- detector residual 图仍有弱空间结构；尚未区分有限迭代、coverage、局部 conditioning 或 measurement null-space 的贡献。
- 只验证 homogeneous-reference initialization；尚未建立非 truth 初始化之间的 basin/stability evidence。
- development case 仍是 noiseless、integer-shift、identity native interpolation、matched known-B scalar working model；没有真实
  detector calibration、negative mismatch 或 robustness evidence。
- `reference_validated=false`、`full_tgv_reference_authorized=false`；本轮不说明 exp040 获得真实三维电磁准确性，也不说明真实
  TGV probe/waist可恢复。

### 改动后总体优先级

- 下一轮主要矛盾（一个）：**非 truth 初始化稳定性尚未建立**。最值得做的是保持同一 data/operator/known B 和当前 GN-scaled
  optimizer，预注册一个确定性、measurement-only 的非 truth complex perturbation initialization；比较 measurement loss/residual
  收敛以及两次 reconstruction 之间仅用于诊断的 global-phase-aligned pairwise difference。必要性是区分快速单次收敛与稳定 basin/
  identifiability，预期作用是决定 exp042 是否需要 observability 分析，而不是继续调 step。
- 次要工作一：加入 measurement-only 的 residual spatial/radial summary 和 gradient norm reduction，用于判断剩余结构是否随迭代
  一致衰减；在主要矛盾未关闭前不据此增加复杂模型。
- 次要工作二：预先定义一个小型 negative-mismatch diagnostic（例如只改变 reconstruction-side detector quadrature 的明确非 matched
  control），但只有 initialization stability 完成后再运行，且不能与 primary matched branch 混称通过。
- 次要工作三：基于 measurement curve 预注册 stopping/continuation 诊断，区分 iteration-budget limited 与 plateau；不得使用 truth
  error 选择 iteration count。
- 明确延后：blind B update、更多 sample-B/scan、noise/stage/subpixel/calibration、full-size、多 seed sweep、复杂正则、waist inference、
  Helmholtz/reference validation和任何真实物理精度声明。
- 建议继续 exp042；当前尚未形成需要新实验编号的新研究问题。如果不同非 truth initialization 收敛到 measurement-equivalent 但
  probe-inconsistent 解，再把 identifiability/null-space 作为新的主要研究问题记录并停止盲目优化。

### 下一轮快速恢复上下文

- 当前 authoritative appended section：本文第 17 节。
- 本轮完成的 Change：Change 01 full q4 intensity Jacobian、measurement-derived GN directional scale、Armijo integration、诊断持久化、
  targeted validation 和一次正式 run。
- 本轮修改文件：
  `configs/experiments/exp042_TGV_3d_multislice_probe_reconstruction.yaml`、
  `src/tgv_ptycho/recon/exp042.py`、`scripts/run_exp042_probe_reconstruction.py`、
  `tests/test_exp042_probe_reconstruction.py`，以及本文末尾第 17 节。
- 当前有效 config：上述 exp042 YAML，source SHA256
  `652E9D06D6D75EBBE38D89C7F4E2A1605E80854C976D1941B92E4117E625E524`；algorithm
  `batch_complex_gradient_descent_gn_scaled_armijo`，60 iterations，curvature 每步重算。
- 最新有效 run：
  `runs/exp042_TGV_3d_multislice_probe_reconstruction_dev_20260820_181010`，complete/validated；run-config SHA256
  `C616A4CC1C3415572C6AC5B3373A1CE7C7C72F0A9E6239FE5F96AF0ED7606353`。
- 已通过的 tests/commands：最终 targeted pytest `7 passed in 13.81s`；修改范围 Ruff `All checks passed!`；8 步 scaled-vs-
  unscaled measurement control；唯一正式 runner exit `0`；config/metadata/metrics/HDF5 tree/三张 PNG read-back 与目视检查通过。
- 已确认不需要重跑的 controls：若 operator/config 未变，本节 replay、field/shift/q4 adjoint、full loss gradient、intensity Jacobian、
  zero residual、truth fixed point、finite-B boundary、determinism无需机械重跑；exp040 R4/R5/R8/R9/R10A/R13/R14A controls 和
  R10B--R14B reference failures无需重算。
- 当前未关闭的主要矛盾：非 truth initialization stability/basin 尚未验证。
- 当前次要矛盾：remaining residual/gradient diagnostics、negative mismatch control、measurement-only stopping/plateau 判断。
- 下一轮推荐的最小改动：只在 exp042 config/module/test/runner 内增加一个预注册确定性 non-truth perturbed initialization；保持
  truth/data/B/scan/operator/optimizer scale 完全不变，保存独立 raw reconstruction 和 pairwise diagnostic，最多运行一个新的受控 run。
- 下一轮最小读取集合：`AGENTS.md`；本文第 3、6、9、10 节与本第 17 节，尤其本恢复块；当前 exp042 YAML/module/test/runner；
  最新 run 的 `config.yaml`、`metrics.json` 和必要 HDF5 reconstruction curves。无需重读第 16 节全文。
- 只有 forward/operator/B/scan 语义改变、恢复摘要与实际 artifact 冲突、出现 regression、latest artifact 不完整，或主要矛盾无法由
  本节定位时，才重新读取 exp040 对应章节并在新 appended section 说明原因和结论。
- 明确不应重新扫描：exp040 全文/`_old.md`、全部历史 runs、整个 `src`/`tests`、notebooks、reports、data、exp041/exp05x
  详细内容和无关 theory notes。

### Git 状态

本轮只产生本地 unstaged/untracked 修改；没有执行 `git add`、commit、push、PR、merge 或 branch 操作。exp042 的
config/script/module/test/本文仍为 untracked，runs 被 Git ignore；staged 为空。开始时已有的 exp030/exp040 文档修改、deleted
notebook 及其他 untracked 用户文件均保持原状态、未覆盖、未删除、未重解释。commit/push/PR 状态均未改变。

## 18. 2026-08-22 13:17：Implementation iteration 03 — truth-free initialization stability

### 本轮目标与明确未做事项

本轮继承第 17 节的唯一主要矛盾：在完全保持 matched truth/data、known B、scan、q4/open detector、loss、GN-scaled
optimizer 和 60-step budget 不变时，加入一个预注册、确定性且不读取 simulation truth 的局部初始化扰动，检查 homogeneous
primary reconstruction 是否表现出 initialization stability，并保存两条 raw reconstruction、truth-free pairwise diagnostic 和一次
新的 timestamped development run。

本轮没有增加 optimizer iteration、没有用 truth 选择 perturbation/seed/stopping、没有恢复 B、没有做更多 seeds sweep、没有改变
sample-B/scan/detector/forward、没有加入 noise/mismatch/stage/subpixel/calibration、没有拟合或输出 `D_waist`，也没有运行 exp040
Helmholtz/reference pipeline。结果仍不形成 scientific pass/fail 或真实物理准确性结论。

### 开始时检查的内容

- 执行 `git -c safe.directory=E:/tgv_ptycho_sim status -sb`，识别并保留所有已有 modified、deleted 和 untracked 用户内容；
  staged 为空。
- 完整读取 `AGENTS.md`；按恢复协议读取本文第 3、6、9、10 节固定边界和最新 authoritative 第 17 节，尤其“改动后总体
  优先级”“下一轮快速恢复上下文”“Git 状态”。
- 在代码修改前锁定本文：`69543` bytes，SHA256
  `6D9DE76C4A61618FE0078E785736272CF09590D61BC06B58C00140AF1D94D07A`，最后章节号 `17`。
- 定向检查当前 exp042 YAML、config validation、`reconstruct_known_b_probe()`、global-phase evaluation、runner metrics/HDF5/figure
  writer 和完整 exp042 targeted test file。
- 读取第 17 节最新有效 run
  `runs/exp042_TGV_3d_multislice_probe_reconstruction_dev_20260820_181010` 的 `config.yaml` 和 `metrics.json`，确认 primary
  config、60-step results、measurement-only scale 与恢复块一致。旧 run 仅作为 provenance/对照，没有输入本轮 optimizer。
- 没有扩大读取范围：没有重读 exp040、没有读取 `_old.md`、没有扫描全部 runs、整个 `src`/`tests`、notebooks、reports、data、
  exp041/exp05x 或无关 theory notes。第 17 节和实际 artifact 足以定位本轮问题。

### 上一轮意见

上一轮 authoritative record 为第 17 节。其未关闭主要矛盾是非 truth initialization stability/basin；推荐最小改动是同一
data/operator/known B/GN optimizer 下增加一个预注册 deterministic non-truth complex perturbation，保存独立 raw reconstruction
和仅消除 global phase 的 pairwise diagnostic，最多运行一个新受控 run。

本轮复核后继续采用该意见。上一轮 operator/gradient/scaling 已关闭且 60-step primary loss/residual 仍下降，所以初始化依赖比继续调
step、引入 negative mismatch 或扩展 blind branch 更必要。

### Change 01

#### 改动前：主要矛盾、次要矛盾与必要性判断

- 主要矛盾：只有 homogeneous-reference initialization 的单次快速收敛证据，无法判断 matched known-B inverse 在局部 basin 内是否
  对非 truth 初始化扰动稳定。
- 次要矛盾（最多三个）：remaining residual/gradient 的结构尚未解释；iteration-budget transient 与 weak observability 尚未区分；
  negative mismatch control 尚未建立。
- 明确延后：更多 seeds/amplitudes sweep、blind `P_B+B`、noise/真实误差、full-size、sample-B/scan redesign、复杂正则、waist
  inference、reference validation 和真实精度声明。
- 必要性：如果不同初始化产生 measurement-equivalent 但 probe-inconsistent 解，继续降低单一 loss 或盲目增加 iteration 不能回答
  identifiability；必须先保存可审计的 paired evidence。

#### 技术决策与理由

- YAML 在看到本轮正式结果前预注册 `initialization_stability_control`：role
  `truth_free_local_basin_diagnostic`、seed `20260843`、相对 primary L2 `0.05`、零均值 complex perturbation、复用全部 primary
  optimizer settings、pairwise alignment 只允许 global phase、`truth_used_by_initialization=false`。
- 新公共函数从 primary homogeneous field 和 seed 直接构造 perturbation：先生成 complex Gaussian field、减去 complex mean、再按
  primary field L2 norm 精确归一到 5%。接口不接受 truth、`D(z)` 或 evaluation metric。
- runner 在同一 run 中依次执行 primary/control 两条 raw reconstruction；两者使用相同 measured `I_stack`、known B、operator、loss、
  60-step GN-scale/Armijo settings。control 没有使用 primary final field warm start。
- pairwise diagnostic 只在两条 optimizer 都完成后计算：保存 control-init/control-final 对 primary 的 global-phase-aligned copy、raw/
  aligned probe difference、final prediction difference、loss/residual absolute difference。aligned copy 不覆盖任何 raw result，也不回写
  optimizer。
- primary/control 各自的 truth alignment 仍单独位于 `simulation_evaluation_only`；truth-free pairwise diagnostic 与 simulation-only
  evaluation 分组保存，避免证据身份混淆。
- convergence figure 叠加两条 loss/residual/simulation-only probe-error curve；增加第 4 张不使用 truth 的 initialization stability
  amplitude/phase comparison figure。

#### 创建或修改的文件

- `configs/experiments/exp042_TGV_3d_multislice_probe_reconstruction.yaml`
- `src/tgv_ptycho/recon/exp042.py`
- `scripts/run_exp042_probe_reconstruction.py`
- `tests/test_exp042_probe_reconstruction.py`
- 本文只在原 `69543` bytes 之后追加本第 18 节。

没有修改 shared forward/optics/shift/IO API，也没有改变项目级 `/entry` schema。

#### 验证命令与结果

实际执行：

```powershell
D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim python -m pytest -q tests/test_exp042_probe_reconstruction.py
D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim python -m ruff check src/tgv_ptycho/recon/exp042.py scripts/run_exp042_probe_reconstruction.py tests/test_exp042_probe_reconstruction.py
```

- 第一次 targeted pytest：`1 failed, 6 passed in 26.91s`。失败断言预设 8 步后 pairwise phase-aligned probe difference 必须从
  `0.05` 下降；实际为 `0.05835972970706823`。两条 measurement loss 均正常下降，失败不是 runtime/operator error，而是测试错误地
  把待检验的稳定性结论写成门槛。
- 修正方法：删除 outcome-directed decrease assertion，只验证 initialization 精确/零均值/确定性、两条 measurement loss 单调、
  diagnostic finite、truth-free 和 artifact 分组；没有改变 config、seed、5% amplitude、optimizer 或正式 run budget。
- 修正后 targeted pytest：`7 passed in 22.42s`。
- 修改范围 Ruff：`All checks passed!`。
- 没有运行 full pytest：没有改变共享 forward/optics/shift/IO，exp042 targeted suite 已覆盖 config、initializer、两条 optimizer、
  pairwise diagnostic、runner、HDF5 和四张 figures；不扩大到全仓既有问题。
- 固定 8-step truth-free 对照：primary/control final loss
  `0.00023847114959227932 / 0.00023943887164664223`；final detector residual
  `0.031151804015628247 / 0.031214947431772146`；pairwise aligned probe difference
  `0.05 -> 0.05835972970706823`；final prediction relative L2 `0.0014006021714425543`；两条 fallback 均为 `0`。
- 冻结 source YAML 后记录 SHA256
  `BAC75A2845C221A25600214AB638BF9E28F74F422DB0E04E1D340FE68F29B1F2`，之后才启动唯一正式 run。

#### 改动后重新评估

实现与可审计 pipeline 的问题判定为**已关闭**：initializer 精确确定、truth-free、重复生成 bitwise identical；两条 reconstruction
正确复用同一 operator/settings，raw/aligned/truth-only 数据身份清楚，tests/Ruff/artifact validation 均通过。

上一轮“初始化稳定性尚未建立”的科学/数值主要矛盾判定为**部分关闭，并暴露新的主要矛盾**：一个 5% 局部 perturbation control
已经执行，但在 8 步和 60 步均未向 primary probe 收敛；与此同时两条 measurement prediction 极接近。由于两条 loss/gradient 在
60 步仍下降，当前证据不能区分 finite-budget transient 与局部 weakly observed/null-like direction，不能宣称 general instability 或
non-identifiability。

### Operator-consistency 检查

本轮未改变 measurement operator，但唯一正式 runner 按当前 config 重放并记录全部 controls：

- truth replay、truth loss、truth gradient norm、truth one-step change、zero-residual update、deterministic repeat：均为 `0.0`。
- field adjoint：`1.4510142628728665e-15`；constant-zero shift adjoint：`5.682234924561534e-16`；q4 physical-weighted
  adjoint：`0.0`。
- full-loss gradient finite-difference error：`1.2347820022705336e-09`；full q4 intensity Jacobian error：
  `1.0858519027590559e-09`。
- detector constant/sum errors：`0.0/0.0`；node geometry normalized error：`6.776263578034403e-15`；minimum intensity
  `0.04091490300503436`，全非负。
- finite-B edge modulation：`0.0`；所有数组 finite；native interpolation mapping 仍明确为 identity。

新增 initialization control 不修改 operator；其 determinism 另行记录为 exact repeat `true`、max abs error `0.0`。

### Development run 与 artifacts

唯一新 run：

`runs/exp042_TGV_3d_multislice_probe_reconstruction_dev_20260822_131449`

实际命令：

```powershell
D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim python scripts/run_exp042_probe_reconstruction.py --config configs/experiments/exp042_TGV_3d_multislice_probe_reconstruction.yaml
```

- runner exit `0`；`run_state.json` 为 `complete`、`artifacts_validated=true`，runtime `76.4710366 s`，figure count `4`。
- run 规范化 config SHA256：`07F8C9AE5D60D4B1C3B8160874C8FDB093E7FF26903ED5706EA8AF005262AFEC`；source
  YAML、run `config.yaml` 与 HDF5 `/entry/config_yaml` 逐项语义相等。source/run 字节哈希差异仅来自 `save_config()` 序列化。
- metadata 保持 known-B/probe-only、`reference_validated=false`、`full_tgv_reference_authorized=false`、
  `truth_used_by_optimizer=false`，并明确 `initialization_stability_control=true`。
- `/entry` 顶层仍为 `config_yaml/data/instrument/metadata/metrics/reconstruction/sample/truth`。新增自然 subgroup：
  `/entry/reconstruction/initialization_stability_control`，其中保存 control raw `P_B_init/P_B_rec`、完整 loss/residual/gradient/step/
  curvature curves、独立 `simulation_evaluation_only`，以及 `truth_free_pairwise_diagnostic` 下两张 pairwise aligned copies 和实际 scalar
  metrics。`/entry/metrics/initialization_stability_control` 保存 initialization provenance、control measurement summary、truth-free pairwise
  summary 和独立 simulation-only summary。没有空 group、没有项目级 schema change。
- HDF5 SHA256：`D78E0742BBEF6B1D27CBA5924093746B080A6E829BC8E97507076111A471BE6C`；metrics SHA256：
  `EB519BF8D66CCCDD23188F015F52D6F2DD2AFDDB9E552D81A7AD00C420C2DDD4`。
- 四张 figure 全部实际读取并目视检查：两条 measurement convergence curve 几乎重合；control simulation-only probe-error curve
  系统性高于 primary；primary detector residual 图与第 17 节一致；新增 stability 图显示 5% fine-scale init perturbation 在 control final
  amplitude/phase 中仍留下可见细粒度差异，但主要 probe structure 与 primary 相同。所有 PNG 可读、finite、非空。

### 当前 metrics

Primary measurement-only：

- loss：`0.03127716305291964 -> 1.9277964661849684e-05`；detector residual：
  `0.3567623010258413 -> 0.008857188126643553`。
- gradient L2：`0.002737010680953316 -> 9.3765771650562e-06`；60 步 loss nonincreasing，stopping
  `iteration_budget`，fallback/backtracking `0/0`。

Control initialization 与 measurement-only：

- requested/actual relative L2：`0.05 / 0.05000000000000001`；perturbation mean abs
  `7.086076117205387e-19`；deterministic repeat exact `true`；truth used by initialization `false`。
- loss：`0.03133195189085838 -> 1.93299503664604e-05`；detector residual：
  `0.357074638166358 -> 0.008869122404370298`。
- gradient L2：`0.0027404734802917866 -> 9.385454989866588e-06`；60 步 loss nonincreasing，stopping
  `iteration_budget`，fallback/backtracking `0/0`。

Truth-free pairwise diagnostic：

- probe global-phase-aligned relative L2：`0.05000000000000001 -> 0.056199000098323494`。
- final prediction relative L2：`0.0007003237596188443`。
- final loss absolute difference：`5.198570461071786e-08`；final detector-residual absolute difference：
  `1.1934277726744283e-05`。

Simulation evaluation only：primary/control final global-phase-aligned truth error 分别为
`0.13937197214033825 / 0.1492569162151173`。这些值只在两条 optimizer 完成后产生，没有参与 initialization、step、stopping 或
result selection。

上述结果没有预注册 pass/fail threshold，不写 Passed/Failed。状态保持
`Development baseline / No scientific pass-fail conclusion`。

### 失败、限制与未关闭问题

- 第一次 targeted test 错误预设 pairwise difference 必须下降；该方法学错误已修正并记录，不能用测试门槛强迫产生“稳定”结论。
- 只执行一个 seed、一个 5% local perturbation；不能推广为全局 basin、所有 initialization 或概率稳定性结论。
- 两条 reconstruction 在 measurement space 极接近、probe space 仍相差约 5.62%，但两条 gradient/loss 仍下降；当前不能区分有限
  iteration transient、ill-conditioning、weak observability 或真正 null space。
- pairwise global phase 是第 9 节允许的 gauge；没有对齐 spatial shift、phase ramp 或 amplitude scale。
- 仍为 noiseless、integer-shift、identity-native interpolation、matched known-B scalar working model；没有 mismatch/robustness/真实
  calibration evidence。
- `reference_validated=false`、`full_tgv_reference_authorized=false`；不说明真实 TGV probe 或 waist 可稳定恢复。

### 改动后总体优先级

- 下一轮主要矛盾（一个）：**pairwise probe difference 是 finite-budget transient，还是 measurement Jacobian 的 weakly observed
  direction，尚未区分**。最值得做的是保持本 run 和 operator 不变，对 final pairwise difference direction 计算 truth-free normalized
  `||J delta P||/||delta P||`、directional GN curvature，并与预注册的 gradient direction 和少量同 seed random directions 比较；同时报告
  已有 nonlinear prediction difference。必要性是直接解释“prediction 近似一致而 probe 不一致”，预期作用是决定是否进入正式
  identifiability/null-space 研究，而不是继续增加 optimizer iterations。
- 次要工作一：补充 measurement-only stopping/continuation diagnostic，判断 60-step gradient reduction 是否已接近 plateau；必须在
  Jacobian-direction 诊断后再决定是否需要一次 continuation run。
- 次要工作二：对 primary/control final detector residual 增加 spatial/radial summary，检查两条解是否留下相同结构。
- 次要工作三：negative mismatch control 继续保持预注册候选，但在局部 conditioning 未解释前不运行。
- 明确延后：更多 initialization seeds/amplitudes sweep、blind B、noise/stage/subpixel/calibration、sample-B/scan redesign、full-size、
  regularization、waist inference、reference validation和真实物理声明。
- 建议继续 exp042 的一个最小 observability/conditioning Change；当前不应继续盲目优化。如果 pairwise direction 被确认显著弱观测，
  应把 identifiability/scan-information 记录为新的研究问题，并在改变 scan/sample-B 前转交 exp041 或新任务，而不是在 exp042 静默改模型。

### 下一轮快速恢复上下文

- 当前 authoritative appended section：本文第 18 节。
- 本轮完成的 Change：Change 01 deterministic 5% zero-mean non-truth initialization、双 matched reconstruction、truth-free pairwise
  diagnostic、双 simulation-only evaluation、HDF5/metrics/四图持久化、targeted validation 和一次正式 run。
- 本轮修改文件：
  `configs/experiments/exp042_TGV_3d_multislice_probe_reconstruction.yaml`、
  `src/tgv_ptycho/recon/exp042.py`、`scripts/run_exp042_probe_reconstruction.py`、
  `tests/test_exp042_probe_reconstruction.py`，以及本文末尾第 18 节。
- 当前有效 config：上述 exp042 YAML，source SHA256
  `BAC75A2845C221A25600214AB638BF9E28F74F422DB0E04E1D340FE68F29B1F2`；primary/control 均为 60-step
  GN-scaled Armijo，control seed `20260843`、relative L2 `0.05`。
- 最新有效 run：
  `runs/exp042_TGV_3d_multislice_probe_reconstruction_dev_20260822_131449`，complete/validated；run-config SHA256
  `07F8C9AE5D60D4B1C3B8160874C8FDB093E7FF26903ED5706EA8AF005262AFEC`。
- 已通过的 tests/commands：最终 targeted pytest `7 passed in 22.42s`；修改范围 Ruff `All checks passed!`；固定 8-step
  paired control；唯一正式 runner exit `0`；config/metadata/metrics/HDF5 tree/四张 PNG read-back 与目视检查通过。第一次 outcome-directed
  test failure 已在本节记录，不应恢复该断言。
- 已确认不需要重跑的 controls：只要 operator/config 未改变，本节和第 17 节 replay、field/shift/q4 adjoint、gradient/Jacobian finite
  difference、zero residual、truth fixed point、finite-B boundary、determinism无需机械重跑；exp040 reference controls无需重算。
- 当前未关闭的主要矛盾：final pairwise difference direction 的 finite-budget transient 与 weak observability/conditioning 尚未区分。
- 当前次要矛盾：measurement-only plateau/stopping、paired residual structure、negative mismatch control。
- 下一轮推荐最小改动：只在 exp042 module/test/runner 中加入 pairwise-difference directional Jacobian sensitivity 与预注册 random/gradient
  controls；优先复用本 run 的 raw primary/control fields 或同一 deterministic fixture，不改变 optimizer、data、B、scan 或 operator，不先运行
  continuation。
- 下一轮最小读取集合：`AGENTS.md`；本文第 3、6、9、10 节与本第 18 节，尤其本恢复块；当前 exp042 module/test/runner；当前 YAML；
  最新 run 的 `config.yaml`、`metrics.json` 及 HDF5 中 primary/control raw `P_B_rec` 和必要 operator provenance。无需重读第 17 节全文。
- 只有 forward/operator/B/scan 语义改变、摘要与 artifact 冲突、出现 regression、latest artifact 不完整，或 conditioning 问题无法由本节
  定位时，才重新读取 exp040 对应章节并在新 appended section 说明原因与结论。
- 明确不应重新扫描：exp040 全文/`_old.md`、全部历史 runs、整个 `src`/`tests`、notebooks、reports、data、exp041/exp05x
  详细内容和无关 theory notes。

### Git 状态

本轮只产生本地 unstaged/untracked 修改；没有执行 `git add`、commit、push、PR、merge 或 branch 操作。exp042 的
config/script/module/test/本文仍为 untracked，runs 被 Git ignore；staged 为空。开始时已有的 exp030/exp040 文档修改、deleted
notebook 和其他 untracked 用户文件保持原状态，未覆盖、未删除、未重解释。commit/push/PR 状态均未改变。

## 19. 2026-08-22 13:43：Implementation iteration 04 — truth-free pairwise Jacobian conditioning

### 本轮目标与明确未做事项

本轮继承第 18 节的唯一主要矛盾：判断 60-step primary/control 最终 probe 差异更符合 finite-budget transient，还是更符合
measurement Jacobian 的弱观测方向。保持上一轮 data、known B、scan、matched operator、optimizer、primary/control initialization 和
60-step budget 全部不变，只增加 truth-free posthoc local Jacobian conditioning diagnostic、持久化、测试和一轮正式 development run。

本轮没有执行 continuation、blind `P_B+B`、waist/D(z) inference、更多 initialization seeds/amplitudes、noise/stage/subpixel/
calibration、sample-B/scan redesign、negative mismatch、full-size pipeline 或 exp040 Helmholtz/reference validation；没有用 truth 选择
direction、random controls、optimizer、步长或停止时机，也没有登记科学 pass/fail threshold。

### 开始时检查的内容

- 执行 `git -c safe.directory=E:/tgv_ptycho_sim status -sb`，确认并保留开始时已有的 exp030/exp040 文档修改、deleted notebook、
  其他 untracked 用户文件，以及仍为 untracked 的 exp042 文件。
- 读取 `AGENTS.md`；按恢复协议读取本文第 3、6、9、10 节固定边界，以及 authoritative 第 18 节的“失败、限制与未关闭问题”、
  “改动后总体优先级”“下一轮快速恢复上下文”和 Git 状态。
- 代码改动前锁定本文为 `89037` bytes，SHA256
  `BB433B66C801C2D51F6E5949C0349A69CE18E11EAC3AD696345C606E8173D50D`，最后章节号 `18`。
- 定向检查当前 exp042 YAML、config validation、`MatchedKnownBProbeOperator.detector_field()`、
  `linear_detector_field()`、`intensity_jacobian_direction()`、loss/gradient、initialization-stability diagnostic、runner metrics/HDF5/
  figure writer 和 exp042 targeted tests。
- 复用上一轮 run
  `runs/exp042_TGV_3d_multislice_probe_reconstruction_dev_20260822_131449` 的同一 HDF5 measured `I_stack` 与 primary/control raw
  `P_B_rec` 做一次不写 artifact 的预运行诊断；没有把 truth 输入 diagnostic。
- 没有重读 exp040、`_old.md`、全部 runs、整个 `src`/`tests`、notebooks、reports、data、exp041/exp05x 或无关 theory notes。
  为确认 latest-run 路径和 HDF5 字段进行的定向 `rg` 同时显示了少量第 17 节匹配行，但没有顺序读取第 17 节，也没有改变本轮优先级；
  找到第 18 节 provenance 后即停止扩展。

### 上一轮意见

第 18 节认为主要矛盾是 final pairwise difference direction 的 finite-budget transient 与 weak observability/conditioning 尚未区分；
建议先实现 pairwise directional Jacobian sensitivity，并以预注册 random/gradient controls 比较，不先做 continuation。次要矛盾是
measurement-only plateau/stopping、paired residual structure 和 negative mismatch control。本轮复核后继续采用该意见：局部 Jacobian
诊断直接回答当前科学解释问题，且比延长优化、更换 scan/B 或增加正则化更小、更可审计。

### Change 01

#### 改动前：主要矛盾、次要矛盾与必要性判断

- 继承的主要矛盾仍成立：两条 reconstruction 的 detector prediction 已很接近，但 final aligned probe relative L2 仍约 `5.62%`；
  仅看 loss/probe curve 不能区分有限迭代与局部弱观测。
- 本 Change 的唯一主要矛盾：在 primary final probe 处，定量判断 aligned control-minus-primary direction 的 intensity-Jacobian gain
  是否低于正常局部方向。
- 次要矛盾最多三个：measurement-only plateau/stopping；paired residual 的空间结构；negative mismatch control。
- 明确延后 continuation、更多 seeds、谱级 identifiability、scan/sample-B 改动、regularization、blind reconstruction 和所有物理腰径结论。
- 当前改动只复用已经通过 finite-difference test 的 exact intensity Jacobian，不改变 forward 或 optimizer；它是回答上一轮主要矛盾所需的
  最小证据。

#### 技术决策与理由

- 预先固定 base probe 为 primary final raw reconstruction；pairwise direction 为 control 先按第 18 节规则 global-phase aligned 后减
  primary，并按 probe L2 归一化为单位方向。
- 定义 `gain(d)=sqrt(mean((Jd_unit)^2))`、`curvature(d)=mean((Jd_unit)^2)`，mean 覆盖完整 25-frame、32x32
  detector stack。比较方向固定为 primary final loss-gradient direction、native-ROI phase-rotation direction `i P_native`，以及 seed
  `20260844` 的 8 个 zero-mean complex-Gaussian unit-L2 directions；无科学 threshold，truth 禁止进入 diagnostic。
- 第一次 targeted test 原先把 `i P_native` 错称为 exact global-phase gauge 并要求 gain `<1e-12`，实际得到
  `0.004009995355199863`。定向复核 operator 后确认：当前 open/reference-plus-residual 参数化仅更新 native ROI，ROI 外的
  homogeneous complex exterior 固定；因此 `i P_native` 不会旋转整个 open field，它相对固定 exterior reference 是可观测的，不能作为
  exact gauge zero control。没有放宽阈值，而是把 config/API/metric/figure 明确更名为
  `native_global_phase_rotation` comparator，并测试其与 direct Jacobian evaluation 一致且非零。
- 这是对本文第 9.1 节通用 global-phase ambiguity 表述的 append-only correction note：对当前固定 complex exterior 的具体 inverse
  参数化，该 ambiguity 被 reference phase 锚定；旧内容不回改。raw reconstruction 始终保留。本轮 pairwise direction 与 native-phase
  direction 的 real inner-product magnitude 为 `3.686287386450715e-18`，因此这项语义修正不解释或制造 pairwise low gain。

#### 创建或修改的文件

- 修改 `configs/experiments/exp042_TGV_3d_multislice_probe_reconstruction.yaml`：增加
  `verification.pairwise_conditioning`，固定 base/direction/norm、seed `20260844`、8 个 random controls 和第五张 figure；source
  SHA256 为 `A3418E10495DCB19FED2A1A19867C906B2DD26735779E646709E6665DA88FE18`。
- 修改 `src/tgv_ptycho/recon/exp042.py`：增加 conditioning config validation、
  `normalized_intensity_jacobian_sensitivity()` 与 `truth_free_pairwise_conditioning_diagnostic()`；没有修改 shared
  forward/optics/shift/IO，也没有改变已有 reconstruction 数值路径。
- 修改 `scripts/run_exp042_probe_reconstruction.py`：计算和保存 conditioning metrics/field，新增 log-scale gain comparison figure，
  metadata 与 artifact validation。
- 修改 `tests/test_exp042_probe_reconstruction.py`：覆盖 config、unit direction、gain-curvature identity、native-phase comparator、
  random determinism、metrics/HDF5 subgroup 和第五张 figure。
- 本节是本文唯一修改；没有修改 exp040/exp041/exp050、README、roadmap、notebooks 或 reports。

#### 验证命令与结果

```powershell
D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim python -m pytest -q tests/test_exp042_probe_reconstruction.py
D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim python -m ruff check src/tgv_ptycho/recon/exp042.py scripts/run_exp042_probe_reconstruction.py tests/test_exp042_probe_reconstruction.py
D:\anaconda3\envs\tgv_ptycho_sim\python.exe -c <只读 latest-run conditioning diagnostic>
D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim python scripts/run_exp042_probe_reconstruction.py --config configs/experiments/exp042_TGV_3d_multislice_probe_reconstruction.yaml
```

- 第一次 targeted pytest：`1 failed, 6 passed in 31.58s`；失败就是上述错误 exact-gauge 假设，实际
  native-phase gain `0.004009995355199863`。修正语义和测试后：`7 passed in 32.47s`。
- 修改范围 Ruff：`All checks passed!`。
- 第一次只读 diagnostic 启动命令用 multiline `conda run ... python -c`，被 conda 在进入 Python 前以“不支持含换行参数”拒绝并
  exit `1`；没有写 artifact。改用已核对的 env Python 直接执行同一命令，exit `0`。
- 没有运行 full pytest：本轮没有修改 shared forward/optics/shift/IO，exp042 targeted suite 已覆盖新增 config、Jacobian
  diagnostic、runner、metrics、HDF5 和五图；不扩大到全仓既有问题。

#### 改动后重新评估

- 本轮主要矛盾为“部分关闭”：pairwise gain `1.0561312508814082e-4`，约为 random median
  `5.510060015880909e-4` 的 `0.1916732753976295`，且 8/8 random directions 的 gain 均更高；这直接支持
  pairwise difference 落在相对弱观测的 local Jacobian direction。由于本轮有意不做 continuation，不能排除其中仍有
  finite-budget transient，故不写成完全关闭或全局 identifiability 结论。
- 新暴露的 native/open global-phase 语义差异不改变本轮 low-gain 结果，但会影响 future gauge interpretation，列为下一轮次要矛盾；
  不是 Jacobian implementation defect。
- 没有证据要求修改 scan/B/operator、引入正则化或扩大为 blind problem；这些继续延后。更系统的 singular-spectrum/identifiability
  已形成候选研究问题，不在本 Change 内扩张。

### Operator-consistency 检查

- truth replay：relative L2 `0.0`；truth loss/gradient norm/one-step relative change 均为 `0.0`。
- linear composite adjoint dot test：relative error `1.4510142628728665e-15`；constant-zero shift adjoint
  `5.682234924561534e-16`；physical-weighted q4 adjoint `0.0`。
- full loss directional gradient relative error：`1.2347820022705336e-9`；intensity Jacobian directional finite-difference
  relative error：`1.0858519027590559e-9`。
- zero-residual update L2 norm：`0.0`；truth-initialized fixed point 为 exact zero-change。
- detector quadrature：constant max error `0.0`、sum relative error `0.0`、minimum intensity
  `0.04091490300503436`、all nonnegative；node-geometry normalized error `6.776263578034403e-15`，q4 weights 为
  16 个 `1/16` positive midpoint weights。
- finite-B boundary：shifted `(B-1)` open-grid edge maximum modulation `0.0`，constant-zero/transparent exterior 保持。
- determinism：truth-data repeat relative L2 `0.0`；test 中同 seed random gain array exact equal；正式 run 与只读旧 artifact diagnostic
  的 conditioning scalars 相同。全部 HDF5 numeric datasets finite。

### Development run 与 artifacts

唯一新 run：

`runs/exp042_TGV_3d_multislice_probe_reconstruction_dev_20260822_134015`

- runner exit `0`；`run_state.json` 为 `complete`、`artifacts_validated=true`，runtime `82.3249638999996 s`，figure
  count `5`。
- run config SHA256 `B93E59ED52670DC1E9D829FF7D70BA9D49B92E69001B54643E2BDC298CFB6581`；source YAML、run
  `config.yaml` 与 HDF5 `/entry/config_yaml` 逐项语义相等。source/run 字节哈希差异只来自 `save_config()` 序列化。
- metadata SHA256 `50B80209B23FB5160D63A197E4117A7C6FA53F990A08F9864143F1F5664D47BD`，继续明确 known-B/
  probe-only、`reference_validated=false`、`full_tgv_reference_authorized=false`、`truth_used_by_optimizer=false`，新增
  `pairwise_conditioning_diagnostic=true`。
- metrics SHA256 `7FF82E9F8C96CB6BABA94908989EBA35512FF46984110876E470BA4341E5F1C1`；HDF5 SHA256
  `AF99052EDB924CE5A1CA218BFF234877321AD6FB12C13A521EDAF2C6A53F61E9`。
- HDF5 `/entry` 顶层仍为 `config_yaml/data/instrument/metadata/metrics/reconstruction/sample/truth`。新增自然字段集中在
  `/entry/reconstruction/initialization_stability_control/truth_free_pairwise_conditioning` 和对应
  `/entry/metrics/initialization_stability_control/truth_free_pairwise_conditioning`；前者额外保存 raw unit-L2 pairwise direction
  `P_B_pairwise_direction_unit_l2`。没有空 group、没有修改项目级 schema、raw `P_B_rec` 没有被 aligned field 覆盖。
- 独立打印并检查完整 HDF5 tree；source/run/HDF5 config semantic equality 均为 true，全部 numeric datasets finite。
- 五张 figure 全部实际读取并目视检查：convergence 两条 measurement curves 几乎重合且单调下降；probe comparison 与 detector
  residual 延续上一轮结构；stability 图保留 control 的细粒度差异；conditioning log bar 图显示 pairwise 低于 8 个 random controls，
  native-phase 和 gradient comparator 均更高。PNG 均非空、可读、无空白或损坏。

### 当前 metrics

- primary measurement-only：loss `0.03127716305291964 -> 1.9277964661849684e-5`；detector relative residual
  `0.3567623010258413 -> 0.008857188126643553`；gradient L2
  `0.002737010680953316 -> 9.3765771650562e-6`；60 iterations、loss nonincreasing、0 backtracking、0 step-scale
  fallback，stopping reason `iteration_budget`。
- control measurement-only：loss `0.03133195189085838 -> 1.93299503664604e-5`；detector relative residual
  `0.357074638166358 -> 0.008869122404370298`；gradient L2
  `0.0027404734802917866 -> 9.385454989866588e-6`；同样 60 iterations、0 backtracking/fallback。
- truth-free pairwise：final prediction relative L2 `0.0007003237596188443`；final raw/aligned probe relative L2
  `0.056199044487549206 / 0.056199000098323494`；loss absolute difference `5.198570461071786e-8`；detector
  residual absolute difference `1.1934277726744283e-5`。
- truth-free conditioning：pairwise gain/curvature `1.0561312508814082e-4 / 1.115413219088328e-8`；random gain
  min/median/max `4.935238554603458e-4 / 5.510060015880909e-4 / 6.031190732290739e-4`；pairwise/random-median
  ratio `0.1916732753976295`，random fraction at/below pairwise `0.0`；gradient gain `0.00888478663562367`；
  native-phase comparator gain `0.003944233199468681`；`truth_used_by_diagnostic=false`。
- simulation evaluation only：primary global-phase-aligned probe relative L2
  `0.47873691495342324 -> 0.13937197214033825`；control
  `0.48200798884046003 -> 0.1492569162151173`。这些 truth-aided values 只用于 deterministic simulation evaluation，
  没有进入 optimizer/diagnostic，也不构成 scientific pass/fail。

### 失败、限制与未关闭问题

- 第一次 exact-gauge test 失败不是通过调 threshold 消除，而是暴露并修正了 native ROI 与 fixed complex exterior 的语义差异；
  本文旧第 9.1 节不回改。当前 simulation-only global-phase alignment 仍作为既有辅助 metric 保存，但 raw fields/metrics 也完整保留，
  且本 run alignment phase 很小。
- random-direction comparison 是局部、有限的 8-direction diagnostic，不是 full Jacobian singular spectrum；它支持 weak-direction
  解释，不能建立全局唯一性、resolution 或 detection limit。
- 没有 continuation，因此 finite-budget contribution 尚未定量分离；两条 reconstruction 都以 `iteration_budget` 停止，loss/gradient/
  probe-error 在第 60 步仍下降。
- development case 仍是小规模、无噪声、integer-shift、matched scalar working-model fixture；exp040 evidence boundary 保持
  `reference_validated=false`、`full_tgv_reference_authorized=false`。
- 本轮没有科学 threshold、Passed/Failed 或 physical accuracy 结论；状态仍为
  `Development baseline / No scientific pass-fail conclusion`。

### 改动后总体优先级

- 下一轮主要矛盾（一个）：在不改变 data/B/scan/operator/optimizer 的前提下，定量分离 pairwise weak-direction 中尚存的
  finite-budget transient。最值得做的是预注册一次从本 run 两个 raw final probes 出发、相同额外预算的 paired continuation-only
  diagnostic，追踪 pairwise projection/gain、prediction difference 和 measurement curves；预期判断差异是否继续明显衰减，而不是为降低
  truth error 调参。
- 次要工作一：在新的 append-only correction 中进一步明确 fixed complex exterior phase reference 与第 9.1 节 generic gauge 假设的适用
  边界；必要时增加 full-open phase tangent 的数学 control，但不改变当前 operator。
- 次要工作二：增加 primary/control detector residual 的 spatial/radial truth-free summary，判断相近 residual norm 是否隐藏不同结构。
- 次要工作三：若 paired continuation 仍沿 low-gain direction 停滞，再考虑 matrix-free `J^T J` Krylov/singular-subspace diagnostic；它
  是 identifiability 研究工具，不应在本轮用来修改 scan/B。
- 明确延后 negative mismatch、更多 seeds/amplitudes、blind B、noise/stage/subpixel/calibration、scan/sample-B redesign、regularization、
  full-size、waist inference、reference validation 和真实物理声明。
- 建议继续 exp042 的一个最小 continuation diagnostic；若 continuation 与 Krylov evidence 均确认结构性弱观测，则停止优化并把
  scan/sample-B identifiability 作为 exp041 或新研究任务，而不是在 exp042 静默扩大模型。

### 下一轮快速恢复上下文

- 当前 authoritative appended section：本文第 19 节。
- 本轮完成的 Change：Change 01 truth-free pairwise local Jacobian gain、gradient/native-phase/8-random controls、gauge-semantics
  correction、metrics/HDF5/第五图持久化、targeted validation 和一次正式 run。
- 本轮修改文件：
  `configs/experiments/exp042_TGV_3d_multislice_probe_reconstruction.yaml`、
  `src/tgv_ptycho/recon/exp042.py`、`scripts/run_exp042_probe_reconstruction.py`、
  `tests/test_exp042_probe_reconstruction.py`，以及本文末尾第 19 节。
- 当前有效 config：上述 exp042 YAML，source SHA256
  `A3418E10495DCB19FED2A1A19867C906B2DD26735779E646709E6665DA88FE18`；primary/control 仍为 60-step
  GN-scaled Armijo，control seed `20260843`、relative L2 `0.05`；conditioning seed `20260844`、8 random directions。
- 最新有效 run：
  `runs/exp042_TGV_3d_multislice_probe_reconstruction_dev_20260822_134015`，complete/validated；run-config SHA256
  `B93E59ED52670DC1E9D829FF7D70BA9D49B92E69001B54643E2BDC298CFB6581`。
- 已通过的 tests/commands：最终 targeted pytest `7 passed in 32.47s`；修改范围 Ruff `All checks passed!`；latest-run
  read-only diagnostic exit `0`；唯一正式 runner exit `0`；config/metadata/metrics/full HDF5 tree/五张 PNG read-back 与目视检查通过。
  第一次 exact-gauge pytest failure 和 multiline `conda run -c` command failure 已记录，不应恢复错误 gauge 断言。
- 已确认不需要重跑的 controls：只要 forward/operator/config 未改变，本节 replay、field/shift/q4 adjoint、gradient/Jacobian finite
  difference、zero residual、truth fixed point、finite-B boundary 和 determinism 不需机械重跑；exp040 reference controls 不需重算。
- 当前未关闭的主要矛盾：pairwise direction 已确认相对 8 个 random controls 弱观测，但 finite-budget contribution 尚未通过固定
  continuation 定量分离。
- 当前次要矛盾：fixed-exterior/generic-gauge 语义边界；paired residual spatial structure；必要时的 matrix-free singular-subspace
  diagnostic。negative mismatch 继续延后。
- 下一轮推荐的最小改动：只在 exp042 config/module/test/runner 内增加一次预注册 paired continuation-only control，从本 run 的两张 raw
  final probes 启动，保持 measurement operator 与额外 iteration budget 配对；不得用 truth 决定 budget、停止或选择结果，最多一个新 run。
- 下一轮最小读取集合：`AGENTS.md`；本文第 3、6、9、10 节与本第 19 节，尤其本恢复块；当前 exp042 YAML/module/test/runner；最新 run
  的 `config.yaml`、`metrics.json` 和 HDF5 primary/control raw `P_B_rec`、loss/residual/gradient curves 与 conditioning subgroup。
- 只有 forward/operator/B/scan/exterior-reference 语义改变、摘要与 artifact 冲突、出现 regression、latest artifact 不完整，或 continuation
  问题无法由本节定位时，才重新读取 exp040 对应章节，并在新 appended section 说明原因、范围和结论。
- 明确不应重新扫描：exp040 全文/`_old.md`、全部历史 runs、整个 `src`/`tests`、notebooks、reports、data、exp041/exp05x
  详细内容和无关 theory notes。

### Git 状态

本轮只产生本地 unstaged/untracked 修改；没有执行 `git add`、commit、push、PR、merge 或 branch 操作。exp042 的
config/script/module/test/本文仍为 untracked，runs 被 Git ignore；staged 为空。开始时已有的 exp030/exp040 文档修改、deleted
notebook 和其他 untracked 用户文件保持原状态，未覆盖、未删除、未重解释。commit/push/PR 状态均未改变。

## 20. 2026-08-22 14:09：Implementation iteration 05 — truth-free equal-budget paired continuation

### 本轮目标与明确未做事项

本轮继承第 19 节唯一未关闭的主要矛盾：pairwise direction 已确认相对随机方向弱观测，但其中 finite-budget contribution 尚未通过
固定 continuation 定量分离。本轮保持 matched data、known B、scan、operator、optimizer、initial primary/control definitions 和前
60-step budget 不变，从两张 same-run raw final probes 出发分别继续完全相同的 60 steps；只用 measurement loss、residual、pairwise
field/prediction difference、start-direction projection 和 local Jacobian gain 解释结果。

本轮没有用 truth 设置 continuation budget、checkpoint、停止或结果选择；没有增加更多 seeds、blind B、noise/stage/subpixel/
calibration、negative mismatch、scan/sample-B redesign、regularization、full-size、waist inference 或 exp040 reference validation；没有
运行第二个 formal run，也没有因看到结果继续追加 iteration。

### 开始时检查的内容

- 执行 `git -c safe.directory=E:/tgv_ptycho_sim status -sb`，开始时用户已有 exp030/exp040 文档修改、deleted notebook 和其他
  untracked 文件均保持；exp042 文件仍为 untracked，staged 为空。
- 读取 `AGENTS.md`；按恢复协议读取本文第 3、6、9、10 节和 authoritative 第 19 节全文，重点复核“失败、限制与未关闭问题”、
  “改动后总体优先级”“下一轮快速恢复上下文”和 Git 状态。
- 代码修改前锁定本文为 `109601` bytes，SHA256
  `27AB9CACA7D8588196350A04DEEBF34B4DA5BD3F5ADFFAC90FEB050187410566`，最后章节号 `19`。
- 定向检查当前 exp042 YAML、config validation、`reconstruct_known_b_probe()` 的 raw `probe_history` 与 endpoint semantics、
  `_align_global_phase()`、`normalized_intensity_jacobian_sensitivity()`、pairwise diagnostic、runner metrics/HDF5/figure writer、artifact
  validation 和 targeted tests。
- 读取第 19 节 latest run
  `runs/exp042_TGV_3d_multislice_probe_reconstruction_dev_20260822_134015` 的 config/metrics 和 HDF5 primary/control raw probes/
  curves/conditioning subgroup，用于正式 run 后的 exact base replay comparison；旧 run 没有作为 truth 或 optimizer feedback。
- 没有扩大读取范围：没有重读 exp040、`_old.md`、全部 runs、整个 `src`/`tests`、notebooks、reports、data、exp041/exp05x 或无关
  theory notes；没有联网。现有 exact operator 和第 19 节 artifact 足以完成问题。

### 上一轮意见

第 19 节认为主要矛盾是 weak pairwise direction 中 remaining finite-budget contribution；建议只做从两张 raw final probes 出发、相同
额外预算的 paired continuation，追踪 pairwise projection/gain、prediction difference 和 measurement curves，不用 truth 决定预算、停止
或选择结果，最多一个新 run。次要矛盾是 fixed-exterior/generic-gauge 语义边界、paired residual spatial structure，以及必要时的
matrix-free singular-subspace diagnostic。本轮复核后完整采用该意见；继续优化之前必须先量化同等预算实际能消除多少 field difference。

### Change 01

#### 改动前：主要矛盾、次要矛盾与必要性判断

- 继承的上一轮主要矛盾仍成立：第 19 节只能证明 direction local gain 较低，尚不知道另一个相同 60-step budget 会使 probe difference
  大幅缩小还是基本保留。
- 本 Change 的唯一主要矛盾：在 data/B/scan/operator/optimizer 全冻结时，定量测量 extra 60 steps 对 pairwise probe difference、
  prediction difference 和 direction/gain 的作用。
- 次要矛盾最多三个：fixed-exterior/generic-gauge 解释；paired residual spatial structure；必要时的 local singular-subspace 分析。
- 明确延后 negative mismatch、更多 seeds/amplitudes、scan/B 改动、regularization、blind reconstruction 和任何 waist/physical claim。
- 当前 equal-budget continuation 直接隔离 iteration budget，且比更换 optimizer/先验或继续随机初始化 sweep 更小、更可审计。

#### 技术决策与理由

- config 在看到新结果前固定 `additional_iterations=60`，与 base budget 完全相等；checkpoint 固定为每 10 steps，共
  `[0,10,20,30,40,50,60]`；复用同一 GN-scaled Armijo settings，truth 和科学 threshold 均禁用。
- 为使新 run 自包含并验证可重复性，runner 先确定性重放 base primary/control 60 steps，再将其 raw `P_B_rec` 逐元素原样作为两条
  continuation 的 `P_B_init`。正式审计同时与第 19 节 HDF5 比较 base fields/curves，避免把 regenerated start 当成未经核对的新条件。
- continuation raw results 分别保存，不覆盖 top-level base reconstruction。truth-free diagnostic 对每一步保存 raw/aligned pairwise relative
  L2、对 start direction 的 real projection fraction/cosine；固定 checkpoint 额外计算 prediction relative L2 和 exact intensity-Jacobian
  RMS gain。
- 沿用第 19 节已登记的 pairwise global-phase alignment 仅用于 pairwise diagnostic，同时保留 raw difference；不把 aligned field 写回
  optimizer。当前 fixed complex exterior 下该 alignment 不是可自由优化的物理 gauge，这一限制继续保留，不在本 Change 改 operator。
- 没有 preregister outcome threshold；测试只断言 start/end/shape/determinism/truth flags/finite/HDF5/figure contracts，不断言 continuation
  必须改善 pairwise metric。

#### 创建或修改的文件

- 修改 `configs/experiments/exp042_TGV_3d_multislice_probe_reconstruction.yaml`：增加
  `reconstruction.paired_continuation_control` 和第六张 `exp042_paired_continuation.png`；source YAML SHA256
  `5860A8E86FBF09383EE7AA49F37A95DE2F3361954ED337987D9CD022630F0471`。
- 修改 `src/tgv_ptycho/recon/exp042.py`：增加 continuation config validation 和
  `truth_free_paired_continuation_diagnostic()`；没有修改 matched operator、forward、gradient、optimizer 或 shared optics/shift/IO。
- 修改 `scripts/run_exp042_probe_reconstruction.py`：执行两条 continuation，验证 raw starts exact，持久化 metrics/HDF5，新增第六图并扩大
  artifact validation；原 base reconstruction HDF5 payload 保持兼容。
- 修改 `tests/test_exp042_probe_reconstruction.py`：覆盖 config、raw start equality、checkpoint/projection invariants、finite gain、tiny runner、
  metrics/HDF5 subgroup 和六图。
- 本节是设计文档唯一修改；没有修改 exp040/exp041/exp050、README、roadmap、notebooks 或 reports。

#### 验证命令与结果

```powershell
D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim python -m pytest -q tests/test_exp042_probe_reconstruction.py
D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim python -m ruff check src/tgv_ptycho/recon/exp042.py scripts/run_exp042_probe_reconstruction.py tests/test_exp042_probe_reconstruction.py
D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim python scripts/run_exp042_probe_reconstruction.py --config configs/experiments/exp042_TGV_3d_multislice_probe_reconstruction.yaml
```

- targeted pytest 第一次即通过：`7 passed in 37.69s`；没有 outcome-directed failure 或阈值调整。
- 修改范围 Ruff：`All checks passed!`。
- 没有运行 full pytest：本轮没有改变 shared forward/optics/shift/IO，exp042 targeted suite 已覆盖新增 public diagnostic、runner、
  config、HDF5 和 figures；不扩大到全仓既有问题。
- 唯一正式 runner exit `0`，没有第二个 timestamped run。

#### 改动后重新评估

- 本轮主要矛盾在当前 development case/固定 equal-budget 定义下“已关闭”：额外 60 steps 后 aligned probe difference
  `0.056199000098323494 -> 0.05523950908525804`，final/initial ratio `0.9829269024113104`，即只缩小约
  `1.71%`；同时 prediction difference ratio 为 `0.6732357044724224`，两条 measurement losses/residuals 均继续明显下降。
  因此 finite budget 确实仍影响 detector-space mismatch，但不是 pairwise probe difference 持续约 5.5% 的主要解释。
- final difference 对 initial weak direction 的 projection fraction `0.9883989122851806`，real cosine
  `0.9981378651221002`；方向几乎保留。pairwise gain 同时降至初始的 `0.6802561737180043`，说明 continuation 进入更弱的
  measurement sensitivity，而不是向一般 random direction 旋转。
- 新证据把问题推进为 local identifiability/singular-subspace 研究问题。继续追加 optimizer budget 不是当前主要工作；若继续 exp042，
  下一 Change 应为 matrix-free spectral diagnostic，而非再做 continuation 或调 optimizer。

### Operator-consistency 检查

- 正式 run 重新执行 matched controls：truth replay relative L2 `0.0`；truth loss/gradient/one-step change 均 `0.0`；zero-residual
  update L2 `0.0`。
- linear composite adjoint relative error `1.4510142628728665e-15`；constant-zero shift adjoint
  `5.682234924561534e-16`；physical-weighted q4 adjoint `0.0`。
- full loss gradient relative error `1.2347820022705336e-9`；intensity-Jacobian directional finite-difference error
  `1.0858519027590559e-9`。
- detector constant/sum errors均 `0.0`，minimum intensity `0.04091490300503436`、all nonnegative，node geometry error
  `6.776263578034403e-15`；finite-B shifted `(B-1)` edge modulation `0.0`。
- determinism：当前 run 的 base primary/control raw `P_B_rec` 和两条 loss curves 与第 19 节 run 均 `array_equal=True`、max absolute
  difference `0.0`；当前 HDF5 中两张 continuation `P_B_init` 与相应 same-run raw base `P_B_rec` 均 exact equal。
- 全部 HDF5 numeric datasets finite；continuation 和 diagnostic 均记录 `truth_used_by_continuation=false`、
  `truth_used_by_diagnostic=false`。

### Development run 与 artifacts

唯一新 run：

`runs/exp042_TGV_3d_multislice_probe_reconstruction_dev_20260822_140501`

- `run_state.json` 为 `complete`、`artifacts_validated=true`，runtime `164.10905879999973 s`，figure count `6`。
- run config SHA256 `586C7DE5C775DC8BB4956D0C04700FB29FB7388788D3D18275D1CC315CED5F5A`；source YAML、run
  `config.yaml` 与 HDF5 `/entry/config_yaml` 语义完全相等。source/run 字节哈希差异只来自 `save_config()` serialization。
- metadata SHA256 `E8448FE37FB748B3795522C63DB01E016E4F2EA27735C6A06F1CD88D41BF8C5E`，新增
  `paired_continuation_control=true`，并保持 known-B/probe-only、`reference_validated=false`、
  `full_tgv_reference_authorized=false` 和 `truth_used_by_optimizer=false`。
- metrics SHA256 `D0E760FBDDDE1741ED23FFEA02EC13C3A026E35C2B9AE2738BCBDD94BF8B7823`；HDF5 SHA256
  `37B9DB85256026C97E720657D93DEF91DD5F7119DE29B3657DDCBC87986D4771`。
- HDF5 `/entry` 顶层没有改变。新增
  `/entry/reconstruction/paired_continuation_control/{primary,control,truth_free_pairwise_diagnostic}`，保存两条 raw starts/finals、完整
  loss/residual/gradient/step/curvature histories、unit start direction 和 continuation diagnostic curves；对应 summary 保存到
  `/entry/metrics/paired_continuation_control`。没有空 group、没有项目级 schema change、没有 truth-aligned continuation field。
- 独立打印并检查完整 HDF5 tree；三份 config semantic equality 为 true，全部 numeric data finite。
- 六张 figures 全部实际读取并目视检查：前五图与 exact replay 一致；新增 continuation 图中两条 loss/residual 平滑下降，pairwise field
  curve 只缓慢下降并有 optimizer-step 交替小振荡，start-direction projection/cosine 保持接近 1；prediction difference 和 Jacobian gain
  在 checkpoint 同步下降。PNG 均非空、可读、无损坏或明显布局截断。

### 当前 metrics

- base 60-step primary/control 与第 19 节完全相同；base aligned probe difference `0.056199000098323494`、prediction difference
  `0.0007003237596188443`、pairwise gain `0.00010561312508814082`。
- primary continuation measurement-only：loss `1.9277964661849684e-5 -> 5.204078917062369e-6`；detector residual
  `0.008857188126643553 -> 0.004601900413388764`；gradient L2
  `9.3765771650562e-6 -> 3.7607971981497366e-6`；60 iterations、loss nonincreasing、0 backtracking/fallback。
- control continuation measurement-only：loss `1.93299503664604e-5 -> 5.2350708308365225e-6`；detector residual
  `0.008869122404370298 -> 0.0046155829491290375`；gradient L2
  `9.385454989866588e-6 -> 3.7665745698635184e-6`；同样 60 iterations、0 backtracking/fallback。
- truth-free pairwise continuation：aligned probe difference `0.056199000098323494 -> 0.05523950908525804`，ratio
  `0.9829269024113104`；prediction difference `0.0007003237596188443 -> 0.00047148295966576803`，ratio
  `0.6732357044724224`；Jacobian gain `0.00010561312508814082 -> 0.00007184398036685964`，ratio
  `0.6802561737180043`；final start-direction projection/cosine
  `0.9883989122851806 / 0.9981378651221002`。
- 上述 continuation metrics 全部 truth-free；本轮没有新增 truth-aided continuation evaluation，也没有 scientific pass/fail threshold。

### 失败、限制与未关闭问题

- 本轮实现、targeted tests、Ruff、唯一正式 run 和 artifact audit 均无失败；没有删除或覆盖任何失败证据。
- equal-budget continuation 明确表明更多相同优化预算不能快速消除 pairwise probe difference，但仍只是一个 matched、noiseless、小规模
  development case；不能把局部 persistence 推广为所有 scan/B/probe 的全局 null space。
- pairwise direction 相对 8 个 random directions weak 且经 continuation 保持，但尚未计算 matrix-free `J^T J` 的低 singular
  subspace；因此“local weak direction”证据充分，“完整谱/秩/分辨率/detection limit”仍未建立。
- 当前 fixed complex exterior phase reference 与旧第 9.1 节 generic global-phase ambiguity 的语义边界仍需 append-only correction；raw
  metrics 保留，本轮结果不依赖把 native phase rotation 当作 gauge。
- 两条 continuation 都仍以 `iteration_budget` 停止；这不再构成本轮主要矛盾，因为另一个等预算只改变 pairwise field 约 1.71%，但也
  不允许宣称无限预算极限已证明。
- 状态仍为 `Development baseline / No scientific pass-fail conclusion`；不形成 exp040 物理准确性、真实 TGV recovery 或 waist 结论。

### 改动后总体优先级

- 下一轮主要矛盾（一个）：判断 persistent pairwise direction 是否实际落入 local low-singular-value subspace，而不是仅相对 8 个 random
  directions 较弱。最值得做的是一个预注册、matrix-free `J^T J` Lanczos/Krylov diagnostic，在当前 base/continuation final probe 处使用
  exact Jacobian action与adjoint，报告 low-mode Ritz values、residuals 和 pairwise subspace overlap；不改变 optimizer、scan 或 B。
- 次要工作一：通过新的 append-only correction 明确 fixed complex exterior 的 phase-reference/gauge 边界；只改解释和 control，不改
  operator。
- 次要工作二：增加 paired detector residual spatial/radial truth-free summary，确认 norm 相近时结构是否也相近。
- 次要工作三：对 Krylov 数值稳定性只做必要的 reorthogonalization/residual controls，不把它发展成新的 reconstruction framework。
- 明确不建议现在做：第三次 continuation、optimizer/regularizer tuning、更多 initialization sweep、negative mismatch、scan/sample-B redesign、
  blind B、noise/experimental calibration、waist inference 或 reference validation。
- 当前主要优化问题已关闭并形成新的 local identifiability 研究问题。建议只以一个独立、受控的 exp042 spectral-diagnostic iteration 继续；
  若 low-mode overlap 得到确认，则停止扩张 exp042，把 scan/sample-B information redesign 交给 exp041 或新任务。

### 下一轮快速恢复上下文

- 当前 authoritative appended section：本文第 20 节。
- 本轮完成的 Change：Change 01 same-run exact base replay、两条 equal 60-step continuation、truth-free pairwise projection/prediction/
  gain trajectories、metrics/HDF5/第六图持久化、targeted validation 和一次正式 run。
- 本轮修改文件：
  `configs/experiments/exp042_TGV_3d_multislice_probe_reconstruction.yaml`、
  `src/tgv_ptycho/recon/exp042.py`、`scripts/run_exp042_probe_reconstruction.py`、
  `tests/test_exp042_probe_reconstruction.py`，以及本文末尾第 20 节。
- 当前有效 config：上述 exp042 YAML，source SHA256
  `5860A8E86FBF09383EE7AA49F37A95DE2F3361954ED337987D9CD022630F0471`；base primary/control 60 steps，
  continuation primary/control 各 60 steps，checkpoint interval 10，其他 optimizer/data/operator settings 不变。
- 最新有效 run：
  `runs/exp042_TGV_3d_multislice_probe_reconstruction_dev_20260822_140501`，complete/validated；run-config SHA256
  `586C7DE5C775DC8BB4956D0C04700FB29FB7388788D3D18275D1CC315CED5F5A`。
- 已通过的 tests/commands：targeted pytest `7 passed in 37.69s`；修改范围 Ruff `All checks passed!`；唯一 formal runner exit `0`；
  section 19 base replay exact comparison、same-run continuation start equality、config/metadata/metrics/full HDF5 tree/六张 PNG read-back 与
  目视检查均通过。
- 已确认不需要重跑的 controls：只要 forward/operator/config/Jacobian action 未改变，本节 truth replay、field/shift/q4 adjoint、loss/Jacobian
  finite difference、zero residual、truth fixed point、finite-B boundary、base replay 和 continuation 不需机械重跑；exp040 reference controls
  不需重算。下一轮 spectral diagnostic 可以直接读取 latest HDF5 raw base/continuation probes与 operator provenance。
- 当前未关闭的主要矛盾：persistent pairwise direction 是否与 matrix-free local low-singular subspace 对齐。
- 当前次要矛盾：fixed-exterior/gauge 解释边界；paired residual spatial structure；Krylov residual/reorthogonalization controls。
  negative mismatch 继续延后。
- 下一轮推荐最小改动：只在 exp042 config/module/test/runner 内增加预注册 matrix-free `J^T J` Lanczos/Krylov diagnostic；复用当前 exact
  Jacobian/adjoint和 latest raw probes，不改 reconstruction，不运行 continuation，最多一个新 timestamped run。
- 下一轮最小读取集合：`AGENTS.md`；本文第 3、6、9、10 节与本第 20 节，尤其本恢复块；当前 exp042 YAML/module/test/runner；latest run
  `config.yaml`、`metrics.json` 与 HDF5 base/continuation raw probes、conditioning/continuation subgroups。无需重读第 19 节全文。
- 只有 forward/operator/B/scan/exterior-reference 或 Jacobian-adjoint 语义改变、摘要与 artifact 冲突、出现 regression、latest artifact 不完整，
  或 Krylov blocker 无法由本节定位时，才重新读取 exp040 对应章节，并记录扩大范围原因、证据和停止点。
- 明确不应重新扫描：exp040 全文/`_old.md`、全部历史 runs、整个 `src`/`tests`、notebooks、reports、data、exp041/exp05x
  详细内容和无关 theory notes。

### Git 状态

本轮只产生本地 unstaged/untracked 修改；没有执行 `git add`、commit、push、PR、merge 或 branch 操作。exp042 的
config/script/module/test/本文仍为 untracked，runs 被 Git ignore；staged 为空。开始时已有的 exp030/exp040 文档修改、deleted
notebook 和其他 untracked 用户文件保持原状态，未覆盖、未删除、未重解释。commit/push/PR 状态均未改变。

### Append-only verification correction note

- Correction：上文“本轮实现、targeted tests、Ruff、唯一正式 run 和 artifact audit 均无失败”只指代码、测试、运行和
  artifacts；第一次文档 append 因补丁锚点复用了第 18/19 节相同 Git 段落，曾把 section 20 插到 section 19 之前。首次校验得到
  heading 顺序 18→20→19、原 109601-byte prefix SHA256
  `6EE2D92D77D0B80C71C05554D52CBA59C13D7D129B793956B15A482B461757AD`，因此明确判定失败，没有接受或隐藏。
- 修复只移除本轮误插入的 section 20；随后先独立验证文档精确恢复为 `109601` bytes、SHA256
  `27AB9CACA7D8588196350A04DEEBF34B4DA5BD3F5ADFFAC90FEB050187410566`，再用第 19 节独有恢复块作为 EOF
  锚点重新追加同一 section 20。最终旧前缀 SHA256 仍为上述 `27AB...410566`、`PrefixMatches=true`，heading 顺序为
  19→20。该修复没有改代码/config/run，没有重跑实验，也没有触碰任何历史 section 字节。

## 21. 2026-08-22 14:45：Implementation iteration 06 — matrix-free local spectral diagnostic

### 本轮目标与明确未做事项

本轮继承第 20 节唯一未关闭的主要矛盾：persistent primary/control pairwise direction 是否实际集中在当前 matched operator 的
local low-singular-value support，而不只是相对 8 个 random directions 的单方向 gain 较低。本轮在看到新结果前固定
`A = J^T J / N_detector`、pairwise/random 两个 start、Krylov dimension 12、two-pass full reorthogonalization、seed
`20260845` 和 lowest-3 Ritz summary；新增严格 real-inner-product `J^T`，只执行 bounded、matrix-free、start-dependent Lanczos
diagnostic。

本轮没有修改或运行 reconstruction、paired continuation、known B、scan、sample-B、forward prediction、loss 或 optimizer；没有使用
truth 选择 start、Krylov dimension、停止或结果；没有增加 blind B、noise/stage/subpixel/calibration、negative mismatch、scan/B redesign、
regularization、full-size、waist inference 或 exp040 reference validation；没有把 Ritz result 称为完整 spectrum、rank、resolution 或
detection limit。

### 开始时检查的内容

- 执行 `git -c safe.directory=E:/tgv_ptycho_sim status -sb`；开始时用户已有 exp030/exp040 文档修改、deleted notebook 和其他
  untracked 文件均保持，exp042 文件仍为 untracked，staged 为空。
- 读取 `AGENTS.md`；按第 20 节恢复协议读取本文第 3、6、9、10 节与 authoritative 第 20 节全文，重点复核“失败、限制与未关闭
  问题”“改动后总体优先级”“下一轮快速恢复上下文”和 append-only correction note。
- 代码修改前锁定本文为 `130313` bytes，SHA256
  `536DB948F769761398D4F7B139F3D1CA1CFE47C4CDC63DBBFE71E5F041C01163`，最后章节号 `20`。
- 定向检查当前 exp042 YAML；`MatchedKnownBProbeOperator.intensity_jacobian_direction()`、
  `linear_detector_field_adjoint()`、q4 residual scatter/crop gradient；conditioning/continuation diagnostics；runner import、HDF5 payload、
  artifact validator、dispatch 和 targeted tests。
- 读取 latest source run
  `runs/exp042_TGV_3d_multislice_probe_reconstruction_dev_20260822_140501` 的 config、metrics、HDF5 data/scan 和两张 raw
  continuation-final probes；源 config/metrics/HDF5 SHA256 分别锁为 `586C7D...CED5F5A`、`D0E760...BF8B7823`、
  `37B9DB...86D4771`。
- 为确认 diagnostic-only payload 可沿用现有可选 group contract，额外定向读取直接 IO writer
  `src/tgv_ptycho/io/save_load.py::save_ptycho_hdf5()` 的签名与可选 group 行为；这是 runner HDF5 blocker 的最小直接依赖，结论是
  可以自然省略本 run 未产生的 truth group，无需修改共享 IO 或项目级 schema，得到答案后停止扩展。
- 没有重读 exp040、`_old.md`、全部 runs、整个 `src`/`tests`、notebooks、reports、data、exp041/exp05x 或无关 theory notes；没有
  联网。现有 exact Jacobian action、field adjoint和第 20 节 artifact 足以完成问题。

### 上一轮意见

第 20 节认为继续优化已不是主要工作，建议只增加预注册、matrix-free `J^T J` Lanczos/Krylov diagnostic，在 current
base/continuation-final probe 处用 exact Jacobian action/adjoint，报告 Ritz values、residuals、pairwise spectral weights/overlaps；不改
optimizer、scan 或 B，不重跑 continuation，最多一个 timestamped run。本轮复核后继续采用该意见。次要问题为 fixed-exterior/gauge
解释边界、paired residual spatial structure 和必要的 Krylov residual/reorthogonalization controls。

### Change 01

#### 改动前：主要矛盾、次要矛盾与必要性判断

- 继承的主要矛盾仍成立：第 20 节证明 pairwise direction 经 equal-budget continuation 几乎保留且 gain 继续下降，但没有谱分解证据。
- 本 Change 的唯一主要矛盾：在 forward/data/B/scan 和两张 raw continuation-final probes 全冻结时，判断 pairwise direction 的
  start-dependent Ritz spectral measure 是否比 deterministic random start 更集中于低曲率 support。
- 次要矛盾最多三个：real `J^T` correctness；Krylov orthogonality/Ritz residual 审计；fixed-exterior 下 pairwise global-phase alignment 的
  解释边界。
- 明确延后 paired residual spatial summary、negative mismatch、更多 starts/dimensions、shift-invert、optimizer/regularizer、scan/B 改动、
  blind reconstruction 和所有 waist/physical claims。
- 只有 exact real adjoint 才能定义可审计的 PSD `J^T J`；从已锁 source probes 运行 diagnostic-only entry 比再做 240 个优化步更小、
  更直接，也避免把新 reconstruction trajectory 混入 local identifiability 判断。

#### 技术决策与理由

- 新 `intensity_jacobian_adjoint()` 使用 detector ROI center-embed、q4 pixel-average Euclidean transpose、`2 r U` intensity adjoint 和已有
  exact field adjoint；其配对定义为
  `sum((J d) * r) = Re(sum(conj(d) * J^T r))`，不把 complex Hermitian equality误当作 real-parameter transpose。
- normal operator 固定为 `A d = J^T(J d) / 25600`，所以 unit-L2 start 的 Rayleigh quotient 等于 detector-stack RMS Jacobian gain
  的平方；全程 matrix-free，不显式形成 Jacobian 或 normal matrix。
- 两条独立 Lanczos 使用 pairwise aligned difference 和 deterministic zero-mean complex Gaussian comparator；dimension 12、two-pass full
  reorthogonalization、relative breakdown tolerance `1e-12`、lowest-3 summary 和 seed 均在正式结果前写入 YAML，不设置 scientific
  pass/fail threshold。
- 每条结果保存 projected tridiagonal、ascending Ritz values、sqrt-clipped gain estimates、terminal-beta residual estimates、start spectral
  weights、weighted quantiles、Rayleigh reconstruction error和 basis real-orthogonality error；这些是 start-dependent bounded Ritz measures，
  不是完整低 singular spectrum。
- runner 进入 `local_spectral_diagnostic_from_prior_run` 模式：先核对源三份 SHA256、complete/validated state、九个 operator config sections、
  exact `I_stack`/scan replay，再读取两张 raw probes。新 run 仍确定性重建相同 operator，但不执行任何 reconstruction/continuation；新
  HDF5 保存 raw source probes，不覆盖 source run。
- append-only correction：本文第 9.1 节是 generic known-B gauge 描述；当前 branch 的 homogeneous exterior/reference complex phase 固定，
  因而 native `P_B` 的任意 global phase rotation 一般是可观测变化。本轮和前轮的 global-phase alignment 只作为预登记 pairwise comparison
  convention，raw probes/metrics 才是 authoritative；不把该 alignment 解释为当前 operator 的自由物理 gauge。

#### 创建或修改的文件

- 修改 `configs/experiments/exp042_TGV_3d_multislice_probe_reconstruction.yaml`：增加 spectral-only execution、锁定 source artifact、
  dimension/start/seed/reorthogonalization/residual 设置和唯一 spectral figure；source YAML SHA256
  `CD464BB9EAA36F452FE647CDA7673E4183F80A17B6D692D259B9C2CD9F4FFEEB`。
- 修改 `src/tgv_ptycho/recon/exp042.py`：增加 real intensity-Jacobian adjoint/dot test、fully reorthogonalized Lanczos 和 truth-free local
  spectral diagnostic；没有修改 forward action、loss、gradient、optimizer 或 shared optics/shift/IO。
- 修改 `scripts/run_exp042_probe_reconstruction.py`：保留旧 full-baseline implementation，新增当前 config 的 spectral-only dispatcher、
  source provenance validation、metrics/HDF5/one-figure writer 和 artifact validation。
- 修改 `tests/test_exp042_probe_reconstruction.py`：增加 real-adjoint dot test、8-real-dimensional diagonal Lanczos exact-spectrum control、
  local diagnostic determinism和 tiny spectral-only artifact contract。
- 本节是设计文档唯一修改；没有修改 exp040/exp041/exp050、README、roadmap、notebooks、reports 或共享 IO 文件。

#### 验证命令与结果

```powershell
D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim python -m ruff check src/tgv_ptycho/recon/exp042.py scripts/run_exp042_probe_reconstruction.py tests/test_exp042_probe_reconstruction.py
D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim python -m pytest -q tests/test_exp042_probe_reconstruction.py
D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim python scripts/run_exp042_probe_reconstruction.py --config configs/experiments/exp042_TGV_3d_multislice_probe_reconstruction.yaml
```

- scoped Ruff 首次发现一个新增 `E501`（module 中 89-character error-message line），在正式 run 前仅机械换行修正；第二次为
  `All checks passed!`。没有隐藏这次 lint failure，也没有改变数值实现或 preregistration。
- targeted pytest 第一次通过：`8 passed in 46.74s`；包括 real-adjoint、diagonal spectrum、determinism、tiny runner、source hashes、HDF5
  和 one-figure contract。
- 没有运行 full pytest：没有修改 shared forward/optics/shift/IO，新增风险由 exp042 targeted suite覆盖；不扩大到全仓既有问题。
- 唯一正式 runner exit `0`；没有第二个 timestamped run，没有根据结果调整 dimension、seed、summary 或 residual interpretation。

#### 改动后重新评估

- 本轮主要矛盾“部分关闭”：pairwise start Rayleigh quotient 是 random start 的 `0.016439041231112783`，weighted-median Ritz value
  ratio 为 `0.15165045585716508`；pairwise projected spectral weight 的 `0.9898857696047444` 落在其最低 Ritz vector，最低三个合计
  `0.9999996071227836`。在当前 bounded Krylov measure 中，persistent direction 明确比 random comparator 更弱并集中于低曲率 support。
- 不能把它评为完全关闭：random start 自身也有 `0.9634422551732417` 权重落在其最低 Ritz vector；pairwise/random lowest Ritz residual
  相对各自最低 Ritz value 约为 `32.6/16.7`，说明最低 eigenvalue 数值尚未按自身量级收敛。当前证据支持“start-dependent low-spectral
  concentration”，不支持完整 low-singular subspace、null space、rank 或 condition number claim。
- pairwise direction 投影到 random-start 12-D Krylov subspace 的总 weight 仅 `4.684108149921883e-05`；其中 `99.52%` 落在 random lowest-3
  Ritz vectors，但总 captured weight 太小，不能把该比例单独解释成跨-start low-subspace overlap 证明。
- 新证据暴露的下一主要矛盾是最低 Ritz component 的 Krylov-dimension/residual convergence；不应通过继续 reconstruction 或换 optimizer
  处理。fixed-exterior/gauge 解释已由本节 correction 关闭；paired residual spatial structure仍为次要项。

### Operator-consistency 检查

- 新 real intensity-Jacobian adjoint dot-test 正式 run relative error
  `1.809942305798957e-15`；测试 grid 同类 control `<1e-11`。
- diagonal Lanczos control 在 8-dimensional real parameter space 精确重放 eigenvalues `1..8`，spectral weights sum `1`、basis
  orthogonality `<1e-12`、maximum Ritz residual `<1e-10`、Rayleigh reconstruction error `<1e-12`。
- 正式 pairwise/random 两条均完成预登记 12 dimensions，无 breakdown、无 negative Ritz value；basis real-orthogonality max error均
  `2.220446049250313e-16`，Rayleigh reconstruction relative error分别 `6.410317894601204e-16/0.0`，spectral weights sum均为 `1.0`。
- source config operator sections exact；source/current `I_stack` 与 scan positions逐元素 exact，relative L2均 `0.0`；新 HDF5 raw primary/control
  probes 与 source HDF5 对应 raw probes逐元素 exact。
- forward action没有改变，且源 run 三重 hashes和 exact data replay均通过，因此第 20 节已通过的 truth replay、field/shift/q4 adjoint、loss/
  Jacobian finite difference、zero residual、truth fixed point、finite-B boundary和 continuation controls没有机械重跑；metrics明确记录
  `inherited_forward_controls_rerun=false`。
- formal run没有 truth input；diagnostic/HDF5/metrics均记录 `truth_used_by_diagnostic=false`。测试中的 deterministic repeat 得到 exact Ritz
  arrays；正式 run 遵守最多一次约束，未用第二个 run 做结果选择。

### Development run 与 artifacts

唯一新 run：

`runs/exp042_TGV_3d_multislice_probe_reconstruction_dev_20260822_144251`

- `run_state.json` 为 `complete`、`artifacts_validated=true`，runtime `17.62988459999906 s`，figure count `1`。
- run config SHA256 `92F6BCC163A8427889CA10E9ACF9632BA34C13928643DB20BA16B9FF81118FE8`；source YAML、run config与
  HDF5 `/entry/config_yaml` 语义完全相等。source/run字节哈希差异只来自 `save_config()` serialization。
- metadata SHA256 `626C3BABED0BAB6F59A2ABCBB557EBEB83D5DF7D046F005D51569A21755C2742`；明确
  `reconstruction_performed_in_this_run=false`、`paired_continuation_performed_in_this_run=false`、known-B/probe-only、
  `reference_validated=false` 和 `full_tgv_reference_authorized=false`。
- metrics SHA256 `3D777C2FF7DF1E72AA4EAA26C82AB542E0E856341B7D26901AD7D64DC0DECE0E`；HDF5 SHA256
  `35C8452199C53931513E4501EC78080B7D5EC513962A8E630E17AE8FCA34C47C`；figure SHA256
  `B4FF0F2D9C6EF82EF212FAF86D381DAA4CB5443E37B10EA652A293EA2961357A`。
- HDF5 `/entry` 为 `config_yaml/data/instrument/metadata/metrics/reconstruction/sample`。新增/本 run自然产生的核心路径为
  `/entry/sample/B_known`、`/entry/reconstruction/local_spectral_diagnostic/{P_B_primary_probe_raw,P_B_control_probe_raw,
  P_B_control_probe_global_phase_aligned_to_primary,P_B_pairwise_direction_unit_l2,P_B_random_start_direction_unit_l2,pairwise_start,
  random_start,...}` 和 mirrored `/entry/metrics/local_spectral_diagnostic`；没有项目级 schema change，没有空 truth group，没有覆盖 source raw
  reconstruction。
- 独立打印并检查完整 HDF5 tree；213 个 numeric datasets全部 finite；raw source equality、三份 current config semantic equality和 spectral
  weight sums均通过。
- 唯一 figure `exp042_local_spectral_diagnostic.png` 非空、可读取并目视检查：Ritz gain/weight/residual/cumulative-weight 四 panel 标签完整，
  无损坏、遮挡或明显布局截断。图只显示 bounded projected measures，不作为后续计算主数据源。

### 当前 metrics

- pairwise direct Jacobian RMS gain `7.184398036685964e-05`，与第 20 节 continuation-final gain一致；random direct gain
  `0.0005603406675177247`，对应 Rayleigh quotient ratio `0.016439041231112783`。
- pairwise lowest Ritz value/gain/weight为
  `1.8946112671261024e-09 / 4.3527132539671185e-05 / 0.9898857696047444`；pairwise lowest-3 weight
  `0.9999996071227836`。
- random lowest Ritz value/gain/weight为
  `1.2493277757836605e-08 / 0.00011177333205123933 / 0.9634422551732417`；random lowest-3 weight
  `0.9950987925271625`。
- pairwise/random spectral-weighted median Ritz values为 `1.8946112671261024e-09 / 1.2493277757836605e-08`，ratio
  `0.15165045585716508`；pairwise weight at/below random weighted median为 `0.9898857696047444`。
- lowest Ritz residual estimates pairwise/random为 `6.179217363498601e-08 / 2.08307988243795e-07`；相对各自 projected spectral radius为
  `0.0005484498026620335 / 0.0018498853484591375`，但相对 lowest Ritz value仍较大，故不宣称最低 eigenvalue converged。
- 上述指标全部 truth-free；没有 scientific threshold、Passed/Failed 或 simulation-truth selection。状态仍为
  `Development baseline / No scientific pass-fail conclusion`。

### 失败、限制与未关闭问题

- 唯一实现失败是正式 run 前 scoped Ruff 首轮的一个新增 `E501`，已机械修复并复验；targeted tests、正式 run和 artifact audit无失败，
  没有删除或覆盖失败证据。
- Krylov dimension仅 12，且两条最低 Ritz residual均大于最低 Ritz value；当前 lowest values和 derived gains不能视为收敛 singular values。
- 两个独立 single-vector Krylov spaces高度 start-dependent；pairwise 对 random Krylov space的总捕获很小，不能用 low-fraction取代共同 subspace
  或完整 spectrum analysis。
- pairwise 与 random 均在 projected lowest mode含大量 weight，说明当前 operator可能有广泛弱方向；一个 random comparator不足以量化全局
  spectral density。本轮不追加更多 starts，避免扩大为无界 spectral survey。
- pairwise alignment 是固定 exterior/reference 下的比较约定而非自由 physical gauge；raw source probes单独保留。
- 仍只是一个 matched、noiseless、小规模 development case；不能推广为全局 null space、rank、resolution、detection limit、真实 TGV recovery、
  waist inference 或 exp040 物理准确性。

### 改动后总体优先级

- 下一轮主要矛盾（一个）：最低-weight Ritz component 是否随预登记 Krylov dimension 扩大而稳定，并使 residual 相对其自身 Ritz value显著
  收敛。最值得做的是复用同一 source probes/operator/seed，事前固定一个 bounded dimension-extension convergence control（例如 12→24，
  只比较前 12 的 weighted summaries与最低 Ritz residual/value），不做 reconstruction；作用是决定本轮 low-spectral evidence 能否关闭，
  不是追逐更小 eigenvalue。
- 次要工作一：paired detector residual 的 spatial/radial truth-free summary；只在 spectral convergence关闭后考虑，用于解释相同 norm下的
  residual structure。
- 次要工作二：若单向 Lanczos low-mode convergence仍差，只记录为新的 eigenproblem/identifiability研究问题；不要静默引入 shift-invert、
  regularized inverse或大框架。
- 次要工作三：把 scan/sample-B information redesign 的需求交回 exp041或新任务；exp042只提供当前 evidence boundary，不自行改设计。
- 明确不建议现在做：更多 reconstruction/continuation、optimizer/regularizer tuning、更多 initialization sweep、blind B、noise/calibration、
  waist inference、full exp040 reference validation，或把 dimension不断扩大直至 metric好看。
- 建议只允许一个预登记、受控的 spectral convergence iteration继续 exp042；若 dimension extension稳定本轮结论则停止扩张 exp042，若不稳定则
  将其明确提升为新研究问题而不是继续无界 Krylov sweep。

### 下一轮快速恢复上下文

- 当前 authoritative appended section：本文第 21 节。
- 本轮完成的 Change：Change 01 real `J^T`、matrix-free normalized `J^T J`、pairwise/random 12-D two-pass Lanczos、source artifact
  hash/exact replay、spectral-only metrics/HDF5/one figure、targeted validation和一次正式 run。
- 本轮修改文件：
  `configs/experiments/exp042_TGV_3d_multislice_probe_reconstruction.yaml`、
  `src/tgv_ptycho/recon/exp042.py`、`scripts/run_exp042_probe_reconstruction.py`、
  `tests/test_exp042_probe_reconstruction.py`，以及本文末尾第 21 节。
- 当前有效 config：上述 exp042 YAML，source SHA256
  `CD464BB9EAA36F452FE647CDA7673E4183F80A17B6D692D259B9C2CD9F4FFEEB`；execution mode spectral-only，source run锁为第 20 节
  `..._20260822_140501`，dimension 12、two-pass full reorthogonalization、seed `20260845`、lowest-3 summary。
- 最新有效 run：
  `runs/exp042_TGV_3d_multislice_probe_reconstruction_dev_20260822_144251`，complete/validated；run-config SHA256
  `92F6BCC163A8427889CA10E9ACF9632BA34C13928643DB20BA16B9FF81118FE8`。
- 已通过的 tests/commands：targeted pytest `8 passed in 46.74s`；第二次 scoped Ruff `All checks passed!`；唯一 formal runner exit `0`；
  source三哈希/operator sections/data/scan/raw probes、current config semantic equality、full HDF5 tree/213 numeric datasets/one PNG read-back与目视检查均通过。
- 已确认不需要重跑的 controls：只要 forward/operator/source hashes不变，第 20 节 truth replay、field/shift/q4 adjoint、loss/Jacobian finite
  difference、zero residual、truth fixed point、finite-B boundary、base replay和continuation均不需重跑；本节 exact real-adjoint与diagonal Lanczos
  controls在仅改变 Krylov dimension时也不需机械重跑。exp040 reference controls不需重算。
- 当前未关闭的主要矛盾：最低-weight Ritz component 的 dimension/residual convergence。
- 当前次要矛盾：paired detector residual spatial structure；若 convergence失败则形成独立 eigenproblem/identifiability问题；scan/B information
  redesign应交回 exp041/新任务。fixed-exterior/gauge解释已由本节 correction关闭，negative mismatch继续延后。
- 下一轮推荐最小改动：只在 exp042 config/module/test/runner中增加预登记 bounded dimension-extension comparison，复用同一 source hashes、raw probes、
  operator、start definitions和seed；不运行 reconstruction/continuation，最多一个新 timestamped run，不根据中间 Ritz result继续追加 dimension。
- 下一轮最小读取集合：`AGENTS.md`；本文第 3、6、9、10 节与本第 21 节，尤其失败/优先级/本恢复块；当前 exp042 YAML/module/test/runner；
  latest spectral run config/metrics/HDF5 local spectral subgroup/figure；source run只需由已锁 hashes和 raw probe paths读取，不需重读第 20 节全文。
- 只有 forward/operator/B/scan/exterior-reference/Jacobian-adjoint语义改变、summary与 artifact冲突、出现 regression、latest artifact不完整，或 dimension
  convergence blocker无法由本节定位时，才重新读取 exp040对应章节，并记录扩大范围原因、证据和停止点。
- 明确不应重新扫描：exp040全文/`_old.md`、全部历史 runs、整个 `src`/`tests`、notebooks、reports、data、exp041/exp05x详细内容和无关
  theory notes。

### Git 状态

本轮只产生本地 unstaged/untracked修改；没有执行 `git add`、commit、push、PR、merge或 branch操作。exp042的 config/script/module/test/本文
仍为 untracked，runs被 Git ignore；staged为空。开始时已有的 exp030/exp040文档修改、deleted notebook和其他 untracked用户文件保持原状态，
未覆盖、未删除、未重解释。commit/push/PR状态均未改变。

## 22. 2026-08-22 15:18：Implementation iteration 07 — bounded 12→24 Krylov convergence control

### 本轮目标与明确未做事项

本轮继承第 21 节唯一未关闭的主要矛盾：最低-weight Ritz component 是否随预登记 Krylov dimension 扩大而稳定，并使 residual
相对其自身 Ritz value 收敛。正式结果前固定只做一次 `12→24` dimension extension，完全复用相同 source run、matched operator、
primary/control raw probes、pairwise/random starts、two-pass full reorthogonalization 和 seed `20260845`；只比较 recurrence prefix、
prefix Ritz summaries、最低 Ritz value/residual/weight 与 weighted median，不设置 truth 或 scientific threshold。

本轮没有运行 reconstruction 或 paired continuation，没有修改 forward、known B、scan、sample-B、loss、optimizer、start definition 或
normal-operator normalization；没有追加第三个 Krylov dimension、第二次正式 run、更多 random starts、shift-invert 或无界 spectral sweep；
没有加入 blind B、noise/stage/subpixel/calibration、regularization、full-size、waist inference 或 exp040 reference validation。结果只用于
frozen exp040 scalar working model 下 matched development operator 的局部数值诊断。

### 开始时检查的内容

- 执行 `git -c safe.directory=E:/tgv_ptycho_sim status -sb`；开始时已有 exp030/exp040 文档修改、deleted notebook 和其他
  untracked 用户文件均保持，exp042 文件仍为 untracked，staged 为空。
- 读取 `AGENTS.md`；按恢复协议定向读取本文第 3、6、9、10 节与 authoritative 第 21 节，特别是“失败、限制与未关闭问题”、
  “改动后总体优先级”和“下一轮快速恢复上下文”。
- 代码修改前锁定本文为 `152562` bytes，SHA256
  `3982EBDE152B277F0A86D5BF5401698C40C4CBA49571535663E486656680C5CC`，最后章节号 `21`。
- 定向检查当前 exp042 YAML、`src/tgv_ptycho/recon/exp042.py` 中 local spectral/Lanczos 相关函数、runner 的 spectral-only
  source/reference loader、metrics/HDF5/figure writer 和 artifact validator，以及对应 targeted tests；没有顺序通读大型目录。
- 读取 12-D reference run
  `runs/exp042_TGV_3d_multislice_probe_reconstruction_dev_20260822_144251` 的 config、metrics 与 HDF5 local spectral subgroup；
  reference config/metrics/HDF5 SHA256 分别锁为
  `92F6BCC163A8427889CA10E9ACF9632BA34C13928643DB20BA16B9FF81118FE8`、
  `3D777C2FF7DF1E72AA4EAA26C82AB542E0E856341B7D26901AD7D64DC0DECE0E`、
  `35C8452199C53931513E4501EC78080B7D5EC513962A8E630E17AE8FCA34C47C`。
- source run 仍为第 20/21 节已锁的
  `runs/exp042_TGV_3d_multislice_probe_reconstruction_dev_20260822_140501`；本轮仅通过既有 locked paths/hashes 和 raw probe
  datasets 读取，没有重新扫描其他 runs。
- 最终 artifact audit 为纠正预期文件名，只定向列举了本轮唯一 run 的六个文件，定位实际
  `outputs/exp042_probe_reconstruction.h5` 和 `figures/exp042_local_spectral_diagnostic.png`；读取范围没有扩展到其他 run 或目录。
- 没有重读 exp040、`_old.md`、整个 `src`/`tests`、notebooks、reports、data、exp041/exp05x 或无关 theory notes；没有联网。

### 上一轮意见

第 21 节建议只允许一个预登记、受控的 convergence iteration：同一 source/operator/probes/starts/seed 下把 dimension 从 12 扩至
24，比较 recurrence prefix 和 lowest Ritz residual/value；不运行 reconstruction，不根据中间结果继续加 dimension。若稳定则停止扩张
exp042，若不稳定则提升为新的 eigenproblem/local-identifiability 研究问题。本轮复核后完整采用该意见。

### Change 01

#### 改动前：主要矛盾、次要矛盾与必要性判断

- 继承的主要矛盾仍成立：12-D pairwise/random 最低 Ritz residual 分别约为其 Ritz value 的 `32.6/16.7` 倍，最低谱端没有按
  自身量级收敛。
- 本 Change 的唯一主要矛盾：冻结全部 data/operator/start provenance 后，判断 12-D low-end Ritz evidence 在唯一一次 24-D
  extension 中是否数值稳定。
- 次要矛盾最多三个：证明 12-D recurrence prefix 没有实现漂移；保留 reference/source artifact 的 exact provenance；明确
  single-vector、start-dependent Ritz measure 的解释边界。
- 明确延后 paired detector residual spatial summary、更多 starts/dimensions、block/filtered/shift-invert eigensolver、optimizer/regularizer、
  scan/B redesign、blind reconstruction 和所有 waist/physical claims。
- 该 control 直接检验第 21 节证据中最薄弱的 residual convergence，且只增加一次 bounded matrix-free extension；在主矛盾未关闭前，
  它比继续优化或解释 residual spatial structure 更必要。

#### 技术决策与理由

- YAML 把 `krylov_dimension` 预登记为 `24`，新增 `dimension_convergence_control`，锁定 12-D reference run 的三份 SHA256、
  reference/extended dimensions `12/24`、recurrence prefix `12`、same starts/operator/seed、single extension only，且明确
  `truth_used_by_control=false`、`scientific_thresholds_preregistered=false`。
- 新 `lanczos_dimension_convergence_diagnostic()` 对 pairwise/random 分别核对 diagonal/off-diagonal/terminal-beta recurrence prefix，
  由 prefix tridiagonal 复算 Ritz values/weights，并比较最低 Ritz value、terminal-beta residual、residual/value、lowest/lowest-3
  spectral weights 和 weighted median；不修改已有 `J`、`J^T`、`J^T J / N_detector` 或 optimizer。
- runner 增加 hash-locked 12-D reference loader；先验证 reference config/operator/source probes/HDF5/metrics exact，再运行唯一 24-D
  diagnostic。real intensity-Jacobian adjoint control 未受代码影响，因此不机械重跑，metrics 明确记录
  `intensity_jacobian_real_adjoint_rerun=false` 并继承 reference error。
- HDF5 在已有 local spectral subgroup 下自然增加 `dimension_convergence_control`，不改项目级 writer/schema；原始 24-D
  pairwise/random Ritz arrays 单独保留。figure 从四 panel 扩为 `2×3`，新增 12/24 lowest Ritz value 与 residual/value 对照。
- 预先约定本次结果无论好坏都不追加第三个 dimension 或第二次 run，避免按结果追逐更小 Ritz value。

#### 创建或修改的文件

- 修改 `configs/experiments/exp042_TGV_3d_multislice_probe_reconstruction.yaml`：预登记 24-D extension、12-D reference
  provenance 和 bounded convergence controls；正式 run 前 source YAML SHA256
  `398646992D974C9572FAA36AF5303AECB25FF1AD2767F36587557EDC9F589A14`。
- 修改 `src/tgv_ptycho/recon/exp042.py`：增加可复用 `lanczos_dimension_convergence_diagnostic()`；未改 forward、Jacobian、
  adjoint、normal operator 或 optimizer。
- 修改 `scripts/run_exp042_probe_reconstruction.py`：增加 locked reference loader、12/24 metrics/HDF5 和六-panel figure；保留旧
  baseline/spectral entry behavior。
- 修改 `tests/test_exp042_probe_reconstruction.py`：对角矩阵增加 `4→8` exact recurrence-prefix control；tiny runner 先建立
  12-D reference artifact，再验证 24-D branch、provenance 和 artifact contract。
- 本节是设计文档唯一修改；没有修改 exp040/exp041/exp050、README、roadmap、notebooks、reports、shared forward/optics/shift/IO。

#### 验证命令与结果

```powershell
D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim python -m ruff check src/tgv_ptycho/recon/exp042.py scripts/run_exp042_probe_reconstruction.py tests/test_exp042_probe_reconstruction.py
D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim python -m pytest -q tests/test_exp042_probe_reconstruction.py
D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim python scripts/run_exp042_probe_reconstruction.py --config configs/experiments/exp042_TGV_3d_multislice_probe_reconstruction.yaml
```

- scoped Ruff：`All checks passed!`。
- targeted pytest：`8 passed in 54.82s`；覆盖 diagonal dimension control、tiny reference/extension runner、hash/probe/recurrence
  exactness、HDF5 和 figure contract。
- 没有运行 full pytest：没有修改 shared forward/optics/shift/IO，风险由 exp042 targeted suite覆盖；不扩大到全仓既有问题。
- 唯一正式 runner exit `0`；没有第二个 timestamped run，没有根据结果更改 dimension、seed、start 或 summary。
- run 后使用 `Get-FileHash -Algorithm SHA256` 和只读 `h5py` audit 核对 hashes、embedded config、source/reference probes、
  recurrence prefixes、JSON/HDF5 metrics、dataset finiteness 和 tree。第一次 `Get-FileHash` 猜测了不存在的
  `outputs/results.h5`/`figures/local_spectral_diagnostic.png`，命令报告 not found；随后只列举唯一 run，改用实际文件名并完成核对。
  这是只读审计路径错误，不是 implementation、test、run 或 artifact failure，也没有产生第二个 run。

#### 改动后重新评估

- 本轮主要矛盾“未关闭”，但已证明不是 source、operator、start、seed 或 recurrence 实现漂移：两个 starts 的 12-D diagonal、
  off-diagonal、terminal beta、prefix Ritz values 和 prefix spectral weights 均逐元素 exact。
- pairwise lowest Ritz value 从 `1.8946112671261024e-09` 漂移到 `1.080440342433596e-09`，ratio
  `0.5702701979981858`；residual 从 `6.179217363498601e-08` 降至 `3.1610938643834744e-08`，但
  residual/value 只从 `32.61469764650841` 降至 `29.257458651195776`，ratio `0.897063617400358`，仍远未按自身量级收敛。
- random lowest Ritz value从 `1.2493277757836605e-08` 漂移到 `3.779972884746253e-09`，ratio
  `0.30256054159807705`；residual虽降至 `7.73805391983961e-08`，residual/value却从 `16.673605780767225`
  升至 `20.47118896293104`，ratio `1.2277601637040187`。
- 因此第 21 节“start-dependent low-spectral concentration”定性观察仍可作为 bounded projected evidence，但最低 Ritz value/gain、
  完整 low-singular subspace、rank、null space、resolution 和 detection-limit claim 均不能关闭。
- 按预登记停止规则，不再扩大 dimension。可靠低端 eigensolver 与 local-identifiability 已形成新的研究问题，不应在 exp042 内通过
  无界 Lanczos sweep 或继续 reconstruction 偷换解决。

### Operator-consistency 检查

- 24-D run 从同一 source HDF5 读取 primary/control raw probes；最终独立 audit 确认 current、source 与 12-D reference 的两张 raw
  probes 均逐元素 exact。
- pairwise/random 12-D recurrence diagonal、前 11 个 off-diagonal、reference terminal beta 均在 24-D recurrence 中 exact；由 prefix
  复算的 Ritz values/weights 也 exact，`both_recurrence_prefixes_exact=true`。
- 两条 24-D Lanczos 均无 breakdown、无 negative Ritz value；basis real-orthogonality max error分别
  `3.3306690738754696e-16` 和 `4.440892098500626e-16`。
- real intensity-Jacobian adjoint没有改变且本 run 不重跑；锁定 reference control error
  `1.809942305798957e-15`，metrics 明确记录 `intensity_jacobian_real_adjoint_rerun=false`。normal operator仍为
  `J^T J / N_detector`。
- source operator config sections、`I_stack`、scan positions 和 raw probes 已由 locked source/reference loader exact 核对；forward/operator
  未变，因此第 20/21 节已通过的 truth replay、field/shift/q4 adjoint、loss/Jacobian finite difference、zero residual、truth fixed point、
  finite-B boundary和 continuation controls没有机械重跑，`inherited_forward_controls_rerun=false`。
- HDF5 external `config.yaml` 与 `/entry/config_yaml` 字节解码后 exact；external metrics 的 184 个 leaves 与 `/entry/metrics`
  无缺失、无不一致；整个 HDF5 的 290 个 numeric datasets全部 finite。
- formal control 不使用 truth，记录 `truth_used_by_diagnostic=false`、`truth_used_by_control=false`；determinism 由相同 start/seed 的
  exact recurrence prefix 和 targeted tests覆盖。

### Development run 与 artifacts

本轮唯一新 run：

`runs/exp042_TGV_3d_multislice_probe_reconstruction_dev_20260822_151048`

- `run_state.json` 为 `complete`、`artifacts_validated=true`，runtime `37.990417299999535 s`，figure count `1`。
- config SHA256 `88A4076B69D314E972BF09FB578D17FAC3A94D531EB9BA170AEB707D7C48B8DF`；metadata SHA256
  `C0B03D7764AF916910081B54D5094551200F062DDA25CA9C90FCA8C1667EF959`；metrics SHA256
  `CD16575308C6798639C84BB51A80C85185626EB54ED11CE692E75E639C942645`。
- HDF5 `outputs/exp042_probe_reconstruction.h5` SHA256
  `30D75F03B84DA52E23E0BC0872516A60E190A3D5A9D9E66B0B533DC788825358`；figure
  `figures/exp042_local_spectral_diagnostic.png` SHA256
  `490350528A27506F6B1E4435235D6C90C500025292FF918B71DC229308058054`。
- metadata/metrics 明确 `reconstruction_performed_in_this_run=false`、`paired_continuation_performed_in_this_run=false`、known-B、
  probe-only、`reference_validated=false`、`full_tgv_reference_authorized=false` 和无 scientific pass/fail conclusion。
- HDF5 `/entry` 为 `config_yaml/data/instrument/metadata/metrics/reconstruction/sample`；新增
  `/entry/reconstruction/local_spectral_diagnostic/dimension_convergence_control` 及 mirrored metrics group，pairwise/random 主 Ritz arrays
  shape均为 `(24,)`；raw probes仍单独保留，没有 truth group、空 group、raw overwrite 或项目级 schema change。
- figure 已读取并目视检查：`2×3` panels、12/24 legend、lowest Ritz value与 residual/value panels标签完整，无损坏、遮挡或明显截断；
  PNG 仅用于人工审阅，不作为计算主数据源。

### 当前 metrics

- pairwise 12→24：lowest Ritz value `1.8946112671261024e-09 → 1.080440342433596e-09`；lowest residual
  `6.179217363498601e-08 → 3.1610938643834744e-08`；residual/value
  `32.61469764650841 → 29.257458651195776`。
- pairwise lowest weight `0.9898857696047444 → 0.9801042780120676`；lowest-3 cumulative weight
  `0.9999996071227836 → 0.9999693745997192`。
- random 12→24：lowest Ritz value `1.2493277757836605e-08 → 3.779972884746253e-09`；lowest residual
  `2.08307988243795e-07 → 7.73805391983961e-08`；residual/value
  `16.673605780767225 → 20.47118896293104`。
- random lowest weight `0.9634422551732417 → 0.9329406196472049`；lowest-3 cumulative weight
  `0.9950987925271625 → 0.9915959925915568`。
- pairwise residual/value只改善约 `10.3%` 且仍为 `29.26`；random反而恶化约 `22.8%`。lowest values显著漂移，故不能解释为
  converged smallest eigenvalues；没有事后设置 threshold、Passed 或 Failed。
- 全部指标 truth-free；状态仍为 `Development baseline / No scientific pass-fail conclusion`。

### 失败、限制与未关闭问题

- implementation、targeted tests、Ruff、唯一正式 run 和 artifact内容验证均无失败；最终只读 hash audit 的第一次命令使用了两个错误
  文件名并报告 not found，已在同一唯一 run 内定位实际文件名并完成审计，未隐藏该诊断过程。
- 12→24 extension 没有使最低 Ritz residual 相对其自身 Ritz value 收敛；pairwise仍为 `29.26`，random升至 `20.47`。
- single-vector Lanczos measure高度 start-dependent；最低 weight虽仍很高，也不能替代共同 low-subspace、global spectral density、rank 或
  null-space analysis。
- 本轮只允许一次 dimension extension；不能再追加 dimension/start直到 metric看起来稳定。若需要可信 smallest modes，应改变研究问题并
  预注册适合 low end 的 solver/control，而不是继续当前 sweep。
- 仍只是一个 matched、noiseless、小规模 known-B development case；不能推广为真实 TGV recovery、D_waist、resolution、detection limit
  或 exp040 真实物理准确性。

### 改动后总体优先级

- 下一轮主要矛盾（一个）：已不再是 exp042 reconstruction baseline 内的优化问题，而是如何对当前 matrix-free PSD normal operator 获得
  可靠、可独立 residual 验证的最低谱端。若继续，最值得做的是新开 local-identifiability/eigenproblem 任务，事前选择并锁定一个适合
  smallest modes 的 block/filtered eigensolver及独立 convergence contract；预期作用是区分真实弱方向与有限 Krylov projection artifact。
- 次要工作一：在新任务设计中先用可显式构造的小矩阵和当前 matrix-free action 双重验证 solver residual/orthogonality；必要性是避免把
  eigensolver误差解释为物理不可辨识性。
- 次要工作二：paired detector residual 的 spatial/radial truth-free summary保持 exp042 次要项；只有可靠 low-end solver结论关闭后再考虑，
  作用是解释相同 residual norm 的结构差异。
- 次要工作三：scan/sample-B information redesign需求交回 exp041或新任务；当前 exp042只提供 evidence boundary，不自行修改设计。
- 明确不建议现在做：第三个 Krylov dimension、更多 random starts、无界 Lanczos sweep、继续 reconstruction/continuation、optimizer/
  regularizer tuning、blind B、noise/calibration、waist inference或 full exp040 validation。
- 建议停止本条 exp042 spectral-convergence 分支的扩张；若用户授权研究最低谱端，应作为新的 local-identifiability研究问题建立独立记录。

### 下一轮快速恢复上下文

- 当前 authoritative appended section：本文第 22 节。
- 本轮完成的 Change：Change 01，locked 12-D reference loader、bounded 24-D extension、pairwise/random recurrence-prefix exact
  comparison、convergence metrics/HDF5/six-panel figure、targeted validation和一次正式 run。
- 本轮修改文件：
  `configs/experiments/exp042_TGV_3d_multislice_probe_reconstruction.yaml`、
  `src/tgv_ptycho/recon/exp042.py`、`scripts/run_exp042_probe_reconstruction.py`、
  `tests/test_exp042_probe_reconstruction.py`，以及本文末尾第 22 节。
- 当前有效 config：上述 exp042 YAML，source SHA256
  `398646992D974C9572FAA36AF5303AECB25FF1AD2767F36587557EDC9F589A14`；source run `..._20260822_140501`、
  reference run `..._20260822_144251`、dimensions `12→24`、two-pass full reorthogonalization、seed `20260845`。
- 最新有效 run：
  `runs/exp042_TGV_3d_multislice_probe_reconstruction_dev_20260822_151048`，complete/validated；run-config SHA256
  `88A4076B69D314E972BF09FB578D17FAC3A94D531EB9BA170AEB707D7C48B8DF`。
- 已通过的 tests/commands：targeted pytest `8 passed in 54.82s`；scoped Ruff `All checks passed!`；唯一 formal runner exit `0`；
  source/reference raw probes、12-D recurrence/Ritz prefix、embedded config、184 JSON/HDF5 metrics leaves、290 numeric datasets和 one PNG
  read-back/目视检查均通过。
- 已确认不需要重跑的 controls：只要 forward/operator/source/reference hashes不变，第 20 节 truth replay、field/shift/q4 adjoint、loss/
  Jacobian finite difference、zero residual、truth fixed point、finite-B boundary、base replay和continuation均不需重跑；第 21 节 real-adjoint、
  diagonal Lanczos和12-D spectral run也不需机械重跑。exp040 reference controls不需重算。
- 当前未关闭的主要矛盾：可靠 lowest-spectrum estimation/local-identifiability；它已超出本条 exp042 bounded Lanczos iteration，不能靠继续
  增加 dimension关闭。
- 当前次要矛盾：显式小矩阵/当前 action 的新 solver双重验证；paired residual spatial structure；scan/B redesign交回 exp041/新任务。
- 下一轮推荐的最小改动：不要继续修改当前 exp042 config/run；若获授权，先建立独立 local-identifiability/eigenproblem 设计与小矩阵
  solver control，再决定是否读取当前 24-D run作为输入，不运行 reconstruction。
- 下一轮最小读取集合：`AGENTS.md`；本文第 3、6、9、10 节与本第 22 节，尤其本恢复块；当前 exp042 module中 `J`/`J^T`/normal action
  和 Lanczos API；最新 24-D run 的 config/metrics/HDF5 local spectral subgroup。新任务设计前无需读取其他历史 iteration全文。
- 只有 forward/operator/B/scan/exterior-reference/Jacobian-adjoint语义改变、summary与 artifact冲突、出现 regression、latest artifact不完整，
  或新 solver blocker需要确认 physical evidence boundary时，才重新读取 exp040对应章节，并记录扩大原因、证据与停止点。
- 明确不应重新扫描：exp040全文/`_old.md`、全部历史 runs、整个 `src`/`tests`、notebooks、reports、data、exp041/exp05x详细内容和
  无关 theory notes。

### Append-only verification

- append 前旧文件为 `152562` bytes，SHA256
  `3982EBDE152B277F0A86D5BF5401698C40C4CBA49571535663E486656680C5CC`，最后章节号 `21`。
- append 后独立按原长度读取 prefix；预期并核对 `PrefixMatches=true`、prefix SHA256仍为上述值，heading顺序为 `21→22`。本节只在
  EOF 增加，没有修改、删除、重排或润色旧 section。若校验与此不符，必须在新 EOF correction note中记录，不能回改旧字节。

### Git 状态

本轮只产生本地 unstaged/untracked修改；没有执行 `git add`、commit、push、PR、merge或 branch操作。exp042的 config/script/module/test/本文
仍为 untracked，runs被 Git ignore；staged为空。开始时已有的 exp030/exp040文档修改、deleted notebook和其他 untracked用户文件保持原状态，
未覆盖、未删除、未重解释。commit/push/PR状态均未改变。

## 23. 2026-08-22 15:37：路线调整记录 — 暂挂最低谱端收敛问题

本节不是新的 Implementation iteration，没有修改代码/config/test，没有运行实验，也不替代第 22 节的数值证据。它只记录本次讨论形成的
近期路线调整；下一次真正实施 Change 时仍须恢复此前约定的完整 iteration 结构、验证、timestamped run、artifact audit 和 append-only记录。

### 暂挂事项与理由

- 第 22 节遗留的“最低谱端是否精确收敛”暂时挂起，状态是**未关闭但不再作为近期主线**；12→24 结果和限制继续有效，不回改、不否定。
- 该问题用于判断当前 matched operator 是否存在数据近乎不可见的局部 probe 方向，并非直接衡量不同 reconstruction 模块作用的消融指标。
- 已有 bounded diagnostic 足以提示潜在 local-identifiability 风险；在主要消融/对照尚未开展时，继续追求 smallest Ritz value 的高精度
  缺少已证明的必要性，可能把精力过早投入独立的高阶 eigenproblem。
- 暂不追加第三个 Krylov dimension、更多 starts、无界 Lanczos sweep 或新 low-end eigensolver。只有直接对照再次出现无法由 residual、
  probe error、初始化或收敛轨迹解释的稳定差异，或者 local-identifiability 本身被明确设为新研究目标时，才参考第 22 节重新开启。

### 新的近期建议路线

- 下一轮优先做低成本、预注册、one-factor-at-a-time 的 known-B probe-only 直接对照。第一项建议是在同一 matched data/operator/B/scan、
  相同 optimizer 和 equal compute budget 下，只改变确定性初始化，检查初始化敏感性；不同时改变 detector、scan、sample-B 或 optimizer。
- 各分支统一保存 detector residual/loss convergence curve、raw reconstructed probe、分支间 probe distance；probe error和 truth-aligned error
  只标记为 `simulation evaluation only`，不得进入 optimizer、步长选择或停止规则。
- 先用这些直接指标判断差异主要来自初始化/优化轨迹，还是在 equal-budget 后仍稳定保留。只有证据需要时，再设计下一项单因素消融；
  不在一个 run 中同时堆叠多个模块变化，也不把 diagnostic branch 称为科学通过结论。
- 当前状态仍为 `Development baseline / No scientific pass-fail conclusion`；blind B、noise/calibration、scan/sample-B redesign、waist inference
  和 exp040 真实物理准确性继续不在近期 exp042 baseline 范围内。

### 下一轮恢复提示

- 路线决定以本第 23 节为最新入口；数值 evidence、latest valid run和未关闭 spectral 限制仍以第 22 节为准。
- 下一轮最小读取集合：`AGENTS.md`、本文 fixed design boundary、第 22 节的 run/evidence boundary、本第 23 节，以及当前 exp042
  YAML/module/runner/test；默认不重读 exp040、全部历史 runs、整个 `src`/`tests`、notebooks 或 reports。
- 下一轮开始前应先把具体初始化分支、seed、equal budget、公共 metrics 和停止规则写入 config/iteration 的改动前判断；看到结果后不追加
  分支或更换指标。
- 本节 append 前文档为 `174064` bytes，SHA256
  `945427079FD82E231313B545D911D7702F14C703A93DA2671D92B026DF26C280`，最后章节号 `22`；append 后须验证原前缀不变且章节顺序
  `22→23`。

## 24. 2026-08-22 16:07：Implementation iteration 08 — equal-budget initialization ablation

### 本轮目标与明确未做事项

本轮按第 23 节路线调整启动一次正式 direct ablation：在同一 deterministic matched data/operator/known-B/scan、同一 GN-scaled Armijo
optimizer和相同 60-iteration maximum budget下，只改变 probe initialization。正式结果前固定四个分支：homogeneous reference，以及
相同 `relative_l2=0.05`、不同 seeds `20260843/20260846/20260847` 的 deterministic zero-mean complex perturbations；统一保存 loss、
detector residual、raw reconstruction、truth-free pairwise probe/prediction distances和 simulation-only probe-error curves。

本轮没有改变 forward、detector q4、finite-B、scan、sample-B、optimizer、iteration budget或 stop rule；没有运行 continuation、conditioning
或 local spectral diagnostic；没有根据 truth 或中间结果选择 seed、分支、步长、停止或指标；没有 blind B、noise/calibration、scan/B redesign、
waist inference、full-size或 exp040 reference validation。第 22 节最低谱端问题继续“未关闭但暂挂”。

### 开始时检查的内容

- 执行 `git -c safe.directory=E:/tgv_ptycho_sim status -sb`；开始时已有 exp030/exp040 文档修改、deleted notebook 和其他
  untracked 用户文件均保持，exp042 文件仍为 untracked，staged 为空。
- 读取 `AGENTS.md`；按恢复协议定向读取本文 fixed design boundary 第 3、6、9、10 节、第 22 节 evidence boundary和第 23 节路线决定，
  重点复核暂挂项、新建议路线和下一轮恢复提示。
- 修改前锁定本文为 `177465` bytes，SHA256
  `8BEBC3F3F0C9182A5297FA357363E3664D43CB7CBF6EC7B0E8640E9758E1C9E3`，最后章节号 `23`。
- 定向检查当前 exp042 YAML；module中的 config validation、deterministic initialization、known-B optimizer、simulation evaluation和现有
  two-branch stability diagnostic；runner的 dispatch、metrics/HDF5/figure/artifact writer；对应 targeted tests。没有顺序通读整个源码目录。
- 没有复用或修改历史 run。正式 run使用当前公共 `build_matched_development_case()` 确定性再生同一
  `canonical_geometry_coarse_matched_dev` fixture，branch仍为 `R8 unified q8 finite-B open q4 scalar working model`，grid
  `96×96` native、`256×256` open、`dx=5e-7 m`、25 scans、q4、seed family在 config中预登记。
- 没有扩大到 exp040、`_old.md`、全部 runs、整个 `src`/`tests`、notebooks、reports、data、exp041/exp05x或无关 theory notes；没有联网。

### 上一轮意见

第 23 节将最低谱端精确收敛暂挂，建议优先做 one-factor-at-a-time known-B probe-only direct control：固定 matched data/operator/B/scan、
optimizer和equal budget，只改变确定性初始化，统一观察 detector residual、probe error、分支距离和收敛曲线，truth只能 post-hoc evaluation。
本轮复核后完整采用；没有参考第 22 节 Ritz values选择任何 branch或参数。

### Change 01

#### 改动前：主要矛盾、次要矛盾与必要性判断

- 继承的路线判断仍成立：基础 reconstruction可运行，但现有 homogeneous+单一 5% perturbation记录不是一个预注册、多 seed、统一
  artifact的直接消融，无法判断 observed difference是否只属于一个 perturbation direction。
- 本 Change 的唯一主要矛盾：相同 matched measurement和equal optimizer budget下，detector fit与 recovered probe对 deterministic
  initialization方向是否敏感。
- 次要矛盾最多三个：保证每个 perturbation精确 5%且确定性；保证各分支 optimizer/settings/curve length公平；严格隔离 truth-free
  diagnostics和 `simulation_evaluation_only` probe error。
- 明确延后 perturbation-magnitude sweep、额外 seed sweep、iteration-budget ablation、operator module ablation、lowest-spectrum solver、
  scan/B redesign、blind reconstruction和所有 physical/waist claims。
- 该 Change直接执行第 23 节提出的最小对照，比继续优化最低谱端或同时改变多个模块更能回答当前 reconstruction baseline的实际稳定性。

#### 技术决策与理由

- config新增 `execution.mode=initialization_ablation_matched` 和 `verification.initialization_ablation`；固定 homogeneous 加三个同幅度 seeds、
  reference branch、60-step equal budget、共同 metrics、truth禁用项和无 scientific threshold。旧 stability continuation、pairwise conditioning与
  spectral controls均设 `enabled=false`，其实现和历史 artifact保持不变。
- `validate_exp042_config()` 增加当前 mode的独立严格验证：branch names/seeds唯一、homogeneous reference唯一、perturbation family、共同5%幅度、
  iteration budget与 reconstruction一致、GN-scaled algorithm/curvature cadence不变、truth/threshold flags正确；旧 execution mode validation继续保留。
- 新 `truth_free_initialization_ablation_diagnostic()` 不接收 truth，计算initial/final raw与 global-phase-aligned symmetric pairwise probe-distance
  matrices、final prediction-distance matrix、reference-branch vectors、maximum pairwise distances、budget/algorithm/curve/monotonicity controls。
  global-phase alignment仅是第 21 节已声明的 diagnostic convention，不解释为当前 fixed-exterior operator的自由 gauge。
- runner新增独立 ablation entry：在同一 in-memory `I_stack`上顺序运行四支；每个 perturbation在 optimizer前重复生成并要求 bitwise exact；
  raw `P_B_init/P_B_rec`与全部 measurement curves按branch保存，truth-aligned field/error只进入各branch的
  `simulation_evaluation_only` subgroup和post-hoc metrics。
- 唯一 figure预登记为 `2×3`：loss、detector residual、raw/aligned simulation-only probe errors以及final aligned-probe/prediction pairwise
  matrices。没有按正式结果增加panel或branch。

#### 创建或修改的文件

- 修改 `configs/experiments/exp042_TGV_3d_multislice_probe_reconstruction.yaml`：当前 source YAML SHA256
  `7918A566C62217E7129B59AC2BFA4721F460CB5A1660EE714C714E8E2D294E53`。
- 修改 `src/tgv_ptycho/recon/exp042.py`：新增 mode validation和truth-free multi-branch diagnostic；没有修改 matched forward、loss、gradient、
  adjoint、optimizer数值实现或 shared optics/shift/IO。
- 修改 `scripts/run_exp042_probe_reconstruction.py`：新增 ablation metadata/runner/HDF5/figure/artifact validation和dispatch；保留历史 full-baseline/
  spectral runner paths。
- 修改 `tests/test_exp042_probe_reconstruction.py`：新增 diagnostic assertions和tiny four-branch runner contract；保留 spectral runner regression。
- 本节是文档唯一修改；没有修改 exp040/exp041/exp050、README、roadmap、notebooks、reports或共享 IO schema。

#### 验证命令与结果

```powershell
D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim python -m ruff check src/tgv_ptycho/recon/exp042.py scripts/run_exp042_probe_reconstruction.py tests/test_exp042_probe_reconstruction.py
D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim python -m pytest -q tests/test_exp042_probe_reconstruction.py
D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim python scripts/run_exp042_probe_reconstruction.py --config configs/experiments/exp042_TGV_3d_multislice_probe_reconstruction.yaml
```

- 第一次 scoped Ruff即 `All checks passed!`。
- targeted pytest首轮为 `1 failed, 8 passed in 54.44s`：新 runner truth replay误写不存在的 `operator.predict()`，而已有公共API是
  `predict_stack()`；此前八项和四分支tiny reconstruction均通过。只替换该API名称后，第二次 targeted pytest为
  `9 passed in 53.79s`，随后 scoped Ruff复验 `All checks passed!`。没有隐藏或删除这次失败证据。
- 实现补丁第一次因选取的函数结尾锚点与实际代码不一致而未应用、文件无部分变化；改用唯一 `validate_exp042_config`锚点后成功。这是
  edit定位失败，不是数值实现或实验失败。
- 没有运行 full pytest：没有修改 shared forward/optics/shift/IO，当前 exp042公共接口、旧 spectral runner和新 branch artifact风险均由
  targeted suite覆盖；不扩大到全仓既有问题。
- 唯一正式 runner exit `0`；没有第二个正式 run，没有根据结果调整branch、seed、budget、metric或figure。

#### 改动后重新评估

- 本轮主要矛盾在当前 matched development case内“已关闭并得到有边界的答案”：四支 detector loss/residual curves几乎重合，final
  detector residual范围 `0.00885487–0.00890643`、max/min ratio `1.005822`；但final probes相对 homogeneous仍差约
  `0.05607–0.05617`，三个perturbed probes彼此最高 phase-aligned distance `0.07984405`。因此 detector fit对这组初始化较稳，raw
  recovered probe并非初始化不敏感。
- final detector predictions彼此最多只差 `0.00102743`，明显小于probe pairwise distance；raw和phase-aligned probe matrices近乎相同，
  说明difference不是单一 global-phase convention造成。
- simulation evaluation only显示 homogeneous final aligned probe error `0.13937197`，三个perturbed为
  `0.14925692/0.14921981/0.15031856`，最大约高 `7.85%`。该truth-aided结果只能描述simulation，不得据此为真实数据选择初始化。
- 新暴露的下一主要矛盾是5%幅度是否过大、以及sensitivity是否随同一perturbation direction的幅度连续缩放；这是比增加更多随机seed或
  恢复谱端分析更直接的one-factor question。60-step curves仍下降是次要confound，但第 20 节对 homogeneous/seed-20260843已有额外equal-budget
  continuation evidence，不应无条件重跑。

### Operator-consistency 检查

- formal run重新执行了当前结论直接依赖的 truth-pair intensity replay：`truth_replay_exact=true`、relative L2 `0.0`。
- 三个5% perturbations在正式 optimizer前各重复生成一次并bitwise exact；HDF5独立核对相对 homogeneous one-sided L2分别为
  `0.049999999999999684/0.04999999999999968/0.049999999999999684`。
- 四支使用同一 `I_stack`、operator instance、known B、optimizer mapping和60-step maximum budget；全部完成60步、stopping reason均
  `iteration_budget`、curve length均61、loss全部nonincreasing、detector residual全部下降、total backtracking均0。
- forward/operator实现未变，因此第 20–22 节已通过的 field/shift/q4 adjoint、loss/Jacobian finite difference、zero residual、truth fixed point、
  finite-B boundary与real-Jacobian adjoint没有机械重跑；metrics明确
  `inherited_forward_adjoint_gradient_controls_rerun=false`。
- source YAML、run `config.yaml`和HDF5 `/entry/config_yaml`语义/embedded text exact；external metrics的163个leaves与
  `/entry/metrics`为0 missing、0 mismatch；HDF5的289个 numeric datasets全部finite。
- truth没有进入 initialization、optimizer、branch selection或stopping；raw reconstruction单独保留，aligned fields只在
  `simulation_evaluation_only`；sample B没有更新，spectral/continuation没有运行。

### Development run 与 artifacts

本轮唯一新 run：

`runs/exp042_TGV_3d_multislice_probe_reconstruction_dev_20260822_155712`

- `run_state.json`为 `complete`、`artifacts_validated=true`，runtime `158.80298949999997 s`，figure count `1`。
- run config SHA256 `C62D3353DD33CF8FCC92D4466AA1FD3A66441974CB45B54D550E21ECD158A389`；metadata SHA256
  `71F0D87CEAE2115175B3F2292FD8C4E0565944165745827E1D092FAF574578E0`；metrics SHA256
  `5932B69E4C6491C0087EBA26479DA884859AADD5EA4F72A27BC41624233EE81B`。
- HDF5 `outputs/exp042_probe_reconstruction.h5` SHA256
  `10B001E2FDECA0038339DF23E4D3C7AEA0EBA3EE0D958D6654BD2198E054CDE5`；figure
  `figures/exp042_initialization_ablation.png` SHA256
  `B93CB577BBAF9CBA7BDDC56FDD4E175EF85BC92EAD0E4EAD83145AE7FAE9B47D`。
- HDF5 `/entry`为 `config_yaml/data/instrument/metadata/metrics/reconstruction/sample/truth`；核心新增路径为
  `/entry/reconstruction/initialization_ablation/branches/<branch>/{P_B_init,P_B_rec,loss_curve,detector_relative_residual_curve,...}`、
  各branch的 `simulation_evaluation_only`、以及 `truth_free_pairwise_diagnostic`的initial/final `4×4` matrices。没有项目级schema change、
  空group或raw overwrite。
- 每支 `P_B_init/P_B_rec` shape `(96,96)`、loss/residual/probe-error curves shape `(61,)`；完整 reconstruction subgroup tree已打印检查。
- figure可读取并目视检查：六panel、四支legend、simulation-only labels、pairwise heatmaps和colorbars完整；底部长branch labels可读，无损坏、
  遮挡或明显截断。PNG不作为计算主数据源。
- 最终只读audit中，直接调用env `python.exe`执行含 `numpy.linalg`脚本曾无traceback退出1；改用完整`conda run`单行审计后相同array检查通过。
  随后的multiline `conda run python -c`又触发Conda已知“不支持含换行argument”AssertionError；最终使用不调用`numpy.linalg`的env解释器脚本完成
  JSON/HDF5递归审计。两次均为只读审计环境/命令问题，未修改artifact、未重跑实验，最终内容审计通过。

### 当前 metrics

- homogeneous：final loss `1.9277964661849684e-05`，final detector residual `0.008857188126643553`，simulation-only raw/aligned
  probe error `0.13937201182884235/0.13937197214033825`。
- seed `20260843`：final loss `1.93299503664604e-05`，residual `0.008869122404370298`，raw/aligned error
  `0.14925691849563977/0.1492569162151173`。
- seed `20260846`：final loss `1.9267890383450204e-05`，residual `0.008854873529613713`，raw/aligned error
  `0.1492199334424988/0.14921980979662705`。
- seed `20260847`：final loss `1.9492900184821872e-05`，residual `0.008906426920264212`，raw/aligned error
  `0.15031858941997456/0.1503185646727466`。
- 各支final/initial loss ratio约 `0.000616–0.000623`，final/initial residual ratio约 `0.02483–0.02495`；所有曲线合理下降。
- perturbation final raw distance to homogeneous为 `0.05615363/0.05606646/0.05617347`；maximum all-pair raw/aligned distances
  `0.0798441292/0.0798440499`，maximum prediction distance `0.00102743119`。
- 本轮没有预注册scientific threshold，不给 Passed/Failed；状态仍为
  `Development baseline / No scientific pass-fail conclusion`。

### 失败、限制与未关闭问题

- 实现阶段有一次未应用的patch锚点失败、一次targeted pytest API-name失败；均在正式run前以最小修改修复并复验。正式run和artifact内容验证
  没有失败。最终audit另有两次只读Windows/Conda命令环境失败，最终替代审计通过；没有删除或覆盖失败证据。
- 只测试一个固定5%幅度和三个perturbation directions；不能由此建立初始化扰动幅度的响应曲线、一般seed distribution或global basin结构。
- 60步结束时loss/residual/probe-error仍在下降；当前差异可能同时包含有限iteration budget和弱ly constrained direction effects。已有第20节
  single-seed continuation证据但不能自动推广到三个新seeds。
- homogeneous在simulation truth error上最好，但truth不能用于真实reconstruction initialization selection；这里没有建立production初始化规则。
- 仍是一个matched、noiseless、小规模case；不能推广为blind recovery、noise robustness、真实TGV、D_waist、resolution/detection limit或
  exp040物理准确性。最低谱端问题保持暂挂，不因本轮现象自动恢复为主线。

### 改动后总体优先级

- 下一轮主要矛盾（一个）：当前init sensitivity是否随同一deterministic perturbation direction的幅度连续缩放，还是5%已经跨入不同优化轨迹。
  最值得做的是固定一个seed/direction，预注册少量幅度级别（具体levels在看到结果前写入下一轮config），保持operator/optimizer/budget不变；
  作用是把“direction variation”和“perturbation magnitude”拆开。
- 次要工作一：在下一次自然产生probe history的run中保存truth-free pairwise checkpoint curves，而不只initial/final matrices；作用是判断branch
  separation发生在早期还是持续积累，避免为本run缺失该字段机械重跑。
- 次要工作二：iteration-budget仍是潜在confound；先复用第20节 homogeneous/seed-20260843的120-step evidence，只有magnitude control仍无法解释时
  再预注册统一budget ablation，不根据当前曲线临时延长。
- 次要工作三：初始化因素关闭后，再选择一个明确的operator/reconstruction module做matched或显式mismatch单因素消融；必须在设计时区分
  “模块贡献”和“模型错配”，不在同一run混合多个变化。
- 明确延后：更多random seeds、第三个Krylov dimension、low-end eigensolver、optimizer sweep、scan/B redesign、blind B、noise/calibration、
  waist inference和full exp040 validation。
- 建议继续exp042 direct-ablation主线；当前问题仍是baseline内部可验证的初始化尺度问题，尚未形成必须新开实验的研究问题。

### 下一轮快速恢复上下文

- 当前authoritative appended section：本文第24节；第23节提供路线决定，第22节spectral问题继续暂挂。
- 本轮完成的Change：Change 01，四支equal-budget initialization ablation mode、strict config validation、truth-free pairwise diagnostic、
  branch metrics/HDF5/six-panel figure、targeted regression和一次正式run。
- 本轮修改文件：
  `configs/experiments/exp042_TGV_3d_multislice_probe_reconstruction.yaml`、
  `src/tgv_ptycho/recon/exp042.py`、`scripts/run_exp042_probe_reconstruction.py`、
  `tests/test_exp042_probe_reconstruction.py`，以及本文末尾第24节。
- 当前有效config：上述exp042 YAML，source SHA256
  `7918A566C62217E7129B59AC2BFA4721F460CB5A1660EE714C714E8E2D294E53`；mode `initialization_ablation_matched`，branches
  homogeneous+seeds `20260843/46/47`，common perturbation `0.05`，budget `60`。
- 最新有效run：
  `runs/exp042_TGV_3d_multislice_probe_reconstruction_dev_20260822_155712`，complete/validated；run-config SHA256
  `C62D3353DD33CF8FCC92D4466AA1FD3A66441974CB45B54D550E21ECD158A389`。
- 已通过commands：修复后targeted pytest `9 passed in 53.79s`；两次scoped Ruff均 `All checks passed!`；唯一formal runner exit `0`；
  truth replay、deterministic 5% inits、equal budgets、raw branch HDF5、embedded config、163 metrics leaves、289 numeric datasets和figure read-back/
  目视检查均通过。
- 已确认不需要重跑的controls：只要forward/operator/config相应sections不变，第20–22节的full adjoint/gradient/fixed-point/finite-B/q4/
  continuation/spectral controls均不需机械重跑；第24节本run也不为补pairwise checkpoint fields重跑。exp040 reference controls不需重算。
- 当前未关闭的主要矛盾：固定direction下perturbation magnitude scaling；当前次要矛盾为truth-free separation trajectory、iteration-budget
  confound和未来单模块消融设计。
- 下一轮推荐最小改动：只把当前ablation branch axis从“同幅度不同seed”改为“同seed不同幅度”，同时让diagnostic保存checkpoint pairwise curves；
  不改forward/B/scan/optimizer，不运行spectral，最多一个新timestamped run。
- 下一轮最小读取集合：`AGENTS.md`；本文第3、6、9、10、23和本第24节；当前YAML；module的initialization generator/ablation diagnostic；
  runner的ablation entry/HDF5/figure validator；对应targeted tests；latest run只读metrics/HDF5 initialization-ablation subgroup和figure。
- 只有forward/operator/B/scan/exterior-reference语义改变、summary与artifact冲突、出现regression、latest artifact不完整，或direct ablation无法解释
  稳定差异时，才重新读取exp040对应章节；如恢复local-identifiability研究，先参考第22节而非重扫exp040。
- 明确不应重新扫描：exp040全文/`_old.md`、全部历史runs、整个`src`/`tests`、notebooks、reports、data、exp041/exp05x详细内容和无关
  theory notes。

### Append-only verification

- append前旧文件为 `177465` bytes，SHA256
  `8BEBC3F3F0C9182A5297FA357363E3664D43CB7CBF6EC7B0E8640E9758E1C9E3`，最后章节号 `23`。
- append后独立按原长度读取prefix；预期并核对 `PrefixMatches=true`、prefix SHA256仍为上述值，heading顺序为 `23→24`。本节只在EOF
  增加；若校验不符，只能在新EOF correction note记录，不能回改旧section。

### Git 状态

本轮只产生本地unstaged/untracked修改；没有执行 `git add`、commit、push、PR、merge或branch操作。exp042的config/script/module/test/本文
仍为untracked，runs被Git ignore；staged为空。开始时已有的exp030/exp040文档修改、deleted notebook和其他untracked用户文件保持原状态，
未覆盖、未删除、未重解释。commit/push/PR状态均未改变。

## 25. 2026-08-22 16:36：Implementation iteration 09 — fixed-direction initialization-magnitude ablation

### 本轮目标与明确未做事项

本轮按第24节的唯一主要建议，把上一轮“同幅度、不同seed”的初始化方向消融收窄为“同seed/同方向、不同幅度”的确定性单因素消融。
预注册且在看到本轮reconstruction结果前固定的幅度为 `0% / 1% / 2.5% / 5%`，固定perturbation seed为 `20260843`；四支使用
完全相同的matched q4/finite-B/open数据、known B、scan、GN-scaled Armijo和60-step budget，每5步保存一次truth-free pairwise
checkpoint。目标是判断第24节观察到的最终probe差异是否随同一扰动方向的幅度连续缩放，而不是继续追逐最低谱端。

本轮明确未做blind `P_B + B`、`D_waist`拟合或腰径估计、sample-B/scan redesign、noise/stage/subpixel/calibration、更多random
directions、iteration-budget continuation、conditioning或local spectral diagnostic；没有修改exp040 frozen evidence，没有运行exp040 Helmholtz/reference
pipeline，也没有把matched inverse development behavior解释为真实三维电磁准确性。状态保持
`Development baseline / No scientific pass-fail conclusion`。

### 开始时检查的内容

- 开始时执行 `git -c safe.directory=E:/tgv_ptycho_sim status -sb`；staged为空。已有exp030/exp040文档修改、deleted notebook和其他
  untracked用户文件均被识别并保留。
- 按恢复协议读取 `AGENTS.md`，并恢复本文第3、6、9、10、23节以及第24节末尾的“失败、限制与未关闭问题”“改动后总体优先级”
  和“下一轮快速恢复上下文”。本轮没有重读exp040、没有读取exp040 old文档、没有扫描全部runs、`src`、`tests`、notebooks、reports
  或data。
- 定向检查当前YAML的execution/reconstruction/ablation/artifact字段，`src/tgv_ptycho/recon/exp042.py` 中
  `validate_exp042_config()`、初始化生成和 `truth_free_initialization_ablation_diagnostic()`，runner中的
  `_run_initialization_ablation()`、HDF5/metrics/figure validator与mode dispatch，以及exp042 targeted tests。实现后仅以一次定向 `rg`
  核对上述符号和checkpoint字段位置；这属于恢复块列出的范围，不是额外无目标扫描。
- 复用第24节已经验证的同一deterministic matched development generator/operator chain；没有拼接不同历史artifact，也没有把历史run作为
  本轮数据输入。正式run重新确定性生成同一组 `P_B_true`、`B_true`、`I_stack`和scan positions。
- 修改代码前锁定本文为 `198567` bytes，SHA256
  `CA6EE52FD0B267D95FCA6A660680676B41020B6AD21887F0D50BC5325E2564C3`，最后章节号为 `24`。

### 上一轮意见

- 第24节未关闭的主要矛盾是：在固定perturbation direction下，reconstruction差异是否随初始化幅度连续缩放；上一轮建议只改变
  branch axis为“同seed不同幅度”，增加checkpoint pairwise curves，不改forward/B/scan/optimizer且最多运行一次正式run。
- 次要矛盾是truth-free separation trajectory、60-step iteration-budget confound和下一项单模块消融的设计；更多seeds、low-end
  eigensolver、optimizer sweep、blind B、noise/calibration、waist inference和full exp040 validation明确延后。
- 复核后上一轮意见仍成立：初始化幅度是进入其他module ablation前最小、最可归因的未关闭变量。本轮完整采用该建议，没有因运行中指标
  临时改变幅度、seed、budget、metric或停止时机。

### Change 01

#### 改动前：主要矛盾、次要矛盾与必要性判断

- 继承的主要矛盾：第24节的5% perturbation不同方向在60步后给出约5.6%--8.0%的probe差异，但尚不知道差异是方向偶然性，还是在
  固定方向上存在连续幅度响应。
- 本Change唯一主要矛盾：固定方向时，初始化扰动幅度是否形成可复现、可审计的probe/prediction separation scaling。
- 次要矛盾最多三个：分离主要在早期出现还是持续增长；相近detector residual能否掩盖probe差异；60步budget是否仍限制最终收敛解释。
- 明确延后：最低谱端精确收敛、更多方向/seed、其他operator module、blind reconstruction与物理robustness。
- 必要性：如果同方向幅度响应并不连续，继续把上一轮差异归因于初始化敏感性会缺少最基本的dose-response control；先关闭该问题比直接
  增加新的模块或复杂优化器更必要。

#### 技术决策与理由

- 新execution mode为 `initialization_magnitude_ablation_matched`，family为 `fixed_direction_magnitude_sweep`；branch固定为
  `homogeneous_reference=0`、`perturb_1pct=0.01`、`perturb_2p5pct=0.025`、`perturb_5pct=0.05`。
- 三个非零branch共用seed `20260843`，以相同归一化zero-mean complex direction乘不同幅度。optimizer开始前计算归一方向pairwise L2
  error，并要求不超过预注册容差 `1e-12`。
- forward truth/data、known B、scan、q4 detector、GN-scaled Armijo和60步budget完全固定；truth不参与branch selection、初始化、optimizer、
  stopping或truth-free diagnostic。
- 每5步保存raw probe、global-phase-aligned diagnostic probe和detector prediction的pairwise/reference distance；aligned量仅是诊断约定，
  不改变raw reconstruction也不进入optimizer。
- HDF5增加自然产生的magnitude-ablation `design`和checkpoint datasets；没有改变项目级 `/entry` 并列schema。保留旧direction-ablation和
  spectral mode的validation/dispatch兼容性。
- 预注册figure固定为2x4共8 panels：loss、detector residual、raw/aligned simulation-only probe error、三类truth-free checkpoint
  reference curves和final aligned probe-distance matrix。

#### 创建或修改的文件

- `configs/experiments/exp042_TGV_3d_multislice_probe_reconstruction.yaml`：切换到固定方向幅度sweep，固定四级幅度、seed、容差、
  checkpoint interval和8-panel artifact contract。当前source文件SHA256为
  `FD16A6234114C37531493108CDA95ED581BCB51D71D761E928B9B290E9A048D3`。
- `src/tgv_ptycho/recon/exp042.py`：扩展config validation与truth-free ablation diagnostic，生成方向一致性、initial magnitude及三类
  checkpoint distance matrices；SHA256为 `AF5EB5BC435DCD2003AE1C844D053209103CEB1C1EA5945AF2CF36B8823616FD`。
- `scripts/run_exp042_probe_reconstruction.py`：增加magnitude mode的metadata/run role、同方向构造与强制检查、HDF5 design/checkpoints、
  8-panel figure和artifact final-checkpoint exact validation；SHA256为
  `C5EE3CEF120861CF274300F3BF16FF94DC19D1FC5514634E6B325BEB21B8A5D1`。
- `tests/test_exp042_probe_reconstruction.py`：覆盖新config、checkpoint shapes/final equality、tiny runner的新branch/design/HDF5 contract，
  并保留spectral runner regression；SHA256为 `410AC96EB010D405DB08561E7EFF4F399B2706272D998A533BA8DCFFC0079C74`。
- 本文仅在EOF追加本第25节；没有修改旧第1--24节或顶部状态。

#### 验证命令与结果

- 首次scoped Ruff：
  `D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim python -m ruff check src/tgv_ptycho/recon/exp042.py scripts/run_exp042_probe_reconstruction.py tests/test_exp042_probe_reconstruction.py`
  报告1个本轮新增 `E501`，原因为89字符checkpoint key；仅拆分字符串字面量后修复，没有改变字段值或数值行为。
- Targeted pytest：
  `D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim python -m pytest -q tests/test_exp042_probe_reconstruction.py`
  结果 `9 passed in 50.79s`。
- 修复后同一scoped Ruff结果 `All checks passed!`。没有运行full pytest：本轮未修改shared forward/optics/shift/IO，且新mode、旧mode兼容与
  runner contract均由targeted tests覆盖；没有理由扩大到全仓既有问题。
- 唯一正式命令：
  `D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim python scripts/run_exp042_probe_reconstruction.py --config configs/experiments/exp042_TGV_3d_multislice_probe_reconstruction.yaml`
  exit `0`，runtime `161.52151960000083 s`，runner报告 `artifacts_validated: true`。
- 独立artifact审计首先用一条PowerShell精简指标表命令时发生parser error（`foreach`输出未先包成数组）；这是只读审计命令错误，不是run、
  code或artifact失败。修正命令后继续，未重跑正式实验。
- 独立审计结果：267个数值datasets全部finite；selected JSON/HDF5 checkpoint/final metrics逐元素exact；三种checkpoint matrices均
  `(13,4,4)`且最后一帧与final matrices逐元素exact；figure可读取并完成目视检查，无空panel、明显截断或不可读轴标签。

#### 改动后重新评估

- 本轮主要矛盾在预注册的单一development direction与0%--5%范围内已关闭。最终raw probe distance-to-reference为
  `0 / 0.011239426962035 / 0.0280937419068912 / 0.0561536311201187`；分别除以1%、2.5%、5%后约为
  `1.12394 / 1.12375 / 1.12307`，显示近似比例缩放。final prediction distance同样为
  `0 / 0.000140070508876904 / 0.000350174990604679 / 0.000700324467521879`。
- 分离不是持续发散：5% raw probe distance由iteration 0的 `0.04996878900157145` 在iteration 5升至
  `0.059301643040744256`，之后缓慢降至iteration 60的 `0.056153631120118744`；prediction distance则从
  `0.0029676155305378886`持续降至 `0.000700324467521879`。这支持“早期更新放大一部分probe差异、后续数据预测继续靠拢”的有限解释。
- 四支loss均单调不增、detector residual均显著下降且最终非常接近，但simulation-only final aligned probe error随幅度从
  `0.13937197214033825`升至 `0.1492569162151173`。因此detector fit相近不能在本case中保证probe解相近；这是一项初始化敏感性
  development observation，不是最低谱端的严格证明，也不能推广到真实TGV数据。
- 新证据没有要求恢复第22节的low-end eigensolver问题；该问题继续作为延后项。下一轮主要矛盾转为：一个明确operator/reconstruction
  module的作用能否通过单因素direct ablation被归因，而不再继续优化本轮初始化曲线。

### Operator-consistency 检查

- Truth replay：正式run重新生成matched data并得到 `truth_replay_exact=true`、relative L2 `0.0`。
- Adjoint dot test、full-loss finite-difference gradient、zero-residual update和truth fixed-point：本轮没有修改forward/adjoint/loss/gradient，
  按第24节恢复块不机械重跑；第20--22节已有controls继续有效。本轮targeted regression覆盖旧mode调用路径。
- Detector quadrature：HDF5记录q4的16个positive weights，shape `(16,)`、每个 `0.0625`、sum `1.0`；`I_stack` shape
  `(25,32,32)`、dtype `float64`、minimum `0.04091490300503436`，没有把sqrt(pixel intensity)复制到q4 nodes。
- Finite-B boundary：沿用同一finite support、transparent exterior、对 `B-1` constant-zero integer shift和open reference-plus-residual
  matched chain；相关operator未改变，因此未重跑第20--22节已有边界/adjoint control。
- Checkpoint/adjoint-related pairing audit：三类checkpoint matrices为 `(13,4,4)`，iterations为
  `0,5,...,60`；raw/aligned/prediction末帧均与final matrices exact。
- Determinism：固定seed `20260843`；非零branch归一方向最大pairwise L2 error
  `3.898284132372964e-15 <= 1e-12`。没有为了证明run级bitwise determinism重复执行第二个正式run；deterministic generator和new
  contracts由targeted tests覆盖。
- 所有truth usage flags在initialization、optimizer和truth-free diagnostic中均为false；truth仅用于明确标记的simulation evaluation。

### Development run 与 artifacts

- 唯一新run：
  `runs/exp042_TGV_3d_multislice_probe_reconstruction_dev_20260822_163049`；`run_state.json`为 `complete`，
  `artifacts_validated=true`，figure count `1`。
- `config.yaml`：run copy SHA256
  `567B8E06D7F02E434C24059FA9E92ADEE90AEF86F652C1CA605CE59039EE50B6`；保留固定幅度、seed、matched operator与60-step budget。
- `metadata.json`：SHA256 `DEE11BE3BE04E1D546187BB636BE6F66B30E7BC28579A3E720E2242891B29161`；记录
  `run_role=initialization_magnitude_ablation_matched`、`initialization_family=fixed_direction_magnitude_sweep`、known-B/probe-only、
  `reference_validated=false`、`full_tgv_reference_authorized=false`和truth usage flags。
- `metrics.json`：SHA256 `B08B0F1B958483A754D78D67CC8EE462B8C701F4FD72809B2A95048ED970BE5D`；JSON/HDF5选定
  branch/checkpoint/final字段独立核对一致。
- HDF5：`outputs/exp042_probe_reconstruction.h5`，SHA256
  `5B3DF1D256357438833F7352C56D8A4957A1FC8D9CD4D1BB2D8DF36EE140FC7B`，size `4168640` bytes。保留
  `/entry/data/I_stack (25,32,32)`、`scan_positions (25,2)`、instrument/sample/truth provenance；在
  `/entry/reconstruction/initialization_ablation/`下保存四支raw `P_B_init/P_B_rec (96,96)`、curves、simulation-only aligned fields、
  `design`和`truth_free_pairwise_diagnostic`，并在 `/entry/metrics/initialization_ablation/`镜像自然产生的metrics。没有空group或项目级schema
  change。
- Figure：`figures/exp042_initialization_magnitude_ablation.png`，SHA256
  `3C3B24D572F6D5C3F9E25E015079C9B15EE8B4E4B2AB53B74431FEF5EAE4F13B`，size `333655` bytes；8个预注册panels完成目视检查。

### 当前 metrics

以下probe-truth error均为 **simulation evaluation only**，不进入optimizer或branch selection：

- `homogeneous_reference`：residual `0.3567623010258413 -> 0.008857188126643553`；loss
  `0.03127716305291964 -> 1.9277964661849684e-05`；aligned probe error
  `0.47873691495342324 -> 0.13937197214033825`。
- `perturb_1pct`：residual `0.3568212842110594 -> 0.008855293893537657`；loss
  `0.0312875059574132 -> 1.9269719820817872e-05`；aligned probe error
  `0.47885458058781016 -> 0.1396780039482004`。
- `perturb_2p5pct`：residual `0.35691302674200454 -> 0.008856468325924901`；loss
  `0.03130359672393973 -> 1.9274831448740914e-05`；aligned probe error
  `0.4795355456582235 -> 0.141749550905277`。
- `perturb_5pct`：residual `0.357074638166358 -> 0.008869122404370298`；loss
  `0.03133195189085838 -> 1.93299503664604e-05`；aligned probe error
  `0.48200798884046003 -> 0.1492569162151173`。
- 四支均完成60步、stopping reason均为 `iteration_budget`、backtracking均为0、loss均单调不增。maximum final pairwise raw/aligned probe
  distance分别为 `0.056153631120118744` / `0.0561535867667631`；maximum final pairwise prediction distance为
  `0.000700324467521879`。
- 以上没有预注册scientific threshold，不给出Passed/Failed。development status保持不变。

### 失败、限制与未关闭问题

- 代码验证唯一失败是首次Ruff的本轮新增 `E501`，已以机械换行关闭；artifact审计唯一命令错误是PowerShell parser error，修正后关闭。
  正式run、targeted tests、最终Ruff和artifact validator均成功。
- 只测试一个deterministic direction和0%--5%幅度；“近似线性”只属于该matched noiseless development case，不能外推到任意方向、幅度、
  噪声或真实数据。
- 四支均在60步budget停止，不能宣称无限迭代后的不同初始化极限解必然不同；不过本轮预注册问题是equal-budget sensitivity，而不是精确极限。
- detector predictions持续靠拢而probe fields保持幅度相关分离，说明当前data fit不足以单独判定probe一致性；这不等于已经定位某个具体operator
  module或严格证明null space/lowest spectrum。
- `reference_validated=false`、`full_tgv_reference_authorized=false`继续成立；没有真实TGV准确性、waist精度、resolution或detection limit结论。

### 改动后总体优先级

- 下一轮主要矛盾（一个）：detector quadrature这一明确模块对当前reconstruction behavior的作用是否可由单因素对照归因。最值得做的最小
  工作是预注册“matched q4 reference vs reconstruction-side q1 point-detector explicit-mismatch diagnostic”，保持同一q4 data、truth/B/scan、
  初始化和equal budget，只改变重建measurement operator的quadrature；q1支必须明确标记为simplified mismatch，不能替代exp040-matched
  baseline。预期作用是直接判断省略pixel integration会如何改变residual、probe error和branch separation。
- 次要工作一：为q4/q1两支保存同一组truth-free prediction/probe checkpoint distances，区分早期transient和持续差异；必要性是避免只看final
  scalar。
- 次要工作二：运行前明确q1 prediction与q4 pixel data的normalization/shape contract，并增加adjoint dot、full-loss gradient和truth replay
  controls；必要性是保证差异来自预注册quadrature mismatch，而不是接口错误。
- 次要工作三：若q1 mismatch暴露明显差异，后续再考虑增加单独的q1/q1 matched diagnostic以区分“point model本身”与“data/model mismatch”；
  本轮不提前实现第三支。
- 明确不建议现在做：恢复lowest-spectrum精确收敛、延长当前四支budget、增加更多seeds、同时移除bandlimit/open mapping或改变B/scan；这些会
  混合因素或偏离direct-ablation主线。建议继续exp042；当前仍是baseline内部operator ablation，尚未形成必须新开实验的问题。

### 下一轮快速恢复上下文

- 当前authoritative appended section：本文第25节；第23节是路线调整，第24节是不同direction的5%初始化消融，第22节spectral问题继续暂挂。
- 本轮完成的Change：Change 01，固定seed/方向的 `0/1/2.5/5%` equal-budget magnitude ablation、checkpoint pairwise diagnostics、HDF5/
  metrics/8-panel figure、targeted regression和唯一正式run。
- 本轮修改文件：
  `configs/experiments/exp042_TGV_3d_multislice_probe_reconstruction.yaml`、
  `src/tgv_ptycho/recon/exp042.py`、`scripts/run_exp042_probe_reconstruction.py`、
  `tests/test_exp042_probe_reconstruction.py`和本文EOF第25节。
- 当前有效config：上述exp042 YAML，source SHA256
  `FD16A6234114C37531493108CDA95ED581BCB51D71D761E928B9B290E9A048D3`；mode
  `initialization_magnitude_ablation_matched`，family `fixed_direction_magnitude_sweep`，seed `20260843`，magnitudes
  `0/0.01/0.025/0.05`，checkpoint interval `5`，budget `60`。
- 最新有效run：
  `runs/exp042_TGV_3d_multislice_probe_reconstruction_dev_20260822_163049`，complete/validated；run-config SHA256
  `567B8E06D7F02E434C24059FA9E92ADEE90AEF86F652C1CA605CE59039EE50B6`。
- 已通过commands：targeted pytest `9 passed in 50.79s`；修复后scoped Ruff `All checks passed!`；唯一formal runner exit `0`；
  truth replay、q4 quadrature、same-direction tolerance、checkpoint final exact、selected JSON/HDF5 equality、267 numeric datasets finite和figure
  read-back/目视检查均通过。首轮Ruff `E501`和只读PowerShell parser error均已记录并关闭。
- 已确认不需要重跑的controls：只要forward/operator/loss/gradient相应实现不变，第20--22节的adjoint/gradient/fixed-point/zero-residual/
  finite-B/open controls不需机械重跑；本轮run不为增加事后metric或延长budget而重跑；exp040 reference controls不需重算。
- 当前未关闭的主要矛盾：下一项单模块direct ablation，推荐detector q4 matched reference对reconstruction-side q1 explicit mismatch。
- 当前次要矛盾：q4/q1 checkpoint trajectory；q1与q4 data的shape/normalization/adjoint contract；未来是否需要q1/q1 matched第三支。
- 下一轮推荐最小改动：先在测试级构造同一q4 `I_stack`上的q4/q1 reconstruction-operator pair，完成q1 branch的shape、adjoint dot、full-loss
  gradient和zero-residual controls；通过后加入两支equal-budget runner，最多一个新timestamped run。不要把q1支称为exp040-matched baseline。
- 下一轮最小读取集合：`AGENTS.md`；本文第3、6、9、10、23、24和本第25节；当前YAML；module中的measurement operator/config
  validation/optimizer/ablation diagnostic；runner的ablation/HDF5/figure validator；对应targeted tests；latest run只读metrics/HDF5
  initialization-ablation subgroup和figure。
- 只有q1复用需要确认exp040 detector node语义、forward/operator API发生变化、summary与artifact冲突、出现regression或latest artifact不完整时，
  才定向重读exp040 R4/R8对应定义；不重读R10--R14 solver/history。若恢复local-identifiability研究，先参考第22节而不是重扫exp040。
- 明确不应重新扫描：exp040全文/`_old.md`、全部历史runs、整个`src`/`tests`、notebooks、reports、data、exp041/exp05x详细内容和无关
  theory notes。

### Append-only verification

- append前旧文件为 `198567` bytes，SHA256
  `CA6EE52FD0B267D95FCA6A660680676B41020B6AD21887F0D50BC5325E2564C3`，最后章节号 `24`。
- append后必须独立按原长度读取prefix；预期并核对prefix bytes与上述SHA256完全不变，heading顺序为 `24 -> 25`。本节只在EOF增加；
  若校验不符，只能在新EOF correction note记录，不能回改旧section。

### Git 状态

本轮只产生本地unstaged/untracked修改；没有执行 `git add`、commit、push、PR、merge或branch操作。exp042 config/script/module/test/本文
仍为untracked，runs被Git ignore；staged为空。开始时已有的exp030/exp040文档修改、deleted notebook和其他untracked用户文件保持原状态，
未覆盖、未删除、未重解释。commit/push/PR状态均未改变。

### Correction note — 本轮 append 定位失败与恢复

- 首次追加第25节时，`apply_patch` 的短上下文命中了第21节末尾相同的Git状态文字，使新section暂时位于第22节之前；首次独立检查得到
  `PrefixMatches=false`（current bytes `220228`、current SHA256
  `63BBFB915883EE2F5523F23D55C3647566D30B8ADFA8E7E7A2F6F1FD70782F29`、旧长度prefix SHA256
  `44335234F9E725AD51508C294B4F51754CA96F08DDCAB5352DD8ABE9913B344B`）。这是文档定位/审计失败，代码、config、tests和正式run均未受影响。
- 第一次纠正按错误的heading范围暂时连同原第22节一起移出；工具状态仍完整保留第22节原248行。随后先原样恢复第22节，并独立确认文档精确回到
  `198567` bytes、SHA256
  `CA6EE52FD0B267D95FCA6A660680676B41020B6AD21887F0D50BC5325E2564C3`，章节顺序恢复为 `22 -> 23 -> 24`；没有改写第1--24节内容。
- 然后以第24节最后9行为唯一EOF anchor重新追加本第25节。main record追加后为 `220228` bytes、SHA256
  `EE1BEC1439B9D0633375233DE0EADBD14CA42C5600EB7443E7C19A68A6279B7B`，第25节count为1、位于第24节之后，原`198567`字节prefix
  SHA256仍为上述值，`PrefixMatches=true`。
- 为解决该直接blocker，额外做了受限只读搜索：只在workspace的 `docs/.codex/.agents` 和VS Code local-history路径查找exp042同名备份线索，未找到；
  随后用已知旧SHA定向定位本任务的单个Codex session JSONL，只读取相关hash/patch上下文，确认第22节夹在误插第25节与第23节之间。
  该搜索没有扫描源码、tests、runs、notebooks、reports、data或其他实验，没有联网；找到恢复依据后即停止扩大范围。
- 本correction note本身仍只追加在EOF。任务结束审计将再次按原长度核对prefix，并在最终回复报告最终文件bytes/SHA256与Git staged状态。

## 26. 2026-08-22 19:28：Implementation iteration 10 — q4-data detector-readout ablation

### 本轮目标与明确未做事项

本轮按第25节的唯一主要建议，建立 `matched_q4` 与 reconstruction-side `mismatch_q1_point` 的确定性单因素对照。两支使用同一套由冻结的
matched q4 operator生成的 `P_B_true`、known `B_true`、scan positions和 `I_stack`，并固定相同的homogeneous initialization、GN-scaled
Armijo optimizer与60-step budget；唯一预注册变化是reconstruction detector readout。目标是判断省略q4 positive pixel integration时，
detector residual、probe reconstruction和truth-free branch separation会发生什么变化，同时保持q1支为明确的negative control。

本轮明确未做q1/q1 matched第三支、blind `P_B + B`、`D_waist`拟合或腰径估计、sample-B/scan redesign、noise/stage/subpixel/calibration、
bandlimit/open/finite-B消融、更多seed、iteration continuation、local spectral diagnostic或最低谱端精确收敛。没有修改exp040 frozen evidence，
没有运行exp040 Helmholtz/reference pipeline，也不把q1 diagnostic或matched inverse behavior解释为真实三维电磁准确性。状态保持
`Development baseline / No scientific pass-fail conclusion`。

### 开始时检查的内容

- 开始时执行 `git -c safe.directory=E:/tgv_ptycho_sim status -sb`；staged为空。已有exp030/exp040文档修改、deleted notebook及其他untracked
  用户内容均被识别并保留。
- 按恢复协议读取 `AGENTS.md`、本文第3、6、9、10节和第25节，重点恢复第25节的“失败、限制与未关闭问题”“改动后总体优先级”及
  “下一轮快速恢复上下文”。没有顺序重读更早implementation records。
- 定向检查当前YAML，`src/tgv_ptycho/recon/exp042.py` 中measurement operator、readout/adjoint、loss/gradient、config validation与truth-free
  diagnostic，runner中的mode dispatch、HDF5/metrics/figure/artifact validator，以及exp042 targeted tests；没有顺序通读整个 `src` 或 `tests`。
- 为确认q1 point readout相对exp040 q4语义的边界，按第25节许可额外定向读取exp040 primary文档R4第463--506行、R8定义第828--904行及
  R8结果边界第1034--1125行。结论是primary data必须继续由q4 positive staggered midpoint quadrature产生，而q1只能作为显式
  reconstruction-side mismatch；答案明确后即停止扩大范围。没有读取exp040 old、R10--R14 solver/history、全部runs、notebooks、reports或data，
  也没有联网。
- 复用已验证的deterministic matched development generator/operator chain；没有从不同历史run拼接truth和data。正式run重新生成同一完整
  operator chain的 `P_B_true`、`B_true`、q4 `I_stack`和scan positions。
- 修改代码前锁定本文为 `222131` bytes，SHA256
  `40566B2011CAAF8B7F039B566487C5B8EAF6E44DDDEAD316EF472DD5EE827D3E`，最后章节号为 `25`。

### 上一轮意见

- 第25节的下一轮主要矛盾是：detector quadrature这一单一模块对当前reconstruction behavior的作用能否通过同一q4 data上的
  `matched_q4` / reconstruction-side `q1 point` explicit mismatch归因。
- 上一轮要求固定truth/B/scan/initialization/optimizer/budget，先建立q1 shape、normalization、transpose、full-loss gradient与replay controls，
  再运行最多一个timestamped run；q1不得称为exp040-matched baseline。q1/q1 matched第三支只在本对照确实暴露差异后再考虑。
- 复核后意见仍成立：初始化方向与幅度问题已分别由第24、25节在equal budget下回答，继续优化初始化或恢复lowest-spectrum问题会偏离当前
  direct-ablation主线。本轮完整采用上一轮建议，未根据结果事后改变branch、budget、metric或stopping。

### Change 01

#### 改动前：主要矛盾、次要矛盾与必要性判断

- 继承的主要矛盾：尚无一个只改变detector readout且具有严格transpose/full-loss gradient证据的q1 negative control，因此任何q4/q1差异都
  可能被operator实现错误混淆。
- 本Change唯一主要矛盾：能否建立复用同一q4 propagation field、只替换readout、且数值adjoint/gradient闭合的
  `mismatch_q1_point` measurement operator。
- 次要矛盾最多三个：point interpolation的确定定义；q1 self-data fixed point与q4-data mismatch的区分；shared linear chain是否逐项相同。
- 明确延后：runner/artifacts、q1/q1 matched第三支、其他operator modules和最低谱端问题。
- 必要性：若q1 readout自身没有配对transpose和gradient，后续branch曲线没有可归因性；因此operator consistency必须先于正式对照run。

#### 技术决策与理由

- `matched_q4` 保持现有16个positive node weights的pixel average，不把pixel `sqrt(I)`复制到q4 nodes。
- `mismatch_q1_point` 复用完全相同的q4 detector propagation field，在每个4x4 node block中对中央2x2复场作等权双线性中点插值，再取
  `abs(.)**2`。这表示pixel中心point detector diagnostic，不是pixel integral。
- 为中点复场插值实现显式Euclidean transpose：pixel adjoint等分回中央2x2 q4 nodes；intensity Jacobian与full-loss residual继续通过这一
  transpose和共享linear adjoint回传，没有使用 `H^-1` 代替band-limited ASM的 `H^H`。
- primary truth/data始终由matched q4 branch生成；q1 self-data仅用于验证其自身fixed point/zero-gradient。q1对q4 truth-data的非零replay
  residual是预注册mismatch signal，不是truth generator的替代。
- q4/q1 operator共享finite-B support、transparent exterior、对 `B-1` 的constant-zero integer shift、P/B复乘、BC band-limited ASM、
  reference-plus-residual open mapping、padding/restriction/crop/native ROI和detector node geometry；只改变readout。

#### 创建或修改的文件

- `src/tgv_ptycho/recon/exp042.py`：增加bilinear midpoint point-field readout及其transpose，扩展measurement operator的intensity、Jacobian、
  adjoint/full-loss路径，增加detector-quadrature consistency与truth-free diagnostic。最终SHA256为
  `CC3B9AB63EBB08E917E8778F4CB64C6828F4ACC6078E83D0BCF291B1E27D36D7`。
- `tests/test_exp042_probe_reconstruction.py`：增加q1 readout/adjoint/Jacobian/gradient/self-fixed-point/q4-data mismatch与shared-chain controls；
  同一文件随后也覆盖Change 02 runner contract。最终SHA256为
  `61785A67B5817FBAF183757865B713E7BE1F21EAA275FEB9F1CEE75A8119E6FF`。

#### 验证命令与结果

- 定向operator pytest调用exp042 test文件中的config、truth replay、linear adjoint、full-loss gradient/fixed-point和新增q1 control五项；结果
  `5 passed, 5 deselected in 17.12s`。
- matched q4 truth replay exact，relative L2 `0.0`；q4 quadrature adjoint relative error
  `1.1149627102382043e-16`。
- q1 point-field readout adjoint relative error `5.344856384280288e-16`；q1 intensity Jacobian adjoint relative error
  `5.670789142949717e-16`；directional Jacobian finite-difference error `2.203922966584297e-09`。
- q1 operator对固定q4 truth-data的full-loss directional gradient error `1.6814699178487923e-08`；q1 self-data replay/loss、zero-gradient和
  truth-initialized one-step fixed point均exact。
- q1对q4 truth-data replay按预期不exact，relative L2 `0.012422040278502295`。shared linear arrays/scalars逐项exact或为同一对象；truth未
  进入controls或optimizer。
- 首次scoped Ruff报告测试文件import排序 `I001`；只做机械import排序后，同一检查通过，没有数值行为变化。

#### 改动后重新评估

- 本Change主要矛盾已关闭：q1 branch具有明确point-readout定义、严格transpose、Jacobian/full-loss gradient证据及self fixed-point controls，
  同时在q4 data上保留可测的显式mismatch。
- 新证据没有暴露shared propagation、finite-B/open或quadrature pairing regression。q4/q1差异可进入equal-budget runner验证，而不需要先扩大到
  其他module或spectral问题。
- 下一Change的主要矛盾转为：两支是否能在同一q4 data、同一初始化与预算下形成可运行、可审计的pipeline和唯一正式run。

### Change 02

#### 继承Change 01的评估与改动前判断

- 继承结果：operator/readout correctness已关闭；当前不再优化q1数学实现。
- 本Change唯一主要矛盾：能否把 `matched_q4` / `mismatch_q1_point` 组成equal-budget、timestamped、HDF5/metrics/figure完整的对照，并保存
  truth-free trajectory以检查差异何时形成。
- 次要矛盾最多三个：JSON/HDF5 checkpoint final exact；两支初始化逐元素相同；figure和metadata能否清楚标记q1的mismatch role。
- 明确延后：q1/q1 matched第三支、更多budgets/seeds、其他operator ablation和任何scientific threshold。
- 必要性：只有运行级artifacts才能回答第25节提出的direct ablation；operator unit control本身不能给出reconstruction behavior。

#### 技术决策与理由

- YAML mode设为 `detector_quadrature_ablation_q4_data`，固定两支、reference branch、homogeneous initialization、60步GN-scaled Armijo和每5步
  checkpoint；q1 model role强制为 `explicit_mismatch_diagnostic`。
- runner只生成一次matched q4 `I_stack`，两支读取相同数组并从逐元素相同的 `P_B_init` 开始。保存raw probe、global-phase-aligned
  simulation-only probe和prediction的pairwise checkpoint distances；aligned量不进入optimizer，也不覆盖raw reconstruction。
- figure预注册为2x4八个panels，展示loss、detector residual、raw/aligned simulation-only probe error、三类truth-free checkpoint曲线和final
  aligned probe-distance matrix。
- HDF5沿用 `/entry` 并列结构，在experiment-specific reconstruction/metrics subgroup自然增加两支、design、controls和diagnostics；没有修改
  project-level writer/schema。

#### 创建或修改的文件

- `configs/experiments/exp042_TGV_3d_multislice_probe_reconstruction.yaml`：加入并启用q4-data detector-readout ablation contract；最终SHA256
  `46D06262F89F736EC3BF74C18ABFCC95FED196E4B9B9CFA24C794CF21B6C224C`。
- `scripts/run_exp042_probe_reconstruction.py`：增加两支run编排、metadata/metrics/HDF5 payload、8-panel figure及artifact validator；最终SHA256
  `3E439A7C7F0AF07CD47138620F8862D544842F64657B4B48FF3694EE929C41C7`。
- `tests/test_exp042_probe_reconstruction.py`：增加tiny runner对branch roles、same initialization、checkpoint和HDF5 subgroup的审计。
- 本文只在真实EOF追加本第26节；没有修改顶部状态或旧第1--25节。

#### 验证命令与结果

- 新tiny runner测试前两次均失败于测试代码把JSON反序列化后的list当作NumPy array读取 `.shape`；runner两次都已完成并通过其artifact
  validator。第一次修复命中了另一处相似断言，第二次才修复正确位置；均为测试断言修复，没有修改算法或生产artifact contract。
- 修复后tiny runner selection结果 `1 passed, 10 deselected in 3.03s`。
- 完整targeted命令：
  `D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim python -m pytest -q tests/test_exp042_probe_reconstruction.py`；结果
  `11 passed in 58.23s`。
- 最终scoped Ruff命令：
  `D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim python -m ruff check src/tgv_ptycho/recon/exp042.py scripts/run_exp042_probe_reconstruction.py tests/test_exp042_probe_reconstruction.py`；
  结果 `All checks passed!`。
- 未运行full pytest：没有修改shared forward/optics/shift/IO，旧mode regression、operator controls与新runner contract均已由11项targeted tests覆盖；
  不扩大到全仓既有问题。
- 唯一正式命令：
  `D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim python scripts/run_exp042_probe_reconstruction.py --config configs/experiments/exp042_TGV_3d_multislice_probe_reconstruction.yaml`；
  exit `0`，runtime `85.78597009999794 s`，`artifacts_validated=true`。
- 独立artifact审计确认177个numeric datasets全部finite，selected JSON/HDF5 metrics exact，三类checkpoint末帧与final matrices exact，figure可读取并
  目视通过；没有为审计重跑正式实验。

#### 改动后重新评估

- 本Change主要矛盾已关闭：两支在同一q4 data、逐元素相同初始化和equal budget下完成，run、metadata、metrics、HDF5、figure及validator完整。
- `matched_q4` final detector residual `0.008857188126643553`，q1 mismatch为 `0.013877636581934633`；simulation-only aligned probe error
  分别为 `0.13937197214033825` 与 `0.1407863889828641`。省略q4 integration产生了可归因的data-fit与probe差异，但差异幅度不支持
  scientific pass/fail或物理外推。
- truth-free raw/aligned final probe distances为 `0.02246895506912404` / `0.0222812134445028`；prediction distance从initial
  `0.010360090974309621`到final `0.010917748156504524`。probe separation从0增长，prediction distance在iteration 5达到约
  `0.01144836`后回落但未低于初始值，说明差异不是只有final scalar或单纯初始化偏差。
- 新的未关闭问题是因果拆分：当前q1分支同时具有“point model”与“q4-data/q1-model mismatch”身份。本run证明省略q4 integration的显式mismatch
  会改变结果，但不能区分point-detector approximation本身与data/model mismatch各自的作用。

### Operator-consistency 检查

- Truth replay：matched q4完整 `I_stack` replay exact，relative L2 `0.0`；truth pair仍是exact-data fixed point。
- Adjoint dot tests：q4 quadrature、q1 complex midpoint readout和q1 intensity Jacobian relative errors分别为
  `1.1149627102382043e-16`、`5.344856384280288e-16`、`5.670789142949717e-16`。
- Gradient tests：q1 Jacobian directional finite-difference error `2.203922966584297e-09`；q1-on-q4-data full-loss directional gradient error
  `1.6814699178487923e-08`。
- Fixed-point / zero-residual：q1 self-data replay、loss、zero-gradient和truth-initialized one-step update均exact；matched q4已有同类exact control。
- Detector quadrature：q4有16个positive weights，每个 `0.0625`、sum `1.0`；q1采用中央2x2复场等权插值后取强度，明确不称为pixel
  integral。q1对q4 truth-data replay relative L2 `0.012422040278502295`，符合negative-control设计。
- Finite-B/open/shared chain：两支finite support、transparent exterior、`B-1` constant-zero shift、复乘、BC propagation、bandlimit、open
  reference-plus-residual、padding/restriction/crop/native ROI和node geometry完全共享；只改变readout。没有把 `H^-1` 当作 `H^H`。
- Determinism：固定配置和generator seed；两支 `P_B_init` 逐元素exact，checkpoint iterations为 `0,5,...,60`，三类checkpoint shape均
  `(13,2,2)`且末帧与final matrices exact。没有为run级bitwise重复性创建第二个正式run。
- Truth usage：所有initialization/optimizer/truth-free diagnostic flags均为false；truth只用于明确标记的simulation evaluation。

### Development run 与 artifacts

- 唯一新run：`runs/exp042_TGV_3d_multislice_probe_reconstruction_dev_20260822_192316`；`run_state.json`为 `complete`，
  `artifacts_validated=true`，runtime `85.78597009999794 s`，figure count `1`。
- `config.yaml`：SHA256 `504147BA360CDE7D2B0D7E60A9056751385169E462172BDE11A0C3927B381176`；记录q4-data、two readouts、
  same initialization和60-step budget。
- `metadata.json`：SHA256 `BF39C6E13B0F315BE19E615F734A352890C0F6B41855FEAC1F8B478EBA08FDE5`；记录
  `run_role=detector_quadrature_ablation_q4_data`、known-B/probe-only、q1 explicit mismatch、`reference_validated=false`、
  `full_tgv_reference_authorized=false`与truth usage flags。
- `metrics.json`：SHA256 `A180113C9BADD55C8A254248AFC2E318C25961E5A262FEBDD774C22C052B58D8`；selected branch/control/
  checkpoint metrics与HDF5逐项exact。
- HDF5：`outputs/exp042_probe_reconstruction.h5`，SHA256
  `3F602088378CA9673644877EDE63CBDC5647FC27129BEC1DDF131F4D4D6119C1`，size `3229280` bytes；保留
  `/entry/data/I_stack (25,32,32)`、`scan_positions (25,2)`、instrument/sample/truth provenance。在
  `/entry/reconstruction/detector_quadrature_ablation/` 保存 `design`、`controls`、两支raw `P_B_init/P_B_rec (96,96)`、curves、
  simulation-only aligned fields和truth-free pairwise diagnostics，并在 `/entry/metrics/detector_quadrature_ablation/` 保存对应metrics。
  没有空group或项目级schema change。
- Figure：`figures/exp042_detector_quadrature_ablation.png`，SHA256
  `0795FD7DD8DC478127457C6D1D8A8462C6C36E1341A75B5BB134A4436C3BEE9D`，size `303154` bytes；8个panels完整、可读、无截断。

### 当前 metrics

以下probe-truth error均为 **simulation evaluation only**，不进入optimizer、branch selection或stopping：

- `matched_q4`：detector residual `0.3567623010258413 -> 0.008857188126643553`；loss
  `0.03127716305291964 -> 1.9277964661849684e-05`；aligned probe error
  `0.47873691495342324 -> 0.13937197214033825`。
- `mismatch_q1_point`：detector residual `0.3576077554391245 -> 0.013877636581934633`；loss
  `0.03142557978995783 -> 4.732607221165744e-05`；aligned probe error
  `0.47873691495342324 -> 0.1407863889828641`。
- 两支均完成60步、backtracking均为0。truth-free initial prediction distance为 `0.010360090974309621`；final raw/aligned probe distance
  为 `0.02246895506912404` / `0.0222812134445028`；final prediction distance为 `0.010917748156504524`。
- 以上没有预注册scientific threshold，不给出Passed/Failed。development status保持不变。

### 失败、限制与未关闭问题

- 验证过程的失败包括首次Ruff `I001`，以及tiny runner测试两次对JSON list误用 `.shape` 的测试断言失败；均已以机械修复关闭。runner从第一次
  tiny调用起即完成并通过artifact validator，正式run、完整targeted suite、最终Ruff和artifact审计均成功。
- 当前是q4-data/q1-model explicit mismatch，只能说明省略matched q4 pixel integration会改变equal-budget behavior；不能区分point detector
  model本身的conditioning/可恢复性与data/model mismatch，也不能把q1 branch当作exp040-matched baseline。
- 两支均在60步budget结束；结果是equal-budget development comparison，不是无限迭代极限或global optimum结论。
- 单一deterministic noiseless matched case不能外推到其他B/scan、noise、真实calibration或真实TGV。
- `reference_validated=false`、`full_tgv_reference_authorized=false`继续成立；没有真实三维电磁准确性、waist精度、resolution或detection limit结论。

### 改动后总体优先级

- 下一轮主要矛盾（一个）：是否增加独立的q1/q1 matched control，以区分“point-detector approximation本身”与“q4-data/q1-model mismatch”。
  最小建议是在不改变truth/B/scan/initialization/optimizer/budget的前提下，新增由同一q1 point operator生成data并由同一q1 operator重建的
  matched支，同时保留本轮q4/q4 reference与q4-data/q1 mismatch作为既有对照。预期作用是把model family effect和mismatch penalty分开。
- 次要工作一：在运行前先完成q1/q1 truth replay、zero-residual/fixed-point、full-loss gradient和normalization controls；必要性是避免第三支接口错误。
- 次要工作二：预注册三支最小比较矩阵和truth-free checkpoints，明确哪些pair比较model effect、哪些pair比较mismatch effect；必要性是防止事后解释。
- 次要工作三：若q1/q1 matched与q4/q4仍有差异，再判断是否值得研究readout conditioning；在此之前不恢复lowest-spectrum eigensolver。
- 明确不建议下一轮同时改bandlimit、open mapping、finite-B、B、scan、optimizer、budget或seed；否则无法保持detector readout的单因素归因。
  建议继续exp042；当前仍是baseline内部operator control，没有形成必须新开实验的问题。

### 下一轮快速恢复上下文

- 当前authoritative appended section：本文第26节；第23节是路线调整，第24/25节分别是初始化方向/幅度消融，第22节lowest-spectrum问题继续暂挂。
- 本轮完成的Change：Change 01，q1 midpoint point readout/transpose/Jacobian/full-loss consistency controls；Change 02，同一q4 data上的
  `matched_q4` / `mismatch_q1_point` equal-budget runner、truth-free checkpoints、HDF5/metrics/8-panel figure和唯一正式run。
- 本轮修改文件：
  `configs/experiments/exp042_TGV_3d_multislice_probe_reconstruction.yaml`、
  `src/tgv_ptycho/recon/exp042.py`、`scripts/run_exp042_probe_reconstruction.py`、
  `tests/test_exp042_probe_reconstruction.py`和本文EOF第26节。
- 当前有效config：上述exp042 YAML，source SHA256
  `46D06262F89F736EC3BF74C18ABFCC95FED196E4B9B9CFA24C794CF21B6C224C`；mode `detector_quadrature_ablation_q4_data`，
  branches `matched_q4` / `mismatch_q1_point`，checkpoint interval `5`，budget `60`。
- 最新有效run：`runs/exp042_TGV_3d_multislice_probe_reconstruction_dev_20260822_192316`，complete/validated；run-config SHA256
  `504147BA360CDE7D2B0D7E60A9056751385169E462172BDE11A0C3927B381176`。
- 已通过commands：operator selection `5 passed, 5 deselected in 17.12s`；tiny runner selection `1 passed, 10 deselected in 3.03s`；完整
  targeted suite `11 passed in 58.23s`；最终scoped Ruff `All checks passed!`；唯一formal runner exit `0`；177 numeric datasets finite、
  JSON/HDF5 selected metrics exact、checkpoint final exact和figure read-back/目视检查均通过。首次Ruff与两次测试断言失败已记录并关闭。
- 已确认不需要重跑的controls：只要shared forward/operator chain和本轮readout/loss实现不变，本轮q4/q1 adjoint/gradient/fixed-point、第20--22节
  finite-B/open controls及本轮正式run不需机械重跑；不为增加事后metric、延长budget或证明bitwise重复性而重跑；exp040 reference controls不重算。
- 当前未关闭的主要矛盾：是否用独立q1/q1 matched第三支拆分point-model effect与q4/q1 mismatch penalty。
- 当前次要矛盾：第三支operator controls；三支pairwise comparison contract；若model effect存在，未来是否研究readout conditioning。
- 下一轮推荐最小改动：先在测试级用当前q1 operator生成q1 self-data并完成replay/gradient/fixed-point，再将其加入三支、同预算runner；最多一个新
  timestamped run。不要改动其他operator module，也不要把任何q1结果称为exp040物理validation。
- 下一轮最小读取集合：`AGENTS.md`；本文第3、6、9、10、23和本第26节；当前YAML；module中的readout/measurement operator、consistency
  metrics和diagnostic；runner的detector-ablation/HDF5/figure validator；对应targeted tests；latest run只读metrics、HDF5 detector subgroup和figure。
- 只有forward/operator/B/scan语义变化、摘要与artifact冲突、出现regression、latest artifact不完整，或q1/q1设计无法由现有point operator回答时，
  才定向重读exp040 R4/R8；不重读R10--R14 solver/history。若恢复local-identifiability研究，先参考第22节而不是重扫exp040。
- 明确不应重新扫描：exp040全文/`_old.md`、全部历史runs、整个 `src`/`tests`、notebooks、reports、data、exp041/exp05x详细内容和无关
  theory notes。

### Append-only verification

- append前旧文件为 `222131` bytes，SHA256
  `40566B2011CAAF8B7F039B566487C5B8EAF6E44DDDEAD316EF472DD5EE827D3E`，最后章节号 `25`。
- 第26节main record追加后文件为 `245567` bytes，SHA256
  `12367AAE69A8CC1A15D91C55BBC0532B8F3FB4DE92E6D95E0F26C6BC68CAA5BB`；独立读取原 `222131` 字节prefix所得SHA256仍为
  `40566B2011CAAF8B7F039B566487C5B8EAF6E44DDDEAD316EF472DD5EE827D3E`，`PrefixMatches=true`。
- heading顺序核对为 `24 -> 25 -> 26`，第26节count为1且位于真实EOF；旧第1--25节及顶部状态没有修改。本verification自身继续只追加在
  第26节EOF，任务结束时再次核对同一原prefix。

### Git 状态

- 最终检查 `git diff --cached --name-status` 无输出，staged为空；本轮没有执行 `git add`、commit、push、PR、merge或branch操作。
- exp042 config/script/module/test/本文仍为本地untracked内容，runs被Git ignore；开始时已有的exp030/exp040文档修改、deleted notebook与其他
  untracked用户文件保持原状态，未覆盖、未删除、未重解释。commit/push/PR状态均未改变。

## 27. 2026-08-22 19:56：Implementation iteration 11 — q4/q1 three-branch data-model pairing control

### 本轮目标与明确未做事项

本轮按第26节的唯一主要建议，在已有 `matched_q4`（q4 data/q4 reconstruction）和 `mismatch_q1_point`（q4 data/q1 reconstruction）之外，
增加独立 `matched_q1_point`（q1 data/q1 reconstruction）simplified matched control。三支共用同一 `P_B_true`、known `B_true`、scan、
homogeneous initialization、GN-scaled Armijo和60-step budget；目标是把“matched detector model family effect”与“q4-data/q1-model mismatch
penalty”分开，而不是继续优化第26节两支曲线。

本轮明确未做blind `P_B + B`、`D_waist`拟合或腰径估计、sample-B/scan redesign、noise/stage/subpixel/calibration、bandlimit/open/finite-B
消融、更多seed、iteration continuation、local spectral diagnostic或最低谱端精确收敛。没有修改exp040 frozen evidence，没有运行exp040
Helmholtz/reference pipeline，也不把q1/q1 simplified matched control称为exp040-matched或真实detector model。状态保持
`Development baseline / No scientific pass-fail conclusion`。

### 开始时检查的内容

- 开始时执行 `git -c safe.directory=E:/tgv_ptycho_sim status -sb`；staged为空。已有exp030/exp040文档修改、deleted notebook及其他untracked
  用户内容均被识别并保留。
- 按连续会话恢复协议使用仍在有效上下文中的 `AGENTS.md` 与本文固定第3、6、9、10、23节边界，并完整读取第26节，重点复核“失败、限制与
  未关闭问题”“改动后总体优先级”和“下一轮快速恢复上下文”。没有机械重读更早implementation records。
- 定向检查当前YAML；`src/tgv_ptycho/recon/exp042.py` 中detector config validation、q1 consistency metrics和truth-free detector diagnostic；runner的
  detector metadata/design/HDF5/figure/validator/run/dispatch；对应targeted tests。为决定q1 control measurement在现有HDF5 writer下的最小安全
  位置，额外只读取 `src/tgv_ptycho/io/save_load.py` 中 `save_ptycho_hdf5()` 第68--130行；结论是不修改共享writer或项目级schema，而把q1 control
  stack放在experiment-specific reconstruction subgroup。答案明确后即停止扩大范围。
- 本轮没有重读exp040，因为第26节已确认q4/q1 node/readout语义且本轮没有改变该语义；没有扫描全部runs、整个 `src`/`tests`、notebooks、
  reports或data，没有联网。
- 复用同一deterministic development generator/operator chain；正式run重新由同一truth/B/scan生成primary q4 stack和q1 control stack，没有
  混用历史artifact或让truth参与initialization、optimizer、branch selection或stopping。
- 修改代码前锁定本文为 `246703` bytes，SHA256
  `61A0CE52796C757858D4209695CB4032F2914FFD80E3813E53685BDBB0B3AD12`，最后章节号为 `26`。

### 上一轮意见

- 第26节未关闭的主要矛盾是：q1 point model本身与q4-data/q1-model mismatch同时存在于同一negative-control branch，尚不能分别归因。
- 上一轮建议先在测试级用当前q1 operator生成q1 self-data并完成replay/gradient/fixed-point，再加入第三支、同预算runner，最多一个新timestamped
  run；要求保留q4/q4 reference和q4/q1 mismatch，不改变truth/B/scan/initialization/optimizer/budget。
- 次要矛盾是第三支operator controls、三支pairwise comparison contract，以及只有matched q4/q1仍有差异时才考虑readout conditioning；
  bandlimit/open/B/scan、lowest-spectrum与更多budgets/seeds明确延后。
- 复核后上一轮意见仍成立。第26节已证明q4/q1 mismatch有可测差异，但因果拆分是继续其他module ablation前最小且必要的control；本轮完整采用
  该建议，没有根据运行结果事后改变branch、budget、metric或停止时机。

### Change 01

#### 改动前：主要矛盾、次要矛盾与必要性判断

- 继承的主要矛盾：缺少q1 data/q1 reconstruction matched branch，导致point-model effect与data/model mismatch混合。
- 本Change唯一主要矛盾：能否注册并验证q4/q4、q4/q1、q1/q1三支data/model pairing，使第三支是自身exact-data fixed point，同时不改变共享
  propagation、B、scan或optimizer。
- 次要矛盾最多三个：q1/q1 full-loss gradient；两条q1 reconstruction支是否严格共用同一operator；三组pair比较的预注册语义。
- 明确延后：runner/formal artifacts、其他operator modules、more cases/seeds和lowest-spectrum。
- 必要性：没有第三支时，第26节约2.23%的probe separation不能判断来自point readout本身还是用q1模型拟合q4数据；补齐self-matched control比
  继续追求更低residual更必要。

#### 技术决策与理由

- execution mode改为 `detector_quadrature_three_branch_control`。三支固定为：
  `matched_q4 = primary_q4_data + positive_pixel_average`；
  `mismatch_q1_point = primary_q4_data + bilinear_midpoint_point_from_q4_nodes`；
  `matched_q1_point = q1_point_control_data + bilinear_midpoint_point_from_q4_nodes`。
- primary `/entry/data/I_stack`仍由冻结matched q4 operator产生；q1 control data由相同truth/B/scan和已有q1 point operator确定性生成，只服务于
  simplified matched diagnostic，不替代exp040 primary data。
- 两条q1 reconstruction branch强制复用同一个operator对象。三支使用逐元素相同初始化和相同reconstruction settings。
- 预注册三组解释：`matched_q4 vs mismatch_q1_point`为共享q4 data上的readout mismatch；`mismatch_q1_point vs matched_q1_point`为共享q1
  reconstruction model时的q4/q1 data pairing差异；`matched_q4 vs matched_q1_point`为各自matched时的detector model family effect。
- 在已有q1 adjoint/Jacobian/self fixed-point controls上，增加q1 self-data full-loss directional gradient control；truth仍不进入optimizer或
  truth-free pairwise diagnostic。

#### 创建或修改的文件

- `configs/experiments/exp042_TGV_3d_multislice_probe_reconstruction.yaml`：注册三支、两种data source、三组comparison contract、q1/q1 included
  flag和新figure filename；最终SHA256
  `B381152274D67E9EBA3F055C171CC4FC8C2FC7F865E5A08248BDC5C92AE98A26`。
- `src/tgv_ptycho/recon/exp042.py`：扩展config validation、q1 self-data gradient/replay controls，并把truth-free detector diagnostic从硬编码两支
  泛化为至少两支的pairwise实现；最终SHA256
  `0CE0A0B05EF559FBD43FB791BBC40180BCEB7988F8A265608C78FE08FEE14F89`。
- `tests/test_exp042_probe_reconstruction.py`：扩展config/operator test为三支，验证q1 self-data replay/gradient及 `(3,3)` pairwise checkpoint；
  同一文件随后也覆盖Change 02 runner contract。最终SHA256
  `AD5DF2D910EA8ADBD7D7772A43DA70177818CAE0A63DC382E5C7AF2C64478CB7`。

#### 验证命令与结果

- Operator/control selection：
  `D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim python -m pytest -q tests/test_exp042_probe_reconstruction.py -k "config_and_shapes or truth_replay_detector_controls_and_determinism or linear_shift_and_quadrature_adjoint_dot_products or full_loss_gradient_and_zero_residual_fixed_point or q1_point_mismatch_controls_and_truth_free_diagnostic"`；
  结果 `5 passed, 6 deselected in 25.40s`。
- matched q4 truth replay exact，relative L2 `0.0`；q1 self replay exact，relative L2/loss/zero-gradient均 `0.0`，truth fixed point exact。
- q4 quadrature、q1 point readout、q1 intensity Jacobian adjoint relative errors分别为 `1.1149627102382043e-16`、
  `5.344856384280288e-16`、`5.670789142949717e-16`。
- q1 Jacobian directional finite-difference error `2.203922966584297e-09`；q1 full-loss gradient error在q4 data上为
  `1.6814699178487923e-08`，在q1 self-data上为 `2.2026043761344724e-09`，均低于预注册 `2e-6`。
- q1 control stack与q4 primary stack shape一致、全部nonnegative且deterministic；两者relative L2为 `0.012422040278502295`，因此第三支不是
  对q4 data的静默复制。

#### 改动后重新评估

- 本Change主要矛盾已关闭：三支data/model pairing被config强制注册，q1/q1 self branch具有exact replay/fixed-point和full-loss gradient证据，
  两条q1 reconstruction支共用同一operator。
- 新证据没有暴露q1 operator、shared linear chain或truth leakage问题；无需重读exp040或扩大到其他module。
- 下一Change的主要矛盾转为：三支能否在同预算正式run中形成完整、可审计的data mapping、pairwise trajectory和artifacts。

### Change 02

#### 继承Change 01的评估与改动前判断

- 继承结果：三支operator/data contract已通过测试级controls；本Change不再改变readout数学定义。
- 本Change唯一主要矛盾：能否完成三支equal-budget run，并用truth-free pairwise evidence实际拆开model-family effect与mismatch penalty。
- 次要矛盾最多三个：q1 control stack的HDF5位置；三支checkpoint/final matrix exact；figure和metadata能否明确q1不属于exp040-matched。
- 明确延后：budget continuation、更多cases/seeds、其他operator ablation与scientific thresholds。
- 必要性：只有第三支实际重建曲线和pairwise probe/prediction轨迹才能回答第26节遗留的因果问题；unit fixed point本身不能量化两种作用。

#### 技术决策与理由

- runner一次生成primary q4与q1 control stacks，按branch `data_source`传给同一optimizer；q4 primary与q4/q1 mismatch逐元素使用同一q4 stack，
  q1/q1使用独立q1 control stack。
- `/entry/data/I_stack`继续保存authoritative q4 primary data；为避免修改共享writer/schema，q1 stack保存在
  `/entry/reconstruction/detector_quadrature_ablation/control_measurements/q1_point_control_data/I_stack`，并携带
  `simulation_matched_control_only`及`exp040_matched=false`标记。
- branch metrics统一改为 `measurement_fit`，显式记录 `data_source` / `data_readout`；metadata和design记录三支axis、comparison contract及q1
  scientific boundary。
- figure保持2x4八个panels，但标题改为branch-specific own-data loss/residual与三支pairing；truth-free checkpoint和final aligned probe matrix扩展为
  三支。

#### 创建或修改的文件

- `scripts/run_exp042_probe_reconstruction.py`：扩展metadata/design/HDF5/metrics/figure/validator/run/dispatch为三支，强制q4 data共享、q1 operator
  identity、q1 control data非复制及三支artifact exact；最终SHA256
  `3848EF89AD1B292B438827C0ED9ACDD01A2043E1B63D5E1C5DBA2BB2B48A1C88`。
- `tests/test_exp042_probe_reconstruction.py`：tiny runner增加第三支、q1 control stack、`(3,3)` checkpoints和data-control assertions。
- 本文仅在真实EOF追加本第27节；没有修改顶部状态或旧第1--26节。

#### 验证命令与结果

- Tiny runner selection：
  `D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim python -m pytest -q tests/test_exp042_probe_reconstruction.py -k detector_quadrature_ablation_runner`；
  结果 `1 passed, 10 deselected in 3.77s`。
- 完整targeted命令：
  `D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim python -m pytest -q tests/test_exp042_probe_reconstruction.py`；结果
  `11 passed in 81.88s`。
- Scoped Ruff命令：
  `D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim python -m ruff check src/tgv_ptycho/recon/exp042.py scripts/run_exp042_probe_reconstruction.py tests/test_exp042_probe_reconstruction.py`；
  结果 `All checks passed!`。本轮代码/test/Ruff没有失败。
- 未运行full pytest：没有修改shared forward/optics/shift/IO；只读检查 `save_ptycho_hdf5()` 后选择experiment-specific subgroup，writer本身未改。
  三支operator、旧exp042 modes regression和runner contract均由11项targeted tests覆盖，因此没有理由扩大到全仓既有问题。
- 唯一正式命令：
  `D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim python scripts/run_exp042_probe_reconstruction.py --config configs/experiments/exp042_TGV_3d_multislice_probe_reconstruction.yaml`；
  exit `0`，runtime `150.99311710000256 s`，`artifacts_validated=true`。
- 首次独立HDF5 numeric traversal审计的callback错误返回tuple，导致 `visititems()`在首个dataset停止并给出无效count `1`；第二次修正命令因
  command-line quote转义错误产生Python `SyntaxError`。第三次以明确返回`None`的visitor成功，得到302个numeric datasets且全部finite。这两次是
  只读审计命令失败，不是代码、run或artifact失败；没有重跑正式实验。
- 独立审计还确认JSON/HDF5 selected final residual exact、三支init逐元素exact、三类checkpoint final exact、q4/q1 stacks和quadrature weights
  正确。Figure read-back finite并完成目视检查，8 panels完整、无截断或不可读标签。

#### 改动后重新评估

- 本Change主要矛盾在本deterministic noiseless case和60-step equal budget下已关闭。final aligned probe distance：
  `matched_q4 vs matched_q1_point = 0.005174988189090093`；
  `mismatch_q1_point vs matched_q1_point = 0.022902953625056812`；
  `matched_q4 vs mismatch_q1_point = 0.0222812134445028`。
- matched q4/q1 model-family separation约0.52%，而共享q1 reconstruction model下由q4/q1 data pairing造成的separation约2.29%，后者约为前者
  4.4倍。因此在本case/equal budget下，第26节观察到的约2.23%分离主要与data/model mismatch一致，而不是point model自身造成同量级变化。
- `matched_q1_point` simulation-only aligned probe error `0.13674882391730617`，小于matched q4的 `0.13937197214033825`；但这是单case、
  两种不同synthetic data上的equal-budget truth-aided evaluation，不能据此宣称q1 detector优于q4或选择新的detector model。
- mismatch branch在共享q4 data上的final residual `0.013877636581934633`，高于matched q4的 `0.008857188126643553`；q1/q1 own-data
  residual为 `0.008709871737336922`。前两者可直接作为同q4 data的fit对照，q1/q1 residual属于不同control data，不能直接当成物理优劣排名。
- detector quadrature的当前因果问题已经足够关闭，不建议继续增加q factors、budgets或seeds来优化这一模块。下一轮主要矛盾应转向另一个单一
  reconstruction operator module，而不是恢复最低谱端问题。

### Operator-consistency 检查

- Truth replay：matched q4 primary stack replay exact，relative L2 `0.0`；q1 control stack由q1 operator生成并self replay exact，relative L2/loss/
  zero-gradient均 `0.0`，truth-initialized fixed point exact。
- Adjoint/gradient：q4 quadrature、q1 point readout、q1 intensity Jacobian adjoint errors分别为 `1.1149627102382043e-16`、
  `5.344856384280288e-16`、`5.670789142949717e-16`；q1 full-loss directional gradient errors在q4/q1 data上分别为
  `1.6814699178487923e-08` / `2.2026043761344724e-09`。
- Detector data controls：q4和q1 stacks均 `(25,32,32)`、float64、finite、nonnegative；relative L2
  `0.012422040278502295`。q4仍使用16个positive weights，每个 `0.0625`、sum `1.0`；q1仍是中央2x2复场平均后取强度，未冒充pixel
  integration。
- Shared chain：三支truth/B/scan、finite-B transparent exterior、`B-1` constant-zero shift、BC propagation、bandlimit、open mapping、padding/
  restriction/crop/native ROI和node geometry不变；两条q1 reconstruction支共用同一operator，三支初始化逐元素exact。
- Checkpoints：iterations为 `0,5,...,60`；raw/aligned probe与prediction matrices均 `(13,3,3)`，末帧与final matrices逐元素exact。
- Determinism/truth boundary：固定generator/config seed；没有创建第二个正式run。truth只用于生成synthetic control data与明确标记的simulation
  evaluation，不参与initialization、optimizer、branch selection、stopping或truth-free diagnostic；q1 branches均为
  `exp040_matched=false`。

### Development run 与 artifacts

- 唯一新run：`runs/exp042_TGV_3d_multislice_probe_reconstruction_dev_20260822_195139`；`run_state.json`为 `complete`，
  `artifacts_validated=true`，runtime `150.99311710000256 s`，figure count `1`。
- `config.yaml`：SHA256 `2B24FB4E3D87C82EA259F3EAA3685E456A7D9F0AFEB22D2EA11E560E66DB6C0B`；记录三支data/model mapping、
  comparison contract、same initialization和60-step budget。
- `metadata.json`：SHA256 `B6C47B761DD0A5C329B426FA0F7925AD04EA7D0739D44800EAD3A139B0760C3C`；记录
  `run_role=detector_quadrature_three_branch_control`、known-B/probe-only、q1 mismatch/matched-control roles、
  `reference_validated=false`、`full_tgv_reference_authorized=false`和truth usage boundary。
- `metrics.json`：SHA256 `DA36CAAB9999CC434A66128CD85702873F7A04167F36B78CED891A82D157BB80`；selected branch/control/
  checkpoint metrics与HDF5逐项exact。
- HDF5：`outputs/exp042_probe_reconstruction.h5`，SHA256
  `4AF86885BA6A5A60F6E76A48ADE5B272E25A5E96FE34D2B03E3774D53848D62E`，size `3942336` bytes；`/entry` children为
  `config_yaml/data/instrument/metadata/metrics/reconstruction/sample/truth`。保留 `/entry/data/I_stack (25,32,32)` 与
  `scan_positions (25,2)`；新增experiment-specific
  `/entry/reconstruction/detector_quadrature_ablation/control_measurements/q1_point_control_data/I_stack (25,32,32)`；三支
  `P_B_init/P_B_rec`均 `(96,96)`，并保存design、operator specs、curves、simulation-only aligned fields和 `(13,3,3)` truth-free diagnostics。
  302个numeric datasets全部finite，没有项目级schema change。
- Figure：`figures/exp042_detector_quadrature_three_branch_control.png`，SHA256
  `D94D19BF8C0A1F3B577919FACB86717F90306F75B3E9B747E2DC48C8457D635E`，size `330288` bytes；8 panels完成read-back和目视检查。

### 当前 metrics

以下probe-truth error均为 **simulation evaluation only**，不进入optimizer、branch selection或stopping：

- `matched_q4`：residual `0.3567623010258413 -> 0.008857188126643553`；loss
  `0.03127716305291964 -> 1.9277964661849684e-05`；aligned probe error
  `0.47873691495342324 -> 0.13937197214033825`。
- `mismatch_q1_point`：residual `0.3576077554391245 -> 0.013877636581934633`；loss
  `0.03142557978995783 -> 4.732607221165744e-05`；aligned probe error
  `0.47873691495342324 -> 0.1407863889828641`。
- `matched_q1_point`：residual `0.3586727143269997 -> 0.008709871737336922`；loss
  `0.031754778094720666 -> 1.8725607778914353e-05`；aligned probe error
  `0.47873691495342324 -> 0.13674882391730617`。
- 三支均完成60步、stopping reason均为 `iteration_budget`、backtracking均为0、loss均单调不增且residual均下降。
- Final aligned probe pairwise：q4/q4 vs q4/q1 `0.0222812134445028`；q4/q4 vs q1/q1 `0.005174988189090093`；q4/q1 vs q1/q1
  `0.022902953625056812`。对应final prediction distances为 `0.010917748156504524`、`0.012430304081549331`、
  `0.005252335770254552`。
- 以上没有预注册scientific threshold，不给出Passed/Failed；development status保持不变。

### 失败、限制与未关闭问题

- 本轮代码、operator tests、tiny runner、完整targeted suite、Ruff和正式run均未失败。独立artifact审计有两次命令级失败：callback提前终止和quote
  `SyntaxError`；修正后得到302个numeric datasets全部finite，已关闭且未重跑正式run。
- q1/q1是simplified matched diagnostic，不是exp040 primary data，也不代表真实point detector calibration；q4 primary branch仍是唯一
  exp040-working-model matched reference。
- matched-family约0.52%与mismatch约2.29%的分解只属于一个deterministic noiseless case和60-step equal budget；不是无限迭代、全局最优、
  多case robustness或真实数据结论。
- q4/q4与q1/q1的own-data residual不能直接作为同一objective下的优劣排名；simulation-only probe error也不能用于选择真实detector model。
- `reference_validated=false`、`full_tgv_reference_authorized=false`继续成立；没有真实三维电磁准确性、waist精度、resolution或detection limit结论。

### 改动后总体优先级

- 下一轮主要矛盾（一个）：BC propagation中的bandlimit/alias-control这一单一模块对当前known-B reconstruction behavior的作用能否用同样的
  matched/mismatch因果结构归因。最小建议是先在测试级构造 `bandlimited data/bandlimited reconstruction` reference、`bandlimited data/
  unbandlimited reconstruction` explicit mismatch与 `unbandlimited data/unbandlimited reconstruction` simplified matched control，保持q4 readout、
  truth/B/scan/initialization/optimizer/budget不变，并为两种transfer分别使用真正的conjugate adjoint。预期作用是继续direct-ablation路线，且不把
  detector quadrature与propagation alias control混在同一Change。
- 次要工作一：运行前定义unbandlimited control transfer和data provenance，并做linear adjoint、full-loss gradient、self fixed-point；必要性是避免
  把transfer实现错误解释为bandlimit作用。
- 次要工作二：复用本轮三支comparison contract和checkpoint矩阵，但使用新的experiment-specific subgroup/figure名称；必要性是保持artifact可归因且
  不覆盖本轮结果。
- 次要工作三：若bandlimit mismatch差异低于数值噪声或control data与primary几乎相同，记录为低作用并转向下一模块，不通过延长budget或调参追metric。
- 明确不建议下一轮同时改变open mapping、finite-B boundary、B、scan、detector readout、optimizer、budget或seed；不恢复lowest-spectrum精确收敛。
  建议继续exp042；当前仍是baseline内部single-module control，没有形成必须新开实验的问题。

### 下一轮快速恢复上下文

- 当前authoritative appended section：本文第27节；第26节是q4/q1两支mismatch，第23节是路线调整，第22节lowest-spectrum问题继续暂挂。
- 本轮完成的Change：Change 01，q4/q4、q4/q1、q1/q1三支data/model pairing与q1 self-data controls；Change 02，三支runner、HDF5 control
  measurement、metrics/8-panel figure、targeted regression、唯一正式run与因果拆分。
- 本轮修改文件：
  `configs/experiments/exp042_TGV_3d_multislice_probe_reconstruction.yaml`、
  `src/tgv_ptycho/recon/exp042.py`、`scripts/run_exp042_probe_reconstruction.py`、
  `tests/test_exp042_probe_reconstruction.py`和本文EOF第27节。共享IO writer只读检查，未修改。
- 当前有效config：上述exp042 YAML，source SHA256
  `B381152274D67E9EBA3F055C171CC4FC8C2FC7F865E5A08248BDC5C92AE98A26`；mode `detector_quadrature_three_branch_control`，
  branches `matched_q4 / mismatch_q1_point / matched_q1_point`，checkpoint interval `5`，budget `60`。
- 最新有效run：`runs/exp042_TGV_3d_multislice_probe_reconstruction_dev_20260822_195139`，complete/validated；run-config SHA256
  `2B24FB4E3D87C82EA259F3EAA3685E456A7D9F0AFEB22D2EA11E560E66DB6C0B`。
- 已通过commands：operator selection `5 passed, 6 deselected in 25.40s`；tiny runner `1 passed, 10 deselected in 3.77s`；完整targeted
  suite `11 passed in 81.88s`；scoped Ruff `All checks passed!`；唯一formal runner exit `0`；302 numeric datasets finite、selected
  JSON/HDF5 exact、three-branch init/checkpoint final exact和figure目视检查均通过。两次只读审计命令失败已记录并关闭。
- 已确认不需要重跑的controls：只要q4/q1 readout/operator及本轮config不变，本轮adjoint/gradient/fixed-point和正式run不需机械重跑；不为增加
  事后metric、延长budget或更多seeds重跑；exp040 reference controls不重算。若下一轮只改变bandlimit control，则先新增transfer相关controls，
  不重算exp040 Helmholtz/reference pipeline。
- 当前未关闭的主要矛盾：下一个single-module direct ablation，推荐BC bandlimit/alias-control的matched/mismatch因果拆分。
- 当前次要矛盾：unbandlimited transfer/adjoint contract；新module的artifact命名和comparison mapping；若作用微弱时及时停止而不调参追metric。
- 下一轮推荐最小改动：先在测试级创建共享geometry/B/scan/q4 readout的bandlimited/unbandlimited operator pair，完成transfer dot/full-loss gradient/
  self replay/fixed-point；通过后加入三支equal-budget runner，最多一个新timestamped run。不要改open/finite-B/detector/optimizer。
- 下一轮最小读取集合：`AGENTS.md`；本文第3、6、9、10、23和本第27节；当前YAML；module中的measurement operator/transfer construction/
  config validation/consistency diagnostic；runner的detector control可复用pattern及新module HDF5/figure validator；对应targeted tests；latest run只读
  metrics/HDF5 detector subgroup/figure作为比较格式。
- 只有bandlimit transfer API语义不清、forward/operator/B/scan发生变化、摘要与artifact冲突、出现regression或latest artifact不完整时，才定向
  重读exp040 R8的bandlimit/operator定义；不重读R10--R14 solver/history。若恢复local-identifiability研究，先参考第22节而不是重扫exp040。
- 明确不应重新扫描：exp040全文/`_old.md`、全部历史runs、整个 `src`/`tests`、notebooks、reports、data、exp041/exp05x详细内容和无关
  theory notes。

### Append-only verification

- append前旧文件为 `246703` bytes，SHA256
  `61A0CE52796C757858D4209695CB4032F2914FFD80E3813E53685BDBB0B3AD12`，最后章节号 `26`。
- 第27节main record追加后文件为 `272032` bytes，SHA256
  `040C4554148A9670C2A72105BEAAB3D4C8E2B7AA94A34F32E5918683F74AD376`；独立读取原 `246703` 字节prefix所得SHA256仍为
  `61A0CE52796C757858D4209695CB4032F2914FFD80E3813E53685BDBB0B3AD12`，`PrefixMatches=true`。
- heading顺序核对为 `25 -> 26 -> 27`，第27节count为1且位于真实EOF；旧第1--26节及顶部状态没有修改。本verification自身继续只追加在
  第27节EOF，任务结束时再次核对同一原prefix。

### Git 状态

- 最终检查 `git diff --cached --name-status` 无输出，staged为空；本轮没有执行 `git add`、commit、push、PR、merge或branch操作。
- exp042 config/script/module/test/本文仍为本地untracked内容，runs被Git ignore；开始时已有的exp030/exp040文档修改、deleted notebook与其他
  untracked用户文件保持原状态，未覆盖、未删除、未重解释。commit/push/PR状态均未改变。

### Correction note — 本轮 verification append 定位失败与恢复

- 第27节main record首次成功追加并确认原prefix不变后，追加verification/Git块时使用了过短且重复的“明确不应重新扫描”anchor，误命中第26节末尾，
  使该块暂时位于第27节之前。随后的独立审计立即得到 `PrefixMatches=false`：文件 `273168` bytes、SHA256
  `6F2EE0B7511202B8B7BBA487270BB2537614EEED3D94CD65F6F17F918A6134B8`，原长度prefix SHA256
  `A63F32A393DE00BDC3C314C55BECDE2DCA00EEE5CD93CC77FCAB1CB5E2F5BEF0`。代码、tests、run和artifacts均未受影响。
- 只精确移除这段误插的1136 bytes后，文档恢复到main record的 `272032` bytes、SHA256
  `040C4554148A9670C2A72105BEAAB3D4C8E2B7AA94A34F32E5918683F74AD376`，原 `246703` 字节prefix SHA256重新精确匹配，章节顺序恢复为
  `25 -> 26 -> 27`；没有改写第1--26节内容。
- 随后使用第27节独有的bandlimit恢复上下文长anchor，把verification、Git状态和本correction note仅追加到真实EOF。任务结束审计将再次按原长度
  核对prefix，并在最终回复报告最终bytes/SHA256。

## 28. 2026-08-23 16:20：路线调整记录 — exp042 development baseline 阶段性挂起

本节不是新的 Implementation iteration；没有修改 reconstruction 代码/config/test，没有运行新实验，也不改变第 16--27 节的数值证据。

### 暂挂决定

- Stage A operator consistency 与 Stage B known-B probe-only development baseline 已经建立；q4/q4 matched branch 可作为后续 Phase 5
  的 raw reconstructed-probe 输入。
- exp042 仍未完成 blind B、多 case、noise/calibration 或真实数据研究，也没有 `D_waist`、resolution 或 detection-limit 结论。
- 当前不继续第 27 节建议的 bandlimit/open/finite-B 等逐模块扩张；第 23 节暂挂的最低谱端问题也继续保留而不恢复。

### 暂挂理由与恢复条件

当前 q4/q4 run 的 detector residual 已由 `0.356762` 降至 `0.008857`，simulation-evaluation-only aligned probe error 由
`0.478737` 降至 `0.139372`。剩余约 14% 的 full-field error 是否会实际限制腰径拟合，不能仅靠继续降低 probe metric 判断；应由
Phase 5 的 `P_B_true`/raw `P_B_rec` 配对参数拟合给出更直接证据。

只有后续出现以下证据之一，才建议定向恢复 exp042：`P_B_true` 可稳定拟合而 matched raw `P_B_rec` 不可拟合；腰径敏感方向与
exp042 已发现的弱方向或 operator mismatch 显著重合；跨实验 artifact handoff 不完整；或新的 measurement design 明确要求重做
known-B reconstruction。恢复时只解决由下游证据指出的一个主要问题，不默认继续全部消融或恢复 lowest-spectrum sweep。

当前 authoritative 状态入口为第 0 节；最新有效 run 仍为
`runs/exp042_TGV_3d_multislice_probe_reconstruction_dev_20260822_195139`。本次治理改动经用户明确授权，在页首新增可更新的第 0 节；
原第 1--27 节文本保持不变。改动前全文为 `274322` bytes，SHA256
`931F49BC2221F9E9753289F3F77C08776FFF8D0F6D957AFE2E186C26E1EEB43F`。以后第 0 节可以同步实时状态，第 1 节以后仍只允许在 EOF
追加新记录。
