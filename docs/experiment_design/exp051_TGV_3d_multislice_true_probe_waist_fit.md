# exp051：3D scalar multislice true-probe waist fitting

## 0. 实时状态与阅读顺序

本节是本文唯一允许持续原位更新的状态入口。第 1 节以后是先于实现和正式运行冻结的设计正文；首次完整构建
完成后，既有文字只能保持不变，新的实现、结果、失败或 correction 只能追加到 EOF。

```text
Scientific status: Passed
Work status: q8 plateau-interval formal complete / artifacts validated
Results available: true
Latest valid run: runs/exp051_TGV_3d_multislice_true_probe_waist_fit_q8_plateau_interval_20260824_161440
Authoritative appended section: section 27
Primary current question: 是否按 exp051 handoff 另行预注册 exp053 matched raw P_B_rec interval fit
reference_validated: false
full_tgv_reference_authorized: false
```

阅读顺序：

- 第 0 节：实时状态、最新有效 run、当前问题和本文阅读索引。
- 第 1--19 节：首次实现前冻结的研究设计、阈值和判定逻辑。
- 第 20--22 节：首次完整实现、waist-fit formal run、artifact audit 和环境复核。
- 第 23--25 节：q8 local-differentiability control 的预注册、实现与 preflight、formal 结果和供 exp04x
  手工写入的反馈。
- 第 26 节：q8 plateau-equivalence set、interval estimator 和 non-smooth identifiability gate 的预注册。
- 第 27 节：第 26 节方案的一次完整实现、preflight corrections、tests/lint、唯一 formal、artifact audit、
  exp04x 手工反馈和最终判定。
- 后续章节：每轮 implementation iteration 原则上只在 EOF 追加一个编号章节，并在该节完整记录本轮的
  预注册、Changes、测试与 lint、development/formal run、artifact audit、结果、限制、Git 状态和下一步；
  只有独立 correction 或确需分离的新证据才另起章节，不得回改历史正文。第 20--25 节保持现状，不追溯合并。

这里的 truth 始终只表示
`simulation truth under the selected exp040 scalar working model`。

## 1. 研究问题与 scope boundary

研究问题固定为：在 exp040 冻结的 scalar multislice working model 中，固定除 `D_waist` 外的全部 TGV 几何、
光学、网格和 reference convention，并直接使用 matched complex `P_B_true` 时，单参数 `D_waist` 是否能被
自洽、稳定且可重复地拟合？目标量为

```text
D_waist = min_z D(z)
```

本实验只建立 oracle fitting baseline。输入是 matched raw `P_B_true`；唯一变化参数是 `D_waist`。不使用
`P_B_rec`，不从 detector intensity 直接拟合，不恢复任意 `D(z)`，不同时拟合 `z_waist`、`D_top`、
`D_bottom`、折射率、传播距离、幅度、phase offset 或其他 nuisance；不研究 noise、stage error、真实 calibration
或真实数据。成功只表示 selected scalar working model 内的 self-consistency，不表示 Maxwell/full-wave、真实器件
或真实计量精度验证。

## 2. Source identity 与不可混用输入

authoritative source 固定为：

```text
source experiment: exp042
source run: runs/exp042_TGV_3d_multislice_probe_reconstruction_dev_20260822_195139
source HDF5: outputs/exp042_probe_reconstruction.h5
target dataset: /entry/truth/P_B_true
source exp040 branch: R8 unified q8 finite-B open q4 scalar working model
development case: canonical_geometry_coarse_matched_dev
source git commit recorded by artifact: 6c1a43fdfe5d172db24474b6fbba0b1f92087666
```

source artifact SHA256 冻结为：

```text
config.yaml    2B24FB4E3D87C82EA259F3EAA3685E456A7D9F0AFEB22D2EA11E560E66DB6C0B
metadata.json  B6C47B761DD0A5C329B426FA0F7925AD04EA7D0739D44800EAD3A139B0760C3C
metrics.json   DA36CAAB9999CC434A66128CD85702873F7A04167F36B78CED891A82D157BB80
run_state.json 329F96D1B95BB6D2960C189F93AE909FFC2075866381A7391988E6F7D037FD4E
HDF5           4AF86885BA6A5A60F6E76A48ADE5B272E25A5E96FE34D2B03E3774D53848D62E
P_B_true bytes FA61264AF0D96BF3393EC461E2147992FDE6926D0132B2346090790E40E8EBFD
```

loader 必须核对 source state 为 `complete`、`artifacts_validated=true`、上述 file hashes 和 dataset identity。
任何 `/entry/reconstruction/.../P_B_rec`、truth-aligned copy 或 detector control data 都不是 exp051 输入；若配置的
target path 包含 `reconstruction` 或以 `P_B_rec` 结尾，必须 hard fail。raw `P_B_rec` 仅属于 exp053 handoff。

## 3. Target probe 的物理与数值身份

冻结 target 为 B plane raw complex field：

```text
plane: B, at z_AB = 0.5 mm after the physical A-exit boundary
dataset: /entry/truth/P_B_true
shape / axis: (96, 96), (y, x)
dtype: complex128
node dx: 0.5 um in both y and x
native FOV: 48 um x 48 um
coordinate convention: x/y = (index - 47.5) * 0.5 um
coordinate endpoints: -23.75 um ... +23.75 um
geometric origin: midway between the four central samples
units: complex field in arbitrary amplitude units; phase in rad
```

probe 由 sample-A raw full field 与 homogeneous reference 的 residual decomposition 产生：内部参考介质
`n_ref=1.5`，外部介质 `n=1.0`，A-to-B 使用同网格 bandlimited、alias-controlled angular spectrum transfer；
NumPy FFT convention 为 forward unnormalized、inverse 含 `1/(ny*nx)`。target native probe 不进行 detector crop、
sample-B modulation 或 q4 detector readout。`open_shape=(256,256)` 和 centered zero padding of
`P_B - homogeneous_probe` 属于后续 exp042 measurement operator，不改变保存的 native `P_B_true` 身份。

## 4. 冻结 sample-A geometry 与 optical parameters

固定参数如下：

```text
model: axisymmetric TGV q8 cell-average scalar multislice
shape: (96, 96), axis (y, x), dx = 0.5 um
thickness: 100 um; target dz: 1 um
slice centers: 0.5, 1.5, ..., 99.5 um
slice widths: 100 values of 1 um; sum exactly 100 um within float64
D_top / D_bottom: 30 / 30 um
z_waist: 50 um
center (x,y): (0,0)
n_glass / n_air: 1.5 / 1.0
wavelength: 532 nm
illumination: unit-amplitude, normal-incidence plane wave
internal / external reference index: 1.5 / 1.0
z_AB: 0.5 mm
interface: q8 staggered midpoint air-area fraction
multislice: centered symmetric split-step, bandlimit=true, internal alias_control=false
AB propagation: bandlimit=true, alias_control=true
complex dtype: complex128; geometry/index dtype: float64
```

`D(z)` 使用共享 piecewise-linear `diameter_profile()` 在 slice centers 采样；q8 只表示 analytic indicator 的
cell-average quadrature，不是 Maxwell effective-medium law。sample B、scan、BC propagation 和 detector 不参与 primary fit。

## 5. 唯一变化参数与 bounds

唯一变化参数为 `D_waist`。本 case 的预注册 fitting interval 为闭区间 `[16,24] um`；它包含 exp040 已有
`18/20/22 um` signature 并留出两侧 margin，同时严格满足 `0 < D_waist <= min(D_top,D_bottom)`。该区间是本
实验的先验工作范围，不宣称是所有真实 TGV 的通用物理范围。非法、非有限或越界候选必须在 forward 前拒绝。

simulation truth 为 `20 um`，只用于 exact replay、simulation evaluation 和最终 accuracy gate；optimizer 不得以 truth
early stop、选择 branch 或调节超参数。

## 6. Frozen profile grid、finite difference 与缓存

Stage B grid 冻结为：

```text
coarse grid: 16,17,...,24 um (1 um spacing; 9 points)
local fine grid: coarse minimum ±0.5 um, clipped to bounds, 0.125 um spacing
primary central finite-difference half-step h1: 0.25 um
step control half-step h2: 0.125 um
```

fine grid 的中心只能由 frozen coarse loss 的 argmin 决定；同 loss 时选择较小 `D_waist`，不得人工换中心。完整
coarse/fine profile 均保存。所有 candidate 通过 exp042 matched development generator 中复用的 exp040 q8 scalar
building blocks 生成，不复制 parallel forward。cache key 是 candidate 的 float64 meter value；同一 key 只生成一次。
HDF5 保存本 run 所有 unique candidate 的 `D_waist`、probe stack 和每条 profile/FD/optimizer track 到 cache index
的 mapping，足以逐点审计和完整复算。

## 7. Primary loss 与 reference/gauge rule

全视场 mask 固定为全 `True`，每个 complex pixel 等权。primary loss 为无量纲 raw complex normalized squared L2：

$$
L(D)=\frac{\sum_{y,x}|P_B(D;y,x)-P_B^{target}(y,x)|^2}
{\sum_{y,x}|P_B^{target}(y,x)|^2}.
$$

分母在一次 run 内固定，必须有限且大于零。固定 unit plane wave、reference indices、carrier、FFT convention 和物理
B plane 已给出绝对 complex reference；primary loss 不允许 global-phase、complex gain、amplitude scale、affine phase、
spatial registration 或结果后选择的 mask。raw complex loss 与任何未来 aligned diagnostic 必须分开保存和命名。
本次不预注册 aligned/downstream-compatible nuisance control，因此正式 run 不计算它。

辅助 replay 诊断预注册为：

$$
e_A=\frac{\| |P|-|T| \|_2}{\||T|\|_2},\qquad
e_\phi=\frac{\||T|(e^{i\arg P}-e^{i\arg T})\|_2}{\|T\|_2}.
$$

二者只解释 replay，不替代 raw primary loss。

## 8. Stage A exact replay gate

任何 optimizer 前，candidate generator 必须在 `D_waist=20 um` 生成一次 replay，并独立再生成一次 deterministic
repeat。必须同时满足：

```text
source hashes/state/path/plane/shape/axis/dtype/dx/FOV/origin/reference exact
raw complex relative L2 <= 1e-12
amplitude relative L2 <= 1e-12
amplitude-weighted phase-sensitive relative L2 <= 1e-12
candidate repeat raw relative L2 <= 1e-14
all arrays finite; target denominator > 0
```

gate 未通过时 Stage B/C 禁止执行；不得改 loss、mask、phase alignment 或 optimizer 掩盖。run 只保存 handoff
诊断并判为 `Inconclusive / artifact_operator_handoff_not_closed`，随后只允许定向检查 exp040→exp042→exp051 provenance。

## 9. Profile minimum 与 numerical-floor gates

coarse 与 fine profile 必须完整、有限、使用同一 primary loss。fine minimum 必须：

```text
unique within tolerance max(1e-20, 100 * replay_floor_loss)
strictly inside [16,24] um by at least one fine step
absolute error <= 0.125 um
minimum loss <= 1e-20
second-best loss / max(replay_floor_loss, float64 eps) >= 1e4
```

其中 `replay_floor_relative = max(raw replay error, deterministic repeat error, float64 eps)`，
`replay_floor_loss = replay_floor_relative**2`。candidate-forward waist signature 取
`min(sqrt(L(20um-h2)), sqrt(L(20um+h2)))`，要求 signature/floor `>=1e6`。这些 gate 区分参数 signature
与 bitwise/deterministic numerical floor；不使用 exp040 旧 detector floor。

## 10. Local Jacobian、curvature 与 FD convergence

在 truth point 用 raw complex fields 定义

$$
J_h=\frac{P(20\,\mu m+h)-P(20\,\mu m-h)}{2h},
$$

报告 normalized Jacobian norm `||J_h||/||P_target||`（单位 `m^-1`）和 loss curvature
`[L(D+h)-2L(D)+L(D-h)]/h^2`（单位 `m^-2`）。预注册 gate：

```text
normalized Jacobian norm at h1 >= 1e4 1/m
loss curvature at h1 finite and > 0
||J_h1 - J_h2|| / ||J_h2|| <= 0.35
both one-sided signatures finite and above replay floor
```

`h1=0.25 um` 提供稳定的中心差分，`h2=0.125 um` 对应 q8 subpixel lateral spacing 引起的 diameter scale
control。FD convergence 失败属于 working-model local derivative 未闭合，不得换 step 追求通过。

## 11. Bounded equal-budget multi-start optimizer

optimizer 是确定性一维 bounded pattern search：

```text
bounds: [16,24] um
starts: [16.5,18.5,21.5,23.5] um
initial pattern half-step: 2 um
evaluation budget: exactly 41 objective calls per start
update: evaluate clipped incumbent-step and incumbent+step; take lowest loss
tie break: retain incumbent, then smaller D if two new candidates tie
no improvement: halve step; improvement: keep step
minimum represented step: no early stop; continue to equal fixed budget
stopping reason: evaluation_budget for every start
```

所有 start 使用相同 bounds、算法、step rule、tolerance-free fixed budget 和 cache；保存每次 evaluation 的 candidate、
loss、cache index、incumbent 和 step。不得只保留最好 branch；最终代表 estimate 是四支 final estimate 的 median，
不是 truth-aided best branch。

## 12. Optimizer 与 accuracy gates

所有 branch 必须满足：

```text
exactly 41 objective calls and stopping_reason=evaluation_budget
final estimate absolute error <= 0.125 um
final estimate relative error <= 0.00625
distance to fine-profile minimum <= 0.125 um
not within 0.125 um of either fitting bound
```

此外四支 final estimate 的 max-min `<=0.125 um`，median estimate 与 fine-profile minimum 差
`<=0.125 um`。所有 branches 都进入 metrics/HDF5，无 seed 或 branch selection。

## 13. Passed、Failed、Inconclusive 互斥逻辑

按以下顺序决定唯一状态：

1. source identity/hash/state、plane/grid/reference 或 Stage A replay gate 未闭合，或正式 evidence 因不可恢复的 runtime/
   artifact 缺口不完整：`Inconclusive`；
2. Stage A 通过但 FD convergence、Jacobian numerical-floor separation 未闭合：`Inconclusive`；
3. Stage A/B/C 均产生有效有限 evidence，但 profile uniqueness/interior/accuracy、optimizer agreement/accuracy、boundary
   或 equal-budget gate 任一失败：`Failed`；
4. 所有预注册 gates 通过且 artifact audit 通过：`Passed`。

三种状态互斥。`Passed` 只能表述为 `single-parameter oracle fit passed within the selected exp040 scalar working model`。

## 14. HDF5 与 artifact contract

每次执行创建新的
`runs/exp051_TGV_3d_multislice_true_probe_waist_fit_<timestamp>/`，包含 `config.yaml`、`metadata.json`、
`metrics.json`、`run_state.json`、`outputs/exp051_true_probe_waist_fit.h5` 和 `figures/`。HDF5 保持 `/entry`
并列结构，至少包括：

```text
/entry/config_yaml
/entry/instrument/...
/entry/sample/sample_a/...
/entry/truth/P_B_true
/entry/truth/D_waist_true_m
/entry/truth/D_z_m
/entry/truth/z_m
/entry/reconstruction/waist_fit/design/...
/entry/reconstruction/waist_fit/cache/D_waist_m
/entry/reconstruction/waist_fit/cache/P_B_candidate
/entry/reconstruction/waist_fit/profile/...
/entry/reconstruction/waist_fit/finite_difference/...
/entry/reconstruction/waist_fit/optimizer/branches/...
/entry/reconstruction/waist_fit/P_B_best_raw
/entry/reconstruction/waist_fit/residual_field_raw
/entry/metrics/...
/entry/metadata/...
```

保存 source run/config/HDF5 paths 与 hashes、target dataset byte hash、all-false provenance flags。不得创建伪造的
calibration/preprocessing。PNG 只供人工检查；正式 profile、probe cache、tracks 和 residual 必须进入 HDF5。

## 15. Figures、runtime 与 memory contract

至少生成并审计：

```text
exp051_loss_profile.png
exp051_local_jacobian.png
exp051_optimizer_tracks.png
exp051_best_fit.png
```

图必须标 SI-derived `um`/dimensionless 单位且不以 wrapped phase 图代替数值结果。streamed multislice 不保存 full
`(nz,ny,nx)` volume；candidate cache 只保存 native `(96,96)` complex probes。预计单 run 小于 5 分钟、峰值工作内存
小于 1 GiB、HDF5 小于 128 MiB；超出只记录实际值，不改变科学阈值。

## 16. Artifact audit contract

formal 结束后必须独立检查：run state complete/validated；外部 JSON 与 HDF5 的 status、estimate、profile、FD、
optimizer counts/reasons 和 provenance flags exact；所有 numeric datasets finite；complex target/best/residual identity；cache
mapping 有效；HDF5 `/entry` tree 不含 calibration/preprocessing；四张 PNG 可读取、有限、无截断。artifact audit 失败按
第 13 节给出 `Inconclusive`，不得重写已有 run；修复后也创建新 timestamped run，除非只是只读审计代码错误。

## 17. exp053 handoff contract

exp053 必须复用本实验冻结的 source case、candidate generator、bounds、raw primary loss、full-field mask/reference
convention、coarse/fine grids、FD steps、equal-budget optimizer 和 failure flags；唯一输入变化是同一 exp042 matched case 的
raw q4/q4 `P_B_rec`。exp053 不得使用 truth-aligned copy。exp051 输出 artifact loader 所需的 source identities、cache
mapping 和 design group，但本实验绝不读取 `P_B_rec` 调参。exp051 成功只建立 oracle baseline，不能证明 reconstruction 后
仍可拟合。

## 18. exp055、exp041 与 exp04x 启发条件

面向 exp055，只记录候选 nuisance：`z_waist`、`D_top/D_bottom`、reference indices、`z_AB`、incident complex scale/
phase 和 grid-origin/reference convention；不在本实验联合拟合。profile 非对称、boundary sensitivity 或 FD direction 可作为
后续 preregistration 依据。

若 true-probe Jacobian/signature 接近 numerical floor或 profile 不唯一，把 probe-space `J_h`、residual direction 和失败 flags
反馈 exp041，以后再设计 sample-B/scan；本实验不 sweep B family/scan。exact replay/provenance 失败可定向质疑并最小完善
exp040/042；true-probe fit 通过而 exp053 matched raw `P_B_rec` 失败才构成恢复 exp042 的直接下游证据。

## 19. 明确未做事项、限制与停止扩张条件

未做：projected exp050、reconstructed-probe exp053、nuisance exp055、detector-intensity fitting、noise/model mismatch、
multi-waist cases、真实 data/calibration、arbitrary `D(z)`、full-wave/reference validation。阶段性假设包括轴对称、圆孔、
分段线性 `D(z)`、scalar monochromatic one-way propagation、q8 cell-average interface、periodic FFT transfer、固定 plane wave
和完全 matched operator。

以下任一出现即停止扩张：exact replay 未闭合；plane/grid/reference identity 不可确认；primary loss 必须引入未预注册
nuisance 才有 minimum；结果依赖事后 mask/bounds/alignment/start selection；需要改变 exp040 核心物理模型或项目级 HDF5
schema；需要转为 detector fitting、noise、model mismatch 或真实数据。保存诊断 artifacts 并按第 13 节判定，不能靠调参推进。

## 20. 2026-08-23 17:26：首次完整设计冻结证书

本节是首次完整构建完成后的第一个 EOF append。冻结对象从首个 `## 1.` 的 UTF-8 起始字节开始，到第 19 节
原始 EOF 结束，不包含可实时更新的第 0 节，也不包含本证书：

```text
frozen body bytes: 15946
frozen body SHA256: A499B6FB546F3A2DDB7B2840834F4E7C8770BD7DA87C7CFEBA9253E6180EEE6D
full file bytes before this certificate: 16981
last frozen section: 19
```

以后验证时，从当前文件的首个 `## 1.` 起读取前 `15946` 个 UTF-8 bytes，并核对上述 SHA256；新增记录只能
继续追加到真实 EOF。第 0 节长度变化不参与 frozen-body hash。

## 21. 2026-08-23：首次实现、formal run 与独立 artifact audit

### 21.1 本轮目标、明确未做事项与上一轮意见

本轮目标是在第 1--19 节预注册冻结后，完成第一个可判定的 3D scalar multislice、matched raw
`P_B_true`、单参数 `D_waist` oracle baseline：实现共享 candidate forward、exact replay、冻结 profile、局部
finite-difference/Jacobian control、equal-budget bounded optimizer、HDF5/JSON/figures 保存和独立审计，并按预注册
互斥规则给出 `Passed`、`Failed` 或 `Inconclusive`。

本轮明确未做 exp050、exp053、exp055、detector-intensity fitting、noise/model mismatch、nuisance fitting、真实数据或
full-wave/reference validation；没有读取 `P_B_rec` 作为 primary input，也没有用 truth 做 optimizer early stopping、分支
选择或超参数选择。首次 implementation iteration 没有上一轮 exp051 reviewer 意见；本轮用户要求及已冻结第 1--19 节
是 controlling specification。

### 21.2 开始检查与实际读取内容

开始时完整读取根目录 `AGENTS.md`，执行
`git -c safe.directory=E:/tgv_ptycho_sim status -sb`；dubious ownership 只在该单条命令中绕过，没有写全局 Git
配置。确认工作区已有 modified、deleted 和 untracked 内容并全部视为用户所有；本轮没有回退、删除或暂存它们。

按定向范围实际读取：exp051 预留全文、`phase.md`、`roadmap.md`、`data_format.md`、exp040 theory note 全文、exp040
指定状态/章节、exp042 指定章节、exp050/053/055 预留边界、exp042 YAML、matched q4 source run 的四个人工 artifacts，
以及 source HDF5 中指定 truth/sample/instrument/metadata/config datasets、attrs 和 tree。源码先用 `rg` 定位并按需读取
exp040 shared forward、multislice A、TGV geometry/3D object、inverse fitting/metrics、IO、exp040/042 runner handoff 和两组
对应测试；未无目标扫描 notebooks、reports、data、全部 runs 或 exp040 old 文档。

### 21.3 改动前判断、阈值冻结与 development 边界

source identity、B-plane/grid/reference convention 和 raw `P_B_true` 的 provenance 足以建立严格 loader；candidate 必须
复用 exp040 R8 scalar working-model operator。第 10--13 节 gates 在正式结果前冻结，特别是
`jacobian_step_relative_l2 <= 0.35`，FD 步长固定为 `h1=0.25 um`、`h2=0.125 um`。没有候选 profile 或 optimizer
formal 结果参与阈值制定；低成本 development 仅为单元测试、小型 synthetic fixture、配置类型和 artifact writer/validator
调试，不产生可作为科学证据的 run。正式运行后没有修改 loss、full-field mask、bounds、alignment、步长或阈值。

### 21.4 Changes

新增 `configs/experiments/exp051_TGV_3d_multislice_true_probe_waist_fit.yaml`、
`src/tgv_ptycho/inverse/exp051.py`、`scripts/run_exp051_true_probe_waist_fit.py` 和
`tests/test_exp051_true_probe_waist_fit.py`；扩展 `src/tgv_ptycho/inverse/waist_fit.py`。在
`src/tgv_ptycho/forward/exp040.py` 增加共享 `build_scalar_working_model_probe()`，并让
`src/tgv_ptycho/recon/exp042.py` 的 matched development case 委托该共享 helper，避免 exp051 复制 forward；该重构没有
改变 exp042 科学算法或输入输出语义。没有修改项目级 HDF5 schema，也没有修改 exp040/042 历史文档结论。

实现包括：source state/hash/dataset identity 强校验及 `P_B_rec` hard fail；raw complex normalized L2；exact replay 四项
控制；完整 coarse/fine profile；两步中心 FD/Jacobian、局部 curvature 和 signature/floor；四起点、每支 41 evaluations
的 fixed-budget bounded pattern search；121 个 native complex128 candidate cache；正式 artifacts、四图与只读 validator。

开发中发现并关闭四个工程问题：严格配置数组比较需要统一 float64；YAML scientific notation 必须解析成数值而非
字符串；Windows Conda 的 `Library/bin` DLL 搜索路径曾导致 Matplotlib `0xc06d007f`；validator 最初把布尔 provenance
字段 `P_B_rec_used=false` 误当成 `P_B_rec` dataset，随后改为只检查 leaf dataset name。它们均在唯一 formal run 前关闭，
没有改变科学 gate。

### 21.5 Operator 与 numerical consistency

Stage A 通过。用 exp051 candidate 路径在 true `D_waist=20 um` 重算得到 `(96,96)`、`complex128` B-plane field；与
source `/entry/truth/P_B_true` 的 shape、dtype、dx、FOV、origin、reference convention 全部相符。四项结果为：

```text
raw complex relative L2                         0.0
amplitude relative L2                           0.0
amplitude-weighted phase-sensitive relative L2 0.0
deterministic repeat relative L2                0.0
```

独立审计再次确认 source target 与 exp051 HDF5 target bitwise equal，source HDF5 hash 与预注册值一致。因此没有证据
质疑 exp040 -> exp042 -> exp051 artifact/operator handoff，exp040/042 保持科学结论不变。

### 21.6 Formal run、artifacts 与实际命令

唯一持久化 formal run 为：

```text
runs/exp051_TGV_3d_multislice_true_probe_waist_fit_20260823_180631
status: complete
artifacts_validated: true
runtime_seconds: 30.1168162
sampled peak RSS: 117346304 bytes
candidate cache count: 121
HDF5 bytes: 18786808
HDF5 SHA256: FDA70E0374AD15BA67C2605B1E8055C01199D5375843F0A65BCD4D9C7475E7C0
```

主要执行与验证命令：

```powershell
python scripts/run_exp051_true_probe_waist_fit.py --config configs/experiments/exp051_TGV_3d_multislice_true_probe_waist_fit.yaml
python -m pytest -q tests/test_exp051_true_probe_waist_fit.py
python -m pytest -q
python -m ruff check src/tgv_ptycho/forward/exp040.py src/tgv_ptycho/recon/exp042.py src/tgv_ptycho/inverse/exp051.py src/tgv_ptycho/inverse/waist_fit.py scripts/run_exp051_true_probe_waist_fit.py tests/test_exp051_true_probe_waist_fit.py
```

targeted tests 为 `6 passed in 3.38s`；scoped Ruff 为 `All checks passed!`。full pytest 为
`312 passed, 12 failed in 102.89s`；12 个失败全部是本任务开始前已经存在的 exp040 R10--R14 family frozen-config/
scientific-contract SHA lock mismatch，本轮没有改这些冻结配置或为通过测试而重写 hash，exp051/exp042 功能测试没有
回归。

### 21.7 Profile、Jacobian/FD 与 optimizer metrics

coarse 和 fine profile 的唯一内点 minimum 都是 `20 um`，minimum loss 为 `0.0`；second-best 是
`20.25 um`，loss `7.086068569797842e-4`，second-best/floor ratio
`3.1912815770463345e12`，profile gate 通过。完整 9 点 coarse、9 点 fine profile 和 cache mapping 已保存，未只保留
最优点。

局部控制结果为：

```text
||J_h1|| / ||P||                 68969.4150558 1/m  (h1 = 0.25 um)
||J_h2|| / ||P||                140161.721663  1/m  (h2 = 0.125 um)
jacobian step relative L2            0.730774315780
preregistered maximum                 0.35
loss curvature                  3.02105988824e10 1/m^2
h2 one-sided signature relative L2    0.0266373015308
signature / replay floor ratio   1.1996374124846794e14
```

signature 很强且远高于 replay floor，但 `J_h1` 与 `J_h2` 的步长收敛指标超过冻结上限，因此 FD/Jacobian gate 失败；
不能把强 profile 或 optimizer agreement 用来覆盖这个 numerical-control failure。

四个 starts 为 `16.5, 18.5, 21.5, 23.5 um`，全部使用相同算法、容差和 41-evaluation budget，停止原因全部是
`evaluation_budget`。final estimates 分别约为 `20, 20, 20, 20 um`；中位代表估计
`19.999999999999998 um`，absolute error `3.3881317890172014e-21 m`，spread
`6.776263578034403e-21 m`，无 boundary hit，optimizer gate 通过。所有分支和完整 tracks 均保留，没有只选最好分支。

### 21.8 HDF5/JSON/figure 独立审计

HDF5 `/entry` 顶层精确为 `config_yaml,data,instrument,metadata,metrics,reconstruction,sample,truth`；没有
`calibration` 或 `preprocessing` group，也没有 leaf dataset 名为 `P_B_rec`。target、best candidate 和 raw residual 均为
`(96,96) complex128`；candidate cache 为 `(121,96,96) complex128`；best probe 与 cache index 51 bitwise equal，且
`P_B_best_raw - P_B_true == residual_field_raw` exact。所有 numeric datasets finite，四支 optimizer 的 cache indices 有效。

外部 `metadata.json` 和 `metrics.json` 与 HDF5 对应 group 逐叶 exact，`config.yaml` 与 `/entry/config_yaml` exact；
`run_state.json` 中 `complete`、`artifacts_validated=true` 以及 config/metadata/metrics/HDF5/figure hashes 与文件复算一致。
四张 PNG 全部可读取、像素有限、标签和色条无截断：loss profile `1120x768`、local Jacobian `1920x608`、optimizer
tracks `1152x768`、best fit `1408x1200`。零 residual panel 的自动色条为对称小范围，raw phase difference 仅显示约
`1e-17 rad` 的浮点余量；这些图只供人工查看，正式数值均在 HDF5/JSON。

自然产生并审计的关键 HDF5 内容包括：instrument probe-grid/reference；sample-A geometry 和 sample-B source identity-only；
raw `P_B_true`、`D_waist_true_m`、`D_z_m`、`z_m`、slice widths；fit design/thresholds/source hashes；121-probe cache 与
映射；profile；两个 complex Jacobian fields 和 FD metrics；四支 optimizer tracks/counts/reasons；best raw probe、raw residual；
并列 metrics/metadata。没有伪造未执行的 calibration/preprocessing。

### 21.9 Formal 判定、限制与未关闭问题

预注册判定为互斥的 **Inconclusive**，interpretation 为
`local_jacobian_numerical_control_not_closed`。原因仅是 `0.730774315780 > 0.35` 的 FD step-convergence gate；exact replay、
profile 和 optimizer gates 均通过，artifact audit 通过。不能将结果提升为 Passed，也不能称为 waist 不可辨识或 fitting
Failed。更不能在看到结果后改变 FD steps、threshold、loss、mask、alignment 或 bounds。

当前证据只表明：在 selected exp040 scalar working model 下，true-probe waist signature 和冻结 profile 很强，optimizer
能精确回到 simulation truth，但 q8 cell-average/interface 离散下所选两个局部步长没有建立一致 Jacobian 极限。
`reference_validated=false`、`full_tgv_reference_authorized=false` 持续为 false；truth 仍只表示
`simulation truth under the selected exp040 scalar working model`。

最可能的后续 nuisance 仍是 `z_waist`、`D_top/D_bottom`、indices、`z_AB`、incident complex scale/phase、grid origin 和
reference convention；本轮没有联合拟合。由于 signature 并不弱，不触发 exp041 information-design sweep。由于 oracle
baseline 尚未 Passed，不启动 exp053；也不启动 exp055。

### 21.10 改动后优先级与下一轮快速恢复

下一步优先属于**继续 exp051 的新预注册 numerical-control iteration**，定向研究 q8 cell-average geometry 对
`D_waist` 的局部可微性/量化台阶，并在查看新结果前登记新的 step family、离散控制和判定规则；也可以把这一直接证据
反馈 exp040 的 R8 operator numerical semantics。若该工作需要改变核心 forward model、物理假设或项目级 schema，应停止
exp051 扩张并新开实验。当前没有证据恢复 exp042 reconstruction，也不应启动 exp053/055。

快速恢复时先读第 0 节、第 10--13 节和本节；复核 formal run 的 `metrics.json` 与
`/entry/reconstruction/waist_fit/finite_difference`；保留 `h1/h2=0.25/0.125 um` 和失败阈值作为本轮不可改写的历史证据。

### 21.11 Append-only verification 与 Git 状态

追加本节并只原位更新第 0 节后，重新从首个 `## 1.` 读取前 `15946` UTF-8 bytes，要求 SHA256 仍为
`A499B6FB546F3A2DDB7B2840834F4E7C8770BD7DA87C7CFEBA9253E6180EEE6D`；本节必须位于真实 EOF。若验证不符，必须
另加 correction note 而不得静默改写。

本轮没有执行 `git add`、commit、push、merge 或 PR；本任务文件保持 local unstaged/untracked。最终 Git status 仍需在
交付前只读复核，并继续区分用户原有修改与本任务变更。

## 22. 2026-08-23：最终环境复核 correction 与交付锁

### 22.1 目标、上一轮意见与明确未做事项

本轮只做第 21 节之后的最终解释器、test/lint、正文锁和 Git 复核，不修改科学实现、阈值或 formal artifacts。上一轮
结论是 formal `Inconclusive` 且 artifact audit 通过；本轮不尝试把它调成 Passed，也不启动 exp053/055 或恢复无关
exp040/042 工作。

### 22.2 检查、判断、Changes 与 numerical consistency

最终 shell 中直接执行未限定解释器的 `python -m pytest -q tests/test_exp051_true_probe_waist_fit.py` 时，发现 `python`
实际指向 `D:\anaconda3\python.exe`（Python 3.12.4、NumPy 1.26.4），而不是 AGENTS 指定的 Python 3.11 项目环境；结果为
`1 failed, 5 passed`，唯一差异是 exact replay raw relative L2 为 `7.450550977139077e-15` 而测试要求 bitwise zero。
同一 base 环境没有 Ruff。该结果说明未激活的 base FFT stack 不能冒充本项目验证环境，不改变已经保存的 formal run 或
预注册 scientific gate；它作为环境敏感性证据保留，不被静默删除。

随后用 `conda run -n tgv_ptycho_sim` 明确进入
`D:\anaconda3\envs\tgv_ptycho_sim\python.exe`（Python 3.11.15、NumPy 2.4.6），不改任何文件，复核为：

```text
targeted pytest: 6 passed in 3.36s
scoped Ruff: All checks passed!
full pytest: 312 passed, 12 failed in 99.47s
```

12 个 full-suite failure 与第 21 节相同，精确为 exp040 R10 stage-A 的 2 项、R10 stage-B、R10 stage-B preflight、
R11、R11 preflight、R12、R13、R14、R14a 和 R14b 的 2 项 frozen config/scientific-contract SHA mismatch。它们不涉及
exp051 tests；本轮不修复用户/历史 exp040 hash locks。正式 run、exact replay、profile、FD/Jacobian、optimizer 和 HDF5
metrics 均未重算或改变。

### 22.3 Artifacts、未关闭问题、优先级与快速恢复

没有新 development/formal run；latest valid run 仍是
`runs/exp051_TGV_3d_multislice_true_probe_waist_fit_20260823_180631`，其 artifact audit 和关键 metrics 仍以第 21 节为准。
未关闭科学问题仍只有 `local_jacobian_numerical_control_not_closed`。下一优先级仍是继续 exp051 的新预注册 q8/local-
differentiability control，或将定向证据反馈 exp040；不启动 exp053/055，也不反馈 exp041 information design。

快速恢复时必须使用名为 `tgv_ptycho_sim` 的 Python 3.11 环境；未激活 base 结果只能作为跨环境数值敏感性诊断，不得
替代项目验证基线。仍保持 `reference_validated=false`、`full_tgv_reference_authorized=false`。

### 22.4 Append-only verification 与 Git 状态

本轮只原位更新第 0 节的 authoritative section 并在 EOF 追加本节。冻结正文的前 `15946` bytes/SHA256 仍须精确匹配
第 20 节证书。最终 `git diff --cached --name-only` 为空；未执行 `git add`、commit、push、merge 或 PR。task 文件仍为
local unstaged/untracked，用户既有 modified/deleted/untracked 文件未被回退、删除或暂存。

## 23. 2026-08-24：q8 local-differentiability control 预注册

### 23.1 研究问题、进入依据与不变边界

本节写在任何新 local-control candidate、breakpoint 或 chord-control 结果产生之前。上一轮唯一未闭合项是固定 q8
midpoint interface 下 `h=0.25/0.125 um` 两个 central Jacobian 的 relative L2 为 `0.730774315780 > 0.35`；同时
exact replay 为零、profile 唯一且 optimizer agreement 通过。新问题固定为：该失败是否来自 q8 subpixel midpoint
indicator 对连续 `D_waist` 的分段常数/跳变离散，而不是 source handoff、probe signature 过弱或 optimizer failure？

本轮仍属于 exp051 numerical-control iteration，不建立新 fit，不重跑上一轮 optimizer，不读取 `P_B_rec`，不改变 raw
complex loss、mask、bounds、reference convention、sample-A 连续几何、传播算子或原 gates。原 exp051 scientific status
无论本轮结果如何都保持 `Inconclusive`；本轮只给出独立的 `DiagnosticPassed/DiagnosticFailed/DiagnosticInconclusive`。
`reference_validated=false` 与 `full_tgv_reference_authorized=false` 不得提升。

### 23.2 冻结 source 与 provenance

primary target 仍是 exp042 matched q4 run 的 raw `/entry/truth/P_B_true`，其 identity 和 hashes 沿用第 2--3 节。新诊断
还必须登记并校验上一轮 exp051 formal artifacts：

```text
source exp051 run: runs/exp051_TGV_3d_multislice_true_probe_waist_fit_20260823_180631
config SHA256:     320046672D5BEC1BBF0714763B0FB0C7639C979BFD1AD82C887AE2E389AA73C0
metadata SHA256:   F1CDAA1E894420CE5C2307FA57C4E1CB8E3619090EB4B09354D940A02623B999
metrics SHA256:    4ED867067D3664CB68FE8DA37B62D7DA3F6A544E545E8E45EA849B44C426522A
run_state SHA256:  7F4CF653B28FD939725F857834A81439D6817F5741029FEA5BE7A01FDB30225D
HDF5 SHA256:       FDA70E0374AD15BA67C2605B1E8055C01199D5375843F0A65BCD4D9C7475E7C0
required prior interpretation: local_jacobian_numerical_control_not_closed
```

若任一 hash、state、target bytes、q8 replay 或 all-false provenance 不符，hard fail，不运行后续控制，也不质疑
`D_waist` 可辨识性。

### 23.3 q8 breakpoint 的解析定义

q8 midpoint rule 在每个 lateral pixel 内使用 `8x8` 固定 nodes，air fraction 是圆内 node count 除以 64。对 slice
center `z_j`，piecewise-linear diameter 可写为

```text
D_j(D_waist) = c_j + alpha_j D_waist,  alpha_j > 0.
```

每个 q8 node 半径 `r_k` 的 inclusion 在 `D_j=2 r_k` 处改变，因此候选 breakpoint 固定为

```text
D_waist_break(j,k) = (2 r_k - c_j) / alpha_j.
```

只保留原 bounds `[16,24] um` 内 finite breakpoints；不得用 probe 结果反推或移动 breakpoint。登记 truth 下方最近
`b_minus`、上方最近 `b_plus`、两侧 gap、每个 breakpoint 的 slice/node multiplicity，以及 equality-at-truth flag。
`<=` boundary convention 保持 source 实现，不做人为 epsilon shift。

由解析 gap 自动派生且在结果前固定以下 probes：

```text
inside_minus = truth - 0.5 * (truth - b_minus)
inside_plus  = truth + 0.5 * (b_plus - truth)
cross_minus  = truth - 1.5 * (truth - b_minus)
cross_plus   = truth + 1.5 * (b_plus - truth)
```

若 1.5 倍 crossing 越过 bounds，hard fail，而不是换系数。inside 两侧必须与 truth 的全部 100 个 q8 fraction slices
bitwise equal，且 B-plane probe raw relative L2 为零；cross 两侧必须出现对应 node-count 变化和非零 raw probe response。

### 23.4 冻结 halving step family 与保存量

step family 在上一轮 `0.25/0.125 um` 的基础上向两侧扩一档并继续二分，选择理由是 q8 lateral node spacing
`dx/q=0.0625 um`，需要跨越和低于可能的 radial breakpoint gap：

```text
h_um = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625]
```

每个 h 都生成固定 q8 `P(D_true-h)` 和 `P(D_true+h)`，不对 phase/scale/shift 对齐。保存 raw one-sided relative L2、
raw loss、central complex `J_h`、`||J_h||/||P_true||`、相邻 halving pair 的 Jacobian relative L2、second-difference
asymmetry、changed fraction-voxel count、changed subpixel-node count、signed discrete air-volume change和 fraction-stack
relative L2。保存完整 q8 candidate probe stack、所有 complex `J_h` 和 diameter-to-cache mapping，不只保留图或摘要。

确定性 repeat 对 truth、inside 和最小 h pair 都要求 raw relative L2 `<=1e-14`。不能因某个 h 更好看而删除其他步长。

### 23.5 chord-cell control 的身份与 gates

仓库已有 `make_tgv_air_fraction_slice_chord_quadrature()`：它对 analytic circular chord 做 Gauss--Legendre cell-area
积分，不是新的材料 effective-medium law。它只作为非 primary numerical comparator；不能替换 q8 source target、不能用于
重新拟合 truth，也不能把与 q8 target 的 mismatch 解释成 measurement error。

control 固定为同一连续 `D(z)`、shape/dx/dz、multislice/AB propagation 和 reference convention，仅把 interface builder
改为 chord-cell GL64。对同一 halving step family 保存 chord candidate stack、central `J_h`、norm、相邻 convergence 和
asymmetry。另在全部 100 slices 对 GL64/GL128 air-fraction geometry 作 order control：stack relative L2 与 integrated
air-volume relative error 都必须 `<=1e-5`。chord B-plane derivative 的最后两个 halving steps 必须满足：

```text
relative L2(J_0.0078125um, J_0.00390625um) <= 0.10
minimum of their normalized Jacobian norms >= 1.0e4 1/m
truth deterministic repeat <= 1.0e-14
```

这些阈值来自 inverse local-linearization 需要和既有 exp040 chord geometry `1e-5` order budget，在任何本轮 chord
field 结果前固定。若 GL64 geometry order 失败，不运行 chord field；q8 attribution仍可独立保存，但总诊断最多
`DiagnosticInconclusive`。

### 23.6 互斥诊断状态与 failure logic

状态固定为：

1. source/replay/provenance、breakpoint ordering、inside exact 或 crossing observation 任一 hard control 失败：
   `DiagnosticFailed`，interpretation 指向 artifact/breakpoint implementation mismatch；
2. q8 hard controls 通过并证明 truth 位于非零宽度 constant plateau，但 chord geometry/derivative convergence 任一未闭合：
   `DiagnosticInconclusive`，interpretation 为 `q8_plateau_attributed__smooth_control_not_closed`；
3. q8 hard controls与 chord controls 全部通过：`DiagnosticPassed`，interpretation 为
   `q8_midpoint_piecewise_constant_at_truth__chord_control_locally_converged`；
4. 如果 q8 没有 nonzero plateau、step family 呈收敛 Jacobian 且 breakpoint 计算也一致，则
   `DiagnosticInconclusive`，因为上一轮 failure 需要另找原因，不能看后发明新 gate。

即使 `DiagnosticPassed`，原 exp051 仍是 `Inconclusive`；它只关闭 failure attribution，并为未来另行预注册可微
candidate representation 或 non-smooth identifiability criterion 提供依据。

### 23.7 实现、artifact 与资源 contract

可复用 breakpoint/step diagnostics 放 `src/tgv_ptycho/inverse/`；shared scalar generator 只允许增加显式 injectable
air-fraction builder，默认必须仍是 q8 midpoint 且 exact replay 不变。新 runner 只负责编排、run 创建、保存和审计。
新 YAML、runner 和 tests 名称使用 `exp051 ... local_differentiability_control`，不覆盖上一轮配置或 run。

新 run 固定为
`runs/exp051_TGV_3d_multislice_true_probe_waist_fit_q8_local_control_<timestamp>/`，至少含
`config.yaml/metadata.json/metrics.json/run_state.json/outputs/exp051_local_differentiability_control.h5/figures`。HDF5
保持 `/entry` 并列结构，诊断数据放 `/entry/data/local_differentiability`，另含 instrument/sample/truth/metrics/metadata；
不得创建 reconstruction、calibration 或 preprocessing 的伪结果。至少三图：breakpoint/response、q8-vs-chord Jacobian
series、representative complex-Jacobian fields。JSON/HDF5 同义数据必须 exact；所有 numeric datasets finite。

预计少于 3 分钟、peak RSS 小于 1 GiB、HDF5 小于 128 MiB。正式执行前先通过 targeted tests 和 scoped Ruff；只创建
一个 formal timestamped run，失败也保留，不调参重跑。

### 23.8 exp04x 手工反馈触发规则与停止扩张

只有 q8 breakpoint/plateau hard controls 通过，才在 exp051 EOF 结果章节写一个独立的“供用户手工转录至 exp04x”块，
至少包含 source model、breakpoint gaps、zero plateau、step-series、chord control 和结论边界。预期有价值的反馈不是否定
R7 的 q4->q8 finite-perturbation convergence，而是明确：该 convergence 不自动授权 q8 midpoint operator 用于连续参数的
local Jacobian；forward accuracy 与 inverse differentiability 是不同 gate。Codex 本轮不编辑 exp04x 文档。

若需要改变 exp040 核心物理模型、项目级 schema、target probe、primary loss，或转成 exp053/P_B_rec、nuisance/noise/
detector fitting，则停止本轮并另开预注册问题。

### 23.9 本轮预注册证书与开始时 Git 状态

本节追加前 exp051 文档为 `32862 bytes`，完整 SHA256 为
`63C084BA2C8A2D803F2828DA4BA04B0175A3AE385255AFE2108182BAD4162B8E`。第 20 节冻结的 15946-byte prefix 仍是
authoritative fixed-body lock；本轮结果只能追加在本节之后。

开始时再次完整读取 `AGENTS.md`，并用单条 safe-directory status 检查工作区。用户既有 modified/deleted/untracked
内容保持不动；本轮不执行 `git add`、commit、push、merge 或 PR。

## 24. 2026-08-24：local-control 实现、preflight 与 formal execution lock

### 24.1 本轮目标、上一轮意见和明确未做事项

本轮执行第 23 节冻结的实现与 development/preflight，只决定唯一 formal 是否可运行。上一轮意见是继续 exp051 的
q8/local-differentiability control，并把有价值的 exp04x 反馈写在本实验文档供用户手工转录；本轮没有编辑任何 exp04x
文档。明确未做新 fit、P_B_rec/exp053、nuisance/exp055、sample-B/scan exp041、noise、detector fitting 或物理模型变更。

### 24.2 开始检查、读取与改动前判断

重新完整读取 `AGENTS.md`，用单条 `git -c safe.directory=E:/tgv_ptycho_sim status -sb` 核对混合工作区；读取 exp051
第 0、21--23 节和 q8/chord 相关 shared source、R7/R8/R11 定向文档证据。`rg` 定位确认：R7 已证明 q4->q8 在固定有限
扰动上的 forward convergence，R11 已有 analytic chord-cell GL64/128 geometry control；但现有 exp04x 记录没有把
`D_waist -> P_B` 的 fixed-q8 local differentiability 当作 inverse gate。改动前判断仍是区分 forward finite-perturbation
accuracy 与 continuous-parameter derivative consistency，不否定 R7/R8，也不把 chord control 升为 exp051 primary model。

### 24.3 Changes 与 operator consistency

新增 local-control YAML、`src/tgv_ptycho/inverse/exp051_local_control.py`、新 runner 和新 tests。共享
`build_scalar_working_model_probe()` 只增加显式 `air_fraction_builder/interface_resolution` numerical-control injection；
默认仍调用 q8 midpoint builder，原 exp042/exp051 candidate path 未变。`make_exp051_candidate_generator()` 同步暴露该
可选注入。`save_ptycho_hdf5()` 增加可选 `data` mapping，用于自然写入既有 `/entry/data`，没有改变项目级 HDF5 tree。

实现了 fixed-node analytic breakpoint map、动态 inside/cross probes、8-step q8/chord complex-Jacobian series、q8
fraction/node/volume transition metrics、chord GL64/128 order control、互斥 diagnostic status、完整 caches/HDF5/JSON/
三图和 artifact validator。默认 q8 true-waist replay 的 array equality 回归测试通过，证明 injection API 没有改变 source
operator。

formal 前锁定文件：

```text
config  bytes/SHA256: 5623 / 357ED1A05265A22AF98F6CE89A6A78AE31393746D13C410A639CD044B336CAAF
module  bytes/SHA256: 27185 / 07D0D3F77578C4CB5C909E713B469624E6B189BED8A5D5EE4195A7907701F05E
runner  bytes/SHA256: 23147 / 98A706085B4C53A4A95D8BE0712C9CE6DACBC132C353E76BF7FEE806CFE5CC0B
tests   bytes/SHA256: 9470 / 9EF72558BB2BB9589D6A103EFB78B6C26EF123437192D8782B8FE3526CECF501
```

### 24.4 Development/preflight、tests 与 lint

development 先只计算 breakpoint map，得到 283665 个 bounds 内 unique breakpoints；随后测试 fixture 执行完整 control
用于验证数组、gate 和 artifact writer。所有第 23 节阈值在这些结果前已冻结，下面任何结果都没有用于删 step、改
threshold 或改变状态逻辑：

```text
lower / upper truth gap: 0.0272989471 nm / 0.143340469 nm
inside-minus / inside-plus q8 fraction and probe: bitwise exact
cross-minus / cross-plus changed q8 nodes: 48 / 16
cross-minus / cross-plus raw probe L2: 5.61466e-5 / 7.43675e-5
q8 final halving-pair J relative L2: 0.505751654
chord GL64/128 fraction-stack L2: 3.09028e-10
chord GL64/128 volume relative error: 1.36709e-11
chord final halving-pair J relative L2: 0.0223858172
development diagnostic status: DiagnosticPassed
```

q8 的 8 个 `zero_pair` 全为 false，因为最小固定 step `3.90625 nm` 仍远大于 sub-nm plateau；动态解析 inside probes
按预注册提供了真正 local-constant exact control。chord 最后两档 normalized Jacobian 为约 `3.12106e5/3.16835e5 1/m`，
通过 `>=1e4 1/m` gate。

实际验证为：

```text
new local-control tests: 5 passed in 22.37s
combined original+new exp051 tests: 11 passed in 24.13s
scoped Ruff: All checks passed!
full pytest: 317 passed, 12 failed in 126.83s
```

full-suite 12 项仍精确是既有 exp040 R10 stage-A 两项、R10 stage-B、R10 stage-B preflight、R11、R11 preflight、
R12、R13、R14、R14a 和 R14b 两项 frozen-config/scientific-contract SHA mismatch；没有新增 exp051/exp042/IO
functional failure，也没有修复历史 hash locks。

### 24.5 Formal authorization、artifact contract 与停止规则

source/prior hashes、default-q8 exact replay、breakpoint implementation、synthetic HDF5/figures validator、targeted tests 和
Ruff 均通过，因此授权按 config SHA `357ED1...CAAF` 执行一次 formal。命令固定为：

```powershell
conda run -n tgv_ptycho_sim python scripts/run_exp051_local_differentiability_control.py --config configs/experiments/exp051_TGV_3d_multislice_local_differentiability_control.yaml
```

formal 不得因结果与 development 不同而重跑或修改 config；失败也保存该 timestamped run。结果必须在 EOF 新章节记录，
并独立审计 JSON/HDF5/figures。exp04x 手工反馈块只有 formal q8 hard controls 通过才允许写。

### 24.6 Append-only verification、Git 与快速恢复

本节追加前文档为 `42462 bytes`，完整 SHA256 为
`03D636F04DFA4BD75C04A23ABBC8322EBC4939E935A2C4EE7A9D3CA4307DECF7`。第 20 节 frozen prefix 不变；本节未回改
第 23 节 gates。没有执行 Git staging/commit/push/merge/PR，用户既有工作区内容保持不动。

快速恢复时先核对 config SHA、combined 11-test result、full-suite 既有 12 failures 和本节 formal authorization；不要把
development run 当正式科学 artifact。

## 25. 2026-08-24：q8 local-control formal 结果、审计与 exp04x 手工反馈

### 25.1 本轮目标、上一轮意见与明确未做事项

本轮只执行第 24 节授权的唯一 formal、独立 artifact audit 和结论记录。上一轮已锁定 config/code/tests hashes、
`DiagnosticPassed` development evidence 和不调参规则。没有修改 config、step family、threshold、builder、plot selection 或
status logic；没有重跑旧 exp051 fit、使用 `P_B_rec`、启动 exp053/055/041，或编辑 exp04x 文档。

### 25.2 Formal run 与 operator/numerical consistency

实际命令：

```powershell
conda run -n tgv_ptycho_sim python scripts/run_exp051_local_differentiability_control.py --config configs/experiments/exp051_TGV_3d_multislice_local_differentiability_control.yaml
```

唯一 formal run：

```text
runs/exp051_TGV_3d_multislice_true_probe_waist_fit_q8_local_control_20260824_134414
run state: complete
artifacts_validated: true
experiment_status: Inconclusive
diagnostic_status: DiagnosticPassed
interpretation: q8_midpoint_piecewise_constant_at_truth__chord_control_locally_converged
runtime: 19.3940182 s
sampled peak RSS: 167161856 bytes
```

执行前 source YAML SHA256 仍为预锁定的 `357ED1...CAAF`；run 内语义相同、规范化重写的 `config.yaml` SHA256 为
`0F673A...6660`，`load_config(run/config)==executed config` 且 `/entry/config_yaml` 与 run config text exact。exp042 raw
`P_B_true` 与新 HDF5 target bitwise equal；q8 true replay 与所有登记 deterministic repeats 为 `0.0`。因此本轮没有引入
新的 exp040->exp042->exp051 handoff 问题。

### 25.3 q8 breakpoint、plateau 与完整 step series

在原 `[16,24] um` bounds 内登记 283665 个 unique q8 fixed-node breakpoints。truth 两侧最近点为：

```text
lower breakpoint: 19.999972701052885 um, gap 0.0272989471164 nm
upper breakpoint: 20.000143340468626 um, gap 0.143340468625 nm
lower / upper node multiplicity: 8 / 16
lower slice index: 38
upper slice indices: 43, 56
equality-at-truth multiplicity: 0
q8 subnode spacing: 62.5 nm
```

解析派生的 inside-minus/inside-plus 在全部 100 fraction slices 和 B-plane raw complex probe 上均 bitwise exact；cross-minus
跨过 48 个 q8 nodes、probe L2 `5.61465625e-5`，cross-plus 跨过 16 nodes、probe L2 `7.43674819e-5`。因此 truth
确实处于非零宽度的 q8 constant plateau，breakpoint prediction、inside exact 和 crossing observation gates 全部通过。

从 `h=0.5` 到 `0.00390625 um` 的 8 个 q8 normalized Jacobian norms 为：

```text
[42324.8, 68969.4, 140162, 185229, 232678, 316043, 329083, 377559] 1/m
```

相邻 halving-pair relative L2 为：

```text
[0.618367, 0.730774, 0.931286, 0.510906, 0.388713, 0.454449, 0.505752]
```

没有向稳定 derivative 收敛。固定 step family 的 8 个 `zero_pair` 均为 false，因为最小 `3.90625 nm` 仍是较窄一侧
plateau gap 的 143 倍；这不与动态 sub-nm inside probe 的 exact-zero 结果矛盾。原 `0.25/0.125 um` failure 被完整复现，
且不再只是“两步选得不好”的未定位现象。

### 25.4 非 primary chord-cell control

GL64 对 GL128 的 100-slice fraction-stack relative L2 为 `3.09028363e-10`，integrated air-volume relative error 为
`1.36708595e-11`，远低于冻结的 `1e-5` gates。chord GL64 的 normalized Jacobian norms 为：

```text
[41035.4, 65334.9, 138956, 201454, 247650, 294625, 312106, 316835] 1/m
```

halving convergence 为：

```text
[0.595049, 0.699525, 0.913945, 0.542530, 0.286247, 0.0855672, 0.0223858]
```

最后一对 `0.0223858 <=0.10`，最后两档 norm 均 `>=1e4 1/m`，chord geometry/derivative gates 通过。这表明同一连续
圆孔几何与同一 propagation 下，连续 cell-area representation 可以建立局部 derivative，而 fixed-q8 midpoint count
不能。chord-center 与 q8-center probe raw L2 为 `0.00301833`；所以 chord 不能事后替换 q8 candidate 去拟合原 q8 target，
否则 exact replay 将被破坏。它在本轮始终只是 attribution control。

### 25.5 HDF5、JSON、figures 与资源审计

HDF5 为 `13026504 bytes`，SHA256
`2F8ED89B335E31F2E195E3ABBCD5A37C69B387F75949381D1A8A90756710D8B1`。`/entry` 精确为：

```text
config_yaml, data, instrument, metadata, metrics, sample, truth
```

`/entry/data` 精确只有 `local_differentiability`；无 reconstruction/calibration/preprocessing，也无 leaf dataset
`P_B_rec`。自然保存了 283665 breakpoints/multiplicities、动态 plateau/cross controls、q8/chord 各 `(17,96,96)`
complex128 candidate cache、各 `(8,96,96)` complex128 Jacobian stack、完整 step/geometry metrics、selected-slice
q8/GL64/GL128 fractions、design 和 gates。全树 331 个 group/dataset path，所有 numeric datasets finite；target 为
`(96,96) complex128`。

`metadata.json` 与 `/entry/metadata`、`metrics.json` 与 `/entry/metrics` 逐叶 exact，external target 与 source target
bitwise exact。三张 PNG 均实际打开检查：breakpoint response、q8-vs-chord convergence 和代表 Jacobian fields 标签/单位
清晰、无截断、像素 finite。runtime、memory、HDF5 size 均在预注册 contract 内。

run artifacts hashes：

```text
config.yaml   0F673AA3AE9FA035054A5B9E2DC7220D2461FEDE95983E8FC3A6C2051B446660
metadata.json 15990370D6B231EA35BA7233987AA55E5917DBBDC3F101F2B6B861972191CD05
metrics.json  A3C84876BE9FED338B05C767D663E6958F746541F5F477BA11DE600E483F2AC0
run_state     D30A300B830F8CCF6013D6419849032CD66A5F6BD01A6BAF87EC9BC2A25E939D
```

### 25.6 判定、限制与未关闭问题

本轮按预注册互斥规则为 **DiagnosticPassed**，关闭了上一轮 FD/Jacobian failure 的主要 attribution：固定 q8 midpoint
count operator 对连续 `D_waist` 是分段常数且在 breakpoints 处跳变；原两个 Jacobian 不是一个光滑 forward 的一致
derivative estimate。原 exp051 scientific status 仍是 **Inconclusive**，不能回写为 Passed，也不能宣称 full-wave、真实
器件或真实计量验证。

这个结论不否定 q8 对有限 `D_waist` 扰动的 forward visibility，也不否定 R7 q4->q8 的既定有限差异 convergence；它只
说明那类 gate 不足以授权 continuous-parameter local Jacobian。plateau 总宽约 `0.17064 nm`，远小于原 `0.125 um`
accuracy budget，因此 q8 仍可能适合 derivative-free/profile-set inverse；但必须另行预注册 plateau-equivalence set、
estimator resolution 和 non-smooth failure logic，不能用本轮结果自动宣布 baseline Passed。

另一条路线是让 upstream 产生 matched chord-cell truth/candidate artifacts；那会改变 source operator，必须在 exp04x/
exp042 handoff 中显式版本化并重跑，不能在 exp051 内静默替换。当前不启动 exp053，因为其 matched raw `P_B_rec` 仍来自
q8 source；也不启动 exp055/041。

### 25.7 供用户手工转录至 exp04x 的反馈块

以下文字是本轮允许产生的定向反馈；Codex 没有编辑 exp04x 文档：

> **来自 exp051 的 inverse-differentiability 反馈（2026-08-24）**  
> source 为 exp042 matched raw q8 `P_B_true`，exp051 exact replay 为 0；证据 run 是
> `runs/exp051_TGV_3d_multislice_true_probe_waist_fit_q8_local_control_20260824_134414`，HDF5 SHA256
> `2F8ED89B...D8B1`。在 `96^2 @ 0.5 um`、100 slices、q8 midpoint cell-average 下，解析枚举 283665 个
> `[16,24] um` 内 `D_waist` breakpoints；truth 最近下/上 gap 仅 `0.02730/0.14334 nm`。两侧 half-gap 内全部 q8
> fractions 与 `P_B` bitwise 不变，跨 breakpoint 后分别改变 48/16 个 subpixel nodes，并产生
> `5.61e-5/7.44e-5` raw probe L2。`h=0.5...0.00390625 um` 的 q8 halving-Jacobian convergence 始终为
> `0.389...0.931` 量级，未形成 local derivative。非 primary analytic chord-cell GL64/128 geometry error 为
> `3.09e-10`、volume error `1.37e-11`，chord Jacobian 尾部 convergence 为 `0.0856 -> 0.0224`，显示同一连续几何
> 在连续 cell-area representation 下可局部收敛。结论边界：R7 q4->q8 finite-perturbation forward convergence 不应
> 回写为错误；但它不自动授权 fixed-q8 operator 用于连续参数 Jacobian。未来 Phase 5 要么对 q8 明确使用 non-smooth/
> plateau-set identifiability gate，要么由 exp04x 显式版本化 continuous cell-average operator 并重新生成 matched truth/
> reconstruction handoff；不得把 chord candidate 静默用于现有 q8 target。该证据仍是 selected scalar working model，
> `reference_validated=false`、`full_tgv_reference_authorized=false`。

### 25.8 改动后优先级与下一轮快速恢复

下一优先级仍属于 exp051，但必须先选择并预注册：建议最小扩张是保持 q8 source，定义 exact/near-exact plateau-equivalence
set、reported interval estimator、interval width相对 `0.125 um` tolerance 和 multi-start interval agreement；这可回答
derivative-free oracle baseline 是否可判定。若决定采用 chord-cell primary truth，则回到 exp04x/042 新 source-version
任务，不在当前 exp051 直接推进。exp053/055/041 均继续等待。

快速恢复时读第 23--25 节和 formal `metrics.json`，特别区分 `experiment_status=Inconclusive` 与
`diagnostic_status=DiagnosticPassed`；不要把 chord control 当 primary input。

### 25.9 Append-only verification 与 Git 状态

本节追加前文档为 `48065 bytes`，完整 SHA256 为
`B256A2ED9E97D59C7B08156FD0DE9F158724AAC20F9100925DE6CDDCCD19E970`。追加后仍需复核第 20 节 15946-byte
frozen-prefix hash；第 23--24 节未回改。没有执行 `git add`、commit、push、merge 或 PR；所有本任务改动保持 local
unstaged/untracked，用户既有 modified/deleted/untracked 文件不动。

## 26. 2026-08-24：q8 plateau-equivalence interval estimator 与 non-smooth gate 预注册

### 26.1 本轮目标、上一轮意见与明确未做事项

第 25 节已经把原 finite-difference failure 归因为 fixed-q8 midpoint operator 在 `D_waist` 上的分段常数/跳变语义；
其下一步意见是继续 exp051，保持 matched q8 source，改用 exact/near-exact plateau-equivalence set、区间估计和
non-smooth identifiability gate。本节在任何新 interval candidate、阈值控制、development 或 formal 结果产生前冻结该
方案。研究问题固定为：在不改变 q8 truth/operator、raw complex loss 和单参数边界的条件下，能否把 target-compatible
`D_waist` 集合稳定地估计为一个不宽于原 `0.125 um` accuracy budget 的区间，并由全部 equal-budget starts 得到同一
区间？

本轮只完成预注册和文档记录，未实现代码、未运行测试或实验、未创建 run。明确不把 chord-cell 改成 primary truth，
不读取 `P_B_rec`，不启动 exp053/055/041，不引入 phase/scale alignment、noise、nuisance、detector fitting 或新物理模型，
也不回改第 1--25 节的历史 gates/results。旧 q8 Jacobian non-convergence 继续是有效历史证据；新 gate 不是宣称它已经
收敛，而是为已确认的 non-smooth operator 使用与其数学语义相符的 set-valued criterion。

### 26.2 开始检查、实际读取和阈值信息边界

开始时核对仓库级 `AGENTS.md` 规则并执行单条
`git -c safe.directory=E:/tgv_ptycho_sim status -sb`；混合工作区中的 modified/deleted/untracked 内容仍全部视为用户所有。
本轮完整重读第 23--25 节，并只读核对最新 q8 local-control formal `metrics.json` 及五个 artifact hashes。允许参与本轮
阈值制定的既有结果只有：source/replay/repeat 均为 `0`；truth plateau 两侧 gap 为 `0.0272989471/0.143340469 nm`；
plateau 总宽约 `0.170639416 nm`；已登记 cross probes 的最小 raw relative L2 为 `5.61465625e-5`；原 accuracy budget
为 `0.125 um`；四起点旧 fit/profile 已到达同一 truth neighborhood。没有生成或查看新的相邻 cell、interval expansion、
bisection 或 multi-start interval 结果。

### 26.3 冻结 source、scope 和不可替换 operator

primary target、sample-A、grid、plane、reference convention、bounds `[16,24] um`、full-field equal weighting 和 raw
complex normalized loss全部沿用第 2--7 节。target 仍只能是：

```text
runs/exp042_TGV_3d_multislice_probe_reconstruction_dev_20260822_195139
outputs/exp042_probe_reconstruction.h5:/entry/truth/P_B_true
operator: exp040 R8 fixed-q8 midpoint scalar working model
truth identity: simulation truth under the selected exp040 scalar working model
```

本轮还冻结第 25 节 q8 attribution artifact 为 required prior：

```text
run: runs/exp051_TGV_3d_multislice_true_probe_waist_fit_q8_local_control_20260824_134414
config SHA256:   0F673AA3AE9FA035054A5B9E2DC7220D2461FEDE95983E8FC3A6C2051B446660
metadata SHA256: 15990370D6B231EA35BA7233987AA55E5917DBBDC3F101F2B6B861972191CD05
metrics SHA256:  A3C84876BE9FED338B05C767D663E6958F746541F5F477BA11DE600E483F2AC0
run_state SHA256:D30A300B830F8CCF6013D6419849032CD66A5F6BD01A6BAF87EC9BC2A25E939D
HDF5 SHA256:     2F8ED89B335E31F2E195E3ABBCD5A37C69B387F75949381D1A8A90756710D8B1
required state: complete / artifacts_validated=true / DiagnosticPassed
required interpretation: q8_midpoint_piecewise_constant_at_truth__chord_control_locally_converged
```

任一 source hash/state、raw target bytes、default-q8 exact replay、plane/grid/reference identity 或 all-false provenance flag
不符时禁止 interval fit。chord GL64/128 只保留为第 23--25 节的 attribution evidence，本轮不得调用 chord builder 生成
primary candidate。`reference_validated=false` 与 `full_tgv_reference_authorized=false` 始终不得提升。

### 26.4 Plateau-equivalence set 的冻结定义

记 fixed-q8 candidate field 为 `F_q8(D)`，target 为 `T`，原 primary loss及其平方根为

$$
L(D)=\frac{\lVert F_{q8}(D)-T\rVert_2^2}{\lVert T\rVert_2^2},\qquad
\rho(D)=\sqrt{L(D)}.
$$

记 `G(D)` 为全部 100 slices 的 q8 air-fraction/node-count fingerprint。第 23 节的解析 breakpoints 加上两个 fitting
bounds 把 `[16,24] um` 分割成有序 cells；每个 cell 内 `G(D)` 和 `F_q8(D)` 均为常数。数学 endpoint convention 服从
source 的 radius `<= D/2` 规则；实现必须另外保存每个 breakpoint 及其两侧 `nextafter` 的实际归属，不能用 epsilon
移动 boundary。

对不使用 truth 选出的 qualifying seed `s`，定义：

- `C_geom(s)`：包含 `s` 且 `G(D)==G(s)` 的最大连通 q8 cell union；
- `C_exact(s)`：包含 `s` 且 raw candidate 与 target `np.array_equal` 的最大连通 cell union；
- `C_tau(s)`：包含 `s` 且 `rho(D) <= tau_rel` 的最大连通 cell union。

primary reported estimator 是 `I_hat=C_tau(s)` 的左右 breakpoint、endpoint closure flag 和区间宽度；它是 set-valued
结果。兼容旧接口可另外报告 `D_hat_mid=(I_left+I_right)/2`，但 midpoint 不是比区间更精确的独立测量，不得宣称在同一
plateau 内恢复了 truth 的具体位置。

阈值在新结果前冻结为：

```text
rho_floor = max(source replay rho, deterministic-repeat rho, float64 eps)
tau_rel   = max(1.0e-12, 100 * rho_floor)
tau_low   = tau_rel / 10
tau_high  = tau_rel * 10
primary loss threshold = tau_rel**2
```

现有 replay 为零，因此预期 `tau_rel=1e-12`；它比第 25 节已知 cross response 至少低 `5.6e7` 倍。`tau_low/high`
只做 threshold-stability control，不能从三者中事后挑最好看的区间。exact、geometry、`tau_low/tau_rel/tau_high` sets
必须分别保存；primary 始终是预先指定的 `tau_rel` set。

### 26.5 冻结 interval estimator、搜索预算和缓存

执行顺序固定如下，任何失败不得靠更换 start、loss、threshold 或搜索分辨率重跑：

1. **Stage A replay**：重复第 8 节 source identity、true-waist replay 和 deterministic repeat gates；未通过立即停止。
2. **Stage B profile screen**：重复第 6 节的 `1 um` coarse grid 和由 coarse argmin 唯一派生的 `0.125 um` fine grid，
   保存所有 raw losses。多个 qualifying grid points 只要属于同一 `I_hat` 就不再按 point-tie 判失败；位于不同 interval
   的 qualifying points 是 non-unique evidence。
3. **Stage C equal-budget search**：原 starts `[16.5,18.5,21.5,23.5] um`、pattern half-step `2 um`、tie rule 和每支
   exactly `41` objective calls 全部不变。每支 final incumbent 必须独立成为 interval seed；不允许只选择最低 loss 的一支。
4. **Stage D cell expansion**：用解析 sorted breakpoint table 把每个 seed 映射到 cell；分别按 `array_equal`、
   `tau_low`、`tau_rel`、`tau_high` 向左右逐 cell 检查 midpoint，直到遇到第一个 non-member。candidate cache 以 float64
   meter value 为 key并跨 branches/thresholds 复用，但每个 logical query、cache hit 和 cell index 都进入 track。每个方向、
   每个 threshold 最多检查 `256` 个相邻 cells；触及上限而尚未找到外侧 non-member 时停止并判 numerical-control
   Inconclusive，不把被截断集合当成窄区间。
5. **Stage E boundary control**：对 primary `tau_rel` interval 的每个边界，使用最后一个 inside-cell midpoint 与第一个
   outside-cell midpoint作 bracket；每支 branch 在每侧固定执行 exactly `24` 次 loss-membership bisection，不 early stop。
   final inner/outer bracket 必须包含解析 breakpoint且宽度 `<=1.0e-12 m`（`1 pm`）。另外保存 breakpoint 本身和两侧
   `nextafter` 的 geometry/field membership，以审计 half-open endpoint 语义。

truth 只在所有 interval 已由 target loss 和 fixed search 得出后用于 simulation accuracy evaluation；不得用于 seed、cell
定位、停止、branch selection 或 threshold selection。所有 unique candidate probes、q8 fingerprints、breakpoint/cell mapping、
profile、optimizer tracks、cell expansion 和 bisection tracks必须进入统一 cache/HDF5，不能只保存摘要。

### 26.6 Non-smooth identifiability gates 与互斥状态

以下 gates 全部在 formal 前冻结：

```text
source/replay/repeat raw relative L2                         <= 1e-14
all breakpoints/cells finite, strictly ordered after grouping, within bounds
C_geom == C_exact == C_tau_low == C_tau == C_tau_high       same endpoint indices
each immediate outside-cell rho / tau_rel                   >= 1e6
each boundary bisection bracket width                       <= 1.0e-12 m
analytic breakpoint lies inside corresponding loss bracket
all 4 starts                                                exactly 41 search calls
all 4 final incumbents                                      rho <= tau_rel
all 4 interval estimates                                    identical endpoint indices/closure
all qualifying coarse/fine/optimizer-track points           lie in the common interval
interval width                                              0 < width <= 0.125 um
truth-to-interval distance (simulation evaluation only)     0
worst-case endpoint absolute error from truth               <= 0.125 um
worst-case endpoint relative error from truth               <= 0.00625
interval distance from either fitting bound                 >= 0.125 um
all saved numeric arrays finite; no P_B_rec primary read
```

`C_geom==C_exact` 检查“同一离散 operator cell 确实给出 target”；三档 `tau` set 相同及 outside separation 检查 near-exact
边界不依赖任意 numerical floor。profile uniqueness 在预注册 grid、四支完整 tracks 和相邻-cell expansion 的联合 evaluation
support 上判定；它不是对全部 283665 个 cells 的全局场注入性证明，这一限制必须写入 formal 结果。q8 的 smooth Jacobian
convergence 不再是本轮通过条件，也不得删除其第 21/25 节 failure 记录。

状态按以下顺序互斥决定：

1. source/provenance/replay 或 artifact completeness 未闭合：`Inconclusive`，interpretation 为
   `artifact_operator_handoff_not_closed`；
2. breakpoint ordering/endpoint convention、determinism、threshold stability、outside separation、bisection agreement 或
   expansion budget 任一 numerical control 未闭合：`Inconclusive`，interpretation 为
   `q8_plateau_interval_numerical_control_not_closed`；
3. controls 完整有效，但任一 equal-budget start 未到达 qualifying set、各 start interval 不一致、screened support 出现
   interval 外的 qualifying component、truth 不在 interval、interval 超过 accuracy/boundary budget：`Failed`，按原因记录
   `interval_search_unstable`、`screened_equivalence_nonunique` 或 `interval_accuracy_failed`，不得挑 best branch；
4. 所有上述 gates 和 artifact audit 通过：`Passed`，interpretation 固定为
   `q8_plateau_interval_single_parameter_oracle_fit_passed`。

正式结果产生前 exp051 继续保持 `Inconclusive`。未来若 Passed，只能表述为 selected exp040 scalar working model 内、
matched true-probe、fixed-q8 operator 的 **interval-valued single-parameter oracle baseline** 通过；不能恢复 sub-plateau 点精度，
也不能表述为 full-wave、真实器件或真实计量验证。

### 26.7 实现、tests、run 与 artifact contract

下一轮实现建议新增且不覆盖现有文件：

```text
configs/experiments/exp051_TGV_3d_multislice_q8_plateau_interval_fit.yaml
src/tgv_ptycho/inverse/exp051_plateau_interval.py
scripts/run_exp051_q8_plateau_interval_fit.py
tests/test_exp051_q8_plateau_interval_fit.py
```

breakpoint helper、raw loss、candidate generator 和 cache 必须复用现有 exp051/exp040 shared implementation；runner 只做
config、编排、timestamped run、保存和审计。development/preflight 只允许 synthetic step-function/cell fixtures、source hash/
replay、类型/shape/HDF5 writer 测试以及最多一次明确标记的 real-case preflight；不得据此修改本节 thresholds。targeted tests
至少覆盖 half-open endpoint、exact/near-exact component expansion、disconnected synthetic alias、256-cell cap、24-step
bisection、threshold stability、multi-start equal-budget interval agreement、bounds/illegal diameter、determinism、all-false flags、
HDF5 layout 和 `P_B_rec` hard fail。正式运行前依次通过新 targeted tests、原+新 exp051 tests、full pytest 和 scoped Ruff；
full-suite 既有失败必须与本轮新增失败分开报告。

formal 只允许在实现/测试/preflight 记录并锁定后创建一次：

```text
runs/exp051_TGV_3d_multislice_true_probe_waist_fit_q8_plateau_interval_<timestamp>/
  config.yaml
  metadata.json
  metrics.json
  run_state.json
  outputs/exp051_q8_plateau_interval_fit.h5
  figures/
```

HDF5 继续使用 `/entry` 并列结构；自然产生的 fit 放
`/entry/reconstruction/waist_fit/plateau_interval/`，至少保存 design/gates/source、sorted breakpoints 与 multiplicities、
四种 equivalence components、endpoint/nextafter membership、三档 threshold、profile、四支 search/expansion/bisection tracks、
完整 candidate/fingerprint cache、reported interval/midpoint、adjacent outside probes、representative raw probe 和 residual。
同时保存原 instrument/sample/truth/metrics/metadata/config_yaml；不得创建伪 calibration/preprocessing 或保存/读取 primary
`P_B_rec`。外部 JSON 与 HDF5 同义 leaves 必须 exact，formal validator 必须审计全树 finite、cache mapping、hashes、flags 和
status。至少三图：global profile+threshold/interval、breakpoint-local staircase+boundary brackets、four-start interval tracks。
预算冻结为 runtime `<180 s`、sampled peak RSS `<1 GiB`、HDF5 `<128 MiB`；超预算记录事实并判 artifact/runtime
Inconclusive，不通过删 evidence 规避。

### 26.8 exp04x/exp053/exp055/exp041 边界与停止扩张

本节没有产生新的 exp04x 反馈，也不编辑 exp04x 文档。formal 若证明 q8 interval gate 通过，结果章节应在同一轮记录一个
供用户手工转录的反馈块：fixed-q8 虽无 smooth local derivative，但在明确 set resolution 下可形成窄且多起点一致的 oracle
interval；若 threshold/adjacent-cell control 失败，则反馈 q8 inverse resolution 仍未闭合。两种情况都不能改写 R7 finite-
perturbation forward convergence，也不能把 chord control 冒充 q8 result。

若 formal Passed，exp051 oracle baseline 才可关闭并把同一 q8 interval estimator、threshold、bounds、search budget 和 failure
flags交给 exp053；exp053 仍必须另行预注册并只换 matched raw `P_B_rec`。若 interval 太宽或出现 screened non-uniqueness，
优先反馈 exp04x 的 q/interface resolution，不直接扩成 exp055 nuisance fit；只有 fixed-parameter oracle interval 已通过，
才讨论 nuisance。由于当前 probe signature 并不接近 floor，不触发 exp041 scan-design sweep。

需要改 target/operator、使用 chord-cell primary truth、改变物理模型或项目级 schema，或转向 detector/noise/model mismatch/
真实数据时立即停止 exp051；这些都需要 upstream source version 或新实验，不得靠本轮调参扩张。

### 26.9 本轮 Changes、预注册证书、Git 与快速恢复

本轮唯一 Change 是原位同步第 0 节并在 EOF 追加本节；没有修改 Python/YAML/tests/runs 或 exp04x 文档。追加前 exp051
文档为 `58462 bytes`，完整 SHA256 为
`E2C75080C45139DAC90EA5A9A6C9310D8563ADAE7C8C0F9B29DA0C8A9F6A74F9`。authoritative frozen-body lock 仍是从首个
`## 1.` 起前 `15946` UTF-8 bytes，预期 SHA256
`A499B6FB546F3A2DDB7B2840834F4E7C8770BD7DA87C7CFEBA9253E6180EEE6D`；第 1--25 节不得回改，后续实现、测试、
preflight、唯一 formal、audit、结果、exp04x 手工反馈、限制和 Git 状态应按第 0 节规则尽量合并记录在一个新的 EOF
编号章节。

没有执行 `git add`、commit、push、merge 或 PR。快速恢复时先读第 0、23--26 节，校验本节 source hashes 和 threshold/
budget/status logic，再实现；不得把第 25 节已知 truth gap 直接硬编码为 estimator boundary，也不得在 formal 后修改
`tau_rel=1e-12`、`24` 次 bisection、`256`-cell cap 或 `0.125 um` interval budget。

## 27. 2026-08-24：q8 plateau-interval 完整实现、formal 与审计

### 27.1 本轮目标、上一轮意见与明确未做事项

本轮按第 26 节冻结设计实现第一个 fixed-q8、matched raw `P_B_true`、set-valued 单参数 `D_waist` oracle estimator：
复用 exp040/exp051 q8 candidate generator 和解析 breakpoint map，完成 exact/near-exact connected component、四起点等预算
interval agreement、fixed-budget boundary bisection、non-smooth gates、HDF5/JSON/figures、tests 和唯一 formal。上一轮意见是
下一步仍留在 exp051，先关闭 q8 plateau-equivalence set、interval resolution 和 non-smooth identifiability，而不是静默换成
chord-cell truth。

明确未做 chord-cell primary、`P_B_rec`/exp053、nuisance/exp055、sample-B/scan/exp041、noise、detector fitting、真实数据、
full-wave/reference validation或项目级 schema 变化；没有编辑 exp04x 文档。旧第 21/25 节 q8 Jacobian non-convergence
保持不变，新结论只说明对这个已确认 non-smooth operator，冻结的 interval-valued gate 是否通过。

### 27.2 开始检查、实际读取和改动前判断

开始时完整读取根目录 `AGENTS.md`，随后执行
`git -c safe.directory=E:/tgv_ptycho_sim status -sb`。工作区已有 modified/deleted/untracked 内容全部保持用户所有，没有
回退、覆盖、删除或暂存。定向读取第 23--26 节、最新 q8 local-control `metrics.json`、现有 exp051 source loader/candidate
generator、q8 breakpoint/fraction-stack helpers、raw loss/profile/pattern search、local-control runner、HDF5 writer及两组 exp051
tests。`rg` 先定位符号，没有扫描无关 runs/notebooks/reports。

改动前判断是：现有 shared forward 和 source provenance 已足够，新增 inverse set estimator 和 orchestration 即可；无需改
exp040/042、`waist_fit.py`、`save_load.py` 或 data-format schema。第 26 节 `tau=1e-12`、三档 threshold control、四起点每支
41 calls、每侧 256-cell cap、双侧各 24 次 bisection、`1 pm` bracket budget、`0.125 um` interval accuracy budget 和状态逻辑
均在任何新 real-case result 前保持冻结。

### 27.3 Changes 与 operator/numerical consistency

新增：

```text
configs/experiments/exp051_TGV_3d_multislice_q8_plateau_interval_fit.yaml
src/tgv_ptycho/inverse/exp051_plateau_interval.py
scripts/run_exp051_q8_plateau_interval_fit.py
tests/test_exp051_q8_plateau_interval_fit.py
```

inverse module 实现了 prior q8-local artifact/hash loader、严格 config validation、283665-breakpoint cell partition、float64
cell mapping、connected-component expansion、fixed-iteration membership bisection、target/raw-loss cache、coarse/fine profile、
原四支 pattern search、geometry node-count fingerprints、endpoint/`nextafter` controls、threshold stability、screened alias、
interval accuracy和互斥状态判定。candidate generator 仍是 default q8 shared path；没有复制或修改 exp040 forward，也没有调用
chord builder。

runner 只负责 config/provenance、timestamped run、资源采样、并列 HDF5 保存、三图和只读 validator。tests 覆盖 forbidden
`P_B_rec`/all-false flags、非法 config/bounds、synthetic disconnected alias、256-cell cap、无 interior float64 cell、24-step
bisection、real q8 replay/threshold/components/multi-start、HDF5 layout/cache/residual/branch budgets。实现后 formal 前最终锁为：

```text
config  5689 bytes  FD507707EC849981F9DD083002F9FC5228D9BAE703A5AA903FB079E8B4393AE5
module 43061 bytes  D0544B991EC9C99836515AD297FDF73780A2123D913F3232A533327C9EC1E945
runner 25277 bytes  75A8D2834B3339DBB3E9E23D74F598ECB59F8EF78CE6EB1E661CE170A58C431C
tests  10229 bytes  5C9B82F4D602EE4420927FBF16194844CE47BB81DDEA7CBC820F310C04A27591
```

### 27.4 Development/preflight corrections 与 formal lock

先通过纯 synthetic/config tests `3 passed, 2 deselected` 和 scoped Ruff。第一次 real fixture 在 candidate forward 前停止：
解析 breakpoints 中存在相邻 float64 值，其间没有可表示的严格 interior midpoint。该问题只涉及 half-open cell 的代表值；
按 `[left,right)` 语义令这种 cell 使用唯一可表示的左端点，并增加回归测试，没有改任何 gate/threshold。

修正后 real estimator 完整运行，但测试最初得到 `3 passed, 2 failed`，status 为 numerical-control Inconclusive。只读检查临时
metrics 定位为 endpoint audit 把“analytic upper breakpoint 的一个 `nextafter(-)` 必须仍在内侧”当成额外要求；第 26 节实际
冻结的是：保存实际 endpoint/nextafter 归属、geometry 与 raw field 逐点一致、24-step loss bracket 包含 analytic breakpoint，
并没有冻结单 ULP 必须跨界。formal 前把实现修正为这些已预注册条件：所有登记点 geometry/exact/tau membership 一致，
analytic closure 与 component closure 一致，transition 由 fixed bisection 控制。`tau`、bisection iterations/resolution、cell cap、
interval tolerance和状态顺序均未改变。修正后的 targeted suite 为 `5 passed`，由此锁定上节文件 hashes并授权唯一 formal。

该 correction 揭示的实际数值语义被保留：upper analytic breakpoint 的 `nextafter(-)` 已属于外侧，但 geometry 与 raw field
一致；loss transition bracket 仍包含 analytic breakpoint，宽度远小于 `1 pm`。因此 interval 是按预注册 resolution 报告的
set estimate，不声称 analytic real-number breakpoint 与每个 float64 ULP 完全同位。

### 27.5 Tests、Ruff 与实际命令

实际执行的验证命令及结果为：

```powershell
conda run -n tgv_ptycho_sim python -m ruff check src/tgv_ptycho/inverse/exp051_plateau_interval.py scripts/run_exp051_q8_plateau_interval_fit.py tests/test_exp051_q8_plateau_interval_fit.py
# 初次报告 7 个本轮 lint 问题；修正后和最终复核均为 All checks passed!

conda run -n tgv_ptycho_sim python -m pytest -q tests/test_exp051_q8_plateau_interval_fit.py -k "config_prior or connected_component or fixed_budget_membership"
# 3 passed, 2 deselected

conda run -n tgv_ptycho_sim python -m pytest -q tests/test_exp051_q8_plateau_interval_fit.py
# 首次：3 passed, 2 errors（无 representable interior midpoint）
# 第二次：3 passed, 2 failed（额外 nextafter assumption 导致 Inconclusive）
# 最终：5 passed in 53.81s

conda run -n tgv_ptycho_sim python -m pytest -q tests/test_exp051_true_probe_waist_fit.py tests/test_exp051_local_differentiability_control.py tests/test_exp051_q8_plateau_interval_fit.py
# 16 passed in 78.23s

conda run -n tgv_ptycho_sim python -m pytest -q
# 322 passed, 12 failed in 178.72s
```

full-suite 12 项仍精确是既有 exp040 R10 stage-A 两项、R10 stage-B、R10 stage-B preflight、R11、R11 preflight、R12、
R13、R14、R14a 和 R14b 两项 frozen-config/scientific-contract SHA mismatch；没有新增 exp051、exp042、shared forward 或 IO
functional failure。本轮未修改这些历史 locks，也未为全仓全绿而触碰无关文件。

### 27.6 唯一 formal run、exact replay 与 interval metrics

formal 实际命令：

```powershell
conda run -n tgv_ptycho_sim python scripts/run_exp051_q8_plateau_interval_fit.py --config configs/experiments/exp051_TGV_3d_multislice_q8_plateau_interval_fit.yaml
```

唯一 formal run：

```text
runs/exp051_TGV_3d_multislice_true_probe_waist_fit_q8_plateau_interval_20260824_161440
run state: complete
artifacts_validated: true
experiment_status: Passed
interpretation: q8_plateau_interval_single_parameter_oracle_fit_passed
runtime: 48.5393926 s
sampled peak RSS: 142585856 bytes
```

source raw `P_B_true`、true-waist replay、replay repeat和 seed/outside deterministic repeats均为 bitwise/relative-L2 `0`；
`tau_low/primary/high = 1e-13/1e-12/1e-11`，primary loss threshold `1e-24`。exact、geometry和三档 near-exact
components 对四个 starts 全部落在同一 q8 cell index `139498`，均为：

```text
I_hat = [19.999972701052885, 20.000143340468626) um
width = 0.170639415741 nm
midpoint = 20.000058020760754 um
midpoint raw loss = 0
truth-to-interval distance = 0
worst endpoint absolute error = 0.143340468625 nm
worst endpoint relative error = 7.16702343e-6
fitting-bound margin = 3.99985665953 um
```

因此区间宽度只占冻结 `0.125 um` budget 的约 `0.1365%`，truth 位于区间内部；primary result 是上述半开区间，不是
midpoint 点精度。左右 immediate outside raw relative L2 为 `7.29413808e-6/7.43674819e-5`，相对 `tau` 为
`7.29413808e6/7.43674819e7`，通过 `>=1e6` separation。四支 starts 均 exactly 41 calls，final seeds 分别为
`20.0/19.999999999999998/19.999999999999998/19.999999999999996 um`，loss 全为零并得到同一 interval；联合 profile/
optimizer support 中 19 个 qualifying evaluations 全在该 interval，没有 screened external component。

四支每侧均 exactly 24 次 bisection；lower/upper final bracket widths 为
`5.08558582e-18/8.51776332e-18 m`，均包含 analytic breakpoint且远低于 `1e-12 m` gate。283665 个 internal
breakpoints形成 283666 个 cells；所有 source、breakpoint、determinism、threshold stability、endpoint convention、outside
separation、bisection、cell budget、equal-budget、multi-start agreement、screened uniqueness、accuracy和资源 gates均为 true。

### 27.7 HDF5、JSON、figures 与资源独立审计

formal HDF5 为 `42187256 bytes`，SHA256
`FD1F5E47300BB8A09329F8E21E1A9A3AFCD84BFEFA3FD9A7AD89705983270B3C`，小于 128 MiB contract；peak RSS和 runtime
也分别小于 1 GiB/180 s。`/entry` 顶层精确为：

```text
config_yaml, data, instrument, metadata, metrics, reconstruction, sample, truth
```

`/entry/data` 为空的通用 writer group；没有 calibration/preprocessing。新 fit 全部位于
`/entry/reconstruction/waist_fit/plateau_interval/`，自然包含 `design/source/partition/profile/optimizer/interval_by_branch/
reported_interval/outside_control/geometry_control/endpoint_field_control/bisection/deterministic_repeat/profile_screen/cache/gates`，
以及 `P_B_interval_midpoint_raw` 和 `residual_field_raw`。保存了 `(177,96,96) complex128` candidate stack、全部 cache
diameter/loss/cell mapping、四支完整 41-call tracks、四种 components、四支双侧完整 bisection tracks、9 组
`(100,96,96) uint8` q8 node-count fingerprints和 283665 breakpoints/multiplicities。

独立只读审计统计 136 groups、1082 datasets、总计 1218 paths；所有 numeric leaves finite。external target 与 source target
bitwise equal，midpoint probe 与 cache mapping exact，`P_B_midpoint-P_B_true==residual` exact。`config.yaml` 与
`/entry/config_yaml` exact；`metadata.json`、`metrics.json` 与 HDF5 对应 groups逐叶无差异。无 leaf `P_B_rec`，all-false
provenance 未提升。三张 PNG 全部实际打开检查：global profile、local staircase/boundary brackets和 four-start intervals均可读、
finite、标签/单位清楚且无截断；global 图上 0 loss 映射到 float tiny 仅用于 log display，数值判断仍来自 HDF5 raw zeros。

run artifact hashes：

```text
config.yaml   88CD812DC716C3AACB3BF7932B9A6A155D5CAE5C52842BED7C2F62F483301C8B
metadata.json 3541FC02427CE5EF20E2DA188398242BD3A3C2E7F6885B860EB95F79814D5E34
metrics.json  20955E3A6C8061A184633896851BA1EACF5E59A7424715EE7AB94C06B51AC418
run_state     DF69EF43C1E719F785962247B9E819DDB28A2D574A7FF796ED28D7800A19B588
```

### 27.8 Formal 判定、证据边界与已知限制

按第 26.6 节互斥规则，本轮为 **Passed**。exp051 现在建立了 selected exp040 scalar working model 内、matched raw true
probe、fixed-q8 operator 的 **interval-valued single-parameter oracle baseline**；它关闭了此前仅因 smooth Jacobian gate 不适配
non-smooth operator而留下的 `Inconclusive`。第 21/25 节 Jacobian failure没有被抹掉或改写，也不能从本结果推断 q8 存在
smooth derivative。

限制仍包括：screened uniqueness只覆盖冻结 coarse/fine grids、四支完整 tracks和相邻-cell expansion，并非逐一 forward
评估全部 283665 cells的全局场注入性证明；interval宽度是 q8 discretization/operator resolution，不是真实仪器 resolution；
upper analytic breakpoint 与 float64 actual transition存在小于 bisection bracket 的 ULP-level offset；完全 matched/noiseless/
true-probe 条件未覆盖 `P_B_rec`、noise、nuisance、calibration或 model mismatch。truth 始终只表示
`simulation truth under the selected exp040 scalar working model`。`reference_validated=false`、
`full_tgv_reference_authorized=false` 持续为 false。

### 27.9 供用户手工转录至 exp04x 的新增反馈块

Codex 本轮没有编辑 exp04x 文档；以下是相对第 25.7 节新增、值得手工反馈的 inverse 证据：

> **来自 exp051 的 q8 plateau-interval 反馈（2026-08-24）**  
> 在 exp042 matched raw q8 `P_B_true`、default-q8 exact replay `0` 的条件下，证据 run
> `runs/exp051_TGV_3d_multislice_true_probe_waist_fit_q8_plateau_interval_20260824_161440` 通过了预注册 non-smooth
> identifiability gate。exact q8 geometry/raw-field set 与 `rho<=1e-13/1e-12/1e-11` 三档 set 均为同一半开区间
> `[19.999972701052885,20.000143340468626) um`，宽 `0.170639416 nm`，远小于 `0.125 um` inverse accuracy budget；
> 四个 starts 每支 41 calls并得到同一 interval。相邻外侧 probe response 为 `7.29e-6/7.44e-5`，相对 primary tau
> 分离 `7.29e6/7.44e7`；双侧 24-step brackets 宽 `5.09e-18/8.52e-18 m`且包含 analytic breakpoints。由此 fixed-q8
> operator 虽不具备可用 smooth local Jacobian，仍可在明确 operator/set resolution 下支持 derivative-free interval-valued
> oracle inverse。需要在 exp04x/Phase-5 handoff 中同时登记 breakpoint/half-open endpoint 和 float64 actual membership：upper
> analytic breakpoint 的一个 `nextafter(-)` 已在外侧，但 q8 geometry 与 raw field 逐点一致且 transition 位于冻结 `1 pm`
> bracket 内。该反馈不否定 R7 finite-perturbation convergence，也不把 chord-cell 升为当前 primary；它只要求 q8 inverse
> claim使用 set resolution而非 smooth derivative或伪 sub-plateau点精度。证据仍限于 selected scalar working model，
> `reference_validated=false`、`full_tgv_reference_authorized=false`。

### 27.10 改动后优先级、handoff 与快速恢复

exp051 的 oracle baseline 已按 interval-valued claim **Passed**，不需要继续在 exp051 调 q8 gate。下一步优先是新开/恢复
**exp053 preregistration**：复用本轮 source case、default-q8 candidate generator、raw loss/full-field reference、
`[16,24] um` bounds、coarse/fine screens、四起点 41-call budget、`tau=1e-12` 三档 controls、256-cell cap、24-step/1-pm
bisection、interval accuracy/status logic和 artifact loader；唯一 primary input 改成同一 exp042 matched case 的 raw
`P_B_rec`。exp053 不得用 truth-aligned probe，也不得用 exp051 truth调 threshold。

exp055继续等待 exp053，不能因 oracle interval Passed就提前扩成 nuisance fit。exp041不触发，因为 q8 true-probe outside
signature远高于 floor；若 exp053失败，应先区分 reconstruction error 与 interval set failure。exp04x只需由用户手工加入上节
反馈；当前没有证据恢复 exp040全部 reference-validation或修改 R7/R8。快速恢复时读第 0、23--27 节和本 formal
`metrics.json`；primary result是 interval，不是 midpoint。

### 27.11 Append-only verification 与 Git 状态

本节追加前 exp051 文档为 `75040 bytes`，完整 SHA256 为
`5641DCE12E39A67CB827C918E2C3567AD9F6A8F9062555B451C4AA09CF289390`。第 1--26 节没有回改；authoritative
15946-byte frozen prefix SHA256 仍应为
`A499B6FB546F3A2DDB7B2840834F4E7C8770BD7DA87C7CFEBA9253E6180EEE6D`。本轮只原位同步第 0 节并在 EOF 追加
本节，符合“一轮一节”规则。

结束时 Git staged为空；未执行 `git add`、commit、push、merge 或 PR。用户既有 modified/deleted/untracked 文件未回退、
删除或擅自暂存；本轮 source/config/script/test/doc均保持 local unstaged/untracked，formal run/HDF5/PNG也未纳入 Git。
