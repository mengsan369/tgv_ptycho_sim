# exp053：3D scalar multislice reconstructed-probe waist fitting

## 0. 实时状态与阅读顺序

本节是首次完整构建后唯一允许持续原位更新的状态入口。第 1 节以后是 formal 前冻结的设计正文；首次冻结后，
新的实现、结果、失败和 correction 只能追加到真实 EOF，不回改既有正文。

```text
Scientific status: Passed
Work status: Formal complete / artifacts validated / numerical closure complete
Results available: true
Latest valid run: runs/exp053_TGV_3d_multislice_reconstructed_probe_q8_cell_interval_20260901_122210
Authoritative appended section: section 16
Primary current question: fixed-q8 reconstructed-probe baseline已闭合；冻结 exp053，暂不启动 exp055
reference_validated: false
full_tgv_reference_authorized: false
```

阅读顺序：第 0 节看实时状态；第 1--9 节看研究问题、source、方法兼容性审计、冻结 estimator、gates 和
artifact contract；第 10 节是首次设计冻结记录；后续编号章节记录实现、测试、formal、审计和 correction。

## 1. 研究问题与依赖链

exp053 在 exp040 selected scalar multislice working model、exp042 单一无噪声 known-B matched q4/q4 reconstruction、
exp051 fixed-q8 单参数 oracle contract 下，把 fitter target 从 raw `P_B_true` 换成 raw `P_B_rec`，回答能否形成稳定、
可重复的 interval-valued `D_waist`，以及相对 exp051 oracle interval 的 reconstruction-induced bias。依赖链为
`exp040 -> exp042 -> exp051 -> exp053`。

即使 Passed，也只说明上述 working-model、单 case、fixed-parameter baseline 闭合；不表示真实三维电磁准确性、真实
计量精度、resolution、detection limit、noise robustness 或 production reconstruction。

## 2. Authoritative source 与禁止输入

primary source 固定为：

```text
run: runs/exp042_TGV_3d_multislice_probe_reconstruction_dev_20260822_195139
HDF5: outputs/exp042_probe_reconstruction.h5
dataset: /entry/reconstruction/detector_quadrature_ablation/branches/matched_q4/P_B_rec
HDF5 SHA256: 4AF86885BA6A5A60F6E76A48ADE5B272E25A5E96FE34D2B03E3774D53848D62E
dataset-byte SHA256: F6323C7A7676CEA2F8FD8D99EC2CEC0CA6C4C9234BD49F122F17BEDE4DDDC738
identity: plane B, axis (y,x), 96x96 complex128, node dx 0.5 um
```

loader 必须实际校验 source 全部文件 hash、complete/`artifacts_validated=true`、branch operator spec、raw dataset bytes、
shape/dtype/finite、embedded config、plane/grid/reference 和 all-false provenance。primary input 只能是上述路径；禁止
truth、`P_B_rec_global_phase_aligned`、任何 aligned/truth-selected copy、q4/q1 mismatch 和 q1/q1 control。

`/entry/truth/P_B_true` 只用于 candidate-operator replay 和明确标记的 simulation evaluation；不得作为 target，不能选择
branch、mask、threshold、stop 或 best start。

## 3. exp051 方法兼容性审计

仍有效的不变量是同一 source case、default-q8 candidate generator、`[16,24] um` bounds、raw full-field complex loss、
full-field equal weighting、无 phase/scale alignment、coarse/fine profile、四 starts、每支 41 calls、q8 breakpoint/half-open
语义、24-step/1-pm boundary control、256-cell cap、0.125-um interval/accuracy budget、deterministic cache、artifact/resource
审计和 oracle comparison contract。

不能继承的是把 `candidate(20 um)` 对 fitter target 的 exact replay 当 gate、用 target replay error 定义 numerical floor、
绝对零损失附近的 equivalence set，以及要求 candidate 对 reconstructed target 达到 `rho<=1e-12`。raw reconstruction mismatch
约 0.139372，若机械乘入 tau 会把 reconstruction error 变成可接受集合宽度，失去拟合意义。

候选定义比较：

- 离散最优 q8 cell 直接回答 reconstructed target 在冻结 operator 上选择哪个 cell，且不需要新误差模型；保留。
- 相对最优 raw-loss near-minimum interval 需要预注册 measurement/reconstruction error model；当前没有，排除为 primary。
- 相对最佳 candidate field 的 q8 field-equivalence interval 由 operator resolution 和 determinism 定义，可把最优 cell 扩写为
  有端点语义的 set-valued estimate；选为 primary。

## 4. 冻结 reconstruction-aware estimator

记 raw target 为 `R=P_B_rec`，candidate 为 `F_q8(D)`：

```text
L_rec(D) = ||F_q8(D)-R||^2 / ||R||^2
```

先在冻结 coarse/fine grid 与四支完整 pattern-search tracks 的联合 support 上选 raw-loss 最低 cell；loss tie tolerance 固定为
`1e-14`，只用于识别数值不可区分的不同 screened cells，不从 0.139 reconstruction mismatch 放大。以最优 cell 的 candidate
field `F_best` 为 reference，定义包含该 cell 的最大连通 q8 component：

```text
C_field(tau) = {connected cells with ||F_q8(D)-F_best||/||F_best|| <= tau}
tau_primary = 1e-12; tau_low = 1e-13; tau_high = 1e-11
```

primary reported interval 是 `C_field(1e-12)` 的解析左右 breakpoint 和实际 closure flags；midpoint 仅是代表值。`C_geom`、
exact-field、low/primary/high components 必须一致。threshold 来自 float64/operator replay 与 exp051 已冻结 field-equivalence
control，不依赖 reconstructed-target residual。

## 5. 搜索、区间和 comparison contract

coarse grid 为 16--24 um 每 1 um；fine grid 由 coarse argmin 唯一派生，半宽 0.5 um、步长 0.125 um。starts 为
16.5/18.5/21.5/23.5 um，initial step 2 um，每支 exactly 41 calls，不 early stop。每支 final incumbent 必须属于同一
best-field component；不得只挑最低的一支。

每个 threshold/component 每方向最多展开 256 cells。primary component 双边各做 24 次 fixed-membership bisection，bracket
必须含解析 breakpoint且宽度不超过 1 pm；breakpoint 与两侧 `nextafter` 的 geometry/field membership 全部保存。

exp051 oracle 固定为 `[19.999972701052885,20.000143340468626) um`。formal 后报告 overlap、interval distance、midpoint
displacement、两端 displacement 和 endpoint-Hausdorff displacement。truth-to-interval distance、endpoint errors 与 directional
diagnostic 均标为 `simulation_evaluation_only`，不反馈进 fitter。

## 6. Gates 与互斥状态

numerical/artifact gates：source 与 oracle hash/state、candidate replay 到 operator reference `<=1e-14`、repeat `<=1e-14`、
breakpoints strictly ordered、geom/exact/three tau components 相同、无 expansion cap、half-open endpoint一致、24-step brackets
`<=1 pm` 且含解析 boundary、全数值 finite、JSON/HDF5/cache mapping一致。

scientific/search gates：四支均 41 calls并进入同一 interval；screened support 不得在 interval 外出现 loss-tied cell；
`0 < width <=0.125 um`；simulation evaluation 中 truth-to-interval distance与最大 endpoint error均 `<=0.125 um`，最大相对
endpoint error `<=0.00625`；interval 离 fitting bounds 至少 0.125 um。

状态顺序：artifact/operator/numerical control 不闭合为 `Inconclusive`；controls 有效但 search 不一致、screened nonunique 或
accuracy 失败为 `Failed`；全部通过为 `Passed / reconstructed_probe_q8_cell_interval_fit_passed`。不得 formal 后改定义或阈值。

## 7. 定向 reconstruction-error 诊断

若 formal 未 Passed 或 truth 不在 reported interval，才计算 simulation-only 诊断：reconstruction error
`P_B_rec-P_B_true` 对冻结 `D_true +/-0.125 um` finite-change directions，以及 `P_best-P_B_true` candidate-manifold direction
的 real cosine 与 complex coherence。两侧都报告，不事后挑方向；该诊断不进入 fit，只用于决定是否以一个明确误差方向恢复
exp042。不能仅凭 full-field 14% error 要求上游优化。

## 8. HDF5、figures 与 artifact contract

不修改项目级 schema。自然结果写入 `/entry/reconstruction/waist_fit/reconstructed_target_q8_cell_interval/`：raw input、source
identity、operator replay/reconstructed mismatch、profiles、四支 tracks、components/bisection、reported interval、best raw
candidate/residual、oracle comparison、gates/resources、完整 cache 和 simulation-only truth comparison。raw target 单独保存，
绝不由 aligned copy覆盖。`/entry/config_yaml`、`metadata`、`metrics` 并列；不伪造 calibration/preprocessing。

固定三图为 global reconstructed-target profile+oracle、local q8 staircase+两 interval、four-start tracks。只有第 7 节触发时
增加 direction diagnostic 图。HDF5/JSON 同义 leaves exact；validator 审计 tree、hash、field identity、numeric finite、cache
mapping、41-call branches、24-step brackets、figures、runtime `<180 s`、RSS `<1 GiB`、HDF5 `<128 MiB`。

## 9. Scope 和上游反馈边界

不扩展 `z_waist`、top/bottom diameter、index、`z_AB`、complex gain、noise、stage、blind-B、真实数据、任意 `D(z)` 或 direct
detector fitting。exp051 反馈只说明 true-target tau 不能机械用于 reconstructed target及本实验的新定义；exp042 只有在 oracle
通过而 reconstructed fit偏差/失败且 direction diagnostic 显著重合时才恢复；exp040 只在 operator/provenance/replay不闭合时反馈；
exp041 只在证据表明 measurement design 未保留 waist direction 时触发。

## 10. 2026-08-31：首次完整设计冻结记录

本轮目标是完成 formal 前方法兼容性审计与预注册。实际读取 `AGENTS.md`、本文件旧预留状态、roadmap Phase 4/5、data-format、
exp051 第 0/17/23--27 节、exp042 第 0/27/28 节及 authoritative metrics/HDF5、exp040 当前状态/R8、exp051 config/module/runner/tests。
source 外部文件和 dataset byte hashes已独立计算；raw target 为 `(96,96) complex128`、finite，未对齐 raw-to-truth relative L2
为 `0.13937201182884226`。Git 因 ownership 初次拒绝，随后只对命令使用 `-c safe.directory=E:/tgv_ptycho_sim`；未改全局配置。

Changes 是本设计正文和
`configs/experiments/exp053_TGV_3d_multislice_reconstructed_probe_q8_cell_interval_fit.yaml`。截至本节尚未实现 Python/tests，
未运行 development/formal，未创建 run，故无 HDF5/JSON/figure audit 或科学结果。当前高重要性问题“exp051 zero-residual
interval 对 reconstructed target 不兼容”已由 best-candidate field-equivalence 定义在方法层关闭；下一步最高优先级是按冻结
配置实现 loader/estimator/runner/tests，通过 real preflight 后只运行一次 formal。没有 `git add`、commit、push、PR 或 merge；
所有改动保持本地 unstaged。

## 11. 2026-08-31：real-case preflight endpoint correction 与 formal lock

本轮目标是实现后执行一次不创建 run 的 real-case preflight，并解决 formal 前唯一的高重要性 numerical-control 问题；上一轮
建议是保持冻结 raw target/loss/search/threshold，先关闭 loader、operator replay 和 endpoint semantics。实际 Changes 为新增
`src/tgv_ptycho/inverse/exp053.py` 与
`scripts/run_exp053_reconstructed_probe_q8_cell_interval_fit.py`，并对第 4 节“实际 closure flags”补充本 correction：primary
report 始终遵循 q8 partition 的规范半开 `[lower,upper)`；direct generator 在 breakpoint/两侧 `nextafter` 的实际 membership
另行完整保存，不用孤立 float64 endpoint 改写 interval closure。

preflight 实际验证 source/oracle loaders、candidate replay、profiles、四支各 41 calls、components、bisection、accuracy 和
direction diagnostic。四支均到达 cell `177657`，best field component 跨 cells `177657--177658`；direct generator 的 upper
breakpoint 仍与 best field相同、`nextafter(+inf)` 才进入 outside field。这是 `radius <= D/2` 与 half-open partition 在孤立
float64 boundary 的表示差异，不是 reconstructed mismatch 定义失效。修正只把 component signature 的 closure固定为规范
`[lower,upper)`，实际 endpoint arrays保持原值；没有改变 interval endpoints、best cell、loss、tau、starts、budget 或 gate。

preflight 尚未作为 scientific formal：临时结果为 reconstructed interval 约
`[21.07016531442902,21.07033167530035) um`、truth distance约 `1.070165 um`，明显提示 formal 可能 Failed；这些数值没有用于
修改任何方法或 threshold。静态首次检查仅有两项 E501，已机械换行；Python compile通过。下一步最高优先级是补齐 targeted
tests，重新运行 corrected preflight 并锁定 tests/lint；只有全部通过才运行一次 formal。当前无 development/formal run、无新
artifact；未执行 `git add`、commit、push、PR 或 merge，改动保持 unstaged。

## 12. 2026-08-31：实现、tests/lint、corrected preflight 与 formal authorization

本轮目标是补齐 tests、关闭 corrected preflight 与 tiny runner artifact contract，并在 formal 前执行提示词要求的 regression；
上一轮建议是不得根据已经可见的约 1.07-um bias 修改方法。实际新增
`tests/test_exp053_reconstructed_probe_q8_cell_interval_fit.py`，覆盖 source/hash/state/raw matched-q4 only、aligned/truth/q1
拒绝、shape/dtype/finite/plane/grid/reference、candidate replay与 reconstructed mismatch分离、固定 tau、四支等预算、规范半开
endpoint和实际 breakpoint membership、bisection、oracle comparison、direction diagnostic、exp051 exact-replay regression、tiny runner
HDF5/JSON/cache/PNG validator。

实际命令与结果：

```text
python -m pytest -q tests/test_exp053_reconstructed_probe_q8_cell_interval_fit.py
  -> 5 passed in 48.85s
python -m pytest -q tests/test_exp051_q8_plateau_interval_fit.py tests/test_exp053_reconstructed_probe_q8_cell_interval_fit.py
  -> 10 passed in 101.03s
python -m ruff check src/tgv_ptycho/inverse/exp053.py scripts/run_exp053_reconstructed_probe_q8_cell_interval_fit.py tests/test_exp053_reconstructed_probe_q8_cell_interval_fit.py
  -> All checks passed!
python -m pytest -q
  -> 345 passed, 12 failed in 246.01s
```

全部正式 Python 命令用 `conda run -n tgv_ptycho_sim` 和独立 `--basetemp` 执行。直接解释器的 tiny runner 曾在
Matplotlib/NumPy DLL delay-load 处两次 native crash；改用仓库既有 Conda 入口后消失。第一次 Conda tiny runner又因 validator
把 `raw_unaligned` 误识别为 aligned copy而失败；validator收窄为明确拒绝 `global_phase_aligned/truth_aligned/P_B_rec_aligned`，
随后 `1 passed`。这些是测试环境/validator correction，未创建正式 run、未改 scientific result。

全量 12 failures全部来自 exp040 R10--R14 formal/preflight/release YAML 与代码内 frozen SHA256 不一致；无 exp053/exp051
failure，且失败数与既有仓库记录相同。本任务不修改上游 config/hash lock。corrected preflight 当前状态为
`Failed / reconstructed_target_interval_accuracy_failed`，但 candidate replay、threshold stability、endpoint、boundary、multi-start、
screened uniqueness等 numerical gates全部通过；失败是预注册 scientific outcome，不阻止保存 formal evidence。

当前主要问题从高重要性的 provenance/method compatibility 降为中重要性的 reconstruction-induced bias。formal authorization
已满足；下一步最高优先级是只运行一次冻结 YAML，随后独立核查 run_state/config/metadata/metrics/HDF5全树/cache/四图。当前无
valid run、无 artifact audit；未执行 `git add`、commit、push、PR 或 merge，所有改动保持 unstaged。

## 13. 2026-08-31：唯一 formal、独立 artifact audit、判定与上游反馈

### 13.1 本轮目标、上一轮建议与实际 Changes

本轮目标是按第 12 节 authorization 执行一次冻结 formal，独立审计所有 artifacts，并回答 reconstruction 后 fixed-q8 waist
interval 是否闭合。上一轮建议是保持已经锁定的 target、loss、threshold、optimizer 和 status logic，不因 preflight bias调参。
本轮没有再改 inverse 方法、YAML 或 tests；formal 后只更新本第 0 节并在真实 EOF追加本节。测试用自定义 basetemp目录已按
精确路径清理；没有删除历史 run 或用户文件。

### 13.2 实际命令、tests/lint 与 formal run

formal 前验证沿用第 12 节：exp053 `5 passed`，exp051+exp053 `10 passed`，scoped Ruff `All checks passed!`，全仓
`345 passed, 12 failed`；12项均为既有 exp040 R10--R14 config hash-lock mismatch。本轮唯一 formal 命令为：

```text
conda run -n tgv_ptycho_sim python scripts/run_exp053_reconstructed_probe_q8_cell_interval_fit.py \
  --config configs/experiments/exp053_TGV_3d_multislice_reconstructed_probe_q8_cell_interval_fit.yaml
```

唯一正式 run：

```text
runs/exp053_TGV_3d_multislice_reconstructed_probe_q8_cell_interval_20260831_143704
run_state: complete
artifacts_validated: true
experiment_status: Failed
interpretation: reconstructed_target_interval_accuracy_failed
runtime: 47.4895022 s
peak sampled RSS: 145,588,224 bytes
```

runtime、memory与 HDF5 size budgets均通过；没有 development run，也没有 formal retry。

### 13.3 Primary interval、loss 与 oracle bias

raw matched-q4 `P_B_rec` 对 source `P_B_true` 的未对齐 relative L2（truth denominator）为 `0.139372011828842`；以
reconstructed target norm为分母为 `0.142182434275985`。candidate generator在 20 um 对 operator reference的 raw/amplitude/
phase-sensitive replay及 deterministic repeat均为 `0`，所以 reconstruction mismatch 与 operator replay已明确分离，`tau_primary`
保持 `1e-12`，未被 14% error放大。

primary reconstructed interval为：

```text
[21.07016531442902, 21.07033167530035) um
width = 0.16636087133046974 nm
best cell component = 177657--177658
representative midpoint = 21.070248494864686 um
```

四 starts 各 exactly 41 calls，全部到达同一 component和同一 loss；equal budget、best-cell consistency、threshold stability、
half-open endpoint、8个24-step bisections、determinism、screened uniqueness、runtime/memory gates全部通过。唯一失败 gate是
`interval_accuracy_pass_simulation_evaluation_only=false`。

raw reconstructed-target loss为：

```text
L_rec(D_true=20 um) = 0.0202158446166449
L_rec(best interval) = 0.0190483880015089
best raw residual relative L2 = 0.138015897640485
absolute loss improvement = 0.001167456615136
relative loss improvement = 5.77495839167047%
```

simulation-only accuracy 与 exp051 oracle comparison为：

```text
truth-to-reconstructed-interval distance = 1.07016531442902 um
lower/upper endpoint absolute error = 1.07016531442902 / 1.07033167530035 um
worst endpoint relative error = 0.0535165837650174
interval overlap with exp051 oracle = false
interval distance = 1.07002197396039 um
midpoint displacement = +1.07019047410393 um
lower/upper endpoint displacement = +1.07019261337613 / +1.07018833483172 um
endpoint-Hausdorff displacement = 1.07019261337613 um
```

该 bias约为 `0.125 um` accuracy budget的 8.56倍，因此 formal按预注册逻辑为 Failed；窄 interval 和四起点一致只说明
optimizer稳定找到 biased q8 cell，不能把稳定性误写为 accuracy通过。

### 13.4 Direction diagnostic 与 exp042恢复依据

由于 truth不在 interval，按冻结 trigger执行 simulation-only diagnostic。reconstruction error 对 `D_true-0.125 um` finite-change
direction的 real cosine为 `-0.4194387867`、complex coherence为 `0.4386450231`；对 `D_true+0.125 um` 分别仅为
`0.0101014567/0.1645796441`。对 `P_best-P_true` candidate-manifold direction为 `0.2577450007/0.3826634030`，real projection
coefficient为 `0.7344688089`。这些数值不进入 fitter，也没有预注册显著性阈值；但与 1.07-um accuracy failure合看，已经把
“是否恢复 exp042”的问题从全场14%误差缩小到 waist-sensitive/candidate-manifold投影这一单一误差方向。

### 13.5 HDF5/JSON/figures 独立审计

HDF5为 `outputs/exp053_reconstructed_probe_q8_cell_interval_fit.h5`，size `40,364,688 bytes`，SHA256
`7BCD5E1D817C497630474228D680B5058E5DD959FC8DB27E2DAA004438E83E8A`。`/entry` children严格为
`config_yaml/data/instrument/metadata/metrics/reconstruction/sample/truth`，`data`为空，无伪 calibration/preprocessing。
experiment-specific root为 `/entry/reconstruction/waist_fit/reconstructed_target_q8_cell_interval`；保存 raw `P_B_rec`、source/
oracle hashes、operator replay、profiles、四支完整 tracks、components、endpoint/nextafter、bisection、reported interval、best raw
candidate/residual、comparison、gates/resources、direction diagnostic和164-entry candidate cache。

独立 traversal得到946个 datasets、855个 numeric datasets，全部 finite。raw target保持 `(96,96) complex128`，byte SHA256
`F6323C7A7676CEA2F8FD8D99EC2CEC0CA6C4C9234BD49F122F17BEDE4DDDC738`；midpoint-cache和 residual mapping逐元素exact；
branch calls为 `[41,41,41,41]`，bisection iterations为8个24；未保存任何 global-phase/truth/`P_B_rec` aligned copy；两个
provenance flags均 false。外部 metadata/metrics 与 HDF5同义树已由 runner validator逐项exact验证。

四张 PNG 的 read-back、finite与目视检查均通过，无截断或不可读标签：global profile明确显示20-um oracle和约21.07-um
reconstructed minimum分离；local staircase显示约-1070-nm oracle displacement；四支 track收敛一致；direction图完整显示三组
projection/coherence。PNG只作人工审阅，全部数值均在 HDF5/metrics。

### 13.6 判定、限制、问题重要性与停止条件

高重要性 source/provenance/aligned-target/operator replay/artifact矛盾全部关闭。当前唯一主要问题为中重要性的
reconstruction-induced waist bias：证据是稳定、唯一但与 truth/oracle相距约1.07 um的 interval，以及 reconstruction error对
waist-sensitive directions的中等投影；影响是 fixed-parameter reconstructed-probe baseline未闭合，不能启动 exp055。停止条件是
不在 exp053 内换 mask、threshold、gain、seed、budget或引入 nuisance，也不为降低 full-field metric做无目标 reconstruction sweep。
低重要性的图形样式和性能优化不影响判定，记录后停止。

限制继续包括 single noiseless known-B matched case、fixed q8 single parameter、raw full-field loss、selected exp040 scalar
working model；`reference_validated=false`、`full_tgv_reference_authorized=false`。本结果不是 full-wave、真实器件、真实计量、
resolution、detection limit、noise robustness或 production reconstruction结论。

### 13.7 按重要性排序的上游手工反馈块

1. **给 exp042（中，最高优先级）**：exp051 oracle通过而 exp053 raw matched-q4 fit在全部 numerical/search gates通过后仍偏
   `+1.07019 um`，并且 error对 `D_true-0.125 um` direction的 `|real cosine|=0.4194`、coherence `0.4386`，对 best
   candidate-manifold direction的 coherence `0.3827`。建议只恢复“压低/约束 reconstruction error在 fixed-q8 waist-sensitive
   candidate-manifold方向的投影”这一问题；不要默认延长60-step budget、换 seed、改 detector branch或恢复全部最低谱端消融。
2. **给 exp051（方法反馈）**：true-probe replay/tau来自零残差 oracle，不能机械替换为 `P_B_rec`；否则14% reconstruction
   error会成为伪容差。exp053采用 raw-loss best screened q8 cell + 相对 best candidate field的固定 `1e-12` connected
   equivalence component，保留四起点、breakpoint、boundary和0.125-um accuracy contract。
3. **给 exp040（无恢复触发）**：candidate replay为0，q8 geometry/threshold/endpoint/boundary全部闭合；exp053 accuracy失败不提供
   exp040 reference-validation失败的新证据，继续保持其 Frozen/Paused 与 all-false reference状态。
4. **给 exp041（无触发）**：当前 measurement/reconstruction至少保留了一个稳定、screened-unique的 waist-related minimum；现有
   证据首先指向 recovered-probe error direction，而不是证明 B/scan design丢失腰径信息，因此不做无目标 B/scan sweep。

### 13.8 最可能下一步与 Git 状态

最高优先级是把第 13.7 的定向证据交给 exp042任务流，限定为 waist-sensitive error-projection control。不要新开 exp055；exp053
只在 exp042产生同一 provenance下的新 raw matched probe后继续做相同冻结 fitter的复验。最多两个次要建议：先在 exp042用
measurement-only可实现的 regularization/early-stop proxy预注册其与该方向的关系；若无法 truth-free控制该方向，则记录为
Inconclusive并停止，而不是调参追 metric。

最终 Git cached diff为空；没有执行 `git add`、commit、push、PR、merge或branch变更。五个 exp053 source/config/script/test/doc
文件均为本地 unstaged/untracked，formal run继续由 Git ignore；开始时已有用户 modified/deleted/untracked内容均未覆盖、删除、
暂存或提交。

## 14. 2026-08-31：exp042 feedback-control raw handoff 与冻结 fitter 复验

### 14.1 目标、上游触发与明确边界

exp042 第 30 节提供新的 validated raw matched-q4 GN-CG probe；本轮只把它接入第 13 节冻结的 exp053 fitter并执行一次 formal复验。
source从 `20260822_195139/.../matched_q4/P_B_rec` 换为
`20260831_183139/.../spectrally_damped_gn_cg/P_B_rec`，raw dataset SHA256为
`194C7B950F8DCF2BF94212A6F63270296BC9EC79AE0C04ED557C864F5FE8EB07`。没有修改 loss、mask、q8 generator、threshold、4 starts、
41-call equal budget、accuracy contract或 status logic；truth仍只作 simulation evaluation。

### 14.2 Change 01 — strict双源接口与 artifact contract correction

改动前主要矛盾是 loader把 matched-q4身份硬绑定到旧 HDF5路径/branch，导致新的同 operator raw field被拒绝。修正后 validator仅允许旧 exact raw path
和新 feedback exact raw path；新 branch必须同时通过 config R8/q4 provenance、HDF5 matched-q4/same B-scan-q4 flags、四个 truth-use false、
run-state raw path/hash、plane/grid/hash检查。旧 contract保持兼容，aligned/truth/q1输入继续禁止。

runner原先硬要求四支 bisection，无法保存 status logic正常产生的 Inconclusive/空 bisection；改为与 result bisection tree exact对比，未放宽
任何 gate。修改 exp053 YAML、inverse loader、runner、test和本文；未改 exp040/exp042算法或项目级 schema。

验证过程如实包含：旧路径 preflight `ValueError`；focused loader `1 passed`；Ruff import-order `I001/E402`后机械修正；完整 suite先
`2 failed, 3 passed`、再 `1 failed, 4 passed`（均为旧结果硬编码/validator contract），最终 `5 passed in 37.69s`；scoped Ruff最终
`All checks passed!`。未运行 full pytest，因为影响面限定于 exp053 source/runner contract且 targeted suite包含真实 locked source与tiny artifact。

### 14.3 唯一 formal、结果与判定

唯一新 formal：

```text
runs/exp053_TGV_3d_multislice_reconstructed_probe_q8_cell_interval_20260831_185434
status=complete
artifacts_validated=true
experiment_status=Inconclusive
interpretation=reconstructed_q8_interval_numerical_control_not_closed
runtime=33.835074300001 s
peak sampled RSS=137359360 bytes
```

reported interval为 `[20.00301349202269, 20.00327036622095) um`，width `0.2568741982597806 nm`。simulation-only truth distance
`3.013492022687817 nm`、oracle interval distance `2.8701515540631407 nm`、endpoint Hausdorff `3.1270257523229213 nm`；accuracy gate已为 true。
相较第 13 节约 `+1.07019 um` bias，这是约三个数量级的降低，但不能据此跳过 numerical gates。

四 starts均 exactly 41 calls；start_00/01/02落在 cell `139603`，start_03落在 `139608`，后者 final best-field relative L2
`0.00016730662496125193`，因此 `all_final_seeds_qualify=false`、`interval_agreement=false`、endpoint/bisection未执行、
`numerical_controls_pass=false`。screened uniqueness、operator replay、threshold stability、determinism、equal-budget、runtime/memory均通过。
`L_rec(true)=0.010410295555232872`、`L_rec(best)=0.010391415826649259`，relative improvement仅 `0.1813563168%`。

### 14.4 Artifact独立审计与上游反馈

HDF5 `32,748,336` bytes，SHA256 `712A67A3A586E90A1CB5D27845FA8332176AD9D3E793AEC993D9CB7D5F11318C`；754 datasets、
466 numeric datasets全部 finite。raw source `(96,96) complex128`与 exp042 raw逐元素 exact，hash `194C...B07`；未保存 aligned target。
四支 tracks各41 calls，bisection group自然为空，config/metadata/metrics/HDF5/run-state hashes一致。四张 PNG read-back与目视检查通过，tracks图明确显示
第4支停在邻近不同 cell；全部数值仍以 HDF5/JSON为准。

- 给 exp042：定向 GN-CG control已把原 waist bias大幅降低，exp042恢复目标可视为已回答；不要继续调 damping/CG或恢复 lowest-spectrum。
- 给 exp040：operator replay、matched chain和 flags闭合，但没有新增 full-wave/reference证据；继续 Frozen/Paused、两个 reference flags为 false。
- 给 exp053：当前唯一主要矛盾是 bounded numerical convergence，不是 accuracy。下一轮只允许先预注册一个所有四 starts同规则、同预算的
  truth-free continuation/budget-extension control；不得按 truth选步数、放宽 qualifier、换 seed或启动 exp055。

### 14.5 快速恢复上下文与 Git状态

- authoritative section为本第14节；当前 config根文件 SHA256
  `DF80204E81946713AF42AE567BC6908B81B59423C8D00445B60AC7A27500855D`，最新 run为 `20260831_185434`。
- 已通过 exp053 focused `1 passed`、最终 targeted `5 passed`和 scoped Ruff；无需重跑 exp042 formal、exp040 reference、exp051 oracle或本 formal。
- 下一轮最小读取：本文第0、13、14节；exp042第0、30节；当前 exp053 YAML/inverse/runner/test；最新两个 run的 run_state/metrics与 raw path。
- 只有 source/operator/hash冲突才重读 exp040；不要扫描全部 runs/src/tests、notebooks/reports/data或其他 exp05x。
- staged为空；未 add/commit/push/PR/merge/branch。用户既有 modified/deleted/untracked内容未回退、覆盖、删除或暂存。

### 14.6 Append-only基线

更新第0节、追加本节前本文为 `24904` bytes、SHA256
`E94F4A78FC1D20B1BE8D8A77EF291D19648E28BCC99951F57E86DEFD41BF7DD1`；第1--13节历史区间为 `23849` bytes、SHA256
`7F314715659D070B44E5F477B68BA5B7B334E60209215483A5E7EB21BC89F945`。下面独立 note记录验证结果。

### 14.7 Append-only独立验证 note

- 首次更新第0节并追加第14节后，本文为 `30514` bytes、SHA256
  `1A26DF7E7F02E87FE6A9050B55F3540F247FAB8DBBA84134D0ED8EE3474175C1`。
- 新文件从新 `## 1.` 起的前 `23849` bytes SHA256仍为
  `7F314715659D070B44E5F477B68BA5B7B334E60209215483A5E7EB21BC89F945`，与本轮前第1--13节历史区间 exact；下一 byte为新增 `LF`，
  随后才是第14节。历史章节未修改、删除、重排或润色。
- heading尾序为 `12→13→14`；除第0节实时状态外，本轮只向旧 EOF追加内容。

## 15. 2026-09-01：多起点最小等预算数值闭合 control

### 15.1 Formal 前主要矛盾、轨迹证据与预注册

本轮由用户明确授权继续 exp053，但不得预先锁死增加预算或更换 optimizer。修改前只读 `AGENTS.md`、exp042 第 0/30 节、
本文第 0/13/14 节、当前 YAML/inverse/runner/test，以及最新 exp042/exp053 run 的锁定 source、metrics 和四支完整 HDF5 tracks；
没有读取 truth 来选择方法、参数、停止点或结果，也没有扫描全部 runs/src/tests、exp040 全文、notebooks/reports/data 或启动 exp055。

唯一主要矛盾是：在 authoritative raw GN-CG probe 上，前三支 41-call pattern search 已到共同 q8 component，`start_03` 因同预算下少完成
一个局部 pattern update而停在邻近 cell `139608`，使 `all_final_seeds_qualify=false`、`interval_agreement=false`，并阻断 endpoint/bisection。
最多三个次要矛盾是：必须保持四支公平同预算；不得把约 3 nm simulation-only accuracy 改善反馈进方法选择；artifact validator/tests 仍把
`41` 写成冻结常数。source/provenance、candidate replay、threshold、screened uniqueness、determinism、runtime/memory和 accuracy gate 均已闭合，
因此本轮不改 source、candidate generator、loss、threshold、starts、seed、interval/status logic 或 accuracy contract。

方法选择只使用 raw reconstructed-target objective 与既有 41-call optimizer tracks。`start_03` 的末端 incumbent 为
`20.00341796875 um`、step为 `0.244140625 nm`；按原 pattern-search 规则，下一个完整更新必须成对评估左右点，其中左点
`20.003173828125 um` 已由其他起点在同一 truth-free objective 下评估且具有更低 loss `0.010391415826649259`，位于共同 best-field
component。更换 optimizer 会改变归因；大幅预算 continuation 超出闭合所需；按单支自适应补调用会破坏 equal-budget。因此预注册唯一 control为：

```text
algorithm: unchanged fixed_budget_bounded_pattern_search
baseline budget: 41 calls/start
extension: exactly one complete symmetric pattern update = 2 calls/start
control budget: 43 calls/start
starts / initial step: unchanged
selection evidence: existing raw-target loss and tracks only
truth use: simulation evaluation only after freeze
```

本 control 的公平比较单位是每支相同的 objective-call count；四支无论是否较早进入 component都执行 exactly 43 calls，停止原因仍必须为
`evaluation_budget`。不得在看到 control结果后改为45或更高 budget，不得换 seed、放宽 `1e-12` qualifier、`1e-14` tie tolerance、boundary/
accuracy gates或把 simulation truth用于 branch/stop选择。若43 calls仍不闭合，formal必须保留为 Failed/Inconclusive并停止本轮。

计划只修改 exp053 YAML、inverse validator/result contract、runner artifact validator、targeted tests和本文第0/15节；先运行 targeted pytest与
scoped Ruff，通过后只执行一个 timestamped formal，再独立审计 JSON/HDF5/figures/provenance。预注册前本文为 `31086` bytes、SHA256
`06E5167181E2360333FA04E0FA7589CD65C202E872FCCA266AA5186D368CA89D`；root config SHA256为
`DF80204E81946713AF42AE567BC6908B81B59423C8D00445B60AC7A27500855D`。后续结果只追加到本节之后，第1--14节不回改。

### 15.2 实现、targeted验证与未绕过的新增 gate 证据

实际 Changes严格限定为：YAML把每支 budget由41改为43并登记 baseline/一个完整 update/两次调用/truth-free selection/equal-budget字段；
inverse validator把该control锁为 exact contract并把它写入 result；runner把同义control写入 metadata/metrics/HDF5，artifact validator不再硬编码41，
而是读取已验证 config budget并逐项核对 closure group；tests锁定43 calls、四支共同 component、八个24-step bisections及禁止 post-hoc 45-call
修改。没有改 `equal_budget_bounded_pattern_search()`、candidate generator、loss、cache、q8 partition、threshold、component qualifier、status logic、
source/oracle hash或 truth boundary。

修改文件及 formal 前最终 SHA256：

- `configs/experiments/exp053_TGV_3d_multislice_reconstructed_probe_q8_cell_interval_fit.yaml`：
  `97B13D90F6579BD726D71DB2B724A59E57ABE2BEB30A1C09EEB314E44EF39EE7`；
- `src/tgv_ptycho/inverse/exp053.py`：`2B6AC541B41920473BF30E92BBC5A92B2DFD8DB709280ED3C9E82AA1789B3864`；
- `scripts/run_exp053_reconstructed_probe_q8_cell_interval_fit.py`：
  `5A02DE4A36BFD55DB4E42B1EFF1B61B04BCC9519ED5F97B48F5550E03FFF5083`；
- `tests/test_exp053_reconstructed_probe_q8_cell_interval_fit.py`：
  `B5ABE793E3B0A715D3D6E4707D56B6931016B8A4F74129FE0697C0C2C0BE3E8A`；
- 本文第0节与真实 EOF 本第15节。

首次 targeted suite得到 `2 failed, 3 passed in 51.33s`。失败不是多起点：四支已经全部43 calls进入 cell `139603`，
`all_final_seeds_qualify=true`、`interval_agreement=true`。失败来自此前被 start_03 gate阻断、从未运行的 lower-boundary control：四支lower
bisection均收敛到 `[2.0003013492022683e-5, 2.0003013492022686e-5] m`，而解析 breakpoint为
`2.000301349202269e-5 m`，解析值正好是 bracket upper的下一个 float64；此外解析 lower breakpoint的 `nextafter(-inf)`仍为member，
违反已冻结的 lower-closed source-convention检查。upper boundary四支全部通过。

本轮没有把该 one-ULP差异事后改成通过，也没有放宽解析-in-bracket、half-open或boundary-resolution gate。只把测试从事前预期 Passed改为锁定
实际、可复现的 `Inconclusive`，并明确断言四支lower fail/upper pass以及 `nextafter(bracket_upper,+inf)==analytic_breakpoint`。第二次 targeted
suite为 `5 passed in 46.60s`。scoped命令与结果：

```powershell
D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim python -m pytest -q tests/test_exp053_reconstructed_probe_q8_cell_interval_fit.py
D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim python -m ruff check src/tgv_ptycho/inverse/exp053.py scripts/run_exp053_reconstructed_probe_q8_cell_interval_fit.py tests/test_exp053_reconstructed_probe_q8_cell_interval_fit.py
```

最终为 `5 passed`、Ruff `All checks passed!`。未运行 full pytest或 exp051+exp053组合回归，因为未修改共享 optimizer、exp051、forward/optics/IO；
真实 source、oracle regression、candidate replay与tiny runner均已在本 targeted suite覆盖。

### 15.3 唯一 formal、结果与科学判定

唯一正式命令为：

```powershell
D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim python scripts/run_exp053_reconstructed_probe_q8_cell_interval_fit.py --config configs/experiments/exp053_TGV_3d_multislice_reconstructed_probe_q8_cell_interval_fit.yaml
```

唯一新 run：

```text
runs/exp053_TGV_3d_multislice_reconstructed_probe_q8_cell_interval_20260901_120005
status=complete
artifacts_validated=true
experiment_status=Inconclusive
interpretation=reconstructed_q8_interval_numerical_control_not_closed
runtime=49.145791000000145 s
peak sampled RSS=142848000 bytes
```

本轮主要矛盾已经闭合：四支均 exactly 43 calls、`stopping_reason=evaluation_budget`、final cell均为 `139603`、final loss均为
`0.010391415826649259`、final best-field relative L2均为 `0`；equal-budget、all-seeds-qualify和interval-agreement gates全部通过。
相对上一 run，`start_03`只通过预注册的一个完整左右评估更新进入共同 component，没有按单支追加调用。

reported interval保持 `[20.00301349202269, 20.00327036622095) um`，width `0.2568741982597806 nm`；simulation-only truth distance
`3.013492022687817 nm`、oracle interval distance `2.8701515540631407 nm`、endpoint Hausdorff `3.1270257523229213 nm`，accuracy gate仍通过。
`L_rec(true)=0.010410295555232872`、`L_rec(best)=0.010391415826649259`，relative improvement `0.18135631676780646%`。

整体没有升级为 Passed：四个lower bisection的解析-in-bracket与 endpoint half-open convention均失败，四个upper bisection均通过；因此
`boundary_bisection_pass=false`、`endpoint_half_open_convention_pass=false`、`numerical_controls_pass=false`。其余 candidate replay、partition、
best-cell consistency、determinism、threshold stability、adjacent cells、cell cap、screened uniqueness、equal budget、multi-start、accuracy、runtime和
memory gates全部通过。该状态说明“多起点预算不足”已解决，但解析 q8 partition边界与实际 source generator的 float64 half-open transition仍未闭合；
不得把约3 nm accuracy结果越过该数值边界写成 Passed。

### 15.4 Artifact、provenance与figure独立审计

run artifact hashes：config `149243947EEA9D6C518CDA2F2B573C2BB309601AEFFB7C2FDDACD3D604BF6900`；metadata
`AA71FACBA4E55B45FA9982C9467328D1A5FCF6972BAA69CA981EAE7B7B05D41A`；metrics
`ED1AF922A8E4B3733922996CCA905BF9BCC5F8A0307D931C0C06E648B29F6EF0`；run-state
`351CC11287B5BEF114D60831136FA9A861BD80241BE2D8E830E0FB33DB8C335B`；HDF5
`25BF9E9C287874BD0836AEA2BF31D95A8D2F1E941F79D16030AE30D3F6A7DF96`，size `36,836,912` bytes。

独立 HDF5 traversal得到968 datasets、874 numeric datasets，全部 finite；`/entry`仍为
`config_yaml/data/instrument/metadata/metrics/reconstruction/sample/truth`，无伪 calibration/preprocessing。metadata/metrics外部 JSON与HDF5递归exact。
raw source仍为 `(96,96) complex128`，dataset-byte SHA256
`194C7B950F8DCF2BF94212A6F63270296BC9EC79AE0C04ED557C864F5FE8EB07`，与 exp042 authoritative raw逐元素exact；source path仍为
`/entry/reconstruction/exp053_feedback_control/branches/spectrally_damped_gn_cg/P_B_rec`。未保存任何 aligned target，两个 reference flags均false；
best-candidate/cache与raw residual mapping逐元素exact。experiment-specific optimizer下新增/记录 `numerical_closure_control`，未改变项目级schema。

四张 PNG均完成SHA/read-back和目视审计，tracks图显示四支在第21个pattern update后重合，其他图无截断、损坏或不可读标签；PNG仅作人工查看：

- four-start `E19F9F0D42DC5577E9AA52A76516C3F604BB4E1B55AF60884A965AA346DF90CC`；
- local staircase `6A05391875CC761B8B2C65607F5491D70F768ECDD8D7F047F16630920AFA9351`；
- global profile `96E394D71F20A420B19CD199C11EC2083C6D939DD2C19A1D434E9A7C5E085135`；
- direction diagnostic `8E0769A9A95290980884963BF808E467A2A883B40BC96813E73D923D0E66B25B`。

### 15.5 下一轮建议、快速恢复上下文与Git状态

下一轮唯一主要矛盾已从 multi-start转为：解析公式生成的 q8 lower breakpoint与 candidate generator实际执行
`diameter_profile -> radius_squared -> <=`时的 source-exact representable transition相差一个 float64 ULP。建议继续 exp053，但先预注册一个
纯数值、truth-free的 boundary-semantics control：分别保存 analytic breakpoint与 source-exact first-inside/first-outside representable values，
用 bounded `nextafter`和node-count/field membership证明映射，再决定是否需要统一修正 partition edge语义。不得简单把“一 ULP内”设为通过、强制
`analytic_breakpoint_in_bracket=true`、修改 `tau`/bisection iterations或只修当前lower endpoint；若不能给出对所有相关breakpoint一致的规则，保持
Inconclusive并停止。暂不启动 exp055，也不返回 exp042/exp040/exp041。

快速恢复上下文：authoritative exp042仍为第30节/run `20260831_183139`；authoritative exp053为本第15节/run
`20260901_120005`。当前 root config SHA256为 `97B13D...9EE7`，raw source path/hash仍为上述 GN-CG path/`194C...B07`。
已通过 targeted `5 passed`与scoped Ruff；本 formal complete/validated。下一轮最小读取为 `AGENTS.md`、本文第0/14/15节、exp042第0/30节、
当前 YAML、`exp051_local_control.q8_waist_breakpoint_map()`、`tgv3d.make_tgv_air_fraction_slice()`、exp051 plateau的partition/cell/bisection函数、
exp053 endpoint control及最新 run的metrics/HDF5。不要扫描全部历史runs/src/tests、exp040全文、notebooks/reports/data或其他exp05x。

Git staged仍为空；未执行 add/commit/push/PR/merge/branch。用户既有 modified/deleted/untracked内容均未回退、覆盖、删除或暂存；新 run由Git ignore，
上一 Inconclusive run与所有失败证据均保留。

### 15.6 Append-only独立验证 note

- 15.1预注册完成、结果追加前，本文为 `34400` bytes、SHA256
  `CFC86142C84D9549ACDB91EEA2F462CA76D182BBD1A17FD94A3FA4DAD827D580`；从 `## 1.` 起的既有区间为 `33321` bytes、SHA256
  `AB788CC3DD6CAE8104BF65025E1937810B7578906AA5271064BE0497A5110F83`。
- 更新第0节并追加15.2--15.5后，本文为 `43400` bytes、SHA256
  `DCF41A99046850DCF7444ED90B5C0F21198FE2DABA1BB648E60D9B8B77768A81`；新文件从新 `## 1.` 起的前 `33321` bytes SHA256仍精确为
  `AB788CC3DD6CAE8104BF65025E1937810B7578906AA5271064BE0497A5110F83`，下一内容才是新增 `### 15.2`。
- 因此除治理约定允许原位更新的第0节外，第1--14节和15.1预注册记录均未修改、删除、重排或润色；结果与本note只追加在预注册EOF之后。

## 16. 2026-09-01：source-exact float64 half-open boundary semantics control

### 16.1 Formal 前主要矛盾、方法比较与预注册

本轮按用户授权继续 exp053并继承第15.5节唯一建议。修改前只读 `AGENTS.md`、本文第0/15节、当前 YAML、latest run state、
`diameter_profile()`、`q8_waist_breakpoint_map()`、`make_tgv_air_fraction_slice()`、partition/cell/bisection与 exp053 endpoint/test路径；
没有读取 truth来选择 boundary规则、ULP budget、branch、停止点或结果，也没有扫描无关历史或启动 exp055。

唯一主要矛盾：`q8_waist_breakpoint_map()`以代数反解 `(2*r-constant)/alpha`生成解析 edge，而 source generator实际执行
`diameter_profile -> (diameter/2)^2 -> node_radius_squared <= radius_squared`；代数等价不保证 float64逐操作等价。第15节已证明当前lower edge的
解析值比实际 transition高一个 ULP，而upper edge恰好一致。最多三个次要矛盾是：必须同时处理 lower/upper且不能只特判当前值；analytic partition仍需
原样保留用于cell topology与诊断；reported interval、bisection、JSON/HDF5/tests必须清楚区分 analytic edge和source-exact representable boundary。

排除的方法：不采用“解析值距离 bracket 一 ULP内即通过”，因为这是事后容差；不强制把
`analytic_breakpoint_in_bracket`写成true；不改 `tau`、24-step bisection、q8 generator或 `<=` source convention；不全局改写 exp051的解析 partition，
因为本问题只要求给当前 connected component建立可验证的执行语义，且共享 oracle artifact必须保持不变。

预注册唯一方法为统一、truth-free的 bounded representable-transition scan：

```text
method: bounded_float64_nextafter_transition_scan
analytic center: existing q8 component edge
scan budget: exactly 8 float64 values in each direction plus center
lower source-exact boundary: first representable member immediately after an outside value
upper source-exact boundary: first representable outside value immediately after a member
membership triplet: tau-primary field / exact-best-field / q8 node-count equality
required behavior: all three memberships agree at every scanned value; exactly one monotone transition; scan cap not hit
reported/bisection boundary: source-exact transition
analytic edge: retained separately as algebraic diagnostic, never coerced into the bracket
truth use: none in mapping, selection, status or stopping
```

`8 ULP/side`是固定审计调用上限，不是“八 ULP内即科学等价”的容差；只要三种 membership不一致、出现多重/非单调 transition、或扫描到cap仍未找到
两侧，就使新的 mapping gate失败并保持 Inconclusive。对 lower与upper调用完全同一函数和同一预算。只有 mapping gate通过时，reported interval才采用
source-exact first-inside/first-outside端点；analytic endpoints、ULP offsets、全扫描数组和两套 in-bracket flags必须全部保存。bisection pass继续要求
24 iterations、原 boundary-resolution阈值及source-exact boundary在 bracket内；旧 analytic-in-bracket值保留为诊断但不伪装为 source transition。

本轮保持 exp042 source/hash、43-call四起点、loss、mask、threshold、tie、component、accuracy/status顺序、runtime/memory和 truth boundary不变。
计划只修改 exp053 YAML/module/runner/test和本文第0/16节；先做 targeted与共享 q8 regression、Ruff，通过后只执行一个 timestamped formal并完整审计。
预注册前本文为 `44218` bytes、SHA256 `3878B73F98AC086FF3F8E75DA61AA8C9C753AC8601564F1B1ECC793953939BED`；
root config SHA256为 `97B13D90F6579BD726D71DB2B724A59E57ABE2BEB30A1C09EEB314E44EF39EE7`。后续结果仅追加到本节之后。

### 16.2 实现、tests/lint与方法边界复核

实际实现严格采用16.1预注册规则。`_source_exact_boundary_scan()`从analytic edge向两侧各生成exactly 8个连续float64 `nextafter`值，加中心共17点；
每点分别计算tau-primary field membership、与best field逐元素exact、q8 node-count stack逐元素exact，要求三条布尔序列完全相同、只有一个符合side方向的
monotone transition、且scan两端已覆盖transition两侧。lower取outside之后第一个representable member，upper取member之后第一个representable outside；
若control失败则保留analytic endpoint并让新增mapping gate失败，不伪造source endpoint。

candidate cell topology仍由原analytic partition和cell midpoints确定；reported interval改为source-exact执行边界，并同时保存
`analytic_lower/upper_m`与`boundary_semantics`。endpoint control使用source-exact half-open值；bisection同时保存analytic和source-exact breakpoint、
两套in-bracket flags及ULP offset，pass只认通过三重membership mapping后的实际source transition。旧analytic lower的
`analytic_breakpoint_in_bracket=false`被如实保留，没有改成true或删除。

修改文件及formal前SHA256：

- config：`16B8FBA20CF7DD97C68FC80BC9AA585BCEFA851F6929C9542F1E259A65DC8B54`；
- `src/tgv_ptycho/inverse/exp053.py`：`250F9145463A52BB02BC63C4110094832862901666E939754BF82651F8EA4054`；
- runner：`18B58D062C871FABE9E0953FC78D06D53B9BEDAC7C4DDB0A9500257BC5F08040`；
- tests：`68DF892B623BB695B08B303D81DA4EDF0D38314FACAB8D0C0DCBDA646B3D893E`；
- 本文第0/16节。

实际验证命令：

```powershell
D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim python -m pytest -q tests/test_exp053_reconstructed_probe_q8_cell_interval_fit.py
D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim python -m pytest -q tests/test_exp051_q8_plateau_interval_fit.py tests/test_exp053_reconstructed_probe_q8_cell_interval_fit.py
D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim python -m ruff check src/tgv_ptycho/inverse/exp053.py scripts/run_exp053_reconstructed_probe_q8_cell_interval_fit.py tests/test_exp053_reconstructed_probe_q8_cell_interval_fit.py
```

结果依次为 `5 passed in 63.18s`、`10 passed in 118.14s`、`All checks passed!`，无失败。组合回归确认exp051 oracle解析partition/replay保持exact；
没有修改共享 exp051 partition/bisection函数或旧oracle artifact。未运行full pytest，因为影响面由真实source targeted、tiny artifact和exp051组合回归覆盖，
且全仓既有exp040 hash-lock failures与本Change无关。

### 16.3 唯一 formal、source-exact证据与判定

唯一正式命令：

```powershell
D:\anaconda3\Scripts\conda.exe run -n tgv_ptycho_sim python scripts/run_exp053_reconstructed_probe_q8_cell_interval_fit.py --config configs/experiments/exp053_TGV_3d_multislice_reconstructed_probe_q8_cell_interval_fit.yaml
```

唯一新 run：

```text
runs/exp053_TGV_3d_multislice_reconstructed_probe_q8_cell_interval_20260901_122210
status=complete
artifacts_validated=true
experiment_status=Passed
interpretation=reconstructed_probe_q8_cell_interval_fit_passed
runtime=57.143496799999866 s
peak sampled RSS=184147968 bytes
```

source-exact scan得到：lower的17点三种membership序列均为前7个outside、后10个member，唯一transition index `7`，first-inside为
`2.0003013492022686e-5 m`，相对analytic `2.000301349202269e-5 m`为 `-1 ULP`；其previous为
`2.0003013492022683e-5 m`且outside。upper序列为前8个member、后9个outside，唯一transition index `8`，first-outside与analytic均为
`2.000327036622095e-5 m`，offset `0 ULP`。两侧membership triplet agreement、unique monotone transition、no-cap与mapping pass全部true。

四支仍exactly43 calls、同到cell `139603`、loss `0.010391415826649259`、best-field relative L2 `0`；八个24-step bisections全部pass。
四个lower保留 `analytic_breakpoint_in_bracket=false`，但source-exact first-inside位于bracket upper且
`source_exact_breakpoint_in_bracket=true`；四个upper的analytic/source-exact均在bracket。由此
`source_exact_boundary_mapping_pass=true`、endpoint half-open、boundary bisection、numerical controls、equal-budget、多起点agreement、screened
uniqueness和accuracy等全部预注册gates为true，formal按未改status顺序为Passed。

最终reported interval：

```text
[20.003013492022685, 20.00327036622095) um
width = 0.256874198263169 nm
source-exact lower/upper ULP offsets = -1 / 0
```

simulation-evaluation-only truth distance为 `3.013492022684429 nm`，oracle interval distance `2.8701515540597525 nm`，midpoint displacement
`+3.0839083610635567 nm`，endpoint Hausdorff `3.1270257523229213 nm`，均低于冻结 `125 nm` accuracy budget。
`L_rec(true)=0.010410295555232872`、`L_rec(best)=0.010391415826649259`，relative improvement仍为 `0.18135631676780646%`。
truth没有进入source-exact mapping、optimizer、branch、threshold、stop或status；direction diagnostic仍只因truth不在极窄reported interval而作为
simulation evaluation执行。

### 16.4 Artifact/HDF5/figures独立审计

run hashes：config `7C9CF03E8587170817DBD9CEE4A714FF5AA93574D9833242C20F55967040EAA7`；metadata
`63CFCE2803480494B6C9A9FCD0EB4C14F061CC9E929292E64CBB8D69E75C1239`；metrics
`832CD202A018470035099D96136FA0D82608B1C97CAF97EDB021033F405A3EF2`；run-state
`3836BD3B99BCB3CBAFCA2184724D813AC9544BAA9FBBFAEF22F8C6C4F5A02AB8`；HDF5
`CF1697FC1B36CFCD9647FFC07A4BC68FD02C97ED1C329AE4313C9719467D0C1F`，size `42,609,752` bytes。

独立 traversal得到1181 datasets、1052 numeric datasets，全部finite；外部metadata/metrics与HDF5递归exact。`/entry`结构不变，无伪
calibration/preprocessing。raw source仍为 `(96,96) complex128`，byte SHA256
`194C7B950F8DCF2BF94212A6F63270296BC9EC79AE0C04ED557C864F5FE8EB07`，与 exp042 authoritative raw逐元素exact；没有aligned target，
两个reference flags均false，best/cache/residual mapping exact。candidate cache为166 fields。

HDF5只在experiment-specific result/metrics/metadata内增加 `source_exact_boundary_control`、两侧17点scan arrays、analytic/source-exact endpoints、
ULP offsets和bisection双重reference flags；没有项目级schema变化，也没有改source/oracle datasets。

四张图完成hash/read-back/目视审计，无截断、损坏或不可读标签；global profile标题为Passed，其余科学内容与数值一致：four-start
`E19F9F0D42DC5577E9AA52A76516C3F604BB4E1B55AF60884A965AA346DF90CC`，local staircase
`6A05391875CC761B8B2C65607F5491D70F768ECDD8D7F047F16630920AFA9351`，global profile
`5AFE60A5708AAEB6BAB801953D061475376A8C9FDFB8A7D283E4D360FE81F426`，direction diagnostic
`8E0769A9A95290980884963BF808E467A2A883B40BC96813E73D923D0E66B25B`。PNG仍不作为计算输入。

### 16.5 结论、限制、下一步与快速恢复上下文

exp053当前研究问题已经闭合：同一validated raw GN-CG probe在固定q8单参数、raw full-field loss下形成多起点一致、source-exact half-open、
artifact-validated且满足冻结accuracy budget的interval。应冻结本实验，不再调budget、threshold、mask、seed、optimizer或boundary semantics。

Passed只覆盖 selected exp040 scalar working model、单一deterministic/noiseless/known-B/matched q4 case与固定参数candidate family；不表示真实三维电磁
准确性、真实计量精度、noise/blind-B robustness、resolution、detection limit或production reconstruction。`reference_validated=false`、
`full_tgv_reference_authorized=false`继续成立。analytic edge与source-exact边界的区别属于float64执行语义，不是新的物理精度证据。

下一步不再属于exp053的小修正。若用户以后明确授权nuisance identifiability，应作为exp055独立任务重新预注册；在当前授权下暂不启动exp055。
也无需继续恢复exp042、exp040或exp041。可选治理工作只是整理发布边界/Git文件清单，但没有自动commit权限。

快速恢复上下文：authoritative section为本第16节，latest valid run为 `20260901_122210`，root config SHA256
`16B8FBA20CF7DD97C68FC80BC9AA585BCEFA851F6929C9542F1E259A65DC8B54`；authoritative exp042 source仍为run `20260831_183139`与raw
hash `194C...B07`。已通过exp053 `5 passed`、exp051+exp053 `10 passed`和scoped Ruff；formal complete/validated/Passed。若只需汇报无需重跑；
若artifact/hash冲突，最小读取为本文第0/14/15/16节、当前YAML/module/runner/test、latest run state/metrics/HDF5及exp042第30节。

Git staged保持为空；未执行add/commit/push/PR/merge/branch。用户既有modified/deleted/untracked内容均未回退、覆盖、删除或暂存；本轮run由Git ignore，
先前Inconclusive与failed证据全部保留。

### 16.6 Append-only独立验证 note

- 16.1预注册完成、结果追加前，本文为 `47985` bytes、SHA256
  `8B28783B242BA0F722D4EDE2240D12ED1899A8DB55DED85501CD74340D576059`；从 `## 1.` 起的既有区间为 `46897` bytes、SHA256
  `7D20AC89792F7C2542CBB1076C589EE0E83CD9E608EFDE41D2BAE951A4D3EB0F`。
- 更新第0节并追加16.2--16.5后，本文为 `56807` bytes、SHA256
  `D3460AAC7A858A6797438133A1CE5F61BC5DF3CBDC695FD92960FBB16BEE03DB`；新文件从新 `## 1.` 起的前 `46897` bytes SHA256仍精确为
  `7D20AC89792F7C2542CBB1076C589EE0E83CD9E608EFDE41D2BAE951A4D3EB0F`，下一内容才是新增 `### 16.2`。
- 因此除第0节实时状态外，第1--15节和16.1预注册记录均未修改、删除、重排或润色；formal结果与本note只追加在预注册EOF之后。
