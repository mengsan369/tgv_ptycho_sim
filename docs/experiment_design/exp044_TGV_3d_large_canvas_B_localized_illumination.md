# exp044：3D TGV large-canvas finite B 与 localized illumination measurement design

## 0. 实时状态与阅读顺序

- Scientific status：`Failed / blind_measurement_reconstruction_not_closed`。
- Work status：`Formal complete / Artifacts validated / Frozen`。
- Results available：`true`。
- Latest valid run：`runs/exp044_TGV_3d_large_canvas_B_localized_illumination_formal_20260903_212213/`。
- Latest development/preflight run：`runs/exp044_TGV_3d_large_canvas_B_localized_illumination_development_20260903_211649/` / `runs/exp044_TGV_3d_large_canvas_B_localized_illumination_preflight_20260903_210607/`。
- Latest formal run：`runs/exp044_TGV_3d_large_canvas_B_localized_illumination_formal_20260903_212213/`。
- Authoritative appended section：`Section 9 (post-formal QA seal; scientific result in Section 8)`。
- Primary current question：把 exp031 的 large-canvas finite nonperiodic B、physical scan windows 与 A-plane localized illumination 思想迁移到 selected exp040 scalar multislice working model 后，是否在 forward 和 known-B 闭合的前提下改善 exp043 暴露的 blind component non-identifiability？
- exp031 source identity：formal `runs/exp031_finite_B_illumination_spot_20260830_211803/`；root config SHA256 `71057707EF5453582C4DB32C4341DD4B208110E62D91FB4BC4D0CCCFC5B86E64`；只迁移 API、物理区域和数值诊断，不迁移二维数值结论。
- exp040 source identity：R8 formal `runs/exp040_TGV_3d_multislice_r8_unified_visibility_20260814_152034/`；R8 root config当前工作区 SHA256 `9EBF713F3E7D9F803517932E480CC4AF5300347644D2D84D9F48257519B96BEB`；继续保持 `reference_validated=false`、`full_tgv_reference_authorized=false`。
- exp042 source identity：authoritative run `runs/exp042_TGV_3d_multislice_probe_reconstruction_dev_20260831_183139/`；HDF5 SHA256 `C48588FC47EED474CF047219A1DB26BCC9242961A58176CA9CCB879495FFA37D`；raw known-B probe SHA256 `194C7B950F8DCF2BF94212A6F63270296BC9EC79AE0C04ED557C864F5FE8EB07`。
- exp043 source identity：formal `runs/exp043_TGV_3d_multislice_blind_probe_B_reconstruction_20260903_153944/`；HDF5 SHA256 `BBDF25C93383FB8EA368DDDC60B013D312E68D67AF87688032B7E50CCBB52B09`；scientific status `Failed / measurement_consistent_but_component_recovery_non_identifiable`。
- Current design matrix：E43 exp043 authoritative bridge；C0 open-multislice plane-wave + legacy B；C1 large master B only；C2 A-entrance Gaussian only；C3 combined primary。
- Selected forward strategy：A-entry Gaussian直接进入足够大的 Cartesian scalar multislice；homogeneous Gaussian reference 与 TGV-induced residual 分开传播；B→detector 使用 finite physical patch、transparent exterior、open residual propagation与 matched q4 positive detector quadrature。
- Selected known-B method：复用 exp042 spectrally damped GN-CG family、measurement-only loss、truth-free homogeneous-reference initialization。
- Selected blind method：首先复用 exp043 alternating exact phase-coordinate block GN-CG；large master 只增加 physical extract/scatter 与 observable-union parameterization，不同时更换 optimizer。
- 当前唯一主要矛盾：已解决为 negative result；C3 blind measurement residual与两初始化prediction repeatability先失败，selected design未建立可解释的component-identifiability改善。
- 下一步动作：exp044停止调参和formal重跑；measurement-design、optimizer或blind-probe waist-fitting均按Section 8建议另开实验。
- 阅读顺序：先读本节，再读Section 8；需要追溯决策时读Sections 4--7，Sections 1--3保留初始预注册历史。

## 1. 初始 source audit、研究问题与 scope

### 1.1 开始前审计

任务开始时已完整读取 `AGENTS.md`、实验任务模板、roadmap 和 data-format；定向读取 exp031 第 0/1--12/24 节、exp040 第 0/R4/R5/R8、exp042 第 0/29/30 节、exp043 第 0/5/6 节及对应 theory、config、公共实现、runner、tests 和 authoritative JSON/HDF5 identities。`git -c safe.directory=E:/tgv_ptycho_sim status -sb` 显示 staged 为空，但工作区包含用户已有 modified、deleted 与大量 untracked 研究文件；全部保留，不回退、不覆盖、不暂存。

exp043 的 forward/operator、P-only、B-only 和 blind measurement gates 已闭合。representative blind detector residual为 `0.0273522822`，aligned probe/B/exit errors为 `0.206064417 / 0.401164927 / 0.305120617`。因此 exp044 不把主要矛盾重新解释为 exp043 实现错误或 iteration 不足。

### 1.2 稳定研究问题

本实验按单一证据链回答：

```text
Forward/operator validation
-> known-B probe reconstruction control
-> blind P/B joint reconstruction primary
```

若 forward/operator 未闭合，不进入 reconstruction 解释；若 selected design 的 known-B probe control 未闭合，blind 结果不解释为 component identifiability；只有前两层闭合后，blind measurement 与 component gates 才形成科学状态。

### 1.3 Scope 与 claim boundary

允许：selected exp040 scalar multislice生成新 probe truth；finite nonperiodic master B与physical scan patches；matched q4 detector data；known-B与blind reconstruction；coverage、dynamic-range、conditioning与simulation-only component comparisons。

禁止：拟合 `D_waist`、geometry nuisance、noise/stage/calibration/真实数据、beam decenter/tilt/aberration/unknown waist plane、恢复 exp040 Helmholtz/reference路线、修改 exp031/040/042/043历史结论，或以 truth 选择 beam/seed/checkpoint/optimizer。任何 Passed 最多表示 selected scalar、noiseless、matched measurement-design scope 内的数值闭合。

## 2. 初始 measurement design 与物理区域定义

### 2.1 C0/C1/C2/C3 因果矩阵

| case | physical incident field | B boundary/design | scientific role |
|---|---|---|---|
| C0 | exp042/043 unit plane wave | exp043 finite 96 um support, constant-zero shifted `B-1` | hash-locked baseline bridge |
| C1 | 与 C0 相同 | 128 um finite aperiodic master；每帧抽取 96 um physical patch | large-canvas single factor |
| C2 | sample-A entrance Gaussian | 与 C0 相同 | localized-illumination single factor |
| C3 | 与 C2 相同 | 与 C1 相同 | combined primary |

C1/C3 master 的中心 `48x48` 个 2 um cells逐元素继承 exp043 seed `20260840` realization；外围由独立 seed `20260944` 生成。这样 zero-position central patch可与 legacy B exact bridge，而非把 large canvas 混成新的中心编码器。C0使用 exp043 authoritative blind artifact，不重新以本实验预算重写 baseline；C1/C2/C3的新 reconstruction 才在 exp044执行。

### 2.2 四类横向区域

1. incident/support：A入口 Gaussian 在 `256x256 @ 0.5 um` open multislice grid上定义；primary `1/e2` intensity diameter `48 um`，另有 `320x320` support control。
2. probe--B active interaction：B面 `192x192=96 um` physical patch；TGV-induced unknown probe residual登记在居中 `96x96=48 um` native window；两者不能混称同一 support。
3. master B 与 scan union：C1/C3 master `256x256=128 um`，25个物理位置 span `18x18 um`，patch extraction coverage union、multiplicity、observable mask和never-visited region分别保存。active reconstruction mask固定为 `coverage>0`，不以 truth error缩小。
4. detector/open padding：q4 node open grid primary `256x256=128 um`，control `320x320=160 um`；native detector ROI固定 `32x32` pixels，即 `64x64 um`。open padding只是数值区域，不是 sample-B truth extent。

### 2.3 Localized beam 与功率口径

selected field 在 sample-A entrance 定义：

$$A(r)=A_0\exp(-r^2/w^2),\quad 2w=48\,\mu\mathrm m.$$

waist plane 就是 A入口；无 curvature、decenter、tilt、astigmatism 或 aberration。primary normalization 为 analytic fixed total Gaussian power，`P0=(48 um)^2`，即单位平面波在 registered native A aperture 上的功率。plane wave没有有限全平面总功率，故跨 plane/Gaussian 同时报告 registered-aperture power、Gaussian analytic/captured power、detector total intensity和energy-normalized metrics，不把浓缩照度造成的 signal 变化解释为几何增益。

Gaussian 必须先经过 q8 scalar multislice 与 A→B；禁止在现成 `P_B_true` 上后乘 B-plane Gaussian。C2/C3 的 homogeneous reference 是同一 Gaussian 在 homogeneous glass 中传播得到的 field；TGV residual单独 open传播，保存 incident、A-exit、B-reference、B-residual和B-total。

## 3. 初始方法、truth boundary、预算与 artifact contract

### 3.1 Forward/operator controls

公共实现复用 `multislice_propagate_streamed_A`、q8 air-fraction slice、angular-spectrum transfer、`ScanWindowPlan`、physical sample-B cells、q4 positive quadrature与 exp042/043 exact JVP/VJP/GN-CG family。新增适配层只承担 arbitrary A-entry field、master-window extraction/scatter和large-master B parameterization。

至少检查：homogeneous/zero-contrast Gaussian identity；plane/large-spot bridge；beam power与captured fraction；TGV residual edge；`256->320` support/padding convergence；A→B reference-plus-residual identity；master central-core identity；positive margin/zero wrap；extract/scatter、propagation、readout和combined intensity derivative；determinism、finite/nonnegative、runtime与memory estimate。

### 3.2 Reconstruction 与 truth boundary

known-B使用measurement-only mean half squared pixel-intensity loss、registered homogeneous-reference initialization和spectrally damped GN-CG。B逐元素固定，truth不进入初始化、stopping、checkpoint或branch selection；raw field冻结后才做 simulation-only gauge-aware probe error。

blind首先使用 exp043 alternating block GN-CG，P block为complex field，B block为active-mask上real phase；large-master B adjoint通过同一 `ScanWindowPlan` scatter-add。C3执行homogeneous/seeded两个truth-free initialization；C1/C2在自身forward与known-B measurement gate通过时执行一个homogeneous branch。formal representative固定为homogeneous final checkpoint，不以 truth 选择。

### 3.3 Development 与停止规则

先做tiny preflight；development只运行C0/C3并使用配置中的缩减预算。只有证据显示原block preconditioner因coverage结构失效，才允许在EOF先追加一个定向Change后尝试coverage-aware scaling或joint/LM之一；不并行撒网算法。formal前必须冻结source/config hashes、矩阵、beam、master/scan、operator、budgets、gates、mask和artifacts。

初始 gate 候选已进入 YAML，formal freeze前可依据 measurement-only development evidence一次性修订；修订必须写入EOF且不得依据blind truth挑阈值。component评价同时报告full-master、observed-union、coverage-weighted、coverage-stratified、low-coverage edge、per-scan/weighted exit与exp043 relative change；full-master error不作为primary gate。

### 3.4 HDF5/JSON/figures

HDF5顶层保持 `/entry/config_yaml,data,instrument,sample,truth,reconstruction,metadata,metrics`；不伪造calibration/preprocessing，不改变项目级schema。experiment-specific trees保存source provenance、design matrix、beam/power、master/patch/coverage/masks、forward controls、known-B、blind、gauge/checkpoints、conditioning和exp043 comparison。raw、measurement-only canonical和`simulation_evaluation_only`严格分离；所有cases均保存，不只保存最佳case。

计划六图：physical regions；forward controls；known-B recovery；blind components；detector prediction/residual；exp043 comparison。PNG只供人工审阅，全部定量值进入 HDF5 与 metrics JSON。

### 快速恢复上下文

```text
Current authoritative section: Section 3
Latest valid/development/preflight/formal run: none / none / none / none
Source exp031: formal 20260830_211803; root config SHA256 710577...86E64
Source exp040: R8 formal 20260814_152034; root config current-worktree SHA256 9EBF71...B96BEB
Source exp042: formal 20260831_183139; HDF5 C48588...A37D; raw P 194C7B...EB07
Source exp043: formal 20260903_153944; HDF5 BBDF25...2B09; Failed component-identifiability baseline
Frozen config SHA256: not yet frozen
Matrix: C0 authoritative bridge; C1 large-only; C2 Gaussian-only; C3 combined primary
Beam: A-entrance Gaussian, 48 um 1/e2 intensity diameter, fixed analytic total P0=(48 um)^2
Master/active/scan: 128 um master, 96 um extracted interaction patch, active mask=25-scan coverage>0, span=18 um
Methods/budget: exp042-family known-B GN-CG; exp043-family blind block GN-CG; formal budget not yet frozen
Forward/known-B/blind: not yet executed / not yet executed / not yet executed
Current single contradiction: whether geometry changes improve B/exit components rather than only detector fit
Minimum next read: Section 0 and Sections 2--3, YAML, forward/recon/runner/tests once created
Next command: conda run --no-capture-output -n tgv_ptycho_sim python -m pytest -q tests/test_exp044_TGV_3d_large_canvas_B_localized_illumination.py
```

### 后续建议

- 继续 exp044：完成实现、preflight、development、formal freeze、authoritative formal与artifact audit。
- 另开实验：blind reconstructed probe 的 waist fitting；sample-B family/scan系统优化；beam calibration/decenter/tilt/aberration；noise/detector calibration/真实数据；exp040 physical/full-wave validation。不得返回改写 exp031/040/042/043。

## 4. Implementation 与 operator correction（2026-09-03）

实现新增 `src/tgv_ptycho/forward/exp044.py`、`src/tgv_ptycho/recon/exp044.py`、`scripts/run_exp044_large_canvas_B_localized_illumination.py` 和 scoped tests。forward 适配层直接组合 exp040 streamed scalar multislice、exp031 `ScanWindowPlan`/extract/scatter、physical phase cells、exp042 q4 readout与 exp043 JVP/VJP；没有复制公共传播或 GN-CG 基础设施。

第一次真实 preflight `runs/exp044_TGV_3d_large_canvas_B_localized_illumination_preflight_20260903_205833/` 在所有 forward 数值已生成后因“known-B 未执行却创建 0-row subplot”失败。该 run 保留且 `run_state.json` 为 failed；修复只涉及 preflight figure分支。

初始 C0=exp043 artifact、C1=source plane+large、C2=open Gaussian+legacy、C3=open Gaussian+large 的安排随后被否决。有效 preflight显示，`256^2` open-A exact plane probe 与 exp043 `96^2` native-A source probe相差 `0.248087109`；因此原 C2并非严格的illumination单因素。formal矩阵改为：

| case | A-forward organization | incident | B design |
|---|---|---|---|
| E43 | exp043/042 native-A authoritative artifact | unit plane | legacy finite B |
| C0 | exp044 open scalar multislice | exact unit plane | legacy finite B |
| C1 | 与 C0 相同 | exact unit plane | large finite master B |
| C2 | 与 C0 相同 | selected A-entry Gaussian | legacy finite B |
| C3 | 与 C0 相同 | selected A-entry Gaussian | large finite master B |

E43只作 hash-locked跨实验 bridge，不混入四例因果矩阵。现在 C0→C1只改 B geometry，C0→C2只改 incident illumination，C2→C3只改 B geometry；E43→C0的差异明确登记为 A-domain/operator organization bridge。`1.6 mm` large-spot在128 µm open FOV内的incident flatness为 `0.00840174`，其归一化probe与exact open-plane probe差异 `0.00357302`，因此 exact plane branch 是可信的limit control；它不要求与E43 native-A source相同。

## 5. Preflight 与 bounded development evidence（2026-09-03）

有效 corrected preflight为 `runs/exp044_TGV_3d_large_canvas_B_localized_illumination_preflight_20260903_210607/`，artifact audit complete。较早 `..._210040/` 同样complete，但属于矩阵修正前的 superseded development evidence。

关键 forward 指标：scan extract/scatter adjoint `3.16849e-16`；minimum master margin `14 px = 7 µm`；Gaussian captured fraction `0.999999808`；A-residual edge fraction `3.45483e-6`；reference+residual identity `3.74454e-16`；localized probe `256→320` support difference `0.0111730`；C3 detector `256→320` padding difference `0.0342515`；四例 replay均为0，linear adjoint不高于 `1.9985e-15`，combined intensity adjoint不高于 `1.5486e-15`，directional derivative不高于 `8.31e-8`。四例均finite/nonnegative/repeat deterministic。

Gaussian固定analytic total power `2.304e-9 m^2`，captured `2.30399956e-9 m^2`。detector total intensity C0/C1/C2/C3分别约 `15597.2 / 15567.5 / 5712.58 / 5713.31`；因此localized case没有靠增加总detector energy获得优势。B-phase random-direction median gain从 C0/C1 的 `4.974e-4 / 4.083e-4`降至 C2/C3 的 `1.175e-4 / 9.490e-5`，是formal前已登记的measurement-only不利信号，不据此取消 C3。

development runs：

- `..._development_20260903_210736/`：缩减3-outer known-B，C0/C3 residual `0.01194 / 0.04104`，未进入blind。
- `..._development_20260903_210952/`：exp042-sized 6-outer known-B，C0闭合 `0.003375 / 0.09550`（residual/probe）；C3为 `0.011822 / 0.214859`。只有C0进入2-sweep blind，residual `0.04727`。
- `..._development_20260903_211230/` 与 `..._211446/`：把 known-B outer增到10后，后期 damped GN-CG 在已很小gradient处无法生成finite direction；两次failed run均保留。该路线被否决，避免把measurement-design实验变成solver尾部调参。
- `..._development_20260903_211649/`：最终有效development。known-B恢复6-outer；预注册新-design gate改为residual `<=0.015`、post-freeze probe `<=0.23`，C0和C3均闭合。C0/C3各执行2-sweep homogeneous blind，loss均单调且finite；residual `0.04727 / 0.09515`。C3早期simulation-only probe/B-weighted/exit为 `0.38899 / 0.43945 / 0.54639`，不用于修改formal设置。

## 6. Formal freeze（2026-09-03）

formal冻结为四例 C0/C1/C2/C3 全部forward与known-B；四例在各自forward和known-B measurement gate闭合后各运行homogeneous blind，C3额外运行预注册seeded-phase branch。representative固定为 homogeneous B 的 final sweep；不以truth选择case、branch、checkpoint或method。

selected Gaussian仍为sample-A entrance `A0 exp(-r^2/w^2)`，`2w=48 µm`，fixed analytic total power `2.304e-9 m^2`。master为 `256^2 @ 0.5 µm=128 µm`，interaction patch `192^2=96 µm`，scan 25 positions/span 18 µm，active mask固定 `coverage>0`；never-observed pixels不优化且不进入primary B gate。detector为256² q4 nodes、32² pixel ROI，320²只作support/padding control。

known-B冻结为6 outer、6 power、8 CG、spectral damping `1e-4`；blind冻结为6 alternating sweeps，每block 1 outer、3 power、4 CG，phase-only unit-modulus B，fixed final。没有授权替代optimizer，因为development未出现block oscillation或operator/preconditioner失效证据。

formal gates冻结为：replay `1e-13`；scan/propagation/intensity adjoint `1e-11/1e-11/1e-10`；directional `2e-5`；A residual edge `5e-3`；localized support `0.02`；detector padding `0.04`；margin `1 µm`；captured fraction `0.999`；known-B detector/probe `0.015/0.23`；blind detector `0.04`、C3 branch prediction repeat `0.01`；probe/B coverage-weighted/exit `0.25/0.35/0.25`；相对E43 B和exit各至少改善10%。这些门槛在formal blind truth未知时冻结。

root config冻结：`10456 bytes`，SHA256 `F22E15CB90F0D12E079AF1EAB0FE96B3182AC8233C379755AEC8BE56F2B9E799`。source implementation hashes：forward `F1A1CF77543A20CC73D3FD5AD3D3B860F880A0F914879F61DBE36FFFC979AB75`；recon `0FB89D624E7D1DCB19F51EA2A62D9988D61EBFE2F6EB70E191E03B897E73F3FD`；runner `F4B53135A694062DD15D881DED133B3437FBBF2DE26E2F9C911C25AE793799F3`；test `EE31B8BA32C53FE05E0EDD4AD88437D57BCF9D12666BB42BB8D89A5BE69C1E0F`。初始文档prefix再次验证为 `13155 bytes`、SHA256 `28B1C57C2C9B362BF7347D9FC399B464560A0AA2B9E83C95738648B785AF2BC5`，逐byte未变。

### 快速恢复上下文

```text
Current authoritative section: Section 6 formal freeze
Latest valid/development/preflight/formal: development 20260903_211649 / same / preflight 20260903_210607 / none
Sources: exp031 config 710577...86E64; exp040 R8 current 9EBF71...B96BEB; exp042 H5 C48588...A37D/raw P 194C7B...EB07; exp043 H5 BBDF25...2B09
Frozen config: F22E15...E799
Matrix: E43 hash bridge; C0 open-plane+legacy; C1 open-plane+large; C2 A-Gaussian+legacy; C3 A-Gaussian+large
Forward/known-B/blind: forward closed / C0+C3 development closed / 2-sweep cost and monotonicity closed, formal not run
Beam: A-entry Gaussian, 48 um 1/e2 intensity diameter, fixed analytic total power
Master/active/scan: 128 um master, 96 um patch, coverage>0 active union, 25 scans, 18 um span, 7 um min margin
Methods: known-B GN-CG 6x(power6,CG8); blind alternating GN-CG 6 sweeps, per-block power3/CG4
Current single contradiction: whether formal C3 closes measurement and improves registered B/exit components versus E43
Minimum next read: Section 0, Sections 4--6, frozen YAML, runner
Next command: conda run --no-capture-output -n tgv_ptycho_sim python scripts/run_exp044_large_canvas_B_localized_illumination.py --config configs/experiments/exp044_TGV_3d_large_canvas_B_localized_illumination.yaml --stage formal
```

### 后续建议

- 继续 exp044：执行唯一authoritative formal、完整artifact/figure audit、tests/lint，并append-only写回结果。
- 另开实验：blind reconstructed-probe waist fitting；sample-B/scan优化；beam calibration；noise/calibration/真实数据；exp040 full-wave physical validation。

## 7. Freeze seal 与 formal authorization（2026-09-03）

Section 0--6组成的 frozen document prefix为 `21215 bytes`，SHA256 `A3FBC7039D2992DD78B310A8B8D788049A4533FAD52EF69B738CE8443087FBF8`。后续只允许在此EOF之后追加；formal前再次确认 frozen root config为 `F22E15CB90F0D12E079AF1EAB0FE96B3182AC8233C379755AEC8BE56F2B9E799`。授权执行一个且仅一个该配置的authoritative formal矩阵；任何状态均保存。若暴露实现错误，保留run并另行append correction，不静默覆盖。

### 快速恢复上下文

```text
Current authoritative section: Section 7 freeze seal
Latest valid/development/preflight/formal: development 20260903_211649 / same / preflight 20260903_210607 / none
Source identities: exp031 710577...86E64; exp040 R8 9EBF71...B96BEB; exp042 H5 C48588...A37D; exp043 H5 BBDF25...2B09
Frozen config SHA256: F22E15...E799
Frozen document prefix: 21215 bytes, A3FBC7...7FBF8
Matrix: E43 external bridge plus causal C0/C1/C2/C3
Forward/known-B/blind: forward closed / selected controls closed / formal pending
Beam/power: A-entry 48 um Gaussian, fixed analytic total power 2.304e-9 m2
Master/active/scan: 128 um master, 96 um patch, active=coverage>0, 25 scans, 7 um margin
Methods/budgets: known-B 6 outer(power6/CG8); blind 6 sweeps(block power3/CG4)
Current single contradiction: formal component identifiability versus E43
Minimum next read: Sections 6--7, frozen YAML, runner
Next command: conda run --no-capture-output -n tgv_ptycho_sim python scripts/run_exp044_large_canvas_B_localized_illumination.py --config configs/experiments/exp044_TGV_3d_large_canvas_B_localized_illumination.yaml --stage formal
```

### 后续建议

- 继续 exp044：执行formal、artifact审计、tests/lint与结果append。
- 另开实验：waist fitting、sample-B/scan优化、beam calibration、noise/experimental pipeline或exp040 physical validation。

## 8. Authoritative formal result 与 artifact audit（2026-09-03）

authoritative formal为 `runs/exp044_TGV_3d_large_canvas_B_localized_illumination_formal_20260903_212213/`。run complete、artifacts validated；HDF5 SHA256 `EA2E615D0260183A623B0656D60D6B144B327740BABDAE559DF92B48A779688A`，metrics SHA256 `84ACDB9153512151E0F4420098C7B30EE3377BC246783F7B5B07FC5EC2501A27`。运行时间 `459.839 s`，冻结root config SHA256精确为 `F22E15CB90F0D12E079AF1EAB0FE96B3182AC8233C379755AEC8BE56F2B9E799`。formal前的Section 0--7 prefix `23123 bytes / 67AB67EDF617289F9F083F80C73B5292E662090264BF29358EDD416A136FB94A` 已在追加本节前验证；允许更新的Section 0不纳入后续append-only锁，Section 1--7 body为 `19784 bytes / 673D905D59ADA1D703F2FA9C34BB20CBF7203B512DED3BD0167290A0C94CA97D`。

### 8.1 Scientific status 与层级原因

最终状态：`Failed / blind_measurement_reconstruction_not_closed`。

Part A 四个case全部通过forward gates；Part B E43 bridge和C0--C3全部通过known-B gates。Part C C3两条固定final branches均finite、loss nonincreasing且完成6 sweeps，但最大detector residual `0.0476316 > 0.04`，pairwise prediction difference `0.0192078 > 0.01`。因此按冻结层级，状态在blind measurement/repeatability处失败，不能进入“measurement consistent but components inaccurate”的状态分支。不得把下述post-freeze component errors单独解释成已闭合的component-identifiability结论。

### 8.2 Part A：forward/operator与measurement design

- E43 authoritative replay为0；open-A C0与E43 native-A source probe差异 `0.248087`，作为明确operator-organization bridge保存。
- C0--C3 replay均0；linear propagation adjoint最大约 `2.00e-15`，combined intensity real-adjoint最大约 `1.55e-15`，directional derivative最大约 `8.31e-8`；均finite/nonnegative/deterministic。
- physical scan-window adjoint `3.16849e-16`；所有patch完全在master内；minimum margin `14 px = 7 µm`；master coverage为1--25，C3 observable `51776` pixels、never observed `13760` pixels。
- Gaussian analytic/captured power `2.304e-9 / 2.30399956e-9 m^2`，captured fraction `0.999999808`；A residual edge `3.45483e-6`；reference+residual identity `3.74454e-16`；localized probe support difference `0.0111730`；detector padding difference `0.0342515`。
- detector total C0/C1/C2/C3为 `15597.2 / 15567.5 / 5712.58 / 5713.31`。localized pair信号更低而非靠额外光子；B-phase random-direction median gain C0/C1/C2/C3为约 `4.974e-4 / 4.083e-4 / 1.175e-4 / 9.490e-5`，selected Gaussian在当前fixed-total口径下恶化该局部conditioning proxy。

### 8.3 Part B：known-B control

E43 authoritative raw bridge residual/probe error为 `0.00339877 / 0.101074`。exp044 C0/C1/C2/C3 的 detector residual分别为 `0.00337533 / 0.00387792 / 0.0118251 / 0.0118220`，post-freeze aligned probe error分别为 `0.0954973 / 0.0986447 / 0.214916 / 0.214859`；均低于冻结 `0.015 / 0.23` gates，loss单调。B逐元素固定，初始化为各case registered homogeneous reference，truth未进入optimizer、spectral damping、stopping或branch selection。

### 8.4 Part C：blind measurement与component diagnostics

fixed-final homogeneous branch结果：

| case | detector residual | probe error | B observed/coverage-weighted error | exit-product error |
|---|---:|---:|---:|---:|
| E43 authoritative | 0.0273523 | 0.206064 | 0.401165 | 0.305121 |
| C0 open-plane + legacy B | 0.0259257 | 0.209220 | 0.408689 | 0.303779 |
| C1 open-plane + large B | 0.0315427 | 0.218637 | 0.395151 / 0.400381 | 0.297771 |
| C2 Gaussian + legacy B | 0.0477987 | 0.294709 | 0.426917 | 0.486713 |
| C3 Gaussian + large B | 0.0476316 | 0.294574 | 0.434773 / 0.429837 | 0.486661 |

C3 illumination-weighted B error `0.426272`、low-coverage error `0.451595`、coverage-weighted exit error `0.486661`。相对E43，C3 coverage-weighted B和exit分别变化为 `-7.15% / -59.50%` improvement（负值表示恶化），probe也恶化。C3 seeded branch residual `0.0414546`，优于固定representative的 `0.0476316`，但二者prediction difference `0.0192078`，且不得以较好branch替换预注册representative。C0/C1的large-B单因素只带来约0.2% B与2.4% exit改善，未达到10%注册门槛；C2/C3的large-B单因素几乎不改变exit，而B略差。现有证据不支持“large canvas + selected localized beam改善blind recovery”；更严格地说，primary在measurement/repeatability先失败。

没有尝试替代optimizer。forward/known-B闭合，但development/formal显示的主要问题与 fixed-total localized signal、B-direction gain下降及两初始化prediction不一致相关；没有证据证明仅是large-master preconditioner失效，也没有授权用joint/LM/variable projection事后救援。

### 8.5 HDF5/JSON/figure审计

HDF5大小 `190,997,720 bytes`，`/entry`精确包含 `config_yaml,data,instrument,sample,truth,reconstruction,metadata,metrics`，无伪造calibration/preprocessing；全部数值dataset有限。C0 source bridge bytes、config/source hashes、JSON/HDF5 status一致。自然新增的experiment-specific树包括：E43/C0--C3 data；source provenance/design matrix；beam/power；master cells、25个实际patch、starts/stops/margins；coverage/illumination coverage/observable与never-observed masks；localized incident/A-exit/reference/residual/B-plane fields；forward/operator/conditioning controls；known-B与blind raw/canonical/simulation-only outputs；loss/gradient/CG/line-search/checkpoint/action/runtime；exp043 comparison。无项目级HDF5 schema变化。

六张PNG均经PIL decode和人工审阅：physical regions/coverage边界清楚；Gaussian power与detector totals可读；known-B四例显示localized error较大但仍闭合；blind C3的B恢复呈illumination-shaped平滑结构而非physical cell pattern；detector residual有明确空间结构；E43/C3 bar chart与定量恶化一致。PNG只作人工检查，定量值均在JSON/HDF5。

### 8.6 Tests、lint与Git

- exp044 scoped：`15 passed`。
- exp031 + scan-window + physical sample-B：`18 passed`。
- exp040 R8 + multislice：`32 passed`。
- exp042 + exp043 + exp044 reconstruction regression：`40 passed`。
- exp044修改范围Ruff：通过。
- 全量pytest：`397 passed, 12 failed in 268.41 s`；12项全部为任务开始前已登记的exp040 R10--R14B frozen-config SHA256 lock mismatches，无exp044新增失败。
- project-wide Ruff诊断：11个既有问题（exp001八个E402，calibration/stage、recon/losses、recon/rpie各一个E501）；不在本实验顺手修改。

Git保持本地unstaged；未执行 `git add`、commit、push、PR或merge。未覆盖、回退或删除用户已有修改；runs/HDF5/PNG未暂存。

### 快速恢复上下文

```text
Current authoritative section: Section 8 formal result
Latest valid/formal: runs/exp044_TGV_3d_large_canvas_B_localized_illumination_formal_20260903_212213
Latest development/preflight: 20260903_211649 / 20260903_210607
Sources: exp031 710577...86E64; exp040 R8 9EBF71...B96BEB; exp042 H5 C48588...A37D/raw P 194C7B...EB07; exp043 H5 BBDF25...2B09
Frozen config: F22E15...E799
Matrix: E43 bridge; C0 open-plane legacy; C1 open-plane large; C2 A-Gaussian legacy; C3 A-Gaussian large
Forward/known-B/blind: Passed / Passed / Failed measurement+repeatability
Beam: A-entry 48 um Gaussian, fixed analytic total power 2.304e-9 m2
Master/active/scan: 128 um master, 96 um patch, coverage>0 mask, 51776 observed/13760 never-observed, 25 scans
Methods/budgets: known-B GN-CG 6 outer; blind alternating block GN-CG 6 sweeps; no alternative solver
Status: Failed / blind_measurement_reconstruction_not_closed
Single contradiction resolved: selected localized design did not improve formal blind recovery; measurement/repeatability failed first
Minimum next read: Section 0 and Section 8; formal metrics.json/run_state.json
Next command: no further exp044 formal command authorized; inspect the authoritative run only
```

### 后续建议

- 继续 exp044：不再调参或重跑；只允许勘误、artifact读取或对本authoritative negative result的解释。
- 新的blind-probe waist-fitting实验：当前C3 raw probe不得直接进入；若以后有通过registered blind measurement/component contract的新raw probe，再以新exp05x定义raw/canonical/gauge下游合同。
- measurement-design后续：另开实验研究beam diameter/power-normalized conditioning、scan/B coding或更合适的localized illumination；不得用本次truth error事后选spot。optimizer-only问题也应独立编号并固定本次data/operator。
- 其他独立实验：beam decenter/tilt/aberration/unknown waist plane；noise/detector calibration/真实数据；exp040 full-wave physical validation。

## 9. Post-formal QA seal（2026-09-03）

formal后只补充了不改变operator、config、run或scientific status的测试：exact open-plane repeat、causal matrix/truth boundary、append-only body hash和invalid incident type。因而Section 6记录的pre-QA test-file hash由最终 `0AFA3D2EB4176098AF330A00BD6485B9E68063979B3663C39EAB03BDD37D7EDA` 取代；forward/recon/runner与root config hashes不变，authoritative HDF5/metrics hashes不变。Section 1--8 append-only body在追加本节前为 `28614 bytes / 7F82672A1269D046EFA19A4776EC3742C3485B9825A89BA2F01879FCE8A5BF97`；Section 1--7已再次逐byte验证为 `19784 bytes / 673D905D59ADA1D703F2FA9C34BB20CBF7203B512DED3BD0167290A0C94CA97D`。

### 快速恢复上下文

```text
Current authoritative section: Section 9 post-formal QA seal; scientific result remains Section 8
Latest valid/formal: exp044..._formal_20260903_212213
Latest development/preflight: 20260903_211649 / 20260903_210607
Frozen config: F22E15...E799; formal H5 EA2E61...688A; metrics 84ACDB...1A27
Matrix: E43 bridge plus C0/C1/C2/C3 causal open-A matrix
Forward/known-B/blind: Passed / Passed / Failed measurement+repeatability
Beam/master/scan: A-entry 48 um fixed-total Gaussian; 128 um master; 96 um patch; coverage>0; 25 scans
Methods: exp042-family known-B GN-CG; exp043-family alternating blind block GN-CG; no alternative optimizer
Current contradiction: none inside exp044; negative result is frozen
Minimum next read: Section 0, Section 8, then this seal
Next command: no new exp044 formal; read authoritative artifacts
```

### 后续建议

- 继续 exp044：只做勘误和artifact解释，不调参、不重跑。
- 新实验：blind-probe waist fitting须等待通过新合同的raw probe；measurement/beam/scan/B和optimizer问题分别预注册；noise、real-data和physical validation保持独立。
