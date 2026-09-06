# exp043：exp042 框架下的 blind probe/B joint reconstruction

## 0. 实时状态与阅读顺序

```text
Scientific status: Failed / measurement_consistent_but_component_recovery_non_identifiable
Work status: Frozen / Closed; formal executed once, artifacts/figures/tests audited
Results available: development/preflight and authoritative formal evidence
Latest valid run: runs/exp043_TGV_3d_multislice_blind_probe_B_reconstruction_20260903_153944
Latest development/preflight run: runs/exp043_TGV_3d_multislice_blind_probe_B_reconstruction_dev_20260903_152530
Latest formal run: runs/exp043_TGV_3d_multislice_blind_probe_B_reconstruction_20260903_153944
Authoritative appended section: Section 6
Primary current question: 已闭合；measurement fit/repeatability 通过，但 frozen B/exit component recovery gates 失败
exp042 source run: runs/exp042_TGV_3d_multislice_probe_reconstruction_dev_20260831_183139
exp042 source HDF5: outputs/exp042_probe_reconstruction.h5
Selected primary method: frozen alternating block spectrally damped GN-CG in exact real B-phase coordinates
Current single main contradiction: selected B/scan/localized-illumination design admits measurement-consistent but component-inaccurate blind factorizations
Next action: no further exp043 computation; take measurement-design/non-identifiability work to exp044, and waist fitting to a new exp05x only after a downstream probe contract is defined
Reading order: Section 0 -> latest authoritative appended section -> earlier append-only sections only as needed
```

本节是全文唯一允许持续原位更新的实时状态入口。第 1 节以后全部 append-only；不得回改、删除、移动、
重排或润色旧记录。每次 implementation、correction、freeze 和 formal result 都只在真实 EOF 追加新编号章节。

## 1. 2026-09-03：初始设计、source audit 与 development preregistration

### 1.1 研究问题与最小变化

exp043 回答：在保持 exp042 authoritative matched-q4 data、measurement operator、mean half squared
pixel-intensity loss 和 spectrally damped GN-CG 框架不变的条件下，只把 sample B 从固定已知量改为未知量，
能否稳定联合恢复 raw B-plane probe `P_B` 与 finite sample B？

相对 exp042 的最小算法差异固定为：

```text
exp042: unknown P_B; B_true fixed; one probe GN-CG block
exp043: unknown P_B and B; alternating probe GN-CG and B GN-CG blocks
```

本轮不把 primary optimizer 改为 ePIE/rPIE，不改变 source data、scan、q4 detector、open propagation、loss、
B support、transparent exterior、FFT/adjoint convention 或 exp040 status。只有 controls 通过但 blind alternating
出现可重复、明确的结构性阻力后，才允许在后续 EOF Change 中预注册至多一到两个定向替代方案。

### 1.2 开始前读取与 Git 保护

开始前已读取：

- `AGENTS.md` 全文；
- `docs/templates/experiment_task_prompt.md` 全文；
- `docs/theory_notes/roadmap.md` 全文；
- exp040 第 0 节、selected R8 working-model 第 16 节和第 19 节冻结结论；
- exp042 第 0、29、30 节，以及 source/operator、loss、truth boundary、gauge、adjoint、初始化、spectral
  damping 和 artifact contract 所需内容；
- exp052 的 known-B/blind 双通道、truth-use、gauge、checkpoint 和 artifact 经验边界；
- exp042 root YAML、公共模块、runner、tests；
- authoritative run 的 `config.yaml`、`metadata.json`、`metrics.json`、`run_state.json` 和 HDF5 实际树。

初始 `git status -sb` 显示当前分支已有用户的 staged 之外的多项 tracked modifications、untracked files，且
`notebooks/00_check_propagation.ipynb` 为 deleted。exp043 不覆盖、回退、删除或暂存任何这些内容；本任务只新增
exp043 文件并保持 unstaged，不执行 add/commit/push/PR/merge。由于 sandbox 用户与仓库 owner 不同，所有只读
Git 命令使用一次性 `git -c safe.directory=E:/tgv_ptycho_sim ...`，未修改全局 Git 配置。

### 1.3 authoritative source 的实际 artifact 审计

source 身份不是从当前 root YAML 推断，而是直接由 run artifact 核对：

```text
source run:
runs/exp042_TGV_3d_multislice_probe_reconstruction_dev_20260831_183139

source HDF5:
outputs/exp042_probe_reconstruction.h5

HDF5 SHA256:
C48588FC47EED474CF047219A1DB26BCC9242961A58176CA9CCB879495FFA37D

run config SHA256:
59860425AFAB3D07D6DA5D44A41085B3FA24E9EAFF6EE317ABE06B6BF0BA7276

metadata SHA256:
5B056A65302924A5C03E8EC5AAEB06B7C388F8888ED2BCCF86D24659C9CE7B2B

metrics SHA256:
195B001CC4EB90EE9B9B37ACACDEB599EF23D6379EEC0CA10025CB211D70615A

run_state SHA256:
75E70687CEA890E8CFFB9473F54080583685942CF45B70D5F89678621031905C

source Git commit:
ac5c20843ff910c1d1619fe590b4b5e18bc38912
```

关键 datasets 已逐一路径、shape、dtype 和 C-order dataset bytes 核对：

| role | HDF5 path | shape / dtype | dataset-byte SHA256 |
|---|---|---|---|
| measured intensity | `/entry/data/I_stack` | `(25,32,32) float64` | `BCE7A54EFBA46F6B856C592364823454864F528AC02A44C1C6BA67E46B105948` |
| scan positions | `/entry/data/scan_positions` | `(25,2) float64` | `63961ADAEC739EFCEABD8F7110A391F094851BFD02475AC23BA1AECEDD8ABCC5` |
| simulation probe truth | `/entry/truth/P_B_true` | `(96,96) complex128` | `FA61264AF0D96BF3393EC461E2147992FDE6926D0132B2346090790E40E8EBFD` |
| simulation B truth | `/entry/truth/B_true` | `(256,256) complex128` | `740EBB9B35D6132744ED56785BEA9D4034774CD1D1819D3B040D2C83D0653537` |
| exp042 authoritative raw probe | `/entry/reconstruction/exp053_feedback_control/branches/spectrally_damped_gn_cg/P_B_rec` | `(96,96) complex128` | `194C7B950F8DCF2BF94212A6F63270296BC9EC79AE0C04ED557C864F5FE8EB07` |

sampling/axis contract 为 field `(ny,nx)`、scan columns `(x,y)` in m、node `dx=0.5 um`、native probe
FOV `48 um × 48 um`、open FOV `128 um × 128 um`、q4 detector pixels `2 um`、native detector ROI
`32×32`。operator branch 是 `R8 unified q8 finite-B open q4 scalar working model`，B support `192×192`
on `256×256` open grid，shift 为 constant-zero shift of `B-1`，exterior transmission 是 `1+0j`。

run state 为 `complete` 且 `artifacts_validated=true`。始终继承：

```text
reference_validated=false
full_tgv_reference_authorized=false
single noiseless matched scalar simulation case
```

### 1.4 source/operator replay contract

exp043 primary 直接从上述 HDF5 读取并使用持久化 `I_stack` 与 `scan_positions`。为了取得 exp042 未持久化的
transfer/reference arrays，loader 从该 run 自己的 `config.yaml` 确定性重建 operator；放行条件是 regenerated
`P_B_true`、`B_true`、scan 和 `I_stack` 全部逐元素 exact，且用持久化 truth 重新 forward 的 prediction 与
持久化 `I_stack` exact。任何 hash、shape、dtype、state、reference flag、support/exterior 或 replay 不闭合都
立即停止算法开发，状态为 `Inconclusive`。

### 1.5 变量、loss 和 block derivatives

probe 参数仍是 exp042 native `96×96 complex128 P_B`。B 参数是 open grid 上的 complex modulation
`M=B-1`；只有配置中已知的 centered `192×192` active support 参与更新，support 外每次 projection 都严格为
zero，即 physical B exterior 始终 `1+0j`。

对 scan `s`：

```text
P_open = P_homogeneous_open + embed(P_B - P_homogeneous_native)
B_s = 1 + S_s M
exit_s = P_open B_s
```

detector 与 exp042 完全相同：open reference-plus-residual ASM、matched q4 positive midpoint pixel average、
centered detector ROI。loss 仍为所有 detector pixels 上 `mean(0.5*(I_pred-I_data)^2)`。

新增 B JVP 为 `H[P_open S_s dM]`；VJP 为 detector/q4/ASM adjoint 后乘 `conj(P_open)`，再用 paired
constant-zero unshift/scatter 累加并投影 active support。P/B normal actions 各自是 mean-normalized
`J_block^T J_block`。P、B block 分别使用自己的 truth-free power-iteration spectral radius、relative damping、
real-inner-product CG 和 Armijo；这反映单位/曲率差异，不更换 optimizer family。

### 1.6 三层 control

1. Known-B / probe-only：直接调用 exp042 已验证的 spectrally damped GN-CG，初始化和全部 settings 从
   source run config 读取；要求 raw output 与 authoritative dataset byte exact，且 residual 重现。
2. Known-probe / B-only：`P_B_true` 只作为 `simulation_diagnostic_only` 的固定已知输入；从 homogeneous B
   初始化，只更新 active B。检查 B JVP/VJP、loss gradient、normal symmetry/PSD、support/exterior、
   deterministic loss/residual 下降。
3. Blind primary：P 使用 exp042 homogeneous-reference initialization；B 使用 YAML 固定的 truth-free
   homogeneous 或 seeded zero-mean weak-phase initialization；固定次序先 P block 后 B block，固定有限 outer
   sweeps，代表分支永远是 YAML 中第一支，checkpoint 永远是 final outer sweep。

truth 不进入 initialization、optimizer、damping、stopping、checkpoint、representative/seed/branch selection
或 method selection。simulation evaluation 只在每支 raw final 已存在后执行。

### 1.7 blind gauge 分析和 primary convention

一般 blind factorization 的代数对称为 `P→cP, B→B/c`；辅助函数必须验证完整 exit product 在该变换下达到
机器精度 invariance。exp042 的实际 parameterization 还固定了 native unknown window 外的 homogeneous
probe reference 和 active support 外的 transparent B exterior。因此 non-unit `c` 无法在允许的 parameter
space 中同时缩放这两个固定 reference；必须另存 `model_representable_reciprocal_scale_prediction_change` 证明
标准 scale gauge 被 boundary/reference pin，而不是未经检查地假设它存在。

primary 使用的 measurement-only canonical convention 因而是 identity：raw P/B 原样复制到 canonical tree，
`factor=1+0j`，不读取 truth，也不改变 exit wave 或 detector prediction。probe norm、active-B RMS amplitude、
active-B mean phase 随 outer sweep 保存，用于检查近零/弱辨识漂移。simulation-only tree 可以在 raw final 冻结后
由 truth 拟合一个 probe least-squares reciprocal gain，并把 inverse gain 补偿到 active B；该 copy 不返回
optimizer，不成为下游 raw-complex probe。

affine/raster ambiguity 不预先通过 truth 校正；由多初始化 pairwise prediction、probe distance、component
error 和 residual 共同诊断。若只能得到宽 equivalence class，即使 measurement residual 很低也不能 Passed。

### 1.8 development/preflight 预算与开放阈值

首次 root YAML 是 development config，不作 scientific pass/fail：

```text
known-B P control: exact source exp042 6 outer × up to 8 CG (unchanged)
known-probe B control: 4 outer, 4 power, up to 4 CG
blind: 2 alternating outer sweeps
each blind P/B block: 1 outer, 3 power, up to 4 CG
blind initializations: homogeneous B and fixed seeded 0.05-rad RMS phase B
checkpoint: fixed final outer sweep
representative: first registered homogeneous-B branch
```

本预算只用于 source/operator/block correctness、cost、gauge 和 alternating obstruction 判断。formal thresholds
在 development evidence 后另行 EOF 预注册；当前 YAML 中的数值仅是待评估候选，`scientific_gates_enabled=false`。
不得把 development status 提升为 scientific status。

### 1.9 formal gate 候选与状态逻辑

formal 前必须冻结 source hashes、method、初始化、budgets、checkpoint、thresholds、status matrix、HDF5/figures
contract 和 root config SHA256。状态顺序固定为：

1. source/hash/operator/JVP/VJP/gradient/support/artifact identity 未闭合：`Inconclusive`；
2. P-only 或 B-only control 未闭合：`Inconclusive`；
3. controls 有效，但 frozen blind measurement、repeatability、component、boundary 或 stability gate 失败：
   `Failed`；
4. 所有 frozen scientific/artifact gates 通过：`Passed`。

若 measurement gates 通过但 P/B component gates 失败，reason 必须是
`measurement_consistent_but_component_recovery_non_identifiable`，不得只凭 residual Passed。

### 1.10 HDF5、JSON、figure 和 artifact contract

每次 run 创建独立 timestamped directory，包含 `config.yaml`、`metadata.json`、`metrics.json`、
`run_state.json`、`outputs/exp043_blind_probe_B_reconstruction.h5` 和五张 PNG。HDF5 `/entry` 并列为：

```text
config_yaml / data / instrument / sample / truth / reconstruction / metadata / metrics
```

不伪造 calibration/preprocessing。`reconstruction` 至少含：

```text
source_provenance
design
operator_controls
known_b_probe_control
known_probe_B_control
blind_joint/branches
blind_joint/representative
gauge
checkpoints
```

raw、measurement-only canonical 和 simulation-evaluation-only aligned P/B 分树保存；所有 branch、block、
loss/residual/gradient/CG/line-search、actions、stopping、truth flags 和 checkpoint rule 均持久化。外部 JSON 与
HDF5 status 必须一致；source data bytes 必须 exact；所有 numeric datasets finite；五张图必须可由 PIL 解码。
项目级 HDF5 schema 不变。

五图固定检查 controls convergence、probe target/init/raw/error、B target/init/raw、detector prediction/residual，
以及 block update/gauge/multi-initialization stability。PNG 只供人工审计，所有数值进入 HDF5/JSON。

### 1.11 首轮实现和非科学验证

新增而未修改 exp042 的文件为：

```text
configs/experiments/exp043_TGV_3d_multislice_blind_probe_B_reconstruction.yaml
src/tgv_ptycho/recon/exp043.py
scripts/run_exp043_blind_probe_B_reconstruction.py
tests/test_exp043_blind_probe_B_reconstruction.py
docs/experiment_design/exp043_TGV_3d_multislice_blind_probe_B_reconstruction.md
```

初次 scoped Ruff 暴露纯 formatting/closure lint，已修正；没有改变科学方法。初次 pytest 的两个失败是
`operator_consistency_metrics` 调用签名错误，以及 exact fixed point 有 `3.90e-33` 浮点 loss 而测试错误要求
bitwise zero。修正后命令和结果：

```text
conda run --no-capture-output -n tgv_ptycho_sim python -m pytest -q tests/test_exp043_blind_probe_B_reconstruction.py
11 passed in 9.64s

conda run --no-capture-output -n tgv_ptycho_sim python -m ruff check src/tgv_ptycho/recon/exp043.py scripts/run_exp043_blind_probe_B_reconstruction.py tests/test_exp043_blind_probe_B_reconstruction.py
All checks passed
```

覆盖 source loader/hash/state/replay、B finite difference、B JVP/VJP adjoint、combined gradient、normal symmetry/
PSD、support/exterior、gauge invariance、canonical prediction invariance、truth-free deterministic repeat、invalid
input、状态矩阵和 HDF5/artifact round trip。尚未执行 authoritative source development/preflight。

### 快速恢复上下文

```text
Current authoritative section: Section 1
Latest valid/development/preflight/formal run: none / none / none / none
Source exp042 run: runs/exp042_TGV_3d_multislice_probe_reconstruction_dev_20260831_183139
Source HDF5: outputs/exp042_probe_reconstruction.h5
Source hashes: HDF5 C48588...A37D; I BCE7A5...5948; scan 63961A...CC5; P truth FA6126...EBFD; B truth 740EBB...3537; raw P rec 194C7B...B07
Frozen root config SHA256: not frozen; development config only
Selected method/budget: alternating block spectrally damped GN-CG; 2 dev sweeps, 1x(3 power + up to 4 CG) per block
P-only/B-only/blind: unit-test interfaces passed / unit-test interfaces passed / authoritative preflight pending
Gauge: identity primary canonical under fixed homogeneous reference + transparent exterior; truth reciprocal alignment postfreeze only
Current single contradiction: authoritative B-only and blind preflight not run
Minimum next read: Section 0, Section 1.3--1.9, root exp043 YAML
Next command: conda run --no-capture-output -n tgv_ptycho_sim python scripts/run_exp043_blind_probe_B_reconstruction.py --config configs/experiments/exp043_TGV_3d_multislice_blind_probe_B_reconstruction.yaml
```

### 后续建议

继续 exp043 才能完成：执行 authoritative development/preflight；核对 blocks、gauge、budget 和 artifacts；冻结并
执行唯一 formal；运行 scoped/combined/full tests 和 Ruff；追加 formal audit 与严格状态。

应新开实验：large-canvas finite nonperiodic B、localized illumination 或 scan/B design 进入 exp044；blind raw
probe 的 `D_waist` fitting 新开 exp05x；noise、calibration、真实数据和 physical model mismatch 分别另开实验。
不返回改写 exp040 或 exp042。

## 2. 2026-09-03：Implementation iteration 01 — 首次 authoritative preflight 与有界 phase-only correction

### 2.1 首次 development/preflight 结果

按第 1 节 development contract 实际执行：

```text
command:
conda run --no-capture-output -n tgv_ptycho_sim python scripts/run_exp043_blind_probe_B_reconstruction.py --config configs/experiments/exp043_TGV_3d_multislice_blind_probe_B_reconstruction.yaml

run:
runs/exp043_TGV_3d_multislice_blind_probe_B_reconstruction_dev_20260903_150810

run state: complete; artifacts_validated=true; scientific_status=Development
runtime: 108.8617213 s
run config SHA256: F036C3867930BBDDE8D7A3C2EACB5AD89CE4DFA454C86E436085D43E863ADAE5
metrics SHA256: DCA7E1067DA649AA719050BAE07413012495ACEF0AB09999DEF46A6767C9F9D9
HDF5 SHA256: F575ED4CE465E768053795BD5E7527360BD3FBFA8ACA0CFB248550177985F9D7
```

source arrays、regenerated operator 和 independent replay 全部 exact。P/B/combined controls 为：

```text
B intensity JVP finite-difference relative error: 1.3134392e-9
B intensity JVP/VJP real-adjoint relative error: 6.3193857e-16
combined loss directional-gradient relative error: 9.6974786e-11
B normal symmetry relative error: 1.3945323e-16
B normal PSD quadratic form: 2.2169270e-7
support/exterior projection error: 0
truth fixed-point loss: 2.8778979e-33
algebraic reciprocal product gauge error: 9.5338389e-17
representable model reciprocal-scale prediction change: 0.17694513
```

最后两项证明标准 factorization scale gauge 在纯代数 product 上存在，但被本 operator 的固定 homogeneous
reference/exterior boundary 显著破缺；primary identity canonical convention 有实际 operator 依据。

known-B P control 与 exp042 authoritative raw probe byte exact：relative L2 `0`、final detector residual
`0.00339877488`、50 operator-action units、29.48 s。该 control 完全闭合。

known-probe B-only control 的 loss 从 `0.0241627992` 单调降至 `0.00011524594`，detector residual 从
`0.31357317` 降至 `0.02165599`，25 action units、18.86 s；但 active-B error 为 `0.40933423`，超过 development
候选 `0.35`。所以 B-only control 在首次预算下未闭合，不能启动 formal。

blind 两支均使用 fixed-final checkpoint 且 truth-free：

| init | initial/final loss | final intensity residual | final amplitude residual | actions | runtime |
|---|---:|---:|---:|---:|---:|
| homogeneous B | `0.08804793 -> 0.000570961` | `0.0482024` | `0.0266201` | 36 | 26.94 s |
| seeded phase B | `0.08713159 -> 0.000564730` | `0.0479386` | `0.0264712` | 36 | 25.02 s |

两支 prediction relative L2 仅 `0.00132787`，probe pairwise complex-gain-aligned difference `0.0118139`；没有
loss increase、line-search failure、two-cycle、unbounded norm drift 或 seed instability。homogeneous representative
的 simulation-only probe/B/exit errors 为 `0.285987 / 0.420233 / 0.324241`。这些结果在 2 sweeps 后仍高于
development 候选 component/residual gates，但曲线持续下降，因此当前证据是有限预算与 B parameterization
不足，尚不是结构性阻力。

### 2.2 figure 与 artifact 审计

HDF5 `/entry` 八个并列 children exact，`reconstruction` 八个要求子树存在，共 852 paths；source dataset
bytes exact，所有 numeric datasets finite，JSON/HDF5 status一致，无 calibration/preprocessing。五张 PNG 均可
解码并逐张人工查看：

- controls 曲线单调，标签和 log scales 清楚；
- probe 图显示 2-sweep raw amplitude 已形成主要结构，但 phase 和中心误差仍明显；
- B 图显示 unconstrained complex B 出现 `~0.67--1.27` amplitude deviation，而 source B 的 amplitude 明确为 1；
- detector prediction 已复现大尺度结构，residual 仍有 scan-structured low-frequency pattern；
- update/gauge 图没有振荡，probe norm 在首 sweep 改变后稳定，active-B RMS 小幅降至约 `0.9864`。

B-only `update_relative_l2_curve` 首项因以 zero modulation norm 作分母而出现 `1.10e17`，这是诊断分母错误，
不影响 optimizer、candidate、loss、residual 或保存的 raw B。后续将分母改为 physical transmission `1+M`
norm；失败 run 原样保留。

### 2.3 Change 01：有证据的 phase-only B projection 与唯一预算扩展

source artifact 明确记录 sample B type 为 `random_phase`、truth active amplitude 为 1，且首次 B reconstruction
的主要可见偏差包含非物理 amplitude drift。因此在第二次 development 运行前登记：

1. B block 仍使用同一 complex JVP/VJP、normal action、spectral damping、CG 和 Armijo；candidate step 后只在
   已知 active support 上投影到 unit modulus，exterior 仍为 `1+0j`。这是 source measurement design 已知的
   phase-only constraint，不读取 `B_true` values，也不改变 optimizer family。
2. projected Armijo 使用实际 projected displacement 与当前 gradient 的 real inner product，拒绝非 descent
   projection；完整 accepted/rejected step 仍保存。
3. B-only update relative denominator 改为 `||1+M||`，只修复诊断尺度。
4. B-only 从 4 扩到 8 outer steps；blind 从 2 扩到 6 alternating sweeps。每个 blind block 仍是 `1 outer +
   3 power + up to 4 CG`，两个原初始化、代表分支、seed、damping、loss、checkpoint 和 gauge 均不变。
5. 这是 formal 前唯一额外预算/parameterization preflight。若 B-only 仍不能形成可解释 control，则停止，不运行
   blind formal；若 controls 闭合但 blind component recovery 显示结构性阻力，则按证据 Failed 或交 exp044，
   不无限加 iteration。

本 Change 在第二次 run 前写入 EOF；未查看 phase-only 结果。formal thresholds 仍未冻结，第二次 run 仍为
`scientific_gates_enabled=false`。

### 快速恢复上下文

```text
Current authoritative section: Section 2
Latest valid/development/preflight run: runs/exp043_TGV_3d_multislice_blind_probe_B_reconstruction_dev_20260903_150810
Latest formal run: none
Source exp042 run/HDF5: 20260831_183139 / outputs/exp042_probe_reconstruction.h5
Source hashes: HDF5 C48588...A37D; I BCE7A5...5948; scan 63961A...CC5; P truth FA6126...EBFD; B truth 740EBB...3537; raw P rec 194C7B...B07
Frozen root config SHA256: not frozen; latest development run config F036C3...ADAE5
Selected method/budget: alternating block damped GN-CG; phase-only B projection; 6 sweeps; 1x(3 power+4 CG) per block
P-only/B-only/blind: Passed compatibility / first budget not closed / first 2-sweep development stable but above candidate gates
Gauge: primary identity pinned-reference convention; algebraic error 9.53e-17, representable-model change 0.176945
Current single contradiction: whether phase-only + bounded budget closes B control and blind residual/components
Minimum next read: Section 0, Section 2.1--2.3, current root YAML
Next command: conda run --no-capture-output -n tgv_ptycho_sim python -m pytest -q tests/test_exp043_blind_probe_B_reconstruction.py && conda run --no-capture-output -n tgv_ptycho_sim python scripts/run_exp043_blind_probe_B_reconstruction.py --config configs/experiments/exp043_TGV_3d_multislice_blind_probe_B_reconstruction.yaml
```

### 后续建议

继续 exp043：先通过 phase-only projection scoped tests，再执行唯一第二次 preflight；依据其 measurement-only
曲线、B-only control、repeatability 和 component diagnostics 冻结或停止 formal。

另开实验：若剩余矛盾指向 B/scan/large canvas/localized illumination，转 exp044；blind reconstructed raw
probe 的 waist fitting 新开 exp05x；noise、calibration、真实数据和 model mismatch 分开立项。不修改 exp040/
exp042 历史状态。

## 3. 2026-09-03：Implementation iteration 02 — projected-complex GN 阻力与 exact phase-coordinate correction

### 3.1 第二次 development/preflight 的保留结果

第 2.3 节登记的 projected complex-B GN 和 8-step/6-sweep budget 已执行：

```text
run: runs/exp043_TGV_3d_multislice_blind_probe_B_reconstruction_dev_20260903_151542
run state: complete; artifacts_validated=true; scientific_status=Development
runtime: 225.5822745 s
run config SHA256: DDCB3A2062679D00406A3E7E500B5F6BCA96D7CEFAA9998461A6DB16434DC539
metrics SHA256: 908B5ED71A1643FB3DCD85583E37BF0361B1612A270A5887C63C129F98DA8D1F
HDF5 SHA256: A2A36FD9B21611401E4DCC80B426685E47F152D46AF8C09EBC2905867E075631
```

source/operator 和 known-B P control 仍全部闭合，P raw 仍 byte exact。naive unit-modulus projection 使 B-only
loss 虽单调，却只从 `0.0241628` 降到 `0.00749455`，residual 仅从 `0.313573` 降到 `0.174638`，显著差于
unconstrained complex-B 第一次 preflight 的 `0.021656`。B active error 为 `0.413363`，control 失败。

blind representative 在 6 sweeps 的 residual 为 `0.0381245`，仅略好于第一次 2-sweep `0.0482024`；两支 final
prediction difference 仍小 `0.00139255`，loss 单调、无 line-search failure。probe/B/exit simulation-only errors
为 `0.227092 / 0.402914 / 0.308474`。phase projection 保证 active B RMS amplitude exact 1，但没有带来匹配的
phase-space GN direction。

因此第二次 preflight 不能用于 formal。失败 run 和全部 raw artifacts 保留，不改写为成功。证据明确指出：
当前 normal equation/C​​G 在 complex B 的 amplitude+phase 空间求方向，随后 nonlinear projection 删除 amplitude
分量，导致 spectral radius、CG curvature 和实际 feasible tangent 不一致。这是 B block 实现/参数化不一致，
不是换 ePIE、增加 sweep 或放宽 gate 的理由。

### 3.2 Change 02：真实 phase 坐标中的最小 GN-CG 修正

在下一次运行前登记唯一修正：

1. B 参数改为 active support 上 real phase `phi`，`B=exp(i phi)`；source config 已知 B type 为
   `random_phase`，不读取 truth values。
2. phase JVP 严格使用 `dB=i B dphi`；complex modulation VJP 通过
   `Re(conj(iB) g_B)` 转为 real phase adjoint。
3. phase normal action 为 `J_phi^T J_phi`；power iteration、damping 和 real-inner-product CG 全部在同一 real
   phase coordinate 内执行。新增 phase JVP finite difference、JVP/VJP adjoint、normal symmetry/PSD gates。
4. accepted candidate 使用 multiplicative `B_new=B exp(-i alpha dphi)`，天然保持 unit modulus 和 transparent
   exterior；Armijo slope 也使用 phase-coordinate gradient/direction，不再事后投影 complex direction。
5. B-only 8 steps、blind 6 sweeps、每 block power/CG、两个 initializations、seeds、representative、fixed-final
   checkpoint、source、loss 和 gauge 全部保持第二次 preflight 不变。没有追加预算。

这是一个由失败诊断直接支持的 block-coordinate correctness correction，仍属于 alternating spectrally damped
GN-CG，不是替代算法。scoped tests 必须先通过。若 exact phase-coordinate B-only control 仍不闭合，exp043 停止
为 `Inconclusive`，不执行 blind formal；若 control 闭合，则只允许按当前 budget 冻结一次 formal。

### 快速恢复上下文

```text
Current authoritative section: Section 3
Latest valid/development/preflight run: runs/exp043_TGV_3d_multislice_blind_probe_B_reconstruction_dev_20260903_151542
Latest formal run: none
Source exp042 run/HDF5: 20260831_183139 / outputs/exp042_probe_reconstruction.h5
Source hashes: HDF5 C48588...A37D; I BCE7A5...5948; scan 63961A...CC5; P truth FA6126...EBFD; B truth 740EBB...3537; raw P rec 194C7B...B07
Frozen root config SHA256: not frozen; second dev config DDCB3A...DC539
Selected method/budget: exact real-phase B block alternating damped GN-CG; B-only 8; blind 6; 3 power + 4 CG per blind block
P-only/B-only/blind: Passed / naive projection failed / stable but not interpretable while B control fails
Gauge: identity pinned-reference primary; postfreeze reciprocal truth alignment only
Current single contradiction: exact phase-coordinate B control correctness and convergence
Minimum next read: Section 0, Section 3.1--3.2, current exp043 module/YAML
Next command: conda run --no-capture-output -n tgv_ptycho_sim python -m pytest -q tests/test_exp043_blind_probe_B_reconstruction.py
```

### 后续建议

继续 exp043：只测试并执行 exact phase-coordinate preflight；control 闭合时冻结唯一 formal，否则以
Inconclusive 结束，不尝试 ePIE/rPIE 或无界调参。

新开实验：若 B-only phase block正确但 B/scan/illumination导致不可辨，进入 exp044；blind raw probe waist
fit 新开 exp05x；noise/calibration/真实数据/model mismatch 分开立项。exp040/exp042 不回改。

## 4. 2026-09-03：Implementation iteration 03 — exact phase-coordinate preflight 与 formal 冻结

### 4.1 第三次 development/preflight 的保留结果

第 3.2 节预注册的 exact real-phase B coordinate 已完成 scoped tests 和一次 development/preflight：

```text
run: runs/exp043_TGV_3d_multislice_blind_probe_B_reconstruction_dev_20260903_152530
run state: complete; artifacts_validated=true; scientific_status=Development
runtime: 244.8681698 s
run config SHA256: 3D2F1278B595931362A81507A65706975130CF2D8A21B145E2D48936737C9817
metrics SHA256: D94287092A08D57640E10840EEDFE533898606BBFB64D2DA74587B2EED56F8AD
HDF5 SHA256: ECE0E24A27D4DB6E6C8F7508F91806ADB605EFEFDD43409EE12D6C36889F95DE
figures: 5; all required files present and hashed
```

source bytes、scan、truth datasets、independent operator replay 和 exp042 known-B P control 均保持闭合；
authoritative raw P 相对误差为 `0` 且 byte exact，P-control detector relative residual 为
`0.00339877488`。新增 phase-coordinate 数值控制通过：JVP finite-difference relative error
`1.23101968e-9`，JVP/VJP real-adjoint error `4.00949145e-16`，normal symmetry error
`2.90424217e-15`，PSD quadratic form `2.39300264e-7`。combined loss directional-gradient error 为
`9.69747855e-11`。这些结果关闭了 projected-complex GN 的实现不一致。

known-probe/B-only 在 8 个固定 steps 内 loss 从 `0.0241627992` 降为 `0.000270073869`，ratio
`0.0111773`，loss 单调，detector relative residual 为 `0.0331517579`，满足下面冻结的 measurement-only
control gates；simulation-only active-B error 仍为 `0.505317210`，表明 measurement fit 与 B component
identity 必须分开判断。

blind 两个非 truth 初始化均完成固定 6 sweeps。预注册 representative `homogeneous_B` 的 detector relative
residual 为 `0.0273522822`、amplitude residual 为 `0.0147777031`、loss 从 `0.0880479290` 降至
`0.000183847013`；`seeded_phase_B` 对应为 `0.0263798692`、`0.0142568564` 和
`0.000171007327`。两支 prediction relative difference 为 `0.00294165192`。representative 的
post-freeze simulation-only aligned P/B/exit errors 为 `0.206064417 / 0.401164927 / 0.305120617`。

因此，minimum viable alternating 方法已具备进入 formal 的数值正确性和 measurement-only 收敛证据；同时
preflight 已显示 component-level B/exit identity 可能不满足合理的恢复门限。此证据不能用于改代表支、挑 checkpoint、
增加预算或放宽门限。它只用于在 formal 前诚实冻结 component-recovery gate，使低 detector residual 不能单独造成
`Passed`。

### 4.2 Formal 冻结合同

以下内容在 formal 命令执行前冻结：

```text
source/operator: exp042 authoritative 20260831_183139 artifacts and hashes from Sections 1--3
primary method: alternating block spectrally damped GN-CG
P coordinate: raw complex residual field under frozen homogeneous reference
B coordinate: real phase on active finite support, B <- B exp(-i alpha dphi)
gauge: identity measurement-only canonical copy under fixed reference and transparent exterior
truth alignment: post-freeze simulation evaluation only
known-B P control: exact authoritative exp042 replay
known-probe B control: 8 steps, 4 power iterations, 4 CG iterations
blind: 6 outer sweeps; one P block then one B block per sweep; 3 power + 4 CG per block
branches: homogeneous_B and seeded_phase_B
representative: homogeneous_B, fixed before truth metrics
checkpoint: fixed final outer sweep
alternative route: none; not authorized by preflight evidence
```

冻结的 scientific gates 为：

1. source/operator、finite/adjoint/gradient/normal/support/gauge controls 和 artifact contract 必须全部闭合；否则
   `Inconclusive`。
2. known-B P control 必须 raw byte exact、relative-L2 `<=1e-14` 且 detector residual `<=0.0035`；否则
   `Inconclusive`。
3. known-probe B control 必须 detector residual `<=0.04`、final/initial loss ratio `<=0.02` 且 loss
   nonincreasing；否则 `Inconclusive`。
4. blind 两支均必须 finite、fixed-final、loss nonincreasing；最大 detector residual `<=0.03`，两支 prediction
   relative difference `<=0.005`。
5. representative simulation-only gauge-aware component gates 为 P error `<=0.25`、active-B error `<=0.35`、
   exit-wave product error `<=0.25`。truth metrics 仅在 raw primary 冻结后计算，不反馈 optimizer、method、
   checkpoint、damping、stopping 或 representative selection。
6. 前三层控制有效但第 4 或第 5 项失败时，status 为 `Failed`；若 measurement gates 通过而 component gates
   失败，reason 固定为 `measurement_consistent_but_component_recovery_non_identifiable`。全部 gates 通过才可
   `Passed`。

operator-action 预算按实现中的 forward/JVP/VJP-equivalent units 记录而非事后归一化；wall time 必须逐支和总计
保存。formal 无论结果为何只执行一次并保留，不以失败为由删除、增算或重命名为 development。

### 4.3 冻结身份与 append-only 边界

在追加本节之前，Sections 0--3 的文件前缀长度为 `28938` bytes，SHA256 为
`436F30B39E679F2615941C444CA92289ABD80B3BCE1D81640805B40E2D1DCA66`。该值已在追加前重新计算一致。
第 0 节按文档合同仍可原位同步实时状态；第 1 节以后不得回改。formal root artifacts 冻结哈希为：

```text
config YAML: 43C88B4C1158D22633CC8C7BF149E02FDF83105A44CAB31D8FF950B92E862C29
src exp043.py: 97FB733F9F7C54A69D809D27B2DBB13F172762F55248C00F9AE05BDF30258185
runner: D5362D0E7F74D86157883462DF42B0C9967DC159981A1656A3D53C178020BF16
tests: E0C7B2B2C4DBD63EAAED0F4F53693E0FB8D6E7089A3C0F868B2236EB2E855A6D
```

唯一获授权的下一条 scientific execution 是：

```powershell
conda run --no-capture-output -n tgv_ptycho_sim python scripts/run_exp043_blind_probe_B_reconstruction.py --config configs/experiments/exp043_TGV_3d_multislice_blind_probe_B_reconstruction.yaml
```

### 快速恢复上下文

```text
Current authoritative section: Section 4
Latest valid/development/preflight run: runs/exp043_TGV_3d_multislice_blind_probe_B_reconstruction_dev_20260903_152530
Latest formal run: pending, execute exactly once
Source exp042 run/HDF5: 20260831_183139 / outputs/exp042_probe_reconstruction.h5
Source hashes: HDF5 C48588...A37D; I BCE7A5...5948; scan 63961A...CC5; P truth FA6126...EBFD; B truth 740EBB...3537; raw P rec 194C7B...B07
Frozen root config SHA256: 43C88B...C29
Selected method/budget: exact real-phase B alternating damped GN-CG; B-only 8x(4 power+4 CG); blind 6 sweeps with 3 power+4 CG per block
P-only/B-only/blind: Passed / Passed measurement control / formal pending
Gauge: primary identity under pinned reference/exterior; reciprocal truth alignment post-freeze simulation-only
Current single contradiction: whether formal measurement consistency also closes registered P/B/exit component recovery gates
Minimum next read: Section 0, Section 4.1--4.3, frozen YAML
Next command: conda run --no-capture-output -n tgv_ptycho_sim python scripts/run_exp043_blind_probe_B_reconstruction.py --config configs/experiments/exp043_TGV_3d_multislice_blind_probe_B_reconstruction.yaml
```

### 后续建议

继续 exp043：只执行一次 frozen formal，审计 JSON/HDF5/figures，运行 scoped、exp042+043、Ruff 和 full
regression，再追加 authoritative result；不按结果改预算、初始化、gauge、代表支或门限。

另开实验：若 formal 指向 finite-B/scan/localized-illumination 下的 component non-identifiability，转 exp044；
blind reconstructed probe 的 waist fitting 新开 exp05x；noise、calibration、真实数据和 physical-model mismatch
分别立项。不得返回改写 exp040 或 exp042。

## 5. 2026-09-03：Authoritative formal result、artifact audit 与实验关闭

### 5.1 Frozen formal 执行身份

第 4 节冻结后的 root YAML、公共模块、runner 和 tests 哈希在执行前再次核对，无漂移。唯一一次 formal 命令为：

```powershell
conda run --no-capture-output -n tgv_ptycho_sim python scripts/run_exp043_blind_probe_B_reconstruction.py --config configs/experiments/exp043_TGV_3d_multislice_blind_probe_B_reconstruction.yaml
```

其唯一新 run 为：

```text
runs/exp043_TGV_3d_multislice_blind_probe_B_reconstruction_20260903_153944
```

run 于 `2026-09-03T07:43:57.072104+00:00` 完成，`run_state.status=complete`，总 runtime
`247.6062463 s`。root frozen YAML SHA256 保持
`43C88B4C1158D22633CC8C7BF149E02FDF83105A44CAB31D8FF950B92E862C29`；run 内持久化 YAML 的
SHA256 为 `BA0A82855634649F9D43C99F3E5706F78C29A0133699E7F2E512FD653224B2AB`。两者经 YAML load
后 semantic equal；hash 差异只来自 runner 的确定性序列化，不是合同变更。

formal metadata 记录当前 Git commit `47d224ef59d59f372f6c40e7376424f3d81d10ca`，source exp042 自己的
artifact provenance 仍记录 source commit `ac5c20843ff910c1d1619fe590b4b5e18bc38912`。operator 仍为
`R8 unified q8 finite-B open q4 scalar working model`，`reference_validated=false`、
`full_tgv_reference_authorized=false`。

### 5.2 Source/operator 与三层 controls

formal 从 source artifact 读取的 `I_stack`、scan、`P_B_true`、`B_true` 均与 exp042 source dataset bytes
逐元素 exact；独立 forward replay relative-L2 为 `0`，regenerated `I_stack` exact，transparent exterior max
error 为 `0`。HDF5 全文件和各 source dataset hashes 与第 1.3 节一致，无 source substitution。

operator controls 全部通过：

```text
B phase JVP finite-difference relative error: 1.2310196799e-9 <= 2e-6
B phase JVP/VJP real-adjoint relative error: 4.0094914479e-16 <= 1e-11
B phase normal symmetry relative error: 2.9042421749e-15 <= 1e-11
B phase normal PSD quadratic form: 2.3930026364e-7 >= -1e-14
combined loss directional-gradient relative error: 9.6974785530e-11 <= 2e-6
support projection exterior max abs: 0
algebraic reciprocal product-gauge error: 9.5338388929e-17
representable pinned-reference prediction change for nonunit test factor: 0.17694512995
```

三层结果如下：

| branch | initial loss | final loss | detector relative residual | amplitude residual | action units | runtime | status |
|---|---:|---:|---:|---:|---:|---:|---|
| known-B / probe-only | `0.0312771631` | `2.83866564e-6` | `0.00339877488` | n/a | `50` | `29.0764 s` | control Passed; raw byte exact |
| known-probe / B-only | `0.0241627992` | `0.000270073869` | `0.0331517579` | `0.0176106257` | `45` | `33.2681 s` | control Passed |
| blind `homogeneous_B` | `0.0880479290` | `0.000183847013` | `0.0273522822` | `0.0147777031` | `108` | `76.9063 s` | measurement gates Passed |
| blind `seeded_phase_B` | `0.0871315869` | `0.000171007327` | `0.0263798692` | `0.0142568564` | `108` | `97.4460 s` | measurement gates Passed |

known-B P control 的 authoritative raw relative-L2 为 `0`、byte exact；stopping reason 为
`iteration_budget`，total backtracking `0`。known-probe B control 完成 8 steps，final/initial loss ratio
`0.0111772592 <= 0.02`，loss nonincreasing，stopping reason `iteration_budget`，backtracking `0`；其
simulation-diagnostic-only active-B error 是 `0.505317210`，但不参与 optimizer 或该 measurement-control
gate。

两个 blind branches 均完成固定 6 sweeps、使用 fixed-final checkpoint、loss nonincreasing、all finite、
backtracking `0`。两支最大 detector residual `0.0273522822 <= 0.03`，truth-free pairwise prediction
relative-L2 `0.00294165192 <= 0.005`；gain-aligned pairwise probe difference 为 `0.0448448835`。因此 source、
operator、P-only、B-only、blind convergence 和 repeatability 的 measurement-only gates 均关闭。

representative `homogeneous_B` 的最后一个 P/B block gradient norms 分别为 `2.01965519e-5` 和
`1.55039676e-5`，最后 update relative-L2 分别为 `0.0354614144` 和 `0.00929690406`；probe norm
final/initial ratio 为 `0.778283719`，active-B mean phase 累积变化 `-0.0284704434 rad`。停止原因是冻结的
`outer_sweep_budget`，并非依据 truth early stop。总登记 action units 为 `50+45+108+108=311`。

### 5.3 Gauge 与 simulation-only component 判定

primary raw 和 measurement-only canonical 结果使用 identity factor `1+0j`；HDF5 明确记录 convention
`identity_under_frozen_homogeneous_reference_and_transparent_exterior`、`uses_simulation_truth=false` 和
`prediction_invariant_by_construction=true`。这是因为 frozen homogeneous open-grid probe reference 与
transparent B exterior 把非单位 reciprocal scaling 钉在可表示模型之外；不使用 truth 强造另一个 raw complex
reference。affine/raster/weak-illumination 类 equivalence 只作为剩余非辨识性解释，不被伪装为已完全 gauge-fixed。

representative raw primary 冻结后才加载 simulation truth。post-freeze least-squares reciprocal alignment factor 为
`1.10630260806 + 0.04310165461j`，只用于 `simulation_evaluation_only` tree：

| metric | raw | simulation-only aligned | frozen gate | result |
|---|---:|---:|---:|---|
| probe relative-L2 | `0.229654659` | `0.206064417` | `<=0.25` | Passed |
| active-B relative-L2 | `0.405081404` | `0.401164927` | `<=0.35` | Failed |
| exit-wave/product relative-L2 | n/a | `0.305120617` | `<=0.25` | Failed |
| prediction relative-L2 | n/a | `0.0273522822` | measurement gate above | Passed |

因此 authoritative scientific status 为：

```text
Failed / measurement_consistent_but_component_recovery_non_identifiable
```

这是数值有效的 `Failed`，不是 source/operator/control 未闭合导致的 `Inconclusive`。blind measurement fit 和
两初始化 prediction repeatability 通过，但 B 与 exit-product component gates 失败，不能把低 detector residual
提升为 joint component recovery `Passed`，也不能把 primary probe 直接声明为 exp053 类 raw-complex waist fitter
的可用输入。

### 5.4 结构性阻力与替代路线决定

formal 将 `structural_resistance_observed=true`。阻力不是 JVP/VJP、adjoint、phase parameterization、Armijo 或
数值发散：P-only/B-only controls 正确、blind loss 单调、无 rejected step，两个初始化收敛到非常接近的 detector
predictions。矛盾是同一 frozen finite-B/scan/localized-illumination measurement design 下，measurement-consistent
factorization 仍保留过大的 B/exit component error；known-probe B-only 虽有合格 residual，也有 `0.5053` 的 active-B
simulation error，提供同方向诊断。

未尝试 joint GN-CG、LM、ePIE/rPIE 或其他替代 optimizer，`alternative_route_attempted=false`。当前证据不支持
在 exp043 内继续增加 iteration 或撒网换优化器；主要限制已从 optimizer correctness 转移到 component
identifiability/measurement design。按预注册停止规则，这一问题进入 exp044，而不是事后放宽 gate 或重跑 formal。

### 5.5 Artifact、HDF5 与 figure audit

formal artifact hashes：

```text
metrics.json: A35B1029CA59DF2045AB5977FD746E6FEA96E7D7E285C9F64307AFDB429A2D52
metadata.json: A40B394D914BFFC1454E3124DCA28999C4986F9B1999D20B55269645988CAC03
HDF5: BBDF25C93383FB8EA368DDDC60B013D312E68D67AF87688032B7E50CCBB52B09
```

`run_state` 记录 `artifacts_validated=true`、`source_dataset_bytes_exact=true`、
`json_hdf5_status_consistent=true`、`required_hdf5_paths_present=true`、`entry_children_exact=true`、
`all_numeric_hdf5_datasets_finite=true`。独立审计枚举到 `1493` 个 HDF5 objects、`1381` 个 datasets；所有 numeric
datasets finite。`/entry` children 准确为：

```text
config_yaml, data, instrument, metadata, metrics, reconstruction, sample, truth
```

`/entry/reconstruction` 实际包含 `source_provenance`、`design`、`operator_controls`、
`known_b_probe_control`、`known_probe_B_control`、`blind_joint`、`gauge` 和 `checkpoints`。blind 两支均保存 raw
P/B init、raw final、measurement-only canonical、prediction、完整 outer/block records 和 checkpoint history；
representative 另存 post-freeze `simulation_evaluation_only` aligned P/B、factor、errors 与 truth-use flags。
`/entry/sample/sample_b`、source data、instrument、truth、JSON mirrors 和 source hash/shape/dtype/sampling/axis/
units/shift convention 均存在。未创建伪造的 `calibration` 或 `preprocessing` group。项目级 HDF5 schema 无变化。

五张 PNG 均存在且 hash 与 run_state 一致。人工检查结论：

- controls 的 loss/residual 曲线均单调并与 JSON 数值一致；
- reconstructed probe 保留 target 主结构，但 aligned residual 仍有明确空间结构；
- reconstructed B 严格保持 unit amplitude 和 transparent exterior，却未恢复 target random phase 细节；
- scan-0 detector prediction 视觉接近 target，residual 小但具有空间结构；
- block updates 有界并衰减，active-B RMS amplitude 恒为 1，probe norm 无无界漂移，两初始化 prediction heatmap
  的非对角差异与 `0.00294165` 一致。

所有图像可读，无空图、明显裁切、错轴或色条缺失。PNG 只用于人工审计；判定数值均已进入 HDF5 和
`metrics.json`。

### 5.6 Tests、lint 与 Git 状态

formal 后实际运行：

```powershell
conda run --no-capture-output -n tgv_ptycho_sim python -m pytest -q tests/test_exp043_blind_probe_B_reconstruction.py
conda run --no-capture-output -n tgv_ptycho_sim python -m pytest -q tests/test_exp042_probe_reconstruction.py tests/test_exp043_blind_probe_B_reconstruction.py
conda run --no-capture-output -n tgv_ptycho_sim python -m ruff check src/tgv_ptycho/recon/exp043.py scripts/run_exp043_blind_probe_B_reconstruction.py tests/test_exp043_blind_probe_B_reconstruction.py
conda run --no-capture-output -n tgv_ptycho_sim python -m pytest -q
```

结果：

```text
exp043 scoped: 11 passed in 9.81 s
exp042 + exp043 regression: 25 passed in 102.00 s
scoped Ruff: All checks passed
full pytest: 382 passed, 12 failed in 284.35 s
```

全量 12 项逐一仍为已有 exp040 R10 Stage-A/Stage-B、R11、R12、R13、R14、R14A、R14B frozen-config
SHA256 lock mismatch；没有 exp043 failure，也没有新增失败类别。本任务没有修锁或触碰 exp040 files。

最终 `git status -sb` 仍显示进入任务前的混合用户工作区及五个 exp043 untracked source/doc/config/test/runner
文件。没有执行 `git add`、commit、push、PR 或 merge；没有改变 staged 状态，没有删除或回退用户内容。由于
`AGENTS.md`、roadmap、phase 已有用户修改，本任务不对其做可选同步，以避免混入或覆盖来源不明的变更。

### 5.7 Append-only 完整性

追加本节前，允许变化的 Section 0 已同步到 formal 结果；严格 append-only 的 Sections 1--4 从其 UTF-8 marker
开始长度为 `35084` bytes，SHA256 为
`DD02E7E5D1DE2C3B73AC8BB704300F361B33992C23A461FD753322B61E6685AD`。追加本节后必须验证这段 prefix
逐 byte 不变。第 4.3 节登记的 pre-Section-4 `28938`-byte whole-file prefix 只因允许原位更新的 Section 0 而不再
作为 whole-file immutable hash；其中 Sections 1--3 的 append-only body 未修改。

### 快速恢复上下文

```text
Current authoritative section: Section 5
Latest valid run: runs/exp043_TGV_3d_multislice_blind_probe_B_reconstruction_20260903_153944
Latest development/preflight run: runs/exp043_TGV_3d_multislice_blind_probe_B_reconstruction_dev_20260903_152530
Latest formal run: runs/exp043_TGV_3d_multislice_blind_probe_B_reconstruction_20260903_153944
Scientific status: Failed / measurement_consistent_but_component_recovery_non_identifiable
Source exp042 run/HDF5: 20260831_183139 / outputs/exp042_probe_reconstruction.h5
Source hashes: HDF5 C48588...A37D; I BCE7A5...5948; scan 63961A...CC5; P truth FA6126...EBFD; B truth 740EBB...3537; raw P rec 194C7B...B07
Frozen root config SHA256: 43C88B...C29
Selected method/budget: exact real-phase B alternating damped GN-CG; P control 50 actions; B control 45; blind 6 sweeps/108 per branch
P-only/B-only/blind: Passed / Passed measurement control / measurement Passed, component Failed
Gauge: primary identity under pinned reference/exterior; truth reciprocal factor 1.1063026+0.0431017j post-freeze simulation-only
Current single contradiction: frozen measurement design permits stable low-residual predictions without registered B/exit component fidelity
Minimum next read: Section 0, Sections 5.1--5.7, formal metrics/run_state
Next command: none for exp043; create exp044 before changing B/scan/illumination design
```

### 后续建议

继续 exp043：无需再运行或调参；只允许勘误、artifact recovery 或回答审计问题，并继续 append-only 记录。若发现
明确实现错误，必须保留当前 Failed run、先追加 correction，再单独决定是否授权 corrected formal。

转入 exp044：研究 large-canvas finite nonperiodic B、localized illumination、scan/sample-B design 与 component
identifiability；保持本 formal 作为 measurement-consistent non-identifiability baseline，不返回改写 exp040/exp042。

新开 blind-probe waist-fitting 实验：只有在独立任务中定义 raw/canonical/gauge/downstream contract 后，才评估
blind reconstructed probe 的 `D_waist` fitting；不得把本 exp043 的 low detector residual 当作授权。noise、
calibration、真实数据和 physical-model mismatch 也分别另开实验。

## 6. 2026-09-03：最终冻结与仓库级状态同步

### 6.1 冻结决定

用户已确认在第 5 节 authoritative result 基础上停止 exp043 的 optimizer development、追加 iteration、方法
替换和 formal rerun。exp043 现在正式处于：

```text
Scientific status: Failed / measurement_consistent_but_component_recovery_non_identifiable
Work status: Frozen / Closed
Authoritative formal: runs/exp043_TGV_3d_multislice_blind_probe_B_reconstruction_20260903_153944
```

冻结理由是研究问题已经得到数值有效的负面回答：source/operator 与三层 controls 闭合，blind measurement
fit/repeatability 通过，而 B/exit component gates 失败。当前主要矛盾属于 frozen scan/B/illumination measurement
design 下的 component identifiability，不再属于 exp043 内的 block implementation 或无目标 optimizer tuning。

本次冻结操作没有新建 run，没有重写 formal config、code、tests、JSON、HDF5 或 figures，也没有改变任何 scientific
gate。未来不得把更多 iteration、更换 seed、改变 representative、放宽 threshold 或试验其他 optimizer 的结果
冒充本次 formal。

### 6.2 冻结身份账本

冻结时再次读取并核对以下身份：

```text
source exp042 HDF5 SHA256:
C48588FC47EED474CF047219A1DB26BCC9242961A58176CA9CCB879495FFA37D

frozen exp043 root formal config SHA256:
43C88B4C1158D22633CC8C7BF149E02FDF83105A44CAB31D8FF950B92E862C29

src/tgv_ptycho/recon/exp043.py SHA256:
97FB733F9F7C54A69D809D27B2DBB13F172762F55248C00F9AE05BDF30258185

scripts/run_exp043_blind_probe_B_reconstruction.py SHA256:
D5362D0E7F74D86157883462DF42B0C9967DC159981A1656A3D53C178020BF16

tests/test_exp043_blind_probe_B_reconstruction.py SHA256:
E0C7B2B2C4DBD63EAAED0F4F53693E0FB8D6E7089A3C0F868B2236EB2E855A6D

formal metrics.json SHA256:
A35B1029CA59DF2045AB5977FD746E6FEA96E7D7E285C9F64307AFDB429A2D52

formal HDF5 SHA256:
BBDF25C93383FB8EA368DDDC60B013D312E68D67AF87688032B7E50CCBB52B09
```

formal `run_state.status=complete`、`artifacts_validated=true`，五张 figures 已人工审计。最终验证基线仍为
`11 passed` exp043 scoped、`25 passed` exp042+exp043、scoped Ruff clean、全量 `382 passed, 12 failed`；12 项
全部是既有 exp040 R10--R14B frozen-config SHA256 lock mismatch。

### 6.3 允许与禁止的后续动作

冻结后 exp043 只允许：

- 回答审计问题或恢复已有 artifacts；
- 对明确文字错误追加 append-only correction；
- 若发现可复现的实现错误，保留当前 Failed run，先追加 correction，再由新的明确授权决定是否执行独立
  corrected formal。

exp043 不再允许：

- 仅为压低 residual 增加 sweep/CG/action budget；
- 依据 truth error 选择 seed、checkpoint、gauge 或 optimizer；
- 静默改变 frozen thresholds 或覆盖 authoritative run；
- 将当前 raw/canonical probe 直接声明为 blind waist fitter 的已授权输入；
- 返回修改 exp040/exp042 的 frozen scientific status。

仓库级 `AGENTS.md`、`docs/theory_notes/roadmap.md` 和 `docs/experiment_design/phase.md` 已各做一处最小状态
同步，保留其中原有用户修改。README、项目级 HDF5 schema 和历史实验文件未修改。

### 6.4 Append-only 完整性

追加本节前，严格 append-only 的 Sections 1--5 长度为 `48739` bytes，SHA256 为
`05DDAF01BB7C71A619EB1592BE9CE265ECDC9B73B1D0302478990CD5C12F5958`。第 0 节只按约定把 work status、
authoritative section 和 next action 同步为最终冻结状态；追加后须验证上述 Sections 1--5 prefix 逐 byte 不变。

### 快速恢复上下文

```text
Current authoritative section: Section 6
Latest valid/formal run: runs/exp043_TGV_3d_multislice_blind_probe_B_reconstruction_20260903_153944
Latest development/preflight run: runs/exp043_TGV_3d_multislice_blind_probe_B_reconstruction_dev_20260903_152530
Scientific/work status: Failed / measurement_consistent_but_component_recovery_non_identifiable; Frozen / Closed
Source exp042 run/HDF5: 20260831_183139 / outputs/exp042_probe_reconstruction.h5
Source hashes: HDF5 C48588...A37D; I BCE7A5...5948; scan 63961A...CC5; P truth FA6126...EBFD; B truth 740EBB...3537; raw P rec 194C7B...B07
Frozen root config SHA256: 43C88B...C29
Selected method/budget: exact real-phase B alternating damped GN-CG; P control 50 actions; B control 45; blind 108 per branch
P-only/B-only/blind: Passed / Passed measurement control / measurement Passed, B and exit components Failed
Gauge: primary identity under pinned reference/exterior; truth alignment post-freeze simulation-only
Current single contradiction: frozen measurement design permits low-residual stable predictions without registered B/exit fidelity
Minimum next read: Section 0, Sections 6.1--6.4, then Section 5 only for detailed metrics
Next exp043 command: none
```

### 后续建议

继续 exp043：不再进行 scientific computation；仅允许上述审计、恢复或 append-only correction。

转入 exp044：独立预注册 measurement-design/identifiability 问题，研究 large-canvas finite nonperiodic B、
localized illumination、scan/sample-B coding 和近零方向，不回改 exp043。

新开 blind-probe waist-fitting 实验：待新的 blind reconstruction 或独立 downstream gauge contract 闭合后，
另开 exp05x；noise、calibration、真实数据和 physical model mismatch 分别立项。
