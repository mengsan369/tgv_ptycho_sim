# exp052：2D projected reconstructed-probe waist fit：known-B 与 blind-ePIE 双通道

## 0. 实时状态与阅读顺序

```text
Overall scientific status: Passed / single combined formal complete and independently audited
Shared source/operator status: Passed
Part A scientific/work status: reconstruction Passed; waist fit Passed; overall Passed
Part B scientific/work status: reconstruction Passed; waist fit Passed; overall Passed
Results available: true; authoritative formal results and artifacts validated
Latest valid run: runs/exp052_TGV_2d_projected_reconstructed_probe_waist_fit_20260902_235549
Latest development/preflight run: runs/exp052_TGV_2d_projected_reconstructed_probe_waist_fit_preflight_20260902_233926
Latest Part A formal run: contained in combined formal runs/exp052_TGV_2d_projected_reconstructed_probe_waist_fit_20260902_235549
Latest Part B formal run: contained in combined formal runs/exp052_TGV_2d_projected_reconstructed_probe_waist_fit_20260902_235549
Latest combined formal run: runs/exp052_TGV_2d_projected_reconstructed_probe_waist_fit_20260902_235549
Authoritative appended section: section 12 formal result, audit and closure
Primary current question: answered within the registered matched 2D scope; both known-B and blind-ePIE raw probes support the frozen single-parameter fit
exp030 source run/datasets: runs/exp030_TGV_2d_effective_phase_20260810_121124; /entry/data/I_stack, /entry/data/scan_positions, /entry/truth/B_true, /entry/truth/P_B_true
exp050 oracle run: runs/exp050_TGV_2d_projected_true_probe_waist_fit_20260902_192303
Current sole issue: exp052 范围内无未闭合问题；1 nm cell和零 displacement不得提升为物理 uncertainty/resolution
Next action: 不重跑 exp052 formal；nuisance 新开 exp054，blind-algorithm或真实数据问题另立实验
```

阅读顺序：先读本节和第 12 节 authoritative result；再读第 11 节冻结证书；需要复核 source、两臂 reconstruction、gauge、fit 与 gate
时依次读第 3--9 节。

## 1. 历史预留职责（完整原文保留）

以下代码块逐字保留本任务启动前占位文档的完整正文；它是历史职责快照，不是当前状态：

````markdown
# exp052：2D projected-model reconstructed-probe waist fitting

```text
Scientific status: Planned / Not run
Work status: Reserved / Not started
Results available: false
```

## 0. 预留职责

在 exp050 的同一 2D projected forward、case 和 fitting contract 下，使用由 projected data/operator matched reconstruction
得到的 raw `P_B_rec` 拟合 `D_waist`，并与 exp050 oracle result 配对比较 reconstruction-induced bias。

当前没有 authoritative `P_B_rec`、YAML、实现、测试、run 或结果。不得直接使用 exp042 的 3D scalar `P_B_rec`；把 3D data
交给 projected fitter 属于以后单独预注册的 cross-model mismatch，而不是本实验 baseline。
````

## 2. 研究问题、双通道和严格范围

研究对象固定为 exp030 已通过的单孔、无噪声、matched 2D projected-phase working model；除 `D_waist` 外，`D_top`、
`D_bottom`、`z_waist`、折射率、传播距离、phase scale 和其他 geometry/optics 参数全部固定。

Part A 回答：在 `B_true`、scan 和 detector operator 已知且 B 在 optimizer 中逐元素固定时，仅从 detector intensity 恢复的 raw
complex128 `P_B_rec` 是否支持唯一、稳定、可复算的 `D_waist` profile？Part B 回答：当 probe 与 B 均未知，genuine blind ePIE
同时更新两者后，其 raw complex128 `P_B_rec` 是否仍支持同一 candidate forward 下的 profile？比较链固定为
exp050 oracle floor → known-B reconstruction → blind joint reconstruction，不允许用较好的 arm 覆盖另一 arm 的状态。

本实验不做 detector-intensity direct fit，不拟合 nuisance，不引入 noise、stage error、calibration 或真实数据，不研究 sample-B family、
scan redesign、大画布、3D scalar/multislice 或 full-wave mismatch。exp030 的 2D effective phase 仍只是 projected diagnostic；任何
`1 nm` profile cell 都不是物理 uncertainty、resolution、detection limit 或真实三维精度。

## 3. Authoritative source、artifact 直接审计和身份锁

### 3.1 exp030 detector/source chain

authoritative source 为：

```text
runs/exp030_TGV_2d_effective_phase_20260810_121124/
HDF5: outputs/exp030_effective_phase.h5
Git commit: e48932e0f5b77a9fc144e8b8ea2eb710952b21cb
shape: (ny,nx)=(384,384); frames=49; scan coordinate order=(x,y)
dx=detector pixel size=2.5e-7 m; wavelength=5.32e-7 m; z_AB=z_BC=1.0e-3 m
```

直接读取 `config.yaml`、`metadata.json`、`metrics.json`、`run_state.json`、HDF5、durable checkpoints 和 figures 后锁定文件 SHA256：

```text
config:    B96667249BF343A94FC6C8347E8DC87EADF2170B439E7AB81E9845B30E00DC0B
metadata:  3344838EE6A4054E7CD55C6CF918BE44AA0A92AF6567C54A0828DB57869D35E8
metrics:   ED74512B30DA9027B2B192A85C3D5C040A4BD88DED49439BC658AA8667E18818
run_state: 1CDC94725B7410EE29C5E66BEC87160CAEB641DB609B3F6DA4F78AB169088A09
HDF5:      350B85BAE727366893EC57331ACF05E78717591F15313B311899FD5949B7DEB6
```

dataset byte hashes：

```text
/entry/data/I_stack:         066901A76B3ED49AD4EA0CD897EE7979A8C54DE38BBFECA55C33DAE7410651D2
/entry/data/scan_positions:   9AD6B71D4D5977F88D01E37DB9C94B5A4D8D5B6A04486FA8CD36A4E711B7D168
/entry/truth/B_true:          9527922B9F41E762A50CA6ADF878F5FA06184CA3C91EACAD15E873F0BA4D8AC9
/entry/truth/P_B_true:        A020FD6FACABC28A3CFE3D05B3A24C7A9284665BB7998D85C69BB305909BDA40
```

四个 source dataset 都没有 `units` attribute；exp052 只在 provenance 中如实记录缺失，并把 SI/field semantic 写到自己的
instrument metadata，不伪造 source attribute。`I_stack` 为 `(49,384,384)` float64；positions 为 `(49,2)` float64；B 与 probe 为
`(384,384)` complex128，全部 finite。

### 3.2 exp050 oracle comparator

oracle 为 `runs/exp050_TGV_2d_projected_true_probe_waist_fit_20260902_192303/`，已直接审计为 complete、artifact validated、Passed：

```text
config/run copy: 800FB8B33040FAFFDAA229DCF715751525F746AE983923321ADBEE0D049E019B
metadata:        14C34B9F82ABF409D9003E469F1ED78C6A9EAD6E728EBE5A18C93911A3255BFD
metrics:         FF7E407E7EC6462601F0A6F094C7D3AF030ED8B3E083D2242839B7EF4F2E29E4
run_state:       4F3097FF1F998001D6060E4D108E5FD1009E47B6544F8B1D021D234DA6BC52B0
HDF5:            BA6E18A99E7A21B8BFA571109A01FFB235B800E860E194D9BEBCB01D82AD69BD
frozen config:   E6691CB1C7093D8F48D55FC721554B65E1E1E1F6A7104D24663CEACF868B7097
frozen body:     13087 bytes / EDBA033AFD74FD7C6A7AAFF95659A842950F042D7A3A760A9C52C56532095352
oracle estimate: 33.300000 um
oracle cell:     [33.2975, 33.3025] um
```

### 3.3 Git/worktree 审计

开始时分支为 `codex/exp053-reconstructed-probe-q8-cell-interval-fit`。工作区已有用户 modified、untracked 和 deleted 内容，包括
AGENTS/roadmap/phase、exp030/040/050/055 文件、sample-B 代码、notebook 删除和多份草稿；没有把它们清理、回退或覆盖。本任务文件保持
local unstaged，不执行 `git add`、commit、push、PR、merge。

## 4. 两臂 reconstruction 冻结合同和 truth 边界

### 4.1 Part A：重新生成 known-B probe-only raw target

exp030 现有 known-B candidate 只有 60 iterations、`beta_probe=0.08`、final residual `2.541822660848331e-05`，且缺少 exp052 所需的
完整 repeat/checkpoint 证据，因此不作为 authoritative target。exp052 从 locked `I_stack`、positions 和 `B_true` 重新生成：

- initialization 固定为 detector mean-amplitude backpropagation，不读取 `P_B_true`；
- `update_probe=true`、`update_object=false`，B 逐元素固定，periodic integer-pixel shifts 和 matched bandlimited ASM；
- correction 为 `adjoint_residual`，denominator 为 ePIE，`beta_probe=0.5`，shuffle seed `20260733`；
- formal 固定 200 iterations，checkpoints 50/100/200，并用同 seed/budget 做一次完整 bitwise repeat；
- 每个 checkpoint 保存 raw P/B、loss prefix、独立 frozen residual、completed iteration、problem signature、PCG64 state；final 也保存完整
  trajectory 和可恢复 state 摘要。

`B_known` 来源就是锁定的 `/entry/truth/B_true` bytes；这里的 truth 标签表示 simulation source 身份，B 作为 Part A 已知实验输入，不能
更新。`P_B_true` 不进入 initialization、update、stop、branch 或 hyperparameter 选择。

### 4.2 Part B：复用固定的 genuine blind-long final，而不按 truth 选 checkpoint

mandatory blind arm 固定复用 exp030 已实际执行的 `blind_long_study/baseline` 1000-iteration final state；它是真正同时更新 P 与 B 的
blind ePIE-family path，不是 known-B、known-probe 或后处理替代。选择规则在 fit 前固定为
`fixed_authoritative_blind_long_final_1000_not_checkpoint_selected`：不在 200/500/1000 中按 probe truth error 或 waist error挑选；final 1000
是 primary target，三个 checkpoint 全部拟合作为稳定性诊断。

```text
raw probe dataset: /entry/reconstruction/operator_consistency_ablation/blind_long_study/baseline/P_B_rec_raw
raw B dataset:     /entry/reconstruction/operator_consistency_ablation/blind_long_study/baseline/B_rec_raw
raw probe SHA256:  B750FB6C98289BB4832413FAC95F36A65800344C7990DED484BC8733CF41590C
raw B SHA256:      2886CC621EC5681E6C90065B1E06B6B522C7D60B68AF636811B678D8BC05F690
durable checkpoint SHA256: 0D4FA5C9C49F29B00967776E3CFB4C06AFC4C421EB0EDD7DADD2624AAA37460C
problem signature: be48207c9651ea6c367fab9cb50fffa263408931b5651808297549c82f15ff11
RNG: PCG64
checkpoint residuals: 200=1.26860506207945e-4; 500=6.353276329996916e-5; 1000=3.866388915066552e-5
```

source settings 锁定 `beta_probe=0.08`、`beta_object=0.5`、adjoint-residual ePIE denominator、radial output-range probe constraint、
`|B|=1`、periodic shifts；`update_probe=true`、`update_object=true`，两个 `uses_simulation_truth_*_as_input=false`。formal 必须把 raw init、raw
final、B、1000-point loss trajectory、checkpoint fields、settings、durable-checkpoint identity、RNG/signature复制到自己的 HDF5，并独立重算
full-stack residual。

### 4.3 Truth 使用边界

两臂 primary target 永远是原始保存的 complex128 `P_B_rec_raw`。`P_B_true/B_true/D_waist_true` 只允许在 reconstruction 与完整 profile
完成后进入明确命名的 `simulation_evaluation_only`：probe/B error、estimate bias 和 gate evaluation。不得用 truth 选择 initialization、
iteration、checkpoint、branch、mask、gauge 或最终结果。

## 5. Shared operator replay、candidate forward 和 gauge 合同

candidate generator 固定复用 exp050 已验证的 exp030 continuous axisymmetric Fresnel--Hankel compact `T-1` operator 加解析无限平面波
reference；严禁用 coarse `A_effective_true` 普通 ASM 代替。detector replay 固定为 periodic integer shift B、probe×B、bandlimited ASM 到 C、
`abs(U_C)^2`。正式 fit 前执行：true-waist candidate replay、deterministic repeat、完整 `I_stack` intensity/amplitude replay和传播 adjoint
inner-product test。

最后一次 preflight 得到：candidate replay relative L2 `1.3783726375438022e-15`，candidate repeat `0`，I-stack intensity/amplitude replay
均为 `0`，adjoint relative error `2.2918847508779016e-15`；shared contract 为 Passed。

实际 symmetry audit 显示两臂 raw field 与 candidate 存在不可辨 global phase。known B 固定 affine ramp；blind radial probe constraint 去除 affine
ramp，blind `|B|=1` constraint 固定 probe/object magnitude scale。因此 primary loss 对每个 candidate 只解析 profile global phase：

```text
min_phi || exp(i phi) P_candidate(D) - P_target_raw ||_2^2 / ||P_target_raw||_2^2
```

全场等权。raw complex loss和解析 complex-gain-profiled loss只作诊断；不 profile magnitude scale或 affine ramp。每个 candidate 保存 phase、
complex-gain diagnostic、raw/primary loss 和 residual。gauge 参数来自 candidate-versus-target correlation，不读取 `P_B_true`，不是新增
geometry nuisance。raw target、best raw candidate和任何 simulation-only truth comparison分树保存。

## 6. 冻结 fitting、搜索预算和 equivalence 语义

两臂与 2+3 个 checkpoints 共用同一个 deterministic candidate cache；cache key 为 float64 `D_waist_m`，保存 candidate complex128 field、
bytes SHA256 和每条 profile 的 cache index/hash映射。注册 bounds 为 `[32.3,34.3] um`：覆盖 exp050 paired comparator 周围 ±1 um，远宽于
reconstructed-target profile cell；不是事后缩小到 truth。

formal 搜索固定为：

```text
global grid: 50 nm step over the full registered bounds
fine grid:   ±50 nm around the deterministic global minimum, 5 nm step
refinement:  ±10 nm around the deterministic fine minimum, 1 nm step
loss tie tolerance: 1e-14
cross-check: three-point local quadratic vertex, no extra candidate budget
```

若 minimum 靠近 bounds，固定宽度局部网格只在 bounds 内平移，不抛弃 boundary 结果。保存 full global/fine/refinement profile、local minima、
equivalence points和分离 components。reported interval 是 refinement best grid cell的两个 midpoint，formal nominal width `1 nm`；它只表示注册
数值单元。若 plateau/multiple components/boundary出现，按 gate Failed，不强造唯一小数。checkpoint estimate spread是不按 truth选 target的
稳定性 control。

preflight 使用较低预算：100 nm global、10 nm fine、2 nm refinement；它只用于 source/gauge/cost/landscape与实现验证，不是 formal result。

## 7. Formal gates 和互斥状态逻辑

### 7.1 Shared gates

- true-probe replay、deterministic repeat、I-stack intensity和amplitude replay均 `<=1e-12`；
- propagation adjoint inner-product relative error `<=1e-12`；
- 所有 source/artifact/dataset hash、shape、dtype、finite、operator身份和 exp050 oracle certificate闭合；
- gauge symmetry可解释，candidate forward可信。任一失败则 Shared=`Inconclusive`，两臂和overall均=`Inconclusive`，停止后续科学计算。

### 7.2 Reconstruction gates

Part A：final full-stack amplitude residual `<=1e-6`，saved-vs-independent差 `<=1e-12`，B max change `<=1e-12`，完整 repeat probe relative
L2 `<=1e-14`且 bitwise equal，update flags正确，checkpoint/state/artifact闭合。

Part B：fixed final residual `<=5e-5`，saved-vs-independent差 `<=1e-12`，probe 和 B 相对 init update 均 `>=1e-6`，200→500→1000
checkpoint residual严格下降，joint-update与truth-free flags、raw hashes、signature/RNG/durable checkpoint闭合。

证据有效但 gate 未达标为该 reconstruction `Failed`；raw target/provenance或 residual不可验证才为 `Inconclusive`。

### 7.3 Fit gates

每臂必须只有一个 registered global local minimum、一个 equivalence component、interior minimum、finite positive curvature；simulation-only
absolute bias `<=10 nm`、truth到reported cell距离 `<=10 nm`、reported cell width `<=1 nm`、quadratic-vs-grid agreement `<=1 nm`；registered
checkpoint estimates spread `<=10 nm`。浮点比较只允许 `32*spacing(max coordinate magnitude)` 的 ULP tolerance；在当前坐标约
`2.1684e-19 m`，远小于任何科学 gate，不改变 `1 nm/10 nm` 含义。

### 7.4 状态矩阵

分别保存 Shared、Part A reconstruction、Part A fit、Part A overall、Part B reconstruction、Part B fit、Part B overall、exp052 overall。
Shared 非 Passed 时 overall Inconclusive；Shared Passed 后，arm 任一有效 gate Failed则该 arm Failed；两个 arm 都 Passed 时且仅当此时 overall
Passed；任一 arm Failed则 overall Failed；没有 Failed但有 Inconclusive则 overall Inconclusive。唯一 profile但 optimizer/search path失败必须
标记 algorithm/search failure，不能写成 parameter non-identifiability。

## 8. 实现、HDF5、figures 和 artifact contract

公共合同位于 `src/tgv_ptycho/inverse/exp052.py`，combined orchestration 位于
`scripts/run_exp052_projected_reconstructed_probe_waist_fit.py`，参数全部位于 exp052 YAML；没有复制 exp030 forward，也没有修改 exp030、
exp050、exp051或exp053冻结数值行为。新增测试覆盖 source/hash/replay、非法 3D/aligned source、gauge、known-B固定/重复/checkpoint、blind
joint provenance、profile/boundary/multi-component、status matrix和 combined artifact round trip。

combined run 必须自然包含 `config.yaml`、`metadata.json`、`metrics.json`、`run_state.json`、一个 HDF5 和七张 figure。HDF5 `/entry` 只含
并列 `config_yaml/data/instrument/sample/truth/reconstruction/metadata/metrics`；不写空洞 calibration/preprocessing。关键 reconstruction 子树：

```text
/entry/reconstruction/known_b_probe
/entry/reconstruction/blind_epie_probe
/entry/reconstruction/waist_fit/known_b
/entry/reconstruction/waist_fit/blind_epie
/entry/reconstruction/waist_fit/checkpoint_consistency
/entry/reconstruction/waist_fit/candidate_cache
/entry/reconstruction/comparison
/entry/reconstruction/source_operator_replay
/entry/reconstruction/source_provenance
```

artifact audit 固定检查 JSON/HDF5 status一致、source bytes、fixed B、blind truth flags、所有 numeric dataset finite、candidate field/hash mapping、
raw/primary residual identity、HDF5 tree和七张 PNG 可读性。项目级 HDF5 schema 不变。

## 9. Development/preflight、缺陷修复与冻结依据

scoped tests 首轮在 combined writer 中发现 NumPy object-string hash array不被 Windows h5py接受；改为写入前确定性字符串列表，不改变
candidate 或 loss。profile boundary 原实现会在局部区间越界时抛错，已改为 bounds内确定性平移，并显式保存 equivalence components；
boundary curvature保存有限 `0` 使 artifact有效而 scientific gate仍失败。Part A checkpoints补存 loss prefix与 RNG/signature state。

首次 full preflight：

```text
runs/exp052_TGV_2d_projected_reconstructed_probe_waist_fit_preflight_20260902_233244
runtime: 171.2979 s
```

shared、两臂 reconstruction和 artifacts均 Passed；两臂 fit曾因 `2 nm` cell由浮点相减成为
`2.0000000000043854 nm` 而被等值 `2 nm` gate误判。没有调整 source、target、reconstruction、profile、truth bias或科学 threshold；只加入上述
ULP比较并补充逐项 gate flags。

最后一次 full preflight：

```text
runs/exp052_TGV_2d_projected_reconstructed_probe_waist_fit_preflight_20260902_233926
runtime: 171.6545 s
scientific_status_if_formal: Passed
shared / A recon / A fit / B recon / B fit / A overall / B overall / overall: all Passed
artifact audit: Passed; 1062 HDF5 paths; all numeric finite; seven figures readable
```

两次 preflight 的 Part A raw hash、Part B raw hash、两臂 residual和两臂 estimate逐项完全相同；只 gate status变化。最后一次证据：Part A
30-iteration residual `9.881649114136478e-6`、fixed-B change `0`、repeat bitwise exact、raw hash
`CC2FDD9F4CC97FDE8B1AF0D6BD29E4EADA63FA510D56EE04244406220661C3C0`；Part B residual
`3.866388915066552e-5`与独立重算 exact，probe/B update `0.288249/0.455307`，locked raw hashes unchanged。两臂 estimate均
`33.300000 um`、checkpoint spread `0`、single component、interior；Part A minimum primary loss `4.3045013196e-8`，Part B
`6.2991599827e-11`。这些是 preflight，不是 authoritative formal结论。

冻结前验证为 exp052 scoped `11 passed`，exp030+exp050+ePIE+exp052 related `47 passed`，修改 Python文件 scoped Ruff
`All checks passed!`。formal 后仍需重跑 scoped、related、全量 pytest和 Ruff；全量既有 exp040 SHA-lock failures必须单列，不得改 lock。

## 10. 唯一 combined formal 预注册

formal layout固定为一个 combined run，同时保存 Part A和Part B；planned formal run count=`1`。命令固定为：

```powershell
conda run --no-capture-output -n tgv_ptycho_sim python scripts/run_exp052_projected_reconstructed_probe_waist_fit.py --config configs/experiments/exp052_TGV_2d_projected_reconstructed_probe_waist_fit.yaml --mode formal
```

formal source/config、两臂 reconstruction rule、checkpoint selection、global-phase gauge、bounds/profile、equivalence、threshold、status matrix、
HDF5/figure/artifact contract均以第 3--8 节和 root YAML为准。正式结果无论 Passed/Failed/Inconclusive都保存；禁止 truth-based checkpoint
选择、挑 seed、缩 bounds、增加 iteration、改 aligned target、放宽 gate、删除失败 arm或重复 formal。失败只能在第 11 节后 append-only
报告，不能回改本冻结 body。

冻结时 root source config SHA256 为 `F25EA91511B2B568DC54D20F5390B27FBB28783EF68EEBB45F31B1B3CDC13E3D`；相同 config 经项目
`save_config()`规范化后的 run-copy SHA256 已由两次 preflight 验证为
`DDACC1343D001C494CAE9C3DFB4340CD88083FE089636A558280E4060577B4A9`，formal run copy必须相同。

## 11. 2026-09-02：formal 冻结证书、快速恢复与后续分流

第 1--10 节已在任何 formal 输出出现前冻结。锁定范围从首个 `## 1.` 的 UTF-8 起始字节开始，到第 10 节原始 EOF 结束；不包含可持续
原位更新的第 0 节，也不包含本证书及其后的 append-only 结果章节。证书为：

```text
frozen body bytes: 18976
frozen body SHA256: 280B4767A4839AE28C2B637613D7391E7568186ABE9F5895A71D66AAC74BD091
formal root config SHA256: F25EA91511B2B568DC54D20F5390B27FBB28783EF68EEBB45F31B1B3CDC13E3D
expected canonical formal run-copy config SHA256: DDACC1343D001C494CAE9C3DFB4340CD88083FE089636A558280E4060577B4A9
last frozen section: 10
planned formal run count: 1
formal layout: single_combined_formal containing Part A and Part B
```

formal 后只允许原位更新第 0 节，并在本节之后的真实 EOF 追加第 12 节；不得回改第 1--11 节改变 source、reconstruction、gauge、fit、
gate、状态或证书。若唯一 formal失败，保留失败证据并停止，不重跑。

### 11.1 快速恢复上下文

```text
Current authoritative section: section 11 formal freeze certificate
Overall status: ready for one formal; latest preflight scientific_status_if_formal=Passed
Part A status: preflight reconstruction/fit Passed; formal raw target pending deterministic 200-iteration rule
Part B status: authoritative blind reconstruction and preflight fit Passed; formal copy/fit pending
Latest valid/development run: runs/exp052_TGV_2d_projected_reconstructed_probe_waist_fit_preflight_20260902_233926
Latest Part A formal run: contained in planned combined formal; not run
Latest Part B formal run: contained in planned combined formal; not run
Latest combined formal run: not run
Frozen config SHA256: F25EA91511B2B568DC54D20F5390B27FBB28783EF68EEBB45F31B1B3CDC13E3D
Frozen body: 18976 bytes / 280B4767A4839AE28C2B637613D7391E7568186ABE9F5895A71D66AAC74BD091
exp030 source: runs/exp030_TGV_2d_effective_phase_20260810_121124/outputs/exp030_effective_phase.h5
exp030 dataset identity: I_stack=066901A7...51D2; scan=9AD6B71D...D168; B_true=9527922B...8AC9; P_B_true=A020FD6F...DA40
exp050 oracle: runs/exp050_TGV_2d_projected_true_probe_waist_fit_20260902_192303; 33.300000 um; [33.2975,33.3025] um
Part A raw target identity: formal deterministic generation rule in section 4.1; preflight raw=CC2FDD9F...C3C0; formal hash pending
Part B raw target identity: /blind_long_study/baseline/P_B_rec_raw; B750FB6C98289BB4832413FAC95F36A65800344C7990DED484BC8733CF41590C
Current sole issue: execute and audit the only combined formal without changing the frozen contract
Minimum next-read set: section 0, section 11, sections 4--7, latest preflight metrics/run_state
Next executable command: conda run --no-capture-output -n tgv_ptycho_sim python scripts/run_exp052_projected_reconstructed_probe_waist_fit.py --config configs/experiments/exp052_TGV_2d_projected_reconstructed_probe_waist_fit.yaml --mode formal
```

### 11.2 后续建议边界

继续 exp052：只执行唯一 formal、独立 artifact/figure审计、tests/Ruff/full pytest、append-only结果和最小治理同步；不得再做 optimizer sweep。

反馈 exp030：只有 exp030 source hash/operator/replay本身被证伪时才反馈；某个 exp052 optimizer或 gate失败不改写 exp030 Passed边界。

新开 exp054：任何 `D_surface`、`z_waist`、折射率、传播距离或其他 geometry/optics nuisance问题属于 exp054。

另开实验：blind-algorithm改进、sample-B design、finite/nonperiodic canvas、detector-direct fitting、noise/真实 calibration/真实数据、3D/full-wave
或 projected-vs-3D mismatch均不得扩入本次 formal。

## 12. 2026-09-03：唯一 combined formal、独立审计与 authoritative 结论

本节是在第 1--10 节冻结并写入第 11 节证书之后追加的唯一 formal 结果。复算冻结 body 仍为
`18976 bytes / 280B4767A4839AE28C2B637613D7391E7568186ABE9F5895A71D66AAC74BD091`；root config 仍为
`F25EA91511B2B568DC54D20F5390B27FBB28783EF68EEBB45F31B1B3CDC13E3D`。结果出现后没有修改 source、target、iteration、
checkpoint、gauge、bounds、profile、threshold、status matrix或 formal artifacts，也没有第二个 formal run。

### 12.1 Formal run、状态与身份

唯一 combined formal 为：

```text
runs/exp052_TGV_2d_projected_reconstructed_probe_waist_fit_20260902_235549/
runtime: 999.6020818 s
run_state: complete
artifacts_validated: true
experiment_status: Passed
Shared / Part A reconstruction / Part A fit / Part A overall: Passed
Part B reconstruction / Part B fit / Part B overall / exp052 overall: Passed
```

formal artifact SHA256：

```text
config.yaml:   DDACC1343D001C494CAE9C3DFB4340CD88083FE089636A558280E4060577B4A9
metadata.json: 10F321B8909F255E5C15C7F7979557261CD0C6E0ABAB77CDE2A7AFFC2E96DC62
metrics.json:  8672B5F74B728EC431C6FCF30E431C8E80C738998D72B18689F4640329EE50FB
run_state:     CABB70D89EA63D26D7E99E6319C29A6F0B9387A1B3652B2E5F5B730C285F88F1
HDF5:          C4CCEEEB8503546911C87D3BCAD5FD6718806D457644CA3F591EEC7894EF3418
```

runs目录中只有两个 `_preflight_` development run和本次一个非-preflight formal目录；planned formal run count=`1`已满足。

### 12.2 Shared、Part A 与 Part B reconstruction 证据

Shared replay与 preflight一致：true-probe candidate replay relative L2 `1.3783726375438022e-15`，repeat `0`，I-stack intensity/amplitude
replay均 `0`，adjoint relative error `2.2918847508779016e-15`。exp030四个 dataset bytes和 exp050 oracle certificate均 exact。

Part A按冻结的 measurement-derived initialization、fixed B、`beta_probe=0.5`、200 iterations运行：

```text
initial detector-amplitude residual: 1.3658545205791736e-1
final/saved/independent residual:     5.344692857844719e-8 / exact match
fixed-B max absolute change:          0
full repeat:                          bitwise exact; probe/B/loss differences all 0
P_B_init SHA256:                      414AA30FC414BCCBAAE60571120C7852E5B0E3A9110059514207B44DACBDCF1B
P_B_rec_raw SHA256:                   04096B56931341C443667088BE264CA14E08D6E6D8787BE0C21EA201475C3FFB
B_known SHA256:                       9527922B9F41E762A50CA6ADF878F5FA06184CA3C91EACAD15E873F0BA4D8AC9
simulation-only probe error:          1.5376877008e-6 global-phase-profiled relative L2
```

50/100/200 checkpoints全部保存 fixed B、对应 loss prefix、independent residual、completed iteration、problem signature和 PCG64 state；三者
fit estimate均为 `33.300000 um`。

Part B formal 没有重选 source，exact复用冻结的 genuine blind-long 1000 final：

```text
initial/final detector-amplitude residual: 1.3788381059142998e-1 / 3.866388915066552e-5
independent residual:                       3.866388915066552e-5 / exact match
probe update relative L2:                   0.28824904511079713
B update relative L2:                       0.4553071708952263
P_B_rec_raw SHA256:                         B750FB6C98289BB4832413FAC95F36A65800344C7990DED484BC8733CF41590C
B_rec_raw SHA256:                           2886CC621EC5681E6C90065B1E06B6B522C7D60B68AF636811B678D8BC05F690
simulation-only probe error:                7.9367870844e-6 global-phase-profiled relative L2
simulation-only B error:                    4.2801884028e-3 global-phase-profiled relative L2
```

200/500/1000 residual严格下降，三个 checkpoint fit均为 `33.300000 um`；joint-update为 true，两个 truth-use flags为 false，durable
checkpoint/hash/signature/RNG闭合。

### 12.3 Oracle→known-B→blind fitting 比较

| arm | estimate (um) | registered cell (um) | minimum primary loss | curvature (m^-2) | checkpoint spread |
|---|---:|---:|---:|---:|---:|
| exp050 oracle | 33.300000 | [33.2975, 33.3025] | replay floor 0 | 1.292005571e9 | n/a |
| Part A known-B | 33.300000 | [33.2995, 33.3005] | 2.3644834652e-12 | 1.311833644e9 | 0 nm |
| Part B blind-ePIE | 33.300000 | [33.2995, 33.3005] | 6.2991599827e-11 | 1.310237013e9 | 0 nm |

registered pairwise displacement全部为 `0 nm`：known-minus-oracle、blind-minus-oracle、blind-minus-known均为零。两臂都是一个 global local
minimum、一个 equivalence component、interior、finite positive curvature；cell width约 `1 nm`。quadratic cross-check相对 grid minimum差分别
为 `9.4091e-6 nm` 和 `0.033088 nm`，远小于冻结的 `1 nm` gate。两臂 raw loss约为 `3.65358/3.65394`，说明 global phase gauge不可
忽略；global-phase primary floor远低，complex-gain diagnostic只作旁证，没有把 truth-aligned field作为 target。

从 persisted shared candidate cache只读计算 `D=33.3 um` 到 `D±1 nm` 的 global-phase-profiled candidate field signature，relative L2分别
为 `2.5610614283e-5` 和 `2.5611148102e-5`，均值 `2.5610881192e-5`。Part A/Part B target mismatch floor为
`1.5376877008e-6/7.9367247544e-6`，相对一格 1 nm signature的比值为 `0.06004/0.30990`。该 ratio是从 HDF5内已保存的 candidate和
minimum loss派生的数值诊断，不是物理 SNR或 detection limit。

### 12.4 独立 artifact/HDF5/figure 审计

runner自审计后又用独立只读脚本复核：`1157` 个 HDF5 paths、`76` 个 shared candidates、`24` 个 profile groups；所有 source/raw/cache
SHA、profile cache index/hash映射、best candidate、raw/primary residual identity、fixed B、blind flags、checkpoint state和 JSON/HDF5 status
通过。所有 numeric datasets finite；没有 `aligned_to_truth` 或 `common_gauge` path。Part A/Part B residual独立重算 exact为
`5.344692857844719e-8` 与 `3.866388915066552e-5`。

HDF5 `/entry` 并列结构为 `config_yaml/data/instrument/sample/truth/reconstruction/metadata/metrics`，没有空洞 calibration/preprocessing。
known-B、blind、waist fits、checkpoint consistency、candidate cache、comparison和 source replay/provenance均存在；项目级 HDF5 schema无变化。

七张 PNG 均可读且 SHA与 run_state一致。人工查看确认 convergence、single-minimum global/local profile、raw amplitude/phase、best residual、
gauge、checkpoint stability和 oracle→known→blind comparison相互一致。convergence右面板没有单独的“Part B”标题，是非科学性标注小缺点；
1000-iteration轴与 paired layout仍可辨，所有数值和 arm身份在 HDF5/metrics中明确。formal后不修改图片或重跑。

### 12.5 实际命令、tests 和 lint

主要实际命令：

```powershell
git -c safe.directory=E:/tgv_ptycho_sim status -sb
conda run --no-capture-output -n tgv_ptycho_sim python scripts/run_exp052_projected_reconstructed_probe_waist_fit.py --config configs/experiments/exp052_TGV_2d_projected_reconstructed_probe_waist_fit.yaml --mode preflight
conda run --no-capture-output -n tgv_ptycho_sim python scripts/run_exp052_projected_reconstructed_probe_waist_fit.py --config configs/experiments/exp052_TGV_2d_projected_reconstructed_probe_waist_fit.yaml --mode formal
conda run --no-capture-output -n tgv_ptycho_sim python -m pytest -q tests/test_exp052_projected_reconstructed_probe_waist_fit.py
conda run --no-capture-output -n tgv_ptycho_sim python -m pytest -q tests/test_exp020_components.py tests/test_exp030_observability.py tests/test_exp050_projected_true_probe_waist_fit.py tests/test_epie_shapes.py tests/test_hdf5_layout.py tests/test_exp052_projected_reconstructed_probe_waist_fit.py
D:\anaconda3\envs\tgv_ptycho_sim\python.exe -m ruff check src/tgv_ptycho/inverse/exp052.py scripts/run_exp052_projected_reconstructed_probe_waist_fit.py tests/test_exp052_projected_reconstructed_probe_waist_fit.py
conda run --no-capture-output -n tgv_ptycho_sim python -m pytest -q
```

两次 preflight路径为 `20260902_233244` 与 `20260902_233926`；第二次仅验证 ULP gate修复，raw hashes/residual/estimate与第一次逐项相同。
最终 exp052 scoped为 `11 passed in 9.48s`；exp020+030+050+ePIE+HDF5+052 related为 `56 passed in 17.44s`；scoped
Ruff `All checks passed!`。全量为 `371 passed, 12 failed in 388.51s`；12 项全部是任务前已登记的 exp040 R10--R14B frozen-config
SHA256 lock mismatch，没有 exp052或新增失败，未修改任何 exp040 lock。

### 12.6 结论、限制、Git 和治理

authoritative scientific status为 `Passed`：在这一个无噪声、matched、integer-pixel periodic、同shape sampling、single-parameter exp030 2D
projected working model内，known-B与冻结的 blind-ePIE raw probe都支持稳定的单参数 waist fit；在注册 1 nm grid上未观察到 estimate
displacement或多解 component。blind arm相对 known-B的新增退化表现为更高 field mismatch floor和更高 floor/1 nm signature ratio，而不是本次
registered estimate位移。

这不证明 production blind reconstruction、真实 calibration/noise/stage error、finite nonperiodic object、subpixel scan、detector integration、
nuisance robustness、actual uncertainty/resolution/detection limit、真实三维 TGV 或 full-wave physics。2D projected model不能替代3D结论。

已最小同步 `AGENTS.md`、`docs/theory_notes/roadmap.md` 和 `docs/experiment_design/phase.md`，没有改写 exp030/050历史结论或 README。
所有本任务修改保持 local unstaged；未执行 `git add`、commit、push、PR或merge，用户原有 modified/untracked/deleted内容均保留。

### 12.7 快速恢复上下文

```text
Current authoritative section: section 12 formal result, audit and closure
Overall / Part A / Part B: Passed / Passed / Passed
Latest valid/formal run: runs/exp052_TGV_2d_projected_reconstructed_probe_waist_fit_20260902_235549
Latest development run: runs/exp052_TGV_2d_projected_reconstructed_probe_waist_fit_preflight_20260902_233926
Frozen config SHA256: F25EA91511B2B568DC54D20F5390B27FBB28783EF68EEBB45F31B1B3CDC13E3D
Frozen body: 18976 bytes / 280B4767A4839AE28C2B637613D7391E7568186ABE9F5895A71D66AAC74BD091
exp030 source: runs/exp030_TGV_2d_effective_phase_20260810_121124/outputs/exp030_effective_phase.h5
exp050 oracle: runs/exp050_TGV_2d_projected_true_probe_waist_fit_20260902_192303; 33.300000 um; [33.2975,33.3025] um
Part A raw target: formal HDF5 /entry/reconstruction/known_b_probe/P_B_rec_raw; 04096B56931341C443667088BE264CA14E08D6E6D8787BE0C21EA201475C3FFB
Part B raw target: formal HDF5 /entry/reconstruction/blind_epie_probe/P_B_rec_raw; B750FB6C98289BB4832413FAC95F36A65800344C7990DED484BC8733CF41590C
Current sole issue: none within exp052; do not reinterpret 1 nm cell as physical uncertainty/resolution
Minimum next-read set: section 0, section 12, section 11 certificate, formal metrics/run_state
Next executable command: git -c safe.directory=E:/tgv_ptycho_sim status -sb
```

### 12.8 后续建议

继续 exp052：无需新增科学计算或重跑 formal；只有发现 artifact/hash/document inconsistency时才在真实 EOF追加纠错章节。

反馈 exp030：当前 source/operator exact闭合，无需反馈；只有未来证伪其 source或operator时才返回 exp030。

新开 exp054：若研究 `D_surface/z_waist` 或其他 projected nuisance，按 exp054单独预注册。

另开实验：blind optimizer改进、sample-B/scan design、finite/nonperiodic canvas、detector-direct、noise/真实 calibration/真实数据、3D/full-wave
和 projected-vs-3D mismatch分别建立新问题，不扩入 exp052。
