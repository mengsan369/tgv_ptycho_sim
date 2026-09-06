# exp050：2D projected-model true-probe waist fitting

## 0. 实时状态与阅读顺序

本节是全文唯一允许持续原位更新的入口。第 1 节以后从本次 formal 设计冻结起 append-only；以后只允许在真实 EOF
追加新编号章节。

```text
Scientific status: Passed / authoritative formal complete and artifacts validated
Work status: Complete / exp050 closed within its registered 2D projected boundary
Results available: development/preflight and authoritative formal
Latest valid run: runs/exp050_TGV_2d_projected_true_probe_waist_fit_20260902_192303
Latest development/preflight run: runs/exp050_TGV_2d_projected_true_probe_waist_fit_preflight_20260902_191340
Latest formal run: runs/exp050_TGV_2d_projected_true_probe_waist_fit_20260902_192303
Authoritative appended section: section 13 (formal result, independent artifact audit, tests and closure)
Primary current question: 固定 exp030 其余参数，matched raw P_B_true 是否支持稳定的单参数 D_waist oracle fit
exp030 source run: runs/exp030_TGV_2d_effective_phase_20260810_121124
exp030 target dataset: outputs/exp030_effective_phase.h5:/entry/truth/P_B_true
Current sole issue: exp050 范围内无未闭合问题；后续 reconstructed-probe 问题属于新实验 exp052
Next action: 不重跑 exp050 formal；若继续 projected 链，新开 exp052 并先审计 matched raw P_B_rec source/operator handoff
```

阅读顺序：先读本节和第 13 节 authoritative result；再按需读第 3--9 节 source/方法/冻结 gate、第 6 节 preflight 和第 12 节
formal 前冻结证书。

## 1. 历史预留快照（原文保留）

以下是本任务启动前预留文件的完整原文；它是历史职责快照，不是当前实时状态：

````text
# exp050：2D projected-model true-probe waist fitting

```text
Scientific status: Planned / Not run
Work status: Reserved / Not started
Results available: false
```

## 0. 预留职责

使用 2D projected TGV forward 产生的 matched `P_B_true`，建立单参数 `D_waist` oracle fitting baseline。
本实验只验证简化 projected model 内的参数化拟合、局部可辨识性和 optimizer；不使用 `P_B_rec`，不把结果解释为
3D scalar/full-wave 物理准确性或真实腰径精度。

当前只有实验编号与职责；geometry、case、loss、参数范围、finite-difference/profile controls、threshold、YAML、实现、测试和 run
均未预注册。后续启动时必须重新读取 exp030 的最终证据边界，并与 exp051 明确区分 forward-model identity。
````

## 2. 研究问题、依赖与严格 scope

问题固定为：在 exp030 已验证的单孔、无噪声、轴对称 2D projected-phase working model 内，固定
`D_top=50 um`、`D_bottom=50 um`、`z_waist=350 um`、厚度、折射率、波长、A-to-B 距离、sampling、FOV、
incident field 和绝对 complex reference，只改变 `D_waist`，直接使用 authoritative raw `P_B_true` 时，能否得到自洽、
唯一于注册 evidence support、数值稳定且可复算的单参数 estimate？

primary input 只能是 exp030 `/entry/truth/P_B_true`。不使用 `P_B_rec`、detector intensity、sample B、scan、ePIE、blind
变量或 nuisance；不使用 exp040/042/051/053 数据。truth 只在 replay 和 simulation-evaluation accuracy 中出现，不参与
profile basin 选择、golden stopping、threshold 选择或 branch selection。Passed 最多表示
`single-parameter true-probe oracle fitting is self-consistent within the exp030 2D projected working model`。

本实验不回答真实三维传播、full-wave 准确性、真实 uncertainty/resolution、noise、calibration、tilt、stage error、
model mismatch 或 detector-direct fitting。

## 3. exp030 authoritative source 实物审计

实际只读核对对象为：

```text
run: runs/exp030_TGV_2d_effective_phase_20260810_121124
config SHA256:    B96667249BF343A94FC6C8347E8DC87EADF2170B439E7AB81E9845B30E00DC0B
metadata SHA256:  3344838EE6A4054E7CD55C6CF918BE44AA0A92AF6567C54A0828DB57869D35E8
metrics SHA256:   ED74512B30DA9027B2B192A85C3D5C040A4BD88DED49439BC658AA8667E18818
run_state SHA256: 1CDC94725B7410EE29C5E66BEC87160CAEB641DB609B3F6DA4F78AB169088A09
HDF5 SHA256:      350B85BAE727366893EC57331ACF05E78717591F15313B311899FD5949B7DEB6
source Git commit: e48932e0f5b77a9fc144e8b8ea2eb710952b21cb
source state: complete / experiment Passed / Stage A--C Passed / Stage D Passed
```

target 身份为 `(384,384) complex128`、axis `(y,x)`、`dx=0.25 um`、FOV `96 um × 96 um`、坐标
`(index-(N-1)/2)*dx`，端点 `-47.875...+47.875 um`，几何原点位于四个中央 samples 之间。field 语义为 arbitrary
complex amplitude，phase 为 rad。source dataset 没有 `units` attribute，旧 run_state 也没有 `artifacts_validated` 字段；
exp050 必须记录“字段不存在、但本轮独立 hash/tree/finite 审计通过”，不得伪造上游声明。

冻结数组 hash：

```text
P_B_true bytes:             A020FD6FACABC28A3CFE3D05B3A24C7A9284665BB7998D85C69BB305909BDA40
radial_source_r_m:           8DD1784D9339DAEC2C93B64E99C3313616A4D065C07D613615CE63463D4EB6FB
radial_source_weight_m:      A7A138C094305EC89BB83D1239AC3840202146A61931091C8A0A17CC970C5188
A_effective_radial_true:     DB236D4BB1262127460A05E7262D56998D3D272032C036C3F53D57A860F89598
P_B_radial_r_m:              03CAFFA47FA56DAD5907340B83EDA2F7F5C3F9C9D894FC9E5F84A3F793E1ED9D
P_B_radial_true:             CDE1CBE5389BA6B19A380DE00A9B59E8BB291AA59807634466BCB678B83FC824
```

HDF5 `/entry` 为 `config_yaml,data,instrument,metadata,metrics,reconstruction,sample,truth`。本实验只读取上述 raw truth、
instrument/sample identity 和 effective-forward arrays；任何 reconstruction subtree 均禁止作为 target。

## 4. Candidate forward 与 exact replay 规则

authoritative A-to-B forward 不是粗 `A_effective_true` 的普通 ASM。它把 projected transmission 写为
`T(r)=1+q(r)`，无限平面波 `1` 解析传播，只对 compact `q=T-1` 做连续轴对称 Fresnel--Hankel 积分，再在已保存
`P_B_radial_r_m` 上得到 radial field并线性插值到 Cartesian B plane。

本轮新增公共 `src/tgv_ptycho/forward/exp030.py`，直接加载 source HDF5 冻结的 9674 个径向节点/权重和 1087 个 B-plane
径向坐标；candidate 只重算 analytic air path、unwrapped phase、radial transmission和同一传播。它从不读取粗
`A_effective_true` 作为 forward input。共享接口的 regression criterion 是 authoritative raw/radial replay 均 `<=1e-12`、
repeat `<=1e-14`；否则 formal fit停止并为 Inconclusive。

## 5. 路线选择及为何不机械复刻 exp051

最终路线是 **global profile-first + local fixed-budget golden cross-check**：

1. 全注册范围等间距 raw complex loss profile 负责发现其他 minimum、边界解或非唯一结构；
2. coarse argmin 唯一派生局部 fine profile；primary estimate 是 fine-profile argmin；
3. 报告区间是该 fine grid 点与左右相邻点中点形成的 5 nm-wide profile cell，只表示注册 profile resolution；
4. coarse minimum 的左右相邻点唯一决定 golden bracket，固定 8 次迭代、不 early stop；它只验证独立搜索方法与 profile
   一致，不把 sub-grid bracket 当物理精度；
5. 两档中心 FD 检查 complex Jacobian direction/norm、curvature、step convergence和 signature/floor separation。

该一维问题已有完整 global screen，不需要四起点 pattern search；golden bracket 由 profile 决定，比另造多个局部初值更直接。
也不采用 exp051 q8 plateau interval：exp030 continuous radial geometry 在 preflight 中没有 fixed-q8 的离散 plateau。没有使用
phase/gain alignment、结果后 mask、profile interpolation或 truth-aided branch selection。若 formal 出现 plateau，按冻结唯一性
gate Failed，而不是事后换 estimator。

primary loss 为全视场、等权、无 alignment 的

$$
L(D)=\frac{\|P_B(D)-P_B^{target}\|_2^2}{\|P_B^{target}\|_2^2}.
$$

## 6. Development/preflight 结果与方法冻结依据

唯一 real-case preflight 为：

```text
runs/exp050_TGV_2d_projected_true_probe_waist_fit_preflight_20260902_191340
mode/status: preflight / Development / artifacts_validated=true
runtime: 5.607 s
candidate cache: 43 fields
```

它使用 100 nm global、20 nm local profile和 6 次 golden，只回答方法选择，不是 authoritative science。结果：

```text
raw Cartesian replay L2:              1.37837263754e-15
raw radial replay L2:                 2.18929767917e-15
amplitude replay L2:                  8.61661215048e-16
phase-sensitive replay L2:            1.09133981760e-15
deterministic repeat L2:               0 (bitwise equal)
global/fine minimum:                   unique at 33.300 um
preflight profile cell:                [33.290, 33.310] um
FD h1/h2:                              125 / 15.625 nm
normalized |J_h1| / |P|:              25414.4125 1/m
normalized |J_h2| / |P|:              25411.6142 1/m
Jacobian step relative L2:             1.57985191e-4
one-sided h2 signature:                3.96999218e-4
signature / replay floor:              2.88020240e11
loss curvature:                        1.29200557e9 1/m^2
profile/golden estimate:               both 33.300 um at preflight precision
```

global profile 从 32.3 到 34.3 um 平滑下降后上升，没有第二 minimum或边界 minimum。preflight figures/HDF5 可读且全数值
finite。由此选择 profile-first 路线，并冻结更密的 50 nm global、5 nm local profile；不开发复杂 optimizer。

## 7. Authoritative formal 冻结配置

formal 运行只允许使用当前 YAML `formal` 节：

```text
bounds: [32.3,34.3] um
global profile: 50 nm spacing, 41 points
fine profile: coarse minimum ±50 nm, 5 nm spacing, 21 points
FD half-steps h1/h2: 125 / 15.625 nm
golden bracket: global argmin 的左右相邻 global points
golden iterations: exactly 8; no early stop
primary estimate: fine-profile minimum
reported resolution interval: fine point左右相邻中点，预期 width 5 nm
seed: 20260902（当前算法 deterministic，不参与 branch selection）
```

范围 `[32.3,34.3] um` 不是通用 TGV prior。它来自 exp030 source quadrature 的既有最大 `D_waist=33.3±1.0 um`
sweep：这保证候选 transition annulus始终落在 source 用 1 nm 节点构造的精细径向区域。更宽范围需要重新设计和验证
quadrature，属于新的 numerical-control iteration，不能在 formal 后扩张。

## 8. 冻结 gates 与数值/物理信息分离

Stage A source/replay：所有文件、dataset和 operator-array hash exact；source state complete/Passed；target shape/dtype/finite/
positive energy；Cartesian raw、amplitude、phase-sensitive、radial replay各 `<=1e-12`；repeat `<=1e-14`。

Stage B profile：global和 fine loss 全有限；两者在 `max(1e-20,100*replay_floor_loss)` 内各只有一个 minimum；fine minimum
为内点；simulation-only absolute error `<=5 nm`；minimum loss `<=1e-20`；fine second-best loss / replay-floor loss `>=1e6`。
profile cell width固定 5 nm，仅为离散 search resolution，不是 uncertainty或 detection limit。

Stage C local numerical control：`||J_h1||/||P|| >=1e4 1/m`；curvature finite且正；
`||J_h1-J_h2||/||J_h2|| <=0.05`；h2 one-sided signature/replay floor `>=1e6`。h1/h2 来自 exp030 已注册 step
family，不在 formal 后更换。

Stage D estimator cross-check：8 次 fixed golden；最终 bracket width `<=5 nm`；estimate simulation-only absolute error
`<=5 nm`；与 fine profile minimum距离 `<=5 nm`；bracket距两 fitting bounds均 `>50 nm`。不得把 golden 最终 bracket
或机器精度 loss floor解释为物理信息。

## 9. Passed / Failed / Inconclusive 互斥逻辑

按顺序：

1. source identity/hash/state、target/operator handoff、raw replay/determinism或 artifact completeness 未闭合：
   `Inconclusive / artifact_operator_handoff_not_closed`；禁止后续 fit；
2. evidence 已产生但 FD/profile numerical resolution或 signature-to-floor 无法解释：
   `Inconclusive / local_numerical_control_not_closed`；
3. replay和 numerical controls有效，但 profile 非唯一/边界/accuracy gate或 fixed-budget cross-check agreement 失败：
   `Failed / registered_profile_or_estimator_stability_gate_failed`；
4. 所有 scientific和 artifact gates 通过：`Passed / exp030_projected_true_probe_single_parameter_oracle_fit_passed`。

profile 唯一但 golden 失败只说明所选 cross-check search 不稳定，不得写成参数不可辨识。formal 失败后保存结果并停止，不修改
threshold、bounds、grid或预算重跑。

## 10. Artifact、HDF5 和 figures contract

formal run 必须包含 `config.yaml`、`metadata.json`、`metrics.json`、`run_state.json`、
`outputs/exp050_projected_true_probe_waist_fit.h5` 和四张 figures。HDF5 `/entry` 并列结构不变；`/entry/data` 为空，
不得伪造 `I_stack`、scan、sample B、calibration或 preprocessing。

自然保存：instrument/axis/units semantic；`sample/sample_a`；raw `truth/P_B_true`、`D_waist_true_m`和 source radial arrays；
`reconstruction/waist_fit` 下的 design、source provenance、target/repeat replay fields、candidate cache、global/fine profile和映射、
FD complex Jacobians/controls、完整 golden evaluations/brackets、best raw candidate和 raw residual；并列 metadata/metrics。项目级
schema 无变化。

四图分别检查 global/local loss landscape和 profile cell、target/best/raw residual、两档 Jacobian及差异、golden bracket/evaluation
track。PNG 只用于人工检查，数值判断全部来自 JSON/HDF5。

formal 后 validator 必须检查 source/target hash、`run_state complete/artifacts_validated=true`、JSON/HDF5 status、target/best/cache/
residual identity、cache mapping、all numeric finite、空 data、无 processing groups、无 `P_B_rec` input flag和四图可读 finite。

## 11. 实现、测试与执行顺序

新增共享 forward `src/tgv_ptycho/forward/exp030.py`、exp050 inverse/source contract
`src/tgv_ptycho/inverse/exp050.py`、runner、真实 YAML和 tests。`waist_fit.py` 只复用 raw loss、replay、profile minimum和 FD
controls，不改变 exp051/053 数值行为。现有 exp030 runner未改；authoritative artifact raw replay作为公共化 regression gate。

formal 前先通过 scoped tests和 Ruff；formal 只运行一次；之后独立审计并再运行 scoped、combined相关测试、全量 pytest和 scoped
Ruff。全量既有 exp040 SHA locks失败必须单独报告，不修改其 lock。所有修改保持 local unstaged，不执行 Git发布操作。

## 12. 2026-09-02：formal 预注册证书、快速恢复与后续建议

第 1--11 节在 formal 前冻结。锁定范围从首个 `## 1.` 的 UTF-8 起始字节开始，到第 11 节原始 EOF 结束，不包含
可实时更新的第 0 节，也不包含本节：

```text
frozen body bytes: 13087
frozen body SHA256: EDBA033AFD74FD7C6A7AAFF95659A842950F042D7A3A760A9C52C56532095352
formal source config SHA256: E6691CB1C7093D8F48D55FC721554B65E1E1E1F6A7104D24663CEACF868B7097
last frozen section: 11
```

正式结果出来后只允许原位更新第 0 节，并在本节之后真实 EOF 追加第 13 节；不得回改第 1--12 节来改变结论。

### 12.1 快速恢复上下文

```text
Current authoritative section: section 12 preregistration; formal result pending
Latest valid run: none
Latest development run: runs/exp050_TGV_2d_projected_true_probe_waist_fit_preflight_20260902_191340
Latest formal run: none
Frozen config SHA256: E6691CB1C7093D8F48D55FC721554B65E1E1E1F6A7104D24663CEACF868B7097
Source artifact: runs/exp030_TGV_2d_effective_phase_20260810_121124/outputs/exp030_effective_phase.h5
Target dataset/bytes: /entry/truth/P_B_true / A020FD6F...BDA40
Current sole issue: run and audit the single frozen formal
Minimum next-read set: section 0, sections 7--9, section 12, formal config
Next executable command: conda run -n tgv_ptycho_sim python scripts/run_exp050_projected_true_probe_waist_fit.py --config configs/experiments/exp050_waist_parametric_fit.yaml --mode formal
```

### 12.2 后续建议（formal 前冻结）

继续 exp050 才能完成：执行上述唯一 formal、artifact/figure audit、tests/lint、状态和治理同步。若 replay或 formal gate失败，
在 exp050 内只记录、定位本次已注册问题，不扩 bounds、不改阈值重跑。

应新开实验：exp050 若闭合，下一步为 exp052 matched raw `P_B_rec` fitting；nuisance 属于 exp054；blind reconstruction、
非周期大画布、detector-direct fitting、noise/真实 calibration、3D physics或 projected-vs-3D model mismatch均不得塞入exp050。

## 13. 2026-09-02：authoritative formal 结果、artifact 审计与实验闭合

本节是在第 1--11 节设计与 gate 冻结、且第 12 节证书写入后追加的唯一 formal 结果。正式配置源文件 SHA256 仍为
`E6691CB1C7093D8F48D55FC721554B65E1E1E1F6A7104D24663CEACF868B7097`；复算第 1--11 节冻结区仍为
`13087 bytes / EDBA033AFD74FD7C6A7AAFF95659A842950F042D7A3A760A9C52C56532095352`。未在结果出现后修改 range、方法、预算、阈值或
状态逻辑，也未产生第二个 formal run。

### 13.1 Formal run 与结论

唯一 authoritative formal 为：

```text
runs/exp050_TGV_2d_projected_true_probe_waist_fit_20260902_192303/
mode: formal
run_state.status: complete
run_state.experiment_status: Passed
run_state.artifacts_validated: true
runtime_seconds: 6.919529200000397
formal metadata Git commit: 47d224ef59d59f372f6c40e7376424f3d81d10ca
```

按第 9 节互斥判定顺序，source identity、operator replay、profile、数值稳定性、独立 estimator 和 artifact gates 全部通过，故
authoritative scientific status 为 `Passed`。其严格含义仅为：

```text
single-parameter true-probe oracle fitting is self-consistent
within the exp030 2D projected working model
```

### 13.2 Source、raw replay 与 determinism

formal 再次从 exp030 authoritative HDF5 的 `/entry/truth/P_B_true` 读取 `(384,384) complex128` raw complex field。formal 内副本与
source 数组逐元素 bitwise 相等，target bytes SHA256 为
`A020FD6FACABC28A3CFE3D05B3A24C7A9284665BB7998D85C69BB305909BDA40`；source 五个文件哈希均与第 3 节注册值一致。source
target 本身没有 `units` attribute，formal 如实记录 `target_dataset_units_attribute_present=false`，只通过 provenance 写明 field
amplitude arbitrary、phase rad，没有伪造 source attribute。

使用公共化后的同一 continuous axisymmetric Fresnel--Hankel compact `T-1` operator，加解析无限平面波 reference，在真实参数处
得到：

| replay control | formal value | frozen maximum | result |
|---|---:|---:|---|
| Cartesian raw complex relative L2 | `1.37837263754e-15` | `1e-12` | Pass |
| radial raw relative L2 | `2.18929767917e-15` | `1e-12` | Pass |
| amplitude relative L2 | `8.61661215048e-16` | `1e-12` | Pass |
| amplitude-weighted phase-sensitive relative L2 | `1.09133981760e-15` | `1e-12` | Pass |
| independent repeat relative L2 | `0`（bitwise exact） | `1e-14` | Pass |

`coarse_A_effective_true_used_as_forward_input=false`；没有 phase/gain alignment、post-result mask、`P_B_rec`、detector data、sample B、scan
或 ePIE。

### 13.3 全局 profile、局部 estimator 与区间语义

primary estimator 是完整 raw-complex profile，而不是从 truth 附近启动一次 optimizer：

- global profile：`32.3--34.3 um`，41 点，`50 nm` step；唯一内点 minimum 为 `33.300 um`，minimum loss
  `1.99905181459e-29`，无其他 registered minimum、plateau 或 boundary solution。
- local fine profile：`33.25--33.35 um`，21 点，`5 nm` step；唯一 minimum 同为 `33.300 um`。第二优点为
  `33.295 um / 1.61591624623e-8`，second-best/replay-floor loss ratio 为 `7.27743980436e7`。
- primary estimate：`D_waist = 33.300000000 um`；相对 simulation truth 的浮点差为 `6.77626357803e-21 m`。
- 报告区间：`[33.2975, 33.3025] um`，宽 `5 nm`。这是冻结 fine grid 的 argmin cell，仅表示本次 profile discretization，明确不是
  uncertainty、resolution、confidence interval 或 detection limit。

固定 8 次迭代的 golden-section 只作独立低成本 cross-check，不控制 primary branch：estimate
`33.2995934691 um`，与 profile 相差 `0.406531 nm`，最终 bracket `[33.2985291572,33.3006577809] um`、宽
`2.128624 nm`；estimate error、method agreement 和 bracket width 均小于冻结 `5 nm` gate。完整 11 次 evaluation、9 个 bracket 状态和
cache index 均已保存。

### 13.4 局部数值控制

两级 centered complex finite difference 使用 `h1=125 nm` 与 `h2=15.625 nm`：

```text
normalized Jacobian h1:       25414.4125356 1/m
normalized Jacobian h2:       25411.6141975 1/m
Jacobian step relative L2:    1.57985191103e-4
h2 one-sided signature:       3.96999217628e-4
signature / replay floor:     2.88020239821e11
loss curvature:               1.29200557123e9 1/m^2
```

相应 frozen gates 分别是 Jacobian norm `>=1e4 1/m`、step relative L2 `<=0.05`、signature/floor `>=1e6`；均有充分 margin。profile
minimum、finite-difference control 和 golden cross-check 指向相同局部结构，没有证据表明 estimate 由径向积分跳变、profile grid
alias 或浮点 floor 产生。

### 13.5 HDF5 与独立 artifact 审计

formal HDF5 为 `outputs/exp050_projected_true_probe_waist_fit.h5`，SHA256
`BA6E18A99E7A21B8BFA571109A01FFB235B800E860E194D9BEBCB01D82AD69BD`。独立只读审计复核了 386 个 HDF5 paths、全部 numeric
dataset finite、JSON/HDF5 status和值一致、source/target hash、target/best/cache/residual identity、76-entry candidate cache 的
global/fine/golden index mapping，以及 replay repeat bitwise identity。保存的 YAML 是 frozen source config 的语义等价重序列化；run
copy SHA256 为 `800FB8B33040FAFFDAA229DCF715751525F746AE983923321ADBEE0D049E019B`。其余 formal artifact hashes 为：

```text
metadata.json: 14C34B9F82ABF409D9003E469F1ED78C6A9EAD6E728EBE5A18C93911A3255BFD
metrics.json:  FF7E407E7EC6462601F0A6F094C7D3AF030ED8B3E083D2242839B7EF4F2E29E4
run_state.json: 4F3097FF1F998001D6060E4D108E5FD1009E47B6544F8B1D021D234DA6BC52B0
```

HDF5 按 `/entry` 并列保存 `config_yaml`、`instrument`、`sample/sample_a`、`truth/P_B_true`、
`truth/D_waist_true_m`、source radial arrays、`reconstruction/waist_fit/{design,source_provenance,replay,candidate_cache,profile,
finite_difference,estimator_crosscheck}`、best raw candidate、raw residual、`metadata` 和 `metrics`。`/entry/data` 为空；不存在
`I_stack`、`scan_positions`、`P_B_rec`、sample B、calibration 或 preprocessing。项目级 HDF5 schema 无变化。

### 13.6 Figures 人工检查

四张 formal PNG 均能打开、hash 与 `run_state.json` 一致、轴和单位可读，且与数值 artifact 相符：

- `exp050_loss_profile.png`：注册全范围和局部 profile 都只有 `33.300 um` 单一内点 minimum；绿色区明确是 5 nm grid cell。
- `exp050_best_fit.png`：target/best raw amplitude 和 phase 肉眼一致；raw amplitude/phase residual 量级约 `1e-14`，未做 alignment。
- `exp050_numerical_controls.png`：两档 complex Jacobian signature 空间结构一致，差值远小于主 signature。
- `exp050_estimator_crosscheck.png`：golden bracket 确定性收缩到 profile minimum 附近，evaluation loss 无隐藏第二 basin。

未发现 blank/corrupt figure、错误轴序、单位误标、裁切遮挡或图与 JSON/HDF5 矛盾。PNG 仅用于人工 QA，未参与 gate。

### 13.7 实际命令、测试与 lint

关键实际命令为：

```powershell
conda run -n tgv_ptycho_sim python scripts/run_exp050_projected_true_probe_waist_fit.py --config configs/experiments/exp050_waist_parametric_fit.yaml --mode formal
conda run -n tgv_ptycho_sim python -m pytest -q tests/test_exp050_projected_true_probe_waist_fit.py
conda run -n tgv_ptycho_sim python -m pytest -q tests/test_exp050_projected_true_probe_waist_fit.py tests/test_exp030_observability.py tests/test_exp051_true_probe_waist_fit.py tests/test_exp051_q8_plateau_interval_fit.py tests/test_exp051_local_differentiability_control.py tests/test_exp053_reconstructed_probe_q8_cell_interval_fit.py
conda run -n tgv_ptycho_sim python -m ruff check src/tgv_ptycho/forward/exp030.py src/tgv_ptycho/inverse/exp050.py scripts/run_exp050_projected_true_probe_waist_fit.py tests/test_exp050_projected_true_probe_waist_fit.py
conda run -n tgv_ptycho_sim python -m pytest -q
```

结果：exp050 scoped `7 passed in 4.24s`；exp030+050+051+053 related regression `45 passed in 131.16s`；scoped Ruff
`All checks passed!`。全量 pytest 为 `360 passed, 12 failed in 258.21s`；12 项全部是任务开始前已登记的 exp040 R10--R14B
frozen-config SHA256 lock mismatch，没有 exp050 或其他新增失败，也未修改任何 exp040 lock。独立 artifact assertion script 使用同一
`tgv_ptycho_sim` 解释器只读运行，结果 `independent_audit=PASS`。

### 13.8 限制、治理与 Git 状态

这是 noiseless、matched、single-parameter、true-probe/oracle、inverse-crime self-consistency。所有其他 geometry/optics 参数固定，且
target 与 candidate 使用同一 exp030 projected operator。它不证明真实 3D TGV、full-wave、model mismatch、实际 calibration/noise/
stage error、detector-direct fit、measurement uncertainty、resolution 或 detection limit；也不能替代 `P_B_rec`、blind B/probe 或
nuisance fitting。exp030 的 2D effective phase 仍只是早期 projected diagnostic。

本任务新增/修改保持 local unstaged；未执行 `git add`、commit、push、PR 或 merge。现有 staged/unstaged/untracked/deleted 用户内容均
保留，runs 继续由 ignore 管理。

### 13.9 快速恢复上下文

```text
Current authoritative section: section 13 formal result and closure
Latest valid run: runs/exp050_TGV_2d_projected_true_probe_waist_fit_20260902_192303
Latest development run: runs/exp050_TGV_2d_projected_true_probe_waist_fit_preflight_20260902_191340
Latest formal run: runs/exp050_TGV_2d_projected_true_probe_waist_fit_20260902_192303
Frozen config SHA256: E6691CB1C7093D8F48D55FC721554B65E1E1E1F6A7104D24663CEACF868B7097
Frozen body: 13087 bytes / EDBA033AFD74FD7C6A7AAFF95659A842950F042D7A3A760A9C52C56532095352
Source artifact: runs/exp030_TGV_2d_effective_phase_20260810_121124/outputs/exp030_effective_phase.h5
Target dataset/bytes: /entry/truth/P_B_true / A020FD6FACABC28A3CFE3D05B3A24C7A9284665BB7998D85C69BB305909BDA40
Current sole issue: none within exp050; reconstructed-probe handoff is exp052
Minimum next-read set: section 0, section 13, section 12 certificate, exp050 formal metrics/run_state
Next executable command: git -c safe.directory=E:/tgv_ptycho_sim status -sb
```

### 13.10 后续建议

继续 exp050：无需新增科学计算或重跑 formal；只有发现 artifact/hash/document inconsistency 时才在真实 EOF 追加纠错章节，不能回改
第 1--13 节或 gate。

新开实验：projected 链的自然下一步是 `exp052` matched raw `P_B_rec` fitting，必须先独立确认 reconstruction source、raw reference 和
operator handoff。多参数 nuisance 属于 `exp054`；blind reconstruction、非周期大画布、detector-direct fitting、noise/真实
calibration、3D/full-wave physics 和 projected-vs-3D model mismatch均应另立实验，不能扩入 exp050。
