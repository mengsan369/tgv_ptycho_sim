# exp055：3D scalar multislice nuisance / multi-parameter identifiability

## 0. 实时状态与阅读顺序

本节是本文唯一允许持续原位更新的实时入口。第 1 节以后全部 append-only；新的预注册、实现、失败、correction、
formal 结果和审计只能追加到真实 EOF，不能回改既有记录。

```text
Scientific status: Failed
Work status: Formal complete / artifacts validated / frozen / P_B_rec stage not authorized
Results available: true
Current authoritative section: section 4
Latest valid run: runs/exp055_TGV_3d_multislice_true_probe_nuisance_identifiability_20260901_193257
Latest preflight run: runs/exp055_TGV_3d_multislice_true_probe_nuisance_identifiability_preflight_20260901_193009
Latest formal run: runs/exp055_TGV_3d_multislice_true_probe_nuisance_identifiability_20260901_193257
Primary current question: 已回答为 Failed；profile/equivalence set 唯一且窄，但四条冻结搜索路径均未到达该 component
Current single primary blocker: 无执行 blocker；scientific failure 是预注册 multistart path inconsistency
Next action: 停止当前 exp055 方向且不启动 P_B_rec；如需研究 optimizer landscape，应另立独立 inverse-method 问题而非改本 formal
reference_validated: false
full_tgv_reference_authorized: false
```

推荐阅读顺序：先读本第 0 节；再读第 1 节的 source audit、方法比较、冻结参数/范围/gates/status/resource
合同；实现后只需继续读后续 authoritative appended section。背景限制见 exp040 第 0/16/19 节、exp042 第 0/30 节、
exp051 第 0/26/27 节和 exp053 第 0/16 节。

## 1. 2026-09-01：true-probe 最小 geometry nuisance 首次完整预注册

### 1.1 本轮目标、上一轮建议与开始时证据

本轮目标是先冻结一个最小、可审计且允许负结果的 Phase-5 问题，再实现和运行；首要目标不是得到 `Passed`。
exp051 第 27 节要求 fixed-parameter oracle 闭合后才讨论 nuisance，exp053 第 16 节已经用 authoritative raw
`P_B_rec` 闭合相应单参数 baseline，因此采用“新开 exp055、但先只用 raw `P_B_true`”的上一轮建议。

先研究完美 `P_B_true` 的原因固定为：如果在完美复场下 `D_waist` 已与 nuisance 退化，B 编码或 blind
reconstruction 不能凭空恢复 candidate family 已丢失的信息；只有 true-probe 闭合而 raw `P_B_rec` 失败时，才有依据把差异归因到
measurement/reconstruction chain。true-probe scientific/artifact contract 闭合前禁止启动 reconstructed stage。

开始时完整读取根目录 `AGENTS.md`，执行
`git -c safe.directory=E:/tgv_ptycho_sim status -sb`，并定向读取 roadmap、exp040 R8/freeze、exp042/051/053
authoritative sections、本文旧全文、exp040 theory/data-format，以及 exp051/053 config、source/candidate/q8
partition/profile/multi-start/runner/tests 相关实现。混合工作区已有 modified、deleted、untracked 内容全部视为用户所有并保留；
本轮只允许新增/修改 exp055 专属文件，保持 unstaged。

authoritative source 已按实际文件而非摘要核对：

```text
exp042 run: runs/exp042_TGV_3d_multislice_probe_reconstruction_dev_20260831_183139
state: complete; artifacts_validated=true
HDF5: outputs/exp042_probe_reconstruction.h5
HDF5 SHA256: C48588FC47EED474CF047219A1DB26BCC9242961A58176CA9CCB879495FFA37D
primary target: /entry/truth/P_B_true, (96,96) complex128
primary target byte SHA256: FA61264AF0D96BF3393EC461E2147992FDE6926D0132B2346090790E40E8EBFD
raw rec path kept separate: /entry/reconstruction/exp053_feedback_control/branches/spectrally_damped_gn_cg/P_B_rec
raw rec byte SHA256: 194C7B950F8DCF2BF94212A6F63270296BC9EC79AE0C04ED557C864F5FE8EB07
source config/metadata/metrics/run-state SHA256:
  59860425...7276 / 5B056A65...7B2B / 195B001C...15A / 75E70687...05C

exp051 oracle run: runs/exp051_TGV_3d_multislice_true_probe_waist_fit_q8_plateau_interval_20260824_161440
state: complete; artifacts_validated=true; Passed
HDF5 SHA256: FD1F5E47300BB8A09329F8E21E1A9A3AFCD84BFEFA3FD9A7AD89705983270B3C
stored raw P_B_true byte SHA256: FA61264A...8EBFD (与 exp042 逐字节同一 truth)

exp053 run: runs/exp053_TGV_3d_multislice_reconstructed_probe_q8_cell_interval_20260901_122210
state: complete; artifacts_validated=true; Passed
HDF5 SHA256: CF1697FC1B36CFCD9647FFC07A4BC68FD02C97ED1C329AE4313C9719467D0C1F
```

因此 primary 输入只允许最新 exp042 HDF5 的 raw truth dataset；禁止 aligned probe、truth-selected copy、raw
`P_B_rec` 或 detector intensity 静默替换。operator identity 固定为 R8 unified q8/finite-B/open/q4 scalar working model，
但本阶段只拟合 B-plane complex probe，不读取或伪造 detector data。

### 1.2 主要矛盾、次要矛盾与参数集审计

唯一主要矛盾是：fixed-q8 对 geometry 参数呈分段常数/非光滑，单参数窄 plateau 不保证加入 nuisance 后
`D_waist` 的投影仍唯一。最多三个次要矛盾是：(1) 参数范围没有真实标定统计；(2) 低维连续参数的全局注入性无法用
单一有限 grid 证明；(3) true-probe 结论必须与 exp040 reference failure 和实际计量 claim 隔离。

formal 结果未知时比较三个最小候选集：

1. `D_waist + tied D_surface + z_waist`：两个 nuisance 直接控制分段线性轴对称孔形，均在既有 sample-A
   config 中有明确 nominal 和 SI 语义；只修改 candidate geometry，不更换传播、材料、B 或 detector operator；预期退化机制是
   surface/waist taper 与 axial waist location 共同保持部分 q8 slice radii。计算量最小且最直接回答“geometry 不再固定”。
2. `D_waist + thickness + z_AB`：会同时改变 slice grid、A 出射边界和 external propagation plane；较小问题无法区分
   geometry nuisance 与 operator/plane 变更，且厚度/距离公差尚无权威来源，故排除。
3. `D_waist + n_glass + global complex gain`：gain 可解析 profile-out，但本 raw loss baseline 明确禁止 phase/scale
   alignment；`n_glass` 还同时改变内部参考传播/载波，现阶段没有冻结 calibration range。采用它会把首轮问题从 minimal
   geometry identifiability 扩成 calibration/operator 研究，故排除。

正式选择候选 1，除 target 外恰好两个 nuisance，不加入第 3 个。`D_top=D_bottom=D_surface` 强制 tied；
thickness、center、materials、illumination、`z_AB` 和所有 sampling/operator 参数保持固定。

### 1.3 范围、参数化和离散合同

nominal 为 `D_waist=20 um, D_surface=30 um, z_waist=50 um`。冻结 working ranges：

```text
D_waist:  [16,24] um（继承 exp051/053 已冻结 fitting bounds）
D_surface:[28,32] um（nominal ±2 um，与 exp040 已使用的 geometry perturbation scale 同量级）
z_waist:  [45,55] um（100-um 厚度中央 nominal 的 ±5-um 最小轴向 working range）
```

后两者明确是 `preregistered working range`，不是经验公差、posterior uncertainty 或实际 calibration accuracy。范围不得在
formal 后缩小或扩张。coarse grid 为 `9 x 5 x 5 = 225` 点，覆盖完整 working box；local grid 为
`9 x 9 x 9 = 729` 点，分别覆盖 `D_waist 19.5--20.5 um @0.125 um`、`D_surface 29.5--30.5 um
@0.125 um`、`z_waist 48--52 um @0.5 um`。两套 grid 均在 formal 前冻结；coarse 用于 distant alias screen，local
用于 profile、连通分量和搜索路径，不得看结果后加点。

q8 fiber 只沿 `D_waist` 在每个 qualifying nuisance pair 上使用 exp051 的解析 partition、最多每侧 256 cells、双侧各
24 次 fixed membership bisection和 `1 pm` bracket；它报告 half-open set resolution，不强行输出 sub-plateau 单点。
float64 source-exact membership 与 analytic breakpoint 若有 ULP 差异必须同时保存，不得用 epsilon 平移边界。

### 1.4 方法比较、primary 方法与 truth 边界

比较的方法不超过三种：

1. **低维 joint grid + profiled connected equivalence set**：能直接表示 q8 plateau、distant alias、离散连通分量和
   `D_waist` 投影；结果可逐点审计。选择为 primary。
2. **bounded continuous derivative-free multi-start 单独优化**：能检查路径依赖，但在 plateau 上的 terminal 点依赖 tie/step，
   且单独使用不展示 equivalence topology；不作为 primary。保留为四角出发、同一 frozen local grid 上的确定性六邻域最陡下降
   path，作为全局/局部非唯一性检查。
3. **local Jacobian/SVD**：对 non-smooth q8 只能描述冻结有限变化方向，不能成为唯一 identifiability gate。只在 nominal
   处用 `0.125/0.125/0.5 um` centered finite changes 构造三列并列归一化后报告 SVD/directions；不进入 status。

primary loss 完全继承 exp051 raw convention：

$$
L(\theta)=\frac{\lVert P_B(\theta)-P_{B,\mathrm{true}}\rVert_2^2}
{\lVert P_{B,\mathrm{true}}\rVert_2^2},\qquad \rho=\sqrt{L}.
$$

mask 是 full native `(96,96)` complex field，所有像素等权，无 phase/scale/spatial alignment。primary equivalence
`rho<=1e-12`，并固定 `1e-13/1e-11` 两档 threshold-stability controls；阈值继承 exp051 已冻结且相对相邻 q8
outside response 至少有 `1e6` separation 的数值合同，不由 exp055 formal 调整。

对每个 local `D_waist`，profile 是对 registered nuisance grid 的最小 raw loss；equivalence nodes按六邻域形成连通分量。
coarse/local 任何相互分离的 qualifying component 都是 non-unique evidence。四个 local-box 角点按固定 lexicographic tie
规则走六邻域 steepest-descent，每支最多 32 moves且使用同一已计算 lattice；不能只选最好的一支。

truth 只允许在 profile/equivalence/search 完成后计算 nominal-to-reported-set distance，字段必须位于
`simulation_evaluation_only`；truth 不参与 grid/range/method/threshold、branch、stop、tie或 status。由于 target 本身来自
nominal simulation，exact source replay是 operator/handoff control，不是用 truth 对拟合结果做 alignment。

### 1.5 Gates、互斥状态、资源和 artifact 合同

numerical/artifact gates 冻结为：source hashes/state/path/flags exact；raw target `(96,96) complex128` finite；nominal replay
与 deterministic repeat `rho<=1e-14`；cache parameter/index/loss/hash exact；三档 threshold component agreement；q8 fiber
外侧 `rho/tau>=1e6`；24-step bracket `<=1 pm`；所有 numeric leaves finite；raw target/best/residual identity exact；外部
JSON 与 HDF5 同义 leaves exact；所有 figures 可 read-back。preflight 只跑冻结的 nominal/repeat/六个 coordinate guards，
只判 correctness/artifact/resource，不产生 scientific status，也不允许用 loss 大小改范围、loss、threshold、grid或方法。

scientific gates 冻结为：qualifying connected component 数不超过 1；四个等合同 paths 都终止于同一 primary component；
equivalence set不触及 local grid边界；profiled `D_waist` 投影是连通 half-open interval，宽度
`0 < width <=0.125 um`，且距全局 fitting bounds 至少 `0.125 um`。`0.125 um` 继承 exp051/053 的既有
inverse accuracy budget，不解释为真实仪器 resolution。没有独立阈值的 SVD condition number 只报告 Diagnostic。

状态按以下顺序互斥：

1. source/provenance/artifact 不完整：`Inconclusive / artifact_operator_handoff_not_closed`；
2. determinism/cache/threshold/fiber/bisection/resource 等 numerical controls 未闭合：
   `Inconclusive / nuisance_numerical_control_not_closed`；
3. 出现多个分离 equivalence components：`Failed / nuisance_equivalence_nonunique`；
4. 四条路径 terminal component 不一致：`Failed / multistart_search_path_inconsistent`；
5. set 触及 local boundary、`D_waist` 投影不连通/为空/过宽或触及全局 bound margin：
   `Failed / waist_profile_not_stably_identifiable`；
6. 上述全部通过：`Passed / minimal_geometry_nuisance_identifiability_passed`。

formal 后禁止修改 status order、bounds、grid、tau、seed/path、budget、loss或阈值。若 finite registered grid 之外的连续
injectivity 仍无证据，限制必须如实写出；不得把 finite screen 提升为全局数学唯一性证明。

HDF5 不改项目级 schema；自然结果写入
`/entry/reconstruction/waist_fit/nuisance_identifiability/true_probe/`，保存 frozen definitions/bounds/units、source
identity/raw `P_B_true`、coarse/local samples、profile、三档 components、四条 path、q8 fibers/bisections、best raw field、
raw residual、cache index/loss/field hashes、controls/gates、local SVD、runtime/memory provenance和
`simulation_evaluation_only`。不写 `I_stack`、scan positions、calibration或 preprocessing；不保存 aligned field。

资源上限冻结为 runtime `600 s`、sampled peak RSS `<1 GiB`、HDF5 `<192 MiB`；candidate full fields不全量写盘，
只保存 raw target/best/residual及每个 cache entry的参数、loss和byte hash。五张图仅供人工审阅，定量数据以 HDF5/JSON为准。

### 1.6 本轮只允许改变的变量、计划文件与冻结证书

本轮 formal 中只允许 `D_waist`、tied `D_surface`、`z_waist` 按冻结 grids/fibers变化。保持 exp042 source、q8 scalar
forward、shape/dx/dz/thickness、center、materials、illumination、`z_AB`、external sampling、loss和 reference flags不变。
计划文件仅为：

```text
configs/experiments/exp055_TGV_3d_multislice_nuisance_identifiability.yaml
src/tgv_ptycho/inverse/exp055.py
scripts/run_exp055_nuisance_identifiability.py
tests/test_exp055_nuisance_identifiability.py
docs/experiment_design/exp055_TGV_3d_multislice_nuisance_identifiability.md
```

config 在任何 exp055 candidate/preflight/formal result 产生前创建并冻结：`7077 bytes`，SHA256
`9DD61506C948DD75D464BE176BA1C786A9885D1D845AA6C830908B5D63F5EC6B`。本文旧 placeholder 在本轮前为
`791 bytes`，SHA256 `D6445D1954EA10630B173D90F7A117909CD06F7CCD3E99645C6B6FD8E3271012`；旧内容只承担
“尚未预注册”的历史职责，本节明确取代其实时状态但不伪造历史运行。

下一步先实现和测试；preflight 后只允许 correctness、接口、artifact 或资源修正，并必须追加 correction，再执行一次对应此冻结合同的
timestamped formal。P_B_true stage 为 Failed/Inconclusive 时停止，不启动 P_B_rec；Passed 且 artifact/numerical/scientific
contract 全闭合时，才评估另追加独立 reconstructed-stage preregistration。

快速恢复上下文：当前 authoritative section 是本第 1 节（预注册，无结果）；latest valid/preflight/formal run均为 none；
root config及 source/raw hash见上；唯一未关闭问题是缺少实现与 evidence run。下一轮最小读取为本文第 0--1 节、冻结 YAML、
exp051 source/candidate与 q8 partition helpers、exp051/053 runner artifact methods；无需重读全部 exp040/042/051/053 历史或扫描 runs。
允许的最小改动仅是上述 exp055 module/runner/test及 correctness-only append；Git staged/commit/push/PR均保持未改变。

## 2. 2026-09-01：实现、targeted/组合回归与 preflight execution lock

### 2.1 本轮目标、上一轮建议与改动前判断

本轮严格采用第 1 节唯一建议：只实现已冻结的 true-probe 三参数 joint profile/equivalence method、source validator、
四起点 lattice path、diagnostic-only SVD、runner/HDF5/figures 和 tests；不生成 formal scientific result。开始证据仍是
exp042 raw truth hash `FA6126...8EBFD` 与 section-1 root config hash `9DD615...5EC6B`。唯一主要矛盾从“无实现”缩小为
“runner artifact contract尚未被真实 preflight闭合”；次要矛盾仅是通用 HDF5 writer不能直接编码 list-of-mapping和 NumPy
Unicode hash arrays，以及 Windows `conda run` 在 Ruff diagnostics 含不可编码字符时发生 GBK wrapper error。

没有采用修改 shared HDF5 schema/writer、缩减 grid 或 candidate evidence 的方案。前两个 artifact 编码问题在 exp055 payload
边界转换为 named groups和 string lists；Ruff改用同一 Conda环境的绝对 Python解释器。它们是 correctness/interface 修正，
没有读取 formal loss landscape，也没有改变 parameter set、bounds、grid、tau、path、budget、threshold或status order。

### 2.2 实际 Changes 与 operator/artifact consistency

实际新增：

```text
src/tgv_ptycho/inverse/exp055.py
scripts/run_exp055_nuisance_identifiability.py
tests/test_exp055_nuisance_identifiability.py
```

inverse module复用 exp051 strict raw-source validator、exp040 `build_scalar_working_model_probe`、exp051 q8 partition/component/
bisection和 raw loss；没有复制 forward。它实现 compact field cache、225-point coarse screen、729-point local joint grid、三档
equivalence mask、六邻域 components、四条 frozen steepest-descent paths、qualifying nuisance fibers、profiled waist projection、
diagnostic finite-change SVD和互斥状态逻辑。full candidate fields只在运行内存 cache中复用；HDF5只写参数/loss/byte hash及
raw target/best/residual，符合 192 MiB contract。

runner强制 root config SHA、timestamped preflight/formal mode、source exact identity、resource sampling、外部 JSON、并列 HDF5、
五图、全树 finite/forbidden-name/read-back和 raw-best-residual exact validator。`/entry/data` 保持空；没有 calibration、
preprocessing、`I_stack`、scan positions或 aligned field。tests覆盖 source/path/第三 nuisance hard fail、真实 nominal exact replay、
synthetic component topology、correctness-only preflight、joint profile/multi-start/status顺序和 tiny runner HDF5 identity。

当前 preflight 前文件锁：

```text
config  7077 bytes  9DD61506C948DD75D464BE176BA1C786A9885D1D845AA6C830908B5D63F5EC6B
module 33528 bytes  6A0C31CF536C970004479B7230FA05678C6E56B952B53826CF6A9A88D02F3309
runner 24680 bytes  94C132F5120C9FA4D0EF9BD155BCC56090BF9B1CEF13D9D80E6FA648587B1979
tests   7925 bytes  0006617A320D62BB969A67E341392569227B965A4B44FD4B1F58F992297AFE66
```

### 2.3 实际命令、失败与验证结果

实际运行：

```powershell
D:\anaconda3\envs\tgv_ptycho_sim\python.exe -m pytest -q tests/test_exp055_nuisance_identifiability.py
# 首次 4 passed, 1 failed：list-of-mapping 无原生 HDF5 dtype
# 第二次 4 passed, 1 failed：NumPy <U64 hash array 无 h5py conversion path
# correctness-only payload 修正后 5 passed in 2.74s

D:\anaconda3\envs\tgv_ptycho_sim\python.exe -m pytest -q tests/test_exp051_q8_plateau_interval_fit.py tests/test_exp053_reconstructed_probe_q8_cell_interval_fit.py tests/test_exp055_nuisance_identifiability.py
# 15 passed in 108.97s

D:\anaconda3\envs\tgv_ptycho_sim\python.exe -m ruff check src/tgv_ptycho/inverse/exp055.py scripts/run_exp055_nuisance_identifiability.py tests/test_exp055_nuisance_identifiability.py
# All checks passed!
```

另一次 `conda run ... ruff` 因 Conda wrapper 的 GBK `UnicodeEncodeError` 未返回 lint diagnostics；随后直接使用登记环境解释器取得
上述有效 Ruff结果。两个 targeted artifact failures均发生在 synthetic tiny writer，未创建项目 runs、未暴露 formal
science，也未绕过失败；最终 runner validator完整通过。

### 2.4 Preflight gate、保持不变项与下一步

现在只授权执行一次第 1 节冻结的七点 correctness-only preflight：nominal、`D_waist ±0.125 um`、
`D_surface ±0.125 um`、`z_waist ±0.5 um`。preflight只要求 source/replay/repeat/finite、HDF5/JSON/cache/figure/resource
闭合；guard loss不进入方法、threshold、formal范围或scientific status。命令固定为：

```powershell
D:\anaconda3\envs\tgv_ptycho_sim\python.exe scripts/run_exp055_nuisance_identifiability.py --config configs/experiments/exp055_TGV_3d_multislice_nuisance_identifiability.yaml --mode preflight
```

若 preflight失败，只允许 correction到 source接口、array shape/dtype、cache/HDF5/figure consistency或资源实现，并在 EOF追加
correction；不得修改科学合同。preflight通过后，先 append 结果和 formal lock，再只运行一次 `--mode formal`。

快速恢复上下文：authoritative section为本第 2 节（实现锁，无scientific result）；latest valid/preflight/formal仍为 none；source、
root config和代码 hashes见第 1/2 节。当前唯一未关闭问题是 preflight artifact；最小读取仅需本文第 0--2 节、当前 exp055
四个实现文件和 source run state。无需重跑 source audit、15-test组合回归或 scoped Ruff。Git staged/commit/push/PR均未改变；
所有 exp055文件保持 local unstaged/untracked。

## 3. 2026-09-01：correctness-only preflight 结果、独立审计与唯一 formal lock

### 3.1 本轮目标、上一轮建议与实际执行

本轮只执行第 2.4 节已冻结的七点 correctness-only preflight；上一轮建议完整采用，没有修改 config/module/runner/test。
开始时的唯一主要矛盾是 source/replay/cache/HDF5/figures/resource 的真实执行证据缺失；guard loss数值明确禁止进入 method、
threshold、formal range或scientific status。实际命令为：

```powershell
D:\anaconda3\envs\tgv_ptycho_sim\python.exe scripts/run_exp055_nuisance_identifiability.py --config configs/experiments/exp055_TGV_3d_multislice_nuisance_identifiability.yaml --mode preflight
```

唯一新 preflight run：

```text
runs/exp055_TGV_3d_multislice_true_probe_nuisance_identifiability_preflight_20260901_193009
status=complete
artifacts_validated=true
preflight_status=PreflightPassed
scientific_status=NotEvaluated
runtime=2.0782465000011143 s
sampled peak RSS=99,983,360 bytes
HDF5 bytes=732,016
```

nominal replay与 deterministic repeat raw relative L2均为 `0`。六个 guards 的 raw relative L2按冻结顺序为
`0.0314146, 0.0266373, 0.0276400, 0.0417049, 0.0146108, 0.0110365`；这些值只证明 candidate调用产生
finite、非静默复用的 fields，不进入科学 gate或后续选择。

### 3.2 Artifact/HDF5/figure 独立审计

runner内置 validator通过后又执行独立只读 traversal。HDF5 `/entry` children精确为
`config_yaml/data/instrument/metadata/metrics/reconstruction/sample/truth`；50 groups、212 datasets，其中116个 numeric
datasets全部 finite；`/entry/data`为空，无 calibration/preprocessing。raw input是 `(96,96) complex128`，byte SHA256
`FA61264A...8EBFD`，与 exp042 authoritative truth逐元素exact；best-raw=residual逐元素exact；cache正好7 entries；
全树没有 aligned、`I_stack`或 scan_positions path。

五张图都通过 hash/read-back/finite检查，并清楚标注 `PRE-FLIGHT ONLY`；PNG不参与计算。artifact hashes：

```text
run config  D2646B02945D09B7547C1A230A6EF77ED6CD10355F96B9DE6E8BBA9FE4FD7D9C
metadata    F8F8DCC7228F75CF3871419E332194D8FBC0D90225F5D3254F1E37DAB1BA55E5
metrics     AB0284CE47D25871F43FAFEFD40F8769BC9621FFC9685A95EA1AC32DCD076E09
run_state   735E45EF8680D4D8855856491F13CD99DC87AEE6289B37EF9F5B33DC3E31643F
HDF5        2BD1D5FAE47371116F5B2E03F77D196A8CC66908710A65971F0B894A9973E379
```

外部 config/metadata/metrics与 HDF5同义字段由 runner逐叶比较无差异；reference flags与 `p_b_rec_used_as_primary_input`
均为 false。没有项目级 schema变化，也没有 aligned field覆盖 raw field。

### 3.3 Formal execution lock 与保持不变项

preflight关闭了唯一 correctness/artifact/resource blocker，且没有产生 scientific result，因此授权一次冻结 formal。正式命令唯一为：

```powershell
D:\anaconda3\envs\tgv_ptycho_sim\python.exe scripts/run_exp055_nuisance_identifiability.py --config configs/experiments/exp055_TGV_3d_multislice_nuisance_identifiability.yaml --mode formal
```

formal继续锁定 section-1 config SHA `9DD615...5EC6B`、section-2 module/runner/test hashes、exp042 source/hash、225+729
grid、三档 tau、四条 path、q8 fiber预算、SVD diagnostic-only、threshold/status order和600 s/1 GiB/192 MiB资源合同。
不得因 preflight guard大小或 formal中间输出改动它们；formal只允许本次一次 timestamped execution。若 numerical blocker出现，
只按已冻结顺序记录 Inconclusive/Failed并停止，不追加 optimizer/grid/seed/threshold sweep。

快速恢复上下文：authoritative section为本第 3 节（preflight result/formal lock，无scientific result）；latest valid formal仍为
none，latest preflight为 `20260901_193009`；source/raw/root config/代码 hashes见第 1--3 节。当前唯一未关闭问题是 formal
scientific evidence。下一轮只需运行上述命令、读 run state/metrics/HDF5/五图并append authoritative result；不需要重复测试、
Ruff、preflight或 source audit。Git staged/commit/push/PR均未改变，历史 run未覆盖或删除。

## 4. 2026-09-01：唯一 formal、独立 artifact audit 与 authoritative Failed 结论

### 4.1 本轮目标、上一轮建议、开始证据与实际 Changes

本轮唯一目标是执行第 3.3 节锁定的 formal、独立审计并按第 1.5 节互斥顺序判定；上一轮“只运行一次、不看结果改参”
完整采用。开始证据为通过的 preflight `20260901_193009`、root config SHA `9DD615...5EC6B`、source raw hash
`FA6126...8EBFD`及 section-2代码锁。formal 前唯一主要矛盾是 scientific evidence缺失；没有次要 blocker。

本轮 science/runtime中唯一允许变化的仍是 frozen grids/fibers中的 `D_waist`、tied `D_surface`、`z_waist`；
source/operator/thickness/materials/illumination/grid/loss/tau/paths/budget/threshold/status order全部保持不变。实际没有修改
config/module/runner/tests，也没有启动 `P_B_rec`、blind B、detector fitting、noise、新 B、Gaussian spot、large canvas或新物理。
唯一 formal命令为：

```powershell
D:\anaconda3\envs\tgv_ptycho_sim\python.exe scripts/run_exp055_nuisance_identifiability.py --config configs/experiments/exp055_TGV_3d_multislice_nuisance_identifiability.yaml --mode formal
```

唯一 formal run：

```text
runs/exp055_TGV_3d_multislice_true_probe_nuisance_identifiability_20260901_193257
status=complete
artifacts_validated=true
experiment_status=Failed
interpretation=multistart_search_path_inconsistent
runtime=257.61678640000173 s
sampled peak RSS=269,029,376 bytes
HDF5 bytes=998,978
```

### 4.2 Primary profile/equivalence 结果与 q8 fiber

225-point full working-box coarse screen与729-point local joint grid全部执行；包含 q8 fiber/bisection后 compact cache共有1008个
unique candidates。best candidate恰为 nominal `(20,30,50) um`，raw loss `0`，field byte hash与 target相同。
source replay和 deterministic repeat均为 `0`。

三档 `rho<=1e-13/1e-12/1e-11` local mask逐点相同；local qualifying component只有一个 node/index
`[4,4,4] = (20,30,50) um`。coarse screen没有额外 qualifying component，因此
`local_component_count=1`、`external_coarse_component_count=0`、`global_screened_component_count=1`；component不触及
local grid boundary。冻结 finite screen在 registered lattice上没有观察到 nuisance-induced disconnected equivalence alias。

该唯一 nuisance pair上的 q8 `D_waist` fiber与 exp051 oracle一致：

```text
projected interval: [19.999972701052886, 20.000143340468625) um
width: 0.17063941574105138 nm
connected: true
fitting-bound margin: 3.999856659531374 um
left/right immediate outside rho/tau: 7.2941381e6 / 7.4367482e7
maximum 24-step bisection bracket width: 8.5177633e-6 pm
truth-to-interval distance (simulation evaluation only): 0
```

因此 source、determinism、threshold stability、q8 fiber、bisection、screened uniqueness、profiled interval和全部 resource gates均
通过。该 interval只表示 fixed-q8/operator set resolution；不是实际 resolution、uncertainty或 detection limit。

### 4.3 四起点路径、primary failure 与 local diagnostic

四条预注册、同一 local lattice/32-move cap/lexicographic tie的六邻域最陡下降路径均没有到达 qualifying component：

| branch | terminal `(D_waist,D_surface,z_waist)` [um] | moves | terminal raw loss | qualifies |
|---|---:|---:|---:|:---:|
| start_00 | `(19.5,29.5,48.0)` | 0 | `8.7076921e-4` | no |
| start_01 | `(19.5,30.25,51.5)` | 3 | `5.8666431e-4` | no |
| start_02 | `(20.375,29.625,51.5)` | 3 | `5.9928041e-4` | no |
| start_03 | `(20.25,30.5,48.0)` | 2 | `1.4848296e-3` | no |

除 start_00 在起点即为六邻域局部最小，其余路径loss单调下降后停在不同 non-qualifying local minima；trajectory/cache/index
逐点exact。由此 `multistart_common_component_pass=false`。状态顺序先确认 numerical controls和screened uniqueness均为 true，
再在 multi-start gate失败，因此 authoritative状态必须是 **Failed / multistart_search_path_inconsistent**，不能因为 exact
best或窄 interval改写为 Passed。

这个负结果的精确含义是：registered exhaustive screen显示单一窄 set，但冻结的最小局部搜索路径对初始化不稳健；它不是
`nuisance_equivalence_nonunique`，也不能证明 continuum上没有grid间退化。diagnostic-only normalized singular values为
`[1.1986147164, 0.9975848854, 0.7537553701]`、condition number `1.5901906161`；它没有进入status，且不能覆盖
non-smooth/global path evidence。

### 4.4 HDF5/JSON/cache/figure 独立审计

独立 traversal得到87 groups、424 datasets，其中313个 numeric datasets全部 finite。`/entry` children仍精确为
`config_yaml/data/instrument/metadata/metrics/reconstruction/sample/truth`；`/entry/data`为空，无伪 detector、calibration或
preprocessing。raw `(96,96) complex128` input hash `FA6126...8EBFD`与 exp042逐字节exact；best也与source exact，
`best-raw=residual` exact；无 aligned、`I_stack`或 scan_positions path。

1008-entry cache的 coarse/local parameter-index-loss mapping、每条 trajectory visited-index/loss、best-cache mapping、64-character
field hashes和左右各24-step bisection均逐项exact。external config与 embedded `config_yaml` normalized exact；external/HDF5
status、interpretation和 interval关键字段exact，runner的全量 JSON/HDF5逐叶 validator亦通过。

五张图完成 hash/read-back/人工目视审计：global/local profile清楚显示 nominal sharp minimum和半开 interval；nuisance pair map显示
唯一 central registered cell；four-start图清楚显示三条短下降与一个零步停滞；SVD图明确标注 Diagnostic-only。所有图标签/单位可读、
无截断或损坏；zero loss以 float tiny作log display，定量判定仍来自HDF5 raw zero。artifact hashes：

```text
run config  D2646B02945D09B7547C1A230A6EF77ED6CD10355F96B9DE6E8BBA9FE4FD7D9C
metadata    00E8F4794A1B17EBB573DAB32D51A70083BF8D0E2EB03E5EB5C2D3A63658BBFC
metrics     7A16809CD4E92765586C90B6C6CBC6633376A8D7172C46349489EBA8793E040E
run_state   5264D15F78608B15F7401B8AFA22DC4FE5D2513AAAC872B3DCF54F07A27A7606
HDF5        031C0950CF5FCF836242F41649517E3CCE793BC9D72DF1706ACA404E72F41B1B
```

### 4.5 Tests、Ruff、失败边界与阶段结论

实际验证记录：

```powershell
D:\anaconda3\envs\tgv_ptycho_sim\python.exe -m pytest -q tests/test_exp055_nuisance_identifiability.py
# 最终 5 passed in 2.74s；此前两次各 4 passed, 1 failed，均为 formal 前 HDF5编码 correctness问题

D:\anaconda3\envs\tgv_ptycho_sim\python.exe -m pytest -q tests/test_exp051_q8_plateau_interval_fit.py tests/test_exp053_reconstructed_probe_q8_cell_interval_fit.py tests/test_exp055_nuisance_identifiability.py
# 15 passed in 108.97s

D:\anaconda3\envs\tgv_ptycho_sim\python.exe -m ruff check src/tgv_ptycho/inverse/exp055.py scripts/run_exp055_nuisance_identifiability.py tests/test_exp055_nuisance_identifiability.py
# All checks passed!

D:\anaconda3\envs\tgv_ptycho_sim\python.exe -m pytest -q
# 353 passed, 12 failed in 263.76s
```

全量12项失败精确仍为既有 exp040 frozen-config SHA256 locks：R10 Stage-A两项、R10 Stage-B、R10 Stage-B preflight、
R11、R11 preflight、R12、R13、R14、R14A和R14B两项；没有 exp055、exp051/053、shared forward或IO新增失败。
没有修改这些无关历史locks。

authoritative scientific status为 **Failed**，Work status为 `Formal complete / artifacts validated / frozen`。这只说明 selected
exp040 scalar working model、axisymmetric/centered/noiseless/matched/fixed-q8 raw true-probe、两个 working-range geometry
nuisance和登记 grids/path下的数值结果；不表示真实3D电磁准确性、真实计量精度、resolution/detection limit或production fitter。
`reference_validated=false`、`full_tgv_reference_authorized=false`持续成立。

### 4.6 下一步、停止条件与快速恢复上下文

true-probe stage因预注册 multi-start gate **Failed**，故不得启动 raw `P_B_rec` 第二阶段，也无需返回 exp042：失败发生在完美
source的 inverse landscape/path robustness，不是 reconstruction chain。当前 evidence也不要求Gaussian illumination、large aperiodic B或
新 forward operator，因此不新开 exp04x。按本任务停止规则，下一步归类为 **停止当前 exp055 方向**；不得在本 experiment内
更换 optimizer、追加 budget/starts/grid、seed chasing或放宽 gate。若未来独立研究“全局 profile唯一但局部搜索易陷”的算法问题，
应作为另一个明确预注册的 inverse-method experiment，而不是把本 Failed formal重跑成 Passed。

快速恢复上下文：authoritative section为本第4节；latest valid/formal run为 `20260901_193257`，latest preflight为
`20260901_193009`；root config SHA `9DD615...5EC6B`，source run/HDF5/raw path/hash为 exp042 `20260831_183139` /
`C48588...37D` / `/entry/truth/P_B_true` / `FA6126...8EBFD`。当前唯一未关闭科学问题是“其他、另行预注册的全局 inverse
method能否在不改数据/forward下稳定到达该唯一 registered component”；它不再属于本 frozen formal。下一轮最小读取为本文第0/4节、
formal run state/metrics和HDF5 paths；不需要重复读取 exp040全文、重跑 source/preflight/formal、targeted/组合/full tests或Ruff。
下一轮允许的最小改动仅是治理性状态同步或独立新实验预注册；Git staged/commit/push/PR均未改变，所有代码/config/doc仍local
unstaged/untracked，历史run未覆盖或删除。

本 authoritative result 同时预登记一个不改变科学结果的治理性 Change：把 `AGENTS.md` 当前阶段/测试基线和
`docs/theory_notes/roadmap.md` Phase-5状态最小同步为本 Failed/stop/P_B_rec-not-authorized结论；不修改其他阶段、阈值、代码、runs
或 HDF5 schema。该同步完成后只追加验证记录，不回改本节 scientific evidence。

## 5. 2026-09-01：post-result 治理状态同步验证

按第 4.6 节预登记范围，只更新 `AGENTS.md` 的 exp055/current-test两条状态和 roadmap 的 Phase-5 current status、exp055表格行、
后续建议段。没有修改其他实验状态、科学阈值、代码/config/tests/runs或HDF5 schema；authoritative scientific section仍是第4节，
状态仍为 `Failed / multistart_search_path_inconsistent`，`P_B_rec`仍未授权。

同步后 hashes：`AGENTS.md` 为
`A14B410AFC28F3BD18C063BB9ED0A8DF4C004342FDCAFD2F08E4DC31C5CE9B82`，roadmap为
`01EBBE1512945FA9CA7CE8F7E0CD0447775A9898131F9205072F0F75BD04E22C`。本节追加前 exp055文档为
`34841 bytes`、SHA256 `7C91FAB7FD0B287939F6CCCC1EDE38C9679D1940BBFE247029F821B16BA178CC`；第1--4节未回改。

最终 `git status -sb` 确认本轮新增 exp055 config/module/runner/test/doc保持 untracked，AGENTS/roadmap为 local unstaged
modified；cached/staged为空。未执行 `git add`、commit、push、PR、merge或branch操作；用户既有 modified/deleted/untracked
内容均未回退、覆盖、删除或暂存，preflight/formal runs由 ignore保留且未覆盖历史run。
