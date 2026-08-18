# exp040 R14：Helmholtz shifted-Laplacian 与两级域分解

本文只记录可独立于某次实验运行成立的数学关系。R14 的离散参数、solver case、阈值、资源上限和运行结果记录在
`docs/experiment_design/exp040_TGV_3d_multislice_forward.md`。

## 1. Helmholtz 线性系统为什么难解

频域标量波动方程经有限元或有限差分离散后可写成

```text
A u = f,
A = K - k² M
```

（整体符号随 weak-form 约定改变，但谱性质相同）。当波数增大时，`A` 是强不定矩阵；离散谱会靠近原点，传播模态
又使误差具有全局性。因此，仅让网格足够精确并不能自动保证 Krylov solver 可扩展。反过来，代数残差很小也只能说明
离散方程被解好，不能证明离散场已逼近连续物理解。

GMRES 使用预条件残差构造 Krylov basis，但最终应另外计算

```text
||A u_j - f|| / ||f||
```

作为 true residual。若只看 callback 给出的 preconditioned residual，非正规矩阵和较强预条件器都可能让它与真实残差
出现明显差异。

## 2. Complex shifted Laplacian

常用的 Helmholtz 预条件 operator 是在 mass term 中加入吸收：

```text
A_shift = K - (1 + i beta) k² M,
beta > 0.
```

复 shift 把靠近实轴的困难谱移入复平面，使 multigrid、ILU 或局部直接分解更稳定；随后以 `A_shift^{-1}` 的近似作用
预条件原始无吸收系统 `A`。`beta` 有基本折衷：吸收太弱时 shifted system 仍难解，吸收太强时它与原系统的相似性下降，
外层 GMRES iterations 会增加。

若用完整全局 LU 精确实现 `A_shift^{-1}`，外层迭代可能很少，但其 fill-in 与内存复杂度仍是原来的 sparse-direct
瓶颈。因而“shifted-Laplacian preconditioner”是否可扩展，取决于内部 inverse 是 bounded-fill、multilevel 或域分解
近似，而不是取决于外层 iteration count 单独有多小。

经典讨论见 Y. A. Erlangga, C. W. Oosterlee and C. Vuik, *A Novel Multigrid Based Preconditioner for
Heterogeneous Helmholtz Problems*, SIAM Journal on Scientific Computing 27(4), 2006。

## 3. Restricted additive Schwarz

把全局 unknowns 分成互不重叠的 core subdomains，并给每个 core 加 overlap，可定义局部 restriction `R_i`、
core-only restriction `R_i^0` 和局部矩阵

```text
A_i = R_i A_shift R_i^T.
```

一层 restricted additive Schwarz (RAS) inverse 为

```text
M_RAS^-1 = sum_i (R_i^0)^T A_i^-1 R_i.
```

Overlap 让相邻 subdomains 交换近场信息；只把 correction 写回 nonoverlapping core，可避免 overlap 区被重复相加。
局部 factor 的尺寸被 block size 限定，因此不会退化成单个全局 LU。不过一层方法只能通过相邻 subdomains 逐步传递
长距离误差，domain 数增加时 iteration count 往往增长。

## 4. 两级 correction 与 coarse-space 选择

令 `Z` 的 columns 张成 coarse space，Galerkin coarse operator 为

```text
E = Z^H A_shift Z.
```

coarse correction 为

```text
u_c = Z E^-1 Z^H r.
```

它可与 RAS 加性或乘性组合。乘性组合先作 coarse correction，再对剩余 residual 作 local correction，因此避免 local
solver 重复处理已被 coarse space 表示的分量。

每个 subdomain 一个常数 basis 是最低成本的两级空间，适合检查“缺少全局低频通信”是否是主要问题。但 Helmholtz
误差本身具有振荡性；当每个 subdomain 跨越多个波长时，piecewise constants 通常不能表示传播方向和局部近零能量
modes。此时 iteration growth 并不说明域分解思路本身失败，而是说明 coarse space 不具备 wave-number robustness。
常见增强包括：

- plane-wave 或 wave-ray coarse bases；
- 基于局部广义特征问题的 spectral/GenEO-type bases；
- sweeping、polarized traces 或 optimized transmission conditions；
- multigrid 中针对波动相位设计的 coarse operators。

这些增强会改变算法假设、setup 成本和 memory model，必须作为新的 comparator 明确固定，不能在同一结果之后逐项加入
直到曲线变好。关于高频 Helmholtz 域分解与吸收的分析，可参见 I. G. Graham, E. A. Spence and E. Vainikko,
*Domain Decomposition Preconditioning for High-Frequency Helmholtz Problems with Absorption*, Mathematics of
Computation 86, 2017；域分解背景见 V. Dolean, P. Jolivet and F. Nataf, *An Introduction to Domain Decomposition
Methods*, SIAM, 2015。

## 5. Storage scaling 与 peak-memory projection

可扩展性需要同时记录：

```text
original sparse matrix arrays
shifted matrix retained by the preconditioner
local/ILU factors and permutations
coarse basis and coarse factor
restart Krylov basis vectors
```

对固定 stencil 和固定 bounded fill，非 coarse 部分通常可先按 unknown count 作线性外推。二维 coarse direct factor 的
fill 往往超线性，不能只按 coarse dimension 线性估算；应从多个 domain sizes 的实际 factor storage 拟合增长，并保留
至少不低于最大实测 scale 的线性延伸这一保守下限。最终还需要 safety factor 覆盖 Python objects、temporary work arrays、
allocator fragmentation 和尚未显式计入的 indices。

OS 报告的 process peak RSS 是重要的 hard control，但它包含进程此前分配的内存，也不等同于某个 solver 对象的净
storage。因而实际 RSS 与按 retained arrays 建立的可解释 memory model 应共同报告。

## 6. Analytic manufactured field 的作用边界

若给定解析场 `u_exact` 并据此构造 source

```text
f = A u_exact,
```

就可以把 iterative error、finite-element approximation error 和 PML coordinate stretch 分开测量。在 material mass
jump 上使用同一 analytic field，可验证未对齐 interface 的 quadrature 与 assembly；它并不复现无源散射场在真实
interface 上的全部 regularity。因此这类 benchmark 能证明 solver/离散实现达到必要条件，但不能替代 canonical
geometry 的最终 domain convergence 与跨模型比较。
