# exp040 理论：sample-B support sensitivity 与 calibration 边界

本文只记录有限 support 假设的通用敏感性分析。具体假设族、阈值、运行结果和实验判定由
`exp040` 实验设计文档记录。

## 1. Sensitivity analysis 不等于 calibration

当样品 B 的有效编码面积、边缘过渡和 complex transmission 未被测量时，simulation 不能识别真实 support。
改变一组合理的 support hypotheses，只能回答 forward prediction 对这些假设有多敏感，不能从中挑选一个
使误差最小的 case 并称其为“标定值”。

Calibration 需要独立观测约束未知参数；sensitivity envelope 只是 conditional model comparison。

## 2. 保持同一内部 realization

比较不同 support 时，编码区内部应保持同一连续或 cell-wise realization。对于 piecewise-constant phase
object，可以从同一组 canonical phase cells 做居中裁剪或扩展，避免每个 case 重新抽取随机对象。

若扩展区域没有实测信息，周期复制只是一种虚拟 hypothesis，不代表发现了新的真实编码区域。其输出只能
用来测试结论的稳健性，不能作为 specimen-specific truth。

## 3. Phase taper 的一种参数化

对 phase-only transmission，可写成

$$
B_{\mathrm{fin}}(x,y)=\exp[iw(x,y)\phi(x,y)],
\qquad 0\le w(x,y)\le1.
$$

$w$ 可取由边缘向内部平滑上升的 separable raised-cosine。这样 support 外 $w=0$、transmission 为 1，
内部 $w=1$、恢复原 phase；同时保持 unit modulus。该参数化表示相位调制逐渐消失，不应与 amplitude
apodization 或真实材料混合层混为一谈。

若真实样品可能吸收，则更一般地需要

$$
B(x,y)=A(x,y)\exp[i\phi(x,y)],
$$

并分别标定 amplitude 与 phase edge profile。

## 4. Sensitivity envelope 的解释

设 nominal hypothesis 的 detector prediction 为 $I_0$，第 $k$ 个 hypothesis 为 $I_k$。可以定义

$$
s_k=\frac{\|I_k-I_0\|_2}
{\max(\|I_0\|_2,\epsilon)}.
$$

$\{s_k\}$ 的范围描述选定假设族内的 model sensitivity。它不是统计 error bar，也不是概率分布；除非已经
为参数定义概率模型并完成不确定性传播，否则不能把它与其他 relative errors 平方相加、相减或解释为
confidence interval。

若所有合理 hypotheses 都保留同一方向的定性结论，可以说该结论对该 envelope 稳健；若结论随 hypotheses
改变，则说明需要收窄先验或取得真实标定数据。

## 5. 真实 B 标定通常需要的量

至少应考虑：

- 有效编码区域及其相对 illumination/scan 的位置；
- 边缘 transition/taper；
- complex transmission 的 amplitude 与 phase；
- exterior/substrate transmission；
- cell size、制造误差、缺陷与非周期性；
- 与 forward wavelength、polarization 和入射角相匹配的响应。

公开数据可以提供材料光学常数、制造方法或典型样品图样，但通常不能替代本项目具体 specimen 的 support、
edge profile 和 complex transmission 标定。
