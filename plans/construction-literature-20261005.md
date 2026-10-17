# 构型创新文献消化（2026-10-05，#9 转入继续研究的第一步）

> 目的：#9 证明"参数化构型调不掉 2 万档的结构性方差与成本"后，按用户指令
> 查文献找原理级方法。只记影响设计决策的结论。

## 一、读到的文献与核心方法

**1. Gârleanu-Pedersen 2013（JF）"Dynamic Trading with Predictable Returns and
Transaction Costs"**（[NBER WP15205](https://ideas.repec.org/p/nbr/nberwo/15205.html)，850+ 引）
- 核心结果：可预测收益 + 线性交易成本下，**最优交易 = 向目标的部分调整**
  （partial adjustment），二次成本下等价于**目标周围的 no-trade region**——
  只有当组合偏离目标足够远（偏离的 alpha 收益 > 边际成本）才交易。
- ⇒ 对 fterm-A：K_hold 的"保留带 ×1.5"是它的粗糙特例。原理化版本 =
  **成本门槛替换规则**：挑战者要替换持仓票，其秩优势必须超过往返成本门槛
  （佣金地板摊薄 + 印花税 0.05% + 过户费，实测约 0.13% 往返）——不是任意带宽，
  而是成本标定的带宽。这是链 #11 的第一候选。

**2. 信号平滑（JPM 传统方法，[转述来源](https://pm-research.com)）**
- 对预测秩做 **EMA 平滑**（decay 可调）再排名——把"每周全重排"变成"渐进重排"，
  换手从结构上下降，且不牺牲信号对慢信息的覆盖（#7：IC 随视界上升 = 慢信息
  真实存在，平滑有理论支撑）。
- ⇒ 链 #11 第二候选：**rank-EMA**（α ∈ {0.3, 0.5} 预声明）。

**3. 带状/滞回再平衡（[arXiv 2026 动量门控层级法](https://arxiv.org)等）**
- 成分股轻微越界不触发调仓（hysteresis）——指数编制的成熟实践。
- ⇒ 已被 K_hold 覆盖（判负），但与 1/2 组合的形态未测。

**4. Fieberg et al.（ML 截面选股，49+ 引）**
- 换手率 0-200% 连续谱上的成本敏感性测量框架——我们 C4 门的经验校准参考。

## 二、创新推导（链 #11 候选，全部预声明后才跑）

| 候选 | 文献依据 | 机制 | 与 #9 被否变体的区别 |
|---|---|---|---|
| **C_hurdle** | Gârleanu-Pedersen no-trade region | 挑战者秩优势 > 成本门槛才换仓（门槛 = 实测往返成本/离散度标定） | K_hold 是任意带宽；C_hurdle 是**成本标定**带宽 |
| **C_ema** | JPM 信号平滑 | pred_rank 的 EMA（α=0.3/0.5）后再排名 | #9 变体全部在**原始秩**上操作；C_ema 平滑**信号本身** |
| **C_hurdle_ema** | 1+2 叠加 | 平滑秩 + 成本门槛 | 交互检验 |

前置依赖：链 #10 的档位读数（若 20 万档过线，构型创新在 20 万档上做；
若不过，构型创新回到 2 万档独立检验——两条路都不堵）。

## 三、来源
- Gârleanu & Pedersen 2013, JF 68(6)（NBER WP15205）
- JPM 信号平滑传统（EMA on signal）
- arXiv 2026 momentum-gated hierarchical rebalancing（hysteresis 实践）
- Fieberg et al. ML cross-sectional equity（换手成本敏感性框架）
