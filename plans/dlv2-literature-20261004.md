# DL-v2 文献消化（2026-10-04，深度研究前的调研纪要）

> 目的：在制作 Final 的深度学习模型 v2 之前，把前人研究的经验吸收进来。
> 来源检索于 2026-10-04；只记录影响设计决策的结论。链 #7 已证：树臂
> IC +0.050 vs TriHeadNet-v1 +0.014——DL-v2 必须解释"差距从哪来、补什么"。

## 一、读过的文献与核心结论

**1. Gu-Kelly-Xiu 2020（RFS）"Empirical Asset Pricing via Machine Learning"**
- NN（3 层深）与树集成同列最优，超额夏普约为线性模型 2 倍；增益来源 =
  **特征间非线性交互 + 集成平均**（"generalization by ensembling"）。
- 主特征 = 动量/短期反转/流动性/波动/规模，且效应**非线性**（如规模的 U 形）。
- 方法论：每月截面、特征逐期标准化、5+ 种子集成、验证集早停。
- ⇒ 对 DL-v2：**集成是增益的一半来源**（我们已有多种子纪律，保留并加强）；
  非线性要喂给模型的是**截面秩/白化特征**，不是原始价格。

**2. Gorishniy et al. 2021/2022（NeurIPS）"Revisiting Deep Learning Models for
Tabular Data" + "On Embeddings for Numerical Features"**
- 表格数据上 GBDT 常胜 DL（与我们 L 臂 vs TriHeadNet-v1 的实测同构）；
- DL 追平/反超的条件：**ResNet 式 MLP 或 FT-Transformer + 数值特征分桶
  embedding**（把连续特征分桶后查表，给网络非线性基数自由度）+ 认真正则 +
  大集成；单改超参不够。
- ⇒ 对 DL-v2：**数值 embedding 是 2022 年文献里最重要的单点创新**——
  我们的 25 个表格特征正是它的用武之地；架构选 ResNet-MLP（比 FT-Transformer
  在中小样本上更稳，Transformer 列为对照臂）。

**3. QPLUM 2019 "First Principles: Cross-Sectional Stock Returns"**
- 截面特征喂 NN 前必须**白化**（rank 后过 inverse erf，即 Rank-Gauss）；
- 截面切片 ergodic：可按(日期×股票)置换增广；ensemble of ensembles。
- ⇒ 对 DL-v2：**Rank-Gauss 白化**为输入标准口径（替代 v1 的全局 z-score——
  v1 把原始量纲特征直接喂 GRU 是明确的设计错误）。

**4. Leippold-Wang-Zhou 2022（JFE）"Machine Learning in the Chinese Stock Market"**
- A 股：短视距可预测性强；**短期反转而非动量**（散户结构）；流动性/波动
  是头部分组特征；卖空约束使异象持续；**交易成本后利润大幅缩水**。
- ⇒ 与我们极性表（rev 正极性）、构型（2 万成本地板）互证；DL-v2 的标签
  与特征集沿用现面板（25 特征已含反转/流动性/波动/规模四族）。

**5. López de Prado 2018（净化/embargo）**——管线已有，保留。

## 二、糟粕清单（明确不采用）

- **逐票时序窗架构**（LSTM/TCN/PatchTST 式）用于截面选股——v1 的失败主因，
  文献亦不支持（时序模型适合单序列预测，不是横截面排序）；
- **端到端原始价格输入**（不做截面白化/秩化）——QPLUM 与 G BX 均不这么做；
- **大 Transformer 直接上小样本表格**——Gorishniy：调参敏感、中小样本不稳定
  （只作对照臂，不作主臂）；
- **单种子单快照读数**——GKX 的集成结论 + 我们自己的种子稳健性铁律；
- GAN/RL 端到端组合管理——复现性差，与本项目预注册纪律冲突。

## 三、DL-v2 设计推导（据此写预注册）

| v1 的问题 | 文献依据 | v2 的对应设计 |
|---|---|---|
| 逐票时序窗（60×25 GRU） | GKX/QPLUM：截面排序要用截面表示 | **逐日截面行模型**：特征行 = (date, code) 的 25 列，日截面独立打分 |
| 全局 z-score 输入 | QPLUM：Rank-Gauss 白化 | **截面秩 → erf⁻¹ 白化**（逐交易日、PIT） |
| 原始量纲直喂 | Gorishniy 2022：数值 embedding | **数值分桶 embedding**（每特征 16 桶可学习查表）作主臂组件 |
| 采样密度 20/日 | 全截面是树臂优势来源 | **全截面训练**（GPU）/ 预声明 CPU 降级路径（每日 cap 200） |
| 单点集成弱 | GKX：ensembling = 一半增益 | **5 种子集成**，双种子集 ×2 |
| 标签 h=5 单一 | 链 #7：IC 随 h 升 | 主标签 h=5（同管线可比），h=10 预声明对照臂 |

## 四、来源

- Gu-Kelly-Xiu 2020: [RFS](https://academic.oup.com/rfs/article-abstract/33/5/2223/5717836) /
  [PDF](https://bryandkelly.github.io/papers/MachineLearning_RFS_Feb2020.pdf) / [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3159577)
- Gorishniy et al. 2021: [arXiv:2106.11959](https://arxiv.org/abs/2106.11959)；
  2022 数值 embedding: [arXiv:2203.05556](https://arxiv.org/abs/2203.05556)；
  Shwartz-Ziv & Armon 2022: [arXiv:2206.13580](https://arxiv.org/abs/2206.13580)
- Grinsztajn et al. 2022 "Why do tree-based models still outperform deep learning
  on tabular data?"
- QPLUM 2019: [First Principles](https://quantresearch.org/First_Principles_Cross-Sectional_Stock_Returns.pdf)
- Leippold-Wang-Zhou 2022（JFE）"Machine Learning in the Chinese Stock Market"
