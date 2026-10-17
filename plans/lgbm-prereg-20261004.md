# 判据链 #4 预注册 · Phase 3 LightGBM 截面臂（2026-10-04 冻结，跑数前定稿）

> 目的：一次性买断"树模型 vs 这套日频特征面板"的问题（路线图 Phase 3）。
> 老盘同构：LightGBM regression、截面打分；**超参 = 老盘 F47 六格结清的现役
> 局部最优整体**（`num_leaves=31 / min_child_samples=200 / lr=0.05 /
> feature_fraction=0.8 / bagging=0.8+freq=1 / lambda_l2=1.0 / 50 轮无早停`），
> 不做网格搜（老盘 F47 教训：超参线已结清）。

## 一、臂与口径

- **L 臂**：LightGBM 回归（objective=regression），特征 = bundle.feature_cols
  全截面（ eligible ∩ fwd_ret 非缺，LightGBM 原生吃 NaN）；标签 = fwd_ret
  （t+1 开盘 → t+1+h 开盘，h=5，hfq，与全链同口径）。
  **冻结偏差声明**：v1 不做行业中性化标签（老盘月频口径的日频移植留待
  v2，先测原始版）。
- **折**：树臂自己的四折年界（不与 #1 折界对齐，声明为构造差异）——
  训练 expanding，测试段 [2019,2021) / [2021,2023) / [2023,2025) /
  [2025,面板末]；OOS = 2019-01-01 → 面板末。每折每种子集训练后对测试段
  全截面打分。
- **双种子集**：{42,7} 与 {123,2024}（老盘 4 seed 拆两组，各 2 种子预测取
  均值=该组集成）。
- **组合层**：S1 基线在**同窗口（2019 起）重跑**配对；L 臂 top_n=2、
  退出带 keep=4（按预测秩）、无 p 门（回归分无量纲门，冻结此差异）。
- **环境**：fterm venv 实测 lightgbm 4.7.0 可用（2026-10-04）。

## 二、判据（冻结；与 #3 同门限，基线换 S1@2019 窗口）

- C2a：vs S1 周配对差 t ≥ 2 且均值 > 0；C2b：总净收益 > S1。
- C3：MDD ≥ S1 − 2pp；C4：总成本 ≤ S1 × 1.25。
- 信息层（入场券，不设门）：OOS 逐日截面 RankIC（pred vs fwd_ret，≥5 只计）。
- adoption_ready = C2a∧C2b∧C3∧C4 在**两套种子集**上都成立。

## 三、产物

预测流缓存 parquet + verdict JSON/txt → `data/reports/lgbm_chain_<stamp>.*`；
证据包 `experiments/lgbm_chain_20261004/`；docs/03 §18 登记。判决词由用户填写。
