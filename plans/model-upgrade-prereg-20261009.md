# 模型升级总预注册 · fterm 信号层 v3（2026-10-09）

> 依据：`docs/audit-model-20261009.md`（深度审计）+ `docs/03-model-iteration.md` §40/§43（5 次模型层尝试全负的结论）。
> 纪律：本预注册**先于任何实验**冻结判据；判决词由用户填写（json `verdict_word` 保持 null）。
> 用户已拍板：立即换版 + 重置前瞻窗；数据全面放开（月频源回原始表提炼日频）；主判据 = 信息层 RankIC/ICIR + 时间稳定性；架构不设限（坚持 DL，允许混合）。

---

## §一 背景与动机（为何这一次与之前 5 次不同）

前 5 次模型层尝试（#14 rank-loss、#18 两把手术刀等）全部失败，§43 结论：「模型层的损失函数与选模指标不是当前杠杆」。

本审计补充了一条 §43 未区分的关键事实（`docs/audit-model-20261009.md` §3）：
- **P0-4**：现有管道对**每一列**做逐日截面 Rank-Gauss（`fterm/model/foldframe.py:95-103`、`infer.py:52-54`）。任何横截面常数特征（市场状态、指数收益、市场广度）秩恒 0.5 → 白化后**恒为 0**。
- ⇒ **「市场状态非秩通道」在现有管道下结构上无法测试**——不是试过失败，是根本没进去过。

因此本次的杠杆**不是损失函数**，而是：
1. **输入管道**：新增旁路通道 `Z`（不做 rank）。
2. **目标结构**：多 horizon 融合（#7 已证 IC 随 h 单调升）。
3. **信息面**：从老库日频原始数据提炼新因子（P1-7）。

---

## §二 冻结声明

**冻结（所有臂逐字不变）**：
- 数据面板 `data/panel/*.parquet`（12 张）+ 交易日历 + PIT 股票池口径。
- 标签口径 `fwd_ret = hfq_open[t+h+1]/hfq_open[t+1] − 1`（h=5 为主标签）。
- 特征白化口径（逐日截面秩 → erf⁻¹）、缺失指示列、`len(elig)<5` 丢弃。
- 折切分与净化语义（`fterm/model/foldframe.py`、`dataset.py::_date_purge_count`）。
- 成本/撮合口径（`fterm/contracts.py`、`backtest/costs.py`）。
- 评估年化常数（`TRADING_DAYS`，见 §五 erratum）。

**变量（每臂只动一个）**：见 §四。

**对照**：**新鲜 F0**——同码同种同折在今日面板上重训的 OOS 预测流（#18 erratum §四口径，**非**历史缓存流）。

---

## §三 主判据（信息层 · 时间稳定性）

用户拍板：主判据 = 信息层 RankIC/ICIR + 时间稳定性。全部在 `fterm/model/stats.py` + `cpcv.py`（Phase 3）实现。

| 判据 | 门槛 | 说明 |
|---|---|---|
| **IC 提升** | 臂 IC − F0 IC 的**逐日配对 t ≥ 2** | 单变量、同日配对 |
| **ICIR** | 臂 ICIR > F0 ICIR | IC 均值 / IC 标准差 |
| **CPCV 稳定性** | `P(IC>0) ≥ 0.9` 且路径间 IC 标准差低于 F0 | `φ=C(N,k)·k/N` 条路径 |
| **制度分层** | 4 层（指数趋势 2×波动 2）regime 分层 IC 的最小值 > 0 | 直击 §40「时间结构」 |
| **滚动 IC** | Newey-West t > 2 且无长段负 IC | 60 日滚动 |
| **过拟合** | `PBO < 0.5` 且 Deflated Sharpe > 0 | Bailey & López de Prado |

**组合层（第二轴，Phase 4）**：`arm_faithful`（真 T3=100 万/top10/keep20 + daily EMA + blocked_cash=0）相对 F0 的**月度配对 t ≥ 2** 且 **MDD ≥ −30%** 且 **周 std 不劣** 且 **blocked_cash = 0**。

---

## §四 实验臂与各自判据

> 每臂 = 一份子预注册 + 一个执行器（`scripts/*_chain.py`）+ 独立可证伪判据。臂按代价排序执行，每臂出读数即登记，不阻塞后续。

### 臂 A — 市场状态非秩通道（regime conditioning）
- 文件：新 `fterm/features/regime.py`、`fterm/model/net3.py`、`scripts/regime_chain.py`；改 `foldframe.py`（旁路 `Z`）、`net2.py`（forward 增 `z`）。
- 特征（横截面常数、t 收盘可得）：`000300/000905` 收益与均线偏离、市场广度（%股>MA20）、截面离散度、全市场换手水平、涨/跌停家数、横截面 pvol 中位数、北向净流入、老库 `regime` 状态。
- 注入：① FiLM（先做）；② 条件 LayerNorm；③ 门控 MoE。
- **判据**：`A_IC−F0_IC` 配对 t≥2 **且** 4 层 regime 分层 IC 最小值 > 0 **且** 滚动 60 日 IC 的 NW-t 较 F0 改善。

### 臂 B — 多 horizon 融合
- 文件：新 `fterm/features/multihorizon.py`、`fterm/model/heads.py`、`losses.py`、`scripts/multihorizon_chain.py`；改 `build.py`、`config.py`、`config.yaml`。
- **判据**：h=5 头 IC 不劣于 F0（配对 t≥0）**且** ≥2 个其他 horizon 头 IC>0 **且** 融合分 ICIR > F0 单头。

### 臂 C — 新数据（从原始数据提炼日频因子）
- C1 分钟级（`baostock_min60`，2020+）；C2 资金面（`margin`/`hk_hold`/`moneyflow`/`block_trade`/`lhb`）；C3 事件面（`forecast`/`insider_trade`/`holder_num`/`restricted_release`/`repurchase`/`kg_edge_event`）；C4 文本（`cninfo`/`fulltext`/`qa_archive`）；C5 行业（`sw_member` → 行业中性化）。
- 去重口径：与现役 25 特征 `max|ρ| ≤ 0.7`；在 fterm 面板/标签上重算 `|IC|≥0.02 & t≥2`（不照抄老盘读数）。
- **判据（每子臂独立）**：增量 IC 配对 t≥2 **且** 加入后 `max|ρ|` 在阈值内 **且** 置换阴性对照 IC 不塌。

### 臂 D — 架构
- D1 截面注意力（先做）；D2 FT-Transformer（对照）；D3 KG 行业链 GNN（最后）；D4 MoE。
- **判据**：IC 配对 t≥2 **且** 训练/推理耗时与显存在 RTX4060 8GB 预算内。

> **erratum §四-1（2026-10-10，Lead 补登；跑数前已实现、非事后改判据）**：
> 臂 D（注意力掩码）与臂 E（listwise 按日截面）**都要求同一交易日的股票同处一个
> batch**，而 F0 用跨日随机混洗（同批每日仅 1–2 只）⇒ 两类改动会失去意义。
> 故 D/E 改用**按日整批**（`fterm/model/daybatch.py`）。
> **但按日整批本身改变训练动态**（批内相关性、有效批大小、梯度噪声）——
> 若只与 F0 比，"批构造"与"架构/损失"两个变量混淆。
> ⇒ 增设 **`F0day` 对照臂**（同样按日整批 + net2 + MSE），
> **判据须同时给出「D/E vs F0」与「D/E vs F0day」两条配对 t**。
> 实测该混淆**非空**：F0day IC +0.0401/ICIR **+0.739** vs F0 +0.0447/ICIR +0.652
> —— 批构造单独就把 ICIR 抬高 0.087（IC 略降）。**故只报 vs F0 的读数不可解释。**

### 臂 E — 损失/训练 + 混合模型
- 新 `losses.py`（ListNet/Softmax listwise、ListMLE、不确定性加权）；改 `train_production.py`（损失可插拔 + CPCV 排序型选模）；混合 = DL + LightGBM stacking（`lgbm_chain.py` 树臂 IC≈0.05）。
- **判据**：listwise IC 配对 t≥2 **且** 尾部（top0.5% 篮子周均收益）不劣。
- **纪律**：任何采纳必须先打败 `fterm/model/baselines.py::run_baselines` 的 DLinear/GRU 基线。

> **erratum §四-2（2026-10-10，Lead 补登）**：本条判据第二项「尾部（top0.5% 篮子
> 周均收益）不劣」在 `scripts/arch_chain.py` 初版**未实现**（只出 IC/ICIR/配对 t/
> regime 分层）⇒ 臂 E 跑完无法按本预注册裁决。已补 `_tail_bucket()`，口径**逐字复用**
> 既有 `scripts/tailloss_chain.py::tail_bucket_weekly`（与 #18 可比），
> 并给 E/E2 出 `tail_not_worse`（对 F0 与 F0day 两对照）。**判据数值一字未改。**
> 另登记一处**未消除的口径错配**：listwise 目标与「val MSE 选模」不同源
> （P0-2 同款错配）——本批**不修**（改选模口径=改基线，须另立预注册），
> 但 E/E2 的读数须在此前提下解读；ListMLE 的损失量级无界（实测同 batch
> 2842 vs ListNet 6.6），故其训练动态与 ListNet 不可直接比。

---

## §五 仪器门（任一失败 ⇒ 该臂读数作废）

1. **阴性置换 tripwire**：置换标签后 `|mean IC| ≤ 0.025`。
2. **新鲜 F0 位连**：同码新鲜 F0 对今日面板的行/标签位连（#14 式，实测应逐位 0）。
3. **流右端点守卫**：OOS 流右端点距面板末 < 15 天 → 响亮失败。
4. **环境谱系登记**：`_meta` 记 python/pandas/numpy/torch 版本（`.venv` 与 `.venv312` 读数不可互比）。
5. **面板竞态守卫**：研究线先 `sync_data --end 20260930` 钉隔离窗。

**erratum（已知口径不一致，本批不动但登记）**：`metrics.py` 用 243、`exposure_rule_chain.market_state` 用 245。

---

## §六 换版与前瞻窗重置（用户已拍板，接受破坏预注册的代价）

1. 新工件 `data/models/dlv3_production.pt`；旧 `dlv2_production.pt` 归档 `data/models/archive/` + manifest。
2. `config.yaml signal.mode` 新增 `dlv3` 分支；`features.*` 新增 `label_horizons`、外部数据开关。
3. 换版守卫：`runner.py` `ema_model_version` kv —— 文件 sha ≠ 账本 kv → 秩平滑清零 + `ema_reset` 告警。
4. 前瞻窗重置：新建 `plans/paper60c-prereg-<日期>.md`，明示「前窗作废、新窗自新工件首个信号日起算」。
5. docs/02 参数基线登记 + docs/03 台账 §44+。

---

## §七 判决词（用户填写）

| 臂 | 判决词 |
|---|---|
| A 市场状态非秩通道 | |
| B 多 horizon 融合 | |
| C 新数据 | |
| D 架构 | |
| E 损失/混合 | |
