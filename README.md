# fterm · A股量化终端（Final）

## 定位（2026-10-11 用户拍板）

> **Final 是一个「每日预测全市场个股相对强弱」的程序。**
> 它每天给出每只股票在当日截面中的**强弱位置（分档）**，
> 回答"这只股票未来会**跑赢还是跑输**其他股票"。

**三个"不"（架构事实，不是谦虚）**：

1. **不预测绝对涨跌幅** —— 输入层对每一列做**逐日截面秩白化**
   （`fterm/model/foldframe.py`），任何市场级信息（指数涨跌、涨跌家数、市场宽度）
   在同日截面**零方差**，对输出零贡献（docs/03 §44 P0-4）⇒ 模型**物理上**只能输出相对位置。
   实测：绝对方向准确率 **0.5196**，而"永远猜多数方向"是 **0.7024** ⇒ **差 18pp**。
2. **不预测大盘方向** —— 同上。
3. **不输出"第几名"** —— 档内排序无分辨力：top-10 内 IC **为负**；
   全截面 IC +0.0889 而 top-50 内仅 +0.0002（docs/03 §48）。

**判据与报告**：判据链 #30（`plans/strength-report-prereg-20261011.md`），
执行器 `scripts/strength_report.py`：

```bash
.venv\Scripts\python.exe scripts\strength_report.py --backfill          # 全史标定
.venv\Scripts\python.exe scripts\strength_report.py --asof 2026-09-30   # 单日报告
```

**实测（2019-01-02 → 2026-09-21，1871 日，5385 只）**：分档单调性 ρ **+0.9758**（h=1）；
判强超额 **+2.36pp**（t=8.14）、判弱超额 **+3.30pp**（t=12.14）；
**判弱比判强准 1.4~1.8 倍**。

> ⚠ **组合层与应用层是已知瓶颈**：判据链 #24 证明"账户收益打败基准"类判据在
> 本项目规模下**结构性不可用**；#29 首次给出被动参照（可投 ETF 年化 **+4.83%~+13.04%**）。
> **预测层与组合层必须分开评估**（详见 docs/03 §49/§53）。

---

个人 A 股现货量化交易终端。**2 万元本金设计基准**，个人单机（Windows + CPU）
低成本长期运行。前身是期货终端（2026-09-26 ~ 2026-10-02）；**2026-10-03
起彻底放弃期货**，域层全部改造为 A 股现货，骨架与工程纪律保留
（改造登记见 [docs/03 §14](docs/03-model-iteration.md)）。

## 快速开始

```bash
# 环境（一次性）：本机 Python 3.13 + venv（复用系统 torch/pandas）
D:\Python313\python.exe -m venv --system-site-packages .venv
.venv\Scripts\python.exe -m pip install pyarrow pyyaml pytest torch

# 1. 数据同步（老库 neweco.db 只读契约 → data/panel/*.parquet，本地无网络依赖）
.venv\Scripts\python.exe scripts\sync_data.py

# 2. 快速验证（walk-forward）
.venv\Scripts\python.exe scripts\train_model.py --quick

# 3. 训练正式模型工件（多种子集成；rule_reversal 模式可跳过）
.venv\Scripts\python.exe scripts\train_model.py

# 4. 回测（忠实单遍：引擎循环内真实账本，熔断/日停逐日联动）
.venv\Scripts\python.exe scripts\backtest_run.py --start 2020-01-01

# 5. 每日决策（收盘后；paper 纸面撮合；A股只做多）
.venv\Scripts\python.exe scripts\run_daily.py

# 全量测试
.venv\Scripts\python.exe -m pytest tests\
```

## 架构

详见 [docs/01-architecture.md](docs/01-architecture.md)。单向数据流：

```
data（老库只读 → 宽面板 + PIT 股票池）→ features（面板化时序/截面特征）
  → model（CrossSectionNet 截面网，可选）→ strategy（rule_reversal 主信号 / 模型门限）
  → risk（sizer 整手手数 + governor 硬约束）→ exec（PaperBroker；券商通道留接口）
  → runtime（SQLite 状态 + runner 调度 + watchdog）
```

## 核心设计（区别于通用框架）

1. **数据契约（L1）**：只读老盘 `neweco.db`（日线/复权因子/换手市值/真实
   涨跌停价/ST 更名/交易日历），本仓不写老库、不 import 老仓代码（同 ECO-W2 模式）。
   全市场宽面板一次同步，逐日 PIT 股票池过滤。
2. **生产模型 = DL-v2 截面网**（`fterm/model/net2.py::CrossSectionNet`，判据链 #8 起）：
   逐日截面秩 → Rank-Gauss 白化 → 数值分桶 embedding → ResNet-MLP 主干 → 回归头；
   **逐股独立打分、无时序窗**。生产工件 `data/models/dlv2_production.pt` 为
   14 网集成（MSE + RankNet 秩项，λ=1）。
   > 注：`fterm/model/net.py::TriHeadNet`（双通路三头，继承期货版）仍在仓内，
   > 但**不是生产信号位**——它只服务 `baselines.py`/`validate.py` 的基线对照，
   > 以及 `signal.mode: model` 这条 legacy 路径（当前生产位是 `dlv2`）。
   > 因此 TriHeadNet 的 vol 头→sizing 设计（`sizer.vol_cap_lots`）在现行生产链
   > 不生效，sizing 实际走 `risk_budget_lots`（ATR 口径）。
3. **A股现货微观结构如实建模**：100 股整手离散（0 手合法）、最低佣金 5 元的
   规模不经济、印花税卖出时间分段（2023-08-28 减半）、过户费分段、真实停板价
   一字板不可成交、T+1（结构保证 + 防御闸留痕）、卖出资金当日可用。
4. **规则主信号优先**：`rule_reversal`（截面 5 日反转秩 + 静默波动闸）是
   **占位基线，未证实 α**——任何采纳声明必须预注册 + 组合层两轴（纪律同老盘）。
5. **忠实单遍回测**：决策在引擎循环内用引擎真实权益/可用现金/真实持仓问询
   （与生产 runner 同账本语义）；熔断（−12%）与日停（−3%）逐日联动。
   两遍法旧口已于 2026-10-08 判废收口（docs/03 §29/§42）。

## 2 万本金风控参数（config.yaml，改动必须登记 docs/02）

| 参数 | 值 | 说明 |
|---|---|---|
| initial_equity | 20000 | 设计基准 |
| max_cash_util | 0.95 | 持仓市值 ≤ 权益 95%（现货全额现金） |
| risk_per_trade | 0.02 | 单票止损风险 ≤ 权益 2%（400 元） |
| atr_k | 1.2 | 止损距离 = 1.2 × ATR14 |
| max_pos_vol_daily | 0.025 | 单仓位预测日波动 ≤ 权益 2.5% |
| daily_stop_loss | 0.03 | 当日浮亏 3% 停开新仓 |
| total_dd_halt | 0.12 | 总回撤 12% 熔断（只拒新开，继承期货版 A/B 判决） |
| max_lots_per_symbol | 10 | 单票手数上限（1手=100股） |
| max_symbols | 3 | 同时持仓股票数上限 |

股票池：全 A 过滤器（剔北交所、ST/退 PIT 剔除、上市满 120 交易日、
流通市值 ≥3 亿、价 ≥2 元）。**现实约束**：200 元股价一手即 2 万 = 全仓；
最低佣金 5 元下小额成交费率 0.5%+——sizer 的 0 手合法输出与现金硬约束
共同保证"打不起就放弃"。这是 2 万本金的数学事实，不是缺陷。

## 定位（2026-10-04 用户拍板：**独立策略线**）

fterm-A 是**独立策略线**（与老盘 NEW ECO 月频、ECO-W2 周频 B0 并行的第三条线），
不是 W2 的附属骨架。差异化：日频终端级基础设施（数据链/纸面执行/看板/判据链
执行器）+ 模型驱动截面选股（DL-v2 **14 网**集成 + C_ema）+ 双 paper 账户轨道
（w2 2万/top2、t3 100万/top10）。**当前生产信号位 = dlv2 双轨纸面**
（2026-10-05 用户拍板"直接代替现役"；判据链 #1-#13 见 docs/03 §15-27），
60 日前瞻验证窗自 10-05 首跑起算，每交易日 20:05 计划任务自动跑。
（2026-10-09/10 模型升级批：**截至立案时点的九次干预全部判负**；随后的第 10–12 次干预
（臂 D 截面注意力 / E ListNet / E2 ListMLE）中 **D 与 E2 首次通过信息层配对 t ≥ 2 判据**，
但被 ICIR 门、过拟合 PBO 门与 Phase 4 组合层（月度配对 t −0.47 / −1.27）三重拦下
<!-- ⚠ 勘误 2026-10-11：PBO 门降级为参考读数（相对量 + N<4 伪读数）⇒ 实为**双重**；
     结论不变（ICIR 与组合层各自独立成立）。见 docs/03 §69 / §70。 -->
⇒ **最终仍不具备换版依据、未换版**。见 docs/03 §45、docs/02 同日登记；前瞻窗未重置。）

## 工程纪律（继承自 NEW-ECO / ECO-W2）

- 改 `config.yaml` 回测/成本/风控默认值 = 换基线，必须在 `docs/02-operations.md`
  §参数基线登记原因与对比数据。
- 测试全量跑（`pytest tests\`），不做"只跑相关文件"。
- 面板内容指纹进模型工件与回测报告，实验可复现。
- `tests/` 里**禁止**真实网络与真实老库：数据层测试全部用合成 SQLite 替身。
- 模型工件自带 config 摘要哈希，配置改动后旧工件拒载（防口径漂移）。

## 纸面 → 实盘路径

1. 当前：`PaperBroker`（纸面撮合，成本口径与回测一致）。
2. 实盘：券商程序化通道（miniQMT/PTrade 类），实现 `exec.broker.Broker`
   四个方法（submit/poll/account/cancel_all）即可接入，业务代码零改动。
   **前置条件**：paper 连续 ≥3 个月、回测与 paper 成交口径偏差 <5%、
   paper 期最大回撤未触熔断。见 docs/02 §上实盘检查单。

## 文档索引

- [docs/00-research.md](docs/00-research.md) — 调研结论（A股版）
- [docs/01-architecture.md](docs/01-architecture.md) — 架构设计与接口契约
- [docs/02-operations.md](docs/02-operations.md) — 运维手册 + 参数基线登记
- [docs/03-model-iteration.md](docs/03-model-iteration.md) — 模型迭代台账
  （§1–13 期货阶段存档，§14 起为 A 股）
