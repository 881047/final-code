# fterm 总体架构设计（A股版）

> A股量化终端。2 万元本金设计基准，个人单机低成本长期运行。
> 前身是期货终端（2026-09-26 ~ 2026-10-02）；2026-10-03 期货域全部移除，
> 架构与工程纪律保留（见 docs/03 §14 改造登记）。

## 设计原则

1. **单进程可拆**：每个模块可独立 CLI 运行（同步 / 训练 / 回测 / 决策），
   也可由调度器串联。无分布式、无外部网络依赖（数据来自本地老库只读契约）。
2. **口径即代码**：佣金/印花税/过户费/滑点/涨跌停规则全部进回测撮合引擎，
   与纸面执行共用同一套规格（`contracts.py`），回测与 paper 不出现两套口径。
3. **防未来函数**：特征只用 t 日收盘前可得数据，标签从 t+1 开盘建仓口径计算；
   面板带内容指纹，训练样本可追溯；股票池过滤逐日 PIT 判定。
4. **小资金现实**：100 股整手离散化（0 手 = 放弃信号是合法输出）、最低佣金
   5 元的规模不经济、T+1、卖出资金当日可用，全部如实建模。

## 模块与职责

```
fterm/
├── config.py          # 配置加载（config.yaml → 冻结 dataclass，多余键报错）
├── contracts.py       # A股交易规格：整手 100 股、佣金（万2.5/最低5元）、
│                      #   印花税（卖出、2023-08-28 减半分段）、过户费（双边分段）
├── data/
│   ├── ashare.py      # 老库 neweco.db 只读契约 → 宽面板（date × ts_code）落
│                      #   data/panel/*.parquet；PIT 股票池掩码；ST/退市风险
│                      #   namechange 逐段展开；后复权价；交易日历
│   └── calendar.py    # 日历校验（A股无夜盘，期货版夜盘归属逻辑已随域移除）
├── features/
│   ├── tsfactors.py   # 面板化时序因子：多窗动量/反转、Parkinson·GK 波动率、
│   │                  #   ATR、日内微观（跳空/CLV/影线）、换手率与市值
│   └── build.py       # 截面特征（xrank/rel/disp）+ 标签 + DatasetBundle
├── model/             # 与期货版同构（TriHeadNet 三头 + 多种子集成 + 净化折叠）
│   ├── dataset.py     #   面板采样（日采样密度轮转 cohort）+ 净化 + RevIN
│   ├── net.py         #   TriHeadNet：线性通路 + GRU 通路 → 方向/分位/波动三头
│   ├── train.py       #   多种子训练 + 集成 + 波动头校准 + 工件摘要哈希
│   ├── baselines.py   #   DLinear / 纯 GRU 基线
│   └── validate.py    #   walk-forward（净化折叠）、方向准确率、逐日截面 RankIC
├── strategy/
│   ├── primary.py     # 规则主信号 rule_reversal：截面反转秩 + 静默波动闸
│   │                  #   （占位基线，未证实 α；采纳前必须预注册+组合层两轴）
│   └── signal.py      # 模型三头 → 只做多置信度门限（分歧惩罚 + 分位宽度加严）
├── risk/
│   ├── sizer.py       # 风险预算手数（ATR 止损）+ 单仓位波动上限（vol 头预测）
│   └── governor.py    # 现金占用上限 / 票数 / 日停 / 12% 总回撤熔断（强平 flag 继承）
├── backtest/
│   ├── engine.py      # 事件驱动：t 收盘信号 → t+1 开盘成交；真实停板价一字板
│   │                  #   掩码；T+1 防御闸；现金硬约束；卖出资金当日可用
│   ├── costs.py       # 佣金+印花税+过户费+滑点（与 paper 同源）
│   └── metrics.py     # 夏普/卡玛/回撤/胜率/成本占比/仓位占用/拒单计数
├── exec/
│   ├── broker.py      # Broker 抽象（BUY/SELL；实盘券商通道实现后接入）
│   └── paper.py       # 纸面撮合（成本口径与回测共用）
├── runtime/
│   ├── runner.py      # 每日调度：面板→特征→推理→信号→风控→订单→审计落库
│   ├── state.py       # SQLite 状态库（权益/持仓市值/成交/决策/心跳/告警；
│   │                  #   期货版 legacy 表就地改名留档）
│   └── watchdog.py    # 进程外探活 + 告警（⚠️ 未接线：无调用方，check_once 供手动探活）
└── research/          # 研究线脚手架（判据链共用：报告 _meta/自证块/复现契约）
    └── scaffold.py
```

> **`web/` 已退役**（2026-10-07 看板线停用）：本仓无 `fterm/web/` 目录，看板改由
> NEW ECO 8787 全账户页承载（`http://127.0.0.1:8787/#/fterm` 双轨页）。
> 本树中不再列出该模块（原 listing 是改造前残留，2026-10-10 文档漂移清理）。

## 数据契约（L1，与 ECO-W2 同款）

- **只读**老库 `D:\NEW ECO\data\neweco.db`（daily / adj_factor / daily_basic /
  stk_limit / stock_basic / namechange / trade_cal），本仓不 import 老仓代码。
- 同步产物为本仓自有面板（`data/panel/*.parquet`），带内容指纹进训练工件。
- 面板起点默认 2017-01-01：老库 `stk_limit` 真实涨跌停价自 2017 起全量覆盖，
  更早年份只有近似兜底口径（昨收 ±10%），不作为回测主区间。

## 每日运行流水（runner）

```
收盘后        sync_panels（老库只读同步）→ 特征构建 → 信号（规则或模型）
              → sizer 手数 → governor 审批 → 次日开盘单队列（PaperBroker）
              → 决策/权益快照/心跳落库（幂等 REPLACE）
常驻          无常驻 web 看板（8790 看板线 2026-10-07 退役，改看 NEW ECO 8787
              全账户页 fterm 双轨页 http://127.0.0.1:8787/#/fterm）；watchdog 未接线
              （现行保障 = 计划任务 fterm_daily + 数据就绪闸 exit 2 留痕）
每周五        全量重训（CPU）+ walk-forward 验证报告落盘
```

## 2 万元本金的默认风控参数（config.yaml，改动必须登记 docs/02）

| 参数 | 默认 | 说明 |
|---|---|---|
| initial_equity | 20000 | 设计基准本金 |
| max_cash_util | 0.95 | 持仓市值/权益上限（现货全额现金） |
| risk_per_trade | 0.02 | 单票风险预算（ATR 止损距离 × 股数 ≤ 权益×此值） |
| max_pos_vol_daily | 0.025 | 单仓位日波动上限（占权益；语义继承期货版重定义） |
| daily_stop_loss | 0.03 | 当日浮亏超 3% 停开新仓 |
| total_dd_halt | 0.12 | 总回撤 12% 熔断，人工确认才能重启 |
| circuit_force_liquidate | false | 熔断只拒新开（docs/03 §13 A/B 判决口径继承） |
| max_lots_per_symbol | 10 | 单票手数上限（1手=100股） |
| max_symbols | 3 | 同时持仓股票数上限 |
| universe | all_a 过滤器 | 剔北交所/ST（PIT）、上市满 120 交易日、流通市值 ≥3 亿、价 ≥2 元 |

## 与老盘 / W2 的关系

- 老盘（NEW ECO）：月频基线 + 周频袖套的生产系统；本仓**只读其数据**，不共享代码。
- W2（ECO-W2）：2 万周频独立第二盘；本仓沿用其"只读老库 L1 契约"模式。
- 本仓定位：把老盘沉淀的纪律（预注册、组合层两轴、PIT、参数基线登记）
  装进一套每日运行的终端骨架；α 的证实或证伪都在这套骨架上以可复现方式发生。
