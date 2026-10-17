"""配置加载：config.yaml → 冻结 dataclass。

纪律：运行期任何模块不得直接读 yaml；统一从这里拿强类型配置，
保证口径单一。改默认值 = 换回测基线（必须在 docs/02-operations.md §参数基线 登记）。

A股版（2026-10-03 期货→A股改造）：account 现金口径（无保证金）、universe 由
显式品种列表改为 PIT 过滤器、signal 主信号为截面反转规则。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REQUIRED_KEYS = ("account", "universe", "data", "features", "model", "backtest", "runtime", "signal")


@dataclass(frozen=True)
class AccountCfg:
    initial_equity: float
    max_cash_util: float           # 持仓市值 / 权益 上限（现货无保证金，现金约束）
    risk_per_trade: float
    atr_k: float
    max_pos_vol_daily: float
    daily_stop_loss: float
    total_dd_halt: float
    circuit_force_liquidate: bool  # 熔断→清仓存量持仓（语义继承期货版 docs/03 §13）
    max_lots_per_symbol: int       # 单票手数上限（1手=100股）
    max_symbols: int
    # ---- 回撤峰值口径（2026-10-10，熔断自锁缺陷修复，docs/03 §36/§47）----
    # none    = 全历史累积峰值（**默认，与修复前逐位一致**；存在"永不解除"的吸收态）
    # rolling = 滚动窗口峰值（无吸收态；不含任何未来信息）
    # 语义实现单一来源：fterm/risk/peak.py。默认值由 `_dc` 的缺省表注入，
    # 故旧 config.yaml（无此二键）仍可加载且行为不变。
    dd_peak_mode: str = "none"
    dd_peak_window: int = 252
    # 自锁可观察性（2026-10-10）：熔断触发且权益连续 N 日无变动 ⇒ 落 warn 告警
    # （缺陷本体 = 吸收态；生产必须能"看见"自己卡住，而非静默空转数年）。
    # 0 = 关闭该告警。**纯观测，不改任何审批行为。**
    dd_selflock_warn_days: int = 60

    def validate(self) -> None:
        assert 0 < self.initial_equity <= 10_000_000, "本金异常"
        assert 0 < self.max_cash_util <= 1, "现金占用上限必须在 (0,1]"
        assert 0 < self.risk_per_trade <= 0.05, "单笔风险预算过大（>5% 不允许）"
        assert 0 < self.atr_k <= 5, "atr_k 异常"
        assert 0 < self.max_pos_vol_daily <= 0.10, "单仓位日波动上限异常"
        assert 0 < self.daily_stop_loss < self.total_dd_halt < 1, "回撤层级必须递增"
        assert self.dd_peak_mode in ("none", "rolling"), \
            "account.dd_peak_mode 只支持 none | rolling（见 fterm/risk/peak.py）"
        assert 1 <= self.dd_peak_window <= 5000, "account.dd_peak_window 须在 [1,5000]"
        assert 0 <= self.dd_selflock_warn_days <= 5000, \
            "account.dd_selflock_warn_days 须在 [0,5000]（0=关闭告警）"


@dataclass(frozen=True)
class UniverseCfg:
    mode: str                      # all_a（PIT 过滤器）| fixed（显式 codes）
    codes: tuple[str, ...]
    exclude_bj: bool
    exclude_st: bool               # namechange PIT 口径（含"退"）
    min_list_days: int
    min_circ_mv_wan: float         # 流通市值下限（万元）
    min_price: float
    max_universe: int              # 打分面截断（0=不截断；截断按 ts_code 稳定序）

    def validate(self) -> None:
        assert self.mode in ("all_a", "fixed"), "universe.mode 只支持 all_a | fixed"
        assert self.min_circ_mv_wan >= 0 and self.min_price >= 0


@dataclass(frozen=True)
class DataCfg:
    old_db: str                    # 老库 neweco.db 只读契约（同 W2 的 L1：只读，不写老库）
    root: str
    start: str                     # 面板起点（默认 2017：stk_limit 真实停板价起覆盖）
    calendar_tz: str

    @property
    def data_dir(self) -> Path:
        return Path(self.root)


@dataclass(frozen=True)
class FeaturesCfg:
    lookback: int
    momentum_windows: tuple[int, ...]
    vol_windows: tuple[int, ...]
    label_horizon: int
    label_quantiles: tuple[float, ...]
    turnover_window: int           # 换手率均值窗（A股特有信息面）
    max_stocks_per_day: int        # 训练采样密度（0=不采样；样本量 = 密度×天数，CPU 预算闸）

    def validate(self) -> None:
        assert self.max_stocks_per_day >= 0


@dataclass(frozen=True)
class ModelCfg:
    name: str
    d_hidden: int
    n_layers: int
    dropout: float
    lr: float
    max_epochs: int
    patience: int
    seeds: tuple[int, ...]
    batch_size: int
    revin: bool
    sample_weight: str             # none | volz
    task: str                      # direction | meta_reversal（标签语义进 digest）


@dataclass(frozen=True)
class BacktestCfg:
    slippage_ticks: int            # 滑点跳数（tick=0.01 元/股）
    signal_exec_lag: int
    limit_updown_block: bool


@dataclass(frozen=True)
class RuntimeCfg:
    state_db: str
    heartbeat_minutes: int
    train_weekday: int


@dataclass(frozen=True)
class SignalCfg:
    mode: str                      # rule_reversal | model | halt | dlv2（校验见 load_config）
    top_n: int                     # 目标持仓数（2 万本金构型）
    exit_buffer: int               # 持仓退出带：跌出 top_n×buffer 才清仓
    rev_window: int                # 反转窗（旧盘极性表：rev_5 正极性）
    quiet_vol: bool                # pvol_20 ≤ 自身 252 日滚动中位数（静默波动闸，继承 H1 口径）
    rebalance_weekday: int         # 信号日（0=周一..4=周五；t 收盘信号 → t+1 开盘执行）
    ema_alpha: float               # dlv2 秩平滑 EMA 系数（新值权重）
    ema_step: str                  # daily | rebalance —— EMA 步进频率（见 docs/02 参数基线）


@dataclass(frozen=True)
class Config:
    account: AccountCfg
    universe: UniverseCfg
    data: DataCfg
    features: FeaturesCfg
    model: ModelCfg
    backtest: BacktestCfg
    runtime: RuntimeCfg
    signal: SignalCfg
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def state_db_path(self) -> Path:
        return Path(self.runtime.state_db)


def _dc(cls: type, src: dict[str, Any], defaults: dict[str, Any] | None = None) -> Any:
    """从 dict 构造 frozen dataclass，多余键报错（防配置漂移）。

    `defaults`：**新增字段的向后兼容缺省值**。旧 config.yaml 没有该键时用它补齐，
    从而"加字段"不构成"换基线"（行为与加字段前逐位一致）。
    **注意**：缺省值只在**键缺失**时生效；显式写了键就以文件为准（可被覆盖）。
    """
    import dataclasses

    fields = {f.name for f in dataclasses.fields(cls)}
    src = dict(src)
    for k, v in (defaults or {}).items():
        assert k in fields, f"{cls.__name__} 的缺省表含未知字段: {k}"
        src.setdefault(k, v)
    missing = fields - set(src)
    extra = set(src) - fields
    assert not missing, f"{cls.__name__} 缺字段: {missing}"
    assert not extra, f"{cls.__name__} 多余字段: {extra}"
    return cls(**src)


def load_config(path: str | Path = "config.yaml") -> Config:
    p = Path(path)
    assert p.exists(), f"配置不存在: {p}"
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    for k in REQUIRED_KEYS:
        assert k in raw, f"config.yaml 缺少段: {k}"

    # 元组化列表字段（冻结 dataclass 不可变约束）
    u = dict(raw["universe"])
    u["codes"] = tuple(u.get("codes") or [])
    f = dict(raw["features"])
    f["momentum_windows"] = tuple(f["momentum_windows"])
    f["vol_windows"] = tuple(f["vol_windows"])
    f["label_quantiles"] = tuple(f["label_quantiles"])
    m = dict(raw["model"])
    m["seeds"] = tuple(m["seeds"])

    cfg = Config(
        account=_dc(AccountCfg, raw["account"],
                    # 新增字段的向后兼容缺省（docs/03 §47）：缺省即"与修复前逐位一致"
                    defaults={"dd_peak_mode": "none", "dd_peak_window": 252,
                              "dd_selflock_warn_days": 60}),
        universe=_dc(UniverseCfg, u),
        data=_dc(DataCfg, raw["data"]),
        features=_dc(FeaturesCfg, f),
        model=_dc(ModelCfg, m),
        backtest=_dc(BacktestCfg, raw["backtest"]),
        runtime=_dc(RuntimeCfg, raw["runtime"]),
        signal=_dc(SignalCfg, raw["signal"]),
        raw=raw,
    )
    cfg.account.validate()
    cfg.universe.validate()
    cfg.features.validate()
    if cfg.universe.mode == "fixed":
        assert cfg.universe.codes, "universe.mode=fixed 需要显式 codes"
        assert all(".SZ" in c or ".SH" in c for c in cfg.universe.codes), "fixed codes 须为 ts_code（如 000001.SZ）"
    assert cfg.signal.top_n <= cfg.account.max_symbols, "top_n 不能超过 max_symbols"
    assert cfg.signal.mode in ("rule_reversal", "model", "halt", "dlv2"), \
        "signal.mode 只支持 rule_reversal | model | halt | dlv2"
    assert 0.0 < cfg.signal.ema_alpha <= 1.0, "signal.ema_alpha 须在 (0,1]（1.0=不平滑）"
    assert cfg.signal.ema_step in ("daily", "rebalance"), \
        "signal.ema_step 只支持 daily | rebalance（链上被采纳构型是 rebalance，生产位现用 daily）"
    if cfg.signal.mode == "dlv2":
        assert raw.get("paper_tracks"), "signal.mode=dlv2 需要 paper_tracks 段"
    return cfg
