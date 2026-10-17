"""端到端集成：合成老库同步 → 特征 → 规则信号 → 引擎回测 → 指标（+期货代码清除验证）。

验证的是"全链路装配正确 + 2万本金约束全程生效 + 期货域彻底移除"，不是策略盈利性。
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from fterm.backtest.engine import BacktestEngine, Target
from fterm.risk.governor import AccountState, RiskGovernor
from fterm.risk.sizer import size_position
from fterm.strategy.primary import rule_reversal_decision


def test_futures_code_removed():
    """彻底放弃期货：期货域模块必须不存在。"""
    for mod in ("fterm.exec.ctp_adapter", "fterm.data.continuous",
                "fterm.data.akshare_src", "fterm.data.store",
                "fterm.features.structure"):
        assert importlib.util.find_spec(mod) is None, f"期货模块残留: {mod}"


def test_full_pipeline(env):
    cfg, panels, cal = env
    from fterm.features.build import build_dataset
    bundle = build_dataset(cfg)
    feats, universe = bundle.features, bundle.universe
    raw_close = panels["close"]

    reb_dates = [d for d in cal if d.weekday() == cfg.signal.rebalance_weekday]
    reb_dates = [d for d in reb_dates if d in feats["atr14"].index]
    assert len(reb_dates) >= 3

    targets: dict[pd.Timestamp, list[Target]] = {}
    positions: dict[str, dict] = {}
    gov = RiskGovernor(cfg)
    keep_n = cfg.signal.top_n * cfg.signal.exit_buffer
    for d in reb_dates:
        equity = cfg.account.initial_equity
        score, eligible = rule_reversal_decision(feats, universe, d,
                                                 rev_window=cfg.signal.rev_window,
                                                 quiet_vol=cfg.signal.quiet_vol)
        ranked = score[eligible].sort_values(ascending=False)
        tgts = []
        for code in list(positions):
            if code not in set(ranked.head(keep_n).index):
                tgts.append(Target(code, 0, 0, "exit_band"))
        for code in ranked.head(cfg.signal.top_n).index:
            if code in positions:
                continue
            px = float(raw_close.at[d, code])
            atr = float(feats["atr14"].at[d, code])
            sz = size_position(cfg, equity, px, atr)
            if sz.lots == 0:
                continue
            tgts.append(Target(code, 1, sz.lots, f"rule_reversal(rank={float(ranked[code]):.2f})"))
        if tgts:
            prices = {t.product: float(raw_close.at[d, t.product]) for t in tgts
                      if pd.notna(raw_close.at[d, t.product])}
            rep = gov.approve(tgts, AccountState(equity=equity, peak_equity=equity,
                                                 day_pnl=0.0, open_positions=dict(positions)),
                              prices=prices)
            for dec in rep.decisions:
                if not dec.approved:
                    continue
                targets.setdefault(d, []).append(dec.target)
                if dec.target.direction == 0:
                    positions.pop(dec.target.product, None)
                else:
                    positions[dec.target.product] = {"direction": 1, "lots": dec.target.lots,
                                                     "value": 0.0}

    bars_map = {}
    for code in raw_close.columns:
        df = pd.DataFrame({
            "open": panels["open"][code], "high": panels["high"][code],
            "low": panels["low"][code], "close": panels["close"][code],
            "up_limit": panels["up_limit"][code], "down_limit": panels["down_limit"][code],
        }).dropna(subset=["close"])
        if len(df):
            bars_map[code] = df
    eng = BacktestEngine(cfg, bars_map)
    start = reb_dates[0]
    result = eng.run(targets, start=start)
    m = result.metrics
    assert m["final_equity"] > 0
    assert m["n_blocked_t1"] == 0                      # T+1 结构保证
    assert np.isfinite(result.equity).all()
    # 权益恒等式：equity = cash + Σ持仓市值（引擎内部恒成立，这里抽查首日）
    assert result.equity.iloc[0] == pytest.approx(cfg.account.initial_equity, rel=1e-9)
