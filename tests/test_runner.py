"""DailyRunner 装配层验收：规则模式 run_once 全流程（FakeBroker）+ 模型工件守卫。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fterm.exec.broker import AccountSnapshot, Broker, FillReport, Order, Side
from fterm.runtime.runner import DailyRunner
from fterm.runtime.state import StateDB


class FakeBroker(Broker):
    """离线 broker：记录订单，账户状态由测试直接摆放。"""

    def __init__(self, equity: float = 20000, positions: dict | None = None):
        self._equity = equity
        self._positions = positions or {}
        self.orders: list[Order] = []

    def submit(self, order: Order) -> str:
        self.orders.append(order)
        return f"fake-{len(self.orders)}"

    def poll(self) -> list[FillReport]:
        return []

    def account(self, mark_prices: dict | None = None) -> AccountSnapshot:
        pos_value = sum(p.get("value", 0.0) for p in self._positions.values())
        return AccountSnapshot(equity=self._equity, available=self._equity - pos_value,
                               pos_value=pos_value, positions=self._positions)

    def cancel_all(self) -> int:
        return 0


def _friday_asof(cfg, bundle):
    dates = bundle.features[f"rev_{cfg.signal.rev_window}"].index
    fridays = [d for d in dates if d.weekday() == cfg.signal.rebalance_weekday]
    return fridays[-2]        # 倒数第二个周五（最后一个周五标签可能截尾，规则模式不依赖）


def test_runner_end_to_end(env):
    cfg, panels, cal = env
    from fterm.features.build import build_dataset
    bundle = build_dataset(cfg)
    asof = _friday_asof(cfg, bundle)
    broker = FakeBroker(equity=20000)
    db = StateDB(cfg.state_db_path)
    runner = DailyRunner(cfg, broker, db)
    summary = runner.run_once(asof=asof)

    # 决策审计落库（幂等 REPLACE）
    assert db.stats()["decisions"] >= 1
    assert db.last_heartbeat() is not None
    # 权益快照已记录（每日一行）
    assert len(db.equity_curve()) == 1
    # 规则模式：top_n=1 → 至多一张买单；理由带秩读数
    buys = [o for o in broker.orders if o.side == Side.BUY]
    assert len(buys) <= cfg.signal.top_n
    for o in buys:
        assert o.lots >= 1 and o.reason
    # 同一 asof 重跑幂等（REPLACE）
    n_before = db.stats()["decisions"]
    runner.run_once(asof=asof)
    assert db.stats()["decisions"] == n_before


def test_runner_exit_band(env):
    """持仓跌出 top_n×exit_buffer 退出带 → 产生清仓决策 + SELL 单。"""
    cfg, panels, cal = env
    from fterm.features.build import build_dataset
    from fterm.strategy.primary import rule_reversal_decision
    bundle = build_dataset(cfg)
    dates = bundle.features[f"rev_{cfg.signal.rev_window}"].index
    # 取 ST 生效（合成库 2024-01-30）前、且过了上市暖机期的周五 ⇒ 池里 3 只都在
    fridays = [d for d in dates if d.weekday() == cfg.signal.rebalance_weekday
               and d < pd.Timestamp("2024-01-30")]
    asof = fridays[-1]
    score, eligible = rule_reversal_decision(bundle.features, bundle.universe, asof,
                                             rev_window=cfg.signal.rev_window,
                                             quiet_vol=cfg.signal.quiet_vol)
    ranked = score[eligible].sort_values(ascending=False)
    keep_n = cfg.signal.top_n * cfg.signal.exit_buffer
    outside = [c for c in ranked.index if c not in set(ranked.head(keep_n).index)]
    assert outside, "3 只股票池与 keep_n=2 必然有 1 只在带外"
    held = {"600036.SH": {"direction": 1, "lots": 3, "value": 5000}}
    held_code = next(c for c in outside if c in held) if any(c in held for c in outside) else None
    if held_code is None:
        held = {outside[0]: {"direction": 1, "lots": 3, "value": 5000}}
        held_code = outside[0]
    broker = FakeBroker(equity=20000, positions=dict(held))
    db = StateDB(cfg.state_db_path)
    summary = DailyRunner(cfg, broker, db).run_once(asof=asof)
    sells = [o for o in broker.orders if o.side == Side.SELL]
    assert any(o.product == held_code for o in sells)
    assert any(d["product"] == held_code and d["direction"] == 0 for d in summary["decisions"])


def test_runner_not_rebalance_day_holds(env):
    cfg, panels, cal = env
    from fterm.features.build import build_dataset
    bundle = build_dataset(cfg)
    dates = bundle.features["rev_2"].index
    monday = next(d for d in dates[15:] if d.weekday() == 0)
    broker = FakeBroker(equity=20000, positions={"000001.SZ": {"direction": 1, "lots": 2, "value": 4000}})
    summary = DailyRunner(cfg, broker, StateDB(cfg.state_db_path)).run_once(asof=monday)
    assert broker.orders == []
    assert all(d["direction"] == 0 and d["reason"] == "not_rebalance_day"
               for d in summary["decisions"])


def test_runner_halt_mode(env):
    """收口态：halt 模式只跑骨架（权益快照+心跳），零决策零订单、不建特征。"""
    cfg, panels, cal = env
    import dataclasses
    from dataclasses import replace
    cfg2 = replace(cfg, signal=replace(cfg.signal, mode="halt"))
    broker = FakeBroker(equity=20000)
    db = StateDB(cfg2.state_db_path)
    summary = DailyRunner(cfg2, broker, db).run_once()
    assert broker.orders == []
    assert all(d["reason"].startswith("halt") for d in summary["decisions"])
    assert len(db.equity_curve()) == 1       # 权益快照照常
    assert db.last_heartbeat() is not None and "halt" in db.last_heartbeat()[1]


def test_runner_model_config_guard(env):
    """mode=model 但无工件 ⇒ 明确报错（工件与配置一致性守卫）。"""
    cfg, panels, cal = env
    import dataclasses
    from dataclasses import replace
    cfg2 = replace(cfg, signal=replace(cfg.signal, mode="model"))
    from fterm.features.build import build_dataset
    build_dataset(cfg2)
    broker = FakeBroker(equity=20000)
    runner = DailyRunner(cfg2, broker, StateDB(cfg2.state_db_path))
    with pytest.raises((FileNotFoundError, AssertionError, RuntimeError)):
        runner.run_once()
