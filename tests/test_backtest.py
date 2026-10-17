"""回测引擎验收：账务恒等式、成本口径（佣金/印花税）、真实停板、现金拒单、T+1 闸、手数离散。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fterm import contracts
from fterm.backtest.costs import fill_price_with_slippage, trade_cost
from fterm.backtest.engine import BacktestEngine, Position, Target
from fterm.backtest.metrics import compute_metrics, max_drawdown
from tests.conftest import _yaml_text


@pytest.fixture
def cfg(tmp_path):
    from fterm.config import load_config

    p = tmp_path / "c.yaml"
    p.write_text(_yaml_text().replace("{old_db}", "x").replace("{root}", str(tmp_path / "d").replace("\\", "/")),
                 encoding="utf-8")
    return load_config(p)


def _bars(dates, price=10.0, up=None, dn=None, crash=None):
    """单票合成日线：open=close=price（常价），可选指定停板价与跳价日。"""
    n = len(dates)
    o = np.full(n, price)
    if crash is not None:
        o[crash[0]] = crash[1]
    df = pd.DataFrame({
        "open": o, "high": o, "low": o, "close": o,
        "up_limit": np.full(n, up if up else price * 1.1),
        "down_limit": np.full(n, dn if dn else price * 0.9),
    }, index=pd.DatetimeIndex(dates))
    return df


class TestCosts:
    def test_slippage_directional(self):
        assert fill_price_with_slippage(10.0, 1, 1) == pytest.approx(10.01)
        assert fill_price_with_slippage(10.0, -1, 1) == pytest.approx(9.99)

    def test_trade_cost_composition(self, cfg):
        d = pd.Timestamp("2024-01-05")
        buy = trade_cost(cfg, 10.0, 1, 10, d)            # 10手 = 1000股 × 10.01 = 10010 元
        shares = contracts.shares_of(10)
        assert buy.fill_price == pytest.approx(10.01)
        assert buy.fee == pytest.approx(max(10010 * 0.00025, 5) + 10010 * 0.00001)
        assert buy.slippage == pytest.approx(0.01 * shares)
        sell = trade_cost(cfg, 10.0, -1, 10, d)
        amount_dn = 9.99 * shares
        assert sell.fee == pytest.approx(max(amount_dn * 0.00025, 5)
                                         + amount_dn * 0.0005 + amount_dn * 0.00001)


class TestEngineAccounting:
    def _run(self, cfg, bars_map, targets, dates):
        eng = BacktestEngine(cfg, bars_map)
        return eng.run(targets, start=dates[0])

    def test_equity_identity_no_trade(self, cfg):
        dates = pd.bdate_range("2024-01-02", periods=5)
        r = self._run(cfg, {"000001.SZ": _bars(dates)}, {}, dates)
        assert (r.equity == cfg.account.initial_equity).all()
        assert r.trades == [] and r.fills == []

    def test_long_pnl_and_costs(self, cfg):
        dates = pd.bdate_range("2024-01-02", periods=6)
        # d1: 买入目标（d2 开盘成交 @10）；d3 起价格跳到 12；d4: 清仓目标（d5 开盘 @12）
        bars = _bars(dates, price=10.0)
        bars.iloc[3:, bars.columns.get_indexer(["open", "high", "low", "close"])] = 12.0
        t = {dates[1]: [Target("000001.SZ", 1, 5, "t_in")],
             dates[4]: [Target("000001.SZ", 0, 0, "t_out")]}
        r = self._run(cfg, {"000001.SZ": bars}, t, dates)
        assert len(r.trades) == 1
        tr = r.trades[0]
        assert tr["product"] == "000001.SZ" and tr["lots"] == 5
        # 中间价口径毛利 = (12 − 10) × 500 股；净利 = 毛利 − 全部费用
        assert tr["raw_pnl"] == pytest.approx(2.0 * 500)
        buy_cost = trade_cost(cfg, 10.0, 1, 5, dates[2])
        sell_cost = trade_cost(cfg, 12.0, -1, 5, dates[5])
        assert tr["pnl"] == pytest.approx(1000 - (buy_cost.fee + sell_cost.fee
                                                  + 0.01 * 500 + 0.01 * 500))
        # 账务恒等式：期末权益 = 本金 + 净盈亏
        assert r.equity.iloc[-1] == pytest.approx(20000 + tr["pnl"], rel=1e-9)

    def test_cash_constraint_blocks(self, cfg):
        dates = pd.bdate_range("2024-01-02", periods=4)
        # 20手 × 100股 × 10元 = 20000 元 + 费用 > 本金 ⇒ blocked_cash
        t = {dates[1]: [Target("000001.SZ", 1, 20, "too_big")]}
        r = self._run(cfg, {"000001.SZ": _bars(dates)}, t, dates)
        assert any(f.action == "blocked_cash" for f in r.fills)
        assert r.equity.iloc[-1] == pytest.approx(20000)

    def test_limit_up_blocks_open(self, cfg):
        dates = pd.bdate_range("2024-01-02", periods=4)
        # 执行日（dates[2]）真实涨停价 = 10.0（low ≥ up ⇒ 一字封板买不进）
        bars = _bars(dates, price=10.0, up=10.0)
        t = {dates[1]: [Target("000001.SZ", 1, 5, "t")]}
        r = self._run(cfg, {"000001.SZ": bars}, t, dates)
        assert any(f.action == "blocked" for f in r.fills)
        assert r.equity.iloc[-1] == pytest.approx(20000)

    def test_limit_down_blocks_close(self, cfg):
        dates = pd.bdate_range("2024-01-02", periods=7)
        bars = _bars(dates, price=10.0, dn=12.0)   # 卖出日 high ≤ dn ⇒ 封死跌停卖不出
        t = {dates[1]: [Target("000001.SZ", 1, 5, "in")],
             dates[4]: [Target("000001.SZ", 0, 0, "out")]}
        r = self._run(cfg, {"000001.SZ": bars}, t, dates)
        # 卖出被拒后持仓保留到回测结束（终值按市价盯市）
        assert any(f.action == "blocked" for f in r.fills)
        fee_b = trade_cost(cfg, 10.0, 1, 5, dates[2]).fee
        assert r.equity.iloc[-1] == pytest.approx(20000 - 10.01 * 500 - fee_b + 10.0 * 500)

    def test_t1_guard(self, cfg):
        dates = pd.bdate_range("2024-01-02", periods=3)
        eng = BacktestEngine(cfg, {"000001.SZ": _bars(dates)})
        positions = {"000001.SZ": Position("000001.SZ", 1, 5, 10.01, 10.0, dates[1])}
        fills, trades = [], []
        eng._execute(Target("000001.SZ", 0, 0, "same_day"), dates[1], positions,
                     15000.0, fills, trades)
        assert fills[0].action == "blocked_t1" and trades == []

    def test_lots_discrete_and_no_short(self, cfg):
        dates = pd.bdate_range("2024-01-02", periods=3)
        eng = BacktestEngine(cfg, {"000001.SZ": _bars(dates)})
        with pytest.raises(AssertionError):
            eng._execute(Target("000001.SZ", -1, 5, "short"), dates[1], {}, 20000.0, [], [])

    def test_flip_position_sell_then_buy(self, cfg):
        """换向 = 先卖后买同日合法（卖出资金当日可用）。"""
        dates = pd.bdate_range("2024-01-02", periods=4)
        eng = BacktestEngine(cfg, {"000001.SZ": _bars(dates, price=10.0)})
        positions = {"000001.SZ": Position("000001.SZ", 1, 5, 10.01, 10.0, dates[0])}
        fills, trades = [], []
        cash = eng._execute(Target("000001.SZ", 0, 0, "flip_out"), dates[1], positions,
                            15000.0, fills, trades)
        assert fills[0].action == "close"
        cash = eng._execute(Target("000001.SZ", 1, 8, "flip_in"), dates[1], positions,
                            cash, fills, trades)
        assert positions["000001.SZ"].lots == 8
        assert fills[1].action == "open"
        assert cash < 15000     # 买 8 手花了更多现金


class TestMetrics:
    def test_drawdown_basic(self):
        eq = pd.Series([100, 120, 90, 95], index=pd.bdate_range("2024-01-02", periods=4))
        mdd, peak, trough = max_drawdown(eq)
        assert mdd == pytest.approx(-0.25)
        assert str(peak.date()) == "2024-01-03"

    def test_metrics_keys(self, cfg):
        dates = pd.bdate_range("2024-01-02", periods=6)
        bars = _bars(dates, price=10.0)
        bars.iloc[3, bars.columns.get_indexer(["open", "high", "low", "close"])] = 12.0
        t = {dates[1]: [Target("000001.SZ", 1, 5, "t_in")],
             dates[4]: [Target("000001.SZ", 0, 0, "t_out")]}
        eng = BacktestEngine(cfg, {"000001.SZ": bars})
        r = eng.run(t, start=dates[0])
        m = compute_metrics(cfg, r.equity, r.trades, r.fills, r.pos_value)
        for k in ("total_return", "ann_return", "sharpe", "max_drawdown", "n_trades",
                  "win_rate", "profit_factor", "total_costs", "position_util_mean",
                  "n_blocked_limit", "n_blocked_cash", "n_blocked_t1", "final_equity"):
            assert k in m, k
        assert m["n_trades"] == 1
        assert m["n_blocked_t1"] == 0
