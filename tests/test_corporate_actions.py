"""除权除息入权益（含权入账）回归钉 —— docs/03 §38 口径变更的行为契约。

旧口径（判据链 #1~#19 全部回测）用未复权价盯市与交收 ⇒ 送转/除息当天持仓市值凭空
蒸发；实测合格截面日收益口径差 +1.5bp/日、年化 +3.3~3.8%，全历史组合层读数系统性
下偏（`backtest_run.py` 里 `bars_hfq` 算了却从没用过，是 dead code，反证本意）。

账户 20,000、单票 1 手×1000 股 ≈ 半仓 ⇒ 一次 10 送 10 的假跳损 = 5,000 = 权益的 25%，
所以断言用 0.99/0.8 两侧分开：新口径不得跳，缺因子列必须退回旧口径且**可观察**。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fterm import contracts
from fterm.backtest.costs import trade_cost
from fterm.backtest.engine import BacktestEngine, Target
from fterm.exec.broker import Order, Side
from fterm.exec.paper import PaperBroker

D = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"])


def _bars(adj_step: float, price_drop: float) -> dict[str, pd.DataFrame]:
    """D1/D2 平（建仓在 D2 开盘），D3 除权：价格 ÷drop、因子 ×step。"""
    close = [10.0, 10.0, 10.0 / price_drop]
    adj = [1.0, 1.0, adj_step]
    df = pd.DataFrame({
        "open": close, "high": [c * 1.02 for c in close], "low": [c * 0.98 for c in close],
        "close": close, "up_limit": [c * 1.1 for c in close],
        "down_limit": [c * 0.9 for c in close], "adj": adj,
    }, index=D)
    return {"000001.SZ": df}


def _run(cfg, bars, lots=10):
    eng = BacktestEngine(cfg, bars)
    # D1 出信号 → D2 开盘建仓（引擎在最后一天不再执行 ⇒ 平仓信号放 D1 之外会丢失）
    tgts = {D[0]: [Target("000001.SZ", 1, lots, "in")]}
    return eng.run(tgts, start=D[0])


class TestEngineCorporateActions:
    def test_bonus_issue_no_phantom_loss(self, cfg):
        r = _run(cfg, _bars(adj_step=2.0, price_drop=2.0))       # 10 送 10
        eq, pv = r.equity, r.pos_value
        assert eq.iloc[1] / eq.iloc[0] > 0.995          # 建仓当日只有费用级摩擦
        assert eq.iloc[2] / eq.iloc[1] > 0.999, f"除权日权益跳损：{eq.to_dict()}"
        assert pv.iloc[2] > 9_900, f"送转后市值必须保住：{pv.to_dict()}"

    def test_cash_dividend_no_phantom_loss(self, cfg):
        step = 1.02040816                                # 除息 2% ⇒ 价格 ÷1.02、因子 ×1.02
        r = _run(cfg, _bars(adj_step=step, price_drop=step))
        assert r.equity.iloc[2] / r.equity.iloc[1] > 0.999, r.equity.to_dict()

    def test_missing_adj_column_falls_back_observably(self, cfg):
        bars = _bars(2.0, 2.0)
        bars["000001.SZ"] = bars["000001.SZ"].drop(columns=["adj"])
        r = _run(cfg, bars)
        ratio_new = _run(cfg, _bars(2.0, 2.0)).equity.iloc[2] / r.equity.iloc[2]
        assert 0.70 < r.equity.iloc[2] / r.equity.iloc[1] < 0.85, \
            f"缺因子列应退回未含权口径（约 −25%）：{r.equity.to_dict()}"
        assert ratio_new > 1.2, "同一数据含权/不含权必须差出一个量级，否则本测无效"

    def test_real_price_space_preserved(self, cfg):
        """成交价/滑点必须留在真实未复权价空间；只有现金与市值按 ratio 兑现。"""
        bars = _bars(2.0, 2.0)["000001.SZ"]
        r = _run(cfg, {"000001.SZ": bars})
        buy = [f for f in r.fills if f.action == "open"][0]
        assert abs(buy.price - (float(bars["open"].iloc[1]) + contracts.TICK)) < 1e-9
        # 现金 = 初始 − 真实成交额 − 建仓费；市值 = 真实收盘 × 股数 × ratio
        assert abs((r.equity.iloc[2] - r.pos_value.iloc[2])
                   - (cfg.account.initial_equity - 10.01 * 1000
                      - trade_cost(cfg, 10.0, 1, 10, D[1]).fee)) < 1e-6
        assert abs(r.pos_value.iloc[2] - 5.0 * 1000 * 2.0) < 1e-6
        assert abs(r.equity.iloc[2] - (r.pos_value.iloc[2]
                                       + (r.equity.iloc[2] - r.pos_value.iloc[2]))) < 1e-9

    def test_close_payout_uses_ratio(self, cfg):
        """D2 开盘建仓 → D3 开盘平仓：到账现金必须是 真实价 × ratio。"""
        eng = BacktestEngine(cfg, _bars(2.0, 2.0))
        r = eng.run({D[0]: [Target("000001.SZ", 1, 10, "in")],
                     D[1]: [Target("000001.SZ", 0, 0, "out")]}, start=D[0])
        sell = [f for f in r.fills if f.action == "close"]
        assert sell, f"应有平仓：{[f.action for f in r.fills]}"
        assert abs(sell[0].price - ((5.0 - contracts.TICK) * 2.0)) < 1e-6, sell[0].price
        assert r.metrics["final_equity"] > cfg.account.initial_equity - 120.0, r.metrics

    def test_faithful_snapshot_value_carries_ratio(self, cfg):
        """忠实快照（governor 的 open_positions.value）必须与 mark 同含权口径。

        pos_value_at 曾是全引擎唯一不带 `_value_ratio` 的估值点：除权后权益按
        ratio 计、忠实快照却按未复权计 ⇒ governor 的现金闸估计系统性偏低
        （2026-10-08 收口批修复）。
        """
        # 4 日：D2 开盘建仓（因子 1.0）→ D3 除权（价格 ÷2、因子 ×2）。注意引擎
        # 在最后一天不问决策 ⇒ 除权日必须是倒数第二天才能被忠实快照看到。
        d4 = D.append(pd.DatetimeIndex(["2024-01-05"]))
        close = [10.0, 10.0, 5.0, 5.0]
        bars = {"000001.SZ": pd.DataFrame({
            "open": close, "high": [c * 1.02 for c in close], "low": [c * 0.98 for c in close],
            "close": close, "up_limit": [c * 1.1 for c in close],
            "down_limit": [c * 0.9 for c in close], "adj": [1.0, 1.0, 2.0, 2.0],
        }, index=d4)}
        eng = BacktestEngine(cfg, bars)
        seen: dict[str, float] = {}

        def propose(ctx):
            if ctx.date == d4[0]:
                return [Target("000001.SZ", 1, 10, "in")]
            if ctx.date == d4[2]:
                seen["value"] = ctx.positions["000001.SZ"]["value"]
            return []

        eng.run({}, start=d4[0], provider=propose, reb_days=set(d4))
        # D4：真实收盘 5.0 × ratio(=2.0) × 1000 股；修复前 = 5.0 × 1000（未含权）
        assert abs(seen["value"] - 10_000.0) < 1e-6, seen


class TestPaperCorporateActions:
    def test_paper_marks_with_ratio(self, cfg):
        pb = PaperBroker(cfg, equity=100_000.0, now=pd.Timestamp("2024-01-03"))
        rep = pb.fill_order_at(Order("000001.SZ", Side.BUY, 10, "t"), 10.0, adj=1.0)
        assert rep.ok
        fee = trade_cost(cfg, 10.0, 1, 10, pd.Timestamp("2024-01-03")).fee
        assert abs(pb.cash - (100_000.0 - 10.01 * 1000 - fee)) < 1e-6
        snap = pb.account(mark_prices={"000001.SZ": 5.0}, adj_prices={"000001.SZ": 2.0})
        assert abs(snap.pos_value - 5.0 * 1000 * 2.0) < 1e-9
        assert snap.available == pb.cash
        assert abs(snap.equity - (snap.available + snap.pos_value)) < 1e-9

    def test_paper_positions_carry_adj_entry(self, cfg):
        pb = PaperBroker(cfg, equity=100_000.0, now=pd.Timestamp("2024-01-03"))
        pb.fill_order_at(Order("000001.SZ", Side.BUY, 10, "t"), 10.0, adj=1.7)
        assert pb.positions["000001.SZ"].adj_entry == 1.7
        c0 = pb.cash
        pb.fill_order_at(Order("000001.SZ", Side.SELL, 10, "t"), 20.0, adj=3.4)
        assert not pb.positions
        # 卖出到账 = (20 − 1tick) × ratio(3.4/1.7=2) × 股数 − 费
        assert pb.cash - c0 > 20.0 * 2 * 1000 - 200.0, (c0, pb.cash)

    def test_state_roundtrip_carries_adj(self, tmp_path):
        from fterm.runtime.state import StateDB
        db = StateDB(tmp_path / "s.db")
        db.save_positions({"000001.SZ": {"lots": 3, "entry_price": 9.9,
                                         "adj_entry": 1.5}})
        got = db.load_positions()
        assert got["000001.SZ"]["adj_entry"] == 1.5
        assert got["000001.SZ"]["adj_missing"] is False
        db.conn.execute("UPDATE positions SET adj_entry = NULL")
        db.conn.commit()
        assert db.load_positions()["000001.SZ"]["adj_missing"] is True   # 缺值可观察，不猜

    def test_legacy_positions_table_gets_column(self, tmp_path):
        """旧库（三列 positions 表）必须就地加列且旧行不丢。"""
        import sqlite3
        from fterm.runtime.state import StateDB
        f = tmp_path / "legacy.db"
        con = sqlite3.connect(f)
        con.executescript("CREATE TABLE positions (code TEXT PRIMARY KEY, lots INTEGER,"
                          " entry_price REAL);"
                          "INSERT INTO positions VALUES ('000001.SZ', 5, 9.9);")
        con.commit()
        con.close()
        db = StateDB(f)
        rows = db.conn.execute("SELECT code, lots, entry_price, adj_entry"
                               " FROM positions").fetchall()
        assert rows == [("000001.SZ", 5, 9.9, None)], rows      # 列已加、历史行保留
        got = db.load_positions()["000001.SZ"]
        assert got["adj_entry"] == 1.0 and got["adj_missing"] is True
