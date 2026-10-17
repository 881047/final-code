"""信号门限 / PaperBroker / StateDB（含期货 legacy 迁移）/ watchdog / 日志可复原性验收。"""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from fterm.backtest.costs import trade_cost
from fterm.config import Config, load_config
from fterm.exec.broker import Order, Side
from fterm.exec.paper import PaperBroker
from fterm.runtime.state import StateDB
from fterm.runtime.watchdog import check_once
from fterm.strategy.signal import SignalProvider
from tests.conftest import _yaml_text


@pytest.fixture
def cfg(tmp_path) -> Config:
    from tests.conftest import write_synthetic_db
    # 本文件 fixture 覆盖 conftest 的 cfg（old_db 占位 "x"）——必须同时把合成库
    # 建在同一路径，依赖 cfg 的 env fixture 才能完成同步。
    write_synthetic_db(tmp_path / "old.db")
    p = tmp_path / "c.yaml"
    p.write_text(_yaml_text().replace("{old_db}", str(tmp_path / "old.db").replace("\\", "/"))
                 .replace("{root}", str(tmp_path / "d").replace("\\", "/")),
                 encoding="utf-8")
    return load_config(p)


def _pred(p_up=0.5, std=0.0, q_lo=-0.01, q_hi=0.01):
    return {"p_up": p_up, "p_up_std": std, "q_lo": q_lo, "q_hi": q_hi, "vol_pred": 0.02}


class TestSignal:
    def test_long_only_gate(self, cfg):
        sp = SignalProvider(cfg, gate=0.06)
        assert sp.decide("000001.SZ", _pred(p_up=0.60)).direction == 1
        d = sp.decide("000001.SZ", _pred(p_up=0.52))
        assert d.direction == 0 and d.reason == "neutral"
        # 现货无做空：低概率也只能观望，不允许 -1
        assert sp.decide("000001.SZ", _pred(p_up=0.30)).direction == 0

    def test_seed_disagreement_forces_flat(self, cfg):
        sp = SignalProvider(cfg, std_gate=0.12)
        d = sp.decide("000001.SZ", _pred(p_up=0.90, std=0.2))
        assert d.direction == 0
        assert "seed_disagreement" in d.reason

    def test_wide_quantile_raises_gate(self, cfg):
        sp = SignalProvider(cfg, gate=0.06, q_width_gate=0.06)
        d = sp.decide("000001.SZ", _pred(p_up=0.55, q_lo=-0.03, q_hi=0.03))
        assert d.direction == 0                      # 0.55 < 0.5 + 0.06 + 0.06
        assert "wide_q" in d.reason


class TestPaperBroker:
    def test_round_trip_costs_match_backtest(self, cfg):
        now = pd.Timestamp("2024-01-05")
        pb = PaperBroker(cfg, equity=20000, now=now)
        pb.bind_price_feed(lambda code: 10.0)
        pb.submit(Order("000001.SZ", Side.BUY, 5, "t_in"))
        pb.execute_pending({"000001.SZ": 10.0})
        pb.submit(Order("000001.SZ", Side.SELL, 5, "t_out"))
        pb.execute_pending({"000001.SZ": 10.0})
        buy_fee = pb.fills[0].fee
        sell_fee = pb.fills[1].fee
        exp_buy = trade_cost(cfg, 10.0, 1, 5, now).fee
        exp_sell = trade_cost(cfg, 10.0, -1, 5, now).fee
        assert buy_fee == pytest.approx(exp_buy)
        assert sell_fee == pytest.approx(exp_sell)
        # 平仓后回到现金：权益 = 本金 − 全部费用 − 双边滑点（买 10.01 / 卖 9.99 ⇒ 0.02×500 股）
        snap = pb.account({"000001.SZ": 10.0})
        assert snap.equity == pytest.approx(20000 - buy_fee - sell_fee - 0.02 * 500, rel=1e-9)

    def test_snapshot_value_semantics(self, cfg):
        pb = PaperBroker(cfg, equity=20000, now=pd.Timestamp("2024-01-05"))
        pb.submit(Order("000001.SZ", Side.BUY, 5, "t"))
        pb.execute_pending({"000001.SZ": 10.0})
        snap = pb.account({"000001.SZ": 12.0})
        assert snap.pos_value == pytest.approx(500 * 12.0)
        assert snap.equity == pytest.approx(snap.available + snap.pos_value)
        assert snap.positions["000001.SZ"]["lots"] == 5

    def test_pending_cancel(self, cfg):
        pb = PaperBroker(cfg, equity=20000)
        pb.submit(Order("000001.SZ", Side.BUY, 5, "t"))
        assert pb.cancel_all() == 1
        assert pb.execute_pending({"000001.SZ": 10.0}) == []

    def test_short_side_absent(self, cfg):
        # 期货版的 OPEN_SHORT/CLOSE_SHORT 不复存在
        assert not hasattr(Side, "OPEN_SHORT") and not hasattr(Side, "CLOSE_SHORT")

    def test_sell_without_position_hard_fails(self, cfg):
        # 2026-10-06 审查修复：无持仓卖出原实现静默空操作却照报成交 → 硬拦
        pb = PaperBroker(cfg, equity=20000, now=pd.Timestamp("2024-01-05"))
        rep = pb.fill_order_at(Order("000001.SZ", Side.SELL, 5, "t"), 10.0)
        assert not rep.ok and rep.msg == "blocked_no_position"
        assert "000001.SZ" not in pb.positions
        assert pb.cash == pytest.approx(20000)

    def test_oversell_hard_fails(self, cfg):
        # 2026-10-06 审查修复：超卖（held < order）原实现按持仓截断并删仓、报告写满额 → 硬拦
        pb = PaperBroker(cfg, equity=20000, now=pd.Timestamp("2024-01-05"))
        pb.fill_order_at(Order("000001.SZ", Side.BUY, 5, "t_in"), 10.0)
        rep = pb.fill_order_at(Order("000001.SZ", Side.SELL, 8, "t_out"), 10.0)
        assert not rep.ok and "oversell" in rep.msg
        assert pb.positions["000001.SZ"].lots == 5      # 持仓不被非法截断


class TestStateDB:
    def test_roundtrip(self, tmp_path):
        db = StateDB(tmp_path / "s.db")
        db.record_equity("2024-01-05", 20100.0, 10000.0, 10100.0)
        db.record_fill("t1", "000001.SZ", "buy", 5, 10.01, 5.35, "oid1")
        db.record_decisions("2024-01-05", [("000001.SZ", 1, 5, True, "rule_reversal(rank=0.9)")])
        assert db.equity_curve() == [("2024-01-05", 20100.0, 10000.0, 10100.0)]
        assert db.stats()["fills"] == 1

    def test_idempotent_writes(self, tmp_path):
        db = StateDB(tmp_path / "s.db")
        db.record_equity("2024-01-05", 1.0, 0.0, 1.0)
        db.record_equity("2024-01-05", 2.0, 0.0, 2.0)
        assert len(db.equity_curve()) == 1
        assert db.equity_curve()[0][1] == 2.0

    def test_futures_legacy_migration(self, tmp_path):
        """期货版 schema（fills 带 contract / equity_curve 带 margin_used）→ 就地改名留档。"""
        p = tmp_path / "legacy.db"
        con = sqlite3.connect(p)
        con.executescript("""
        CREATE TABLE fills (ts TEXT, product TEXT, contract TEXT, side TEXT, lots INTEGER,
            price REAL, fee REAL, order_id TEXT, PRIMARY KEY (ts, product, side, lots));
        CREATE TABLE equity_curve (
            date TEXT PRIMARY KEY, equity REAL, margin_used REAL, cash REAL);
        """)
        con.execute("INSERT INTO fills VALUES ('2026-09-26','C','C2601','open_long',1,2301,1.3,'x')")
        con.execute("INSERT INTO equity_curve VALUES ('2026-09-26',20000,2300,17700)")
        con.commit()
        con.close()
        db = StateDB(p)                      # 迁移发生在这里
        db.record_fill("t1", "000001.SZ", "buy", 5, 10.01, 5.35, "oid1")   # 新 schema 可写
        legacy = db.conn.execute(
            "SELECT product, contract FROM fills_legacy_futures").fetchall()
        assert legacy == [("C", "C2601")]    # 期货历史行保留（审计留痕）
        assert db.conn.execute("SELECT margin_used FROM equity_curve_legacy_futures").fetchone()[0] == 2300


class TestWatchdog:
    def test_fresh_db_reports_missing_heartbeat(self, cfg, tmp_path):
        StateDB(tmp_path / "s.db")
        issues = check_once(cfg, now=pd.Timestamp.now())
        assert any("从未心跳" in i for i in issues)

    def test_stale_heartbeat_alerts(self, cfg, tmp_path):
        db = StateDB(cfg.state_db_path)
        stale = (pd.Timestamp.now() - pd.Timedelta(hours=1)).isoformat()
        db.heartbeat(stale, "old")
        issues = check_once(cfg, now=pd.Timestamp.now())
        assert any("无心跳" in i for i in issues)
        assert db.recent_alerts()


class TestLogRecoverability:
    """H1 纪律继承：决策 reason 必须能复原判据原始读数（前瞻验证要从日志复原方向）。"""

    def test_reversal_reason_carries_rank(self, env):
        cfg, panels, cal = env
        from fterm.features.build import build_dataset
        from fterm.runtime.runner import DailyRunner
        from fterm.exec.paper import PaperBroker
        from fterm.runtime.state import StateDB
        bundle = build_dataset(cfg)
        asof = [d for d in bundle.features["rev_2"].index if d.weekday() == 4][-1]
        broker = PaperBroker(cfg, equity=20000, now=asof)
        runner = DailyRunner(cfg, broker, StateDB(cfg.state_db_path))
        summary = runner.run_once(asof=asof)
        for d in summary["decisions"]:
            if d["direction"] == 1:
                assert "rank=" in d["reason"]          # 原始秩可复原
            if d["reason"].startswith("not_rebalance"):
                raise AssertionError("asof 应是周五（rebalance_weekday=4）")
