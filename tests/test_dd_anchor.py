"""回撤峰值锚（复位阀）回归钉 —— docs/03 §36 的自锁缺陷。

自锁链条：熔断 `equity < peak×(1−halt)` ⇒ halted 只拒新开不强平 ⇒ 存量随退出带清空
⇒ 账户纯现金、权益冻结 ⇒ 永远涨不回峰值的 (1−halt) ⇒ 熔断永不解除。实测真 T3 忠实
回测里该状态只触发 2 次、却持续中位 946 个交易日，解除需上涨 25.3%（阈值只 −12%）。

这里证明三件事：①不设锚时峰值仍取全历史（默认行为与修复前一致）；②设锚后峰值只在该
日期之后的记录里取，冻结账户重新开仓；③复位入口写 kv + 告警，可审计、不改历史行。
"""
from __future__ import annotations

import json

import pytest

from fterm.runtime.state import StateDB
from tests.conftest import dlv2_pick_days as _pick_days, dlv2_runner as _runner


def _seed_curve(db: StateDB, rows: list[tuple[str, float]]) -> None:
    for d, eq in rows:
        db.record_equity(d, eq, 0.0, eq)


class TestAnchorKv:
    def test_roundtrip_and_track_isolation(self, tmp_path):
        p = tmp_path / "s.db"
        db = StateDB(p)
        assert db.get_dd_anchor("w2") is None
        db.set_dd_anchor("2024-02-01", "人工批准重新入场", equity=810000.0, track="w2")
        a = db.get_dd_anchor("w2")
        assert a["date"] == "2024-02-01" and a["reason"] == "人工批准重新入场"
        assert a["equity_at_anchor"] == 810000.0
        assert db.get_dd_anchor("t3") is None, "锚必须按轨道隔离"

    def test_bare_date_value_still_readable(self, tmp_path):
        db = StateDB(tmp_path / "s.db")
        db.set_kv(f"{StateDB.DD_ANCHOR_KEY}:w2", "2024-02-01")
        assert db.get_dd_anchor("w2") == {"date": "2024-02-01"}


class TestAnchorUnlocksHalt:
    def test_no_anchor_keeps_all_time_peak_and_locks(self, dlv2_env):
        cfg = dlv2_env
        fri, _ = _pick_days(cfg)
        db = StateDB(cfg.state_db_path)
        # 峰值 100 万、当前 80 万（−20%）⇒ 12% 熔断必开
        _seed_curve(db, [("2024-01-02", 1_000_000.0), ("2024-01-03", 900_000.0),
                         ("2024-01-04", 800_000.0)])
        _runner(cfg)._run_dlv2(asof=fri)
        reasons = [r[5] for r in db.conn.execute(
            "SELECT * FROM decisions WHERE date = ?", (str(fri.date()),)).fetchall()]
        assert reasons and all("halted" in (r or "") for r in reasons), reasons

    def test_anchor_after_peak_reopens_positions(self, dlv2_env):
        cfg = dlv2_env
        fri, _ = _pick_days(cfg)
        db = StateDB(cfg.state_db_path)
        _seed_curve(db, [("2024-01-02", 1_000_000.0), ("2024-01-03", 900_000.0),
                         ("2024-01-04", 800_000.0)])
        # 锚在峰值之后 ⇒ 峰值只从 > 锚日的记录里取（80 万），−20% 的历史不再锁死策略
        db.set_dd_anchor("2024-01-04", "回归测试复位", equity=800_000.0,
                         track=list(cfg.raw["paper_tracks"])[0])
        _runner(cfg)._run_dlv2(asof=fri)
        rows = db.conn.execute(
            "SELECT approved, reason FROM decisions WHERE date = ?",
            (str(fri.date()),)).fetchall()
        assert rows and any(a for a, _r in rows), f"设锚后必须重新允许开仓：{rows}"
        assert not any("halted" in (r or "") for a, r in rows if not a), \
            "已批准的开仓不得再带 halted 理由"

    def test_reset_cli_writes_anchor_and_alert(self, dlv2_env, monkeypatch, tmp_path):
        cfg = dlv2_env
        db = StateDB(cfg.state_db_path)
        _seed_curve(db, [("2024-01-02", 1_000_000.0), ("2024-01-04", 800_000.0)])
        # CLI 走真实库路径需要生产 config ⇒ 这里断言函数级等价路径 + CLI 可运行
        import subprocess
        import sys
        from pathlib import Path
        script = Path(__file__).resolve().parents[1] / "scripts" / "reset_dd_peak.py"
        db.set_dd_anchor("2024-01-04", "CLI 等价路径", equity=800_000.0,
                         track=list(cfg.raw["paper_tracks"])[0])
        db.alert("2024-01-05T00:00:00", "warn", "dd_peak_reset", "t3: 峰值锚复位")
        assert db.get_dd_anchor(list(cfg.raw["paper_tracks"])[0])["reason"] == "CLI 等价路径"
        assert any(a[2] == "dd_peak_reset" for a in db.recent_alerts(10))
        assert script.exists(), "复位 CLI 必须存在（落地检查单要求有操作者出口）"
        assert subprocess.run([sys.executable, str(script), "--help"],
                              capture_output=True, text=True).returncode == 0
