"""dlv2 生产位 paper 撮合审计（2026-10-05）：开盘价成交 / 封板弃单 / 同日重跑幂等 /
调仓频率闸 / 轨道级 top/keep / 成交落库。

artifact 用 monkeypatch 注入假预测（predict_day ⇒ 按代码字典序递减的固定秩，绕过 .pt
工件与 digest 守卫）；合成面板来自 conftest env（3 只沪深股 40 个交易日，bdate_range
⇒ 周一~周五连续）。封板场景直接改写本仓面板 parquet（stk_limit 真值口径）。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fterm.data import ashare as sa
from fterm.exec.broker import Order, Side
from fterm.exec.paper import PaperBroker
from fterm.features.build import build_dataset
from fterm.runtime.runner import DailyRunner
from fterm.runtime.state import StateDB, fill_cash_delta
from tests.conftest import dlv2_pick_days as _pick_days, dlv2_runner as _runner


def test_friday_generates_pending_with_created(dlv2_env):
    cfg = dlv2_env
    fri, _ = _pick_days(cfg)
    res = _runner(cfg)._run_dlv2(asof=fri)
    assert res["mode"] == "dlv2" and res["tracks"]
    db = StateDB(cfg.runtime.state_db)
    pend = db.load_pending()
    assert pend, "调仓日（周五）必须生成挂单"
    assert all(p["created"] == str(fri.date()) for p in pend)
    # 轨道级 top/keep：w2 top_n=1 ⇒ 恰好 1 张买单（假预测第一名 = 字典序最小）
    assert len(pend) == 1 and pend[0]["side"] == "buy"
    assert pend[0]["code"] == "000001.SZ"


def test_next_day_fills_at_open_and_records(dlv2_env):
    cfg = dlv2_env
    fri, mon = _pick_days(cfg)
    r = _runner(cfg)
    r._run_dlv2(asof=fri)
    db = StateDB(cfg.runtime.state_db)
    r._run_dlv2(asof=mon)
    fills = db.recent_fills(50)
    assert fills, "次日开盘必须有成交落库"
    assert all(f[0] == str(mon.date()) for f in fills)
    op = sa.load_panel(cfg, "open")
    for _ts, prod, side, lots, price, _fee in fills:
        if side == "buy":
            # 成交价 = 开盘价 + 买向 1 tick 滑点 —— 原 bug 是误用 close 价
            assert abs(price - (float(op.at[mon, prod]) + 0.01)) < 1e-6
    # 同日重跑幂等：不再产生新成交、权益行不重复
    n_fills, n_curve = len(db.recent_fills(50)), len(db.equity_curve())
    r._run_dlv2(asof=mon)
    assert len(db.recent_fills(50)) == n_fills
    assert len(db.equity_curve()) == n_curve


def test_non_rebalance_day_no_new_pending(dlv2_env):
    cfg = dlv2_env
    fri, mon = _pick_days(cfg)
    r = _runner(cfg)
    r._run_dlv2(asof=fri)
    db = StateDB(cfg.runtime.state_db)
    assert db.load_pending()
    r._run_dlv2(asof=mon)                    # 周一：成交 + 清空，且不重排
    assert db.load_pending() == []
    dec_dates = {row[0] for row in db.conn.execute(
        "SELECT DISTINCT date FROM decisions").fetchall()}
    assert dec_dates == {str(fri.date())}, "仅调仓日记录决策"


def test_limit_block_drops_buy_and_alerts(dlv2_env):
    cfg = dlv2_env
    fri, mon = _pick_days(cfg)
    # 构造周一 000001.SZ 全天封死涨停：low = up_limit（一字板，买不进）
    up = sa.load_panel(cfg, "up_limit")
    low = sa.load_panel(cfg, "low")
    op = sa.load_panel(cfg, "open")
    code = "000001.SZ"
    up.at[mon, code] = float(up.at[mon, code])          # 保持真值
    low.at[mon, code] = float(up.at[mon, code])         # low >= up ⇒ 全天封死
    op.at[mon, code] = float(up.at[mon, code])          # 开盘即封
    pdir = Path(cfg.data.root) / "panel"
    up.to_parquet(pdir / "up_limit.parquet")
    low.to_parquet(pdir / "low.parquet")
    op.to_parquet(pdir / "open.parquet")

    r = _runner(cfg)
    r._run_dlv2(asof=fri)
    db = StateDB(cfg.runtime.state_db)
    assert [p["code"] for p in db.load_pending()] == [code]
    r._run_dlv2(asof=mon)
    assert code not in db.load_positions(), "封死涨停买单必须被弃"
    assert db.load_pending() == []
    alerts = db.recent_alerts(5)
    assert any(a[2] == "fill_blocked" for a in alerts), "弃单必须进告警表可观察"
    assert db.recent_fills(50) == [], "唯一买单被封 ⇒ 无成交"


def test_tracks_isolated(dlv2_env):
    cfg = dlv2_env
    fri, _ = _pick_days(cfg)
    _runner(cfg)._run_dlv2(asof=fri)
    w2 = StateDB(cfg.runtime.state_db)
    t3 = StateDB(Path(cfg.data.root) / "state" / "t3.db")
    pw, pt = w2.load_pending(), t3.load_pending()
    assert pw and pt, "两条轨道各自生成挂单"
    assert len(pt) == 2 and len(pw) == 1          # top_n=2 vs top_n=1
    assert all(p["created"] == str(fri.date()) for p in pt)
    assert (Path(cfg.data.root) / "state" / "t3.db").exists(), "t3 轨独立库文件"


# ---------------------------------------------------------------------------
# 现金持久化（2026-10-08 P0-A 修复的回归钉）
# 修前缺陷：runner 每日用 `tr["equity"]`（初始本金）重建 PaperBroker，只从库里
# 恢复持仓不恢复现金 ⇒ 现金与持仓市值双计，持仓成交次日起纸面权益 ≈ 本金 + 市值
# （w2 顶满仓 ≈ 2 倍本金），前瞻窗主判据/governor peak 与日停/sizer 风险预算全失真。
# ---------------------------------------------------------------------------

def _next_day(cfg, d, k=1):
    dates = build_dataset(cfg).features[f"rev_{cfg.signal.rev_window}"].index
    return dates[dates.get_loc(d) + k]


def _curve_map(db):
    return {r[0]: (round(r[1], 6), round(r[2], 6), round(r[3], 6))
            for r in db.conn.execute(
                "SELECT date, equity, pos_value, cash FROM equity_curve")}


def test_cash_persists_across_days(dlv2_env):
    cfg = dlv2_env
    fri, mon = _pick_days(cfg)
    tue = _next_day(cfg, mon)
    r = _runner(cfg)
    r._run_dlv2(asof=fri)
    r._run_dlv2(asof=mon)
    r._run_dlv2(asof=tue)
    db = StateDB(cfg.runtime.state_db)
    row = [x for x in db.equity_curve() if x[0] == str(tue.date())]
    assert row, "非调仓日照常记录权益"
    _d, equity, pos_value, cash = row[0]
    assert abs(equity - (cash + pos_value)) < 1e-6, "权益恒等式 equity = cash + pos_value"
    assert pos_value > 0, "周一建仓后周二应仍有市值"
    assert cash < 20000 - 0.5 * pos_value, "买入占用的现金必须真的从现金里扣掉"
    assert equity < 1.2 * 20000, f"修前此处 ≈ 本金+市值 ≈ {20000 + 2 * pos_value:.0f}（虚增）"


def test_cash_rerun_idempotent(dlv2_env):
    cfg = dlv2_env
    fri, mon = _pick_days(cfg)
    r = _runner(cfg)
    r._run_dlv2(asof=fri)
    r._run_dlv2(asof=mon)
    db = StateDB(cfg.runtime.state_db)
    curve1, fills1 = _curve_map(db), db.recent_fills(50)
    r._run_dlv2(asof=mon)                     # 同日重跑（计划任务"修复后重跑即幂等"）
    assert _curve_map(db) == curve1, "同日重跑不得改动已落账权益/现金"
    assert db.recent_fills(50) == fills1, "同日重跑不得重复撮合"


def test_cash_recovery_survives_missing_today_row(dlv2_env):
    """崩溃模拟：成交已落库但当日权益行未写 ⇒ 重跑必须按 fills 重放现金，而非重演成交。"""
    cfg = dlv2_env
    fri, mon = _pick_days(cfg)
    r = _runner(cfg)
    r._run_dlv2(asof=fri)
    r._run_dlv2(asof=mon)
    db = StateDB(cfg.runtime.state_db)
    fills = db.fills_on(str(mon.date()))
    assert fills, "前置：周一确有成交"
    cash_expected = 20000 + sum(fill_cash_delta(s, l, p, f) for _c, s, l, p, f in fills)
    db.conn.execute("DELETE FROM equity_curve WHERE date = ?", (str(mon.date()),))
    db.conn.commit()
    r._run_dlv2(asof=mon)
    row = [x for x in db.equity_curve() if x[0] == str(mon.date())]
    assert row and abs(row[0][3] - cash_expected) < 1e-6, "现金须由 fills 表精确重放"
    assert len(db.fills_on(str(mon.date()))) == len(fills), "重放不得追加成交"


def test_fill_cash_delta_matches_broker_cash(cfg):
    """重放公式与 `PaperBroker._fill` 的现金算术逐位同源（单一口径承诺）。"""
    now = pd.Timestamp("2024-01-05")
    pb = PaperBroker(cfg, equity=20000, now=now)
    buy = pb.fill_order_at(Order("000001.SZ", Side.BUY, 2, "t"), 10.0)
    assert buy.ok
    cash_after_buy = pb.cash
    assert abs(cash_after_buy
               - (20000 + fill_cash_delta("buy", 2, buy.price, buy.fee))) < 1e-9
    sell = pb.fill_order_at(Order("000001.SZ", Side.SELL, 2, "t"), 11.0)
    assert sell.ok
    assert abs(pb.cash
               - (cash_after_buy + fill_cash_delta("sell", 2, sell.price, sell.fee))) < 1e-9


# ---------------------------------------------------------------------------
# 秩平滑 EMA 语义（2026-10-08 入配置 + 换版守卫）
# ---------------------------------------------------------------------------

def _set_ema_step(cfg, step):
    import dataclasses
    return dataclasses.replace(cfg, signal=dataclasses.replace(cfg.signal, ema_step=step))


def test_ema_step_daily_vs_rebalance(dlv2_env):
    """daily 每日步进（生产现用）vs rebalance 只在调仓日步进（判据链构型）。"""
    fri, mon = _pick_days(dlv2_env)

    def ema_asof_after_monday(cfg):
        r = _runner(cfg)
        r._run_dlv2(asof=fri)
        r._run_dlv2(asof=mon)
        return StateDB(cfg.state_db_path).get_kv("ema_asof")

    assert ema_asof_after_monday(_set_ema_step(dlv2_env, "daily")) == str(mon.date()), \
        "daily：非调仓日也必须步进平滑"
    assert ema_asof_after_monday(_set_ema_step(dlv2_env, "rebalance")) == str(fri.date()), \
        "rebalance：非调仓日不得步进（链上被采纳构型的口径）"


def test_ema_reset_on_artifact_version_change(dlv2_env):
    """换版 ⇒ 旧工件的残余秩必须清零（新旧两版工件的秩尺度不可比）。"""
    cfg = dlv2_env
    db = StateDB(cfg.state_db_path)
    db.set_kv("ema", json.dumps({"999999.SZ": 0.97}))
    db.set_kv("ema_asof", "2024-01-04")
    db.set_kv("ema_model_version", "0" * 64)        # 与磁盘上的工件 sha 不符
    r = _runner(cfg)
    r._run_dlv2(asof=_pick_days(cfg)[0])
    ema = json.loads(db.get_kv("ema"))
    assert "999999.SZ" not in ema, "换版后旧平滑残余必须清空"
    assert any(a[2] == "ema_reset" for a in db.recent_alerts(20)), "换版清零须可观察"


def test_artifact_version_mismatch_alerts(dlv2_env):
    """账本登记的 model_version 与实际加载工件不一致 ⇒ 响亮告警（不静默换信号源）。"""
    cfg = dlv2_env
    db = StateDB(cfg.state_db_path)
    db.set_kv("model_version", json.dumps({"sha256": "1" * 64, "chain": "#14"}))
    _runner(cfg)._run_dlv2(asof=_pick_days(cfg)[0])
    assert any(a[2] == "artifact_version_mismatch" for a in db.recent_alerts(20))


def test_ema_state_lives_in_main_ledger(dlv2_env):
    """EMA 状态库 = runtime.state_db（原实现取 tracks 字典第一个值，依赖 YAML 键序）。"""
    cfg = dlv2_env
    other = dict(cfg.raw["paper_tracks"])
    order = ["t3", "w2"] if list(other)[0] == "w2" else ["w2", "t3"]
    cfg.raw["paper_tracks"] = {k: other[k] for k in order}   # 反转轨道顺序
    assert list(cfg.raw["paper_tracks"])[0] != "w2"
    _runner(cfg)._run_dlv2(asof=_pick_days(cfg)[0])
    assert StateDB(cfg.state_db_path).get_kv("ema_asof"), "EMA 必须落在主账本而非首轨道库"
