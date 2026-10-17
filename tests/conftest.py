"""共享 fixture：合成老库（neweco.db 只读契约的离线替身）+ 配置工厂。

纪律：测试禁真实网络、禁真实老库——数据源测试全部走合成 SQLite。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fterm.config import Config, load_config  # noqa: E402
from fterm.data import ashare as sa  # noqa: E402

CODES = ["000001.SZ", "600000.SH", "000002.SZ"]
BJ_CODE = "920202.BJ"


def _yaml_text(**over) -> str:
    d = dict(
        initial_equity=20000, max_cash_util=0.95, risk_per_trade=0.02, atr_k=1.2,
        max_pos_vol_daily=0.025, daily_stop_loss=0.03, total_dd_halt=0.12,
        circuit_force_liquidate=False, max_lots_per_symbol=10, max_symbols=3,
        lookback=10, momentum_windows=[2, 3], vol_windows=[2], label_horizon=2,
        turnover_window=3, max_stocks_per_day=0,
        d_hidden=8, n_layers=1, dropout=0.1, lr=0.003, max_epochs=3, patience=2,
        seeds=[7, 21], batch_size=64,
        top_n=1, exit_buffer=2, rev_window=2, quiet_vol=False, rebalance_weekday=4,
        ema_alpha=0.5, ema_step="daily",
    )
    d.update(over)
    return f"""
account:
  initial_equity: {d['initial_equity']}
  max_cash_util: {d['max_cash_util']}
  risk_per_trade: {d['risk_per_trade']}
  atr_k: {d['atr_k']}
  max_pos_vol_daily: {d['max_pos_vol_daily']}
  daily_stop_loss: {d['daily_stop_loss']}
  total_dd_halt: {d['total_dd_halt']}
  circuit_force_liquidate: {str(d['circuit_force_liquidate']).lower()}
  max_lots_per_symbol: {d['max_lots_per_symbol']}
  max_symbols: {d['max_symbols']}
universe:
  mode: fixed
  codes: {list(CODES)}
  exclude_bj: true
  exclude_st: true
  min_list_days: 5
  min_circ_mv_wan: 10000
  min_price: 2.0
  max_universe: 0
data:
  old_db: "{{old_db}}"
  root: "{{root}}"
  start: "2024-01-01"
  calendar_tz: "Asia/Shanghai"
features:
  lookback: {d['lookback']}
  momentum_windows: {d['momentum_windows']}
  vol_windows: {d['vol_windows']}
  label_horizon: {d['label_horizon']}
  label_quantiles: [0.1, 0.9]
  turnover_window: {d['turnover_window']}
  max_stocks_per_day: {d['max_stocks_per_day']}
model:
  name: trihead_gru
  d_hidden: {d['d_hidden']}
  n_layers: {d['n_layers']}
  dropout: {d['dropout']}
  lr: {d['lr']}
  max_epochs: {d['max_epochs']}
  patience: {d['patience']}
  seeds: {d['seeds']}
  batch_size: {d['batch_size']}
  revin: false
  sample_weight: none
  task: direction
backtest:
  slippage_ticks: 1
  signal_exec_lag: 1
  limit_updown_block: true
runtime:
  state_db: "{{root}}/state/fterm.db"
  heartbeat_minutes: 10
  train_weekday: 5
signal:
  mode: rule_reversal
  top_n: {d['top_n']}
  exit_buffer: {d['exit_buffer']}
  rev_window: {d['rev_window']}
  quiet_vol: {str(d['quiet_vol']).lower()}
  rebalance_weekday: {d['rebalance_weekday']}
  ema_alpha: {d['ema_alpha']}
  ema_step: {d['ema_step']}
notify:
  email_to: ""
"""


@pytest.fixture
def cfg(tmp_path) -> Config:
    p = tmp_path / "config.yaml"
    old_db = str(tmp_path / "old.db").replace("\\", "/")
    root = str(tmp_path / "data").replace("\\", "/")
    p.write_text(_yaml_text().replace("{old_db}", old_db).replace("{root}", root),
                 encoding="utf-8")
    return load_config(p)


def write_synthetic_db(path: Path, n_days: int = 40, st_code: str | None = "000002.SZ"):
    """老库最小替身：3 只沪深股 × n_days 日线 + ST 更名段 + 1 只北交所股。"""
    if path.exists():
        path.unlink()          # 幂等：cfg fixture 与 env fixture 可能各建一次
    rng = np.random.RandomState(0)
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    con = sqlite3.connect(path)
    con.executescript("""
    CREATE TABLE stock_basic (ts_code TEXT, symbol TEXT, name TEXT, area TEXT,
        industry TEXT, market TEXT, list_date TEXT, delist_date TEXT);
    CREATE TABLE trade_cal (cal_date TEXT, is_open INTEGER);
    CREATE TABLE daily (ts_code TEXT, trade_date TEXT, open REAL, high REAL, low REAL,
        close REAL, pre_close REAL, change REAL, pct_chg REAL, vol REAL, amount REAL);
    CREATE TABLE adj_factor (ts_code TEXT, trade_date TEXT, adj_factor REAL);
    CREATE TABLE daily_basic (ts_code TEXT, trade_date TEXT, turnover_rate REAL,
        turnover_rate_f REAL, volume_ratio REAL, pe REAL, pe_ttm REAL, pb REAL,
        dv_ratio REAL, dv_ttm REAL, total_mv REAL, circ_mv REAL);
    CREATE TABLE stk_limit (trade_date TEXT, ts_code TEXT, up_limit REAL, down_limit REAL);
    CREATE TABLE namechange (ts_code TEXT, name TEXT, start_date TEXT, end_date TEXT,
        ann_date TEXT, change_reason TEXT);
    """)
    basics = [(c, c[:6], f"股{i}", "深圳" if c.endswith("SZ") else "上海",
               "银行" if i == 0 else "地产", "主板", "19990101", None)
              for i, c in enumerate(CODES)]
    basics.append((BJ_CODE, BJ_CODE[:6], "安达股份", "北京", "汽车配件", "北交所",
                   "20260929", None))
    con.executemany("INSERT INTO stock_basic VALUES (?,?,?,?,?,?,?,?)", basics)
    con.executemany("INSERT INTO trade_cal VALUES (?,1)",
                    [(d.strftime("%Y%m%d"),) for d in dates])
    for i, code in enumerate(CODES):
        base = 10.0 + 5 * i
        px = base * np.exp(np.cumsum(rng.normal(0.0005, 0.012, n_days)))
        for j, d in enumerate(dates):
            o = px[j] * (1 + rng.normal(0, 0.003))
            h = max(o, px[j]) * (1 + abs(rng.normal(0, 0.004)))
            l = min(o, px[j]) * (1 - abs(rng.normal(0, 0.004)))
            c = float(px[j])
            ds = d.strftime("%Y%m%d")
            con.execute("INSERT INTO daily VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (code, ds, round(o, 2), round(h, 2), round(l, 2), round(c, 2),
                         round(px[j - 1], 2) if j else round(c, 2), 0.0, 0.0,
                         100000.0, 1e6))
            con.execute("INSERT INTO adj_factor VALUES (?,?,?)", (code, ds, 1.0 + 0.001 * j))
            con.execute("INSERT INTO daily_basic VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        (code, ds, 1.5 + 0.1 * i, 1.6, 1.0, 8.0, 8.1, 1.0, 2.0, 2.0,
                         500000.0, 300000.0 + 10000 * i))
            con.execute("INSERT INTO stk_limit VALUES (?,?,?,?)",
                        (ds, code, round(c * 1.1, 2), round(c * 0.9, 2)))
    if st_code:
        mid = dates[n_days // 2].strftime("%Y%m%d")
        con.execute("INSERT INTO namechange VALUES (?,?,?,?,?,?)",
                    (st_code, "*ST测试", mid, None, mid, "风险警示"))
    con.commit()
    con.close()
    return dates


def sync_env(cfg: Config):
    """对 cfg 指向的合成老库跑一次面板同步，返回 (panels, trade_cal)。"""
    results = sa.sync_panels(cfg)
    assert all(not v.startswith("FAIL") for v in results.values()), results
    return sa.load_panels(cfg), sa.load_trade_cal(cfg)


@pytest.fixture
def env(tmp_path, cfg):
    """合成老库 + 已同步面板。"""
    write_synthetic_db(tmp_path / "old.db")
    panels, cal = sync_env(cfg)
    return cfg, panels, cal


@pytest.fixture
def dlv2_env(env, monkeypatch):
    """dlv2 生产位的双轨纸面环境：假工件文件 + 两条测试轨道 + 假预测注入。

    2026-10-08 从 `tests/test_dlv2_paper.py` 上移到此处 —— `test_dd_anchor.py` 也需要
    同一环境，跨测试模块私有 fixture 不可见（复制一份就会分叉，同 run_two_pass 的教训）。
    """
    import fterm.model.infer as infer_mod
    cfg, panels, cal = env
    (Path(cfg.data.root) / "models").mkdir(parents=True, exist_ok=True)
    (Path(cfg.data.root) / "models" / "dlv2_production.pt").touch()   # 走主路径（非 skeleton）
    cfg.raw["paper_tracks"] = {
        "w2": {"equity": 20000, "top_n": 1, "keep_n": 2, "max_symbols": 2,
               "max_lots": 10, "state_db": str(cfg.runtime.state_db)},
        "t3": {"equity": 100000, "top_n": 2, "keep_n": 4, "max_symbols": 3,
               "max_lots": 20, "state_db": str(Path(cfg.data.root) / "state" / "t3.db")},
    }

    class _FakeArtifact:
        """predict_day ⇒ 按代码字典序给递减分（第一名可预期 = 字典序最小的代码）。"""

        def predict_day(self, feats, eligible, asof):
            codes = sorted(c for c in eligible.index if bool(eligible[c]))
            return pd.Series({c: 1.0 - i / max(1, len(codes)) for i, c in enumerate(codes)})

    monkeypatch.setattr(infer_mod, "load_artifact", lambda path: _FakeArtifact())
    return cfg


def dlv2_runner(cfg):
    from fterm.exec.paper import PaperBroker
    from fterm.runtime.runner import DailyRunner
    from fterm.runtime.state import StateDB
    return DailyRunner(cfg, PaperBroker(cfg), StateDB(cfg.runtime.state_db))


def dlv2_pick_days(cfg):
    """(周五, 下周一)：暖机期后、atr14 已定义且在 ST 生效前的第一个周五。"""
    from fterm.features.build import build_dataset
    bundle = build_dataset(cfg)
    dates = bundle.features[f"rev_{cfg.signal.rev_window}"].index
    fridays = [d for d in dates if d.weekday() == 4 and 15 <= dates.get_loc(d) <= 25]
    fri = fridays[0]
    mon = dates[dates.get_loc(fri) + 1]
    assert mon.weekday() == 0
    return fri, mon
