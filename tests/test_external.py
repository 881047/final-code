"""外部日频因子提取验收（Phase 0b）。

纪律：不连真实老库（`_read` 用 monkeypatch 注入合成长表）；钉 PIT 语义与因子口径。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fterm.data import external as ex

CODES = ["000001.SZ", "600000.SH"]


@pytest.fixture
def cal():
    return pd.bdate_range("2024-01-01", periods=10)      # 10 个工作日


@pytest.fixture
def panels(cal):
    rng = np.random.default_rng(0)
    def mk(vals):
        return pd.DataFrame(np.array(vals, dtype="float32").reshape(len(cal), len(CODES)),
                            index=cal, columns=CODES)
    close = mk(np.tile([10.0, 20.0], len(cal)))
    return {
        "close": close,
        "amount": mk(np.tile([1e8, 2e8], len(cal))),          # 元
        "circ_mv": mk(np.tile([3e5, 6e5], len(cal))),         # 万元
        "turnover": mk(np.tile([2.0, 1.0], len(cal))),
        "high": close * 1.02, "low": close * 0.98,
        "up_limit": close * 1.1, "down_limit": close * 0.9,
    }


class TestDateConv:
    def test_yyyyyymmdd(self):
        s = pd.Series([20240102, "20240103", None, "bad"])
        out = ex._d(s)
        assert out.iloc[0] == pd.Timestamp("2024-01-02")
        assert out.iloc[1] == pd.Timestamp("2024-01-03")
        assert pd.isna(out.iloc[2]) and pd.isna(out.iloc[3])


class TestPivot:
    def test_alignment(self, cal):
        long = pd.DataFrame({
            "trade_date": [cal[0], cal[0], cal[2]],
            "ts_code": [CODES[0], CODES[1], CODES[0]],
            "v": [1.0, 2.0, 3.0],
        })
        w = ex._pivot(long, "v", cal)
        assert w.shape == (10, 2)
        assert w.loc[cal[0], CODES[1]] == 2.0
        assert np.isnan(w.loc[cal[1], CODES[0]])      # 未观测日 = NaN
        assert w.loc[cal[2], CODES[0]] == 3.0


class TestPitFfill:
    def test_value_available_next_trading_day(self, cal):
        # 公告日 = cal[2]（周三）；值应自 cal[3] 起可用，cal[2] 当日不可用
        long = pd.DataFrame({
            "ts_code": [CODES[0]],
            "ann": [cal[2]],
            "v": [5.0],
        })
        w = ex._pit_ffill(long, "ann", "v", cal)
        assert np.isnan(w.loc[cal[1], CODES[0]])
        assert np.isnan(w.loc[cal[2], CODES[0]])      # 公告当日不可用（次一交易日起）
        assert w.loc[cal[3], CODES[0]] == 5.0
        assert w.loc[cal[9], CODES[0]] == 5.0         # 前向填充

    def test_later_announcement_overrides(self, cal):
        long = pd.DataFrame({
            "ts_code": [CODES[0], CODES[0]],
            "ann": [cal[1], cal[5]],
            "v": [1.0, 9.0],
        })
        w = ex._pit_ffill(long, "ann", "v", cal)
        assert w.loc[cal[2], CODES[0]] == 1.0
        assert w.loc[cal[6], CODES[0]] == 9.0

    def test_empty_returns_nan_panel(self, cal):
        long = pd.DataFrame({"ts_code": [], "ann": [], "v": []})
        w = ex._pit_ffill(long, "ann", "v", cal)
        assert w.empty or w.isna().all().all()


class TestBroadcast:
    def test_cross_sectional_constant(self, panels):
        s = pd.Series(np.arange(len(panels["close"]), dtype="float64"),
                      index=panels["close"].index)
        b = ex._broadcast(s, panels, "float32")
        assert b.shape == panels["close"].shape
        # 同日所有股票同值
        assert b.iloc[3, 0] == b.iloc[3, 1] == 3.0


class TestBuildFlow:
    def test_margin_and_moneyflow(self, cal, panels, monkeypatch):
        # 注入合成 margin / moneyflow，其余表返回空
        def fake_read(cfg, sql):
            # 老库日期格式为 YYYYMMDD（int/str），与真实数据一致
            def ds(ts):
                return ts.strftime("%Y%m%d")
            if "FROM margin" in sql:
                return pd.DataFrame({
                    "trade_date": [ds(cal[1]), ds(cal[1]), ds(cal[6])],
                    "ts_code": [CODES[0], CODES[1], CODES[0]],
                    "rzye": [30000.0, 60000.0, 33000.0],   # 万元
                    "rqye": [1000.0, 2000.0, 1100.0],
                    "rzmre": [3000.0, 6000.0, 3300.0],
                    "rzche": [0.0, 0.0, 0.0],
                })
            if "FROM moneyflow" in sql:
                return pd.DataFrame({
                    "trade_date": [ds(cal[1])], "ts_code": [CODES[0]],
                    "buy_lg_amount": [2e7], "sell_lg_amount": [1e7],
                    "buy_elg_amount": [1e7], "sell_elg_amount": [0.0],
                })
            return pd.DataFrame()

        monkeypatch.setattr(ex, "_read", fake_read)
        out = ex.build_flow(monkeypatch_cfg(), panels, cal)

        # 融资余额 / 流通市值（均为万元）：30000/300000 = 0.1
        assert out["mrg_rzye_ratio"].loc[cal[1], CODES[0]] == pytest.approx(0.1, rel=1e-5)
        # 主力净额 = 2e7+1e7-1e7-0 = 2e7；/ amount(1e8) = 0.2
        assert out["mf_lg_net_ratio"].loc[cal[1], CODES[0]] == pytest.approx(0.2, rel=1e-5)
        # 融资余额 5 日变化：cal[6] / cal[1] 不是 shift(5) 对齐（应逐日历 shift）
        assert "mrg_rzye_chg5" in out


def monkeypatch_cfg():
    """build_flow 只用到 cfg 透传给 _read；monkeypatch 后可为任意对象。"""
    return object()
