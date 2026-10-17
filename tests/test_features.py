"""特征层验收：面板 PIT 安全、波动率估计量、截面特征、标签对齐、bundle 装配。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fterm.data import ashare as sa
from fterm.features import build as fb
from fterm.features import tsfactors as ts


def _flat_panels(n=60, m=4, seed=0):
    rng = np.random.RandomState(seed)
    idx = pd.bdate_range("2024-01-02", periods=n)
    cols = [f"{i:06d}.SZ" for i in range(m)]
    px = pd.DataFrame(10 + np.abs(rng.normal(0, 0.02, (n, m))).cumsum(axis=0),
                      index=idx, columns=cols)
    spread = np.abs(rng.normal(0.01, 0.005, (n, m)))
    panels = {
        "open": (px * (1 - 0.002)).astype("float32"),
        "high": (px * (1 + 0.004)).astype("float32"),
        "low": (px * (1 - 0.004)).astype("float32"),
        "close": px.astype("float32"),
        "adj": pd.DataFrame(1.0, index=idx, columns=cols).astype("float32"),
        "amount": pd.DataFrame(1e6, index=idx, columns=cols).astype("float32"),
        "vol": pd.DataFrame(1e5, index=idx, columns=cols).astype("float32"),
        "turnover": pd.DataFrame(1.5, index=idx, columns=cols).astype("float32"),
        "circ_mv": pd.DataFrame(3e5, index=idx, columns=cols).astype("float32"),
        "up_limit": (px * 1.1).astype("float32"),
        "down_limit": (px * 0.9).astype("float32"),
        "st_flag": pd.DataFrame(0, index=idx, columns=cols, dtype="int8"),
    }
    return panels


class TestTSFactors:
    def test_shapes_and_pit_safety(self):
        panels = _flat_panels()
        c = sa.hfq(panels)["close"]
        mom = ts.momentum(c, (5, 10))["mom_5"]
        assert mom.shape == c.shape
        assert mom.isna().sum().sum() == 5 * c.shape[1]      # 前 5 行暖机为 NaN
        # PIT：改动未来某天，之前所有行必须逐位不变
        c2 = c.copy()
        c2.iloc[-1] = c2.iloc[-1] * 2
        mom2 = ts.momentum(c2, (5, 10))["mom_5"]
        pd.testing.assert_frame_equal(mom.iloc[:-1], mom2.iloc[:-1])

    def test_parkinson_vol_positive(self):
        panels = _flat_panels()
        hfq = sa.hfq(panels)
        v = ts.parkinson_vol(hfq["high"], hfq["low"], 5)
        assert (v.dropna() >= 0).all().all()

    def test_gk_close_to_parkinson(self):
        panels = _flat_panels()
        hfq = sa.hfq(panels)
        gk = ts.gk_vol(hfq["high"], hfq["low"], hfq["close"], hfq["open"], 5)
        pk = ts.parkinson_vol(hfq["high"], hfq["low"], 5)
        diff = (gk - pk).abs().mean().mean()
        scale = pk.mean().mean()
        assert diff / scale < 0.5     # 同源日内信息，量级须一致（合成数据容忍度）

    def test_atr_positive_raw_price(self):
        panels = _flat_panels()
        a = ts.atr(panels["high"], panels["low"], panels["close"])
        assert (a.dropna() > 0).all().all()

    def test_micro_columns(self):
        panels = _flat_panels()
        hfq = sa.hfq(panels)
        micro = ts.intraday_micro(hfq["open"], hfq["high"], hfq["low"], hfq["close"])
        assert set(micro) == {"gap", "clv", "upper_wick", "lower_wick"}
        assert ((micro["clv"].dropna() >= -1) & (micro["clv"].dropna() <= 1)).all().all()
        # 上下影线非负（合成数据 open/close 在 H/L 区间内）
        assert (micro["upper_wick"].dropna() >= -1e-6).all().all()

    def test_liquidity_columns(self):
        panels = _flat_panels()
        liq = ts.liquidity(panels["turnover"], panels["amount"], panels["circ_mv"], 3)
        assert set(liq) == {"turnover_3", "turnover_chg_5", "amount_log", "size_log"}


class TestCrossSectional:
    def test_xrank_rel_disp(self):
        panels = _flat_panels(m=5, seed=3)
        hfq = sa.hfq(panels)
        mom = ts.momentum(hfq["close"], (20,))["mom_20"]
        feats = {"mom_20": mom, "pvol_20": ts.parkinson_vol(hfq["high"], hfq["low"], 5)}
        out = fb.add_cross_sectional(feats)
        xr = out["xrank_mom_20"]
        # 排名 ∈ (0,1]；每行秩覆盖 {1/5..5/5}（无并列时）
        row = xr.iloc[-1].dropna()
        assert ((row > 0) & (row <= 1)).all()
        assert len(set(row.round(6))) == 5
        # rel_mom：每行截面和为 0（减去行均值）
        assert out["rel_mom_20"].iloc[-1].sum() == pytest.approx(0.0, abs=1e-6)
        # disp：同日全市场同值
        d = out["disp_20"].iloc[-1]
        assert d.nunique() == 1


class TestLabels:
    def test_fwd_ret_alignment(self):
        panels = _flat_panels()
        hfq = sa.hfq(panels)
        o = hfq["open"]
        h = 2
        fwd = o.shift(-(h + 1)) / o.shift(-1) - 1.0
        # 手工验证一个格子
        t = 10
        expect = o.iloc[t + h + 1, 0] / o.iloc[t + 1, 0] - 1
        assert fwd.iloc[t, 0] == pytest.approx(expect, rel=1e-6)
        # 尾部 h+1 行必须全 NaN（无未来数据）
        assert fwd.iloc[-(h + 1):].isna().all().all()

    def test_labels_bundle_match(self, env):
        cfg, panels, cal = env
        labels = fb.add_labels(cfg, panels)
        assert set(labels) == {"fwd_ret", "fwd_dir", "fwd_vol"}
        assert labels["fwd_dir"].dropna().isin([-1.0, 1.0]).all().all()


class TestBuildDataset:
    def test_full_bundle(self, env):
        cfg, panels, cal = env
        bundle = fb.build_dataset(cfg)
        assert "rev_2" in bundle.features               # 反转列（rev_window=2）
        assert "turnover_3" in bundle.features          # A股流动性面
        assert "xrank_mom_20" not in bundle.features    # 测试配置无 mom_20 窗
        assert bundle.universe.dtypes.eq(bool).all()
        assert bundle.fingerprints, "数据指纹必须随 bundle 走（可复现纪律）"
        # 股票池：上市暖机期外每天都有 True（ST 段 000002 被剔，其余在场）
        assert bundle.universe.any(axis=1).iloc[cfg.universe.min_list_days:].all()

    def test_reversal_col_matches_config_window(self, env):
        cfg, panels, cal = env
        bundle = fb.build_dataset(cfg)
        col = f"rev_{cfg.signal.rev_window}"
        c = sa.hfq(panels)["close"]
        expect = -(np.log(c) - np.log(c).shift(cfg.signal.rev_window))
        got = bundle.features[col].astype("float64")
        assert np.allclose(got.to_numpy(), expect.to_numpy(), atol=1e-6, equal_nan=True)
