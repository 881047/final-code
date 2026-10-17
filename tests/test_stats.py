"""信息层统计工具验收（Phase 3）。

纪律：纯合成数据、无网络、无真实老库；每个读数钉一个已知解析值或单调性。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from fterm.model import stats as st


class TestICIR:
    def test_known_value(self):
        x = np.array([0.1, 0.2, 0.3, 0.4])
        # mean=0.25, std(ddof=1)=sqrt(0.05/3)=0.129099...
        assert st.icir(x) == pytest.approx(0.25 / 0.1290994448, rel=1e-6)

    def test_too_short_and_zero_std(self):
        assert st.icir(np.array([0.1])) is None
        assert st.icir(np.array([0.2, 0.2, 0.2])) is None

    def test_nan_filtered(self):
        x = np.array([0.1, np.nan, 0.3, 0.5])
        assert st.icir(x) == pytest.approx(st.icir(np.array([0.1, 0.3, 0.5])))


class TestNeweyWest:
    def test_iid_close_to_naive_t(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0.05, 0.1, size=4000)
        # NW 用总体方差（ddof=0）作 γ0；对照口径必须一致
        naive = x.mean() / math.sqrt((x.std(ddof=0) ** 2) / x.size)
        nw = st.newey_west_t(x, lags=0)   # lags=0 ⇒ 退化为朴素 t
        assert nw == pytest.approx(naive, rel=1e-9)

    def test_positive_autocorr_inflates_se(self):
        rng = np.random.default_rng(1)
        e = rng.normal(0, 1, size=3000)
        x = np.empty_like(e)
        x[0] = e[0]
        for i in range(1, len(e)):      # AR(1) φ=0.7
            x[i] = 0.7 * x[i - 1] + e[i]
        x = x + 0.5                     # 非零均值
        t_naive = x.mean() / (x.std(ddof=1) / math.sqrt(x.size))
        t_nw = st.newey_west_t(x, lags=20)
        assert abs(t_nw) < abs(t_naive)   # 正自相关 ⇒ NW 更保守

    def test_short_returns_none(self):
        assert st.newey_west_t(np.array([1.0, 2.0, 3.0])) is None


class TestRegimeStratified:
    def test_grouping(self):
        ic = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        reg = np.array(["up", "up", "down", "down", "up", "down"])
        out = st.regime_stratified_ic(ic, reg)
        assert out["up"]["n"] == 3
        assert out["up"]["mean"] == pytest.approx((0.1 + 0.2 + 0.5) / 3)
        assert out["down"]["mean"] == pytest.approx((0.3 + 0.4 + 0.6) / 3)
        assert st.regime_min_ic(out) == pytest.approx((0.1 + 0.2 + 0.5) / 3)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            st.regime_stratified_ic(np.array([0.1, 0.2]), np.array(["a"]))


class TestDeflatedSharpe:
    def test_dsr_increases_with_sharpe(self):
        lo = st.deflated_sharpe(0.02, n_trials=10, n_obs=1000)
        hi = st.deflated_sharpe(0.10, n_trials=10, n_obs=1000)
        assert hi.dsr > lo.dsr

    def test_dsr_decreases_with_trials(self):
        few = st.deflated_sharpe(0.05, n_trials=2, n_obs=1000)
        many = st.deflated_sharpe(0.05, n_trials=500, n_obs=1000)
        assert many.sr0 > few.sr0        # 尝试越多，期望最大 Sharpe 越高
        assert many.dsr < few.dsr

    def test_dsr_in_unit_interval(self):
        d = st.deflated_sharpe(0.08, n_trials=50, n_obs=2000)
        assert 0.0 <= d.dsr <= 1.0


class TestPBO:
    def test_dominant_strategy_low_pbo(self):
        rng = np.random.default_rng(7)
        T, N = 400, 6
        M = rng.normal(0.0, 1.0, size=(T, N))
        M[:, 3] += 2.0                   # 策略 3 全程占优
        out = st.pbo_cscv(M, n_blocks=8)
        assert out["pbo"] is not None
        assert out["pbo"] < 0.2

    def test_noise_pbo_exceeds_dominant(self):
        rng = np.random.default_rng(11)
        noise = rng.normal(0.0, 1.0, size=(400, 6))   # 无任何真实优势
        dom = noise.copy()
        dom[:, 3] += 2.0                              # 策略 3 全程占优
        p_noise = st.pbo_cscv(noise, n_blocks=8)["pbo"]
        p_dom = st.pbo_cscv(dom, n_blocks=8)["pbo"]
        assert p_noise is not None and p_dom is not None
        assert p_noise > p_dom                        # 噪声下更易选出「假最优」
        assert p_noise < 0.6                          # 纯噪声 PBO 不极端（有限样本波动）

    def test_single_strategy_returns_none(self):
        out = st.pbo_cscv(np.random.default_rng(0).normal(size=(100, 1)))
        assert out["pbo"] is None


class TestSummaries:
    def test_summarize_ic(self):
        x = np.array([0.05, 0.1, -0.02, 0.08, 0.03])
        s = st.summarize_ic(x)
        assert s["n"] == 5
        assert s["mean"] == pytest.approx(x.mean())
        assert s["pos_frac"] == pytest.approx(0.8)

    def test_paired_t_identical_is_none_or_zero(self):
        a = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        out = st.paired_t(a, a)
        assert out["mean_diff"] == pytest.approx(0.0)
        assert out["t"] is None          # 零方差 ⇒ NW t 无定义

    def test_paired_t_positive(self):
        rng = np.random.default_rng(3)
        a = rng.normal(0.10, 0.05, size=500)
        b = rng.normal(0.05, 0.05, size=500)
        out = st.paired_t(a, b)
        assert out["mean_diff"] > 0
        assert out["t"] > 3
