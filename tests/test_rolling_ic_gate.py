"""滚动 IC 门的口径验收（2026-10-10 补入回归钉；docs/03 §45 已知缺口 #3）。

## 背景（为什么需要这个测试）

`fterm/model/stats.py::rolling_ic_stability` 返回的 `nw_t_of_daily_ic`
**名不副实**：它是**日频 IC 序列**的 Newey-West t（`newey_west_t(a)`），
与 `summarize_ic()["t"]` **同一式** —— 实测二者逐位相等
（F0 两处均 +14.110418023514088）。而预注册 §三 的「滚动 IC」门写的是
「Newey-West t > 2 且无长段负 IC（**60 日滚动**）」。
⇒ 该门与其上层摘要读数**同源，不构成独立信息**。

## 本测试钉住的口径（三件）

① **旧键数值不得变**：`nw_t_of_daily_ic` 仍等于 `newey_west_t(日频IC)`，
   且**逐位等于** `summarize_ic()["t"]`（这正是"同源"的证据，也是既有读数的锚）。
   `stability_arch_20261010.json` 引它作证 ⇒ 改它就会改既有读数。
② **新键是真的滚动 t**：`nw_t_of_rolling_mean` = 对 **window 日滚动均值序列**
   取 NW-t，且**不等于**旧键（否则等于没修）。
③ **滞后阶数的诚实登记**：滚动序列由重叠窗口构成，自相关可延伸至约 `window-1` 阶，
   模块经验式缺省（T≈1815 时 L=7）对重叠窗口偏乐观 ⇒ 显式 `lags=window-1`
   必须给出**更保守（更小）**的 |t|。执行器两者都报，不改门槛。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fterm.model import stats as st  # noqa: E402

F0_STREAM = ROOT / "experiments" / "external_20261009" / "stream_F0_f4.parquet"


def _f0_daily_ic() -> np.ndarray:
    """F0 流的逐日 IC 序列（缺证据包则跳过）。"""
    pd = pytest.importorskip("pandas")
    if not F0_STREAM.exists():
        pytest.skip("缺 F0 流缓存（证据包未入库）")
    sys.path.insert(0, str(ROOT / "scripts"))
    from stability_chain import ic_series  # noqa: E402
    df = pd.read_parquet(F0_STREAM)
    df["date"] = pd.to_datetime(df["date"])
    return ic_series(df).to_numpy()


class TestLegacyKeyUnchanged:
    """① 旧键数值必须逐位保留（§45 正是引它作证「同源」）。"""

    def test_legacy_key_equals_summarize_ic_t(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0.05, 0.1, size=800)
        r = st.rolling_ic_stability(a, window=60)
        assert r["nw_t_of_daily_ic"] == st.summarize_ic(a)["t"]
        assert r["nw_t_of_daily_ic"] == st.newey_west_t(a)

    def test_legacy_key_on_real_f0_stream(self):
        """实测锚：F0 两处均为 +14.110418023514088（§45 原载数字）。"""
        ic = _f0_daily_ic()
        r = st.rolling_ic_stability(ic, window=60)
        assert r["nw_t_of_daily_ic"] == st.summarize_ic(ic)["t"]
        assert r["nw_t_of_daily_ic"] == pytest.approx(14.110418023514088, rel=1e-12)


class TestNewRollingKeyIsReal:
    """② 新键必须是**滚动均值序列**的 t，且与旧键不同。"""

    def test_new_key_is_nw_of_rolling_mean(self):
        rng = np.random.default_rng(1)
        a = rng.normal(0.05, 0.1, size=800)
        r = st.rolling_ic_stability(a, window=60)
        roll = st.rolling_mean(a, 60)
        valid = roll[np.isfinite(roll)]
        assert r["nw_t_of_rolling_mean"] == st.newey_west_t(valid)
        assert r["n_rolling"] == valid.size == a.size - 59

    def test_new_key_differs_from_legacy_on_autocorrelated_series(self):
        """反例守卫：若新键仍等于旧键，则等于没修（缺口 #3 未闭合）。"""
        rng = np.random.default_rng(2)
        e = rng.normal(0, 1, size=1200)
        a = np.empty_like(e)
        a[0] = e[0]
        for i in range(1, len(e)):       # AR(1) φ=0.6 ⇒ 日频 IC 强自相关
            a[i] = 0.6 * a[i - 1] + e[i]
        a = a * 0.1 + 0.05
        r = st.rolling_ic_stability(a, window=60)
        assert r["nw_t_of_rolling_mean"] is not None
        assert r["nw_t_of_rolling_mean"] != r["nw_t_of_daily_ic"]

    def test_matches_standalone_helper(self):
        rng = np.random.default_rng(3)
        a = rng.normal(0.05, 0.1, size=500)
        assert (st.rolling_ic_stability(a, 60)["nw_t_of_rolling_mean"]
                == st.rolling_ic_nw_t(a, 60))

    def test_too_short_returns_none(self):
        assert st.rolling_ic_nw_t(np.arange(10.0), window=60) is None

    def test_new_key_on_real_f0_stream(self):
        """实测读数：F0 滚动均值 NW-t = +24.104（≠ 日频 +14.110）。"""
        ic = _f0_daily_ic()
        r = st.rolling_ic_stability(ic, window=60)
        assert r["nw_t_of_rolling_mean"] == pytest.approx(24.104, abs=0.01)
        assert r["nw_t_of_rolling_mean"] != r["nw_t_of_daily_ic"]


class TestOverlapLagHonesty:
    """③ 重叠窗口 ⇒ 缺省滞后阶数偏乐观，`lags=window-1` 必须更保守。"""

    def test_lag59_more_conservative(self):
        ic = _f0_daily_ic()
        opt = st.rolling_ic_nw_t(ic, window=60)
        rob = st.rolling_ic_nw_t(ic, window=60, lags=59)
        assert opt is not None and rob is not None
        assert abs(rob) < abs(opt), "重叠窗口下高阶滞后必须更保守（|t| 更小）"

    def test_explicit_lags_matches_newey_west(self):
        rng = np.random.default_rng(4)
        a = rng.normal(0.05, 0.1, size=400)
        assert (st.rolling_ic_nw_t(a, window=60, lags=59)
                == st.newey_west_t(st.rolling_mean(a, 60)[np.isfinite(
                    st.rolling_mean(a, 60))], lags=59))


class TestLongestNegativeRun:
    """§三「无长段负 IC」的直读量。"""

    def test_basic(self):
        assert st.longest_negative_run(np.array([1.0, -1, -1, 1, -1])) == 2
        assert st.longest_negative_run(np.array([1.0, 2.0])) == 0
        assert st.longest_negative_run(np.array([-1.0, -2, -3])) == 3

    def test_nan_breaks_run(self):
        """NaN 视为中断（既不延续负段，也不计为负）。"""
        assert st.longest_negative_run(np.array([-1.0, np.nan, -1.0, -1.0])) == 2

    def test_empty(self):
        assert st.longest_negative_run(np.array([])) == 0

    def test_on_real_f0_rolling_series(self):
        ic = _f0_daily_ic()
        roll = st.rolling_mean(ic, 60)
        assert st.longest_negative_run(roll) == 12


class TestExecutorReportsBoth:
    """执行器必须两个读数都报（旧键保读数、新键补门）。"""

    def test_stability_chain_reports_new_keys(self):
        src = (ROOT / "scripts" / "stability_chain.py").read_text(encoding="utf-8")
        assert '"nw_t_of_rolling_mean"' in src
        assert '"nw_t_of_rolling_mean_lag59"' in src
        assert "st.rolling_ic_nw_t(" in src
        assert '"longest_negative_run"' in src
        # 旧键逐字保留
        assert '"nw_t": roll["nw_t_of_daily_ic"]' in src
