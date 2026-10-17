r"""时序因子向量化改造的等价性回归（2026-10-08 B6 改造的安全网）。

背景：`atr()` 与 `intraday_micro()` 原用 `pd.concat(axis=0).groupby(level=0).max()/min()`
求多帧逐元素极值 —— 先纵向拼接（3× 内存）再分组归并，纯属绕路。
改成 `np.fmax/np.fmin.reduce` 后必须**逐位一致**，本文件钉住这一点。

⚠️ 本文件存在的理由：这个改造有个隐蔽陷阱 —— `np.maximum` **不跳 NaN**，
而 pandas 的 `groupby.max()` 默认 `skipna=True`。`atr()` 的首行 `prev=NaN`，
若误用 `maximum` 会让整行变 NaN，**看起来"跑通了"但数值全错**。
改造前 `tests/` 对 atr 只有一条 `atr > 0` 的弱断言，抓不到这个问题。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fterm.features import tsfactors as tf  # noqa: E402


def _panels(n_days=60, n_codes=25, seed=5):
    """合成 OHLC 宽面板，含 NaN（模拟停牌/缺失）。"""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n_days)
    cols = [f"{i:06d}.SZ" for i in range(n_codes)]
    close = pd.DataFrame(rng.uniform(5, 40, (n_days, n_codes)), index=idx, columns=cols)
    high = close * (1 + rng.uniform(0, 0.03, (n_days, n_codes)))
    low = close * (1 - rng.uniform(0, 0.03, (n_days, n_codes)))
    open_ = close.shift(1).fillna(close)
    # 制造缺失：约 5% 单元为 NaN
    for df in (close, high, low, open_):
        df[rng.random((n_days, n_codes)) < 0.05] = np.nan
    return open_, high, low, close


def atr_reference(high, low, close, window=14):
    """旧实现（逐帧 concat + groupby）—— 仅用于对照。"""
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()],
                   axis=0).groupby(level=0).max()
    return tr.rolling(window).mean().astype("float32")


def intraday_reference(open_, high, low, close):
    """旧实现（hi_oc/lo_oc 两处 concat + groupby）。"""
    rng = (high - low).replace(0, np.nan)
    hi_oc = pd.concat([open_, close], axis=0).groupby(level=0).max()
    lo_oc = pd.concat([open_, close], axis=0).groupby(level=0).min()
    return {
        "gap": np.log(open_.astype(float) / close.shift(1).astype(float)).astype("float32"),
        "clv": (((close - low) - (high - close)) / rng).astype("float32"),
        "upper_wick": ((high - hi_oc) / rng).astype("float32"),
        "lower_wick": ((lo_oc - low) / rng).astype("float32"),
    }


@pytest.mark.parametrize("window", [5, 14])
def test_atr_bitwise_identical(window):
    open_, high, low, close = _panels()
    new = tf.atr(high, low, close, window)
    old = atr_reference(high, low, close, window)
    assert new.shape == old.shape
    np.testing.assert_array_equal(new.to_numpy(), old.to_numpy(),
                                  err_msg="atr 数值不一致（多半是误用了 np.maximum 而非 np.fmax）")


def test_intraday_micro_bitwise_identical():
    open_, high, low, close = _panels()
    new = tf.intraday_micro(open_, high, low, close)
    old = intraday_reference(open_, high, low, close)
    for k in old:
        np.testing.assert_array_equal(new[k].to_numpy(), old[k].to_numpy(),
                                      err_msg=f"intraday_micro[{k}] 数值不一致")


def test_fmax_is_required_not_maximum():
    """反向断言：用 np.maximum 会得到**不同**结果 —— 证明这条不是冗余检查。

    atr 的首行 prev=NaN ⇒ 三个候选里有两个是 NaN。
    groupby.max()/fmax 跳过 NaN 得到 high-low；maximum 传染 NaN。
    """
    open_, high, low, close = _panels()
    prev = close.shift(1)
    frames = [high - low, (high - prev).abs(), (low - prev).abs()]
    with_fmax = tf._elementwise_extreme(frames, np.fmax)
    with_max = tf._elementwise_extreme(frames, np.maximum)
    # 首行必须不同（fmax 有值，maximum 全 NaN）
    assert with_fmax.iloc[0].notna().any(), "fmax 首行不应全为 NaN"
    assert with_max.iloc[0].isna().all(), "maximum 首行应被 NaN 传染（这正是不能用它的原因）"
    assert not np.array_equal(with_fmax.to_numpy(), with_max.to_numpy(), equal_nan=True)


def test_atr_positive_and_finite():
    """原有语义不退化：非 NaN 处 ATR 应为正且有限。"""
    open_, high, low, close = _panels()
    a = tf.atr(high, low, close, 14)
    v = a.to_numpy()
    finite = v[np.isfinite(v)]
    assert finite.size > 0
    assert (finite > 0).all()
