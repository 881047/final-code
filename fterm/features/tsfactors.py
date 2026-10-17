"""频段一：日频时序因子（面板版 —— 每列一只股票，逐列滚动，全市场一次算完）。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def momentum(close: pd.DataFrame, windows: tuple[int, ...]) -> dict[str, pd.DataFrame]:
    """多窗口对数动量：ln(P_t / P_{t-w})。列=股票。"""
    logp = np.log(close.astype(float))
    return {f"mom_{w}": (logp - logp.shift(w)).astype("float32") for w in windows}


def reversal(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """短期反转：-ln(P_t / P_{t-window})（旧盘 31 因子极性表：rev 正极性）。"""
    logp = np.log(close.astype(float))
    return (-(logp - logp.shift(window))).astype("float32").rename(columns=lambda c: c)


def parkinson_vol(high: pd.DataFrame, low: pd.DataFrame, window: int) -> pd.DataFrame:
    """Parkinson 高低价波动率（比收盘-收盘更高效利用日内信息）。"""
    hl = (np.log(high.astype(float) / low.astype(float)) ** 2)
    return (np.sqrt(hl.rolling(window).mean() / (4 * np.log(2)))).astype("float32")


def gk_vol(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame,
           open_: pd.DataFrame, window: int) -> pd.DataFrame:
    """Garman-Klass 波动率（含开收信息的 OHLC 估计量）。"""
    log_hl = np.log(high.astype(float) / low.astype(float)) ** 2
    log_co = np.log(close.astype(float) / open_.astype(float)) ** 2
    gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return np.sqrt(gk.rolling(window).mean().clip(lower=0)).astype("float32")


def _elementwise_extreme(frames: list[pd.DataFrame], ufunc) -> pd.DataFrame:
    """多帧逐元素极值，替代 `pd.concat(axis=0).groupby(level=0).max()/min()`。

    2026-10-08 性能改造 B6：原实现为求 3 帧逐元素最大值，先把 3 帧**纵向拼接**
    （单面板 49MB → 中间量 147MB）再做 groupby 归并，纯属绕路。
    实测 2367 日 × 5467 股面板：1.05s → 0.75s（**仅 1.40×**，内存基本持平，
    因 `rolling` 才是主导项）—— 收益远小于 fold_frame 的 7000×，故优先级低。

    ⚠️ **必须用 `np.fmax`/`np.fmin`，不能用 `np.maximum`/`np.minimum`**：
    后者不跳 NaN，而 pandas 的 `groupby.max()/min()` 默认 `skipna=True`。
    首行 `prev=NaN` 时用 maximum 会让整行变 NaN，与现状实现**数值不等**
    （由 tests/test_tsfactors_equivalence.py 逐位钉住；这个坑在基准脚本里第一次跑就抓到过）。

    dtype 不做提升（`np.asarray` 保持原 dtype），以免 float64 中间量改变末位精度。
    """
    arrs = [np.asarray(f) for f in frames]
    out = ufunc.reduce(arrs)
    ref = frames[0]
    return pd.DataFrame(out, index=ref.index, columns=ref.columns)


def atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Average True Range（风控手数计算用；未复权口径——止损盯的是真实交易价）。"""
    prev = close.shift(1)
    tr = _elementwise_extreme([high - low, (high - prev).abs(), (low - prev).abs()], np.fmax)
    return tr.rolling(window).mean().astype("float32")


def intraday_micro(open_: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame,
                   close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """日内微观结构（后复权口径，除权日不留假跳空）。

    gap        隔夜跳空（对数）
    clv        收盘位置 ∈ [-1, 1]
    upper_wick / lower_wick  上下影线占整日区间比
    """
    rng = (high - low).replace(0, np.nan)
    # 2026-10-08 B6：同 atr 的反模式（concat + groupby 求两帧逐元素极值）→ fmax/fmin
    hi_oc = _elementwise_extreme([open_, close], np.fmax)
    lo_oc = _elementwise_extreme([open_, close], np.fmin)
    return {
        "gap": np.log(open_.astype(float) / close.shift(1).astype(float)).astype("float32"),
        "clv": (((close - low) - (high - close)) / rng).astype("float32"),
        "upper_wick": ((high - hi_oc) / rng).astype("float32"),
        "lower_wick": ((lo_oc - low) / rng).astype("float32"),
    }


def ret_tstat(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """动量质量（信息比率式）：滚动窗口内日对数收益 mean/std。"""
    r = np.log(close.astype(float)).diff()
    mu = r.rolling(window).mean()
    sd = r.rolling(window).std()
    return (mu / sd.replace(0, np.nan)).astype("float32")


def liquidity(turnover: pd.DataFrame, amount: pd.DataFrame, circ_mv: pd.DataFrame,
              window: int) -> dict[str, pd.DataFrame]:
    """A股特有信息面：换手率（热度/拥挤）、成交额（规模 proxy）、流通市值（规模）。"""
    return {
        f"turnover_{window}": turnover.rolling(window).mean().astype("float32"),
        "turnover_chg_5": (turnover / turnover.shift(5) - 1.0).astype("float32"),
        "amount_log": np.log1p(amount.astype(float)).astype("float32"),
        "size_log": np.log1p(circ_mv.astype(float)).astype("float32"),
    }


def weekday_panel(index: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
    """星期效应占位（0=周一..4=周五 → /4 归一到 [0,1]；截面常数特征）。"""
    v = pd.Series(index.dayofweek / 4.0, index=index, dtype="float32")
    return pd.DataFrame({c: v for c in columns})
