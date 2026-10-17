r"""两遍法 equity/peak 提取的等价性回归（2026-10-08 B5 + 去重改造的安全网）。

背景：`gate_chain.decision_pass` 与 `capital_chain.decision_pass` 里每个调仓日做

    hist = eq_curve.loc[:d]
    equity  = float(hist.iloc[-2]) if len(hist) >= 2 else float(hist.iloc[-1])
    peak    = float(hist.max())
    day_pnl = float(hist.iloc[-1] - hist.iloc[-2]) if len(hist) >= 2 else 0.0

重算整个前缀最大值（单次 O(D)、总量 O(D×R)）。改成 `cummax()` 一次算好 + O(1) 查表后
**必须逐位一致**，本文件钉住这一点，并专门覆盖两个易错边界：

1. **d 不在索引时**：`eq_curve.loc[:d]` 会退到「最后一个 ≤ d 的日期」，不是报错也不是空；
2. **前缀只有 1 个点**：equity 退化为当日权益、day_pnl = 0。

改造前 `tests/` 对这两处**零覆盖** —— 改错了不会有测试变红。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fterm.backtest.two_pass import EquityCurveView  # noqa: E402


def reference_context(eq_curve, d):
    """**旧实现**逐字复刻：返回 (equity, peak, day_pnl)；无数据时抛 IndexError。"""
    hist = eq_curve.loc[:d]
    equity = float(hist.iloc[-2]) if len(hist) >= 2 else float(hist.iloc[-1])
    peak = float(hist.max())
    day_pnl = float(hist.iloc[-1] - hist.iloc[-2]) if len(hist) >= 2 else 0.0
    return equity, peak, day_pnl


def _curve(n=120, seed=17):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    # 有涨有跌，保证 cummax 与当期值不同（否则 peak 恒等于 last，测不出差异）
    vals = 100000 + np.cumsum(rng.normal(0, 800, n))
    return pd.Series(vals, index=idx)


def test_bitwise_identical_on_every_day():
    eq = _curve()
    view = EquityCurveView(eq)
    for d in eq.index:
        ref = reference_context(eq, d)
        got = view.context(d)
        assert got is not None
        np.testing.assert_array_equal(np.array(got), np.array(ref), err_msg=f"{d} 不一致")


def test_identical_when_d_not_in_index():
    """边界 1：d 不在索引 ⇒ 退到最后一个 ≤ d 的日期（不是报错/空）。"""
    eq = _curve()
    view = EquityCurveView(eq)
    # 构造一个确定不在索引里的日期（周末；逐日试探以兼容任意日历）
    d_gap = eq.index[40] + pd.Timedelta(days=1)
    while d_gap in eq.index:
        d_gap += pd.Timedelta(days=1)
    assert d_gap not in eq.index
    ref = reference_context(eq, d_gap)
    got = view.context(d_gap)
    np.testing.assert_array_equal(np.array(got), np.array(ref))

    # d 远晚于全部数据 ⇒ 等价于取最后一个点
    d_far = eq.index[-1] + pd.Timedelta(days=30)
    np.testing.assert_array_equal(np.array(view.context(d_far)),
                                  np.array(reference_context(eq, d_far)))


def test_prefix_of_one_point():
    """边界 2：前缀只有 1 个点 ⇒ equity=当日、day_pnl=0。"""
    eq = _curve()
    view = EquityCurveView(eq)
    d0 = eq.index[0]
    ref = reference_context(eq, d0)
    got = view.context(d0)
    assert got == ref
    assert got[2] == 0.0, "前缀仅 1 点时 day_pnl 必须为 0"
    assert got[0] == float(eq.iloc[0]), "前缀仅 1 点时 equity 退化为当日权益"


def test_returns_none_before_any_data():
    """d 早于全部数据 ⇒ 返回 None（旧实现此时 IndexError，故调用方需跳过）。"""
    eq = _curve()
    view = EquityCurveView(eq)
    assert view.context(eq.index[0] - pd.Timedelta(days=1)) is None
    with pytest.raises(IndexError):
        reference_context(eq, eq.index[0] - pd.Timedelta(days=1))


def test_peak_is_running_max_not_global_max():
    """peak 必须是**截至当日**的累积最大值，不是全期最大值。"""
    eq = _curve()
    view = EquityCurveView(eq)
    # 找一个"历史最高点之后回落"的日子：peak 应等于历史最高，且 > 当日值
    run_max = eq.cummax()
    i = int(np.argmax((run_max.to_numpy() - eq.to_numpy()) > 1e-9))
    if i == 0:
        pytest.skip("合成序列无回落点")
    d = eq.index[i]
    equity, peak, _ = view.context(d)
    assert peak == float(run_max.iloc[i])
    assert peak > float(eq.iloc[i])


def test_reb_days_order_independent():
    """`run_decision_pass` 内部强制升序 ⇒ set 输入与 list 输入访问顺序一致。

    原实现里 capital_chain 传的是 **set**（无序），靠 CPython 集合迭代顺序"碰巧"
    接近插入顺序；本模块显式 `sorted()`，消除这个隐患。
    """
    from fterm.backtest.two_pass import run_decision_pass

    eq = _curve(60)
    view = EquityCurveView(eq)
    days = list(eq.index[5:50:5])

    class _EmptyGov:
        """只记录被调用的日期顺序；不做任何审批（decisions 为空）。"""
        def __init__(self):
            self.seen = []

        def approve(self, tgts, state, prices=None):
            class _R:
                decisions = []
            return _R()

    def _run(reb_days):
        gov = _EmptyGov()
        order = []
        run_decision_pass(
            reb_days=reb_days, cfg=None, governor=gov, raw_close=pd.DataFrame(),
            initial_equity=1.0, equity_view=view,
            targets_for=lambda d, e, p: (order.append(d), [])[1],
        )
        return order

    from_list = _run(days)
    from_set = _run(set(days))
    assert from_list == sorted(days), "调仓日必须按升序访问"
    assert from_set == sorted(days), "set 输入也必须按升序访问（内部已排序）"
    assert from_list == from_set
