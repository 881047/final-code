"""交易日历校验（A股现货无夜盘，期货版夜盘归属逻辑随期货域一并移除）。

不维护完整交易所日历（避免外部依赖）：日历来自老库 trade_cal（sync 落盘），
本模块只做数据完整性校验——周末绝不能是交易日。
"""

from __future__ import annotations

import pandas as pd


def assert_trading_days(index: pd.DatetimeIndex, name: str = "panel") -> None:
    """校验索引：唯一、单调递增、无周末。"""
    assert index.is_unique, f"{name}: 交易日索引有重复"
    assert index.is_monotonic_increasing, f"{name}: 交易日索引非递增"
    weekdays = set(index.dayofweek)
    assert not (weekdays & {5, 6}), f"{name}: 出现周末交易日（日历/数据错位）"
