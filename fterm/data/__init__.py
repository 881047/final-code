"""数据层 —— 老库 neweco.db 只读同步 → 本仓宽面板（date × ts_code）。

对外入口：ashare.sync_panels()（同步，scripts/sync_data.py 调用）、
ashare.load_panels()（特征/回测加载）、ashare.universe_mask()（PIT 股票池）。
"""

from __future__ import annotations

from . import ashare, calendar  # noqa: F401
