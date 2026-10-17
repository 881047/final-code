r"""股票池布尔面板的**向量化查询**（2026-10-08 性能改造 B2）。

## 为什么有这个模块

全仓有三处用「逐日逐股 `universe.at[d, c]` 标量取数」筛选股票池：

| 位置 | 形态 |
|---|---|
| `scripts/dlv2_chain.py` `fold_frame` | 每个交易日 × 每只股票两次 `.at`（universe + fwd_ret） |
| `scripts/gate_chain.py` `targets_for` | 每个调仓日 `row[[c for c in row.index if bool(universe.at[d, c])]]` |
| `scripts/capital_chain.py` `targets_for` | 同上（12 组合 ⇒ 调用次数 ×12） |

pandas 的 `.at` 每次都走一遍标签哈希与标量装箱，在 2367 日 × 5467 股规模下
复杂度虽标称 O(1)/格，常数极大。实测（`scripts/bench_profile.py`）：
60 日扫描 10.5–10.7 s ⇒ 全史外推 **415–423 s（约 7 分钟）**；
改为布尔矩阵整数切片后 **0.0014 s**（同 60 日），命中数完全一致（49492）。

## 语义保持

`universe` 由 `build_features` 产出，`universe_mask` 内已 `fillna(False)`
⇒ 缺列 / NaN 一律视为 **False**（不可入选）。本模块用 `to_numpy(dtype=bool)`
保持同一语义；对 `columns` 中不存在的 code 返回 False。

行序 / 列序一律保持调用方传入的顺序（`get_indexer` 是 C 级、保持输入序）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class UniverseIndex:
    """`universe` 布尔宽面板的向量化查询器。

    构造一次（O(T×N) 拷贝），之后每次查询都是整数索引，无标签哈希。
    """

    def __init__(self, universe: pd.DataFrame):
        #: (T, N) bool 数组；fillna 已由上游完成，此处只做 dtype 归一
        self._arr = universe.to_numpy(dtype=bool)
        self._row_pos = {d: i for i, d in enumerate(universe.index)}
        self.columns = universe.columns

    def __len__(self) -> int:
        return self._arr.shape[0]

    def row_pos(self, d) -> int | None:
        """日期 → 行号；面板中不存在该日期返回 None。"""
        return self._row_pos.get(d)

    def row(self, d) -> np.ndarray | None:
        """整行布尔数组（长度 N，列序 = universe.columns）；日期不存在返回 None。"""
        i = self._row_pos.get(d)
        return None if i is None else self._arr[i]

    def mask_for(self, d, codes, strict: bool = True) -> np.ndarray:
        """`codes` 中入选者的布尔掩码（长度 = len(codes)，**保持输入顺序**）。

        用于 `row = row[mask]` 这类「按现有 index 过滤」的场景
        （各 chain 的 targets_for）。面板中不存在的 code ⇒ False。

        strict=True（默认）：日期不在面板中时**抛 KeyError** ——
        与原 `universe.at[d, c]` 的行为一致（宁可大声失败，不静默返回空池，
        否则"某天没数据"会被误读成"当天无票可选"）。
        strict=False：日期缺失返回全 False（用于按日遍历、缺日应跳过的场景）。
        """
        codes = list(codes)
        out = np.zeros(len(codes), dtype=bool)
        i = self._row_pos.get(d)
        if i is None:
            if strict:
                raise KeyError(f"UniverseIndex: 日期 {d} 不在股票池面板中（面板未对齐）")
            return out
        if not codes:
            return out
        pos = self.columns.get_indexer(pd.Index(codes))
        ok = pos >= 0
        out[ok] = self._arr[i, pos[ok]]
        return out

    def positions(self, d) -> np.ndarray:
        """该日入选股票的**列位置**数组（升序，等价于按 universe.columns 顺序遍历）。

        fold_frame 用这条快路径：列序与 universe.columns 一致，
        故无需任何 code→位置映射，直接整数切片。

        日期缺失返回空数组（非 strict）——fold_frame 的日循环已用 index 一致性
        校验保证 d 必然存在，此处无需再抛错。
        """
        i = self._row_pos.get(d)
        if i is None:
            return np.empty(0, dtype=np.int64)
        return np.flatnonzero(self._arr[i])
