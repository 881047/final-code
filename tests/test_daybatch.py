"""按日整批采样验收（臂 D/E 共用基础设施）。

钉：① 每批含**完整交易日**（不切裂日内截面）；② 并集覆盖全部样本且不重不漏；
③ 打乱只打乱日的顺序、不动日内；④ 边界情形（空/单日/单样本日）。
"""
from __future__ import annotations

import numpy as np
import pytest

from fterm.model.daybatch import day_batches, n_days


def _mk():
    # 3 天，样本数 4 / 2 / 3（模拟每日截面大小不同）
    return np.array([0, 0, 0, 0, 1, 1, 2, 2, 2])


class TestIntegrity:
    def test_batches_are_whole_days(self):
        d = _mk()
        for idx in day_batches(d, batch_days=1, shuffle=False):
            days_in_batch = np.unique(d[idx])
            # 每个 batch 恰好含 batch_days 个完整日
            assert len(days_in_batch) == 1
            for day in days_in_batch:
                assert (d[idx] == day).sum() == (d == day).sum()

    def test_union_is_complete_and_disjoint(self):
        d = _mk()
        seen = []
        for idx in day_batches(d, batch_days=2, shuffle=True, seed=3):
            seen.append(idx)
        all_idx = np.concatenate(seen)
        assert sorted(all_idx.tolist()) == list(range(len(d)))   # 不重不漏

    def test_batch_days_grouping(self):
        d = _mk()
        batches = list(day_batches(d, batch_days=2, shuffle=False))
        assert len(batches) == 2                 # 3 天按 2 天一批 ⇒ 2 批
        assert len(np.unique(d[batches[0]])) == 2
        assert len(np.unique(d[batches[1]])) == 1

    def test_shuffle_preserves_intraday_order(self):
        d = _mk()
        for idx in day_batches(d, batch_days=1, shuffle=True, seed=1):
            # 日内索引必须连续升序（未被内部打乱）
            assert np.all(np.diff(idx) == 1)


class TestEdges:
    def test_empty(self):
        assert list(day_batches(np.array([], dtype=int))) == []

    def test_single_day(self):
        d = np.zeros(5, dtype=int)
        b = list(day_batches(d, batch_days=3))
        assert len(b) == 1 and len(b[0]) == 5

    def test_singleton_days(self):
        d = np.array([0, 1, 2, 3])
        b = list(day_batches(d, batch_days=2, shuffle=False))
        assert len(b) == 2
        assert [len(x) for x in b] == [2, 2]

    def test_n_days(self):
        assert n_days(_mk()) == 3
