"""CPCV 组合净化交叉验证验收（Phase 3）。

钉三件事：① 路径数公式；② 划分无泄漏（train∩test=∅ 且 purge/embargo 生效）；
③ 样本索引映射正确。
"""
from __future__ import annotations

import numpy as np
import pytest

from fterm.model.cpcv import cpcv_date_splits, cpcv_sample_folds, n_paths


class TestNPaths:
    def test_formula(self):
        # C(6,2)*2/6 = 15*2/6 = 5
        assert n_paths(6, 2) == pytest.approx(5.0)
        # C(8,2)*2/8 = 28*2/8 = 7
        assert n_paths(8, 2) == pytest.approx(7.0)


class TestDateSplits:
    def _dates(self, T=60):
        return list(np.arange(T))

    def test_no_overlap_and_count(self):
        dates = self._dates(60)
        splits = list(cpcv_date_splits(dates, n_groups=6, n_test_groups=2,
                                        label_span=6, embargo=0))
        from math import comb
        assert len(splits) == comb(6, 2)
        for tr, te in splits:
            assert set(tr.tolist()).isdisjoint(set(te.tolist()))

    def test_purge_and_embargo_effective(self):
        dates = self._dates(60)
        T = 60
        # 手工重建分组边界
        bounds = np.linspace(0, T, 7, dtype=int)
        for tr, te in cpcv_date_splits(dates, n_groups=6, n_test_groups=2,
                                       label_span=6, embargo=3):
            tr_set = set(tr.tolist())
            # 对每个测试块，检查 [s-6, e+3) 内无训练样本
            for g in range(6):
                s, e = bounds[g], bounds[g + 1]
                if set(range(s, e)).issubset(set(te.tolist())):
                    for pos in range(max(0, s - 6), min(T, e + 3)):
                        assert pos not in tr_set, f"泄漏：位置 {pos} 未净化"

    def test_illegal_params_raise(self):
        with pytest.raises(ValueError):
            list(cpcv_date_splits(list(range(30)), n_groups=1, n_test_groups=1))
        with pytest.raises(ValueError):
            list(cpcv_date_splits(list(range(30)), n_groups=5, n_test_groups=5))
        with pytest.raises(ValueError):
            list(cpcv_date_splits(list(range(3)), n_groups=6, n_test_groups=2))


class TestSampleFolds:
    def test_mapping_and_min_train(self):
        # 10 个交易日，每日 3 个样本 → 30 样本
        dates = [np.datetime64("2020-01-01") + np.timedelta64(d, "D") for d in range(10) for _ in range(3)]
        folds = list(cpcv_sample_folds(dates, n_groups=5, n_test_groups=1,
                                       label_span=1, embargo=0, min_train=5))
        assert len(folds) > 0
        for tr, te in folds:
            assert set(tr).isdisjoint(set(te))
            assert len(tr) >= 5
            assert len(te) > 0

    def test_min_train_filters(self):
        dates = [np.datetime64("2020-01-01") + np.timedelta64(d, "D") for d in range(10) for _ in range(3)]
        folds = list(cpcv_sample_folds(dates, n_groups=5, n_test_groups=1,
                                       label_span=1, min_train=10_000))
        assert folds == []           # 训练量永不达标 ⇒ 全过滤
