"""spearman_rank_ic 的 NaN 处理回归钉（2026-10-09，臂 B 事故）。

事故背景：臂 B 首跑产出**全 NaN 预测**，但 `spearman_rank_ic` 对全 NaN 输入
仍返回 +0.0578（看似正常），因为 `np.argsort` 把 NaN 排到末尾并赋予非恒定秩，
退化成"行序 vs 收益序"的伪相关。该伪读数一度让臂 B 被误判为 adoption_ready=True。

本测试钉住：任一侧非有限的样本对被**剔除**；全 NaN ⇒ None。
"""
from __future__ import annotations

import numpy as np
import pytest

from fterm.model.validate import spearman_rank_ic


class TestNaNHandling:
    def test_all_nan_returns_none(self):
        pairs = [(float("nan"), float(i)) for i in range(20)]
        assert spearman_rank_ic(pairs) is None

    def test_all_nan_y_returns_none(self):
        pairs = [(float(i), float("nan")) for i in range(20)]
        assert spearman_rank_ic(pairs) is None

    def test_nan_pairs_dropped_not_ranked(self):
        # 前 10 对有效且完全同序；后 10 对含 NaN（应被剔除，不影响结果）
        valid = [(float(i), float(i)) for i in range(10)]
        noise = [(float("nan"), float(i)) for i in range(10)]
        assert spearman_rank_ic(valid + noise) == pytest.approx(1.0)

    def test_too_few_valid_returns_none(self):
        pairs = [(float(i), float(i)) for i in range(4)] + [(float("nan"), 1.0)] * 10
        assert spearman_rank_ic(pairs) is None

    def test_perfect_order_still_one(self):
        pairs = [(float(i), float(i)) for i in range(20)]
        assert spearman_rank_ic(pairs) == 1.0

    def test_reverse_order_minus_one(self):
        pairs = [(float(i), float(19 - i)) for i in range(20)]
        assert abs(spearman_rank_ic(pairs) + 1.0) < 1e-12
