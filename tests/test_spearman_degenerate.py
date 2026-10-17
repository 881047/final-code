"""核心指标函数的退化输入守卫（2026-10-10 verifier 复核抓到的残留缺陷）。

## 背景

`fterm/model/validate.py::spearman_rank_ic` 是**全项目核心 IC 函数**。2026-10-09
修过一次 NaN（臂 B 首跑全 NaN 流曾产出伪 IC +2.74）；但**同类残留未修**：
原守卫 `ra.std() == 0` **不可达**——`_ranks` 赋的是 1..n 整数秩，std 恒 > 0。
⇒ **常数预测**仍产伪 IC（实测常数 pred 对 20 个不同 y 返回 +1.000000）。

本测试钉住"任一侧无变异 ⇒ None"，并顺带钉住**口径缺陷的已知边界**
（整数秩而非并列平均秩），使该口径差有据可查、不静默漂移。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fterm.model.validate import _ranks, spearman_rank_ic  # noqa: E402


class TestDegenerateGuard:
    def test_constant_pred_returns_none(self):
        """常数预测（退化）必须返回 None —— 原实现返回伪 IC +1.0。"""
        y = np.arange(20, dtype=float)
        assert spearman_rank_ic(list(zip(np.zeros(20), y))) is None

    def test_constant_target_returns_none(self):
        y = np.arange(20, dtype=float)
        assert spearman_rank_ic(list(zip(y, np.zeros(20)))) is None

    def test_both_constant_returns_none(self):
        assert spearman_rank_ic(list(zip(np.ones(20), np.full(20, 3.0)))) is None

    def test_normal_input_still_works(self):
        """正常输入不受影响（含完全单调）。"""
        y = np.arange(20, dtype=float)
        assert spearman_rank_ic(list(zip(y, y * 2.0 + 1.0))) == pytest.approx(1.0)
        assert spearman_rank_ic(list(zip(y, -y))) == pytest.approx(-1.0)

    def test_nearly_constant_is_not_none(self):
        """只有一个点不同 ⇒ 仍有变异 ⇒ 不应被误判为退化。"""
        y = np.arange(20, dtype=float)
        p = np.zeros(20)
        p[0] = 1.0
        v = spearman_rank_ic(list(zip(p, y)))
        assert v is not None

    def test_too_few_samples_still_none(self):
        assert spearman_rank_ic([(1.0, 2.0)] * 4) is None

    def test_all_nan_still_none(self):
        """2026-10-09 那次修复的回归（不能因本次改动回退）。"""
        assert spearman_rank_ic([(float("nan"), 1.0)] * 10) is None


class TestRankConvention:
    """口径缺陷的**已知边界**（登记用，非判据）。

    `_ranks` 用 `argsort(stable)` 赋整数秩、**不做并列平均**，故非标准 Spearman。
    y_ret 并列很重（F0 每日平均约 155 对并列），读数与标准 Spearman 有微差
    （verifier 实测 IC 均值差 ~1e-05、NW-t 差 ~1e-03 量级）。
    远小于判据阈值（配对 t≥2），**不改判决方向**，故本批不修
    （改并列口径 = 改基线，须另立预注册 + 全量重算）。此处钉住该边界。
    """

    def test_ranks_are_integer_not_averaged(self):
        a = np.array([1.0, 2.0, 2.0, 3.0])
        r = _ranks(a)
        assert sorted(r.tolist()) == [1.0, 2.0, 3.0, 4.0], "整数秩（并列不平均）"

    def test_ties_differ_from_standard_spearman(self):
        """有并列时与标准 Spearman（并列平均秩）存在微差 ⇒ 登记该口径差。"""
        rng = np.random.default_rng(0)
        y = np.round(rng.normal(size=200), 1)      # 制造大量并列
        p = rng.normal(size=200)
        got = spearman_rank_ic(list(zip(p, y)))

        def avg_rank(x):
            order = np.argsort(x, kind="stable")
            r = np.empty(len(x))
            r[order] = np.arange(1, len(x) + 1)
            # 并列平均
            _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
            sums = np.zeros(len(cnt))
            np.add.at(sums, inv, r)
            return sums[inv] / cnt[inv]

        std = float(np.corrcoef(avg_rank(p), avg_rank(y))[0, 1])
        assert abs(got - std) < 0.01, "微差应远小于判据阈值"
        assert got is not None
