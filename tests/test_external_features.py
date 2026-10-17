"""外部因子入模（臂 C）验收：共线剔除、bundle 扩展、筛选读回。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fterm.features.build import DatasetBundle
from fterm.features.external_features import extend_bundle, max_abs_rho

N_DAYS, N_CODES = 40, 8


def _bundle(seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=N_DAYS)
    codes = [f"{600000+i}.SH" for i in range(N_CODES)]
    base = pd.DataFrame(rng.normal(size=(N_DAYS, N_CODES)), index=dates, columns=codes)
    feats = {"feat_a": base, "feat_b": pd.DataFrame(rng.normal(size=(N_DAYS, N_CODES)), index=dates, columns=codes)}
    uni = pd.DataFrame(True, index=dates, columns=codes)
    return DatasetBundle(feats, {}, uni, ("feat_a", "feat_b"), {}), base, dates, codes


class TestMaxAbsRho:
    def test_identical_is_one(self):
        b, base, dates, codes = _bundle()
        mask = b.universe.astype(bool)
        assert max_abs_rho(base, base, mask) > 0.99

    def test_independent_is_low(self):
        b, base, dates, codes = _bundle()
        rng = np.random.default_rng(99)
        other = pd.DataFrame(rng.normal(size=(N_DAYS, N_CODES)), index=dates, columns=codes)
        assert max_abs_rho(other, base, b.universe.astype(bool)) < 0.5


class TestExtendBundle:
    def test_adds_and_rejects_collinear(self):
        b, base, dates, codes = _bundle()
        rng = np.random.default_rng(7)
        indep = pd.DataFrame(rng.normal(size=(N_DAYS, N_CODES)), index=dates, columns=codes)
        collinear = base * 1.0 + rng.normal(scale=1e-6, size=(N_DAYS, N_CODES))   # 与 feat_a 共线
        ext = {"indep_x": indep, "dup_x": collinear}
        new, rhos = extend_bundle(b, ext, rho_max=0.7)
        assert "indep_x" in new.feature_cols          # 独立因子被并入
        assert "dup_x" not in new.feature_cols        # 共线因子被剔除
        assert rhos["dup_x"] > 0.7
        assert rhos["indep_x"] < 0.7

    def test_original_features_preserved(self):
        b, base, dates, codes = _bundle()
        new, _ = extend_bundle(b, {})
        assert set(new.feature_cols) == {"feat_a", "feat_b"}
        assert new.feature_cols == tuple(sorted(new.feature_cols))

    def test_original_bundle_untouched(self):
        b, base, dates, codes = _bundle()
        before = tuple(b.feature_cols)
        extend_bundle(b, {"x": base * 0 + 1.0})       # 常数因子（ρ 无定义 → 低）
        assert tuple(b.feature_cols) == before        # frozen dataclass 不被原地改
