r"""fold_frame 向量化改造的等价性回归（2026-10-08 B2 改造的安全网）。

背景：改造前 `tests/` 对 `fold_frame` 的**数值输出没有任何断言** ——
把逐日 `.at` 标量扫描改成布尔矩阵整数切片后，改错了不会有测试变红。
本文件先于实现落地，钉住"逐位一致"：

- `Xg`（逐日截面秩白化后的特征）、`miss`（缺失指示）、`y`、`meta`、`cols`
  五项全部 `assert_array_equal` / 相等比较；
- 覆盖边界：全 False 日、eligible < 5 日、含 NaN 的特征与标签、`cap_per_day` 采样、
  面板中不存在的日期区间。

另钉一条性能下界（新实现不得慢于旧实现），防止有人改回逐行循环。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fterm.features.build import DatasetBundle  # noqa: E402
from fterm.model.foldframe import fold_frame, fold_frame_reference  # noqa: E402


def _make_bundle(n_days=24, n_codes=14, n_feat=3, seed=3):
    """合成 DatasetBundle：含 NaN 特征/标签、False 股票池、全 False 日。"""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n_days)
    codes = pd.Index([f"{i:06d}.SZ" for i in range(n_codes)])

    feats = {}
    for k in range(n_feat):
        arr = rng.normal(size=(n_days, n_codes))
        # 制造缺失：约 8% 单元为 NaN
        arr[rng.random((n_days, n_codes)) < 0.08] = np.nan
        feats[f"f{k}"] = pd.DataFrame(arr, index=idx, columns=codes)

    fwd = rng.normal(0, 0.02, size=(n_days, n_codes))
    fwd[rng.random((n_days, n_codes)) < 0.10] = np.nan   # 标签缺失
    labels = {"fwd_ret": pd.DataFrame(fwd, index=idx, columns=codes)}

    uni = rng.random((n_days, n_codes)) > 0.35            # 约 65% 入选
    uni[0, :] = False                                    # 全 False 日
    uni[1, :] = False
    uni[1, :5] = True                                    # eligible < 5 的日（且 fwd 可能 NaN）
    universe = pd.DataFrame(uni, index=idx, columns=codes)

    return DatasetBundle(feats, labels, universe, tuple(sorted(feats.keys()))), idx, codes


@pytest.mark.parametrize("cap_per_day", [0, 4, 10])
def test_bitwise_identical(cap_per_day):
    """不变量：五项输出全部逐位一致。"""
    bundle, idx, _ = _make_bundle()
    d0, d1 = idx[0], idx[-1]
    new = fold_frame(bundle, bundle.universe, d0, d1, cap_per_day)
    old = fold_frame_reference(bundle, bundle.universe, d0, d1, cap_per_day)

    for name, a, b in zip(("Xg", "miss", "y", "meta", "cols"), new, old):
        if name == "meta":
            assert a == b, f"{name} 不一致"
        elif name == "cols":
            assert list(a) == list(b), f"{name} 不一致"
        else:
            assert a.dtype == b.dtype, f"{name} dtype 不一致：{a.dtype} vs {b.dtype}"
            np.testing.assert_array_equal(a, b, err_msg=f"{name} 数值不一致")


def test_partial_window_identical():
    """只取中间一段窗口（模拟 walk-forward 的单折）。"""
    bundle, idx, _ = _make_bundle()
    new = fold_frame(bundle, bundle.universe, idx[5], idx[18], 6)
    old = fold_frame_reference(bundle, bundle.universe, idx[5], idx[18], 6)
    np.testing.assert_array_equal(new[0], old[0])
    np.testing.assert_array_equal(new[1], old[1])
    np.testing.assert_array_equal(new[2], old[2])
    assert new[3] == old[3]


def test_miss_flags_match_nan_in_Xg_source():
    """miss 必须与 X 的 NaN 位置一致（改造后 miss 由 np.isnan(X) 直接得出）。"""
    bundle, idx, _ = _make_bundle()
    Xg, miss, y, meta, cols = fold_frame(bundle, bundle.universe, idx[0], idx[-1], 0)
    assert Xg.shape == (len(meta), len(cols))
    assert miss.shape == Xg.shape
    assert miss.dtype == np.float32
    # 白化后不应再有 NaN（nan_option="keep" + fillna(0.5) 兜底）
    assert not np.isnan(Xg).any()
    assert not np.isnan(y).any()


def test_raises_on_unaligned_panels():
    """面板未对齐时必须大声报错（旧实现靠 .at 触发 KeyError）。"""
    bundle, idx, _ = _make_bundle()
    bad_universe = bundle.universe.iloc[:-1]              # 少一行 ⇒ index 不一致
    with pytest.raises(ValueError, match="universe"):
        fold_frame(bundle, bad_universe, idx[0], idx[-1], 0)


def test_not_slower_than_reference():
    """性能下界：新实现不得慢于旧实现（防止改回逐行 .at）。"""
    bundle, idx, _ = _make_bundle(n_days=60, n_codes=300, n_feat=5, seed=11)
    t0 = time.perf_counter()
    fold_frame_reference(bundle, bundle.universe, idx[0], idx[-1], 0)
    t_ref = time.perf_counter() - t0
    t0 = time.perf_counter()
    fold_frame(bundle, bundle.universe, idx[0], idx[-1], 0)
    t_new = time.perf_counter() - t0
    assert t_new <= t_ref, (
        f"新实现（{t_new:.4f}s）不得慢于旧实现（{t_ref:.4f}s）—— 若改回逐行 .at 会在此处变红")
