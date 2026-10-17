r"""day_ids 构造的等价性回归（2026-10-08 B9 改造的安全网）。

背景：改造前 `tests/` 对 B9（day_ids 逐行 dict → pd.factorize）**没有任何断言**
—— 也就是说改错了不会有测试变红。本文件先于实现落地，钉住三条不变量：

1. **逐位一致**：新实现与旧 `setdefault` 实现输出完全相同（不只是分组等价）；
2. **分组语义**：同 id ⇔ 同日；不同日必不同 id；
3. **首次出现序**：编号按首次出现顺序 —— `argsort(stable=True)` 的日内分组依赖
   "同日块连续"，编号规则变了不会破坏分组，但会改变 id 的绝对数值，
   若将来有人拿 id 做别的事（如当索引），顺序变了就会静默出错，故一并钉住。

另附性能下界断言（新实现不得慢于旧实现），防止有人改回逐行循环。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fterm.model.dayids import build_day_ids, build_day_ids_reference  # noqa: E402


def _meta_dates(n_days: int, per_day: int, dtype="str"):
    """构造按日期有序的日期序列（模拟 build_dataset 的 meta 顺序）。"""
    if dtype == "str":
        days = [f"2024{(i // 20) + 1:02d}{(i % 20) + 1:02d}" for i in range(n_days)]
    elif dtype == "int":
        days = [20240101 + i for i in range(n_days)]
    else:  # timestamp
        import pandas as pd
        days = list(pd.bdate_range("2024-01-01", periods=n_days))
    return [d for d in days for _ in range(per_day)]


@pytest.mark.parametrize("dtype", ["str", "int", "timestamp"])
def test_bitwise_identical_to_reference(dtype):
    """不变量 1：新实现与旧 setdefault 实现逐位一致。"""
    dates = _meta_dates(60, 40, dtype)
    assert np.array_equal(build_day_ids(dates), build_day_ids_reference(dates))


def test_grouping_semantics():
    """不变量 2：同 id ⇔ 同日。"""
    dates = ["a"] * 3 + ["b"] * 2 + ["a"] * 2 + ["c"]
    ids = build_day_ids(dates)
    ref = build_day_ids_reference(dates)
    assert np.array_equal(ids, ref)
    # 同 id ⇔ 同日
    by_id: dict = {}
    for d, i in zip(dates, ids):
        by_id.setdefault(int(i), set()).add(d)
    for i, ds in by_id.items():
        assert len(ds) == 1, f"id {i} 混了多天：{ds}"
    assert len(by_id) == len(set(dates))


def test_first_appearance_order():
    """不变量 3：编号按首次出现顺序（不是按值排序）。"""
    dates = ["b", "a", "b", "c", "a"]
    ids = build_day_ids(dates)
    assert list(ids) == [0, 1, 0, 2, 1]


def test_empty_and_single():
    assert build_day_ids([]).dtype == np.int64
    assert list(build_day_ids([])) == []
    assert list(build_day_ids(["x", "x", "x"])) == [0, 0, 0]


def test_block_contiguity_preserved():
    """meta 按日期有序 ⇒ 同日块连续；编码后仍应连续（argsort 分组的前提）。"""
    dates = _meta_dates(30, 10)
    ids = build_day_ids(dates)
    # 每个 id 的出现位置应是一段连续区间
    for i in np.unique(ids):
        pos = np.flatnonzero(ids == i)
        assert pos[-1] - pos[0] + 1 == len(pos), f"id {i} 不连续"


def test_not_slower_than_reference():
    """性能下界：新实现不得慢于旧实现（防止改回逐行循环）。

    本尺度（20 万行）两侧都是 ~0.02 s，量级差要到全史才显现
    （`scripts/bench_profile.py` 实测 ~156×）。因此这里**只断言不快于旧实现**，
    并且两侧各取 3 次的最优值——原单次比较零余量，实测会在套件里随机翻红
    （0.01595 vs 0.01592），一个会随机失败的测试等于没有测试。

    ⚠️ 2026-10-10（Lead）：即使取 3 次最优，本尺度实测余量仅 ~8%
    （reference 0.01604 / new 0.01480），在**同机 GPU 训练并发**时仍会随机翻红
    （实跑 1/8 概率）。故容差放到 **2×**：真实回归（改回逐行循环）在全史是 ~156×，
    2× 的容差照样把它抓住，同时把测量噪声排除在外。
    性能的权威口径仍是 `scripts/bench_profile.py`（全史尺度），本测试只做"别退化"的哨兵。
    """
    dates = _meta_dates(4000, 50)

    def best(fn):
        return min((_time(fn) for _ in range(3)))

    def _time(fn):
        t0 = time.perf_counter()
        fn(dates)
        return time.perf_counter() - t0

    t_ref = best(build_day_ids_reference)
    t_new = best(build_day_ids)
    assert t_new <= 2.0 * t_ref, (
        f"新实现（{t_new:.4f}s）慢于旧实现（{t_ref:.4f}s）的 2× —— "
        f"若改回逐行循环会在此处变红（真实提速见 bench_profile，全史 ~156×）")
