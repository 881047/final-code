r"""训练行的「日组 id」构造（2026-10-08 性能改造 B9）。

## 为什么有这个模块

`scripts/rankloss_chain.py` 与 `scripts/train_production.py` 原本各自用**逐行
`dict.setdefault`** 构造 day_ids：

    day_keys = {}
    for i, (d, _) in enumerate(meta):
        day_ids[i] = day_keys.setdefault(d, len(day_keys))

1200 万行实测 **11.4–11.8 s**（纯 Python 解释器逐行开销）。本模块用 `pd.factorize`
做同一件事（C 级），实测 **0.07 s（≈160×）**。

## 语义等价性（关键）

`pd.factorize(..., sort=False)` 的编码规则是**按首次出现顺序**编号 ——
与 `setdefault(d, len(keys))` 的编号规则**完全相同**。因此新旧实现不仅"分组等价"，
而是 **逐位相同**（`build_day_ids_reference` 保留旧实现供回归对照）。

day_ids 的唯一用途是喂给 `torch.argsort(db, stable=True)` 做**日内分组**，
故绝对数值本无所谓；但我们仍保持逐位一致，以便改造后可用"结果完全相同"来验收。

## 使用约束

`meta` 必须**按日期有序**（同日块连续）—— 这是调用方 `build_dataset` 的既有保证，
也是 `argsort(stable=True)` 依赖的前提。本模块不做排序，只做编码。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_day_ids(dates) -> np.ndarray:
    """日期序列 → int64 日组 id。同 id ⇔ 同日；编号按首次出现顺序。

    dates 可为任意可迭代对象（list / np.ndarray / pandas Series 均可），
    元素类型不限（str / int / Timestamp 都支持），只要同一天的取值可哈希且相等。
    """
    codes, _ = pd.factorize(pd.Index(list(dates)), sort=False)
    return np.asarray(codes, dtype="int64")


def build_day_ids_reference(dates) -> np.ndarray:
    """**旧实现**（逐行 dict.setdefault）。

    仅供 `tests/test_dayids_equivalence.py` 做逐位对照；生产代码不得调用
    （这正是本模块要替换掉的那段慢代码）。
    """
    keys: dict = {}
    out = np.empty(len(dates), dtype="int64")
    for i, d in enumerate(dates):
        out[i] = keys.setdefault(d, len(keys))
    return out
