"""按日整批采样（模型升级臂 D/E 的共用基础设施，2026-10-09）。

## 为什么需要它

F0 的训练用**跨日随机混洗**（`DataLoader(shuffle=True)`，BATCH=4096）。在该口径下，
同一交易日的样本被摊到许多 batch 里，平均每个 batch 每日只有 **~2–9 个样本**
（BATCH ÷ 训练窗交易日数：fold0 ≈8.5 / fold1 ≈4.2 / fold2 ≈2.8 / fold3 ≈2.1，
随折变化；均 ≪ 完整截面 ~4300；`plans/rankloss-prereg-20261007.md` 自述同日对
覆盖 ~0.24%/epoch/日）。这对**逐股独立**的 net2 无害，但会**使以下两类改动失去意义**：

  - **臂 D 截面注意力**：注意力掩码按 `day_id` 分块，若同批每日仅几只股票，
    等于没有截面信息可注意；
  - **臂 E listwise 损失**（ListNet/ListMLE）：按日截面做概率分布对齐，
    需要**完整的当日截面**才有定义。

因此 D/E 必须用**按日整批**（每个 batch 含整数个完整交易日）。

## 与 F0 的比较纪律（关键）

按日整批本身会改变训练动态（批内相关性、有效批大小、梯度噪声结构）。
若用「D/E（按日整批）vs F0（随机混洗）」，则**批构造方式**与**架构/损失**混淆。
故本基础设施配套一个 **`F0_day` 对照臂**：同样的按日整批、同样的 net2 + MSE。
判据必须同时给出 D/E 相对 **F0** 与相对 **F0_day** 的配对 t。
"""
from __future__ import annotations

import numpy as np


def day_batches(day_ids: np.ndarray, batch_days: int = 2, shuffle: bool = True,
                seed: int = 0):
    """产出按日整批的索引数组：每批含 `batch_days` 个完整交易日的全部样本。

    day_ids 必须**按日期块连续**（同日样本相邻）——由 `build_day_ids` + `fold_frame`
    的产出保证（day_ids 按首次出现顺序编号，故数组**非递减**）。本函数据此切块，
    不做排序（排序会打乱日内结构）。
    """
    d = np.asarray(day_ids)
    if d.ndim != 1:
        raise ValueError("day_batches: day_ids 需为 1 维")
    if d.size == 0:
        return
    change = np.flatnonzero(np.diff(d)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [d.size]])
    day_slices = [(int(s), int(e)) for s, e in zip(starts, ends)]
    if shuffle:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(day_slices))
        day_slices = [day_slices[i] for i in perm]
    for i in range(0, len(day_slices), batch_days):
        chunk = day_slices[i:i + batch_days]
        idx = np.concatenate([np.arange(s, e) for s, e in chunk])
        if idx.size:
            yield idx


def n_days(day_ids: np.ndarray) -> int:
    return int(np.unique(np.asarray(day_ids)).size)
