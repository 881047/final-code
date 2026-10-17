"""多 horizon 标签（模型升级臂 B，2026-10-09）。

依据：
  - 判据链 #7 已证 **IC 随 horizon 单调升**（ss42_7：h1 IC 0.032 → h21 IC 0.078，t 10.8→27.1）；
  - §43 把「多 horizon 融合」列为尚未测试的方向。
  - 机制：h=5 是信噪比最差档之一；多任务共享表示可让短窗（高噪声、高信息量）与
    长窗（低噪声、更稳）互相正则，融合分应比单头更稳（ICIR 提升）。

口径与主标签完全一致（`features/build.py::add_labels`）：
  fwd_ret_h = hfq_open[t+h+1] / hfq_open[t+1] − 1（t+1 开盘建仓、持有 h 日）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config

DEFAULT_HORIZONS: tuple[int, ...] = (1, 3, 5, 10, 20)


def build_multihorizon_labels(cfg: Config, panels: dict[str, pd.DataFrame],
                              horizons: tuple[int, ...] = DEFAULT_HORIZONS,
                              ) -> dict[str, pd.DataFrame]:
    """返回 {f"fwd_ret_h{h}": 宽面板}（PIT：只用到未来 h+1 日的开盘价，标签自然截尾）。"""
    from ..data import ashare as sa

    hfq = sa.hfq(panels)
    o = hfq["open"]
    out: dict[str, pd.DataFrame] = {}
    for h in horizons:
        if h < 1:
            raise ValueError(f"multihorizon: 非法 horizon {h}")
        out[f"fwd_ret_h{h}"] = (o.shift(-(h + 1)) / o.shift(-1) - 1.0).astype("float32")
    return out


def fused_score(head_preds: dict[str, pd.Series], weights: dict[str, float] | None = None,
                standardize: bool = True) -> pd.Series:
    """多头融合分（默认等权；可选按 IC 权重）。

    各头输出量纲不同（不同 horizon 的收益尺度不同），standardize=True 时先做截面 z-score
    再加权，避免长 horizon 头（方差更大）支配融合分。
    """
    if not head_preds:
        raise ValueError("fused_score: 空输入")
    if weights is None:
        weights = {k: 1.0 for k in head_preds}
    total = None
    for k, s in head_preds.items():
        w = float(weights.get(k, 1.0))
        if w == 0:
            continue
        v = s.astype("float64")
        if standardize:
            sd = v.std(ddof=0)
            v = (v - v.mean()) / (sd if sd > 1e-12 else 1.0)
        total = w * v if total is None else total + w * v
    if total is None:
        raise ValueError("fused_score: 全零权重")
    return total
