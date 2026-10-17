"""外部日频因子入模（模型升级臂 C，2026-10-09）。

把 `experiments/newdata_20261009/screen_*.json` 筛出的**幸存因子**（7 个，见
docs/audit-model-20261009.md §7.2）作为特征并入 `DatasetBundle`，供臂 C 的
增量 IC 检验使用。

设计：
  - **不翻转符号**——截面秩白化后符号由网络自行学习，人工翻转只会增加口径风险；
  - 入模前与现役特征做 **max|ρ| ≤ 0.7** 去重（沿用 alpha_pool_screen 口径）；
  - 缺失值保留 NaN（由 fold_frame 的缺失指示列承载，不插补）。

筛选口径（预注册冻结）：|IC| ≥ 0.02 且 |t| ≥ 2 且 max|ρ| vs 现役 ≤ 0.7。
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from .build import DatasetBundle

# 2026-10-09 筛选幸存者（experiments/newdata_20261009/screen_20261009_1117.json）
SURVIVOR_COLS: tuple[str, ...] = (
    "lhb_on", "lhb_org_net_ratio", "mrg_rqye_ratio", "bt_amount_ratio",
    "unlock_ratio_90d", "hk_ratio_chg5", "mrg_rzye_chg5",
)


def load_survivor_panels(cfg: Config, ref: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """加载幸存因子面板并对齐到 ref（index × columns）。缺文件响亮报错。"""
    from ..data.external import load_external

    ext = load_external(cfg, names=list(SURVIVOR_COLS))
    return {c: ext[c].reindex(index=ref.index, columns=ref.columns) for c in SURVIVOR_COLS}


def max_abs_rho(panel: pd.DataFrame, base: pd.DataFrame, mask: pd.DataFrame) -> float:
    """panel 与 base 的逐日截面 Spearman |ρ| 的时间均值（去重口径）。"""
    p = panel.where(mask).rank(axis=1, pct=True).to_numpy(dtype="float64")
    b = base.where(mask).rank(axis=1, pct=True).to_numpy(dtype="float64")
    valid = np.isfinite(p) & np.isfinite(b)
    n = valid.sum(axis=1)
    P = np.where(valid, p, 0.0); B = np.where(valid, b, 0.0)
    mp = P.sum(1) / np.maximum(n, 1); mb = B.sum(1) / np.maximum(n, 1)
    dp = np.where(valid, P - mp[:, None], 0.0); db = np.where(valid, B - mb[:, None], 0.0)
    rho = (dp * db).sum(1) / (np.sqrt((dp * dp).sum(1)) * np.sqrt((db * db).sum(1)) + 1e-12)
    rho[n < 5] = np.nan
    return float(np.nanmean(np.abs(rho)))


def extend_bundle(bundle: DatasetBundle, ext_panels: dict[str, pd.DataFrame],
                  rho_max: float = 0.7) -> tuple[DatasetBundle, dict[str, float]]:
    """把外部因子并入 bundle 的 features 与 feature_cols。

    返回 (新 bundle, {因子: max|ρ| vs 现役})。超过 rho_max 的因子被**剔除**（不并入）。
    """
    ref = bundle.features[bundle.feature_cols[0]]
    mask = bundle.universe.astype(bool)
    feats = dict(bundle.features)
    cols = list(bundle.feature_cols)
    rhos: dict[str, float] = {}
    for name, panel in ext_panels.items():
        r = max_abs_rho(panel, ref, mask)
        rhos[name] = r
        if r > rho_max:
            continue                      # 共线剔除
        feats[name] = panel.astype("float32")
        cols.append(name)
    new = replace(bundle, features=feats, feature_cols=tuple(sorted(cols)))
    return new, rhos


def load_screen_survivors(cfg: Config, stamp: str | None = None) -> list[str]:
    """从证据包读回筛选幸存者（默认最新一份）。"""
    root = Path(cfg.data.root).resolve().parents[0] / "experiments"
    files = sorted(root.glob("newdata_*/screen_*.json"))
    if not files:
        raise FileNotFoundError("external_features: 未找到 newdata 筛选证据包（先跑 scripts/newdata_chain.py）")
    f = files[-1] if stamp is None else next(p for p in files if stamp in p.name)
    return list(json.loads(f.read_text(encoding="utf-8"))["survivors"])
