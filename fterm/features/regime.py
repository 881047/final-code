"""市场状态非秩通道特征（模型升级臂 A，2026-10-09）。

这些特征**横截面常数**（同日全市场同值），因此**绝不能**进入逐日截面秩白化通道
（秩恒 0.5 → 白化后恒 0，见 docs/audit-model-20261009.md P0-4）。它们只经
`fold_frame(..., z_cols=...)` 的旁路通道进入 `net3.RegimeCrossSectionNet` 的 FiLM。

来源：
  - `data/external/*.parquet`（regime_*/mkt_*，由 scripts/sync_external.py 提取）
  - `data/index/000905.SH.parquet`（中证500 基准，指数收益/均线偏离）
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..data import ashare as sa

# 旁路通道列（顺序冻结；新增列需在预注册登记）
REGIME_COLS: tuple[str, ...] = (
    "regime_state", "regime_score", "regime_drawdown", "regime_vol",
    "mkt_breadth_ma20", "mkt_limitup_cnt", "mkt_limitdn_cnt",
    "mkt_disp", "mkt_turnover",
    "idx_ret_20d", "idx_ma_dev",
)


def _broadcast(series: pd.Series, ref: pd.DataFrame) -> pd.DataFrame:
    v = series.reindex(ref.index).to_numpy(dtype="float64")
    return pd.DataFrame(np.repeat(v[:, None], ref.shape[1], axis=1),
                        index=ref.index, columns=ref.columns, dtype="float32")


def _index_features(cfg: Config, ref: pd.DataFrame, code: str = "000905.SH") -> dict[str, pd.DataFrame]:
    """指数特征（基准 000905）：20 日收益、相对 MA20 偏离。"""
    p = Path(cfg.data.root) / "index" / f"{code}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"regime: 缺指数文件 {p}（先跑 scripts/sync_index.py）")
    idx = pd.read_parquet(p)
    idx.index = pd.DatetimeIndex(idx.index)
    close = idx["close"].astype("float64")
    ret20 = close / close.shift(20) - 1.0
    ma_dev = close / close.rolling(20, min_periods=10).mean() - 1.0
    return {"idx_ret_20d": _broadcast(ret20, ref), "idx_ma_dev": _broadcast(ma_dev, ref)}


def build_regime_features(cfg: Config,
                          panels: dict[str, pd.DataFrame] | None = None) -> dict[str, pd.DataFrame]:
    """返回 REGIME_COLS 对应的宽面板（横截面常数）。缺列响亮报错。"""
    if panels is None:
        panels = sa.load_panels(cfg)
    ref = panels["close"]
    ext_dir = Path(cfg.data.root) / "external"
    out: dict[str, pd.DataFrame] = {}
    for c in REGIME_COLS:
        p = ext_dir / f"{c}.parquet"
        if p.exists():
            out[c] = pd.read_parquet(p).reindex(index=ref.index, columns=ref.columns).astype("float32")
    out.update({k: v for k, v in _index_features(cfg, ref).items() if k in REGIME_COLS})
    missing = [c for c in REGIME_COLS if c not in out]
    if missing:
        raise FileNotFoundError(f"regime: 缺旁路特征 {missing}（先跑 scripts/sync_external.py）")
    return {c: out[c] for c in REGIME_COLS}


def market_regime_labels(cfg: Config) -> pd.Series:
    """把 regime 状态转成 2×2 制度分层标签（趋势 × 波动），供 regime 分层 IC 用。

    趋势 = 指数 20 日收益正负；波动 = 指数 20 日已实现波动相对自身中位数高低。
    """
    p = Path(cfg.data.root) / "index" / "000905.SH.parquet"
    idx = pd.read_parquet(p)
    idx.index = pd.DatetimeIndex(idx.index)
    close = idx["close"].astype("float64")
    ret = close.pct_change()
    trend = (close / close.shift(20) - 1.0) > 0
    vol = ret.rolling(20, min_periods=10).std()
    vol_hi = vol > vol.expanding(min_periods=60).median()
    lab = pd.Series(index=close.index, dtype=object)
    lab[(trend) & (~vol_hi)] = "up_lowvol"
    lab[(trend) & (vol_hi)] = "up_highvol"
    lab[(~trend) & (~vol_hi)] = "down_lowvol"
    lab[(~trend) & (vol_hi)] = "down_highvol"
    return lab
