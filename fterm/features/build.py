"""特征矩阵组装 + 标签生成（面板化，point-in-time 安全）。

输出 DatasetBundle：
    features      dict[feature_col → 面板 DataFrame(date × ts_code)]
    labels        dict[fwd_ret/fwd_dir/fwd_vol → 面板]
    universe      PIT 股票池掩码面板（bool）
    feature_cols  特征列清单（进模型工件）
    fingerprints  数据指纹（可复现纪律）

标签口径（执行现实口径，继承期货版）：fwd_ret = hfq_open[t+h+1] / hfq_open[t+1] − 1，
即 t+1 开盘建仓、持有 h 日、t+1+h 开盘口径的可实现收益（复权价 ⇒ 含分红送转）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import Config
from ..data import ashare as sa
from . import tsfactors as ts


@dataclass(frozen=True)
class DatasetBundle:
    features: dict[str, pd.DataFrame]
    labels: dict[str, pd.DataFrame]
    universe: pd.DataFrame
    feature_cols: tuple[str, ...]
    fingerprints: dict[str, str] = field(default_factory=dict)


def build_features(cfg: Config, panels: dict[str, pd.DataFrame] | None = None,
                   universe: pd.DataFrame | None = None) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """全市场特征面板（不含截面 rank 与标签；截面需全市场对齐后统一加）。"""
    if panels is None:
        panels = sa.load_panels(cfg)
    if universe is None:
        universe = sa.universe_mask(cfg, panels)
    hfq = sa.hfq(panels)
    o, h, l, c = hfq["open"], hfq["high"], hfq["low"], hfq["close"]
    feats: dict[str, pd.DataFrame] = {}
    feats.update(ts.momentum(c, cfg.features.momentum_windows))
    feats[f"rev_{cfg.signal.rev_window}"] = ts.reversal(c, cfg.signal.rev_window)
    for w in cfg.features.vol_windows:
        feats[f"pvol_{w}"] = ts.parkinson_vol(h, l, w)
        feats[f"gkvol_{w}"] = ts.gk_vol(h, l, c, o, w)
    feats["atr14"] = ts.atr(panels["high"], panels["low"], panels["close"])   # 未复权：风控用真实价
    feats.update(ts.intraday_micro(o, h, l, c))
    feats[f"tstat_5"] = ts.ret_tstat(c, 5)
    feats["tstat_20"] = ts.ret_tstat(c, 20)
    feats.update(ts.liquidity(panels["turnover"], panels["amount"], panels["circ_mv"],
                              cfg.features.turnover_window))
    feats["weekday"] = ts.weekday_panel(c.index, c.columns)
    return feats, universe


def add_cross_sectional(feats: dict[str, pd.DataFrame], mom_col: str = "mom_20",
                        vol_col: str = "pvol_20") -> dict[str, pd.DataFrame]:
    """截面特征：每交易日全市场排名 ∈ (0,1) + 相对动量 + 截面离散度（制度状态）。

    mom_col/vol_col 由 build_dataset 按配置的最大窗口给出（生产 = mom_20/pvol_20）。
    """
    mom, vol = feats[mom_col], feats[vol_col]
    mom_rank = mom.rank(axis=1, pct=True)
    vol_rank = vol.rank(axis=1, pct=True)
    out = dict(feats)
    out[f"xrank_{mom_col}"] = mom_rank.astype("float32")
    out[f"xrank_{vol_col}"] = vol_rank.astype("float32")
    out[f"rel_{mom_col}"] = (mom.sub(mom.mean(axis=1), axis=0)).astype("float32")
    disp = mom.std(axis=1).to_numpy(dtype="float32")     # 截面离散度：同日全市场同值
    out["disp_20"] = pd.DataFrame(np.repeat(disp[:, None], mom.shape[1], axis=1),
                                  index=mom.index, columns=mom.columns, dtype="float32")
    return out


def add_labels(cfg: Config, panels: dict[str, pd.DataFrame],
               horizon: int | None = None) -> dict[str, pd.DataFrame]:
    """标签面板：fwd_ret = hfq_open[t+h+1]/hfq_open[t+1] − 1（t+1 开盘建仓可实现收益）。"""
    h = horizon if horizon is not None else cfg.features.label_horizon
    hfq = sa.hfq(panels)
    o, c = hfq["open"], hfq["close"]
    fwd_ret = (o.shift(-(h + 1)) / o.shift(-1) - 1.0).astype("float32")
    fwd_vol = (np.log(c / c.shift(1)).rolling(h).std().shift(-h)).astype("float32")
    return {"fwd_ret": fwd_ret, "fwd_dir": np.sign(fwd_ret).astype("float32"),
            "fwd_vol": fwd_vol}


def build_dataset(cfg: Config, compute_fingerprints: bool = True) -> DatasetBundle:
    """全流程：面板 → 特征 → 截面 → 标签 → 股票池 + 数据指纹。

    compute_fingerprints：runner 每日生产路径传 **False**——12 张全史面板逐列
    序列化哈希是分钟级纯开销，且决策路径无任何消费方（2026-10-06 审查）；
    研究/训练/回测脚本保持默认 True（工件与报告要记录输入指纹，复现纪律）。
    """
    panels = sa.load_panels(cfg)
    feats, universe = build_features(cfg, panels, universe=None)
    mom_col = f"mom_{max(cfg.features.momentum_windows)}"
    vol_col = f"pvol_{max(cfg.features.vol_windows)}"
    feats = add_cross_sectional(feats, mom_col=mom_col, vol_col=vol_col)
    labels = add_labels(cfg, panels)

    fps = ({f: sa.df_fingerprint(panels[f]) for f in sorted(panels)}
           if compute_fingerprints else {})
    return DatasetBundle(feats, labels, universe,
                         tuple(sorted(feats.keys())), fps)
