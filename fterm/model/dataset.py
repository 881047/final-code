"""滚动窗口张量化（净化折叠 / RevIN / 样本权重）—— 面板采样版。

样本定义：(股票 code, 日期 t) →
    x[code,t] = 特征面板 t-lookback+1 .. t 行（含 t）的该列堆叠，全非 NaN 才成样本
    y[code,t] = (fwd_dir, fwd_ret, fwd_vol)
    w[code,t] = 样本权重（volz: |fwd_ret|/fwd_vol，clip [0.25,4]）

净化（López de Prado 2018，继承期货版 docs/03 P0）：fwd_ret 标签跨 h+1 根 bar。
训练样本日期位置 ≥ 测试首日位置 − (h+1) 的一律剔除，防标签窗口伸入测试段。

采样密度（A股特有约束）：全市场 5000+ 股 × 全史日期在 CPU 上不可训练，
max_stocks_per_day 控制每日采样数——按日期序轮转确定性地切股票 cohort
（同一数据同一结果；不引入任何未来信息），0 = 不采样（研究/冒烟用）。

RevIN-lite（继承期货版 P1）：连续特征逐窗口 z-score；xrank_* 截面语义，
保持全局缩放。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset

BINARY_FEATURES: set[str] = set()
BINARY_FEATURE_PREFIXES = ("xrank_",)
REVIN_EPS = 1e-6


def revin_mask_for(feature_cols: tuple[str, ...]) -> np.ndarray:
    """True = 连续特征（做窗口内实例标准化），False = 截面/二值特征（保持全局缩放）。"""
    return np.array([
        not (c in BINARY_FEATURES or c.startswith(BINARY_FEATURE_PREFIXES))
        for c in feature_cols
    ], dtype=bool)


@dataclass(frozen=True)
class WindowSample:
    x: np.ndarray            # (T, F)
    y_dir: float
    y_ret: float
    y_vol: float
    product: str             # ts_code（沿用字段名，下游评估/日志不改）
    date: pd.Timestamp
    w: float = 1.0


def _day_cohort(codes: list[str], i: int, k: int) -> list[str]:
    """第 i 个交易日的确定性股票 cohort：按日期序轮转，全程均匀覆盖、无未来信息。"""
    n = len(codes)
    if k <= 0 or k >= n:
        return codes
    off = (i * k) % n
    idx = [(off + j) % n for j in range(k)]
    return [codes[j] for j in idx]


def stack_samples(features: dict[str, pd.DataFrame], labels: dict[str, pd.DataFrame],
                  feature_cols: tuple[str, ...], lookback: int, universe: pd.DataFrame,
                  weight_mode: str = "none", max_stocks_per_day: int = 0,
                  meta_primary: pd.DataFrame | None = None) -> list[WindowSample]:
    """面板 → (股票,日期) 样本流。按日期排序（时间顺序对 walk-forward 关键）。

    meta_primary（task=meta_reversal）：反转主信号方向面板（±1/0）。
    提供时方向头标签改为 meta 任务 y_dir = sign(fwd_ret × 主信号方向)
    （"跟随主信号是否赚钱"），主信号为 0/NaN 的样本剔除（元标签无定义）。
    """
    cols = list(feature_cols)
    f0 = features[cols[0]]
    dates = f0.index
    codes = list(f0.columns)
    ret, vol = labels["fwd_ret"], labels["fwd_vol"]
    dirp = labels["fwd_dir"]

    samples: list[WindowSample] = []
    for i in range(lookback - 1, len(dates) - 1):
        d = dates[i]
        coh = _day_cohort(codes, i, max_stocks_per_day)
        coh = [c for c in coh if c in universe.columns]
        if not coh:
            continue
        # 2026-10-08 B2：按日一次取整段布尔掩码（原为逐股 universe.at[d, code]）；
        # 此处不建全量 UniverseIndex —— cohort 很小（≤max_stocks_per_day），
        # 全量索引的 O(T×N) 拷贝反而比这里的 O(k) 取值更贵。
        uni_ok = universe.loc[d, coh].to_numpy(dtype=bool)
        # (lookback × n_cohort) 行块 × F 特征 → 逐股拼 (lookback × F)
        rows = slice(i - lookback + 1, i + 1)
        mats = {c: features[c].iloc[rows][coh].to_numpy(dtype=np.float64) for c in cols}
        r_i = ret.loc[d, coh].to_numpy(dtype=np.float64)
        v_i = vol.loc[d, coh].to_numpy(dtype=np.float64)
        p_i = (meta_primary.loc[d, coh].to_numpy(dtype=np.float64)
               if meta_primary is not None else None)
        for j, code in enumerate(coh):
            if not uni_ok[j]:
                continue
            window = np.stack([mats[c][:, j] for c in cols], axis=1)
            if np.isnan(window).any() or np.isnan(r_i[j]) or np.isnan(v_i[j]):
                continue
            if p_i is not None:
                pd_ = p_i[j]
                if np.isnan(pd_) or pd_ == 0:
                    continue        # 主信号未定义/中性：元标签无定义
                y_dir = float(np.sign(r_i[j] * pd_))
            else:
                y_dir = float(dirp.at[d, code])
            if weight_mode == "volz":
                w = float(np.clip(abs(r_i[j]) / (v_i[j] + 1e-6), 0.25, 4.0))
            else:
                w = 1.0
            samples.append(WindowSample(window, y_dir, float(r_i[j]),
                                        float(v_i[j]), code, d, w))
    return samples


class FtermDataset(TorchDataset):
    def __init__(self, samples: list[WindowSample], mean: np.ndarray, std: np.ndarray,
                 revin_mask: np.ndarray | None = None):
        self.samples = samples
        self.mean = mean
        self.std = std
        self.revin_mask = revin_mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        x = (s.x - self.mean) / self.std
        if self.revin_mask is not None:
            x = x.copy()
            cont = x[:, self.revin_mask]
            mu = cont.mean(axis=0, keepdims=True)
            sd = cont.std(axis=0, keepdims=True)
            x[:, self.revin_mask] = (cont - mu) / (sd + REVIN_EPS)
        return (
            torch.from_numpy(x.astype(np.float32)),
            torch.tensor(s.y_dir, dtype=torch.float32),
            torch.tensor(s.y_ret, dtype=torch.float32),
            torch.tensor(np.log1p(s.y_vol * 100), dtype=torch.float32),  # vol 对数稳定化
            torch.tensor(s.w, dtype=torch.float32),
        )


def fit_scaler(samples: list[WindowSample]) -> tuple[np.ndarray, np.ndarray]:
    """在给定样本集上算逐特征 mean/std（必须只传训练索引的样本）。"""
    shapes = {s.x.shape for s in samples}
    assert len(shapes) == 1, f"样本窗口形状不一致: {shapes}（上游数据流损坏）"
    allx = np.stack([s.x for s in samples])          # (N, T, F)
    flat = allx.reshape(-1, allx.shape[-1])
    mean = np.nanmean(flat, axis=0)
    std = np.nanstd(flat, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def _date_purge_count(samples: list[WindowSample], boundary_idx: int, label_span: int) -> int:
    """boundary_idx 段首之前要剔除的训练样本数（按日期位置净化 label_span 个交易日）。"""
    dates = sorted({s.date for s in samples})
    first = samples[boundary_idx].date
    p = dates.index(first)
    if p < label_span:
        return boundary_idx        # 训练段不足：全保（上游保证 min_train）
    cutoff = dates[p - label_span]
    return sum(1 for s in samples[:boundary_idx] if s.date >= cutoff)


def time_split(samples: list[WindowSample], train_ratio: float = 0.7,
               val_ratio: float = 0.15, label_span: int = 1) -> tuple[list[int], list[int], list[int]]:
    """按日期顺序切 train/val/test 索引（禁止随机打散）。train 与后续段按 label_span 净化。"""
    n = len(samples)
    i1, i2 = int(n * train_ratio), int(n * (train_ratio + val_ratio))
    purge = _date_purge_count(samples, i1, label_span)
    return list(range(i1 - purge)), list(range(i1, i2)), list(range(i2, n))


class WalkForwardFolds:
    """expanding-window walk-forward + 净化：train = [0, k*fold) 净化后，test = [k*fold, (k+1)*fold)。"""

    def __init__(self, samples: list[WindowSample], n_folds: int = 4, min_train: int = 200,
                 label_span: int = 1):
        self.samples = samples
        self.label_span = label_span
        n = len(samples)
        self.edges = []
        test_start = n - n_folds * ((n - min_train) // n_folds)
        fold = (n - test_start) // n_folds
        for k in range(n_folds):
            tr_end = test_start + k * fold
            te_end = tr_end + fold if k < n_folds - 1 else n
            self.edges.append((tr_end, te_end))

    def __len__(self) -> int:
        return len(self.edges)

    def fold(self, k: int) -> tuple[list[int], list[int]]:
        tr_end, te_end = self.edges[k]
        purge = _date_purge_count(self.samples, tr_end, self.label_span)
        return list(range(tr_end - purge)), list(range(tr_end, te_end))
