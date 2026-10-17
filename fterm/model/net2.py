"""DL-v2 —— 截面原生深度模型（判据链 #8，预注册 plans/dlv2-prereg-20261004.md）。

文献依据 plans/dlv2-literature-20261004.md：
  - 输入白化：逐交易日截面秩 → Rank-Gauss（erf⁻¹），PIT 安全（QPLUM 2019）
  - 数值分桶 embedding（每特征可学习查表，Gorishniy 2022）
  - ResNet-MLP 主干（Gorishniy 2021：表格 DL 唯一稳定追平 GBDT 的形态）
  - 5 种子集成（GKX 2020：ensembling = 一半增益）
与 v1（逐票时序窗 GRU）的本质差异：日截面独立打分，无时序窗。
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

N_BINS = 16          # 每特征分桶数（冻结）
EMBED_DIM = 8        # 桶 embedding 维度（冻结）
HIDDEN = 256         # ResNet 宽度（冻结）
N_BLOCKS = 3         # 残差块数（冻结）
DROPOUT = 0.1


def rank_gauss(x: np.ndarray) -> np.ndarray:
    """列向量截面秩 → erf⁻¹ 白化。输入 (N, F)，逐列独立（= 逐特征全期秩）。

    ⚠️ 调用方必须保证传入的是**当日截面内**的秩分位（PIT 由上游 stacker 保证，
    本函数只做单调白化）。秩 pct ∈ (0,1) → sqrt(2)·erfinv(2p−1) ∈ 正态分位。
    """
    import pandas as pd

    p = np.asarray(x, dtype="float64")
    # 向量化秩（2026-10-06：原逐列循环 + 循环内 import + 逐列建 Series）。
    # DataFrame.rank(axis=0) 与逐列 Series.rank 同为 average 并列语义 ⇒ 数值逐位一致。
    r = pd.DataFrame(p).rank(axis=0, pct=True, na_option="keep").fillna(0.5).to_numpy()
    r = np.clip(r, 1e-4, 1 - 1e-4)
    return (np.sqrt(2.0) * _erfinv(2.0 * r - 1.0)).astype("float32")


def _erfinv(v: np.ndarray) -> np.ndarray:
    import math

    try:
        from scipy.special import erfinv
        return erfinv(v)
    except Exception:
        # 无 scipy 的兜底（向量化牛顿法，精度 1e-6 足够）
        import numpy as _np

        _verf = _np.vectorize(math.erf, otypes=["float64"])
        a = _np.asarray(v, dtype="float64")
        x = _np.copy(a)
        for _ in range(6):
            x = x - (_verf(x) - a) / (2.0 / _np.sqrt(_np.pi) * _np.exp(-x * x) + 1e-12)
        return x


class NumericEmbedding(nn.Module):
    """每特征 N_BINS 桶可学习查表 + 线性投影残差（Gorishniy 2022 轻量版）。"""

    def __init__(self, n_features: int, hidden: int, bins: int = N_BINS, dim: int = EMBED_DIM):
        super().__init__()
        self.bins = bins
        self.embed = nn.ModuleList([nn.Embedding(bins, dim) for _ in range(n_features)])
        self.proj = nn.Linear(n_features * dim, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, bin_idx: torch.Tensor, lin: torch.Tensor) -> torch.Tensor:
        embs = [e(bin_idx[:, j]) for j, e in enumerate(self.embed)]
        e = torch.cat(embs, dim=-1)
        return self.norm(self.proj(e)) + lin


class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = DROPOUT):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(self.drop(torch.relu(self.bn(x))))
        h = self.fc2(self.drop(torch.relu(h)))
        return x + h


class CrossSectionNet(nn.Module):
    """DL-v2 主网络：数值 embedding → ResNet 主干 → 回归头。"""

    def __init__(self, n_features: int, hidden: int = HIDDEN, n_blocks: int = N_BLOCKS,
                 embed_dim: int = EMBED_DIM, bins: int = N_BINS,
                 dropout: float = DROPOUT, n_embed_features: int | None = None):
        super().__init__()
        self.n_features = n_features
        # embedding 只覆盖基础特征列；缺失指示等附加列仅走线性分支
        n_embed = n_embed_features if n_embed_features is not None else n_features
        self.nemb = NumericEmbedding(n_embed, hidden, bins, embed_dim)
        self.inp = nn.Linear(n_features, hidden)
        self.blocks = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden, 1)

    def forward(self, x_rankgauss: torch.Tensor, bin_idx: torch.Tensor) -> torch.Tensor:
        lin = self.inp(x_rankgauss)
        h = self.nemb(bin_idx, lin)
        h = self.blocks(h)
        return self.head(torch.relu(h)).squeeze(-1)


def fit_bins(x_train: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """训练折分位数桶边界（PIT：只 fit 训练段）。返回 (F, bins+1) 边界。"""
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = np.nanquantile(x_train, qs, axis=0)
    return edges.astype("float64")


def to_bin_idx(x_rankgauss: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """白化值 → 桶索引（edges 为 bins−1 条内边界 → 索引 0..bins−1）。"""
    idx = np.zeros_like(x_rankgauss, dtype="int64")
    for j in range(x_rankgauss.shape[1]):
        idx[:, j] = np.searchsorted(edges[:, j], x_rankgauss[:, j], side="right")
    return np.clip(idx, 0, edges.shape[1])
