"""DL-v3 —— 市场状态条件化截面网（模型升级臂 A，2026-10-09）。

动机（docs/audit-model-20261009.md P0-4 / P1-1）：
现有 `net2.CrossSectionNet` 只能吃**逐日截面秩白化**后的特征。任何横截面常数
特征（指数收益、市场广度、涨跌停家数、regime 状态）在同一天所有股票上取值相同，
其截面秩恒为 0.5 → erf⁻¹(0)=0 → 白化后**恒为 0**，信息在进网前被完全消灭。
因此「市场状态非秩通道」（§40 点名的未测方向）在现有管道下**结构上无法测试**。

本模块提供：
  - `ZScaler`：旁路通道的标准化（**只 fit 训练段**，PIT）。
  - `RegimeCrossSectionNet`：主干与 net2 逐字相同（Rank-Gauss 输入 + 数值分桶
    embedding + ResNet-MLP），但在每个残差块后插入 **FiLM** 条件化
    `h ← γ(z)⊙h + β(z)`，让网络按市场状态调制其表示。

设计要点：
  - FiLM 的 γ 偏置初始化 1、β 偏置初始化 0、权重初始化 0 ⇒ 训练起点**等价于
    无条件模型**（net2），条件化只能带来增量，不会破坏已学到的信号。
  - 参数增量仅 `2 × z_dim × hidden × n_blocks`（z_dim≈8、hidden=256、blocks=3
    ⇒ ≈12k 参数），对 8GB 显存与 CPU 均无压力。
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .net2 import DROPOUT, EMBED_DIM, HIDDEN, N_BINS, N_BLOCKS, NumericEmbedding, ResBlock


class ZScaler:
    """旁路通道标准化器（PIT：只 fit 训练段）。"""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype("float64")
        self.std = std.astype("float64")

    @classmethod
    def fit(cls, z: np.ndarray) -> "ZScaler":
        z = np.asarray(z, dtype="float64")
        mean = np.nanmean(z, axis=0)
        std = np.nanstd(z, axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean, std)

    def transform(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype="float64")
        out = (z - self.mean) / self.std
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")


class FiLMBlock(nn.Module):
    """残差块 + FiLM 条件化：h ← ResBlock(h)；h ← γ(z)⊙h + β(z)。"""

    def __init__(self, hidden: int, z_dim: int, dropout: float = DROPOUT):
        super().__init__()
        self.block = ResBlock(hidden, dropout)
        self.to_gamma = nn.Linear(z_dim, hidden)
        self.to_beta = nn.Linear(z_dim, hidden)
        # 起点 = 恒等条件化（γ≡1, β≡0），保证不劣于无条件模型
        nn.init.zeros_(self.to_gamma.weight); nn.init.ones_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight); nn.init.zeros_(self.to_beta.bias)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = self.block(h)
        return self.to_gamma(z) * h + self.to_beta(z)


class RegimeCrossSectionNet(nn.Module):
    """DL-v3 主网络：数值 embedding → FiLM 条件化 ResNet 主干 → 回归头。"""

    def __init__(self, n_features: int, z_dim: int, hidden: int = HIDDEN,
                 n_blocks: int = N_BLOCKS, embed_dim: int = EMBED_DIM, bins: int = N_BINS,
                 dropout: float = DROPOUT, n_embed_features: int | None = None):
        super().__init__()
        self.n_features = n_features
        self.z_dim = z_dim
        n_embed = n_embed_features if n_embed_features is not None else n_features
        self.nemb = NumericEmbedding(n_embed, hidden, bins, embed_dim)
        self.inp = nn.Linear(n_features, hidden)
        self.blocks = nn.ModuleList([FiLMBlock(hidden, z_dim, dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden, 1)

    def forward(self, x_rankgauss: torch.Tensor, bin_idx: torch.Tensor,
                z: torch.Tensor) -> torch.Tensor:
        lin = self.inp(x_rankgauss)
        h = self.nemb(bin_idx, lin)
        for blk in self.blocks:
            h = blk(h, z)
        return self.head(torch.relu(h)).squeeze(-1)


def count_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())
