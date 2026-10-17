"""多头网络（模型升级臂 B，2026-10-09）。

与 `net2.CrossSectionNet` 共享同一套主干（Rank-Gauss 输入 + 数值分桶 embedding +
ResNet-MLP），但把单一回归头换成**每个 horizon 一个头**，训练时用
`losses.MultitaskLoss`（不确定性加权）联合优化。

设计要点：
  - 主干共享 ⇒ 表示层被迫同时服务多个 horizon，起到正则作用；
  - 各头独立 Linear(1) ⇒ 不增加主干容量，参数增量 = Σ(hidden+1) ≈ 5×257；
  - 推理时可只用 h=5 头（与 F0 可比），或输出融合分（`multihorizon.fused_score`）。
"""
from __future__ import annotations

import torch
from torch import nn

from .net2 import DROPOUT, EMBED_DIM, HIDDEN, N_BINS, N_BLOCKS, NumericEmbedding, ResBlock


class MultiHorizonNet(nn.Module):
    """共享主干 + 多 horizon 头。"""

    def __init__(self, n_features: int, horizons: tuple[int, ...] = (1, 3, 5, 10, 20),
                 hidden: int = HIDDEN, n_blocks: int = N_BLOCKS, embed_dim: int = EMBED_DIM,
                 bins: int = N_BINS, dropout: float = DROPOUT,
                 n_embed_features: int | None = None):
        super().__init__()
        self.horizons = tuple(horizons)
        self.n_features = n_features
        n_embed = n_embed_features if n_embed_features is not None else n_features
        self.nemb = NumericEmbedding(n_embed, hidden, bins, embed_dim)
        self.inp = nn.Linear(n_features, hidden)
        self.blocks = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.heads = nn.ModuleDict({f"h{h}": nn.Linear(hidden, 1) for h in self.horizons})

    def forward(self, x_rankgauss: torch.Tensor, bin_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        lin = self.inp(x_rankgauss)
        h = self.nemb(bin_idx, lin)
        h = self.blocks(h)
        h = torch.relu(h)
        return {name: head(h).squeeze(-1) for name, head in self.heads.items()}
