"""DL-v2-LN —— BatchNorm → LayerNorm 单开关变体（判据链 #42，2026-10-11）。

预注册：`plans/ln-prereg-20261011.md`（**先于跑数冻结**）
诊断依据：`experiments/arch_audit_20261011/README.md`（P0）

## 为什么新建模块而不改 `net2.py`
`net3.py`（FiLM 条件化）与 `net4.py`（截面注意力）都 **import 了 net2 的组件**
（`net3.py:11` / `net4.py:12` 均自述"主干与 net2 逐字相同"）
⇒ 改 `net2.py` 的原件会污染那两条已判负的干预线，破坏可追溯性。
⇒ 本模块**只新增**，原件逐字不动。

## 唯一的变量
`ResBlock` 的归一化层：`nn.BatchNorm1d(dim)` → `nn.LayerNorm(dim)`。
其余（深度 3、宽度 256、NumericEmbedding、Rank-Gauss 输入、损失、优化器、种子）**逐字不变**。

## ⭐ 等价性设计（承自 net3/net4 的既有惯例）
net3/net4 都写有「**初始化时条件化 = 无条件化**（γ≡1, β≡0）」。
本模块沿用同一原则：**`elementwise_affine=False`**
⇒ LayerNorm 不引入额外的**仿射自由度**（不加 weight/bias）
⇒ 参数量差异**仅为** BN 侧的 `2×dim`，差异可完全解释（仪器门 G4 可账平）。
"""
from __future__ import annotations

import torch
from torch import nn

from fterm.model.net2 import (DROPOUT, EMBED_DIM, HIDDEN, N_BINS, N_BLOCKS,  # noqa: F401
                              NumericEmbedding)

__all__ = ["ResBlockLN", "CrossSectionNetLN", "LN_N_PARAMS_DELTA"]


class ResBlockLN(nn.Module):
    """与 `net2.ResBlock` **逐字同构**，仅把 BatchNorm1d 换成 LayerNorm。

    结构对照（`net2.py:73-84`）：
        net2 : x + fc2(drop(relu(fc1(drop(relu(bn(x)))))))
        本模块: x + fc2(drop(relu(fc1(drop(relu(ln(x)))))))
    """

    def __init__(self, dim: int, dropout: float = DROPOUT):
        super().__init__()
        # ⭐ elementwise_affine=False ⇒ 无 weight/bias ⇒ 与 BN 的参数差异可完全账平
        self.bn = nn.LayerNorm(dim, elementwise_affine=False)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(self.drop(torch.relu(self.bn(x))))
        h = self.fc2(self.drop(torch.relu(h)))
        return x + h


class CrossSectionNetLN(nn.Module):
    """DL-v2 主网络：数值 embedding → **LN** ResNet 主干 → 回归头。

    与 `net2.CrossSectionNet` 的唯一差异是 `ResBlockLN` 取代 `ResBlock`；
    `NumericEmbedding`（**内含 LayerNorm**，见 `net2.py:65`）直接复用 net2 的实现。
    """

    def __init__(self, n_features: int, hidden: int = HIDDEN, n_blocks: int = N_BLOCKS,
                 embed_dim: int = EMBED_DIM, bins: int = N_BINS,
                 dropout: float = DROPOUT, n_embed_features: int | None = None):
        super().__init__()
        self.n_features = n_features
        n_embed = n_embed_features if n_embed_features is not None else n_features
        self.nemb = NumericEmbedding(n_embed, hidden, bins, embed_dim)
        self.inp = nn.Linear(n_features, hidden)
        self.blocks = nn.Sequential(*[ResBlockLN(hidden, dropout)
                                      for _ in range(n_blocks)])
        self.head = nn.Linear(hidden, 1)

    def forward(self, x_rankgauss: torch.Tensor, bin_idx: torch.Tensor) -> torch.Tensor:
        lin = self.inp(x_rankgauss)
        h = self.nemb(bin_idx, lin)
        h = self.blocks(h)
        return self.head(torch.relu(h)).squeeze(-1)


#: 单个 ResBlock 的参数差异（BN 侧多 weight+bias；LN 侧 elementwise_affine=False ⇒ 0）
#: ⚠ 口径：**BN 参数数 − LN 参数数**（正数表示 LN 变体更少）
LN_N_PARAMS_DELTA = 2 * HIDDEN * N_BLOCKS
