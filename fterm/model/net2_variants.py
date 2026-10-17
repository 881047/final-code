"""DL-v2 变体 —— **A2 分桶数** 与 **A3 激活函数**（判据链 #46，2026-10-11）。

预注册：`plans/a2a3-prereg-20261011.md`（**先于跑数冻结**）

## 为什么新建模块而不改 `net2.py`
`net3.py`（FiLM 条件化）与 `net4.py`（截面注意力）都 **import 了 net2 的组件**
（两者自述"主干与 net2 逐字相同"）⇒ 改 `net2.py` 的原件会污染那两条已判负的干预线。
⇒ 本模块**只新增**，原件逐字不动（`tests/test_net2_ln.py::test_production_net2_is_locked` 已钉）。

## ⭐ 单变量纪律（每个变体只改一个东西）
  `CrossSectionNetBins`  —— 只改 `bins`（A2）；其余（hidden/blocks/dropout/embed_dim）不变
  `CrossSectionNetAct`   —— 只改激活（ReLU → PReLU）；其余不变

## ⚠ 参数量须可完全账平
  A2：`bins×8×25` 的 embedding 表随 bins 变（这是 A2 的**内在成本**，非额外变量）
  A3：PReLU 每块多 `dim` 个参数 ⇒ 3 块共 `3×hidden`（预注册 §6-3 已登记）
"""
from __future__ import annotations

import torch
from torch import nn

from fterm.model.net2 import (DROPOUT, EMBED_DIM, HIDDEN, N_BINS,  # noqa: F401
                              N_BLOCKS, NumericEmbedding)

__all__ = ["ResBlockAct", "CrossSectionNetBins", "CrossSectionNetAct"]


class ResBlockAct(nn.Module):
    """与 `net2.ResBlock` **逐字同构**，仅把 `relu` 换成 `PReLU`（A3 变量）。"""

    def __init__(self, dim: int, dropout: float = DROPOUT, init: float = 0.25):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        # ⭐ 每块两个 PReLU（与 forward 里两处 relu 对齐）⇒ 参数量 +2×dim/块
        self.act1 = nn.PReLU(num_parameters=dim, init=init)
        self.act2 = nn.PReLU(num_parameters=dim, init=init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(self.drop(self.act1(self.bn(x))))
        h = self.fc2(self.drop(self.act2(h)))
        return x + h


class CrossSectionNetBins(nn.Module):
    """A2 变体：**只**改分桶数（`bins`）。其余与 `net2.CrossSectionNet` 相同。"""

    def __init__(self, n_features: int, hidden: int = HIDDEN, n_blocks: int = N_BLOCKS,
                 embed_dim: int = EMBED_DIM, bins: int = N_BINS,
                 dropout: float = DROPOUT, n_embed_features: int | None = None):
        super().__init__()
        self.n_features = n_features
        n_embed = n_embed_features if n_embed_features is not None else n_features
        self.bins = int(bins)
        self.nemb = NumericEmbedding(n_embed, hidden, self.bins, embed_dim)
        self.inp = nn.Linear(n_features, hidden)
        # ⚠ **直接复用 `net2.ResBlock`**，不套 Sequential —— 套了会让
        #    `blocks.0.bn.*` 变成 `blocks.0.0.bn.*`，破坏 state_dict 键名
        #    （`arch_chain` 与生产加载都依赖原键名）。
        from fterm.model.net2 import ResBlock
        self.blocks = nn.Sequential(*[ResBlock(hidden, dropout)
                                      for _ in range(n_blocks)])
        self.head = nn.Linear(hidden, 1)

    def forward(self, x_rankgauss: torch.Tensor, bin_idx: torch.Tensor) -> torch.Tensor:
        lin = self.inp(x_rankgauss)
        h = self.nemb(bin_idx, lin)
        h = self.blocks(h)
        return self.head(torch.relu(h)).squeeze(-1)


class CrossSectionNetAct(nn.Module):
    """A3 变体：**只**改激活（ReLU → PReLU）。其余与 `net2.CrossSectionNet` 相同。

    ⚠ `forward` 的**两处 relu**：`blocks` 内的（已由 `ResBlockAct` 接管）
      与 `head` 前的 `torch.relu(h)` —— 后者是**输出头激活**，
      预注册 §2 的 A3 变量**只针对残差块内的激活** ⇒ 输出头保持 `torch.relu` 不变。
    """

    def __init__(self, n_features: int, hidden: int = HIDDEN, n_blocks: int = N_BLOCKS,
                 embed_dim: int = EMBED_DIM, bins: int = N_BINS,
                 dropout: float = DROPOUT, n_embed_features: int | None = None):
        super().__init__()
        self.n_features = n_features
        n_embed = n_embed_features if n_embed_features is not None else n_features
        self.nemb = NumericEmbedding(n_embed, hidden, bins, embed_dim)
        self.inp = nn.Linear(n_features, hidden)
        self.blocks = nn.Sequential(*[ResBlockAct(hidden, dropout)
                                      for _ in range(n_blocks)])
        self.head = nn.Linear(hidden, 1)

    def forward(self, x_rankgauss: torch.Tensor, bin_idx: torch.Tensor) -> torch.Tensor:
        lin = self.inp(x_rankgauss)
        h = self.nemb(bin_idx, lin)
        h = self.blocks(h)
        return self.head(torch.relu(h)).squeeze(-1)
