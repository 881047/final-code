"""DL-v4 —— 截面自注意力网（模型升级臂 D1，2026-10-09）。

动机（docs/audit-model-20261009.md P1-4）：
现有 `net2.CrossSectionNet` 对**每只股票独立打分**——同一交易日的股票之间没有任何
交互。但截面排序的本质是**相对定价**：同行业/同题材/同风格的股票之间存在横截面
传导（补涨、比价、拥挤度）。注意力机制正是建模"股票看股票"的自然形式。

设计：
  - 主干与 net2 逐字相同（Rank-Gauss 输入 + 数值分桶 embedding + ResNet-MLP）；
  - 在主干之后插入**一层截面自注意力**（`nn.MultiheadAttention`），
    用 `day_id` 构造**块对角掩码**——只允许同一交易日的股票互相注意（PIT：绝不跨日）；
  - 残差 + LayerNorm，使注意力是"增量"而非替换（初始化即可退化为 net2 行为）；
  - 单头输出，与 F0 可比。

代价：注意力矩阵为 O(B²)（B=batch 行）。BATCH=4096 ⇒ 4096²×heads ≈ 67M/头，
RTX4060 8GB 可承受（约 0.3GB/头）；若显存不足需降 batch 或按日分块。
"""
from __future__ import annotations

import torch
from torch import nn

from .net2 import DROPOUT, EMBED_DIM, HIDDEN, N_BINS, N_BLOCKS, NumericEmbedding, ResBlock


class CrossSectionAttentionNet(nn.Module):
    """DL-v4：主干 + 截面自注意力（同 day_id 块内）。"""

    def __init__(self, n_features: int, hidden: int = HIDDEN, n_blocks: int = N_BLOCKS,
                 n_heads: int = 4, embed_dim: int = EMBED_DIM, bins: int = N_BINS,
                 dropout: float = DROPOUT, n_embed_features: int | None = None):
        super().__init__()
        self.n_features = n_features
        n_embed = n_embed_features if n_embed_features is not None else n_features
        self.nemb = NumericEmbedding(n_embed, hidden, bins, embed_dim)
        self.inp = nn.Linear(n_features, hidden)
        self.blocks = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.attn = nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x_rankgauss: torch.Tensor, bin_idx: torch.Tensor,
                day_ids: torch.Tensor) -> torch.Tensor:
        lin = self.inp(x_rankgauss)
        h = self.nemb(bin_idx, lin)
        h = self.blocks(h)
        # 块对角掩码：同一天为 False（可见），跨天为 True（屏蔽）。
        # 对角线恒可见 ⇒ 每行至少有 1 个有效 key，softmax 不会 NaN。
        block = day_ids[:, None] != day_ids[None, :]
        a, _ = self.attn(h.unsqueeze(0), h.unsqueeze(0), h.unsqueeze(0), attn_mask=block)
        h = self.attn_norm(h + a.squeeze(0))
        return self.head(torch.relu(h)).squeeze(-1)
