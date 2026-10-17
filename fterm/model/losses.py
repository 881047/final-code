"""训练损失族（模型升级臂 E，2026-10-09）。

背景：#14 已试 MSE + RankNet(pairwise)；#18 证「选模指标与损失加权不是杠杆」。
但**listwise**（ListNet/Softmax、ListMLE）从未试过——pairwise 只用到相邻对
（`train_production.py` 同日对覆盖仅 ~0.24%/epoch/日），listwise 直接对**整日截面**
做概率分布对齐，是机制上不同的一类。

本模块全部按**日截面**（同一交易日为一组）计算，PIT 安全：
  - `listnet_loss`：Plackett-Luce / ListNet 交叉熵（Softmax 归一）。
  - `listmle_loss`：ListMLE（对最优排列的负对数似然）。
  - `ranknet_loss`：batch 内同日相邻对（与 #14 逐字同款，供对照复现）。
  - `uncertainty_weight`：多任务不确定性加权（Kendall et al. 2018）Σ(1/2σᵢ²)Lᵢ + logσᵢ。
  - `MultitaskLoss`：把上述组装成可插拔的训练目标。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _grouped_softmax(out: torch.Tensor, y: torch.Tensor, day_ids: torch.Tensor,
                     temp: float = 1.0) -> torch.Tensor:
    """按日截面做 ListNet 交叉熵。

    out/y/day_ids 同长（batch 行）。对每个 day 组：
      p_i = softmax(out_i)（模型分），q_i = softmax(y_i / temp)（标签分，Y 为标准化的 fwd_ret）。
      loss = −Σ q_i log p_i。
    组内样本 < 2 的日跳过（无排序定义）。
    """
    total, n_groups = out.new_zeros(()), 0
    for g in torch.unique(day_ids):
        m = day_ids == g
        if int(m.sum()) < 2:
            continue
        o, t = out[m], y[m]
        logp = F.log_softmax(o, dim=0)
        q = F.softmax(t / temp, dim=0)
        total = total - (q * logp).sum()
        n_groups += 1
    return total / max(n_groups, 1)


def listnet_loss(out: torch.Tensor, y: torch.Tensor, day_ids: torch.Tensor,
                 temp: float = 1.0) -> torch.Tensor:
    """ListNet / Plackett-Luce 截面交叉熵。"""
    return _grouped_softmax(out, y, day_ids, temp)


def listmle_loss(out: torch.Tensor, y: torch.Tensor, day_ids: torch.Tensor) -> torch.Tensor:
    """ListMLE：按标签降序的最优排列的负对数似然（组内）。"""
    total, n_groups = out.new_zeros(()), 0
    for g in torch.unique(day_ids):
        m = day_ids == g
        if int(m.sum()) < 2:
            continue
        o = out[m]
        order = torch.argsort(y[m], descending=True)
        o = o[order]
        # log-sum-exp 后缀：−Σ_k (o_k − logsumexp(o_k..o_{K−1}))
        K = o.numel()
        rev_cum = torch.logcumsumexp(o.flip(0), dim=0).flip(0)
        total = total - (o - rev_cum).sum()
        n_groups += 1
    return total / max(n_groups, 1)


def ranknet_loss(out: torch.Tensor, y: torch.Tensor, day_ids: torch.Tensor) -> torch.Tensor:
    """batch 内同日相邻对 pairwise（与 train_production.py 逐字同款，供对照复现）。"""
    order = torch.argsort(day_ids, stable=True)
    si, sj = order[:-1], order[1:]
    m = (day_ids[si] == day_ids[sj]) & (y[si] != y[sj])
    if not bool(m.any()):
        return out.sum() * 0.0
    i_, j_ = si[m], sj[m]
    sign = torch.sign(y[i_] - y[j_])
    return F.softplus(-(out[i_] - out[j_]) * sign).mean()


def mse_loss(out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(out, y)


class MultitaskLoss(torch.nn.Module):
    """多任务可插拔损失：主头 + 若干 horizon 头，按不确定性加权（Kendall 2018）。

    parts: {任务名: (预测, 标签)}；weights: {任务名: 固定权重 或 None}。
    固定权重时 loss = Σ wᵢ · Lᵢ；None 时启用可学习 log σᵢ 做不确定性加权。
    """

    def __init__(self, task_names: list[str], base_loss: str = "mse",
                 fixed_weights: dict[str, float] | None = None):
        super().__init__()
        self.task_names = list(task_names)
        self.base_loss = base_loss
        self.fixed = fixed_weights or {}
        learnable = [t for t in self.task_names if self.fixed.get(t) is None]
        self.log_var = torch.nn.ParameterDict(
            {t: torch.nn.Parameter(torch.zeros(())) for t in learnable})

    def _one(self, out, y, day_ids):
        if self.base_loss == "mse":
            return mse_loss(out, y)
        if self.base_loss == "listnet":
            return listnet_loss(out, y, day_ids)
        if self.base_loss == "listmle":
            return listmle_loss(out, y, day_ids)
        if self.base_loss == "ranknet":
            return ranknet_loss(out, y, day_ids)
        raise ValueError(f"未知 base_loss: {self.base_loss}")

    def forward(self, parts: dict[str, tuple[torch.Tensor, torch.Tensor]],
                day_ids: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        total = None
        logs: dict[str, float] = {}
        for t in self.task_names:
            if t not in parts:
                continue
            out, y = parts[t]
            L = self._one(out, y, day_ids)
            w = self.fixed.get(t)
            if w is None:
                # 不确定性加权：0.5·exp(−logvar)·L + 0.5·logvar
                lv = self.log_var[t]
                term = 0.5 * torch.exp(-lv) * L + 0.5 * lv
            else:
                term = w * L
            total = term if total is None else total + term
            logs[t] = float(L.detach())
        if total is None:
            raise ValueError("MultitaskLoss: parts 与 task_names 无交集")
        return total, logs
