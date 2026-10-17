"""TriHeadNet —— 双通路三头混合网络（本项目模型创新点）。

通路 A（线性，DLinear 血统）：逐特征时间维线性映射，保留可线性外推的
        趋势/自相关结构；参数少、不易过拟合，是"复杂模型必须打败"的内嵌基线。
通路 B（非线性，GRU）：捕捉阈值/交互等非线性时序结构。
融合：两通路拼接 → 小 MLP → 三头输出：
    dir_logit  未来 H 日方向（训练：BCE，推理：sigmoid 概率）
    q_lo/q_hi  未来收益分位（训练：pinball loss @ τ=0.1/0.9）
    vol_logit  未来 H 日波动（训练：MSE on log1p(vol*100)）

设计动机（调研结论）：外生特征已进特征矩阵（期限结构/截面 rank）；
方向+分位+波动三头让下游 sizing 直接拿到风险预算所需的 vol 预测，
不必再用历史波动兜底。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TriHeadNet(nn.Module):
    def __init__(self, n_features: int, lookback: int, d_hidden: int = 48,
                 n_layers: int = 1, dropout: float = 0.1, use_linear: bool = True,
                 use_gru: bool = True):
        super().__init__()
        assert use_linear or use_gru, "至少启用一条通路"
        self.use_linear, self.use_gru = use_linear, use_gru
        d_in = 0
        if use_linear:
            # 逐通道时间线性（DLinear 通道独立思想的轻量版：共享投影）
            self.linear_path = nn.Conv1d(n_features, 32, kernel_size=lookback)
            d_in += 32
        if use_gru:
            self.gru = nn.GRU(n_features, d_hidden, n_layers, batch_first=True,
                              dropout=dropout if n_layers > 1 else 0.0)
            d_in += d_hidden
        self.trunk = nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(), nn.Dropout(dropout))
        self.head_dir = nn.Linear(64, 1)
        self.head_q = nn.Linear(64, 2)     # q_lo, q_hi
        self.head_vol = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: (B, T, F)
        feats = []
        if self.use_linear:
            feats.append(self.linear_path(x.transpose(1, 2)).squeeze(-1))  # (B,32)
        if self.use_gru:
            out, _ = self.gru(x)
            feats.append(out[:, -1, :])                                     # (B,H)
        z = self.trunk(torch.cat(feats, dim=1))
        return {
            "dir_logit": self.head_dir(z).squeeze(-1),
            "q": self.head_q(z),
            "vol": self.head_vol(z).squeeze(-1),
        }


def trihead_loss(out: dict[str, torch.Tensor], y_dir: torch.Tensor, y_ret: torch.Tensor,
                 y_vol: torch.Tensor, taus: tuple[float, float] = (0.1, 0.9),
                 w: tuple[float, float, float] = (1.0, 0.5, 0.3),
                 sample_w: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    """BCE(方向) + pinball(分位) + MSE(波动) 加权复合损失。

    sample_w：逐样本权重（volz 模式 = |fwd_ret|/fwd_vol clip [0.25,4]，docs/03 P2）。
    加权口径 = Σ(loss_i × w_i) / Σw_i（与无权重时的均值口径严格一致退化）。
    """
    bce_v = nn.functional.binary_cross_entropy_with_logits(out["dir_logit"], (y_dir > 0).float(),
                                                           reduction="none")
    q_lo, q_hi = out["q"][:, 0], out["q"][:, 1]
    pin_v = _pinball(q_lo, y_ret, taus[0], reduction_none=True) + \
        _pinball(q_hi, y_ret, taus[1], reduction_none=True)
    vol_v = (out["vol"] - y_vol) ** 2

    if sample_w is None:
        bce = bce_v.mean()
        pin = pin_v.mean()
        vol_mse = vol_v.mean()
    else:
        sw = sample_w / sample_w.sum().clamp(min=1e-9) * len(sample_w)
        bce = (bce_v * sw).mean()
        pin = (pin_v * sw).mean()
        vol_mse = (vol_v * sw).mean()
    loss = w[0] * bce + w[1] * pin + w[2] * vol_mse
    return loss, {"bce": float(bce.detach()), "pinball": float(pin.detach()),
                  "vol_mse": float(vol_mse.detach())}


def _pinball(pred: torch.Tensor, target: torch.Tensor, tau: float,
             reduction_none: bool = False) -> torch.Tensor:
    d = target - pred
    per = torch.maximum(tau * d, (tau - 1.0) * d)
    return per if reduction_none else per.mean()


@torch.no_grad()
def predict_direction(net: TriHeadNet, x: torch.Tensor) -> torch.Tensor:
    """方向概率（0..1）。"""
    return torch.sigmoid(net(x)["dir_logit"])
