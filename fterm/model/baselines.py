"""基线模型 —— "复杂模型必须打败的东西"（调研纪律）。

基线1 DLinear-only：只留线性通路（use_gru=False）
基线2 GRU-only：只留非线性通路（use_linear=False）
两者与 TriHeadNet 共用训练/验证管线，指标同口径直接可比。
"""

from __future__ import annotations

import torch

from ..config import Config
from .dataset import FtermDataset, fit_scaler
from .net import TriHeadNet, trihead_loss
from .validate import FoldMetric, _eval_fold


def _fit_variant(cfg: Config, samples, use_linear: bool, use_gru: bool,
                 epochs: int = 15, rmask=None) -> TriHeadNet:
    mean, std = fit_scaler(samples[: int(len(samples) * 0.7)])
    torch.manual_seed(cfg.model.seeds[0])
    net = TriHeadNet(len(samples[0].x[0]), cfg.features.lookback, cfg.model.d_hidden,
                     cfg.model.n_layers, cfg.model.dropout,
                     use_linear=use_linear, use_gru=use_gru)
    from torch.utils.data import DataLoader

    dl = DataLoader(FtermDataset(samples, mean, std, rmask),
                    batch_size=cfg.model.batch_size, shuffle=True)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.model.lr)
    net.train()
    for _ in range(epochs):
        for xb, ydir, yret, yvol, sw in dl:
            opt.zero_grad()
            loss, _ = trihead_loss(net(xb), ydir, yret, yvol, sample_w=sw)
            loss.backward()
            opt.step()
    return net


def run_baselines(cfg: Config, samples, test_samples) -> dict[str, FoldMetric]:
    """在同一测试段上跑两个基线，返回指标（与主模型同 FoldMetric 口径）。"""
    out = {}
    for name, ul, ug in (("dlinear", True, False), ("gru_only", False, True)):
        net = _fit_variant(cfg, samples, ul, ug)
        mean, std = fit_scaler(samples[: int(len(samples) * 0.7)])
        m = _eval_fold(net, test_samples, mean, std)
        out[name] = m
    return out
