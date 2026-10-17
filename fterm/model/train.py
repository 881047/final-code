"""训练循环：多种子训练 + 种子集成（种子稳健性纪律）。

铁律（继承自旧项目事故教训）：单种子的单快照增益不可信。本实现默认训练
多个种子，推理时对方向概率取**集成均值**，并把种子间分歧度（std）随
预测一起输出 —— 分歧大的信号置信度打折。

模型工件（.pt）自带：feature_cols / mean / std / lookback / revin_mask /
vol_cal / config 摘要哈希，推理端不得重新 fit 任何统计量。
配置改动（features/model dataclass 任一字段）→ digest 失效 → 拒载 → 必须重训。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import Config
from .dataset import FtermDataset, WindowSample, fit_scaler, revin_mask_for, time_split
from .net import TriHeadNet, trihead_loss


@dataclass
class TrainReport:
    seed: int
    best_val_loss: float
    epochs_run: int


@dataclass
class SeedEnsemble:
    nets: list[TriHeadNet]
    mean: np.ndarray
    std: np.ndarray
    feature_cols: tuple[str, ...]
    lookback: int
    revin_mask: np.ndarray | None = None
    vol_cal: float = 1.0             # 波动头校准因子（训练段 realized/pred 中位数比）

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """x: (T,F) 未标准化窗口 → 各头集成均值 + 方向分歧度。"""
        xn = (x - self.mean) / self.std
        if self.revin_mask is not None:
            cont = xn[:, self.revin_mask]
            mu = cont.mean(axis=0, keepdims=True)
            sd = cont.std(axis=0, keepdims=True)
            xn = xn.copy()
            xn[:, self.revin_mask] = (cont - mu) / (sd + 1e-6)
        t = torch.from_numpy(xn.astype(np.float32)).unsqueeze(0)
        dirs, qs, vols = [], [], []
        for net in self.nets:
            net.eval()
            out = net(t)
            dirs.append(float(torch.sigmoid(out["dir_logit"])))
            qs.append(out["q"][0].numpy())
            vols.append(float(out["vol"]))
        d = np.array(dirs)
        raw_vol = np.expm1(float(np.mean(vols))) / 100.0 * self.vol_cal   # 校准后的日波
        return {
            "p_up": float(d.mean()),
            "p_up_std": float(d.std()),      # 种子分歧：大 → 降置信
            "q_lo": float(np.mean([q[0] for q in qs])),
            "q_hi": float(np.mean([q[1] for q in qs])),
            "vol_pred": float(np.log1p(max(raw_vol, 0.0) * 100)),  # 保持 log1p 口径契约
        }


def _config_digest(cfg: Config) -> str:
    """摘要取自冻结 dataclass（运行期真实口径），而非 yaml 原文——
    进程内 replace 出的篡改配置同样会被工件校验拦住。"""
    payload = json.dumps({
        "features": dataclasses.asdict(cfg.features),
        "model": dataclasses.asdict(cfg.model),
    }, sort_keys=True, default=list).encode()
    return hashlib.sha256(payload).hexdigest()[:10]


def _train_one(cfg: Config, samples: list[WindowSample], seed: int,
               n_features: int, rmask: np.ndarray | None = None) -> tuple[TriHeadNet, TrainReport]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    label_span = cfg.features.label_horizon + 1
    tr_idx, va_idx, _ = time_split(samples, label_span=label_span)
    mean, std = fit_scaler([samples[i] for i in tr_idx])   # 只 fit 训练段
    dl_tr = DataLoader(FtermDataset([samples[i] for i in tr_idx], mean, std, rmask),
                       batch_size=cfg.model.batch_size, shuffle=True)
    dl_va = DataLoader(FtermDataset([samples[i] for i in va_idx], mean, std, rmask),
                       batch_size=512, shuffle=False)

    net = TriHeadNet(n_features, cfg.features.lookback, cfg.model.d_hidden,
                     cfg.model.n_layers, cfg.model.dropout)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.model.lr)
    best_loss, best_state, bad, epochs_run = float("inf"), None, 0, 0
    for ep in range(cfg.model.max_epochs):
        net.train()
        for xb, ydir, yret, yvol, sw in dl_tr:
            opt.zero_grad()
            out = net(xb)
            loss, _ = trihead_loss(out, ydir, yret, yvol, sample_w=sw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            vloss = 0.0
            n = 0
            for xb, ydir, yret, yvol, sw in dl_va:
                out = net(xb)
                l, _ = trihead_loss(out, ydir, yret, yvol, sample_w=sw)
                vloss += float(l.detach()) * len(xb)
                n += len(xb)
            vloss /= max(n, 1)
        epochs_run = ep + 1
        if vloss < best_loss - 1e-5:
            best_loss, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.model.patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, TrainReport(seed, best_loss, epochs_run)


def train_ensemble(cfg: Config, samples: list[WindowSample],
                   feature_cols: tuple[str, ...]) -> tuple[SeedEnsemble, list[TrainReport]]:
    nets, reports = [], []
    rmask = revin_mask_for(feature_cols) if cfg.model.revin else None
    for seed in cfg.model.seeds:
        net, rep = _train_one(cfg, samples, seed, len(feature_cols), rmask=rmask)
        nets.append(net)
        reports.append(rep)
    # 集成 scaler 与各网训练 scaler 必须同源：time_split 的训练段（含净化）
    # （2026-10-06 审查修复：原实现用"前 70% 含待净化样本"另 fit 一份，
    # 工件 mean/std 与网络实际训练统计量有细微漂移；不影响已落盘工件——
    # dlv2_production.pt 走 train_production.py 独立管线，不经过本函数。）
    tr_idx, _, _ = time_split(samples, label_span=cfg.features.label_horizon + 1)
    mean, std = fit_scaler([samples[i] for i in tr_idx])
    ens = SeedEnsemble(nets, mean, std, feature_cols, cfg.features.lookback,
                       revin_mask=rmask)
    # 波动头校准：只用训练段（无泄漏）。真实数据实测校准前中位预测 1.7%/日 vs 真实 ~0.9%。
    ens.vol_cal = _fit_vol_calibration(ens, [samples[i] for i in tr_idx])
    return ens, reports


def _fit_vol_calibration(ens: SeedEnsemble, train_samples: list[WindowSample],
                         max_n: int = 800) -> float:
    """cal = median(realized) / median(pred)，clip [0.5, 2]（防极端外推）。"""
    sub = train_samples[-max_n:]
    preds, reals = [], []
    for s in sub:
        out = ens.predict(s.x)
        preds.append(np.expm1(out["vol_pred"]) / 100.0)
        reals.append(max(s.y_vol, 1e-6))
    med_p, med_r = float(np.median(preds)), float(np.median(reals))
    if med_p <= 0:
        return 1.0
    return float(np.clip(med_r / med_p, 0.5, 2.0))


def save_ensemble(cfg: Config, ens: SeedEnsemble, path: Path, fingerprints: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dicts": [{k: v for k, v in n.state_dict().items()} for n in ens.nets],
        "mean": ens.mean, "std": ens.std, "vol_cal": ens.vol_cal,
        "revin_mask": ens.revin_mask,
        "feature_cols": list(ens.feature_cols), "lookback": ens.lookback,
        "config_digest": _config_digest(cfg),
        "fingerprints": fingerprints,
        "hp": dataclasses.asdict(cfg.model),
    }
    torch.save(payload, path)


def load_ensemble(path: Path, cfg: Config) -> SeedEnsemble:
    payload = torch.load(path, weights_only=False)
    digest = _config_digest(cfg)
    assert payload["config_digest"] == digest, (
        f"模型工件与当前配置不一致（{payload['config_digest']} vs {digest}），"
        "改 features/model 配置后必须重训"
    )
    rmask = payload.get("revin_mask")
    rmask = np.asarray(rmask, dtype=bool) if rmask is not None else None
    nets = []
    for sd in payload["state_dicts"]:
        n = TriHeadNet(len(payload["feature_cols"]), payload["lookback"],
                       d_hidden=payload["hp"]["d_hidden"], n_layers=payload["hp"]["n_layers"],
                       dropout=payload["hp"]["dropout"])
        n.load_state_dict(sd)
        nets.append(n)
    return SeedEnsemble(nets, payload["mean"], payload["std"],
                        tuple(payload["feature_cols"]), payload["lookback"],
                        revin_mask=rmask, vol_cal=payload.get("vol_cal", 1.0))
