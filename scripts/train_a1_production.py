# -*- coding: utf-8 -*-
"""A1 生产工件训练 —— ListMLE 秩目标 6 网集成（判据链 #124 采纳对象的部署件）。

配方（冻结，与 objective_ab_20261013/ab_chain.py 的 A1 臂逐字同源）：
  net2 + ListMLE、按日整批（batch_days=1, shuffle=True, seed=seed*1000+ep）、
  LR=1e-3 / WD=1e-5 / patience=5 / max_epochs=30 / val=后 15% 行。
训练窗 = 全史扩展（cfg.data.start → 面板末），与 dlv2_production 的全史口径对齐；
6 种子 (7,21,99,5,13,123) = 预注册 §0 冻结的采纳种子集。

产出 data/models/a1_production.pt：kind=a1_production，6 state_dicts +
feature_cols + bin_edges + 面板指纹（谱系核实用，同 train_production 口径）。

判定语义：pred（推理）侧按种子分别出（predict_day_perseed）⇒ 采纳对象
D = 0.5·rank_pct(Ā) + 0.5·rank_pct(dlv2) 的 Ā 用**逐种子秩均值**（与判读一致），
不用 raw-pred 均值后再取秩。
"""
import dataclasses
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from fterm.config import load_config  # noqa: E402
from fterm.data import ashare as sa  # noqa: E402
from fterm.features.build import build_dataset  # noqa: E402
from fterm.model.daybatch import day_batches  # noqa: E402
from fterm.model.dayids import build_day_ids  # noqa: E402
from fterm.model.foldframe import fold_frame  # noqa: E402
from fterm.model.losses import listmle_loss  # noqa: E402
from fterm.model.net2 import N_BINS, CrossSectionNet, fit_bins, to_bin_idx  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = (7, 21, 99, 5, 13, 123)          # 预注册 §0 冻结
LR, WD, PATIENCE, MAX_EPOCHS = 1e-3, 1e-5, 5, 30
VAL_FRAC = 0.85


def enable_determinism() -> dict:
    os_environ = __import__("os").environ
    os_environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    return {"cudnn.deterministic": True, "device": str(DEVICE)}


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 种子 × 3 epochs × 2024+ 子集")
    args = ap.parse_args()

    seeds = (7, 21) if args.smoke else SEEDS
    max_epochs = 3 if args.smoke else MAX_EPOCHS
    out_name = "a1_production_smoke.pt" if args.smoke else "a1_production.pt"

    enable_determinism()
    cfg = load_config(ROOT / "config.yaml")
    panels = sa.load_panels(cfg)
    bundle = build_dataset(cfg)
    fps = {f: sa.df_fingerprint(panels[f]) for f in sorted(panels)}

    end = pd.Timestamp(panels["close"].index[-1]) + pd.Timedelta(days=1)
    if args.smoke:
        cut = pd.Timestamp("2024-01-01")
        feats = {k: df.loc[cut:] for k, df in bundle.features.items()}
        labels = {k: df.loc[cut:] for k, df in bundle.labels.items()}
        uni = bundle.universe.loc[cut:]
        bundle = dataclasses.replace(bundle, features=feats, labels=labels, universe=uni)
        end = pd.Timestamp(panels["close"].index[-1]) + pd.Timedelta(days=1)

    # ---- 全史训练帧（fold_frame 同口径：日内秩白化 + 缺失指示）----
    t0 = time.time()
    X, m, y, meta, cols = fold_frame(bundle, bundle.universe, pd.Timestamp(cfg.data.start), end, 0)
    val_cut = int(len(X) * VAL_FRAC)
    edges = fit_bins(np.concatenate([X, np.zeros((1, X.shape[1]))], axis=0), n_bins=N_BINS)
    Xin = np.concatenate([X, m], axis=1)
    Xb = to_bin_idx(X, edges)
    day_tr = build_day_ids(d for d, _ in meta)
    print(f"帧就绪：{len(X)} 行 / {len(cols)} 特征（train {val_cut} / val {len(X)-val_cut}，"
          f"{time.time()-t0:.0f}s）", flush=True)

    xt = torch.from_numpy(Xin[:val_cut]).float().to(DEVICE)
    bt = torch.from_numpy(Xb[:val_cut]).long().to(DEVICE)
    yt = torch.from_numpy(y[:val_cut]).float().to(DEVICE)
    dt = torch.from_numpy(day_tr[:val_cut]).long().to(DEVICE)
    train_days = day_tr[:val_cut]
    xv = torch.from_numpy(Xin[val_cut:]).float().to(DEVICE)
    bv = torch.from_numpy(Xb[val_cut:]).long().to(DEVICE)
    yv = torch.from_numpy(y[val_cut:]).float().to(DEVICE)

    state_dicts = []
    t00 = time.time()
    for i, seed in enumerate(seeds):
        torch.manual_seed(seed)
        net = CrossSectionNet(Xin.shape[1], n_embed_features=len(cols)).to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
        best, best_state, bad, ep_run = float("inf"), None, 0, 0
        for ep in range(max_epochs):
            net.train()
            for idx in day_batches(train_days, batch_days=1, shuffle=True,
                                   seed=seed * 1000 + ep):
                ti = torch.from_numpy(idx)
                opt.zero_grad()
                out = net(xt[ti].to(DEVICE), bt[ti].to(DEVICE))
                loss = listmle_loss(out, yt[ti].to(DEVICE), dt[ti].to(DEVICE))
                if not torch.isfinite(loss):
                    raise SystemExit(f"损失非有限（seed={seed} ep={ep}）⇒ 判废")
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                vl = torch.nn.functional.mse_loss(net(xv, bv), yv).item()
            ep_run = ep + 1
            if vl < best - 1e-6:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        state_dicts.append({"seed": seed, "state": net.state_dict(),
                            "val_loss": best, "epochs": ep_run})
        print(f"  [{i+1}/{len(seeds)}] seed {seed} val {best:.6f} ep {ep_run} "
              f"累计 {time.time()-t00:.0f}s", flush=True)

    payload = {
        "kind": "a1_production",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "state_dicts": state_dicts,
        "feature_cols": list(cols),
        "n_embed_features": len(cols),
        "bin_edges": edges,
        "fingerprints": fps,
        "n_samples": int(len(X)),
        "recipe": {"loss": "listmle", "day_batch": True, "LR": LR, "WD": WD,
                   "PATIENCE": PATIENCE, "MAX_EPOCHS": MAX_EPOCHS, "VAL_FRAC": VAL_FRAC,
                   "seeds": list(seeds), "source": "objective_ab_20261013.A1 冻结配方"},
        "hp": dataclasses.asdict(cfg.features),
    }
    out = ROOT / "data" / "models" / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"== 完成 → {out}（{time.time()-t0:.0f}s）==", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
