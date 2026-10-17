# -*- coding: utf-8 -*-
"""RL2 生产工件训练 —— RankNet pairwise 秩项 + MSE 6 网集成（T3 工程单步骤 1）。

plans/tri-production-plan-20261014.md 步骤 1；依据 #126 六门 ALL PASS（信息层）。
配方（冻结，与 rankloss_chain.py #14 冻结臂逐字同源）：
  net2、损失 = MSE + 1.0×RankNet 相邻对（batch 内按日稳定排序取相邻、同日且 y 不等者
  构成对），早停 = val MSE；LR=1e-3 / WD=1e-5 / BATCH=4096 / patience=5 / max_epochs=30 /
  val=后 15% 行；种子 (7,21,99,5,13,123) = RL2 流同名种子集。
训练窗 = 全史（cfg.data.start → 面板末），与 a1_production 全史口径对齐。

产出 data/models/rl_production.pt：kind=rl_production，6 state_dicts + feature_cols +
bin_edges + 面板指纹。判定语义：pred 侧按种子分别出（逐种子秩均值，与判读一致）。
已知边界（工程单 §4）：报告流（RL2 parquet）与本工件为**不同管线**，允许合法实现级
差异；对账用逐日截面秩相关 ≥0.999（tri 步骤 2），不要求逐位。
"""
import dataclasses
import json
import os
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
from fterm.model.foldframe import fold_frame  # noqa: E402
from fterm.model.net2 import N_BINS, CrossSectionNet, fit_bins, to_bin_idx  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = (7, 21, 99, 5, 13, 123)          # RL2 流同名种子集（工程单步骤 1）
LR, WD, PATIENCE, MAX_EPOCHS, BATCH = 1e-3, 1e-5, 5, 30, 4096   # dlv2_chain 冻结
VAL_FRAC = 0.85
LAMBDA_RANK = 1.0                        # rankloss_chain #14 冻结


def enable_determinism() -> dict:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    return {"cudnn.deterministic": True, "device": str(DEVICE)}


def rl2_loss(out, yb, db):
    """MSE + λ·RankNet 相邻对（rankloss_chain.py:129-142 逐字语义）。"""
    mse = torch.nn.functional.mse_loss(out, yb)
    order = torch.argsort(db, stable=True)
    si, sj = order[:-1], order[1:]
    m = (db[si] == db[sj]) & (yb[si] != yb[sj])
    if bool(m.any()):
        i_, j_ = si[m], sj[m]
        sign = torch.sign(yb[i_] - yb[j_])
        rank = torch.nn.functional.softplus(-(out[i_] - out[j_]) * sign).mean()
    else:
        rank = out.sum() * 0.0
    return mse + LAMBDA_RANK * rank


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 种子 × 3 epochs × 2024+ 子集")
    args = ap.parse_args()

    seeds = (7, 21) if args.smoke else SEEDS
    max_epochs = 3 if args.smoke else MAX_EPOCHS
    out_name = "rl_production_smoke.pt" if args.smoke else "rl_production.pt"

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

    t0 = time.time()
    X, m, y, meta, cols = fold_frame(bundle, bundle.universe,
                                     pd.Timestamp(cfg.data.start), end, 0)
    val_cut = int(len(X) * VAL_FRAC)
    edges = fit_bins(np.concatenate([X, np.zeros((1, X.shape[1]))], axis=0), n_bins=N_BINS)
    Xin = np.concatenate([X, m], axis=1)
    Xb = to_bin_idx(X, edges)
    # 逐日 id（RankNet 相邻对需要；同 build_day_ids 口径）
    d2i = {}
    day_ids = np.empty(len(meta), dtype="int64")
    for i, (d, _c) in enumerate(meta):
        day_ids[i] = d2i.setdefault(d, len(d2i))
    print(f"帧就绪：{len(X)} 行 / {len(cols)} 特征 / {len(d2i)} 日"
          f"（train {val_cut} / val {len(X)-val_cut}，{time.time()-t0:.0f}s）", flush=True)

    xt = torch.from_numpy(Xin[:val_cut]).float().to(DEVICE)
    bt = torch.from_numpy(Xb[:val_cut]).long().to(DEVICE)
    yt = torch.from_numpy(y[:val_cut]).float().to(DEVICE)
    dt = torch.from_numpy(day_ids[:val_cut]).long().to(DEVICE)
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
            perm = torch.randperm(val_cut, device=DEVICE)
            for lo in range(0, val_cut, BATCH):
                idx = perm[lo:lo + BATCH]
                opt.zero_grad()
                out = net(xt[idx], bt[idx])
                loss = rl2_loss(out, yt[idx], dt[idx])
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
        "kind": "rl_production",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "state_dicts": state_dicts,
        "feature_cols": list(cols),
        "n_embed_features": len(cols),
        "bin_edges": edges,
        "fingerprints": fps,
        "n_samples": int(len(X)),
        "recipe": {"loss": "mse+ranknet_adjacent(lambda=1)", "batch": BATCH,
                   "LR": LR, "WD": WD, "PATIENCE": PATIENCE, "MAX_EPOCHS": MAX_EPOCHS,
                   "VAL_FRAC": VAL_FRAC, "seeds": list(seeds),
                   "source": "rankloss_chain #14 冻结配方（T3 工程单步骤 1）"},
        "hp": dataclasses.asdict(cfg.features),
    }
    out = ROOT / "data" / "models" / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"== 完成 → {out}（{time.time()-t0:.0f}s）==", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
