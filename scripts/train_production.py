"""生产训练器 —— DL-v2 全史扩展窗（到最新面板日），14 网（5+5+2+2 种子集构成）。

判据链 #13 收官后用户拍板：候选直接入主生产信号位（paper 双账户盘面）。
判据链 #14 换版（用户 2026-10-07 选 b 填判决）：训练损失升级为
MSE + RankNet pairwise 秩项（λ=1，batch 内同日相邻对，与 #14 臂逐字同款），
早停/选模仍为纯 val MSE（与 #14 口径一致）。全史样本（无 holdout 泄漏风险——
标签 t+6 自然截尾）、逐日截面白化、数值分桶。
产出 data/models/dlv2_production.pt（20 state_dicts + feature_cols + 桶边界 + 指纹）。
"""
import dataclasses
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fterm.config import load_config  # noqa: E402
from fterm.data import ashare as sa  # noqa: E402
from fterm.features.build import build_dataset  # noqa: E402
from fterm.model.dayids import build_day_ids  # noqa: E402
from fterm.model.net2 import CrossSectionNet, fit_bins, to_bin_idx  # noqa: E402
from dlv2_chain import fold_frame  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED_SETS = [(7, 21, 99, 5, 13), (123, 2024, 42, 314, 2027),
             (101, 202), (303, 404)]          # 14 网池（5+5+2+2，与 #13/#14 的
                                              # 4 套 OOS 流构成一致；docstring
                                              # 曾误写"20 网"——20 是流口径非网数）
MAX_EPOCHS, PATIENCE, BATCH, LR, WD = 30, 5, 4096, 1e-3, 1e-5
VAL_FRAC = 0.10
LAMBDA_RANK = 1.0    # 冻结（#14 预注册 §一）：rank 项与 mse 项等权


def train_all(cfg, bundle, universe, seeds_all):
    feats = bundle.features
    cols = list(bundle.feature_cols)
    latest = feats[cols[0]].index.max()
    print(f"训练窗: 面板起点 → {latest.date()}（device={DEVICE}）", flush=True)

    # 全史样本（无 cap；标签 NaN 自然截尾最近 6 日）
    Xg, miss, y, meta, _ = fold_frame(bundle, universe,
                                      feats[cols[0]].index.min(), latest + pd.Timedelta(days=1), 0)
    edges = fit_bins(Xg)
    Xb = to_bin_idx(Xg, edges)
    Xin = np.concatenate([Xg, miss.astype("float32")], axis=1)
    n_feat = Xin.shape[1]
    n_val = max(1, int(len(Xin) * VAL_FRAC))
    cut = len(Xin) - n_val
    print(f"样本 {len(Xin)}（特征 {n_feat}，val {n_val}）", flush=True)

    xt = torch.from_numpy(Xin[:cut]).float().to(DEVICE)
    bt = torch.from_numpy(Xb[:cut]).long().to(DEVICE)
    yt = torch.from_numpy(y[:cut]).float().to(DEVICE)
    # 训练行 → 日组 id（meta 按日期有序 ⇒ 同日块连续；秩项按日内相邻对取）
    # 2026-10-08 B9：逐行 dict.setdefault → pd.factorize（C 级），输出逐位一致、快 ~160×。
    # 等价性由 tests/test_dayids_equivalence.py 钉住；实现见 fterm/model/dayids.py。
    day_ids = build_day_ids(d for d, _ in meta[:cut])
    dt = torch.from_numpy(day_ids).long().to(DEVICE)
    xv = torch.from_numpy(Xin[cut:]).float().to(DEVICE)
    bv = torch.from_numpy(Xb[cut:]).long().to(DEVICE)
    yv = torch.from_numpy(y[cut:]).float().to(DEVICE)

    state_dicts = []
    total = sum(len(s) for s in seeds_all)
    done = 0
    t00 = time.time()
    for set_i, seeds in enumerate(SEED_SETS):
        for seed in seeds:
            torch.manual_seed(seed)
            net = CrossSectionNet(n_feat, n_embed_features=len(cols)).to(DEVICE)
            opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
            best, best_state, bad, ep_run = float("inf"), None, 0, 0
            for ep in range(MAX_EPOCHS):
                net.train()
                perm = torch.randperm(len(xt), device=DEVICE)
                for i in range(0, len(xt), BATCH):
                    idx = perm[i:i + BATCH]
                    opt.zero_grad()
                    out = net(xt[idx], bt[idx])
                    mse = torch.nn.functional.mse_loss(out, yt[idx])
                    # RankNet 相邻对（与 #14 臂逐字同款）：batch 内按日稳定排序，
                    # 相邻元素中同日且 y 不等者构成对；损失对日内偏移不变
                    db = dt[idx]
                    order = torch.argsort(db, stable=True)
                    si, sj = order[:-1], order[1:]
                    m = (db[si] == db[sj]) & (yt[idx][si] != yt[idx][sj])
                    if bool(m.any()):
                        i_, j_ = si[m], sj[m]
                        sign = torch.sign(yt[idx][i_] - yt[idx][j_])
                        rank = torch.nn.functional.softplus(
                            -(out[i_] - out[j_]) * sign).mean()
                    else:
                        rank = out.sum() * 0.0
                    loss = mse + LAMBDA_RANK * rank
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
            state_dicts.append({"seed": seed, "seed_set": set_i, "state": net.state_dict(),
                                "val_loss": best, "epochs": ep_run})
            done += 1
            print(f"  [{done}/{total}] seed {seed}（set{set_i}）val {best:.6f} "
                  f"ep {ep_run}  累计 {time.time()-t00:.0f}s", flush=True)
    return state_dicts, edges, cols, len(Xin)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 网 × 2024+ 子集 × 3 epochs 冒烟")
    args = ap.parse_args()
    cfg = load_config(ROOT / "config.yaml")
    panels = sa.load_panels(cfg)
    bundle = build_dataset(cfg)
    universe = bundle.universe
    fps = {f: sa.df_fingerprint(panels[f]) for f in sorted(panels)}

    global SEED_SETS, MAX_EPOCHS
    out_name = "dlv2_production.pt"
    if args.smoke:
        SEED_SETS = [(7, 21)]
        MAX_EPOCHS = 3
        out_name = "dlv2_production_smoke.pt"
        # 2024+ 子集：全部面板/标签/股票池截到 cut（frozen dataclass 用 replace 重建）
        cut = pd.Timestamp("2024-01-01")
        feats = {k: df.loc[cut:] for k, df in bundle.features.items()}
        labels = {k: df.loc[cut:] for k, df in bundle.labels.items()}
        uni = bundle.universe.loc[cut:]
        bundle = dataclasses.replace(bundle, features=feats, labels=labels, universe=uni)
        # 2026-10-09 修复：局部 universe 此前未随 smoke 切片更新（仍指向全史面板），
        # 与切片后的 bundle.features 索引不一致 ⇒ fold_frame 索引校验响亮失败。
        # 该 bug 使 `--smoke` 路径自加入以来一直不可用（不影响生产全量路径）。
        universe = uni
        print(f"SMOKE: 2024+ 子集，{len(SEED_SETS)} 组 × {MAX_EPOCHS} epochs", flush=True)

    state_dicts, edges, cols, n_samples = train_all(cfg, bundle, universe, SEED_SETS)
    payload = {
        "kind": "dlv2_production",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "state_dicts": state_dicts,
        "feature_cols": list(cols),
        "n_embed_features": len(cols),
        "bin_edges": edges,
        "fingerprints": fps,
        "n_samples": n_samples,
        "hp": dataclasses.asdict(cfg.model),
        "digest_src": {"features": dataclasses.asdict(cfg.features),
                       "model": dataclasses.asdict(cfg.model)},
    }
    out = ROOT / "data" / "models" / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"工件 → {out}（{len(state_dicts)} 网）", flush=True)


if __name__ == "__main__":
    main()
