"""判据链 · 臂 C 执行器：外部日频因子入模的增量 IC（2026-10-09）。

预注册：plans/model-upgrade-prereg-20261009.md §四 臂 C。
对照：F0（现役 25 特征）vs C（+7 个幸存外部因子，共线 >0.7 者剔除）。
唯一变量 = 特征集；折/种子/超参逐字相同。

主判据：C 相对 F0 的逐日 IC 配对 t ≥ 2 且加入后 max|ρ| 在阈值内。
用法：python scripts/external_chain.py [--folds 4]
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from fterm.config import load_config  # noqa: E402
from fterm.data import ashare as sa  # noqa: E402
from fterm.features.build import build_dataset  # noqa: E402
from fterm.features.external_features import extend_bundle, load_survivor_panels  # noqa: E402
from fterm.features.regime import market_regime_labels  # noqa: E402
from fterm.model import stats as st  # noqa: E402
from fterm.model.foldframe import fold_frame  # noqa: E402
from fterm.model.net2 import CrossSectionNet, fit_bins, to_bin_idx  # noqa: E402
from fterm.model.validate import spearman_rank_ic  # noqa: E402

OUT_DIR = ROOT / "experiments" / f"external_{datetime.now().strftime('%Y%m%d')}"
FOLD_EDGES = ["2019-01-01", "2021-01-01", "2023-01-01", "2025-01-01"]
SEEDS = (7, 21)
LABEL_SPAN = 6
MAX_EPOCHS, PATIENCE, BATCH, LR, WD = 30, 5, 4096, 1e-3, 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ic_series(df: pd.DataFrame) -> pd.Series:
    out = {}
    for d, g in df.groupby("date"):
        v = spearman_rank_ic(list(zip(g["pred"], g["y_ret"])))
        if v is not None:
            out[d] = v
    return pd.Series(out).sort_index()


def train_fold(cfg, bundle, universe, d0, d1, seeds) -> pd.DataFrame:
    d1_pad = d1 + pd.Timedelta(days=1)
    Xtr, mtr, ytr, meta_tr, cols = fold_frame(
        bundle, universe, pd.Timestamp(cfg.data.start), d0, 0)
    cut = d0 - pd.Timedelta(days=LABEL_SPAN * 2)
    keep = np.array([d < cut for d, _ in meta_tr])
    Xtr, mtr, ytr = Xtr[keep], mtr[keep], ytr[keep]
    Xte, mte, yte, meta_te, _ = fold_frame(bundle, universe, d0, d1_pad, 0)

    edges = fit_bins(np.concatenate([Xtr, np.zeros((1, Xtr.shape[1]))], axis=0))
    Xtr_in = np.concatenate([Xtr, mtr], axis=1)
    Xte_in = np.concatenate([Xte, mte], axis=1)
    Xtr_bin, Xte_bin = to_bin_idx(Xtr, edges), to_bin_idx(Xte, edges)
    n_feat = Xtr_in.shape[1]

    val_cut = int(len(Xtr) * 0.85)
    preds = np.zeros(len(Xte_in))
    for seed in seeds:
        torch.manual_seed(seed)
        net = CrossSectionNet(n_feat, n_embed_features=len(cols)).to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
        xt = torch.from_numpy(Xtr_in[:val_cut]).float(); bt = torch.from_numpy(Xtr_bin[:val_cut]).long()
        yt = torch.from_numpy(ytr[:val_cut]).float()
        xv = torch.from_numpy(Xtr_in[val_cut:]).float(); bv = torch.from_numpy(Xtr_bin[val_cut:]).long()
        yv = torch.from_numpy(ytr[val_cut:]).float()
        best, best_state, bad = float("inf"), None, 0
        ds = TensorDataset(xt, bt, yt)
        for ep in range(MAX_EPOCHS):
            net.train()
            for xb, bb, yb in DataLoader(ds, batch_size=BATCH, shuffle=True):
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(net(xb.to(DEVICE), bb.to(DEVICE)), yb.to(DEVICE))
                loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                vl = torch.nn.functional.mse_loss(net(xv.to(DEVICE), bv.to(DEVICE)), yv.to(DEVICE)).item()
            if vl < best - 1e-6:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        if best_state:
            net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            acc = []
            for i in range(0, len(Xte_in), 8192):
                xb = torch.from_numpy(Xte_in[i:i + 8192]).float()
                bb = torch.from_numpy(Xte_bin[i:i + 8192]).long()
                acc.append(net(xb.to(DEVICE), bb.to(DEVICE)).cpu().numpy())
            preds += np.concatenate(acc) / len(seeds)
    return pd.DataFrame([{"date": d, "product": c, "pred": float(p), "y_ret": y}
                         for (d, c), p, y in zip(meta_te, preds, yte)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config(ROOT / "config.yaml")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"== 面板/特征（device={DEVICE}）==", flush=True)
    panels = sa.load_panels(cfg)
    base = build_dataset(cfg)
    ref = base.features[base.feature_cols[0]]
    ext = load_survivor_panels(cfg, ref)
    ext_bundle, rhos = extend_bundle(base, ext)
    added = [c for c in ext_bundle.feature_cols if c not in base.feature_cols]
    print(f"现役 {len(base.feature_cols)} 特征 → 扩展 {len(ext_bundle.feature_cols)}"
          f"（新增 {len(added)}：{added}）", flush=True)
    print(f"max|ρ| vs 现役: { {k: round(v, 3) for k, v in rhos.items()} }", flush=True)

    dates = panels["close"].index
    edges = [pd.Timestamp(x) for x in FOLD_EDGES][:args.folds] + [dates[-1] + pd.Timedelta(days=1)]
    arms = {"F0": base, "C": ext_bundle}
    streams = {}
    for name, bnd in arms.items():
        cache = OUT_DIR / f"stream_{name}_f{args.folds}.parquet"
        if cache.exists():
            streams[name] = pd.read_parquet(cache)
            streams[name]["date"] = pd.to_datetime(streams[name]["date"])
            print(f"  {name}: 缓存 {len(streams[name])}", flush=True)
            continue
        t0, parts = time.time(), []
        for k in range(len(edges) - 1):
            print(f"  [{name}] fold{k} ...", flush=True)
            parts.append(train_fold(cfg, bnd, bnd.universe, edges[k], edges[k + 1], SEEDS))
        streams[name] = pd.concat(parts, ignore_index=True)
        streams[name].to_parquet(cache)
        print(f"  {name}: {len(streams[name])} 条（{time.time()-t0:.0f}s）", flush=True)

    ic = {n: ic_series(df) for n, df in streams.items()}
    reg = market_regime_labels(cfg).dropna()
    common = ic["C"].index.intersection(ic["F0"].index)
    pt = st.paired_t(ic["C"].loc[common].to_numpy(), ic["F0"].loc[common].to_numpy())
    s_c, s_f0 = st.summarize_ic(ic["C"].to_numpy()), st.summarize_ic(ic["F0"].to_numpy())
    cidx = ic["C"].index.intersection(reg.index)
    strat_c = st.regime_stratified_ic(ic["C"].loc[cidx].to_numpy(), reg.loc[cidx].to_numpy())

    pass_t = bool(pt["t"] is not None and pt["t"] >= 2.0 and (pt["mean_diff"] or 0) > 0)
    pass_rho = bool(max(rhos.values()) <= 0.7) if rhos else True

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    verdict = {"prereg": "plans/model-upgrade-prereg-20261009.md#臂C", "device": str(DEVICE),
               "added_cols": added, "rhos": rhos,
               "F0": s_f0, "C": s_c, "paired_t": pt,
               "regime_min_ic": st.regime_min_ic(strat_c),
               "gates": {"C_ic_paired_t>=2": pass_t, "C_maxrho<=0.7": pass_rho},
               "adoption_ready": bool(pass_t and pass_rho), "verdict_word": ""}
    (OUT_DIR / f"verdict_{stamp}.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    lines = [f"判据链 · 臂 C（外部因子入模）读数 {stamp}",
             f"新增特征 {added}",
             f"F0: IC {s_f0['mean']:+.4f} t {s_f0['t']:+.2f} ICIR {s_f0['icir']:+.3f} n={s_f0['n']}",
             f"C : IC {s_c['mean']:+.4f} t {s_c['t']:+.2f} ICIR {s_c['icir']:+.3f} n={s_c['n']}",
             f"配对 t（C−F0）= {pt['t']:+.2f}，ΔIC = {pt['mean_diff']:+.4f}",
             f"门：IC配对t≥2 = {pass_t}；maxρ≤0.7 = {pass_rho}",
             f"adoption_ready = {verdict['adoption_ready']}", "判决词：＿＿＿（由用户填写）"]
    text = "\n".join(lines)
    (OUT_DIR / f"verdict_{stamp}.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
