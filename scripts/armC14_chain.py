"""判据链 · 臂 C 的 14 种子确证性实验（2026-10-09）。

预注册：plans/armC-14seed-prereg-20261009.md（**判据与种子池已冻结，一次性判定**）。

与 external_chain.py 的差异：
  1. 种子池 = 生产位 14 网（逐字相同）；
  2. **同时训练 F0_14 与 C_14**（对照必须同为 14 种子，否则特征与集成规模混淆）；
  3. **逐折增量落盘**（`parts/{arm}_f{k}.parquet`）——112 次训练耗时数小时，
     中断后可从断点续跑同一配置，不必从头再来。

用法：
  python scripts/armC14_chain.py            # 全量（可重复调用，已完成的折自动跳过）
  python scripts/armC14_chain.py --report   # 只做评估出裁决（不训练）
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

OUT_DIR = ROOT / "experiments" / "armC14_20261009"
PARTS = OUT_DIR / "parts"
# 冻结（预注册 §二）：与 scripts/train_production.py::SEED_SETS 逐字相同的 14 网池
SEEDS14 = (7, 21, 99, 5, 13, 123, 2024, 42, 314, 2027, 101, 202, 303, 404)
FOLD_EDGES = ["2019-01-01", "2021-01-01", "2023-01-01", "2025-01-01"]
LABEL_SPAN = 6
MAX_EPOCHS, PATIENCE, BATCH, LR, WD = 30, 5, 4096, 1e-3, 1e-5
VAL_FRAC = 0.85
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ic_series(df: pd.DataFrame) -> pd.Series:
    out = {}
    for d, g in df.groupby("date"):
        v = spearman_rank_ic(list(zip(g["pred"], g["y_ret"])))
        if v is not None:
            out[d] = v
    return pd.Series(out).sort_index()


def train_fold(cfg, bundle, d0, d1, seeds) -> pd.DataFrame:
    """一个折 × 全部种子：返回该折的逐行 OOS 预测（种子均值）。"""
    d1_pad = d1 + pd.Timedelta(days=1)
    Xtr, mtr, ytr, meta_tr, cols = fold_frame(bundle, bundle.universe,
                                              pd.Timestamp(cfg.data.start), d0, 0)
    cut = d0 - pd.Timedelta(days=LABEL_SPAN * 2)
    keep = np.array([d < cut for d, _ in meta_tr])
    Xtr, mtr, ytr = Xtr[keep], mtr[keep], ytr[keep]
    Xte, mte, yte, meta_te, _ = fold_frame(bundle, bundle.universe, d0, d1_pad, 0)

    edges = fit_bins(np.concatenate([Xtr, np.zeros((1, Xtr.shape[1]))], axis=0))
    Xtr_in = np.concatenate([Xtr, mtr], axis=1)
    Xte_in = np.concatenate([Xte, mte], axis=1)
    Xtr_bin, Xte_bin = to_bin_idx(Xtr, edges), to_bin_idx(Xte, edges)
    n_feat = Xtr_in.shape[1]

    val_cut = int(len(Xtr) * VAL_FRAC)
    xt = torch.from_numpy(Xtr_in[:val_cut]).float()
    bt = torch.from_numpy(Xtr_bin[:val_cut]).long()
    yt = torch.from_numpy(ytr[:val_cut]).float()
    xv = torch.from_numpy(Xtr_in[val_cut:]).float()
    bv = torch.from_numpy(Xtr_bin[val_cut:]).long()
    yv = torch.from_numpy(ytr[val_cut:]).float()

    preds = np.zeros(len(Xte_in))
    for si, seed in enumerate(seeds):
        torch.manual_seed(seed)
        net = CrossSectionNet(n_feat, n_embed_features=len(cols)).to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
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
        print(f"        seed {seed} ({si+1}/{len(seeds)}) val {best:.6f}", flush=True)
    if not np.isfinite(preds).all():
        raise ValueError(f"armC14: 预测含 NaN（{int(np.isnan(preds).sum())}/{len(preds)}）")
    return pd.DataFrame({"date": [d for d, _ in meta_te], "product": [c for _, c in meta_te],
                         "pred": preds, "y_ret": yte})


def run_arm(cfg, bundle, name, edges):
    """跑一个臂的全部折（逐折增量落盘，已完成的折跳过）。"""
    parts = []
    for k in range(len(edges) - 1):
        p = PARTS / f"{name}_f{k}.parquet"
        if p.exists():
            df = pd.read_parquet(p); df["date"] = pd.to_datetime(df["date"])
            print(f"      [{name}] fold{k} 已缓存 {len(df)} 条，跳过", flush=True)
        else:
            t0 = time.time()
            print(f"      [{name}] fold{k} 训练中（{len(SEEDS14)} 种子）...", flush=True)
            df = train_fold(cfg, bundle, edges[k], edges[k + 1], SEEDS14)
            df.to_parquet(p)
            print(f"      [{name}] fold{k} 完成 {len(df)} 条（{time.time()-t0:.0f}s）", flush=True)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="只评估出裁决（不训练）")
    args = ap.parse_args()
    cfg = load_config(ROOT / "config.yaml")
    PARTS.mkdir(parents=True, exist_ok=True)

    if not args.report:
        print(f"== 臂 C 14 种子确证性实验（device={DEVICE}）==", flush=True)
        print("预注册：plans/armC-14seed-prereg-20261009.md（判据与种子池已冻结）", flush=True)
        panels = sa.load_panels(cfg)
        base = build_dataset(cfg)
        ref = base.features[base.feature_cols[0]]
        ext_bundle, rhos = extend_bundle(base, load_survivor_panels(cfg, ref))
        added = [c for c in ext_bundle.feature_cols if c not in base.feature_cols]
        print(f"F0 {len(base.feature_cols)} 特征 / C {len(ext_bundle.feature_cols)} 特征"
              f"（新增 {len(added)}）", flush=True)
        dates = panels["close"].index
        edges = [pd.Timestamp(x) for x in FOLD_EDGES] + [dates[-1] + pd.Timedelta(days=1)]
        for name, bnd in (("F0", base), ("C", ext_bundle)):
            sb = run_arm(cfg, bnd, name, edges)
            sb.to_parquet(OUT_DIR / f"stream_{name}_14.parquet")
            print(f"  {name}: {len(sb)} 条 OOS 汇总完成", flush=True)

    # ---- 评估（一次性判定）----
    sf0 = pd.read_parquet(OUT_DIR / "stream_F0_14.parquet"); sf0["date"] = pd.to_datetime(sf0["date"])
    sc = pd.read_parquet(OUT_DIR / "stream_C_14.parquet"); sc["date"] = pd.to_datetime(sc["date"])
    # 仪器门：流必须 100% 非空
    for nm, df in (("F0_14", sf0), ("C_14", sc)):
        if df["pred"].notna().sum() != len(df) or df["y_ret"].notna().sum() != len(df):
            raise ValueError(f"仪器门失败：{nm} 流含 NaN（预测 {int(df['pred'].isna().sum())} / "
                             f"标签 {int(df['y_ret'].isna().sum())}）——读数作废")

    ic_f0, ic_c = ic_series(sf0), ic_series(sc)
    common = ic_c.index.intersection(ic_f0.index)
    pt = st.paired_t(ic_c.loc[common].to_numpy(), ic_f0.loc[common].to_numpy())
    s_f0, s_c = st.summarize_ic(ic_f0.to_numpy()), st.summarize_ic(ic_c.to_numpy())

    reg = market_regime_labels(cfg).dropna()
    cidx = ic_c.index.intersection(reg.index)
    strat_c = st.regime_stratified_ic(ic_c.loc[cidx].to_numpy(), reg.loc[cidx].to_numpy())
    min_ic = st.regime_min_ic(strat_c)

    # PBO：以逐日多空分位差为表现序列
    def ls(df, q=0.1):
        rows = {}
        for d, g in df.groupby("date"):
            if len(g) < 20:
                continue
            k = max(1, int(len(g) * q))
            s = g.sort_values("pred")
            rows[d] = float(s["y_ret"].tail(k).mean() - s["y_ret"].head(k).mean())
        return pd.Series(rows).sort_index()
    lsdf = pd.concat({"F0": ls(sf0), "C": ls(sc)}, axis=1).dropna()
    pbo = st.pbo_cscv(lsdf.to_numpy(), n_blocks=8) if len(lsdf) >= 16 else {"pbo": None}

    gates = {
        "主判据 t>=2 且 ΔIC>0": bool(pt["t"] is not None and pt["t"] >= 2.0 and (pt["mean_diff"] or 0) > 0),
        "辅1 ICIR(C)>ICIR(F0)": bool(s_c["icir"] is not None and s_f0["icir"] is not None
                                     and s_c["icir"] > s_f0["icir"]),
        "辅2 regime 最小分层IC>0": bool(min_ic is not None and min_ic > 0),
        "辅3 PBO<0.5": bool(pbo.get("pbo") is not None and pbo["pbo"] < 0.5),
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    verdict = {"prereg": "plans/armC-14seed-prereg-20261009.md", "device": str(DEVICE),
               "seeds": list(SEEDS14), "n_seeds": len(SEEDS14), "folds": len(FOLD_EDGES),
               "F0_14": s_f0, "C_14": s_c, "paired_t": pt,
               "regime_stratified_C": {k: v["mean"] for k, v in strat_c.items()},
               "regime_min_ic": min_ic, "pbo": pbo, "gates": gates,
               "adoption_ready": bool(gates["主判据 t>=2 且 ΔIC>0"]),
               "verdict_word": ""}
    (OUT_DIR / f"verdict_{stamp}.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    lines = [f"臂 C · 14 种子确证性实验读数 {stamp}（device={DEVICE}, 种子 {len(SEEDS14)}, 4 折）",
             "预注册 plans/armC-14seed-prereg-20261009.md（一次性判定）",
             f"F0_14: IC {s_f0['mean']:+.4f} t {s_f0['t']:+.2f} ICIR {s_f0['icir']:+.3f} n={s_f0['n']}",
             f"C_14 : IC {s_c['mean']:+.4f} t {s_c['t']:+.2f} ICIR {s_c['icir']:+.3f} n={s_c['n']}",
             f"配对 t（C_14 − F0_14）= {pt['t']:+.2f}，ΔIC = {pt['mean_diff']:+.4f}（n={pt['n']}）",
             f"regime 分层 IC: { {k: (round(v['mean'], 4) if v['mean'] is not None else None) for k, v in strat_c.items()} }",
             f"regime 最小分层 IC = {min_ic}",
             f"PBO = {pbo.get('pbo')}",
             "门：" + "；".join(f"{k}={v}" for k, v in gates.items()),
             f"adoption_ready = {verdict['adoption_ready']}",
             "判决词：＿＿＿（由用户填写）"]
    text = "\n".join(lines)
    (OUT_DIR / f"verdict_{stamp}.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f"证据包 → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
