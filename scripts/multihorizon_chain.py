"""判据链 · 臂 B 执行器：多 horizon 融合（2026-10-09）。

预注册：plans/model-upgrade-prereg-20261009.md §四 臂 B。
依据：判据链 #7 已证 IC 随 horizon 单调升（h1 0.032 → h21 0.078）。

设计：
  - F0：单头 h=5（net2），**复用臂 A/臂 C 同一份 F0 流缓存**（配置逐字相同）；
  - B ：`MultiHorizonNet`（共享主干 + h∈{1,3,5,10,20} 五头）+ `MultitaskLoss`
        （不确定性加权，Kendall 2018）；
  - 样本集以**最长 horizon（h=20）**的 eligibility 为准（其 NaN 集是其余 horizon 的超集，
    保证多头标签全部非 NaN）；
  - 评估：h=5 头 IC（与 F0 直接可比）、融合分 IC/ICIR、逐日配对 t。

用法：python scripts/multihorizon_chain.py --folds 4
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
from fterm.features.multihorizon import build_multihorizon_labels, fused_score  # noqa: E402
from fterm.features.regime import market_regime_labels  # noqa: E402
from fterm.model import stats as st  # noqa: E402
from fterm.model.dayids import build_day_ids  # noqa: E402
from fterm.model.foldframe import fold_frame  # noqa: E402
from fterm.model.heads import MultiHorizonNet  # noqa: E402
from fterm.model.losses import MultitaskLoss  # noqa: E402
from fterm.model.net2 import fit_bins, to_bin_idx  # noqa: E402
from fterm.model.validate import spearman_rank_ic  # noqa: E402

OUT_DIR = ROOT / "experiments" / f"multihorizon_{datetime.now().strftime('%Y%m%d')}"
FOLD_EDGES = ["2019-01-01", "2021-01-01", "2023-01-01", "2025-01-01"]
HORIZONS = (1, 3, 5, 10, 20)
SEEDS = (7, 21)
LABEL_SPAN = 6
MAX_EPOCHS, PATIENCE, BATCH, LR, WD = 30, 5, 4096, 1e-3, 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_multi_y(bundle, meta, horizons) -> dict[str, np.ndarray]:
    """按 meta 的 (date, code) 取各 horizon 的 y（向量化索引）。

    ⚠️ 必须先把标签面板 **reindex 到 ref 的 index/columns**：标签面板由 `open×adj`
    派生（5471 列），与特征参照面板（5467 列）列集不同；不重对齐会索引错位、
    取到含 NaN 的格子（2026-10-09 臂 B 首跑即因此产出全 NaN 预测与伪 IC）。
    """
    ref = bundle.features[bundle.feature_cols[0]]
    master, codes = ref.index, ref.columns
    ri = master.get_indexer(pd.DatetimeIndex([d for d, _ in meta]))
    ci = codes.get_indexer(pd.Index([c for _, c in meta]))
    if (ri < 0).any() or (ci < 0).any():
        raise ValueError(f"build_multi_y: 索引未命中（ri<0 {int((ri<0).sum())}, ci<0 {int((ci<0).sum())}）")
    out = {}
    for h in horizons:
        panel = bundle.labels[f"fwd_ret_h{h}"].reindex(index=master, columns=codes)
        arr = panel.to_numpy(dtype="float64")
        out[f"h{h}"] = arr[ri, ci]
    return out


def ic_series(df: pd.DataFrame, col: str = "pred") -> pd.Series:
    out = {}
    for d, g in df.groupby("date"):
        v = spearman_rank_ic(list(zip(g[col], g["y_ret"])))
        if v is not None:
            out[d] = v
    return pd.Series(out).sort_index()


def train_fold(cfg, bundle, universe, d0, d1, seeds) -> pd.DataFrame:
    d1_pad = d1 + pd.Timedelta(days=1)
    # 以最长 horizon 的 eligibility 为准（保证多头标签全非 NaN）
    Xtr, mtr, _, meta_tr, cols = fold_frame(bundle, universe, pd.Timestamp(cfg.data.start),
                                            d0, 0, y_key="fwd_ret_h20")
    cut = d0 - pd.Timedelta(days=LABEL_SPAN * 2)
    keep = np.array([d < cut for d, _ in meta_tr])
    Xtr, mtr = Xtr[keep], mtr[keep]
    meta_tr = [m for m, k in zip(meta_tr, keep) if k]
    Xte, mte, _, meta_te, _ = fold_frame(bundle, universe, d0, d1_pad, 0, y_key="fwd_ret_h20")

    Ytr = build_multi_y(bundle, meta_tr, HORIZONS)
    Yte = build_multi_y(bundle, meta_te, HORIZONS)

    def _common(X, m, meta, Y):
        """多头共同样本集：任一 horizon 为 NaN 即剔除。

        仅按最长 horizon（h=20）过滤**不够**——个股在 t+2 停牌而 t+1/t+21 正常交易时，
        h=1 会是 NaN（2026-10-09 臂 B 首跑实测 1018/1,091,266）。多任务损失必须无 NaN。
        """
        k = np.ones(len(meta), dtype=bool)
        for v in Y.values():
            k &= np.isfinite(v)
        return X[k], m[k], [mm for mm, kk in zip(meta, k) if kk], {n: v[k] for n, v in Y.items()}

    Xtr, mtr, meta_tr, Ytr = _common(Xtr, mtr, meta_tr, Ytr)
    Xte, mte, meta_te, Yte = _common(Xte, mte, meta_te, Yte)
    if len(meta_tr) == 0 or len(meta_te) == 0:
        raise ValueError("multihorizon: 多头共同样本集为空")

    # 标签按训练段各 horizon 的 std 标准化（PIT：只用训练段统计），使多任务损失可等权。
    # ⚠️ 2026-10-09：原用 Kendall(2018) 可学习不确定性加权，实测**单调发散**
    # （train_loss 从 ~0 一路降到 −12，log σ→−∞），因 L 可趋 0 时该项无下界——
    # 属该加权的已知退化模式。故改用「标准化标签 + 固定等权」，并在台账登记此偏离。
    for h in HORIZONS:
        s = float(np.std(Ytr[f"h{h}"]))
        s = s if s > 1e-12 else 1.0
        Ytr[f"h{h}"] = Ytr[f"h{h}"] / s
        Yte[f"h{h}"] = Yte[f"h{h}"] / s

    edges = fit_bins(np.concatenate([Xtr, np.zeros((1, Xtr.shape[1]))], axis=0))
    Xtr_in = np.concatenate([Xtr, mtr], axis=1)
    Xte_in = np.concatenate([Xte, mte], axis=1)
    Xtr_bin, Xte_bin = to_bin_idx(Xtr, edges), to_bin_idx(Xte, edges)
    n_feat = Xtr_in.shape[1]

    val_cut = int(len(Xtr) * 0.85)
    # 逐日 id（供 listnet/ranknet 口径扩展；MSE 主用时不参与计算）
    # 用项目既有助手（pd.factorize 的 C 级实现；pandas 3.0 不接受裸 list，故必须走它）
    day_ids_tr = build_day_ids(d for d, _ in meta_tr)

    preds_h5 = np.zeros(len(Xte_in))
    preds_all = {f"h{h}": np.zeros(len(Xte_in)) for h in HORIZONS}
    for seed in seeds:
        torch.manual_seed(seed)
        net = MultiHorizonNet(n_feat, HORIZONS, n_embed_features=len(cols)).to(DEVICE)
        mtl = MultitaskLoss([f"h{h}" for h in HORIZONS], base_loss="mse",
                            fixed_weights={f"h{h}": 1.0 for h in HORIZONS}).to(DEVICE)
        opt = torch.optim.AdamW(list(net.parameters()) + list(mtl.parameters()), lr=LR, weight_decay=WD)
        xt = torch.from_numpy(Xtr_in[:val_cut]).float()
        bt = torch.from_numpy(Xtr_bin[:val_cut]).long()
        dt = torch.from_numpy(day_ids_tr[:val_cut])
        yt = {f"h{h}": torch.from_numpy(Ytr[f"h{h}"][:val_cut]).float() for h in HORIZONS}
        xv = torch.from_numpy(Xtr_in[val_cut:]).float()
        bv = torch.from_numpy(Xtr_bin[val_cut:]).long()
        dv = torch.from_numpy(day_ids_tr[val_cut:])
        yv = {f"h{h}": torch.from_numpy(Ytr[f"h{h}"][val_cut:]).float() for h in HORIZONS}
        best, best_state, bad = float("inf"), None, 0
        n = len(xt)
        ep_loss = float("nan")
        for ep in range(MAX_EPOCHS):
            net.train(); mtl.train()
            perm = torch.randperm(n)
            for i in range(0, n, BATCH):
                idx = perm[i:i + BATCH]
                opt.zero_grad()
                outs = net(xt[idx].to(DEVICE), bt[idx].to(DEVICE))
                parts = {k: (v, yt[k][idx].to(DEVICE)) for k, v in outs.items()}
                loss, _ = mtl(parts, dt[idx].to(DEVICE))
                loss.backward(); opt.step()
                ep_loss = float(loss)
            net.eval(); mtl.eval()
            with torch.no_grad():
                ov = net(xv.to(DEVICE), bv.to(DEVICE))
                vl, _ = mtl({k: (v, yv[k].to(DEVICE)) for k, v in ov.items()}, dv.to(DEVICE))
            vl = float(vl)
            print(f"      seed {seed} ep{ep} train_loss={ep_loss:.6f} val_loss={vl:.6f}", flush=True)
            if vl < best - 1e-6:
                best, bad = vl, 0
                best_state = ({k: v.detach().clone() for k, v in net.state_dict().items()},
                              {k: v.detach().clone() for k, v in mtl.state_dict().items()})
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        if best_state:
            net.load_state_dict(best_state[0]); mtl.load_state_dict(best_state[1])
        else:
            print(f"      ⚠️ seed {seed} 无有效 best_state（val 从未改善/NaN）", flush=True)
        net.eval()
        with torch.no_grad():
            acc = {f"h{h}": [] for h in HORIZONS}
            for i in range(0, len(Xte_in), 8192):
                xb = torch.from_numpy(Xte_in[i:i + 8192]).float()
                bb = torch.from_numpy(Xte_bin[i:i + 8192]).long()
                o = net(xb.to(DEVICE), bb.to(DEVICE))
                for k, v in o.items():
                    acc[k].append(v.cpu().numpy())
        for h in HORIZONS:
            preds_all[f"h{h}"] += np.concatenate(acc[f"h{h}"]) / len(seeds)
        preds_h5 += np.concatenate(acc["h5"]) / len(seeds)
    # 响亮失败守卫：预测含 NaN ⇒ 结果不可用（2026-10-09 曾因此产出伪 IC）
    for nm, arr in [("pred", preds_h5)] + [(f"h{h}", preds_all[f"h{h}"]) for h in HORIZONS]:
        if not np.isfinite(arr).all():
            raise ValueError(
                f"multihorizon: {nm} 预测含 NaN（{int(np.isnan(arr).sum())}/{len(arr)}）——"
                "网络发散或标签含 NaN，读数不可用")
    return pd.DataFrame({
        "date": [d for d, _ in meta_te], "product": [c for _, c in meta_te],
        "pred": preds_h5, "y_ret": Yte["h5"],
        **{f"pred_h{h}": preds_all[f"h{h}"] for h in HORIZONS},
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config(ROOT / "config.yaml")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"== 面板/特征（device={DEVICE}）==", flush=True)
    panels = sa.load_panels(cfg)
    bundle = build_dataset(cfg)
    mh = build_multihorizon_labels(cfg, panels, HORIZONS)
    import dataclasses
    bundle = dataclasses.replace(bundle, labels={**bundle.labels, **mh})
    print(f"多 horizon 标签 {list(mh)}；特征 {len(bundle.feature_cols)}", flush=True)

    dates = panels["close"].index
    edges = [pd.Timestamp(x) for x in FOLD_EDGES][:args.folds] + [dates[-1] + pd.Timedelta(days=1)]

    cache_b = OUT_DIR / f"stream_B_f{args.folds}.parquet"
    if cache_b.exists():
        sb = pd.read_parquet(cache_b); sb["date"] = pd.to_datetime(sb["date"])
        print(f"  B: 缓存 {len(sb)}", flush=True)
    else:
        t0, parts = time.time(), []
        for k in range(len(edges) - 1):
            print(f"  [B] fold{k} ...", flush=True)
            parts.append(train_fold(cfg, bundle, bundle.universe, edges[k], edges[k + 1], SEEDS))
        sb = pd.concat(parts, ignore_index=True); sb.to_parquet(cache_b)
        print(f"  B: {len(sb)} 条（{time.time()-t0:.0f}s）", flush=True)

    # F0 对照必须与折数匹配（否则是「单折 vs 四折」的错配比较）
    f0_candidates = [
        ROOT / "experiments" / "external_20261009" / f"stream_F0_f{args.folds}.parquet",
        ROOT / "experiments" / "regime_20261009" / f"stream_F0_f{args.folds}.parquet",
        ROOT / "experiments" / "regime_20261009" / "stream_F0.parquet",   # 早期 f1 冒烟命名
    ]
    f0_path = next((p for p in f0_candidates if p.exists()), None)
    if f0_path is None:
        raise FileNotFoundError(f"缺与 folds={args.folds} 匹配的 F0 缓存；候选 {f0_candidates}")
    print(f"  F0 对照 ← {f0_path.name}", flush=True)
    sf0 = pd.read_parquet(f0_path); sf0["date"] = pd.to_datetime(sf0["date"])

    ic_f0 = ic_series(sf0, "pred")
    ic_b5 = ic_series(sb, "pred")
    # 融合分：逐日对五头做截面 z-score 后等权
    fused = sb.groupby("date", group_keys=False).apply(
        lambda g: fused_score({f"h{h}": g[f"pred_h{h}"] for h in HORIZONS}))
    sb_f = sb.assign(pred=fused.to_numpy())
    ic_bf = ic_series(sb_f, "pred")

    reg = market_regime_labels(cfg).dropna()
    common = ic_b5.index.intersection(ic_f0.index)
    pt5 = st.paired_t(ic_b5.loc[common].to_numpy(), ic_f0.loc[common].to_numpy())
    common_f = ic_bf.index.intersection(ic_f0.index)
    ptf = st.paired_t(ic_bf.loc[common_f].to_numpy(), ic_f0.loc[common_f].to_numpy())
    s_f0 = st.summarize_ic(ic_f0.to_numpy())
    s_b5 = st.summarize_ic(ic_b5.to_numpy())
    s_bf = st.summarize_ic(ic_bf.to_numpy())
    per_head = {f"h{h}": st.summarize_ic(ic_series(sb, f"pred_h{h}").to_numpy()) for h in HORIZONS}

    gates = {
        "C_h5_ic_not_worse": bool(s_b5["mean"] is not None and s_f0["mean"] is not None
                                  and s_b5["mean"] >= s_f0["mean"] - 1e-4),
        "C_ge2_other_heads_ic>0": bool(sum(1 for h in HORIZONS if h != 5
                                          and per_head[f"h{h}"]["mean"] is not None
                                          and per_head[f"h{h}"]["mean"] > 0) >= 2),
        "C_fused_icir>F0": bool(s_bf["icir"] is not None and s_f0["icir"] is not None
                                and s_bf["icir"] > s_f0["icir"]),
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    verdict = {"prereg": "plans/model-upgrade-prereg-20261009.md#臂B", "device": str(DEVICE),
               "horizons": list(HORIZONS), "F0": s_f0, "B_h5": s_b5, "B_fused": s_bf,
               "per_head": per_head, "paired_t_h5_vs_F0": pt5, "paired_t_fused_vs_F0": ptf,
               "gates": gates, "adoption_ready": bool(all(gates.values())), "verdict_word": ""}
    (OUT_DIR / f"verdict_{stamp}.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    lines = [f"判据链 · 臂 B（多 horizon 融合）读数 {stamp}（device={DEVICE}, folds={args.folds}）",
             f"F0  单头h5: IC {s_f0['mean']:+.4f} t {s_f0['t']:+.2f} ICIR {s_f0['icir']:+.3f}",
             f"B   h5头 : IC {s_b5['mean']:+.4f} t {s_b5['t']:+.2f} ICIR {s_b5['icir']:+.3f}",
             f"B  融合分: IC {s_bf['mean']:+.4f} t {s_bf['t']:+.2f} ICIR {s_bf['icir']:+.3f}",
             f"配对 t（B_h5−F0）= {pt5['t']:+.2f}；配对 t（B_融合−F0）= {ptf['t']:+.2f}",
             "各头 IC: " + ", ".join(f"h{h} {per_head[f'h{h}']['mean']:+.4f}" for h in HORIZONS),
             "门：" + ", ".join(f"{k}={v}" for k, v in gates.items()),
             f"adoption_ready = {verdict['adoption_ready']}", "判决词：＿＿＿（由用户填写）"]
    text = "\n".join(lines)
    (OUT_DIR / f"verdict_{stamp}.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
