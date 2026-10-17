"""判据链 · 臂 A 执行器：市场状态非秩通道（2026-10-09，模型升级）。

预注册：plans/model-upgrade-prereg-20261009.md §四 臂 A。
审计依据：docs/audit-model-20261009.md P0-4（秩化通道消灭市场状态）。

设计：同一批折、同一批种子、同一份白化特征，**唯一变量** = 是否给模型喂
市场状态旁路通道 Z（FiLM 条件化）。两臂：
  - F0（对照）：net2.CrossSectionNet（无条件，忽略 Z）
  - A （处理）：net3.RegimeCrossSectionNet（FiLM 条件化于 Z）

主判据（信息层）：A 相对 F0 的逐日 IC 配对 t ≥ 2 且 regime 分层 IC 最小值 > 0。
判决词由用户填写。
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
from fterm.features.regime import REGIME_COLS, build_regime_features, market_regime_labels  # noqa: E402
from fterm.model import stats as st  # noqa: E402
from fterm.model.foldframe import fold_frame  # noqa: E402
from fterm.model.net2 import CrossSectionNet, fit_bins, to_bin_idx  # noqa: E402
from fterm.model.net3 import RegimeCrossSectionNet, ZScaler, count_params  # noqa: E402

REPORTS = ROOT / "data" / "reports"
OUT_DIR = ROOT / "experiments" / f"regime_{datetime.now().strftime('%Y%m%d')}"
FOLD_EDGES = ["2019-01-01", "2021-01-01", "2023-01-01", "2025-01-01"]
SEEDS = (7, 21)
LABEL_SPAN = 6
MAX_EPOCHS, PATIENCE, BATCH, LR, WD = 30, 5, 4096, 1e-3, 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _rank_ic_series(df: pd.DataFrame) -> pd.Series:
    """逐日截面 Spearman IC 序列。"""
    from fterm.model.validate import spearman_rank_ic

    out = {}
    for d, g in df.groupby("date"):
        v = spearman_rank_ic(list(zip(g["pred"], g["y_ret"])))
        if v is not None:
            out[d] = v
    return pd.Series(out).sort_index()


def _train_fold(cfg, bundle, universe, z_cols, d0, d1, seeds, conditioned: bool) -> pd.DataFrame:
    """一个测试段：训练 seeds 网集成，返回逐行 OOS 预测。"""
    d1_pad = d1 + pd.Timedelta(days=1)
    Xtr, mtr, ytr, meta_tr, cols, Ztr = fold_frame(
        bundle, universe, pd.Timestamp(cfg.data.start), d0, 0, z_cols=z_cols)
    cut = d0 - pd.Timedelta(days=LABEL_SPAN * 2)
    keep = np.array([d < cut for d, _ in meta_tr])
    Xtr, mtr, ytr, Ztr = Xtr[keep], mtr[keep], ytr[keep], Ztr[keep]
    meta_tr = [m for m, k in zip(meta_tr, keep) if k]
    Xte, mte, yte, meta_te, _, Zte = fold_frame(
        bundle, universe, d0, d1_pad, 0, z_cols=z_cols)

    edges = fit_bins(np.concatenate([Xtr, np.zeros((1, Xtr.shape[1]))], axis=0))
    Xtr_bin, Xte_bin = to_bin_idx(Xtr, edges), to_bin_idx(Xte, edges)
    Xtr_in = np.concatenate([Xtr, mtr], axis=1)
    Xte_in = np.concatenate([Xte, mte], axis=1)
    n_feat = Xtr_in.shape[1]

    zs = ZScaler.fit(Ztr)
    Ztr_s, Zte_s = zs.transform(Ztr), zs.transform(Zte)

    val_cut = int(len(Xtr) * 0.85)
    preds = np.zeros(len(Xte_in))
    for seed in seeds:
        torch.manual_seed(seed)
        if conditioned:
            net = RegimeCrossSectionNet(n_feat, len(z_cols), n_embed_features=len(cols)).to(DEVICE)
        else:
            net = CrossSectionNet(n_feat, n_embed_features=len(cols)).to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
        xt = torch.from_numpy(Xtr_in[:val_cut]).float()
        bt = torch.from_numpy(Xtr_bin[:val_cut]).long()
        zt = torch.from_numpy(Ztr_s[:val_cut]).float()
        yt = torch.from_numpy(ytr[:val_cut]).float()
        xv = torch.from_numpy(Xtr_in[val_cut:]).float()
        bv = torch.from_numpy(Xtr_bin[val_cut:]).long()
        zv = torch.from_numpy(Ztr_s[val_cut:]).float()
        yv = torch.from_numpy(ytr[val_cut:]).float()

        def fwd(x, b, z):
            return net(x, b, z) if conditioned else net(x, b)

        best, best_state, bad = float("inf"), None, 0
        ds = TensorDataset(xt, bt, zt, yt)
        for ep in range(MAX_EPOCHS):
            net.train()
            for xb, bb, zb, yb in DataLoader(ds, batch_size=BATCH, shuffle=True):
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(
                    fwd(xb.to(DEVICE), bb.to(DEVICE), zb.to(DEVICE)), yb.to(DEVICE))
                loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                vl = torch.nn.functional.mse_loss(
                    fwd(xv.to(DEVICE), bv.to(DEVICE), zv.to(DEVICE)), yv.to(DEVICE)).item()
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
                zb = torch.from_numpy(Zte_s[i:i + 8192]).float()
                acc.append(fwd(xb.to(DEVICE), bb.to(DEVICE), zb.to(DEVICE)).cpu().numpy())
            preds += np.concatenate(acc) / len(seeds)
    return pd.DataFrame([{"date": d, "product": c, "pred": float(p), "y_ret": y}
                         for (d, c), p, y in zip(meta_te, preds, yte)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=4, help="折数（调试用 1）")
    args = ap.parse_args()
    cfg = load_config(ROOT / "config.yaml")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"== 面板/特征（device={DEVICE}）==", flush=True)
    panels = sa.load_panels(cfg)
    bundle = build_dataset(cfg)
    z_panels = build_regime_features(cfg, panels)
    # 把旁路特征挂进 bundle.features（**不进 feature_cols**，避免被秩化）
    bundle.features.update(z_panels)
    universe = bundle.universe
    dates = panels["close"].index
    edges = [pd.Timestamp(x) for x in FOLD_EDGES][:args.folds] + [dates[-1] + pd.Timedelta(days=1)]
    z_cols = list(REGIME_COLS)

    arms = {"F0": False, "A": True}
    streams: dict[str, pd.DataFrame] = {}
    for name, cond in arms.items():
        # 缓存键必须含折数（否则 1 折冒烟缓存会被 4 折正式跑误用）
        cache = OUT_DIR / f"stream_{name}_f{args.folds}.parquet"
        if cache.exists():
            streams[name] = pd.read_parquet(cache)
            streams[name]["date"] = pd.to_datetime(streams[name]["date"])
            print(f"  {name}: 缓存 {len(streams[name])} 条", flush=True)
            continue
        t0, parts = time.time(), []
        for k in range(len(edges) - 1):
            print(f"  [{name}] fold{k} 训练中（{len(SEEDS)} 种子，conditioned={cond}）...", flush=True)
            parts.append(_train_fold(cfg, bundle, universe, z_cols,
                                     edges[k], edges[k + 1], SEEDS, cond))
        df = pd.concat(parts, ignore_index=True)
        df.to_parquet(cache)
        streams[name] = df
        print(f"  {name}: {len(df)} 条 OOS（{time.time()-t0:.0f}s）", flush=True)

    ic = {name: _rank_ic_series(df) for name, df in streams.items()}
    reg = market_regime_labels(cfg)

    def _strat(ic_s: pd.Series) -> dict:
        common = ic_s.index.intersection(reg.dropna().index)
        return st.regime_stratified_ic(ic_s.loc[common].to_numpy(),
                                       reg.loc[common].to_numpy())

    sum_f0, sum_a = st.summarize_ic(ic["F0"].to_numpy()), st.summarize_ic(ic["A"].to_numpy())
    common = ic["A"].index.intersection(ic["F0"].index)
    pt = st.paired_t(ic["A"].loc[common].to_numpy(), ic["F0"].loc[common].to_numpy())
    strat_a = _strat(ic["A"])
    min_ic = st.regime_min_ic(strat_a)
    roll = st.rolling_ic_stability(ic["A"].to_numpy(), window=60)
    roll_f0 = st.rolling_ic_stability(ic["F0"].to_numpy(), window=60)

    pass_t = bool(pt["t"] is not None and pt["t"] >= 2.0 and (pt["mean_diff"] or 0) > 0)
    pass_regime = bool(min_ic is not None and min_ic > 0)
    pass_roll = bool(roll["nw_t_of_daily_ic"] is not None
                     and roll_f0["nw_t_of_daily_ic"] is not None
                     and abs(roll["nw_t_of_daily_ic"]) >= abs(roll_f0["nw_t_of_daily_ic"]))

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    verdict = {
        "prereg": "plans/model-upgrade-prereg-20261009.md#臂A",
        "device": str(DEVICE), "seeds": list(SEEDS), "folds": args.folds,
        "z_cols": z_cols,
        "F0": {"summary": sum_f0, "regime_stratified": _strat(ic["F0"]),
               "rolling": {"nw_t": roll_f0["nw_t_of_daily_ic"],
                           "frac_neg": roll_f0["frac_negative_windows"]}},
        "A": {"summary": sum_a, "regime_stratified": strat_a,
              "rolling": {"nw_t": roll["nw_t_of_daily_ic"],
                          "frac_neg": roll["frac_negative_windows"]}},
        "paired_t": pt,
        "regime_min_ic": min_ic,
        "gates": {"C_ic_paired_t>=2": pass_t, "C_regime_min_ic>0": pass_regime,
                  "C_rolling_nw_t_not_worse": pass_roll},
        "adoption_ready": bool(pass_t and pass_regime and pass_roll),
        "verdict_word": "",   # 待用户填写
    }
    (OUT_DIR / f"verdict_{stamp}.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    lines = [f"判据链 · 臂 A（市场状态非秩通道）读数 {stamp}（device={DEVICE}, folds={args.folds}）",
             f"F0 无条件: IC {sum_f0['mean']:+.4f} t {sum_f0['t']:+.2f} ICIR {sum_f0['icir']:+.3f} n={sum_f0['n']}",
             f"A  条件化: IC {sum_a['mean']:+.4f} t {sum_a['t']:+.2f} ICIR {sum_a['icir']:+.3f} n={sum_a['n']}",
             f"配对 t（A−F0）= {pt['t']:+.2f}，ΔIC = {pt['mean_diff']:+.4f}（n={pt['n']}）",
             f"regime 分层 IC: { {k: round(v['mean'], 4) if v['mean'] is not None else None for k, v in strat_a.items()} }",
             f"regime 最小分层 IC = {min_ic}",
             f"滚动 NW-t: F0 {roll_f0['nw_t_of_daily_ic']:.2f} → A {roll['nw_t_of_daily_ic']:.2f}",
             f"门：IC配对t≥2 = {pass_t}；regime 最小IC>0 = {pass_regime}；滚动不劣 = {pass_roll}",
             f"adoption_ready = {verdict['adoption_ready']}",
             "判决词：＿＿＿（由用户填写）"]
    text = "\n".join(lines)
    (OUT_DIR / f"verdict_{stamp}.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f"证据包 → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
