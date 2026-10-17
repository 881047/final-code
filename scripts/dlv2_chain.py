"""判据链 #8 · DL-v2 执行器 —— 判据冻结于 plans/dlv2-prereg-20261004.md。

截面原生深度模型（Rank-Gauss 白化 + 数值 embedding + ResNet-MLP，5 种子集成），
四折年界（同 #4），组合层 top2/keep4 周频两遍法 vs S1@2019，双种子集。
算力降级路径：GPU 不可用时训练采样每日 cap 200 行（预测仍全截面）。
判决词由用户填写。
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from fterm.config import load_config  # noqa: E402
from fterm.data import ashare as sa  # noqa: E402
from fterm.features.build import build_dataset  # noqa: E402
from fterm.model.foldframe import fold_frame  # noqa: E402
from fterm.model.net2 import (CrossSectionNet, fit_bins, rank_gauss,  # noqa: E402
                              to_bin_idx)
from fterm.model.validate import spearman_rank_ic  # noqa: E402
from ablation_chain import CHAIN1_JSON  # noqa: E402
from gate_chain import run_model_arm  # noqa: E402
from judgment_chain import paired_weekly_t  # noqa: E402

REPORTS = Path(__file__).resolve().parents[1] / "data" / "reports"
FOLD_EDGES = ["2019-01-01", "2021-01-01", "2023-01-01", "2025-01-01"]
SEED_SETS = [(7, 21, 99, 5, 13), (123, 2024, 42, 314, 2027)]
START_OOS = "2019-01-01"
LABEL_SPAN = 6

# ---- 冻结（预注册 §二） ----
CPU_CAP_PER_DAY = 200       # 降级路径：无 GPU 时训练采样密度
MAX_EPOCHS = 30
PATIENCE = 5
BATCH = 4096
LR = 1e-3
WD = 1e-5
C2A_T, C3_TOL, C4_RATIO = 2.0, 0.02, 1.25

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_GPU = DEVICE.type == "cuda"


# fold_frame / _erfinv_vec 已于 2026-10-08 B2 改造移到 fterm/model/foldframe.py
# （逐日逐股 .at 标量扫描 → 布尔矩阵整数切片，实测外推 415–423s → 0.1s，≈7000×）。
# 数值等价性由 tests/test_fold_frame_equivalence.py 逐位钉住；本文件从该模块导入。
def train_fold_seedset(cfg, bundle, universe, d0, d1, seeds, cap_per_day):
    """一个测试段 × 一套种子：训练 5 网集成，返回逐行集成预测。"""
    d1_pad = d1 + pd.Timedelta(days=1)
    Xtr_raw, mtr, ytr, meta_tr, cols = fold_frame(bundle, universe, pd.Timestamp(cfg.data.start), d0, cap_per_day)
    # 净化：去掉标签窗口伸入测试段的行（fwd_ret 到 d0 之前 label_span 日的样本）
    cut = d0 - pd.Timedelta(days=LABEL_SPAN * 2)
    keep = np.array([d < cut for d, _ in meta_tr])
    Xtr_raw, mtr, ytr, meta_tr = Xtr_raw[keep], mtr[keep], ytr[keep], [m for m, k in zip(meta_tr, keep) if k]
    # 测试段（预测全截面）
    Xte_raw, mte, yte, meta_te, _ = fold_frame(bundle, universe, d0, d1_pad, 0)

    edges = fit_bins(np.concatenate([Xtr_raw, np.zeros((1, Xtr_raw.shape[1]))], axis=0))
    Xtr = Xtr_raw                            # fold_frame 已做逐日截面白化
    Xtr_bin = to_bin_idx(Xtr, edges)
    Xte_bin = to_bin_idx(Xte_raw, edges)
    Xtr_in = np.concatenate([Xtr, mtr], axis=1)      # + 缺失指示（冻结）
    Xte_in = np.concatenate([Xte_raw, mte], axis=1)
    n_feat = Xtr_in.shape[1]

    preds = np.zeros(len(Xte_in), dtype="float64")
    val_cut = int(len(Xtr) * 0.85)
    for seed in seeds:
        torch.manual_seed(seed)
        net = CrossSectionNet(n_feat, n_embed_features=len(cols)).to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
        xt = torch.from_numpy(Xtr_in[:val_cut]).float()
        bt = torch.from_numpy(Xtr_bin[:val_cut]).long()
        yt = torch.from_numpy(ytr[:val_cut]).float()
        xv = torch.from_numpy(Xtr_in[val_cut:]).float()
        bv = torch.from_numpy(Xtr_bin[val_cut:]).long()
        yv = torch.from_numpy(ytr[val_cut:]).float()
        best, best_state, bad = float("inf"), None, 0
        ds = TensorDataset(xt, bt, yt)
        for ep in range(MAX_EPOCHS):
            net.train()
            dl = DataLoader(ds, batch_size=BATCH, shuffle=True)
            for xb, bb, yb in dl:
                opt.zero_grad()
                out = net(xb.to(DEVICE), bb.to(DEVICE))
                loss = torch.nn.functional.mse_loss(out, yb.to(DEVICE))
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                vl = torch.nn.functional.mse_loss(net(xv.to(DEVICE), bv.to(DEVICE)),
                                                  yv.to(DEVICE)).item()
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
            out = []
            for i in range(0, len(Xte_in), 8192):
                xb = torch.from_numpy(Xte_in[i:i + 8192]).float()
                bb = torch.from_numpy(Xte_bin[i:i + 8192]).long()
                out.append(net(xb.to(DEVICE), bb.to(DEVICE)).cpu().numpy())
            preds += np.concatenate(out) / len(seeds)
        print(f"      seed {seed} val {best:.6f}", flush=True)
    rows = [{"date": d, "product": c, "pred": float(p), "y_ret": y}
            for (d, c), p, y in zip(meta_te, preds, yte)]
    return pd.DataFrame(rows)


def main():
    cfg = load_config(ROOT / "config.yaml")
    REPORTS.mkdir(parents=True, exist_ok=True)
    chain1 = json.loads(CHAIN1_JSON.read_text(encoding="utf-8"))
    start_ts = pd.Timestamp(START_OOS)
    cap = 0 if USE_GPU else CPU_CAP_PER_DAY
    print(f"== 面板/特征（device={DEVICE}, cap/day={cap or '全截面'}）==", flush=True)
    panels = sa.load_panels(cfg)
    bundle = build_dataset(cfg)
    dates = panels["close"].index
    universe = bundle.universe
    edges = [pd.Timestamp(x) for x in FOLD_EDGES] + [dates[-1] + pd.Timedelta(days=1)]

    results, eqs = {}, {}
    for ss in SEED_SETS:
        tag = f"dlv2_ss{'_'.join(map(str, ss))}"
        cache = REPORTS / f"{tag}.parquet"
        if cache.exists():
            df = pd.read_parquet(cache)
            df["date"] = pd.to_datetime(df["date"])
            print(f"  {tag}: 缓存 {len(df)} 条", flush=True)
        else:
            parts = []
            t0 = time.time()
            for k in range(len(edges) - 1):
                print(f"    fold{k} 训练中（{len(SEED_SETS[0])} 种子）...", flush=True)
                parts.append(train_fold_seedset(cfg, bundle, universe,
                                                edges[k], edges[k + 1], ss, cap))
            df = pd.concat(parts, ignore_index=True)
            df.to_parquet(cache)
            print(f"  {tag}: {len(df)} 条 OOS 预测 ({time.time()-t0:.0f}s)", flush=True)
        ics = [spearman_rank_ic(list(zip(g["pred"], g["y_ret"])))
               for _, g in df.groupby("date")]
        ics = [x for x in ics if x is not None]
        ic_mean = float(np.mean(ics))
        panel = df.pivot_table(index="date", columns="product", values="pred")
        eq, m = run_model_arm(-1e9, 2, 3, cfg, panel, bundle, panels, dates, start_ts)
        eqs[tag] = eq
        results[tag] = {"rank_ic": ic_mean, "n_days": len(ics),
                        "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                    for k, v in m.items()},
                        "C1": bool(ic_mean > 0.02)}
        print(f"  {tag}: IC {ic_mean:+.4f} ret {m['total_return']:+.2%}", flush=True)

    from ablation_chain import run_rule_arm
    eq_s1, m_s1 = run_rule_arm("S1", cfg, bundle, panels, dates, start_ts)
    fridays = [d for d in dates if d >= start_ts
               and d.weekday() == cfg.signal.rebalance_weekday and d in eq_s1.index]
    for tag, r in results.items():
        p = paired_weekly_t(eqs[tag], eq_s1, fridays)
        r["C2a"] = {**p, "pass": bool(p["t"] >= C2A_T and p["mean"] > 0)}
        r["C2b"] = bool(r["metrics"]["total_return"] > m_s1["total_return"])
        r["C3"] = bool(r["metrics"]["max_drawdown"] >= m_s1["max_drawdown"] - C3_TOL)
        r["C4"] = bool(r["metrics"]["total_costs"] <= m_s1["total_costs"] * C4_RATIO)
        r["adoption_ready"] = bool(r["C1"] and r["C2a"]["pass"] and r["C2b"] and r["C3"] and r["C4"])
    dual = bool(all(r["adoption_ready"] for r in results.values()) and len(results) == 2)

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    verdict = {
        "prereg": "plans/dlv2-prereg-20261004.md",
        "device": str(DEVICE), "cap_per_day": cap or "full",
        "oos_window": {"start": START_OOS, "end": chain1["oos_window"]["end"]},
        "frozen": {"H": "h=5 主标签", "SEED_SETS": [list(s) for s in SEED_SETS],
                   "MAX_EPOCHS": MAX_EPOCHS, "BATCH": BATCH, "LR": LR},
        "baseline": {"name": "S1@2019", **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                           for k, v in m_s1.items()}},
        "results": results, "dual_seed_agree": dual,
        "verdict_word": "",   # 待用户填写
    }
    (REPORTS / f"dlv2_chain_{stamp}.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [f"判据链 #8 读数 {stamp}（判据冻结于 plans/dlv2-prereg-20261004.md；device={DEVICE}）",
             f"基线 S1@2019: {m_s1['total_return']:+.2%} / MDD {m_s1['max_drawdown']:+.2%} / 成本 {m_s1['total_costs']:.0f}"]
    for tag, r in results.items():
        lines.append(f"{tag}: IC {r['rank_ic']:+.4f}(n={r['n_days']}) "
                     f"ret {r['metrics']['total_return']:+.2%} MDD {r['metrics']['max_drawdown']:+.2%} "
                     f"成本 {r['metrics']['total_costs']:.0f} "
                     f"C1={r['C1']} C2a t={r['C2a']['t']:+.2f} C2b={r['C2b']} "
                     f"C3={r['C3']} C4={r['C4']} → adoption_ready={r['adoption_ready']}")
    lines.append(f"dual_seed_agree = {dual}")
    lines.append("判决词：＿＿＿（由用户填写）")
    text = "\n".join(lines)
    (REPORTS / f"dlv2_chain_{stamp}.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f"报告 → data/reports/dlv2_chain_{stamp}.json/.txt", flush=True)


if __name__ == "__main__":
    main()
