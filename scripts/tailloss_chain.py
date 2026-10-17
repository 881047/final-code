"""判据链 #18 · 模型层两把手术刀（预注册 plans/tailloss-prereg-20261008.md）。

单变量臂：
  F0  = #14 B 臂逐字复现（选模 val MSE、成对权重 1）——自证锚点
  M1  = 只改选模/早停指标 → val RankIC（逐日截面秩相关均值）
  M2  = 只改成对权重 → 尾部聚焦 w = ½(|yᵢ|+|yⱼ|) 批内归一
  NEG = 日内置换标签的阴性对照（tripwire |IC| ≤ 0.025）

其余（折边界/净化/分桶/网络/优化器/λ=1/相邻对构造/5 网集成）与 #14 逐字相同。
组合层裁判用 #15 忠实臂（真 T3 + ema_step=daily + blocked_cash=0），不用 gate_chain
那个"无 EMA 且档位错接"的旧评测口。

用法：  .venv312\\Scripts\\python.exe -u scripts/tailloss_chain.py --arm M1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fterm.config import load_config                       # noqa: E402
from fterm.data import ashare as sa                        # noqa: E402
from fterm.features.build import build_dataset             # noqa: E402
from fterm.model.dayids import build_day_ids               # noqa: E402
from fterm.model.net2 import CrossSectionNet, fit_bins, to_bin_idx  # noqa: E402
from fterm.model.validate import spearman_rank_ic          # noqa: E402

import sync_index                                           # noqa: E402
import tierfix_replay as TF                                # noqa: E402
from dlv2_chain import (BATCH, CPU_CAP_PER_DAY, LABEL_SPAN,  # noqa: E402
                        LR, USE_GPU, WD, fold_frame)
from dlv2_chain import FOLD_EDGES, MAX_EPOCHS, PATIENCE, START_OOS  # noqa: E402
from ddhalt_chain import weekly_stats                      # noqa: E402

REPORTS = ROOT / "data" / "reports"
EXDIR = ROOT / "experiments" / "tailloss_20261008"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = (7, 21, 99, 5, 13)              # 与缓存流同集 ⇒ 三方逐流可比
LAMBDA_RANK = 1.0                        # 冻结（#14 预注册）
X0_TOL = 2.79e-3                         # GPU atomicAdd 包络（docs/02 2026-10-08 登记）
TRIPWIRE_IC = 0.025
PAIR_T = 2.0
MDD_FLOOR = -0.30
CAP_TRAIN = 0 if USE_GPU else CPU_CAP_PER_DAY
REF_STREAM = ROOT / "data" / "reports" / "rankloss_ss7_21_99_5_13.parquet"
BASELINE_NAME = "F_RL2_daily"            # #15 存档的忠实臂基线（组合层裁判对象）


def _mean_daily_rankic(pred: np.ndarray, y: np.ndarray, day_ids: np.ndarray) -> float:
    """val 块上"逐日截面秩相关的均值"——与链上 IC 口径一致，只用于选模。"""
    out = []
    i0 = 0
    for n_d in np.bincount(day_ids):
        if n_d >= 8:
            ic = spearman_rank_ic(list(zip(pred[i0:i0 + n_d], y[i0:i0 + n_d])))
            if ic is not None:
                out.append(ic)
        i0 += int(n_d)
    return float(np.mean(out)) if out else float("-inf")


def train_fold(cfg, bundle, universe, d0, d1, seeds, cap, *, select="mse",
                tail_w=False, shuffle_labels=False):
    """#14 `train_fold_rankloss` 的单变量变体。

    select: "mse"（现行，取最小 val MSE）| "rankic"（取最大逐日 val RankIC）
    tail_w: True ⇒ 成对项按 ½(|yᵢ|+|yⱼ|) 加权并在批内归一到均值 1（梯度集中到尾部）
    """
    Xtr_raw, mtr, ytr, meta_tr, cols = fold_frame(
        bundle, universe, pd.Timestamp(cfg.data.start), d0, cap)
    cut = d0 - pd.Timedelta(days=LABEL_SPAN * 2)
    keep = np.array([d < cut for d, _ in meta_tr])
    Xtr_raw, mtr, ytr = Xtr_raw[keep], mtr[keep], ytr[keep]
    meta_tr = [m for m, k in zip(meta_tr, keep) if k]
    Xte_raw, mte, yte, meta_te, _ = fold_frame(bundle, universe, d0, d1, 0)

    edges = fit_bins(np.concatenate([Xtr_raw, np.zeros((1, Xtr_raw.shape[1]))], axis=0))
    Xtr_bin = to_bin_idx(Xtr_raw, edges)
    Xte_bin = to_bin_idx(Xte_raw, edges)
    Xtr_in = np.concatenate([Xtr_raw, mtr], axis=1)
    Xte_in = np.concatenate([Xte_raw, mte], axis=1)
    day_ids = build_day_ids(d for d, _ in meta_tr)

    y_fit = ytr
    if shuffle_labels:
        rng = np.random.RandomState(20261007)
        y_fit = ytr.copy()
        i0 = 0
        for n_d in np.bincount(day_ids):
            if n_d > 1:
                y_fit[i0:i0 + n_d] = y_fit[i0:i0 + n_d][rng.permutation(n_d)]
            i0 += int(n_d)

    val_cut = int(len(Xtr_raw) * 0.85)
    preds = np.zeros(len(Xte_in), dtype="float64")
    ep_log = []
    for seed in seeds:
        torch.manual_seed(seed)
        net = CrossSectionNet(Xtr_in.shape[1], n_embed_features=len(cols)).to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
        xt = torch.from_numpy(Xtr_in[:val_cut]).float().to(DEVICE)
        bt = torch.from_numpy(Xtr_bin[:val_cut]).long().to(DEVICE)
        yt = torch.from_numpy(y_fit[:val_cut]).float().to(DEVICE)
        dt = torch.from_numpy(day_ids[:val_cut]).long().to(DEVICE)
        xv = torch.from_numpy(Xtr_in[val_cut:]).float().to(DEVICE)
        bv = torch.from_numpy(Xtr_bin[val_cut:]).long().to(DEVICE)
        yv = torch.from_numpy(y_fit[val_cut:]).float().to(DEVICE)
        dv = torch.from_numpy(day_ids[val_cut:]).long().to(DEVICE)
        best = float("inf") if select == "mse" else float("-inf")
        best_state, bad, ep_run = None, 0, 0
        ds = TensorDataset(xt, bt, yt, dt)
        for ep in range(MAX_EPOCHS):
            net.train()
            for xb, bb, yb, db in DataLoader(ds, batch_size=BATCH, shuffle=True):
                opt.zero_grad()
                out = net(xb, bb)
                mse = torch.nn.functional.mse_loss(out, yb)
                order = torch.argsort(db, stable=True)
                si, sj = order[:-1], order[1:]
                m = (db[si] == db[sj]) & (yb[si] != yb[sj])
                if bool(m.any()):
                    i_, j_ = si[m], sj[m]
                    sign = torch.sign(yb[i_] - yb[j_])
                    per = torch.nn.functional.softplus(
                        -(out[i_] - out[j_]) * sign)
                    if tail_w:
                        w = 0.5 * (yb[i_].abs() + yb[j_].abs())
                        w = w / (w.mean() + 1e-12)          # 批内归一到均值 1（不改 λ 的语义）
                        rank = (per * w.detach()).mean()
                    else:
                        rank = per.mean()
                else:
                    rank = out.sum() * 0.0
                (mse + LAMBDA_RANK * rank).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                o = net(xv, bv)
                if select == "mse":
                    score = torch.nn.functional.mse_loss(o, yv).item()
                    better = score < best - 1e-9
                else:
                    score = _mean_daily_rankic(o.cpu().numpy(), yv.cpu().numpy(),
                                               dv.cpu().numpy())
                    better = score > best + 1e-9
            ep_run = ep + 1
            ep_log.append({"seed": seed, "ep": ep_run, "select": select,
                           "score": round(float(score), 6)})
            if better:
                best, bad = score, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        if best_state:
            net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            outs = [net(torch.from_numpy(Xte_in[i:i + 8192]).float().to(DEVICE),
                        torch.from_numpy(Xte_bin[i:i + 8192]).long().to(DEVICE)).cpu().numpy()
                    for i in range(0, len(Xte_in), 8192)]
            preds += np.concatenate(outs) / len(seeds)
        print(f"      seed {seed} select={select} best={best:.6f} ep={ep_run}", flush=True)
    df = pd.DataFrame([{"date": d, "product": c, "pred": float(p), "y_ret": y}
                       for (d, c), p, y in zip(meta_te, preds, yte)])
    df["fold"] = int(os.environ.get("TAIL_FOLD", -1))
    return df, ep_log


def daily_ic(df: pd.DataFrame) -> pd.Series:
    return pd.Series({d: spearman_rank_ic(list(zip(g["pred"], g["y_ret"])))
                      for d, g in df.groupby("date")}).dropna()


def tail_bucket_weekly(df: pd.DataFrame, q: float = 0.005) -> dict:
    """事实 2 的机制读数：最优 q 分位篮子的周均收益（周五出信号、下周同批收益）。"""
    out = {}
    for frac, name in ((q, "top0.5pct"), (0.01, "top1pct")):
        rows = []
        for d, g in df.groupby("date"):
            k = max(3, int(round(len(g) * frac)))
            rows.append({"date": d, "bucket_ret": float(
                g.nlargest(k, "pred")["y_ret"].mean())})
        s = pd.DataFrame(rows).set_index("date")["bucket_ret"]
        wk = s.resample("W-FRI").mean().dropna()
        out[name] = {"n_weeks": int(len(wk)), "mean_week": round(float(wk.mean()), 6),
                     "std_week": round(float(wk.std()), 6),
                     "pct_pos_weeks": round(float((wk > 0).mean()), 4)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="M1", choices=["F0", "M1", "M2", "NEG", "NEG_M1"])
    args = ap.parse_args()
    EXDIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "config.yaml")
    panels = sa.load_panels(cfg)
    assert panels["close"].index.max() <= pd.Timestamp("2026-09-30"), "越过前瞻窗隔离线"
    bundle = build_dataset(cfg)
    universe = bundle.universe
    # 终端折边界（2026-10-08 勘误）：原实现漏加 `dates[-1]+1` ⇒ 只训 3 折、流止于
    # 2024-12-31，缺 2025-01 起的 418 个交易日（rankloss_chain/dlv2_chain 均有此边界，
    # 预注册 §二"F0 四条 OOS 流"即指 4 折）⇒ 信号层配对的 fillna(0) 把缺失日记成
    # 负差（M1 t=−9.1/M2 t=−12.7 被污染）、组合层在无信号区走 21 个月——
    # M1/M2 首批流与读数全部作废重训（docs/03 §43）。
    edges = [pd.Timestamp(x) for x in FOLD_EDGES] + \
        [panels["close"].index[-1] + pd.Timedelta(days=1)]

    kw = {"F0": dict(select="mse", tail_w=False),
          "M1": dict(select="rankic", tail_w=False),
          "M2": dict(select="mse", tail_w=True),
          # 阴性对照 ×2（预注册 §二"每臂做一次"）：各自与对应主臂同 kw + 日内置换标签
          "NEG_M1": dict(select="rankic", tail_w=False, shuffle_labels=True),
          "NEG": dict(select="mse", tail_w=True, shuffle_labels=True)}[args.arm]
    parts, ep_logs = [], []
    t00 = time.time()
    for i in range(len(edges) - 1):
        os.environ["TAIL_FOLD"] = str(i)
        print(f"  fold{i} {edges[i].date()}→{edges[i+1].date()} {kw}", flush=True)
        df, elog = train_fold(cfg, bundle, universe, edges[i], edges[i + 1],
                              SEEDS, CAP_TRAIN, **kw)
        df["fold"] = i
        parts.append(df)
        ep_logs += elog
        print(f"    fold{i} 完成 {len(df)} 行（累计 {time.time()-t00:.0f}s）", flush=True)
    stream = pd.concat(parts, ignore_index=True)
    stream["date"] = pd.to_datetime(stream["date"])
    assert stream["date"].max() <= pd.Timestamp("2026-09-30")
    # 响亮失败守卫（2026-10-08 勘误补）：流必须覆盖到数据尾端（标签尾差 ~6 交易日
    # ⇒ 右端点距面板末 <15 自然日）。缺终端折时这里会拦住，而不是让下游的
    # fillna(0) 配对与"无信号区组合层"静默吞掉 21 个月缺口。
    tail_gap = (panels["close"].index.max() - stream["date"].max()).days
    assert tail_gap < 15, (
        f"流右端点 {stream['date'].max().date()} 距面板末 "
        f"{panels['close'].index.max().date()} 有 {tail_gap} 天 ⇒ 折边界不完整"
        f"（缺终端折/缺折），拒绝出数")
    out_path = EXDIR / f"tailloss_{args.arm}_ss7_21_99_5_13.parquet"
    stream.to_parquet(out_path)

    ic = daily_ic(stream)
    res = {"arm": args.arm, "kw": {k: bool(v) for k, v in kw.items()},
           "n_rows": int(len(stream)), "train_seconds": round(time.time() - t00, 1),
           "mean_daily_ic": round(float(ic.mean()), 6),
           "ic_by_fold": {}, "epochs_selected": {}, "stream": str(out_path)}
    for f, g in stream.groupby("fold"):
        res["ic_by_fold"][str(f)] = round(float(daily_ic(g).mean()), 6)
    res["epochs_selected"] = {"min": int(min(e["ep"] for e in ep_logs)),
                              "max": int(max(e["ep"] for e in ep_logs)),
                              "mean": round(float(np.mean([e["ep"] for e in ep_logs])), 2)}

    # 机制读数：尾部篮子周均收益（事实 2 的直击面），本臂与 #14 缓存 B 流并比
    ref = pd.read_parquet(REF_STREAM)
    ref["date"] = pd.to_datetime(ref["date"])
    res["tail_bucket_this_arm"] = tail_bucket_weekly(stream)
    res["tail_bucket_F0_cached"] = tail_bucket_weekly(ref)
    # ---- 自证 F0 / 对照读数 ----
    ic_ref = daily_ic(ref)
    if args.arm == "F0":
        j = ref.merge(stream, on=["date", "product"], suffixes=("_ref", "_new"))
        max_dev = float(np.abs(j["pred_ref"] - j["pred_new"]).max())
        res["x0_selfcheck"] = {"n_matched": int(len(j)), "max_abs_dev": max_dev,
                               "tolerance": X0_TOL, "ok": bool(max_dev <= X0_TOL)}
        print(f"  F0 自证: 匹配 {len(j)} 行 max|Δpred|={max_dev:.2e} "
              f"包络 {X0_TOL:.1e} ⇒ {'✓' if max_dev <= X0_TOL else '✗ 判废'}", flush=True)
    else:
        d_ic = ic.reindex(ic_ref.index).fillna(0) - ic_ref.fillna(0)
        res["signal_layer_vs_F0cached"] = {
            "mean_diff_ic": round(float(d_ic.mean()), 6),
            "paired_t": round(float(d_ic.mean() / (d_ic.std(ddof=1) / np.sqrt(len(d_ic)))), 3),
            "n_days": int(len(d_ic)), "pass_t2": bool(
                d_ic.mean() > 0 and d_ic.mean() / (d_ic.std(ddof=1) / np.sqrt(len(d_ic)))
                >= PAIR_T)}
        print(f"  信号层 vs #14 缓存 B 流: ΔIC={d_ic.mean():+.5f} "
              f"t={res['signal_layer_vs_F0cached']['paired_t']:+.2f}", flush=True)

    if args.arm in ("NEG", "NEG_M1"):
        res["negative_control"] = {"abs_mean_ic": round(abs(float(ic.mean())), 6),
                                   "tripwire": TRIPWIRE_IC,
                                   "ok": bool(abs(float(ic.mean())) <= TRIPWIRE_IC),
                                   "verdict": "塌到噪声 ✓" if abs(float(ic.mean()))
                                   <= TRIPWIRE_IC else "未塌 ⇒ 疑似泄漏，本臂判废"}
        print(f"  阴性对照: |IC|={abs(float(ic.mean())):.5f} ⇒ "
              f"{res['negative_control']['verdict']}", flush=True)

    # ---- 组合层裁判：忠实臂（真 T3 + daily EMA）----
    if args.arm in ("M1", "M2"):
        panel = stream.pivot_table(index="date", columns="product", values="pred")
        dates = panels["close"].index
        start_ts = pd.Timestamp(START_OOS)
        s = pd.Series(dates, index=dates)
        month_ends = [d for d in s.groupby([dates.year, dates.month]).last() if d >= start_ts]
        bench = sync_index.load(cfg, "000905.SH")["close"]
        bench = bench.loc[bench.index >= start_ts].astype("float64")
        eq, m = TF.arm_faithful(panel, TF.T3, cfg, bundle, panels, dates, start_ts,
                                ema_step="daily")
        res["portfolio_faithful"] = {"total_return": round(float(m["total_return"]), 6),
                                     "max_drawdown": round(float(m["max_drawdown"]), 6),
                                     "total_costs": round(float(m["total_costs"]), 1),
                                     "position_util_mean": round(
                                         float(m.get("position_util_mean", 0)), 4),
                                     "n_blocked_cash": int(m.get("n_blocked_cash", 0)),
                                     **weekly_stats(eq),
                                     **{("beta_" + k): v for k, v in
                                        TF.beta_ir(eq, bench).items()}}
        print(f"  组合层(忠实): {m['total_return']:+.2%} MDD {m['max_drawdown']:+.2%} "
              f"成本 {m['total_costs']:.0f} 周std {res['portfolio_faithful']['weekly_std_ann']} "
              f"正周 {res['portfolio_faithful']['pct_positive_weeks']} "
              f"blocked_cash {res['portfolio_faithful']['n_blocked_cash']}", flush=True)
        eq.to_frame("equity").to_parquet(EXDIR / f"eq_{args.arm}_faithful.parquet")

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    (REPORTS / f"tailloss_{args.arm}_{stamp}.json").write_text(
        json.dumps({"prereg": "plans/tailloss-prereg-20261008.md",
                    "seeds": list(SEEDS), "results": res, "verdict_word": None},
                   ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
