# -*- coding: utf-8 -*-
"""#132 · 判弱因子买入前过滤（模型外规则）双臂 A/B（B 系列 B9 · 用户拍板提前）。

预注册：plans/avoid-fin-prereg-20261014.md（冻结 commit d6b678c，先于本执行器）。
因子 = fin_netprofit_yoy（#80 PIT 复核通过；极性低值=弱，#79 W0）。
F1 = 买入候选 head(50) 内剔除「因子值存在且当日截面 pct-rank ≤ 0.10」者，顺延；
缺失 = 放行。A0 = 现行买入（同批重跑）。退出侧不动（keep4，#128 已证无自由度）。
判据：G0 / S1（被过滤 vs 实际买入配对差）/ C1 / S2 / FR 过滤率（预注册 §1）。
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fterm.config import load_config                      # noqa: E402
from fterm.data import ashare as sa                       # noqa: E402
from fterm.features.build import build_dataset            # noqa: E402
from fterm.research.scaffold import report_meta           # noqa: E402
from monthly_chain import monthly_log_paired              # noqa: E402
from fterm.model.stats import newey_west_t                # noqa: E402
from fterm.backtest.engine import Target                  # noqa: E402
import tierfix_replay as TF                               # noqa: E402

OUT = ROOT / "experiments" / "avoid_fin_20261014"
OUT.mkdir(parents=True, exist_ok=True)
START_OOS = "2019-01-01"
MAX_DATA_DATE = pd.Timestamp("2026-09-30")
W2_TIER = dict(equity=20000, top_n=2, keep_n=4, max_symbols=2, max_lots=10)
K_CAND = 50                   # 候选带宽（§0 冻结）
LOW_Q = 0.10                  # 低分位阈值（§0 冻结）
ROUND_TRIP_BP = 17.1
NOTIONAL_FRAC = 9500.0 / 20000.0
HALF_SPLIT = pd.Timestamp("2023-01-01")

t0 = time.time()


def log(*a):
    print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)


def main() -> int:
    cfg0 = load_config(ROOT / "config.yaml")
    log("== 面板（研究线 worktree）==")
    panels = sa.load_panels(cfg0)
    if panels["close"].index.max() != MAX_DATA_DATE:
        raise SystemExit(f"[G8 门] 面板右端点 {panels['close'].index.max().date()} != "
                         f"{MAX_DATA_DATE.date()}")
    bundle = build_dataset(cfg0, compute_fingerprints=False)
    dates = panels["close"].index
    start_ts = pd.Timestamp(START_OOS)
    s = pd.Series(dates, index=dates)
    month_ends = [d for d in s.groupby([dates.year, dates.month]).last() if d >= start_ts]
    raw_close = panels["close"]
    reb_days = [d for d in dates if d >= start_ts
                and d.weekday() == cfg0.signal.rebalance_weekday]

    rl2 = pd.read_parquet(ROOT / "data" / "reports" / "rankloss_ss7_21_99_5_13.parquet")
    rl2["date"] = pd.to_datetime(rl2["date"])
    assert rl2["date"].max() <= MAX_DATA_DATE
    panel = rl2.pivot_table(index="date", columns="product", values="pred").astype("float32")

    # ---- 因子面板（#80 PIT 复核版；宽表 index=date, columns=product）----
    fin = pd.read_parquet(ROOT / "data" / "external" / "fin_netprofit_yoy.parquet")
    fin.index = pd.to_datetime(fin.index)
    fin = fin[~fin.index.duplicated()]
    # 每日截面 pct-rank（有值股票内；缺失保持 NaN = 放行）
    fin_rank = fin.rank(axis=1, pct=True)
    log(f"fin 面板 {fin.shape}，{fin.notna().mean(axis=1).median():.1%} 中位日覆盖")

    filtered_events = []

    def buy_filter_f1(ranked, smooth, date):
        cand = ranked.head(K_CAND).index
        fr = fin_rank.loc[date] if date in fin_rank.index else None
        kept, skipped = [], []
        for c in cand:
            r = None if fr is None else fr.get(c)
            if r is not None and not pd.isna(r) and r <= LOW_Q:
                skipped.append({"date": str(date.date()), "code": c,
                                "fin_rank": round(float(r), 4)})
            else:
                kept.append(c)
        filtered_events.extend(skipped)
        return kept

    out = {"prereg": "plans/avoid-fin-prereg-20261014.md",
           "stamp": datetime.now().strftime("%Y%m%d_%H%M"),
           "tier": W2_TIER, "start_ts": START_OOS, "K_cand": K_CAND, "low_q": LOW_Q,
           "meta": report_meta(cfg0, panels, bundle, harness="faithful",
                               dividend_basis="hfq_inclusive", ema_step="daily",
                               stream_paths=["data/reports/rankloss_ss7_21_99_5_13.parquet"],
                               window_max=MAX_DATA_DATE, tier=W2_TIER,
                               extra={"prereg_commit": "d6b678c"})}

    runs = {}
    for nm, bfn in (("A0", None), ("F1", buy_filter_f1)):
        t_a = time.time()
        stats = {}
        eq, m = TF.arm_faithful(panel, W2_TIER, cfg0, bundle, panels, dates, start_ts,
                                ema_step="daily", stats=stats, buy_filter_fn=bfn)
        runs[nm] = {"equity": eq, "metrics": m, "stats": stats,
                    "seconds": round(time.time() - t_a, 1)}
        eq.to_frame("equity").to_parquet(OUT / f"eq_{nm}.parquet")
        log(f"[{nm}] ret={m.get('total_return', float('nan')):+.4f} "
            f"MDD={m.get('max_drawdown', float('nan')):+.4f} "
            f"blocked={m.get('n_blocked_cash', '?')} targets={m.get('n_targets', '?')} "
            f"({runs[nm]['seconds']}s)")
    log(f"过滤事件总数 = {len(filtered_events)}")
    (OUT / "filtered_events.json").write_text(
        json.dumps(filtered_events, ensure_ascii=False, indent=0), encoding="utf-8")
    if not filtered_events:
        log("[FR] 全史零过滤 ⇒ 机制不可达，登记关闭（预注册 §2）")
        out["verdict"] = {"FR": 0, "mechanism": "unreachable", "pass": False}
        (OUT / "avoid_fin.json").write_text(json.dumps(out, ensure_ascii=False, indent=1,
                                                       default=str), encoding="utf-8")
        return 0

    # ---- G0 确定性 + A0 逐位锚（buy_filter_fn=None 双跑） ----
    eq0b, _ = TF.arm_faithful(panel, W2_TIER, cfg0, bundle, panels, dates, start_ts,
                              ema_step="daily")
    g0_bit = float((eq0b - runs["A0"]["equity"]).abs().max())
    eq1b, _ = TF.arm_faithful(panel, W2_TIER, cfg0, bundle, panels, dates, start_ts,
                              ema_step="daily", buy_filter_fn=buy_filter_f1)
    g0_det = float((eq1b - runs["F1"]["equity"]).abs().max())
    log(f"[G0] A0 双跑 max|Δeq|={g0_bit:.2e} | F1 确定性 max|Δeq|={g0_det:.2e}")

    # ---- S1：被过滤标的 vs 实际买入标的的持有段收益（触发日收盘→下次调仓日收盘） ----
    reb_arr = pd.DatetimeIndex(reb_days)

    def hold_rets(evs):
        rets = {}
        for e in evs:
            d0 = pd.Timestamp(e["date"]); c = e["code"]
            nxt = reb_arr[reb_arr > d0]
            if len(nxt) == 0 or c not in raw_close.columns:
                continue
            p0, p1 = raw_close.at[d0, c], raw_close.at[nxt[0], c]
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                rets[(e["date"], c)] = (p1 / p0 - 1.0) * 1e4
        return rets

    # 实际买入标的：F1 的成交可从 equity 无法反推 ⇒ 用 A0 成交+过滤差集近似；
    # 精确做法：被过滤标的的"同窗收益" vs "kept 前 top2 同窗收益"逐事件配对
    skipped_rets = hold_rets(filtered_events)
    # 对照：同日 kept 候选的 top2（实际买入近似）
    kept_rets = []
    by_date = {}
    for e in filtered_events:
        by_date.setdefault(e["date"], []).append(e["code"])
    for dstr, skipped_codes in by_date.items():
        d0 = pd.Timestamp(dstr)
        nxt = reb_arr[reb_arr > d0]
        if len(nxt) == 0:
            continue
        row = panel.loc[d0].dropna()
        if row.empty:
            continue
        rk = row.rank(pct=True, ascending=False).sort_values()
        fr = fin_rank.loc[d0] if d0 in fin_rank.index else None
        kept = []
        for c in rk.index[:K_CAND]:
            r = None if fr is None else fr.get(c)
            hit = (r is not None) and (not pd.isna(r)) and (r <= LOW_Q)
            if not hit and c not in skipped_codes:
                kept.append(c)
            if len(kept) >= W2_TIER["top_n"]:
                break
        for c in kept:
            p0, p1 = raw_close.at[d0, c], raw_close.at[nxt[0], c]
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                kept_rets.append((p1 / p0 - 1.0) * 1e4)
    sk = np.asarray(list(skipped_rets.values()), "float64")
    kp = np.asarray(kept_rets, "float64")
    n_pair = min(len(sk), len(kp))
    diff = sk[:n_pair] - kp[:n_pair]
    t_s1 = newey_west_t(diff) if n_pair >= 5 else None
    s1 = {"n_skipped": int(len(sk)), "n_kept": int(len(kp)),
          "skipped_mean_bp": float(sk.mean()) if len(sk) else None,
          "kept_mean_bp": float(kp.mean()) if len(kp) else None,
          "diff_mean_bp": float(diff.mean()) if n_pair else None,
          "nw_t": (float(t_s1) if t_s1 is not None and np.isfinite(t_s1) else None)}
    log(f"[S1] skipped={s1['skipped_mean_bp'] if len(sk) else float('nan'):+.1f}bp "
        f"kept={s1['kept_mean_bp'] if len(kp) else float('nan'):+.1f}bp "
        f"diff={s1['diff_mean_bp'] if n_pair else float('nan'):+.1f}bp t={t_s1}")

    # ---- C1 成本门 ----
    mp = monthly_log_paired(runs["F1"]["equity"], runs["A0"]["equity"], month_ends)
    e_f = float(runs["F1"]["equity"].iloc[-1]); e_0 = float(runs["A0"]["equity"].iloc[-1])
    gross = e_f / e_0 - 1.0
    n_sell_f = len(filtered_events)          # F1 的额外卖出 ≈ 过滤引发的补位换手
    drag = n_sell_f * NOTIONAL_FRAC * (ROUND_TRIP_BP / 1e4)
    c1 = {"monthly": mp, "gross_diff_full": round(gross, 6),
          "cost_drag_full": round(drag, 6), "net_diff_full": round(gross - drag, 6),
          "net_positive": bool(gross - drag > 0)}
    log(f"[C1] gross={gross:+.4f} drag={drag:.4f} net={c1['net_diff_full']:+.4f} monthly_t={mp.get('t')}")

    # ---- S2 稳健 ----
    dirs = []
    for lo, hi in ((start_ts, HALF_SPLIT), (HALF_SPLIT, dates[-1])):
        ea, e0 = runs["F1"]["equity"], runs["A0"]["equity"]
        idx = ea.index[(ea.index >= lo) & (ea.index < hi)]
        if len(idx) < 2:
            dirs.append(0); continue
        g = ea.loc[idx[-1]] / ea.loc[idx[0]] - e0.loc[idx[-1]] / e0.loc[idx[0]]
        dirs.append(1 if g > 0 else (-1 if g < 0 else 0))
    s2 = {"half_dirs": dirs, "same_sign": bool(dirs[0] == dirs[1] and dirs[0] != 0)}

    # ---- 判据表 ----
    t_ok = t_s1 is not None and t_s1 >= 1.64
    s1_pass = bool(t_ok and s1["diff_mean_bp"] is not None
                   and s1["diff_mean_bp"] >= 11.5)
    c1_pass = bool(mp.get("t") is not None and mp["t"] >= 2
                   and mp.get("mean_log", 0) > 0 and c1["net_positive"])
    mdd_f = runs["F1"]["metrics"].get("max_drawdown", 0)
    mdd_0 = runs["A0"]["metrics"].get("max_drawdown", 0)
    f_flag = "ok" if (mdd_f - mdd_0) > -0.05 else ("warn" if (mdd_f - mdd_0) > -0.10
                                                   else "recommend_reject")
    gates = {"G0": {"a0_bitwise": bool(g0_bit == 0.0), "det": bool(g0_det <= 1e-9),
                    "blocked_zero": bool(runs["A0"]["metrics"].get("n_blocked_cash", 1) == 0
                                         and runs["F1"]["metrics"].get("n_blocked_cash", 1) == 0)},
             "FR": len(filtered_events),
             "S1": {**s1, "pass": s1_pass},
             "C1": {**c1, "pass": c1_pass},
             "S2": {**s2, "pass": s2["same_sign"]},
             "F_mdd_flag": f_flag}
    gates["pass"] = bool(gates["G0"]["a0_bitwise"] and gates["G0"]["det"]
                         and gates["G0"]["blocked_zero"] and s1_pass and c1_pass
                         and s2["same_sign"])
    out["gates"] = gates
    out["runs_summary"] = {nm: {"total_return": runs[nm]["metrics"].get("total_return"),
                                "max_drawdown": runs[nm]["metrics"].get("max_drawdown"),
                                "n_targets": runs[nm]["metrics"].get("n_targets")}
                           for nm in ("A0", "F1")}
    (OUT / "avoid_fin.json").write_text(json.dumps(out, ensure_ascii=False, indent=1,
                                                   default=str), encoding="utf-8")
    log("verdict:", json.dumps(gates, ensure_ascii=False, default=str))
    log(f"结果 → {OUT / 'avoid_fin.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
