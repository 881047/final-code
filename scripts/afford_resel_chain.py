# -*- coding: utf-8 -*-
"""#131 · 可负担性带内重选（买不起顺延）双臂 A/B（B 系列 B7）。

预注册：plans/afford-resel-prereg-20261014.md（冻结 commit c530c01，先于本执行器）。
B7 臂 = buy_filter_fn 返回 ranked.head(50) 全列表（无剔除、纯顺延）——买入循环对
size_position=0（买不起）者自然跳过。A0 = 现行 head(2)（同批重跑，逐位锚）。
判据：G0 / D1 部署门（空仓补救 ≥30%）/ S1 顺延标的容忍平（t ≥ −1.64）/ C1 / S2（§1）。
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
import tierfix_replay as TF                               # noqa: E402

OUT = ROOT / "experiments" / "afford_20261014"
OUT.mkdir(parents=True, exist_ok=True)
START_OOS = "2019-01-01"
MAX_DATA_DATE = pd.Timestamp("2026-09-30")
W2_TIER = dict(equity=20000, top_n=2, keep_n=4, max_symbols=2, max_lots=10)
K_CAND = 50                   # 顺延带宽（§0 冻结）
ROUND_TRIP_BP = 17.1
NOTIONAL_FRAC = 9500.0 / 20000.0
HALF_SPLIT = pd.Timestamp("2023-01-01")

t0 = time.time()


def log(*a):
    print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)


def main() -> int:
    cfg0 = load_config(ROOT / "config.yaml")
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

    rl2 = pd.read_parquet(ROOT / "data" / "reports" / "rankloss_ss7_21_99_5_13.parquet")
    rl2["date"] = pd.to_datetime(rl2["date"])
    assert rl2["date"].max() <= MAX_DATA_DATE
    panel = rl2.pivot_table(index="date", columns="product", values="pred").astype("float32")

    def b7_cands(ranked, smooth, date):
        return list(ranked.head(K_CAND).index)   # 无剔除，纯顺延（§0 冻结）

    out = {"prereg": "plans/afford-resel-prereg-20261014.md",
           "stamp": datetime.now().strftime("%Y%m%d_%H%M"),
           "tier": W2_TIER, "start_ts": START_OOS, "K": K_CAND,
           "meta": report_meta(cfg0, panels, bundle, harness="faithful",
                               dividend_basis="hfq_inclusive", ema_step="daily",
                               stream_paths=["data/reports/rankloss_ss7_21_99_5_13.parquet"],
                               window_max=MAX_DATA_DATE, tier=W2_TIER,
                               extra={"prereg_commit": "c530c01"})}

    runs = {}
    for nm, bfn in (("A0", None), ("B7", b7_cands)):
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

    # ---- G0 ----
    eq0b, _ = TF.arm_faithful(panel, W2_TIER, cfg0, bundle, panels, dates, start_ts,
                              ema_step="daily")
    eq7b, _ = TF.arm_faithful(panel, W2_TIER, cfg0, bundle, panels, dates, start_ts,
                              ema_step="daily", buy_filter_fn=b7_cands)
    g0_bit = float((eq0b - runs["A0"]["equity"]).abs().max())
    g0_det = float((eq7b - runs["B7"]["equity"]).abs().max())
    blocked_ok = all(runs[k]["metrics"].get("n_blocked_cash", 1) == 0 for k in ("A0", "B7"))
    log(f"[G0] A0 双跑={g0_bit:.2e} B7 确定性={g0_det:.2e} blocked_ok={blocked_ok}")

    # ---- D1 部署门：持仓日占比（equity 变动日≈有成交；更稳的口径：targets 数差 + 权益曲线非平坦日）----
    # 直接口径：从 metrics 拿不到持仓日 ⇒ 用月度权益变动近似：变更日占比
    def active_days(eq):
        chg = eq.diff().abs() > 1e-6
        return int(chg.sum()), int(len(eq))
    a_chg, a_tot = active_days(runs["A0"]["equity"])
    b_chg, b_tot = active_days(runs["B7"]["equity"])
    d1 = {"A0_active_days": a_chg, "B7_active_days": b_chg, "total_days": b_tot,
          "rescue_ratio_vs_A0": round(b_chg / max(a_chg, 1), 3)}
    # 空仓补救（口径：A0 空仓的调仓周中 B7 建仓的比例，以月度活跃差近似部署提升）
    deploy_lift = (b_chg - a_chg) / max(a_tot - a_chg, 1)
    d1["deploy_lift_on_idle"] = round(deploy_lift, 3)
    d1["pass"] = bool(deploy_lift >= 0.30)
    log(f"[D1] A0 活跃 {a_chg}/{a_tot} → B7 {b_chg}/{b_tot} | 空仓补救率 {deploy_lift:.1%} → "
        f"{'PASS' if d1['pass'] else 'FAIL'}")

    # ---- S1：顺延买入标的（rank 排名 >2 的建仓）持有段收益，容忍平 ----
    # 独立口径：对 A0 空仓的调仓日（A0 只试 head(2) 且全不可买），找 B7 视角首个
    # 可负担候选（px ≤ 单票预算上限 9500 元 ⇒ px ≤ 95 元/股），算持有段收益
    reb_arr = pd.DatetimeIndex([d for d in dates if d >= start_ts
                                and d.weekday() == cfg0.signal.rebalance_weekday])
    lot_budget = 9500.0
    late_rets = []
    for d in reb_arr:
        row = panel.loc[d].dropna()
        if row.empty:
            continue
        rk = row.rank(pct=True, ascending=False).sort_values().index[:K_CAND]
        # A0 视角：head(2) 是否全部买不起
        top2_afford = []
        for c in rk[:2]:
            px = raw_close.at[d, c] if c in raw_close.columns else np.nan
            if pd.notna(px) and px * 100 <= lot_budget:
                top2_afford.append(c)
        if len(top2_afford) >= 1:
            continue                                  # A0 能买 ⇒ 非 B7 目标场景
        # B7 顺延：第 3+ 名中首个可负担
        for c in rk[2:]:
            px = raw_close.at[d, c] if c in raw_close.columns else np.nan
            if pd.notna(px) and px * 100 <= lot_budget:
                nxt = reb_arr[reb_arr > d]
                if len(nxt):
                    p0, p1 = raw_close.at[d, c], raw_close.at[nxt[0], c]
                    if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                        late_rets.append((p1 / p0 - 1.0) * 1e4)
                break
    arr = np.asarray(late_rets, "float64")
    t_s1 = newey_west_t(arr) if arr.size >= 5 else None
    s1 = {"n_late_buys": int(arr.size),
          "mean_bp": float(arr.mean()) if arr.size else None,
          "nw_t": (float(t_s1) if t_s1 is not None and np.isfinite(t_s1) else None)}
    s1_pass = bool(t_s1 is not None and t_s1 >= -1.64)
    log(f"[S1] 顺延建仓 n={arr.size} mean={s1['mean_bp'] if arr.size else float('nan'):+.1f}bp "
        f"t={t_s1} → {'PASS(容忍平)' if s1_pass else 'FAIL(显著为负)'}")

    # ---- C1 成本门（增量开仓口径：新增建仓笔数 × 单票名义 × 单边 8.55bp） ----
    mp = monthly_log_paired(runs["B7"]["equity"], runs["A0"]["equity"], month_ends)
    e_b = float(runs["B7"]["equity"].iloc[-1]); e_0 = float(runs["A0"]["equity"].iloc[-1])
    gross = e_b / e_0 - 1.0
    extra_opens = max(len(late_rets), 0)
    drag = extra_opens * NOTIONAL_FRAC * (ROUND_TRIP_BP / 2 / 1e4)   # 单边
    c1 = {"monthly": mp, "gross_diff_full": round(gross, 6),
          "extra_opens": extra_opens, "cost_drag_full": round(drag, 6),
          "net_diff_full": round(gross - drag, 6), "net_positive": bool(gross - drag > 0)}
    c1_pass = bool(mp.get("t") is not None and mp["t"] >= 2
                   and mp.get("mean_log", 0) > 0 and c1["net_positive"])
    log(f"[C1] gross={gross:+.4f} drag={drag:.4f} net={c1['net_diff_full']:+.4f} monthly_t={mp.get('t')}")

    # ---- S2 ----
    dirs = []
    for lo, hi in ((start_ts, HALF_SPLIT), (HALF_SPLIT, dates[-1])):
        ea, e0 = runs["B7"]["equity"], runs["A0"]["equity"]
        idx = ea.index[(ea.index >= lo) & (ea.index < hi)]
        if len(idx) < 2:
            dirs.append(0); continue
        g = ea.loc[idx[-1]] / ea.loc[idx[0]] - e0.loc[idx[-1]] / e0.loc[idx[0]]
        dirs.append(1 if g > 0 else (-1 if g < 0 else 0))
    s2 = {"half_dirs": dirs, "pass": bool(dirs[0] == dirs[1] and dirs[0] != 0)}

    mdd_b = runs["B7"]["metrics"].get("max_drawdown", 0)
    mdd_0 = runs["A0"]["metrics"].get("max_drawdown", 0)
    f_flag = "ok" if (mdd_b - mdd_0) > -0.05 else ("warn" if (mdd_b - mdd_0) > -0.10
                                                   else "recommend_reject")
    gates = {"G0": {"a0_bitwise": bool(g0_bit == 0.0), "det": bool(g0_det <= 1e-9),
                    "blocked_ok": bool(blocked_ok)},
             "D1": d1, "S1": {**s1, "pass": s1_pass}, "C1": {**c1, "pass": c1_pass},
             "S2": s2, "F_mdd_flag": f_flag}
    gates["pass"] = bool(gates["G0"]["a0_bitwise"] and gates["G0"]["det"] and blocked_ok
                         and d1["pass"] and s1_pass and c1_pass and s2["pass"])
    out["gates"] = gates
    out["runs_summary"] = {nm: {"total_return": runs[nm]["metrics"].get("total_return"),
                                "max_drawdown": runs[nm]["metrics"].get("max_drawdown"),
                                "n_targets": runs[nm]["metrics"].get("n_targets")}
                           for nm in ("A0", "B7")}
    (OUT / "afford.json").write_text(json.dumps(out, ensure_ascii=False, indent=1,
                                                default=str), encoding="utf-8")
    log("verdict pass:", gates["pass"], "| flags:", json.dumps(gates, ensure_ascii=False,
                                                               default=str)[:600])
    log(f"结果 → {OUT / 'afford.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
