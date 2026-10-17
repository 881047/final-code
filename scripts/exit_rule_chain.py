# -*- coding: utf-8 -*-
"""#128 · 判弱侧退出引擎四臂 A/B（B 系列 B1 · 批次二）。

预注册：plans/exit-rule-prereg-20261014.md（冻结 commit 89adb7c，先于本执行器）。
harness = tierfix_replay.arm_faithful（已参数化 exit_fn，默认路径逐位不变）。
四臂：A0=keep4 基线（同批重跑）/ A1=AVOID-10 / A2=AVOID-30 / A3=混合（keep4 或 0.10）。
w2 构型；RL2 10-14 新部署流；ema_step=daily；面板 ≤2026-09-30（G8 竞态闸）。
判据：G0 仪器 / S1 信号层 / C1 成本门 / S2 稳健 / NEG 阴性 / TR 触发率（预注册 §1）。
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

OUT = ROOT / "experiments" / "exit_rule_20261014"
OUT.mkdir(parents=True, exist_ok=True)
START_OOS = "2019-01-01"
MAX_DATA_DATE = pd.Timestamp("2026-09-30")
W2_TIER = dict(equity=20000, top_n=2, keep_n=4, max_symbols=2, max_lots=10)
AVOID_10, AVOID_30 = 0.10, 0.30            # avoid_value_chain 同款
ROUND_TRIP_BP = 17.1                        # #31 冻结：2 万档单票 9,500 元往返
NOTIONAL_FRAC = 9500.0 / 20000.0            # 单票名义占权益（成本折算用）
NEG_SEED = 20261015
S1_MIN_ABS_BP = 11.5                        # #31 锚 −23.0bp 保守折半
HALF_SPLIT = pd.Timestamp("2023-01-01")     # S2 前后半段界

t0 = time.time()


def log(*a):
    print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)


def make_keep_exit(events: list, day_hold: list):
    """A0 的事件统计版（语义与 exit_fn=None 逐位一致：跌出 keep_n ⇒ 卖，仅多记录）。"""
    def _fn(ranked, smooth, positions, tier, date):
        keep = set(ranked.head(tier["keep_n"]).index)
        outs = []
        for c in positions:
            if c not in keep:
                events.append({"date": str(date.date()), "code": c})
                outs.append(Target(c, 0, 0, f"exit_band(keep_{tier['keep_n']})"))
        day_hold.append(len(positions))
        return outs
    return _fn


def make_avoid_exit(threshold: float, events: list, day_hold: list):
    """AVOID 阈值退出（A1/A2）。缺失于平滑面/预测面 ⇒ 卖（fail-safe）。"""
    def _fn(ranked, smooth, positions, tier, date):
        outs = []
        for c in positions:
            v = smooth.get(c)
            trig = (v is None) or (v <= threshold) or (c not in ranked.index)
            if trig:
                events.append({"date": str(date.date()), "code": c,
                               "smooth": (None if v is None else round(float(v), 4))})
                outs.append(Target(c, 0, 0, f"avoid_exit({threshold})"))
        day_hold.append(len(positions))
        return outs
    return _fn


def make_hybrid_exit(threshold: float, events: list, day_hold: list):
    """A3 混合：跌出 keep4 或平滑秩 ≤ threshold，任一触发。"""
    def _fn(ranked, smooth, positions, tier, date):
        keep = set(ranked.head(tier["keep_n"]).index)
        outs = []
        for c in positions:
            v = smooth.get(c)
            trig = (c not in keep) or (v is None) or (v <= threshold) \
                   or (c not in ranked.index)
            if trig:
                reason = "hybrid_keep" if c not in keep else "hybrid_avoid"
                events.append({"date": str(date.date()), "code": c,
                               "smooth": (None if v is None else round(float(v), 4)),
                               "reason": reason})
                outs.append(Target(c, 0, 0, f"hybrid_exit({reason})"))
        day_hold.append(len(positions))
        return outs
    return _fn


def make_random_exit(p_neg: float, events: list, day_hold: list, seed=NEG_SEED):
    """NEG：同概率随机退出带（种子固定，propose 序确定 ⇒ 可复现）。"""
    rng = np.random.default_rng(seed)
    def _fn(ranked, smooth, positions, tier, date):
        outs = []
        for c in positions:
            if rng.random() < p_neg:
                events.append({"date": str(date.date()), "code": c})
                outs.append(Target(c, 0, 0, "random_exit"))
        day_hold.append(len(positions))
        return outs
    return _fn


def main() -> int:
    cfg0 = load_config(ROOT / "config.yaml")
    log("== 面板（研究线 worktree）==")
    panels = sa.load_panels(cfg0)
    if panels["close"].index.max() != MAX_DATA_DATE:
        raise SystemExit(f"[G8 门] 面板右端点 {panels['close'].index.max().date()} != "
                         f"{MAX_DATA_DATE.date()}；必须在研究线 worktree 内跑")
    bundle = build_dataset(cfg0, compute_fingerprints=False)
    dates = panels["close"].index
    start_ts = pd.Timestamp(START_OOS)
    s = pd.Series(dates, index=dates)
    month_ends = [d for d in s.groupby([dates.year, dates.month]).last() if d >= start_ts]
    raw_close = panels["close"]                     # hfq 含权口径（#38）
    reb_days = [d for d in dates if d >= start_ts
                and d.weekday() == cfg0.signal.rebalance_weekday]

    rl2 = pd.read_parquet(ROOT / "data" / "reports" / "rankloss_ss7_21_99_5_13.parquet")
    rl2["date"] = pd.to_datetime(rl2["date"])
    assert rl2["date"].max() <= MAX_DATA_DATE, "RL2 流越 G8 线"
    panel = rl2.pivot_table(index="date", columns="product", values="pred").astype("float32")
    log(f"RL2 面板 {panel.shape}，{len(reb_days)} 个调仓日")

    arms = {}
    # A0 基线（exit_fn=None，G0 逐位锚；同批重跑）
    arms["A0"] = {"exit_fn": None, "events": None, "holds": None}
    for nm, th in (("A1", AVOID_10), ("A2", AVOID_30)):
        ev, hd = [], []
        arms[nm] = {"exit_fn": make_avoid_exit(th, ev, hd), "events": ev, "holds": hd}
    ev, hd = [], []
    arms["A3"] = {"exit_fn": make_hybrid_exit(AVOID_10, ev, hd), "events": ev, "holds": hd}
    # A0 的事件统计版（语义同 None，仅多记录；用于卖笔差）
    ev0, hd0 = [], []
    arms["A0E"] = {"exit_fn": make_keep_exit(ev0, hd0), "events": ev0, "holds": hd0}

    out = {"prereg": "plans/exit-rule-prereg-20261014.md",
           "stamp": datetime.now().strftime("%Y%m%d_%H%M"),
           "tier": W2_TIER, "start_ts": START_OOS,
           "meta": report_meta(cfg0, panels, bundle, harness="faithful",
                               dividend_basis="hfq_inclusive", ema_step="daily",
                               stream_paths=["data/reports/rankloss_ss7_21_99_5_13.parquet"],
                               window_max=MAX_DATA_DATE, tier=W2_TIER,
                               extra={"prereg_commit": "89adb7c"})}
    runs = {}
    for nm in ("A0", "A0E", "A1", "A2", "A3"):
        t_a = time.time()
        stats = {}
        eq, m = TF.arm_faithful(panel, W2_TIER, cfg0, bundle, panels, dates, start_ts,
                                ema_step="daily", stats=stats,
                                exit_fn=arms[nm]["exit_fn"])
        runs[nm] = {"equity": eq, "metrics": m, "stats": stats,
                    "events": arms[nm]["events"], "day_holds": arms[nm]["holds"],
                    "seconds": round(time.time() - t_a, 1)}
        if nm != "A0E":
            eq.to_frame("equity").to_parquet(OUT / f"eq_{nm}.parquet")
        log(f"[{nm}] ret={m.get('total_return', float('nan')):+.4f} "
            f"MDD={m.get('max_drawdown', float('nan')):+.4f} "
            f"blocked={stats.get('n_blocked_cash', '?')} targets={m.get('n_targets', '?')} "
            f"events={len(arms[nm]['events'] or [])} ({runs[nm]['seconds']}s)")
    # A0 双跑对账（G0 ①：None 路径与 keep 事件版语义逐位一致）
    eq_same = float((runs["A0"]["equity"] - runs["A0E"]["equity"]).abs().max())
    log(f"[G0] A0(None) vs A0E(keep 记录版) max|Δeq|={eq_same:.2e}")
    if runs["A0"]["metrics"].get("n_blocked_cash", 1) != 0:
        raise SystemExit("[G0] A0 blocked_cash != 0 ⇒ 全链作废")

    # ---- G0 确定性自证：A1 同输入重跑 ----
    eq1b, _ = TF.arm_faithful(panel, W2_TIER, cfg0, bundle, panels, dates, start_ts,
                              ema_step="daily", exit_fn=arms["A1"]["exit_fn"])
    det_max = float((eq1b - runs["A1"]["equity"]).abs().max())
    log(f"[G0] A1 确定性重跑 max|Δeq|={det_max:.2e}")
    g0_det_ok = det_max <= 1e-9

    # ---- NEG 臂（同换手率：p_neg 由 A1 触发数标定） ----
    n1 = len(runs["A1"]["events"])
    n_slots = 2 * len(reb_days)                     # 上限口径（max 2 持仓）
    p_neg = min(0.9, n1 / max(n_slots, 1))
    ev, hd = [], []
    arms["NEG"] = {"exit_fn": make_random_exit(p_neg, ev, hd), "events": ev, "holds": hd}
    stats = {}
    eqN, mN = TF.arm_faithful(panel, W2_TIER, cfg0, bundle, panels, dates, start_ts,
                              ema_step="daily", stats=stats, exit_fn=arms["NEG"]["exit_fn"])
    runs["NEG"] = {"equity": eqN, "metrics": mN, "stats": stats,
                   "events": ev, "day_holds": hd, "p_neg": round(p_neg, 4)}
    eqN.to_frame("equity").to_parquet(OUT / "eq_NEG.parquet")
    log(f"[NEG] p_neg={p_neg:.4f} 事件={len(ev)} ret={mN.get('total_return', float('nan')):+.4f}")

    # ---- S1：退出触发标的的持有段收益（触发日收盘 → 下次调仓日收盘；机会成本口径） ----
    reb_arr = pd.DatetimeIndex(reb_days)
    s1 = {}
    for nm in ("A1", "A2", "A3", "NEG"):
        evs = runs[nm]["events"] or []
        rets = []
        for e in evs:
            d0 = pd.Timestamp(e["date"])
            nxt = reb_arr[reb_arr > d0]
            if len(nxt) == 0 or e["code"] not in raw_close.columns:
                continue
            p0, p1 = raw_close.at[d0, e["code"]], raw_close.at[nxt[0], e["code"]]
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                rets.append((p1 / p0 - 1.0) * 1e4)
        arr = np.asarray(rets, "float64")
        t_v = newey_west_t(arr) if arr.size >= 5 else None
        s1[nm] = {"n_events_used": int(arr.size), "mean_bp": (float(arr.mean()) if arr.size else None),
                  "nw_t": (float(t_v) if t_v is not None and np.isfinite(t_v) else None)}
        log(f"[S1 {nm}] n={s1[nm]['n_events_used']} mean={s1[nm]['mean_bp'] if arr.size else float('nan'):+.1f}bp "
            f"t={s1[nm]['nw_t'] if t_v is not None else float('nan'):+.2f}")

    # ---- TR 触发率（近似口径：事件 / (2×调仓日)） ----
    tr = {nm: round(len(runs[nm]["events"] or []) / max(n_slots, 1), 4)
          for nm in ("A1", "A2", "A3", "NEG")}

    # ---- C1 成本门：月度配对 + 成本调整（卖笔差 × NOTIONAL_FRAC × 17.1bp） ----
    def sell_cost_drag(n_sells_a, n_sells_b, eq_base):
        """Δ卖笔 × 单票名义占比 × 往返成本（占权益比例）。"""
        return abs(n_sells_a - n_sells_b) * NOTIONAL_FRAC * (ROUND_TRIP_BP / 1e4)

    c1 = {}
    for nm in ("A1", "A2", "A3", "NEG"):
        mp = monthly_log_paired(runs[nm]["equity"], runs["A0"]["equity"], month_ends)
        e_a = float(runs[nm]["equity"].iloc[-1]); e_0 = float(runs["A0"]["equity"].iloc[-1])
        gross_diff = e_a / e_0 - 1.0
        n_sell_a = len(runs[nm]["events"] or [])
        n_sell_0 = len(runs["A0E"]["events"])
        drag = abs(n_sell_a - n_sell_0) * NOTIONAL_FRAC * (ROUND_TRIP_BP / 1e4)
        c1[nm] = {"monthly": mp,
                  "gross_diff_full": round(gross_diff, 6),
                  "n_exit_events": n_sell_a, "n_exit_events_A0": n_sell_0,
                  "cost_drag_full": round(drag, 6),
                  "net_diff_full": round(gross_diff - drag, 6),
                  "net_positive": bool(gross_diff - drag > 0)}
        log(f"[C1 {nm}] gross={gross_diff:+.4f} drag={drag:.4f} net={c1[nm]['net_diff_full']:+.4f} "
            f"monthly_t={mp.get('t')}")

    # ---- S2 稳健门：前后半段月度毛差方向 ----
    s2 = {}
    for nm in ("A1", "A2", "A3"):
        dirs = []
        for lo, hi in ((start_ts, HALF_SPLIT), (HALF_SPLIT, dates[-1])):
            ea = runs[nm]["equity"]; e0 = runs["A0"]["equity"]
            idx = ea.index[(ea.index >= lo) & (ea.index < hi)]
            if len(idx) < 2:
                dirs.append(0); continue
            g = ea.loc[idx[-1]] / ea.loc[idx[0]] - e0.loc[idx[-1]] / e0.loc[idx[0]]
            dirs.append(1 if g > 0 else (-1 if g < 0 else 0))
        s2[nm] = {"half_dirs": dirs, "same_sign": bool(dirs[0] == dirs[1] and dirs[0] != 0)}
        log(f"[S2 {nm}] halves={dirs} same={s2[nm]['same_sign']}")

    # ---- 判据表（预注册 §1） ----
    blocked_ok = all(runs[k]["metrics"].get("n_blocked_cash", 1) == 0
                     for k in ("A0", "A0E", "A1", "A2", "A3"))
    gates = {"G0": {"a0_none_vs_keep_bitwise": bool(eq_same == 0.0),
                    "det_rerun_max": det_max,
                    "blocked_cash_zero": bool(blocked_ok),
                    "pass": bool(eq_same == 0.0 and g0_det_ok and blocked_ok)}}
    for nm in ("A1", "A2", "A3"):
        t_v = s1[nm]["nw_t"]; mean_bp = s1[nm]["mean_bp"] or 0.0
        s1_pass = bool(t_v is not None and t_v <= -1.64 and abs(mean_bp) >= S1_MIN_ABS_BP)
        c1_pass = bool(c1[nm]["net_positive"]
                       and (c1[nm]["monthly"].get("t") is not None
                            and c1[nm]["monthly"]["t"] >= 2
                            and c1[nm]["monthly"].get("mean_log", 0) > 0))
        s2_pass = s2[nm]["same_sign"]
        neg_ok = True
        if nm == "A1":
            t1, tn = s1["A1"]["nw_t"], s1["NEG"]["nw_t"]
            neg_ok = bool(t1 is not None and tn is not None and t1 < tn)
        mdd_a = runs[nm]["metrics"].get("max_drawdown", 0)
        mdd_0 = runs["A0"]["metrics"].get("max_drawdown", 0)
        f_flag = "ok" if (mdd_a - mdd_0) > -0.05 else ("warn" if (mdd_a - mdd_0) > -0.10
                                                       else "recommend_reject")
        arm_tr = tr[nm]
        mech = "ok" if not (nm == "A1" and arm_tr < 0.01) else "unreachable"
        gates[nm] = {"S1": s1_pass, "C1": c1_pass, "S2": s2_pass,
                     "NEG": (neg_ok if nm == "A1" else None),
                     "TR": arm_tr, "mech_reachable": mech != "unreachable",
                     "F_mdd_flag": f_flag,
                     "pass": bool(s1_pass and c1_pass and s2_pass and neg_ok
                                  and mech != "unreachable")}
    gates["overall"] = {nm: gates[nm]["pass"] for nm in ("A1", "A2", "A3")}
    out["runs_summary"] = {nm: {"total_return": runs[nm]["metrics"].get("total_return"),
                                "max_drawdown": runs[nm]["metrics"].get("max_drawdown"),
                                "n_targets": runs[nm]["metrics"].get("n_targets"),
                                "n_exit_events": len(runs[nm]["events"] or []),
                                "seconds": runs[nm].get("seconds")} for nm in runs}
    out["S1"] = s1; out["TR"] = tr; out["C1"] = c1; out["S2"] = s2; out["gates"] = gates
    (OUT / "exit_rule.json").write_text(json.dumps(out, ensure_ascii=False, indent=1,
                                                   default=str), encoding="utf-8")
    for nm in ("A1", "A2", "A3"):
        (OUT / f"exits_{nm}.json").write_text(
            json.dumps(runs[nm]["events"] or [], ensure_ascii=False, indent=0),
            encoding="utf-8")
    log("判据表:", json.dumps(gates["overall"], ensure_ascii=False))
    log(f"结果 → {OUT / 'exit_rule.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
