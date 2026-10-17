"""判据链 #5 · 月频重调度执行器 —— 判据冻结于 plans/monthly-prereg-20261004.md。

L_monthly（月末调仓，top2/keep4）vs S1 基线，月度对数收益配对（#2 erratum
整改口径），双种子集 OOS 流复用 #4 缓存，零训练。判决词由用户填写。
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

from fterm.features.universe import UniverseIndex  # noqa: E402
from fterm import contracts  # noqa: E402
from fterm.backtest.engine import BacktestEngine, Target  # noqa: E402
from fterm.backtest.two_pass import run_two_pass  # noqa: E402
from fterm.config import load_config  # noqa: E402
from fterm.data import ashare as sa  # noqa: E402
from fterm.features.build import build_dataset  # noqa: E402
from fterm.risk.governor import AccountState, RiskGovernor  # noqa: E402
from fterm.risk.sizer import size_position  # noqa: E402
from ablation_chain import CHAIN1_JSON  # noqa: E402
from judgment_chain import build_bars  # noqa: E402

REPORTS = Path(__file__).resolve().parents[1] / "data" / "reports"
PREDS = {"ss42_7": ROOT / "data/reports/lgbm_preds_ss42_7.parquet",
         "ss123_2024": ROOT / "data/reports/lgbm_preds_ss123_2024.parquet"}
START_OOS = "2019-01-01"

# ---- 冻结（预注册 §二） ----
C2A_T = 2.0
C3_TOL = 0.02
C4_RATIO = 1.25
TOP_N = 2
KEEP_N = 4


def run_monthly_arm(p_up_panel, cfg0, bundle, panels, dates, start_ts):
    from dataclasses import replace
    cfg_ = cfg0
    feats, universe = bundle.features, bundle.universe
    uni_mask = UniverseIndex(universe)
    raw_close = panels["close"]
    eng = BacktestEngine(cfg_, build_bars(panels))
    # 月末交易日（冻结：每月最后一个面板交易日）
    s = pd.Series(dates, index=dates)
    month_ends = s.groupby([dates.year, dates.month]).last()
    reb_all = [d for d in month_ends if d >= start_ts]

    def targets_for(d, equity, positions):
        if d not in p_up_panel.index:
            return []
        row = p_up_panel.loc[d].dropna()
        # 2026-10-08 B2：向量化（原逐格 universe.at[d, c]；见 fterm/features/universe.py）
        row = row[uni_mask.mask_for(d, row.index)]
        ranked = row.sort_values(ascending=False)
        keep = list(ranked.head(KEEP_N).index)
        tgts = [Target(c, 0, 0, f"exit_band(keep_{KEEP_N})") for c in list(positions) if c not in keep]
        for code in ranked.head(TOP_N).index:
            if code in positions:
                continue
            px = float(raw_close.at[d, code]) if pd.notna(raw_close.at[d, code]) else None
            if px is None:
                continue
            atr = feats["atr14"].at[d, code] if code in feats["atr14"].columns else np.nan
            sz = size_position(cfg_, equity, px, float(atr) if np.isfinite(atr) else 0.0)
            if sz.lots > 0:
                tgts.append(Target(code, 1, sz.lots, f"monthly(rank={float(ranked[code]):.3f})"))
        return tgts

    # 2026-10-08 去重 + B5：两遍法骨架移到 fterm/backtest/two_pass.py
    # （原实现与本仓另 5 个 chain 脚本逐行重复 ⇒ 风控口径有静默分叉风险；
    #  peak 原为每调仓日重算前缀最大值，现用 cummax O(1) 查表）
    r, n_rej, t2 = run_two_pass(
        engine=eng, reb_days=reb_all, cfg=cfg_, raw_close=raw_close,
        initial_equity=cfg_.account.initial_equity, start_ts=start_ts,
        targets_for=lambda d, e, p: targets_for(d, e, p))
    m = dict(r.metrics)
    m["n_governor_rejected"] = n_rej
    m["n_targets"] = int(sum(len(v) for v in t2.values()))
    return r.equity, m


def monthly_log_paired(eq_c: pd.Series, eq_b: pd.Series, month_ends) -> dict:
    """C2a'：月末权益快照的月度对数收益差配对 t。"""
    rc, rb = [], []
    prev = None
    for f in month_ends:
        if f not in eq_c.index or f not in eq_b.index:
            continue
        if prev is not None and eq_c[prev] > 0 and eq_b[prev] > 0:
            rc.append(float(np.log(eq_c[f] / eq_c[prev])))
            rb.append(float(np.log(eq_b[f] / eq_b[prev])))
        prev = f
    rc, rb = np.array(rc), np.array(rb)
    diff = rc - rb
    if len(diff) < 3 or diff.std(ddof=1) == 0:
        return {"n": int(len(diff)), "mean": float(diff.mean()) if len(diff) else float("nan"),
                "t": float("nan")}
    t = float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))))
    return {"n": int(len(diff)), "mean_log": float(diff.mean()), "t": t}


def main():
    cfg = load_config(ROOT / "config.yaml")
    REPORTS.mkdir(parents=True, exist_ok=True)
    chain1 = json.loads(CHAIN1_JSON.read_text(encoding="utf-8"))
    start_ts = pd.Timestamp(START_OOS)

    print("== 面板/特征 + S1 基线（同窗口重跑）==", flush=True)
    panels = sa.load_panels(cfg)
    bundle = build_dataset(cfg)
    dates = panels["close"].index
    from ablation_chain import run_rule_arm
    eq_s1, m_s1 = run_rule_arm("S1", cfg, bundle, panels, dates, start_ts)
    print(f"  S1: {m_s1['final_equity']:.0f} ({m_s1['total_return']:+.2%}) "
          f"MDD {m_s1['max_drawdown']:+.2%} 成本 {m_s1['total_costs']:.0f}", flush=True)

    results = {}
    for tag, path in PREDS.items():
        pr = pd.read_parquet(path)
        pr["date"] = pd.to_datetime(pr["date"])
        panel = pr.pivot_table(index="date", columns="product", values="pred")
        t0 = time.time()
        eq, m = run_monthly_arm(panel, cfg, bundle, panels, dates, start_ts)
        print(f"  L_monthly_{tag}: {m['final_equity']:.0f} ({m['total_return']:+.2%}) "
              f"MDD {m['max_drawdown']:+.2%} 成本 {m['total_costs']:.0f} "
              f"目标 {m['n_targets']}  ({time.time()-t0:.0f}s)", flush=True)
        s = pd.Series(dates, index=dates)
        month_ends = [d for d in s.groupby([dates.year, dates.month]).last() if d >= start_ts]
        p = monthly_log_paired(eq, eq_s1, month_ends)
        c2a = bool(p["t"] >= C2A_T and p["mean_log"] > 0)
        c2b = bool(m["total_return"] > m_s1["total_return"])
        c3 = bool(m["max_drawdown"] >= m_s1["max_drawdown"] - C3_TOL)
        c4 = bool(m["total_costs"] <= m_s1["total_costs"] * C4_RATIO)
        results[tag] = {
            "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                        for k, v in m.items()},
            "C2a": {**p, "pass": c2a},
            "C2b": {"total_return": m["total_return"], "base": m_s1["total_return"], "pass": c2b},
            "C3": {"mdd": m["max_drawdown"], "base": m_s1["max_drawdown"], "pass": c3},
            "C4": {"costs": m["total_costs"], "base": m_s1["total_costs"], "pass": c4},
            "adoption_ready": bool(c2a and c2b and c3 and c4)}
    dual = bool(all(r["adoption_ready"] for r in results.values()) and len(results) == 2)

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    verdict = {
        "prereg": "plans/monthly-prereg-20261004.md",
        "oos_window": {"start": START_OOS, "end": chain1["oos_window"]["end"]},
        "frozen": {"C2A_T": C2A_T, "C3_TOL": C3_TOL, "C4_RATIO": C4_RATIO,
                   "TOP_N": TOP_N, "KEEP_N": KEEP_N},
        "baseline": {"name": "S1@2019", **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                           for k, v in m_s1.items()}},
        "results": results, "dual_seed_agree": dual,
        "verdict_word": "",   # 待用户填写
    }
    (REPORTS / f"monthly_chain_{stamp}.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [f"判据链 #5 读数 {stamp}（判据冻结于 plans/monthly-prereg-20261004.md）",
             f"基线 S1@2019: {m_s1['total_return']:+.2%} / MDD {m_s1['max_drawdown']:+.2%} / 成本 {m_s1['total_costs']:.0f}"]
    for tag, r in results.items():
        lines.append(f"L_monthly_{tag}: ret {r['metrics']['total_return']:+.2%} "
                     f"MDD {r['metrics']['max_drawdown']:+.2%} 成本 {r['metrics']['total_costs']:.0f} "
                     f"C2a t={r['C2a']['t']:+.2f}(n={r['C2a']['n']}月) "
                     f"C2b={r['C2b']['pass']} C3={r['C3']['pass']} C4={r['C4']['pass']} "
                     f"→ adoption_ready={r['adoption_ready']}")
    lines.append(f"dual_seed_agree = {dual}")
    lines.append("判决词：＿＿＿（由用户填写；通过 ≠ 采纳——带宽确认见链 #6）")
    text = "\n".join(lines)
    (REPORTS / f"monthly_chain_{stamp}.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f"报告 → data/reports/monthly_chain_{stamp}.json/.txt", flush=True)


if __name__ == "__main__":
    main()
