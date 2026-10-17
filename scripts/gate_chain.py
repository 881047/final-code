"""判据链 #3 · Phase 2 门限经济学执行器 —— 判据冻结于 plans/gate-prereg-20261004.md。

三变体（B_g55/B_g60/B_top1）× 双种子集，OOS 预测流直接用判据链 #1 存档
（零训练），两遍法组合层，配对基 = 判据链 #2 的 S*（S1）。判决词由用户填写。
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
from ablation_chain import run_rule_arm, cfg_variant, CHAIN1_JSON  # noqa: E402
from judgment_chain import build_bars, paired_weekly_t  # noqa: E402

REPORTS = Path(__file__).resolve().parents[1] / "data" / "reports"
PREDS_DIR = ROOT / "experiments" / "judgment_chain_20261003"
SEED_SETS = [(7, 21, 99), (1, 2, 3)]

# ---- 冻结（预注册 §一/§二） ----
VARIANTS = {"B_g55": 0.55, "B_g60": 0.60, "B_top1": 0.50}
C2A_T = 2.0
C3_TOL = 0.02
C4_RATIO = 1.25


def run_model_arm(gate: float, top_n: int, max_symbols: int, cfg0, p_up_panel,
                  bundle, panels, dates, start_ts):
    from dataclasses import replace
    cfg_ = replace(cfg0, account=replace(cfg0.account, max_symbols=max_symbols),
                   signal=replace(cfg0.signal, top_n=top_n))
    feats, universe = bundle.features, bundle.universe
    uni_mask = UniverseIndex(universe)
    raw_close = panels["close"]
    keep_n = top_n * cfg0.signal.exit_buffer
    eng = BacktestEngine(cfg_, build_bars(panels))
    reb_all = [d for d in dates if d >= start_ts and d.weekday() == cfg_.signal.rebalance_weekday]

    def targets_for(d, equity, positions):
        if d not in p_up_panel.index:
            return []
        row = p_up_panel.loc[d].dropna()
        # 2026-10-08 B2：向量化（原逐格 universe.at[d, c]；见 fterm/features/universe.py）
        row = row[uni_mask.mask_for(d, row.index)]
        ranked = row[row >= gate].sort_values(ascending=False)
        keep = list(ranked.head(keep_n).index)
        tgts = [Target(c, 0, 0, f"exit_band(keep_{keep_n})") for c in list(positions) if c not in keep]
        for code in ranked.head(top_n).index:
            if code in positions:
                continue
            px = float(raw_close.at[d, code]) if pd.notna(raw_close.at[d, code]) else None
            if px is None:
                continue
            atr = feats["atr14"].at[d, code] if code in feats["atr14"].columns else np.nan
            s = size_position(cfg_, equity, px, float(atr) if np.isfinite(atr) else 0.0)
            if s.lots > 0:
                tgts.append(Target(code, 1, s.lots, f"p={float(ranked[code]):.3f}"))
        return tgts

    # 2026-10-08 去重 + B5：两遍法骨架移到 fterm/backtest/two_pass.py。
    # 原实现与另一条 chain 逐行重复 ⇒ 风控口径有静默分叉风险（改一处忘另一处）；
    # 且 peak 原为每个调仓日重算前缀最大值（单次 O(D)、总量 O(D×R)），
    # 现用 cummax 一次算好、O(1) 查表。等价性见 tests/test_two_pass_equivalence.py。
    r, n_rej, t2 = run_two_pass(
        engine=eng, reb_days=reb_all, targets_for=targets_for, cfg=cfg_,
        raw_close=raw_close, initial_equity=cfg_.account.initial_equity,
        start_ts=start_ts)
    m = dict(r.metrics)
    m["n_governor_rejected"] = n_rej
    m["n_targets"] = int(sum(len(v) for v in t2.values()))
    return r.equity, m


def main():
    cfg = load_config(ROOT / "config.yaml")
    REPORTS.mkdir(parents=True, exist_ok=True)
    chain1 = json.loads(CHAIN1_JSON.read_text(encoding="utf-8"))
    start_ts = pd.Timestamp(chain1["oos_window"]["start"])

    print("== 面板/特征 + S1 基线 ==", flush=True)
    panels = sa.load_panels(cfg)
    bundle = build_dataset(cfg)
    dates = panels["close"].index
    eq_s1, m_s1 = run_rule_arm("S1", cfg, bundle, panels, dates, start_ts)
    print(f"  S1 基线: {m_s1['final_equity']:.0f} ({m_s1['total_return']:+.2%}) "
          f"MDD {m_s1['max_drawdown']:+.2%} 成本 {m_s1['total_costs']:.0f}", flush=True)

    variants = {}
    for ss in SEED_SETS:
        tag = f"ss{'_'.join(map(str, ss))}"
        pr = pd.read_parquet(PREDS_DIR / f"jc_preds_direction_{tag}.parquet")
        pr["date"] = pd.to_datetime(pr["date"])
        panel = pr.pivot_table(index="date", columns="product", values="p_up")
        for vname, gate in VARIANTS.items():
            top_n = 1 if vname == "B_top1" else 2
            t0 = time.time()
            eq, m = run_model_arm(gate, top_n, 1 if vname == "B_top1" else 3,
                                  cfg, panel, bundle, panels, dates, start_ts)
            variants[f"{vname}_{tag}"] = {"metrics": m, "equity": eq}
            print(f"  {vname}_{tag}: {m['final_equity']:.0f} ({m['total_return']:+.2%}) "
                  f"MDD {m['max_drawdown']:+.2%} 成本 {m['total_costs']:.0f} "
                  f"目标 {m['n_targets']}  ({time.time()-t0:.0f}s)", flush=True)

    fridays = [d for d in dates if d >= start_ts
               and d.weekday() == cfg.signal.rebalance_weekday and d in eq_s1.index]

    print("== 判据（冻结常数）==", flush=True)
    results = {}
    for vname in VARIANTS:
        per_ss = {}
        for ss in SEED_SETS:
            key = f"{vname}_ss{'_'.join(map(str, ss))}"
            m, eq = variants[key]["metrics"], variants[key]["equity"]
            p = paired_weekly_t(eq, eq_s1, fridays)
            c2a = bool(p["t"] >= C2A_T and p["mean"] > 0)
            c2b = bool(m["total_return"] > m_s1["total_return"])
            c3 = bool(m["max_drawdown"] >= m_s1["max_drawdown"] - C3_TOL)
            c4 = bool(m["total_costs"] <= m_s1["total_costs"] * C4_RATIO)
            per_ss[f"ss{'_'.join(map(str, ss))}"] = {
                "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                            for k, v in m.items()},
                "C2a": {**p, "pass": c2a}, "C2b": {"total_return": m["total_return"],
                                                   "base": m_s1["total_return"], "pass": c2b},
                "C3": {"mdd": m["max_drawdown"], "base": m_s1["max_drawdown"], "pass": c3},
                "C4": {"costs": m["total_costs"], "base": m_s1["total_costs"], "pass": c4},
                "adoption_ready": bool(c2a and c2b and c3 and c4)}
        results[vname] = {"per_seed_set": per_ss,
                          "dual_seed_agree": bool(all(v["adoption_ready"] for v in per_ss.values()))}

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    verdict = {
        "prereg": "plans/gate-prereg-20261004.md",
        "oos_window": chain1["oos_window"],
        "baseline": "S1（判据链 #2 S*）",
        "frozen_constants": {"C2A_T": C2A_T, "C3_TOL": C3_TOL, "C4_RATIO": C4_RATIO,
                             "VARIANTS": VARIANTS, "SEED_SETS": [list(s) for s in SEED_SETS]},
        "baseline_metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                             for k, v in m_s1.items()},
        "results": results,
        "verdict_word": "",   # 待用户填写
    }
    (REPORTS / f"gate_chain_{stamp}.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [f"判据链 #3 读数 {stamp}（判据冻结于 plans/gate-prereg-20261004.md）",
             f"基线 S1: {m_s1['total_return']:+.2%} / MDD {m_s1['max_drawdown']:+.2%} / 成本 {m_s1['total_costs']:.0f}"]
    for vname, res in results.items():
        lines.append(f"{vname}: dual_seed_agree={res['dual_seed_agree']}")
        for ss, r in res["per_seed_set"].items():
            lines.append(f"  {ss}: ret {r['metrics']['total_return']:+.2%} "
                         f"MDD {r['metrics']['max_drawdown']:+.2%} "
                         f"成本 {r['metrics']['total_costs']:.0f} "
                         f"C2a t={r['C2a']['t']:+.2f} C2b={r['C2b']['pass']} "
                         f"C3={r['C3']['pass']} C4={r['C4']['pass']} "
                         f"→ adoption_ready={r['adoption_ready']}")
    lines.append("判决词：＿＿＿（由用户填写）")
    text = "\n".join(lines)
    (REPORTS / f"gate_chain_{stamp}.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f"报告 → data/reports/gate_chain_{stamp}.json/.txt", flush=True)


if __name__ == "__main__":
    main()
