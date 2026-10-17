"""判据链 #10 · 本金档位执行器 —— 判据冻结于 plans/capital-prereg-20261005.md。

T1（20万/top10/keep20）与 T2（20万/top2/keep4 纯放大）× 6 条缓存预测流，
基线 S1@同档重跑，月度对数配对 C2a''。零训练。判决词由用户填写。
"""
import json
import sys
import time
from dataclasses import replace
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
from ablation_chain import CHAIN1_JSON, run_rule_arm  # noqa: E402
from judgment_chain import build_bars  # noqa: E402
from monthly_chain import monthly_log_paired  # noqa: E402

REPORTS = Path(__file__).resolve().parents[1] / "data" / "reports"
START_OOS = "2019-01-01"

# ---- 冻结（预注册 §一/§二） ----
TIERS = {
    "T1": dict(equity=200_000, top_n=10, keep_n=20, max_symbols=10, max_lots=10),
    "T2": dict(equity=200_000, top_n=2, keep_n=4, max_symbols=2, max_lots=60),
}
S1_TIER = dict(equity=200_000, top_n=2, keep_n=4, max_symbols=2, max_lots=60)
C4_RATIO, C3_TOL, C2A_T = 1.25, 0.02, 2.0
STREAMS = {
    "L": [f"data/reports/lgbm_preds_ss{s}.parquet" for s in
          ("42_7", "123_2024", "5_13", "2027_314")],
    "D": [f"data/reports/dlv2_ss{s}.parquet" for s in
          ("7_21_99_5_13", "123_2024_42_314_2027")],
}


def run_tier(tier: dict, p_up_panel, cfg0, bundle, panels, dates, start_ts):
    cfg_ = replace(cfg0,
                   account=replace(cfg0.account, initial_equity=tier["equity"],
                                   max_symbols=tier["max_symbols"],
                                   max_lots_per_symbol=tier["max_lots"]),
                   signal=replace(cfg0.signal, top_n=tier["top_n"]))
    feats, universe = bundle.features, bundle.universe
    uni_mask = UniverseIndex(universe)
    raw_close = panels["close"]
    eng = BacktestEngine(cfg_, build_bars(panels))
    reb_all = {d for d in dates if d >= start_ts and d.weekday() == cfg_.signal.rebalance_weekday}

    def targets_for(d, equity, positions):
        if d not in p_up_panel.index:
            return []
        row = p_up_panel.loc[d].dropna()
        # 2026-10-08 B2：向量化（原逐格 universe.at[d, c]；见 fterm/features/universe.py）
        row = row[uni_mask.mask_for(d, row.index)]
        ranked = row.sort_values(ascending=False)
        keep = set(ranked.head(tier["keep_n"]).index)
        tgts = [Target(c, 0, 0, f"exit_band(keep_{tier['keep_n']})")
                for c in list(positions) if c not in keep]
        for code in ranked.head(tier["top_n"]).index:
            if code in positions:
                continue
            px = float(raw_close.at[d, code]) if pd.notna(raw_close.at[d, code]) else None
            if px is None:
                continue
            atr = feats["atr14"].at[d, code] if code in feats["atr14"].columns else np.nan
            s = size_position(cfg_, equity, px, float(atr) if np.isfinite(atr) else 0.0)
            if s.lots > 0:
                tgts.append(Target(code, 1, s.lots, f"tier(rank={float(ranked[code]):.3f})"))
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
    start_ts = pd.Timestamp(START_OOS)

    print("== 面板/特征 + S1@各档基线 ==", flush=True)
    panels = sa.load_panels(cfg)
    bundle = build_dataset(cfg)
    dates = panels["close"].index
    baselines = {}
    for name, tier in (("S1_T1", TIERS["T1"]), ("S1_T2", S1_TIER)):
        # S1 规则构型（top2）在该档的基线：直接用 run_tier 的锚语义不可复用
        # （S1 是规则臂），这里用规则版 runner：借用 run_rule_arm 的 tier 变体
        from fterm.strategy.primary import rule_reversal_decision
        cfg_t = replace(cfg,
                        account=replace(cfg.account, initial_equity=tier["equity"],
                                        max_symbols=tier["max_symbols"],
                                        max_lots_per_symbol=tier["max_lots"]),
                        signal=replace(cfg.signal, top_n=tier["top_n"]))
        eq, m = _rule_tier_arm(cfg_t, tier, bundle, panels, dates, start_ts)
        baselines[name] = {"equity": eq, "metrics": m}
        print(f"  {name}: {m['total_return']:+.2%} / MDD {m['max_drawdown']:+.2%} / "
              f"成本 {m['total_costs']:.0f}", flush=True)
    s = pd.Series(dates, index=dates)
    month_ends = [d for d in s.groupby([dates.year, dates.month]).last() if d >= start_ts]

    results = {}
    for tier_name, tier in TIERS.items():
        bl = baselines[f"S1_{tier_name}"]
        for model, paths in STREAMS.items():
            for path in paths:
                tag = Path(path).stem.replace("lgbm_preds_", "L_").replace("dlv2_ss", "D_")
                pr = pd.read_parquet(ROOT / path)
                pr["date"] = pd.to_datetime(pr["date"])
                panel = pr.pivot_table(index="date", columns="product", values="pred")
                key = f"{tier_name}:{model}:{tag}"
                t0 = time.time()
                eq, m = run_tier(tier, panel, cfg, bundle, panels, dates, start_ts)
                p_m = monthly_log_paired(eq, bl["equity"], month_ends)
                c4 = bool(m["total_costs"] <= bl["metrics"]["total_costs"] * C4_RATIO)
                c3 = bool(m["max_drawdown"] >= bl["metrics"]["max_drawdown"] - C3_TOL)
                c2b = bool(m["total_return"] > bl["metrics"]["total_return"])
                c2a = bool(p_m["t"] >= C2A_T and p_m["mean_log"] > 0)
                results[key] = {
                    "metrics": {k: (float(x) if isinstance(x, (int, float, np.floating)) else x)
                                for k, x in m.items()},
                    "C2a": {**p_m, "pass": c2a}, "C2b": c2b, "C3": c3, "C4": c4,
                    "pass_all": bool(c4 and c3 and c2b and c2a)}
                print(f"  {key}: {m['total_return']:+.2%} MDD {m['max_drawdown']:+.2%} "
                      f"成本 {m['total_costs']:.0f} 目标 {m['n_targets']} "
                      f"C2a'' t={p_m['t']:+.2f} → {'PASS' if results[key]['pass_all'] else 'fail'} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    per_model = {}
    for tier_name in TIERS:
        for model in STREAMS:
            keys = [k for k in results if k.startswith(f"{tier_name}:{model}:")]
            per_model[f"{tier_name}/{model}"] = bool(
                all(results[k]["pass_all"] for k in keys))
    per_tier = {tn: bool(per_model[f"{tn}/L"] and per_model[f"{tn}/D"]) for tn in TIERS}
    passing = [tn for tn in TIERS if per_tier[tn]]
    selected = None
    if passing:
        def mean_ret(tn):
            vals = [results[k]["metrics"]["total_return"] for k in results
                    if k.startswith(tn + ":")]
            return float(np.mean(vals))
        selected = max(passing, key=mean_ret)

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    verdict = {
        "prereg": "plans/capital-prereg-20261005.md",
        "oos_window": {"start": START_OOS, "end": chain1["oos_window"]["end"]},
        "frozen": {"TIERS": TIERS, "S1_TIER": S1_TIER,
                   "C4_RATIO": C4_RATIO, "C3_TOL": C3_TOL, "C2A_T": C2A_T},
        "baselines": {k: {"metrics": {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv)
                                      for kk, vv in v["metrics"].items()}}
                      for k, v in baselines.items()},
        "results": results,
        "per_model_adoption": per_model,
        "per_tier_cross_model": per_tier,
        "passing_tiers": passing,
        "selected": selected,
        "selection_rule": "过线档中取两模型全部种子集总收益均值最高；过线≠采纳（前瞻纸面+入金拍板在前）",
        "verdict_word": "",   # 待用户填写
    }
    (REPORTS / f"capital_chain_{stamp}.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [f"判据链 #10 读数 {stamp}（判据冻结于 plans/capital-prereg-20261005.md）"]
    for k, v in baselines.items():
        lines.append(f"基线 {k}: {v['metrics']['total_return']:+.2%} / "
                     f"MDD {v['metrics']['max_drawdown']:+.2%} / 成本 {v['metrics']['total_costs']:.0f}")
    for tn in TIERS:
        lines.append(f"{tn}: per-model L={per_model[f'{tn}/L']} D={per_model[f'{tn}/D']} "
                     f"cross={per_tier[tn]}")
        for k in sorted(results):
            if k.startswith(tn + ":"):
                r = results[k]
                lines.append(f"   {k}: {r['metrics']['total_return']:+.2%} "
                             f"MDD {r['metrics']['max_drawdown']:+.2%} "
                             f"成本 {r['metrics']['total_costs']:.0f} "
                             f"C2a'' t={r['C2a']['t']:+.2f}")
    lines.append(f"过线档: {passing or '∅'}；选中: {selected or '∅'}")
    lines.append("判决词：＿＿＿（由用户填写；过线≠采纳：前瞻纸面 ≥60 交易日 + 入金拍板在前）")
    text = "\n".join(lines)
    (REPORTS / f"capital_chain_{stamp}.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f"报告 → data/reports/capital_chain_{stamp}.json/.txt", flush=True)


def _rule_tier_arm(cfg_t, tier, bundle, panels, dates, start_ts):
    """S1 规则构型在某档位的两遍法回测（与 run_tier 同撮合同口径）。"""
    from fterm.strategy.primary import rule_reversal_decision
    feats, universe = bundle.features, bundle.universe
    raw_close = panels["close"]
    eng = BacktestEngine(cfg_t, build_bars(panels))
    pvmed = feats["pvol_20"].rolling(252, min_periods=126).median()
    reb_all = {d for d in dates if d >= start_ts and d.weekday() == cfg_t.signal.rebalance_weekday}

    def targets_for(d, equity, positions):
        score, eligible = rule_reversal_decision(
            feats, universe, d, rev_window=cfg_t.signal.rev_window,
            quiet_vol=cfg_t.signal.quiet_vol, pvol_med=pvmed)
        ranked = score[eligible].sort_values(ascending=False)
        keep = set(ranked.head(tier["keep_n"]).index)
        tgts = [Target(c, 0, 0, f"exit_band(keep_{tier['keep_n']})")
                for c in list(positions) if c not in keep]
        for code in ranked.head(tier["top_n"]).index:
            if code in positions:
                continue
            px = float(raw_close.at[d, code]) if pd.notna(raw_close.at[d, code]) else None
            if px is None:
                continue
            atr = feats["atr14"].at[d, code] if code in feats["atr14"].columns else np.nan
            sz = size_position(cfg_t, equity, px, float(atr) if np.isfinite(atr) else 0.0)
            if sz.lots > 0:
                tgts.append(Target(code, 1, sz.lots, f"S1tier(rank={float(ranked[code]):.3f})"))
        return tgts

    # 2026-10-08 去重 + B5：与 run_tier 同一骨架，收敛到 fterm/backtest/two_pass.py
    # （原实现三处重复；此处无曲线时兜底权益是 tier["equity"]，与 run_tier 的
    #  cfg_.account.initial_equity 等价 —— 那边已把 initial_equity 设成 tier 值）
    r, n_rej, _t2 = run_two_pass(
        engine=eng, reb_days=reb_all, targets_for=targets_for, cfg=cfg_t,
        raw_close=raw_close, initial_equity=tier["equity"], start_ts=start_ts)
    m = dict(r.metrics)
    m["n_governor_rejected"] = n_rej
    return r.equity, m


if __name__ == "__main__":
    main()
