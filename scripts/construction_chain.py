"""判据链 #9 · 构型重设计执行器 —— 判据冻结于 plans/construction-prereg-20261004.md。

4 预声明构型（K_hold/K_div4/K_biweek/K_combo）× 6 条已缓存 OOS 预测流
（LightGBM 4 套 + DL-v2 2 套），两遍法组合回测，锚 K0 引用 #4/§22 存档读数。
判据：per-model adoption_ready（C4+C3+C2b+C2a' 月度对数配对）× 跨模型泛化。
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
from judgment_chain import build_bars, paired_weekly_t  # noqa: E402
from monthly_chain import monthly_log_paired  # noqa: E402

REPORTS = Path(__file__).resolve().parents[1] / "data" / "reports"
START_OOS = "2019-01-01"

# ---- 冻结（预注册 §二/§三） ----
VARIANTS = ("K_hold", "K_div4", "K_biweek", "K_combo")
TOP_N, KEEP_N, HOLD_BAND = 2, 4, 6           # K0 口径 + K_hold 保留带
DIV_TOP, DIV_KEEP = 4, 8                     # K_div4 口径
BIWEEK_F = 10                                # K_biweek/K_combo cadence
C4_RATIO, C3_TOL, C2A_T = 1.25, 0.02, 2.0
STREAMS = {
    "L": [f"data/reports/lgbm_preds_ss{s}.parquet" for s in
          ("42_7", "123_2024", "5_13", "2027_314")],
    "D": [f"data/reports/dlv2_ss{s}.parquet" for s in
          ("7_21_99_5_13", "123_2024_42_314_2027")],
}


def run_variant(variant: str, p_up_panel, cfg0, bundle, panels, dates, start_ts):
    from dataclasses import replace
    top_n = DIV_TOP if variant == "K_div4" else TOP_N
    keep_n = DIV_KEEP if variant == "K_div4" else KEEP_N
    band = HOLD_BAND if variant in ("K_hold", "K_combo") else keep_n
    max_symbols = 4 if variant == "K_div4" else 3
    cfg_ = replace(cfg0, account=replace(cfg0.account, max_symbols=max_symbols),
                   signal=replace(cfg0.signal, top_n=top_n))
    feats, universe = bundle.features, bundle.universe
    uni_mask = UniverseIndex(universe)
    raw_close = panels["close"]
    eng = BacktestEngine(cfg_, build_bars(panels))
    oos_days = [d for d in dates if d >= start_ts]
    if variant in ("K_biweek", "K_combo"):
        reb_all = set(oos_days[::BIWEEK_F])
    else:
        reb_all = {d for d in oos_days if d.weekday() == cfg_.signal.rebalance_weekday}

    def targets_for(d, equity, positions):
        if d not in p_up_panel.index:
            return []
        row = p_up_panel.loc[d].dropna()
        # 2026-10-08 B2：向量化（原逐格 universe.at[d, c]；见 fterm/features/universe.py）
        row = row[uni_mask.mask_for(d, row.index)]
        ranked = row.sort_values(ascending=False)
        rank_of = {c: i + 1 for i, c in enumerate(ranked.index)}
        # 退出：held 票秩超出保留带（K_hold/K_combo 用加宽带）
        tgts = [Target(c, 0, 0, f"exit(band_{band})") for c in list(positions)
                if rank_of.get(c, 10**9) > band]
        # 入场：rank 前 top_n 且未持有（保留带内的 held 不占新入场名额）
        for code in ranked.head(top_n).index:
            if code in positions:
                continue
            px = float(raw_close.at[d, code]) if pd.notna(raw_close.at[d, code]) else None
            if px is None:
                continue
            atr = feats["atr14"].at[d, code] if code in feats["atr14"].columns else np.nan
            s = size_position(cfg_, equity, px, float(atr) if np.isfinite(atr) else 0.0)
            if s.lots > 0:
                tgts.append(Target(code, 1, s.lots, f"{variant}(rank={rank_of[code]})"))
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


def main():
    cfg = load_config(ROOT / "config.yaml")
    REPORTS.mkdir(parents=True, exist_ok=True)
    chain1 = json.loads(CHAIN1_JSON.read_text(encoding="utf-8"))
    start_ts = pd.Timestamp(START_OOS)

    print("== 面板/特征 + S1 基线 ==", flush=True)
    panels = sa.load_panels(cfg)
    bundle = build_dataset(cfg)
    dates = panels["close"].index
    eq_s1, m_s1 = run_rule_arm("S1", cfg, bundle, panels, dates, start_ts)
    print(f"  S1@2019: {m_s1['total_return']:+.2%} / MDD {m_s1['max_drawdown']:+.2%} / "
          f"成本 {m_s1['total_costs']:.0f}", flush=True)
    s = pd.Series(dates, index=dates)
    month_ends = [d for d in s.groupby([dates.year, dates.month]).last() if d >= start_ts]

    results = {}
    for model, paths in STREAMS.items():
        for path in paths:
            tag = Path(path).stem.replace("lgbm_preds_", "L_").replace("dlv2_ss", "D_")
            pr = pd.read_parquet(ROOT / path)
            pr["date"] = pd.to_datetime(pr["date"])
            panel = pr.pivot_table(index="date", columns="product", values="pred")
            for v in VARIANTS:
                key = f"{model}:{tag}:{v}"
                t0 = time.time()
                eq, m = run_variant(v, panel, cfg, bundle, panels, dates, start_ts)
                p_m = monthly_log_paired(eq, eq_s1, month_ends)
                c4 = bool(m["total_costs"] <= m_s1["total_costs"] * C4_RATIO)
                c3 = bool(m["max_drawdown"] >= m_s1["max_drawdown"] - C3_TOL)
                c2b = bool(m["total_return"] > m_s1["total_return"])
                c2a = bool(p_m["t"] >= C2A_T and p_m["mean_log"] > 0)
                results[key] = {
                    "metrics": {k: (float(x) if isinstance(x, (int, float, np.floating)) else x)
                                for k, x in m.items()},
                    "C2a": {**p_m, "pass": c2a}, "C2b": c2b, "C3": c3, "C4": c4,
                    "pass_all": bool(c4 and c3 and c2b and c2a)}
                print(f"  {key}: {m['total_return']:+.2%} MDD {m['max_drawdown']:+.2%} "
                      f"成本 {m['total_costs']:.0f} 目标 {m['n_targets']} "
                      f"C2a' t={p_m['t']:+.2f} → {'PASS' if results[key]['pass_all'] else 'fail'} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    # ---- 判据汇总（冻结规则） ----
    per_model = {}
    for model in STREAMS:
        keys = [k for k in results if k.startswith(model + ":")]
        per_model[model] = {v: bool(all(results[k]["pass_all"] for k in keys if k.endswith(":" + v)))
                            for v in VARIANTS}
    cross_model = {v: bool(per_model["L"][v] and per_model["D"][v]) for v in VARIANTS}
    passing = [v for v in VARIANTS if cross_model[v]]
    selected = None
    if passing:
        def mean_ret(v):
            vals = [results[k]["metrics"]["total_return"] for k in results if k.endswith(":" + v)]
            return float(np.mean(vals))
        selected = max(passing, key=mean_ret)

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    verdict = {
        "prereg": "plans/construction-prereg-20261004.md",
        "oos_window": {"start": START_OOS, "end": chain1["oos_window"]["end"]},
        "frozen": {"VARIANTS": list(VARIANTS), "TOP_N": TOP_N, "KEEP_N": KEEP_N,
                   "HOLD_BAND": HOLD_BAND, "DIV": [DIV_TOP, DIV_KEEP], "BIWEEK_F": BIWEEK_F,
                   "C4_RATIO": C4_RATIO, "C3_TOL": C3_TOL, "C2A_T": C2A_T},
        "baseline": {"name": "S1@2019", **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                           for k, v in m_s1.items()}},
        "results": results,
        "per_model_adoption": per_model,
        "cross_model": cross_model,
        "passing_variants": passing,
        "selected": selected,
        "selection_rule": "跨模型成立者中取两模型全部种子集总收益均值最高",
        "verdict_word": "",   # 待用户填写
    }
    (REPORTS / f"construction_chain_{stamp}.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [f"判据链 #9 读数 {stamp}（判据冻结于 plans/construction-prereg-20261004.md）",
             f"基线 S1@2019: {m_s1['total_return']:+.2%} / MDD {m_s1['max_drawdown']:+.2%} / 成本 {m_s1['total_costs']:.0f}"]
    for v in VARIANTS:
        lines.append(f"{v}: per-model L={per_model['L'][v]} D={per_model['D'][v]} "
                     f"cross={cross_model[v]}")
        for k in sorted(results):
            if k.endswith(":" + v):
                r = results[k]
                lines.append(f"   {k}: {r['metrics']['total_return']:+.2%} "
                             f"MDD {r['metrics']['max_drawdown']:+.2%} "
                             f"成本 {r['metrics']['total_costs']:.0f} "
                             f"C2a' t={r['C2a']['t']:+.2f}")
    lines.append(f"跨模型成立臂: {passing or '∅'}；选择规则选中: {selected or '∅'}")
    if not passing:
        lines.append("⇒ 构型族在现有约束下判负（预注册 §三）：候选维持 halt，瓶颈移交本金档位另立预注册")
    lines.append("判决词：＿＿＿（由用户填写；过线构型进生产前必须前瞻纸面 ≥60 交易日）")
    text = "\n".join(lines)
    (REPORTS / f"construction_chain_{stamp}.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f"报告 → data/reports/construction_chain_{stamp}.json/.txt", flush=True)


if __name__ == "__main__":
    main()
