"""判据链 #11 · 构型 v2（文献方法）执行器 —— 判据冻结于 plans/construction-v2-prereg-20261005.md。

C_ema（预测秩 EMA α=0.5）/ C_half（半仓入场渐进）/ C_ema_half（叠加），
平台 = T1 档（20万/top10/keep20），基线 = S1_T1（复用 #10 存档判据与数值）。
零训练。判决词由用户填写。
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
from ablation_chain import CHAIN1_JSON  # noqa: E402
from judgment_chain import build_bars  # noqa: E402
from monthly_chain import monthly_log_paired  # noqa: E402
from capital_chain import _rule_tier_arm, S1_TIER  # noqa: E402

REPORTS = Path(__file__).resolve().parents[1] / "data" / "reports"
START_OOS = "2019-01-01"
T1 = dict(equity=200_000, top_n=10, keep_n=20, max_symbols=10, max_lots=10)
EMA_ALPHA = 0.5                                   # 冻结（预注册 §一）
C4_RATIO, C3_TOL, C2A_T = 1.25, 0.02, 2.0
BASE = {"ret": -0.3727, "mdd": -0.5920, "costs": 27210}   # S1_T1（#10 存档）
STREAMS = {
    "L": [f"data/reports/lgbm_preds_ss{s}.parquet" for s in
          ("42_7", "123_2024", "5_13", "2027_314")],
    "D": [f"data/reports/dlv2_ss{s}.parquet" for s in
          ("7_21_99_5_13", "123_2024_42_314_2027")],
}


def run_v2(variant: str, p_up_panel, cfg0, bundle, panels, dates, start_ts):
    cfg_ = replace(cfg0,
                   account=replace(cfg0.account, initial_equity=T1["equity"],
                                   max_symbols=T1["max_symbols"],
                                   max_lots_per_symbol=T1["max_lots"]),
                   signal=replace(cfg0.signal, top_n=T1["top_n"]))
    feats, universe = bundle.features, bundle.universe
    uni_mask = UniverseIndex(universe)
    raw_close = panels["close"]
    eng = BacktestEngine(cfg_, build_bars(panels))
    reb_all = sorted({d for d in dates if d >= start_ts
                      and d.weekday() == cfg_.signal.rebalance_weekday})
    # C_ema：每股票预测百分位在调仓日序列上的 EMA（冻结 α=0.5，PIT：只用历史调仓日）
    smooth = {}                                   # code → ema 值
    use_ema = variant in ("C_ema", "C_ema_half")
    half_entry = variant in ("C_half", "C_ema_half")
    half_flags: dict[str, bool] = {}              # code → True=只建了半仓

    def targets_for(d, equity, positions):
        if d not in p_up_panel.index:
            return []
        row = p_up_panel.loc[d].dropna()
        # 2026-10-08 B2：向量化（原逐格 universe.at[d, c]；见 fterm/features/universe.py）
        row = row[uni_mask.mask_for(d, row.index)]
        pct = row.rank(pct=True)
        if use_ema:
            for c in pct.index:
                v = float(pct[c])
                smooth[c] = v if c not in smooth else EMA_ALPHA * v + (1 - EMA_ALPHA) * smooth[c]
            ranked = pd.Series(smooth).reindex(pct.index).dropna().sort_values(ascending=False)
        else:
            ranked = row.sort_values(ascending=False)
        keep = set(ranked.head(T1["keep_n"]).index)
        tgts = [Target(c, 0, 0, "exit_band(keep_20)") for c in list(positions) if c not in keep]
        n_full = sum(1 for c in positions if not half_flags.get(c))
        for code in ranked.head(T1["top_n"]).index:
            if code in positions:
                continue
            px = float(raw_close.at[d, code]) if pd.notna(raw_close.at[d, code]) else None
            if px is None:
                continue
            atr = feats["atr14"].at[d, code] if code in feats["atr14"].columns else np.nan
            s = size_position(cfg_, equity, px, float(atr) if np.isfinite(atr) else 0.0)
            if s.lots <= 0:
                continue
            if half_entry and n_full >= T1["top_n"]:
                lots = max(1, s.lots // 2)
                half_flags[code] = True
            else:
                lots = s.lots
                half_flags[code] = False
            n_full += 0 if half_flags.get(code) else 1
            tgts.append(Target(code, 1, lots, f"{variant}(rank={float(ranked[code]):.3f})"))
        # 半仓票若仍在 top_n，本调仓日补足（部分调整的第二步）
        for code in list(positions):
            if half_flags.get(code) and code in set(ranked.head(T1["top_n"]).index):
                px = float(raw_close.at[d, code])
                atr = feats["atr14"].at[d, code] if code in feats["atr14"].columns else np.nan
                s = size_position(cfg_, equity, px, float(atr) if np.isfinite(atr) else 0.0)
                if s.lots > positions[code]["lots"]:
                    tgts.append(Target(code, 1, s.lots, f"{variant}(topup)"))
                half_flags[code] = False
        return tgts

    # 2026-10-08 去重 + B5：两遍法骨架移到 fterm/backtest/two_pass.py
    # （原实现与本仓另 5 个 chain 脚本逐行重复 ⇒ 风控口径有静默分叉风险；
    #  peak 原为每调仓日重算前缀最大值，现用 cummax O(1) 查表）
    r, n_rej, t2 = run_two_pass(
        engine=eng, reb_days=reb_all, cfg=cfg_, raw_close=raw_close,
        initial_equity=T1["equity"], start_ts=start_ts,
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

    print("== 面板/特征 + S1_T1 基线 ==", flush=True)
    panels = sa.load_panels(cfg)
    bundle = build_dataset(cfg)
    dates = panels["close"].index
    eq_s1, m_s1 = _rule_tier_arm(replace(cfg,
                                         account=replace(cfg.account, initial_equity=T1["equity"],
                                                         max_symbols=T1["max_symbols"],
                                                         max_lots_per_symbol=T1["max_lots"]),
                                         signal=replace(cfg.signal, top_n=T1["top_n"])),
                                 T1, bundle, panels, dates, start_ts)
    print(f"  S1_T1: {m_s1['total_return']:+.2%} / MDD {m_s1['max_drawdown']:+.2%} / "
          f"成本 {m_s1['total_costs']:.0f}（#10 存档 −37.27%/−59.20%/27210）", flush=True)
    s = pd.Series(dates, index=dates)
    month_ends = [d for d in s.groupby([dates.year, dates.month]).last() if d >= start_ts]

    results = {}
    for model, paths in STREAMS.items():
        for path in paths:
            tag = Path(path).stem.replace("lgbm_preds_", "L_").replace("dlv2_ss", "D_")
            pr = pd.read_parquet(ROOT / path)
            pr["date"] = pd.to_datetime(pr["date"])
            panel = pr.pivot_table(index="date", columns="product", values="pred")
            for v in ("C_ema", "C_half", "C_ema_half"):
                key = f"{model}:{tag}:{v}"
                t0 = time.time()
                eq, m = run_v2(v, panel, cfg, bundle, panels, dates, start_ts)
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
                      f"C2a'' t={p_m['t']:+.2f} → {'PASS' if results[key]['pass_all'] else 'fail'} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    per_model = {}
    for model in STREAMS:
        keys = [k for k in results if k.startswith(model + ":")]
        per_model[model] = {v: bool(all(results[k]["pass_all"] for k in keys
                                        if k.endswith(":" + v)))
                            for v in ("C_ema", "C_half", "C_ema_half")}
    cross = {v: bool(per_model["L"][v] and per_model["D"][v])
             for v in ("C_ema", "C_half", "C_ema_half")}
    passing = [v for v in ("C_ema", "C_half", "C_ema_half") if cross[v]]
    selected = None
    if passing:
        selected = max(passing, key=lambda v: float(np.mean(
            [results[k]["metrics"]["total_return"] for k in results if k.endswith(":" + v)])))

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    verdict = {
        "prereg": "plans/construction-v2-prereg-20261005.md",
        "oos_window": {"start": START_OOS, "end": chain1["oos_window"]["end"]},
        "frozen": {"EMA_ALPHA": EMA_ALPHA, "T1": T1, "C4_RATIO": C4_RATIO,
                   "C3_TOL": C3_TOL, "C2A_T": C2A_T},
        "baseline": {"name": "S1_T1", **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                         for k, v in m_s1.items()}},
        "results": results,
        "per_model_adoption": per_model,
        "cross_model": cross,
        "passing_variants": passing,
        "selected": selected,
        "verdict_word": "",   # 待用户填写
    }
    (REPORTS / f"construction_v2_{stamp}.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [f"判据链 #11 读数 {stamp}（判据冻结于 plans/construction-v2-prereg-20261005.md）",
             f"基线 S1_T1: {m_s1['total_return']:+.2%} / MDD {m_s1['max_drawdown']:+.2%} / "
             f"成本 {m_s1['total_costs']:.0f}"]
    for v in ("C_ema", "C_half", "C_ema_half"):
        lines.append(f"{v}: per-model L={per_model['L'][v]} D={per_model['D'][v]} cross={cross[v]}")
        for k in sorted(results):
            if k.endswith(":" + v):
                r = results[k]
                lines.append(f"   {k}: {r['metrics']['total_return']:+.2%} "
                             f"MDD {r['metrics']['max_drawdown']:+.2%} "
                             f"成本 {r['metrics']['total_costs']:.0f} "
                             f"C2a'' t={r['C2a']['t']:+.2f}")
    lines.append(f"跨模型成立臂: {passing or '∅'}；选中: {selected or '∅'}")
    lines.append("判决词：＿＿＿（由用户填写；过线≠采纳：前瞻纸面 ≥60 交易日 + 入金拍板在前）")
    text = "\n".join(lines)
    (REPORTS / f"construction_v2_{stamp}.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f"报告 → data/reports/construction_v2_{stamp}.json/.txt", flush=True)


if __name__ == "__main__":
    main()
