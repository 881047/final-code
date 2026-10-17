"""判据链 #12 · D 定制构型执行器 —— 判据冻结于 plans/dlv2-hurdle-prereg-20261005.md。

C_ema + 换仓门槛 H（no-trade region 离散实现），D 专属双种子集；X0=H0 锚
（须与 #11 D:C_ema 逐位一致——确定性自证）；X1=H5、X2=H10 @T1；X3=H10 @T3。
基线 S1_T1（引用 #10/#11 存档）+ S1_T3 同构重跑。判决词由用户填写。
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
EMA_ALPHA = 0.5
T1 = dict(equity=200_000, top_n=10, keep_n=20, max_symbols=10, max_lots=10)
T3 = dict(equity=1_000_000, top_n=10, keep_n=20, max_symbols=10, max_lots=50)
C4_RATIO, C3_TOL, C2A_T = 1.25, 0.02, 2.0
D_STREAMS = [f"data/reports/dlv2_ss{s}.parquet" for s in
             ("7_21_99_5_13", "123_2024_42_314_2027")]
S1_T1_BASE = {"total_return": -0.3727, "max_drawdown": -0.5920, "total_costs": 27210.0}


def run_cema(hurdle: int, tier: dict, p_up_panel, cfg0, bundle, panels, dates, start_ts):
    cfg_ = replace(cfg0,
                   account=replace(cfg0.account, initial_equity=tier["equity"],
                                   max_symbols=tier["max_symbols"],
                                   max_lots_per_symbol=tier["max_lots"]),
                   signal=replace(cfg0.signal, top_n=tier["top_n"]))
    feats, universe = bundle.features, bundle.universe
    uni_mask = UniverseIndex(universe)
    raw_close = panels["close"]
    eng = BacktestEngine(cfg_, build_bars(panels))
    reb_all = sorted({d for d in dates if d >= start_ts
                      and d.weekday() == cfg_.signal.rebalance_weekday})
    smooth: dict[str, float] = {}

    def targets_for(d, equity, positions):
        if d not in p_up_panel.index:
            return []
        row = p_up_panel.loc[d].dropna()
        # 2026-10-08 B2：向量化（原逐格 universe.at[d, c]；见 fterm/features/universe.py）
        row = row[uni_mask.mask_for(d, row.index)]
        pct = row.rank(pct=True)
        for c in pct.index:
            v = float(pct[c])
            smooth[c] = v if c not in smooth else EMA_ALPHA * v + (1 - EMA_ALPHA) * smooth[c]
        ranked = pd.Series(smooth).reindex(pct.index).dropna().sort_values(ascending=False)
        rank_of = {c: i + 1 for i, c in enumerate(ranked.index)}
        keep = set(ranked.head(tier["keep_n"]).index)
        tgts = [Target(c, 0, 0, f"exit_band(keep_{tier['keep_n']})")
                for c in list(positions) if c not in keep]
        for code in ranked.head(tier["top_n"]).index:
            if code in positions:
                if hurdle > 0 and rank_of[code] > tier["top_n"]:
                    # 持仓已滑出 top_n：仅当接替者秩优势 ≥ H 才换（no-trade region）
                    better = [c for c in ranked.head(tier["top_n"]).index
                              if c not in positions
                              and rank_of[c] <= rank_of[code] - hurdle]
                    if not better:
                        continue        # no-trade：保留持仓
                    tgts.append(Target(code, 0, 0, f"hurdle_replace(H={hurdle})"))
                    for c in better[:1]:
                        px = float(raw_close.at[d, c]) if pd.notna(raw_close.at[d, c]) else None
                        if px is None:
                            continue
                        atr = feats["atr14"].at[d, c] if c in feats["atr14"].columns else np.nan
                        s2 = size_position(cfg_, equity, px, float(atr) if np.isfinite(atr) else 0.0)
                        if s2.lots > 0:
                            tgts.append(Target(c, 1, s2.lots, f"hurdle_in(H={hurdle})"))
                continue
            px = float(raw_close.at[d, code]) if pd.notna(raw_close.at[d, code]) else None
            if px is None:
                continue
            atr = feats["atr14"].at[d, code] if code in feats["atr14"].columns else np.nan
            s = size_position(cfg_, equity, px, float(atr) if np.isfinite(atr) else 0.0)
            if s.lots > 0:
                tgts.append(Target(code, 1, s.lots, f"H{hurdle}(rank={rank_of[code]})"))
        return tgts

    # 2026-10-08 去重 + B5：两遍法骨架移到 fterm/backtest/two_pass.py
    # （原实现与本仓另 5 个 chain 脚本逐行重复 ⇒ 风控口径有静默分叉风险；
    #  peak 原为每调仓日重算前缀最大值，现用 cummax O(1) 查表）
    r, n_rej, t2 = run_two_pass(
        engine=eng, reb_days=reb_all, cfg=cfg_, raw_close=raw_close,
        initial_equity=tier["equity"], start_ts=start_ts,
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

    print("== 面板/特征 + S1_T3 基线 ==", flush=True)
    panels = sa.load_panels(cfg)
    bundle = build_dataset(cfg)
    dates = panels["close"].index
    from capital_chain import S1_TIER, _rule_tier_arm
    eq_s1t3, m_s1t3 = _rule_tier_arm(
        replace(cfg, account=replace(cfg.account, initial_equity=T3["equity"],
                                     max_symbols=T3["max_symbols"],
                                     max_lots_per_symbol=T3["max_lots"]),
                signal=replace(cfg.signal, top_n=S1_TIER["top_n"])),
        S1_TIER, bundle, panels, dates, start_ts)
    print(f"  S1_T3: {m_s1t3['total_return']:+.2%} / MDD {m_s1t3['max_drawdown']:+.2%} / "
          f"成本 {m_s1t3['total_costs']:.0f}", flush=True)
    s = pd.Series(dates, index=dates)
    month_ends = [d for d in s.groupby([dates.year, dates.month]).last() if d >= start_ts]

    arms = [("X0", 0, T1), ("X1", 5, T1), ("X2", 10, T1), ("X3", 10, T3)]
    results = {}
    for arm, hurdle, tier in arms:
        bl = S1_T1_BASE if tier is T1 else {
            "total_return": m_s1t3["total_return"], "max_drawdown": m_s1t3["max_drawdown"],
            "total_costs": m_s1t3["total_costs"]}
        for path in D_STREAMS:
            tag = Path(path).stem.replace("dlv2_ss", "D_")
            pr = pd.read_parquet(ROOT / path)
            pr["date"] = pd.to_datetime(pr["date"])
            panel = pr.pivot_table(index="date", columns="product", values="pred")
            key = f"{arm}:{tag}"
            t0 = time.time()
            eq, m = run_cema(hurdle, tier, panel, cfg, bundle, panels, dates, start_ts)
            p_m = monthly_log_paired(eq, eq_s1t3 if tier is T3 else eq_s1t3, month_ends)
            # X0-X2 基线为 S1_T1：配对需同档——S1_T1 权益曲线在 #10/#11 未存盘，
            # 此处用 #11 的同口径读数做基线校验 + 月度配对改对 S1_T1 重跑一次
            c4 = bool(m["total_costs"] <= bl["total_costs"] * C4_RATIO)
            c3 = bool(m["max_drawdown"] >= bl["max_drawdown"] - C3_TOL)
            c2b = bool(m["total_return"] > bl["total_return"])
            c2a = bool(p_m["t"] >= C2A_T and p_m["mean_log"] > 0) if tier is T3 else None
            results[key] = {
                "metrics": {k: (float(x) if isinstance(x, (int, float, np.floating)) else x)
                            for k, x in m.items()},
                "C2a_T3": {**p_m, "pass": c2a} if tier is T3 else None,
                "C2b": c2b, "C3": c3, "C4": c4,
                "pass_all_T3": bool(c4 and c3 and c2b and c2a) if tier is T3 else None}
            print(f"  {key}: {m['total_return']:+.2%} MDD {m['max_drawdown']:+.2%} "
                  f"成本 {m['total_costs']:.0f} 目标 {m['n_targets']} ({time.time()-t0:.0f}s)",
                  flush=True)

    # X0 确定性自证：与 #11 D:C_ema 读数逐位比对
    chain11 = json.loads((ROOT / "experiments/construction_v2_20261005/construction_v2_20261004_2345.json")
                         .read_text(encoding="utf-8"))
    ref = {k.split(":")[1]: v["metrics"] for k, v in chain11["results"].items()
           if k.endswith(":C_ema")}
    x0_ok = True
    for tag, r in results.items():
        if not tag.startswith("X0"):
            continue
        refm = ref.get("D_" + tag.split(":")[1])
        same = refm and all(abs(refm[kk] - r["metrics"][kk]) < 1e-6
                            for kk in ("total_return", "max_drawdown", "total_costs"))
        print(f"  自证 {tag}: {'逐位一致 ✓' if same else '漂移 ✗'}", flush=True)
        x0_ok = x0_ok and bool(same)

    # T1 臂（X1/X2）的 C2a''：月度对数配对 vs S1_T1——需 S1_T1 权益曲线，此处重跑
    from capital_chain import _rule_tier_arm
    eq_s1t1, m_s1t1 = _rule_tier_arm(
        replace(cfg, account=replace(cfg.account, initial_equity=T1["equity"],
                                     max_symbols=T1["max_symbols"],
                                     max_lots_per_symbol=T1["max_lots"]),
                signal=replace(cfg.signal, top_n=T1["top_n"])),
        T1, bundle, panels, dates, start_ts)
    s = pd.Series(dates, index=dates)
    month_ends = [d for d in s.groupby([dates.year, dates.month]).last() if d >= start_ts]
    print(f"  S1_T1 重跑: {m_s1t1['total_return']:+.2%}（存档 −37.27%）", flush=True)
    for arm, hurdle, tier in arms:
        if tier is not T1:
            continue
        for path in D_STREAMS:
            tag = Path(path).stem.replace("dlv2_ss", "D_")
            key = f"{arm}:{tag}"
            if key not in results:
                continue
            # 重跑该臂以取权益曲线（等价于上方回测，缓存确定性）
            pr = pd.read_parquet(ROOT / path)
            pr["date"] = pd.to_datetime(pr["date"])
            panel = pr.pivot_table(index="date", columns="product", values="pred")
            eq, m = run_cema(hurdle, tier, panel, cfg, bundle, panels, dates, start_ts)
            p_m = monthly_log_paired(eq, eq_s1t1, month_ends)
            c2a = bool(p_m["t"] >= C2A_T and p_m["mean_log"] > 0)
            c2b = bool(m["total_return"] > m_s1t1["total_return"])
            c3 = bool(m["max_drawdown"] >= m_s1t1["max_drawdown"] - C3_TOL)
            c4 = bool(m["total_costs"] <= m_s1t1["total_costs"] * C4_RATIO)
            results[key].update({"C2a_T1": {**p_m, "pass": c2a}, "C2b_T1": c2b,
                                 "C3_T1": c3, "C4_T1": c4,
                                 "pass_all_T1": bool(c2a and c2b and c3 and c4)})

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    verdict = {
        "prereg": "plans/dlv2-hurdle-prereg-20261005.md",
        "oos_window": {"start": START_OOS, "end": chain1["oos_window"]["end"]},
        "frozen": {"EMA_ALPHA": EMA_ALPHA, "HURDLES": {"X0": 0, "X1": 5, "X2": 10, "X3": 10},
                   "TIERS": {"T1": T1, "T3": T3}, "C4_RATIO": C4_RATIO,
                   "C3_TOL": C3_TOL, "C2A_T": C2A_T},
        "x0_determinism_ok": x0_ok,
        "baselines": {"S1_T3": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                for k, v in m_s1t3.items()},
                      "S1_T1_rerun": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                      for k, v in m_s1t1.items()},
                      "S1_T1_archived": S1_T1_BASE},
        "results": results,
        "verdict_word": "",   # 待用户填写
    }
    (REPORTS / f"dlv2_hurdle_{stamp}.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [f"判据链 #12 读数 {stamp}（判据冻结于 plans/dlv2-hurdle-prereg-20261005.md）",
             f"X0 确定性自证: {'✓ 逐位一致' if x0_ok else '✗ 漂移（全链判废）'}",
             f"基线 S1_T3: {m_s1t3['total_return']:+.2%} / MDD {m_s1t3['max_drawdown']:+.2%} / "
             f"成本 {m_s1t3['total_costs']:.0f}   S1_T1: {m_s1t1['total_return']:+.2%} / "
             f"MDD {m_s1t1['max_drawdown']:+.2%} / 成本 {m_s1t1['total_costs']:.0f}"]
    for arm, hurdle, tier in arms:
        tname = "T1" if tier is T1 else "T3"
        lines.append(f"{arm} (H={hurdle}, {tname}):")
        for k in sorted(results):
            if k.startswith(arm + ":"):
                r = results[k]
                line = f"   {k}: {r['metrics']['total_return']:+.2%} " \
                       f"MDD {r['metrics']['max_drawdown']:+.2%} 成本 {r['metrics']['total_costs']:.0f}"
                if tier is T3:
                    line += f"  C2a'' t={r['C2a_T3']['t']:+.2f} C2b={r['C2b']} " \
                            f"C3={r['C3']} C4={r['C4']} → {r['pass_all_T3']}"
                else:
                    line += f"  C2a'' t={r['C2a_T1']['t']:+.2f} C2b={r['C2b_T1']} " \
                            f"C3={r['C3_T1']} C4={r['C4_T1']} → {r['pass_all_T1']}"
                lines.append(line)
    lines.append("判决词：＿＿＿（由用户填写；过线≠采纳：前瞻纸面 ≥60 交易日 + 入金拍板在前）")
    text = "\n".join(lines)
    (REPORTS / f"dlv2_hurdle_{stamp}.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f"报告 → data/reports/dlv2_hurdle_{stamp}.json/.txt", flush=True)


if __name__ == "__main__":
    main()
