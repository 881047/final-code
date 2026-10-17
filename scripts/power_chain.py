"""判据链 #13 · C2a 功效补齐执行器 —— 判据冻结于 plans/power-prereg-20261005.md。

①训练 +2 套种子集 {101,202}/{303,404}（GPU 全截面，同 #8 折界/架构/口径）；
②四套种子集预测取均值 → 集成面板 → T3 组合层两遍法 vs S1_T3；
③判据：C2a''' 月度对数配对 t≥2 + C2b/C3'/C4'（对集成曲线）；per-set 透明读数。
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

from fterm.config import load_config  # noqa: E402
from fterm.data import ashare as sa  # noqa: E402
from fterm.features.build import build_dataset  # noqa: E402
from fterm.model.validate import spearman_rank_ic  # noqa: E402
from ablation_chain import CHAIN1_JSON, run_rule_arm  # noqa: E402
from dlv2_chain import train_fold_seedset  # noqa: E402
from gate_chain import run_model_arm  # noqa: E402
from monthly_chain import monthly_log_paired  # noqa: E402

REPORTS = Path(__file__).resolve().parents[1] / "data" / "reports"
FOLD_EDGES = ["2019-01-01", "2021-01-01", "2023-01-01", "2025-01-01"]
NEW_SETS = [(101, 202), (303, 404)]
EXISTING = [f"data/reports/dlv2_ss{s}.parquet" for s in
            ("7_21_99_5_13", "123_2024_42_314_2027")]
START_OOS = "2019-01-01"

# ---- 冻结（预注册 §二） ----
T3 = dict(equity=1_000_000, top_n=10, keep_n=20, max_symbols=10, max_lots=50)
C4_RATIO, C3_TOL, C2A_T = 1.25, 0.02, 2.0


def main():
    cfg = load_config(ROOT / "config.yaml")
    REPORTS.mkdir(parents=True, exist_ok=True)
    chain1 = json.loads(CHAIN1_JSON.read_text(encoding="utf-8"))
    start_ts = pd.Timestamp(START_OOS)

    print("== 面板/特征 ==", flush=True)
    panels = sa.load_panels(cfg)
    bundle = build_dataset(cfg)
    dates = panels["close"].index
    universe = bundle.universe
    edges = [pd.Timestamp(x) for x in FOLD_EDGES] + [dates[-1] + pd.Timedelta(days=1)]

    # ---- ①训练 +2 套新种子集（带缓存）----
    all_panels = {}
    for path in EXISTING:
        tag = Path(path).stem.replace("dlv2_ss", "ss")
        df = pd.read_parquet(ROOT / path)
        df["date"] = pd.to_datetime(df["date"])
        all_panels[tag] = df
        print(f"  {tag}: 已有 {len(df)} 条", flush=True)
    for ss in NEW_SETS:
        tag = f"ss{'_'.join(map(str, ss))}"
        cache = REPORTS / f"dlv2_{tag}.parquet"
        if cache.exists():
            df = pd.read_parquet(cache)
            df["date"] = pd.to_datetime(df["date"])
            all_panels[tag] = df
            print(f"  {tag}: 缓存 {len(df)} 条", flush=True)
            continue
        parts = []
        t0 = time.time()
        for k in range(len(edges) - 1):
            d0, d1 = edges[k], edges[k + 1]
            dfk = train_fold_seedset(cfg, bundle, universe, d0, d1, ss, 0)
            dfk["fold"] = k
            parts.append(dfk)
            print(f"    fold{k} 训练完成 ({time.time()-t0:.0f}s)", flush=True)
        df = pd.concat(parts, ignore_index=True)
        df.to_parquet(cache)
        all_panels[tag] = df
        print(f"  {tag}: {len(df)} 条 OOS 预测 ({time.time()-t0:.0f}s)", flush=True)

    # ---- ②四套集成均值面板 ----
    base = None
    for tag, df in all_panels.items():
        p = df.pivot_table(index="date", columns="product", values="pred")
        if base is None:
            base = p.copy()
        else:
            base = base.add(p, fill_value=0)
    mean_panel = base.div(len(all_panels)).astype("float32")
    print(f"  集成面板: {mean_panel.shape}（{len(all_panels)} 套均值）", flush=True)

    # per-set 透明 IC
    per_set_ic = {}
    for tag, df in all_panels.items():
        ics = [spearman_rank_ic(list(zip(g["pred"], g["y_ret"])))
               for _, g in df.groupby("date")]
        ics = [x for x in ics if x is not None]
        per_set_ic[tag] = float(np.mean(ics))
    print(f"  per-set IC: {per_set_ic}", flush=True)

    # ---- ③T3 组合层（集成曲线 vs S1_T3）----
    print("== T3 组合层（4 集成均值 vs S1_T3）==", flush=True)
    from capital_chain import S1_TIER, _rule_tier_arm
    from dataclasses import replace
    cfg_s1 = replace(cfg,
                     account=replace(cfg.account, initial_equity=S1_TIER["equity"],
                                     max_symbols=S1_TIER["max_symbols"],
                                     max_lots_per_symbol=S1_TIER["max_lots"]),
                     signal=replace(cfg.signal, top_n=S1_TIER["top_n"]))
    eq_s1, m_s1 = _rule_tier_arm(cfg_s1, S1_TIER, bundle, panels, dates, start_ts)
    print(f"  S1_T3: {m_s1['total_return']:+.2%} / MDD {m_s1['max_drawdown']:+.2%} / "
          f"成本 {m_s1['total_costs']:.0f}", flush=True)

    eq_mean, m_mean = run_model_arm(-1e9, 10, 10, cfg, mean_panel, bundle, panels,
                                    dates, start_ts)
    print(f"  集成: {m_mean['total_return']:+.2%} / MDD {m_mean['max_drawdown']:+.2%} / "
          f"成本 {m_mean['total_costs']:.0f}", flush=True)
    s = pd.Series(dates, index=dates)
    month_ends = [d for d in s.groupby([dates.year, dates.month]).last() if d >= start_ts]
    p_m = monthly_log_paired(eq_mean, eq_s1, month_ends)
    c2a = bool(p_m["t"] >= C2A_T and p_m["mean_log"] > 0)
    c2b = bool(m_mean["total_return"] > m_s1["total_return"])
    c3 = bool(m_mean["max_drawdown"] >= m_s1["max_drawdown"] - C3_TOL)
    c4 = bool(m_mean["total_costs"] <= m_s1["total_costs"] * C4_RATIO)
    adoption = bool(c2a and c2b and c3 and c4)

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    verdict = {
        "prereg": "plans/power-prereg-20261005.md",
        "oos_window": {"start": START_OOS, "end": chain1["oos_window"]["end"]},
        "frozen": {"NEW_SETS": [list(s) for s in NEW_SETS], "T3": T3,
                   "C4_RATIO": C4_RATIO, "C3_TOL": C3_TOL, "C2A_T": C2A_T},
        "per_set_ic": per_set_ic,
        "ensemble": {"metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                 for k, v in m_mean.items()},
                     "C2a'''": {**p_m, "pass": c2a}, "C2b": c2b, "C3": c3, "C4": c4,
                     "adoption_ready": adoption},
        "baseline": {"name": "S1_T3", **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                         for k, v in m_s1.items()}},
        "verdict_word": "",   # 待用户填写
    }
    (REPORTS / f"power_chain_{stamp}.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [f"判据链 #13 读数 {stamp}（判据冻结于 plans/power-prereg-20261005.md）",
             f"基线 S1_T3: {m_s1['total_return']:+.2%} / MDD {m_s1['max_drawdown']:+.2%} / "
             f"成本 {m_s1['total_costs']:.0f}",
             f"per-set IC: {per_set_ic}",
             f"4 集成均值（T3/top10/keep20）: {m_mean['total_return']:+.2%} / "
             f"MDD {m_mean['max_drawdown']:+.2%} / 成本 {m_mean['total_costs']:.0f}",
             f"C2a''' t={p_m['t']:+.2f}(n={p_m['n']}) C2b={c2b} C3={c3} C4={c4} "
             f"→ adoption_ready={adoption}"]
    lines.append("判决词：＿＿＿（由用户填写；过线≠采纳：前瞻纸面 ≥60 交易日 + 入金拍板在前）")
    text = "\n".join(lines)
    (REPORTS / f"power_chain_{stamp}.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f"报告 → data/reports/power_chain_{stamp}.json/.txt", flush=True)


if __name__ == "__main__":
    main()
