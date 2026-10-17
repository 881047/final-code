"""判据链 #6 · 带宽确认执行器 —— 判据冻结于 plans/bandwidth-prereg-20261004.md。

新增种子集 {5,13}/{2027,314}，复用 lgbm_chain 的折级训练与组合层（同构 #4），
回答"IC 0.050 是否种子稳健"。判决词由用户填写。
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
from ablation_chain import CHAIN1_JSON  # noqa: E402
from lgbm_chain import train_fold, predict_range, FOLD_EDGES  # noqa: E402
from gate_chain import run_model_arm  # noqa: E402
from judgment_chain import paired_weekly_t  # noqa: E402

REPORTS = Path(__file__).resolve().parents[1] / "data" / "reports"
SEED_SETS = [(5, 13), (2027, 314)]          # 冻结（预注册 §一）
START_OOS = "2019-01-01"

# ---- 冻结判据（预注册 §二） ----
B1_IC_FLOOR = 0.02
B3_IC_MEAN = 0.03
B3_IC_RANGE = 0.04


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

    # S1@2019 基线（同 #4 口径重跑）
    from ablation_chain import run_rule_arm
    eq_s1, m_s1 = run_rule_arm("S1", cfg, bundle, panels, dates, start_ts)
    print(f"  S1@2019: {m_s1['final_equity']:.0f} ({m_s1['total_return']:+.2%})", flush=True)

    results = {}
    for ss in SEED_SETS:
        tag = f"ss{'_'.join(map(str, ss))}"
        cache = REPORTS / f"lgbm_preds_{tag}.parquet"
        if cache.exists():
            df = pd.read_parquet(cache)
            df["date"] = pd.to_datetime(df["date"])
            print(f"  L_{tag}: 缓存 {len(df)} 条", flush=True)
        else:
            rows = []
            t0 = time.time()
            for k in range(len(edges) - 1):
                d0, d1 = edges[k], edges[k + 1]
                models = []
                for seed in ss:
                    m, cols = train_fold(bundle, universe, d0, seed)
                    models.append(m)
                    print(f"    fold{k} seed{seed} 训练完成 ({time.time()-t0:.0f}s)", flush=True)
                seg = [pd.DataFrame(predict_range(bundle, universe, m, cols, d0, d1))
                       for m in models]
                dfk = seg[0].copy()
                for extra in seg[1:]:
                    dfk["pred"] += extra["pred"].to_numpy()
                dfk["pred"] /= len(seg)
                dfk["fold"] = k
                rows.append(dfk)
            df = pd.concat(rows, ignore_index=True)
            df.to_parquet(cache)
            print(f"  L_{tag}: {len(df)} 条 OOS 预测 ({time.time()-t0:.0f}s)", flush=True)
        panel = df.pivot_table(index="date", columns="product", values="pred")
        ics = [spearman_rank_ic(list(zip(g["pred"], g["y_ret"])))
               for _, g in df.groupby("date")]
        ics = [x for x in ics if x is not None]
        eq, m = run_model_arm(-1e9, 2, 3, cfg, panel, bundle, panels, dates, start_ts)
        results[tag] = {"rank_ic": float(np.mean([x for x in ics if x is not None])),
                        "n_days": len(ics),
                        "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                    for k, v in m.items()},
                        "B1": bool(np.mean([x for x in ics if x is not None]) > B1_IC_FLOOR),
                        "B2": bool(m["total_return"] > m_s1["total_return"])}
        print(f"  L_{tag}: IC {results[tag]['rank_ic']:+.4f} "
              f"ret {m['total_return']:+.2%} MDD {m['max_drawdown']:+.2%}", flush=True)

    # B3：四套种子集合并（含 #4 的两套；#4 verdict JSON 键名无 L_ 前缀）
    chain4 = json.loads((ROOT / "experiments/lgbm_chain_20261004/lgbm_chain_20261004_0230.json")
                        .read_text(encoding="utf-8"))
    c4_keys = [k for k in chain4["results"] if k.startswith("ss")]
    all_ics = [chain4["results"][t]["info"]["rank_ic"] for t in c4_keys] + \
              [results[t]["rank_ic"] for t in results]
    b3 = {"ics": {**{t: chain4["results"][t]["info"]["rank_ic"] for t in c4_keys},
                  **{t: results[t]["rank_ic"] for t in results}},
          "mean": float(np.mean(all_ics)),
          "range": float(np.max(all_ics) - np.min(all_ics)),
          "pass": bool(np.mean(all_ics) > B3_IC_MEAN
                       and (max(all_ics) - min(all_ics)) < B3_IC_RANGE)}
    robust = bool(all(r["B1"] and r["B2"] for r in results.values()) and b3["pass"])

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    verdict = {
        "prereg": "plans/bandwidth-prereg-20261004.md",
        "oos_window": {"start": START_OOS, "end": chain1["oos_window"]["end"]},
        "frozen": {"B1_IC_FLOOR": B1_IC_FLOOR, "B3_IC_MEAN": B3_IC_MEAN,
                   "B3_IC_RANGE": B3_IC_RANGE, "SEED_SETS": [list(s) for s in SEED_SETS]},
        "results": results, "B3": b3, "robust": robust,
        "verdict_word": "",   # 待用户填写
    }
    (REPORTS / f"bandwidth_chain_{stamp}.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [f"判据链 #6 读数 {stamp}（判据冻结于 plans/bandwidth-prereg-20261004.md）",
             f"基线 S1@2019: {m_s1['total_return']:+.2%}"]
    for tag, r in results.items():
        lines.append(f"L_{tag}: IC {r['rank_ic']:+.4f} (n={r['n_days']}) "
                     f"ret {r['metrics']['total_return']:+.2%} "
                     f"MDD {r['metrics']['max_drawdown']:+.2%} B1={r['B1']} B2={r['B2']}")
    lines.append(f"B3 四套 IC: {b3['ics']}  mean {b3['mean']:+.4f} range {b3['range']:.4f} → {b3['pass']}")
    lines.append(f"带宽稳健判定 robust = {robust}（True ⇒ IC 0.050 为种子稳健读数；"
                 f"False ⇒ 信号层正式收口走路径 b）")
    lines.append("判决词：＿＿＿（由用户填写）")
    text = "\n".join(lines)
    (REPORTS / f"bandwidth_chain_{stamp}.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f"报告 → data/reports/bandwidth_chain_{stamp}.json/.txt", flush=True)


if __name__ == "__main__":
    main()
