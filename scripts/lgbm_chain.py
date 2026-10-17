"""判据链 #4 · Phase 3 LightGBM 截面臂执行器 —— 判据冻结于 plans/lgbm-prereg-20261004.md。

老盘现役超参（F47 结清值，不网格搜）；四折年界 expanding；双种子集
{42,7}/{123,2024}；组合层两遍法，配对基 = S1 同窗口重跑。判决词由用户填写。
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
from fterm.data import ashare as sa  # noqa: E402
from fterm.features.build import build_dataset  # noqa: E402
from fterm.config import load_config  # noqa: E402
from fterm.model.validate import spearman_rank_ic  # noqa: E402
from ablation_chain import CHAIN1_JSON  # noqa: E402
from gate_chain import run_model_arm  # noqa: E402
from judgment_chain import paired_weekly_t  # noqa: E402

REPORTS = Path(__file__).resolve().parents[1] / "data" / "reports"
PREDS_DIR = ROOT / "experiments" / "judgment_chain_20261003"

# ---- 冻结（预注册 §一） ----
FOLD_EDGES = ["2019-01-01", "2021-01-01", "2023-01-01", "2025-01-01"]  # 测试段起点
LGB_PARAMS = {"num_leaves": 31, "min_child_samples": 200, "learning_rate": 0.05,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
              "lambda_l2": 1.0, "n_estimators": 50}
SEED_SETS = [(42, 7), (123, 2024)]
START_OOS = "2019-01-01"
C2A_T, C3_TOL, C4_RATIO = 2.0, 0.02, 1.25


def fold_frame(bundle, panels, d0, d1):
    """[d0,d1) 测试段的 (date, code, pred 输入) 长表由训练侧拼；这里只给日期边界。"""
    return d0, d1


def train_fold(bundle, universe, train_end: pd.Timestamp, seed: int):
    """expanding 训练：全部 date < train_end − label_span 的 eligible 行。"""
    uni_mask = UniverseIndex(universe)
    import lightgbm as lgb

    feats, labels = bundle.features, bundle.labels
    cols = list(bundle.feature_cols)
    fwd = labels["fwd_ret"]
    span = 6  # label_span = h+1
    dates = [d for d in feats[cols[0]].index if d < train_end - pd.Timedelta(days=0)]
    # 净化：expanding 全史，但末端留 label_span 个交易日不进训练（防标签越折）
    pos = dates.index(train_end) if train_end in dates else len(dates)
    train_dates = dates[:max(0, pos - span)]
    Xs, ys = [], []
    for d in train_dates:
        # 2026-10-08 B2：universe 侧走整数切片（原逐格 .at）；
        # fwd 的 notna 仍逐格判定（缺失语义与 universe 不同，不合并）
        elig = [c for c in universe.columns[uni_mask.positions(d)]
                if pd.notna(fwd.at[d, c])]
        if len(elig) < 10:
            continue
        Xs.append(pd.DataFrame({c: feats[c].loc[d, elig] for c in cols}))
        ys.append(fwd.loc[d, elig])
    X = pd.concat(Xs, axis=0)
    y = pd.concat(ys, axis=0).astype("float64")
    model = lgb.LGBMRegressor(objective="regression", random_state=seed, verbose=-1,
                              deterministic=True, force_row_wise=True, **LGB_PARAMS)
    model.fit(X, y)
    return model, cols


def predict_range(bundle, universe, model, cols, d0, d1):
    uni_mask = UniverseIndex(universe)
    feats = bundle.features
    fwd = bundle.labels["fwd_ret"]
    rows = []
    for d in [d for d in feats[cols[0]].index if d0 <= d < d1]:
        # 2026-10-08 B2：universe 侧走整数切片（原逐格 .at）；
        # fwd 的 notna 仍逐格判定（缺失语义与 universe 不同，不合并）
        elig = [c for c in universe.columns[uni_mask.positions(d)]
                if pd.notna(fwd.at[d, c])]
        if len(elig) < 5:
            continue
        X = pd.DataFrame({c: feats[c].loc[d, elig] for c in cols})
        pred = model.predict(X)
        for c, p, y in zip(elig, pred, [float(fwd.at[d, c]) for c in elig]):
            rows.append({"date": d, "product": c, "pred": float(p), "y_ret": y})
    return rows


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

    # ---- L 臂 OOS 预测流（双种子集，缓存）----
    all_rows = {}
    for ss in SEED_SETS:
        tag = f"ss{'_'.join(map(str, ss))}"
        cache = REPORTS / f"lgbm_preds_{tag}.parquet"
        if cache.exists():
            all_rows[tag] = pd.read_parquet(cache)
            all_rows[tag]["date"] = pd.to_datetime(all_rows[tag]["date"])
            print(f"  L_{tag}: 缓存 {len(all_rows[tag])} 条", flush=True)
            continue
        rows = []
        t0 = time.time()
        for k in range(len(edges) - 1):
            d0, d1 = edges[k], edges[k + 1]
            models = []
            for seed in ss:
                m, cols = train_fold(bundle, universe, d0, seed)
                models.append(m)
                print(f"    fold{k} seed{seed} 训练完成 ({time.time()-t0:.0f}s)", flush=True)
            # 集成 = 组内种子预测均值
            seg = []
            for m in models:
                seg.append(pd.DataFrame(predict_range(bundle, universe, m, cols, d0, d1)))
            df = seg[0].copy()
            for extra in seg[1:]:
                df["pred"] += extra["pred"].to_numpy()
            df["pred"] /= len(seg)
            df["fold"] = k
            rows.append(df)
        out = pd.concat(rows, ignore_index=True)
        out.to_parquet(cache)
        all_rows[tag] = out
        print(f"  L_{tag}: {len(out)} 条 OOS 预测 ({time.time()-t0:.0f}s)", flush=True)

    # ---- 信息层（入场券读数）----
    info = {}
    for tag, df in all_rows.items():
        ics = [spearman_rank_ic(list(zip(g["pred"], g["y_ret"])))
               for _, g in df.groupby("date")]
        ics = [x for x in ics if x is not None]
        arr = np.array(ics)
        info[tag] = {"rank_ic": float(arr.mean()), "n_days": len(arr)}

    # ---- 组合层（S1 同窗口重跑配对）----
    print("== 组合层（S1@2019 窗口重跑 + L 臂两遍法）==", flush=True)
    from ablation_chain import run_rule_arm
    eq_s1, m_s1 = run_rule_arm("S1", cfg, bundle, panels, dates, start_ts)
    print(f"  S1: {m_s1['final_equity']:.0f} ({m_s1['total_return']:+.2%}) "
          f"MDD {m_s1['max_drawdown']:+.2%} 成本 {m_s1['total_costs']:.0f}", flush=True)
    arms = {}
    for tag, df in all_rows.items():
        panel = df.pivot_table(index="date", columns="product", values="pred")
        eq, m = run_model_arm(-1e9, 2, 3, cfg, panel, bundle, panels, dates, start_ts)
        arms[tag] = {"metrics": m, "equity": eq}
        print(f"  L_{tag}: {m['final_equity']:.0f} ({m['total_return']:+.2%}) "
              f"MDD {m['max_drawdown']:+.2%} 成本 {m['total_costs']:.0f} "
              f"目标 {m['n_targets']}", flush=True)

    fridays = [d for d in dates if d >= start_ts
               and d.weekday() == cfg.signal.rebalance_weekday and d in eq_s1.index]
    results = {}
    for tag, a in arms.items():
        m, eq = a["metrics"], a["equity"]
        p = paired_weekly_t(eq, eq_s1, fridays)
        c2a = bool(p["t"] >= C2A_T and p["mean"] > 0)
        c2b = bool(m["total_return"] > m_s1["total_return"])
        c3 = bool(m["max_drawdown"] >= m_s1["max_drawdown"] - C3_TOL)
        c4 = bool(m["total_costs"] <= m_s1["total_costs"] * C4_RATIO)
        results[tag] = {"info": info[tag],
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
        "prereg": "plans/lgbm-prereg-20261004.md",
        "oos_window": {"start": START_OOS, "end": chain1["oos_window"]["end"]},
        "frozen": {"LGB_PARAMS": LGB_PARAMS, "FOLD_EDGES": FOLD_EDGES,
                   "SEED_SETS": [list(s) for s in SEED_SETS],
                   "C2A_T": C2A_T, "C3_TOL": C3_TOL, "C4_RATIO": C4_RATIO},
        "baseline": {"name": "S1@2019", **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                           for k, v in m_s1.items()}},
        "results": results, "dual_seed_agree": dual,
        "verdict_word": "",   # 待用户填写
    }
    (REPORTS / f"lgbm_chain_{stamp}.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [f"判据链 #4 读数 {stamp}（判据冻结于 plans/lgbm-prereg-20261004.md）",
             f"基线 S1@2019: {m_s1['total_return']:+.2%} / MDD {m_s1['max_drawdown']:+.2%} / 成本 {m_s1['total_costs']:.0f}"]
    for tag, r in results.items():
        lines.append(f"L_{tag}: IC {r['info']['rank_ic']:+.4f} (n={r['info']['n_days']}日) "
                     f"ret {r['metrics']['total_return']:+.2%} MDD {r['metrics']['max_drawdown']:+.2%} "
                     f"成本 {r['metrics']['total_costs']:.0f} "
                     f"C2a t={r['C2a']['t']:+.2f} C2b={r['C2b']['pass']} "
                     f"C3={r['C3']['pass']} C4={r['C4']['pass']} "
                     f"→ adoption_ready={r['adoption_ready']}")
    lines.append(f"dual_seed_agree = {dual}")
    lines.append("判决词：＿＿＿（由用户填写）")
    text = "\n".join(lines)
    (REPORTS / f"lgbm_chain_{stamp}.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f"报告 → data/reports/lgbm_chain_{stamp}.json/.txt", flush=True)


if __name__ == "__main__":
    main()
