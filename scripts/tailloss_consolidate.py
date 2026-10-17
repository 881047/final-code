"""判据链 #18 · 裁判合并器（2026-10-09 收口批 v2；预注册 plans/tailloss-prereg-20261008.md + erratum §四）。

v2 erratum（2026-10-09，用户拍板前按推荐路径执行、判决词留用户）：
预注册 §二的 F0 自证门（fresh preds 与 #14 缓存逐位一致 ≤2.79e-3）被证实**结构性不可达**：
  ① 2.79e-3 是同代码背靠背包络（rankloss_chain.run_x0 erratum 已登记），被误用作跨实现门；
  ② #14 缓存本体在 P0-B 修复前的面板几何上（2021-09 幽灵日使 fwd_ret 按位置 shift 错位，
     实测标签差/行差全部集中于 2021-09-13~17，11,897 行）；
  ③ 跨实现微差（早停 epsilon 1e-9 vs 1e-6、张量预上 GPU）经 epoch 选择放大。
锚点改为**同代码新鲜 F0**（同面板同种子，干净单变量对照）；对 #14 缓存降级为谱系登记。
仪器门 = NEG/NEG_M1 tripwire + 新鲜 F0 流对今日面板的行/标签位连。

产出：F0/M1/M2 信号层配对（对新鲜 F0）、M1/M2 组合层在当前仪器重评 + 对当前 #15 基线
月度配对三门、M3 条件判定。判决词由用户填写。
"""
from __future__ import annotations

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

from fterm.config import load_config                      # noqa: E402
from fterm.data import ashare as sa                       # noqa: E402
from fterm.features.build import build_dataset            # noqa: E402
from fterm.model.foldframe import fold_frame              # noqa: E402
from fterm.model.validate import spearman_rank_ic         # noqa: E402

import tierfix_replay as TF                               # noqa: E402
from ddhalt_chain import weekly_stats                     # noqa: E402
from monthly_chain import monthly_log_paired              # noqa: E402

REPORTS = ROOT / "data" / "reports"
EXDIR = ROOT / "experiments" / "tailloss_20261008"
BASELINE_CURVE = ROOT / "experiments" / "tierfix_20261008" / "eq_F_RL2_daily.parquet"
CACHE_STREAM = REPORTS / "rankloss_ss7_21_99_5_13.parquet"
ARMS = ("F0", "M1", "M2", "NEG_M1", "NEG")
PAIR_T, MDD_FLOOR = 2.0, -0.30


def latest_report(arm: str) -> dict:
    files = sorted(REPORTS.glob(f"tailloss_{arm}_*.json"))
    assert files, f"缺 {arm} 臂报告（先跑 scripts/tailloss_chain.py --arm {arm}）"
    return json.loads(files[-1].read_text(encoding="utf-8"))


def load_stream(arm: str) -> pd.DataFrame:
    p = EXDIR / f"tailloss_{arm}_ss7_21_99_5_13.parquet"
    assert p.exists(), f"缺 {arm} 流缓存: {p}"
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df


def daily_ic(df: pd.DataFrame) -> pd.Series:
    return pd.Series({d: spearman_rank_ic(list(zip(g["pred"], g["y_ret"])))
                      for d, g in df.groupby("date")}).dropna()


def main() -> int:
    cfg = load_config(ROOT / "config.yaml")
    panels = sa.load_panels(cfg)
    assert panels["close"].index.max() <= TF.MAX_DATA_DATE, "越过前瞻窗隔离线"
    bundle = build_dataset(cfg, compute_fingerprints=False)
    dates = panels["close"].index
    start_ts = pd.Timestamp(TF.START_OOS)
    s = pd.Series(dates, index=dates)
    month_ends = [d for d in s.groupby([dates.year, dates.month]).last() if d >= start_ts]

    reports = {arm: latest_report(arm) for arm in ARMS}
    results: dict[str, dict] = {}
    streams = {arm: load_stream(arm) for arm in ("F0", "M1", "M2")}

    # ---- 仪器门 1：新鲜 F0 流对今日面板的行/标签位连（#14 式位连检查）----
    # 窗口必须对齐流的 OOS 覆盖（START_OOS 起）——从 data.start 起算会把训练段
    # 的 112 万行误判成"仅在重算"（2026-10-09 自查修正）。
    d0, d1 = pd.Timestamp(TF.START_OOS), panels["close"].index[-1] + pd.Timedelta(days=1)
    _Xg, _miss, y, meta, _ = fold_frame(bundle, bundle.universe, d0, d1, 0)
    recom = pd.DataFrame([{"date": d, "product": c, "y_new": yy} for (d, c), yy in zip(meta, y)])
    f0 = streams["F0"][["date", "product", "y_ret"]]
    j = recom.merge(f0, on=["date", "product"], how="outer", suffixes=("", "_f0"), indicator=True)
    both = j[j["_merge"] == "both"]
    dlab = float((both["y_new"] - both["y_ret"]).abs().max())
    n_only_recom = int((j["_merge"] == "left_only").sum())
    n_only_f0 = int((j["_merge"] == "right_only").sum())
    results["_instrument_rowlabel_check"] = {
        "label_max_abs_diff": dlab, "rows_only_recompute": n_only_recom,
        "rows_only_f0": n_only_f0, "n_matched": int(len(both)),
        "pass": bool(dlab == 0.0 and n_only_recom == 0 and n_only_f0 == 0),
        "note": "新鲜 F0 流 vs 今日面板 fold_frame 重算（同代码同数据 ⇒ 必须逐位）"}

    # ---- 仪器门 2：NEG/NEG_M1 tripwire（原样引用）----
    for arm in ("NEG_M1", "NEG"):
        nc = reports[arm]["results"].get("negative_control", {})
        results[f"_tripwire_{arm}"] = {
            "abs_mean_ic": nc.get("abs_mean_ic"), "tripwire": nc.get("tripwire"),
            "ok": bool(nc.get("ok")), "verdict": nc.get("verdict")}

    # ---- 谱系登记：新鲜 F0 vs #14 缓存（不设门，erratum §四的取证本体）----
    ref = pd.read_parquet(CACHE_STREAM)
    ref["date"] = pd.to_datetime(ref["date"])
    jc = ref.merge(streams["F0"], on=["date", "product"], suffixes=("_ref", "_new"))
    jc["dev"] = (jc["pred_ref"] - jc["pred_new"]).abs()
    by_fold = {str(int(k)): {"max": round(float(g["dev"].max()), 6),
                             "mean": round(float(g["dev"].mean()), 6)}
               for k, g in jc.groupby("fold_new")}
    results["_lineage_vs_cache"] = {
        "n_matched": int(len(jc)),
        "max_abs_dev": round(float(jc["dev"].max()), 6),
        "envelope_2_79e3_rows_pct": round(float((jc["dev"] > 2.79e-3).mean()), 4),
        "by_fold": by_fold,
        "erratum": "预注册 §二 F0 门结构性不可达：①同码包络被误用作跨实现门；"
                   "②缓存本体在 P0-B 前面板几何上（偏差集中 2021-09-13~17，标签差 11,897 行）；"
                   "③跨实现微差经 epoch 选择放大。锚点改新鲜 F0（erratum §四）"}

    # ---- 信号层：各臂 vs 新鲜 F0（同代码同数据，干净单变量）----
    ic_f0 = daily_ic(streams["F0"])
    signal = {}
    for arm in ("M1", "M2"):
        d_ic = daily_ic(streams[arm]).reindex(ic_f0.index).fillna(0) - ic_f0.fillna(0)
        t = float(d_ic.mean() / (d_ic.std(ddof=1) / np.sqrt(len(d_ic))))
        signal[arm] = {"mean_diff_ic": round(float(d_ic.mean()), 6),
                       "paired_t": round(t, 3), "n_days": int(len(d_ic)),
                       "pass_t2": bool(d_ic.mean() > 0 and t >= PAIR_T)}
        results[arm] = {"signal_layer_vs_F0_fresh": signal[arm]}
        print(f"  {arm} 信号层(对新鲜F0): ΔIC={d_ic.mean():+.5f} t={t:+.2f} "
              f"pass={signal[arm]['pass_t2']}", flush=True)

    # ---- 组合层：当前仪器重评 + 对当前 #15 基线三门 ----
    assert BASELINE_CURVE.exists(), f"基线曲线不存在: {BASELINE_CURVE}"
    eq_base = pd.read_parquet(BASELINE_CURVE)["equity"]
    eq_base.index = pd.to_datetime(eq_base.index)
    base_stats = weekly_stats(eq_base)
    base_mdd = float((eq_base / eq_base.cummax() - 1.0).min())
    results["_baseline_F_RL2_daily"] = {
        "file": str(BASELINE_CURVE.name), "lineage": "#15 忠实臂（收口批 2229）",
        "total_return": round(float(eq_base.iloc[-1] / eq_base.iloc[0] - 1.0), 6),
        "max_drawdown": round(base_mdd, 6), **base_stats}

    for arm in ("M1", "M2"):
        t0 = time.time()
        panel = streams[arm].pivot_table(index="date", columns="product", values="pred")
        eq, m = TF.arm_faithful(panel, TF.T3, cfg, bundle, panels, dates, start_ts,
                                ema_step="daily")
        eq.to_frame("equity").to_parquet(EXDIR / f"eq_{arm}_faithful_r2.parquet")
        st = weekly_stats(eq)
        mdd = float((eq / eq.cummax() - 1.0).min())
        p = monthly_log_paired(eq, eq_base, month_ends)
        gates = {
            "paired_t_ge2": bool(p["t"] >= PAIR_T and p["mean_log"] > 0),
            "mdd_ge_floor": bool(mdd >= MDD_FLOOR),
            "weekly_std_not_worse": bool(
                st.get("weekly_std_ann") is not None and base_stats.get("weekly_std_ann")
                is not None and st["weekly_std_ann"] <= base_stats["weekly_std_ann"]),
        }
        gates["all_pass"] = bool(all(gates.values()))
        results[arm].update({
            "total_return": round(float(m["total_return"]), 6),
            "max_drawdown": round(mdd, 6),
            "total_costs": round(float(m["total_costs"]), 1),
            "n_blocked_cash": int(m.get("n_blocked_cash", 0)),
            **st,
            "paired_vs_baseline": {"t": round(float(p["t"]), 3),
                                   "mean_log": round(float(p["mean_log"]), 6),
                                   "n": int(p["n"])},
            "portfolio_gates": gates,
            "note": "组合层=收口批仪器重评;旧 json 内 portfolio_faithful 为旧仪器+旧折口径,留档不采"})
        print(f"  {arm} 组合层: ret {m['total_return']:+.2%} MDD {mdd:+.2%} "
              f"周std {st.get('weekly_std_ann')} | 配对 t={p['t']:+.2f} 门={gates} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ---- M3 条件判定 ----
    d1v, d2v = signal["M1"]["mean_diff_ic"], signal["M2"]["mean_diff_ic"]
    results["_m3_condition"] = {
        "rule": "预注册 §一：仅当 M1/M2 同向（同改善）才跑",
        "m1_diff_ic": d1v, "m2_diff_ic": d2v,
        "decision": "跑 M3" if (d1v > 0 and d2v > 0) else
        "不跑：两臂信号层未同向改善（组合无依据）"}

    # ---- 汇总 ----
    instr_ok = (results["_instrument_rowlabel_check"]["pass"]
                and all(results[f"_tripwire_{a}"]["ok"] for a in ("NEG_M1", "NEG")))
    results["_instrument_ok"] = {"rowlabel_ok": results["_instrument_rowlabel_check"]["pass"],
                                 "neg_ok": all(results[f"_tripwire_{a}"]["ok"]
                                               for a in ("NEG_M1", "NEG")),
                                 "ok": bool(instr_ok),
                                 "note": "仪器门任一失败 ⇒ 全链读数作废"}
    results["_gate_summary"] = {
        arm: {"signal": signal[arm]["pass_t2"],
              "portfolio": results[arm]["portfolio_gates"]} for arm in ("M1", "M2")}

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    payload = {"prereg": "plans/tailloss-prereg-20261008.md", "erratum": "§四(2026-10-09)",
               "baseline": results["_baseline_F_RL2_daily"],
               "results": results, "verdict_word": None}
    (REPORTS / f"tailloss_consolidated_{stamp}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (EXDIR / f"tailloss_consolidated_{stamp}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("存档 →", REPORTS / f"tailloss_consolidated_{stamp}.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
