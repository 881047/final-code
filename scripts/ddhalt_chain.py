"""判据链 #16 · 回撤熔断阈值标定（预注册 plans/ddhalt-prereg-20261008.md）。

唯一变量 = `account.total_dd_halt`；其余（档位/top/keep/EMA 语义/成本/数据窗）
全部沿用 #15 忠实臂。产出的是**标定曲线**（阈值 → 收益/回撤/IR/仓位），
不是"挑一个最好的阈值"。

自证 V0 必须与 #15 存档的 F_RL2_daily / F_MSE2_daily 逐位一致；
阴性对照 V4（≈关闭熔断）若与 V0 相同 ⇒ 本链对阈值不敏感，仪器有问题，判废。
"""
from __future__ import annotations

import argparse
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

from fterm.config import load_config                      # noqa: E402
from fterm.data import ashare as sa                       # noqa: E402
from fterm.features.build import build_dataset            # noqa: E402

import sync_index                                        # noqa: E402
import tierfix_replay as TF                               # noqa: E402
from fterm.research.scaffold import report_meta           # noqa: E402
from monthly_chain import monthly_log_paired              # noqa: E402

REPORTS = ROOT / "data" / "reports"
OUTDIR = ROOT / "experiments" / "ddhalt_20261008"
# 默认网格 = #16（plans/ddhalt-prereg-20261008.md）。#17（plans/ddhalt2-prereg-20261008.md）
# 用 --grid 0.02,0.05,0.08,0.10,0.12 --anchor 0.12 --prereg plans/ddhalt2-prereg-20261008.md
# ——同一执行器跑多条链，避免像 run_two_pass 那样复制出六份而各自漂移（docs/02 已登记过该教训）。
THRESHOLDS = [("V0", 0.12), ("V1", 0.20), ("V2", 0.25), ("V3", 0.35), ("V4", 0.99)]
ANCHOR_THR = 0.12
PREREG = "plans/ddhalt-prereg-20261008.md"
OUTTAG = "ddhalt"
MODELS = {"RL2": "rank-loss 2 流", "MSE2": "MSE 2 流"}
MDD_FLOOR = -0.30          # 冻结硬约束：窗内 MDD 劣于 −30% 的阈值视为不可用
PAIR_T = 2.0               # 沿用 C2a 的 t 门槛（配对 V* vs V0）


def load_v0_reference() -> dict:
    """#15 忠实批存档里的 F_RL2_daily / F_MSE2_daily 读数（V0 自证基准）。"""
    files = sorted(REPORTS.glob("tierfix_faithful_*.json"))
    assert files, "找不到 #15 忠实批存档，无法自证 V0"
    d = json.loads(files[-1].read_text(encoding="utf-8"))
    out = {}
    for tag, key in (("RL2", "F_RL2_daily"), ("MSE2", "F_MSE2_daily")):
        r = d["results"][key]
        out[tag] = (str(files[-1].name), {k: float(r[k]) for k in
                                          ("total_return", "max_drawdown", "total_costs")})
    return out


def weekly_stats(eq: pd.Series) -> dict:
    """周收益分布 = "周频稳定"的度量本体（预注册 §三；只作透明读数，不设门）。"""
    wk = eq.resample("W-FRI").last().pct_change().dropna()
    if len(wk) < 12:
        return {}
    sd = float(wk.std())
    return {"n_weeks": int(len(wk)),
            "weekly_std_ann": round(sd * np.sqrt(52), 4),
            "pct_positive_weeks": round(float((wk > 0).mean()), 4),
            "worst_week": round(float(wk.min()), 4),
            "p05_week": round(float(wk.quantile(0.05)), 4),
            "IR_weekly_ann": round(float(wk.mean() / sd * np.sqrt(52)), 3) if sd > 0 else None}


def main() -> int:
    global THRESHOLDS, ANCHOR_THR, PREREG, OUTTAG, OUTDIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default=None, help="逗号分隔的 total_dd_halt 网格")
    ap.add_argument("--anchor", type=float, default=ANCHOR_THR, help="配对与自证的锚点阈值")
    ap.add_argument("--prereg", default=PREREG)
    ap.add_argument("--outtag", default=OUTTAG)
    a = ap.parse_args()
    if a.grid:
        vals = [float(x) for x in a.grid.split(",")]
        THRESHOLDS = [(f"A{i}" if v != a.anchor else "V0", v) for i, v in enumerate(vals)]
    ANCHOR_THR, PREREG, OUTTAG = a.anchor, a.prereg, a.outtag
    OUTDIR = ROOT / "experiments" / f"{OUTTAG}_20261008"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "config.yaml")
    panels = sa.load_panels(cfg)
    assert panels["close"].index.max() <= TF.MAX_DATA_DATE, "面板越过前瞻窗隔离线"
    bundle = build_dataset(cfg, compute_fingerprints=False)
    dates = panels["close"].index
    start_ts = pd.Timestamp(TF.START_OOS)
    s = pd.Series(dates, index=dates)
    month_ends = [d for d in s.groupby([dates.year, dates.month]).last() if d >= start_ts]

    mse = TF.load_panel_map(TF.MSE_STREAMS)
    rl = TF.load_panel_map(TF.RL_STREAMS)
    panels_mean = {"RL2": TF.mean_panel(list(rl.values())),
                   "MSE2": TF.mean_panel([mse["ss7_21_99_5_13"],
                                          mse["ss123_2024_42_314_2027"]])}
    # 基准指数统一走 sync_index（L1 契约同面板；缺面板大声报错，不再内联 SQL）
    bench = sync_index.load(cfg, "000905.SH")["close"]
    bench = bench.loc[bench.index >= start_ts].astype("float64")
    ref = load_v0_reference()

    results: dict[str, dict] = {"_v0_reference": {k: {"file": v[0], **v[1]}
                                                 for k, v in ref.items()}}
    curves: dict[str, pd.Series] = {}
    ok_all = True
    for mk in MODELS:
        for tag, thr in THRESHOLDS:
            st: dict = {}
            cfg_t = replace(cfg, account=replace(cfg.account, total_dd_halt=thr))
            t0 = time.time()
            eq, m = TF.arm_faithful(panels_mean[mk], TF.T3, cfg_t, bundle, panels,
                                    dates, start_ts, ema_step="daily", stats=st)
            key = f"{tag}_{mk}"
            curves[key] = eq
            rr = st.get("reject_by_reason", {})
            ir = TF.beta_ir(eq, bench)
            results[key] = {"total_dd_halt": thr,
                            "total_return": round(float(m["total_return"]), 6),
                            "max_drawdown": round(float(m["max_drawdown"]), 6),
                            "total_costs": round(float(m["total_costs"]), 1),
                            "position_util_mean": round(float(m.get("position_util_mean", 0)), 4),
                            "n_blocked_cash": int(m.get("n_blocked_cash", 0)),
                            "reject_by_reason": rr,
                            "halted_share": round(rr.get("halted", 0) /
                                                  max(1, sum(rr.values())), 4),
                            "beta": ir.get("beta"), "resid_IR": ir.get("resid_IR_ann"),
                            **weekly_stats(eq)}
            print(f"  {key} (halt={thr:.2%}): {m['total_return']:+.2%} "
                  f"MDD {m['max_drawdown']:+.2%} 均仓位 "
                  f"{results[key]['position_util_mean']:.3f} halted占比 "
                  f"{results[key]['halted_share']:.2f} IR {ir.get('resid_IR_ann')} "
                  f"blocked_cash {results[key]['n_blocked_cash']} ({time.time()-t0:.0f}s)",
                  flush=True)
            if results[key]["n_blocked_cash"] != 0:
                print(f"  !! {key} blocked_cash≠0 ⇒ 忠实账本未闭合", flush=True)
                ok_all = False
            # V0 自证
            _fn, rv = ref[mk]
            drift = max(abs(results[key][k] - rv[k]) for k in
                        ("total_return", "max_drawdown", "total_costs"))
            if tag == "V0":
                results[key]["anchor_selfcheck_abs_drift"] = round(drift, 9)
                results[key]["v0_selfcheck_ok"] = bool(drift < 1e-6)
                print(f"     V0 自证漂移 {drift:.2e} ⇒ {'✓' if drift < 1e-6 else '✗ 判废'}",
                      flush=True)
                if drift >= 1e-6:
                    ok_all = False
        # 配对 V* vs V0（同模型对象）
        anchor_key = next(f"{t}_{mk}" for t, _v in THRESHOLDS if t == "V0")
        base = curves[anchor_key]
        for tag, thr in THRESHOLDS:
            if tag == "V0":
                continue
            p = monthly_log_paired(curves[f"{tag}_{mk}"], base, month_ends)
            r = results[f"{tag}_{mk}"]
            r["paired_vs_V0"] = {"t": round(float(p["t"]), 4),
                                 "mean_log": round(float(p["mean_log"]), 6),
                                 "n": int(p["n"]),
                                 "pass": bool(p["t"] >= PAIR_T and p["mean_log"] > 0)}
            r["mdd_floor_ok"] = bool(results[f"{tag}_{mk}"]["max_drawdown"] >= MDD_FLOOR)
            print(f"     {tag}_{mk} vs V0: t={p['t']:+.2f} "
                  f"MDD可用={r['mdd_floor_ok']}", flush=True)
    # 阴性对照：V4 与 V0 不得相同
    # 阴性对照：网格两端与锚点不得逐位相同（相同 ⇒ 仪器对阈值不敏感，本链判废）
    for mk in MODELS:
        anchor_key = next(f"{t}_{mk}" for t, _v in THRESHOLDS if t == "V0")
        far_key = next((f"{t}_{mk}" for t, v in THRESHOLDS
                        if abs(v - ANCHOR_THR) >= 0.5), None)
        if far_key is None:      # 本段网格不含远端：与 #16 存档的最松档比
            far = json.loads(sorted(REPORTS.glob("ddhalt_*.json"))[-1].read_text(
                encoding="utf-8"))["results"].get(f"V4_{mk}")
            if far is None:
                results[f"_negative_control_{mk}"] = {"skipped": "无可比远端读数"}
                continue
            same = all(abs(results[anchor_key][k] - far[k]) < 1e-9 for k in
                       ("total_return", "max_drawdown", "total_costs"))
            label = f"{far_key or 'V4(#16 存档)'} vs {anchor_key}"
        else:
            same = all(abs(results[far_key][k] - results[anchor_key][k]) < 1e-9
                       for k in ("total_return", "max_drawdown", "total_costs"))
            label = f"{far_key} vs {anchor_key}"
        results[f"_negative_control_{mk}"] = {"compared": label, "identical": bool(same),
                                              "verdict": "仪器对阈值不敏感 ⇒ 本链判废"
                                              if same else "有响应 ✓"}
        if same:
            ok_all = False
            print(f"  !! {mk}: {label} 逐位相同 ⇒ 本链对熔断阈值不敏感，判废", flush=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    tagp = OUTTAG
    payload = {"prereg": PREREG, "tier": TF.T3, "grid": dict(THRESHOLDS),
               "anchor": ANCHOR_THR,
               "ema_step": "daily", "thresholds": dict(THRESHOLDS),
               "_meta": report_meta(cfg, panels, bundle, harness="faithful",
                                    dividend_basis="hfq_inclusive", ema_step="daily",
                                    stream_paths=list(TF.RL_STREAMS.values()),
                                    window_max=TF.MAX_DATA_DATE, tier=TF.T3,
                                    extra={"anchor_note": "V0 对 #15 忠实批存档自证",
                                           "negative_control": "网格远端 vs 锚点逐位比较"}),
               "instrument_ok": bool(ok_all), "results": results, "verdict_word": None}
    txt = json.dumps(payload, ensure_ascii=False, indent=1, default=str)
    (REPORTS / f"{tagp}_{stamp}.json").write_text(txt, encoding="utf-8")
    (OUTDIR / f"{tagp}_{stamp}.json").write_text(txt, encoding="utf-8")
    lines = [f"回撤熔断阈值标定 {PREREG} {stamp}",
             f"仪器自证: {'全部通过 ✓' if ok_all else '存在失败项 ⇒ 读数不作数 ✗'}", ""]
    for k, v in results.items():
        if isinstance(v, dict) and "total_return" in v:
            lines.append(f"{k} halt={v['total_dd_halt']:.0%}: ret={v['total_return']:+.4f} "
                         f"MDD={v['max_drawdown']:+.4f} util={v['position_util_mean']:.3f} "
                         f"halted={v['halted_share']:.2f} IR={v['resid_IR']} "
                         f"周std={v.get('weekly_std_ann')} 正周={v.get('pct_positive_weeks')} "
                         f"最差周={v.get('worst_week')} 周IR={v.get('IR_weekly_ann')} "
                         f"paired={v.get('paired_vs_V0', {})} MDD可用={v.get('mdd_floor_ok')}")
        else:
            lines.append(f"{k}: {v}")
    lines.append("判决词：＿＿＿（由用户填写；本链是标定曲线，单点胜出不构成采纳依据）")
    (REPORTS / f"ddhalt_{stamp}.txt").write_text("\n".join(lines), encoding="utf-8")
    for k, eq in curves.items():
        eq.to_frame("equity").to_parquet(OUTDIR / f"eq_{k}.parquet")
    print(f"存档 → {REPORTS}/{tagp}_{stamp}.json|txt + {OUTDIR}", flush=True)
    return 0 if ok_all else 4


if __name__ == "__main__":
    sys.exit(main())
