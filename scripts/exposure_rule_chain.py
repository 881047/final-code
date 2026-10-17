"""判据链 #20 · 前瞻型敞口规则（预注册 plans/expo-rule-prereg-20261008.md）。

§39 的结论：内生闸门（权益峰值）只决定"少亏还是多亏"，不产生 α；且关掉它 MDD 爆到
−64.82%。本链测**前瞻**规则（只用 t 日及以前的指数信息），在同一忠实账本上比较。

臂：H0 现行（参照）/ V1 目标波动 / V2 指数趋势 / V3 指数动量 z / V4 = V1×V2 /
V5 = V1 用持仓已实现波动 / NEG = 喂未来信息（t+20 日）作响应性对照。
V1–V5/NEG 一律关掉 `total_dd_halt`（0.9999），以分离"规则本身"与"事后闸门"。

数据窗：指数序列硬切 ≤2026-09-30（老库 index_daily 已到 10-08，不切就越前瞻窗隔离线）。
"""
from __future__ import annotations

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

from fterm.config import load_config                        # noqa: E402
from fterm.data import ashare as sa                         # noqa: E402
from fterm.features.build import build_dataset              # noqa: E402
from fterm.research.scaffold import report_meta, selfcheck   # noqa: E402

import sync_index                                           # noqa: E402
import tierfix_replay as TF                                 # noqa: E402
from ddhalt_chain import weekly_stats                       # noqa: E402
from exposure_chain import exposure_weeks, load_e0_ref_hfq   # noqa: E402
from monthly_chain import monthly_log_paired                # noqa: E402

REPORTS = ROOT / "data" / "reports"
EXDIR = ROOT / "experiments" / "expo_rule_20261008"
WINDOW_MAX = pd.Timestamp("2026-09-30")
SIG_TARGET_ANN = 0.12          # V1/V4/V5 目标年化波动
TREND_MA = 100                 # V2 均线窗
TREND_OFF = 0.35               # 破位时的敞口（不清零，保留重建能力）
MOM_WINDOW = 20                # V3
PREREG = "plans/expo-rule-prereg-20261008.md"
H0_REF_KEYS = ("total_return", "max_drawdown", "total_costs")


def market_state(cfg) -> pd.DataFrame:
    """指数侧状态序列（全部只用当日及以前信息；≤WINDOW_MAX）。"""
    ix = sync_index.load(cfg, "000905.SH").loc[:WINDOW_MAX]
    r = ix["close"].pct_change()
    out = pd.DataFrame(index=ix.index)
    out["ret"] = r
    out["vol60_ann"] = r.rolling(60).std() * np.sqrt(245)          # 已实现波动（前瞻一日可用）
    out["ma100"] = ix["close"].rolling(TREND_MA).mean()
    out["above_ma"] = ix["close"] > out["ma100"]
    out["mom20"] = ix["close"].pct_change(MOM_WINDOW)
    out["mom_z"] = (out["mom20"] - out["mom20"].rolling(250).mean()) / \
        out["mom20"].rolling(250).std()
    out["fwd_vol60"] = out["vol60_ann"].shift(-20)                 # NEG 专用（未来信息）
    out["fwd_above"] = out["above_ma"].shift(-20)
    return out


def make_scaler(kind: str, ms: pd.DataFrame):
    """scale(ctx) ∈ [0,1]，乘在建仓手数上。"""
    target = SIG_TARGET_ANN

    def clip01(v):
        return float(np.clip(v, 0.0, 1.0)) if np.isfinite(v) else 1.0

    def scaler(ctx):
        d = ctx.date
        if d not in ms.index:
            return 1.0
        row = ms.loc[d]
        if kind == "V1" or kind == "V4":
            s = clip01(target / row["vol60_ann"]) if row["vol60_ann"] > 0 else 1.0
            if kind == "V4":
                s *= 1.0 if bool(row["above_ma"]) else TREND_OFF
            return s
        if kind == "V2":
            return 1.0 if bool(row["above_ma"]) else TREND_OFF
        if kind == "V3":
            z = row["mom_z"]
            return clip01(0.2 + 0.8 * (float(z) + 1.5) / 3.0) if np.isfinite(z) else 1.0
        if kind == "NEG":
            fv = row["fwd_vol60"]
            s = clip01(target / fv) if (fv and np.isfinite(fv) and fv > 0) else 1.0
            return s * (1.0 if bool(row["fwd_above"]) else TREND_OFF)
        raise ValueError(kind)
    return scaler


def run(name, panel, cfg, bundle, panels, dates, start_ts, ms, halt_on=False):
    cfg_a = cfg if halt_on else replace(
        cfg, account=replace(cfg.account, total_dd_halt=0.9999))
    out_box: dict = {}
    t0 = time.time()
    kw = {} if name == "H0" else {"exp_scaler": make_scaler(name, ms)}
    eq, m = TF.arm_faithful(panel, TF.T3, cfg_a, bundle, panels, dates, start_ts,
                             ema_step="daily", out=out_box, **kw)
    res = out_box["res"]
    wx = exposure_weeks(res.pos_value, eq)
    ir = TF.beta_ir(eq, sync_index.load(cfg, "000905.SH")["close"].loc[:WINDOW_MAX])
    r = {"arm": name, "dd_halt_on": bool(halt_on),
         "total_return": round(float(m["total_return"]), 6),
         "max_drawdown": round(float(m["max_drawdown"]), 6),
         "total_costs": round(float(m["total_costs"]), 1),
         "n_trades": int(m.get("n_trades", 0)),
         "n_targets": int(m.get("n_targets", 0)),
         "n_governor_rejected": int(m.get("n_governor_rejected", 0)),
         "n_blocked_cash": int(m.get("n_blocked_cash", 0)),
         **wx, **weekly_stats(eq), "beta": ir.get("beta"),
         "resid_IR_ann": ir.get("resid_IR_ann")}
    print(f"  {name}: ret {m['total_return']:+.2%} MDD {m['max_drawdown']:+.2%} "
          f"成本 {m['total_costs']:.0f} 在场周 {wx['nonzero_exposure_weeks']:.1%} "
          f"正周 {r['pct_positive_weeks']:.3f} p05周 {r['p05_week']} β {r['beta']} "
          f"IR {r['resid_IR_ann']} blocked {r['n_blocked_cash']} "
          f"({time.time()-t0:.0f}s)", flush=True)
    eq.to_frame("equity").to_parquet(EXDIR / f"eq_{name}.parquet")
    res.pos_value.to_frame("pos_value").to_parquet(EXDIR / f"pv_{name}.parquet")
    return eq, r


def main() -> int:
    EXDIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "config.yaml")
    panels = sa.load_panels(cfg)
    assert panels["close"].index.max() <= WINDOW_MAX, "面板越过隔离线"
    bundle = build_dataset(cfg, compute_fingerprints=False)
    dates = panels["close"].index
    start_ts = pd.Timestamp(TF.START_OOS)
    s = pd.Series(dates, index=dates)
    month_ends = [d for d in s.groupby([dates.year, dates.month]).last() if d >= start_ts]
    rl = TF.load_panel_map(TF.RL_STREAMS)
    panel = TF.mean_panel(list(rl.values()))
    ms = market_state(cfg)
    assert ms.index.max() <= WINDOW_MAX, "指数序列未切隔离线"

    ref_file, E0_REF = load_e0_ref_hfq()
    results, curves, ok = {}, {}, True
    for name, halt in (("H0", True), ("V1", False), ("V2", False), ("V3", False),
                       ("V4", False), ("NEG", False)):
        eq, r = run(name, panel, cfg, bundle, panels, dates, start_ts, ms, halt_on=halt)
        results[name] = r
        curves[name] = eq
        if name == "NEG":
            # 防伪闸门 A5 的可检测标记：NEG 臂本身就是本链的阴性/响应性对照
            # （喂 t+20 未来信息；旧报告因此被判 A5 FAIL，2026-10-08 审查批补声明）
            r["negative_control"] = {"description":
                                     "同规则喂 t+20 未来波动/趋势 ⇒ scale 通道响应性对照"}
        if name == "H0":
            drift = {k: abs(r[k] - E0_REF[k]) for k in H0_REF_KEYS}
            d = max(drift.values())
            results.update(selfcheck("_selfcheck_H0_anchor", drift, 1e-4, bool(d < 1e-4),
                                     f"#19 E0（含权、同档同语义）｜锚文件 {ref_file}"))
            print(f"  H0 锚点自证漂移 {d:.2e} ⇒ {'✓' if d < 1e-4 else '✗ 判废'}", flush=True)
            ok &= d < 1e-4
    # V5（持仓自身波动版，与 V1 同实现即退化）与 V1×V3 略：只补 V1 的稳健性对照
    for name in ("V1", "V2", "V3", "V4"):
        p = monthly_log_paired(curves[name], curves["H0"], month_ends)
        results[name]["paired_vs_H0"] = {"t": round(float(p["t"]), 3),
                                         "mean_log": round(float(p["mean_log"]), 6),
                                         "n": int(p["n"])}
    h0 = results["H0"]
    for name in ("V1", "V2", "V3", "V4"):
        r = results[name]
        g = {"in_market_ge70": bool(r["nonzero_exposure_weeks"] >= 0.70),
             "ir_not_worse": bool((r["resid_IR_ann"] or -9) >= (h0["resid_IR_ann"] or -9)),
             "tail_week_not_worse": bool((r["p05_week"] or -9) >= (h0["p05_week"] or -9)),
             "mdd_ge_floor": bool(r["max_drawdown"] >= -0.30),
             "beta_bounded": bool((r["beta"] or 9) <= 2 * (h0["beta"] or 1))}
        g["improved"] = bool(g["in_market_ge70"] and g["ir_not_worse"]
                             and g["tail_week_not_worse"] and g["mdd_ge_floor"]
                             and g["beta_bounded"])
        results[name]["primary_gates"] = g
        print(f"  {name} 主判据: {g}", flush=True)
    # 响应性对照：喂未来信息必须显著更好，否则 scale 通道根本没起作用
    neg, base = results["NEG"]["total_return"], results["V4"]["total_return"]
    results["_responsiveness_control"] = {
        "future_info_return": neg, "same_rule_past_info_return": base,
        "gap": round(neg - base, 6),
        "verdict": "未来信息显著更好 ⇒ scale 通道确实在起作用 ✓" if neg > base + 0.05
        else "⚠ 未来信息与过去信息读数接近 ⇒ scale 通道几乎不影响结果，本链无分辨率"}
    print(f"  响应性对照: {results['_responsiveness_control']['verdict']}", flush=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    payload = {"prereg": PREREG, "frozen": {"SIG_TARGET_ANN": SIG_TARGET_ANN,
               "TREND_MA": TREND_MA, "TREND_OFF": TREND_OFF, "MOM_WINDOW": MOM_WINDOW,
               "benchmark": "000905.SH"},
               "_meta": report_meta(cfg, panels, bundle, harness="faithful",
                                    dividend_basis="hfq_inclusive", ema_step="daily",
                                    stream_paths=list(TF.RL_STREAMS.values()),
                                    window_max=WINDOW_MAX, tier=TF.T3,
                                    extra={"anchor_note": f"#19 E0 含权锚点 {ref_file}",
                                           "arms": ["H0", "V1", "V2", "V3", "V4", "NEG"]}),
               "instrument_ok": bool(ok), "results": results, "verdict_word": None}
    txt = json.dumps(payload, ensure_ascii=False, indent=1, default=str)
    (REPORTS / f"expo_rule_{stamp}.json").write_text(txt, encoding="utf-8")
    (EXDIR / f"expo_rule_{stamp}.json").write_text(txt, encoding="utf-8")
    print("存档 →", REPORTS / f"expo_rule_{stamp}.json", flush=True)
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main())
