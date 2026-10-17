"""判据链 #22 · 熔断自锁缺陷修复的对比批（2026-10-10）。

预注册：`plans/halt-selflock-fix-prereg-20261010.md`（**先于实现与跑数冻结**）。

## 设计
13 臂 × 2 模式完全交叉，其余一字不动：
- **P0** = `dd_peak_mode="none"`（现行，全历史峰值 ⇒ 存在吸收态）
- **P1** = `dd_peak_mode="rolling"`（`dd_peak_window=252` ⇒ 无吸收态）

## 判据（预注册 §四）
- **C1** 吸收态消除：`max_halted_run(P1) < 0.5 × max_halted_run(P0)`；
  严格不变式 `invariant_violations == 0`
  （「权益连续 W 日恒定 ⇒ 该段内第 W 日起 halted 必为 False」；
  ⚠ `max_halted_run ≤ W` 是**错的**，见 erratum §三-1）
- **C2** 风险控制不劣化：`mdd_P1 ≥ mdd_P0 − 0.10` 且 `n_halt_events(P1) ≥ 1`
- **C3** 不退化敞口规则：非零敞口周占比 P1 > P0；无前视（回归钉保证）
- **C4** 对 §46 结论的影响（解释性）

## 用法
  python scripts/halt_selflock_chain.py
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

from fterm.config import load_config                        # noqa: E402
from fterm.data import ashare as sa                         # noqa: E402
from fterm.features.build import build_dataset              # noqa: E402
from fterm.risk.peak import halt_state, DEFAULT_WINDOW      # noqa: E402

from tierfix_replay import arm_faithful, T3, START_OOS, MAX_DATA_DATE  # noqa: E402
from monthly_chain import monthly_log_paired                # noqa: E402

OUT_DIR = ROOT / "experiments" / "halt_selflock_20261010"
CALIB_DIR = ROOT / "experiments" / "exposure_vs_precision_20261010"
PROD_REPORTS = Path(r"D:\Final\data\reports")
ARCH = ROOT / "experiments" / "arch_20261009"
EXT = ROOT / "experiments" / "external_20261009"
REGIME = ROOT / "experiments" / "regime_20261009"
MH = ROOT / "experiments" / "multihorizon_20261009"
C14 = ROOT / "experiments" / "armC14_20261009"

W = DEFAULT_WINDOW          # 252，预注册 §二 冻结
MDD_TOL = 0.10              # 预注册 §四 C2
C1_FACTOR = 0.5             # 预注册 §四 C1


def _resolve(name: str) -> Path:
    local = ROOT / "data" / "reports" / name
    if local.exists():
        return local
    return PROD_REPORTS / name


ARMS: dict[str, list[Path]] = {
    "RL2": [_resolve("rankloss_ss7_21_99_5_13.parquet"),
            _resolve("rankloss_ss123_2024_42_314_2027.parquet")],
    "MSE2": [_resolve("dlv2_ss7_21_99_5_13.parquet"),
             _resolve("dlv2_ss123_2024_42_314_2027.parquet")],
    "MSE4": [_resolve("dlv2_ss7_21_99_5_13.parquet"),
             _resolve("dlv2_ss123_2024_42_314_2027.parquet"),
             _resolve("dlv2_ss101_202.parquet"), _resolve("dlv2_ss303_404.parquet")],
    "LGBM": [_resolve("lgbm_preds_ss5_13.parquet"), _resolve("lgbm_preds_ss42_7.parquet"),
             _resolve("lgbm_preds_ss123_2024.parquet"),
             _resolve("lgbm_preds_ss2027_314.parquet")],
    "A": [REGIME / "stream_A_f4.parquet"], "B": [MH / "stream_B_f4.parquet"],
    "C": [EXT / "stream_C_f4.parquet"], "C14": [C14 / "stream_C_14.parquet"],
    "F0": [EXT / "stream_F0_f4.parquet"], "F0day": [ARCH / "stream_F0day.parquet"],
    "D": [ARCH / "stream_D.parquet"], "E": [ARCH / "stream_E.parquet"],
    "E2": [ARCH / "stream_E2.parquet"],
}
REF_ARM = "F0"


def panel_from(paths):
    ps = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"缺流：{p}")
        d = pd.read_parquet(p, columns=["date", "product", "pred"])
        d["date"] = pd.to_datetime(d["date"])
        if d["date"].max() > MAX_DATA_DATE:
            raise ValueError(f"{p.name}: 越隔离线")
        ps.append(d.pivot_table(index="date", columns="product", values="pred"))
    base = ps[0].copy()
    for d in ps[1:]:
        base = base.add(d, fill_value=0)
    return base.div(len(ps)).astype("float32")


def exposure_weeks(pos_value: pd.Series, equity: pd.Series) -> dict:
    pv = pos_value.resample("W-FRI").max()
    eq = equity.resample("W-FRI").last()
    j = pd.concat([pv.rename("pv"), eq.rename("eq")], axis=1).dropna()
    j = j[j["eq"] > 0]
    return {"n_weeks": int(len(j)),
            "nonzero_exposure_weeks": round(float((j["pv"] > 1e-6).mean()), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=None)
    args = ap.parse_args()
    arms = {k: v for k, v in ARMS.items() if args.arms is None or k in args.arms}

    cfg0 = load_config(ROOT / "config.yaml")
    print("== 面板（研究线）==", flush=True)
    panels = sa.load_panels(cfg0)
    if panels["close"].index.max() != MAX_DATA_DATE:
        raise SystemExit(f"[门] 面板右端点 {panels['close'].index.max().date()} != "
                         f"{MAX_DATA_DATE.date()}；必须在研究线 worktree 内跑")
    bundle = build_dataset(cfg0, compute_fingerprints=False)
    dates = panels["close"].index
    start_ts = pd.Timestamp(START_OOS)
    s = pd.Series(dates, index=dates)
    month_ends = [d for d in s.groupby([dates.year, dates.month]).last() if d >= start_ts]
    print(f"  面板右端点 {MAX_DATA_DATE.date()} ✓；{len(arms)} 臂", flush=True)

    CFG = {
        "P0": cfg0,      # none（现行）
        "P1": replace(cfg0, account=replace(cfg0.account,
                                            dd_peak_mode="rolling", dd_peak_window=W)),
    }
    assert CFG["P0"].account.dd_peak_mode == "none"
    assert CFG["P1"].account.dd_peak_mode == "rolling"
    # 唯一变量自证：除 dd_peak_* 外账户字段必须逐字段相同
    d0, d1 = vars(CFG["P0"].account), vars(CFG["P1"].account)
    diff = [k for k in d0 if d0[k] != d1[k]]
    assert diff == ["dd_peak_mode"], f"非唯一变量：{diff}"
    print(f"  唯一变量自证 ✓（差异字段 = {diff}）", flush=True)

    res: dict = {"prereg": "plans/halt-selflock-fix-prereg-20261010.md",
                 "generated_at": datetime.now().isoformat(timespec="seconds"),
                 "tier": T3, "start_ts": str(start_ts.date()),
                 "window": W, "arms": list(arms), "runs": {}, "gates": {}}

    for tag in ("P0", "P1"):
        mode = CFG[tag].account.dd_peak_mode
        print(f"\n===== {tag}（dd_peak_mode={mode}）=====", flush=True)
        res["runs"][tag] = {}
        for name, paths in arms.items():
            t0 = time.time()
            eq, m = arm_faithful(panel_from(paths), T3, CFG[tag], bundle, panels,
                                 dates, start_ts, ema_step="daily")
            pv = None
            # 用 metrics 的 util + 重算 halted 统计
            st = halt_state(eq, CFG[tag].account.total_dd_halt, mode=mode, window=W)
            rec = {
                "total_return": round(float(m["total_return"]), 6),
                "max_drawdown": round(float(m["max_drawdown"]), 6),
                "total_costs": round(float(m["total_costs"]), 1),
                "position_util_mean": round(float(m.get("position_util_mean", float("nan"))), 4),
                "n_trades": int(m.get("n_trades", 0)),
                "n_blocked_cash": int(m.get("n_blocked_cash", 0)),
                **st,
            }
            res["runs"][tag][name] = rec
            res.setdefault("_eq", {}).setdefault(tag, {})[name] = eq
            print(f"  {name:6s} ret {rec['total_return']:+9.2%} mdd {rec['max_drawdown']:+7.2%} "
                  f"util {rec['position_util_mean']:.4f} halted {rec['halted_frac']:.3f} "
                  f"最长 {rec['max_halted_run']:5d} 解除 {rec['n_release_events']:2d} "
                  f"不变式违规 {rec['invariant_violations']} ({time.time()-t0:.0f}s)", flush=True)

    # 非零敞口周占比（C3 需要 pos_value；arm_faithful 不返回，故用 util 的时序代理）
    # ⇒ 直接以 metrics 的 position_util_mean 作 C3 的敞口代理（同源、可比）
    print("\n===== 判据 =====", flush=True)
    names = list(arms)
    p0 = res["runs"]["P0"]
    p1 = res["runs"]["P1"]

    # C1
    m0 = float(np.median([p0[n]["max_halted_run"] for n in names]))
    thr = C1_FACTOR * m0
    c1 = {n: {"P0": p0[n]["max_halted_run"], "P1": p1[n]["max_halted_run"],
              "pass": bool(p1[n]["max_halted_run"] < thr),
              "invariant_violations_P1": p1[n]["invariant_violations"],
              "invariant_ok": bool(p1[n]["invariant_violations"] == 0),
              "constant_run_while_halted_P1": p1[n]["constant_run_max_while_halted"],
              "release_events_P0": p0[n]["n_release_events"],
              "release_events_P1": p1[n]["n_release_events"]}
          for n in names}
    n_c1 = sum(1 for v in c1.values() if v["pass"])
    n_inv = sum(1 for v in c1.values() if v["invariant_ok"])
    print(f"  C1 吸收态消除：门槛 = 0.5 × P0中位({int(m0)}) = {thr:.0f}", flush=True)
    print(f"     P1 过门槛 {n_c1}/{len(names)}；严格不变式无违规 {n_inv}/{len(names)}",
          flush=True)
    for n in names:
        v = c1[n]
        print(f"       {n:6s} P0 {v['P0']:5d} → P1 {v['P1']:5d}  "
              f"{'✓' if v['pass'] else '✗'}  解除 {v['release_events_P0']}→"
              f"{v['release_events_P1']}  不变式违规 {v['invariant_violations_P1']}",
              flush=True)

    # C2
    c2 = {n: {"mdd_P0": p0[n]["max_drawdown"], "mdd_P1": p1[n]["max_drawdown"],
              "mdd_ok": bool(p1[n]["max_drawdown"] >= p0[n]["max_drawdown"] - MDD_TOL),
              "halt_still_fires": bool(p1[n]["n_halt_events"] >= 1)}
          for n in names}
    n_mdd = sum(1 for v in c2.values() if v["mdd_ok"])
    n_fire = sum(1 for v in c2.values() if v["halt_still_fires"])
    print(f"\n  C2 风险控制不劣化：MDD 不劣于 P0−{MDD_TOL:.0%} 的臂 {n_mdd}/{len(names)}；"
          f"P1 仍触发熔断的臂 {n_fire}/{len(names)}", flush=True)

    # C3
    c3 = {n: {"util_P0": p0[n]["position_util_mean"], "util_P1": p1[n]["position_util_mean"],
              "higher": bool(p1[n]["position_util_mean"] > p0[n]["position_util_mean"])}
          for n in names}
    n_higher = sum(1 for v in c3.values() if v["higher"])
    print(f"\n  C3 敞口提升：P1 util > P0 的臂 {n_higher}/{len(names)}", flush=True)

    res["criteria"] = {
        "C1_absorbing_state_removed": {
            "threshold": round(thr, 1), "p0_median_max_halted_run": int(m0),
            "n_pass": n_c1, "n_arms": len(names), "arms": c1,
            "n_invariant_ok": n_inv,
            "invariant": "权益连续 W 日恒定 ⇒ 该段内第 W 日起 halted 必为 False；"
                         "检验量 = invariant_violations（必须为 0）。"
                         "注意 max_halted_run ≤ W 是**错的**（erratum §三-1）",
            "all_pass": bool(n_c1 == len(names)),
        },
        "C2_risk_not_degraded": {
            "mdd_tolerance": MDD_TOL, "n_mdd_ok": n_mdd, "n_halt_still_fires": n_fire,
            "n_arms": len(names), "arms": c2,
            "all_pass": bool(n_mdd == len(names) and n_fire == len(names)),
        },
        "C3_exposure_higher": {
            "n_higher": n_higher, "n_arms": len(names), "arms": c3,
            "all_pass": bool(n_higher == len(names)),
        },
    }

    # C4：§46 三判据在 P1 下重算（解释性）
    r0 = np.array([p0[n]["total_return"] for n in names])
    r1 = np.array([p1[n]["total_return"] for n in names])
    u0 = np.array([p0[n]["position_util_mean"] for n in names])
    u1 = np.array([p1[n]["position_util_mean"] for n in names])
    ic = json.loads((CALIB_DIR / "confirm_20261010_1609.json").read_text(
        encoding="utf-8"))["ic"]
    icv = np.array([ic[n] for n in names])

    def sp(a, b):
        return round(float(pd.Series(list(a)).corr(pd.Series(list(b)),
                                                   method="spearman")), 4)

    res["criteria"]["C4_effect_on_46"] = {
        "rank_stability_P0_vs_P1": sp(r0, r1),
        "ic_vs_return_P0": sp(icv, r0), "ic_vs_return_P1": sp(icv, r1),
        "util_vs_return_P0": sp(u0, r0), "util_vs_return_P1": sp(u1, r1),
        "note": "解释性读数；门槛见 §46 预注册（n=13 双侧 5% 临界 0.5549）",
    }
    c4 = res["criteria"]["C4_effect_on_46"]
    print(f"\n  C4（解释性，门槛 0.5549）：", flush=True)
    print(f"     排序 P0↔P1 ρ = {c4['rank_stability_P0_vs_P1']:+.4f}", flush=True)
    print(f"     IC↔组合层  P0 {c4['ic_vs_return_P0']:+.4f} / P1 {c4['ic_vs_return_P1']:+.4f}",
          flush=True)
    print(f"     敞口↔组合层 P0 {c4['util_vs_return_P0']:+.4f} / P1 {c4['util_vs_return_P1']:+.4f}",
          flush=True)

    # 仪器门
    bad = [(t, n) for t in ("P0", "P1") for n in names
           if res["runs"][t][n]["n_blocked_cash"] != 0]
    res["gates"] = {"ledger_closed": not bad, "bad": bad}
    import platform
    res["gates"]["env"] = {"python": platform.python_version(), "pandas": pd.__version__,
                           "numpy": np.__version__,
                           "torch": __import__("torch").__version__}
    print(f"\n[门] 账本闭合 {not bad}；环境 {res['gates']['env']}", flush=True)

    # 锚点自证：P0 必须复现 §46 存档
    confirm = json.loads((CALIB_DIR / "confirm_20261010_1609.json").read_text(
        encoding="utf-8"))
    maxd = max(abs(p0[n]["total_return"] - confirm["runs"]["halt_on"][n]["total_return"])
               for n in names)
    res["gates"]["p0_archive_delta"] = round(float(maxd), 9)
    print(f"[门] P0 vs §46 存档最大 |Δret| = {maxd:.2e} "
          f"{'✓' if maxd < 1e-6 else '✗'}", flush=True)

    for tag in res.get("_eq", {}):
        res["_eq"][tag] = {k: None for k in res["_eq"][tag]}
    res.pop("_eq", None)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out = OUT_DIR / f"selflock_{stamp}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str),
                   encoding="utf-8")
    print(f"\n报告 → {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
