"""判据链 #24 · 评估口径与采纳判据的执行器（2026-10-10）。

预注册：`plans/eval-criteria-rebuild-prereg-20261010.md`（**先于本脚本跑数冻结**）。

## 为什么需要执行器而不是文档
预注册 §三 的 R1–R5 若只写在文档里，就会重演本仓最贵的一类错误：
**"口径写在文档、代码却按老口径跑"**（§41/§42 登记的"两处语义静默分叉"）。
故把 R1–R4 落成**可执行函数**，任何组合层读数都必须过它。

## 本脚本实现
- `active_span(eq)`          —— R1 有效样本口径
- `significance(eq)`         —— R2 显著性（NW-t 多滞后 + 块自举 CI）
- `mde(n, sd)` / `power_report` —— R3 功效口径
- `breadth_table(...)`       —— R4 宽度口径（逐 k 超额 + t）
- `audit(eq, label)`         —— 综合判定（能否作为业绩证据）

## 用法
  python scripts/eval_criteria_chain.py --stream RL2
  python scripts/eval_criteria_chain.py --stream RL2 --tier w2
"""
from __future__ import annotations

import argparse
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

from fterm.config import load_config                        # noqa: E402
from fterm.data import ashare as sa                         # noqa: E402
from fterm.features.build import build_dataset              # noqa: E402
from fterm.model.stats import newey_west_t                  # noqa: E402

from tierfix_replay import arm_faithful, START_OOS, T3, MAX_DATA_DATE  # noqa: E402

OUT_DIR = ROOT / "experiments" / "eval_criteria_20261010"
PROD_REPORTS = Path(r"D:\Final\data\reports")

# ---- R3 的常数（预注册 §三 冻结）----
Z_A2 = 1.959963984540054      # 双侧 α=0.05
Z_B = 0.8416212335729143      # 80% 功效
BOOT_B = 20_000               # R2 自举次数（预注册 §五-4）
BOOT_BLOCK = 6                # 块长
BOOT_SEED = 20261010          # 固定种子（预注册 §五-4）

STREAMS = {
    "RL2": ["rankloss_ss7_21_99_5_13.parquet",
            "rankloss_ss123_2024_42_314_2027.parquet"],
    "MSE2": ["dlv2_ss7_21_99_5_13.parquet",
             "dlv2_ss123_2024_42_314_2027.parquet"],
}
W2_TIER = dict(equity=20000, top_n=2, keep_n=4, max_symbols=2, max_lots=10)


def _resolve(name: str) -> Path:
    local = ROOT / "data" / "reports" / name
    return local if local.exists() else PROD_REPORTS / name


def _nw_t(x: np.ndarray, lags: int | None = None) -> float | None:
    """`newey_west_t` 的安全包装。

    ⚠ 上游在**退化输入**（T<5 或方差≈0，如恒定权益序列）时返回 **None**，
    不是数字。直接 `float(...)` 会 TypeError（本函数存在的原因，由回归钉抓出）。
    语义：None ⇒ **无法判定显著性** ⇒ R2 **不过**（失败安全，不静默放过）。
    """
    try:
        v = newey_west_t(x, lags=lags) if lags is not None else newey_west_t(x)
    except Exception:
        return None
    if v is None:
        return None
    v = float(v)
    return v if np.isfinite(v) else None


# ------------------------------------------------------------------ R1
def active_span(eq: pd.Series, halt: float = 0.12) -> dict:
    """R1 有效样本口径：权益有变动的天数 / 熔断状态。

    **判据**：`active_span < 0.5` 的读数不得作为业绩证据（预注册 §三 R1）。
    """
    d = eq.diff().abs()
    active = (d > 1e-9)
    active.iloc[0] = False
    peak = eq.cummax()
    h = (eq < peak * (1 - halt))
    rel = [i for i in range(1, len(h)) if h.iloc[i - 1] and not h.iloc[i]]
    return {
        "n_days": int(len(eq)),
        "active_days": int(active.sum()),
        "active_span": round(float(active.mean()), 4),
        "halted_frac": round(float(h.mean()), 4),
        "first_halt": str(eq.index[h.argmax()].date()) if h.any() else None,
        "last_release": (str(eq.index[rel[-1]].date()) if rel else None),
        "passes_R1": bool(active.mean() >= 0.5),
    }


# ------------------------------------------------------------------ R2
def significance(eq: pd.Series, lags_list=(None, 0, 3, 6, 12),
                 b: int = BOOT_B, block: int = BOOT_BLOCK,
                 seed: int = BOOT_SEED) -> dict:
    """R2 显著性：月度对数收益的 NW-t（多滞后）+ 移动块自举 CI。

    **判据**：CI 含 0 或 NW-t < 2 ⇒ 不得声称"优于基准"。
    """
    mo = eq.resample("ME").last().pct_change().dropna()
    lg = np.log1p(mo.to_numpy())
    n = len(lg)
    if n < 4:
        return {"n_months": n, "error": "样本不足"}
    out = {"n_months": n, "mean_log": round(float(lg.mean()), 6),
           "sd_log": round(float(lg.std(ddof=1)), 6)}
    ts = {}
    for L in lags_list:
        key = "default" if L is None else f"lags{L}"
        v = _nw_t(lg, lags=L)
        ts[key] = (round(v, 4) if v is not None else None)
    out["nw_t"] = ts
    # 移动块自举
    rng = np.random.RandomState(seed)
    boot = np.empty(b)
    for i in range(b):
        idx = []
        while len(idx) < n:
            s = rng.randint(0, n)
            idx.extend(((s + np.arange(block)) % n).tolist())
        boot[i] = lg[np.array(idx[:n])].mean()
    lo, hi = np.quantile(boot, [0.025, 0.975])
    out["boot_ci95"] = [round(float(lo), 6), round(float(hi), 6)]
    out["boot_contains_zero"] = bool(lo <= 0 <= hi)
    # 失败安全：任一 NW-t 无法算出（None）⇒ 不得判过
    t_def = ts.get("default")
    out["passes_R2"] = bool(t_def is not None
                            and (not out["boot_contains_zero"])
                            and abs(t_def) >= 2)
    return out


# ------------------------------------------------------------------ R3
def mde(n: int, sd: float, z_a2: float = Z_A2, z_b: float = Z_B) -> float:
    """R3 MDE（80% 功效，双侧 α=0.05）= (z+z)·σ/√n。"""
    if n <= 0:
        return float("nan")
    return (z_a2 + z_b) * sd / np.sqrt(n)


def n_needed(effect: float, sd: float) -> float:
    """达到 80% 功效所需样本量。"""
    if effect == 0:
        return float("inf")
    return ((Z_A2 + Z_B) * sd / abs(effect)) ** 2


def power_report(exc: np.ndarray, per_year: int = 243) -> dict:
    """R3 功效口径：给定逐期超额序列，报 MDE 与"需多少期才能检出实测效应"。

    退化输入（效应≈0 或 σ≈0）时**不抛异常**：`n_needed_80pct` 置 None
    （语义 = 无法达到），`effect_detectable_at_n` 置 False（失败安全）。
    """
    e = np.asarray(exc, dtype="float64")
    e = e[np.isfinite(e)]
    n = len(e)
    if n < 2:
        return {"n": n, "error": "样本不足", "effect_detectable_at_n": False}
    sd = float(e.std(ddof=1))
    md = mde(n, sd)
    mean = float(e.mean())
    need = n_needed(mean, sd)
    return {
        "n": n, "mean": round(mean, 8), "sd": round(sd, 8),
        "mde": round(md, 8),
        "mde_annualized": round(float(np.expm1(md) * per_year), 4)
        if np.isfinite(md) else None,
        "n_needed_80pct": (round(need, 1) if np.isfinite(need) else None),
        "years_needed_80pct": (round(need / per_year, 1)
                               if np.isfinite(need) else None),
        "effect_detectable_at_n": bool(np.isfinite(md) and abs(mean) >= md),
    }


# ------------------------------------------------------------------ R4
def breadth_table(panel: pd.DataFrame, labels: pd.DataFrame, universe: pd.DataFrame,
                  dates, start_ts, ks=(1, 2, 3, 5, 10, 20, 50, 100, 450)) -> dict:
    """R4 宽度口径：逐 k 报「篮子日均 / 超额 / NW-t / sd / MDE」。

    **判据**：目标 k 的 t < 2 ⇒ 该 k 下的组合层读数不构成对信号质量的证据。
    """
    idx = [x for x in dates if x >= start_ts and x in panel.index and x in labels.index]
    codes = [c for c in panel.columns if c in labels.columns]
    mask = universe.reindex(index=idx, columns=codes).fillna(False).astype(bool)
    Pm = panel.loc[idx, codes].where(mask).to_numpy()
    Ym = labels.reindex(index=idx, columns=codes).astype("float64").where(mask).to_numpy()
    mkt = np.nanmean(np.where(np.isfinite(Ym), Ym, np.nan), axis=1)

    out = {}
    for k in ks:
        bkt = np.full(len(idx), np.nan)
        exc = np.full(len(idx), np.nan)
        for i in range(len(idx)):
            pr, yr = Pm[i], Ym[i]
            m = np.isfinite(pr) & np.isfinite(yr)
            if m.sum() < k + 5:
                continue
            pr_m, yr_m = pr[m], yr[m]
            order = np.argsort(-pr_m)[:k]
            bkt[i] = yr_m[order].mean()
            exc[i] = bkt[i] - yr_m.mean()
        ok = np.isfinite(exc)
        e = exc[ok]
        t = _nw_t(e)
        pr_ = power_report(e)
        out[k] = {
            "basket_bp": round(float(bkt[ok].mean()) * 1e4, 2),
            "excess_bp": round(float(e.mean()) * 1e4, 2),
            "nw_t": (round(t, 3) if t is not None else None),
            "sd_bp": round(float(e.std(ddof=1)) * 1e4, 2),
            "mde_bp": round(pr_["mde"] * 1e4, 2) if pr_.get("mde") else None,
            "years_needed_80pct": pr_["years_needed_80pct"],
            "detectable": bool(t is not None and abs(t) >= 2),
        }
    return {"n_days": len(idx), "by_k": out}


# ------------------------------------------------------------------ 综合
def audit(eq: pd.Series, label: str, halt: float = 0.12) -> dict:
    """综合判定：该读数能否作为业绩证据（R1 + R2 + R3）。"""
    r1 = active_span(eq, halt)
    r2 = significance(eq)
    return {
        "label": label, "R1": r1, "R2": r2,
        "usable_as_evidence": bool(r1["passes_R1"] and r2.get("passes_R2", False)),
        "note": "usable_as_evidence = R1 过 且 R2 过（预注册 §三）",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default="RL2", choices=list(STREAMS))
    ap.add_argument("--tier", default="t3", choices=["t3", "w2"])
    args = ap.parse_args()

    cfg = load_config(ROOT / "config.yaml")
    print("== 面板 ==", flush=True)
    panels = sa.load_panels(cfg)
    if panels["close"].index.max() != MAX_DATA_DATE:
        raise SystemExit(f"[门] 面板右端点 {panels['close'].index.max().date()} != "
                         f"{MAX_DATA_DATE.date()}；研究线须在 worktree 内跑")
    bundle = build_dataset(cfg, compute_fingerprints=False)
    dates = panels["close"].index
    start_ts = pd.Timestamp(START_OOS)

    ps = []
    for n in STREAMS[args.stream]:
        d = pd.read_parquet(_resolve(n), columns=["date", "product", "pred"])
        d["date"] = pd.to_datetime(d["date"])
        ps.append(d.pivot_table(index="date", columns="product", values="pred"))
    panel = ps[0].copy()
    for x in ps[1:]:
        panel = panel.add(x, fill_value=0)
    panel = (panel / len(ps)).astype("float32")

    tier = T3 if args.tier == "t3" else W2_TIER
    print(f"== {args.stream} @ {args.tier} 档 ==", flush=True)
    t0 = time.time()
    eq, m = arm_faithful(panel, tier, cfg, bundle, panels, dates, start_ts,
                         ema_step="daily")
    print(f"  收益 {m['total_return']:+.2%}  MDD {m['max_drawdown']:+.2%}  "
          f"({time.time()-t0:.0f}s)", flush=True)

    doc = {
        "prereg": "plans/eval-criteria-rebuild-prereg-20261010.md",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stream": args.stream, "tier": args.tier, "tier_params": tier,
        "total_return": round(float(m["total_return"]), 6),
        "max_drawdown": round(float(m["max_drawdown"]), 6),
    }
    a = audit(eq, f"{args.stream}@{args.tier}")
    doc["audit"] = a
    r1, r2 = a["R1"], a["R2"]
    print(f"\n[R1 有效样本] active_days {r1['active_days']}/{r1['n_days']} "
          f"= {r1['active_span']:.1%}  halted {r1['halted_frac']:.1%}  "
          f"⇒ {'过' if r1['passes_R1'] else '**不过（<50% ⇒ 不得作为业绩证据）**'}",
          flush=True)
    _td = r2.get("nw_t", {}).get("default")
    _td_s = f"{_td:+.2f}" if _td is not None else "n/a(退化输入)"
    print(f"[R2 显著性] NW-t(缺省) {_td_s}  "
          f"CI95 [{r2['boot_ci95'][0]:+.5f}, {r2['boot_ci95'][1]:+.5f}]  "
          f"含0={r2['boot_contains_zero']}  ⇒ "
          f"{'过' if r2['passes_R2'] else '**不过（不得声称优于基准）**'}", flush=True)
    print(f"\n⇒ **usable_as_evidence = {a['usable_as_evidence']}**", flush=True)

    print("\n== R4 宽度口径（逐 k）==", flush=True)
    bt = breadth_table(panel, bundle.labels["fwd_ret"], bundle.universe, dates, start_ts)
    doc["R4_breadth"] = bt
    print(f"{'k':>5s} {'篮子bp':>9s} {'超额bp':>9s} {'NW t':>7s} {'sd bp':>9s} "
          f"{'MDE bp':>8s} {'需年数':>9s} {'可检出':>7s}")
    print("-" * 74)
    for k, v in bt["by_k"].items():
        yn = v["years_needed_80pct"]
        tn = v["nw_t"]
        print(f"{k:>5d} {v['basket_bp']:9.1f} {v['excess_bp']:9.1f} "
              f"{(f'{tn:+.2f}' if tn is not None else 'n/a'):>7s} "
              f"{v['sd_bp']:9.1f} "
              f"{(f'{v['mde_bp']:.1f}' if v.get('mde_bp') else 'n/a'):>8s} "
              f"{(f'{yn:,.0f}' if yn is not None else 'inf'):>9s} "
              f"{'是' if v['detectable'] else '否':>7s}")
    k_target = tier["top_n"]
    tk = bt["by_k"].get(k_target)
    if tk:
        tn = tk["nw_t"]
        print(f"\n  目标持仓数 k={k_target}：NW t = "
              f"{(f'{tn:+.2f}' if tn is not None else 'n/a')} ⇒ "
              f"{'可检出' if tk['detectable'] else '**不可检出（该 k 的读数不构成信号质量证据）**'}",
              flush=True)
        doc["R4_target_k"] = {"k": k_target, **tk}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out = OUT_DIR / f"eval_{args.stream}_{args.tier}_{stamp}.json"
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str),
                   encoding="utf-8")
    print(f"\n报告 → {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
