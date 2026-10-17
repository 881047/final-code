"""§45 留痕补全：阴性对照分辨率标定（每流 5 置换种子的 z）+ 生产位月度配对 t。

## 为什么需要这个脚本（填补 docs/03 §45 的两个留痕缺口）

审查（scribe，2026-10-10）指出 §45 有两组数字**无法追溯**：
  ③ 阴性对照的 z 值（F0day +3.39 / F0 +0.74 / D −0.07 / E2 −1.83）——
     全仓检索只有 D 的 −0.073 落在 phase4_20261010_1243.json，其余三值无 artifact。
     根因：`phase4_arch_chain.py` 的阴性对照只跑 `args.arms[0]`（单臂）。
  ④ 对生产位的三个月度配对 t（F0 −1.50 / D −1.69 / E2 −2.20）——无任何 json 留痕。

本脚本把这两组读数**落盘**，使 §45 的每个数字都可追溯到 artifact。

## 口径（逐字复用既有实现，不重造）
- 组合层引擎：`tierfix_replay.arm_faithful`（真 T3=100万/top10/keep20/max_lots50 + daily EMA）
- 月度配对 t：`monthly_chain.monthly_log_paired`
- 生产位：`data/reports/rankloss_ss{7_21_99_5_13,123_2024_42_314_2027}.parquet`
  的 2 流均值（= 判据链采纳的 RL2 构型）
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

WT = Path(r"C:\Users\27817\WorkBuddy\Worktrees\Final\master-d5a41331")
REPORTS = Path(r"D:\Final\data\reports")
OUT = WT / "experiments" / "arch_20261009"

sys.path.insert(0, str(WT))
sys.path.insert(0, str(WT / "scripts"))

from fterm.config import load_config                      # noqa: E402
from fterm.data import ashare as sa                       # noqa: E402
from fterm.features.build import build_dataset            # noqa: E402
from tierfix_replay import arm_faithful, T3, START_OOS, mean_panel  # noqa: E402
from monthly_chain import monthly_log_paired              # noqa: E402

SEEDS = (1, 2, 3, 4, 5)
STREAMS = {
    "F0": WT / "experiments/external_20261009/stream_F0_f4.parquet",
    "F0day": OUT / "stream_F0day.parquet",
    "D": OUT / "stream_D.parquet",
    "E": OUT / "stream_E.parquet",
    "E2": OUT / "stream_E2.parquet",
}


def to_panel(path: Path) -> pd.DataFrame:
    d = pd.read_parquet(path, columns=["date", "product", "pred"])
    d["date"] = pd.to_datetime(d["date"])
    return d.pivot_table(index="date", columns="product", values="pred")


def shuffled(panel: pd.DataFrame, seed: int) -> pd.DataFrame:
    """日内置换预测秩（阴性对照），与 phase4_arch_chain.py 同实现。"""
    rng = np.random.RandomState(seed)
    v = panel.to_numpy(dtype="float64", copy=True)
    for i in range(v.shape[0]):
        row = v[i]
        ok = ~np.isnan(row)
        if ok.sum() > 1:
            v[i][ok] = row[ok][rng.permutation(int(ok.sum()))]
    return pd.DataFrame(v, index=panel.index, columns=panel.columns)


def main() -> int:
    cfg = load_config(WT / "config.yaml")
    panels = sa.load_panels(cfg)
    bundle = build_dataset(cfg, compute_fingerprints=False)
    dates = panels["close"].index
    start_ts = pd.Timestamp(START_OOS)
    s = pd.Series(dates, index=dates)
    month_ends = [d for d in s.groupby([dates.year, dates.month]).last() if d >= start_ts]

    out = {"generated_at": datetime.now().isoformat(timespec="seconds"),
           "purpose": "§45 留痕补全（审查发现的两处不可追溯读数）",
           "tier": T3, "ema_step": "daily", "seeds": list(SEEDS),
           "engine": "tierfix_replay.arm_faithful", "paired": "monthly_chain.monthly_log_paired"}

    # ---- 先算真实组合层读数（每流一次）----
    real, eqs = {}, {}
    for name, p in STREAMS.items():
        t0 = time.time()
        eq, m = arm_faithful(to_panel(p), T3, cfg, bundle, panels, dates, start_ts,
                             ema_step="daily")
        eqs[name] = eq
        real[name] = {"total_return": float(m["total_return"]),
                      "max_drawdown": float(m["max_drawdown"]),
                      "n_blocked_cash": int(m.get("n_blocked_cash", 0)),
                      "position_util_mean": float(m.get("position_util_mean", float("nan")))}
        print(f"  real {name:6s} {m['total_return']:+.2%} ({time.time()-t0:.0f}s)", flush=True)

    # ---- ③ 阴性对照分辨率标定：每流 vs 自身 5 个置换种子的分布 ----
    print("== ③ 阴性对照分辨率标定（每流 5 置换种子）==", flush=True)
    calib = {}
    for name, p in STREAMS.items():
        panel = to_panel(p)
        vals = []
        for sd in SEEDS:
            _eq, m = arm_faithful(shuffled(panel, sd), T3, cfg, bundle, panels, dates,
                                  start_ts, ema_step="daily")
            vals.append(float(m["total_return"]))
        a = np.asarray(vals)
        sd_ = float(a.std(ddof=1))
        z = (real[name]["total_return"] - float(a.mean())) / sd_ if sd_ > 0 else float("nan")
        calib[name] = {"real_return": round(real[name]["total_return"], 6),
                       "shuffled_returns": [round(x, 6) for x in vals],
                       "shuffled_mean": round(float(a.mean()), 6),
                       "shuffled_std": round(sd_, 6),
                       "z_of_real_vs_shuffled": round(z, 3)}
        print(f"  {name:6s} real {real[name]['total_return']:+.2%} | perm mean "
              f"{a.mean():+.2%} sd {sd_:.2%} => z={z:+.2f}", flush=True)
    out["negative_control_calibration"] = calib

    # ---- ④ 对生产位（RL2 = rank-loss 2 流均值）的月度配对 t ----
    print("== ④ 生产位 RL2 组合层 + 各臂月度配对 t ==", flush=True)
    def pmap(p):
        d = pd.read_parquet(p, columns=["date", "product", "pred"])
        d["date"] = pd.to_datetime(d["date"])
        return d.pivot_table(index="date", columns="product", values="pred")

    rl2 = mean_panel([pmap(REPORTS / "rankloss_ss7_21_99_5_13.parquet"),
                      pmap(REPORTS / "rankloss_ss123_2024_42_314_2027.parquet")])
    eq_rl, m_rl = arm_faithful(rl2, T3, cfg, bundle, panels, dates, start_ts, ema_step="daily")
    out["production_RL2"] = {
        "total_return": round(float(m_rl["total_return"]), 6),
        "max_drawdown": round(float(m_rl["max_drawdown"]), 6),
        "total_costs": round(float(m_rl["total_costs"]), 1),
        "position_util_mean": round(float(m_rl.get("position_util_mean", float("nan"))), 4),
        "n_blocked_cash": int(m_rl.get("n_blocked_cash", 0)),
        "archive_ref": "tierfix_faithful_20261008_2229.json 的 F_RL2_daily"
                       "（+1.046006 / −0.220555 / 77355.5 / 0.2482）",
    }
    print(f"  生产位 RL2 {m_rl['total_return']:+.2%} MDD {m_rl['max_drawdown']:+.2%} "
          f"util {m_rl.get('position_util_mean'):.4f}", flush=True)

    vs_prod = {}
    for name, eq in eqs.items():
        mp = monthly_log_paired(eq, eq_rl, month_ends)
        vs_prod[name] = {"n": int(mp["n"]), "mean_log": round(float(mp["mean_log"]), 6),
                         "t": round(float(mp["t"]), 4)}
        print(f"  {name:6s} vs 生产位: t={mp['t']:+.2f} mean_log {mp['mean_log']:+.6f}", flush=True)
    out["monthly_paired_vs_production_RL2"] = vs_prod

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    dest = OUT / f"phase4_negcalib_vsprod_{stamp}.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n报告 → {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
