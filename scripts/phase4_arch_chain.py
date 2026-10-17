"""判据链 · Phase 4 组合层校验（模型升级批，2026-10-10）。

预注册 `plans/model-upgrade-prereg-20261009.md §三`：

    组合层（第二轴，Phase 4）：`arm_faithful`（真 T3=100 万/top10/keep20 +
    daily EMA + blocked_cash=0）相对 F0 的 **月度配对 t ≥ 2** 且 **MDD ≥ −30%**
    且 **周 std 不劣** 且 **blocked_cash = 0**。

## 为什么需要这一步

信息层（Phase 2/3）只证明"预测排序里有增量信息"；本项目的采纳纪律要求
**组合层第二轴**独立成立（docs/03 §40/§43：信息层读数与组合层读数会分叉，
例如 #18 的 M1 信息层 t=+1.85 而组合层 t=−0.04）。

## 与既有执行器的关系

- 组合层引擎**逐字复用** `scripts/tierfix_replay.py::arm_faithful`
  （忠实单遍账本：决策在引擎循环内问真实权益/现金/持仓，blocked_cash 必须为 0）
  与 `scripts/monthly_chain.py::monthly_log_paired`（月末对数收益差配对 t）。
  本脚本**不复制**这两处的任何口径，只负责把 Phase 2/3 产出的 OOS 预测流
  接成 `arm_faithful` 需要的 date×product 面板，并出四门读数。
- 本脚本**不训练**、不改判据常数、不碰生产配置。

## 用法
  python scripts/phase4_arch_chain.py --arms D E2     # 对指定臂跑组合层
  python scripts/phase4_arch_chain.py --arms D --no-neg   # 跳过阴性对照（省时）
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

from fterm.config import load_config                      # noqa: E402
from fterm.data import ashare as sa                       # noqa: E402
from fterm.features.build import build_dataset            # noqa: E402

import sync_index                                        # noqa: E402
import dlv2_hurdle_chain as HC                            # noqa: E402
from monthly_chain import monthly_log_paired              # noqa: E402
from tierfix_replay import (arm_faithful, T3, START_OOS,  # noqa: E402
                            MAX_DATA_DATE)

OUT_DIR = ROOT / "experiments" / "arch_20261009"
F0_STREAM = ROOT / "experiments" / "external_20261009" / "stream_F0_f4.parquet"
MDD_FLOOR = -0.30          # 预注册 §三 组合层第二项
C2A_T = HC.C2A_T           # 月度配对 t 门槛（= 2.0）


def stream_to_panel(path: Path) -> pd.DataFrame:
    """OOS 预测流（date,product,pred）→ date×product 面板（arm_faithful 的输入口径）。

    与 `tierfix_replay.load_panel_map` 逐字同口径（pivot_table + 右端点守卫）。
    """
    df = pd.read_parquet(path, columns=["date", "product", "pred"])
    df["date"] = pd.to_datetime(df["date"])
    hi = df["date"].max()
    assert hi <= MAX_DATA_DATE, f"{path.name}: 预测流含 >{MAX_DATA_DATE.date()} 的数据 {hi}"
    if df["pred"].notna().sum() != len(df):
        raise ValueError(f"仪器门失败：{path.name} 含 NaN，读数作废")
    return df.pivot_table(index="date", columns="product", values="pred")


def weekly_std(eq: pd.Series) -> float:
    """周频收益标准差（预注册 §三「周 std 不劣」）。"""
    r = eq.resample("W-FRI").last().pct_change().dropna()
    return float(r.std(ddof=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["D"])
    ap.add_argument("--no-neg", action="store_true", help="跳过阴性对照")
    ap.add_argument("--with-f0day", action="store_true",
                    help="额外给 F0day 组合层基准（erratum §四-1 要求双基准）")
    args = ap.parse_args()

    cfg = load_config(ROOT / "config.yaml")
    print("== 面板 / 特征 ==", flush=True)
    panels = sa.load_panels(cfg)
    assert panels["close"].index.max() <= MAX_DATA_DATE, "面板已越过前瞻窗隔离线"
    bundle = build_dataset(cfg, compute_fingerprints=False)
    dates = panels["close"].index
    start_ts = pd.Timestamp(START_OOS)
    s = pd.Series(dates, index=dates)
    month_ends = [d for d in s.groupby([dates.year, dates.month]).last() if d >= start_ts]

    bench = sync_index.load(cfg, "000905.SH")["close"]
    bench = bench.loc[bench.index >= start_ts].astype("float64")

    # ---- 参照臂：F0（随机混洗基线）。预注册 §三明定组合层相对 F0。----
    print("== 参照臂 F0（组合层，daily EMA，真 T3）==", flush=True)
    t0 = time.time()
    panel_f0 = stream_to_panel(F0_STREAM)
    eq_f0, m_f0 = arm_faithful(panel_f0, T3, cfg, bundle, panels, dates, start_ts,
                               ema_step="daily")
    print(f"  F0: {m_f0['total_return']:+.2%} MDD {m_f0['max_drawdown']:+.2%} "
          f"blocked_cash {m_f0.get('n_blocked_cash')} ({time.time()-t0:.0f}s)", flush=True)
    if int(m_f0.get("n_blocked_cash", 0)) != 0:
        print("  !! F0 有 blocked_cash ⇒ 忠实账本未闭合，本批读数不作数", flush=True)

    results: dict = {
        "prereg": "plans/model-upgrade-prereg-20261009.md#§三-组合层",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "tier": T3, "ema_step": "daily", "start_ts": str(start_ts.date()),
        "mdd_floor": MDD_FLOOR, "monthly_t_threshold": C2A_T,
        "F0": {
            "total_return": round(float(m_f0["total_return"]), 6),
            "max_drawdown": round(float(m_f0["max_drawdown"]), 6),
            "total_costs": round(float(m_f0["total_costs"]), 1),
            "n_blocked_cash": int(m_f0.get("n_blocked_cash", 0)),
            "position_util_mean": round(float(m_f0.get("position_util_mean", float("nan"))), 4),
            "weekly_std": round(weekly_std(eq_f0), 6),
        },
    }
    print(f"  F0 周 std {results['F0']['weekly_std']:.6f} "
          f"均仓位 {results['F0']['position_util_mean']:.3f}", flush=True)

    # ---- 第二基准 F0day（erratum §四-1：D/E 必须同时给 vs F0 与 vs F0day）----
    # 组合层对齐同一形状：月度配对 t 也须给出两条。
    eq_f0day = None
    if args.with_f0day:
        p_f0d = OUT_DIR / "stream_F0day.parquet"
        if p_f0d.exists():
            print("== 第二基准 F0day（组合层，daily EMA，真 T3）==", flush=True)
            t0 = time.time()
            eq_f0day, m_f0d = arm_faithful(stream_to_panel(p_f0d), T3, cfg, bundle,
                                           panels, dates, start_ts, ema_step="daily")
            results["F0day"] = {
                "total_return": round(float(m_f0d["total_return"]), 6),
                "max_drawdown": round(float(m_f0d["max_drawdown"]), 6),
                "total_costs": round(float(m_f0d["total_costs"]), 1),
                "n_blocked_cash": int(m_f0d.get("n_blocked_cash", 0)),
                "position_util_mean": round(float(m_f0d.get("position_util_mean", float("nan"))), 4),
                "weekly_std": round(weekly_std(eq_f0day), 6),
            }
            print(f"  F0day: {m_f0d['total_return']:+.2%} MDD {m_f0d['max_drawdown']:+.2%} "
                  f"周std {results['F0day']['weekly_std']:.6f} "
                  f"blocked_cash {results['F0day']['n_blocked_cash']} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if int(m_f0d.get("n_blocked_cash", 0)) != 0:
                print("  !! F0day 有 blocked_cash ⇒ 忠实账本未闭合", flush=True)

    for arm in args.arms:
        p = OUT_DIR / f"stream_{arm}.parquet"
        if not p.exists():
            print(f"  跳过 {arm}：缺 {p}", flush=True)
            continue
        print(f"== 臂 {arm}（组合层，daily EMA，真 T3）==", flush=True)
        t0 = time.time()
        panel = stream_to_panel(p)
        eq, m = arm_faithful(panel, T3, cfg, bundle, panels, dates, start_ts,
                             ema_step="daily")
        wstd = weekly_std(eq)
        mp = monthly_log_paired(eq, eq_f0, month_ends)
        rec = {
            "total_return": round(float(m["total_return"]), 6),
            "max_drawdown": round(float(m["max_drawdown"]), 6),
            "total_costs": round(float(m["total_costs"]), 1),
            "n_blocked_cash": int(m.get("n_blocked_cash", 0)),
            "position_util_mean": round(float(m.get("position_util_mean", float("nan"))), 4),
            "weekly_std": round(wstd, 6),
            "monthly_paired_vs_F0": {
                "n": int(mp["n"]),
                "mean_log": round(float(mp["mean_log"]), 6),
                "t": round(float(mp["t"]), 4),
            },
            "gates": {
                "C_monthly_t>=2": bool(mp["t"] >= C2A_T),
                "C_mdd>=-30%": bool(float(m["max_drawdown"]) >= MDD_FLOOR),
                "C_weekly_std_not_worse": bool(wstd <= results["F0"]["weekly_std"] + 1e-12),
                "C_blocked_cash==0": bool(int(m.get("n_blocked_cash", 0)) == 0),
            },
        }
        # 第二基准（erratum §四-1）：同一形状再给一条 vs F0day 的月度配对 t。
        if eq_f0day is not None:
            mpd = monthly_log_paired(eq, eq_f0day, month_ends)
            rec["monthly_paired_vs_F0day"] = {
                "n": int(mpd["n"]),
                "mean_log": round(float(mpd["mean_log"]), 6),
                "t": round(float(mpd["t"]), 4),
            }
            rec["gates"]["C_monthly_t>=2_vs_F0day"] = bool(mpd["t"] >= C2A_T)
            rec["gates"]["C_weekly_std_not_worse_vs_F0day"] = bool(
                wstd <= results["F0day"]["weekly_std"] + 1e-12)
        rec["adoption_ready"] = bool(all(
            v for k, v in rec["gates"].items() if k != "C_weekly_std_not_worse_vs_F0day"))
        results[arm] = rec
        g = rec["gates"]
        print(f"  {arm}: {m['total_return']:+.2%} MDD {m['max_drawdown']:+.2%} "
              f"月度配对 t={mp['t']:+.2f} (n={mp['n']}) 周std {wstd:.6f} "
              f"blocked_cash {rec['n_blocked_cash']}", flush=True)
        if "monthly_paired_vs_F0day" in rec:
            print(f"      vs F0day: 月度配对 t={rec['monthly_paired_vs_F0day']['t']:+.2f} "
                  f"mean_log {rec['monthly_paired_vs_F0day']['mean_log']:+.6f}", flush=True)
        print(f"  门：月度t>=2 {g['C_monthly_t>=2']} | MDD>=-30% {g['C_mdd>=-30%']} | "
              f"周std不劣 {g['C_weekly_std_not_worse']} | blocked_cash=0 "
              f"{g['C_blocked_cash==0']} ⇒ adoption_ready={rec['adoption_ready']}",
              flush=True)
        if rec["n_blocked_cash"] != 0:
            print(f"  !! {arm} 有 blocked_cash ⇒ 忠实账本未闭合，读数不作数", flush=True)

    # ---- 阴性对照（闸门 A5）：置换预测秩 ⇒ 组合层读数必须显著塌掉 ----
    #
    # ⚠️ 口径更正（2026-10-10，Lead）：初版把门槛写成 `real - shuffled > 0.10`——
    # 该 0.10 是从 #15 的 RL2 臂（真实 +104.6% vs 置换 −5.5%，gap 1.10）抄来的**绝对差**，
    # 属于「跨语境错配 bar」（见记忆 acceptance-bar-calibrate-first）。
    # 本批的臂信息量远小于 RL2（D 的全截面 IC 仅 +0.056），绝对差必然小得多，
    # 用 0.10 判会把「信号弱」误读成「仪器无分辨率」。
    # 正确口径：**同一流自置换的分布 z 值**（5 个置换种子），并**另行标定仪器分辨率**
    # ——只要有一个已知有信号的流（F0day）z 显著为正，仪器就有分辨率。
    if not args.no_neg:
        from exposure_chain import shuffled_stream
        arm0 = args.arms[0]
        if arm0 in results:
            print(f"== 阴性对照：置换预测秩（{arm0} 流，5 个置换种子）==", flush=True)
            t0 = time.time()
            panel0 = stream_to_panel(OUT_DIR / f"stream_{arm0}.parquet")
            real = float(results[arm0]["total_return"])
            neg_rets = []
            for seed in (1, 2, 3, 4, 5):
                rng = np.random.RandomState(seed)
                v = panel0.to_numpy(dtype="float64", copy=True)
                for i in range(v.shape[0]):
                    row = v[i]
                    ok = ~np.isnan(row)
                    if ok.sum() > 1:
                        v[i][ok] = row[ok][rng.permutation(int(ok.sum()))]
                neg_panel = pd.DataFrame(v, index=panel0.index, columns=panel0.columns)
                _eq, m_neg = arm_faithful(neg_panel, T3, cfg, bundle, panels, dates,
                                          start_ts, ema_step="daily")
                neg_rets.append(float(m_neg["total_return"]))
                print(f"   置换种子 {seed}: {m_neg['total_return']:+.2%}", flush=True)
            na = np.asarray(neg_rets)
            sd = float(na.std(ddof=1))
            z = (real - float(na.mean())) / sd if sd > 0 else float("nan")
            results["_negative_control"] = {
                "arm": arm0,
                "real_return": round(real, 6),
                "shuffled_returns": [round(x, 6) for x in neg_rets],
                "shuffled_mean": round(float(na.mean()), 6),
                "shuffled_std": round(sd, 6),
                "z_of_real_vs_shuffled": round(z, 3),
                "note": "|z|<2 ⇒ 该臂组合层读数与其自身置换分布不可区分；"
                        "这只是『无证据』，不等于『证明有害』。仪器分辨率由 F0day 的 z 另行标定。",
            }
            print(f"   真实 {real:+.2%} vs 置换 mean {na.mean():+.2%} sd {sd:.2%} "
                  f"⇒ z={z:+.2f} ({time.time()-t0:.0f}s)", flush=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out = OUT_DIR / f"phase4_{stamp}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str),
                   encoding="utf-8")
    print(f"报告 → {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
