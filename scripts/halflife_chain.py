"""判据链 #7 · 信号半衰期与黄金调仓期执行器 —— 判据冻结于 plans/halflife-prereg-20261004.md。

①IC(h) 衰减曲线 + 半衰期（h ∈ {1,2,3,5,8,10,15,21}）；
②组合层调仓频率扫描 f ∈ {5,10,15,21}（top2/keep4，两遍法，双种子集），
f* = 总净收益均值最大者，黄金期成立 = f* 在两套种子集各自同时优于 5 与 21。
零训练（复用 #4 OOS 预测流缓存）。判决词由用户填写。
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
from fterm import contracts  # noqa: E402
from fterm.backtest.engine import BacktestEngine, Target  # noqa: E402
from fterm.backtest.two_pass import run_two_pass  # noqa: E402
from fterm.config import load_config  # noqa: E402
from fterm.data import ashare as sa  # noqa: E402
from fterm.features.build import build_dataset  # noqa: E402
from fterm.model.validate import spearman_rank_ic  # noqa: E402
from fterm.risk.governor import AccountState, RiskGovernor  # noqa: E402
from fterm.risk.sizer import size_position  # noqa: E402
from ablation_chain import CHAIN1_JSON  # noqa: E402
from judgment_chain import build_bars  # noqa: E402

REPORTS = Path(__file__).resolve().parents[1] / "data" / "reports"
PREDS = {"ss42_7": ROOT / "data/reports/lgbm_preds_ss42_7.parquet",
         "ss123_2024": ROOT / "data/reports/lgbm_preds_ss123_2024.parquet"}
START_OOS = "2019-01-01"

# ---- 冻结（预注册 §一/§二） ----
H_GRID = (1, 2, 3, 5, 8, 10, 15, 21)
F_GRID = (5, 10, 15, 21)
TOP_N = 2
KEEP_N = 4


def ic_curve(preds: pd.DataFrame, hfq_open: pd.DataFrame, h_grid) -> dict:
    """IC(h)：OOS 逐日截面 spearman(pred, fwd_ret_h) 均值与 t（向量化 join 版）。"""
    out = {}
    long_p = preds.rename(columns={"product": "code"})[["date", "code", "pred"]]
    for h in h_grid:
        fwd = (hfq_open.shift(-(h + 1)) / hfq_open.shift(-1) - 1.0)
        long_f = fwd.stack().rename("y").rename_axis(["date", "code"]).reset_index()
        m = long_p.merge(long_f, on=["date", "code"], how="inner")
        if len(m) < 100:
            out[h] = {"ic": float("nan"), "t": float("nan"), "n_days": 0}
            continue
        r1 = m.groupby("date")["pred"].rank()
        r2 = m.groupby("date")["y"].rank()
        m = m.assign(r1=r1.to_numpy(), r2=r2.to_numpy())
        # 逐日 Pearson（秩的）= E[r1 r2] − E[r1]E[r2]，除以组内标准差
        e12 = (m["r1"] * m["r2"]).groupby(m["date"]).mean()
        e1 = m.groupby("date")["r1"].mean()
        e2 = m.groupby("date")["r2"].mean()
        sd1 = m.groupby("date")["r1"].std(ddof=0)
        sd2 = m.groupby("date")["r2"].std(ddof=0)
        n = m.groupby("date").size()
        ic_daily = (e12 - e1 * e2) / (sd1 * sd2)
        ic_daily = ic_daily[n >= 5].dropna()
        arr = ic_daily.to_numpy()
        out[h] = {"ic": float(arr.mean()) if len(arr) else float("nan"),
                  "t": float(arr.mean() / (arr.std(ddof=1) / np.sqrt(len(arr)))) if len(arr) > 2 else float("nan"),
                  "n_days": len(arr)}
    return out


def half_life(curve: dict) -> str:
    ic1 = curve[1]["ic"]
    for h in sorted(curve):
        if curve[h]["ic"] < 0.5 * ic1:
            return str(h)
    return ">21"


def run_freq_arm(freq: int, p_up_panel, cfg0, bundle, panels, dates, start_ts):
    feats, universe = bundle.features, bundle.universe
    uni_mask = UniverseIndex(universe)
    raw_close = panels["close"]
    eng = BacktestEngine(cfg0, build_bars(panels))
    oos_days = [d for d in dates if d >= start_ts]
    reb_all = set(oos_days[::freq])          # 冻结：统一 cadence 族，OOS 首日起每 f 日

    def targets_for(d, equity, positions):
        if d not in p_up_panel.index:
            return []
        row = p_up_panel.loc[d].dropna()
        # 2026-10-08 B2：向量化（原逐格 universe.at[d, c]；见 fterm/features/universe.py）
        row = row[uni_mask.mask_for(d, row.index)]
        ranked = row.sort_values(ascending=False)
        keep = list(ranked.head(KEEP_N).index)
        tgts = [Target(c, 0, 0, f"exit_band(keep_{KEEP_N})") for c in list(positions) if c not in keep]
        for code in ranked.head(TOP_N).index:
            if code in positions:
                continue
            px = float(raw_close.at[d, code]) if pd.notna(raw_close.at[d, code]) else None
            if px is None:
                continue
            atr = feats["atr14"].at[d, code] if code in feats["atr14"].columns else np.nan
            s = size_position(cfg0, equity, px, float(atr) if np.isfinite(atr) else 0.0)
            if s.lots > 0:
                tgts.append(Target(code, 1, s.lots, f"f{freq}(rank={float(ranked[code]):.3f})"))
        return tgts

    # 2026-10-08 去重 + B5：两遍法骨架移到 fterm/backtest/two_pass.py
    # （原实现与本仓另 5 个 chain 脚本逐行重复 ⇒ 风控口径有静默分叉风险；
    #  peak 原为每调仓日重算前缀最大值，现用 cummax O(1) 查表）
    r, n_rej, t2 = run_two_pass(
        engine=eng, reb_days=reb_all, cfg=cfg0, raw_close=raw_close,
        initial_equity=cfg0.account.initial_equity, start_ts=start_ts,
        targets_for=lambda d, e, p: targets_for(d, e, p))
    m = dict(r.metrics)
    m["n_governor_rejected"] = n_rej
    m["n_targets"] = int(sum(len(v) for v in t2.values()))
    return m


def main():
    cfg = load_config(ROOT / "config.yaml")
    REPORTS.mkdir(parents=True, exist_ok=True)
    chain1 = json.loads(CHAIN1_JSON.read_text(encoding="utf-8"))
    start_ts = pd.Timestamp(START_OOS)

    print("== 面板/特征 ==", flush=True)
    panels = sa.load_panels(cfg)
    bundle = build_dataset(cfg)
    hfq = sa.hfq(panels)
    dates = panels["close"].index

    curves, sweeps = {}, {}
    for tag, path in PREDS.items():
        pr = pd.read_parquet(path)
        pr["date"] = pd.to_datetime(pr["date"])
        t0 = time.time()
        curves[tag] = ic_curve(pr, hfq["open"], H_GRID)
        curves[tag]["h_half"] = half_life(curves[tag])
        print(f"  {tag} 衰减曲线 {time.time()-t0:.0f}s: "
              + " ".join(f"h{h}={curves[tag][h]['ic']:+.4f}" for h in H_GRID)
              + f"  h_half={curves[tag]['h_half']}", flush=True)
        panel = pr.pivot_table(index="date", columns="product", values="pred")
        for f in F_GRID:
            m = run_freq_arm(f, panel, cfg, bundle, panels, dates, start_ts)
            sweeps.setdefault(f, {})[tag] = m
            print(f"    f={f:2d} {tag}: {m['total_return']:+.2%} "
                  f"MDD {m['max_drawdown']:+.2%} 成本 {m['total_costs']:.0f} "
                  f"目标 {m['n_targets']}", flush=True)

    # ---- f* 与黄金期判定（冻结规则） ----
    mean_ret = {f: float(np.mean([sweeps[f][t]["total_return"] for t in PREDS])) for f in F_GRID}
    f_star = max(mean_ret, key=mean_ret.get)
    golden = all(sweeps[f_star][t]["total_return"] > sweeps[5][t]["total_return"]
                 and sweeps[f_star][t]["total_return"] > sweeps[21][t]["total_return"]
                 for t in PREDS)
    if f_star == 5 or f_star == 21:
        shape = "单调（端点最优）"
    else:
        shape = "黄金期成立" if golden else "无黄金期（f* 未同时优于两端）"

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    verdict = {
        "prereg": "plans/halflife-prereg-20261004.md",
        "oos_window": {"start": START_OOS, "end": chain1["oos_window"]["end"]},
        "frozen": {"H_GRID": list(H_GRID), "F_GRID": list(F_GRID),
                   "TOP_N": TOP_N, "KEEP_N": KEEP_N},
        "ic_curves": curves,
        "sweep": {str(f): {t: {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                               for k, v in m.items()} for t, m in sweeps[f].items()}
                  for f in F_GRID},
        "mean_total_return": mean_ret,
        "f_star": f_star, "golden": bool(golden), "shape": shape,
        "verdict_word": "",   # 待用户填写
    }
    (REPORTS / f"halflife_chain_{stamp}.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [f"判据链 #7 读数 {stamp}（判据冻结于 plans/halflife-prereg-20261004.md）"]
    for tag in PREDS:
        lines.append(f"{tag}: " + " ".join(f"h{h}={curves[tag][h]['ic']:+.4f}" for h in H_GRID)
                     + f"  h_half={curves[tag]['h_half']} 交易日")
    lines.append("")
    lines.append("频率扫描（总净收益，两套种子集）:")
    for f in F_GRID:
        lines.append(f"  f={f:2d}: " + "  ".join(
            f"{t} {sweeps[f][t]['total_return']:+.2%} (MDD {sweeps[f][t]['max_drawdown']:+.2%}, "
            f"成本 {sweeps[f][t]['total_costs']:.0f})" for t in PREDS)
            + f"  均值 {mean_ret[f]:+.2%}")
    lines.append(f"f* = {f_star}（均值最大）；判定 = {shape}")
    lines.append("判决词：＿＿＿（由用户填写；f* 进生产前须 #3 门槛确认性预注册）")
    text = "\n".join(lines)
    (REPORTS / f"halflife_chain_{stamp}.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f"报告 → data/reports/halflife_chain_{stamp}.json/.txt", flush=True)


if __name__ == "__main__":
    main()
