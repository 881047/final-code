"""量化"含权入账"口径变更对组合层读数的影响（docs/03 §38 的 erratum 取证）。

不覆盖既有存档：产物全部写 `experiments/hfq_delta_20261008/`。
对照对象 = #19 的 E0/E2 与 #15 的 F_* 读数（都出自 JSON 存档，未含权口径）。
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path("D:/Final")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fterm.config import load_config                     # noqa: E402
from fterm.data import ashare as sa                      # noqa: E402
from fterm.features.build import build_dataset           # noqa: E402

import sync_index                                         # noqa: E402
import tierfix_replay as TF                              # noqa: E402
from ddhalt_chain import weekly_stats                    # noqa: E402
from exposure_chain import exposure_weeks                # noqa: E402
from monthly_chain import monthly_log_paired             # noqa: E402

OUTDIR = ROOT / "experiments" / "hfq_delta_20261008"


def load_prev(dir_name: str, key: str, pinned: str | None = None) -> dict:
    """读上一代存档读数。pinned 指定文件名时钉死对象（防 glob-latest 变成循环对比）。"""
    if pinned:
        p = ROOT / "data" / "reports" / pinned
        assert p.exists(), f"钉定的上一代存档不存在: {p}"
        files = [p]
    else:
        files = sorted((ROOT / "data" / "reports").glob(f"{dir_name}_*.json"))
        assert files, "找不到上一代存档"
    d = json.loads(files[-1].read_text(encoding="utf-8"))["results"]
    return {k: float(d[key][k]) for k in
            ("total_return", "max_drawdown", "total_costs")}, str(files[-1].name)


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "config.yaml")
    panels = sa.load_panels(cfg)
    assert panels["close"].index.max() <= TF.MAX_DATA_DATE
    bundle = build_dataset(cfg, compute_fingerprints=False)
    dates = panels["close"].index
    start_ts = pd.Timestamp(TF.START_OOS)
    s = pd.Series(dates, index=dates)
    month_ends = [d for d in s.groupby([dates.year, dates.month]).last() if d >= start_ts]
    rl = TF.mean_panel(list(TF.load_panel_map(TF.RL_STREAMS).values()))
    mse = TF.load_panel_map(TF.MSE_STREAMS)
    mse2 = TF.mean_panel([mse["ss7_21_99_5_13"], mse["ss123_2024_42_314_2027"]])
    # 基准指数统一走 sync_index（与 expo/其他链同源，不再内联 SQL）
    bench = sync_index.load(cfg, "000905.SH")["close"]
    bench = bench.loc[bench.index >= start_ts].astype("float64")

    arms = {"RL2_daily_hfq": (rl, "daily"),      # （旧写法 "rebalance" and "daily" 恒为
                "MSE2_daily_hfq": (mse2, "daily"),    #  "daily"——2026-10-08 审查批去混淆）
                "RL2_weekly_hfq": (rl, "rebalance")}
    out, curves = {}, {}
    for name, (panel, step) in arms.items():
        out_box: dict = {}
        t0 = time.time()
        eq, m = TF.arm_faithful(panel, TF.T3, cfg, bundle, panels, dates, start_ts,
                                 ema_step=step, out=out_box)
        res = out_box["res"]
        wx = exposure_weeks(res.pos_value, eq)
        ir = TF.beta_ir(eq, bench)
        out[name] = {"ema_step": step,
                     "total_return": round(float(m["total_return"]), 6),
                     "max_drawdown": round(float(m["max_drawdown"]), 6),
                     "total_costs": round(float(m["total_costs"]), 1),
                     "n_trades": int(m.get("n_trades", 0)),
                     "n_blocked_cash": int(m.get("n_blocked_cash", 0)),
                     **wx, **weekly_stats(eq),
                     "beta": ir.get("beta"), "resid_IR_ann": ir.get("resid_IR_ann")}
        curves[name] = eq
        print(f"  {name}: ret {m['total_return']:+.2%} MDD {m['max_drawdown']:+.2%} "
              f"成本 {m['total_costs']:.0f} 非零敞口周 {wx['nonzero_exposure_weeks']:.1%} "
              f"正周 {out[name]['pct_positive_weeks']:.3f} IR {ir.get('resid_IR_ann')} "
              f"blocked {out[name]['n_blocked_cash']} ({time.time()-t0:.0f}s)", flush=True)

    # 与上一代仪器存档对比 ⇒ 谱系差。钉死文件防循环：exposure E0 的锚点就是本脚本
    # 的 RL2_daily_hfq，若 glob-latest，重跑 exposure 后 delta 恒为 0。
    # 注：exposure_20261008_1939 已是含权口径 ⇒ 本 delta ≠ 口径差（§38 已测得 +3.00pp/年），
    # 而是"上一代（审查批前、未登记环境）vs 本代（干净树 + 登记环境）"的仪器谱系差。
    prev, prev_file = load_prev("exposure_20261008", "E0", pinned="exposure_20261008_1939.json")
    d = {k: round(out["RL2_daily_hfq"][k] - prev[k], 6) for k in prev}
    years = (dates[-1] - dates[int(dates.searchsorted(start_ts))]).days / 365.25
    out["_delta_vs_prev_generation"] = {
        "new": "RL2_daily_hfq", "prev_file": prev_file,
        "prev_kind": "上一代仪器含权存档（审查批前，环境未登记）",
        "prev_archive": prev, "delta": d,
        "annualized_pp": round(
            (out["RL2_daily_hfq"]["total_return"]
             - prev["total_return"]) / years * 100, 2)}
    print(f"  谱系差(vs {prev_file})：总收益 {d['total_return']:+.4f}（≈"
          f"{out['_delta_vs_prev_generation']['annualized_pp']:+.2f}pp/年）"
          f" 成本 {d['total_costs']:+.0f}", flush=True)
    p = monthly_log_paired(curves["RL2_daily_hfq"], curves["MSE2_daily_hfq"], month_ends)
    out["_P2_hfq_daily"] = {"t": round(float(p["t"]), 4),
                            "mean_log": round(float(p["mean_log"]), 6), "n": int(p["n"]),
                            "pass": bool(p["t"] >= 2.0 and p["mean_log"] > 0)}
    print(f"  P2(含权, daily 步进) RL2−MSE2: t={p['t']:+.2f} n={p['n']}", flush=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    (OUTDIR / f"hfq_delta_{stamp}.json").write_text(
        json.dumps({"note": "含权入账口径差取证；不覆盖 #15/#19 存档。"
                            "2026-10-08 审查批后重跑：本 delta = 含权口径差 + "
                            "arm_faithful 空预测日修复的尾部仪器差（docs/03 §41）",
                    "results": out, "verdict_word": None},
                   ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for k, eq in curves.items():
        eq.to_frame("equity").to_parquet(OUTDIR / f"eq_{k}.parquet")
    print("存档 →", OUTDIR, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
