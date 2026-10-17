"""判据链 #19 · 敞口持续性结构（预注册 plans/exposure-prereg-20261008.md）。

§36 已证：现行二值熔断一次触发就锁 1892 个交易日，零变动日 65.2% 与 halted 日 67.0%
几乎重合 ⇒ 此前测的不是周频策略。本链用单变量对照三种非自锁形态，并把
"非零敞口周占比"提为主判据的第一条（否则又会把长期空仓误读成风控好）。

E0 现行二值 / E1 冷却期重锚(N=20) / E2 连续降仓(L=0.20) / E3 滚动峰值(252日)
+ 阴性对照（置换标签流，敞口结构指标应基本不变）。
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
from fterm.research.scaffold import report_meta, selfcheck  # noqa: E402

import sync_index                                         # noqa: E402
import tierfix_replay as TF                               # noqa: E402
from ddhalt_chain import weekly_stats                     # noqa: E402
from monthly_chain import monthly_log_paired              # noqa: E402

REPORTS = ROOT / "data" / "reports"
EXDIR = ROOT / "experiments" / "exposure_20261008"
HALT = 0.12
COOLDOWN_N = 20
DD_LIMIT = 0.20
ROLL_WIN = 252
MIN_NONZERO_WEEKS = 0.60
# 锚点自证基准 = #17 存档 V0_RL2 的**全精度**读数。
# 教训（第二次踩同一坑，见 docs/03 §31 的 X0_REF）：参考值必须从存档取原始浮点数，
# 抄自己写在文档里的四舍五入值会把"成本 88152.0 vs 88151.7"变成 0.3 的假漂移，
# 在 1e-4 容差下直接把整条链判废——仪器错杀自己。
E0_REF_FILE = "ddhalt2"


class CooldownReanchor:
    """E1：连续 halted 满 N 个问价日 ⇒ 把峰值重锚在当前权益（风险预算随时间重置）。"""

    def __init__(self, halt=HALT, n=COOLDOWN_N):
        self.halt, self.n, self.run, self.peak = halt, n, 0, 0.0

    def __call__(self, dates, equities, cur):
        self.peak = max(self.peak, cur)
        if cur < self.peak * (1 - self.halt):
            self.run += 1
            if self.run >= self.n:
                self.peak = cur
                self.run = 0
        else:
            self.run = 0
        return self.peak


def rolling_peak(win=ROLL_WIN):
    def f(dates, equities, cur):
        return float(max(equities[-win:])) if equities else cur
    return f


def make_scaler(limit=DD_LIMIT):
    """E2：目标敞口 × clip((L − dd)/L, 0, 1)，dd 相对全历史峰值。"""
    box = {"peak": 0.0}

    def scaler(ctx):
        box["peak"] = max(box["peak"], ctx.equity)
        dd = 1.0 - ctx.equity / box["peak"] if box["peak"] > 0 else 0.0
        return float(np.clip((limit - dd) / limit, 0.0, 1.0))
    return scaler


def exposure_weeks(pos_value: pd.Series, equity: pd.Series) -> dict:
    """一等指标：非零敞口周占比（周内有任意一天 pos_value>0 即算持仓周）。"""
    pv = pos_value.resample("W-FRI").max()
    eq = equity.resample("W-FRI").last()
    j = pd.concat([pv.rename("pv"), eq.rename("eq")], axis=1).dropna()
    j = j[j["eq"] > 0]
    return {"n_weeks": int(len(j)),
            "nonzero_exposure_weeks": float((j["pv"] > 1e-6).mean()),
            "mean_weekly_util": float((j["pv"] / j["eq"]).mean())}


def shuffled_stream(panel: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """阴性对照流：日内置换预测秩（敞口规则不变，信号结构被毁）。"""
    rng = np.random.RandomState(20261008)
    out = panel.copy()
    vals = np.array(out.to_numpy(dtype="float64"), copy=True)   # parquet 零拷贝数组是只读的
    for i in range(vals.shape[0]):
        row = vals[i]
        ok = ~np.isnan(row)
        if ok.sum() > 1:
            vals[i][ok] = row[ok][rng.permutation(int(ok.sum()))]
    return pd.DataFrame(vals, index=out.index, columns=out.columns)


def load_e0_ref() -> dict:
    import json
    files = sorted(REPORTS.glob(f"{E0_REF_FILE}_*.json"))
    assert files, "找不到 #17 存档，无法自证 E0"
    v = json.loads(files[-1].read_text(encoding="utf-8"))["results"]["V0_RL2"]
    return str(files[-1].name), {k: float(v[k]) for k in
                                 ("total_return", "max_drawdown", "total_costs")}


def load_e0_ref_hfq():
    """锚点 = scripts/hfq_delta.py 的 RL2_daily_hfq（独立实现，同数据同档同语义）。"""
    files = sorted((ROOT / "experiments" / "hfq_delta_20261008").glob("hfq_delta_*.json"))
    if not files:
        raise FileNotFoundError("hfq_delta 存档不存在")
    v = json.loads(files[-1].read_text(encoding="utf-8"))["results"]["RL2_daily_hfq"]
    return str(files[-1].name), {k: float(v[k]) for k in
                                 ("total_return", "max_drawdown", "total_costs")}


def run_arm(name, panel, cfg, bundle, panels, dates, start_ts, no_halt=False):
    """no_halt=True：把二值熔断开到 99.99%（实质关闭），用于"结构对照"——
    敞口持续性若随信号崩塌，说明它是内生指标（熔断本身依赖权益），不能当独立门。"""
    kwargs = {}
    cfg_a = replace(cfg, account=replace(cfg.account, total_dd_halt=0.9999))         if no_halt and name != "E2" else cfg
    if name == "E1":
        kwargs["peak_fn"] = CooldownReanchor()
    elif name == "E2":
        cfg_a = replace(cfg, account=replace(cfg.account, total_dd_halt=0.9999))
        kwargs["exp_scaler"] = make_scaler()
    elif name == "E3":
        kwargs["peak_fn"] = rolling_peak()
    out: dict = {}
    st: dict = {}
    t0 = time.time()
    eq, m = TF.arm_faithful(panel, TF.T3, cfg_a, bundle, panels, dates, start_ts,
                             ema_step="daily", stats=st, out=out, **kwargs)
    res = out["res"]
    wx = exposure_weeks(res.pos_value, eq)
    # 基准指数统一走 sync_index（与各链同源；缺面板大声报错提示先跑 scripts/sync_index.py）
    bench = sync_index.load(cfg, "000905.SH")["close"]
    bench = bench.loc[bench.index >= start_ts].astype("float64")
    bench_ir = TF.beta_ir(eq, bench)
    r = {"total_return": round(float(m["total_return"]), 6),
         "max_drawdown": round(float(m["max_drawdown"]), 6),
         "total_costs": round(float(m["total_costs"]), 1),
         "n_trades": int(m.get("n_trades", 0)),
         "position_util_mean": round(float(m.get("position_util_mean", 0)), 4),
         "n_blocked_cash": int(m.get("n_blocked_cash", 0)),
         "n_governor_rejected": int(m.get("n_governor_rejected", 0)),
         "reject_by_reason": st.get("reject_by_reason", {}),
         **wx, **weekly_stats(eq),
         "beta": bench_ir.get("beta"), "resid_IR_ann": bench_ir.get("resid_IR_ann"),
         "ann_alpha_wk": bench_ir.get("ann_alpha_wk"),
         "tracking_err_ann": bench_ir.get("ann_vol")}
    print(f"  {name}: ret {m['total_return']:+.2%} MDD {m['max_drawdown']:+.2%} "
          f"成本 {m['total_costs']:.0f} 非零敞口周 {wx['nonzero_exposure_weeks']:.1%} "
          f"均周仓位 {wx['mean_weekly_util']:.3f} 正周 {r.get('pct_positive_weeks')} "
          f"周std {r.get('weekly_std_ann')} p05周 {r.get('p05_week')} "
          f"IR {r.get('resid_IR_ann')} blocked {r['n_blocked_cash']} "
          f"({time.time()-t0:.0f}s)", flush=True)
    eq.to_frame("equity").to_parquet(EXDIR / f"eq_{name}.parquet")
    res.pos_value.to_frame("pos_value").to_parquet(EXDIR / f"pv_{name}.parquet")
    return eq, r


def main() -> int:
    EXDIR.mkdir(parents=True, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-neg", action="store_true")
    args = ap.parse_args()
    cfg = load_config(ROOT / "config.yaml")
    panels = sa.load_panels(cfg)
    assert panels["close"].index.max() <= TF.MAX_DATA_DATE, "越过前瞻窗隔离线"
    bundle = build_dataset(cfg, compute_fingerprints=False)
    dates = panels["close"].index
    start_ts = pd.Timestamp(TF.START_OOS)
    s = pd.Series(dates, index=dates)
    month_ends = [d for d in s.groupby([dates.year, dates.month]).last() if d >= start_ts]
    rl = TF.load_panel_map(TF.RL_STREAMS)
    panel = TF.mean_panel(list(rl.values()))

    # 口径变更（含权入账，docs/03 §38）后，未含权的 #17 存档不再是合法锚点 ⇒
    # 锚点改为**独立脚本** hfq_delta 的含权读数（不同代码路径、同数据同档 ⇒ 交叉自证）
    anchor_note = "未含权 #17 存档已随口径变更失效；本链锚点 = hfq_delta 独立实现的含权读数"
    try:
        ref_file, E0_REF = load_e0_ref_hfq()
    except FileNotFoundError:
        ref_file, E0_REF = load_e0_ref()
        anchor_note += "｜⚠ 回退到未含权锚点（先跑 scripts/hfq_delta.py 再复跑本链）"
    results, curves, ok = {}, {}, True
    for name in ("E0", "E1", "E2", "E3"):
        eq, r = run_arm(name, panel, cfg, bundle, panels, dates, start_ts)
        results[name] = r
        curves[name] = eq
        if r["n_blocked_cash"] != 0:
            ok = False
            print(f"  !! {name} blocked_cash≠0 ⇒ 忠实账本未闭合", flush=True)
        if name == "E0":
            drift = max(abs(r[k] - E0_REF[k]) for k in E0_REF)
            results.update(selfcheck(
        "_selfcheck_E0_anchor", {"total_return": abs(r["total_return"] - E0_REF["total_return"]),
                                 "max_drawdown": abs(r["max_drawdown"] - E0_REF["max_drawdown"]),
                                 "total_costs": abs(r["total_costs"] - E0_REF["total_costs"])},
        1e-4, bool(drift < 1e-4),
        f"{anchor_note}｜锚点文件 {ref_file}（同数据同档同语义 ⇒ 必须逐位一致）"))
            print(f"  E0 锚点自证漂移 {drift:.2e} ⇒ "
                  f"{'✓' if drift < 1e-4 else '✗ 判废'}", flush=True)
            if drift >= 1e-4:
                ok = False
    for name in ("E1", "E2", "E3"):
        p = monthly_log_paired(curves[name], curves["E0"], month_ends)
        results[name]["paired_vs_E0"] = {"t": round(float(p["t"]), 3),
                                         "mean_log": round(float(p["mean_log"]), 6),
                                         "n": int(p["n"])}
    # 主判据三条同时成立才算改进
    for name in ("E1", "E2", "E3"):
        r, e0 = results[name], results["E0"]
        c1 = r["nonzero_exposure_weeks"] >= MIN_NONZERO_WEEKS
        c2 = (r.get("resid_IR_ann") or -9) >= (e0.get("resid_IR_ann") or -9)
        c3 = (r.get("p05_week") or -9) >= (e0.get("p05_week") or -9)
        results[name]["primary_gates"] = {"exposure_ok": bool(c1), "ir_not_worse": bool(c2),
                                          "tail_week_not_worse": bool(c3),
                                          "improved": bool(c1 and c2 and c3)}
        print(f"  {name} 主判据: 敞口{c1} IR不劣{c2} 尾部周不劣{c3} ⇒ 改进"
              f"{'✓' if (c1 and c2 and c3) else '✗'}", flush=True)
    # ---- 对照一：阴性（置换信号，现行风险层） ----
    if not args.no_neg:
        eq, r = run_arm("NEG_shuffled", shuffled_stream(panel, bundle.universe),
                        cfg, bundle, panels, dates, start_ts)
        results["NEG_shuffled"] = r
        same = abs(r["nonzero_exposure_weeks"] - results["E0"]["nonzero_exposure_weeks"])
        results["_negative_control"] = {
            "exposure_share_shuffled": r["nonzero_exposure_weeks"],
            "exposure_share_E0": results["E0"]["nonzero_exposure_weeks"],
            "abs_gap": round(float(same), 4),
            "verdict": "敞口结构指标与信号无关 ✓" if same < 0.15
            else "敞口占比随信号崩塌 ⇒ 预注册 §二对该指标的独立性假设**被证伪**"}
        print(f"  阴性对照(带熔断): {results['_negative_control']['verdict']}", flush=True)
        # ---- 对照二：关掉熔断后的结构对照（区分"没票可买"与"权益门把账户锁死"） ----
        eq, r_nohalt = run_arm("E0_nohalt", panel, cfg, bundle, panels, dates,
                               start_ts, no_halt=True)
        results["_structure_control_E0_nohalt"] = {
            k: r_nohalt[k] for k in ("total_return", "max_drawdown",
                                     "nonzero_exposure_weeks", "mean_weekly_util",
                                     "position_util_mean", "pct_positive_weeks",
                                     "resid_IR_ann")}
        eq, r_shuf_nh = run_arm("NEG_nohalt", shuffled_stream(panel, bundle.universe),
                                cfg, bundle, panels, dates, start_ts, no_halt=True)
        results["_structure_control_NEG_nohalt"] = {
            k: r_shuf_nh[k] for k in ("total_return", "max_drawdown",
                                      "nonzero_exposure_weeks", "mean_weekly_util",
                                      "pct_positive_weeks", "resid_IR_ann")}
        gap = abs(r_nohalt["nonzero_exposure_weeks"] - r_shuf_nh["nonzero_exposure_weeks"])
        results["_structure_verdict"] = {
            "E0_nohalt_exposure": r_nohalt["nonzero_exposure_weeks"],
            "NEG_nohalt_exposure": r_shuf_nh["nonzero_exposure_weeks"],
            "abs_gap": round(float(gap), 4),
            "reading": ("熔断关闭后两者敞口占比都 ≥0.7 ⇒ 空仓完全由权益门造成，"
                        "与信号质量无关 ⇒ 非零敞口周占比只能当诊断，不能当独立门"
                        if min(r_nohalt["nonzero_exposure_weeks"],
                              r_shuf_nh["nonzero_exposure_weeks"]) >= 0.7
                        else "关掉熔断后仍有臂低敞口 ⇒ 空仓另有原因（候选池/sizer/停牌），"
                             "需继续分解"),
            "criterion": "两臂 no-halt 敞口占比是否都 ≥0.7（不是看两者之差）"}
        print(f"  结构对照: {results['_structure_verdict']['reading']}", flush=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    payload = {"prereg": "plans/exposure-prereg-20261008.md", "tier": TF.T3,
               "_meta": report_meta(
                   cfg, panels, bundle, harness="faithful",
                   dividend_basis="hfq_inclusive", ema_step="daily",
                   stream_paths=list(TF.RL_STREAMS.values()),
                   window_max=panels["close"].index.max(), tier=TF.T3,
                   extra={"anchor_note": anchor_note, "arms": ["E0", "E1", "E2", "E3"]}),
               "frozen": {"COOLDOWN_N": COOLDOWN_N, "DD_LIMIT": DD_LIMIT,
                          "ROLL_WIN": ROLL_WIN, "MIN_NONZERO_WEEKS": MIN_NONZERO_WEEKS},
               "instrument_ok": bool(ok), "results": results, "verdict_word": None}
    txt = json.dumps(payload, ensure_ascii=False, indent=1, default=str)
    (REPORTS / f"exposure_{stamp}.json").write_text(txt, encoding="utf-8")
    (EXDIR / f"exposure_{stamp}.json").write_text(txt, encoding="utf-8")
    print("存档 →", REPORTS / f"exposure_{stamp}.json", flush=True)
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main())
