"""判据链 · 稳定性读数执行器（模型升级 Phase 3，2026-10-09）。

把「信息层 RankIC/ICIR + 时间稳定性」这一主判据落成可复用读数：对任意一组
OOS 预测流（parquet: date, product, pred, y_ret）输出

  1. IC 汇总：均值 / Newey-West t / ICIR / 正比例
  2. 滚动稳定性：60 日滚动 IC、NW-t、负窗口占比
  3. 制度分层 IC：趋势 × 波动 2×2 四层（最小值 > 0 为通过）
  4. 多流对比：逐日配对 t + PBO（组合对称交叉验证）
  5. **CPCV 稳定性**（2026-10-10 补，docs/03 §45 已知缺口 #1）：预注册
     `plans/model-upgrade-prereg-20261009.md §三` 的「`P(IC>0) ≥ 0.9` 且路径间
     IC 标准差低于 F0」门 —— 本脚本是**该门唯一的执行器**（`fterm/model/cpcv.py`
     此前零调用点）。多流时同时给出每臂相对 F0 的门读数。

用法：
  python scripts/stability_chain.py --streams a.parquet b.parquet [--names F0 A]

## 两处口径澄清（2026-10-10，必读；均**只增键、不改旧值**）

① **滚动 NW-t（§45 缺口 #3）**：`rolling.nw_t` 历史上装的是**日频 IC 的 NW-t**
   （与 `ic_summary.t` 同式，实测逐位相等：F0 两处均 +14.110418023514088）——
   名不副实，且与上层摘要**同源、不构成独立信息**。
   **旧键 `nw_t` 数值一字未改**（`stability_arch_20261010.json` 须逐位可复现，
   §45 正是引它作证）。新增 `nw_t_of_rolling_mean`（对 60 日滚动均值序列取 NW-t，
   即该门真正要的读数）与 `nw_t_of_rolling_mean_lag59`（重叠窗口的稳健滞后阶数）。
   `longest_negative_run` 为 §三「无长段负 IC」的直读量。

② **CPCV 门**：判据取**划分级**读数而非路径级 —— 单流下路径级恒退化为
   `P=1.0 / std=0`（见 `fterm/model/cpcv.py::cpcv_ic_stability` 文档），
   若照抄路径级则任何臂无条件通过（与缺口 #3 同型陷阱）。路径级读数一并留痕。
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fterm.config import load_config           # noqa: E402
from fterm.features.regime import market_regime_labels  # noqa: E402
from fterm.model import cpcv as cv              # noqa: E402
from fterm.model import stats as st             # noqa: E402
from fterm.model.validate import spearman_rank_ic  # noqa: E402

OUT_DIR = ROOT / "experiments"
ROLL_WINDOW = 60          # §三「60 日滚动」；与既有调用一致
CPCV_N_GROUPS = 6         # §三 φ=C(N,k)·k/N ⇒ N=6,k=2 ⇒ φ=5
CPCV_N_TEST_GROUPS = 2
CPCV_P_MIN = 0.9          # §三「P(IC>0) ≥ 0.9」


def ic_series(df: pd.DataFrame) -> pd.Series:
    """逐日截面 Spearman IC 序列。"""
    out = {}
    for d, g in df.groupby("date"):
        v = spearman_rank_ic(list(zip(g["pred"], g["y_ret"])))
        if v is not None:
            out[d] = v
    return pd.Series(out).sort_index()


def daily_ls_return(df: pd.DataFrame, q: float = 0.1) -> pd.Series:
    """每日多空分位差收益（top q 均值 − bottom q 均值），供 PBO 用。"""
    rows = {}
    for d, g in df.groupby("date"):
        if len(g) < 20:
            continue
        k = max(1, int(len(g) * q))
        s = g.sort_values("pred")
        rows[d] = float(s["y_ret"].tail(k).mean() - s["y_ret"].head(k).mean())
    return pd.Series(rows).sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", nargs="+", required=True)
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(ROOT / "config.yaml")
    names = args.names or [Path(p).stem for p in args.streams]
    reg = market_regime_labels(cfg).dropna()

    streams, ics = {}, {}
    for name, p in zip(names, args.streams):
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"])
        streams[name] = df
        ics[name] = ic_series(df)

    report: dict = {"generated_at": datetime.now().isoformat(timespec="seconds"),
                    "streams": {}, "pairs": {}}
    for name in names:
        ic = ics[name]
        s = st.summarize_ic(ic.to_numpy())
        roll = st.rolling_ic_stability(ic.to_numpy(), window=ROLL_WINDOW)
        common = ic.index.intersection(reg.index)
        strat = st.regime_stratified_ic(ic.loc[common].to_numpy(), reg.loc[common].to_numpy())
        report["streams"][name] = {
            "ic_summary": s,
            "rolling": {"nw_t": roll["nw_t_of_daily_ic"],
                        "frac_negative_windows": roll["frac_negative_windows"],
                        # 2026-10-10 新增（§45 缺口 #3）：旧键 nw_t 是**日频 IC 的 NW-t**，
                        # 与 ic_summary.t 同源。以下两键才是「滚动 IC 序列」的 NW-t。
                        "nw_t_of_rolling_mean": roll["nw_t_of_rolling_mean"],
                        "nw_t_of_rolling_mean_lag59": st.rolling_ic_nw_t(
                            ic.to_numpy(), window=ROLL_WINDOW, lags=ROLL_WINDOW - 1),
                        "n_rolling": roll["n_rolling"],
                        "longest_negative_run": st.longest_negative_run(
                            roll["rolling"])},
            "regime_stratified": strat,
            "regime_min_ic": st.regime_min_ic(strat),
            "n_days": int(len(ic)),
        }

    # ---- §三 CPCV 稳定性门（2026-10-10 补；§45 缺口 #1）----
    # 判据取**划分级**读数：单流下路径级 `path_ic_std` 恒为 0.0，照抄则 C2 变成
    # `0.0 < 0.0` ⇒ **恒假**，门对任何臂（含 F0 自比）无条件失败。
    # 读数不可算（流太短等）时**响亮留痕且不产出 pass=True**（失败安全）——
    # 与 arch_chain 的 `tail` / judgment_chain 的 `regime_error` 同款处置，
    # 保证新增门不会让既有调用崩溃、也不会静默报"通过"。
    cpcv_err = {}
    for name in names:
        try:
            report["streams"][name]["cpcv"] = cv.cpcv_ic_stability(
                ics[name].to_numpy(), n_groups=CPCV_N_GROUPS,
                n_test_groups=CPCV_N_TEST_GROUPS)
        except Exception as e:      # 留痕，不静默
            cpcv_err[name] = f"{type(e).__name__}: {e}"
            report["streams"][name]["cpcv"] = {"error": cpcv_err[name]}
            print(f"⚠ CPCV 门读数不可算（{name}）：{cpcv_err[name]}", flush=True)
    if len(names) >= 2:
        base = names[0]
        f0_cpcv = report["streams"][base]["cpcv"]
        gate_arms = {}
        for n in names:
            if n == base:
                continue
            if "error" in report["streams"][n]["cpcv"] or "error" in f0_cpcv:
                gate_arms[n] = {"pass": False, "error": "读数不可算（见 streams.*.cpcv.error）"}
            else:
                gate_arms[n] = cv.cpcv_gate(report["streams"][n]["cpcv"], f0_cpcv,
                                            p_min=CPCV_P_MIN)
        report["cpcv_gate"] = {
            "reference": base,
            "p_min": CPCV_P_MIN,
            "n_groups": CPCV_N_GROUPS, "n_test_groups": CPCV_N_TEST_GROUPS,
            "reading_level": "split（划分级；单流下路径级退化，见 cpcv_ic_stability）",
            "arms": gate_arms,
        }
        if cpcv_err:
            report["cpcv_gate"]["errors"] = cpcv_err

    if len(names) >= 2:
        base = names[0]
        for other in names[1:]:
            common = ics[base].index.intersection(ics[other].index)
            pt = st.paired_t(ics[other].loc[common].to_numpy(), ics[base].loc[common].to_numpy())
            report["pairs"][f"{other}_minus_{base}"] = pt

    if len(names) >= 2:
        ls = pd.concat({n: daily_ls_return(streams[n]) for n in names}, axis=1).dropna()
        if len(ls) >= 16:
            report["pbo"] = st.pbo_cscv(ls.to_numpy(), n_blocks=8)
        for name in names:
            if name in ls:
                r = ls[name]
                dsr = st.deflated_sharpe(
                    sharpe=float(r.mean() / (r.std(ddof=1) + 1e-12)),
                    n_trials=len(names), n_obs=len(r))
                report["streams"][name]["ls_sharpe_daily"] = float(r.mean() / (r.std(ddof=1) + 1e-12))
                report["streams"][name]["deflated_sharpe"] = {"dsr": dsr.dsr, "sr0": dsr.sr0}

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out = Path(args.out) if args.out else OUT_DIR / f"stability_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    lines = [f"稳定性读数 {stamp}"]
    for name in names:
        r = report["streams"][name]
        s = r["ic_summary"]
        c = r["cpcv"]
        lines.append(
            f"{name:6s} IC {s['mean']:+.4f} t {s['t']:+.2f} ICIR {s['icir']:+.3f} "
            f"正比例 {s['pos_frac']:.2f} | 滚动NW-t {r['rolling']['nw_t']:+.2f} "
            f"负窗 {r['rolling']['frac_negative_windows']:.2f} | regime最小IC {r['regime_min_ic']}")
        lines.append(
            f"       [滚动IC门] 日频IC_NW-t {r['rolling']['nw_t']:+.2f}（=上层t，同源）"
            f" → 滚动均值NW-t {r['rolling']['nw_t_of_rolling_mean']:+.2f}"
            f"（lag59 {r['rolling']['nw_t_of_rolling_mean_lag59']:+.2f}）"
            f" | 最长负段 {r['rolling']['longest_negative_run']} 日")
        if "error" in c:
            lines.append(f"       [CPCV门] 不可算：{c['error']}")
        else:
            lines.append(
                f"       [CPCV门] 划分级 P(IC>0) {c['p_ic_pos_splits']:.3f} "
                f"std {c['split_ic_std']:.6f}（{c['n_splits']} 划分）"
                f" | 路径级 P {c['p_ic_pos_paths']:.3f} std {c['path_ic_std']:.6f}"
                f"（{c['n_paths']} 路径，退化={c['path_std_degenerate']}）")
    if "cpcv_gate" in report:
        g = report["cpcv_gate"]
        lines.append(f"CPCV 门（基准 {g['reference']}，门槛 P≥{g['p_min']}，{g['reading_level']}）：")
        for n, v in g["arms"].items():
            if "error" in v:
                lines.append(f"  {n:6s} 不可算：{v['error']}（门 = False）")
                continue
            lines.append(
                f"  {n:6s} C1 P(IC>0)={v['c1_p_ic_pos_ge']['value']:.3f} "
                f"≥{g['p_min']} → {v['c1_p_ic_pos_ge']['ok']} | "
                f"C2 std {v['c2_split_ic_std_below_f0']['value']:.6f} "
                f"< F0 {v['c2_split_ic_std_below_f0']['f0']:.6f} → "
                f"{v['c2_split_ic_std_below_f0']['ok']} | 门 = {v['pass']}")
    for k, v in report["pairs"].items():
        lines.append(f"配对 {k}: ΔIC {v['mean_diff']:+.4f} t {v['t']:+.2f} n={v['n']}")
    if "pbo" in report:
        lines.append(f"PBO = {report['pbo'].get('pbo')}（splits={report['pbo'].get('n_splits')}）")
    text = "\n".join(lines)
    print(text, flush=True)
    print(f"报告 → {out}", flush=True)


if __name__ == "__main__":
    main()
