"""判据链执行器 —— 判据冻结于 plans/judgment-chain-prereg-20261003.md，本脚本只实现。

四臂（A rule_reversal 基线 / B model_dir / C model_meta / D random 阴性对照），
双种子集，净化 walk-forward OOS → 组合层两轴（收益配对 t + MDD 容差）→
判据布尔 → verdict JSON。**判决词由用户填写，本脚本不产出采纳结论。**

断点续跑：walk-forward 预测流按 (task, seed_set) 缓存为 parquet，已存在则跳过训练。
"""
import argparse
import dataclasses
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fterm.features.universe import UniverseIndex  # noqa: E402
from fterm import contracts
from fterm.backtest.engine import BacktestEngine, Target
from fterm.backtest.two_pass import run_two_pass
from fterm.config import load_config
from fterm.data import ashare as sa
from fterm.features.build import build_dataset
from fterm.model.dataset import stack_samples
from fterm.model.validate import spearman_rank_ic, walk_forward_predictions
from fterm.risk.governor import AccountState, RiskGovernor
from fterm.risk.sizer import size_position
from fterm.strategy.primary import rule_reversal_decision

# ---- 冻结判据（预注册 §三；改这里 = 违反预注册） ----
RANKIC_MIN = 0.01
RANKIC_T_MIN = 2.0
DIRACC_MIN = 0.52
QCOV_RANGE = (0.15, 0.25)
PAIRED_T_MIN = 2.0
MDD_TOL = 0.02
COST_RATIO_TOL = 1.25
RANDOM_RANKIC_ABS = 0.01
SEED_SETS = [(7, 21, 99), (1, 2, 3)]
WF_FOLDS = 4
WF_EPOCHS = 15
SAMPLE_DENSITY = 20
RAND_SEED = 7

REPORTS = Path(__file__).resolve().parents[1] / "data" / "reports"


# ---------------------------------------------------------------------------
# 信息层：walk-forward OOS 预测流（带缓存）
# ---------------------------------------------------------------------------
def get_preds(cfg, bundle, task: str, seed_set: tuple[int, ...]) -> pd.DataFrame:
    cache = REPORTS / f"jc_preds_{task}_ss{'_'.join(map(str, seed_set))}.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        df["date"] = pd.to_datetime(df["date"])
        return df
    meta = None
    if task == "meta":
        meta = np.sign(bundle.features[f"rev_{cfg.signal.rev_window}"])
    samples = stack_samples(bundle.features, bundle.labels, bundle.feature_cols,
                            cfg.features.lookback, bundle.universe,
                            max_stocks_per_day=SAMPLE_DENSITY, meta_primary=meta)
    print(f"    样本 {len(samples)}（task={task}）", flush=True)
    cfg_r = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, task=task, seeds=seed_set))
    t0 = time.time()
    preds = walk_forward_predictions(cfg_r, samples, n_folds=WF_FOLDS,
                                     feature_cols=bundle.feature_cols, epochs=WF_EPOCHS)
    print(f"    walk-forward {len(preds)} 条预测，耗时 {time.time()-t0:.0f}s", flush=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(preds).to_parquet(cache)
    return pd.DataFrame(preds)


def info_layer_metrics(preds: pd.DataFrame, cfg=None) -> dict:
    """C1 读数：逐日截面 RankIC（均值+t）、dir_acc、分位覆盖率。

    2026-10-09（模型升级 Phase 3）扩展：追加 **ICIR / 滚动 IC 稳定性 / 制度分层 IC**
    ——用户拍板的主判据「信息层 RankIC/ICIR + 时间稳定性」。旧键全部保留（向后兼容）。
    cfg 提供时追加制度分层（趋势×波动 2×2）；缺省跳过。
    """
    from fterm.model import stats as _st

    by_day = {d: g for d, g in preds.groupby("date")}
    pairs = [(d, spearman_rank_ic(list(zip(g["p_up"], g["y_ret"])))) for d, g in by_day.items()]
    pairs = [(d, v) for d, v in pairs if v is not None]
    ic_arr = np.array([v for _, v in pairs])
    ic_mean = float(ic_arr.mean()) if len(ic_arr) else float("nan")
    ic_t = float(ic_arr.mean() / (ic_arr.std(ddof=1) / np.sqrt(len(ic_arr)))) if len(ic_arr) > 2 else float("nan")
    # dir_acc：p_up>=0.5 预测涨、<0.5 预测跌，对真实符号
    pred_sign = np.where(preds["p_up"] >= 0.5, 1, -1)
    true_sign = np.where(preds["y_ret"] >= 0, 1, -1)
    dir_acc = float((pred_sign == true_sign).mean()) if len(preds) else float("nan")
    cov = float(((preds["q_lo"] <= preds["y_ret"]) & (preds["y_ret"] <= preds["q_hi"])).mean())
    out = {"rank_ic": ic_mean, "rank_ic_t": ic_t, "n_days": len(ic_arr),
           "dir_acc": dir_acc, "q_cov": cov, "n_preds": int(len(preds))}
    # ---- Phase 3 扩展（主判据）----
    out["icir"] = _st.icir(ic_arr)
    roll = _st.rolling_ic_stability(ic_arr, window=60)
    out["rolling_nw_t"] = roll["nw_t_of_daily_ic"]
    out["frac_negative_windows"] = roll["frac_negative_windows"]
    if cfg is not None:
        try:
            from fterm.features.regime import market_regime_labels
            reg = market_regime_labels(cfg).dropna()
            ser = pd.Series([v for _, v in pairs], index=[d for d, _ in pairs])
            common = ser.index.intersection(reg.index)
            strat = _st.regime_stratified_ic(ser.loc[common].to_numpy(), reg.loc[common].to_numpy())
            out["regime_stratified"] = {k: v["mean"] for k, v in strat.items()}
            out["regime_min_ic"] = _st.regime_min_ic(strat)
        except Exception as e:      # 制度标签不可用时不影响主读数
            out["regime_error"] = str(e)
    return out


def c1_pass(m: dict) -> bool:
    return bool(m["rank_ic"] > RANKIC_MIN and m["rank_ic_t"] > RANKIC_T_MIN
                and m["dir_acc"] > DIRACC_MIN
                and QCOV_RANGE[0] <= m["q_cov"] <= QCOV_RANGE[1])


# ---------------------------------------------------------------------------
# 组合层：统一 decision pass（两遍法）
# ---------------------------------------------------------------------------
def build_bars(panels):
    raw_close = panels["close"]
    up = panels["up_limit"].reindex(columns=raw_close.columns)
    dn = panels["down_limit"].reindex(columns=raw_close.columns)
    # 复权因子随行带上：含权入账（除权除息入权益，docs/03 §38）。价格仍是真实未复权价。
    adj_pan = panels["adj"].reindex(index=raw_close.index, columns=raw_close.columns) \
        if "adj" in panels else None
    bars = {}
    for code in raw_close.columns:
        d = {
            "open": panels["open"][code], "high": panels["high"][code],
            "low": panels["low"][code], "close": panels["close"][code],
            "up_limit": up[code], "down_limit": dn[code],
        }
        if adj_pan is not None:
            d["adj"] = adj_pan[code]
        df = pd.DataFrame(d).dropna(subset=["close"])
        if len(df):
            bars[code] = df
    return bars


def make_targets(arm: str, cfg, bundle, panels, dates, rng, p_up_panel=None):
    """返回 targets_by_date 构造闭包（读 OOS 预测面板 / 规则 / 随机）。"""
    feats, universe = bundle.features, bundle.universe
    uni_mask = UniverseIndex(universe)
    raw_close = panels["close"]
    keep_n = cfg.signal.top_n * cfg.signal.exit_buffer

    def _sz(cfg_, equity, code, d):
        px = float(raw_close.at[d, code]) if pd.notna(raw_close.at[d, code]) else None
        if px is None:
            return None
        atr = feats["atr14"].at[d, code] if code in feats["atr14"].columns else np.nan
        s = size_position(cfg_, equity, px, float(atr) if np.isfinite(atr) else 0.0)
        return None if s.lots == 0 else (s.lots, px)

    def targets_for(d: pd.Timestamp, equity: float, cfg_, positions: dict):
        tgts = []
        if arm == "A":
            score, eligible = rule_reversal_decision(
                feats, universe, d, rev_window=cfg.signal.rev_window,
                quiet_vol=cfg.signal.quiet_vol, pvol_med=PVMED)
            ranked = score[eligible].sort_values(ascending=False)
        elif arm in ("B", "C"):
            if d not in p_up_panel.index:
                return tgts
            row = p_up_panel.loc[d].dropna()
            # 2026-10-08 B2：向量化（原逐格 universe.at[d, c]；见 fterm/features/universe.py）
            row = row[uni_mask.mask_for(d, row.index)]
            ranked = row[row >= 0.5].sort_values(ascending=False)   # 冻结：p_up≥0.5 门
        else:   # D random：无退出带，全换仓
            # 2026-10-08 B2：universe 侧走整数切片（原逐格 .at）；
            # raw_close 的 notna 仍逐格判定（其缺失语义与 universe 不同，不合并）
            pool = [c for c in universe.columns[uni_mask.positions(d)]
                    if pd.notna(raw_close.at[d, c])]
            pick = list(pd.Series(pool).sample(n=min(cfg.signal.top_n, len(pool)),
                                               random_state=rng))
            for code in list(positions):
                if code not in pick:
                    tgts.append(Target(code, 0, 0, "random_rotate"))
            for code in pick:
                if code in positions:
                    continue
                s = _sz(cfg_, equity, code, d)
                if s:
                    tgts.append(Target(code, 1, s[0], f"random(rank=NA)"))
            return tgts

        keep = list(ranked.head(keep_n).index)
        for code in list(positions):
            if code not in keep:
                tgts.append(Target(code, 0, 0, f"exit_band(rank_out_of_{keep_n})"))
        for code in ranked.head(cfg.signal.top_n).index:
            if code in positions:
                continue
            s = _sz(cfg_, equity, code, d)
            if s:
                score_v = float(ranked[code])
                tgts.append(Target(code, 1, s[0], f"{arm}(rank={score_v:.3f})"))
        return tgts

    # 2026-10-08 去重 + B5：本函数原还内联了一份两遍法 decision_pass，
    # 已收敛到 fterm/backtest/two_pass.py（调用方 run_arm 直接用 run_two_pass）。
    # 本函数现在只负责产出 targets_for。
    return targets_for


def run_arm(arm, cfg, bundle, panels, dates, start_ts, p_up_panel=None, rng_seed=7):
    eng_bars = build_bars(panels)
    eng = BacktestEngine(cfg, eng_bars)
    rng = np.random.RandomState(rng_seed)
    tf = make_targets(arm, cfg, bundle, panels, dates, rng, p_up_panel=p_up_panel)
    # 2026-10-08 去重 + B5：两遍法骨架收敛到 fterm/backtest/two_pass.py
    reb = [d for d in dates
           if d >= start_ts and d.weekday() == cfg.signal.rebalance_weekday]
    r, n_rej, t2 = run_two_pass(
        engine=eng, reb_days=reb, cfg=cfg, raw_close=panels["close"],
        initial_equity=cfg.account.initial_equity, start_ts=start_ts,
        targets_for=lambda d, e, p: tf(d, e, cfg, p))
    m = dict(r.metrics)
    m["n_governor_rejected"] = n_rej
    m["n_targets"] = int(sum(len(v) for v in t2.values()))
    return r, m


def paired_weekly_t(eq_c: pd.Series, eq_a: pd.Series, fridays) -> dict:
    """C2：每个调仓周五的权益环比配对差 t。"""
    rc, ra = [], []
    prev = None
    for f in fridays:
        if f not in eq_c.index or f not in eq_a.index:
            continue
        if prev is not None:
            rc.append(eq_c[f] / eq_c[prev] - 1.0)
            ra.append(eq_a[f] / eq_a[prev] - 1.0)
        prev = f
    rc, ra = np.array(rc), np.array(ra)
    diff = rc - ra
    if len(diff) < 3 or diff.std(ddof=1) == 0:
        return {"n": int(len(diff)), "mean": float(diff.mean()) if len(diff) else float("nan"),
                "t": float("nan")}
    t = float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))))
    return {"n": int(len(diff)), "mean": float(diff.mean()), "t": t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="忽略缓存重跑 walk-forward")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "config.yaml")
    REPORTS.mkdir(parents=True, exist_ok=True)

    print("== 1/5 面板与特征 ==", flush=True)
    panels = sa.load_panels(cfg)
    bundle = build_dataset(cfg)
    dates = panels["close"].index
    global PVMED
    PVMED = (bundle.features["pvol_20"].rolling(252, min_periods=126).median()
             if cfg.signal.quiet_vol else None)
    fps = {f: sa.df_fingerprint(panels[f]) for f in sorted(panels)}

    # ---- 信息层（B/C × 双种子集）----
    print("== 2/5 walk-forward OOS（B direction / C meta_reversal，双种子集）==", flush=True)
    info, preds_panels = {}, {}
    for task, tag in (("direction", "B"), ("meta", "C")):
        for ss in SEED_SETS:
            key = f"{tag}_ss{'_'.join(map(str, ss))}"
            print(f"  {key} ...", flush=True)
            if args.fresh:
                cache = REPORTS / f"jc_preds_{task}_ss{'_'.join(map(str, ss))}.parquet"
                if cache.exists():
                    cache.unlink()
            pr = get_preds(cfg, bundle, task, ss)
            info[key] = info_layer_metrics(pr)
            panel = pr.pivot_table(index="date", columns="product", values="p_up")
            preds_panels[key] = panel
            print(f"    {info[key]}", flush=True)

    oos_start = min(p.index.min() for p in preds_panels.values())
    oos_end = max(p.index.max() for p in preds_panels.values())
    print(f"  OOS 窗口: {oos_start.date()} → {oos_end.date()}", flush=True)

    # ---- 组合层（同窗口 OOS 起跑；两遍法）----
    print("== 3/5 组合层两轴（A/D 全口径，B/C 读 OOS 面板；同窗口配对）==", flush=True)
    arm_metrics, arm_equity = {}, {}
    for arm in ("A", "D"):
        r, m = run_arm(arm, cfg, bundle, panels, dates, oos_start, rng_seed=RAND_SEED)
        arm_metrics[arm], arm_equity[arm] = m, r.equity
        print(f"  {arm}: {m['final_equity']:.0f} ({m['total_return']:+.2%}) MDD {m['max_drawdown']:.2%} "
              f"成本 {m['total_costs']:.0f} 拒 {m['n_governor_rejected']}", flush=True)
    for key, panel in preds_panels.items():
        r, m = run_arm(key[0], cfg, bundle, panels, dates, oos_start,
                       p_up_panel=panel, rng_seed=RAND_SEED)
        arm_metrics[key], arm_equity[key] = m, r.equity
        print(f"  {key}: {m['final_equity']:.0f} ({m['total_return']:+.2%}) MDD {m['max_drawdown']:.2%} "
              f"成本 {m['total_costs']:.0f} 拒 {m['n_governor_rejected']}", flush=True)

    fridays = [d for d in dates if d >= oos_start and d.weekday() == cfg.signal.rebalance_weekday
               and d in arm_equity["A"].index]

    # ---- C5 阴性对照：随机分数的逐日截面 RankIC（对 fwd_ret，eligible 集合）----
    rng = np.random.RandomState(RAND_SEED)
    fwd = bundle.labels["fwd_ret"]
    # 2026-10-08 B2：universe 侧走整数切片（原逐格 bundle.universe.at）；
    # 本处与 run_arm 内的 uni_mask 不是同一作用域，故就地新建。
    uni_ic = UniverseIndex(bundle.universe)
    ics = []
    for d in fridays:
        pool = [c for c in bundle.universe.columns[uni_ic.positions(d)]
                if pd.notna(fwd.at[d, c])]
        if len(pool) < 5:
            continue
        r = rng.random_sample(len(pool))
        ic = spearman_rank_ic(list(zip(r, [float(fwd.at[d, c]) for c in pool])))
        if ic is not None:
            ics.append(ic)
    random_ic = float(np.mean(ics)) if ics else float("nan")

    # ---- 判据布尔（冻结常数）----
    print("== 4/5 判据 ==", flush=True)
    criteria = {"C5_random_ic": {"rank_ic": random_ic,
                                 "pass": bool(abs(random_ic) < RANDOM_RANKIC_ABS)}}
    cand_tags = ("B", "C")
    for tag in cand_tags:
        ss_results = {f"ss{'_'.join(map(str, ss))}": {
            "info": info[f"{tag}_ss{'_'.join(map(str, ss))}"],
            "c1": c1_pass(info[f"{tag}_ss{'_'.join(map(str, ss))}"]),
        } for ss in SEED_SETS}
        # C2/C3/C4 per seed set（组合层用该种子集的 OOS 面板）
        for ss in SEED_SETS:
            key = f"{tag}_ss{'_'.join(map(str, ss))}"
            m, eq = arm_metrics[key], arm_equity[key]
            base_m, base_eq = arm_metrics["A"], arm_equity["A"]
            p = paired_weekly_t(eq, base_eq, fridays)
            ss_results[f"ss{'_'.join(map(str, ss))}"]["C2"] = {
                **p, "pass": bool(p["t"] > PAIRED_T_MIN and p["mean"] > 0)}
            ss_results[f"ss{'_'.join(map(str, ss))}"]["C3"] = {
                "mdd": m["max_drawdown"], "mdd_base": base_m["max_drawdown"],
                "pass": bool(m["max_drawdown"] >= base_m["max_drawdown"] - MDD_TOL)}
            ss_results[f"ss{'_'.join(map(str, ss))}"]["C4"] = {
                "costs": m["total_costs"], "costs_base": base_m["total_costs"],
                "pass": bool(m["total_costs"] <= base_m["total_costs"] * COST_RATIO_TOL)}
        # 双种子集同向：C1 与 C2 在两套种子集上都 PASS
        dual = all(
            ss_results[f"ss{'_'.join(map(str, ss))}"]["c1"]
            and ss_results[f"ss{'_'.join(map(str, ss))}"]["C2"]["pass"]
            for ss in SEED_SETS)
        criteria[tag] = {"per_seed_set": ss_results, "dual_seed_agree": bool(dual),
                         "adoption_ready": bool(dual and criteria["C5_random_ic"]["pass"])}

    # ---- 落盘 ----
    print("== 5/5 报告 ==", flush=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    verdict = {
        "prereg": "plans/judgment-chain-prereg-20261003.md",
        "frozen_constants": {"RANKIC_MIN": RANKIC_MIN, "RANKIC_T_MIN": RANKIC_T_MIN,
                             "DIRACC_MIN": DIRACC_MIN, "QCOV_RANGE": list(QCOV_RANGE),
                             "PAIRED_T_MIN": PAIRED_T_MIN, "MDD_TOL": MDD_TOL,
                             "COST_RATIO_TOL": COST_RATIO_TOL,
                             "RANDOM_RANKIC_ABS": RANDOM_RANKIC_ABS,
                             "SEED_SETS": [list(s) for s in SEED_SETS],
                             "WF_FOLDS": WF_FOLDS, "WF_EPOCHS": WF_EPOCHS,
                             "SAMPLE_DENSITY": SAMPLE_DENSITY},
        "panel_fingerprints": fps,
        "oos_window": {"start": str(oos_start.date()), "end": str(oos_end.date())},
        "arm_metrics": {k: {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv)
                            for kk, vv in v.items()} for k, v in arm_metrics.items()},
        "info_layer": info,
        "criteria": criteria,
        "verdict_word": "",   # 待用户填写：采纳 / 不采纳 / 续观察
    }
    (REPORTS / f"judgment_chain_{stamp}.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [f"判据链读数 {stamp}（判据冻结于 plans/judgment-chain-prereg-20261003.md）",
             f"OOS 窗口 {oos_start.date()} → {oos_end.date()}",
             f"C5 阴性对照: 随机 RankIC {random_ic:+.4f}（门 |IC|<{RANDOM_RANKIC_ABS}）→ "
             f"{'PASS' if criteria['C5_random_ic']['pass'] else 'FAIL（全链判废）'}"]
    for tag, name in (("B", "model_dir"), ("C", "model_meta")):
        c = criteria[tag]
        lines.append(f"臂 {tag} {name}: dual_seed_agree={c['dual_seed_agree']} "
                     f"adoption_ready={c['adoption_ready']}")
        for ss, res in c["per_seed_set"].items():
            i = res["info"]
            lines.append(f"  {ss}: IC {i['rank_ic']:+.4f}(t={i['rank_ic_t']:+.1f}) "
                         f"acc {i['dir_acc']:.3f} cov {i['q_cov']:.2f} C1={res['c1']} "
                         f"C2 t={res['C2']['t']:+.2f}(n={res['C2']['n']}) C3 MDD "
                         f"{res['C3']['mdd']:.2%} vs {res['C3']['mdd_base']:.2%} "
                         f"C4 成本 {res['C4']['costs']:.0f}/{res['C4']['costs_base']:.0f}")
    lines.append("判决词：＿＿＿（由用户填写；判'采纳'须谨慎——OOS 段交易笔数有限，功效受限）")
    text = "\n".join(lines)
    (REPORTS / f"judgment_chain_{stamp}.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f"报告 → data/reports/judgment_chain_{stamp}.json/.txt", flush=True)


if __name__ == "__main__":
    main()
