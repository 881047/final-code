"""判据链 #2 · Phase 1 归因消融执行器 —— 判据冻结于 plans/ablation-prereg-20261004.md。

七臂（A0 锚 / S1 去静默闸 / S2 去退出带 / S3 top4 分散 / S4 top1 集中 /
E 静态持有地板 / D 随机阴性对照），同一 OOS 窗口（读判据链 #1 verdict JSON），
两遍法回测，全部规则臂零训练。新基线候选 S* 按冻结顺序 S1>S2>S3>S4 取
第一个"周配对差 vs D 均值>0 且 t>0"者；判决词由用户填写。
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
from fterm.risk.governor import AccountState, RiskGovernor  # noqa: E402
from fterm.risk.sizer import size_position  # noqa: E402
from fterm.strategy.primary import rule_reversal_decision  # noqa: E402
from judgment_chain import build_bars, paired_weekly_t, RAND_SEED  # noqa: E402

REPORTS = Path(__file__).resolve().parents[1] / "data" / "reports"
CHAIN1_JSON = ROOT / "experiments" / "judgment_chain_20261003" / "judgment_chain_20261003_2200.json"

# ---- 冻结（预注册 §一/§三） ----
S_ORDER = ["S1", "S2", "S3", "S4"]


def cfg_variant(cfg, name: str):
    from dataclasses import replace
    if name == "A0":
        return cfg
    if name == "S1":
        return replace(cfg, signal=replace(cfg.signal, quiet_vol=False))
    if name == "S2":
        return replace(cfg, signal=replace(cfg.signal, exit_buffer=1))
    if name == "S3":
        return replace(cfg, account=replace(cfg.account, max_symbols=4),
                       signal=replace(cfg.signal, top_n=4))
    if name == "S4":
        return replace(cfg, signal=replace(cfg.signal, top_n=1))
    return cfg


def run_rule_arm(name: str, cfg0, bundle, panels, dates, start_ts, rng_seed=RAND_SEED):
    """规则臂（A0/S1-S4）与静态臂 E、随机臂 D 的统一两遍法 runner。"""
    cfg_ = cfg_variant(cfg0, name)
    feats, universe = bundle.features, bundle.universe
    uni_mask = UniverseIndex(universe)
    raw_close = panels["close"]
    keep_n = cfg_.signal.top_n * cfg_.signal.exit_buffer
    eng = BacktestEngine(cfg_, build_bars(panels))

    pvmed = None
    if cfg_.signal.quiet_vol:
        pvmed = feats["pvol_20"].rolling(252, min_periods=126).median()

    def targets_for(d, equity, positions, rng):
        if name == "E":
            if positions or d != first_reb[0]:
                return []
            # 2026-10-08 B2：universe 侧走整数切片（原逐格 .at）；
            # raw_close 的 notna 仍逐格判定（缺失语义与 universe 不同，不合并）
            pool = [c for c in universe.columns[uni_mask.positions(d)]
                    if pd.notna(raw_close.at[d, c])]
            pick = list(pd.Series(pool).sample(n=min(2, len(pool)), random_state=rng))
            tgts = []
            for code in pick:
                px = float(raw_close.at[d, code])
                atr = feats["atr14"].at[d, code] if code in feats["atr14"].columns else np.nan
                s = size_position(cfg_, equity, px, float(atr) if np.isfinite(atr) else 0.0)
                if s.lots > 0:
                    tgts.append(Target(code, 1, s.lots, "static_hold"))
            return tgts
        if name == "D":
            # 2026-10-08 B2：universe 侧走整数切片（原逐格 .at）；
            # raw_close 的 notna 仍逐格判定（缺失语义与 universe 不同，不合并）
            pool = [c for c in universe.columns[uni_mask.positions(d)]
                    if pd.notna(raw_close.at[d, c])]
            pick = list(pd.Series(pool).sample(n=min(cfg_.signal.top_n, len(pool)),
                                               random_state=rng))
            tgts = [Target(c, 0, 0, "random_rotate") for c in list(positions) if c not in pick]
            for code in pick:
                if code in positions:
                    continue
                px = float(raw_close.at[d, code])
                atr = feats["atr14"].at[d, code] if code in feats["atr14"].columns else np.nan
                s = size_position(cfg_, equity, px, float(atr) if np.isfinite(atr) else 0.0)
                if s.lots > 0:
                    tgts.append(Target(code, 1, s.lots, "random"))
            return tgts
        score, eligible = rule_reversal_decision(
            feats, universe, d, rev_window=cfg_.signal.rev_window,
            quiet_vol=cfg_.signal.quiet_vol, pvol_med=pvmed)
        ranked = score[eligible].sort_values(ascending=False)
        keep = list(ranked.head(keep_n).index)
        tgts = [Target(c, 0, 0, f"exit_band(keep_{keep_n})") for c in list(positions) if c not in keep]
        for code in ranked.head(cfg_.signal.top_n).index:
            if code in positions:
                continue
            px = float(raw_close.at[d, code]) if pd.notna(raw_close.at[d, code]) else None
            if px is None:
                continue
            atr = feats["atr14"].at[d, code] if code in feats["atr14"].columns else np.nan
            s = size_position(cfg_, equity, px, float(atr) if np.isfinite(atr) else 0.0)
            if s.lots > 0:
                tgts.append(Target(code, 1, s.lots, f"{name}(rank={float(ranked[code]):.3f})"))
        return tgts

    # E 的"首调仓日"：OOS 窗口内第一个调仓日（预注册 §一）
    reb_all = [d for d in dates if d >= start_ts and d.weekday() == cfg_.signal.rebalance_weekday]
    first_reb = reb_all[:1]

    # 2026-10-08 去重 + B5：两遍法骨架移到 fterm/backtest/two_pass.py
    # （原实现与本仓另 5 个 chain 脚本逐行重复 ⇒ 风控口径有静默分叉风险；
    #  peak 原为每调仓日重算前缀最大值，现用 cummax O(1) 查表）
    r, n_rej, t2 = run_two_pass(
        engine=eng, reb_days=reb_all, cfg=cfg_, raw_close=raw_close,
        initial_equity=cfg_.account.initial_equity, start_ts=start_ts,
        targets_for_factory=lambda: (lambda d, e, p: targets_for(d, e, p, np.random.RandomState(rng_seed))))
    m = dict(r.metrics)
    m["n_governor_rejected"] = n_rej
    m["n_targets"] = int(sum(len(v) for v in t2.values()))
    return r.equity, m


def main():
    cfg = load_config(ROOT / "config.yaml")
    REPORTS.mkdir(parents=True, exist_ok=True)
    chain1 = json.loads(CHAIN1_JSON.read_text(encoding="utf-8"))
    start_ts = pd.Timestamp(chain1["oos_window"]["start"])
    print(f"OOS 窗口起点（对齐判据链 #1）: {start_ts.date()}", flush=True)

    print("== 面板与特征 ==", flush=True)
    panels = sa.load_panels(cfg)
    bundle = build_dataset(cfg)
    dates = panels["close"].index

    arms = ["A0", "S1", "S2", "S3", "S4", "E", "D"]
    eqs, metrics = {}, {}
    for arm in arms:
        t0 = time.time()
        eqs[arm], metrics[arm] = run_rule_arm(arm, cfg, bundle, panels, dates, start_ts)
        m = metrics[arm]
        print(f"  {arm:3s}: {m['final_equity']:.0f} ({m['total_return']:+.2%}) "
              f"MDD {m['max_drawdown']:+.2%} 成本 {m['total_costs']:.0f} "
              f"目标 {m['n_targets']} 拒 {m['n_governor_rejected']}  ({time.time()-t0:.0f}s)",
              flush=True)

    fridays = [d for d in dates if d >= start_ts
               and d.weekday() == cfg.signal.rebalance_weekday and d in eqs["D"].index]
    paired = {a: paired_weekly_t(eqs[a], eqs["D"], fridays) for a in arms if a != "D"}

    # ---- 归因读数 + S* 选择（冻结顺序） ----
    attribution = {}
    for v, base in (("S1", "A0"), ("S2", "A0"), ("S3", "A0"), ("S4", "A0")):
        attribution[v] = {
            "paired_mean_improvement_vs_A0": paired[v]["mean"] - paired["A0"]["mean"],
            "mdd_improvement_vs_A0": metrics[v]["max_drawdown"] - metrics["A0"]["max_drawdown"],
            "return_improvement_vs_A0": metrics[v]["total_return"] - metrics["A0"]["total_return"],
        }
    s_star = None
    for arm in S_ORDER:
        p = paired[arm]
        if p["mean"] > 0 and p["t"] > 0:
            s_star = arm
            break
    e_floor = {
        "E_return": metrics["E"]["total_return"],
        "beats_all_rotating": bool(all(metrics["E"]["total_return"] > metrics[a]["total_return"]
                                       for a in arms if a != "E")),
        "beats_D": bool(metrics["E"]["total_return"] > metrics["D"]["total_return"]),
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    verdict = {
        "prereg": "plans/ablation-prereg-20261004.md",
        "oos_window": chain1["oos_window"],
        "arms": {a: {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                     for k, v in metrics[a].items()} for a in arms},
        "paired_vs_D": paired,
        "attribution": attribution,
        "s_star": s_star,
        "s_selection_rule": "S1>S2>S3>S4 第一个 vs D 周配对差均值>0 且 t>0",
        "e_floor": e_floor,
        "verdict_word": "",   # 待用户填写
    }
    (REPORTS / f"ablation_chain_{stamp}.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [f"判据链 #2 读数 {stamp}（判据冻结于 plans/ablation-prereg-20261004.md）",
             f"OOS {start_ts.date()} 起；配对基 = D（随机全换仓）"]
    for a in arms:
        p = paired.get(a)
        ps = f"vsD mean {p['mean']:+.5f} (t {p['t']:+.2f}, n={p['n']})" if p else "（配对基）"
        lines.append(f"  {a:3s} {metrics[a]['total_return']:+.2%} MDD {metrics[a]['max_drawdown']:+.2%} "
                     f"成本 {metrics[a]['total_costs']:.0f}  {ps}")
    lines.append(f"归因: " + "; ".join(
        f"{v}:{attribution[v]['paired_mean_improvement_vs_A0']:+.5f}/周"
        for v in S_ORDER))
    lines.append(f"S*（新基线候选）= {s_star or '∅（构造族判负）'}")
    lines.append(f"E 静态地板: {e_floor['E_return']:+.2%}"
                 f"（{'打赢全部轮动臂' if e_floor['beats_all_rotating'] else '未打赢全部轮动臂'}，"
                 f"{'打赢' if e_floor['beats_D'] else '未打赢'} D）")
    lines.append("判决词：＿＿＿（由用户填写；S* 进生产前须判据链 #3 确认性预注册）")
    text = "\n".join(lines)
    (REPORTS / f"ablation_chain_{stamp}.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f"报告 → data/reports/ablation_chain_{stamp}.json/.txt", flush=True)


if __name__ == "__main__":
    main()
