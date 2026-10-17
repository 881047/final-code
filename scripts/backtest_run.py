"""回测 CLI：python scripts/backtest_run.py [--start 2020-01-01]

流程：面板 → 特征 → 样本 → 训练（含多种子集成）→ 逐日推理（仅用当日可得信息）
    → 信号 → sizer → governor → BacktestEngine 撮合 → 指标报告。
报告落盘 data/reports/backtest_*.txt（回测证据存档纪律）。

忠实单遍（2026-10-08 收口批，docs/03 §29/§42）：决策在引擎循环内用**引擎真实账本**
（真实权益/可用现金/真实持仓）问询，与生产 runner、判据链忠实臂同一套账本语义。
旧实现的两遍法先列目标再过引擎，被引擎按现金拒掉的单在决策环里已算持仓
⇒ 幻象持仓 ⇒ 长期欠投资（实测同口径 n_blocked_cash 192 笔）——本 CLI 是管线
验收工具，不得再站在被 §29 判废的评测口上出验收证据。
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fterm.features.universe import UniverseIndex  # noqa: E402
from fterm.backtest.engine import BacktestEngine, Target
from fterm.backtest.two_pass import run_faithful
from fterm.config import load_config
from fterm.data import ashare as sa
from fterm.features.build import build_dataset
from fterm.model.dataset import stack_samples
from fterm.model.train import train_ensemble
from fterm.strategy.primary import rule_reversal_decision
from fterm.strategy.signal import SignalProvider


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--signal", default=None, choices=["model", "rule_reversal"],
                    help="信号源；缺省取 config signal.mode")
    ap.add_argument("--force-liquidate", action="store_true",
                    help="A/B 处理臂：熔断→清仓存量持仓（不改 config.yaml）")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "config.yaml")
    if args.force_liquidate:
        from dataclasses import replace
        cfg = replace(cfg, account=replace(cfg.account, circuit_force_liquidate=True))
    sig_mode = args.signal or cfg.signal.mode

    print("== 1/5 构建特征（面板）==")
    bundle = build_dataset(cfg)
    feats, labels, universe = bundle.features, bundle.labels, bundle.universe
    uni_mask = UniverseIndex(universe)

    ens = None
    if sig_mode == "model":
        samples = stack_samples(feats, labels, bundle.feature_cols, cfg.features.lookback,
                                universe, weight_mode=cfg.model.sample_weight,
                                max_stocks_per_day=cfg.features.max_stocks_per_day)
        print(f"样本 {len(samples)}")
        assert len(samples) >= 200, "样本不足，先 sync_data"
        print("== 2/5 训练（多种子集成，只用训练段）==")
        ens, _ = train_ensemble(cfg, samples, bundle.feature_cols)
    else:
        print("== 2/5 训练：跳过（rule_reversal 规则信号，无 ML；占位基线未证实 α）==")

    print("== 3/5 引擎 + 忠实决策环 ==")
    signals = SignalProvider(cfg)  # model 模式用；rule_reversal 走 primary
    panels = sa.load_panels(cfg)
    raw_close = panels["close"]
    all_dates = panels["close"].index
    start_ts = pd.Timestamp(args.start)

    def _atr(code: str, d: pd.Timestamp) -> float:
        v = feats["atr14"].at[d, code] if (d in feats["atr14"].index and code in feats["atr14"].columns) else np.nan
        return float(v) if np.isfinite(v) else 0.0

    def _px(code: str, d: pd.Timestamp) -> float | None:
        v = raw_close.at[d, code] if (d in raw_close.index and code in raw_close.columns) else np.nan
        return float(v) if np.isfinite(v) and v > 0 else None

    # 静默波动中位数面板：全史一次预计算（滚动窗口 PIT 安全），逐调仓日复用
    pvol_med = (feats["pvol_20"].rolling(252, min_periods=126).median()
                if cfg.signal.quiet_vol else None)
    reb_dates = [d for d in all_dates
                 if d >= start_ts and d.weekday() == cfg.signal.rebalance_weekday]

    def propose(ctx) -> list[Target]:
        """忠实模式决策：t 日收盘出目标，引擎按 t+1 开盘撮合（挂单语义同生产 runner）。"""
        d = ctx.date
        if d.weekday() != cfg.signal.rebalance_weekday:
            return []                       # 非信号日：维持现状（引擎幂等跳过）
        tgts: list[Target] = []
        if sig_mode == "rule_reversal":
            score, eligible = rule_reversal_decision(
                feats, universe, d, rev_window=cfg.signal.rev_window,
                quiet_vol=cfg.signal.quiet_vol, pvol_med=pvol_med)
            ranked = score[eligible].sort_values(ascending=False)
            keep_n = cfg.signal.top_n * cfg.signal.exit_buffer
            keep = set(ranked.head(keep_n).index)
            for code in list(ctx.positions):
                if code not in keep:
                    tgts.append(Target(code, 0, 0, f"exit_band(rank_out_of_{keep_n})"))
            for code in ranked.head(cfg.signal.top_n).index:
                if code in ctx.positions:
                    continue                # 同向持仓不加仓
                px = _px(code, d)
                if px is None:
                    continue
                sz = size_position(cfg, ctx.equity, px, _atr(code, d))
                if sz.lots == 0:
                    continue
                tgts.append(Target(code, 1, sz.lots,
                                   f"rule_reversal(rank={float(ranked[code]):.3f})"))
        else:
            cols = [c for c in ens.feature_cols if c in feats]
            cands = []
            # 只在入选列上迭代（B2 向量化口径）；持仓过滤保留——二者语义不同。
            for code in universe.columns[uni_mask.positions(d)]:
                if code in ctx.positions:
                    continue
                win = np.stack([feats[c][code].loc[:d].tail(ens.lookback).to_numpy()
                                for c in cols], axis=1)
                if len(win) < ens.lookback or np.isnan(win).any():
                    continue
                cands.append((code, ens.predict(win)))
            cands.sort(key=lambda x: x[1]["p_up"], reverse=True)
            taken = 0
            for code, pred in cands:
                if taken >= cfg.account.max_symbols:
                    break
                sig = signals.decide(code, pred)
                if sig.direction == 0:
                    continue
                px = _px(code, d)
                if px is None:
                    continue
                sz = size_position(cfg, ctx.equity, px, _atr(code, d),
                                   vol_pred_log=pred["vol_pred"])
                if sz.lots == 0:
                    continue
                tgts.append(Target(code, 1, sz.lots,
                                   sig.reason + f"(p={pred['p_up']:.2f})"))
                taken += 1
        return tgts

    # 停板面板可能缺最近上市的股票（无 stk_limit 记录）⇒ 对齐到 close 列，
    # 缺口留给引擎的"昨收 ±10%"兜底近似（docs/02 登记口径）
    up_lim = panels["up_limit"].reindex(columns=raw_close.columns)
    dn_lim = panels["down_limit"].reindex(columns=raw_close.columns)
    bars_map = {}
    for code in raw_close.columns:
        df = pd.DataFrame({
            "open": panels["open"][code], "high": panels["high"][code],
            "low": panels["low"][code], "close": panels["close"][code],
            "up_limit": up_lim[code], "down_limit": dn_lim[code],
        }).dropna(subset=["close"])
        if len(df):
            bars_map[code] = df
    eng = BacktestEngine(cfg, bars_map)

    from fterm.risk.sizer import size_position
    result, n_rej, n_tgt = run_faithful(
        engine=eng, reb_days=reb_dates, propose=propose, cfg=cfg,
        raw_close=raw_close, start_ts=start_ts)

    print(f"   目标 {n_tgt} 个，governor 拒绝 {n_rej} 个")
    print("== 5/5 指标 ==")
    from fterm.backtest.metrics import metrics_summary
    summary = metrics_summary(result.metrics)
    print(summary)

    reports_dir = root / "data" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    universe_desc = "fixed" if cfg.universe.mode == "fixed" else "all_a(PIT过滤器)"
    header = (f"回测区间 {args.start} ~ {all_dates.max().date()}  信号: {sig_mode}\n"
              f"股票池: {universe_desc}  本金: {cfg.account.initial_equity:.0f}\n"
              f"评测口: 忠实单遍（引擎真实账本，docs/03 §29/§42）\n"
              f"数据指纹: {bundle.fingerprints}\n"
              f"决策数: {n_tgt}（governor 拒绝 {n_rej}）\n"
              f"⚠️ 规则基线未证实 α：任何采纳声明前必须预注册+组合层两轴（纪律同老盘）\n")
    (reports_dir / f"backtest_{stamp}.txt").write_text(header + summary, encoding="utf-8")
    print(f"报告 → data/reports/backtest_{stamp}.txt")


if __name__ == "__main__":
    main()
