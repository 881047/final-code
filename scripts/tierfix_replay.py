"""判据链 #15 · 仪器修正重放批（2026-10-08，预注册 plans/tierfix-prereg-20261008.md）。

不训练、不改判据常数、不碰生产：只读缓存 pred 流，把 #13/#14 的组合层臂
搬到 **真 T3 档**（100 万/top10/keep20/max_lots 50）重放，并加一条
`ema_step=daily` 的生产语义孪生臂。

自证 X0：单流 ss7@T3(hurdle=10) 必须复现 #12 存档读数
（+48.00% / MDD −28.54% / 成本 221220.362），否则本脚本全部读数作废。
"""
from __future__ import annotations

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
from fterm.features.universe import UniverseIndex         # noqa: E402

import sync_index                                        # noqa: E402
import dlv2_hurdle_chain as HC                            # noqa: E402
from capital_chain import S1_TIER, _rule_tier_arm         # noqa: E402
from fterm.research.scaffold import report_meta, selfcheck as scaffold_selfcheck  # noqa: E402
from monthly_chain import monthly_log_paired              # noqa: E402

REPORTS = ROOT / "data" / "reports"
OUTDIR = ROOT / "experiments" / "tierfix_20261008"
START_OOS = HC.START_OOS
T3 = HC.T3
C2A_T, C3_TOL, C4_RATIO = HC.C2A_T, HC.C3_TOL, HC.C4_RATIO
MAX_DATA_DATE = pd.Timestamp("2026-09-30")       # 前瞻窗隔离：研究线不得读其后的数据

MSE_STREAMS = {
    "ss7_21_99_5_13": "data/reports/dlv2_ss7_21_99_5_13.parquet",
    "ss123_2024_42_314_2027": "data/reports/dlv2_ss123_2024_42_314_2027.parquet",
    "ss101_202": "data/reports/dlv2_ss101_202.parquet",
    "ss303_404": "data/reports/dlv2_ss303_404.parquet",
}
RL_STREAMS = {
    "ss7_21_99_5_13": "data/reports/rankloss_ss7_21_99_5_13.parquet",
    "ss123_2024_42_314_2027": "data/reports/rankloss_ss123_2024_42_314_2027.parquet",
}
# 自证基准 X0_REF —— 2026-10-08 重新钉定（erratum，见 plans/tierfix-prereg-20261008.md §六）
# 旧基准（#12 存档 +48.00% / −28.54% / 221220.362）经二分定位确认**无效**：
#   10-06 审查批 ffea601 给 RiskGovernor 加了批内现金累加 `committed`，修掉了
#   "风控批了但引擎 blocked_cash、决策环却以为已持仓 ⇒ 幻象持仓导致长期欠投资"
#   的缺陷。实测该臂 n_blocked_cash 192 → 0、同流同档读数 +48.00% → +149.69%。
#   跨树验证：9e7e5c6 树 + 今天面板 = 逐位复现 +48.00%（说明输入未漂移，差异全在代码）；
#             ffea601 树 与 HEAD 跑出同一组数（+1.496875/−0.240336/300619.916）。
X0_REF = {"total_return": 1.4968753848070593, "max_drawdown": -0.24033625792564528,
          "total_costs": 300619.9162546077}
X0_REF_PROVENANCE = "ffea601..HEAD 语义（批内 committed 现金累加已生效），2026-10-08 跨树核对"


def content_sha(df: pd.DataFrame) -> str:
    """兼容别名：内容哈希统一走 scaffold（唯一写入侧，2026-10-08 审查批收敛）。"""
    from fterm.research.scaffold import content_sha as _sha
    return _sha(df)


def repro_context(cfg, panels: dict, bundle, paths: list[str], *,
                  harness: str = "faithful", ema_step: str = "daily") -> dict:
    """复现契约（scaffold 统一写入侧）——补 harness/含权口径/EMA 语义/数据窗右端点，
    使防伪闸门 A1/A6/A7/A8 可核对（旧实现缺这些字段，#15 存档被判不可采信是事实）。"""
    return report_meta(cfg, panels, bundle, harness=harness,
                       dividend_basis="hfq_inclusive", ema_step=ema_step,
                       stream_paths=paths, window_max=MAX_DATA_DATE, tier=T3,
                       extra={"anchor_note": X0_REF_PROVENANCE})


def load_panel_map(paths: dict[str, str]) -> dict[str, pd.DataFrame]:
    out = {}
    for tag, rel in paths.items():
        df = pd.read_parquet(ROOT / rel)
        df["date"] = pd.to_datetime(df["date"])
        hi = df["date"].max()
        assert hi <= MAX_DATA_DATE, f"{tag}: 预测流含 >{MAX_DATA_DATE.date()} 的数据 {hi}"
        out[tag] = df.pivot_table(index="date", columns="product", values="pred")
    return out


def mean_panel(panels: list[pd.DataFrame]) -> pd.DataFrame:
    """与 #13/#14 同款集成均值：逐流 pivot 后相加除以流数。"""
    base = None
    for p in panels:
        base = p.copy() if base is None else base.add(p, fill_value=0)
    return base.div(len(panels)).astype("float32")


def reb_ranking(smooth: dict[str, float], raw_row: pd.Series | None) -> pd.Series:
    """调仓日排名（模块级纯函数，便于单测）。

    - `raw_row` = 当日预测截面（**未过 universe 掩码**，与旧实现 reindex 对象一致）
      ⇒ smooth 限制到当日预测列，资格/预测消失的持仓自然掉出 keep；
    - `raw_row` is None（预测流缺口日，如标签前视窗 `fwd_ret` 需未来 6 日导致的
      窗口尾部周五）⇒ 用最近一次平滑秩整体排名 —— 与生产 runner 的持久化 EMA
      状态同语义。旧实现 reindex 到空 ⇒ keep=∅ ⇒ **整本清仓**（2026-10-08 审查批
      勘误，docs/03 §41；`run_cema` 该场景保持持仓，三处语义自此对齐）。
    """
    s = pd.Series(smooth, dtype="float64")
    if raw_row is None:
        return s.dropna().sort_values(ascending=False)
    return s.reindex(raw_row.index).dropna().sort_values(ascending=False)


def daily_step_panel(pred: pd.DataFrame, universe: pd.DataFrame,
                     alpha: float = 0.5) -> pd.DataFrame:
    """生产语义孪生：逐交易日（非仅周五）步进秩 EMA。

    与 `runner._run_dlv2` 一致：当日 eligible 截面内取秩 pct →
    `ema[c] = alpha·pct + (1−alpha)·ema[c]`（首次入选取原 pct）→
    仅输出当日 eligible 的平滑值。判据链的 C_ema 只在调仓日步进（半衰期≈1 周），
    本函数产出半衰期≈1 交易日的对照对象；下游用 EMA_ALPHA=1.0 直通、不再二次平滑。
    """
    uni = UniverseIndex(universe)
    cols = universe.columns
    ema: dict[str, float] = {}
    rows = {}
    for d in pred.index:
        pos = uni.row_pos(d)
        if pos is None:
            continue
        s = pred.loc[d].dropna()
        if len(s) == 0:
            continue
        s = s[uni.mask_for(d, s.index, strict=False)]      # 当日 eligible（向量化，同各链）
        if len(s) == 0:
            continue
        pct = s.rank(pct=True)
        for c, v in pct.items():
            ema[c] = float(v) if c not in ema else alpha * float(v) + (1 - alpha) * ema[c]
        rows[d] = pd.Series({c: ema[c] for c in pct.index})
    out = pd.DataFrame(rows).T.reindex(columns=cols).sort_index()
    return out.astype("float32")


def gates(eq: pd.Series, m: dict, bl_eq: pd.Series, bl_m: dict,
          month_ends: list[pd.Timestamp]) -> dict:
    p_m = monthly_log_paired(eq, bl_eq, month_ends)
    return {
        "total_return": round(float(m["total_return"]), 6),
        "max_drawdown": round(float(m["max_drawdown"]), 6),
        "total_costs": round(float(m["total_costs"]), 1),
        "n_trades": int(m.get("n_trades", 0)),
        "position_util_mean": round(float(m.get("position_util_mean", float("nan"))), 4),
        "n_blocked_cash": int(m.get("n_blocked_cash", 0)),
        "n_targets": int(m.get("n_targets", 0)),
        "n_governor_rejected": int(m.get("n_governor_rejected", 0)),
        "C2a": {"t": round(float(p_m["t"]), 4), "mean_log": round(float(p_m["mean_log"]), 6),
                "n": int(p_m["n"]), "pass": bool(p_m["t"] >= C2A_T and p_m["mean_log"] > 0)},
        "C2b": bool(m["total_return"] > bl_m["total_return"]),
        "C3": bool(m["max_drawdown"] >= bl_m["max_drawdown"] - C3_TOL),
        "C4": bool(m["total_costs"] <= bl_m["total_costs"] * C4_RATIO),
    }


def beta_ir(eq: pd.Series, bench: pd.Series) -> dict:
    """周频 β / α / 残差 IR（透明读数，不是门）。"""
    wk = eq.resample("W-FRI").last().pct_change().dropna()
    bk = bench.resample("W-FRI").last().pct_change().dropna()
    j = pd.concat([wk, bk], axis=1, keys=["p", "b"]).dropna()
    if len(j) < 24:
        return {}
    x, y = j["b"].to_numpy(), j["p"].to_numpy()
    # ddof=1 与 np.cov 的默认一致（旧实现 var 用 ddof=0，β 有 1/(n-1) 量级的微小偏倚）
    beta = float(np.cov(x, y)[0, 1] / np.var(x, ddof=1))
    resid = y - (beta * x + (y.mean() - beta * x.mean()))
    sd = float(resid.std())
    return {"n_weeks": int(len(j)), "beta": round(beta, 3),
            "ann_return": round(float((1 + y.mean()) ** 52 - 1), 4),
            "ann_alpha_wk": round(float((y.mean() - beta * x.mean()) * 52), 4),
            "resid_IR_ann": round(float((y.mean() - beta * x.mean()) / sd * np.sqrt(52)), 3)
            if sd > 0 else None}


def arm_faithful(panel: pd.DataFrame, tier: dict, cfg, bundle, panels, dates,
                 start_ts, ema_step: str = "rebalance", alpha: float = 0.5,
                 stats: dict | None = None, peak_fn=None, exp_scaler=None,
                 exit_fn=None, buy_filter_fn=None, out: dict | None = None,
                 exempt_batch_exits: bool = False, reb_mod: int = 1):
    """忠实单遍：决策在引擎循环内问真实账本（equity/cash/positions 全同构于生产 runner）。

    与 `HC.run_cema` 的差别只有一个但性质严重：旧法先列目标再过引擎，风控按估计市值
    批单、引擎按真实现金拒单，被拒的单在决策视图里已算持仓 ⇒ 幻象持仓 ⇒ 欠投资
    （docs/03 §29，实测 `n_blocked_cash` 192 → 0）。
    `ema_step`：rebalance = 判据链 #11/#12/#13 被采纳构型（半衰期≈1 周）；
    daily = 生产位语义（半衰期≈1 交易日，`signal.ema_step=daily`）。

    `peak_fn`：显式传入时优先（#19/#20 的 E1/E3 实验用）。**未传入时按
    `cfg.account.dd_peak_mode` 解析**（`fterm/risk/peak.py`，单一实现来源）：
    `none` ⇒ None（走 `run_faithful` 原有 running-max 分支，**与修复前逐位一致**）；
    `rolling` ⇒ 滚动窗口峰值（**无吸收态**，docs/03 §36 缺陷的结构性修复）。

    `reb_mod`：>1 时调仓日取子序列（每 reb_mod 个取一；#137 双周臂用 2）。
    默认 1 ⇒ reb 列表逐元素不变 ⇒ 全路径逐位不变（由 #137 G0 证明）。
    """
    from dataclasses import replace as _replace

    from fterm.backtest.engine import BacktestEngine, Target
    from fterm.backtest.two_pass import run_faithful
    from fterm.risk.peak import peak_fn_from_cfg
    from fterm.risk.sizer import size_position
    from fterm.data import ashare as _sa
    from judgment_chain import build_bars        # 各链共用的行情构造（sys.path 已含 scripts/）

    cfg_ = _replace(cfg,
                    account=_replace(cfg.account, initial_equity=tier["equity"],
                                     max_symbols=tier["max_symbols"],
                                     max_lots_per_symbol=tier["max_lots"]),
                    signal=_replace(cfg.signal, top_n=tier["top_n"]))
    feats, universe = bundle.features, bundle.universe
    uni = UniverseIndex(universe)
    raw_close = panels["close"]
    eng = BacktestEngine(cfg_, build_bars(panels))
    reb_all = [d for d in dates if d >= start_ts
               and d.weekday() == cfg_.signal.rebalance_weekday]
    if reb_mod > 1:
        reb_all = reb_all[::reb_mod]       # #137 低频调仓臂：每 reb_mod 个调仓日取一
    step_days = set(dates) if ema_step == "daily" else set(reb_all)
    smooth: dict[str, float] = {}
    # 忠实模式需要"上一交易日权益"做日停参照 → 由引擎 ctx 提供；此处只需逐日步进 EMA
    pred_by_day = {d: panel.loc[d].dropna() for d in panel.index}

    def propose(ctx):
        d = ctx.date
        raw_row = pred_by_day.get(d)
        if d in step_days and raw_row is not None:
            row = raw_row[uni.mask_for(d, raw_row.index)]
            if len(row):
                pct = row.rank(pct=True)
                for c, v in pct.items():
                    smooth[c] = float(v) if c not in smooth else \
                        alpha * float(v) + (1 - alpha) * smooth[c]
        if d not in reb_set:
            return []          # 非调仓日：只步进平滑，不出单（与生产 runner 同节律）
        ranked = reb_ranking(smooth, raw_row)
        if exit_fn is None:
            # 现行退出语义（跌出 keep_n ⇒ 卖）——exit_fn 未传入时逐位不变（#128 G0 钉）
            keep = set(ranked.head(tier["keep_n"]).index)
            tgts = [Target(c, 0, 0, f"exit_band(keep_{tier['keep_n']})")
                    for c in list(ctx.positions) if c not in keep]
        else:
            # #128 判弱侧退出引擎：自定义退出规则（预注册 plans/exit-rule-prereg-20261014.md）
            tgts = list(exit_fn(ranked=ranked, smooth=smooth,
                                positions=list(ctx.positions), tier=tier, date=d))
        if buy_filter_fn is None:
            buy_cands = list(ranked.head(tier["top_n"]).index)
        else:
            # #132 判弱因子买入前过滤（预注册 plans/avoid-fin-prereg-20261014.md）：
            # 返回过滤后的候选序列（同 ranked 序），买入循环顺延尝试
            buy_cands = list(buy_filter_fn(ranked=ranked, smooth=smooth, date=d))
        for code in buy_cands:
            if code in ctx.positions:
                continue
            px = float(raw_close.at[d, code]) if pd.notna(raw_close.at[d, code]) else None
            if px is None:
                continue
            atr = feats["atr14"].at[d, code] if code in feats["atr14"].columns else np.nan
            s = size_position(cfg_, ctx.equity, px, float(atr) if np.isfinite(atr) else 0.0)
            if exp_scaler is not None:      # #19 连续降仓：按回撤深度缩目标敞口，而非开/关
                s_lots = int(np.floor(s.lots * float(exp_scaler(ctx))))
                if s_lots != s.lots:
                    from fterm.backtest.engine import Target as _T
                    if s_lots > 0:
                        tgts.append(_T(code, 1, s_lots,
                                       f"scaled(ema={float(smooth[code]):.3f})"))
                    continue
            if s.lots > 0:
                tgts.append(Target(code, 1, s.lots, f"faithful(ema={float(smooth[code]):.3f})"))
        # #131 疑点诊断（§157②）：零行为变化的成交日志（stats["daylog"]，仅诊断用）
        if stats is not None and "daylog" in stats:
            tries = []
            for code in buy_cands[:K_LOG]:
                px_t = float(raw_close.at[d, code]) if pd.notna(raw_close.at[d, code]) else None
                lots_t = 0
                if px_t is not None:
                    atr_t = feats["atr14"].at[d, code] if code in feats["atr14"].columns else np.nan
                    lots_t = int(size_position(cfg_, ctx.equity, px_t,
                                               float(atr_t) if np.isfinite(atr_t) else 0.0).lots)
                tries.append({"code": code, "px": px_t, "lots": lots_t,
                              "held": code in ctx.positions})
            stats["daylog"].append({
                "date": str(d.date()), "cash": float(ctx.cash),
                "equity": float(ctx.equity),
                "positions": {c: (p.get("lots", "?") if isinstance(p, dict)
                                  else getattr(p, "lots", "?"))
                              for c, p in ctx.positions.items()},
                "ranked_head": [str(c) for c in ranked.head(10).index],
                "n_cands": len(buy_cands),
                "tries": tries,
                "n_exit_tgts": sum(1 for t in tgts if t.lots == 0),
                "n_buy_tgts": sum(1 for t in tgts if t.lots > 0)})
        return tgts

    reb_set = set(reb_all)
    ask = set(dates) if ema_step == "daily" else reb_all
    K_LOG = 8                    # #131 疑点诊断：daylog 只记前 8 个候选的定价/手数
    if peak_fn is None:
        # 单一实现来源（fterm/risk/peak.py）：none ⇒ None（默认分支，逐位不变）
        peak_fn = peak_fn_from_cfg(cfg_)
    res, n_rej, n_tgt = run_faithful(engine=eng, reb_days=reb_all, propose=propose,
                                     cfg=cfg_, raw_close=raw_close, start_ts=start_ts,
                                     ask_days=ask, stats=stats, peak_fn=peak_fn,
                                     exempt_batch_exits=exempt_batch_exits)
    if out is not None:
        out["res"] = res
    m = dict(res.metrics)
    m["n_governor_rejected"] = n_rej
    m["n_targets"] = n_tgt
    return res.equity, m


def _faithful_mode(cfg, bundle, panels, dates, start_ts, month_ends, bench,
                   mse, rl, mse2, mse4, rl2, eq_s1, m_s1) -> int:
    """忠实单遍账本下的关键配对（docs/03 §31）。"""
    results: dict[str, dict] = {}
    results["_repro"] = repro_context(cfg, panels, bundle,
                                      list(MSE_STREAMS.values()) + list(RL_STREAMS.values()))
    results["baseline_S1_twopass"] = {
        "total_return": round(float(m_s1["total_return"]), 6),
        "max_drawdown": round(float(m_s1["max_drawdown"]), 6),
        "total_costs": round(float(m_s1["total_costs"]), 1),
        "note": "规则基线仍是两遍法口径（未重做），只作水平参考，不参与忠实配对判定"}
    curves: dict[str, pd.Series] = {}
    arms = [("F_RL2_rebalance", rl2, "rebalance"), ("F_MSE2_rebalance", mse2, "rebalance"),
            ("F_MSE4_rebalance", mse4, "rebalance"),
            ("F_RL2_daily", rl2, "daily"), ("F_MSE2_daily", mse2, "daily")]
    for name, panel, step in arms:
        t0 = time.time()
        eq, m = arm_faithful(panel, T3, cfg, bundle, panels, dates, start_ts, ema_step=step)
        curves[name] = eq
        results[name] = {"total_return": round(float(m["total_return"]), 6),
                         "max_drawdown": round(float(m["max_drawdown"]), 6),
                         "total_costs": round(float(m["total_costs"]), 1),
                         "n_trades": int(m.get("n_trades", 0)),
                         "position_util_mean": round(float(m.get("position_util_mean", 0)), 4),
                         "n_blocked_cash": int(m.get("n_blocked_cash", 0)),
                         "n_targets": int(m["n_targets"]),
                         "n_governor_rejected": int(m["n_governor_rejected"]),
                         "ema_step": step,
                         "beta_vs_000905": beta_ir(eq, bench)}
        r = results[name]
        print(f"  {name}: {m['total_return']:+.2%} MDD {m['max_drawdown']:+.2%} "
              f"成本 {m['total_costs']:.0f} 均仓位 {r['position_util_mean']:.3f} "
              f"blocked_cash {r['n_blocked_cash']} 拒绝 {r['n_governor_rejected']} "
              f"β {r['beta_vs_000905'].get('beta')} IR {r['beta_vs_000905'].get('resid_IR_ann')} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if r["n_blocked_cash"] != 0:
            print(f"  !! {name} 仍有 blocked_cash ⇒ 忠实账本未闭合，读数不作数", flush=True)
    # ---- 阴性对照（闸门 A5）：日内置换预测秩，信号结构被毁 ⇒ 读数必须显著塌掉 ----
    from exposure_chain import shuffled_stream      # 延迟导入：exposure 顶层 import 本模块
    t0 = time.time()
    eq_neg, m_neg = arm_faithful(shuffled_stream(rl2, bundle.universe), T3, cfg,
                                 bundle, panels, dates, start_ts, ema_step="daily")
    real_ret = float(results["F_RL2_daily"]["total_return"])
    gap = real_ret - float(m_neg["total_return"])
    results["NEG_shuffled_daily"] = {
        "total_return": round(float(m_neg["total_return"]), 6),
        "max_drawdown": round(float(m_neg["max_drawdown"]), 6),
        "total_costs": round(float(m_neg["total_costs"]), 1),
        "n_blocked_cash": int(m_neg.get("n_blocked_cash", 0))}
    results["_negative_control"] = {
        "negative_control": "NEG_shuffled_daily（置换预测秩，对照 F_RL2_daily）",
        "real_return": round(real_ret, 6),
        "shuffled_return": round(float(m_neg["total_return"]), 6),
        "gap": round(gap, 6),
        "verdict": "真实信号显著优于置换 ⇒ 仪器对信号结构有分辨率 ✓" if gap > 0.10
        else "⚠ 置换流读数与真实流接近 ⇒ 仪器无分辨率，本批判废"}
    print(f"  阴性对照: 真实 {real_ret:+.2%} vs 置换 {m_neg['total_return']:+.2%} "
          f"⇒ {results['_negative_control']['verdict']} ({time.time()-t0:.0f}s)", flush=True)
    # ---- 仪器确定性自证（#15 忠实批无外部锚点 ⇒ 同输入重放必须逐位一致）----
    # 2026-10-08 审查批补：防伪闸门 A3 要求报告自带可核验的自证块；跨批锚点由
    # #16 ddhalt 的 V0 对本档 F_RL2_daily/F_MSE2_daily 承担。
    t0 = time.time()
    eq2, _m2 = arm_faithful(rl2, T3, cfg, bundle, panels, dates, start_ts, ema_step="daily")
    ref_eq = curves["F_RL2_daily"]
    dmax = (float((eq2 - ref_eq).abs().max())
            if len(eq2) == len(ref_eq) and (eq2.index == ref_eq.index).all() else float("inf"))
    results.update(scaffold_selfcheck(
        "_rerun_determinism_F_RL2_daily",
        {"equity_max_abs_diff": dmax}, 1e-9, bool(dmax <= 1e-9),
        "同输入重复执行 arm_faithful(F_RL2_daily) ⇒ 仪器确定性自证（docs/03 §41）"))
    print(f"  确定性自证: max|Δequity|={dmax:.2e} ({time.time()-t0:.0f}s)", flush=True)
    for tag, (a, b) in {"P2_faithful_rebalance": ("F_RL2_rebalance", "F_MSE2_rebalance"),
                        "P2_faithful_daily": ("F_RL2_daily", "F_MSE2_daily")}.items():
        p_m = monthly_log_paired(curves[a], curves[b], month_ends)
        results[tag] = {**{k: round(float(v), 5) for k, v in p_m.items()
                           if isinstance(v, (int, float, np.floating))},
                        "n": int(p_m["n"]),
                        "pass": bool(p_m["t"] >= C2A_T and p_m["mean_log"] > 0)}
        print(f"  {tag}: t={p_m['t']:+.2f} mean_log={p_m['mean_log']:+.5f} n={p_m['n']} "
              f"pass={results[tag]['pass']}", flush=True)
    _dump(results, curves, suffix="faithful")
    return 0


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "config.yaml")
    print("== 面板 / 特征 / 基准 ==", flush=True)
    panels = sa.load_panels(cfg)
    assert panels["close"].index.max() <= MAX_DATA_DATE, "面板已越过前瞻窗隔离线"
    bundle = build_dataset(cfg, compute_fingerprints=False)
    universe = bundle.universe
    dates = panels["close"].index
    start_ts = pd.Timestamp(START_OOS)
    s = pd.Series(dates, index=dates)
    month_ends = [d for d in s.groupby([dates.year, dates.month]).last() if d >= start_ts]

    # S1 基线在**真 T3 档**重跑（#13 当时把基线放 20 万、把集成放 2 万）。
    # 调用形态与 dlv2_hurdle_chain.py:126-131 逐字相同（含 tier 仍传 S1_TIER 本身），
    # 读数须对得上存档 S1_T3（−20.44% / MDD −38.31% / 成本 183,879）⇒ 基线复现。
    cfg_s1 = replace(cfg,
                     account=replace(cfg.account, initial_equity=T3["equity"],
                                     max_symbols=T3["max_symbols"],
                                     max_lots_per_symbol=T3["max_lots"]),
                     signal=replace(cfg.signal, top_n=S1_TIER["top_n"]))
    eq_s1, m_s1 = _rule_tier_arm(cfg_s1, S1_TIER, bundle, panels, dates, start_ts)
    print(f"  S1@真T3: {m_s1['total_return']:+.2%} / MDD {m_s1['max_drawdown']:+.2%} / "
          f"成本 {m_s1['total_costs']:.0f}", flush=True)

    # 基准指数：000905.SH（中证500）——统一走 sync_index（L1 契约：只读老库 → 本仓面板），
    # 缺面板时大声报错提示先跑 scripts/sync_index.py（不再四处内联 SQL 查询）。
    bench = sync_index.load(cfg, "000905.SH")["close"]
    bench = bench.loc[bench.index >= start_ts].astype("float64")

    mse = load_panel_map(MSE_STREAMS)
    rl = load_panel_map(RL_STREAMS)
    mse2 = mean_panel([mse["ss7_21_99_5_13"], mse["ss123_2024_42_314_2027"]])
    mse4 = mean_panel(list(mse.values()))
    rl2 = mean_panel(list(rl.values()))

    if "--faithful" in sys.argv:
        return _faithful_mode(cfg, bundle, panels, dates, start_ts, month_ends, bench,
                              mse, rl, mse2, mse4, rl2, eq_s1, m_s1)
    results: dict[str, dict] = {"baseline_S1_T3": {
        "total_return": round(float(m_s1["total_return"]), 6),
        "max_drawdown": round(float(m_s1["max_drawdown"]), 6),
        "total_costs": round(float(m_s1["total_costs"]), 1),
        "n_blocked_cash": int(m_s1.get("n_blocked_cash", 0)),
        "position_util_mean": round(float(m_s1.get("position_util_mean", float("nan"))), 4)}}
    results["_repro"] = repro_context(cfg, panels, bundle,
                                      list(MSE_STREAMS.values()) + list(RL_STREAMS.values()),
                                      harness="two_pass", ema_step="mixed")

    # ---- X0 自证：单流 ss7@T3，hurdle=10（#12 的 X3 原样）----
    t0 = time.time()
    eq, m = HC.run_cema(10, T3, mse["ss7_21_99_5_13"], cfg, bundle, panels, dates, start_ts)
    drift = {k: round(abs(float(m[k]) - X0_REF[k]), 8) for k in X0_REF}
    x0_ok = all(v < 1e-4 for v in drift.values())
    results["X0_single_stream_D7_T3"] = {**m, "x0_ref": X0_REF,
                                         "x0_ref_provenance": X0_REF_PROVENANCE,
                                         "abs_drift": drift,
                                         "x0_determinism_ok": bool(x0_ok)}
    print(f"  X0 自证: {m['total_return']:+.2%} / MDD {m['max_drawdown']:+.2%} / "
          f"成本 {m['total_costs']:.0f} ⇒ 逐位一致={x0_ok} 漂移={drift} "
          f"({time.time()-t0:.0f}s)", flush=True)
    if not x0_ok:
        print("!! X0 不复现 ⇒ 重放路径本身有问题，停止出数", flush=True)
        _dump(results, {})
        return 3

    # ---- R1：MSE 4 流集成 @真T3（#13 原判据对象的正确档位版）----
    # ---- R2/R3：rank-loss 2 流 vs MSE 2 流 @真T3（#14 P2 的正确档位版）----
    # 四门一律对同档 S1 基线；P2 的"损失单变量"配对在下方单独算（R2 vs R3）。
    arms = [("R1_MSE4_ensemble_T3", mse4, "rebalance"),
            ("R2_RANKLOSS2_T3", rl2, "rebalance"),
            ("R3_MSE2_T3", mse2, "rebalance")]
    curves: dict[str, pd.Series] = {}
    for name, panel, step in arms:
        t0 = time.time()
        eq, m = HC.run_cema(0, T3, panel, cfg, bundle, panels, dates, start_ts)
        curves[name] = eq
        g = gates(eq, m, eq_s1, m_s1, month_ends)
        g["ema_step"] = step
        g["beta_vs_000905"] = beta_ir(eq, bench)
        results[name] = g
        print(f"  {name}: {m['total_return']:+.2%} MDD {m['max_drawdown']:+.2%} "
              f"成本 {m['total_costs']:.0f} C2a t={g['C2a']['t']:.2f} "
              f"β={g['beta_vs_000905'].get('beta')} IR={g['beta_vs_000905'].get('resid_IR_ann')} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # P2 配对：R2 vs R3（同为 2 流、同档、同语义，只差训练损失）
    t0 = time.time()
    eq_mse2 = curves["R3_MSE2_T3"]
    p2 = monthly_log_paired(curves["R2_RANKLOSS2_T3"], eq_mse2, month_ends)
    results["P2_rankloss_vs_MSE_at_T3"] = {**{k: (round(float(v), 5) if isinstance(v, (int, float, np.floating)) else v) for k, v in p2.items()},
                                            "pass": bool(p2["t"] >= C2A_T and p2["mean_log"] > 0)}
    print(f"  P2 配对(R2−R3): t={p2['t']:+.2f} mean_log={p2['mean_log']:+.5f} n={p2['n']} "
          f"pass={results['P2_rankloss_vs_MSE_at_T3']['pass']} ({time.time()-t0:.0f}s)", flush=True)

    # ---- R4：生产语义（daily 步进）双臂，alpha=1.0 直通预平滑面板 ----
    orig_alpha = HC.EMA_ALPHA
    HC.EMA_ALPHA = 1.0
    try:
        for name, panel in (("R4a_RANKLOSS2_dailyT3", rl2), ("R4b_MSE2_dailyT3", mse2)):
            t0 = time.time()
            dsp = daily_step_panel(panel, universe)
            eq, m = HC.run_cema(0, T3, dsp, cfg, bundle, panels, dates, start_ts)
            curves[name] = eq
            g = gates(eq, m, eq_s1, m_s1, month_ends)
            g["ema_step"] = "daily"
            g["beta_vs_000905"] = beta_ir(eq, bench)
            results[name] = g
            print(f"  {name}: {m['total_return']:+.2%} MDD {m['max_drawdown']:+.2%} "
                  f"成本 {m['total_costs']:.0f} C2a t={g['C2a']['t']:.2f} "
                  f"β={g['beta_vs_000905'].get('beta')} "
                  f"IR={g['beta_vs_000905'].get('resid_IR_ann')} ({time.time()-t0:.0f}s)",
                  flush=True)
        p4 = monthly_log_paired(curves["R4a_RANKLOSS2_dailyT3"], curves["R4b_MSE2_dailyT3"],
                                month_ends)
        results["P2_daily_step_paired"] = {**{k: (round(float(v), 5) if isinstance(v, (int, float, np.floating)) else v) for k, v in p4.items()},
                                           "pass": bool(p4["t"] >= C2A_T and p4["mean_log"] > 0)}
        print(f"  P2(daily 步进配对): t={p4['t']:+.2f} n={p4['n']}", flush=True)
    finally:
        HC.EMA_ALPHA = orig_alpha

    _dump(results, curves)
    return 0


def _dump(results: dict, curves: dict, suffix: str = "") -> None:
    """统一存档（faithful / two-pass 两路合并；旧 _dump_results 并入，2026-10-08 审查批）。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    tag = f"tierfix_{suffix}_{stamp}" if suffix else f"tierfix_{stamp}"
    # results["_repro"] 提升为 payload["_meta"]：闸门 A8 只查顶层/_meta 的数据窗右端点
    meta = results.pop("_repro", None)
    payload = {"prereg": "plans/tierfix-prereg-20261008.md", "oos_window": START_OOS,
               "tier": T3, "_meta": meta, "x0_determinism_ok": bool(
                   results.get("X0_single_stream_D7_T3", {}).get("x0_determinism_ok")),
               "results": results, "verdict_word": None}
    txt = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    (REPORTS / f"{tag}.json").write_text(txt, encoding="utf-8")
    (OUTDIR / f"{tag}.json").write_text(txt, encoding="utf-8")
    lines = [f"#15 重放批（{suffix or 'two-pass'}，真 T3 档） {stamp}", ""]
    for k, v in results.items():
        lines.append(f"{k}: {v}")
    (REPORTS / f"{tag}.txt").write_text("\n".join(lines), encoding="utf-8")
    for name, eq in curves.items():
        eq.to_frame("equity").to_parquet(OUTDIR / f"eq_{name}.parquet")
    print(f"存档 → {REPORTS}/{tag}.json|txt + {OUTDIR}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
