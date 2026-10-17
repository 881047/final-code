"""回测指标 —— 收益/风险/成本三轴，小资金视角重点看回撤与成本占比。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from .engine import Fill

TRADING_DAYS = 243    # A股年交易日近似


def max_drawdown(equity: pd.Series) -> tuple[float, pd.Timestamp, pd.Timestamp]:
    """返回 (最大回撤比例, 峰值日, 谷底日)。"""
    peak = equity.cummax()
    dd = equity / peak - 1.0
    trough = dd.idxmin()
    peak_date = equity.loc[:trough].idxmax()
    return float(dd.min()), peak_date, trough


def compute_metrics(cfg: Config, equity: pd.Series, closed_trades: list[dict],
                    fills: list[Fill], pos_value: pd.Series) -> dict:
    ret = equity.pct_change().dropna()
    total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) > 1 else 0.0
    ann_ret = float((1 + total_ret) ** (TRADING_DAYS / max(len(ret), 1)) - 1.0) if len(ret) else 0.0
    vol = float(ret.std() * np.sqrt(TRADING_DAYS)) if len(ret) > 1 else 0.0
    sharpe = float(ann_ret / vol) if vol > 1e-9 else 0.0
    mdd, peak_d, trough_d = max_drawdown(equity)
    calmar = float(ann_ret / abs(mdd)) if mdd < -1e-9 else 0.0

    wins = [t for t in closed_trades if t["pnl"] > 0]
    gross_win = sum(t["pnl"] for t in wins)
    gross_loss = -sum(t["pnl"] for t in closed_trades if t["pnl"] <= 0)
    total_costs = sum(t["costs"] for t in closed_trades)
    gross_pnl = sum(t["raw_pnl"] for t in closed_trades)
    n_cash_blocks = sum(1 for f in fills if f.action == "blocked_cash")
    n_limit_blocks = sum(1 for f in fills if f.action == "blocked")
    n_t1_blocks = sum(1 for f in fills if f.action == "blocked_t1")

    return {
        "total_return": total_ret,
        "ann_return": ann_ret,
        "ann_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "mdd_peak": str(peak_d.date()) if len(equity) else "",
        "mdd_trough": str(trough_d.date()) if len(equity) else "",
        "calmar": calmar,
        "n_trades": len(closed_trades),
        "win_rate": len(wins) / len(closed_trades) if closed_trades else 0.0,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 1e-9 else float("inf") if gross_win > 0 else 0.0,
        "total_costs": total_costs,
        "cost_ratio": total_costs / gross_pnl if abs(gross_pnl) > 1e-9 else float("nan"),
        "position_util_mean": float((pos_value / equity.reindex(pos_value.index)).mean()) if len(pos_value) else 0.0,
        "position_util_max": float((pos_value / equity.reindex(pos_value.index)).max()) if len(pos_value) else 0.0,
        "n_blocked_limit": n_limit_blocks,
        "n_blocked_cash": n_cash_blocks,
        "n_blocked_t1": n_t1_blocks,
        "final_equity": float(equity.iloc[-1]) if len(equity) else cfg.account.initial_equity,
    }


def metrics_summary(m: dict) -> str:
    lines = [
        f"final equity = {m['final_equity']:.0f}  total return = {m['total_return']:+.2%}",
        f"ann ret = {m['ann_return']:+.2%}  vol = {m['ann_vol']:.2%}  sharpe = {m['sharpe']:.2f}  calmar = {m['calmar']:.2f}",
        f"maxDD = {m['max_drawdown']:.2%} ({m['mdd_peak']} → {m['mdd_trough']})",
        f"trades = {m['n_trades']}  win = {m['win_rate']:.1%}  PF = {m['profit_factor']:.2f}",
        f"costs = {m['total_costs']:.0f} 元 (占毛利 {m['cost_ratio']:.1%})",
        f"position util mean/max = {m['position_util_mean']:.1%} / {m['position_util_max']:.1%}  blocked(limit/cash/t1) = {m['n_blocked_limit']}/{m['n_blocked_cash']}/{m['n_blocked_t1']}",
    ]
    return "\n".join(lines)
