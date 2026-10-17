"""手数 sizer —— A股小资金的核心约束：100 股整手 + 最低佣金 5 元。

风险预算法：股数 = 权益 × risk_per_trade / (atr_k × ATR14)，向下取整手
    （止损距离 = atr_k × ATR14，未复权真实价口径）
波动率上限法：手数 = floor(权益 × max_pos_vol_daily / (价 × 100 × 预测日波))
    用模型 vol 头预测替代历史波动（TriHeadNet 三头设计的下游用途）
最终手数 = min(两法) 且 ≥0，再 cap 到 max_lots_per_symbol。
0 手是合法输出（预算不足放弃信号）——最低佣金约束下，预算不足的单强行
成交会让成本占比失控（1手×1000元 → 5元佣金 = 0.5%/边）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import contracts
from ..config import Config


@dataclass(frozen=True)
class SizeResult:
    lots: int
    method_used: str          # risk_budget / vol_target / zero
    stop_distance: float      # 每股止损距离（元）
    risk_amount: float        # 该手数下的实际风险金额（止损打满口径，元）


def risk_budget_lots(cfg: Config, equity: float, price: float, atr_value: float,
                     atr_k: float | None = None) -> int:
    """风险预算法手数（100 股整手离散化向下取整）。atr_k 缺省取配置值。"""
    k = cfg.account.atr_k if atr_k is None else atr_k
    if not np.isfinite(atr_value) or atr_value <= 0 or not np.isfinite(price) or price <= 0:
        return 0
    stop_dist = k * atr_value                       # 元/股
    shares = equity * cfg.account.risk_per_trade / stop_dist
    return int(shares // contracts.LOT_SIZE)


def vol_cap_lots(cfg: Config, equity: float, price: float,
                 vol_pred_log: float | None = None) -> int:
    """单仓位波动上限法手数。vol_pred_log 是 log1p(vol*100) 口径（模型 vol 头，已校准）。

    单手日波动金额 ≈ price × 100 × daily_vol。
    lots = floor(equity × max_pos_vol_daily / 单手日波动金额)。
    语义继承期货版 2026-09-26 重定义：否决"预测波动贡献超上限"的持仓。
    """
    import math

    if vol_pred_log is None or not np.isfinite(price) or price <= 0:
        return 0
    vol_daily = math.expm1(vol_pred_log) / 100.0        # 逆 np.log1p(v*100)（自然对数）
    if not np.isfinite(vol_daily) or vol_daily <= 0:
        return 0
    per_lot_daily_amt = price * contracts.LOT_SIZE * vol_daily
    if per_lot_daily_amt <= 0:
        return 0
    return int(equity * cfg.account.max_pos_vol_daily / per_lot_daily_amt)


def size_position(cfg: Config, equity: float, price: float, atr_value: float,
                  vol_pred_log: float | None = None) -> SizeResult:
    """双约束取小。0 手合法（预算不足放弃信号）。"""
    lots_rb = risk_budget_lots(cfg, equity, price, atr_value)
    lots_vt = (vol_cap_lots(cfg, equity, price, vol_pred_log)
               if vol_pred_log is not None else lots_rb)
    lots = min(lots_rb, lots_vt)
    lots = min(lots, cfg.account.max_lots_per_symbol)
    if lots <= 0:
        return SizeResult(0, "zero", cfg.account.atr_k * atr_value, 0.0)
    stop_distance = cfg.account.atr_k * atr_value
    return SizeResult(lots, "risk_budget" if lots_rb <= lots_vt else "vol_target",
                      stop_distance, stop_distance * contracts.shares_of(lots))
