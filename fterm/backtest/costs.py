"""成本模型 —— 回测与实盘共用的唯一口径（与 contracts 同源）。

A股现货：费用 = 佣金（万2.5，单边最低5元）+ 印花税（卖出，时间分段）+ 过户费（双边）。
滑点 = 参考价 ± n×tick（0.01 元/股），成交价按跳取整。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .. import contracts
from ..config import Config


@dataclass(frozen=True)
class TradeCost:
    fee: float            # 全费用（元）：佣金+印花税+过户费
    slippage: float       # 滑点损耗（元，正数=损失）
    fill_price: float     # 含滑点的成交价（元/股，跳取整）


def fill_price_with_slippage(ref_price: float, direction: int,
                             slippage_ticks: int) -> float:
    """买 +滑点、卖 -滑点（不利方向）。direction ∈ {+1,-1}。"""
    raw = ref_price + direction * slippage_ticks * contracts.TICK
    return round(raw, 2)


def trade_cost(cfg: Config, ref_price: float, direction: int, lots: int,
               date: pd.Timestamp, value_ratio: float = 1.0) -> TradeCost:
    """一笔成交的全成本。direction=+1 买入 / -1 卖出；lots 以手计。

    `value_ratio`（≥0，默认 1.0）= 含权入账的**价值倍数**：持仓在除权除息日按
    `adj[d]/adj[entry]` 累积的权利（送转股数 + 现金分红等效）。成交价与现金流仍按
    **真实未复权价**计（现金交收、整手、tick 滑点、停板判定都不许离开真实价），
    只有"这笔交易实际动了多少钱"的名义额按倍数放大 ⇒ 佣金/印花税/过户费按真实
    成交额计（送转后股数确实变多，据此计费是正确的；纯现金分红情形下会略微高估
    费用，方向保守）。
    """
    assert direction in (1, -1), "direction 必须是 ±1"
    assert lots >= 1, "lots 必须 ≥1（0 手不产生成交）"
    shares = contracts.shares_of(lots)
    fp = fill_price_with_slippage(ref_price, direction, cfg.backtest.slippage_ticks)
    amount = fp * shares * value_ratio
    fee = contracts.trade_fee(amount, direction, date)
    slip_cost = abs(fp - ref_price) * shares * value_ratio
    return TradeCost(fee=fee, slippage=slip_cost, fill_price=fp)
