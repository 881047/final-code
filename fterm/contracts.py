"""A股现货交易规格 —— 回测撮合与纸面执行共用的唯一口径来源。

与期货版（multiplier/保证金/手数乘数）的本质差异：
  - 现货全额现金交收，无保证金、无杠杆、无做空（T+1）
  - 整手 = 100 股，手数是唯一的仓位离散单位
  - 成本 = 佣金（万 2.5，单边最低 5 元）+ 印花税（卖出，时间分段）+ 过户费（双边）

费率依据（docs/02-operations.md §参数基线 登记处）：
  - 佣金万 2.5 / 最低 5 元：2026 年主流互联网券商常见水平（近似值）
  - 印花税：2023-08-28 起卖出 0.05%（财政部 2023 年减半公告）；此前 0.1%
  - 过户费：2022-04-29 起双边 0.001%；此前 0.002%（中国结算公告口径）
涨跌停：优先用老库 stk_limit 真实停板价（2017 起有全量），缺表时按主板 ±10%
近似回退（近似口径只允许出现在数据缺失的兜底路径，见 data.ashare）。
"""

from __future__ import annotations

import pandas as pd

LOT_SIZE = 100                # 整手股数
TICK = 0.01                   # 最小变动价位（元/股）
COMMISSION_RATE = 0.00025     # 佣金万 2.5（双边）
MIN_COMMISSION = 5.0          # 单笔佣金下限（元）
TRANSFER_FEE_RATE_NEW = 0.00001   # 过户费 2022-04-29 起（双边）
TRANSFER_FEE_RATE_OLD = 0.00002   # 2022-04-29 前
STAMP_DUTY_CUT_DATE = pd.Timestamp("2023-08-28")  # 印花税减半生效日（卖出）
STAMP_DUTY_NEW = 0.0005       # 2023-08-28 起卖出印花税
STAMP_DUTY_OLD = 0.0010       # 此前卖出印花税


def transfer_fee_rate(date: pd.Timestamp) -> float:
    return TRANSFER_FEE_RATE_NEW if pd.Timestamp(date) >= pd.Timestamp("2022-04-29") else TRANSFER_FEE_RATE_OLD


def stamp_duty_rate(date: pd.Timestamp) -> float:
    """卖出印花税率（买入不收）。时间分段，见模块头依据。"""
    return STAMP_DUTY_NEW if pd.Timestamp(date) >= STAMP_DUTY_CUT_DATE else STAMP_DUTY_OLD


def shares_of(lots: int) -> int:
    """手数 → 股数。"""
    return int(lots) * LOT_SIZE


def trade_fee(amount: float, side: int, date: pd.Timestamp) -> float:
    """一笔成交的全费用（元）。

    side=+1 买入：佣金 + 过户费；side=-1 卖出：佣金 + 印花税 + 过户费。
    amount = 成交金额（元，不含费用口径的股数×价格）。
    """
    assert side in (1, -1), "side 必须是 ±1"
    commission = max(amount * COMMISSION_RATE, MIN_COMMISSION)
    duty = amount * stamp_duty_rate(date) if side == -1 else 0.0
    transfer = amount * transfer_fee_rate(date)
    return commission + duty + transfer
