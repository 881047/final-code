"""Broker 抽象 —— 纸面执行与未来实盘通道共用接口。

A股实盘切换路径（期货版 CTP/openctp 随期货域移除）：paper.PaperBroker →
券商程序化通道（如 miniQMT/PTrade，接入时实现本接口即可），业务代码（runner）
只依赖本接口，不感知通道差异（架构设计承诺）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class Side(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True)
class Order:
    product: str            # ts_code（如 000001.SZ）
    side: Side
    lots: int               # 手数（1手=100股）
    reason: str = ""


@dataclass
class FillReport:
    order: Order
    price: float
    lots: int
    fee: float
    ts: str
    ok: bool
    msg: str = ""


@dataclass
class AccountSnapshot:
    equity: float
    available: float        # 可用资金（现金）
    pos_value: float        # 持仓市值
    positions: dict[str, dict] = field(default_factory=dict)   # code → {direction, lots, value}


class Broker(ABC):
    """同步风格接口（个人终端日频足够）。"""

    @abstractmethod
    def submit(self, order: Order) -> str:
        """提交订单，返回 order_id。"""

    @abstractmethod
    def poll(self) -> list[FillReport]:
        """拉取成交回报。"""

    @abstractmethod
    def account(self, mark_prices: dict[str, float] | None = None) -> AccountSnapshot:
        """当前账户快照。mark_prices 提供时按该价组盯市（runner 收盘口径）。"""

    @abstractmethod
    def cancel_all(self) -> int:
        """撤销全部未成交单（收盘前风控用），返回撤销数量。"""
