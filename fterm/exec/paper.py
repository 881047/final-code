"""PaperBroker —— 本地纸面撮合：收盘价/实时价驱动，成本口径与回测同源。

状态全部在内存 + state 库（由 runner 落库），重启后从 state 恢复持仓。
滑点/费用公式与 backtest.costs 完全共用（单一口径承诺）。
A股现货语义：只做多、全额现金交收、卖出资金当日可用（先卖后买同日合法）。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .. import contracts
from ..backtest.costs import fill_price_with_slippage, trade_cost
from ..config import Config
from .broker import AccountSnapshot, Broker, FillReport, Order, Side


@dataclass
class PaperPosition:
    product: str
    direction: int
    lots: int
    entry_price: float
    adj_entry: float = 1.0    # 入场日复权因子（含权入账基准，docs/03 §38）


class PaperBroker(Broker):
    def __init__(self, cfg: Config, equity: float | None = None,
                 now: pd.Timestamp | None = None):
        self.cfg = cfg
        self.cash = equity if equity is not None else cfg.account.initial_equity
        self.positions: dict[str, PaperPosition] = {}
        self.pending: dict[str, Order] = {}
        self.fills: list[FillReport] = []
        self._order_seq = 0
        self.now = now    # 费用时间分段（印花税）所需；None 时用当前日期

    def _fee_date(self) -> pd.Timestamp:
        return self.now if self.now is not None else pd.Timestamp.now().normalize()

    def _price_feed(self, product: str) -> float:
        """行情价源：由 runner 注入（QuoteFeed），测试注入固定价。"""
        raise NotImplementedError("PaperBroker 需要 price_fn 注入（见 bind_price_feed）")

    def bind_price_feed(self, price_fn) -> None:
        self._price_feed_real = price_fn

    def submit(self, order: Order) -> str:
        self._order_seq += 1
        oid = f"paper-{self._order_seq:06d}"
        self.pending[oid] = order
        return oid

    def fill_order_at(self, order: Order, ref_price: float,
                      adj: float = 1.0) -> FillReport:
        """按给定参考价撮合单个订单（paper 日频链：昨日挂单 → 今日开盘价执行）。

        `adj` = 今日复权因子。买入把它记进 `adj_entry`；卖出按 `adj/adj_entry` 把
        除权除息的权利兑现进现金（送转股数 + 现金分红等效），成交价与滑点仍按真实
        未复权参考价计（现金交收/整手/tick 一律不离真实价）。
        """
        rep = self._fill(f"paper-live-{self._order_seq}", order, ref_price, adj)
        self._order_seq += 1
        return rep

    def execute_pending(self, ref_prices: dict[str, float],
                        adj_prices: dict[str, float] | None = None) -> list[FillReport]:
        """以给定参考价撮合全部挂单（回测/收盘复盘中直接喂价）。"""
        reports = []
        for oid, order in list(self.pending.items()):
            ref = ref_prices.get(order.product)
            if ref is None:
                continue
            rep = self._fill(oid, order, ref, (adj_prices or {}).get(order.product, 1.0))
            reports.append(rep)
            del self.pending[oid]
        return reports

    def _fill(self, oid: str, order: Order, ref_price: float,
              adj: float = 1.0) -> FillReport:
        direction = 1 if order.side == Side.BUY else -1
        cur0 = self.positions.get(order.product)
        # 卖出的价值倍数 = 今日因子 / 入场因子（建仓时恒为 1.0）
        ratio = 1.0 if direction == 1 else (
            (adj / (cur0.adj_entry or 1.0)) if (cur0 is not None and adj and adj > 0) else 1.0)
        if not (ratio > 0):
            ratio = 1.0
        price = fill_price_with_slippage(ref_price, direction,
                                         self.cfg.backtest.slippage_ticks)
        cur = self.positions.get(order.product)
        if direction == -1 and (cur is None or cur.lots < order.lots):
            # 卖侧硬拦在费用计算之前：被拦单零现金影响（blocked_cash 同语义）。
            # 2026-10-06 审查修复：原实现无持仓静默空操作却照报成交、
            # 超卖按持仓截断且报告写满额手数。
            msg = "blocked_no_position" if cur is None else f"blocked_oversell(held={cur.lots})"
            rep = FillReport(order, price, 0, 0.0, ts="", ok=False, msg=msg)
            self.fills.append(rep)
            return rep
        cost = trade_cost(self.cfg, ref_price, direction, order.lots,
                          self._fee_date(), value_ratio=ratio)
        eff_price = price * ratio      # 含权等效成交价（现金按这个动）
        if direction == 1:
            needed = price * contracts.shares_of(order.lots) + cost.fee
            if needed > self.cash + 1e-6:
                rep = FillReport(order, price, order.lots, 0.0, ts="", ok=False,
                                 msg="blocked_cash")
                self.fills.append(rep)
                return rep                # 现金硬校验（与回测引擎同语义，2026-10-05 首跑抓到）
        self.cash -= cost.fee    # 费用即扣；买卖价差（含滑点）按市值口径进出
        shares = contracts.shares_of(order.lots)
        if direction == 1:
            self.cash -= price * shares
            if cur is None:
                self.positions[order.product] = PaperPosition(order.product, 1, order.lots,
                                                              price, adj_entry=adj or 1.0)
            else:    # 加仓
                tot = cur.lots + order.lots
                cur.entry_price = (cur.entry_price * cur.lots + price * order.lots) / tot
                cur.adj_entry = ((cur.adj_entry or 1.0) * cur.lots + (adj or 1.0)
                                 * order.lots) / tot
                cur.lots = tot
        else:    # 卖出：先减仓后加现金（同日先卖后买合法；仓位合法性已在上方硬拦）
            if cur.lots == order.lots:
                self.cash += eff_price * contracts.shares_of(cur.lots)
                del self.positions[order.product]
            else:
                self.cash += eff_price * contracts.shares_of(order.lots)
                cur.lots -= order.lots
        rep = FillReport(order, eff_price, order.lots, cost.fee, ts="", ok=True,
                         msg=f"slip={cost.slippage:.1f}"
                         + (f" ratio={ratio:.4f}" if ratio != 1.0 else ""))
        self.fills.append(rep)
        return rep

    def poll(self) -> list[FillReport]:
        out = self.fills[:]
        self.fills.clear()
        return out

    def account(self, mark_prices: dict[str, float] | None = None,
                adj_prices: dict[str, float] | None = None) -> AccountSnapshot:
        """`adj_prices`：当日复权因子。给出则市值按含权口径计（与回测引擎同源）。"""
        marks = mark_prices or {}
        adjs = adj_prices or {}
        pos_value = 0.0
        pos_out = {}
        for p, pos in self.positions.items():
            px = marks.get(p, pos.entry_price)
            r = adjs.get(p)
            ratio = (r / (pos.adj_entry or 1.0)) if (r and r > 0) else 1.0
            value = px * contracts.shares_of(pos.lots) * ratio
            pos_value += value
            pos_out[p] = {"direction": pos.direction, "lots": pos.lots, "value": value}
        return AccountSnapshot(equity=self.cash + pos_value, available=self.cash,
                               pos_value=pos_value, positions=pos_out)

    def cancel_all(self) -> int:
        n = len(self.pending)
        self.pending.clear()
        return n
