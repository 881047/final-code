"""RiskGovernor —— 组合级硬约束审批（2 万元本金的最后防线）。

审批顺序（全部通过才放行；A股现货只做多，direction ∈ {0, +1}）：
  1. total_dd_halt  总回撤熔断：权益 < 峰值×(1-12%) → 全部拒绝（人工确认才重置）
  2. daily_stop     当日浮亏 > 3% → 只许卖不许买
  3. max_symbols    同时持仓股票数上限
  4. cash_util      持仓市值（含新单，按最近收盘价估值）≤ 权益 × max_cash_util
  5. lots cap       单票手数上限（防御性，sizer 已 cap）

拒绝也要产出记录（含原因）——可审计性优先。熔断强平语义继承期货版
docs/03 §13 A/B 判决：flag 默认 off（只拒新开）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..backtest.engine import Target
from ..config import Config


@dataclass
class AccountState:
    equity: float
    peak_equity: float
    day_pnl: float                      # 当日已实现+浮动（正负）
    open_positions: dict[str, dict]     # code → {direction, lots, value}
    day_start_equity: float | None = None   # 日初（前一记录日收盘）权益；None=退回 peak 口径（兼容旧调用方）
    available_cash: float | None = None     # 引擎/账户真实可用现金；None=不另设现金闸（兼容旧调用方）


@dataclass
class Decision:
    target: Target
    approved: bool
    reason: str


@dataclass
class GovernorReport:
    decisions: list[Decision] = field(default_factory=list)

    @property
    def approved(self) -> list[Target]:
        return [d.target for d in self.decisions if d.approved]


class RiskGovernor:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def approve(self, targets: list[Target], state: AccountState,
                prices: dict[str, float] | None = None,
                exempt_batch_exits: bool = False) -> GovernorReport:
        """prices：审批时点估值价（最近收盘）。开仓审批必需——缺价即拒（不盲批）。

        exempt_batch_exits（#135 仪器开关，默认 False = 现行语义逐位不变）：
        True 时本批**已获批准的平仓**即时释放槽位与市值/现金——
        max_symbols 计数扣除、value_now 与可用现金加回平仓市值。
        语义依据：生产引擎同日先执行卖出后执行买入（卖出回款当日可用），
        而审批的 open_positions/available_cash 快照含本批待平仓 ⇒ 现行默认语义
        在"全平仓+新建仓"的换仓日必然全部拒买（#131 诊断：B7 拒单 16,628 次；
        A0 亦有 539 次同源拒单）。该开关把审批与执行的同批语义对齐；
        是否将其启用到生产位是独立拍板，本参数仅为判据链仪器而设。
        """
        cfg = self.cfg.account
        prices = prices or {}
        report = GovernorReport()
        halted = state.equity < state.peak_equity * (1 - cfg.total_dd_halt)
        # 日停分母 = 日初权益（"当日浮亏 3%"的正确基数）。原用 peak_equity：
        # 回撤越深 peak 相对日初越高 ⇒ 阈值越负 ⇒ 日停越松（2026-10-06 审查修复）。
        day_denom = (state.day_start_equity if state.day_start_equity is not None
                     else state.peak_equity)
        day_blocked = state.day_pnl < -day_denom * cfg.daily_stop_loss

        committed = 0.0     # 本批已批准开仓的市值累加（批内第二单须看到第一单，2026-10-05 首跑教训）
        exited: set[str] = set()          # exempt_batch_exits：本批已批准平仓
        exited_value = 0.0                # 同上：被平持仓的市值（估值口径）
        for t in targets:
            if halted:
                if t.direction == 0 or self._reduces_risk(t, state):
                    report.decisions.append(Decision(t, True, "halt_reduce_only"))
                else:
                    report.decisions.append(Decision(t, False, f"halted(总回撤>{cfg.total_dd_halt:.0%})"))
                continue
            if day_blocked and not self._reduces_risk(t, state):
                report.decisions.append(Decision(t, False, f"day_stop(当日亏损>{cfg.daily_stop_loss:.0%})"))
                continue
            if t.direction != 0:
                held_eff = (set(state.open_positions) - exited) if exempt_batch_exits \
                    else set(state.open_positions)
                if t.product not in held_eff and len(held_eff) >= cfg.max_symbols:
                    report.decisions.append(Decision(t, False, f"max_symbols(>{cfg.max_symbols})"))
                    continue
                if t.lots > cfg.max_lots_per_symbol:
                    report.decisions.append(Decision(t, False, f"lots_cap(>{cfg.max_lots_per_symbol})"))
                    continue
                est = self._value_estimate(t, prices)
                if est is None:
                    report.decisions.append(Decision(t, False, "no_price_for_valuation"))
                    continue
                value_now = sum(p.get("value", 0.0) for p in state.open_positions.values())
                value_now -= state.open_positions.get(t.product, {}).get("value", 0.0)
                if exempt_batch_exits:
                    value_now -= exited_value
                if value_now + est + committed > state.equity * cfg.max_cash_util:
                    report.decisions.append(Decision(t, False, f"cash_util(>{cfg.max_cash_util:.0%})"))
                    continue
                # 可用现金硬闸（2026-10-08 忠实回测）：审批时就用引擎给的真实可用现金
                # 卡一遍 ⇒ "风控批了、引擎因现金不足拒单、决策环却记成持仓"的幻象持仓
                # 通道被关掉（docs/03 §29）。available_cash=None 时不启用（旧调用方兼容）。
                cash_eff = (state.available_cash + exited_value) \
                    if (exempt_batch_exits and state.available_cash is not None) \
                    else state.available_cash
                if (cash_eff is not None
                        and est + committed > cash_eff + 1e-6):
                    report.decisions.append(Decision(t, False,
                                                     f"cash_short(可用{state.available_cash:.0f})"))
                    continue
                committed += est
            elif exempt_batch_exits and t.direction == 0:
                # 本批平仓即时释放槽位/市值/现金（豁免开关下；默认路径零变化）
                if t.product in state.open_positions and t.product not in exited:
                    exited.add(t.product)
                    exited_value += state.open_positions[t.product].get("value", 0.0)
            report.decisions.append(Decision(t, True, "ok"))

        # 熔断强平（继承 docs/03 §13）：halted 时对当日没有清仓目标的存量持仓补发清零目标
        if halted and cfg.circuit_force_liquidate:
            covered = {t.product for t in targets if t.direction == 0}
            for p in state.open_positions:
                if p not in covered:
                    report.decisions.append(
                        Decision(Target(p, 0, 0, "circuit_force_liquidate"), True, "halt_force_liquidate"))
        return report

    @staticmethod
    def _reduces_risk(t: Target, state: AccountState) -> bool:
        if t.direction == 0:
            return True    # 清仓/无持仓空目标：纯降风险
        cur = state.open_positions.get(t.product)
        if cur is None:
            return False
        return t.lots < cur.get("lots", 0)

    @staticmethod
    def _value_estimate(t: Target, prices: dict[str, float]) -> float | None:
        """持仓市值粗估 = 最近收盘价 × 100 股 × 手数。缺价返回 None（审批拒绝）。"""
        from .. import contracts

        px = prices.get(t.product)
        if px is None or px <= 0:
            return None
        return px * contracts.shares_of(t.lots)
