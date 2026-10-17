"""事件驱动回测引擎（日频，A股现货）。

撮合规则（与实盘 paper 引擎同构，成本公式与实盘共用 costs 模块）：
  - t 日收盘产生目标 → t+1 日开盘执行（signal_exec_lag=1，天然满足 T+1）
  - 成交价 = t+1 开盘 ± slippage（不利方向，跳取整）
  - 一字涨跌停不可成交：买被拒 = low ≥ 真实涨停价；卖被拒 = high ≤ 真实跌停价
    （真实停板价来自老库 stk_limit；缺表回退昨收 ±10% 近似——docs/02 登记的兜底口径）
  - 现货全额现金：买入需 cash ≥ 成交金额 + 费用，不足 → 拒单（blocked_cash）
  - 手数离散：整数手（1手=100股），0 手 = 放弃信号（合法状态）
  - T+1 防御闸：当日买入的手数当日不可卖（结构上不触发，触发即记录 blocked_t1）

账务（现货全额交收，等式恒成立 equity = cash + Σ持仓市值）：
  买入：cash -= 成交金额 + 买入费
  卖出：cash += 成交金额 − 卖出费（含印花税）——卖出资金当日可用（先卖后买同日合法）
  盯市：每日收盘按 close（未复权）重估持仓市值
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import contracts
from ..config import Config
from .costs import trade_cost


@dataclass(frozen=True)
class Target:
    """strategy+risk 产出的目标仓位。direction 0 = 清仓离场；只做多（1）。"""
    product: str
    direction: int          # 0/+1
    lots: int
    reason: str = ""


@dataclass
class Fill:
    date: pd.Timestamp
    product: str
    direction: int
    lots: int
    price: float
    fee: float
    slippage: float
    action: str             # open / close / adjust / blocked / blocked_cash / blocked_t1


@dataclass
class Position:
    product: str
    direction: int
    lots: int
    entry_price: float      # 含滑点成交价（盯市基准）
    entry_ref: float        # 执行时点中间价（成本归因基准）
    entry_date: pd.Timestamp
    open_fee: float = 0.0
    open_slip: float = 0.0
    adj_entry: float = 1.0        # 入场日复权因子（含权入账的价值基准，docs/03 §38）


@dataclass
class BacktestResult:
    equity: pd.Series
    pos_value: pd.Series
    fills: list[Fill]
    trades: list[dict]      # 已平仓回合（净盈亏含全部成本）
    metrics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ExecContext:
    """引擎在决策日交给决策方的真实账本（忠实回测的唯一入口）。

    `prev_equity` = 上一交易日收盘权益（日停闸的日初参照，与生产 runner 同源）。
    """
    date: pd.Timestamp
    equity: float
    cash: float
    positions: dict[str, dict]
    prev_equity: float | None = None


class BacktestEngine:
    def __init__(self, cfg: Config, bars_map: dict[str, pd.DataFrame]):
        self.cfg = cfg
        self.bars = {p: df.sort_index() for p, df in bars_map.items()}
        idx = set()
        for df in self.bars.values():
            idx |= set(df.index)
        self.calendar = pd.DatetimeIndex(sorted(idx))
        # 2026-10-08 性能改造：列值按需缓存为 ndarray（**惰性 + 按 code**，
        # 只在实际交易的标的上分配；全市场一次性预取会是数百 MB 的纯浪费）。
        # 目的：把 _limit_prices / mark 里的 `df.index.get_indexer([date])`（每次分配
        # 一个列表并走一遍标签哈希）与 `df.at[date, col]`（两次标签查找）换成
        # C 级二分 + 整数索引。
        self._arr_cache: dict[str, tuple] = {}

    def _arrays(self, code: str) -> tuple:
        """(index, open, close, up_limit|None, down_limit|None)，惰性构建并缓存。"""
        a = self._arr_cache.get(code)
        if a is None:
            df = self.bars[code]
            a = (
                df.index,
                df["open"].to_numpy(dtype="float64"),
                df["close"].to_numpy(dtype="float64"),
                df["up_limit"].to_numpy(dtype="float64") if "up_limit" in df.columns else None,
                df["down_limit"].to_numpy(dtype="float64") if "down_limit" in df.columns else None,
            )
            self._arr_cache[code] = a
        return a

    def _row_pos(self, code: str, date: pd.Timestamp) -> int:
        """date 在 code 行情表中的行号；不在表中抛错（与原 get_indexer 的失败语义一致）。

        用 searchsorted（C 级二分，O(log n)、零分配）替代 get_indexer([date])
        （构造列表 + 标签哈希）。二者都是"找不到就报错"，只是报错点不同。
        """
        idx = self._arrays(code)[0]
        pos = int(idx.searchsorted(date))
        assert pos > 0, f"{code} {date}: 无昨收参照（数据起点不足）"
        assert pos < len(idx) and idx[pos] == date, f"{code} {date}: 该日不在行情表中"
        return pos

    def adj_at(self, code: str, date: pd.Timestamp) -> float:
        """该日复权因子（`adj` 列缺失或无值 ⇒ 1.0，即不含权、退回旧口径）。

        含权入账的全部数学都建立在 `ratio = adj[d]/adj[entry]` 上：送转股数与现金
        分红都折成"价值倍数"，成交价/现金交收/整手/tick 滑点/停板判定一律留在
        真实未复权价空间（docs/03 §38）。
        """
        df = self.bars.get(code)
        if df is None or "adj" not in df.columns:
            return 1.0
        try:
            v = df["adj"].at[date]
        except KeyError:
            return 1.0
        f = float(v)
        return f if (f == f and f > 0) else 1.0      # NaN/非正 ⇒ 保守退回 1.0

    def _value_ratio(self, pos: "Position", date: pd.Timestamp) -> float:
        r = self.adj_at(pos.product, date) / (pos.adj_entry or 1.0)
        return r if r > 0 else 1.0

    def _limit_prices(self, code: str, date: pd.Timestamp) -> tuple[float, float, float, float]:
        """(up_limit, down_limit, open, prev_close)。停板价优先真实表，缺失回退昨收 ±10%。"""
        _, open_arr, close_arr, up_arr, dn_arr = self._arrays(code)
        pos = self._row_pos(code, date)
        o = float(open_arr[pos])
        prev_close = float(close_arr[pos - 1])
        up = up_arr[pos] if up_arr is not None else np.nan
        dn = dn_arr[pos] if dn_arr is not None else np.nan
        up = float(up) if np.isfinite(up) else prev_close * 1.1     # 兜底近似，docs/02 登记
        dn = float(dn) if np.isfinite(dn) else prev_close * 0.9
        return up, dn, o, prev_close

    def run(self, targets_by_date: dict[pd.Timestamp, list[Target]],
            start: pd.Timestamp | None = None,
            provider=None, reb_days=None) -> BacktestResult:
        """执行回测。两种驱动方式：

        ① `targets_by_date`（旧）：预先把 `{日: [Target]}` 全列出——决策时**看不到
           引擎真实账本**，于是风控批过的单可能因现金不足被引擎拒（blocked_cash），
           而决策环已把它记成持仓 ⇒ 幻象持仓 ⇒ 长期欠投资（docs/03 §29）。
        ② `provider(ctx) -> list[Target]`（忠实模式，ctx=`ExecContext`）：在循环内、
           盯市之后用**引擎当日的真实权益/可用现金/真实持仓**问一次决策，
           拒单不会再变成幻象持仓。`reb_days` 限定问价日（其余日不动仓）。
        """
        cfg = self.cfg
        cash = cfg.account.initial_equity
        positions: dict[str, Position] = {}
        fills: list[Fill] = []
        closed_trades: list[dict] = []
        equity_hist, value_hist = {}, {}
        dates = [d for d in self.calendar if start is None or d >= start]
        assert len(dates) > 2, "回测区间过短"
        if provider is not None:
            reb_set = set(reb_days) if reb_days is not None else set(dates)
            targets_by_date = {}

        def mark(date: pd.Timestamp) -> tuple[float, float]:
            """盯市：持仓按当日收盘估值（该日无行情的标的跳过，不计市值）。"""
            value = 0.0
            for p, pos in positions.items():
                # 2026-10-08：原为 `date in df.index` + `df.at[date, "close"]`
                # （两次标签查找/次/持仓/日）→ C 级二分 + 整数索引。
                idx, _o, close_arr, _u, _d = self._arrays(p)
                i = int(idx.searchsorted(date))
                if i < len(idx) and idx[i] == date:
                    # 含权：市值按 adj[d]/adj[entry] 放大（除权除息不再凭空蒸发）
                    value += (float(close_arr[i]) * contracts.shares_of(pos.lots)
                              * self._value_ratio(pos, date))
            return cash + value, value

        def pos_value_at(date: pd.Timestamp, pos: "Position") -> float:
            """该日收盘市值（无行情则按入场价，与 mark 同语义）。

            2026-10-08 收口批修复：有行情时补上含权 ratio——原实现是全引擎唯一
            不带 `_value_ratio` 的估值点，除权事件后忠实快照（governor 的
            open_positions.value）与 mark() 的权益口径分叉，现金闸估计偏低。
            无行情日退回 entry_price×shares = 入场时点价值（ratio≡1），口径不变。
            """
            idx, _o, close_arr, _u, _d = self._arrays(pos.product)
            i = int(idx.searchsorted(date))
            if i < len(idx) and idx[i] == date:
                px = float(close_arr[i]) * self._value_ratio(pos, date)
            else:
                px = pos.entry_price
            return px * contracts.shares_of(pos.lots)

        for i, date in enumerate(dates[:-1]):
            nxt = dates[i + 1]
            # 1) 当日收盘盯市
            equity_hist[date], value_hist[date] = mark(date)

            # 2) 忠实模式：用引擎真实账本问决策（等价于生产 runner 的 snap/positions）
            if provider is not None and date in reb_set:
                snap_pos = {p: {"direction": s.direction, "lots": s.lots,
                                "value": pos_value_at(date, s)}
                            for p, s in positions.items()}
                targets_by_date[date] = list(provider(ExecContext(
                    date=date, equity=equity_hist[date], cash=cash,
                    positions=snap_pos,
                    prev_equity=equity_hist[dates[i - 1]] if i >= 1 else None)))

            # 3) t 日目标 → t+1 开盘执行
            for tgt in targets_by_date.get(date, []):
                if tgt.product not in self.bars or nxt not in self.bars[tgt.product].index:
                    continue
                cash = self._execute(tgt, nxt, positions, cash, fills, closed_trades)

        # 终值盯市
        last = dates[-1]
        equity_hist[last], value_hist[last] = mark(last)

        eq = pd.Series(equity_hist).sort_index()
        pv = pd.Series(value_hist).sort_index()
        from .metrics import compute_metrics

        return BacktestResult(eq, pv, fills, closed_trades,
                              compute_metrics(cfg, eq, closed_trades, fills, pv))

    # ---------- 执行细节 ----------

    def _execute(self, tgt: Target, exec_date: pd.Timestamp, positions: dict[str, Position],
                 cash: float, fills: list[Fill], closed_trades: list[dict]) -> float:
        """执行一个目标，返回更新后的现金。只做多：direction ∈ {0, +1}。"""
        assert tgt.direction in (0, 1), "A股现货只做多，direction 只能是 0/+1"
        cfg = self.cfg
        df = self.bars[tgt.product]
        h, l = float(df.at[exec_date, "high"]), float(df.at[exec_date, "low"])
        up, dn, o, _prev = self._limit_prices(tgt.product, exec_date)
        limit_up = l >= up - contracts.TICK / 2     # 一字/全天封板：买不进
        limit_dn = h <= dn + contracts.TICK / 2     # 一字/全天封板：卖不出
        cur = positions.get(tgt.product)
        cur_dir = cur.direction if cur else 0
        cur_lots = cur.lots if cur else 0

        if tgt.direction == cur_dir and tgt.lots == cur_lots:
            return cash    # 与现状一致，无操作

        # ---- 平仓段（清仓/减仓；换向 = 先卖后买，卖出资金当日可用）----
        if cur and (tgt.direction != cur_dir or tgt.lots < cur_lots):
            full_close = tgt.direction == 0 or tgt.lots == 0
            close_lots = cur_lots if full_close else cur_lots - tgt.lots
            if limit_dn:
                fills.append(Fill(exec_date, tgt.product, -1, close_lots, o, 0, 0, "blocked"))
                return cash    # 停板关不上：现状保持
            if exec_date == cur.entry_date:
                fills.append(Fill(exec_date, tgt.product, -1, close_lots, o, 0, 0, "blocked_t1"))
                return cash    # T+1 防御闸（结构上不触发，触发即留痕）
            cash += self._do_close(cur, close_lots, o, exec_date, cfg, fills, closed_trades)
            if full_close:
                positions.pop(tgt.product, None)
                cur = None
            else:
                cur.lots -= close_lots

        # ---- 开仓段（新开/加仓）----
        if tgt.direction != 0:
            add_lots = tgt.lots - (cur.lots if cur else 0)
            if add_lots <= 0:
                return cash
            if limit_up:
                fills.append(Fill(exec_date, tgt.product, 1, add_lots, o, 0, 0, "blocked"))
                return cash
            cost = trade_cost(cfg, o, 1, add_lots, exec_date)   # 建仓时 ratio≡1
            needed = cost.fill_price * contracts.shares_of(add_lots) + cost.fee
            if needed > cash + 1e-6:
                fills.append(Fill(exec_date, tgt.product, 1, add_lots, o, 0, 0, "blocked_cash"))
                return cash
            if cur is None:
                fills.append(Fill(exec_date, tgt.product, 1, add_lots,
                                  cost.fill_price, cost.fee, cost.slippage, "open"))
                positions[tgt.product] = Position(tgt.product, 1, add_lots,
                                                  cost.fill_price, o, exec_date,
                                                  cost.fee, cost.slippage,
                                                  adj_entry=self.adj_at(tgt.product,
                                                                        exec_date))
            else:    # 加仓：合并持仓（加权入场价与参考价；T+1 口径取最早入场日）
                tot = cur.lots + add_lots
                cur.entry_price = (cur.entry_price * cur.lots + cost.fill_price * add_lots) / tot
                cur.entry_ref = (cur.entry_ref * cur.lots + o * add_lots) / tot
                cur.lots = tot
                cur.open_fee += cost.fee
                cur.open_slip += cost.slippage
                # 加仓合并价值基准（按手数加权，与 entry_price 同规则）
                tot_adj = cur.adj_entry * (tot - add_lots) + self.adj_at(tgt.product,
                                                                        exec_date) * add_lots
                cur.adj_entry = tot_adj / tot
            cash -= needed    # 全额交收：股票市值进持仓、现金扣成交金额+费
        return cash

    def _do_close(self, cur: Position, close_lots: int, ref_price: float,
                  exec_date: pd.Timestamp, cfg: Config, fills: list[Fill],
                  closed_trades: list[dict]) -> float:
        """卖出 → 返回现金增量 = 成交金额 − 卖出全费用（佣金+印花税+过户费）。

        报告口径（closed_trades）以中间价核算：raw_pnl_mid − 全部费用（双边费+滑点），
        与权益变动严格一致。
        """
        shares_close = contracts.shares_of(close_lots)
        shares_all = contracts.shares_of(cur.lots)
        # 含权入账：ratio = adj[卖出日]/adj[入场日]，等效成交价 = 真实价 × ratio。
        # 现金到账按等效成交价计（送转的额外股数、除息的现金分红都折算进这里），
        # 费用也按真实成交额计；停板判定/整手/tick 滑点仍在真实价空间。
        ratio = self._value_ratio(cur, exec_date)
        cost = trade_cost(cfg, ref_price, -1, close_lots, exec_date, value_ratio=ratio)
        eff_fill = cost.fill_price * ratio
        frac = close_lots / cur.lots
        raw_mid = (ref_price * ratio - cur.entry_ref) * cur.direction * shares_close
        fee_share = cur.open_fee * frac
        slip_share = cur.open_slip * frac
        fills.append(Fill(exec_date, cur.product, -1, close_lots, eff_fill,
                          cost.fee, cost.slippage, "close" if close_lots == cur.lots else "adjust"))
        closed_trades.append({
            "product": cur.product, "direction": cur.direction, "lots": close_lots,
            "entry_date": cur.entry_date, "entry_price": cur.entry_price,
            "exit_date": exec_date, "exit_price": eff_fill,
            "value_ratio": round(ratio, 6),
            "raw_pnl": raw_mid,
            "costs": fee_share + cost.fee + slip_share + cost.slippage,
            "pnl": raw_mid - fee_share - cost.fee - slip_share - cost.slippage,
            "hold_days": int((exec_date - cur.entry_date).days),
        })
        assert shares_close <= shares_all
        # 含权：现金到账 = 等效成交价 × 股数 − 全费用（送转/分红的权利在此兑现）
        return eff_fill * shares_close - cost.fee
