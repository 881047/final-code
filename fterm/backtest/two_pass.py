r"""两遍法决策骨架（2026-10-08 去重 + B5 复杂度修复）。

## 为什么有这个模块

`scripts/gate_chain.py`（`decision_pass` @74-105）与 `scripts/capital_chain.py`
（@81-112）各自内联了一份**近乎逐行相同**的两遍法骨架：

    第一遍：decision_pass(None)          → eng.run → r1
    第二遍：decision_pass(r1.equity)     → eng.run → r

差异只有两处：① `reb_all` 的类型（gate 是 list、capital 是 set）；
② 无曲线时的兜底权益来源（capital 的 `tier["equity"]` 与 `cfg_.account.initial_equity`
等价——capital 在 @50-54 已把 cfg_ 的 initial_equity 设成 tier 值）。

复制多份的风险是**风控口径静默分叉**：改一处忘另一处，两条实验链的熔断/日停联动
就会不一致，而结果仍然"看起来正常"。故收敛到这里。

## B5：peak 的复杂度

原实现每个调仓日做 `hist = eq_curve.loc[:d]; peak = float(hist.max())` ——
重算整个前缀最大值，单次 O(D)、总量 O(D×R)。但**峰值就是累积最大值**，
用 `cummax()` 一次算好即可 O(1) 查表（`EquityCurveView`）。

## 语义复刻（改造前后必须逐位一致）

`eq_curve.loc[:d]` 在 **d 不在索引**时会退到「最后一个 ≤ d 的日期」，
而不是报错或返回空 —— 本模块用 `searchsorted(d, side="right") - 1` 精确复刻。
原式 `equity = hist.iloc[-2] if len(hist) >= 2 else hist.iloc[-1]`：
即「前一日权益」，前缀只有 1 个点时退化为当日权益、`day_pnl = 0`。
"""
from __future__ import annotations

import pandas as pd


class EquityCurveView:
    """权益曲线的 O(1) 查询视图（替代逐调仓日的 `eq_curve.loc[:d]` 切片 + `.max()`）。"""

    __slots__ = ("_eq", "_idx", "_run_max")

    def __init__(self, eq_curve: pd.Series):
        self._eq = eq_curve
        self._idx = eq_curve.index
        self._run_max = eq_curve.cummax()

    def context(self, d) -> tuple[float, float, float] | None:
        """→ (equity, peak, day_pnl)，与 `hist = eq_curve.loc[:d]` 的原式完全一致。

        equity  = 前一日权益（前缀仅 1 点时退化为当日权益）
        peak    = 截至该日的累积最大值
        day_pnl = 当日损益（前缀仅 1 点时 0.0）

        返回 None 表示 d 之前没有任何数据（原实现此时会 IndexError）。
        """
        pos = int(self._idx.searchsorted(d, side="right")) - 1
        if pos < 0:
            return None
        eq = self._eq
        last = float(eq.iloc[pos])
        peak = float(self._run_max.iloc[pos])
        if pos >= 1:
            prev = float(eq.iloc[pos - 1])
            return prev, peak, last - prev
        return last, peak, 0.0


def run_decision_pass(*, reb_days, cfg, governor, raw_close,
                      initial_equity: float, equity_view: EquityCurveView | None = None,
                      targets_for=None, targets_for_factory=None):
    """单遍决策：逐调仓日生成 targets → 过风控审批 → 累积获批 targets 与持仓。

    参数
    ----
    reb_days      : 调仓日集合（**list 或 set 均可** —— 内部强制升序，
                    set 输入无序会让前缀语义错乱，这是原实现的一个隐患）
    targets_for   : `(d, equity, positions) -> list[Target]`
    targets_for_factory : 可选；`() -> targets_for`，**每遍调用一次**。
                    用于需要"每遍独立随机状态"的场景（ablation_chain 的 E/D 臂
                    每遍新建 `np.random.RandomState(rng_seed)`）。
                    提供时优先于 targets_for。
    cfg           : 配置（传给 RiskGovernor）
    governor      : RiskGovernor 实例（每遍新建，保持原实现的"每遍独立审批状态"）
    raw_close     : 收盘价宽面板（取成交价用）
    initial_equity: `equity_view is None`（第一遍）时的权益与峰值
    equity_view   : 第二遍传入；None 表示第一遍

    返回 `(targets_by_date, n_rej)`。
    """
    # 延迟导入：避免 fterm.backtest ←→ fterm.risk 的潜在循环
    from fterm import contracts
    from fterm.risk.governor import AccountState, RiskGovernor

    if targets_for_factory is not None:
        targets_for = targets_for_factory()
    if targets_for is None:
        raise TypeError("run_decision_pass: 必须提供 targets_for 或 targets_for_factory")

    days = sorted(reb_days)          # ⚠️ set 输入必须排序
    targets, positions, n_rej = {}, {}, 0
    for d in days:
        if equity_view is not None:
            ctx = equity_view.context(d)
            if ctx is None:
                continue             # d 之前无数据（原实现会 IndexError）
            equity, peak, day_pnl = ctx
        else:
            equity = peak = initial_equity
            day_pnl = 0.0
        tgts = targets_for(d, equity, positions)
        if not tgts:
            continue
        prices = {t.product: (float(raw_close.at[d, t.product])
                              if pd.notna(raw_close.at[d, t.product]) else 0.0)
                  for t in tgts}
        rep = governor.approve(tgts, AccountState(equity=equity, peak_equity=peak,
                                                  day_pnl=day_pnl,
                                                  open_positions=dict(positions)),
                               prices=prices)
        for dec in rep.decisions:
            if not dec.approved:
                n_rej += 1
                continue
            t = dec.target
            targets.setdefault(d, []).append(t)
            if t.direction == 0:
                positions.pop(t.product, None)
            else:
                px = prices.get(t.product) or 0.0
                positions[t.product] = {"direction": 1, "lots": t.lots,
                                        "value": px * contracts.shares_of(t.lots)}
    return targets, n_rej


def run_faithful(*, engine, reb_days, propose, cfg, raw_close, start_ts=None,
                 ask_days=None, stats=None, peak_fn=None,
                 exempt_batch_exits=False):
    """忠实单遍回测：决策在**引擎循环内**问，账本取引擎当日真实权益/可用现金/真实持仓。

    与 `run_two_pass` 的区别就是 docs/03 §29 的缺陷本体：旧法先把 `{日: [Target]}`
    全部列出再过引擎，风控按"估计市值"批单、引擎按"真实现金"拒单，被拒的单在决策环
    的 `positions` 视图里却已算持仓 ⇒ 幻象持仓 ⇒ 长期欠投资（实测同一条臂
    `n_blocked_cash` 192 笔、平均仓位利用率只有 0.39）。

    `propose(ctx) -> list[Target]`：ctx = `ExecContext(date, equity, cash, positions,
    prev_equity)`，与生产 runner 拿到的 `snap.equity / snap.available / broker.positions`
    一一同构 ⇒ 回测与纸面第一次共用同一套账本语义。

    `ask_days`：向决策方问价的日子。缺省 = `reb_days`（只在调仓日问，即判据链
    #11/#12/#13 的 C_ema 周步进语义）。生产位 `signal.ema_step=daily` 是**每日步进
    秩平滑、只在周五出单** ⇒ 必须 `ask_days=全部交易日` 并由 `propose` 自己在非
    调仓日返回空目标；否则"每日步进"这一构型在回测里根本无法出现
    （docs/03 §31 登记的未闭合项之一，2026-10-08 闭合）。

    返回 `(result, n_rej, n_targets)`。
    """
    from fterm.risk.governor import AccountState, RiskGovernor

    governor = RiskGovernor(cfg)
    state_box = {"n_rej": 0, "n_targets": 0, "peak": 0.0, "hist": []}

    def provider(ctx):
        state_box["hist"].append((ctx.date, ctx.equity))
        if peak_fn is None:
            state_box["peak"] = max(state_box["peak"], ctx.equity)
        else:
            # peak_fn(dates, equities, cur_equity) -> peak：#19 用它承载非自锁的
            # 再入场形态（滚动峰值 / 冷却重锚）。缺省 = 全历史累积峰值（现行语义）。
            ds = [d for d, _e in state_box["hist"]]
            es = [e for _d, e in state_box["hist"]]
            state_box["peak"] = float(peak_fn(ds, es, ctx.equity))
        tgts = list(propose(ctx))
        if not tgts:
            return []
        prices = {t.product: (float(raw_close.at[ctx.date, t.product])
                              if (t.product in raw_close.columns
                                  and pd.notna(raw_close.at[ctx.date, t.product])) else 0.0)
                  for t in tgts}
        day_start = ctx.prev_equity if ctx.prev_equity is not None else ctx.equity
        rep = governor.approve(tgts, AccountState(
            equity=ctx.equity, peak_equity=state_box["peak"],
            day_pnl=ctx.equity - day_start, open_positions=dict(ctx.positions),
            day_start_equity=ctx.prev_equity, available_cash=ctx.cash), prices=prices,
            exempt_batch_exits=exempt_batch_exits)
        approved = []
        for dec in rep.decisions:
            if not dec.approved:
                state_box["n_rej"] += 1
                if stats is not None:
                    r = stats.setdefault("reject_by_reason", {})
                    key = dec.reason.split("(")[0]
                    r[key] = r.get(key, 0) + 1
                continue
            approved.append(dec.target)
        state_box["n_targets"] += len(approved)
        return approved

    res = engine.run({}, start=start_ts, provider=provider,
                     reb_days=(ask_days if ask_days is not None else reb_days))
    return res, state_box["n_rej"], state_box["n_targets"]


def run_two_pass(*, engine, reb_days, targets_for, cfg, raw_close,
                 initial_equity: float, start_ts=None):
    """完整两遍法：第一遍无曲线 → 跑引擎 → 第二遍带曲线 → 再跑引擎。

    返回 `(result, n_rej, targets)`：
    result 为第二遍的 BacktestResult，targets 为第二遍获批的 `{date: [Target]}`，
    供调用方统计 `n_targets` 等指标。
    """
    gov1 = _new_governor(cfg)
    t1, _ = run_decision_pass(reb_days=reb_days, targets_for=targets_for, cfg=cfg,
                              governor=gov1, raw_close=raw_close,
                              initial_equity=initial_equity, equity_view=None)
    r1 = engine.run(t1, start=start_ts)

    gov2 = _new_governor(cfg)
    view = EquityCurveView(r1.equity)
    t2, n_rej = run_decision_pass(reb_days=reb_days, targets_for=targets_for, cfg=cfg,
                                  governor=gov2, raw_close=raw_close,
                                  initial_equity=initial_equity, equity_view=view)
    r = engine.run(t2, start=start_ts)
    return r, n_rej, t2


def _new_governor(cfg):
    from fterm.risk.governor import RiskGovernor
    return RiskGovernor(cfg)
