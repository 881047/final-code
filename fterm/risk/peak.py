"""回撤峰值口径 —— 熔断自锁缺陷（docs/03 §36）的**单一实现来源**。

## 缺陷本体

`RiskGovernor.approve` 的熔断判据是 `equity < peak_equity × (1 − total_dd_halt)`。
当 `peak_equity` 取**全历史累积最大值**时（现行默认），下列链条不可逆：

    halted ⇒ 只拒新开、不强平（circuit_force_liquidate=false）
           ⇒ 存量持仓随 keep 带自然清空 ⇒ 账户变纯现金
           ⇒ 权益**冻结**（无持仓则无市值变动）⇒ peak 不变
           ⇒ 判据**永远为真** ⇒ **halted 成为吸收态**

实测（docs/03 §36 / §46）：解除所需权益涨幅**中位 25.3%**（阈值本身只要求回到 −12%）、
单次 halted 持续**中位 946 天、最长 1892 天**（≈全样本）；
13 臂的 halted 日占比 **58–84%**，其中 3 条臂**从未解除过**（`n_release_events = 0`）。

## 两种口径

| mode | 峰值定义 | 性质 |
|---|---|---|
| **`none`**（默认） | 全历史累积最大权益 | **与修复前逐位一致**；存在吸收态 |
| `rolling` | 权益曲线**滚动窗口**最大值（`dd_peak_window` 交易日） | **无吸收态** |

**`rolling` 为什么无吸收态**：窗口有限 ⇒ 账户空转满 `window` 日后，
窗口内全部权益相等 ⇒ `peak → equity` ⇒ 回撤 → 0 ⇒ 自动解除。
故 `max_halted_run ≤ window + 清仓耗时`（清仓由调仓日退出带完成，≤1 个调仓周期）。

> ⚠ **`rolling` 不是 docs/03 §40 判负的那类"前瞻型敞口规则"**：
> 那些按指数趋势/波动**预测**敞口；本口径**只由已实现权益**决定，
> **不含任何未来信息**（`test_halt_selflock_fix.py::TestNoLookahead` 钉住）。

## 为什么只此一处实现

回测（`fterm/backtest/two_pass.run_faithful` 的 `peak_fn`）与生产
（`fterm/runtime/runner._peak_equity`）**必须同口径**，否则会重演
docs/03 §41/§42 登记的"两处语义静默分叉"（改一处忘另一处，读数仍"看起来正常"）。
"""
from __future__ import annotations

PEAK_MODES = ("none", "rolling")
DEFAULT_WINDOW = 252          # 约 1 年交易日（与 #19/#20 的 ROLL_WIN 同口径）


def rolling_peak_fn(window: int = DEFAULT_WINDOW):
    """回测侧 `peak_fn(dates, equities, cur_equity) -> peak`：滚动窗口最大值。

    契约与 `run_faithful` 的 `peak_fn` 一致（`state_box["hist"]` 已先 append 当日权益，
    故 `equities[-1] == cur`；此处仍显式并入 `cur` 以对两种调用顺序都稳健）。
    """
    if window < 1:
        raise ValueError(f"rolling_peak_fn: 窗口必须 ≥1，收到 {window}")

    def f(dates, equities, cur):
        hist = list(equities)[-window:]
        return float(max(hist + [cur])) if hist else float(cur)

    return f


def peak_fn_from_cfg(cfg):
    """按 `account.dd_peak_mode` 解析回测侧 peak_fn。

    `none` 返回 **None** —— 这是刻意的：`run_faithful` 在 `peak_fn is None` 时
    走原有的"running max"分支（`state_box["peak"] = max(peak, ctx.equity)`），
    从而**保证默认口径与修复前逐位一致**（回归钉
    `test_halt_selflock_fix.py::TestDefaultBitIdentical` 钉住）。
    """
    mode = str(getattr(cfg.account, "dd_peak_mode", "none"))
    if mode == "none":
        return None
    if mode == "rolling":
        return rolling_peak_fn(int(getattr(cfg.account, "dd_peak_window",
                                            DEFAULT_WINDOW)))
    raise ValueError(f"未知 dd_peak_mode={mode!r}，只支持 {PEAK_MODES}")


def peak_equity(equities, current: float, mode: str = "none",
                window: int = DEFAULT_WINDOW) -> float:
    """生产侧口径：由**已实现**权益历史 + 当前权益算峰值。

    与 `runner._peak_equity` 原实现（`max([r[1] for r in curve] + [current])`）
    在 `mode="none"` 下**逐位等价**；`rolling` 时只取窗口内。
    """
    hist = list(equities)
    if mode == "none":
        return float(max(hist + [current])) if hist else float(current)
    if mode == "rolling":
        tail = hist[-window:] if window >= 1 else []
        return float(max(tail + [current])) if tail else float(current)
    raise ValueError(f"未知 dd_peak_mode={mode!r}，只支持 {PEAK_MODES}")


def halt_state(eq, halt: float, mode: str = "none",
               window: int = DEFAULT_WINDOW):
    """由权益曲线算熔断状态序列 + 连续段统计（诊断/标定共用，避免口径分叉）。

    返回 dict：`halted_frac` / `max_halted_run` / `n_halt_events` /
    `n_release_events` / `first_halt` / `release_lag_max`。
    `halted` 由 `equity < peak×(1−halt)` **直接重算**，不依赖任何告警表或日志
    （预注册 §五-4 的测量管线要求）。

    ## ⚠ 不变式的严格形式（预注册 erratum §三-1，由回归钉抓出的自身错误）

    **`max_halted_run ≤ window` 是错的**：连续 halted 段**包含权益冻结之前**
    就已经 halted 的下跌期，冻结只是该段尾部。故

        max_halted_run ≤ (冻结前已 halted 天数) + (解除滞后)

    严格成立的不变式是（**形式见下，已由 `_verify_invariant.py` 实测验证**）：

        **若权益连续 `window` 日恒定，则该恒定段内第 `window` 日起 halted 必为 False。**

    理由：此时滚动窗口内只剩该恒定值 ⇒ `peak = equity` ⇒ 回撤 = 0 ⇒ 判据为假。

    ⇒ 本函数回报两个**正确**的可检验量：
      - `constant_run_max_while_halted`：**halted 段内**最长恒定子段
        （P1 实测 168–248 日，均 < W=252 ⇒ 恒定段还没满窗口就因再入场结束）
      - `invariant_violations`：长度 ≥ `window` 的恒定段中，**段内第 window 日起
        仍 halted** 的个数。**必须为 0**，否则实现有 bug。

    ⚠ **不要用 `max_halted_run ≤ window` 当检验**（erratum §三-1）：
    连续 halted 段包含冻结**之前**的下跌期，故 `max_halted_run` 可略超 `window`
    （实测 249–253）。它只能与 P0 比较，**无固定上界**。
    """
    import numpy as np
    import pandas as pd

    eq = pd.Series(eq).astype("float64")
    if mode == "none":
        peak = eq.cummax()
    elif mode == "rolling":
        peak = eq.rolling(int(window), min_periods=1).max()
    else:
        raise ValueError(f"未知 dd_peak_mode={mode!r}，只支持 {PEAK_MODES}")
    h = (eq < peak * (1.0 - halt)).to_numpy()
    runs, cur = [], 0
    for v in h:
        if v:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    n_rel = int(sum(1 for i in range(1, len(h)) if h[i - 1] and not h[i]))
    n_ent = int(sum(1 for i in range(1, len(h)) if not h[i - 1] and h[i]))

    # ---- 严格不变式的检验量（见 docstring）----
    v = eq.to_numpy(dtype="float64")
    longest_const_in_halt = 0
    k = 0
    while k < len(h):
        if h[k]:
            s = k
            while k < len(h) and h[k]:
                k += 1
            seg = v[s:k]
            best = cur_c = 1 if len(seg) else 0
            for t in range(1, len(seg)):
                if abs(seg[t] - seg[t - 1]) < 1e-9:
                    cur_c += 1
                    best = max(best, cur_c)
                else:
                    cur_c = 1
            longest_const_in_halt = max(longest_const_in_halt, best)
        else:
            k += 1

    violations = 0
    n_const_ge_w = 0
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and abs(v[j + 1] - v[i]) < 1e-9:
            j += 1
        if j - i + 1 >= int(window):
            n_const_ge_w += 1
            for t in range(i + int(window) - 1, j + 1):
                if h[t]:
                    violations += 1
                    break
        i = j + 1

    return {
        "n_days": int(len(h)),
        "halted_days": int(h.sum()),
        "halted_frac": round(float(h.mean()), 4) if len(h) else 0.0,
        "max_halted_run": int(max(runs)) if runs else 0,
        "n_halt_events": n_ent,
        "n_release_events": n_rel,
        "first_halt": (str(eq.index[h.argmax()].date()) if h.any() else None),
        "constant_run_max_while_halted": int(longest_const_in_halt),
        "n_constant_runs_ge_window": int(n_const_ge_w),
        "invariant_violations": int(violations),
    }
