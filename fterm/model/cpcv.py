"""CPCV —— 组合净化交叉验证（2026-10-09，模型升级 Phase 3）。

文献：López de Prado, *Advances in Financial Machine Learning*, ch.7（Combinatorial
Purged Cross-Validation）。目的：把「单一 walk-forward 路径」升级为「多条回测路径」，
从而能度量 **IC 的时间稳定性**（主判据）与 **过拟合概率 PBO**。

与既有实现的区别：
  - `fterm/model/dataset.py::WalkForwardFolds` 是 **expanding** 单路径（每折只有一段测试）。
  - 本模块是 **组合** 多路径：N 组取 k 组测试，共 C(N,k) 种划分，可重建 φ=C(N,k)·k/N 条路径。

净化语义（与既有口径一致，不得改）：
  - 标签 `fwd_ret` 跨 `label_span = label_horizon + 1` 个交易日。
  - **purge**：测试块起点前 `label_span` 个交易日内的训练样本剔除（防标签窗口伸入测试段）。
  - **embargo**：测试块终点后 `embargo` 个交易日内的训练样本剔除（防序列相关泄漏）。
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


def n_paths(n_groups: int, n_test_groups: int) -> float:
    """CPCV 可重建的回测路径数 φ = C(N,k)·k/N。"""
    from math import comb
    return comb(n_groups, n_test_groups) * n_test_groups / n_groups


def cpcv_date_splits(dates: list, n_groups: int = 6, n_test_groups: int = 2,
                     label_span: int = 6, embargo: int = 0):
    """在**交易日位置**上枚举 CPCV 划分。

    dates：升序唯一交易日列表（长度 T）。
    yield (train_pos, test_pos) —— 两个 int 数组，值为 dates 的位置索引。

    要求 n_groups ≥ 2、1 ≤ n_test_groups < n_groups；否则抛错（大声失败）。
    """
    T = len(dates)
    if n_groups < 2 or not (1 <= n_test_groups < n_groups):
        raise ValueError(f"cpcv: 非法参数 n_groups={n_groups}, n_test_groups={n_test_groups}")
    if T < n_groups:
        raise ValueError(f"cpcv: 样本日数 {T} < 组数 {n_groups}")

    bounds = np.linspace(0, T, n_groups + 1, dtype=int)
    groups = [np.arange(bounds[i], bounds[i + 1]) for i in range(n_groups)]

    for combo in combinations(range(n_groups), n_test_groups):
        test_pos = np.concatenate([groups[i] for i in combo])
        test_pos.sort()
        train_mask = np.ones(T, dtype=bool)
        train_mask[test_pos] = False
        # 逐测试块做 purge（前 label_span）与 embargo（后 embargo）
        for gi in combo:
            s, e = bounds[gi], bounds[gi + 1]
            lo = max(0, s - label_span)
            hi = min(T, e + embargo)
            train_mask[lo:hi] = False
        train_pos = np.nonzero(train_mask)[0]
        if train_pos.size == 0:
            continue
        yield train_pos, test_pos


def cpcv_sample_folds(sample_dates, n_groups: int = 6, n_test_groups: int = 2,
                      label_span: int = 6, embargo: int = 0, min_train: int = 200):
    """把 CPCV 划分映射到**样本索引**（对齐 `WalkForwardFolds.fold` 的返回语义）。

    sample_dates：长度 N 的样本日期序列（**必须按日期升序**，与 stack/fold_frame 一致）。
    yield (train_idx, test_idx) —— list[int]，可直接用于重训。

    min_train：训练样本数不足的划分直接跳过（与既有 min_train 纪律一致）。
    """
    sd = np.asarray([np.datetime64(d) for d in sample_dates])
    uniq = np.unique(sd)
    pos_of = {d: i for i, d in enumerate(uniq)}
    sample_pos = np.array([pos_of[d] for d in sd], dtype=int)
    for tr_pos, te_pos in cpcv_date_splits(list(uniq), n_groups, n_test_groups,
                                           label_span, embargo):
        te_set = set(te_pos.tolist())
        tr_set = set(tr_pos.tolist())
        test_idx = np.nonzero(np.isin(sample_pos, list(te_set)))[0].tolist()
        train_idx = np.nonzero(np.isin(sample_pos, list(tr_set)))[0].tolist()
        if len(train_idx) < min_train or not test_idx:
            continue
        yield train_idx, test_idx


# ---------------------------------------------------------------- 稳定性门（§三）

def cpcv_groups(T: int, n_groups: int = 6) -> list:
    """把 T 个**时序位置**均分为 n_groups 个连续块（与 `cpcv_date_splits` 同边界）。

    边界口径逐字复用 `cpcv_date_splits` 的 `np.linspace(0, T, n_groups+1, dtype=int)`，
    保证分组与净化划分**同源**（否则两条路径的组界会静默分叉）。
    """
    if n_groups < 2:
        raise ValueError(f"cpcv_groups: 非法 n_groups={n_groups}")
    if T < n_groups:
        raise ValueError(f"cpcv_groups: 样本日数 {T} < 组数 {n_groups}")
    bounds = np.linspace(0, T, n_groups + 1, dtype=int)
    return [np.arange(bounds[i], bounds[i + 1]) for i in range(n_groups)]


def cpcv_path_assignments(n_groups: int = 6, n_test_groups: int = 2) -> list:
    """把 C(N,k) 个划分**铺砖**成 φ=C(N-1,k-1) 条路径（AFML ch.7 口径）。

    返回 [{组号: 划分号}] 列表，长度 φ。每个 (划分, 测试组) 对被恰好用一次，
    每条路径恰好覆盖全部 N 组各一次 —— 即每条路径是一段完整长度的 OOS 序列。
    """
    from math import comb
    if n_groups < 2 or not (1 <= n_test_groups < n_groups):
        raise ValueError(f"cpcv: 非法参数 n_groups={n_groups}, "
                         f"n_test_groups={n_test_groups}")
    combos = list(combinations(range(n_groups), n_test_groups))
    per_group = {g: [s for s, c in enumerate(combos) if g in c] for g in range(n_groups)}
    phi = comb(n_groups - 1, n_test_groups - 1)
    out = []
    for j in range(phi):
        out.append({g: per_group[g][j] for g in range(n_groups)})
    return out


def cpcv_ic_stability(daily_ic, n_groups: int = 6, n_test_groups: int = 2,
                      split_daily_ic: dict | None = None) -> dict:
    """预注册 §三「CPCV 稳定性」门的读数：`P(IC>0)` 与路径间 IC 标准差。

    输入 `daily_ic`：**按时间升序**的逐日 IC 序列（长度 T）。
    返回两层读数（务必都看，理由见下）：

      - `split_ic` / `p_ic_pos_splits` / `split_ic_std`：C(N,k) 个**划分级**读数。
        每个划分的测试集是 k 个连续组的并集，其 IC = 该并集内日频 IC 的均值。
        划分之间测试段**不重叠** ⇒ 这是真正独立变动的分布。
      - `path_ic` / `p_ic_pos_paths` / `path_ic_std`：φ 条**路径级**读数，
        按 AFML ch.7 铺砖口径**逐位置**重组：路径 p 中第 g 组的每个交易日的 IC
        取自「测试第 g 组的那个划分」的该位置值（见 `cpcv_path_assignments`）。

    ⚠️ **单流退化（2026-10-10 实测，必须登记）**：`split_daily_ic=None`（默认）时
    只有一条预测流，同一天的预测值与标签**不随划分变化** ⇒ 每条路径逐位置都等于
    原序列 ⇒ `path_ic` 各项**恒等于全样本 IC**，`path_ic_std` **恰为 0.0**，
    `p_ic_pos_paths` 退化为 `1.0`（全样本 IC>0 时）。
    **故路径级读数在单流下不构成独立信息**（与 §45 已知缺口 #3 同型陷阱）。
    真正有分辨力的是**划分级**读数；`path_std_degenerate` 为 True 即命中该退化。
    路径级只有在「每个划分各训一个模型」时才独立 —— 届时用 `split_daily_ic`
    传入 `{划分号: 该划分模型下的全长日频 IC 序列}`，路径级读数才有分辨力。
    """
    v = np.asarray(daily_ic, dtype="float64")
    v = v[np.isfinite(v)]
    T = v.size
    gs = cpcv_groups(T, n_groups)
    combos = list(combinations(range(n_groups), n_test_groups))
    split_ic = np.array([float(v[np.concatenate([gs[g] for g in c])].mean())
                         for c in combos], dtype="float64")
    # 每个划分的「全长日频 IC」：单流下所有划分共用同一条序列（⇒ 退化）。
    if split_daily_ic is None:
        per_split = {s: v for s in range(len(combos))}
    else:
        per_split = {int(s): np.asarray(a, dtype="float64") for s, a in split_daily_ic.items()}
        missing = [s for s in range(len(combos)) if s not in per_split]
        if missing:
            raise ValueError(f"cpcv_ic_stability: split_daily_ic 缺划分 {missing}")
        bad = [s for s, a in per_split.items() if a.size != T]
        if bad:
            raise ValueError(f"cpcv_ic_stability: split_daily_ic 长度 != T({T})：{bad}")
    # AFML 铺砖：路径 p 的第 g 组切片取自划分 a[g] 在该组上的**逐日值**。
    path_ic = []
    for a in cpcv_path_assignments(n_groups, n_test_groups):
        ser = np.empty(T, dtype="float64")
        for g in range(n_groups):
            ser[gs[g]] = per_split[a[g]][gs[g]]
        path_ic.append(float(np.nanmean(ser)))
    path_ic = np.asarray(path_ic, dtype="float64")
    sd = float(split_ic.std(ddof=1)) if split_ic.size > 1 else None
    pd_ = float(path_ic.std(ddof=1)) if path_ic.size > 1 else None
    return {
        "n_groups": int(n_groups), "n_test_groups": int(n_test_groups),
        "n_splits": int(split_ic.size), "n_paths": int(path_ic.size),
        "n_days": int(T),
        "split_ic": split_ic.tolist(),
        "split_ic_mean": float(split_ic.mean()) if split_ic.size else None,
        "split_ic_std": sd,
        "p_ic_pos_splits": float((split_ic > 0).mean()) if split_ic.size else None,
        "path_ic": path_ic.tolist(),
        "path_ic_mean": float(path_ic.mean()) if path_ic.size else None,
        "path_ic_std": pd_,
        "p_ic_pos_paths": float((path_ic > 0).mean()) if path_ic.size else None,
        "path_std_degenerate": bool(pd_ is not None and pd_ == 0.0),
    }


def cpcv_gate(reading: dict, f0_reading: dict, p_min: float = 0.9) -> dict:
    """把 §三 CPCV 门的两条判据落成布尔读数（**不设默认通过**）。

    判据（`plans/model-upgrade-prereg-20261009.md §三`）：
      ① `P(IC>0) ≥ p_min`；② 路径间 IC 标准差**低于 F0**。

    ⚠️ 因单流退化（见 `cpcv_ic_stability`），两条判据都取**划分级**读数
    （`p_ic_pos_splits` / `split_ic_std`）。若照抄**路径级**读数，C2 会变成
    `0.0 < 0.0` ⇒ **恒为假**，门对任何臂（含 F0 自比）**无条件失败**——
    一个永不开门的门同样不构成独立信息（与 §45 缺口 #3 同型：读数看似在场、
    实则与上层同源/无分辨力）。路径级读数一并回传留痕。
    """
    p = reading.get("p_ic_pos_splits")
    pf0 = f0_reading.get("p_ic_pos_splits")
    s = reading.get("split_ic_std")
    sf0 = f0_reading.get("split_ic_std")
    c1 = bool(p is not None and p >= p_min)
    c2 = bool(s is not None and sf0 is not None and s < sf0)
    return {
        "c1_p_ic_pos_ge": {"value": p, "threshold": p_min, "ok": c1},
        "c2_split_ic_std_below_f0": {"value": s, "f0": sf0, "ok": c2},
        "f0_p_ic_pos_splits": pf0,
        "pass": bool(c1 and c2),
        "note": "单流退化：path_ic_std 恒 0（路径级 C2 会恒假），判据取划分级读数",
    }
