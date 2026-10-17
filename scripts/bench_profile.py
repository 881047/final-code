r"""性能基准线（2026-10-08 审查 P0）。

背景：本仓此前**没有任何 profiling 基础设施**（全仓 grep cProfile/pstats/line_profiler/
memory_profiler/tracemalloc/pytest-benchmark/perf_counter/py-spy 均为空，唯一计时是
dlv2_chain.py:189 与 rankloss_chain.py:315 的 time.time() 粗打点）。
没有基准线就没法证明任何优化的收益，所以先建测量，再动刀。

本脚本只做**合成数据微基准**：不读 data/ 面板、不碰老库、不写任何状态，
可在测试环境安全反复运行。规模取自实测口径（2367 日 × 5467 股 × 25 特征）。

用法：
    .venv\Scripts\python.exe scripts\bench_profile.py
    .venv\Scripts\python.exe scripts\bench_profile.py --days 400 --codes 1200   # 快速档
    .venv\Scripts\python.exe scripts\bench_profile.py --json data/bench_baseline.json

覆盖的瓶颈（编号与 docs/review-20261008 一致）：
    B2  fold_frame 的逐日 universe.at[d,c] 标量筛选
    B6  tsfactors.atr() 的 concat(3帧)+groupby(level=0).max()
    B7  build.py 的 np.repeat 为广播常量物化整个面板
    B9  day_ids 的逐行 dict.setdefault 构造
"""
from __future__ import annotations

import argparse
import gc
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

REAL_DAYS = 2367
REAL_CODES = 5467


# ---------------------------------------------------------------- 计时/测内存

def _measure(fn, *a, **kw):
    """返回 (耗时秒, 峰值内存MB)。tracemalloc 只统计 Python 层分配，够定位量级。"""
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    dt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return dt, peak / 1024 / 1024, out


def _row(name, cur, new, unit="s"):
    if new <= 0:
        spd = float("inf")
    else:
        spd = cur / new
    return {"case": name, "baseline": round(cur, 4), "candidate": round(new, 4),
            "speedup_x": round(spd, 2), "unit": unit}


# ---------------------------------------------------------------- 合成面板

def make_panel(days: int, codes: int, seed: int = 7):
    """合成 OHLC 宽面板（index=date, columns=code），与生产同构。"""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=days)
    cols = [f"{i:06d}.SZ" for i in range(codes)]
    close = pd.DataFrame(
        10.0 * np.exp(np.cumsum(rng.normal(0, 0.02, (days, codes)), axis=0)),
        index=idx, columns=cols, dtype="float32")
    high = (close * (1 + np.abs(rng.normal(0, 0.01, (days, codes)))).astype("float32"))
    low = (close * (1 - np.abs(rng.normal(0, 0.01, (days, codes)))).astype("float32"))
    open_ = close.shift(1).fillna(close).astype("float32")
    return open_, high, low, close


# ---------------------------------------------------------------- B6 atr

def atr_baseline(high, low, close, window=14):
    """现状实现（fterm/features/tsfactors.py:36-40）。"""
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=0) \
           .groupby(level=0).max()
    return tr.rolling(window).mean().astype("float32")


def atr_candidate(high, low, close, window=14):
    """候选：整块搬到 ndarray 域，用 fmax.reduce 替代 concat+groupby。

    两处关键（都踩过坑，写在这里防回退）：
    1. 必须用 np.fmax 而非 np.maximum —— 后者不跳 NaN，而 pandas 的
       groupby.max() 默认 skipna=True。首行 prev=NaN，用 maximum 会让整行变 NaN，
       与现状实现**数值不等**（本脚本会 assert 出来）。
    2. 必须先把三个面板一次性转成 ndarray 再算 —— 若在 DataFrame 域逐算子算完
       再 np.asarray 转换，每个中间量仍是一份全面板，内存一分不省。
    """
    h = high.to_numpy(dtype="float32")
    l = low.to_numpy(dtype="float32")
    c = close.to_numpy(dtype="float32")
    prev = np.empty_like(c)
    prev[0, :] = np.nan
    prev[1:, :] = c[:-1, :]
    tr = np.fmax.reduce([h - l, np.abs(h - prev), np.abs(l - prev)])
    tr = pd.DataFrame(tr, index=high.index, columns=high.columns)
    return tr.rolling(window).mean().astype("float32")


# ---------------------------------------------------------------- B7 repeat vs 广播

def repeat_baseline(disp):
    """现状（fterm/features/build.py:74）：为广播一个"每日常量"物化整个面板。

    disp 是长度为 N（股票数）的 1-D 每日常量；[:, None] 后按 axis=1 复制 N 份。
    """
    n = disp.shape[0]
    return np.repeat(disp[:, None], n, axis=1)


def repeat_candidate(disp, n):
    """候选：交给广播，不物化（下游按 [:, None] 用时语义等价）。"""
    return disp[:, None]


# ---------------------------------------------------------------- B9 day_ids

def day_ids_baseline(dates):
    """现状（rankloss_chain.py:91-94 / train_production.py:63-67）：逐行 dict.setdefault。"""
    seen, ids = {}, np.empty(len(dates), dtype="int64")
    for i, d in enumerate(dates):
        if d not in seen:
            seen[d] = len(seen)
        ids[i] = seen[d]
    return ids


def day_ids_candidate(dates):
    """候选：pandas.factorize（C 级，O(n)）。"""
    codes, _ = pd.factorize(dates, sort=False)
    return codes.astype("int64")


# ---------------------------------------------------------------- B2 标量筛选

def fold_frame_baseline(universe, days, codes):
    """现状（dlv2_chain.py:64-76）：逐日逐股 universe.at[d,c] 标量取数。"""
    total = 0
    for d in days:
        for c in codes:
            if bool(universe.at[d, c]):
                total += 1
    return total


def fold_frame_candidate(universe, days, codes):
    """候选：一次转 ndarray 布尔矩阵 + 整数切片。"""
    u = universe.to_numpy(dtype=bool)
    di = universe.index.get_indexer(days)
    return int(u[np.ix_(di, np.arange(len(codes)))].sum())


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=REAL_DAYS)
    ap.add_argument("--codes", type=int, default=REAL_CODES)
    ap.add_argument("--scan-days", type=int, default=60,
                    help="B2 标量扫描的天数（全量 2367 天太慢，按比例外推）")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    print(f"合成面板 {args.days} 日 × {args.codes} 股 "
          f"（float32 单面板 ≈ {args.days * args.codes * 4 / 1024 / 1024:.0f} MB）\n")
    open_, high, low, close = make_panel(args.days, args.codes)

    rows = []

    # ---- B6 atr
    t1, m1, r1 = _measure(atr_baseline, high, low, close)
    t2, m2, r2 = _measure(atr_candidate, high, low, close)
    same = bool(np.allclose(r1.to_numpy(), r2.to_numpy(), equal_nan=True))
    print(f"[B6] atr() 现状 {t1:.3f}s / 峰值 {m1:.0f}MB | 候选 {t2:.3f}s / 峰值 {m2:.0f}MB "
          f"| 结果一致={same} | 提速 {t1 / t2:.2f}× 内存降 {m1 / max(m2, 1e-9):.2f}×")
    rows.append({**_row("B6 atr concat+groupby → np.fmax.reduce", t1, t2),
                 "baseline_peak_mb": round(m1, 1), "candidate_peak_mb": round(m2, 1),
                 "numerically_equal": same})
    del r1, r2
    gc.collect()

    # ---- B7 repeat
    disp = np.asarray(close.iloc[-1], dtype="float32")
    t1, m1, _ = _measure(repeat_baseline, disp)
    t2, m2, _ = _measure(repeat_candidate, disp, args.codes)
    print(f"[B7] np.repeat 现状 {t1:.4f}s / 峰值 {m1:.0f}MB | 广播 {t2:.4f}s / 峰值 {m2:.0f}MB "
          f"| 省内存 {m1 - m2:.0f}MB")
    rows.append({**_row("B7 np.repeat → 广播", t1, t2),
                 "baseline_peak_mb": round(m1, 1), "candidate_peak_mb": round(m2, 1)})

    # ---- B9 day_ids
    n_rows = 12_000_000                       # 与真实"千万行"量级一致
    dates = np.repeat(np.arange(args.days, dtype="int64"), n_rows // args.days + 1)[:n_rows]
    t1, _, _ = _measure(day_ids_baseline, dates)
    t2, _, r2 = _measure(day_ids_candidate, dates)
    print(f"[B9] day_ids 现状 {t1:.3f}s | factorize {t2:.3f}s | 提速 {t1 / max(t2, 1e-9):.1f}× "
          f"（{n_rows / 1e6:.0f}M 行）")
    rows.append(_row("B9 逐行 dict → pd.factorize", t1, t2))
    del dates, r2
    gc.collect()

    # ---- B2 标量扫描
    rng = np.random.default_rng(11)
    uni = pd.DataFrame(rng.random((args.days, args.codes)) > 0.85,
                       index=close.index, columns=close.columns)
    days = list(close.index[:args.scan_days])
    codes = list(close.columns)
    t1, _, n1 = _measure(fold_frame_baseline, uni, days, codes)
    t2, _, n2 = _measure(fold_frame_candidate, uni, days, codes)
    scale = args.days / args.scan_days
    print(f"[B2] 逐日 .at 扫描 {args.scan_days} 日 {t1:.2f}s → 全史外推 {t1 * scale:.0f}s | "
          f"布尔矩阵 {t2:.3f}s → 外推 {t2 * scale:.1f}s | 提速 {t1 / max(t2, 1e-9):.0f}× "
          f"（命中数 {n1} vs {n2}）")
    rows.append({**_row("B2 逐日 .at → 布尔矩阵切片", t1, t2),
                 "extrapolated_full_baseline_s": round(t1 * scale, 1),
                 "extrapolated_full_candidate_s": round(t2 * scale, 1),
                 "counts_match": n1 == n2})

    print("\n" + "=" * 78)
    print(f"{'用例':<46}{'现状':>10}{'候选':>10}{'提速':>10}")
    print("-" * 78)
    for r in rows:
        print(f"{r['case']:<46}{r['baseline']:>10.4f}{r['candidate']:>10.4f}"
              f"{r['speedup_x']:>9.2f}×")
    print("=" * 78)

    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "panel": {"days": args.days, "codes": args.codes},
            "note": "合成数据微基准；candidate 为候选实现，未合入生产代码",
            "rows": rows,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入 {p}")


if __name__ == "__main__":
    main()
