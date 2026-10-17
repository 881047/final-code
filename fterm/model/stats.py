"""信息层统计工具（2026-10-09，模型升级 Phase 3）。

审计 P0-1 的直接修复：本项目此前只有「平均 RankIC」，无法度量 ICIR / 滚动 IC 稳定性 /
制度分层 IC / 过拟合概率——而这正是本次升级的主判据。

全部函数为纯 numpy（scipy 仅作可选加速，缺省走 math.erf 兜底），无状态、可单测。

口径说明（务必遵守）：
  - IC 一律指**逐日截面** Spearman RankIC（每日一个值），IC 序列 = 逐日 IC。
  - ICIR = mean(IC) / std(IC)（不年化；与项目既有读数口径一致，仅作同尺度比较）。
  - Newey-West 用 Bartlett 权重，滞后阶数默认 L = floor(4*(T/100)^(2/9))（Newey-West 1994 经验式）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------- 基础

def _norm_ppf(p: float) -> float:
    """标准正态分位。优先 scipy，缺省用 erfinv 恒等式。"""
    p = min(max(float(p), 1e-12), 1 - 1e-12)
    try:
        from scipy.stats import norm
        return float(norm.ppf(p))
    except Exception:
        return float(math.sqrt(2.0) * _erfinv(2.0 * p - 1.0))


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def _erfinv(v: float) -> float:
    a = float(v)
    x = a
    for _ in range(8):
        x = x - (math.erf(x) - a) / (2.0 / math.sqrt(math.pi) * math.exp(-x * x) + 1e-12)
    return x


def icir(daily_ic: np.ndarray) -> float | None:
    """IC 信息比 = mean/std（样本 <2 或 std=0 → None）。"""
    a = np.asarray(daily_ic, dtype="float64")
    a = a[np.isfinite(a)]
    if a.size < 2:
        return None
    sd = float(a.std(ddof=1))
    if sd < 1e-12:
        return None
    return float(a.mean() / sd)


def newey_west_t(x: np.ndarray, lags: int | None = None) -> float | None:
    """序列均值的 Newey-West（Bartlett 核）t 统计量。

    用于「滚动 IC 是否显著非零」——IC 序列存在自相关，朴素 t 会高估显著性。
    """
    a = np.asarray(x, dtype="float64")
    a = a[np.isfinite(a)]
    T = a.size
    if T < 5:
        return None
    if lags is None:
        lags = int(math.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
    lags = max(0, min(lags, T - 1))
    mu = a.mean()
    dev = a - mu
    gamma0 = float((dev * dev).mean())
    s = gamma0
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        gk = float((dev[k:] * dev[:-k]).mean())
        s += 2.0 * w * gk
    var_mean = s / T
    if var_mean <= 1e-18:
        return None
    return float(mu / math.sqrt(var_mean))


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """滚动均值（尾部不足窗口则 min_periods=window，前缀为 NaN）。"""
    a = np.asarray(x, dtype="float64")
    out = np.full(a.shape, np.nan)
    if window <= 0 or a.size < window:
        return out
    c = np.cumsum(np.insert(np.nan_to_num(a, nan=0.0), 0, 0.0))
    cnt = np.cumsum(np.insert(np.isfinite(a).astype("float64"), 0, 0.0))
    for i in range(window - 1, a.size):
        n = cnt[i + 1] - cnt[i + 1 - window]
        if n == window:
            out[i] = (c[i + 1] - c[i + 1 - window]) / window
    return out


def longest_negative_run(x: np.ndarray) -> int:
    """最长连续负值段长度（预注册 §三「无长段负 IC」的直读量）。

    只统计**有限值**；NaN 视为中断（不计入负段，也不延续）。
    """
    a = np.asarray(x, dtype="float64")
    best = cur = 0
    for v in a:
        if np.isfinite(v) and v < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def rolling_ic_nw_t(daily_ic: np.ndarray, window: int = 60,
                    lags: int | None = None) -> float | None:
    """**滚动 IC 序列**的 Newey-West t —— 预注册 §三「滚动 IC」门的正确实现。

    口径：先对日频 IC 序列做 `window` 日滚动均值（不足窗口的前缀为 NaN，与
    `rolling_mean` 同口径），再对该**滚动均值序列**取 `newey_west_t`。

    ⚠️ 与 `rolling_ic_stability()["nw_t_of_daily_ic"]` **不是**同一个量：后者是
    日频 IC 的 NW-t（`newey_west_t(a)`，与 `summarize_ic()["t"]` 同式）——
    实测二者逐位相等（F0 两处均 +14.110418023514088），故它**不构成**预注册 §三
    「滚动 IC」门相对上层摘要 t 的独立信息（docs/03 §45 已知缺口 #3，2026-10-10 补）。

    ⚠️ 滞后阶数：滚动序列由**重叠窗口**构成，自相关可延伸至约 `window-1` 阶；
    `lags=None` 走模块经验式（`newey_west_t` 缺省，T=1815 时 L=7），对重叠窗口
    **偏乐观**。要稳健读数须显式传 `lags=window-1`（执行器两者都报，不改门槛）。
    """
    a = np.asarray(daily_ic, dtype="float64")
    roll = rolling_mean(a, window)
    valid = roll[np.isfinite(roll)]
    return newey_west_t(valid, lags=lags)


def rolling_ic_stability(daily_ic: np.ndarray, window: int = 60) -> dict:
    """滚动 IC 的稳定性读数：滚动均值序列 + 其 NW-t + 负段占比。

    ⚠️ 命名澄清（2026-10-10 补，docs/03 §45 已知缺口 #3）：
      - `nw_t_of_daily_ic`（**名即字面，保留不改**）：对**日频 IC 序列本身**取 NW-t，
        与 `summarize_ic()["t"]` 同式 ⇒ 实测逐位相等；它**不是**滚动序列的 t。
        保留原名与数值是为了不改任何既有读数（三个执行器按键名取值）。
      - `nw_t_of_rolling_mean`（**新**）：对 **window 日滚动均值序列**取 NW-t，
        即预注册 §三「滚动 IC」门真正要的读数（同 `rolling_ic_nw_t()`）。
    """
    a = np.asarray(daily_ic, dtype="float64")
    roll = rolling_mean(a, window)
    valid = roll[np.isfinite(roll)]
    nw = newey_west_t(a)
    neg_frac = float((valid < 0).mean()) if valid.size else None
    return {
        "rolling": roll,
        "rolling_mean_of_roll": float(valid.mean()) if valid.size else None,
        "nw_t_of_daily_ic": nw,
        "nw_t_of_rolling_mean": newey_west_t(valid),
        "n_rolling": int(valid.size),
        "frac_negative_windows": neg_frac,
    }


# ---------------------------------------------------------------- 制度分层

def regime_stratified_ic(daily_ic: np.ndarray, regimes: np.ndarray) -> dict[str, dict]:
    """按制度标签分层统计 IC（每层：n / mean / std / icir / t）。

    regimes 与 daily_ic 等长；NaN/None 层被跳过。返回 {层名: 读数}。
    """
    a = np.asarray(daily_ic, dtype="float64")
    r = np.asarray(regimes, dtype=object)
    if a.shape != r.shape:
        raise ValueError(f"regime_stratified_ic: 长度不一致 {a.shape} vs {r.shape}")
    out: dict[str, dict] = {}
    for lvl in sorted({x for x in r if x is not None and (not isinstance(x, float) or np.isfinite(x))}, key=str):
        m = (r == lvl) & np.isfinite(a)
        v = a[m]
        out[str(lvl)] = {
            "n": int(v.size),
            "mean": float(v.mean()) if v.size else None,
            "std": float(v.std(ddof=1)) if v.size > 1 else None,
            "icir": icir(v),
            "t": newey_west_t(v),
        }
    return out


def regime_min_ic(strat: dict[str, dict]) -> float | None:
    """分层 IC 的最小值（判据：> 0 表示所有制度下 IC 同号为正）。"""
    vals = [d["mean"] for d in strat.values() if d.get("mean") is not None]
    return float(min(vals)) if vals else None


# ---------------------------------------------------------------- 过拟合

@dataclass
class DeflatedSharpe:
    sharpe: float
    sr0: float           # 期望最大 Sharpe（零假设）
    dsr: float           # 通缩后 Sharpe（概率尺度 0-1）
    n_trials: int
    n_obs: int


def deflated_sharpe(sharpe: float, n_trials: int, n_obs: int,
                    skew: float = 0.0, kurt: float = 3.0,
                    sr_variance: float | None = None) -> DeflatedSharpe:
    """通缩 Sharpe（Bailey & López de Prado 2014）。

    sharpe：待检验策略的（可年化可日频，需与 n_obs 同尺度）。
    n_trials：尝试过的策略数（多重检验惩罚）。
    sr_variance：候选策略 Sharpe 的方差；缺省用 1/n_obs 的保守估计。
    dsr > 0.95 视为「在多重检验下仍显著」。
    """
    sr = float(sharpe)
    if n_trials < 2 or n_obs < 3:
        return DeflatedSharpe(sr, 0.0, _norm_cdf(sr * math.sqrt(max(n_obs - 1, 1))), max(n_trials, 1), n_obs)
    var_sr = float(sr_variance) if sr_variance is not None else (1.0 / n_obs)
    var_sr = max(var_sr, 1e-12)
    g = 0.5772156649015329  # Euler-Mascheroni
    e = math.e
    z1 = _norm_ppf(1.0 - 1.0 / n_trials)
    z2 = _norm_ppf(1.0 - 1.0 / (n_trials * e))
    sr0 = math.sqrt(var_sr) * ((1.0 - g) * z1 + g * z2)
    denom = math.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr, 1e-12))
    dsr = _norm_cdf((sr - sr0) * math.sqrt(max(n_obs - 1, 1)) / denom)
    return DeflatedSharpe(sr, sr0, dsr, n_trials, n_obs)


#: PBO 的**最小候选策略数**（判据链 #44，2026-10-11）。
#: ω = rank/(N+1) ⇒ N=2 时 ω ∈ {1/3, 2/3}，"IS 最优落到中位数下"近乎必然
#: ⇒ 该读数是**伪高**的。实测纯噪声 N=2 得 0.74~0.83，N=4 仍达 0.943。
#: ⚠ 这是**本项目选定的口径**，非 Bailey et al. 原文规定。
PBO_MIN_CANDIDATES = 4


def pbo_cscv(perf_matrix: np.ndarray, n_blocks: int = 8) -> dict:
    """PBO —— 组合对称交叉验证（Bailey et al. 2015）。

    perf_matrix：(T, N) —— T 个时点 × N 个候选策略的**逐时点表现**（如日/周收益）。
    把 T 均分为 n_blocks 个连续块，枚举所有 C(S, S/2) 个 IS/OOS 划分：
      IS 最优策略在 OOS 的相对秩 ω ∈ (0,1)，logit λ=log(ω/(1−ω))；
      PBO = P(λ ≤ 0)（IS 最优在 OOS 落到中位数以下的比例）。
    PBO < 0.5 视为可接受（低于抛硬币）。

    ⚠️ **读数前提（2026-10-11 判据链 #44 补入，违反任一 ⇒ 该数字不可用于判据）**：

    ① **PBO 是"相对量"，不是"绝对过拟合度"。**
       ω = rank/(N+1) 是被选中策略在**候选集内**的相对秩 ⇒ 读数必然依赖
       "候选集里其他策略有多强"。**同一个策略对不同对手可算出天差地别的数字**：
         本项目实测（`experiments/ln_20261011/PBO_CALIBER_AUDIT.md` §3）：
           对**强**基线 F0    （LS 日均 0.007596）⇒ PBO = **0.743**
           对**弱**基线 F0day（LS 日均 0.005759）⇒ PBO = **0.000**
       ⇒ **报 PBO 必须同时报「对手是谁」**，否则该数字无意义。
       ⇒ **本项目的正确用法**（§44 已确立）：看对**强基线**的读数。

    ② **N（候选策略数）决定读数有无意义。**
       ω 的取值档位为 N+1 ⇒ N=2 时 ω ∈ {1/3, 2/3}，"IS 最优落到中位数下"
       **近乎必然**（实测纯噪声 N=2 即得 0.74~0.83 的**伪高**读数）。
       ⇒ **N < 4 一律拒绝**（本函数已加守卫；判据链 #44）。
       ⚠ 这是**本项目选定的口径**，非 Bailey 原文规定。

    ③ **纯噪声也会给出高 PBO**（实测 N=4 纯噪声 ⇒ 0.943）。⇒ PBO 高**不等于**
       "过拟合"，可能只是"候选集里没有稳定最优"（N 小、或策略同质）。

    **历史**：守卫原为 `N < 2`（`stats.py:243`），未拦住上述 N=2/3 的伪读数。
    #42 曾据 N=2 的 0.829 给一条**减参**的臂记下"严重过拟合"错误罪名。
    """
    M = np.asarray(perf_matrix, dtype="float64")
    if M.ndim != 2:
        raise ValueError("pbo_cscv: 需要 (T, N) 矩阵")
    T, N = M.shape
    if N < PBO_MIN_CANDIDATES:
        return {"pbo": None, "n_splits": 0, "n_candidates": int(N),
                "reason": f"候选策略过少（N={N} < {PBO_MIN_CANDIDATES}）：PBO 读数无意义"
                           f"（ω 仅 {N + 1} 档；见 docstring 读数前提②）"}
    S = int(n_blocks)
    if S % 2 == 1:
        S -= 1
    if S < 2 or T < S:
        return {"pbo": None, "n_splits": 0, "reason": f"样本不足（T={T}, S={S}）"}
    bounds = np.linspace(0, T, S + 1, dtype=int)
    blocks = [np.arange(bounds[i], bounds[i + 1]) for i in range(S)]
    from itertools import combinations

    lam: list[float] = []
    for is_combo in combinations(range(S), S // 2):
        is_idx = np.concatenate([blocks[i] for i in is_combo])
        oos_idx = np.concatenate([blocks[i] for i in range(S) if i not in is_combo])
        if is_idx.size == 0 or oos_idx.size == 0:
            continue
        is_perf = np.nanmean(M[is_idx], axis=0)
        oos_perf = np.nanmean(M[oos_idx], axis=0)
        best = int(np.nanargmax(is_perf))
        # OOS 相对秩 ω ∈ (0,1)
        ranks = _ranks_avg(oos_perf)
        omega = float(ranks[best] / (N + 1.0))
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        lam.append(math.log(omega / (1.0 - omega)))
    if not lam:
        return {"pbo": None, "n_splits": 0, "reason": "无有效划分"}
    lam_a = np.asarray(lam)
    return {
        "pbo": float((lam_a <= 0).mean()),
        "n_splits": int(lam_a.size),
        "n_candidates": int(N),
        "lambda_mean": float(lam_a.mean()),
        "lambda_std": float(lam_a.std(ddof=1)) if lam_a.size > 1 else None,
    }


def _ranks_avg(a: np.ndarray) -> np.ndarray:
    """平均秩（并列取均值），1-based。"""
    a = np.asarray(a, dtype="float64")
    order = np.argsort(a, kind="stable")
    ranks = np.empty(a.size, dtype="float64")
    sa = a[order]
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sa[j + 1] == sa[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def summarize_ic(daily_ic: np.ndarray) -> dict:
    """主判据汇总：IC 均值 / t / ICIR / 正比例。"""
    a = np.asarray(daily_ic, dtype="float64")
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0, "mean": None, "t": None, "icir": None, "pos_frac": None}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "t": newey_west_t(a),
        "icir": icir(a),
        "pos_frac": float((a > 0).mean()),
    }


def paired_t(a: np.ndarray, b: np.ndarray) -> dict:
    """逐日配对差 (a−b) 的 t（同日配对；NW 校正）。返回 mean_diff / t / n。"""
    x = np.asarray(a, dtype="float64")
    y = np.asarray(b, dtype="float64")
    n = min(x.size, y.size)
    d = x[:n] - y[:n]
    d = d[np.isfinite(d)]
    if d.size < 2:
        return {"n": int(d.size), "mean_diff": None, "t": None}
    return {"n": int(d.size), "mean_diff": float(d.mean()), "t": newey_west_t(d)}
