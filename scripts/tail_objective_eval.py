"""判据链 #26 裁决执行器：按预注册 §六 判据出数（2026-10-10）。

预注册：`plans/tail-objective-prereg-20261010.md`（**先于跑数冻结**；含 4 处已登记勘误）。

## ⚠ 口径演进史（**读本文件前必看**）
1. **初稿**：主指标 = `hit1_at5_volmat`（波动率档内均衡命中率）
   ⇒ 自查否决（无偏估计需逆概率加权，会把精度与选股分布混淆）⇒ 勘误 1。
2. **二稿**：主指标 = `hit1_at5`（原始区域命中率），选它的理由是**功效**
   （`_calib_metric_power.py`：效应/MDE **4.07**，而 `exc1` 仅 1.33）⇒ §6.1。
3. **三稿（当前）**：`hit1_at5` **被撤回** —— 负向自检（"取负应变差"）失败，
   实测取负后**反升**（0.1309→0.1534）；追查证明它度量的是
   **"预测值扎堆在收益分布极端"**，不是"选出赢家"（勘误 3，见 §6.3）。
   ⇒ 改为**两层分开报、不合成**：
     · **极端性层** `extreme = hit_top5 + hit_bot5`（可测 ⇒ 只能声称"改变了行为"）
     · **方向性层** `exc1`（金额版）/ `pct_argmax`、`net_hit`（排名版）
       ⇒ 按 #24 R3 纪律**必须报效应/MDE**，< 2 者判「功效不足」，**不得**裁决

## ⚠ 两层方向性指标**符号相反**（实测）
`pct_argmax` 是 **RL > MSE**（0.5352 vs 0.4934，均值口径），而 `exc1` 是
**MSE > RL**（+139.9bp vs +8.6bp）⇒ **不存在单一的"尾部能力"量**，
故**禁止**合成单一分数，也禁止只挑一个报（预注册 §6.3）。

## 本脚本
1. 载入 C0 / T1 / T2（+ NEG_T1）流，**并强制 E8 日集守卫**（防误用 1 折前哨流）；
2. 指标：极端性层 + 方向性两版 + 描述性（IC / 尾内 IC / 档内成分校正 `hit_excess_*`）；
3. 逐日配对差 + NW-t（多滞后）+ 移动块自举 CI95（B=20,000/block=6/种子 20261010）+ MDE；
4. 全样本与**后段**（2021-01-01 起，#25 regime 纪律）各报一遍；
5. 组合层（w2/t3 忠实账本）**只登记不裁决**（预注册 §六 C5）；
6. 输出 `verdict_tail_objective_*.json`（判决词留空，由用户填写）。

用法：python scripts/tail_objective_eval.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fterm.config import load_config                     # noqa: E402
from fterm.data import ashare as sa                      # noqa: E402
from fterm.features.build import build_dataset           # noqa: E402
from fterm.model.stats import newey_west_t               # noqa: E402
from fterm.model.validate import spearman_rank_ic        # noqa: E402

EXDIR = ROOT / "experiments" / "tail_objective_20261010"
PROD = Path(r"D:\Final\data\reports")
FULL = pd.Timestamp("2019-01-02")
LATE = pd.Timestamp("2021-01-01")
SEEDSET = "7_21_99_5_13"
NQ = 5
TAIL_Q = 0.95
Z2 = 1.959963984540054 + 0.8416212335729143
BOOT_B, BOOT_BLOCK, BOOT_SEED = 20_000, 6, 20261010
CRIT_T = 2.24            # Bonferroni：2 臂 × 1 主指标（预注册 §六 C6）
TRIPWIRE_LIFT = 1.50     # G5 阴性对照 tripwire（预注册 §七）
TIERS = {
    "w2": dict(equity=20000, top_n=2, keep_n=4, max_symbols=2, max_lots=10),
    "t3": dict(equity=1_000_000, top_n=10, keep_n=20, max_symbols=10, max_lots=50),
}


def load(path: Path) -> pd.DataFrame:
    d = pd.read_parquet(path, columns=["date", "product", "pred"])
    d["date"] = pd.to_datetime(d["date"])
    return d.pivot_table(index="date", columns="product", values="pred").astype("float32")


def qcut_rows(X: np.ndarray, nq: int = NQ) -> np.ndarray:
    out = np.full(X.shape, -1, dtype=np.int8)
    for i in range(X.shape[0]):
        r = X[i]
        ok = np.isfinite(r)
        n = int(ok.sum())
        if n < nq * 10:
            continue
        pct = np.empty(n)
        pct[np.argsort(r[ok], kind="stable")] = (np.arange(n) + 0.5) / n
        out[i, ok] = np.minimum((pct * nq).astype(np.int8), nq - 1)
    return out


def metrics(P: pd.DataFrame, idx, mask: np.ndarray, Y: np.ndarray,
            Q: dict[str, np.ndarray | None]) -> dict[str, np.ndarray]:
    """逐日指标。Q = {字段名: 逐日 5 档矩阵}（供 C2c 的档内基准调整）。

    主指标 `hit1_at5` = 逐日 0/1（模型 top-1 是否落在当日真实 top-5%）⇒ 可直接配对。

    C2c 指标 `hit_excess_<字段>` = `hit1_at5` − `base_q`，其中 `base_q` 是**当日该股所属档**
    的"随机命中率"（= 档内真实 top-5% 占比）。这是**逐日、无偏、按该臂自身选股分布调整**
    的成分校正量（间接标准化）：
        E[hit] = Σ_q P(选在 q)·P(命中|选在 q)，  E[base_q] = Σ_q P(选在 q)·base_q
      ⇒ E[excess] = Σ_q P(选在 q)·(该档命中率 − 该档基准)
    对照实测（`calib_stratified.json`）**基准按档差 5 倍**（pvol：0.0175→0.0931）
    ⇒ 不做这层调整就无法区分"精度"与"选股分布偏移"。
    为什么不用逆概率加权：那需要估计"模型选该档的概率"，会把精度与选股分布**再混一次**
    （预注册 §6.1 勘误已登记同一理由）。
    """
    Pm = np.where(mask, P.reindex(index=idx).to_numpy("float64"), np.nan)
    mkt = np.nanmean(Y, axis=1)
    n = len(idx)
    keys = ("exc1", "exc2", "exc10", "exc50", "ic_all", "ic_t50",
            "hit1_at1", "hit1_at5", "hit1_at10", "hit2any_at5",
            "hit_bot5", "extreme", "net_hit", "pct_argmax", "pct_spread")
    out = {k: np.full(n, np.nan) for k in keys}
    for tag in Q:
        out[f"hit_excess_{tag}"] = np.full(n, np.nan)
    for i in range(n):
        pr, yr = Pm[i], Y[i]
        m = np.isfinite(pr) & np.isfinite(yr)
        if int(m.sum()) < 205:
            continue
        pr_m, yr_m = pr[m], yr[m]
        pc = np.empty(len(yr_m))
        pc[np.argsort(yr_m, kind="stable")] = (np.arange(len(yr_m)) + 0.5) / len(yr_m)
        o = np.argsort(-pr_m)
        for k in (1, 2, 10, 50):
            out[f"exc{k}"][i] = yr_m[o[:k]].mean() - mkt[i]
        out["hit1_at1"][i] = float(pc[o[0]] >= 0.99)
        out["hit1_at5"][i] = float(pc[o[0]] >= 0.95)
        out["hit1_at10"][i] = float(pc[o[0]] >= 0.90)
        out["hit2any_at5"][i] = float(pc[o[:2]].max() >= 0.95)
        # ---- 勘误 3（预注册 §6.1）：hit1_at5 **不度量方向能力**，只度量"下注在极端"。
        # 故把两个维度拆开报：extreme（极端性，可测）/ net_hit 与 pct_argmax（方向性）。
        out["hit_bot5"][i] = float(pc[o[0]] <= 0.05)
        out["extreme"][i] = float(pc[o[0]] >= 0.95) + float(pc[o[0]] <= 0.05)
        out["net_hit"][i] = float(pc[o[0]] >= 0.95) - float(pc[o[0]] <= 0.05)
        out["pct_argmax"][i] = pc[o[0]]
        out["pct_spread"][i] = pc[o[0]] - pc[o[-1]]
        v = spearman_rank_ic(list(zip(pr_m, yr_m)))
        if v is not None:
            out["ic_all"][i] = v
        if len(o) > 50:
            v = spearman_rank_ic(list(zip(pr_m[o[:50]], yr_m[o[:50]])))
            if v is not None:
                out["ic_t50"][i] = v
        tail = pc >= TAIL_Q
        for tag, Qd in Q.items():
            if Qd is None:
                continue
            qm = Qd[i][m]
            q_pick = int(qm[o[0]])
            if q_pick < 0:
                continue
            sel = qm == q_pick
            if int(sel.sum()) < 20:
                continue
            base_q = float(tail[sel].mean())
            out[f"hit_excess_{tag}"][i] = float(pc[o[0]] >= TAIL_Q) - base_q
    return out


def _nw_t(x: np.ndarray, lags=None):
    try:
        v = newey_west_t(x, lags=lags) if lags is not None else newey_west_t(x)
    except Exception:
        return None
    if v is None:
        return None
    v = float(v)
    return v if np.isfinite(v) else None


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    x = np.asarray(a, "float64") - np.asarray(b, "float64")
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 5:
        return {"n": n, "error": "样本不足"}
    sd = float(x.std(ddof=1))
    mde = Z2 * sd / np.sqrt(n)
    ts = {}
    for L in (None, 0, 3, 6, 12):
        v = _nw_t(x, lags=L)
        ts["default" if L is None else f"lags{L}"] = round(v, 3) if v is not None else None
    rng = np.random.RandomState(BOOT_SEED)
    boot = np.empty(BOOT_B)
    for i in range(BOOT_B):
        ix = []
        while len(ix) < n:
            st = rng.randint(0, n)
            ix.extend(((st + np.arange(BOOT_BLOCK)) % n).tolist())
        boot[i] = x[np.array(ix[:n])].mean()
    lo, hi = np.quantile(boot, [0.025, 0.975])
    t_def = ts["default"]
    return {
        "n": n,
        "mean_diff": float(x.mean()),
        "sd_diff": sd,
        "nw_t": ts,
        "nw_t_default": t_def,
        "boot_ci95": [round(float(lo), 8), round(float(hi), 8)],
        "boot_contains_zero": bool(lo <= 0 <= hi),
        "mde": mde,
        "ratio_abs_mean_over_mde": (round(abs(float(x.mean())) / mde, 3) if mde > 0 else None),
        "passes_t2": bool(t_def is not None and abs(t_def) >= 2),
        "passes_bar": bool(t_def is not None and t_def >= CRIT_T and not (lo <= 0 <= hi)),
    }


def check_day_sets(stream_meta: dict, min_days: int = 1800,
                   max_tail_gap: int = 15,
                   panel_end: pd.Timestamp = pd.Timestamp("2026-09-30")) -> None:
    """E8 日集守卫（2026-10-10，由独立复核抓出；**独立成函数以便测试能失败**）。

    三条件，任一不满足 ⇒ `SystemExit`：
      ① C0 流日数 ≥ `min_days`（4 折 ≈1871 ⇒ 1800 是宽松下界）—— 防误用 1 折前哨流
         （前哨只有 ≈418 天，会让判据在 418 天上**静默出数**）；
      ② C0 流右端点距面板末 < `max_tail_gap` 天 —— 防缺终端折；
      ③ 各臂流日数彼此相差 ≤5 —— 防配对日集不同源。

    由 `tests/test_tail_objective_gates.py` 用**合成流**逐条证伪（门必须能失败）。
    """
    if "C0" not in stream_meta:
        raise SystemExit("[E8 门] 无 C0 流")
    n0 = stream_meta["C0"]["n_days"]
    if n0 < min_days:
        m = stream_meta["C0"]
        raise SystemExit(
            f"[E8 门] C0 流只有 {n0} 个交易日（4 折应 ≈1871）⇒ 疑似误用前哨流"
            f"（{m['file']}，{m.get('first')}..{m.get('last')}）。拒绝出数。")
    last = pd.Timestamp(stream_meta["C0"]["last"])
    gap = (panel_end - last).days
    if gap >= max_tail_gap:
        raise SystemExit(f"[E8 门] C0 流右端点 {last.date()} 距面板末 "
                         f"{panel_end.date()} 有 {gap} 天 ⇒ 折边界不完整，拒绝出数")
    for a, m in stream_meta.items():
        if abs(m["n_days"] - n0) > 5:
            raise SystemExit(f"[E8 门] {a} 流日数 {m['n_days']} 与 C0 的 {n0} "
                             f"不一致 ⇒ 配对日集不同源，拒绝出数")


def main() -> int:
    cfg = load_config(ROOT / "config.yaml")
    panels = sa.load_panels(cfg)
    if panels["close"].index.max() != pd.Timestamp("2026-09-30"):
        raise SystemExit("[G1] 面板右端点 != 2026-09-30；须在研究线 worktree 内跑")
    bundle = build_dataset(cfg, compute_fingerprints=False)
    dates = panels["close"].index
    uni, labels = bundle.universe, bundle.labels["fwd_ret"]
    vol = bundle.features["pvol_20"]
    size = bundle.features["size_log"]
    turn = bundle.features["turnover_20"]

    arms = {}
    stream_meta = {}
    for arm in ("C0", "T1", "T2"):
        f = EXDIR / f"stream_{arm}_ss{SEEDSET}.parquet"
        if not f.exists():
            print(f"[缺失] {f.name} —— 该臂还没跑完，跳过")
            continue
        arms[arm] = load(f)
        stream_meta[arm] = {"file": f.name, "n_days": int(len(arms[arm])),
                            "first": str(arms[arm].index.min().date()),
                            "last": str(arms[arm].index.max().date()),
                            "n_stocks": int(arms[arm].shape[1])}
    if "C0" not in arms:
        raise SystemExit("[门] 无 C0 新鲜对照流 ⇒ 单变量对照不成立，拒绝出数")

    # ---- E8 日集守卫（独立函数 ⇒ 由 tests 证伪；见 check_day_sets 文档）----
    check_day_sets(stream_meta)
    base = arms["C0"]
    tail_gap = (pd.Timestamp("2026-09-30") - base.index.max()).days
    print(f"[E8 门] 各流日数一致（C0={len(base)}），右端点距面板末 {tail_gap} 天 ✓")
    codes = [c for c in base.columns if c in labels.columns]

    doc: dict = {
        "prereg": "plans/tail-objective-prereg-20261010.md",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "crit_t": CRIT_T,
        # E11/小项：口径名已随勘误 3 更新（复核指出 docstring 与 main_metric 曾滞后）
        "main_metric": "RETRACTED: hit1_at5（勘误 3）→ 两层分报 extreme / exc1 / pct_argmax",
        "layers": {"extremeness": "extreme = hit_top5 + hit_bot5",
                   "directionality_mag": "exc1",
                   "directionality_rank": "pct_argmax / net_hit"},
        "stream_meta": stream_meta,
        "boot": {"B": BOOT_B, "block": BOOT_BLOCK, "seed": BOOT_SEED},
        "segments": {}, "arms_present": sorted(arms),
    }

    for seg, start in (("full", FULL), ("late", LATE)):
        idx = [d for d in dates if d >= start and d in base.index and d in labels.index]
        msk = uni.reindex(index=idx, columns=codes).fillna(False).to_numpy(dtype=bool)
        Y = np.where(msk, labels.reindex(index=idx, columns=codes).to_numpy("float64"), np.nan)

        def q_of(feat):
            Fv = np.where(msk, feat.reindex(index=idx, columns=codes).to_numpy("float64"), np.nan)
            return qcut_rows(Fv)

        Q = {"pvol": q_of(vol), "size": q_of(size), "turn": q_of(turn)}
        M = {a: metrics(p, idx, msk, Y, Q) for a, p in arms.items()}
        print("\n" + "=" * 110)
        print(f"===== {seg}（{len(idx)} 日：{idx[0].date()} .. {idx[-1].date()}）=====")
        print("=" * 110)

        keys = ("hit1_at5", "hit1_at1", "hit1_at10", "hit2any_at5",
                "exc1", "exc2", "exc10", "ic_all", "ic_t50",
                "hit_bot5", "extreme", "net_hit", "pct_argmax", "pct_spread")
        strat = ("hit_excess_pvol", "hit_excess_size", "hit_excess_turn")

        print("\n① 水平（诊断）")
        show = keys + strat
        print(f"{'臂':>5s} " + " ".join(f"{k[:11]:>12s}" for k in show))
        print("-" * (6 + 13 * len(show)))
        lv = {}
        for a, m in M.items():
            lv[a] = {}
            for k in show:
                x = m[k][np.isfinite(m[k])]
                lv[a][k] = round(float(x.mean()), 6) if len(x) else None
            print(f"{a:>5s} " + " ".join(f"{lv[a][k]:12.4f}" for k in show))
        doc["segments"][seg] = {"n_days": len(idx), "levels": lv, "tests": {}}

        print("\n② 配对检验（vs C0，逐日配对；主判据 bar = NW-t ≥ 2.24 且 CI 不含 0）")
        print(f"{'臂':>5s} {'指标':>16s} {'效应':>12s} {'NW-t':>8s} {'CI95 含0':>9s} "
              f"{'MDE':>12s} {'效应/MDE':>9s} {'过':>4s}")
        print("-" * 86)
        for a in arms:
            if a == "C0":
                continue
            for k in show:
                pr = paired(M[a][k], M["C0"][k])
                if "error" in pr:
                    continue
                scale = 1e4 if k.startswith("exc") else 1.0
                doc["segments"][seg]["tests"][f"{a}|{k}"] = pr
                td = pr["nw_t_default"]
                print(f"{a:>5s} {k:>16s} {pr['mean_diff']*scale:12.4f} "
                      f"{(td if td is not None else float('nan')):8.2f} "
                      f"{str(pr['boot_contains_zero']):>9s} {pr['mde']*scale:12.4f} "
                      f"{(pr['ratio_abs_mean_over_mde'] if pr['ratio_abs_mean_over_mde'] else float('nan')):9.2f} "
                      f"{'✓' if pr['passes_bar'] else '✗':>4s}")

    # ---------------- 判据汇总 ----------------
    print("\n" + "=" * 110)
    print("③ 判据汇总（预注册 §六）")
    print("=" * 110)
    crit: dict = {}
    for a in arms:
        if a == "C0":
            continue
        strat = {tag: doc["segments"]["full"]["tests"].get(f"{a}|hit_excess_{tag}", {})
                 for tag in ("pvol", "size", "turn")}
        # ⚠ 小项修复（独立复核抓出）：原 `all(... if v)` 在**空序列**时为 True ⇒ fail-open。
        # 改为：三个字段都必须存在且 t > 0；缺任一 ⇒ 不过。
        strat_ok = bool(strat) and all(
            v and (v.get("nw_t_default") or 0) > 0 for v in strat.values()) and len(strat) == 3
        # 撤回后口径（预注册 §6.3）：两层分开，不合成
        cext = doc["segments"]["full"]["tests"].get(f"{a}|extreme", {})
        cextl = doc["segments"]["late"]["tests"].get(f"{a}|extreme", {})
        # 方向性两版
        cdir_rank = doc["segments"]["full"]["tests"].get(f"{a}|pct_argmax", {})
        cdir_mag = doc["segments"]["full"]["tests"].get(f"{a}|exc1", {})
        crit[a] = {
            # C1′ 极端性：可测维度
            "C1prime_extreme_t": cext.get("nw_t_default"),
            "C1prime_extreme_effect": cext.get("mean_diff"),
            "C1prime_changed": bool(cext.get("nw_t_default") is not None
                                    and abs(cext["nw_t_default"]) >= CRIT_T),
            # C2′ 方向性：按 #24 R3 纪律，只报功效
            "C2prime_mag_effect_over_mde": cdir_mag.get("ratio_abs_mean_over_mde"),
            "C2prime_mag_t": cdir_mag.get("nw_t_default"),
            "C2prime_mag_is_powered": bool(
                (cdir_mag.get("ratio_abs_mean_over_mde") or 0) >= 2),
            "C2prime_rank_t": cdir_rank.get("nw_t_default"),
            "C2prime_rank_effect_over_mde": cdir_rank.get("ratio_abs_mean_over_mde"),
            # 独立复核 ERRATUM-5：rank 版与 mag 版**都**功效不足 ⇒ 方向性层整体判不出
            "C2prime_rank_is_powered": bool(
                (cdir_rank.get("ratio_abs_mean_over_mde") or 0) >= 2),
            "C2prime_directionality_verdict": (
                "有功效" if ((cdir_mag.get("ratio_abs_mean_over_mde") or 0) >= 2
                             and (cdir_rank.get("ratio_abs_mean_over_mde") or 0) >= 2)
                else "**方向性层判不出（两版均功效不足）**"),
            # C3′ regime
            "C3prime_extreme_late_t": cextl.get("nw_t_default"),
            "C4prime_layers_agree": None,
            # C2c 暴露成分校正
            "C2c_excess_all_positive": bool(strat_ok),
            "C2c_t": {k: v.get("nw_t_default") for k, v in strat.items()},
        }
        c = crit[a]
        print(f"  [{a}] C1′ 极端性 t={c['C1prime_extreme_t']} "
              f"（{'改变了行为' if c['C1prime_changed'] else '未改变'}） | "
              f"C2′ 方向性 mag(exc1) t={c['C2prime_mag_t']} 效应/MDE="
              f"{c['C2prime_mag_effect_over_mde']} ⇒ "
              f"{'有功效' if c['C2prime_mag_is_powered'] else '**功效不足**'} | "
              f"rank(pct_argmax) t={c['C2prime_rank_t']} 效应/MDE="
              f"{c['C2prime_rank_effect_over_mde']} ⇒ "
              f"{'有功效' if c['C2prime_rank_is_powered'] else '**功效不足**'} | "
              f"⇒ {c['C2prime_directionality_verdict']} | "
              f"C3′ 后段 t={c['C3prime_extreme_late_t']} | "
              f"C2c 全正={c['C2c_excess_all_positive']} {c['C2c_t']}")
    doc["criteria"] = crit
    # C7 条件臂：撤回后口径（预注册 §6.3 + 复核建议）——只有"行为确实改变"**且**
    # "方向性两版都有功效且同向为正"才谈得上条件臂；**绝不**用无功效读数触发。
    # ⚠ C2c（hit_excess）**不参与**触发：独立复核证它挡不住平凡规则
    #（最高 pvol / 最高 mom 零训练规则在三维上全为正 t 3.16–4.93）⇒ 降级为诊断项。
    doc["C7_condition_arm"] = bool(any(
        c["C1prime_changed"] and c.get("C2prime_mag_is_powered")
        and c.get("C2prime_rank_is_powered")
        and (c.get("C2prime_mag_t") or 0) > 0 and (c.get("C2prime_rank_t") or 0) > 0
        for c in crit.values()))
    doc["C2c_status"] = ("诊断项（**不作判据**）：独立复核证 `hit_excess` 无偏但尺度随"
                         "拾取分布改变（同一精度 2.3× 差异），且挡不住最高 pvol/mom 平凡规则")
    # ⚠ E13（预注册 §6.3，由独立复核抓出、lead 独立复核 28 对确认）：
    # C7 的触发条件在本批数据上**不可达（0/28）** —— 两版方向性指标功效都 <2，
    # 且 max 分别落在**不同的对**上（mag 最高 2.08 的对 rank 仅 0.10；rank 最高 3.04 的对 mag 仅 0.40）。
    # ⇒ 一个**永不触发**的条款与**永真**条款同样不可 falsify。
    # 处置（在 T2/NEG 读数存在之前登记）：**保留条款，但显式声明其性质**。
    doc["C7_note"] = ("E13：C7 是**低概率条件条款**；**不触发 ≠ 判负，也不构成任何证据**。"
                      "实测本批可达性 0/28（两版方向性指标功效均 <2）。"
                      "C7 的输出只报两层读数 + 声明功效不足，不由它推导采纳。")
    doc["metric_erratum"] = ("预注册 §6.1 勘误 3：主判据 hit1_at5 被撤回"
                             "（它度量极端性而非方向能力；MSE 的 top5/bot5 命中都高、"
                             "取负后 hit_top5 反升）。撤回后按 §6.3 两层分开报。")

    # ---------------- 阴性对照 G5 ----------------
    # ⚠ E6 修复（2026-10-10，由独立复核抓出）：原实现 `abs(ic) <= 0.025` 在 ic 为
    # **NaN** 时求值为 False ⇒ **完美的阴性对照被判「泄漏」**（常数预测的极限形态
    # 会让所有逐日 IC 变 NaN）。改为 fail-closed 的正确语义：
    #   ① IC 全 NaN ⇒ 无法计算 ⇒ 视为「塌到噪声」（常数预测必然无信息），**不是**泄漏；
    #   ② 命中率 tripwire 同时检查「档内均衡 lift」（相对当日基准），避免只比绝对 0.075。
    negf = EXDIR / f"stream_NEG_T1_ss{SEEDSET}.parquet"
    if negf.exists():
        Neg = load(negf)
        # ⚠ 修复（2026-10-11）：NEG_T1 是 **单折**（fold3）流 ⇒ 其 (日期, 代码) 网格
        # 与 C0 的 4 折网格**不同源**（实测 418 日 / 5164 股 vs C0 1874 日 / 5385 股）。
        # 必须用 **NEG 自身**的代码集与日期集，绝不能沿用 C0 的 codes（否则
        # reindex 后列不齐 ⇒ broadcast ValueError，或更糟：静默对不齐）。
        # G5 的目的是"这条置换标签流是否塌到噪声"，本来就与 C0 无关。
        neg_codes = [c for c in Neg.columns if c in labels.columns]
        idx = [d for d in dates if d >= FULL and d in Neg.index and d in labels.index]
        msk = uni.reindex(index=idx, columns=neg_codes).fillna(False).to_numpy(dtype=bool)
        Y = np.where(msk, labels.reindex(index=idx, columns=neg_codes).to_numpy("float64"), np.nan)
        Vn = np.where(msk, vol.reindex(index=idx, columns=neg_codes).to_numpy("float64"), np.nan)
        Qn = {"pvol": qcut_rows(Vn)}
        Mn = metrics(Neg, idx, msk, Y, Qn)
        hit = float(np.nanmean(Mn["hit1_at5"]))
        icv = Mn["ic_all"][np.isfinite(Mn["ic_all"])]
        ic = float(icv.mean()) if len(icv) else None
        # 档内均衡：hit − 当日基准（负 ⇒ 优于随机；tripwire 用绝对值）
        excess = Mn["hit_excess_pvol"]
        exc_mean = float(np.nanmean(excess)) if np.isfinite(excess).any() else None
        trip_hit = 0.05 * TRIPWIRE_LIFT
        # fail-closed 语义：IC 无法计算（全 NaN）⇒ 不算违规
        ic_ok = (ic is None) or (abs(ic) <= 0.025)
        hit_ok = hit <= trip_hit
        # 档内均衡 lift ≈ 1 + excess/0.05 ⇒ tripwire 1.5 对应 excess ≤ 0.025
        exc_ok = (exc_mean is None) or (exc_mean <= 0.025)
        ok = bool(ic_ok and hit_ok and exc_ok)
        doc["G5_negative_control"] = {
            "hit1_at5": round(hit, 5), "ic_all": (round(ic, 5) if ic is not None else None),
            "ic_nan_days": int(np.isnan(Mn["ic_all"]).sum()),
            "hit_excess_pvol_mean": (round(exc_mean, 5) if exc_mean is not None else None),
            "tripwire_hit": round(trip_hit, 5), "tripwire_ic": 0.025,
            "ic_ok": bool(ic_ok), "hit_ok": bool(hit_ok), "excess_ok": bool(exc_ok),
            "ok": ok,
            "note": "E6 修复：IC 全 NaN ⇒ 视为塌到噪声（fail-closed 语义修正）；"
                    "并加档内均衡（hit_excess）检查",
        }
        print(f"\n[G5 阴性对照 NEG_T1] 命中率 {hit:.4f}（tripwire ≤ {trip_hit:.4f}）"
              f" ic={ic if ic is not None else 'NaN(视为无信息)'}（≤0.025）"
              f" 档内超额均值 {exc_mean if exc_mean is not None else 'n/a'}"
              f" ⇒ {'✓ 塌到噪声' if ok else '✗ 未塌 ⇒ 疑似泄漏'}")
        # ---- E6 ② 拾取档诊断（复核要求：解释 0.075 tripwire 的依据）----
        # 逐日记录"该臂 top-1 落在哪个 pvol 档"，报分布 + 各档 lift。
        # 用途：若阴性对照的命中率偏高**只是因为它偏向高基准档**，
        # 则 hit_excess（已扣档内基准）应接近 0；若 hit_excess 也偏高 ⇒ 真有问题。
        Pmn = np.where(msk, Neg.reindex(index=idx).to_numpy("float64"), np.nan)
        pick_q = np.full(len(idx), -1, dtype=int)
        lift_by_q: dict[int, list] = {q: [] for q in range(NQ)}
        base_by_q: dict[int, list] = {q: [] for q in range(NQ)}
        for i in range(len(idx)):
            pr, yr = Pmn[i], Y[i]
            m = np.isfinite(pr) & np.isfinite(yr) & (Qn["pvol"][i] >= 0)
            if int(m.sum()) < 205:
                continue
            pr_m, yr_m = pr[m], yr[m]
            qm = Qn["pvol"][i][m]
            pc = np.empty(len(yr_m))
            pc[np.argsort(yr_m, kind="stable")] = (np.arange(len(yr_m)) + 0.5) / len(yr_m)
            o = int(np.argmax(pr_m))
            pick_q[i] = int(qm[o])
            for q in range(NQ):
                sel = qm == q
                if int(sel.sum()) < 20:
                    continue
                bq = float((pc[sel] >= 0.95).mean())
                base_by_q[q].append(bq)
                if qm[o] == q:
                    lift_by_q[q].append(float(pc[o] >= 0.95) / bq if bq > 0 else np.nan)
        dist = {int(q): int((pick_q == q).sum()) for q in range(NQ)}
        lift_mean = {int(q): (float(np.nanmean(lift_by_q[q])) if lift_by_q[q] else None)
                     for q in range(NQ)}
        base_mean = {int(q): (float(np.mean(base_by_q[q])) if base_by_q[q] else None)
                     for q in range(NQ)}
        doc["G5_negative_control"].update({
            "pick_stratum_dist": dist,
            "lift_by_stratum": lift_mean,
            "base_by_stratum": base_mean,
            "note2": "E6②：若命中率偏高只因偏向高基准档 ⇒ hit_excess 应≈0；"
                     "两者都高才判泄漏",
        })
        print(f"      [E6② 拾取档] 分布 {dist}；各档 lift {lift_mean}")
    else:
        doc["G5_negative_control"] = {"ok": None, "note": "NEG_T1 未跑"}

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out = EXDIR / f"verdict_tail_objective_{stamp}.json"
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n判决材料（判决词留空，由用户填写）→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
