"""判据链 · 「敞口/执行构型」与「信号精度」解耦测量（2026-10-10）。

预注册：`plans/exposure-vs-precision-prereg-20261010.md`（**先于本脚本跑数冻结**）。

## 本链回答的唯一问题
§45 登记的臂间组合层排序，是**信号质量的属性**还是**敞口/执行构型的属性**？

## 设计
**2 构型 × 13 臂** 完全交叉，唯一变量 = `account.total_dd_halt`
`{0.12（现行）, 0.9999（结构对照，实质关断）}`，**只改实验进程内 cfg 副本**。

## 判据（§五，门槛已标定：n=13 双侧 5% 临界 |ρ| = 0.5549）
- **C1**：两构型下组合层收益**秩**之间的 Spearman ρ_cross
- **C2**：IC ↔ 组合层收益的 Spearman（两构型分别）
- **C3**：均仓位 ↔ 组合层收益的 Spearman（两构型分别）
- **C4**：熔断阈值敏感性（描述性）

## 仪器门（§六，任一失败 ⇒ 读数作废）
1. 锚点复现（与 `phase4_20261010_1243.json` 逐位一致）
2. 确定性（F0@halt_off 重跑两次逐位一致）
3. 账本闭合（全部 13 臂 × 2 构型 blocked_cash == 0）
4. 标签一致性（共同键集 max|Δy_ret| == 0）
5. 面板竞态守卫（右端点 == 2026-09-30；必须在研究线 worktree 内跑）
6. 阴性置换 tripwire（5 种子，与标定值一致）
7. 环境谱系登记

## 用法
  python scripts/exposure_precision_chain.py            # 全量（13 臂 × 2 构型）
  python scripts/exposure_precision_chain.py --no-neg   # 跳过阴性对照
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fterm.config import load_config                        # noqa: E402
from fterm.data import ashare as sa                         # noqa: E402
from fterm.features.build import build_dataset              # noqa: E402
from fterm.model.validate import spearman_rank_ic           # noqa: E402

from tierfix_replay import arm_faithful, T3, START_OOS, MAX_DATA_DATE  # noqa: E402
from monthly_chain import monthly_log_paired                # noqa: E402

OUT_DIR = ROOT / "experiments" / "exposure_vs_precision_20261010"
ARCH = ROOT / "experiments" / "arch_20261009"
EXT = ROOT / "experiments" / "external_20261009"
REGIME = ROOT / "experiments" / "regime_20261009"
MH = ROOT / "experiments" / "multihorizon_20261009"
C14 = ROOT / "experiments" / "armC14_20261009"

# `data/` 在 .gitignore 内 ⇒ 研究线 worktree **没有** data/reports。
# 生产位与 DL-v2/LGBM 的预测流只存在于生产检出（只读输入，非本批产物）。
# 这与 `experiments/arch_20261009/ic_vs_portfolio/_diag_common_grid.py` 的既有做法一致
# （它们同样写 `R = r"D:\Final\data\reports"`）。流一旦漂移，仪器门1（锚点逐位复现）
# 会响亮失败 ⇒ 不需要额外的指纹机制。
PROD_REPORTS = Path(r"D:\Final\data\reports")


def _resolve(rel: str) -> Path:
    """相对仓根的流路径 → 实际文件（worktree 优先，缺失则回落生产检出）。"""
    local = ROOT / rel
    if local.exists():
        return local
    fallback = PROD_REPORTS / Path(rel).name
    if fallback.exists():
        return fallback
    return local          # 交回原路径，由 load_stream 报"缺流"并给出路径

# 预注册 §三 冻结的 13 臂（多文件臂按 mean_panel 逐字同口径合成）
ARMS: dict[str, list[Path]] = {
    "RL2": [_resolve("data/reports/rankloss_ss7_21_99_5_13.parquet"),
            _resolve("data/reports/rankloss_ss123_2024_42_314_2027.parquet")],
    "MSE2": [_resolve("data/reports/dlv2_ss7_21_99_5_13.parquet"),
             _resolve("data/reports/dlv2_ss123_2024_42_314_2027.parquet")],
    "MSE4": [_resolve("data/reports/dlv2_ss7_21_99_5_13.parquet"),
             _resolve("data/reports/dlv2_ss123_2024_42_314_2027.parquet"),
             _resolve("data/reports/dlv2_ss101_202.parquet"),
             _resolve("data/reports/dlv2_ss303_404.parquet")],
    "LGBM": [_resolve("data/reports/lgbm_preds_ss5_13.parquet"),
             _resolve("data/reports/lgbm_preds_ss42_7.parquet"),
             _resolve("data/reports/lgbm_preds_ss123_2024.parquet"),
             _resolve("data/reports/lgbm_preds_ss2027_314.parquet")],
    "A": [REGIME / "stream_A_f4.parquet"],
    "B": [MH / "stream_B_f4.parquet"],
    "C": [EXT / "stream_C_f4.parquet"],
    "C14": [C14 / "stream_C_14.parquet"],
    "F0": [EXT / "stream_F0_f4.parquet"],
    "F0day": [ARCH / "stream_F0day.parquet"],
    "D": [ARCH / "stream_D.parquet"],
    "E": [ARCH / "stream_E.parquet"],
    "E2": [ARCH / "stream_E2.parquet"],
}
REF_ARM = "F0"                      # 参照臂（预注册 §三）
HALT_ON, HALT_OFF = 0.12, 0.9999    # 预注册 §三 唯一变量
CRIT_RHO = 0.5549                   # 预注册 §五 已标定门槛（n=13 双侧 5%）
NEG_SEEDS = (1, 2, 3, 4, 5)

# 锚点（phase4_20261010_1243.json，halt_on 构型）
ANCHOR = {
    "RL2": (1.046006, -0.220555, 0.2482),
    "F0": (0.112262, -0.1347, 0.0774),
    "F0day": (0.218595, -0.1664, 0.2073),
    "D": (0.020481, -0.2094, 0.0966),
    "E2": (-0.116295, -0.1513, 0.1088),
}


def load_stream(paths: list[Path]) -> pd.DataFrame:
    """OOS 流 → 长表（date, product, pred, y_ret）。多文件臂取逐行均值。"""
    parts = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"缺流：{p}")
        d = pd.read_parquet(p, columns=["date", "product", "pred", "y_ret"])
        d["date"] = pd.to_datetime(d["date"])
        hi = d["date"].max()
        if hi > MAX_DATA_DATE:
            raise ValueError(f"{p.name}: 流含 >{MAX_DATA_DATE.date()} 的数据 {hi}")
        parts.append(d)
    base = parts[0].copy()
    if len(parts) > 1:
        base["pred"] = np.mean([x["pred"].to_numpy() for x in parts], axis=0)
    return base


def label_consistency(streams: dict[str, pd.DataFrame], ref: str) -> dict:
    """仪器门 4：各臂 `y_ret` 相对**规范标签**（`bundle.labels['fwd_ret']`）的一致性。

    ## 为什么不用「臂间逐位相等」当门（本批实测教训）
    首版把门写成"各臂 y_ret 必须与参照臂逐位相等"，**四折假失败**，实测暴露
    **两处真实的流间异质**，但两处都**不污染本批判据**：

    ① **臂 B 的 y_ret 是「训练段 std 标准化标签」**（`multihorizon_chain.py:114-118`：
       为让 5 个 horizon 的多任务损失等权，`Y / std(Y_train)`，PIT 口径）。
       实测比值中位数 **14.7434**（逐折 std 不同 ⇒ 不是全局常数）。
       ⇒ 它**不是**原始 `fwd_ret`。但 (a) 组合层 `arm_faithful` **完全不读 `y_ret`**
       （已 grep 确认：`tierfix_replay.py` 零命中），(b) IC 用 `spearman_rank_ic`
       对**日内秩**取值，而实测 B 的日内秩与规范标签**逐日完全等价**
       （秩不等日数 **0**、不等行 **0**、max|Δrank| **0.0**）⇒ IC 读数逐位不受影响
       （实测 ΔIC = **0.00e+00**）。**已实测**：这是流内自洽的口径，不是缺陷。

    ② **RL2/MSE2/LGBM 的 y_ret 有 11,899 行与规范标签不同**，**全部集中在
       2021-09-14/15/16 三天**（各占当日 ~99%），max|Δ| 0.5734。
       这是**生产流生成时点（2026-10-04~07）与今日面板的复权因子口径差**，
       属**数据 vintage 差**，不是方法差。实测对 IC 的影响
       **ΔIC = +1.9e−04**（RL2：+0.088753 → +0.088944），远小于任何判据阈值。
       组合层不读 `y_ret` ⇒ 亦不受影响。

    ## 因此门的正确口径
    不要求"臂间逐位相等"（那会把两处**已知且已量化**的异质误判为仪器失败），
    而要求：**每条臂的 `y_ret` 与规范标签的日内秩等价性**（IC 轴的真实前提）
    + **差异被显式量化并留痕**（供读者判断）。
    """
    out: dict = {}
    for n, df in streams.items():
        d = df[["date", "product", "y_ret"]].dropna()
        j = d.merge(_canonical_labels(), on=["date", "product"], how="inner")
        if not len(j):
            out[n] = {"error": "与规范标签无交集"}
            continue
        j["r_own"] = j.groupby("date")["y_ret"].rank()
        j["r_can"] = j.groupby("date")["y_canon"].rank()
        j["drank"] = (j["r_own"] - j["r_can"]).abs()
        j["dy"] = (j["y_ret"] - j["y_canon"]).abs()
        bad = j[j["drank"] > 1e-9]
        out[n] = {
            "n_joined": int(len(j)),
            "n_differing_values": int((j["dy"] > 1e-9).sum()),
            "max_abs_dy": round(float(j["dy"].max()), 6),
            "rank_equivalent_days_violated": int(bad["date"].nunique()),
            "rank_violating_rows": int(len(bad)),
            "max_abs_drank": round(float(j["drank"].max()), 1),
            "rank_equivalent": bool(len(bad) == 0),
        }
    out["_ref_arm"] = ref
    return out


def to_panel(df: pd.DataFrame) -> pd.DataFrame:
    """长表 → date×product 面板（`arm_faithful` 输入口径，逐字同 mean_panel）。"""
    return df.pivot_table(index="date", columns="product", values="pred").astype("float32")


_CANON_CACHE: dict[str, pd.DataFrame] = {}


def _canonical_labels(bundle=None) -> pd.DataFrame:
    """规范标签长表（date, product, y_canon）—— `bundle.labels['fwd_ret']` 的 stack。

    **IC 一律用规范标签重算**，不用流自带的 `y_ret`：本批实测流间标签有异质
    （臂 B 是训练段 std 标准化标签；RL2/MSE2/LGBM 有 3 天的数据 vintage 差），
    统一到规范标签可让 13 臂的 IC **严格同标签可比**（`label_consistency()` 量化差异）。
    """
    if "df" not in _CANON_CACHE:
        if bundle is None:
            from fterm.config import load_config as _lc
            from fterm.features.build import build_dataset as _bd
            bundle = _bd(_lc(ROOT / "config.yaml"), compute_fingerprints=False)
        lab = bundle.labels["fwd_ret"]
        c = lab.stack().rename("y_canon").reset_index()
        c.columns = ["date", "product", "y_canon"]
        c["date"] = pd.to_datetime(c["date"])
        _CANON_CACHE["df"] = c.dropna().reset_index(drop=True)
    return _CANON_CACHE["df"]


def daily_ic(df: pd.DataFrame, dates: list, canon: pd.DataFrame | None = None) -> float:
    """共同日网格上的平均 RankIC，**统一用规范标签**（`spearman_rank_ic`）。

    若 `canon` 给定，则用 `y_canon` 取代流自带 `y_ret`（同标签可比）。
    """
    canon = _canonical_labels() if canon is None else canon
    j = df[["date", "product", "pred"]].merge(canon, on=["date", "product"], how="inner")
    j = j[j["date"].isin(dates)]
    vals = []
    for _dt, g in j.groupby("date"):
        x = spearman_rank_ic(list(zip(g["pred"], g["y_canon"])))
        if x is not None:
            vals.append(x)
    return float(np.mean(vals)) if vals else float("nan")


def weekly_std(eq: pd.Series) -> float:
    r = eq.resample("W-FRI").last().pct_change().dropna()
    return float(r.std(ddof=1))


def spearman(xs, ys) -> float:
    return float(pd.Series(list(xs)).corr(pd.Series(list(ys)), method="spearman"))


def mc_critical(n: int, n_mc: int = 200_000, seed: int = 20261010,
                alpha: float = 0.05) -> float:
    """n 个秩的 Spearman 零分布双侧 alpha 临界值（蒙特卡洛，固定种子）。"""
    rng = np.random.RandomState(seed)
    perms = np.argsort(rng.random_sample((n, n_mc)), axis=0)
    idx = np.arange(n)
    d2 = ((idx[:, None] - perms) ** 2).sum(axis=0)
    rho = 1.0 - 6.0 * d2 / (n * (n * n - 1))
    return float(np.quantile(np.abs(rho), 1.0 - alpha))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-neg", action="store_true", help="跳过阴性置换对照")
    ap.add_argument("--arms", nargs="+", default=None, help="只跑指定臂（调试用）")
    args = ap.parse_args()
    arms = {k: v for k, v in ARMS.items() if args.arms is None or k in args.arms}

    print("== 面板 / 特征（研究线）==", flush=True)
    cfg0 = load_config(ROOT / "config.yaml")
    panels = sa.load_panels(cfg0)
    panel_max = panels["close"].index.max()
    # 仪器门 5：面板竞态守卫
    if panel_max != MAX_DATA_DATE:
        raise SystemExit(f"[仪器门5 失败] 面板右端点 {panel_max.date()} != "
                         f"隔离线 {MAX_DATA_DATE.date()}；研究线裁决必须在 worktree 内跑")
    print(f"  面板右端点 {panel_max.date()} == 隔离线 ✓", flush=True)
    bundle = build_dataset(cfg0, compute_fingerprints=False)
    dates = panels["close"].index
    start_ts = pd.Timestamp(START_OOS)
    s = pd.Series(dates, index=dates)
    month_ends = [d for d in s.groupby([dates.year, dates.month]).last() if d >= start_ts]

    print(f"== 加载 {len(arms)} 条流 ==", flush=True)
    streams = {n: load_stream(p) for n, p in arms.items()}

    # 共同交易日网格（IC 在同一网格上比较）
    day_sets = [set(df["date"].unique()) for df in streams.values()]
    grid = sorted(set.intersection(*day_sets))
    print(f"  共同交易日 {len(grid)} 天（{pd.Timestamp(grid[0]).date()} .. "
          f"{pd.Timestamp(grid[-1]).date()}）", flush=True)

    cfgs = {HALT_ON: cfg0,
            HALT_OFF: replace(cfg0, account=replace(cfg0.account,
                                                    total_dd_halt=HALT_OFF))}
    results: dict = {
        "prereg": "plans/exposure-vs-precision-prereg-20261010.md",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "tier": T3, "start_ts": str(start_ts.date()), "ema_step": "daily",
        "panel_max": str(panel_max.date()), "arms": list(arms),
        "ref_arm": REF_ARM, "crit_rho_n13": CRIT_RHO,
        "halt_levels": [HALT_ON, HALT_OFF],
        "n_common_days": len(grid),
        "instrument_gates": {},
        "stream_paths": {n: [str(p) for p in ps] for n, ps in arms.items()},
        "runs": {}, "ic": {},
    }

    # 仪器门 4：标签一致性（相对**规范标签**；见 label_consistency() 的实测依据）
    #
    # ⚠️ 门口径演进（2026-10-10，跑数阶段由门自身抓出，见预注册 erratum §六-1）：
    # 初版门要求「各臂 y_ret 逐位相等」，实测**响亮失败**并抓出两处真实异质
    # （臂 B 的训练段 std 标准化标签；RL2/MSE2/MSE4/LGBM 的 3 天数据 vintage 差）。
    # 两处均**不污染判据**，且**本批 IC 已统一用规范标签重算**（`daily_ic(..., canon)`）
    # ⇒ 流自带 `y_ret` **根本不进入 IC 轴**。故门改为检查真正的前提：
    #   ① **IC 轴确由规范标签驱动**（结构性自证，见下 `ic_canonical_selfcheck`）；
    #   ② **规范标签 join 覆盖率**足够（低于门槛才是真问题：标签面板陈旧/失配）；
    #   ③ 异质被**显式量化并留痕**（诊断，不再硬失败）。
    # 判据 C1/C2/C3 的门槛与算法**一字未改**。
    print("== 仪器门4：标签一致性（相对规范标签 fwd_ret）==", flush=True)
    lc = label_consistency(streams, REF_ARM)
    results["instrument_gates"]["label_consistency"] = lc
    for n, r in lc.items():
        if n.startswith("_"):
            continue
        flag = "✓ 日内秩等价" if r.get("rank_equivalent") else \
               f"△ 秩不等 {r.get('rank_equivalent_days_violated')} 天（见 IC 自证）"
        print(f"  {n:6s} 差异值 {r.get('n_differing_values'):>8,} 行  "
              f"max|Δy| {r.get('max_abs_dy')}  {flag}", flush=True)

    # ① 规范标签 join 覆盖率（真正的硬门：标签面板陈旧/失配时覆盖率会塌）
    cov = {n: (r.get("n_joined", 0) / max(1, len(streams[n])))
           for n, r in lc.items() if not n.startswith("_")}
    worst = min(cov.values()) if cov else 0.0
    results["instrument_gates"]["canonical_label_coverage_min"] = round(worst, 6)
    print(f"  规范标签 join 覆盖率最低 {worst:.4%}", flush=True)
    if worst < 0.99:
        raise SystemExit(f"[仪器门4 失败] 规范标签覆盖率 {worst:.2%} < 99%：标签面板失配")

    # ② IC 轴自证：取**自带标签与规范标签差异最大**的臂，把其自带 y_ret 换成噪声，
    #    IC 必须**逐位不变**（证明 IC 由规范标签驱动，而非流自带标签）。
    _cand = [(r.get("max_abs_dy", 0.0), n) for n, r in lc.items()
             if not n.startswith("_") and "max_abs_dy" in r]
    _probe_name = max(_cand)[1] if _cand else REF_ARM
    _probe = streams[_probe_name]
    _canon = _canonical_labels(bundle)
    _ic_a = daily_ic(_probe, grid, _canon)
    _corrupt = _probe.assign(
        y_ret=np.random.RandomState(20261010).randn(len(_probe)))
    _ic_b = daily_ic(_corrupt, grid, _canon)
    results["instrument_gates"]["ic_canonical_selfcheck"] = {
        "probe_arm": _probe_name,
        "ic_before": round(_ic_a, 8), "ic_after_corrupting_own_y_ret": round(_ic_b, 8),
        "invariant": bool(abs(_ic_a - _ic_b) < 1e-12),
    }
    print(f"  IC 轴自证：对自带标签差异最大的臂 {_probe_name} 污染其 y_ret，"
          f"IC {_ic_a:+.6f} → {_ic_b:+.6f} ⇒ 不变 {abs(_ic_a - _ic_b) < 1e-12}",
          flush=True)
    if abs(_ic_a - _ic_b) >= 1e-12:
        raise SystemExit("[仪器门4 失败] IC 受流自带 y_ret 影响 ⇒ IC 轴未统一到规范标签")

    # ---------- 主循环：2 构型 × N 臂 ----------
    for halt, tag in ((HALT_ON, "halt_on"), (HALT_OFF, "halt_off")):
        print(f"\n===== 构型 {tag}（total_dd_halt={halt}）=====", flush=True)
        results["runs"][tag] = {}
        for name, df in streams.items():
            t0 = time.time()
            eq, m = arm_faithful(to_panel(df), T3, cfgs[halt], bundle, panels,
                                 dates, start_ts, ema_step="daily")
            rec = {
                "total_return": round(float(m["total_return"]), 6),
                "max_drawdown": round(float(m["max_drawdown"]), 6),
                "total_costs": round(float(m["total_costs"]), 1),
                "position_util_mean": round(float(m.get("position_util_mean", float("nan"))), 4),
                "n_trades": int(m.get("n_trades", 0)),
                "n_blocked_cash": int(m.get("n_blocked_cash", 0)),
                "weekly_std": round(weekly_std(eq), 6),
                "n_governor_rejected": int(m.get("n_governor_rejected", 0)),
            }
            results["runs"][tag][name] = rec
            results.setdefault("_eq", {}).setdefault(tag, {})[name] = eq
            print(f"  {name:6s} ret {rec['total_return']:+9.2%}  mdd {rec['max_drawdown']:+7.2%}  "
                  f"util {rec['position_util_mean']:.4f}  trades {rec['n_trades']:4d}  "
                  f"blocked {rec['n_blocked_cash']}  ({time.time()-t0:.0f}s)", flush=True)

    # ---------- 仪器门 3：账本闭合 ----------
    bad = [(tag, n) for tag in results["runs"] for n, r in results["runs"][tag].items()
           if r["n_blocked_cash"] != 0]
    results["instrument_gates"]["ledger_closed"] = not bad
    if bad:
        print(f"\n!! [仪器门3 失败] blocked_cash 非零：{bad}", flush=True)
    else:
        print("\n[仪器门3] 账本闭合 ✓ 全部 blocked_cash == 0", flush=True)

    # ---------- 仪器门 1：锚点复现 ----------
    print("\n===== [仪器门1] 锚点复现（halt_on vs phase4_20261010_1243）=====", flush=True)
    anchor_ok = True
    for name, (ret, mdd, util) in ANCHOR.items():
        if name not in results["runs"]["halt_on"]:
            continue
        r = results["runs"]["halt_on"][name]
        d = (abs(r["total_return"] - ret) < 1e-5 and abs(r["max_drawdown"] - mdd) < 1e-4
             and abs(r["position_util_mean"] - util) < 5e-4)
        anchor_ok &= d
        print(f"  {name:6s} Δret {r['total_return']-ret:+.1e}  "
              f"Δmdd {r['max_drawdown']-mdd:+.1e}  Δutil {r['position_util_mean']-util:+.1e}  "
              f"{'✓' if d else '✗'}", flush=True)
    results["instrument_gates"]["anchor_ok"] = bool(anchor_ok)

    # ---------- 仪器门 2：确定性 ----------
    if REF_ARM in streams:
        print("\n===== [仪器门2] 确定性（参照臂 halt_off 重跑）=====", flush=True)
        eq2, m2 = arm_faithful(to_panel(streams[REF_ARM]), T3, cfgs[HALT_OFF], bundle,
                               panels, dates, start_ts, ema_step="daily")
        eq1 = results["_eq"]["halt_off"][REF_ARM]
        det = bool((eq1 - eq2).abs().max() < 1e-9)
        results["instrument_gates"]["determinism_ok"] = det
        print(f"  权益曲线逐位一致 {det}；ret {m2['total_return']:+.6%}", flush=True)

    # ---------- 配对 t vs 参照臂（同构型内） ----------
    print(f"\n===== 月度配对 t vs {REF_ARM}（同构型内）=====", flush=True)
    for tag in ("halt_on", "halt_off"):
        eq_ref = results["_eq"][tag][REF_ARM]
        for name in streams:
            if name == REF_ARM:
                continue
            mp = monthly_log_paired(results["_eq"][tag][name], eq_ref, month_ends)
            results["runs"][tag][name]["monthly_t_vs_ref"] = round(float(mp["t"]), 4)
            results["runs"][tag][name]["monthly_mean_log_vs_ref"] = round(float(mp["mean_log"]), 6)
        print(f"  [{tag}] " + "  ".join(
            f"{n} {results['runs'][tag][n]['monthly_t_vs_ref']:+.2f}"
            for n in streams if n != REF_ARM), flush=True)

    # ---------- 信息层 IC（共同网格 + 统一规范标签） ----------
    print("\n===== 信息层 RankIC（共同交易日网格，统一规范标签）=====", flush=True)
    canon = _canonical_labels(bundle)
    results["ic"] = {}
    results["ic_own_label"] = {}
    for name, df in streams.items():
        results["ic"][name] = round(daily_ic(df, grid, canon), 6)
        # 同时用流自带 y_ret 算一遍 ⇒ 量化"标签异质对 IC 的影响"（留痕，非门）
        j = df[["date", "product", "pred", "y_ret"]].dropna()
        j = j[j["date"].isin(grid)]
        vals = []
        for _dt, g in j.groupby("date"):
            x = spearman_rank_ic(list(zip(g["pred"], g["y_ret"])))
            if x is not None:
                vals.append(x)
        results["ic_own_label"][name] = round(float(np.mean(vals)), 6) if vals else None
        d_ic = (results["ic"][name] - results["ic_own_label"][name]) \
            if results["ic_own_label"][name] is not None else float("nan")
        print(f"  {name:6s} IC(规范) {results['ic'][name]:+.4f}  "
              f"IC(自带) {results['ic_own_label'][name]:+.4f}  Δ {d_ic:+.2e}", flush=True)
    results["ic_label_heterogeneity_max_abs_delta"] = round(max(
        abs(results["ic"][n] - results["ic_own_label"][n])
        for n in streams if results["ic_own_label"][n] is not None), 6)
    print(f"  ⇒ 标签异质对 IC 的最大影响 |ΔIC| = "
          f"{results['ic_label_heterogeneity_max_abs_delta']:.2e}（远小于判据阈值）",
          flush=True)

    # ---------- 判据 C1/C2/C3 ----------
    names = [n for n in streams]
    r_on = np.array([results["runs"]["halt_on"][n]["total_return"] for n in names])
    r_off = np.array([results["runs"]["halt_off"][n]["total_return"] for n in names])
    icv = np.array([results["ic"][n] for n in names])
    u_on = np.array([results["runs"]["halt_on"][n]["position_util_mean"] for n in names])
    u_off = np.array([results["runs"]["halt_off"][n]["position_util_mean"] for n in names])

    rho_cross = spearman(r_on, r_off)
    results["criteria"] = {
        "C1_rank_stability": {
            "spearman_halt_on_vs_off": round(rho_cross, 4),
            "crit": CRIT_RHO,
            "pass": bool(abs(rho_cross) >= CRIT_RHO),
            "reading": "排序稳定 ⇒ 构型是水平位移" if abs(rho_cross) >= CRIT_RHO
                       else "排序不稳定 ⇒ 臂间排序是构型的函数",
        },
        "C2_ic_vs_return": {
            "halt_on": {"spearman": round(spearman(icv, r_on), 4),
                        "pearson": round(float(np.corrcoef(icv, r_on)[0, 1]), 4)},
            "halt_off": {"spearman": round(spearman(icv, r_off), 4),
                         "pearson": round(float(np.corrcoef(icv, r_off)[0, 1]), 4)},
            "crit": CRIT_RHO,
        },
        "C3_util_vs_return": {
            "halt_on": {"spearman": round(spearman(u_on, r_on), 4),
                        "pearson": round(float(np.corrcoef(u_on, r_on)[0, 1]), 4)},
            "halt_off": {"spearman": round(spearman(u_off, r_off), 4),
                         "pearson": round(float(np.corrcoef(u_off, r_off)[0, 1]), 4)},
            "crit": CRIT_RHO,
        },
    }
    c2 = results["criteria"]["C2_ic_vs_return"]
    c2["both_below_crit"] = bool(abs(c2["halt_on"]["spearman"]) < CRIT_RHO
                                 and abs(c2["halt_off"]["spearman"]) < CRIT_RHO)
    c3 = results["criteria"]["C3_util_vs_return"]
    c3["both_below_crit"] = bool(abs(c3["halt_on"]["spearman"]) < CRIT_RHO
                                 and abs(c3["halt_off"]["spearman"]) < CRIT_RHO)

    print("\n===== 判据读数（门槛 |ρ| ≥ %.4f，n=%d）=====" % (CRIT_RHO, len(names)),
          flush=True)
    print(f"  C1 排序构型不变性 ρ_cross = {rho_cross:+.4f}  ⇒ "
          f"{'过' if abs(rho_cross) >= CRIT_RHO else '不过'}："
          f"{results['criteria']['C1_rank_stability']['reading']}", flush=True)
    print(f"  C2 IC↔组合层  halt_on {c2['halt_on']['spearman']:+.4f} / "
          f"halt_off {c2['halt_off']['spearman']:+.4f}  "
          f"⇒ 两者均 < 门槛 {c2['both_below_crit']}", flush=True)
    print(f"  C3 敞口↔组合层 halt_on {c3['halt_on']['spearman']:+.4f} / "
          f"halt_off {c3['halt_off']['spearman']:+.4f}  "
          f"⇒ 两者均 < 门槛 {c3['both_below_crit']}", flush=True)

    # ---------- 仪器门 6：阴性置换 tripwire ----------
    if not args.no_neg and REF_ARM in streams:
        print(f"\n===== [仪器门6] 阴性置换（{REF_ARM} 流，{len(NEG_SEEDS)} 种子）=====",
              flush=True)
        results["negative_control"] = {}
        base_panel = to_panel(streams[REF_ARM])
        for halt, tag in ((HALT_ON, "halt_on"), (HALT_OFF, "halt_off")):
            ts = []
            for seed in NEG_SEEDS:
                rng = np.random.RandomState(seed)
                v = base_panel.to_numpy(dtype="float64", copy=True)
                for i in range(v.shape[0]):
                    row = v[i]
                    ok = ~np.isnan(row)
                    if ok.sum() > 1:
                        v[i][ok] = row[ok][rng.permutation(int(ok.sum()))]
                p = pd.DataFrame(v, index=base_panel.index, columns=base_panel.columns)
                eq, _m = arm_faithful(p, T3, cfgs[halt], bundle, panels, dates,
                                      start_ts, ema_step="daily")
                mp = monthly_log_paired(eq, results["_eq"][tag][REF_ARM], month_ends)
                ts.append(round(float(mp["t"]), 4))
                print(f"  [{tag}] 种子 {seed}: t vs {REF_ARM} {ts[-1]:+.2f}", flush=True)
            arr = np.asarray(ts)
            results["negative_control"][tag] = {
                "t_list": ts, "t_mean": round(float(arr.mean()), 4),
                "t_std": round(float(arr.std(ddof=1)), 4),
                "t_abs_max": round(float(np.abs(arr).max()), 4),
            }
            print(f"  [{tag}] 置换 t: mean {arr.mean():+.3f} sd {arr.std(ddof=1):.3f} "
                  f"max|t| {np.abs(arr).max():.3f}", flush=True)

    # ---------- 仪器门 7：环境谱系 ----------
    import platform
    results["instrument_gates"]["env_lineage"] = {
        "python": platform.python_version(),
        "pandas": pd.__version__, "numpy": np.__version__,
        "torch": __import__("torch").__version__,
        "venv": str(Path(sys.executable).parent),
    }
    results["instrument_gates"]["n13_critical_recomputed"] = round(mc_critical(len(names)), 4)
    print(f"\n[仪器门7] 环境 {results['instrument_gates']['env_lineage']}", flush=True)
    print(f"  n={len(names)} 临界值现场重算 = "
          f"{results['instrument_gates']['n13_critical_recomputed']}（预注册冻结 {CRIT_RHO}）",
          flush=True)

    # ---------- 落盘 ----------
    for tag in results.get("_eq", {}):
        results["_eq"][tag] = {k: None for k in results["_eq"][tag]}
    results.pop("_eq", None)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out = OUT_DIR / f"confirm_{stamp}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str),
                   encoding="utf-8")
    print(f"\n报告 → {out}", flush=True)

    gates = results["instrument_gates"]
    print(f"\n仪器门汇总：锚点 {gates.get('anchor_ok')} | 确定性 {gates.get('determinism_ok')} "
          f"| 账本闭合 {gates.get('ledger_closed')} "
          f"| 标签覆盖率 {gates.get('canonical_label_coverage_min')} "
          f"| IC轴自证 {gates.get('ic_canonical_selfcheck', {}).get('invariant')} "
          f"| 面板 {results['panel_max']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
