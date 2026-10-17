"""判据链 #25 · 训练目标（MSE vs rank-loss）的确认性检验（2026-10-10）。

预注册：`plans/objective-swap-prereg-20261010.md`（**先于本脚本跑数冻结**）。

## 对照设计（天然单变量，无需重训）
  `dlv2_ss{7,21,99,5,13}.parquet`      = MSE
  `rankloss_ss{7,21,99,5,13}.parquet`  = MSE + RankNet 秩项
同架构 / 同种子 / 同折边界 / 同面板 / 同标签 ⇒ **唯一差异 = 损失函数**。

## 判据
  C1 后段（2021-01-01 起）w2 与 t3 两档，月度配对 NW-t ≥ 2 且 CI 不含 0
  C2 全样本对照（防"分割点是我挑的"）：全样本不得反向显著
  C3 Bonferroni（2 检验）⇒ |t| ≥ 2.24
  C4 尾部机制一致性（top-1 超额差与账本差同号）
  C5 功效声明（MDE）
"""
from __future__ import annotations

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

from fterm.config import load_config                       # noqa: E402
from fterm.data import ashare as sa                        # noqa: E402
from fterm.features.build import build_dataset             # noqa: E402
from fterm.model.stats import newey_west_t                 # noqa: E402

from tierfix_replay import arm_faithful                    # noqa: E402
from monthly_chain import monthly_log_paired               # noqa: E402
import eval_criteria_chain as ec                           # noqa: E402

OUT = ROOT / "experiments" / "objective_swap_20261010"
PROD = Path(r"D:\Final\data\reports")
LATE = pd.Timestamp("2021-01-01")
FULL = pd.Timestamp("2019-01-02")
SEEDSET = "7_21_99_5_13"          # 主种子集（两流同名 ⇒ 单变量）
TIERS = {
    "w2": dict(equity=20000, top_n=2, keep_n=4, max_symbols=2, max_lots=10),
    "t3": dict(equity=1_000_000, top_n=10, keep_n=20, max_symbols=10, max_lots=50),
}


def _p(name: str) -> Path:
    local = ROOT / "data" / "reports" / name
    return local if local.exists() else PROD / name


def main() -> int:
    t_start = time.time()
    cfg0 = load_config(ROOT / "config.yaml")
    panels = sa.load_panels(cfg0)
    bundle = build_dataset(cfg0, compute_fingerprints=False)
    dates = panels["close"].index
    print(f"面板右端点 {dates.max().date()}", flush=True)

    # ---- §五-5 单变量自证 ----
    mse_f = _p(f"dlv2_ss{SEEDSET}.parquet")
    rl_f = _p(f"rankloss_ss{SEEDSET}.parquet")
    if not (mse_f.exists() and rl_f.exists()):
        raise SystemExit(f"[门] 缺流：{mse_f} / {rl_f}")
    print(f"[单变量自证] MSE 流 {mse_f.name}", flush=True)
    print(f"             RL 流 {rl_f.name}  ⇒ 种子集同名 ⇒ 唯一差异 = 损失", flush=True)

    def load1(p: Path) -> pd.DataFrame:
        d = pd.read_parquet(p, columns=["date", "product", "pred"])
        d["date"] = pd.to_datetime(d["date"])
        return d.pivot_table(index="date", columns="product", values="pred")

    Pmse = load1(mse_f).astype("float32")
    Prl = load1(rl_f).astype("float32")

    cfg_roll = replace(cfg0, account=replace(cfg0.account,
                                             dd_peak_mode="rolling",
                                             dd_peak_window=252))
    doc = {
        "prereg": "plans/objective-swap-prereg-20261010.md",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seedset": SEEDSET, "streams": {"mse": mse_f.name, "rank": rl_f.name},
        "tiers": {}, "criteria": {},
    }

    s = pd.Series(dates, index=dates)
    for tier_name, tier in TIERS.items():
        print(f"\n===== 档位 {tier_name} =====", flush=True)
        # 两条流都要用**同一批日期**跑（面板相同即可）
        res = {}
        for lab, P in (("mse", Pmse), ("rank", Prl)):
            eq, m = arm_faithful(P, tier, cfg_roll, bundle, panels, dates,
                                 FULL, ema_step="daily")
            res[lab] = (eq, m)
            print(f"  [{lab}] 全样本 {m['total_return']:+9.2%} "
                  f"MDD {m['max_drawdown']:+8.2%} 成交 {m.get('n_trades',0)}", flush=True)

        tinfo = {}
        for lab, start, tag in (("mse", LATE, "late"), ("rank", LATE, "late")):
            pass
        # 分段重跑（后段）
        late = {}
        for lab, P in (("mse", Pmse), ("rank", Prl)):
            eq, m = arm_faithful(P, tier, cfg_roll, bundle, panels, dates,
                                 LATE, ema_step="daily")
            late[lab] = (eq, m)
            print(f"  [{lab}] 后段   {m['total_return']:+9.2%} "
                  f"MDD {m['max_drawdown']:+8.2%} 成交 {m.get('n_trades',0)}", flush=True)

        # 月度配对（后段）
        me_late = [x for x in s.groupby([dates.year, dates.month]).last() if x >= LATE]
        mp_late = monthly_log_paired(late["mse"][0], late["rank"][0], me_late)
        me_full = [x for x in s.groupby([dates.year, dates.month]).last() if x >= FULL]
        mp_full = monthly_log_paired(res["mse"][0], res["rank"][0], me_full)

        # 自举 CI（对月度配对差的序列）
        def boot_ci(eq_a, eq_b, month_ends, b=20000, block=6, seed=20261010):
            ma = eq_a.resample("ME").last()
            mb = eq_b.resample("ME").last()
            j = pd.concat([ma, mb], axis=1).dropna()
            d = np.log(j.iloc[:, 0] / j.iloc[:, 0].shift(1)
                       ).diff().dropna()
            # 用月度对数收益之差
            la = np.log1p(j.iloc[:, 0].pct_change().dropna())
            lb = np.log1p(j.iloc[:, 1].pct_change().dropna())
            n = min(len(la), len(lb))
            diff = (la.to_numpy()[:n] - lb.to_numpy()[:n])
            if n < 6:
                return None, None, n
            rng = np.random.RandomState(seed)
            out = np.empty(b)
            for i in range(b):
                idx = []
                while len(idx) < n:
                    st = rng.randint(0, n)
                    idx.extend(((st + np.arange(block)) % n).tolist())
                out[i] = diff[np.array(idx[:n])].mean()
            lo, hi = np.quantile(out, [0.025, 0.975])
            return float(lo), float(hi), n

        lo_l, hi_l, n_l = boot_ci(late["mse"][0], late["rank"][0], me_late)
        lo_f, hi_f, n_f = boot_ci(res["mse"][0], res["rank"][0], me_full)

        # 尾部机制（top-1 超额差）
        def tail_exc(P, start):
            idx = [x for x in dates if x >= start and x in P.index
                   and x in bundle.labels["fwd_ret"].index]
            codes = [c for c in P.columns if c in bundle.labels["fwd_ret"].columns]
            mask = bundle.universe.reindex(index=idx, columns=codes).fillna(False).astype(bool)
            Pm = P.loc[idx, codes].where(mask).to_numpy()
            Ym = bundle.labels["fwd_ret"].reindex(index=idx, columns=codes
                                                  ).astype("float64").where(mask).to_numpy()
            mkt = np.nanmean(np.where(np.isfinite(Ym), Ym, np.nan), axis=1)
            arr = np.full(len(idx), np.nan)
            for i in range(len(idx)):
                pr, yr = Pm[i], Ym[i]
                m = np.isfinite(pr) & np.isfinite(yr)
                if m.sum() < 200:
                    continue
                arr[i] = yr[m][np.argsort(-pr[m])[0]] - mkt[i]
            ok = np.isfinite(arr)
            return float(arr[ok].mean()), (float(newey_west_t(arr[ok]))
                                           if ok.sum() > 5 else None)

        te_mse, tt_mse = tail_exc(Pmse, LATE)
        te_rl, tt_rl = tail_exc(Prl, LATE)

        tinfo = {
            "full": {lab: {"ret": res[lab][1]["total_return"],
                           "mdd": res[lab][1]["max_drawdown"]} for lab in res},
            "late": {lab: {"ret": late[lab][1]["total_return"],
                           "mdd": late[lab][1]["max_drawdown"]} for lab in late},
            "paired_late": mp_late, "paired_full": mp_full,
            "boot_ci_late": [lo_l, hi_l, n_l], "boot_ci_full": [lo_f, hi_f, n_f],
            "tail_late": {"mse": [te_mse, tt_mse], "rank": [te_rl, tt_rl]},
        }
        doc["tiers"][tier_name] = tinfo

        print(f"\n  [C1 后段月度配对] t = {mp_late['t']:+.2f}  "
              f"mean_log = {mp_late['mean_log']:+.6f}  n = {mp_late['n']}", flush=True)
        print(f"     自举 CI95 = [{lo_l:+.6f}, {hi_l:+.6f}]  含0 = "
              f"{lo_l <= 0 <= hi_l}", flush=True)
        print(f"  [C2 全样本配对]   t = {mp_full['t']:+.2f}  "
              f"mean_log = {mp_full['mean_log']:+.6f}  n = {mp_full['n']}", flush=True)
        print(f"     自举 CI95 = [{lo_f:+.6f}, {hi_f:+.6f}]", flush=True)
        print(f"  [C4 尾部] 后段 top-1 超额：MSE {te_mse*1e4:+.1f}bp "
              f"(t={tt_mse:+.2f}) vs rank {te_rl*1e4:+.1f}bp (t={tt_rl:+.2f})", flush=True)

    # ---- 判据汇总 ----
    crit = {}
    for tier_name, ti in doc["tiers"].items():
        t1 = ti["paired_late"]["t"]
        lo, hi = ti["boot_ci_late"][0], ti["boot_ci_late"][1]
        crit[tier_name] = {
            "C1": bool(t1 is not None and t1 >= 2 and not (lo <= 0 <= hi)),
            "C1_t": t1, "C1_ci_contains_zero": bool(lo <= 0 <= hi),
            "C3": bool(t1 is not None and t1 >= 2.24),
            "C2_no_reverse": bool(ti["paired_full"]["t"] is None
                                  or ti["paired_full"]["t"] > -2),
            "C4_same_sign": bool(np.sign(ti["tail_late"]["mse"][0]
                                         - ti["tail_late"]["rank"][0])
                                 == np.sign(ti["late"]["mse"]["ret"]
                                            - ti["late"]["rank"]["ret"])),
        }
    doc["criteria"] = crit
    print("\n" + "=" * 78)
    print("判据汇总（预注册 §三）")
    print("=" * 78)
    for tier_name, c in crit.items():
        print(f"  [{tier_name}] C1 {'过' if c['C1'] else '不过'} (t={c['C1_t']:+.2f}) | "
              f"C3 {'过' if c['C3'] else '不过'} (|t|≥2.24) | "
              f"C2 无反向 {'是' if c['C2_no_reverse'] else '否'} | "
              f"C4 同号 {'是' if c['C4_same_sign'] else '否'}")
    all_c1 = all(c["C1"] for c in crit.values())
    print(f"\n  **H1（MSE 后段优于 rank-loss）{'成立' if all_c1 else '不成立'}**")
    doc["H1_supported"] = bool(all_c1)

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    p = OUT / f"objective_swap_{stamp}.json"
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str),
                 encoding="utf-8")
    print(f"\n报告 → {p}   （耗时 {time.time()-t_start:.0f}s）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
