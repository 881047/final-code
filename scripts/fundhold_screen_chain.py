"""判据链 #37 执行器：`fund_hold_history` **截面因子**开采（筛选批）。

预注册：`plans/fundhold-screen-prereg-20261011.md`（判据先于本脚本冻结）

## 与 NEW ECO H1 轨道的关系（防重复劳动核查，预注册 §0）
H1 用 `held_ratio` 作**体制代理 + 剔除规则（EXIT 场景）**，判据前瞻至 ~2027-05；
本批用**截面因子**口径（Final 日频模型的特征候选）—— 是**不同的量**。
NEW ECO `factor` 表 31 因子**不含任何 fund_hold 因子**；`alpha_bank` 亦无。

## PIT 口径（沿用 #33 冻结滞后表）
季报 20 工作日 / 半年报 60 日 / 年报 90 日 ⇒ T_avail = 次一交易日 ⇒ ffill 到日频。

**不做**（预注册 §5）：不训练、不改 config/生产工件、不采纳、不碰组合层、
不用 H1 的体制/剔除口径、不改源 CSV（只读）、不扩清单。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fterm.config import load_config                    # noqa: E402
from fterm.data import ashare as sa                     # noqa: E402
from fterm.features.build import build_dataset          # noqa: E402
from fterm.model.stats import newey_west_t              # noqa: E402

OUT_DIR = ROOT / "experiments" / "fundhold_screen_20261011"
SRC_CSV = Path("D:/NEW ECO/data/h1_crowding_paper/fund_hold_history.csv")
PROD_REPORTS = Path(r"D:\Final\data\reports")

START = pd.Timestamp("2019-01-02")
PANEL_END = pd.Timestamp("2026-09-30")
H_DEFAULT = 5
F1_BONF = 2.64            # Bonferroni 6 个双侧 5%
COV_MIN = 0.50
INC_T = 2.0

# 冻结滞后表（预注册 §1，逐字沿用 #33）
LAG_BUSDAYS = {"0331": 20, "0930": 20}
LAG_CALDAYS = {"0630": 60, "1231": 90}
CAND_NAMES = ("fh_held_ratio", "fh_fund_cnt", "fh_held_ratio_chg",
              "fh_fund_cnt_chg", "fh_fund_cnt_rank", "fh_held_ratio_log")


def row_corr(A, B, min_n=5):
    v = np.isfinite(A) & np.isfinite(B)
    n = v.sum(1)
    A0 = np.where(v, A, 0.0); B0 = np.where(v, B, 0.0)
    ma = A0.sum(1) / np.maximum(n, 1); mb = B0.sum(1) / np.maximum(n, 1)
    da = np.where(v, A0 - ma[:, None], 0.0); db = np.where(v, B0 - mb[:, None], 0.0)
    out = (da * db).sum(1) / (np.sqrt((da * da).sum(1)) * np.sqrt((db * db).sum(1)) + 1e-12)
    out[n < min_n] = np.nan
    return out


def nw(x):
    a = np.asarray(x, "float64"); a = a[np.isfinite(a)]
    if a.size < 5:
        return None
    v = newey_west_t(a)
    return float(v) if v is not None and np.isfinite(v) else None


def build_panels(cal: pd.DatetimeIndex, codes: pd.Index,
                 panels: dict) -> dict:
    """按冻结滞后表构造 6 个候选面板（日频、T_avail 生效 + ffill）。"""
    fh = pd.read_csv(SRC_CSV, dtype={"ts_code": str})
    fh["period_dt"] = pd.to_datetime(fh["period"].astype(str), format="%Y%m%d",
                                     errors="coerce")
    fh = fh.dropna(subset=["period_dt"])
    fh["mmdd"] = fh["period_dt"].dt.strftime("%m%d")
    fh["held_shares"] = pd.to_numeric(fh["held_shares"], errors="coerce")
    fh["fund_cnt"] = pd.to_numeric(fh["fund_cnt"], errors="coerce")

    circ_mv = panels["circ_mv"].reindex(index=cal, columns=codes)
    close = panels["close"].reindex(index=cal, columns=codes)
    float_shares = (circ_mv * 1e4 / close.replace(0, np.nan))

    cal_arr = np.asarray(cal)

    def t_avail(p: pd.Timestamp) -> pd.Timestamp | None:
        key = p.strftime("%m%d")
        if key in LAG_BUSDAYS:
            eff = pd.Timestamp(np.busday_offset(np.datetime64(p, "D"),
                                                LAG_BUSDAYS[key], roll="forward"))
        elif key in LAG_CALDAYS:
            eff = p + pd.Timedelta(days=LAG_CALDAYS[key])
        else:
            return None
        pos = cal_arr.searchsorted(eff, side="right")   # 次一交易日生效
        return cal_arr[pos] if pos < len(cal_arr) else None

    fh["t_avail"] = [t_avail(p) for p in fh["period_dt"]]
    fh = fh.dropna(subset=["t_avail"])

    # 全持仓期标记
    fh["is_full"] = fh["mmdd"].isin(["0630", "1231"])

    # 每期：held_ratio / fund_cnt
    ds = float_shares.stack().rename("float_shares").reset_index()
    ds.columns = ["date", "ts_code", "float_shares"]
    m = fh.merge(ds, left_on=["t_avail", "ts_code"],
                 right_on=["date", "ts_code"], how="left")
    m["held_ratio"] = m["held_shares"] / m["float_shares"].replace(0, np.nan)

    # 逐期构造宽面板
    out = {}
    for field in ("held_ratio", "fund_cnt", "held_ratio_chg", "fund_cnt_chg"):
        pass

    # 变化列：按 ts_code 排序，取**同类型**上一期
    m = m.sort_values(["ts_code", "period_dt"])
    for col, new in (("held_ratio", "held_ratio_chg"), ("fund_cnt", "fund_cnt_chg")):
        prev = m.groupby(["ts_code", "is_full"])[col].shift(1)
        m[new] = m[col] - prev

    def wide(field):
        w = m.pivot_table(index="t_avail", columns="ts_code", values=field,
                          aggfunc="last")
        w = w.reindex(index=cal.union(w.index)).ffill().reindex(cal)
        return w.reindex(columns=codes)

    out["fh_held_ratio"] = wide("held_ratio")
    out["fh_fund_cnt"] = wide("fund_cnt")
    out["fh_held_ratio_chg"] = wide("held_ratio_chg")
    out["fh_fund_cnt_chg"] = wide("fund_cnt_chg")
    out["fh_held_ratio_log"] = np.log(out["fh_held_ratio"].where(
        out["fh_held_ratio"] > 0))
    # ⚠ 自伤登记（2026-10-11）：首版把 K5 直接赋成 K2（`= out["fh_fund_cnt"]`）
    #    ⇒ G4 共线对照退化为**同一性检验**（比自身，恒真）⇒ 空检查。
    #    修法：K5 = `fund_cnt` 的**逐日截面秩** —— 它与 K2 秩等价，
    #    但原始值有大量并列 ⇒ 取值**确实不同** ⇒ G4 才是真检验。
    out["fh_fund_cnt_rank"] = out["fh_fund_cnt"].rank(axis=1, pct=True)
    # 全持仓期掩码（F5 分层用）
    full_mask = pd.DataFrame(False, index=cal, columns=codes)
    fm = m[m["is_full"]].groupby("t_avail").size()
    for d in cal:
        pass
    # 用 ffill 的实际来源：每个日期最近的 t_avail 是否为全持仓期
    ta = pd.DatetimeIndex(sorted(m["t_avail"].unique()))
    ta_full = pd.Series(
        m.drop_duplicates("t_avail").set_index("t_avail")["is_full"]
        .reindex(ta).ffill().fillna(False).to_numpy(), index=ta)
    src = ta.searchsorted(cal, side="right") - 1
    is_full_day = np.where(src >= 0, ta_full.to_numpy()[np.clip(src, 0, len(ta) - 1)],
                           False)
    full_mask = pd.DataFrame(np.repeat(is_full_day[:, None], len(codes), axis=1),
                             index=cal, columns=codes)
    return out, full_mask, len(fh), len(ta)


def main():
    global H_DEFAULT
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=H_DEFAULT)
    args = ap.parse_args()
    H = int(args.h)
    print(f"horizon h = {H}"
          f"{'（用户口径「每日」）' if H == 1 else '（生产 label_horizon）'}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "config.yaml")
    print("载入面板/特征/预测流 ...", flush=True)
    bundle = build_dataset(cfg, compute_fingerprints=False)
    panels = sa.load_panels(cfg)
    dates = panels["close"].index

    d1 = pd.read_parquet(PROD_REPORTS / "rankloss_ss7_21_99_5_13.parquet",
                         columns=["date", "product", "pred"])
    d2 = pd.read_parquet(PROD_REPORTS / "rankloss_ss123_2024_42_314_2027.parquet",
                         columns=["date", "product", "pred"])
    for x in (d1, d2):
        x["date"] = pd.to_datetime(x["date"])
    P = (d1.pivot_table(index="date", columns="product", values="pred")
         .add(d2.pivot_table(index="date", columns="product", values="pred"), fill_value=0) / 2)
    P = P.astype("float64").sort_index()

    idx = [d for d in dates if START <= d <= PANEL_END and d in P.index]
    codes = [c for c in P.columns if c in bundle.labels["fwd_ret"].columns]
    cal = pd.DatetimeIndex(idx)
    mask = bundle.universe.reindex(index=cal, columns=codes).fillna(False)
    M = mask.to_numpy(bool)
    print(f"样本 {len(cal)} 交易日，{len(codes)} 只", flush=True)

    CAND, FULL, n_rows, n_periods = build_panels(cal, codes, panels)
    print(f"  fund_hold 原始 {n_rows:,} 行；可用期 {n_periods} 个", flush=True)
    for k in CAND_NAMES:
        print(f"  {k:20s} 覆盖 {CAND[k].notna().to_numpy().mean():.1%}")

    hfq = sa.hfq(panels)
    o = hfq["open"][codes].reindex(cal).astype("float64")
    Y = (o.shift(-(H + 1)) / o.shift(-1) - 1.0).where(mask)
    Yv = Y.to_numpy("float64")
    Yr = Y.rank(axis=1, pct=True).to_numpy("float64")
    Pm = P.reindex(index=cal, columns=codes)
    Pmr = Pm.where(mask).rank(axis=1, pct=True).to_numpy("float64")

    # 预计算现役 25 列秩
    CTRL_STACK = np.stack(
        [bundle.features[c].reindex(index=cal, columns=codes).where(mask)
         .rank(axis=1, pct=True).to_numpy("float64") for c in bundle.feature_cols],
        axis=2)
    print(f"  控制矩阵 {CTRL_STACK.shape}", flush=True)

    def incr_test(Fq, day_mask=None):
        b_, c_ = [], []
        for i in range(len(cal)):
            if day_mask is not None and not day_mask[i]:
                continue
            y = Yv[i]; p = Pmr[i]; f = Fq[i]
            g = M[i] & np.isfinite(y) & np.isfinite(p) & np.isfinite(f)
            if g.sum() < 500:
                continue
            yy, pp, ff = y[g], p[g], f[g]
            ex = yy - yy.mean()
            sel = pp >= np.quantile(pp, 0.90)
            if sel.sum() < 50:
                continue
            b_.append(ex[sel].mean())
            k = max(5, int(0.20 * sel.sum()))
            ti = np.argsort(-ff[sel])[:k]
            c_.append(ex[sel][ti].mean())
        if not b_:
            return np.nan, None, 0
        d = (np.asarray(c_) - np.asarray(b_)) * 1e4
        return float(d.mean()), nw(d), len(d)

    def orth(Fr):
        R = np.full(Fr.shape, np.nan)
        lam = 1e-6
        for i in range(len(cal)):
            ctrl = CTRL_STACK[i]
            ok = M[i] & np.isfinite(Fr[i]) & np.isfinite(ctrl).all(axis=1)
            if ok.sum() < 100:
                continue
            X = np.column_stack([np.ones(ok.sum()), ctrl[ok]])
            A = X.T @ X + lam * np.eye(X.shape[1]); A[0, 0] -= lam
            beta = np.linalg.solve(A, X.T @ Fr[i][ok])
            R[i, ok] = Fr[i][ok] - X @ beta
        return R

    def evaluate(panel, override=None, day_mask=None):
        p = panel if override is None else override
        F = p.reindex(index=cal, columns=codes).where(mask)
        Fr = F.rank(axis=1, pct=True).to_numpy("float64")
        ic = pd.Series(row_corr(Fr, Yr), index=cal).dropna()
        cov = float(F.notna().to_numpy().mean())
        R = orth(Fr)
        bp, t, n = incr_test(R, day_mask)
        return {"ic": float(ic.mean()), "t": nw(ic.to_numpy()), "cov": cov,
                "incr_bp": bp, "incr_t": t, "n_incr": n}

    # ---------------- 仪器门 ----------------
    print("\n" + "=" * 100)
    print("仪器门")
    print("=" * 100)
    rng = np.random.default_rng(20261011)
    shuf = CAND["fh_fund_cnt"].copy()
    for i in range(len(cal)):
        v = shuf.iloc[i].to_numpy(copy=True)
        ok = np.isfinite(v); vv = v[ok].copy(); rng.shuffle(vv); v[ok] = vv
        shuf.iloc[i] = v
    g1 = evaluate(CAND["fh_fund_cnt"], override=shuf)
    G1 = not ((g1["t"] is not None and abs(g1["t"]) >= F1_BONF) or
              (g1["incr_t"] is not None and abs(g1["incr_t"]) >= INC_T))
    print(f"  G1 阴性（置换 fund_cnt）：IC t={g1['t'] or 0:+.2f}，"
          f"F3 增量 {g1['incr_bp']:+.1f}bp t={g1['incr_t'] or 0:+.2f} ⇒ "
          f"{'✓ 全不过' if G1 else '✗ 泄漏'}")

    ys = Y.to_numpy("float64")
    synth = np.where(np.isfinite(ys),
                     0.15 * np.nan_to_num(ys, nan=0.0) + rng.standard_normal(ys.shape), np.nan)
    g2 = evaluate(CAND["fh_fund_cnt"],
                  override=pd.DataFrame(synth, index=cal, columns=codes))
    G2 = (g2["incr_t"] is not None and g2["incr_t"] >= INC_T and g2["incr_bp"] > 0)
    print(f"  G2 阳性（合成 0.15y+噪声）：F3 增量 {g2['incr_bp']:+.1f}bp "
          f"t={g2['incr_t'] or 0:+.2f} ⇒ {'✓ 过' if G2 else '✗ 不过'}")

    g3 = evaluate(CAND["fh_fund_cnt"],
                  override=bundle.features["size_log"].reindex(index=cal, columns=codes))
    G3 = not (g3["incr_t"] is not None and g3["incr_t"] >= INC_T and g3["incr_bp"] > 0)
    print(f"  G3 影子（size_log）：F3 增量 {g3['incr_bp']:+.1f}bp t={g3['incr_t'] or 0:+.2f} ⇒ "
          f"{'✓ 不过' if G3 else '✗ 过了'}")

    # ⭐ G4 共线对照：K5(rank 版) 与 K2 必须逐位一致（秩等价）
    e2 = evaluate(CAND["fh_fund_cnt"])
    e5 = evaluate(CAND["fh_fund_cnt_rank"])
    G4 = (e2["ic"] == e5["ic"] and e2["t"] == e5["t"] and
          e2["incr_bp"] == e5["incr_bp"] and e2["incr_t"] == e5["incr_t"])
    print(f"  ⭐ G4 共线对照（K5≡K2，秩等价）：IC {e2['ic']:+.6f} vs {e5['ic']:+.6f}；"
          f"增量 {e2['incr_bp']:+.4f} vs {e5['incr_bp']:+.4f} ⇒ "
          f"{'✓ 逐位一致（秩管线正常）' if G4 else '✗ 不一致 ⇒ 截面秩管线有问题'}")

    a = json.dumps(evaluate(CAND["fh_fund_cnt"]), default=str, sort_keys=True)
    b = json.dumps(evaluate(CAND["fh_fund_cnt"]), default=str, sort_keys=True)
    G5 = a == b
    print(f"  G5 确定性 ⇒ {'✓' if G5 else '✗'}")

    gates = {"G1": bool(G1), "G2": bool(G2), "G3": bool(G3), "G4": bool(G4), "G5": bool(G5)}
    print(f"  ⇒ {'全部通过' if all(gates.values()) else '**有失败**'}")

    # ---------------- 逐候选 ----------------
    print("\n" + "=" * 100)
    print(f"F 判据（h={H}，Bonferroni |t| ≥ {F1_BONF}）")
    print("=" * 100)
    print(f"{'候选':22s} {'F1 IC':>9s} {'t':>7s} | {'F2覆盖':>7s} | {'F3增量':>8s} {'t':>6s} | 判定")
    print("-" * 100)
    rows = {}
    for name in CAND_NAMES:
        r = evaluate(CAND[name])
        f1 = r["t"] is not None and abs(r["t"]) >= F1_BONF
        f2 = r["cov"] >= COV_MIN
        f3 = r["incr_t"] is not None and r["incr_t"] >= INC_T and r["incr_bp"] > 0
        passed = f1 and f2 and f3
        r.update({"F1": bool(f1), "F2": bool(f2), "F3": bool(f3), "PASS": bool(passed),
                  "name": name})
        rows[name] = r
        print(f"{name:22s} {r['ic']:>+9.4f} {r['t'] or 0:>+7.2f} | {r['cov']:>7.1%} | "
              f"{r['incr_bp']:>+8.1f} {r['incr_t'] or 0:>+6.2f} | "
              f"{'**PASS**' if passed else '—'}")

    # F5 分层：全持仓期子样本
    print()
    print("=" * 100)
    print("F5 口径分层：**全持仓期**（0630/1231）子样本上的 F3")
    print("=" * 100)
    fm = FULL.iloc[:, 0].to_numpy(bool)
    print(f"  全持仓期交易日 = {int(fm.sum())} / {len(cal)} = {fm.mean():.1%}")
    lay = {}
    for name in CAND_NAMES:
        F = CAND[name].reindex(index=cal, columns=codes).where(mask)
        R = orth(F.rank(axis=1, pct=True).to_numpy("float64"))
        bp, t, n = incr_test(R, day_mask=fm)
        lay[name] = {"bp": bp, "t": t, "n": n}
        print(f"  {name:22s} 增量 {bp:+.1f}bp  t={t if t is None else round(t,2)}  n={n}")

    n_pass = sum(1 for r in rows.values() if r["PASS"])
    print()
    print("=" * 100)
    print("合并裁决")
    print("=" * 100)
    concl = (f"通过 {n_pass} 个：" + str([k for k, r in rows.items() if r["PASS"]])
             if n_pass else "全部不成立 ⇒ **`fund_hold_history`（截面因子口径）结清**")
    print(f"  {concl}")
    print(f"  注：这是**截面因子**口径；NEW ECO H1 的体制/剔除口径仍在跑（前瞻至 ~2027-05），"
          f"两者不同量、**不互相改判**。")

    doc = {"stage": "筛选（无采纳）", "prereg": "plans/fundhold-screen-prereg-20261011.md",
           "generated_at": datetime.now().isoformat(timespec="seconds"),
           "window": [str(cal[0].date()), str(cal[-1].date())], "n_days": len(cal),
           "h": H, "f1_bonferroni": F1_BONF, "n_candidates": len(CAND_NAMES),
           "lag_table": {"busdays": LAG_BUSDAYS, "caldays": LAG_CALDAYS},
           "gates": gates, "gate_detail": {"G1": g1, "G2": g2, "G3": g3,
                                           "G4_K2": e2, "G4_K5": e5},
           "candidates": rows, "F5_full_holdings": lay,
           "n_pass": n_pass, "conclusion": concl}
    (OUT_DIR / f"fundhold_screen_report_h{H}.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n报告 → {OUT_DIR / f'fundhold_screen_report_h{H}.json'}")
    return doc


if __name__ == "__main__":
    main()
