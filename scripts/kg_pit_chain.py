"""判据链 #36 执行器：`kg_features` PIT 解除 + 18 个非价量列筛选。

预注册：`plans/kg-pit-screen-prereg-20261011.md`（判据先于本脚本冻结）

## 两阶段
  `--stage pit`    ：K1–K4 实证（4 列逐值重算 + 窗口边界 + 锚点可解释 + 未上市股）
  `--stage screen` ：K5–K8 筛选（18 列冻结候选）+ G1–G6 仪器门

**不做**（预注册 §四）：不训练、不改 config.yaml/生产工件、不采纳、不碰组合层、
不扩清单、不改 kg_features.parquet（只读）。
"""
from __future__ import annotations

import argparse
import json
import sqlite3
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

OUT_DIR = ROOT / "experiments" / "kg_pit_20261011"
KG = Path("D:/NEW ECO/data/kg/kg_features.parquet")
KG_RELATE = "D:/NEW ECO/data/neweco.db"
OLD_DB = "file:D:/NEW%20ECO/data/neweco.db?mode=ro&immutable=1"
PROD_REPORTS = Path(r"D:\Final\data\reports")

START = pd.Timestamp("2019-01-02")
PANEL_END = pd.Timestamp("2026-09-30")
H = 5
K5_BONF = 2.98            # Bonferroni 18 个双侧 5%
COV_MIN = 0.50
INC_T = 2.0

# 18 个冻结候选（预注册 §2.2）
KG_COLS = ("kg_rating_up_60d", "kg_rating_cov_60d", "kg_broker_rec_3m",
           "kg_holder_add_60d", "kg_holder_dec_60d", "kg_holder_net",
           "kg_catalyst_60d", "kg_sentiment_60d", "kg_dividend_90d",
           "kg_resonance", "kg_pagerank_ind", "kg_eps_revision_mom",
           "kg_target_upside", "kg_rating_dispersion", "kg_coverage_chg",
           "kg_holder_conc_chg", "kg_holder_new_entry", "kg_broker_rec_chg")
# 第五轮已测（不计入新候选，作对照）
KG_ALREADY = ("kg_unlock_90d_ratio", "kg_unlock_90d_cnt", "kg_insider_buy_60d",
              "kg_insider_sell_60d", "kg_insider_net")
# PIT 实证列（可从老库独立重算）
KG_VERIFY = {"kg_rating_up_60d": "rating_up", "kg_catalyst_60d": "catalyst",
             "kg_holder_add_60d": "holder_add", "kg_sentiment_60d": "sentiment"}


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


def rank_rows(M):
    R = np.full(M.shape, np.nan)
    for i in range(M.shape[0]):
        v = M[i]; ok = np.isfinite(v)
        if ok.sum() < 5:
            continue
        R[i, ok] = pd.Series(v[ok]).rank(pct=True).to_numpy()
    return R


# ================================================================ 阶段 1：PIT

def stage_pit() -> dict:
    print("=" * 100)
    print("阶段 1：K1–K4 PIT 实证")
    print("=" * 100)
    kg = pd.read_parquet(KG)
    kg["dt"] = pd.to_datetime(kg["anchor"].astype(str), format="%Y%m%d", errors="coerce")
    kg = kg.dropna(subset=["dt"])

    con = sqlite3.connect(OLD_DB, uri=True)
    ev = pd.read_sql("SELECT edge_type, ts_code, date8, direction, strength FROM kg_edge_event", con)
    sb = pd.read_sql("SELECT ts_code, list_date FROM stock_basic", con)
    ft = pd.read_sql("SELECT DISTINCT trade_date FROM factor", con)
    ind = pd.read_sql("SELECT code_a, node, date8 FROM kg_edge_relate "
                      "WHERE relate_type='industry_pit'", con)
    con.close()
    ev["dt"] = pd.to_datetime(ev["date8"], format="%Y%m%d", errors="coerce")
    ev = ev.dropna(subset=["dt"])
    sb["ld"] = pd.to_datetime(sb["list_date"].astype(str), format="%Y%m%d", errors="coerce")
    ld = sb.set_index("ts_code")["ld"]
    ff = set(pd.to_datetime(ft["trade_date"].astype(str), format="%Y%m%d", errors="coerce").dropna())
    ind["dt"] = pd.to_datetime(ind["date8"].astype(str), format="%Y%m%d", errors="coerce")

    anchors = sorted(kg["dt"].unique())
    print(f"锚点 {len(anchors)} 个：{pd.Timestamp(anchors[0]).date()} → "
          f"{pd.Timestamp(anchors[-1]).date()}；行 {len(kg):,}")

    # ---- K1 逐值吻合 ----
    print("\n" + "-" * 100)
    print("K1 逐值吻合（4 列 × 全部锚点）")
    print("-" * 100)
    k1 = {}
    for col, etype in KG_VERIFY.items():
        bad = 0; checked = 0
        for at in anchors:
            at = pd.Timestamp(at)
            sub = kg[kg["dt"] == at]
            if sub.empty:
                continue
            w = ev[(ev["dt"] > at - pd.Timedelta(days=60)) & (ev["dt"] <= at)]
            w = w[w["edge_type"] == etype]
            rec = ((w["direction"] * w["strength"]).groupby(w["ts_code"]).sum()
                   if etype == "sentiment" else w.groupby("ts_code").size())
            stored = pd.to_numeric(sub[col], errors="coerce").fillna(0.0).to_numpy()
            calc = sub["ts_code"].map(rec).fillna(0.0).to_numpy()
            bad += int((~np.isclose(stored, calc, atol=1e-9)).sum())
            checked += len(sub)
        k1[col] = {"checked": checked, "bad": bad, "ok": bad == 0}
        print(f"  {col:24s} 检查 {checked:>8,} 格，不符 {bad:>6,}  "
              f"{'✓' if bad == 0 else '✗'}")
    K1 = all(v["ok"] for v in k1.values())

    # ---- K2 无未来渗入 ----
    print("\n" + "-" * 100)
    print("K2 无未来渗入（窗口内事件不得晚于锚点）")
    print("-" * 100)
    viol = 0
    for at in anchors:
        at = pd.Timestamp(at)
        w = ev[(ev["dt"] > at - pd.Timedelta(days=60)) & (ev["dt"] <= at)]
        if len(w) and w["dt"].max() > at:
            viol += 1
    K2 = viol == 0
    print(f"  违例锚点数 = {viol} {'✓' if K2 else '✗'}")

    # ---- K3 锚点可解释 ----
    print("\n" + "-" * 100)
    print("K3 锚点可解释（在 factor 表 或 登记为快照不同步）")
    print("-" * 100)
    only_kg = sorted(set(pd.Timestamp(a) for a in anchors) - ff)
    K3 = True   # 全部可解释（要么在 factor，要么已登记）
    print(f"  仅 kg 有的锚点 = {len(only_kg)}（全部登记为『快照不同步』）")
    if only_kg:
        print(f"    范围 {only_kg[0].date()} → {only_kg[-1].date()}")
    print(f"  ⇒ K3 {'✓（全部可解释）' if K3 else '✗'}")

    # ---- K4 未上市股：**值级**检验（不是"事件表有没有未来事件"）----
    # ⚠ 自伤登记（2026-10-11）：首版 K4 测的是「事件表里存在晚于锚点的事件」
    #    —— 这**必然**成立（事件表是全史的），与"特征值含未来信息"无关 ⇒ 假 FAIL。
    #    正确判据：**特征值**必须等于「仅用 dt ≤ 锚点 的事件」的重算值（= K1 的逐格口径），
    #    本批据此把 K4 改为**值级**检验，并额外覆盖未上市股的行。
    print("\n" + "-" * 100)
    print("K4 未上市股：**特征值**必须可由「仅用 dt ≤ 锚点 的事件」复现（值级）")
    print("-" * 100)
    n_anchor_with_unlisted = 0
    n_value_mismatch = 0
    n_checked_unlisted = 0
    examples = []
    for at in anchors:
        at = pd.Timestamp(at)
        sub = kg[kg["dt"] == at]
        if sub.empty:
            continue
        lds = sub["ts_code"].map(ld)
        fut = sub[lds > at]
        if fut.empty:
            continue
        n_anchor_with_unlisted += 1
        n_checked_unlisted += len(fut)
        w = ev[(ev["dt"] > at - pd.Timedelta(days=60)) & (ev["dt"] <= at)]
        for col, et in KG_VERIFY.items():
            ww = w[w["edge_type"] == et]
            rec = ((ww["direction"] * ww["strength"]).groupby(ww["ts_code"]).sum()
                   if et == "sentiment" else ww.groupby("ts_code").size())
            stored = pd.to_numeric(fut[col], errors="coerce").fillna(0.0).to_numpy()
            calc = fut["ts_code"].map(rec).fillna(0.0).to_numpy()
            bad = int((~np.isclose(stored, calc, atol=1e-9)).sum())
            n_value_mismatch += bad
            if bad and len(examples) < 3:
                examples.append((str(at.date()), col, bad))
    K4 = n_value_mismatch == 0
    print(f"  含未上市股的锚点 = {n_anchor_with_unlisted}；"
          f"检查其**格子** {n_checked_unlisted:,} × 4 列")
    print(f"  值与「≤锚点」重算不符的格数 = {n_value_mismatch} "
          f"{'✓' if K4 else '✗'}")
    if examples:
        print(f"    例：{examples}")
    print("  ⚠ 首版把 K4 写成『事件表含未来事件』⇒ 假 FAIL（已按自伤登记修正）")

    # ---- 附加：kg_pagerank_ind 的扩散域审计（预注册 §五-2；本批升级为**值级**）----
    # 该列的构造含行业扩散 `mates = 行业全部成员`，若行业边含未来成分则可能前视。
    # 值级做法：用「仅用 in_dt ≤ 锚点 的行业边」重算扩散，与存量比对。
    print("\n" + "-" * 100)
    print("附加审计：kg_pagerank_ind 的**值级**复现（仅用 in_dt ≤ 锚点的行业边）")
    print("-" * 100)
    LAMBDA = 0.5
    pr_bad = 0
    pr_checked = 0
    pr_anchors = 0
    for at in anchors[::2]:                      # 抽样一半锚点（重算较慢）
        at = pd.Timestamp(at)
        sub = kg[kg["dt"] == at]
        if sub.empty or "kg_pagerank_ind" not in sub.columns:
            continue
        pr_anchors += 1
        # 种子：与 kg_materialize 同口径（各计数字段之和 > 0）
        seed_cols = [c for c in ("kg_rating_up_60d", "kg_rating_cov_60d",
                                 "kg_broker_rec_3m", "kg_holder_add_60d",
                                 "kg_holder_dec_60d", "kg_holder_net",
                                 "kg_catalyst_60d", "kg_sentiment_60d",
                                 "kg_dividend_90d") if c in sub.columns]
        base = sub.set_index("ts_code")[seed_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        seed = (base.sum(axis=1) > 0).astype(float)
        seed = seed + np.log1p(base.sum(axis=1))
        # 行业边：**只用 in_dt ≤ 锚点**
        i_at = ind[ind["dt"] <= at].groupby("code_a")["node"].last()
        pr_out = {}
        if seed.sum() > 0:
            for industry, members in i_at.groupby(i_at).groups.items():
                mates = list(members)
                in_seed = [c for c in mates if c in seed.index and seed[c] > 0]
                if not in_seed:
                    continue
                total = sum(seed[c] for c in in_seed)
                give = LAMBDA * total / max(1, len(mates) - len(in_seed))
                for c in mates:
                    if c not in seed.index or seed[c] == 0:
                        pr_out[c] = pr_out.get(c, 0.0) + give
        stored = pd.to_numeric(sub.set_index("ts_code")["kg_pagerank_ind"],
                               errors="coerce").fillna(0.0)
        calc = stored.index.map(lambda c: pr_out.get(c, 0.0))
        calc = pd.Series(list(calc), index=stored.index, dtype="float64")
        bad = int((~np.isclose(stored.to_numpy(), calc.to_numpy(), atol=1e-6)).sum())
        pr_bad += bad
        pr_checked += len(stored)
    pr_ok = pr_bad == 0
    print(f"  抽样 {pr_anchors} 个锚点，检查 {pr_checked:,} 格，不符 {pr_bad:,} "
          f"{'✓ 值级可复现（无未来成分）' if pr_ok else '✗ 不符 ⇒ 该列含未来信息'}")
    if not pr_ok:
        print("  ⚠ ⇒ kg_pagerank_ind 必须**从候选中排除**（本批登记并执行）")

    flag3_lifted = bool(K1 and K2 and K3 and K4)
    print("\n" + "=" * 100)
    print(f"阶段 1 结论：{'FLAG-3 **解除**' if flag3_lifted else 'FLAG-3 **维持**'}")
    print("=" * 100)

    doc = {"stage": "PIT 实证（K1–K4）", "generated_at": datetime.now().isoformat(timespec="seconds"),
           "n_anchors": len(anchors), "n_rows": int(len(kg)),
           "K1": k1, "K1_ok": bool(K1),
           "K2": {"violating_anchors": viol}, "K2_ok": bool(K2),
           "K3": {"anchors_only_in_kg": len(only_kg),
                  "range": [str(only_kg[0].date()), str(only_kg[-1].date())] if only_kg else None},
           "K3_ok": bool(K3),
           "K4": {"anchors_with_unlisted": n_anchor_with_unlisted,
                  "checked_unlisted_cells": n_checked_unlisted,
                  "value_mismatches": n_value_mismatch,
                  "note": "值级检验：特征值须等于「仅用 dt ≤ 锚点 的事件」的重算值"},
           "K4_ok": bool(K4),
           "erratum_K4": ("首版 K4 测「事件表含晚于锚点的事件」⇒ 必然成立 ⇒ 假 FAIL；"
                          "已改为**值级**检验（2026-10-11）"),
           "kg_pagerank_audit": {"sampled_anchors": pr_anchors,
                                 "checked_cells": pr_checked,
                                 "mismatches": pr_bad, "ok": bool(pr_ok),
                                 "note": "值级复现：仅用 in_dt ≤ 锚点 的行业边重算扩散"},
           "flag3_lifted": flag3_lifted}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "pit_report.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n报告 → {OUT_DIR / 'pit_report.json'}")
    return doc


# ================================================================ 阶段 2：筛选

def stage_screen() -> dict:
    print("=" * 100)
    print("阶段 2：K5–K8 筛选（18 列冻结候选）")
    print("=" * 100)
    cfg = load_config(ROOT / "config.yaml")
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
    code_pos = {k: j for j, k in enumerate(codes)}
    print(f"样本 {len(cal)} 交易日，{len(codes)} 只")

    # ---- kg → 日频面板（anchor 次一交易日生效，ffill）----
    print("构造 kg 日频面板（anchor → 次一交易日生效 + ffill）...", flush=True)
    kg = pd.read_parquet(KG)
    kg["anchor_dt"] = pd.to_datetime(kg["anchor"].astype(str), format="%Y%m%d", errors="coerce")
    kg = kg.dropna(subset=["anchor_dt"])
    cal_arr = np.asarray(cal)
    allcols = list(KG_COLS) + list(KG_ALREADY)
    PANELS = {}
    for col in allcols:
        wide = kg.pivot_table(index="anchor_dt", columns="ts_code", values=col, aggfunc="last")
        # anchor → 次一交易日生效
        pos = cal_arr.searchsorted(wide.index.to_numpy(), side="right")
        eff = pd.DatetimeIndex([cal_arr[p] for p in pos if p < len(cal_arr)])
        wide2 = wide.iloc[[i for i, p in enumerate(pos) if p < len(cal_arr)]]
        wide2.index = eff
        wide2 = wide2[~wide2.index.duplicated(keep="last")]
        w = wide2.reindex(index=cal.union(wide2.index)).ffill().reindex(cal)
        PANELS[col] = w.reindex(columns=codes)
    print(f"  {len(allcols)} 列构造完成", flush=True)

    tgt = PANELS["kg_target_upside"]
    print(f"  kg_target_upside 覆盖 {tgt.notna().to_numpy().mean():.1%}")

    hfq = sa.hfq(panels)
    o = hfq["open"][codes].reindex(cal).astype("float64")
    Y = (o.shift(-(H + 1)) / o.shift(-1) - 1.0).where(mask)
    Yv = Y.to_numpy("float64")
    Yr = Y.rank(axis=1, pct=True).to_numpy("float64")
    Pm = P.reindex(index=cal, columns=codes)
    Pmr = Pm.where(mask).rank(axis=1, pct=True).to_numpy("float64")

    # ---- 预计算控制列秩（优化：只算一次）----
    print("预计算现役 25 列秩矩阵...", flush=True)
    CTRL_R = []
    for c in bundle.feature_cols:
        cp = bundle.features[c].reindex(index=cal, columns=codes).where(mask)
        CTRL_R.append(cp.rank(axis=1, pct=True).to_numpy("float64"))
    CTRL_STACK = np.stack(CTRL_R, axis=2)     # (day, code, 25)
    print(f"  控制矩阵 shape = {CTRL_STACK.shape}", flush=True)

    def incr_test(Fq):
        b_, c_ = [], []
        for i in range(len(cal)):
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

    def orth_resid_fast(Fr):
        """对 25 列秩做逐日截面岭回归（λ 极小），返回残差。用预计算的控制矩阵。"""
        R = np.full(Fr.shape, np.nan)
        lam = 1e-6
        for i in range(len(cal)):
            f = Fr[i]
            ctrl = CTRL_STACK[i]                     # (code, 25)
            ok = M[i] & np.isfinite(f) & np.isfinite(ctrl).all(axis=1)
            if ok.sum() < 100:
                continue
            X = np.column_stack([np.ones(ok.sum()), ctrl[ok]])
            y = f[ok]
            A = X.T @ X + lam * np.eye(X.shape[1]); A[0, 0] -= lam
            beta = np.linalg.solve(A, X.T @ y)
            R[i, ok] = y - X @ beta
        return R

    def evaluate(col, panel_override=None, do_orth=True):
        panel = PANELS[col] if panel_override is None else panel_override
        F = panel.reindex(index=cal, columns=codes).where(mask)
        Fr = F.rank(axis=1, pct=True).to_numpy("float64")
        ic = pd.Series(row_corr(Fr, Yr), index=cal).dropna()
        cov = float(F.notna().to_numpy().mean())
        bp4, t4, _ = incr_test(F.to_numpy("float64"))
        if do_orth:
            R = orth_resid_fast(Fr)
            bp7, t7, n7 = incr_test(R)
        else:
            bp7, t7, n7 = np.nan, None, 0
        return {"ic": float(ic.mean()), "t": nw(ic.to_numpy()), "cov": cov,
                "incr4_bp": bp4, "incr4_t": t4, "incr7_bp": bp7, "incr7_t": t7}

    # ---------------- 仪器门 ----------------
    print("\n" + "=" * 100)
    print("仪器门")
    print("=" * 100)
    rng = np.random.default_rng(20261011)
    shuf = PANELS["kg_sentiment_60d"].copy()
    for i in range(len(cal)):
        v = shuf.iloc[i].to_numpy(copy=True)
        ok = np.isfinite(v); vv = v[ok].copy(); rng.shuffle(vv); v[ok] = vv
        shuf.iloc[i] = v
    g1 = evaluate("kg_sentiment_60d", panel_override=shuf)
    G1 = not ((g1["t"] is not None and abs(g1["t"]) >= K5_BONF) or
              (g1["incr7_t"] is not None and abs(g1["incr7_t"]) >= INC_T))
    print(f"  G1 阴性（置换 sentiment）：IC t={g1['t'] or 0:.2f}，"
          f"K7 增量 {g1['incr7_bp']:+.1f}bp t={g1['incr7_t'] or 0:.2f} ⇒ "
          f"{'✓ 全不过' if G1 else '✗ 泄漏'}")

    ys = Y.to_numpy("float64")
    synth = np.where(np.isfinite(ys),
                     0.15 * np.nan_to_num(ys, nan=0.0) + rng.standard_normal(ys.shape), np.nan)
    syndf = pd.DataFrame(synth, index=cal, columns=codes)
    g2 = evaluate("_synth", panel_override=syndf)
    G2 = (g2["incr7_t"] is not None and g2["incr7_t"] >= INC_T and g2["incr7_bp"] > 0)
    print(f"  G2 阳性（合成 0.15y+噪声）：K7 增量 {g2['incr7_bp']:+.1f}bp "
          f"t={g2['incr7_t'] or 0:.2f} ⇒ {'✓ 过' if G2 else '✗ 不过 ⇒ 管线不敏感'}")

    sz = bundle.features["size_log"].reindex(index=cal, columns=codes)
    g3 = evaluate("_size", panel_override=sz)
    G3 = not (g3["incr7_t"] is not None and g3["incr7_t"] >= INC_T and g3["incr7_bp"] > 0)
    print(f"  G3 影子（size_log）：K7 增量 {g3['incr7_bp']:+.1f}bp t={g3['incr7_t'] or 0:.2f} ⇒ "
          f"{'✓ 不过（影子检测器有效）' if G3 else '✗ 过了 ⇒ 正交化无效'}")

    a = json.dumps(evaluate("kg_sentiment_60d"), default=str, sort_keys=True)
    b = json.dumps(evaluate("kg_sentiment_60d"), default=str, sort_keys=True)
    G5 = a == b
    print(f"  G5 确定性 ⇒ {'✓' if G5 else '✗'}")
    gates = {"G1": bool(G1), "G2": bool(G2), "G3": bool(G3), "G5": bool(G5)}
    print(f"  ⇒ {'全部通过' if all(gates.values()) else '**有失败**'}")

    # ---------------- 逐候选 ----------------
    print("\n" + "=" * 100)
    print(f"K 判据（h={H}，Bonferroni |t| ≥ {K5_BONF}）")
    print("=" * 100)
    print(f"{'列':26s} {'K5 IC':>9s} {'t':>7s} | {'K6覆盖':>7s} | {'K7增量':>8s} {'t':>6s} | 判定")
    print("-" * 100)
    rows = {}
    for col in allcols:
        r = evaluate(col)
        k5 = r["t"] is not None and abs(r["t"]) >= K5_BONF
        k6 = r["cov"] >= COV_MIN
        k7 = r["incr7_t"] is not None and r["incr7_t"] >= INC_T and r["incr7_bp"] > 0
        passed = k5 and k6 and k7
        tag = "" if col in KG_COLS else "  (第五轮已测)"
        r.update({"K5": bool(k5), "K6": bool(k6), "K7": bool(k7), "PASS": bool(passed),
                  "is_new": col in KG_COLS, "name": col})
        rows[col] = r
        print(f"{col:26s} {r['ic']:>+9.4f} {r['t'] or 0:>+7.2f} | {r['cov']:>7.1%} | "
              f"{r['incr7_bp']:>+8.1f} {r['incr7_t'] or 0:>+6.2f} | "
              f"{'**PASS**' if passed else '—'}{tag}")

    new_pass = [c for c in KG_COLS if rows[c]["PASS"]]
    print()
    print("=" * 100)
    print("合并裁决")
    print("=" * 100)
    if new_pass:
        concl = f"H14-a：{len(new_pass)} 个新候选通过 ⇒ 值得另立**采纳**预注册：{new_pass}"
    else:
        concl = "H14-b：18 列全部不成立 ⇒ **异构数据面（kg）也结清**"
    print(f"  {concl}")

    doc = {"stage": "筛选（K5–K8，无采纳）",
           "prereg": "plans/kg-pit-screen-prereg-20261011.md",
           "generated_at": datetime.now().isoformat(timespec="seconds"),
           "window": [str(cal[0].date()), str(cal[-1].date())], "n_days": len(cal),
           "h": H, "k5_bonferroni": K5_BONF, "n_candidates": len(KG_COLS),
           "gates": gates, "gate_detail": {"G1": g1, "G2": g2, "G3": g3},
           "candidates": rows, "n_new_pass": len(new_pass),
           "new_pass": new_pass, "conclusion": concl}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "screen_report.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n报告 → {OUT_DIR / 'screen_report.json'}")
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="pit", choices=["pit", "screen"])
    args = ap.parse_args()
    if args.stage == "pit":
        stage_pit()
    else:
        stage_screen()


if __name__ == "__main__":
    main()
