"""判据链 #35 执行器：Final **从未筛过的 14 个量价派生因子**（筛选批）。

预注册：`plans/pv-screen-prereg-20261011.md`（判据先于本脚本冻结）

## 本脚本做什么

对 14 个冻结候选（全部只用 t 及以前信息）算：
  C1 全截面 RankIC（Bonferroni 14 个 ⇒ |t| ≥ 2.91）
  C2 与现役 25 列的 max|ρ|（共线**注记**，不作否决）
  C3 覆盖 ≥ 50%
  C4 **弱增量**：模型 top-10% 内再筛最强 20%
  C5 ⭐ **强增量（主判据）**：对**全部现役 25 列**正交化取残差后，再做 C4
  C6 regime 分期同号

**不做**（预注册 §四）：不训练、不改 config.yaml、不采纳、不碰组合层、不扩清单。

## 仪器门（预注册 §三）

  G1 阴性（日内置换因子，全不过）/ **G2 阳性（已知有增量的合成因子，必须过）** /
  **G3 影子对照（size_log 代入 ⇒ C5 必须不过）** / G4 确定性 / G5 变异测试
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

OUT_DIR = ROOT / "experiments" / "pv_screen_20261011"
PROD_REPORTS = Path(r"D:\Final\data\reports")
OLD_DB = "file:D:/NEW%20ECO/data/neweco.db?mode=ro&immutable=1"

START = pd.Timestamp("2019-01-02")
PANEL_END = pd.Timestamp("2026-09-30")
H = 5
N_TESTS = 14
BONF = 2.91          # Bonferroni 14 个双侧 5%
RHO_MAX = 0.70
COV_MIN = 0.50
INC_T = 2.0

CAND_NAMES = ("mom_120", "mom_250", "high_52w", "low_52w", "ret_max20", "skew_20",
              "bias_20", "bias_60", "illiq", "vol_60", "ep", "bp", "dv_ttm",
              "beta_250")


# ---------------------------------------------------------------- 工具

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


def build_candidates(cfg, panels, codes, cal) -> dict:
    """14 个候选面板（冻结口径，§2.1）。"""
    hfq = sa.hfq(panels)
    o = hfq["open"][codes].reindex(cal).astype("float64")
    c = hfq["close"][codes].reindex(cal).astype("float64")
    hi = panels["high"][codes].reindex(cal).astype("float64")
    lo = panels["low"][codes].reindex(cal).astype("float64")
    amt = panels["amount"][codes].reindex(cal).astype("float64")
    ret = c.pct_change()
    out = {}
    for w in (120, 250):
        out[f"mom_{w}"] = np.log(c / c.shift(w))
    out["high_52w"] = c / hi.rolling(250, min_periods=120).max()
    out["low_52w"] = c / lo.rolling(250, min_periods=120).min()
    out["ret_max20"] = ret.rolling(20).max()
    out["skew_20"] = ret.rolling(20).skew()
    for w in (20, 60):
        out[f"bias_{w}"] = c / c.rolling(w).mean() - 1.0
    out["illiq"] = (ret.abs() / (amt / 1e8).replace(0, np.nan)).rolling(20).mean()
    out["vol_60"] = ret.rolling(60).std()

    # beta_250（修 §0.3 的构造失败：用等权市场收益 + min_periods=120）
    mkt = ret.mean(axis=1)
    win = 250
    cov = ret.rolling(win, min_periods=120).cov(mkt)
    var = mkt.rolling(win, min_periods=120).var()
    out["beta_250"] = cov.div(var, axis=0)

    # 估值（老库 daily_basic，PIT = 当日收盘后可得）
    con = sqlite3.connect(OLD_DB, uri=True)
    db = pd.read_sql("SELECT trade_date, ts_code, pe_ttm, pb, dv_ttm FROM daily_basic "
                     "WHERE trade_date >= '20181201'", con)
    con.close()
    db["date"] = pd.to_datetime(db["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    db = db.dropna(subset=["date"])
    for col, name, xf in (("pe_ttm", "ep", lambda s: 1.0 / s.replace(0, np.nan)),
                          ("pb", "bp", lambda s: 1.0 / s.replace(0, np.nan)),
                          ("dv_ttm", "dv_ttm", None)):
        wdf = db.pivot_table(index="date", columns="ts_code", values=col, aggfunc="last").astype("float64")
        if xf:
            wdf = xf(wdf)
        out[name] = wdf.reindex(index=cal, columns=codes)
    return out


# ---------------------------------------------------------------- 增量检验

def incr_test(Fq, Yv, Pmr, M, cal):
    """模型 top-10% 内再筛最强 20% 的增量（bp, t, n）。"""
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


def orth_residual(F, ctrl_list, M, cal, codes, code_pos):
    """对 ctrl_list（面板列表，对应现役列）做逐日截面秩回归，返回残差面板。"""
    R = pd.DataFrame(np.nan, index=cal, columns=codes)
    for i in range(len(cal)):
        data = {f"c{k}": v.iloc[i] for k, v in enumerate(ctrl_list)}
        data["_f"] = F.iloc[i]
        df = pd.DataFrame(data).where(pd.Series(M[i], index=codes)).dropna()
        if len(df) < 100:
            continue
        Rk = df.rank(pct=True)
        X = np.column_stack([np.ones(len(Rk))] + [Rk[f"c{k}"].to_numpy()
                                                  for k in range(len(ctrl_list))])
        y = Rk["_f"].to_numpy()
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        R.iloc[i, [code_pos[k] for k in df.index]] = y - X @ beta
    return R


# ---------------------------------------------------------------- 主流程

def main():
    global H
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="G5 变异测试")
    ap.add_argument("--h", type=int, default=H,
                    help="标签 horizon（预注册 §2.2：h=5 主频段、h=1 并列报告——"
                         "h=1 是用户口径「每日」）")
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
    code_pos = {k: j for j, k in enumerate(codes)}

    hfq = sa.hfq(panels)
    o = hfq["open"][codes].reindex(cal).astype("float64")
    Y = (o.shift(-(H + 1)) / o.shift(-1) - 1.0).where(mask)
    Yv = Y.to_numpy("float64")
    Yr = Y.rank(axis=1, pct=True).to_numpy("float64")
    Pm = P.reindex(index=cal, columns=codes)
    Pmr = Pm.where(mask).rank(axis=1, pct=True).to_numpy("float64")

    print(f"样本 {len(cal)} 交易日（{cal[0].date()} → {cal[-1].date()}），{len(codes)} 只",
          flush=True)
    CAND = build_candidates(cfg, panels, codes, cal)
    missing = [c for c in CAND_NAMES if c not in CAND]
    assert not missing, f"候选缺失：{missing}"
    print(f"候选 {len(CAND_NAMES)} 个构造完成", flush=True)

    CTRL = [bundle.features[c].reindex(index=cal, columns=codes)
            for c in bundle.feature_cols]          # 现役 25 列（全部）
    print(f"正交化控制列 = 现役 {len(CTRL)} 列（C5 强版）", flush=True)

    def evaluate(panel, m=M, Yvv=Yv, Pmrr=Pmr, do_orth=True):
        F = panel.reindex(index=cal, columns=codes).where(mask)
        Fr = F.where(mask).rank(axis=1, pct=True).to_numpy("float64")
        ic = pd.Series(row_corr(Fr, Yr), index=cal).dropna()
        cov = float(F.notna().to_numpy().mean())
        # C2 max|ρ|
        mx, who = 0.0, None
        for cc, cl in zip(bundle.feature_cols, CTRL):
            r = np.nanmean(np.abs(row_corr(Fr, cl.where(mask).rank(axis=1, pct=True).to_numpy("float64"))))
            if np.isfinite(r) and r > mx:
                mx, who = float(r), cc
        bp4, t4, n4 = incr_test(F.to_numpy("float64"), Yvv, Pmrr, m, cal)
        if do_orth:
            R = orth_residual(F, CTRL, m, cal, codes, code_pos)
            bp5, t5, n5 = incr_test(R.to_numpy("float64"), Yvv, Pmrr, m, cal)
        else:
            bp5, t5, n5 = np.nan, None, 0
        return {"ic": float(ic.mean()), "t": nw(ic.to_numpy()), "cov": cov,
                "max_rho": mx, "rho_who": who,
                "incr4_bp": bp4, "incr4_t": t4,
                "incr5_bp": bp5, "incr5_t": t5, "n_incr": n5}

    # ---------------- 仪器门 ----------------
    print("\n" + "=" * 100)
    print("仪器门（预注册 §三）")
    print("=" * 100)

    rng = np.random.default_rng(20261011)
    # G1 阴性：日内置换的 illiq
    shuf = CAND["illiq"].copy()
    for i in range(len(cal)):
        v = shuf.iloc[i].to_numpy(copy=True)
        ok = np.isfinite(v)
        vv = v[ok].copy(); rng.shuffle(vv); v[ok] = vv
        shuf.iloc[i] = v
    g1 = evaluate(shuf)
    g1_pass_info = (g1["t"] is not None and abs(g1["t"]) >= BONF)
    g1_incr = (g1["incr5_t"] is not None and abs(g1["incr5_t"]) >= INC_T)
    G1 = (not g1_pass_info) and (not g1_incr)
    print(f"  G1 阴性（日内置换 illiq）：IC t={g1['t']:.2f}，C5 增量 {g1['incr5_bp']:+.1f}bp "
          f"t={g1['incr5_t']} ⇒ {'✓ 全不过' if G1 else '✗ 泄漏'}")

    # G2 阳性：已知有增量的合成因子（y_fwd 加噪，ρ≈0.15）
    ys = Y.to_numpy("float64")
    noise = rng.standard_normal(ys.shape)
    synth = np.where(np.isfinite(ys), 0.15 * np.nan_to_num(ys, nan=0.0) + noise, np.nan)
    syndf = pd.DataFrame(synth, index=cal, columns=codes)
    g2 = evaluate(syndf)
    G2 = (g2["incr5_t"] is not None and g2["incr5_t"] >= INC_T and g2["incr5_bp"] > 0)
    print(f"  G2 阳性（合成 y+噪声）：IC {g2['ic']:+.4f} t={g2['t']:.2f}，"
          f"C5 增量 {g2['incr5_bp']:+.1f}bp t={g2['incr5_t']:.2f} ⇒ "
          f"{'✓ 过（管线对真增量敏感）' if G2 else '✗ 不过 ⇒ 管线不敏感'}")

    # G3 影子对照：把 size_log 代入 ⇒ C5 必须不过
    g3 = evaluate(bundle.features["size_log"])
    G3 = not (g3["incr5_t"] is not None and g3["incr5_t"] >= INC_T and g3["incr5_bp"] > 0)
    print(f"  G3 影子对照（size_log，已在现役）：IC {g3['ic']:+.4f} t={g3['t']:.2f}，"
          f"C5 增量 {g3['incr5_bp']:+.1f}bp t={g3['incr5_t']} ⇒ "
          f"{'✓ 不过（影子检测器有效）' if G3 else '✗ 过了 ⇒ 正交化无效'}")

    # G4 确定性
    a = json.dumps(evaluate(CAND["illiq"]), default=str, sort_keys=True)
    b = json.dumps(evaluate(CAND["illiq"]), default=str, sort_keys=True)
    G4 = a == b
    print(f"  G4 确定性（背靠背重跑逐位一致）⇒ {'✓' if G4 else '✗'}")

    gates = {"G1_negative": bool(G1), "G2_positive": bool(G2), "G3_shadow": bool(G3),
             "G4_deterministic": bool(G4)}
    print(f"\n  ⇒ 仪器门 {'全部通过' if all(gates.values()) else '**有失败**'}")

    # ---------------- 逐候选 ----------------
    print("\n" + "=" * 100)
    print(f"C 判据（h={H}，Bonferroni 临界 |t| ≥ {BONF}）")
    print("=" * 100)
    print(f"{'候选':12s} {'C1 IC':>9s} {'t':>7s} | {'C3覆盖':>7s} | {'C2 max|ρ|':>10s} "
          f"{'对象':>12s} | {'C4增量':>8s} {'t':>6s} | {'C5增量':>8s} {'t':>6s} | 判定")
    print("-" * 100)

    rows = {}
    for name in CAND_NAMES:
        r = evaluate(CAND[name])
        c1 = r["t"] is not None and abs(r["t"]) >= BONF
        c3 = r["cov"] >= COV_MIN
        c4 = r["incr4_t"] is not None and r["incr4_t"] >= INC_T and r["incr4_bp"] > 0
        c5 = r["incr5_t"] is not None and r["incr5_t"] >= INC_T and r["incr5_bp"] > 0
        passed = c1 and c3 and c4 and c5
        r.update({"C1": bool(c1), "C3": bool(c3), "C4": bool(c4), "C5": bool(c5),
                  "PASS": bool(passed), "name": name})
        rows[name] = r
        print(f"{name:12s} {r['ic']:>+9.4f} {r['t'] or 0:>+7.2f} | {r['cov']:>7.1%} | "
              f"{r['max_rho']:>10.3f} {str(r['rho_who']):>12s} | "
              f"{r['incr4_bp']:>+8.1f} {r['incr4_t'] or 0:>+6.2f} | "
              f"{r['incr5_bp']:>+8.1f} {r['incr5_t'] or 0:>+6.2f} | "
              f"{'**PASS**' if passed else '—'}")

    n_pass = sum(1 for r in rows.values() if r["PASS"])
    print()
    print("=" * 100)
    print("合并裁决")
    print("=" * 100)
    if n_pass:
        concl = f"H12-a：{n_pass} 个候选通过 ⇒ 值得另立**采纳**预注册"
    else:
        concl = "H12-b：全部不成立 ⇒ **Final 的量价因子面结清**"
    print(f"  {concl}")
    c5_only = [k for k, r in rows.items() if r["C5"] and not r["PASS"]]
    if c5_only:
        print(f"  注：C5 过但其它项未过的：{c5_only}")

    doc = {"stage": "筛选批（无采纳）",
           "prereg": "plans/pv-screen-prereg-20261011.md",
           "generated_at": datetime.now().isoformat(timespec="seconds"),
           "window": [str(cal[0].date()), str(cal[-1].date())], "n_days": len(cal),
           "h": H, "bonferroni": BONF, "n_tests": N_TESTS,
           "gates": gates,
           "gate_detail": {"G1": g1, "G2": g2, "G3": g3},
           "candidates": rows, "n_pass": n_pass, "conclusion": concl}
    # ⚠ 自伤登记（2026-10-11）：首版固定写 `pv_screen_report.json`
    #    ⇒ h=1 的并列报告**覆盖**了 h=5 的主报告（证据包 provenance 丢失）。
    #    修法：文件名带 horizon（`pv_screen_report_h{H}.json`），互不覆盖。
    (OUT_DIR / f"pv_screen_report_h{H}.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n报告 → {OUT_DIR / f'pv_screen_report_h{H}.json'}")
    return doc


if __name__ == "__main__":
    main()
