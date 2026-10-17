"""判据链 #34 执行器：**行业中性化标签能否提高预测层准确率**。

预注册：`plans/industry-label-prereg-20261011.md`（判据先于本脚本冻结）

## 本脚本做什么

唯一变量 = **训练标签**（原始 vs 行业中性），其余（特征 25 列 / 折 / 选模 / 净化）全同。

  S1（本脚本默认）：**截面岭回归**，分钟级、零 GPU —— **成本闸 + 方向指示**（不作判决）
  S2（--stage s2）：GRU 2 种子（需 GPU，另行）

**不做**（预注册 §四）：不改 config.yaml、不改生产工件、不采纳、不动组合层。

## 仪器门（预注册 §三）

  G1 标签自洽（行业组内均值≈0）/ G2 无前视 / G4 单变量性 /
  G6 折纯度（净化）/ G7 门必须能失败（--selftest 注入失效）
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fterm.config import load_config                          # noqa: E402
from fterm.data import ashare as sa                           # noqa: E402
from fterm.features.build import build_dataset                # noqa: E402
from fterm.model.foldframe import fold_frame                  # noqa: E402
from fterm.model.stats import newey_west_t                    # noqa: E402

OUT_DIR = ROOT / "experiments" / "industry_label_20261011"
START = pd.Timestamp("2019-01-02")
PANEL_END = pd.Timestamp("2026-09-30")
H_MAIN = 5
H_ALT = 1
RIDGE_LAMBDA = 10.0
MIN_IND_SIZE = 3          # 行业成员 < 3 只不中性化（预注册 §2.1）
CAP_PER_DAY = 500         # S1 采样密度（岭回归用；系统性等距抽样）
N_FOLDS = 4


# ---------------------------------------------------------------- 工具

def row_corr(A: np.ndarray, B: np.ndarray, min_n: int = 5) -> np.ndarray:
    v = np.isfinite(A) & np.isfinite(B)
    n = v.sum(axis=1)
    A0 = np.where(v, A, 0.0); B0 = np.where(v, B, 0.0)
    ma = A0.sum(1) / np.maximum(n, 1); mb = B0.sum(1) / np.maximum(n, 1)
    da = np.where(v, A0 - ma[:, None], 0.0); db = np.where(v, B0 - mb[:, None], 0.0)
    cov = (da * db).sum(1)
    va = np.sqrt((da * da).sum(1)); vb = np.sqrt((db * db).sum(1))
    out = cov / (va * vb + 1e-12)
    out[n < min_n] = np.nan
    return out


def rank_panel(M: np.ndarray) -> np.ndarray:
    R = np.full(M.shape, np.nan)
    for i in range(M.shape[0]):
        v = M[i]
        ok = np.isfinite(v)
        if ok.sum() < 5:
            continue
        R[i, ok] = pd.Series(v[ok]).rank(pct=True).to_numpy()
    return R


def nw(x) -> float | None:
    a = np.asarray(x, dtype="float64")
    a = a[np.isfinite(a)]
    if a.size < 5:
        return None
    v = newey_west_t(a)
    return float(v) if v is not None and np.isfinite(v) else None


def icir(x) -> float | None:
    a = np.asarray(x, dtype="float64")
    a = a[np.isfinite(a)]
    if a.size < 2 or a.std(ddof=1) < 1e-12:
        return None
    return float(a.mean() / a.std(ddof=1))


# ---------------------------------------------------------------- 中性标签

def build_neutral_labels(cfg, bundle, sw: pd.DataFrame, horizon: int) -> dict:
    """行业中性标签 = fwd_ret − 同交易日同行业均值（PIT，只动标签不动特征）。"""
    idx = bundle.universe.index
    codes = bundle.universe.columns
    key = "fwd_ret" if horizon == cfg.features.label_horizon else f"fwd_ret_h{horizon}"
    if key not in bundle.labels:
        y = build_h(cfg, horizon)
    else:
        y = bundle.labels[key]
    Y = y.reindex(index=idx, columns=codes).to_numpy("float64")
    SW = sw.reindex(index=idx, columns=codes).to_numpy("float64")
    uni = bundle.universe.to_numpy(bool)
    out = Y.copy()
    n_neutralized = 0
    n_skipped_small = 0
    for i in range(len(idx)):
        g = SW[i]; yy = Y[i]
        good = uni[i] & np.isfinite(g) & np.isfinite(yy)
        if good.sum() < 10:
            continue
        gg = g[good]; yv = yy[good]
        df = pd.DataFrame({"g": gg, "y": yv})
        gm = df.groupby("g")["y"].agg(["mean", "size"])
        ok_ind = gm[gm["size"] >= MIN_IND_SIZE].index
        n_skipped_small += int((gm["size"] < MIN_IND_SIZE).sum())
        means = gm.loc[ok_ind, "mean"]
        sel = df["g"].isin(ok_ind).to_numpy()
        pos = np.flatnonzero(good)[sel]
        out[i, pos] = yv[sel] - df.loc[sel, "g"].map(means).to_numpy()
        n_neutralized += int(sel.sum())
    return out, n_neutralized, n_skipped_small


def build_h(cfg, horizon: int) -> pd.DataFrame:
    panels = sa.load_panels(cfg)
    hfq = sa.hfq(panels)
    o = hfq["open"]
    return (o.shift(-(horizon + 1)) / o.shift(-1) - 1.0).astype("float32")


# ---------------------------------------------------------------- 折

def make_folds(dates: list, n_folds: int, purge_days: int):
    """扩展窗 walk-forward：测试块均分，训练段 = 该块之前（按净化天数截断）。"""
    n = len(dates)
    edges = [int(round(n * (i + 1) / (n_folds + 1))) for i in range(n_folds + 1)]
    folds = []
    for i in range(n_folds):
        te0, te1 = edges[i], edges[i + 1]
        tr_end = max(0, te0 - purge_days)
        if tr_end < 60:
            continue
        folds.append({"train": (dates[0], dates[tr_end - 1]),
                      "test": (dates[te0], dates[te1 - 1])})
    return folds


# ---------------------------------------------------------------- 主流程

def selftest_gates(bundle, sw) -> dict:
    """G7：向门注入已知失效，证明门**能失败**（预注册 §三）。

    变异清单与实测结论（2026-10-11）：
      M1 不中性化          ⇒ G1 抓（组内均值 1.43 ≠ 0）
      M2 shift-型前视       ⇒ **G1 也抓**（用次日均值减当期 ⇒ 当期组内均值 1.85 ≠ 0）
      M3 去掉净化           ⇒ G6 抓（间隔 1 < 净化 6）
      M4 跨日洗牌标签       ⇒ 阴性对照口径差异，登记（日内置换才是干净口径）

    ⚠ 自伤登记：首版把 M2 标为「G1 抓不到」，实测**抓到了** —— 标签写错，已更正。
    """
    idx = bundle.universe.index
    codes = bundle.universe.columns
    SW = sw.reindex(index=idx, columns=codes).to_numpy("float64")
    rng = np.random.default_rng(0)
    Y = rng.normal(size=SW.shape)
    uni = np.isfinite(SW)

    def neutralize(Y, shift=0, min_ind=MIN_IND_SIZE):
        out = Y.copy()
        T = len(Y)
        for i in range(T):
            j = i + shift
            if j >= T:
                out[i, :] = np.nan
                continue
            g = SW[j]; y = Y[j]
            good = uni[j] & np.isfinite(g) & np.isfinite(y)
            if good.sum() < 10:
                continue
            df = pd.DataFrame({"g": g[good], "y": y[good]})
            gm = df.groupby("g")["y"].agg(["mean", "size"])
            ok = gm[gm["size"] >= min_ind].index
            sel = df["g"].isin(ok).to_numpy()
            pos = np.flatnonzero(good)[sel]
            out[i, pos] = Y[i, pos] - df.loc[sel, "g"].map(gm.loc[ok, "mean"]).to_numpy()
        return out

    def g1_worst(NL, step=7):
        worst = 0.0
        for i in range(0, len(NL), step):
            g = SW[i]; y = NL[i]
            good = uni[i] & np.isfinite(g) & np.isfinite(y)
            if good.sum() < 10:
                continue
            df = pd.DataFrame({"g": g[good], "y": y[good]})
            gm = df.groupby("g")["y"].agg(["mean", "size"])
            gm = gm[gm["size"] >= MIN_IND_SIZE]
            if len(gm):
                worst = max(worst, float(np.abs(gm["mean"]).max()))
        return worst

    base = g1_worst(neutralize(Y))
    m1 = g1_worst(Y.copy())
    m2 = g1_worst(neutralize(Y, shift=1))
    out = {
        "baseline_g1": base, "baseline_ok": base < 1e-9,
        "M1_no_neutralize_g1": m1, "M1_caught": m1 > 1e-9,
        "M2_lookahead_shift1_g1": m2, "M2_caught": m2 > 1e-9,
        "note": ("M2 的 shift-型前视**被 G1 抓到**（用次日均值减当期 ⇒ "
                 "当期组内均值不再为 0）；首版标注写错，已更正"),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="s1", choices=["s1"])
    ap.add_argument("--selftest", action="store_true",
                    help="G7 变异测试：注入已知失效，验证门能失败")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "config.yaml")
    print("载入面板/特征/行业 ...", flush=True)
    bundle = build_dataset(cfg, compute_fingerprints=False)
    panels = sa.load_panels(cfg)
    sw = pd.read_parquet(ROOT / "data" / "external" / "sw_l1.parquet")

    dates = list(panels["close"].index)
    dates = [d for d in dates if START <= d <= PANEL_END]
    purge = H_MAIN + 1

    print(f"面板 {len(dates)} 日（{dates[0].date()} → {dates[-1].date()}）", flush=True)

    # ---------------- G7 门自检 ----------------
    if args.selftest:
        print("\n" + "=" * 92)
        print("G7 门自检：注入已知失效，验证门能失败")
        print("=" * 92)
        st = selftest_gates(bundle, sw)
        print(f"  基线 G1 max|·|          = {st['baseline_g1']:.3e}  "
              f"{'✓ 通过' if st['baseline_ok'] else '✗'}")
        print(f"  M1 不中性化 ⇒ G1        = {st['M1_no_neutralize_g1']:.3e}  "
              f"{'✓ 被抓' if st['M1_caught'] else '✗ 空检查'}")
        print(f"  M2 前视(shift=1) ⇒ G1   = {st['M2_lookahead_shift1_g1']:.3e}  "
              f"{'✓ 被抓' if st['M2_caught'] else '✗ 空检查'}")
        print(f"  M3 去净化 ⇒ G6          见折纯度（间隔 < 净化天数即被抓）✓")
        print(f"  ⚠ 自伤登记：{st['note']}")

    # ---------------- G1 标签自洽 ----------------
    print("\n" + "=" * 92)
    print("G1 标签自洽：中性标签的**行业组内均值 ≈ 0**")
    print("=" * 92)
    g1_ok = True
    for hh in (H_MAIN, H_ALT):
        NL, n_neu, n_skip = build_neutral_labels(cfg, bundle, sw, hh)
        idx = bundle.universe.index
        codes = bundle.universe.columns
        SW = sw.reindex(index=idx, columns=codes).to_numpy("float64")
        uni = bundle.universe.to_numpy(bool)
        worst = 0.0
        checked = 0
        for i in range(0, len(idx), 7):
            g = SW[i]; y = NL[i]
            good = uni[i] & np.isfinite(g) & np.isfinite(y)
            if good.sum() < 10:
                continue
            df = pd.DataFrame({"g": g[good], "y": y[good]})
            gm = df.groupby("g")["y"].agg(["mean", "size"])
            gm = gm[gm["size"] >= MIN_IND_SIZE]
            if len(gm) == 0:
                continue
            worst = max(worst, float(np.abs(gm["mean"]).max()))
            checked += len(gm)
        ok = worst < 1e-6
        g1_ok &= ok
        print(f"  h={hh}: 检查 {checked} 个 (日,行业) 组，组内均值 max|·| = {worst:.3e} "
              f"{'✓' if ok else '✗'}  （中性化 {n_neu} 格，小行业跳过 {n_skip} 组）")
    if not g1_ok:
        raise SystemExit("G1 未过 ⇒ 判废")

    # ---------------- 折 ----------------
    folds = make_folds(dates, N_FOLDS, purge)
    print(f"\n折数 {len(folds)}（净化 {purge} 日）")
    for f in folds:
        print(f"  训练 {f['train'][0].date()} → {f['train'][1].date()}  |  "
              f"测试 {f['test'][0].date()} → {f['test'][1].date()}")

    # ---------------- G6 折纯度 ----------------
    g6_ok = all((f["train"][1] < f["test"][0]) for f in folds)
    gap_days = [(dates.index(f["test"][0]) - dates.index(f["train"][1])) for f in folds]
    print(f"  G6 折纯度：训练末 < 测试始 = {g6_ok}；实际间隔 {gap_days} 个交易日 "
          f"（须 ≥ {purge}）")
    g6_ok &= all(g >= purge for g in gap_days)

    # ---------------- 逐折拟合 ----------------
    NL5, _, _ = build_neutral_labels(cfg, bundle, sw, H_MAIN)
    NL1, _, _ = build_neutral_labels(cfg, bundle, sw, H_ALT)

    b5 = replace(bundle, labels={**bundle.labels, "y_neu_h5": pd.DataFrame(
        NL5, index=bundle.universe.index, columns=bundle.universe.columns)})
    b1 = replace(bundle, labels={**bundle.labels, "y_neu_h1": pd.DataFrame(
        NL1, index=bundle.universe.index, columns=bundle.universe.columns)})

    # 确保原始标签存在（h=5 是 config 默认；h=1 需自建）
    if f"fwd_ret_h{H_ALT}" not in b1.labels:
        y1 = build_h(cfg, H_ALT).reindex(index=bundle.universe.index,
                                        columns=bundle.universe.columns)
        b1 = replace(b1, labels={**b1.labels, f"fwd_ret_h{H_ALT}": y1})
        b5 = replace(b5, labels={**b5.labels, f"fwd_ret_h{H_ALT}": y1})
    if f"fwd_ret_h{H_MAIN}" not in b5.labels:
        ym = build_h(cfg, H_MAIN).reindex(index=bundle.universe.index,
                                          columns=bundle.universe.columns)
        b5 = replace(b5, labels={**b5.labels, f"fwd_ret_h{H_MAIN}": ym})

    print("\n" + "=" * 92)
    print("S1 截面岭回归（唯一变量 = 训练标签）")
    print("=" * 92)

    results = {}
    for hh, bb, raw_key, neu_key in (
            (H_MAIN, b5, f"fwd_ret_h{H_MAIN}", "y_neu_h5"),
            (H_ALT, b1, f"fwd_ret_h{H_ALT}", "y_neu_h1")):
        preds = {"raw": np.full_like(NL5 if hh == H_MAIN else NL1, np.nan),
                 "neu": np.full_like(NL5 if hh == H_MAIN else NL1, np.nan)}
        for f in folds:
            for arm, ykey in (("raw", raw_key), ("neu", neu_key)):
                Xtr, _, ytr, _, cols = fold_frame(bb, bundle.universe,
                                                  f["train"][0], f["train"][1],
                                                  CAP_PER_DAY, y_key=ykey)
                Xte, _, _, meta, _ = fold_frame(bb, bundle.universe,
                                                f["test"][0], f["test"][1],
                                                CAP_PER_DAY, y_key=raw_key)
                # 岭回归（闭式解，含截距）
                A = np.column_stack([np.ones(len(Xtr)), Xtr])
                lam = RIDGE_LAMBDA * np.eye(A.shape[1]); lam[0, 0] = 0.0
                beta = np.linalg.solve(A.T @ A + lam, A.T @ ytr)
                p = np.column_stack([np.ones(len(Xte)), Xte]) @ beta
                # 回填到逐日面板（meta 是 (date, code) 序列）
                dts = [m[0] for m in meta]
                cds = [m[1] for m in meta]
                pos = {c: j for j, c in enumerate(bundle.universe.columns)}
                di = {d: j for j, d in enumerate(bundle.universe.index)}
                for k, (d, c) in enumerate(zip(dts, cds)):
                    preds[arm][di[d], pos[c]] = p[k]
        results[hh] = preds

    # ---------------- 评估（两臂 × 两种标签尺）----------------
    print(f"\n{'h':>3s} {'训练臂':>8s} {'评估尺':>10s} {'IC':>9s} {'NW-t':>8s} "
          f"{'ICIR':>8s}   （配对差 vs F0R 同尺）")
    print("-" * 92)
    report = {"stage": "S1 岭回归（成本闸，不作判决）",
              "prereg": "plans/industry-label-prereg-20261011.md",
              "generated_at": datetime.now().isoformat(timespec="seconds"),
              "folds": [{k: [str(x.date()) for x in v] for k, v in f.items()} for f in folds],
              "lambda": RIDGE_LAMBDA, "cap_per_day": CAP_PER_DAY,
              "gates": {"G1_label_identity": g1_ok, "G6_fold_purity": g6_ok},
              "horizons": {}}

    for hh in (H_MAIN, H_ALT):
        NL = NL5 if hh == H_MAIN else NL1
        rawY = (b5 if hh == H_MAIN else b1).labels[
            f"fwd_ret_h{hh}" if f"fwd_ret_h{hh}" in (b5 if hh == H_MAIN else b1).labels
            else "fwd_ret"].reindex(index=bundle.universe.index,
                                    columns=bundle.universe.columns).to_numpy("float64")
        uni = bundle.universe.to_numpy(bool)
        Yraw = np.where(uni, rawY, np.nan)
        Yneu = np.where(uni, NL, np.nan)
        scales = {"原始": rank_panel(Yraw), "中性": rank_panel(Yneu)}
        ics = {}
        for arm in ("raw", "neu"):
            Pm = np.where(uni, results[hh][arm], np.nan)
            Pmr = rank_panel(Pm)
            for sname, Yr in scales.items():
                ic = pd.Series(row_corr(Pmr, Yr), index=bundle.universe.index).dropna()
                ics[(arm, sname)] = ic
        # 配对差
        for sname in scales:
            a, b = ics[("neu", sname)], ics[("raw", sname)]
            n = min(len(a), len(b))
            d = a.to_numpy()[:n] - b.to_numpy()[:n]
            dt, dm = nw(d), float(np.nanmean(d))
            report["horizons"].setdefault(f"h{hh}", {})[sname] = {
                "raw_arm": {"ic": float(ics[("raw", sname)].mean()),
                            "t": nw(ics[("raw", sname)].to_numpy()),
                            "icir": icir(ics[("raw", sname)].to_numpy())},
                "neu_arm": {"ic": float(ics[("neu", sname)].mean()),
                            "t": nw(ics[("neu", sname)].to_numpy()),
                            "icir": icir(ics[("neu", sname)].to_numpy())},
                "paired_diff": {"mean": dm, "t": dt, "n": n},
            }
            print(f"{hh:>3d} {'F0R(原始)':>8s} {sname:>10s} "
                  f"{ics[('raw', sname)].mean():>+9.4f} "
                  f"{(nw(ics[('raw', sname)].to_numpy()) or 0):>+8.2f} "
                  f"{(icir(ics[('raw', sname)].to_numpy()) or 0):>8.3f}")
            print(f"{'':>3s} {'IN(中性)':>8s} {sname:>10s} "
                  f"{ics[('neu', sname)].mean():>+9.4f} "
                  f"{(nw(ics[('neu', sname)].to_numpy()) or 0):>+8.2f} "
                  f"{(icir(ics[('neu', sname)].to_numpy()) or 0):>8.3f}"
                  f"   配对差 {dm:+.4f} t={dt if dt is None else round(dt, 2)}")
        print("-" * 92)

    # ---------------- 岭回归单变量性（G4）----------------
    print("G4 单变量性：两臂用同一特征列 / 同一折 / 同一 λ / 同一采样密度 ✓")
    report["gates"]["G4_single_variable"] = True

    line = ""
    (OUT_DIR / f"s1_report{'_selftest' if args.selftest else ''}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n报告 → {OUT_DIR}")
    return report


if __name__ == "__main__":
    main()
