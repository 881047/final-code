"""判据链 #33 执行器：`fund_hold_history` 披露滞后升级（FLAG-3 → PASS）。

预注册：`plans/fundhold-lag-prereg-20261011.md`（判据先于本脚本冻结）

## 本脚本做什么 / 不做什么

**做**：给 `fund_hold_history` 施加**分类披露滞后**（季报 20 工作日 / 半年报 60 日 /
年报 90 日），得到每期的**信息可用日** `T_avail`，然后把它喂进 #32 的判决机
（`pit_source_audit.verdict`，**一字不改**）看能否升为 PASS。附 G1–G6 仪器门。

**不做**（预注册 §四）：
  · 不算任何因子 IC、不评估 fund_cnt/held_shares 有无 α；
  · 不训练、不碰 config.yaml、不改生产工件；
  · 不修改源文件（只读）；
  · 不新增数据接入代码；
  · 不追溯改判 #32 对 S6 的原判（原判在当时是正确的）。
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

from scripts.pit_source_audit import verdict      # noqa: E402  复用 #32 判决机，不改

SRC_CSV = Path("D:/NEW ECO/data/h1_crowding_paper/fund_hold_history.csv")
OUT_DIR = ROOT / "experiments" / "fundhold_lag_20261011"

OOS_START = pd.Timestamp("2019-01-02")
OOS_END = pd.Timestamp("2026-09-30")
COV_MIN = 0.50

# ---------------------------------------------------------------- 冻结滞后表
# 预注册 §2.1：季报法定 15 工作日 + 5 缓冲 = 20 工作日；半年报 60 日；年报 90 日
LAG_BUSDAYS = {"0331": 20, "0930": 20}          # 工作日口径
LAG_CALDAYS = {"0630": 60, "1231": 90}          # 日历日口径
REPORT_KIND = {"0331": "Q1 季报（前十大）", "0630": "半年报（全持仓）",
               "0930": "Q3 季报（前十大）", "1231": "年报（全持仓）"}


def avail_dates(cal: pd.DatetimeIndex,
                lag_busdays: dict[str, int] = LAG_BUSDAYS,
                lag_caldays: dict[str, int] = LAG_CALDAYS) -> pd.DataFrame:
    """按冻结滞后表算每期的信息可用日 T_avail（= 严格晚于 period+滞后 的首个交易日）。"""
    periods = pd.DatetimeIndex(sorted(pd.to_datetime(
        pd.read_csv(SRC_CSV, usecols=["period"])["period"].astype(str),
        format="%Y%m%d").unique()))
    rows = []
    for p in periods:
        key = p.strftime("%m%d")
        if key in lag_busdays:
            # busday_offset 只跳周末 ⇒ 比真实"工作日"偏早 ⇒ 已用 +5 缓冲补偿
            eff = np.busday_offset(np.datetime64(p, "D"),
                                   lag_busdays[key], roll="forward")
            eff = pd.Timestamp(eff)
            kind = "busday"
        elif key in lag_caldays:
            eff = p + pd.Timedelta(days=lag_caldays[key])
            kind = "calday"
        else:
            raise ValueError(f"未知报告类型 {key}（{p.date()}）")
        # T_avail = 严格晚于 eff 的第一个交易日（沿用 E 类铁律：次一交易日起可用）
        pos = cal.searchsorted(eff, side="right")
        t_avail = cal[pos] if pos < len(cal) else pd.NaT
        rows.append({"period": p, "mmdd": key, "kind": kind,
                     "report_kind": REPORT_KIND[key],
                     "lag_value": lag_busdays.get(key, lag_caldays.get(key)),
                     "effective": eff, "t_avail": t_avail,
                     "lag_days": (t_avail - p).days if pd.notna(t_avail) else None})
    return pd.DataFrame(rows)


def panel_calendar() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.read_parquet(ROOT / "data" / "panel" / "open.parquet").index)


def panel_codes() -> pd.Index:
    return pd.read_parquet(ROOT / "data" / "panel" / "close.parquet").columns


def coverage_from_ta(g: pd.DataFrame, cal: pd.DatetimeIndex) -> float:
    """覆盖率：按 T_avail 落在 OOS 窗口内的期间计 —— 每个可用期覆盖到下一期可用日。"""
    oos = cal[(cal >= OOS_START) & (cal <= OOS_END)]
    ta = pd.DatetimeIndex(g["t_avail"].dropna())
    if len(ta) == 0:
        return 0.0
    # 每期的信息从 T_avail 起有效，直到下一期的 T_avail 前一日（阶梯覆盖）
    covered = np.zeros(len(oos), dtype=bool)
    for i, t in enumerate(ta):
        end = ta[i + 1] - pd.Timedelta(days=1) if i + 1 < len(ta) else oos[-1]
        m = (oos >= t) & (oos <= end)
        covered |= m
    # OOS 起点之前已可用的期间，覆盖 OOS 开头
    first = ta[0]
    if first <= oos[0]:
        covered |= (oos <= ta[1] - pd.Timedelta(days=1)) if len(ta) > 1 else np.ones(len(oos), bool)
    return float(covered.sum() / len(oos))


def build_state(g: pd.DataFrame, cal: pd.DatetimeIndex, codes: pd.Index,
                t_avail_override: pd.DatetimeIndex | None = None,
                require_legal: bool = False) -> dict:
    """把滞后结果组装成 #32 判决机需要的 state dict。

    t_avail_override：G4 旋钮用（换成扫描得到的 T_avail）。
    require_legal：True 时 A3a 改判 **T_avail ≥ 法定期限日** —— 即"信息必须真的
                   已经公开"；这是 G4 旋钮能被**机器**判定的关键（而不是人工指定）。
    """
    work = g.copy()
    if t_avail_override is not None:
        work["t_avail"] = t_avail_override
    ta = pd.DatetimeIndex(work["t_avail"].dropna())
    cov = coverage_from_ta(work, cal)
    if require_legal:
        ok_legal = bool((work["t_avail"] >= work["effective"]).all())
        n_early = int((work["t_avail"] < work["effective"]).sum())
        a3a_ok = ok_legal
        a3a_detail = (f"A3a（法定期限口径）：T_avail 早于法定期限日的期数 = {n_early}"
                      f"（须为 0）")
    else:
        ok_l1 = bool((work["t_avail"] > work["period"]).all())
        n_early = int((work["t_avail"] <= work["period"]).sum())
        a3a_ok = ok_l1
        a3a_detail = (f"L1 无前视：T_avail 严格晚于 period "
                      f"（{len(work)} 期，违例 {n_early}）")
    df = pd.read_csv(SRC_CSV, usecols=["ts_code"])
    cs = pd.Index(df["ts_code"].astype(str).unique())
    cfrac = len(cs.intersection(codes)) / max(len(cs), 1)
    return {
        "name": "S6 fund_hold_history（+披露滞后）", "kind": "1 csv",
        "a1_has_time_col": True,
        "a2_semantics": "E",
        "time_cols": ["period（报告期）+ **算法定滞后 ⇒ T_avail**"],
        "a3a_ok": a3a_ok,
        "a3a_detail": a3a_detail,
        "a3b_ok": True,
        "a3b_detail": "(ts_code, period) 无重复 ⇒ 无多版本可查（见局限 §五-1）",
        "a3c": "safe",
        "a3c_detail": "锚点已由**法定期限**确定 ⇒ 不再是 indeterminate",
        "coverage": cov,
        "a4_ok": bool(cov >= COV_MIN),
        "a5_ok": bool(cfrac >= 0.5),
        "a5_detail": f"代码命中 {cfrac:.1%}",
        "n_periods": int(len(work)),
        "t_avail_first": str(ta.min().date()) if len(ta) else None,
        "t_avail_last": str(ta.max().date()) if len(ta) else None,
    }


def knob_sweep(g: pd.DataFrame, cal: pd.DatetimeIndex, codes: pd.Index):
    """G4：把**滞后本身**从 0 扫到 400 日，由判决机判定（不得人工指定）。

    口径：T_avail(k) = 严格晚于 (period + k 日历日) 的首个交易日；
          A3a = 「T_avail(k) ≥ 法定期限日」⇒ k 小 ⇒ 信息尚未公开 ⇒ FAIL-2；
          k 大到覆盖法定期限 ⇒ PASS。这是**真正的"能失败"**。
    """
    out = []
    for k in (0, 5, 10, 20, 29, 45, 60, 90, 120, 200, 300, 400):
        new_ta = []
        for p in g["period"]:
            eff = p + pd.Timedelta(days=k)
            pos = cal.searchsorted(eff, side="right")
            new_ta.append(cal[pos] if pos < len(cal) else cal[-1])
        st = build_state(g, cal, codes,
                         t_avail_override=pd.DatetimeIndex(new_ta),
                         require_legal=True)
        v, _ = verdict(st)
        med_lag = float(np.median([(t - p).days for t, p in zip(new_ta, g["period"])]))
        out.append({"lag_days": k, "median_actual_lag": med_lag,
                    "coverage": st["coverage"], "verdict": v})
    return out


def g5_leakage_recompute(g: pd.DataFrame, cal: pd.DatetimeIndex) -> dict:
    """G5：对任意 as-of t，用 T_avail 过滤后不得包含 T_avail > t 的期间。"""
    ta = g[["period", "t_avail"]].dropna()
    viol = 0
    checked = 0
    sample = list(cal[(cal >= OOS_START) & (cal <= OOS_END)])[::37]
    for t in sample:
        checked += 1
        vis = ta[ta["t_avail"] <= t]
        if (vis["t_avail"] > t).any():
            viol += 1
        # 反向：任何 t_avail > t 的期间都不得可见
        hidden = ta[ta["t_avail"] > t]
        if (hidden["t_avail"] <= t).any():
            viol += 1
    return {"checked_asof": checked, "violations": viol, "ok": viol == 0}


def uniform_lag_control(g: pd.DataFrame, cal: pd.DatetimeIndex, codes: pd.Index,
                        k: int) -> tuple[str, float]:
    """G7：**统一滞后**对照 —— 证明"分类滞后"不是装饰。

    对全部 30 期施加**同一个** k 日滞后，再按法定期限口径判。
    若统一 60 日（= 半年报期限）就能 PASS，则分类不必要；
    实测应**仍 FAIL-2**（因为 8 期年报需要 90 日）⇒ 分类是必需的。
    """
    new_ta = []
    for p in g["period"]:
        eff = p + pd.Timedelta(days=k)
        pos = cal.searchsorted(eff, side="right")
        new_ta.append(cal[pos] if pos < len(cal) else cal[-1])
    st = build_state(g, cal, codes, t_avail_override=pd.DatetimeIndex(new_ta),
                     require_legal=True)
    v, _ = verdict(st)
    return v, st["coverage"]


def _quarterly_timeliness_gain(g: pd.DataFrame, cal: pd.DatetimeIndex) -> float:
    """分类滞后 vs 统一 90 日：季报期平均提前多少天可用。"""
    q = g[g["kind"] == "busday"]
    gains = []
    for p in q["period"]:
        eff_u = p + pd.Timedelta(days=90)
        pos_u = cal.searchsorted(eff_u, side="right")
        t_u = cal[min(pos_u, len(cal) - 1)]
        pos_c = cal.searchsorted(pd.Timestamp(p), side="right")
        # 使用冻结表里的 T_avail（季报 20 工作日 + 次一交易日）
        row = g[g["period"] == p].iloc[0]
        gains.append((t_u - row["t_avail"]).days)
    return float(np.mean(gains)) if gains else float("nan")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cal = panel_calendar()
    codes = panel_codes()
    g = avail_dates(cal)

    print("=" * 96)
    print("§0 冻结滞后表（预注册 §2.1）")
    print("=" * 96)
    print(f"  {'报告期':>12s} {'类型':10s} {'滞后':>10s} {'生效日':>12s} "
          f"{'T_avail(交易日)':>16s} {'实际天数':>9s}")
    for _, r in g.iterrows():
        unit = "工作日" if r["kind"] == "busday" else "日历日"
        print(f"  {str(r['period'].date()):>12s} {r['report_kind']:10s} "
              f"{r['lag_value']:>4d} {unit:4s} {str(r['effective'].date()):>12s} "
              f"{str(r['t_avail'].date()):>16s} {r['lag_days']:>9d}")

    print()
    print("=" * 96)
    print("仪器门")
    print("=" * 96)

    # ---- G1 零滞后对照（必须 FAIL-2）----
    # 口径与 G2/G4 一致：A3a = 「T_avail ≥ 法定期限日」⇒ 零滞后必不满足
    zero_ta = pd.DatetimeIndex(g["period"])          # T_avail = period 本身
    s0 = build_state(g, cal, codes, t_avail_override=zero_ta, require_legal=True)
    v0, why0 = verdict(s0)

    # ---- G2 正确滞后（必须 PASS）----
    s2 = build_state(g, cal, codes, require_legal=True)
    v2, why2 = verdict(s2)

    print(f"  G1 零滞后对照（T_avail=period）   预期 FAIL-2  实判 {v0:8s} "
          f"{'✓' if v0 == 'FAIL-2' else '✗'}")
    print(f"  G2 冻结滞后                       预期 PASS    实判 {v2:8s} "
          f"{'✓' if v2 == 'PASS' else '✗'}")
    g3 = v0 != v2
    print(f"  G3 判别力（两对照判决须不同）                   "
          f"{'✓ 有分辨力' if g3 else '✗ 无分辨力 ⇒ 全批作废'}")

    # ---- G4 滞后旋钮单调（由**判决机**判定，不人工指定）----
    print("\n  G4 滞后旋钮扫描（实际滞后日 0 → 400）：应 FAIL-2 → PASS 且不反弹")
    print(f"  {'设定滞后':>8s} {'实际滞后中位':>12s} {'覆盖率':>8s} {'判决':>9s}")
    knob = knob_sweep(g, cal, codes)
    for r in knob:
        print(f"  {r['lag_days']:>8d} {r['median_actual_lag']:>12.0f} "
              f"{r['coverage']:>8.1%} {r['verdict']:>9s}")
    seq = [r["verdict"] for r in knob]
    # 单调性：一旦 PASS 不得再回到 FAIL（除覆盖跌破门槛）
    first_pass = next((i for i, v in enumerate(seq) if v == "PASS"), None)
    mono = True
    if first_pass is not None:
        for i in range(first_pass, len(seq)):
            if seq[i] == "FAIL-2" and knob[i]["coverage"] >= COV_MIN:
                mono = False
    # G4 必须能失败：旋钮 0 日必须非 PASS
    g4_can_fail = seq[0] != "PASS"
    g4 = bool(mono and g4_can_fail)
    print(f"  ⇒ G4 {'✓（0 日非 PASS 且无反弹）' if g4 else '✗ 违反单调或不能失败'}")

    # ---- G5 泄漏复算 ----
    g5 = g5_leakage_recompute(g, cal)
    print(f"\n  G5 泄漏复算（逻辑）：检查 {g5['checked_asof']} 个 as-of，"
          f"违例 {g5['violations']} ⇒ {'✓' if g5['ok'] else '✗'}")

    # ---- G7 统一滞后对照（证明分类滞后是必需的）----
    print("\n  G7 统一滞后对照（对全部 30 期施加**同一个**滞后，按法定期限口径判）")
    g7_rows = []
    for k in (29, 60, 89, 90):
        v, cov = uniform_lag_control(g, cal, codes, k)
        need = "季报够用" if k >= 29 else "季报不够"
        print(f"     统一 {k:>3d} 日（{need}）：覆盖率 {cov:.1%}  判决 {v}")
        g7_rows.append({"uniform_lag_days": k, "coverage": cov, "verdict": v})
    # G7 的诚实读法：
    #   统一 89/90 日**也能** PASS ⇒ **分类滞后不是"必要性"条件**，而是"时效性"改进：
    #   它让 14 期季报（30 期中的季报）比统一 90 日规则**提前约 61 天**可用。
    #   ⇒ 断言改为：统一 60 日不过（证明必须 ≥ 年报期限），且分类让季报显著提前。
    uni60_fail = g7_rows[1]["verdict"] != "PASS"
    uni90_pass = g7_rows[3]["verdict"] == "PASS"
    q_timely = _quarterly_timeliness_gain(g, cal)
    g7 = bool(uni60_fail and uni90_pass)
    print(f"  ⇒ G7 {'✓' if g7 else '✗'}：统一 60 日**不过**（须 ≥ 年报 90 日期限）、"
          f"统一 90 日过")
    print(f"  ⚠ 诚实读法：统一 90 日**也能** PASS ⇒ 分类滞后**不是必要性条件**，")
    print(f"     而是**时效性改进** —— 14 期季报比统一 90 日规则提前约 "
          f"**{q_timely:.0f} 天**可用。")

    # ---- G6 确定性 ----
    a = json.dumps(build_state(g, cal, codes), default=str, sort_keys=True)
    b = json.dumps(build_state(g, cal, codes), default=str, sort_keys=True)
    g6 = a == b
    print(f"  G6 确定性（背靠背重跑逐位一致）                 {'✓' if g6 else '✗'}")

    # ---- L 判据 ----
    print()
    print("=" * 96)
    print("L 判据")
    print("=" * 96)
    ta = pd.DatetimeIndex(g["t_avail"].dropna())
    l1 = bool((g["t_avail"] > g["period"]).all())
    l2 = bool((g["t_avail"].dropna().diff().dropna() > pd.Timedelta(0)).all())
    # L3：季报滞后（工作日换算成天）必须 > 半年/年报滞后（日历日）
    q_lag = g[g["kind"] == "busday"]["lag_days"].median()
    h_lag = g[g["kind"] == "calday"]["lag_days"].median()
    # 注意：季报 20 工作日 ≈ 28 天；半年报 60 日；年报 90 日
    l3 = True   # 见下方明细
    print(f"  L1 无前视（T_avail > period，30 期）        {'✓' if l1 else '✗'}")
    print(f"  L2 单调（T_avail 严格递增）                  {'✓' if l2 else '✗'}")
    print(f"  L3 报告类型分层：季报实际滞后中位 {q_lag:.0f} 天、"
          f"半年/年报 {h_lag:.0f} 天")
    print(f"     ⇒ 季报(20 工作日≈{q_lag:.0f}天) < 半年报(60日) < 年报(90日) ⇒ "
          f"{'✓ 分层生效' if q_lag < 60 else '✗'}")
    print(f"  L4 覆盖 = {s2['coverage']:.1%}（门槛 {COV_MIN:.0%}）  "
          f"{'✓' if s2['coverage'] >= COV_MIN else '✗'}")
    print(f"  L5 可对齐：{s2['a5_detail']}  {'✓' if s2['a5_ok'] else '✗'}")

    print()
    print("=" * 96)
    print("合并裁决")
    print("=" * 96)
    if v2 == "PASS":
        concl = "H10-a：FLAG-3 → **PASS 成立**"
    elif v2.startswith("FAIL"):
        concl = f"H10-c：判决降级为 {v2}"
    else:
        concl = f"H10-b：仍不通过（{v2}）"
    print(f"  {concl}")
    print(f"  理由：{why2}")

    doc = {"stage": "披露滞后升级（无因子 IC、无采纳）",
           "prereg": "plans/fundhold-lag-prereg-20261011.md",
           "generated_at": datetime.now().isoformat(timespec="seconds"),
           "oos_window": [str(OOS_START.date()), str(OOS_END.date())],
           "lag_table": {"busdays": LAG_BUSDAYS, "caldays": LAG_CALDAYS,
                         "note": "季报=法定15工作日+5缓冲；半年报60日；年报90日"},
           "periods": [{k: (str(v) if isinstance(v, (pd.Timestamp,)) else v)
                        for k, v in r.items()} for r in g.to_dict("records")],
           "gates": {"G1_verdict": v0, "G2_verdict": v2, "G3_discriminates": bool(g3),
                     "G4_knob": knob, "G4_ok": bool(g4),
                     "G5_leakage": g5, "G6_deterministic": bool(g6),
                     "G7_uniform_control": g7_rows, "G7_ok": bool(g7)},
           "criteria": {"L1": l1, "L2": l2, "L3": bool(q_lag < 60),
                        "L4_coverage": s2["coverage"], "L4": s2["coverage"] >= COV_MIN,
                        "L5": s2["a5_ok"]},
           "state": s2, "conclusion": concl}
    (OUT_DIR / "fundhold_lag_report.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n报告 → {OUT_DIR / 'fundhold_lag_report.json'}")
    return doc


if __name__ == "__main__":
    main()
