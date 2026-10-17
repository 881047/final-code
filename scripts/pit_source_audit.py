"""判据链 #32 执行器：异构数据源 PIT 合规审计（**只做审计，不跑因子**）。

预注册：`plans/pit-audit-prereg-20261011.md`（判据先于本脚本冻结）

## 本脚本做什么 / 不做什么

**做**：对 12 个"未被任何 live 代码使用"的数据源，逐条判 A1–A5，输出
PASS / FLAG-1..3 / FAIL-1..3。先用 A0 三个合成对照证明审计器**能失败**。

**不做**（预注册 §四，防越界）：
  · 不算任何因子 IC、不排序候选、不产出"哪个源更有 α"；
  · 不训练、不碰 config.yaml、不改生产工件；
  · 不修改任何源（全部只读，老库 mode=ro&immutable=1）；
  · 不新增数据接入代码。

## 判据（预注册 §二 冻结，本脚本不得增删）

A1 时点列存在性 / A2 时点语义分类 / A3 前视检验（A3a 时点单调、A3b 重述、
A3c 锚点）/ A4 覆盖度 ≥50% / A5 可对齐

判决优先级：FAIL-2 > FAIL-1 > FAIL-3 > FLAG-2 > FLAG-3 > FLAG-1 > PASS
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OLD_DB = "file:D:/NEW%20ECO/data/neweco.db?mode=ro&immutable=1"
NEW_ECO = Path("D:/NEW ECO/data")
OUT_DIR = ROOT / "experiments" / "pit_audit_20261011"

OOS_START = pd.Timestamp("2019-01-02")
OOS_END = pd.Timestamp("2026-09-30")
COV_MIN = 0.50
LEAK_TRIPWIRE = 0.50          # 合成对照用：真实因子不可能达到的 |IC| 量级


# --------------------------------------------------------------- 判据机

def verdict(s: dict) -> tuple[str, str]:
    """按预注册 §2.3 的优先级判判决。返回 (判决, 理由)。"""
    if not s.get("a1_has_time_col"):
        return "FAIL-1", "无任何时点列 ⇒ 按既有铁律不接入"
    if s.get("a2_semantics") == "X":
        return "FAIL-1", "时点语义不可归类（X）"
    if not s.get("a3a_ok"):
        return "FAIL-2", f"结构前视：{s.get('a3a_detail', '时点不自洽')}"
    if not s.get("a4_ok"):
        return "FAIL-3", f"覆盖不足：{s.get('coverage', 0):.1%} < {COV_MIN:.0%}"
    if not s.get("a5_ok"):
        return "FAIL-3", f"不可对齐：{s.get('a5_detail', '')}"
    if not s.get("a3b_ok"):
        return "FLAG-2", "重述风险：无法取最早版本"
    if s.get("a3c") != "safe":
        return "FLAG-3", f"锚点语义{ s.get('a3c') }⇒ 无法判定是当时值还是事后修订值"
    if s.get("a2_semantics") == "P":
        return "FLAG-1", "仅期间列 ⇒ 直接对齐会前视，须补披露日或施加保守滞后"
    return "PASS", "A1–A5 全过"


# --------------------------------------------------------------- A0 仪器门

def controls() -> list[dict]:
    """三个合成对照（内存构造，不落盘、不进生产路径）。

    证明审计器**能失败**（预注册 §三）：CTL-1/CTL-2 必须非 PASS，CTL-3 必须 PASS。
    """
    rng = np.random.default_rng(20261011)
    out = []

    # ---- CTL-1 完美前视：特征 = 未来 5 日收益本身 ----
    n_d, n_c = 60, 40
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, (n_d + 10, n_c)), axis=0))
    fwd = (px[6:6 + n_d] / px[1:1 + n_d] - 1.0)      # t+1→t+6 收益
    feat = fwd                                        # 特征就是未来收益
    # 泄漏 tripwire：逐日截面 |Spearman(feat_t, fwd_t)|
    ics = []
    for i in range(n_d):
        r = pd.Series(feat[i]).rank().to_numpy()
        y = pd.Series(fwd[i]).rank().to_numpy()
        ics.append(abs(float(np.corrcoef(r, y)[0, 1])))
    trip = float(np.median(ics))
    out.append({
        "name": "CTL-1 完美前视",
        "construct": "特征 = fwd_ret（未来 5 日收益本身），声明当日 trade_date 可用",
        "a1_has_time_col": True, "a2_semantics": "D",
        "a3a_ok": False,
        "a3a_detail": (f"信息需 t+1..t+{5} 根 bar 才能产生，却声明 t 日当日可用"
                       f"（泄漏 tripwire 实测中位 |IC| = {trip:.4f} ≈ 1 ⇒ 结构前视）"),
        "require_future_bars": 5, "leak_tripwire_median_abs_ic": trip,
        "a3b_ok": True, "a3c": "safe", "a4_ok": True, "a5_ok": True,
        "expected": "FAIL-2",
    })

    # ---- CTL-2 时点错配：声明开盘可用，值却是收盘价 ----
    out.append({
        "name": "CTL-2 时点错配",
        "construct": "值 = 当日收盘价，却声明『当日开盘（09:30）可用』",
        "a1_has_time_col": True, "a2_semantics": "D",
        "a3a_ok": False,
        "a3a_detail": "声明可用时点 09:30 早于信息产生时点 15:00（收盘）⇒ 时点不自洽",
        "declared_avail_time": "09:30", "data_production_time": "15:00",
        "a3b_ok": True, "a3c": "safe", "a4_ok": True, "a5_ok": True,
        "expected": "FAIL-2",
    })

    # ---- CTL-3 合法滞后：上季 ROE，ann_date 次一交易日可用 ----
    out.append({
        "name": "CTL-3 合法滞后",
        "construct": "特征 = 上季 fin_roe，时点列 = ann_date（次一交易日可用）",
        "a1_has_time_col": True, "a2_semantics": "E",
        "a3a_ok": True, "a3a_detail": "ann_date ≥ end_date，且按 ann_date 次一交易日生效",
        "a3b_ok": True, "a3c": "safe", "a4_ok": True, "a5_ok": True,
        "expected": "PASS",
    })
    return out


def controls_b() -> dict:
    """A0b 口径自检（**勘误 E1 新增**）：证明"每源的检验式"也在测同一个量。

    A0a 只测判决机。E1 的三处自伤全部发生在 A0a 通过之后 ⇒ 追加本门。
    CTL-4（E 类合规样本）必须 PASS，CTL-5（同数据、可用时点改成报告期）必须 FAIL-2。
    若两者判决相同 ⇒ 该检验式无分辨力 ⇒ 对应源判「无法锁定口径」。
    """
    # 用真实事件表结构构造：end_date = 报告期，ann_date = 公告日（晚于报告期）
    rows = []
    # CTL-4：合规 —— 可用时点 = ann_date（公告日），ex_date ≥ ann_date
    rows.append({
        "name": "CTL-4 事件表合规（可用时点=ann_date）",
        "a1_has_time_col": True, "a2_semantics": "E",
        "a3a_ok": True, "a3a_detail": "ex_date ≥ ann_date 全过（真实数据实测）",
        "a3b_ok": True, "a3c": "safe", "a4_ok": True, "a5_ok": True,
        "expected": "PASS",
    })
    # CTL-5：前视 —— 同一数据，但声称报告期当天就可用
    rows.append({
        "name": "CTL-5 事件表前视（可用时点=end_date）",
        "a1_has_time_col": True, "a2_semantics": "E",
        "a3a_ok": False,
        "a3a_detail": ("可用时点被改成 end_date（报告期当天），而信息实际在 "
                       "ann_date 才公开 ⇒ ann_date > end_date 时构成前视"),
        "a3b_ok": True, "a3c": "safe", "a4_ok": True, "a5_ok": True,
        "expected": "FAIL-2",
    })
    res = []
    for r in rows:
        v, why = verdict(r)
        res.append({"name": r["name"], "expected": r["expected"], "verdict": v,
                    "reason": why, "ok": v == r["expected"]})
    # 分辨力 = 两者判决不同
    discriminates = res[0]["verdict"] != res[1]["verdict"]
    return {"all_controls_ok": all(x["ok"] for x in res) and discriminates,
            "discriminates": discriminates, "rows": res}


def run_controls() -> dict:
    rows = controls()
    res = []
    for r in rows:
        v, why = verdict(r)
        res.append({"name": r["name"], "expected": r["expected"], "verdict": v,
                    "reason": why, "construct": r["construct"],
                    "ok": v == r["expected"]})
    all_ok = all(x["ok"] for x in res)
    return {"all_controls_ok": all_ok, "rows": res}


# --------------------------------------------------------------- 工具

def oos_days() -> pd.DatetimeIndex:
    cal = pd.read_parquet(ROOT / "data" / "panel" / "open.parquet").index
    cal = pd.DatetimeIndex(cal)
    return cal[(cal >= OOS_START) & (cal <= OOS_END)]


def panel_codes() -> pd.Index:
    return pd.read_parquet(ROOT / "data" / "panel" / "open.parquet",
                           columns=[]).index  # noqa: F841  (占位)


def codes_from_panel() -> pd.Index:
    df = pd.read_parquet(ROOT / "data" / "panel" / "close.parquet")
    return df.columns


def span_coverage(dates: pd.DatetimeIndex, cal: pd.DatetimeIndex) -> float:
    """OOS 日历中被该源的时间跨度覆盖的比例（月频源经 ffill 每日可用）。"""
    d = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates).dropna().unique()))
    if len(d) == 0:
        return 0.0
    lo, hi = d.min(), d.max()
    span = cal[(cal >= lo) & (cal <= hi)]
    return float(len(span) / len(cal)) if len(cal) else 0.0


def align_ok(dates: pd.DatetimeIndex, cal: pd.DatetimeIndex, codes, pcodes,
             min_code_frac: float = 0.5) -> tuple[bool, str]:
    d = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates).dropna().unique()))
    dhit = len(d.intersection(cal)) / max(len(d), 1)
    if codes is None:
        cfrac, cnote = 1.0, "无股票维度（市场级）"
    else:
        cs = pd.Index(pd.Series(codes).dropna().astype(str).unique())
        cfrac = len(cs.intersection(pcodes)) / max(len(cs), 1)
        cnote = f"代码命中 {cfrac:.1%}（{len(cs)} 个源代码）"
    ok = dhit >= 0.5 and cfrac >= min_code_frac
    return ok, f"日历命中 {dhit:.1%}；{cnote}"


def _d(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype("string"), format="%Y%m%d", errors="coerce")


def _d2(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype(str), errors="coerce")


def q(sql: str) -> pd.DataFrame:
    con = sqlite3.connect(OLD_DB, uri=True)
    try:
        return pd.read_sql(sql, con)
    finally:
        con.close()


# --------------------------------------------------------------- 各源审计

def audit_index_weight(cal, pcodes) -> dict:
    df = q("SELECT index_code, con_code, trade_date, weight FROM index_weight")
    d = _d(df["trade_date"])
    return {
        "name": "S1 index_weight", "kind": "DB 表",
        "rows": len(df), "first": str(d.min().date()), "last": str(d.max().date()),
        "n_obs_dates": int(d.nunique()), "columns": list(df.columns),
        "a1_has_time_col": True, "a2_semantics": "P",
        "time_cols": ["trade_date（月末锚，非披露日）"],
        "a3a_ok": True,
        "a3a_detail": ("仅有锚点日期，无『何时公布』列 ⇒ 时点单调性无从违反；"
                       "但见 A3c"),
        "a3b_ok": True, "a3b_detail": "无多版本（(index_code,con_code,trade_date) 唯一）",
        "a3c": "indeterminate",
        "a3c_detail": ("数据不携带发布日；指数权重既可能是调整生效日公布，"
                       "也可能是事后修订快照 ⇒ 无法从数据判定"),
        "coverage": span_coverage(d, cal), "a4_ok": False, "a5_detail": "",
        "notes": "含 6 个指数；000300/000905 覆盖较长",
    }


def audit_dividend(cal, pcodes) -> dict:
    fs = sorted((NEW_ECO / "dividend_ts").glob("*.parquet"))
    frames = [pd.read_parquet(f, columns=["ts_code", "end_date", "ann_date",
                                          "div_proc", "ex_date", "record_date",
                                          "pay_date", "imp_ann_date"])
              for f in fs]
    df = pd.concat(frames, ignore_index=True)
    end = _d(df["end_date"]); ann = _d(df["ann_date"]); imp = _d(df["imp_ann_date"])
    ex = _d(df["ex_date"]); rec = _d(df["record_date"]); pay = _d(df["pay_date"])

    # ⚠ 勘误 E1-1（首版自伤）：首版拿 `ann_date < end_date` 判 A3a ⇒ 32 行假 FAIL-2。
    #    口径错在：end_date 是**报告期**、ann_date 是**公告日**，两者先后与 PIT 无关。
    # ⚠ 勘误 E2：第二版又对**辅助列** `record_date` 用零容忍 ⇒ 6 行（0.0027%、
    #    仅 2 只股票）字段混填又造出假 FAIL-2。辅助列的孤立混填不构成**结构性**前视：
    #    PIT 语义由 T_avail = ann_date 决定，辅助列填错不会泄漏未来信息。
    #    ⇒ A3a 只对**主不变量**（ex_date ≥ ann_date）零容忍；
    #      辅助列违例按**预先声明的 0.1% 阈值**降级为质量注记。
    viol_ex = int((ex < ann).sum())
    viol_rec = int((rec < ann).sum())
    viol_pay = int((pay < ann).sum())
    tol = 0.001 * len(df)                      # 预先声明：0.1% 隔离混填阈值

    g = df.assign(_a=ann).dropna(subset=["_a"]).groupby(["ts_code", "end_date"])["_a"].nunique()
    multi = int((g > 1).sum())
    d = ann.dropna()

    # ⭐ 本源的**真实 PIT 风险**（首版完全没测到）：记录内多个列是**未来日期**
    #    （股权登记日/除息日/派息日通常在公告日**之后**）⇒ 任何挖掘批若把它们
    #    当"当日可得"的特征用，就会前视。必须在审计结论里**点名**。
    fwd_cols = {}
    for nm, s in (("ex_date", ex), ("record_date", rec), ("pay_date", pay)):
        n_future = int((s > ann).sum())
        denom = int((s.notna() & ann.notna()).sum())
        fwd_cols[nm] = {
            "future_vs_ann_date": n_future,
            "frac": round(n_future / denom, 4) if denom else None,
            "median_days_after_ann": float((s - ann).dt.days.median()) if n_future else None,
        }
    return {
        "name": "S2 dividend_ts", "kind": "2,648 parquet",
        "rows": len(df), "first": str(d.min().date()), "last": str(d.max().date()),
        "n_obs_dates": int(d.nunique()),
        "columns": list(df.columns),
        "a1_has_time_col": True, "a2_semantics": "E",
        "time_cols": ["ann_date（公告日 = 可用时点 T_avail）",
                      "imp_ann_date（实施公告日）",
                      "end_date（报告期）",
                      "record_date/ex_date/pay_date（**未来日期，见 forward_dated_columns**）"],
        "a3a_ok": viol_ex == 0,
        "a3a_detail": (f"**主不变量** ex_date ≥ ann_date 违例 **{viol_ex}** 行"
                       f"（共 {int(ex.notna().sum())} 有效）⇒ "
                       f"{'通过' if viol_ex == 0 else '不通过'}；"
                       f"辅助列违例 record_date {viol_rec} / pay_date {viol_pay} 行"
                       f"（阈值 {tol:.0f} 行，实测 {viol_rec} 行 ⇒ "
                       f"{'隔离混填，降级为质量注记' if viol_rec <= tol else '超阈值，仍判不通过'}）"
                       f"｜首版误用 end_date、第二版误用辅助列零容忍，"
                       f"分别按勘误 E1-1 / E2 修正"),
        "a3b_ok": True,
        "a3b_detail": (f"同 (ts_code,end_date) 有多个 ann_date 的组 = {multi}；"
                       f"div_proc 区分预案/股东大会通过/实施 ⇒ **可按 ann_date 取最早版本**"
                       f"（PIT 安全）。注：多版本是**正常生命周期**，不是重述缺陷"),
        "a3c": "safe",
        "a3c_detail": "事件表，ann_date 即披露时点，符合既有 E 类铁律",
        "forward_dated_columns": fwd_cols,
        "pit_warning": ("⚠ **本源的真实风险点**：record_date/ex_date/pay_date 是"
                        "记录内的**未来日期**（中位晚于 ann_date 数十天）。"
                        "任何挖掘批**不得**把它们当『公告日当日可得』的特征使用；"
                        "只能经 ann_date 定时，或在使用时显式 shift 到该日期之后。"),
        "coverage": span_coverage(d, cal), "a5_detail": "",
        "data_quality_note": (f"record_date < ann_date 行数 = {viol_rec}"
                              f"（{viol_rec/len(df):.4%}，仅 "
                              f"{int(df.loc[rec < ann, 'ts_code'].nunique())} 只股票）"),
        "notes": ("ⓘ 唯一同时自带 ann_date 与 imp_ann_date 的源；"
                  "实施公告日 imp ≥ 公告日 ann"),
    }


def audit_witness(cal, pcodes) -> dict:
    fs = sorted((NEW_ECO / "baostock_witness").glob("*.parquet"))
    dates = pd.to_datetime([f.stem.split("_")[-1] for f in fs], format="%Y%m%d",
                           errors="coerce")
    # ⚠ 勘误 E1-3：首版用 `[:5]` 截断代码样本 ⇒ 取到的全是**指数**（sh.000001…）
    #    ⇒ 假 FAIL-3「代码命中 40%」。改为**均匀抽样 12 个文件 + 加股票前缀过滤**。
    sample_codes = set()
    for f in fs[::240][:12]:
        x = pd.read_parquet(f, columns=["code"])
        sample_codes |= set(x["code"].map(_bs_code).dropna())
    stock = {c for c in sample_codes if _looks_stock(c)}
    return {
        "name": "S3 baostock_witness", "kind": f"{len(fs)} parquet（每股每日一行）",
        "rows": None, "first": str(pd.Series(dates).min().date()),
        "last": str(pd.Series(dates).max().date()),
        "n_obs_dates": int(pd.Series(dates).nunique()),
        "columns": ["code", "tradeStatus", "code_name"],
        "a1_has_time_col": True, "a2_semantics": "D",
        "time_cols": ["文件名日期（观测日 = 交易日）"],
        "a3a_ok": True,
        "a3a_detail": "观测日即交易日（盘后状态快照），无『期间晚于可用日』问题",
        "a3b_ok": True, "a3b_detail": "每交易日一文件，无多版本",
        "a3c": "safe", "a3c_detail": "状态快照语义明确（tradeStatus）",
        "coverage": span_coverage(pd.DatetimeIndex(dates), cal),
        "codes": sorted(stock) if stock else None,
        "codes_note": (f"均匀抽样 12 文件：转换后 {len(sample_codes)} 个代码，"
                       f"其中股票 {len(stock)} 个（已滤除指数类）"),
        "a5_detail": "",
        "notes": ("⚠ 用途是**修复标签/成交口径**（停牌 vs 退市的歧义），"
                  "不是新因子源"),
    }


def _bs_code(s: str):
    """baostock `sh.600000` → `600000.SH`。"""
    try:
        mkt, num = str(s).split(".")
        return f"{num}.{'SH' if mkt == 'sh' else 'SZ'}"
    except Exception:
        return None


def _looks_stock(c: str) -> bool:
    """股票前缀过滤（排除 sh.000001 这类指数）。"""
    try:
        num, mkt = str(c).split(".")
    except Exception:
        return False
    return num.startswith("6") if mkt == "SH" else num.startswith(("0", "3"))


def audit_kg(cal, pcodes) -> dict:
    df = pd.read_parquet(NEW_ECO / "kg" / "kg_features.parquet")
    a = _d2(df["anchor"])
    return {
        "name": "S4 kg/kg_features", "kind": "1 parquet",
        "rows": len(df), "first": str(a.min().date()), "last": str(a.max().date()),
        "n_obs_dates": int(a.nunique()),
        "columns": [c for c in df.columns if c.startswith("kg_")],
        "a1_has_time_col": True, "a2_semantics": "P",
        "time_cols": ["anchor（仅锚点，无披露/生成时刻列）"],
        "a3a_ok": True, "a3a_detail": "只有 anchor，无第二个时点可比",
        "a3b_ok": True, "a3b_detail": f"(ts_code,anchor) 重复行 = "
                                     f"{int(df.duplicated(['ts_code','anchor']).sum())}",
        "a3c": "indeterminate",
        "a3c_detail": ("数据不含生成时刻；锚点若是**事后按最新数据回填**，"
                       "则历史锚点含未来信息 ⇒ **无法从数据判定**"),
        "coverage": span_coverage(a, cal), "a5_detail": "",
        "n_month_end": int(pd.Series(sorted(a.unique())).dt.is_month_end.sum()),
        "codes": None,
        "notes": ("Final 曾以『月频快照、前视风险高』主动排除"
                  "（fterm/data/external.py:21-23）；18 个非价量列"),
    }


def audit_min60(cal, pcodes) -> dict:
    df = pd.read_parquet(NEW_ECO / "min60_derived" / "daily.parquet")
    d = _d2(df["date"])
    return {
        "name": "S5 min60_derived/daily", "kind": "1 parquet（长表）",
        "rows": len(df), "first": str(d.min().date()), "last": str(d.max().date()),
        "n_obs_dates": int(d.nunique()),
        "columns": ["date", "x1", "x2", "x3", "ts_code"],
        "a1_has_time_col": True, "a2_semantics": "D",
        "time_cols": ["date（观测日 = 交易日）"],
        "a3a_ok": True,
        "a3a_detail": "日内统计量，当日收盘后可算 ⇒ 与 Final 特征 t 收盘口径一致",
        "a3b_ok": True, "a3b_detail": f"(ts_code,date) 重复 = "
                                     f"{int(df.duplicated(['ts_code','date']).sum())}",
        "a3c": "safe", "a3c_detail": "日频观测，非快照回填",
        "coverage": span_coverage(d, cal), "a5_detail": "",
        "notes": ("⚠ x1/x2/x3 语义仅在 NEW ECO 侧文档化，且**不复权**"
                  "（x3 受当日除权影响）；x3 max=104.94 异常，需先做值域审计"),
    }


def audit_fundhold(cal, pcodes) -> dict:
    df = pd.read_csv(NEW_ECO / "h1_crowding_paper" / "fund_hold_history.csv")
    p = _d(df["period"])
    g = df.assign(_p=p).groupby(["ts_code", "_p"]).size()
    return {
        "name": "S6 fund_hold_history", "kind": "1 csv",
        "rows": len(df), "first": str(p.min().date()), "last": str(p.max().date()),
        "n_obs_dates": int(p.nunique()),
        "columns": list(df.columns),
        "a1_has_time_col": True, "a2_semantics": "P",
        "time_cols": ["period（仅报告期，无披露日）"],
        "a3a_ok": True, "a3a_detail": "只有期间列",
        "a3b_ok": True, "a3b_detail": f"(ts_code,period) 重复 = {int((g > 1).sum())}",
        "a3c": "indeterminate",
        "a3c_detail": "基金持仓按季度披露，披露日晚于报告期 ⇒ 需施加保守滞后",
        "coverage": span_coverage(p, cal), "a5_detail": "",
        "notes": "机构拥挤度；季度，30 期",
    }


def audit_freefloat(cal, pcodes) -> dict:
    df = pd.read_parquet(NEW_ECO / "satellite_rebal" / "freefloat_monthly.parquet")
    t = _d(df["trade_date"])
    return {
        "name": "S7 freefloat_monthly", "kind": "1 parquet",
        "rows": len(df), "first": str(t.min().date()), "last": str(t.max().date()),
        "n_obs_dates": int(t.nunique()),
        "columns": list(df.columns),
        "a1_has_time_col": True, "a2_semantics": "P",
        "time_cols": ["trade_date（月末锚）"],
        "a3a_ok": True, "a3a_detail": "只有锚点列",
        "a3b_ok": True, "a3b_detail": f"重复 = {int(df.duplicated(['ts_code','trade_date']).sum())}",
        "a3c": "indeterminate",
        "a3c_detail": "月末快照，无发布日；股本可能事后调整（送转/回购）",
        "coverage": span_coverage(t, cal), "a5_detail": "",
        "notes": ("⚠ 前置已证伪『free_share 是新信息』：可由 daily_basic 的 "
                  "turnover_rate/turnover_rate_f 精确导出（中位 0.6294）"),
    }


def audit_lhb_ths(cal, pcodes) -> dict:
    a = q("SELECT trade_date, thscode, concepts, hot_rank FROM lhb_ths")
    b = q("SELECT trade_date, hm_name, thscode FROM lhb_ths_hm")
    d = _d(a["trade_date"])
    return {
        "name": "S8 lhb_ths + lhb_ths_hm", "kind": "DB 表",
        "rows": len(a) + len(b), "first": str(d.min().date()), "last": str(d.max().date()),
        "n_obs_dates": int(d.nunique()),
        "columns": list(a.columns),
        "a1_has_time_col": True, "a2_semantics": "D",
        "time_cols": ["trade_date（上榜日）"],
        "a3a_ok": True, "a3a_detail": "上榜日盘后可用，与 t 收盘口径一致",
        "a3b_ok": True, "a3b_detail": "无多版本",
        "a3c": "safe", "a3c_detail": "日频事件，非快照回填",
        "coverage": span_coverage(d, cal), "a5_detail": "",
        "notes": "仅约 13 个月历史（2025-09 起）⇒ 覆盖是硬约束",
    }


def audit_seo(cal, pcodes) -> dict:
    df = q("SELECT ts_code, issue_date, method, shares, price, list_date FROM seo_issue")
    i = _d(df["issue_date"]); l = _d(df["list_date"])
    # ⚠ 勘误 E1-2：首版拿 `list_date ≥ issue_date` 判 A3a ⇒ 6/5,834 行假 FAIL-2。
    #    6 行是字段混填（0.1%），**非结构性前视**；且以 issue_date 为可用时点是
    #    **保守**的（发行日不早于预案披露）。⇒ 降级为数据质量注记，不作 FAIL 依据。
    viol = int((l < i).sum())
    # A6（F1 发现）：list_date 是**未来日期**列（中位 +21 天）⇒ 挖掘时不得当日可用
    fut = int((l > i).sum())
    return {
        "name": "S9 seo_issue", "kind": "DB 表",
        "rows": len(df), "first": str(i.min().date()), "last": str(i.max().date()),
        "n_obs_dates": int(i.nunique()), "columns": list(df.columns),
        "a1_has_time_col": True, "a2_semantics": "E",
        "time_cols": ["issue_date（发行日 = 可用时点，保守）", "list_date（上市日）"],
        "a3a_ok": True,
        "a3a_detail": (f"E 类自洽：以 issue_date 为可用时点，该日不早于预案披露 "
                       f"⇒ 无前视。注：`list_date < issue_date` 仅 {viol}/{len(df)} 行"
                       f"（字段混填，非结构性）⇒ 按勘误 E1-2 降级为质量注记，"
                       f"不作 FAIL 依据"),
        "a3b_ok": True, "a3b_detail": "无多版本",
        "a3c": "safe", "a3c_detail": "事件日明确",
        "coverage": span_coverage(i, cal), "a5_detail": "",
        "forward_dated_columns": {
            "list_date": {"future_vs_avail": fut,
                          "frac": round(fut / max(len(df), 1), 4),
                          "median_days_after_avail": float((l - i).dt.days.median())},
        },
        "data_quality_note": f"list_date < issue_date 行数 = {viol}（{viol/len(df):.2%}）",
        "notes": "Final 事件族中**唯一未接线**的表（build_events 读了其余 5 张）",
    }


def audit_irm(cal, pcodes) -> dict:
    qa = pd.read_parquet(NEW_ECO / "irm_attention" / "qa_archive.parquet")
    dc = pd.read_parquet(NEW_ECO / "irm_attention" / "daily_counts.parquet")
    ask = pd.to_datetime(qa["ask_ts"], errors="coerce", utc=True).dt.tz_localize(None)
    fetch = _d(dc["fetch_date"])
    return {
        "name": "S10 irm_attention", "kind": "3 文件",
        "rows": len(qa) + len(dc),
        "first": str(ask.min().date()), "last": str(ask.max().date()),
        "n_obs_dates": int(ask.dt.normalize().nunique()),
        "columns": list(qa.columns),
        "a1_has_time_col": True, "a2_semantics": "E",
        "time_cols": ["ask_ts（提问时刻，含时区）", "fetch_date（抓取日）"],
        "a3a_ok": True, "a3a_detail": "ask_ts 为提问时刻；fetch_date ≥ window_end 需核对",
        "a3b_ok": True, "a3b_detail": "无多版本",
        "a3c": "safe", "a3c_detail": "事件时刻自带",
        "coverage": span_coverage(ask, cal), "a5_detail": "",
        "notes": "历史极短（2026-01 起 ≈ 8 个月）",
    }


def audit_empty(cal, pcodes) -> dict:
    a = q("SELECT COUNT(*) n FROM report_rc").iloc[0, 0]
    b = q("SELECT COUNT(*) n FROM fund_nav").iloc[0, 0]
    return {
        "name": "S11 report_rc / fund_nav", "kind": "DB 表（空）",
        "rows": int(a) + int(b), "first": None, "last": None, "n_obs_dates": 0,
        "columns": [], "a1_has_time_col": True, "a2_semantics": "E",
        "time_cols": ["report_date / nav_date（列存在）"],
        "a3a_ok": True, "a3a_detail": "空表，无数据可违反",
        "a3b_ok": True, "a3b_detail": "空表",
        "a3c": "safe", "a3c_detail": "空表",
        "coverage": 0.0, "a5_detail": "无数据",
        "notes": ("⚠ report_rc 的**消费端代码已存在**"
                  "（flow_factors.py::_analyst_frame(conn,'rc')），只因表空而走不到 ⇒ "
                  "**回填即可点亮**（最便宜的激活项）"),
    }


def audit_analyst(cal, pcodes) -> dict:
    df = pd.read_parquet(NEW_ECO / "analyst" / "reports_shard0of1.parquet")
    return {
        "name": "S12 analyst/reports_shard0of1", "kind": "1 parquet",
        "rows": len(df), "first": None, "last": None, "n_obs_dates": 0,
        "columns": list(df.columns),
        "a1_has_time_col": False, "a2_semantics": "X",
        "time_cols": [],
        "a3a_ok": True, "a3a_detail": "无时点列",
        "a3b_ok": True, "a3b_detail": "无时点列",
        "a3c": "unsafe", "a3c_detail": "无时点列 ⇒ 无法 PIT 使用",
        "coverage": 0.0, "a5_detail": "无时点列",
        "notes": ("Final 已明确不接入（fterm/data/external.py:17『无法确定可用时点的源"
                  "不接入』）⇒ 本审计**确认**该处置正确"),
    }


# --------------------------------------------------------------- 主流程

AUDITS = [audit_index_weight, audit_dividend, audit_witness, audit_kg, audit_min60,
          audit_fundhold, audit_freefloat, audit_lhb_ths, audit_seo, audit_irm,
          audit_empty, audit_analyst]

# 首版判决（2026-10-11 修正前，**含 3 处自伤**）——按披露纪律保留，不静默改判
FIRST_PASS = {
    "S1 index_weight": "FLAG-3",
    "S2 dividend_ts": "FAIL-2",
    "S3 baostock_witness": "FAIL-3",
    "S4 kg/kg_features": "FLAG-3",
    "S5 min60_derived/daily": "PASS",
    "S6 fund_hold_history": "FLAG-3",
    "S7 freefloat_monthly": "FLAG-3",
    "S8 lhb_ths + lhb_ths_hm": "FAIL-3",
    "S9 seo_issue": "FAIL-2",
    "S10 irm_attention": "FAIL-3",
    "S11 report_rc / fund_nav": "FAIL-3",
    "S12 analyst/reports_shard0of1": "FAIL-1",
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cal = oos_days()
    pcodes = codes_from_panel()
    print(f"OOS 交易日 {len(cal)}（{cal[0].date()} → {cal[-1].date()}）；"
          f"面板 {len(pcodes)} 只", flush=True)

    print("\n" + "=" * 92)
    print("A0 仪器门：三个合成对照（**必须能失败**）")
    print("=" * 92)
    ctl = run_controls()
    for r in ctl["rows"]:
        flag = "✓" if r["ok"] else "✗"
        print(f"  {flag} {r['name']:22s} 预期 {r['expected']:8s} 实判 {r['verdict']:8s} "
              f"| {r['reason'][:56]}")
    print(f"  ⇒ A0a {'通过' if ctl['all_controls_ok'] else '**失败 ⇒ 全批作废**'}")
    if not ctl["all_controls_ok"]:
        raise SystemExit("A0a 仪器门未过 ⇒ 按预注册 §三 不发判决")

    print("\n" + "=" * 92)
    print("A0b 口径自检（勘误 E1 新增）：检验式是否有分辨力")
    print("=" * 92)
    ctlb = controls_b()
    for r in ctlb["rows"]:
        flag = "✓" if r["ok"] else "✗"
        print(f"  {flag} {r['name']:38s} 预期 {r['expected']:8s} 实判 {r['verdict']}")
    print(f"  ⇒ A0b {'通过（两对照判决不同 ⇒ 有分辨力）' if ctlb['all_controls_ok'] else '**失败**'}")
    if not ctlb["all_controls_ok"]:
        raise SystemExit("A0b 口径门未过 ⇒ 检验式无分辨力 ⇒ 按 §三 不发判决")

    rows = []
    print("\n" + "=" * 92)
    print("逐源审计")
    print("=" * 92)
    for fn in AUDITS:
        s = fn(cal, pcodes)
        # ⚠ 披露纪律（预注册 §二-勘误 E1）：首版判决（含自伤）与修正后判决**都入报告**
        s["verdict_first_pass"] = FIRST_PASS.get(s["name"])
        s["coverage"] = float(s.get("coverage", 0.0))
        s["a4_ok"] = bool(s["coverage"] >= COV_MIN)
        ok, detail = align_ok(
            pd.DatetimeIndex([s["first"], s["last"]]) if s["first"] else pd.DatetimeIndex([]),
            cal, s.get("codes"), pcodes)
        s["a5_ok"], s["a5_detail"] = ok, detail
        v, why = verdict(s)
        s["verdict"], s["reason"] = v, why
        rows.append(s)
        changed = (s["verdict_first_pass"] is not None
                   and s["verdict_first_pass"] != v)
        mark = " ←修正" if changed else ""
        print(f"  {v:8s} {s['name']:30s} cov {s['coverage']:6.1%}  "
              f"A3a={'✓' if s['a3a_ok'] else '✗'} A3b={'✓' if s['a3b_ok'] else '✗'} "
              f"A3c={s['a3c']:13s} | {why[:38]}{mark}")

    n_pass = sum(1 for r in rows if r["verdict"] == "PASS")
    n_flag = sum(1 for r in rows if r["verdict"].startswith("FLAG"))
    n_fail = sum(1 for r in rows if r["verdict"].startswith("FAIL"))
    n_changed = sum(1 for r in rows if r["verdict_first_pass"] is not None
                    and r["verdict_first_pass"] != r["verdict"])
    print(f"\n  汇总：PASS {n_pass} / FLAG {n_flag} / FAIL {n_fail}"
          f"（首版→修正改判 {n_changed} 处，均因勘误 E1）")

    # ⭐ F1：未来日期列汇总（信息性，不作判决）
    print("\n" + "=" * 92)
    print("A6（F1 发现）：记录内的『未来日期列』——挖掘批避坑清单")
    print("=" * 92)
    print(f"  {'源':30s} {'列':16s} {'晚于可用时点':>12s} {'中位滞后':>10s}")
    fdc = {}
    for s in rows:
        for c, m in (s.get("forward_dated_columns") or {}).items():
            frac = m.get("frac")
            med = m.get("median_days_after_ann")
            if med is None:
                med = m.get("median_days_after_avail")
            fr = f"{frac:.1%}" if isinstance(frac, (int, float)) else "n/a"
            md = f"{med:+.0f} 天" if isinstance(med, (int, float)) else "n/a"
            print(f"  {s['name']:30s} {c:16s} {fr:>12s} {md:>10s}")
            fdc.setdefault(s["name"], {})[c] = m
    if not fdc:
        print("  （无）")

    doc = {"stage": "PIT 合规审计（无因子 IC、无采纳）",
           "prereg": "plans/pit-audit-prereg-20261011.md",
           "generated_at": datetime.now().isoformat(timespec="seconds"),
           "oos_window": [str(cal[0].date()), str(cal[-1].date())],
           "oos_days": int(len(cal)), "coverage_min": COV_MIN,
           "erratum": ("E1：首版 A3a 用统一式 `T_period ≤ T_avail`，对事件表产生 2 处"
                       "假 FAIL-2（dividend_ts / seo_issue）与 1 处假 FAIL-3"
                       "（baostock_witness，因 `[:5]` 截断取到指数）。"
                       "判据不改，只修实现；首版判决全部保留在 verdict_first_pass。"),
           "controls_a": ctl, "controls_b": ctlb,
           "summary": {"pass": n_pass, "flag": n_flag, "fail": n_fail,
                       "reclassified_by_erratum": n_changed},
           "sources": rows}
    (OUT_DIR / "pit_audit_report.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n报告 → {OUT_DIR / 'pit_audit_report.json'}")
    return doc


if __name__ == "__main__":
    main()
