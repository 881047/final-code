"""判据链 #33 回归钉：`fund_hold_history` 披露滞后升级（防静默回退）。

预注册：`plans/fundhold-lag-prereg-20261011.md`
执行器：`scripts/fundhold_pit_lag.py`
证据包：`experiments/fundhold_lag_20261011/`

本钉保护：
  1. **滞后表**（季报 20 工作日 / 半年报 60 日 / 年报 90 日）不得被改小或混用口径；
  2. **T_avail 严格晚于 period**（E 类铁律"次一交易日起可用"）；
  3. **G1/G4/G7 三个对照**必须能失败（防"门是空检查"）；
  4. **A3a 由机器判定**，不得人工指定 `a3a_ok`（#32 的 E1/E2 教训）。

只读：不访问老库、不取网络、不写任何数据。
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.fundhold_pit_lag import (  # noqa: E402
    LAG_BUSDAYS, LAG_CALDAYS, avail_dates, build_state, panel_calendar,
    panel_codes, uniform_lag_control,
)
from scripts.pit_source_audit import verdict  # noqa: E402

SRC = (ROOT / "scripts" / "fundhold_pit_lag.py").read_text(encoding="utf-8")
CSV = Path("D:/NEW ECO/data/h1_crowding_paper/fund_hold_history.csv")

pytestmark = pytest.mark.skipif(not CSV.exists(),
                                reason="fund_hold_history.csv 不存在（跨仓数据）")


# ---------------------------------------------------------------- 滞后表冻结

def test_lag_table_frozen():
    """冻结滞后表：季报 20 工作日、半年报 60 日、年报 90 日。"""
    assert LAG_BUSDAYS == {"0331": 20, "0930": 20}
    assert LAG_CALDAYS == {"0630": 60, "1231": 90}


def test_quarterly_lag_exceeds_legal_15_busdays():
    """季报必须 ≥ 法定 15 工作日（本批取 20，含 +5 缓冲）。"""
    for k, v in LAG_BUSDAYS.items():
        assert v >= 15, f"{k} 滞后 {v} 工作日 < 法定 15 ⇒ 可能前视"


def test_annual_meets_legal_90_days_and_semiannual_meets_60():
    assert LAG_CALDAYS["1231"] >= 90
    assert LAG_CALDAYS["0630"] >= 60


def test_string_literals_stripped_in_ast_view():
    """自证：AST 视图剔除字符串字面量（#32 的教训，防说明文字触发假失败）。"""
    tree = ast.parse(SRC)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = ""
    code = ast.unparse(tree)
    assert "法定" not in code, "字符串字面量未被剔除 ⇒ 元测量会误判"


# ---------------------------------------------------------------- T_avail 语义

def test_t_avail_strictly_after_period():
    """E 类铁律：T_avail 必须严格晚于 period。"""
    cal = panel_calendar()
    g = avail_dates(cal)
    assert (g["t_avail"] > g["period"]).all()


def test_t_avail_strictly_after_effective_date():
    """⭐ 真正的 E 类铁律：T_avail 必须**严格晚于法定期限日**（次一交易日起可用）。

    变异测试 M3 证明首版钉是**空检查**：只断言 `T_avail > period`，
    而滞后本身（20 工作日 / 60 日）保证该式恒真 ⇒ 把 `searchsorted(side="right")`
    改成 `side="left"` 仍然通过。
    实测：`eff` 恰为交易日时（8 抽样期中 5 期）两种 side 结果**不同**，
    差异正是"当日可用"与"次一交易日可用"—— 即该钉要防的正是这个。
    """
    cal = panel_calendar()
    g = avail_dates(cal)
    bad = g[~(g["t_avail"] > g["effective"])]
    assert bad.empty, (
        f"T_avail 未严格晚于法定期限日（违反次一交易日铁律）："
        f"{bad[['period', 'effective', 't_avail']].to_dict('records')}")


def test_t_avail_strictly_increasing():
    """不同报告期的 T_avail 必须严格递增。"""
    cal = panel_calendar()
    g = avail_dates(cal)
    d = g["t_avail"].diff().dropna()
    assert (d > pd.Timedelta(0)).all()


def test_all_four_report_types_present():
    cal = panel_calendar()
    g = avail_dates(cal)
    assert set(g["mmdd"]) == {"0331", "0630", "0930", "1231"}


def test_quarterly_lag_shorter_than_annual():
    """分类滞后：季报的实际滞后必须**短于**年报（否则分类没生效）。"""
    cal = panel_calendar()
    g = avail_dates(cal)
    q = g[g["kind"] == "busday"]["lag_days"].median()
    a = g[g["mmdd"] == "1231"]["lag_days"].median()
    assert q < a, f"季报滞后 {q} 未短于年报 {a}"


# ---------------------------------------------------------------- 仪器门

def test_g1_zero_lag_must_fail():
    """G1：零滞后（T_avail = period）必须 FAIL-2 —— 门必须能失败。"""
    cal = panel_calendar()
    codes = panel_codes()
    g = avail_dates(cal)
    s = build_state(g, cal, codes, t_avail_override=pd.DatetimeIndex(g["period"]),
                    require_legal=True)
    v, _ = verdict(s)
    assert v == "FAIL-2", f"零滞后被判 {v} ⇒ 门无分辨力"


def test_g2_correct_lag_passes():
    """G2：冻结滞后必须 PASS。"""
    cal = panel_calendar()
    g = avail_dates(cal)
    s = build_state(g, cal, panel_codes(), require_legal=True)
    v, _ = verdict(s)
    assert v == "PASS", f"冻结滞后被判 {v}"


def test_g3_discriminates():
    """G3：零滞后与正确滞后判决必须不同。"""
    cal = panel_calendar()
    codes = panel_codes()
    g = avail_dates(cal)
    v0, _ = verdict(build_state(g, cal, codes,
                                t_avail_override=pd.DatetimeIndex(g["period"]),
                                require_legal=True))
    v2, _ = verdict(build_state(g, cal, codes, require_legal=True))
    assert v0 != v2


def test_g4_knob_zero_is_not_pass_and_no_rebound():
    """G4：旋钮在 0 日必须非 PASS，且 PASS 后不得反弹回 FAIL-2（覆盖未跌破门槛时）。"""
    cal = panel_calendar()
    codes = panel_codes()
    g = avail_dates(cal)
    seq = []
    for k in (0, 5, 20, 60, 90, 200):
        ta = []
        for p in g["period"]:
            pos = cal.searchsorted(p + pd.Timedelta(days=k), side="right")
            ta.append(cal[min(pos, len(cal) - 1)])
        s = build_state(g, cal, codes, t_avail_override=pd.DatetimeIndex(ta),
                        require_legal=True)
        seq.append((k, verdict(s)[0], s["coverage"]))
    assert seq[0][1] != "PASS", "旋钮 0 日就 PASS ⇒ 门是空检查"
    first_pass = next((i for i, r in enumerate(seq) if r[1] == "PASS"), None)
    assert first_pass is not None, "滞后加到 200 日仍不 PASS"
    for k, v, cov in seq[first_pass:]:
        assert not (v == "FAIL-2" and cov >= 0.50), f"滞后 {k} 日反弹回 FAIL-2"


def test_g7_uniform_60_fails_90_passes():
    """G7：统一 60 日不过（须 ≥ 年报期限）、统一 90 日过。"""
    cal = panel_calendar()
    codes = panel_codes()
    g = avail_dates(cal)
    v60, _ = uniform_lag_control(g, cal, codes, 60)
    v90, _ = uniform_lag_control(g, cal, codes, 90)
    assert v60 != "PASS", "统一 60 日就通过 ⇒ 与年报 90 日期限矛盾"
    assert v90 == "PASS", "统一 90 日不过 ⇒ 法定期限口径有误"


# ---------------------------------------------------------------- #32 的教训

def test_a3a_is_computed_not_hand_assigned():
    """#32 的 E1/E2 教训：A3a 必须由数据算出，不得人工赋值。

    变异测试 M4（把 `a3a_ok = ok_legal` 改成 `= True`）本钉须**直接**抓到。
    """
    tree = ast.parse(SRC)
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build_state":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Dict):
                    keys = [k.value for k in sub.keys if isinstance(k, ast.Constant)]
                    if "a3a_ok" in keys:
                        val = sub.values[keys.index("a3a_ok")]
                        assert not (isinstance(val, ast.Constant)
                                    and val.value is True), \
                            "a3a_ok 被硬编码为 True ⇒ 门失效"
                        found = True
        if isinstance(node, ast.Assign):
            # 抓 `a3a_ok = ...` 的赋值：右边不得是裸 True
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "a3a_ok":
                    assert not (isinstance(node.value, ast.Constant)
                                and node.value.value is True), \
                        "a3a_ok 被硬编码为 True（赋值语句）⇒ 门失效"
                    found = True
    assert found, "未找到 a3a_ok 的判定逻辑 ⇒ 判定可疑"
    assert "n_early" in SRC, "缺少「早于期限」的计数 ⇒ 判定可疑"


def test_lag_classification_effective_in_output():
    """⭐ 分类滞后必须**真的生效**（不是只写在常量表里）。

    变异测试 M5（把半年报滞后改成 200 日）本钉须**直接**抓到：
    断言半年报的实际滞后日 < 年报的实际滞后日（M5 会让它反转）。
    """
    cal = panel_calendar()
    g = avail_dates(cal)
    semi = g[g["mmdd"] == "0630"]["lag_days"].median()
    ann = g[g["mmdd"] == "1231"]["lag_days"].median()
    assert semi < ann, (
        f"半年报实际滞后 {semi} 天 未短于 年报 {ann} 天 ⇒ 分类滞后未生效"
        f"（或常量表被改坏）")


def test_year_end_lag_is_actually_90_days():
    """年报的实际滞后必须 ≈ 法定 90 日（防常量表被悄悄改小）。"""
    cal = panel_calendar()
    g = avail_dates(cal)
    ann = g[g["mmdd"] == "1231"]["lag_days"]
    assert (ann >= 90).all(), f"年报实际滞后存在 < 90 的期：{ann.tolist()}"


def test_no_factor_ic_computed():
    """越界守卫：本批不算因子 IC。"""
    tree = ast.parse(SRC)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = ""
    code = ast.unparse(tree)
    for banned in ("spearmanr", "summarize_ic", "newey_west_t", "rank_ic"):
        assert banned not in code, f"出现禁用的因子评估符号：{banned}"


def test_source_csv_is_read_only():
    """不修改源文件：不得出现写 CSV 的调用。"""
    tree = ast.parse(SRC)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = ast.unparse(node.func)
            assert not f.endswith(".to_csv"), "出现写 CSV 调用"
