"""判据链 #32 回归钉：PIT 审计器的判据机 + 口径修正（防静默回退）。

预注册：`plans/pit-audit-prereg-20261011.md`
执行器：`scripts/pit_source_audit.py`
证据包：`experiments/pit_audit_20261011/`

本钉保护三件事：
  1. **判决优先级**（§2.3）：FAIL-2 > FAIL-1 > FAIL-3 > FLAG-2 > FLAG-3 > FLAG-1 > PASS；
  2. **E1/E2 的两次自伤修法**不得回退（事件表按 E 类自洽式、辅助列按阈值降级、
     代码对齐不得截断）；
  3. **A0a/A0b 仪器门**必须能失败（CTL-1..5 判决符合预期）。

只读：不访问老库、不取网络、不依赖 data/。
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.pit_source_audit import (  # noqa: E402
    controls, controls_b, run_controls, verdict,
)

SRC = (ROOT / "scripts" / "pit_source_audit.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------- 判决优先级

def test_fail1_beats_everything_when_no_time_col():
    """无时点列 ⇒ FAIL-1（哪怕其他项都过）。"""
    v, _ = verdict({"a1_has_time_col": False, "a3a_ok": True, "a4_ok": True,
                    "a5_ok": True, "a3b_ok": True, "a3c": "safe"})
    assert v == "FAIL-1"


def test_fail2_beats_flags():
    """结构前视优先于一切 FLAG。"""
    v, _ = verdict({"a1_has_time_col": True, "a2_semantics": "E", "a3a_ok": False,
                    "a4_ok": True, "a5_ok": True, "a3b_ok": False, "a3c": "unsafe"})
    assert v == "FAIL-2"


def test_priority_fail1_over_fail3():
    """FAIL-1（无时点）优先于 FAIL-3（覆盖不足）。"""
    v, _ = verdict({"a1_has_time_col": False, "a4_ok": False, "a5_ok": False})
    assert v == "FAIL-1"


def test_flag2_beats_flag3_and_flag1():
    v, _ = verdict({"a1_has_time_col": True, "a2_semantics": "P", "a3a_ok": True,
                    "a4_ok": True, "a5_ok": True, "a3b_ok": False, "a3c": "unsafe"})
    assert v == "FLAG-2"


def test_flag3_beats_flag1():
    """锚点不可判定（FLAG-3）优先于仅期间列（FLAG-1）。"""
    v, _ = verdict({"a1_has_time_col": True, "a2_semantics": "P", "a3a_ok": True,
                    "a4_ok": True, "a5_ok": True, "a3b_ok": True,
                    "a3c": "indeterminate"})
    assert v == "FLAG-3"


def test_flag1_when_only_period_column():
    v, _ = verdict({"a1_has_time_col": True, "a2_semantics": "P", "a3a_ok": True,
                    "a4_ok": True, "a5_ok": True, "a3b_ok": True, "a3c": "safe"})
    assert v == "FLAG-1"


def test_pass_requires_all():
    v, _ = verdict({"a1_has_time_col": True, "a2_semantics": "E", "a3a_ok": True,
                    "a4_ok": True, "a5_ok": True, "a3b_ok": True, "a3c": "safe"})
    assert v == "PASS"


def test_semantics_x_is_fail1():
    v, _ = verdict({"a1_has_time_col": True, "a2_semantics": "X"})
    assert v == "FAIL-1"


# ---------------------------------------------------------------- 仪器门

def test_a0a_controls_expected():
    """CTL-1/2 必须非 PASS、CTL-3 必须 PASS。"""
    ctl = run_controls()
    assert ctl["all_controls_ok"] is True
    got = {r["name"]: r["verdict"] for r in ctl["rows"]}
    assert got["CTL-1 完美前视"] == "FAIL-2"
    assert got["CTL-2 时点错配"] == "FAIL-2"
    assert got["CTL-3 合法滞后"] == "PASS"


def test_a0a_can_fail():
    """A0a 必须**能**失败：若把 CTL-1 的 a3a_ok 翻成 True，它就不再 FAIL-2。"""
    rows = controls()
    ctl1 = [r for r in rows if r["name"].startswith("CTL-1")][0]
    assert verdict(ctl1)[0] == "FAIL-2"
    mutated = dict(ctl1, a3a_ok=True, a3c="safe", a2_semantics="D")
    mutated["a3a_detail"] = "（变异测试：假装无前视）"
    assert verdict(mutated)[0] == "PASS", "门没有分辨力：这是空检查"


def test_a0b_discriminates_event_tables():
    """A0b：合规/前视两样本必须给出**不同**判决（口径有分辨力）。"""
    ctlb = controls_b()
    assert ctlb["discriminates"] is True
    assert ctlb["all_controls_ok"] is True
    got = {r["name"]: r["verdict"] for r in ctlb["rows"]}
    assert got["CTL-4 事件表合规（可用时点=ann_date）"] == "PASS"
    assert got["CTL-5 事件表前视（可用时点=end_date）"] == "FAIL-2"


# ---------------------------------------------------------------- E1/E2 修法不得回退

def _dict_value_in_func(func_name: str, key: str) -> str | None:
    """取某函数返回 dict 里 `key` 的**值表达式**源码（AST 级）。"""
    for node in ast.walk(ast.parse(SRC)):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            for sub in ast.walk(node):
                if not isinstance(sub, ast.Dict):
                    continue
                keys = [k.value for k in sub.keys if isinstance(k, ast.Constant)]
                if key in keys:
                    return ast.unparse(sub.values[keys.index(key)])
    return None


def test_e1_1_dividend_uses_effective_date_not_report_period():
    """E1-1：div 的 A3a 不得再用 end_date（报告期）与 ann_date 比较。

    ⚠ 变异测试 M4 证明首版钉是**空检查**：只查 `ex < ann` 子串存在，
    却没查 `a3a_ok` 是否**消费**该判定 ⇒ 把 `a3a_ok: viol_ex == 0` 改成
    `True` 仍然通过。⇒ 改为断言**值表达式引用 viol_ex**。
    """
    code = _code_only(SRC)
    assert "ann < end" not in code, "A3a 回退成 ann_date vs end_date 比较"
    assert "ex < ann" in code, "主不变量 ex_date ≥ ann_date 检查缺失"
    v = _dict_value_in_func("audit_dividend", "a3a_ok")
    assert v is not None, "audit_dividend 未返回 a3a_ok"
    assert "viol_ex" in v, \
        f"a3a_ok 未消费主不变量判定（当前值表达式 = {v}）⇒ E2 可静默回退"
    assert "首版误用 end_date" in SRC, "勘误说明被删除 ⇒ 无法追溯"


def test_e2_aux_columns_have_declared_threshold():
    """E2：辅助列（record_date/pay_date）必须走**声明阈值**，不得零容忍。"""
    code = _code_only(SRC)
    assert "tol = 0.001 * len(df)" in code, "辅助列阈值声明缺失 ⇒ E2 会回退"
    v = _dict_value_in_func("audit_dividend", "a3a_ok")
    assert v is not None and "viol_ex" in v, "主不变量必须保持零容忍"


def test_e1_2_seo_not_fail_on_field_mislabel():
    """E1-2：seo_issue 的 6 行字段混填不得作为 FAIL 依据。"""
    assert "data_quality_note" in SRC
    assert "a3a_ok\": True" in SRC or 'a3a_ok": True' in SRC


def _code_only(src: str) -> str:
    """去掉注释/docstring/**字符串字面量**，只留真正的可执行表达式。

    ⚠ 记忆 `acceptance-bar-calibrate-first`：毒样本字面量、甚至**说明文字本身**
    都会触发过宽的字符串规则 ⇒ 元测量必须 AST 实现。
    本钉首版正是被勘误说明文字里的 `[:5]` 触发了**两次**假失败：
      ① docstring 里的说明；
      ② 报告 doc 里披露勘误的**字符串字面量**。
    ⇒ 必须把 `ast.Constant(str)` 也剔除，只检查真正的切片/调用表达式。
    """
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = ""
    return ast.unparse(tree)


def _head_truncations_of_sorted(src: str) -> list[str]:
    """找出"对 sorted(...) 结果取头部切片"的写法（E1-3 的缺陷形态）。"""
    hits = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Subscript):
            continue
        sl = node.slice
        if not (isinstance(sl, ast.Slice) and sl.lower is None
                and isinstance(sl.upper, ast.Constant)):
            continue
        v = node.value
        if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) \
                and v.func.id == "sorted":
            hits.append(ast.unparse(node))
    return hits


def _defined_names(src: str) -> set[str]:
    """源码里所有**定义的标识符名**（含函数/类/赋值目标）。"""
    names = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
    return names


def test_e1_3_no_truncated_code_sampling():
    """E1-3：不得对有序代码列表取头部切片（会取到指数）⇒ 假 FAIL-3。

    ⚠ 变异测试 M5 证明首版钉是**空检查**：用 `"_looks_stock" in SRC` 子串判定，
    而 `_looks_stock` 是 `_looks_stock_disabled` 的**子串** ⇒ 把函数改名即绕过。
    ⇒ 改为断言**精确标识符**存在于定义名集合。
    """
    code = _code_only(SRC)
    assert _head_truncations_of_sorted(SRC) == [], \
        "出现 sorted(...) 头部切片 ⇒ 截断采样回退"
    assert "_looks_stock" in _defined_names(SRC), \
        "指数过滤函数 `_looks_stock` 不存在（注意：必须精确名，子串不算）"


def test_e1_3_pin_is_ast_based_not_string_match():
    """自证：本钉必须无视**字符串字面量**里的 `[:5]`（否则会假失败两次）。

    这个自证本身是本批最有价值的钉子之一：它证明"测量器不会把自己的
    说明文字当成违规"，正是记忆里那条元测量教训的直接落地。
    """
    assert "[:5]" in SRC, "前提：源码/说明里确实留有 [:5] 的勘误文字"
    assert "[:5]" not in _code_only(SRC), \
        "AST 视图里仍残留 [:5] ⇒ 剔除字符串字面量失败 ⇒ 会造假失败"
    # 真缺陷（对 sorted 结果取头部切片）仍必须被抓到
    poison = "codes = sorted(x)[:5]"
    assert _head_truncations_of_sorted(poison) != [], "真缺陷漏检 ⇒ 门是空检查"


def test_first_pass_verdicts_preserved():
    """披露纪律：首版判决必须保留在 FIRST_PASS（不静默改判）。"""
    assert "FIRST_PASS" in SRC
    tree = ast.parse(SRC)
    names = [n.targets[0].id for n in ast.walk(tree)
             if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)]
    assert "FIRST_PASS" in names
    assert 'FIRST_PASS = {' in SRC


# ---------------------------------------------------------------- 越界守卫（预注册 §四）

def test_audit_does_not_compute_factor_ic():
    """本链**只做审计**：不得出现因子 IC / 采纳逻辑。"""
    code = _code_only(SRC)
    for banned in ("spearmanr", "summarize_ic", "rank_ic", "newey_west_t",
                   "pbo_cscv", "deflated_sharpe"):
        assert banned not in code, f"审计器出现了禁用的因子评估符号：{banned}"


def test_audit_never_writes_to_old_db():
    """老库必须只读（mode=ro&immutable=1），且不得出现写语句。"""
    code = _code_only(SRC)
    assert "mode=ro" in SRC
    assert "immutable=1" in SRC
    for banned in ("INSERT", "UPDATE", "DELETE", "DROP", "CREATE"):
        assert banned not in code, f"审计器出现写语句：{banned}"
