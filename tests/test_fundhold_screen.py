"""判据链 #37 回归钉：`fund_hold_history` 截面因子筛选（防静默回退）。

预注册：`plans/fundhold-screen-prereg-20261011.md`
执行器：`scripts/fundhold_screen_chain.py`

本钉保护：
  1. ⭐ **G4 共线对照是真检验**（不得回退成"直接赋值 K2"那种同一性检验）；
  2. **滞后表沿用 #33**（季报 20 工作日 / 半年报 60 / 年报 90）；
  3. **候选清单冻结**（6 个）；门槛冻结；
  4. **仪器门方向正确**；**产物按 horizon 分文件**；
  5. **越界守卫**：不训练 / 不写 config / 不碰组合层 / 不改源 CSV / 不用 H1 体制口径。

只读。
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

import scripts.fundhold_screen_chain as chain  # noqa: E402

SRC = (ROOT / "scripts" / "fundhold_screen_chain.py").read_text(encoding="utf-8")
CSV = Path("D:/NEW ECO/data/h1_crowding_paper/fund_hold_history.csv")
needs_csv = pytest.mark.skipif(not CSV.exists(), reason="缺 fund_hold_history（跨仓数据）")


def _norm(src: str) -> str:
    return ast.unparse(ast.parse(src))


def _code(src: str) -> str:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and \
                    isinstance(body[0].value, ast.Constant) and \
                    isinstance(body[0].value.value, str):
                body[0].value.value = ""
    return ast.unparse(tree)


# ---------------------------------------------------------------- 清单与门槛

def test_candidate_list_frozen():
    assert len(chain.CAND_NAMES) == 6
    assert set(chain.CAND_NAMES) == {
        "fh_held_ratio", "fh_fund_cnt", "fh_held_ratio_chg",
        "fh_fund_cnt_chg", "fh_fund_cnt_rank", "fh_held_ratio_log"}


def test_thresholds_frozen():
    assert chain.F1_BONF == 2.64
    assert chain.COV_MIN == 0.50
    assert chain.INC_T == 2.0


def test_lag_table_matches_chain33():
    """滞后表必须与 #33 冻结值一致（季报 20 工作日 / 半年报 60 / 年报 90）。"""
    assert chain.LAG_BUSDAYS == {"0331": 20, "0930": 20}
    assert chain.LAG_CALDAYS == {"0630": 60, "1231": 90}
    for k, v in chain.LAG_BUSDAYS.items():
        assert v >= 15, f"{k} 滞后 {v} < 法定 15 工作日"


# ---------------------------------------------------------------- ⭐ G4 真检验

def test_g4_colinear_check_is_not_tautology():
    """⭐ K5 必须是 `fund_cnt` 的**截面秩**，不得直接赋值 K2。

    ⚠ 自伤登记（2026-10-11）：首版写 `out["fh_fund_cnt_rank"] = out["fh_fund_cnt"]`
    ⇒ G4 共线对照退化为**同一性检验**（比自身、恒真）⇒ 空检查、零信息。
    修法：K5 = `fund_cnt` 的逐日截面秩（取值确实不同 —— 实测 5311/5311 全不同，
    但秩化后完全相同 ⇒ 才是真检验）。

    ⚠ 本钉用 **AST 结构**判定（不用子串）：`ast.unparse` 会把 `"` 规范成 `'`，
    子串比对会假失败（#35/#36 同款自伤已出现三次）。
    """
    found = None
    for node in ast.walk(ast.parse(SRC)):
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            if isinstance(t, ast.Subscript):
                # 形如 out["fh_fund_cnt_rank"] = ...
                txt = ast.unparse(t)
                if "fh_fund_cnt_rank" in txt and txt.startswith("out["):
                    found = node.value
    assert found is not None, "未找到 K5 的定义"
    vt = ast.unparse(found)
    assert ".rank(" in vt, f"K5 未取截面秩（当前 = {vt[:80]}）⇒ G4 是同一性检验"
    assert "pct=True" in vt, "K5 未用 pct 秩"
    assert "自伤登记" in SRC, "缺 G4 自伤勘误说明"


@needs_csv
def test_g4_columns_differ_in_value_but_match_in_rank():
    """G4 的前提：K2/K5 **取值不同**但**秩化后相同**（否则门无分辨力）。"""
    fh = pd.read_csv(CSV, dtype={"ts_code": str})
    fh["fund_cnt"] = pd.to_numeric(fh["fund_cnt"], errors="coerce")
    g = fh[fh["period"] == fh["period"].max()].set_index("ts_code")["fund_cnt"].dropna()
    r = g.rank(pct=True)
    assert not np.allclose(g.to_numpy(), r.to_numpy()), \
        "K2 与 K5 取值相同 ⇒ G4 无法分辨（门失效）"
    assert np.allclose(g.rank(pct=True).to_numpy(), r.rank(pct=True).to_numpy()), \
        "秩化后不相同 ⇒ 秩等价性不成立"


# ---------------------------------------------------------------- 仪器门方向

def test_gate_directions():
    """门方向用 **AST 规范化**视图判定（引号/括号都会被 unparse 改动）。"""
    code = _norm(SRC)
    assert "G1 = not (" in code, "G1 断言方向不对"
    assert "G2 = g2['incr_t'] is not None and g2['incr_t'] >= INC_T" in code, \
        "G2 断言方向不对"
    assert "G3 = not (g3['incr_t'] is not None and g3['incr_t'] >= INC_T" in code, \
        "G3 断言方向不对"
    assert "G4 = e2['ic'] == e5['ic']" in code, "G4 断言缺失"


def test_gate_probes():
    code = _norm(SRC)
    assert "0.15 * np.nan_to_num" in code, "G2 未用合成因子探针"
    assert "bundle.features['size_log']" in code, "G3 未用 size_log 探针"


# ---------------------------------------------------------------- 产物管理（#35 教训）

def test_report_filename_is_horizon_specific():
    assert "fundhold_screen_report_h{H}.json" in SRC, \
        "报告未按 horizon 分文件 ⇒ 并列报告会互相覆盖"
    assert 'fundhold_screen_report.json"' not in SRC, "仍存在固定文件名写法"
    assert "--h" in SRC and "type=int" in SRC, "缺 --h 参数"


# ---------------------------------------------------------------- 越界守卫

def test_no_training_or_config_write():
    mods = set()
    for node in ast.walk(ast.parse(SRC)):
        if isinstance(node, ast.Import):
            for a in node.names:
                mods.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    assert "torch" not in mods, "出现 torch ⇒ 越界训练"
    assert "yaml" not in mods, "出现 yaml ⇒ 可能写 config"
    for node in ast.walk(ast.parse(SRC)):
        if isinstance(node, ast.Call):
            txt = ast.unparse(node)
            if "config.yaml" in txt:
                assert ast.unparse(node.func).endswith("load_config")


def test_source_csv_is_read_only():
    code = _code(SRC)
    for banned in ("to_parquet", "to_csv"):
        assert banned not in code, f"出现写文件调用：{banned}"


def test_no_h1_regime_logic():
    """越界：不得实现 H1 的体制/剔除口径（那是他仓在跑的前瞻判据）。"""
    code = _code(SRC)
    for banned in ("crowded", "regime_at", "neutralize_and_drop", "DROP_N"):
        assert banned not in code, f"出现 H1 体制口径符号：{banned}"


def test_no_portfolio_layer_or_adoption():
    code = _code(SRC).lower()
    for banned in ("sharpe", "max_drawdown", "total_return", "profit_factor",
                   "adoption_ready"):
        assert banned not in code, f"出现越界符号：{banned}"


def test_docstring_stripping_self_consistency():
    assert "sharpe" in _code("def f():\n    return {'sharpe': 1.0}\n").lower()
    assert "config.yaml" not in _code('def f():\n    """不改 config.yaml"""\n    return 1\n')
