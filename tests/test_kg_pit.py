"""判据链 #36 回归钉：kg_features PIT 解除 + 18 列筛选（防静默回退）。

预注册：`plans/kg-pit-screen-prereg-20261011.md`
执行器：`scripts/kg_pit_chain.py`

本钉保护：
  1. **K4 必须是值级检验**（不得回退成"事件表含未来事件"那种必然 FAIL 的写法）；
  2. **候选清单冻结**（18 个新 + 5 个第五轮对照）；
  3. **门槛冻结**（K5 Bonferroni 2.98 / 覆盖 50% / 增量 t≥2）；
  4. **K7 必须正交化全部现役列**；
  5. **仪器门方向正确**（G1 全不过 / G2 必过 / G3 必不过）；
  6. **越界守卫**：不训练 / 不写 config / 不碰组合层 / 不改 kg parquet（只读）。

只读：不写数据。
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.kg_pit_chain as chain  # noqa: E402

SRC = (ROOT / "scripts" / "kg_pit_chain.py").read_text(encoding="utf-8")
KG = Path("D:/NEW ECO/data/kg/kg_features.parquet")
needs_kg = pytest.mark.skipif(not KG.exists(), reason="缺 kg_features（跨仓数据）")


def _norm(src: str) -> str:
    """AST 规范化（引号统一）—— #35 教训：不得拿未规范化子串比对。"""
    return ast.unparse(ast.parse(src))


def _code(src: str) -> str:
    """剔 docstring、保留普通字符串 —— #34 教训：越界守卫要能看见字符串。"""
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

def test_candidate_lists_frozen():
    """候选清单冻结：18 个新 + 5 个第五轮对照。"""
    assert len(chain.KG_COLS) == 18, f"新候选应为 18，实为 {len(chain.KG_COLS)}"
    assert len(chain.KG_ALREADY) == 5
    assert "kg_holder_conc_chg" in chain.KG_COLS
    assert "kg_pagerank_ind" in chain.KG_COLS


def test_thresholds_frozen():
    assert chain.K5_BONF == 2.98, "Bonferroni 门槛被改"
    assert chain.COV_MIN == 0.50
    assert chain.INC_T == 2.0
    assert chain.H == 5


def test_pit_verify_columns_frozen():
    """K1 的实证列冻结为 4 个（可从老库独立重算者）。"""
    assert set(chain.KG_VERIFY) == {
        "kg_rating_up_60d", "kg_catalyst_60d", "kg_holder_add_60d", "kg_sentiment_60d"}


# ---------------------------------------------------------------- K4 值级（自伤防回退）

def test_k4_is_value_level_not_event_level():
    """⭐ K4 必须是**值级**检验。

    首版 K4 测「事件表里存在晚于锚点的事件」⇒ 这**必然**成立（事件表是全史的）
    ⇒ 假 FAIL（79 个锚点）。修法 = 断言**特征值**可由「仅用 dt ≤ 锚点 的事件」复现。
    """
    code = _code(SRC)
    assert "值级" in SRC, "缺 K4 值级勘误说明"
    assert "n_value_mismatch" in code, "K4 未做值级比对"
    assert "ev['dt'] <= at" in code or 'ev["dt"] <= at' in code, "K4 未限制 dt ≤ 锚点"
    # 不得回退成"事件表含未来事件即 FAIL"
    assert "n_future_violation" not in code, "K4 回退成事件级检验（必然 FAIL）"
    assert "anchors_with_future_events" not in code, "K4 回退成事件级口径"


def test_erratum_k4_recorded():
    """勘误须在报告中留痕（不得静默改判）。"""
    assert "erratum_K4" in SRC, "K4 勘误未登记"
    assert "假 FAIL" in SRC


# ---------------------------------------------------------------- 稀疏度诊断

@needs_kg
def test_sparsity_diagnosis_exists():
    """执行器须能报告稀疏度（否则会把"空列/极稀疏列"当成"没信息"）。"""
    assert "cov" in SRC
    assert "to_numpy().mean()" in SRC or "notna().to_numpy().mean()" in SRC


# ---------------------------------------------------------------- K7 用全部现役列

def test_k7_orthogonalizes_all_existing_columns():
    """⭐ K7 主判据必须对**全部现役列**正交化，不得切片截断。

    ⚠ 变异测试 M-5 证明首版钉是**空检查**：断言子串
    `"for c in bundle.feature_cols"` **是否存在**，而把 `bundle.feature_cols`
    改成 `bundle.feature_cols[:3]` 后该子串**仍是前缀** ⇒ 钉看不到。
    （与 #34 的 M5 同型：子串匹配被前缀绕过。）
    ⇒ 改为 **AST 检查**：遍历 `bundle.feature_cols` 的 for 语句，其 iter 不得是 Subscript。
    """
    code = _code(SRC)
    assert "CTRL_STACK" in code, "缺预计算控制矩阵"
    assert "orth_resid_fast" in code, "K7 未调用正交化"

    found = False
    for node in ast.walk(ast.parse(SRC)):
        if not isinstance(node, ast.For):
            continue
        it = ast.unparse(node.iter)
        if "bundle.feature_cols" not in it:
            continue
        found = True
        assert not isinstance(node.iter, ast.Subscript), \
            f"控制列被切片截断：for c in {it} ⇒ K7 未用全部现役列"
        assert it.strip() == "bundle.feature_cols", \
            f"控制列来源被改动：{it}"
    assert found, "未找到遍历 bundle.feature_cols 的 for 语句"


# ---------------------------------------------------------------- 仪器门

def test_gate_directions():
    """G1 全不过 / G2 必过 / G3 必不过（AST 规范化后比对）。

    ⚠ 必须在 `_norm` 视图上比对：`ast.unparse` 把 `"` 规范成 `'`
    （#35 同款测试自伤，本批已避免）。
    """
    code = _norm(SRC)
    assert "G1 = not (" in code, "G1 断言方向不对"
    assert "G2 = g2['incr7_t'] is not None and g2['incr7_t'] >= INC_T" in code, \
        "G2 断言方向不对"
    assert "G3 = not (g3['incr7_t'] is not None and g3['incr7_t'] >= INC_T" in code, \
        "G3 断言方向不对"


def test_gate_probes_are_correct():
    """G2 用合成因子、G3 用 size_log（已在现役里的影子）。"""
    code = _norm(SRC)
    assert "0.15 * np.nan_to_num" in code, "G2 未用合成因子探针"
    assert "bundle.features['size_log']" in code, "G3 未用 size_log 探针"


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


def test_kg_parquet_is_read_only():
    """不得写 kg_features.parquet（只读出）。"""
    code = _code(SRC)
    for banned in ("to_parquet", "to_csv"):
        assert banned not in code, f"出现写文件调用：{banned}"


def test_no_portfolio_layer_or_adoption():
    code = _code(SRC).lower()
    for banned in ("sharpe", "max_drawdown", "total_return", "profit_factor",
                   "adoption_ready"):
        assert banned not in code, f"出现越界符号：{banned}"


def test_docstring_stripping_self_consistency():
    """自证：普通字符串保留、docstring 剔除（#34 的 M-H 教训）。"""
    assert "sharpe" in _code("def f():\n    return {'sharpe': 1.0}\n").lower()
    assert "config.yaml" not in _code('def f():\n    """不改 config.yaml"""\n    return 1\n')
