"""判据链 #35 回归钉：量价派生因子筛选（防静默回退）。

预注册：`plans/pv-screen-prereg-20261011.md`
执行器：`scripts/pv_screen_chain.py`

本钉保护：
  1. **候选清单冻结**（14 个，不得增删）；构造口径（后复权 / min_periods）；
  2. **判据门槛冻结**（Bonferroni 2.91 / ρ≤0.7 / 覆盖≥50% / 增量 t≥2）；
  3. **C5 主判据必须对全部现役列正交化**（不得偷工减料成正交化 1-2 列）；
  4. **仪器门必须能失败**：G2 阳性（合成因子）必须过、G3 影子（size_log）必须不过；
  5. **越界守卫**：不训练 / 不改 config / 不碰组合层 / 不采纳。

只读：不写任何数据文件。
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

import scripts.pv_screen_chain as chain  # noqa: E402

SRC = (ROOT / "scripts" / "pv_screen_chain.py").read_text(encoding="utf-8")
PANEL = ROOT / "data" / "panel" / "close.parquet"
needs_data = pytest.mark.skipif(not PANEL.exists(), reason="缺面板（数据层）")


def _docstring_stripped(src: str) -> str:
    """剔 docstring，保留普通字符串（#34 教训：越界守卫要能看见字符串）。"""
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
    """候选清单冻结为 14 个（预注册 §2.1）。"""
    assert len(chain.CAND_NAMES) == 14
    assert set(chain.CAND_NAMES) == {
        "mom_120", "mom_250", "high_52w", "low_52w", "ret_max20", "skew_20",
        "bias_20", "bias_60", "illiq", "vol_60", "ep", "bp", "dv_ttm", "beta_250"}


def test_thresholds_frozen():
    """门槛冻结：Bonferroni 2.91（14 个）、ρ≤0.7、覆盖≥50%、增量 t≥2、h=5。"""
    assert chain.N_TESTS == 14
    assert chain.BONF == 2.91
    assert chain.RHO_MAX == 0.70
    assert chain.COV_MIN == 0.50
    assert chain.INC_T == 2.0
    assert chain.H == 5


def test_c5_orthogonalizes_against_all_existing_columns():
    """⭐ C5 主判据必须对**全部现役列**正交化，不得只对少数列。

    这是 A1 教训（"有信息"≠"有增量"）的落地；若偷懒只正交化 1-2 列，
    `illiq` 会假过（实测：正交 3 列 +9.7bp t=2.29，正交 25 列 +2.2bp t=1.14）。
    """
    code = _docstring_stripped(SRC)
    assert "CTRL = [bundle.features[c] for c in bundle.feature_cols]" in code or \
           "for c in bundle.feature_cols" in code, \
        "C5 未使用全部 feature_cols 作控制列"
    assert "orth_residual(F, CTRL" in code, "C5 未调用 orth_residual(…, CTRL)"

    # 变异测试 M-1 模拟：把 CTRL 换成前 3 列 ⇒ 本钉须能抓
    poisoned = SRC.replace(
        "CTRL = [bundle.features[c].reindex(index=cal, columns=codes)\n"
        "            for c in bundle.feature_cols]",
        "CTRL = [bundle.features[c].reindex(index=cal, columns=codes)\n"
        "            for c in bundle.feature_cols[:3]]")
    assert poisoned != SRC, "锚点未找到 ⇒ 本钉的变异自证失效"
    assert "for c in bundle.feature_cols[:3]]" not in SRC, "控制列被截断"


# ---------------------------------------------------------------- 构造口径

@needs_data
def test_construction_is_pit_and_correct():
    """构造口径：`beta_250` 必须用 min_periods=120（修 §0.3 的构造失败）。"""
    code = _docstring_stripped(SRC)
    assert "min_periods=120" in code, "beta_250/high_52w 缺 min_periods ⇒ 会构造失败"
    assert "cov(mkt)" in code, "beta_250 未用等权市场收益"


@needs_data
def test_beta_250_constructs_with_valid_days():
    """`beta_250` 必须有有效截面（#0.3 标定时为 0 日 ⇒ 本批已修）。"""
    from fterm.config import load_config
    from fterm.data import ashare as sa
    cfg = load_config(ROOT / "config.yaml")
    panels = sa.load_panels(cfg)
    codes = list(panels["close"].columns[:400])
    cal = panels["close"].index[-300:]
    cand = chain.build_candidates(cfg, panels, codes, cal)
    b = cand["beta_250"]
    valid = int(b.notna().any(axis=1).sum())
    assert valid > 50, f"beta_250 有效日仅 {valid} ⇒ 构造仍失败"


# ---------------------------------------------------------------- 仪器门

def _synth(n_d=80, n_c=800, seed=0):
    """合成面板：n_c 必须 > 500，否则 `incr_test` 的 `good.sum() < 500` 门槛会全跳过。

    ⚠ 首版用 300 只/天 ⇒ `n=0` ⇒ 测试"失败"其实是**合成数据尺寸不足**（测试 bug）。
    """
    rng = np.random.default_rng(seed)
    cal = pd.DatetimeIndex([pd.Timestamp("2020-01-01") + pd.Timedelta(days=i)
                            for i in range(n_d)])
    return rng, cal


def test_incr_test_can_fail():
    """增量检验必须**能失败**：对纯噪声因子，增量不该显著为正。"""
    rng, cal = _synth(seed=0)
    n_d, n_c = len(cal), 800
    noise = rng.standard_normal((n_d, n_c))
    Yv = rng.standard_normal((n_d, n_c)) * 0.02       # 收益率量级
    Pmr = rng.standard_normal((n_d, n_c))
    M = np.ones((n_d, n_c), dtype=bool)
    bp, t, n = chain.incr_test(noise, Yv, Pmr, M, cal)
    assert n > 0, "增量检验未产出观测（合成尺寸不足？）"
    assert t is None or t < 2.0, f"纯噪声因子增量 t={t} ≥ 2 ⇒ 检验无效"


def test_incr_test_detects_real_signal():
    """增量检验必须**能通过**：对真正携带 fwd 信息的因子应显著（G2 阳性对照同型）。"""
    rng, cal = _synth(seed=1)
    n_d, n_c = len(cal), 800
    Yv = rng.standard_normal((n_d, n_c)) * 0.02
    Pmr = rng.standard_normal((n_d, n_c))
    M = np.ones((n_d, n_c), dtype=bool)
    # 因子含半数 fwd 信息 ⇒ 在 top-10% 内应能挑出更强的
    real = 0.5 * Yv + rng.standard_normal((n_d, n_c)) * 0.02
    bp, t, n = chain.incr_test(real, Yv, Pmr, M, cal)
    assert n > 0, "增量检验未产出观测"
    assert bp > 0, f"真信号增量为负（bp={bp}）"
    assert t is not None and t >= 2.0, f"真信号未通过增量检验（t={t}）"


def _norm(src: str) -> str:
    """AST 规范化（统一引号/空格），用于**稳健**的子串比对。

    ⚠ 首版钉直接拿未规范化的源码子串比对，被 `ast.unparse` 的引号规范化
    （`"` → `'`）判失败 ⇒ 是**测试 bug**。本函数对两侧都做规范化。
    """
    return ast.unparse(ast.parse(src))


def test_gates_present_and_must_be_able_to_fail():
    """G1/G2/G3 三门在源码中存在，且断言方向正确。"""
    assert "G1 阴性" in SRC and "G2 阳性" in SRC and "G3 影子对照" in SRC
    code = _norm(SRC)
    # G2 必须断言「过」（增量 t ≥ 门槛且 bp>0）
    assert "G2 = g2['incr5_t'] is not None and g2['incr5_t'] >= INC_T" in code, \
        "G2 断言方向不对"
    # G3 必须断言「不过」（取反）
    assert "G3 = not (g3['incr5_t'] is not None and g3['incr5_t'] >= INC_T" in code, \
        "G3 断言方向不对"


def test_gate_g3_uses_size_log():
    """G3 影子对照必须用**已在现役里**的因子（size_log）作探针。"""
    code = _norm(SRC)
    assert "g3 = evaluate(bundle.features['size_log'])" in code, \
        "G3 未用 size_log 作影子探针"


# ---------------------------------------------------------------- 越界守卫

def test_no_training_or_config_write():
    """越界：不训练模型、不写 config。"""
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
                assert ast.unparse(node.func).endswith("load_config"), \
                    f"对 config.yaml 的非读取调用：{txt[:60]}"


def test_no_portfolio_layer_or_adoption():
    """越界：不碰组合层、不采纳。"""
    code = _docstring_stripped(SRC).lower()
    for banned in ("sharpe", "max_drawdown", "total_return", "profit_factor",
                   "adoption_ready"):
        assert banned not in code, f"出现越界符号：{banned}"


def test_self_consistency_of_docstring_stripping():
    """自证：普通字符串必须**保留**（否则越界守卫变空检查，#34 的 M-H 教训）。"""
    poison = "def f():\n    return {'sharpe': 1.0}\n"
    assert "sharpe" in _docstring_stripped(poison).lower()
    doc = 'def f():\n    """说明：不改 config.yaml"""\n    return 1\n'
    assert "config.yaml" not in _docstring_stripped(doc)


# ---------------------------------------------------------------- 产物管理（自伤防回退）

def test_report_filename_is_horizon_specific():
    """⭐ 产物文件名必须带 horizon（防并列报告互相覆盖）。

    自伤登记（2026-10-11）：首版固定写 `pv_screen_report.json` ⇒ 跑 h=1 并列报告时
    **覆盖**了 h=5 主报告 ⇒ 证据包 provenance 丢失。
    """
    assert "pv_screen_report_h{H}.json" in SRC, \
        "报告文件名未带 horizon ⇒ 并列报告会互相覆盖"
    assert 'pv_screen_report.json"' not in SRC, "仍存在固定文件名写法"
    assert "--h" in SRC and "type=int" in SRC, "缺 --h 参数 ⇒ 无法跑并列报告"


def test_h_parameter_actually_changes_H():
    """h 参数必须真的改变 H（而不是只打印）。"""
    found = False
    for node in ast.walk(ast.parse(SRC)):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign):
                    for t in sub.targets:
                        if isinstance(t, ast.Name) and t.id == "H":
                            found = True
                            assert "args.h" in ast.unparse(sub.value), "H 未被 --h 赋值"
    assert found, "main 里未把 --h 赋给 H"
