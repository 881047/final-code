"""判据链 #34 回归钉：行业中性化标签链（防静默回退）。

预注册：`plans/industry-label-prereg-20261011.md`
执行器：`scripts/industry_label_chain.py`

本钉保护：
  1. **中性标签构造**（预注册 §2.1）：逐日截面减**同行业当日均值**；
     无行业者**保留原值**（不得改置 NaN）；小行业（<3 只）不中性化；
  2. **G1 标签自洽**：组内均值 ≈ 0（这是"中性化真的生效"的唯一证据）；
  3. **G6 折纯度**：净化 ≥ h+1 日；
  4. **G7 门能失败**：M1（不中性化）/M2（shift 前视）必须被 G1 抓；
  5. **越界守卫**：不改 config.yaml / 不采纳 / 不碰组合层。

只读：训练用合成数据 + 真实面板（不写生产文件）。
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

import scripts.industry_label_chain as chain  # noqa: E402

SRC = (ROOT / "scripts" / "industry_label_chain.py").read_text(encoding="utf-8")
SW = ROOT / "data" / "external" / "sw_l1.parquet"
PANEL = ROOT / "data" / "panel" / "close.parquet"

needs_data = pytest.mark.skipif(not (SW.exists() and PANEL.exists()),
                                reason="缺 sw_l1 / 面板（数据层）")


def _code_only(src: str) -> str:
    """剔除注释与**字符串字面量**（#32/#33 教训：说明文字会触发自己的规则）。"""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = ""
    return ast.unparse(tree)


# ---------------------------------------------------------------- 构造（预注册 §2.1）

def _func_source(name: str) -> str:
    """取某函数的源码段（用于检查其**内部**行为）。"""
    tree = ast.parse(SRC)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"未找到函数 {name}")


def test_no_industry_keeps_original_value():
    """⭐ 无行业代码者必须**保留原值**，不得置 NaN（预注册 §2.1 冻结）。

    这条是 #34 最关键的口径：两种构造下配对差 +0.0004(t=0.46) vs −0.0044(t=−2.44)。

    ⚠ 变异测试 M-C 证明首版钉是**空检查**：只查 `out = Y.copy()` 子串**是否存在**，
    而变异是**追加**一行把无行业格置 NaN ⇒ 子串仍在 ⇒ 钉看不到。
    ⇒ 改为 AST 检查 `build_neutral_labels` 体内**不得对 out 赋 NaN**。
    """
    body = _func_source("build_neutral_labels")
    assert "out = Y.copy()" in body, "中性标签未以 copy(Y) 起始 ⇒ 无行业者可能被置 NaN"
    # 不得出现把 out 的切片赋成 NaN 的写法
    for node in ast.walk(ast.parse(SRC)):
        if isinstance(node, ast.FunctionDef) and node.name == "build_neutral_labels":
            for sub in ast.walk(node):
                if not isinstance(sub, ast.Assign):
                    continue
                for t in sub.targets:
                    txt = ast.unparse(t)
                    if txt.startswith("out") and "nan" in ast.unparse(sub.value).lower():
                        raise AssertionError(
                            f"build_neutral_labels 把 out 置 NaN（破坏 §2.1）: {txt}")
    assert chain.MIN_IND_SIZE == 3, "小行业阈值未冻结"


def test_small_industry_threshold_is_three():
    assert chain.MIN_IND_SIZE == 3


def test_lambda_and_cap_frozen():
    assert chain.RIDGE_LAMBDA == 10.0
    assert chain.CAP_PER_DAY == 500


# ---------------------------------------------------------------- G1 标签自洽

@needs_data
def test_g1_neutralization_actually_zeroes_group_mean():
    """G1：中性化后每个 (日,行业) 组内均值必须 ≈ 0。"""
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"])
    codes = pd.Index([f"{i:06d}.SZ" for i in range(1, 61)])
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(3, 60))
    SWv = np.tile(np.repeat(np.arange(1, 7), 10).astype(float), (3, 1))  # 6 行业 × 10 只
    sw = pd.DataFrame(SWv, index=idx, columns=codes)

    class FakeBundle:
        universe = pd.DataFrame(True, index=idx, columns=codes)
        labels = {"fwd_ret": pd.DataFrame(Y, index=idx, columns=codes)}

    class FakeCfg:
        class features:
            label_horizon = 5

    out, _, _ = chain.build_neutral_labels(FakeCfg, FakeBundle, sw, 5)
    for i in range(3):
        df = pd.DataFrame({"g": SWv[i], "y": out[i]})
        gm = df.groupby("g")["y"].mean()
        assert np.abs(gm.to_numpy()).max() < 1e-9, "组内均值非 0 ⇒ 中性化未生效"


def test_g1_can_fail_when_not_neutralized():
    """G1 必须**能失败**：不中性化时组内均值显著 ≠ 0。"""
    idx = pd.DatetimeIndex(["2024-01-02"])
    codes = pd.Index([f"{i:06d}.SZ" for i in range(1, 61)])
    rng = np.random.default_rng(1)
    Y = rng.normal(size=(1, 60))
    SWv = np.repeat(np.arange(1, 7), 10).astype(float).reshape(1, -1)
    df = pd.DataFrame({"g": SWv[0], "y": Y[0]})
    gm = df.groupby("g")["y"].mean()
    assert np.abs(gm.to_numpy()).max() > 1e-6, "未中性化却组内均值为 0 ⇒ 测试无意义"


# ---------------------------------------------------------------- 折纯度

def test_fold_purity_enforced():
    """G6：折的训练末必须早于测试始，且间隔 ≥ 净化天数。"""
    dates = [pd.Timestamp("2020-01-01") + pd.Timedelta(days=i) for i in range(400)]
    folds = chain.make_folds(dates, 4, 6)
    assert folds, "未生成折"
    for f in folds:
        assert f["train"][1] < f["test"][0], "训练段与测试段重叠 ⇒ 泄漏"
        gap = dates.index(f["test"][0]) - dates.index(f["train"][1])
        assert gap >= 6, f"间隔 {gap} < 净化 6"


def test_fold_purity_can_fail():
    """G6 必须能失败：净化 0 时不得通过 ≥6 的检查。"""
    dates = [pd.Timestamp("2020-01-01") + pd.Timedelta(days=i) for i in range(400)]
    folds = chain.make_folds(dates, 4, 0)
    gaps = [dates.index(f["test"][0]) - dates.index(f["train"][1]) for f in folds]
    assert min(gaps) < 6, "净化 0 却仍有 ≥6 间隔 ⇒ 检查是空转"


# ---------------------------------------------------------------- G7 门自检

@needs_data
def test_g7_selftest_gates_catch_mutations():
    """G7：M1（不中性化）与 M2（shift 前视）必须被 G1 抓。"""
    import json
    sw = pd.read_parquet(SW)
    cfg_mod = __import__("fterm.config", fromlist=["load_config"])
    from fterm.config import load_config
    from fterm.features.build import build_dataset
    cfg = load_config(ROOT / "config.yaml")
    bundle = build_dataset(cfg, compute_fingerprints=False)
    # 只取前 200 日 / 前 400 只做门自检（快）
    sub = bundle.universe.iloc[:200, :400]
    lb = {k: v.iloc[:200, :400] for k, v in bundle.labels.items()}
    fake = type(bundle)(features=bundle.features, labels=lb, universe=sub,
                        feature_cols=bundle.feature_cols)
    st = chain.selftest_gates(fake, sw)
    assert st["baseline_ok"], f"基线 G1 未过：{st['baseline_g1']}"
    assert st["M1_caught"], "M1（不中性化）未被 G1 抓 ⇒ 门是空检查"
    assert st["M2_caught"], "M2（shift 前视）未被 G1 抓 ⇒ 门是空检查"


# ---------------------------------------------------------------- 越界守卫

def _imported_modules(src: str) -> set[str]:
    """所有 import 的模块名（含别名解析前的原名）。"""
    mods = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            for a in node.names:
                mods.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mods.add(node.module.split(".")[0])
    return mods


def _called_attr_names(src: str) -> set[str]:
    """所有 `x.method(...)` 形式调用的 method 名。"""
    out = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            out.add(node.func.attr)
    return out


def _docstring_and_comments_stripped(src: str) -> str:
    """剔除 **docstring**，但**保留**普通字符串字面量。

    这个区分是本批的关键（#32/#33 教训的镜像）：
      · 越界守卫要抓被禁符号，而它们常**以字符串字面量**出现（`{'sharpe': 1.0}`）
        ⇒ 不能剔除所有字符串（否则 M-H 漏检）；
      · 但 docstring 里会**说明**"不改 config.yaml" ⇒ 若不剔除 docstring 会假失败。
    ⇒ 只剔除 docstring（模块/类/函数首条 Expr-Constant），保留其余字符串。
    """
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


def test_no_config_yaml_write():
    """越界守卫：不得**写**配置（读取 `load_config` 合法）。

    ⚠ 变异测试 M-G 证明首版钉是**空检查**：查字面串 `yaml.dump`，
    而变异注入 `import yaml as _y; _y.dump({})` ⇒ 不含该字样。
    ⇒ 改为 AST 检查：不得 import yaml、不得调用 dump 系列。
    ⚠ 第二版又太宽（凡出现 `config.yaml` 即失败），但脚本**本来就该读它**。
    ⇒ 第三版：只禁「写」——凡引用 `config.yaml` 的调用，其函数名必须是 `load_config`。
    """
    assert "yaml" not in _imported_modules(SRC), "出现 import yaml ⇒ 可能写配置"
    for m in ("safe_dump", "to_yaml"):
        assert m not in _called_attr_names(SRC), f"出现写配置调用：{m}"

    # 逐调用检查：凡参数/接收者里含 'config.yaml' 的调用，只能是 load_config
    for node in ast.walk(ast.parse(SRC)):
        if not isinstance(node, ast.Call):
            continue
        txt = ast.unparse(node)
        if "config.yaml" not in txt:
            continue
        fname = ast.unparse(node.func)
        assert fname.endswith("load_config"), \
            f"对 config.yaml 的非读取调用：{fname}（{txt[:60]}）"


def test_no_portfolio_layer_conclusions():
    """越界守卫：不碰组合层（不报「按这个交易会赚多少」）。

    ⚠ 变异测试 M-H 证明首版钉是**空检查**：查的是剔除**所有字符串**后的代码，
    而被禁符号恰恰以字符串形式出现（如 `{'sharpe': 1.0}`）⇒ 被剔掉了。
    ⇒ 这是 #32/#33 教训的**镜像**：那里剔字符串防假失败，这里要留字符串才抓得到。
    ⇒ 改为在**保留普通字符串、只剔 docstring**的视图上查。
    """
    code = _docstring_and_comments_stripped(SRC).lower()
    for banned in ("sharpe", "max_drawdown", "total_return", "profit_factor"):
        assert banned not in code, f"出现组合层符号：{banned}"


def test_no_adoption_flag():
    code = _docstring_and_comments_stripped(SRC).lower()
    for banned in ("adoption_ready", "promote_to_factor"):
        assert banned not in code, f"出现采纳符号：{banned}"


def test_string_retention_pin_is_self_consistent():
    """自证：越界守卫必须能**看到字符串字面量**（否则又变回 M-H 空检查）。"""
    poison = "def f():\n    return {'sharpe': 1.0}\n"
    view = _docstring_and_comments_stripped(poison).lower()
    assert "sharpe" in view, "普通字符串被剔除 ⇒ M-H 会重现空检查"
    doc = 'def f():\n    """说明：不改 config.yaml"""\n    return 1\n'
    assert "config.yaml" not in _docstring_and_comments_stripped(doc), \
        "docstring 未被剔除 ⇒ 会假失败"
