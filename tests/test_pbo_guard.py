"""判据链 #44 回归钉：PBO 门加固（防静默回退到"给伪读数开门"）。

预注册：`plans/pbo-guard-prereg-20261011.md`
被测：`fterm/model/stats.py::pbo_cscv`

本钉保护：
  1. ⭐ **拒绝 N<4**（N=2/3 必须是伪读数）；
  2. **N≥4 正常返回**，且携带 `n_candidates`（使调用方无法忽略 N）；
  3. ⭐ **docstring 含读数前提**（相对量 / 对手 / N 的影响）——防"只改代码不改说明"；
  4. ⭐ **数值等价**：重构**不得**改变任何 N≥4 输入的读数（防"顺手改了算法"）。
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fterm.model import stats as st        # noqa: E402

STATS_SRC = (ROOT / "fterm" / "model" / "stats.py").read_text(encoding="utf-8")
BASELINE = Path(r"C:\Users\27817\AppData\Local\Temp\pbo_baseline.json")


# ---------------------------------------------------------------- 1. 拒绝 N<4

def test_rejects_n2_and_n3():
    """⭐ N=2 / N=3 必须被拒绝：ω 只有 N+1 档 ⇒ 读数是伪高。"""
    rng = np.random.default_rng(7)
    for n in (2, 3):
        M = rng.normal(size=(300, n))
        r = st.pbo_cscv(M, n_blocks=8)
        assert r["pbo"] is None, f"N={n} 未被拒绝（pbo={r['pbo']}）"
        assert r["reason"], f"N={n} 缺 reason"
        assert str(n) in r["reason"], f"reason 未指明实际 N：{r['reason']}"


def test_rejects_n1_and_n0():
    for n in (1, 0):
        M = np.zeros((300, n)) if n else np.zeros((300, 0))
        r = st.pbo_cscv(M, n_blocks=8)
        assert r["pbo"] is None, f"N={n} 未被拒绝"


def test_guard_uses_constant_not_magic_number():
    """守卫必须用常量（便于审计），且该常量为 4。"""
    assert st.PBO_MIN_CANDIDATES == 4
    assert "N < PBO_MIN_CANDIDATES" in STATS_SRC, \
        "守卫未使用 PBO_MIN_CANDIDATES（写死了数字）"


# ---------------------------------------------------------------- 2. N>=4 正常 + n_candidates

def test_accepts_n4():
    """N=4 必须正常返回（守卫不能过严）。"""
    rng = np.random.default_rng(11)
    M = rng.normal(size=(300, 4))
    r = st.pbo_cscv(M, n_blocks=8)
    assert r["pbo"] is not None, "N=4 被误拒"
    assert r["n_candidates"] == 4
    assert 0.0 <= r["pbo"] <= 1.0


def test_reports_n_candidates():
    """返回值必须携带 `n_candidates`（使调用方无法忽略 N）。"""
    rng = np.random.default_rng(3)
    for n in (4, 6, 10, 16):
        r = st.pbo_cscv(rng.normal(size=(300, n)), n_blocks=8)
        assert r.get("n_candidates") == n, f"N={n} 未正确报告 n_candidates"


# ---------------------------------------------------------------- 3. docstring 读数前提

def test_docstring_has_reading_caveats():
    """⭐ docstring 必须写入读数前提（相对量 / 对手 / N）。"""
    doc = st.pbo_cscv.__doc__ or ""
    assert "读数前提" in doc, "docstring 缺「读数前提」段"
    assert "相对量" in doc, "docstring 未写明 PBO 是相对量"
    assert "对手" in doc, "docstring 未写明须报对手"
    assert "PBO_MIN_CANDIDATES" in doc or "N < 4" in doc, \
        "docstring 未写明 N 的门槛"


def test_docstring_warns_pure_noise():
    """docstring 必须含**独立的一条**「纯噪声也给高 PBO」警示。

    ⚠ 变异测试 M-5 暴露：前两版都只查"纯噪声"三字 ⇒ **无效**：
      · 首版全 docstring 查 ⇒ §10 历史段落里也有该词；
      · 二版限定在「读数前提」段内 ⇒ **前提②里也有该词**（"实测纯噪声 N=2 即得…"）
        ⇒ 删掉前提③仍能通过。
    ⇒ 改为断言 **前提③ 本身存在**（锚定序号，不靠关键词计数）。
    """
    doc = st.pbo_cscv.__doc__ or ""
    i = doc.find("读数前提")
    assert i >= 0, "docstring 缺「读数前提」段"
    j = doc.find("历史")
    seg = doc[i:j if j > i else len(doc)]
    assert "③" in seg, "前提段缺少第③条（纯噪声警示）"
    # 前提③ 的正文里必须出现"纯噪声"（即警示本身写在③里，而非只在别处）
    third = seg.split("③", 1)[1]
    third = third.split("④")[0]
    assert "纯噪声" in third, \
        "⚠「纯噪声」警示不在前提③内（变异 M-5 未被抓）"
    # ⭐ M-5b：③ 的**标题**本身也是警示主体 ⇒ 删掉标题里的"纯噪声"也必须被抓
    title = third.split("（", 1)[0]
    assert "纯噪声" in title, \
        "⚠ 前提③ 的标题未含「纯噪声」（变异 M-5b 未被抓）⇒ 警示被弱化"


def test_source_has_no_bare_n_lt_2_guard():
    """源码里不得再有 `if N < 2:` 形式的旧守卫（变异 M-1 的锚）。

    ⚠ 只查**代码**（AST），不查 docstring —— docstring 里出现 "N < 4" 是**期望**的。
    """
    tree = ast.parse(STATS_SRC)
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "pbo_cscv"), None)
    assert fn is not None, "未找到 pbo_cscv"
    for node in ast.walk(fn):
        if isinstance(node, ast.Compare):
            # 只看左侧**恰为标识符 N** 的比较（避免误匹配 M.ndim 等）
            left = ast.unparse(node.left).strip()
            if left != "N":
                continue
            txt = ast.unparse(node)
            assert not (txt.startswith("N < 2") or txt.startswith("N < 3")), \
                f"仍存在旧守卫：{txt}"
            assert "PBO_MIN_CANDIDATES" in txt, \
                f"守卫未用常量：{txt}"


# ---------------------------------------------------------------- 4. 数值等价（G4）

def test_numeric_equivalence_with_baseline():
    """⭐ 重构**不得**改变任何 N≥4 输入的读数。"""
    if not BASELINE.exists():
        pytest.skip("缺基准快照（先跑 pbo_baseline.py）")
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    for key, exp in base.items():
        name, nbs = key.rsplit("_nb", 1)
        nb = int(nbs)
        n = int(exp["n_candidates"]) if "n_candidates" in exp else None
        # 基准快照里 N<4 的用例已被新守卫拒绝 ⇒ 只校验有 pbo 的
        if exp.get("pbo") is None:
            continue
        M = _regen(name)
        got = st.pbo_cscv(M, n_blocks=nb)
        assert got["pbo"] == exp["pbo"], \
            f"{key} 数值漂移：基准 {exp['pbo']} vs 现在 {got['pbo']}"
        assert got["n_splits"] == exp["n_splits"], f"{key} n_splits 漂移"
        assert got["lambda_mean"] == exp["lambda_mean"], f"{key} lambda_mean 漂移"


def _regen(name: str) -> np.ndarray:
    """按 `pbo_baseline.py` **同一条共享 rng 流、同序**重建输入。

    ⚠ 必须与基准脚本**同序**消耗同一个 rng（基准里所有用例共用
    `rng = default_rng(20261011)`，按固定顺序依次抽样）
    ⇒ 否则输入不同、数值必然漂移（曾因此假失败一次 —— G4 门**确实**在工作）。
    """
    cases = [
        ("n4_gaussian", lambda r: r.normal(size=(400, 4))),
        ("n6_gaussian", lambda r: r.normal(size=(400, 6))),
        ("n8_with_signal", lambda r: r.normal(size=(400, 8))
         + np.arange(8)[None, :] * 0.3),
        ("n10_trend", lambda r: np.cumsum(r.normal(size=(400, 10)), axis=0) * 0.1),
        ("n5_dominant", lambda r: np.column_stack([r.normal(size=(400, 4)),
                                                    r.normal(size=(400, 1)) + 5.0])),
        ("n16_mixed", lambda r: r.normal(size=(400, 16))),
    ]
    rng = np.random.default_rng(20261011)
    for nm, fn in cases:
        M = fn(rng)
        if nm == name:
            return M
    raise AssertionError(f"未知的基准用例 {name}")


import pytest    # noqa: E402  (放在函数后以免打乱上面的定义顺序)
