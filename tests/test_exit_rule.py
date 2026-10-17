# -*- coding: utf-8 -*-
"""#128 回归钉：退出规则纯函数语义 + arm_faithful 参数化结构（AST 级）。

变异测试证明能失败：
  · 把 tierfix_replay.py 的 "exit_fn is None" 分支改坏（如删掉 exit_band 行）⇒ test_source_structure 红；
  · 把 make_avoid_exit 的 <= 改成 < ⇒ test_avoid_threshold 红（边界值 0.10 恰好触发与否翻转）。
"""
import ast
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from exit_rule_chain import (  # noqa: E402
    AVOID_10, AVOID_30, S1_MIN_ABS_BP, W2_TIER,
    make_avoid_exit, make_hybrid_exit, make_keep_exit,
)


class _T:
    def __init__(self, code, side, lots, reason):
        self.code, self.side, self.lots, self.reason = code, side, lots, reason


def _ranked(codes_scores):
    s = pd.Series(codes_scores)
    return s.sort_values(ascending=False)


# ---------- 纯函数语义 ----------

def test_keep_exit_sells_out_of_band():
    ev, hd = [], []
    fn = make_keep_exit(ev, hd)
    # 5 个标的 ⇒ keep4 = {A,B,C,D}；E 掉出；持仓 C 在带内不卖、E 在带外卖
    ranked = _ranked({"A": 0.9, "B": 0.8, "C": 0.3, "D": 0.2, "E": 0.1})
    outs = fn(ranked=ranked, smooth={}, positions=["C", "E"], tier=W2_TIER,
              date=pd.Timestamp("2026-01-02"))
    assert [t.product for t in outs] == ["E"]          # C 在 keep4 内不卖；E 跌出 ⇒ 卖
    assert ev[0]["code"] == "E"


def test_avoid_threshold_boundary():
    # 边界：smooth == threshold 恰好触发（<=），变异成 < 会红
    ev, hd = [], []
    fn = make_avoid_exit(0.10, ev, hd)
    ranked = _ranked({"A": 0.9, "X": 0.10})
    outs = fn(ranked=ranked, smooth={"A": 0.9, "X": 0.10}, positions=["X"], tier=W2_TIER,
              date=pd.Timestamp("2026-01-02"))
    assert len(outs) == 1 and ev[0]["smooth"] == 0.10
    ev2, _ = [], []
    fn2 = make_avoid_exit(0.10, ev2, [])
    outs2 = fn2(ranked=_ranked({"A": 0.9, "Y": 0.10001}), smooth={"A": 0.9, "Y": 0.10001},
                positions=["Y"], tier=W2_TIER, date=pd.Timestamp("2026-01-02"))
    assert len(outs2) == 0                           # 高于阈值不卖


def test_avoid_missing_in_smooth_failsafe():
    ev, hd = [], []
    fn = make_avoid_exit(0.10, ev, hd)
    ranked = _ranked({"A": 0.9})
    outs = fn(ranked=ranked, smooth={"A": 0.9}, positions=["Z"], tier=W2_TIER,
              date=pd.Timestamp("2026-01-02"))       # Z 不在平滑面 ⇒ fail-safe 卖
    assert [t.product for t in outs] == ["Z"] and ev[0]["smooth"] is None


def test_hybrid_triggers_on_either():
    ev, hd = [], []
    fn = make_hybrid_exit(0.10, ev, hd)
    # 5 标的：keep4={A,B,C,X}；B 在带内但 smooth≤0.10 ⇒ hybrid_avoid；D 掉出带 ⇒ hybrid_keep
    ranked = _ranked({"A": 0.9, "B": 0.8, "C": 0.3, "X": 0.2, "D": 0.05})
    outs = fn(ranked=ranked, smooth={"B": 0.08, "D": 0.5}, positions=["B", "D"],
              tier=W2_TIER, date=pd.Timestamp("2026-01-02"))
    reasons = {t.product: t.reason for t in outs}
    assert reasons == {"B": "hybrid_exit(hybrid_avoid)", "D": "hybrid_exit(hybrid_keep)"}


# ---------- 常量与结构（AST 级，变异可红） ----------

def test_constants_frozen():
    assert AVOID_10 == 0.10 and AVOID_30 == 0.30
    assert W2_TIER == dict(equity=20000, top_n=2, keep_n=4, max_symbols=2, max_lots=10)
    assert S1_MIN_ABS_BP == 11.5


def test_source_structure():
    src = (ROOT / "scripts" / "tierfix_replay.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "arm_faithful")
    kwargs = {a.arg for a in fn.args.kwonlyargs} | {a.arg for a in fn.args.args}
    assert "exit_fn" in kwargs, "arm_faithful 缺 exit_fn 参数"
    assert "exit_fn is None" in src, "缺默认分支守卫"
    assert 'f"exit_band(keep_' in src, "默认分支的 exit_band 语义被改动"
    # 默认分支必须在 if exit_fn is None 之内（防 exit_fn 无条件接管）
    for node in ast.walk(fn):
        if isinstance(node, ast.If) and getattr(node.test, "left", None) is not None:
            seg = ast.get_source_segment(src, node)
            if seg and "exit_fn is None" in seg and "exit_band" in seg:
                break
    else:
        pytest.fail("exit_band 未被 if exit_fn is None 守卫")


def test_exit_chain_arms_frozen():
    src = (ROOT / "scripts" / "exit_rule_chain.py").read_text(encoding="utf-8")
    assert '"A0"' in src and '"A1"' in src and '"A2"' in src and '"A3"' in src \
           and '"NEG"' in src
    assert "20261015" in src, "NEG 种子漂移"
