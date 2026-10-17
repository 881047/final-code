"""tierfix_replay.reb_ranking —— 调仓日排名的纯函数钉（2026-10-08 审查批）。

背景（docs/03 §41 勘误）：预测流因标签前视窗天然缺失最后 6 个交易日，
arm_faithful 旧实现在"调仓日无预测行"时 reindex 到空 ⇒ keep=∅ ⇒ 整本清仓。
修复后该场景回退到最近一次平滑秩（与生产 runner 的持久化 EMA 状态同语义）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tierfix_replay import reb_ranking  # noqa: E402


def _row(codes: list[str]) -> pd.Series:
    return pd.Series({c: 1.0 for c in codes})


def test_no_pred_row_falls_back_to_persisted_smooth():
    """回归钉：无预测行的调仓日不得返回空（旧实现 = 意外全清仓）。"""
    smooth = {"A": 0.9, "B": 0.5, "C": 0.7}
    ranked = reb_ranking(smooth, None)
    assert list(ranked.index) == ["A", "C", "B"]      # 降序
    assert len(ranked) == 3


def test_no_pred_row_with_empty_smooth_is_empty():
    """首个调仓日本就无任何平滑状态：空是真语义（无票可选），不是缺陷。"""
    ranked = reb_ranking({}, None)
    assert ranked.empty


def test_with_pred_row_reindexes_to_row_and_drops_unknown():
    """有预测行：限制到当日预测列（资格/预测消失的持仓自然掉出 keep），与旧行为一致。"""
    smooth = {"A": 0.9, "B": 0.5, "C": 0.7, "GONE": 0.99}
    ranked = reb_ranking(smooth, _row(["A", "B", "C"]))
    assert "GONE" not in ranked.index
    assert list(ranked.index) == ["A", "C", "B"]


def test_with_pred_row_codes_absent_from_smooth_are_dropped():
    """预测行里有从未步进过的新代码：无平滑值 ⇒ 不参与排名（dropna 语义）。"""
    smooth = {"A": 0.9}
    ranked = reb_ranking(smooth, _row(["A", "NEW"]))
    assert list(ranked.index) == ["A"]


def test_empty_smooth_with_row_is_empty():
    assert reb_ranking({}, _row(["A", "B"])).empty


@pytest.mark.parametrize("row", [None, _row(["A", "B", "C"])])
def test_output_is_sorted_descending_float64(row):
    smooth = {"A": 0.3, "B": 0.8, "C": 0.1}
    ranked = reb_ranking(smooth, row)
    assert ranked.dtype == "float64"
    assert list(ranked.index) == ["B", "A", "C"]
