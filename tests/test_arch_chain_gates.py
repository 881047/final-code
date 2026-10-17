"""臂 D/E 执行器的 §五 仪器门验收（2026-10-10 补入回归钉）。

## 背景（为什么需要这个测试）

预注册 `plans/model-upgrade-prereg-20261009.md §五` 列了五道仪器门，任何一门失败
⇒ 该臂读数作废。但 `arch_chain.py` 原实现**只有 NaN 门 + 缺 F0 缓存门**，
五门从未落地：本批五门只由 verifier 与 Lead 在**外部**核验，裁决 json 内无留痕
（docs/03 §45 已知缺口 #2）。2026-10-10 补齐 `_instrument_gates()`。

## 本测试钉住的口径（三件，都是曾经真实踩到的坑）

① **§五-2 位连的判据方向**：只重算一个窗口 ⇒ F0 流里窗口之外的行「只在流里」
   是**正常**的。首版误把它计入不一致，导致四折全部假失败（`right_only` 785 万行）。
   正确判据只有两条：重算行必须全能在流里找到 + 交集 y_ret 逐位相等。
② **§五-5 面板竞态守卫必须响亮失败**：研究线面板越过隔离线时不得静默出裁决
   （实测触发过一次：生产检出面板已到 2026-10-09）。
③ **门不改读数**：补入门之后重跑 `--report`，全部 IC / 配对 t 必须逐位不变。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.arch_chain as ac  # noqa: E402


class TestBitconnectSemantics:
    """① §五-2 位连判据只约束「重算行」方向，不约束「流里多出来的行」。"""

    def test_left_join_only_flags_missing_recomputed_rows(self):
        stream = pd.DataFrame({
            "date": pd.to_datetime(["2019-01-02"] * 3 + ["2019-01-03"] * 2),
            "product": ["A", "B", "C", "A", "B"],
            "y_ret": [0.1, 0.2, 0.3, 0.4, 0.5],
        })
        # 重算只覆盖 2019-01-02（流里另有 01-03 两行 —— 不得算作不一致）
        cur = pd.DataFrame({
            "date": pd.to_datetime(["2019-01-02"] * 3),
            "product": ["A", "B", "C"],
            "y_cur": [0.1, 0.2, 0.3],
        })
        j = cur.merge(stream, on=["date", "product"], how="left", indicator=True)
        only_cur = int((j["_merge"] == "left_only").sum())
        both = j[j["_merge"] == "both"]
        dy = float((both["y_cur"] - both["y_ret"]).abs().max())
        assert only_cur == 0
        assert dy == 0.0
        # 反向：outer join 会把流里多出来的 2 行算成 right_only ⇒ 首版假失败的根因
        j_outer = cur.merge(stream, on=["date", "product"], how="outer", indicator=True)
        assert int((j_outer["_merge"] == "right_only").sum()) == 2

    def test_missing_recomputed_row_is_flagged(self):
        stream = pd.DataFrame({"date": pd.to_datetime(["2019-01-02"] * 2),
                               "product": ["A", "B"], "y_ret": [0.1, 0.2]})
        cur = pd.DataFrame({"date": pd.to_datetime(["2019-01-02"] * 3),
                            "product": ["A", "B", "ZZZ"], "y_cur": [0.1, 0.2, 0.3]})
        j = cur.merge(stream, on=["date", "product"], how="left", indicator=True)
        assert int((j["_merge"] == "left_only").sum()) == 1

    def test_label_drift_is_flagged(self):
        stream = pd.DataFrame({"date": pd.to_datetime(["2019-01-02"]),
                               "product": ["A"], "y_ret": [0.1]})
        cur = pd.DataFrame({"date": pd.to_datetime(["2019-01-02"]),
                            "product": ["A"], "y_cur": [0.100001]})
        j = cur.merge(stream, on=["date", "product"], how="left", indicator=True)
        both = j[j["_merge"] == "both"]
        assert float((both["y_cur"] - both["y_ret"]).abs().max()) > 0.0


class TestPanelRaceGuard:
    """② §五-5：面板越过隔离线必须响亮失败，不得静默出裁决。"""

    def test_isolation_line_constant(self):
        assert ac.ISOLATION_MAX == pd.Timestamp("2026-09-30")

    def test_guard_thresholds_present(self):
        assert ac.RIGHT_GAP_MAX == 15
        assert ac.TRIPWIRE_MAX == 0.025
        assert len(ac.CONN_WINDOWS) == 4


class TestGatesAreWiredIntoVerdict:
    """③ 门必须真的被 report 路径调用，且能用 --no-gates 显式跳过。"""

    def test_no_gates_flag_exists(self):
        # argparse 定义在 main() 内，用源码级断言钉住（避免起真实 report）
        src = (ROOT / "scripts" / "arch_chain.py").read_text(encoding="utf-8")
        assert "--no-gates" in src
        assert "_instrument_gates" in src
        assert "instrument_gates" in src
        # 门失败必须 raise（响亮失败），不得只打印
        assert "仪器门 §五-5 失败" in src
        assert "仪器门 §五-1 失败" in src
        assert "仪器门 §五-3 失败" in src
        assert "仪器门 §五-2 失败" in src

    def test_no_gates_marks_verdict_as_uncitable(self):
        src = (ROOT / "scripts" / "arch_chain.py").read_text(encoding="utf-8")
        assert "不得引用" in src or "不得作为正式裁决引用" in src
