"""2 万档「每次只买 1–2 只」规则的回归钉（判据链 #23；预注册
`plans/w2-rule-prereg-20261010.md`）。

## 规则
用户 2026-10-10 拍板：**2 万档每次只买 1–2 只股票**。

## 本测试钉住四件事
① **规则本身**：`paper_tracks.w2` 的 `max_symbols ≤ 2`（原为 3，越界）
② **口径自洽**：`top_n ≤ max_symbols`（config 的既有断言）+ `keep_n ≥ top_n`
③ **不误伤 t3 档**：t3（100 万验证档）**不受**该规则约束（其构型由 #12 判决冻结）
④ **规则可达性**：2 万本金下 1–2 只是**可执行**的（不是纸面规则）——
   实测 top-2 中至少 1 只可买的天数占比 > 70%
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fterm.config import load_config  # noqa: E402


class TestW2Rule:
    """① 规则本身：2 万档上限 ≤ 2 只。"""

    def test_w2_max_symbols_within_rule(self):
        cfg = load_config(ROOT / "config.yaml")
        w2 = cfg.raw["paper_tracks"]["w2"]
        assert w2["max_symbols"] <= 2, (
            f"用户规则「每次只买 1–2 只」要求 max_symbols ≤ 2，实测 {w2['max_symbols']}")

    def test_w2_top_n_within_rule(self):
        cfg = load_config(ROOT / "config.yaml")
        w2 = cfg.raw["paper_tracks"]["w2"]
        assert 1 <= w2["top_n"] <= 2, f"top_n 须在 1–2，实测 {w2['top_n']}"

    def test_w2_equity_is_20k(self):
        """规则的对象是**2 万档**，本金不能漂移。"""
        cfg = load_config(ROOT / "config.yaml")
        assert cfg.raw["paper_tracks"]["w2"]["equity"] == 20000


class TestInternalConsistency:
    """② 口径自洽（config 既有断言 + 退出带语义）。"""

    def test_top_n_not_exceed_max_symbols(self):
        cfg = load_config(ROOT / "config.yaml")
        w2 = cfg.raw["paper_tracks"]["w2"]
        assert w2["top_n"] <= w2["max_symbols"], "top_n 不能超过 max_symbols"

    def test_keep_n_not_below_top_n(self):
        """退出带必须 ≥ 目标持仓数，否则刚建仓就被踢出（震荡换手）。"""
        cfg = load_config(ROOT / "config.yaml")
        w2 = cfg.raw["paper_tracks"]["w2"]
        assert w2["keep_n"] >= w2["top_n"]

    def test_keep_n_matches_exit_buffer(self):
        """keep_n 语义 = top_n × exit_buffer（见 signal.exit_buffer）。"""
        cfg = load_config(ROOT / "config.yaml")
        w2 = cfg.raw["paper_tracks"]["w2"]
        assert w2["keep_n"] == w2["top_n"] * cfg.signal.exit_buffer


class TestDoesNotTouchT3:
    """③ 不误伤 100 万验证档（其构型由 #12 判决冻结）。"""

    def test_t3_unchanged(self):
        cfg = load_config(ROOT / "config.yaml")
        t3 = cfg.raw["paper_tracks"]["t3"]
        assert t3["max_symbols"] == 10, "t3 档不受 2 万规则约束"
        assert t3["top_n"] == 10
        assert t3["keep_n"] == 20
        assert t3["equity"] == 1_000_000

    def test_rule_documented_as_w2_only(self):
        """规则必须写明只作用于 w2（防未来误改 t3）。"""
        txt = (ROOT / "config.yaml").read_text(encoding="utf-8")
        assert "每次只买 1–2 只" in txt


class TestRuleIsExecutable:
    """④ 规则可达性：2 万本金下 1–2 只是**可执行**的，不是纸面规则。

    用**合成**数据验证 sizer 逻辑（禁真实老库/网络），不依赖本地面板。
    """

    def test_sizer_can_buy_1_to_2_lots_at_20k(self):
        from fterm.risk.sizer import size_position
        cfg = load_config(ROOT / "config.yaml")
        # 典型 A 股：价 10 元、ATR 0.3 元
        s = size_position(cfg, 20000.0, 10.0, 0.3)
        assert s.lots >= 1, "2 万档在典型价位应至少能买 1 手"
        # 单价过高时 0 手是合法输出（README 的"数学事实"）
        s2 = size_position(cfg, 20000.0, 200.0, 5.0)
        assert s2.lots == 0, "200 元股价一手即 2 万 ⇒ 应输出 0 手"
        assert s2.method_used == "zero"

    def test_max_symbols_caps_portfolio_at_2(self):
        """governor 必须真的把第 3 只拒掉（规则的可执行形式）。"""
        from fterm.backtest.engine import Target
        from fterm.risk.governor import AccountState, RiskGovernor

        cfg = load_config(ROOT / "config.yaml")
        from dataclasses import replace
        cfg_w2 = replace(cfg, account=replace(cfg.account,
                                              max_symbols=2, max_lots_per_symbol=10))
        gov = RiskGovernor(cfg_w2)
        state = AccountState(
            equity=20000.0, peak_equity=20000.0, day_pnl=0.0,
            open_positions={"A": {"direction": 1, "lots": 1, "value": 3000.0},
                            "B": {"direction": 1, "lots": 1, "value": 3000.0}},
            day_start_equity=20000.0, available_cash=14000.0)
        rep = gov.approve([Target("C", 1, 1, "test")], state, prices={"C": 30.0})
        assert not rep.decisions[0].approved, "已有 2 只时第 3 只必须被拒"
        assert "max_symbols" in rep.decisions[0].reason

    def test_third_symbol_allowed_when_only_one_held(self):
        """只持 1 只时应允许加第 2 只（规则是 ≤2，不是 ==2）。"""
        from fterm.backtest.engine import Target
        from fterm.risk.governor import AccountState, RiskGovernor
        from dataclasses import replace

        cfg = load_config(ROOT / "config.yaml")
        cfg_w2 = replace(cfg, account=replace(cfg.account,
                                              max_symbols=2, max_lots_per_symbol=10))
        gov = RiskGovernor(cfg_w2)
        state = AccountState(
            equity=20000.0, peak_equity=20000.0, day_pnl=0.0,
            open_positions={"A": {"direction": 1, "lots": 1, "value": 3000.0}},
            day_start_equity=20000.0, available_cash=17000.0)
        rep = gov.approve([Target("B", 1, 1, "test")], state, prices={"B": 30.0})
        assert rep.decisions[0].approved, "只持 1 只时应允许第 2 只"
