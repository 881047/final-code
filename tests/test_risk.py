"""风控栈验收：sizer 离散化与 governor 全部硬约束（A股现金口径）。"""

from __future__ import annotations

import pytest

from fterm.backtest.engine import Target
from fterm.risk.governor import AccountState, RiskGovernor
from fterm.risk.sizer import risk_budget_lots, size_position, vol_cap_lots


@pytest.fixture
def cfg(tmp_path):
    from fterm.config import load_config
    from tests.conftest import _yaml_text

    p = tmp_path / "c.yaml"
    p.write_text(_yaml_text().replace("{old_db}", "x").replace("{root}", str(tmp_path / "d").replace("\\", "/")),
                 encoding="utf-8")
    return load_config(p)


class TestSizer:
    def test_risk_budget_discrete(self, cfg):
        # 权益 2万 × 2% = 400 元预算；止损距离 = 1.2 × ATR=2.0 → 2.4 元/股
        # 股数 = 400/2.4 = 166.7 → 手数 floor(166.7/100) = 1
        lots = risk_budget_lots(cfg, 20000, 20.0, 2.0)
        assert lots == 1
        # 预算不足半手 → 0 手
        assert risk_budget_lots(cfg, 20000, 20.0, 15.0) == 0

    def test_vol_cap(self, cfg):
        # 单手日波金额 = 20 × 100 × 0.02 = 40 元；上限 = 20000×0.025=500 → 12 手
        lots = vol_cap_lots(cfg, 20000, 20.0, vol_pred_log=None, )
        assert lots == 0          # 无 vol 预测 → 0（合法）
        import math
        v = math.log1p(0.02 * 100)
        lots2 = vol_cap_lots(cfg, 20000, 20.0, vol_pred_log=v)
        assert lots2 == 12

    def test_size_position_caps(self, cfg):
        import math
        v = math.log1p(0.02 * 100)
        # vol 上限 12 手 > 配置 max_lots_per_symbol=10 → 截到 10
        sz = size_position(cfg, 20000, 20.0, atr_value=0.01, vol_pred_log=v)
        assert sz.lots == 10
        assert sz.method_used == "vol_target"

    def test_zero_is_legal(self, cfg):
        sz = size_position(cfg, 20000, 20.0, atr_value=0.0)
        assert sz.lots == 0 and sz.method_used == "zero"

    def test_2w_constraint_end_to_end(self, cfg):
        """2 万本金全链：手数×100股×价 不超过权益（现金硬约束由 governor/引擎保证，
        sizer 只做风险口径——这里验证风险预算产出的手数在 2 万下是可成交的）。"""
        sz = size_position(cfg, 20000, 20.0, atr_value=1.0)
        assert sz.lots >= 1
        assert sz.lots * 100 * 20.0 <= 20000 * 0.95    # 与 max_cash_util 对齐


class TestGovernor:
    def _state(self, equity=20000, peak=20000, day_pnl=0.0, positions=None):
        return AccountState(equity=equity, peak_equity=peak, day_pnl=day_pnl,
                            open_positions=positions or {})

    def test_normal_pass(self, cfg):
        rep = RiskGovernor(cfg).approve([Target("000001.SZ", 1, 5, "t")],
                                        self._state(), prices={"000001.SZ": 20.0})
        assert rep.approved[0].lots == 5

    def test_total_dd_halt_blocks_open_allows_reduce(self, cfg):
        state = self._state(equity=17000, peak=20000)   # 回撤 15% > 12%
        rep = RiskGovernor(cfg).approve(
            [Target("000001.SZ", 1, 5, "open"), Target("600000.SH", 0, 0, "close")],
            state, prices={"000001.SZ": 20.0})
        by_code = {d.target.product: d for d in rep.decisions}
        assert by_code["000001.SZ"].approved is False
        assert "halted" in by_code["000001.SZ"].reason
        assert by_code["600000.SH"].approved is True

    def test_daily_stop_blocks_new_only(self, cfg):
        state = self._state(day_pnl=-700)               # > 3% × 20000
        rep = RiskGovernor(cfg).approve([Target("000001.SZ", 1, 5, "open")],
                                        state, prices={"000001.SZ": 20.0})
        assert rep.decisions[0].approved is False
        assert "day_stop" in rep.decisions[0].reason

    def test_daily_stop_denominator_is_day_start(self, cfg):
        # 分母修复钉死（2026-10-06）：日初 24000、peak 25000、当日浮亏 −740
        # ⇒ 新口径（3%×日初=720）拦截；旧 peak 口径（3%×peak=750）会放行。
        # 未触发熔断：23260 > 25000×0.88 = 22000。
        state = AccountState(equity=23260, peak_equity=25000, day_pnl=-740,
                             open_positions={}, day_start_equity=24000)
        rep = RiskGovernor(cfg).approve([Target("000001.SZ", 1, 5, "open")],
                                        state, prices={"000001.SZ": 20.0})
        assert rep.decisions[0].approved is False
        assert "day_stop" in rep.decisions[0].reason

    def test_max_symbols(self, cfg):
        state = self._state(positions={"600000.SH": {"direction": 1, "lots": 1, "value": 5000},
                                       "000002.SZ": {"direction": 1, "lots": 1, "value": 5000},
                                       "600036.SH": {"direction": 1, "lots": 1, "value": 5000}})
        rep = RiskGovernor(cfg).approve([Target("000001.SZ", 1, 5, "open")],
                                        state, prices={"000001.SZ": 20.0})
        assert "max_symbols" in rep.decisions[0].reason

    def test_cash_util_gate(self, cfg):
        # 已持有市值 16000（80%），再加 5手×20元=10000 → 26000 > 95%×20000 ⇒ 拒
        state = self._state(positions={"600000.SH": {"direction": 1, "lots": 8, "value": 16000}})
        rep = RiskGovernor(cfg).approve([Target("000001.SZ", 1, 5, "open")],
                                        state, prices={"000001.SZ": 20.0})
        assert "cash_util" in rep.decisions[0].reason

    def test_lots_cap(self, cfg):
        rep = RiskGovernor(cfg).approve([Target("000001.SZ", 1, 11, "t")],
                                        self._state(), prices={"000001.SZ": 20.0})
        assert "lots_cap" in rep.decisions[0].reason

    def test_no_price_rejected(self, cfg):
        rep = RiskGovernor(cfg).approve([Target("000001.SZ", 1, 5, "t")],
                                        self._state(), prices={})
        assert "no_price" in rep.decisions[0].reason

    def test_halt_reduce_only_direction(self, cfg):
        # 熔断中：持仓减仓允许（lots 变小），加仓拒绝
        state = self._state(equity=17000, peak=20000,
                            positions={"000001.SZ": {"direction": 1, "lots": 8, "value": 16000}})
        rep = RiskGovernor(cfg).approve(
            [Target("000001.SZ", 1, 5, "reduce"), Target("000001.SZ", 1, 10, "add")],
            state, prices={"000001.SZ": 20.0})
        by_lots = {d.target.lots: d for d in rep.decisions}
        assert by_lots[5].approved is True
        assert by_lots[10].approved is False


class TestGovernorForceLiquidate:
    def _cfg(self, cfg, flag: bool):
        from dataclasses import replace
        return replace(cfg, account=replace(cfg.account, circuit_force_liquidate=flag))

    def test_flag_off_keeps_current_behavior(self, cfg):
        state = AccountState(equity=17000, peak_equity=20000, day_pnl=0,
                             open_positions={"000001.SZ": {"direction": 1, "lots": 5, "value": 10000}})
        rep = RiskGovernor(self._cfg(cfg, False)).approve([Target("600000.SH", 1, 5, "x")],
                                                          state, prices={"600000.SH": 20.0})
        assert all(d.reason != "halt_force_liquidate" for d in rep.decisions)

    def test_halt_liquidates_uncovered_positions(self, cfg):
        state = AccountState(equity=17000, peak_equity=20000, day_pnl=0,
                             open_positions={"000001.SZ": {"direction": 1, "lots": 5, "value": 10000}})
        rep = RiskGovernor(self._cfg(cfg, True)).approve([Target("600000.SH", 1, 5, "x")],
                                                         state, prices={"600000.SH": 20.0})
        liq = [d for d in rep.decisions if d.reason == "halt_force_liquidate"]
        assert len(liq) == 1 and liq[0].target.product == "000001.SZ"

    def test_no_duplicate_for_strategy_close(self, cfg):
        state = AccountState(equity=17000, peak_equity=20000, day_pnl=0,
                             open_positions={"000001.SZ": {"direction": 1, "lots": 5, "value": 10000}})
        rep = RiskGovernor(self._cfg(cfg, True)).approve([Target("000001.SZ", 0, 0, "strategy_close")],
                                                         state, prices={"000001.SZ": 20.0})
        assert sum(1 for d in rep.decisions if d.reason == "halt_force_liquidate") == 0

    def test_not_halted_no_liquidation(self, cfg):
        state = AccountState(equity=19000, peak_equity=20000, day_pnl=0,
                             open_positions={"000001.SZ": {"direction": 1, "lots": 5, "value": 10000}})
        rep = RiskGovernor(self._cfg(cfg, True)).approve([], state, prices={})
        assert all(d.reason != "halt_force_liquidate" for d in rep.decisions)

    def test_held_product_increase_target_rejected_still_liquidated(self, cfg):
        state = AccountState(equity=17000, peak_equity=20000, day_pnl=0,
                             open_positions={"000001.SZ": {"direction": 1, "lots": 5, "value": 10000}})
        rep = RiskGovernor(self._cfg(cfg, True)).approve([Target("000001.SZ", 1, 10, "add")],
                                                         state, prices={"000001.SZ": 20.0})
        rejected = [d for d in rep.decisions if not d.approved]
        liq = [d for d in rep.decisions if d.reason == "halt_force_liquidate"]
        assert rejected and liq and liq[0].target.product == "000001.SZ"
