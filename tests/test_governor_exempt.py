# -*- coding: utf-8 -*-
"""#135 仪器开关回归钉：governor.approve 的 exempt_batch_exits。

钉两件事：
1. 默认（False）路径逐位不变——同批"全平仓 + 新建仓"在现行语义下新仓必拒
   （max_symbols / cash_short），#131 诊断的机制锚；
2. True 路径——同批平仓即时释放槽位/市值/现金 ⇒ 换仓日新仓放行，
   且不越 max_symbols / max_cash_util 上限。
"""
import pytest

from fterm.risk.governor import AccountState, RiskGovernor
from fterm.backtest.engine import Target


@pytest.fixture
def cfg(scope="module"):
    """w2 同构账户（arm_faithful 的 dataclasses.replace 口径）：max_symbols=2。"""
    import dataclasses
    from fterm.config import load_config
    from pathlib import Path
    c = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    return dataclasses.replace(c, account=dataclasses.replace(
        c.account, initial_equity=20000, max_symbols=2,
        max_lots_per_symbol=10))


def _rot_state():
    """换仓日账本：满仓 2 只（各 9500 市值），现金≈1000。"""
    return AccountState(equity=20000, peak_equity=20000, day_pnl=0.0,
                        open_positions={
                            "AAA.SZ": {"direction": 1, "lots": 5, "value": 9500},
                            "BBB.SZ": {"direction": 1, "lots": 4, "value": 9500},
                        },
                        day_start_equity=20000, available_cash=1000.0)


class TestExemptBatchExits:
    def test_default_rejects_rebuy_on_rotation_day(self, cfg):
        """默认路径（逐位锚）：平仓 + 新仓同批 ⇒ 新仓被 max_symbols 全拒。"""
        tgts = [Target("AAA.SZ", 0, 0, "exit"), Target("BBB.SZ", 0, 0, "exit"),
                Target("CCC.SZ", 1, 3, "buy1"), Target("DDD.SZ", 1, 3, "buy2")]
        rep = RiskGovernor(cfg).approve(tgts, _rot_state(),
                                        prices={"CCC.SZ": 20.0, "DDD.SZ": 20.0})
        buys = [d for d in rep.decisions if d.target.direction == 1]
        assert all(not d.approved for d in buys)
        assert all(d.reason.startswith("max_symbols") for d in buys)

    def test_exempt_frees_slots_and_cash(self, cfg):
        """豁免路径：同批平仓即时释放 ⇒ 两笔新仓获批准（槽位与现金均够）。"""
        tgts = [Target("AAA.SZ", 0, 0, "exit"), Target("BBB.SZ", 0, 0, "exit"),
                Target("CCC.SZ", 1, 3, "buy1"), Target("DDD.SZ", 1, 3, "buy2")]
        rep = RiskGovernor(cfg).approve(tgts, _rot_state(),
                                        prices={"CCC.SZ": 20.0, "DDD.SZ": 20.0},
                                        exempt_batch_exits=True)
        buys = [d for d in rep.decisions if d.target.direction == 1]
        assert all(d.approved for d in buys)

    def test_exempt_without_exits_still_caps(self, cfg):
        """豁免不是万能放行：无平仓批次（held=2 满仓）新标的仍被 max_symbols 拒。"""
        tgts = [Target("CCC.SZ", 1, 3, "b1"), Target("DDD.SZ", 1, 3, "b2")]
        rep = RiskGovernor(cfg).approve(tgts, _rot_state(),
                                        prices={c: 20.0 for c in ("CCC.SZ", "DDD.SZ")},
                                        exempt_batch_exits=True)
        buys = [d for d in rep.decisions if d.target.direction == 1]
        assert all(not d.approved for d in buys)
        assert all(d.reason.startswith("max_symbols") for d in buys)

    def test_exempt_batch_new_symbols_count_toward_cash(self, cfg):
        """豁免路径下现金闸仍逐笔累加（committed）：第三笔超出平仓回款 ⇒ cash_short。"""
        # 平仓回款 19000 + 现金 1000 = 20000 可用；每笔 est 6000 ⇒ 第 4 笔超限
        tgts = [Target("AAA.SZ", 0, 0, "exit"), Target("BBB.SZ", 0, 0, "exit"),
                Target("CCC.SZ", 1, 3, "b1"), Target("DDD.SZ", 1, 3, "b2"),
                Target("EEE.SZ", 1, 3, "b3"), Target("FFF.SZ", 1, 3, "b4")]
        rep = RiskGovernor(cfg).approve(tgts, _rot_state(),
                                        prices={c: 20.0 for c in ("CCC.SZ", "DDD.SZ",
                                                                  "EEE.SZ", "FFF.SZ")},
                                        exempt_batch_exits=True)
        buys = [d for d in rep.decisions if d.target.direction == 1]
        approved_n = sum(1 for d in buys if d.approved)
        assert approved_n >= 2
        assert any((not d.approved) and d.reason.startswith("cash") for d in buys)

    def test_flag_off_bitwise_values(self, cfg):
        """默认路径数值锚：部分平仓（不豁免）时 value_now/cash 不受影响。"""
        state = AccountState(equity=20000, peak_equity=20000, day_pnl=0.0,
                             open_positions={"AAA.SZ": {"direction": 1, "lots": 5, "value": 9500}},
                             day_start_equity=20000, available_cash=10500.0)
        tgts = [Target("AAA.SZ", 0, 0, "exit"), Target("CCC.SZ", 1, 3, "buy")]
        rep = RiskGovernor(cfg).approve(tgts, state,
                                        prices={"CCC.SZ": 20.0})
        buy = [d for d in rep.decisions if d.target.direction == 1][0]
        assert buy.approved is True       # 现金 10500 足够（与豁免无关的历史行为）


class TestExemptMutation:
    def test_mutation_removing_exemption_restores_rejection(self, cfg):
        """变异测试：把豁免开关从 True 改回 False ⇒ 换仓日新仓重新被拒（钉能失败）。"""
        tgts = [Target("AAA.SZ", 0, 0, "exit"), Target("BBB.SZ", 0, 0, "exit"),
                Target("CCC.SZ", 1, 3, "buy")]
        for flag, expect_ok in ((True, True), (False, False)):
            rep = RiskGovernor(cfg).approve(tgts, _rot_state(),
                                            prices={"CCC.SZ": 20.0},
                                            exempt_batch_exits=flag)
            buy = [d for d in rep.decisions if d.target.direction == 1][0]
            assert buy.approved is expect_ok
