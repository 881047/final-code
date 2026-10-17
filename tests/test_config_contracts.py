"""配置契约与 A 股交易规格（成本口径）测试。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from fterm import contracts
from fterm.config import load_config
from tests.conftest import _yaml_text


class TestConfig:
    def test_load_real_config(self):
        p = Path(__file__).resolve().parents[1] / "config.yaml"
        cfg = load_config(p)
        assert cfg.account.initial_equity == 20000
        assert cfg.signal.mode in ("rule_reversal", "model", "halt", "dlv2")
        assert cfg.data.old_db.endswith("neweco.db")

    def test_halt_mode_accepted(self, tmp_path):
        raw = yaml.safe_load(_yaml_text())
        raw["signal"]["mode"] = "halt"
        p = tmp_path / "c.yaml"
        p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
        cfg = load_config(p)
        assert cfg.signal.mode == "halt"

    def test_unknown_signal_mode_rejected(self, tmp_path):
        raw = yaml.safe_load(_yaml_text())
        raw["signal"]["mode"] = "rule_carry"      # 期货时代残留模式必须被拒
        p = tmp_path / "c.yaml"
        p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
        with pytest.raises(AssertionError, match="signal.mode"):
            load_config(p)

    def test_missing_section_fails(self, tmp_path):
        raw = yaml.safe_load(_yaml_text())
        del raw["signal"]
        p = tmp_path / "c.yaml"
        p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
        with pytest.raises(AssertionError, match="缺少段"):
            load_config(p)

    def test_extra_field_fails(self, tmp_path):
        raw = yaml.safe_load(_yaml_text())
        raw["signal"]["basis_z_min"] = 2.0        # 期货版残留字段必须被拒
        p = tmp_path / "c.yaml"
        p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
        with pytest.raises(AssertionError, match="多余字段"):
            load_config(p)

    def test_ema_keys_required_and_validated(self, tmp_path):
        """秩平滑语义必须显式入配置（2026-10-08）：alpha/step 不许再是代码常量。"""
        raw = yaml.safe_load(_yaml_text())
        del raw["signal"]["ema_step"]
        p = tmp_path / "c.yaml"
        p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
        with pytest.raises(AssertionError, match="缺字段"):
            load_config(p)

        raw = yaml.safe_load(_yaml_text())
        raw["signal"]["ema_step"] = "weekly"       # 只实现 daily | rebalance
        p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
        with pytest.raises(AssertionError, match="ema_step"):
            load_config(p)

        raw["signal"]["ema_step"] = "daily"
        raw["signal"]["ema_alpha"] = 1.5           # 新值权重必须 ∈ (0,1]
        p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
        with pytest.raises(AssertionError, match="ema_alpha"):
            load_config(p)

    def test_real_config_ema_semantics(self):
        """生产 config 的 EMA 语义必须是显式选择的（当前拍板：daily）。"""
        p = Path(__file__).resolve().parents[1] / "config.yaml"
        cfg = load_config(p)
        assert cfg.signal.ema_step in ("daily", "rebalance")
        assert 0.0 < cfg.signal.ema_alpha <= 1.0

    def test_account_validation(self, tmp_path):
        raw = yaml.safe_load(_yaml_text())
        raw["account"]["total_dd_halt"] = 0.02    # 必须 > daily_stop_loss
        p = tmp_path / "c.yaml"
        p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
        with pytest.raises(AssertionError, match="递增"):
            load_config(p)

    def test_fixed_codes_format(self, tmp_path):
        raw = yaml.safe_load(_yaml_text())
        raw["universe"]["codes"] = ["C", "M"]     # 期货品种代码不允许再进股票池
        p = tmp_path / "c.yaml"
        p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
        with pytest.raises(AssertionError, match="ts_code"):
            load_config(p)


class TestContracts:
    def test_lot_shares(self):
        assert contracts.shares_of(3) == 300
        assert contracts.LOT_SIZE == 100

    def test_commission_floor(self):
        d = pd.Timestamp("2026-01-05")
        # 1000 元成交额：万2.5=0.25 元 < 5 元下限 ⇒ 取 5 元（小资金成本大头，docs/02 登记）
        buy = contracts.trade_fee(1000.0, 1, d)
        assert buy == pytest.approx(5.0 + 1000 * contracts.transfer_fee_rate(d), rel=1e-9)
        # 5 万元成交额：佣金 = 12.5 元，超下限
        big = contracts.trade_fee(50000.0, 1, d)
        assert big == pytest.approx(50000 * 0.00025 + 50000 * contracts.transfer_fee_rate(d), rel=1e-9)

    def test_stamp_duty_time_segments(self):
        before = pd.Timestamp("2023-08-27")
        after = pd.Timestamp("2023-08-28")
        assert contracts.stamp_duty_rate(before) == 0.001
        assert contracts.stamp_duty_rate(after) == 0.0005
        # 买入永不收印花税
        assert contracts.trade_fee(10000, 1, before) == contracts.trade_fee(10000, 1, after)
        sell_before = contracts.trade_fee(10000, -1, before)
        sell_after = contracts.trade_fee(10000, -1, after)
        assert sell_before > sell_after

    def test_transfer_fee_time_segments(self):
        assert contracts.transfer_fee_rate(pd.Timestamp("2022-04-28")) == 0.00002
        assert contracts.transfer_fee_rate(pd.Timestamp("2022-04-29")) == 0.00001

    def test_fill_price_tick_rounding(self):
        from fterm.backtest.costs import fill_price_with_slippage
        up = fill_price_with_slippage(10.005, 1, 1)
        dn = fill_price_with_slippage(10.005, -1, 1)
        assert round(up, 2) == up and round(dn, 2) == dn
        assert up > 10.005 > dn                    # 滑点方向：买抬卖压
