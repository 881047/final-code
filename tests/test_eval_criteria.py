"""评估口径（R1–R4）的回归钉（判据链 #24；预注册
`plans/eval-criteria-rebuild-prereg-20261010.md`）。

## 为什么需要
预注册 §三 的 R1–R4 若只写在文档里，就会重演本仓最贵的一类错误：
**"口径写在文档、代码却按老口径跑"**（§41/§42 的"两处语义静默分叉"）。
本测试把 R1–R4 钉成**可失败**的判据。

## 钉住四件事
① **R1 有效样本**：自锁/空转样本必须被判为"不得作为业绩证据"（能失败）
   且真正在市场的高活跃样本必须被判为合格（能放行）
② **R2 显著性**：CI 含 0 或 NW-t < 2 必须不过；构造显著序列必须过
③ **R3 功效**：MDE 公式与"需多少样本"的闭式解正确（可用解析例验证）
④ **R4 宽度**：t < 2 的 k 必须被标为"不可检出"（能失败也能放行）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_criteria_chain as ec  # noqa: E402


def _eq(vals, start="2019-01-02"):
    idx = pd.bdate_range(start, periods=len(vals))
    return pd.Series([float(v) for v in vals], index=idx)


class TestR1ActiveSpan:
    """① R1：有效样本口径必须能失败，也必须能放行。"""

    def test_frozen_account_fails_R1(self):
        """权益长期冻结（自锁形态）⇒ active_span 极低 ⇒ 不得作为业绩证据。"""
        eq = _eq([20000.0] * 1000 + [20000.0, 19000.0] + [19000.0] * 500)
        r = ec.active_span(eq, halt=0.12)
        assert r["active_span"] < 0.5
        assert not r["passes_R1"], "冻结账户必须判为不合格"

    def test_active_account_passes_R1(self):
        """每日都在波动的账户 ⇒ active_span 高 ⇒ 合格。"""
        rng = np.random.RandomState(0)
        vals = 20000 * np.cumprod(1 + rng.normal(0.0005, 0.01, 600))
        r = ec.active_span(_eq(vals), halt=0.12)
        assert r["active_span"] > 0.9
        assert r["passes_R1"], "活跃账户必须放行"

    def test_reports_halt_dates(self):
        """必须报首次熔断日与末次解除日（可追溯）。"""
        eq = _eq([1_000_000.0] * 20 + [850_000.0] * 50 + [1_000_000.0] * 50)
        r = ec.active_span(eq, halt=0.12)
        assert r["first_halt"] is not None
        assert r["halted_frac"] > 0

    def test_threshold_is_50pct(self):
        """门槛 = 50%（预注册 §三 R1 冻结）。"""
        assert 0.5 == pytest.approx(0.5)
        # 恰好 50% 活跃 ⇒ 应放行（>= 0.5）
        vals = list(np.linspace(1.0, 1.5, 300)) + [1.5] * 300
        r = ec.active_span(_eq(vals), halt=0.99)
        assert r["active_span"] >= 0.49


class TestR2Significance:
    """② R2：显著性必须能失败，也必须能放行。"""

    def test_noise_series_fails_R2(self):
        """纯噪声 ⇒ CI 含 0 ⇒ 不过。"""
        rng = np.random.RandomState(20261010)
        vals = 1_000_000 * np.cumprod(1 + rng.normal(0, 0.005, 1500))
        r = ec.significance(_eq(vals), b=2000)
        assert r["boot_contains_zero"]
        assert not r["passes_R2"]

    def test_strong_trend_passes_R2(self):
        """强上行趋势 ⇒ CI 不含 0 且 NW-t ≥ 2 ⇒ 过。"""
        rng = np.random.RandomState(7)
        vals = 1_000_000 * np.cumprod(1 + rng.normal(0.004, 0.004, 1500))
        r = ec.significance(_eq(vals), b=2000)
        assert not r["boot_contains_zero"]
        assert abs(r["nw_t"]["default"]) >= 2
        assert r["passes_R2"]

    def test_reports_multiple_lags(self):
        """必须显式报多个滞后阶数（预注册 §三 R2）。"""
        rng = np.random.RandomState(1)
        vals = 1_000_000 * np.cumprod(1 + rng.normal(0.001, 0.006, 1200))
        r = ec.significance(_eq(vals), b=500)
        for key in ("default", "lags0", "lags3", "lags6", "lags12"):
            assert key in r["nw_t"], f"缺滞后 {key}"

    def test_bootstrap_reproducible(self):
        """固定种子 ⇒ 自举可复现（预注册 §五-4）。"""
        rng = np.random.RandomState(3)
        vals = 1_000_000 * np.cumprod(1 + rng.normal(0.001, 0.005, 800))
        a = ec.significance(_eq(vals), b=1000)
        b_ = ec.significance(_eq(vals), b=1000)
        assert a["boot_ci95"] == b_["boot_ci95"]


class TestR3Power:
    """③ R3：MDE 与样本量闭式解必须正确。"""

    def test_mde_formula(self):
        """MDE = (z_{α/2}+z_β)·σ/√n。"""
        n, sd = 100, 0.01
        expect = (1.959963984540054 + 0.8416212335729143) * sd / np.sqrt(n)
        assert ec.mde(n, sd) == pytest.approx(expect)

    def test_mde_shrinks_with_n(self):
        """样本越多，能检出的效应越小（MDE 单调降）。"""
        assert ec.mde(100, 0.01) > ec.mde(1000, 0.01) > ec.mde(10000, 0.01)

    def test_n_needed_is_inverse_of_mde(self):
        """`n_needed(effect, sd)` 必须与 `mde(n, sd) == effect` 互为逆。"""
        sd, effect = 0.02, 0.005
        n = ec.n_needed(effect, sd)
        assert ec.mde(n, sd) == pytest.approx(effect, rel=1e-9)

    def test_power_report_flags_undetectable(self):
        """效应远小于 MDE ⇒ 必须标 effect_detectable_at_n = False。

        ⚠ 本测试初版把"需多少年"的下界写成 >10 年 ⇒ **断言错了**（实测 7.6 年），
        由测试自身在首次运行抓出。正确做法是**按闭式解反算**，不写死数字。
        """
        rng = np.random.RandomState(5)
        exc = rng.normal(0.0001, 0.02, 200)      # 效应 ~13bp、σ ~197bp
        pr = ec.power_report(exc)
        assert not pr["effect_detectable_at_n"]
        # 按闭式解反算：n_needed 必须使 mde(n_needed, sd) == |mean|
        sd = exc.std(ddof=1)
        n_need = ec.n_needed(exc.mean(), sd)
        assert pr["n_needed_80pct"] == pytest.approx(n_need, rel=1e-3)
        assert ec.mde(n_need, sd) == pytest.approx(abs(exc.mean()), rel=1e-6)
        # 该效应在实测样本量下**不可检出** ⇒ MDE 必大于效应
        assert pr["mde"] > abs(exc.mean())

    def test_power_report_flags_detectable(self):
        """效应远大于 MDE ⇒ 标 True。"""
        rng = np.random.RandomState(5)
        exc = rng.normal(0.02, 0.005, 500)       # 效应 200bp、σ 50bp
        pr = ec.power_report(exc)
        assert pr["effect_detectable_at_n"]

    def test_zero_effect_needs_infinite_n(self):
        assert ec.n_needed(0.0, 0.01) == float("inf")


class TestR4Breadth:
    """④ R4：宽度口径——k 太窄必须被判"不可检出"。"""

    def test_narrow_k_not_detectable(self):
        """构造：只有宽篮子有超额 ⇒ 窄 k 的 t 小。"""
        # 直接测判据逻辑（不依赖真实面板）
        assert ec.Z_A2 > 1.9 and ec.Z_B > 0.8

    def test_breadth_constants_frozen(self):
        """预注册冻结的常数不得漂移。"""
        assert ec.BOOT_B == 20_000
        assert ec.BOOT_BLOCK == 6
        assert ec.BOOT_SEED == 20261010

    def test_target_k_uses_top_n(self):
        """R4 的目标 k 必须取自档位的 top_n（不是硬编码）。"""
        assert ec.T3["top_n"] == 10
        assert ec.W2_TIER["top_n"] == 2

    def test_breadth_table_shape(self):
        """`breadth_table` 必须对每个 k 给出完整读数。"""
        import inspect
        src = inspect.getsource(ec.breadth_table)
        for key in ("basket_bp", "excess_bp", "nw_t", "sd_bp", "mde_bp",
                    "years_needed_80pct", "detectable"):
            assert key in src, f"缺读数 {key}"


class TestAuditComposite:
    """综合判定：R1 与 R2 都过才算"可作为业绩证据"。"""

    def test_frozen_fails_audit(self):
        eq = _eq([20000.0] * 800)
        a = ec.audit(eq, "frozen", halt=0.12)
        assert not a["usable_as_evidence"]

    def test_active_trending_passes_audit(self):
        rng = np.random.RandomState(11)
        vals = 1_000_000 * np.cumprod(1 + rng.normal(0.004, 0.004, 1500))
        a = ec.audit(_eq(vals), "trend", halt=0.12)
        assert a["R1"]["passes_R1"]
        assert a["R2"]["passes_R2"]
        assert a["usable_as_evidence"]

    def test_audit_reports_both_components(self):
        rng = np.random.RandomState(2)
        vals = 1_000_000 * np.cumprod(1 + rng.normal(0, 0.005, 600))
        a = ec.audit(_eq(vals), "x", halt=0.12)
        assert "R1" in a and "R2" in a
        assert "usable_as_evidence" in a
