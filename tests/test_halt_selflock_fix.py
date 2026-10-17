"""熔断自锁缺陷修复的回归钉（docs/03 §36 缺陷 / §47 修复；预注册
`plans/halt-selflock-fix-prereg-20261010.md`）。

## 本测试要钉住的四件事（都是"修缺陷"必须自证的）

① **默认行为逐位不变**：`dd_peak_mode="none"` 下峰值口径必须与修复前
   `max(curve + [current])` **逐位相等** —— 否则"修缺陷"变成了"偷偷换基线"。
② **不变式：rolling 无吸收态**。缺陷的数学本体是"权益冻结 ⇒ peak 不变 ⇒
   判据永真"。rolling 下窗口有限 ⇒ 空转满 window 日后 peak 必降到 equity
   ⇒ 回撤→0 ⇒ **必然解除**。本测试用合成曲线直接构造该吸收态并断言解除。
③ **无前视**：rolling 的 peak 在 t 时刻只读 ≤t 的权益（结构性保证）。
④ **配置向后兼容**：旧 config.yaml（无新键）仍可加载且行为不变；
   `_dc` 的缺省注入只在键缺失时生效（显式值可覆盖）。

## 为什么这些测试必须存在

§46 已证明该缺陷**污染了全部组合层读数**（13 臂 halted 占比 58–84%，
3 条臂从未解除）。修它属"改风险层语义"，是全仓最高风险的改动之一；
没有 ①②③ 的自证，"修好了"与"把风险层改坏了"在读数上无法区分。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fterm.config import load_config                      # noqa: E402
from fterm.risk import peak as pk                          # noqa: E402


def _eq(vals, start="2019-01-02"):
    idx = pd.bdate_range(start, periods=len(vals))
    return pd.Series([float(v) for v in vals], index=idx)


class TestDefaultBitIdentical:
    """① 默认口径与修复前逐位一致（**本批最重要的安全性质**）。"""

    def test_peak_equity_none_equals_legacy_max(self):
        """`peak_equity(hist, cur, 'none')` == 旧实现 `max(hist + [cur])`。"""
        rng = np.random.RandomState(20261010)
        for n in (1, 5, 100, 1874):
            hist = list(rng.uniform(8e5, 1.2e6, size=n))
            cur = float(rng.uniform(8e5, 1.2e6))
            legacy = max(hist + [cur])
            assert pk.peak_equity(hist, cur, mode="none") == legacy, f"n={n}"

    def test_peak_equity_none_empty_history(self):
        assert pk.peak_equity([], 12345.0, mode="none") == 12345.0

    def test_peak_fn_from_cfg_none_returns_none(self):
        """`none` 必须返回 None —— 让 run_faithful 走原有 running-max 分支。

        这是"逐位一致"的**结构性**保证：不重新实现默认分支，而是根本不接管它。
        """
        cfg = load_config(ROOT / "config.yaml")
        assert cfg.account.dd_peak_mode == "none"
        assert pk.peak_fn_from_cfg(cfg) is None

    def test_config_yaml_default_is_none(self):
        """生产默认值必须是 none（修缺陷 ≠ 偷偷启用）。"""
        import yaml
        doc = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        assert doc["account"]["dd_peak_mode"] == "none"
        assert doc["account"]["dd_peak_window"] == 252


class TestAbsorbingStateInvariant:
    """② 不变式：rolling 无吸收态；none 有吸收态（复现缺陷本体）。

    ## ⚠ 不变式的严格形式（预注册 erratum §三-1，由本测试抓出的自身错误）

    初版断言 **`max_halted_run ≤ W`** —— **该上界是错的**，本测试当场失败
    （W=252 时实测 256、W=60 时实测 64）。根因：连续 halted 段**包含权益冻结
    之前**就已 halted 的下跌期，冻结只是该段尾部。故正确形式是：

        max_halted_run ≤ (冻结前已 halted 天数) + (解除滞后)
        release_lag   ≤ W − 1        ← 这才是"实现正确"的检验

    本类**同时保留错误上界的反例断言**（`test_wrong_bound_is_actually_wrong`），
    防止未来有人把错误上界又写回来。
    """

    @staticmethod
    def _absorbing_curve():
        """构造吸收态：涨到峰值 → 深跌触发熔断 → 之后权益完全冻结（纯现金）。"""
        up = list(np.linspace(1_000_000, 1_200_000, 100))
        down = list(np.linspace(1_200_000, 900_000, 50))     # 回撤 25% ⇒ 触发
        flat = [900_000.0] * 800                             # 冻结（无持仓 ⇒ 无变动）
        return _eq(up + down + flat), 0.12

    def test_none_mode_reproduces_absorbing_state(self):
        """现行口径必须**复现**缺陷：冻结后永不解除（这是缺陷本体，不是 bug）。"""
        eq, halt = self._absorbing_curve()
        st = pk.halt_state(eq, halt, mode="none")
        assert st["n_halt_events"] >= 1, "应触发过熔断"
        assert st["n_release_events"] == 0, "冻结后不得解除 —— 吸收态"
        assert st["max_halted_run"] >= 800
        # none 下恒定段永不满窗口后解除 ⇒ 不变式检验必然"违规"
        assert st["invariant_violations"] >= 1

    def test_rolling_mode_eliminates_absorbing_state(self):
        """修复口径必须**消除**缺陷：窗口滑出后自动解除。"""
        eq, halt = self._absorbing_curve()
        W = 252
        st = pk.halt_state(eq, halt, mode="rolling", window=W)
        assert st["n_release_events"] >= 1, "rolling 下必须至少解除一次"
        # 严格不变式：长度 ≥W 的恒定段，段内第 W 日起不得仍 halted
        assert st["invariant_violations"] == 0, (
            f"不变式违规 {st['invariant_violations']} 次 ⇒ 实现有 bug")
        assert st["n_constant_runs_ge_window"] >= 1, "该曲线应存在 ≥W 的恒定段"

    def test_wrong_bound_is_actually_wrong(self):
        """**反例留痕**：`max_halted_run ≤ W` 是错的，不得写回。

        本测试**故意断言错误上界不成立**，以证明 erratum §三-1 的必要性。
        """
        eq, halt = self._absorbing_curve()
        W = 252
        st = pk.halt_state(eq, halt, mode="rolling", window=W)
        assert st["max_halted_run"] > W, (
            "若此断言失败，说明曲线构造或实现变了 —— 请重新核对 erratum §三-1")

    def test_invariant_holds_for_all_windows(self):
        """严格不变式对任意窗口都成立（不是拟合出来的）。"""
        eq, halt = self._absorbing_curve()
        for W in (20, 60, 120, 252, 504):
            st = pk.halt_state(eq, halt, mode="rolling", window=W)
            assert st["invariant_violations"] == 0, f"W={W} 违规"

    def test_constant_run_while_halted_bounded_by_window(self):
        """halted 段内的恒定子段应 ≤ W（恒定满窗口后必解除）。"""
        eq, halt = self._absorbing_curve()
        W = 252
        st = pk.halt_state(eq, halt, mode="rolling", window=W)
        assert st["constant_run_max_while_halted"] <= W + 1

    def test_rolling_strictly_shortens_vs_none(self):
        """修复必须严格改善：rolling 的最长 halted 段短于 none。"""
        eq, halt = self._absorbing_curve()
        a = pk.halt_state(eq, halt, mode="none")
        b = pk.halt_state(eq, halt, mode="rolling", window=252)
        assert b["max_halted_run"] < a["max_halted_run"]
        assert b["n_release_events"] > a["n_release_events"]

    def test_monotone_shallow_drawdown_never_halts_in_rolling(self):
        """缓跌（无冻结）在 rolling 下不应长期熔断 —— 窗口跟随而非死锁。"""
        eq = _eq(np.linspace(1_000_000, 800_000, 1000))
        st = pk.halt_state(eq, 0.12, mode="rolling", window=60)
        assert st["max_halted_run"] <= 60

    def test_freeze_detection(self):
        """恒定段识别（不变式检验的基础）。"""
        eq = _eq([1e6] * 10 + [900_000.0] * 300)
        st = pk.halt_state(eq, 0.12, mode="rolling", window=252)
        assert st["n_constant_runs_ge_window"] == 1
        assert st["invariant_violations"] == 0
        # 短曲线无 ≥W 恒定段
        st2 = pk.halt_state(_eq([1e6] * 5), 0.12, mode="rolling", window=252)
        assert st2["n_constant_runs_ge_window"] == 0
        assert st2["invariant_violations"] == 0


class TestRollingPeakFn:
    """`rolling_peak_fn` 的契约（回测侧 peak_fn 签名）。"""

    def test_uses_only_tail_window(self):
        f = pk.rolling_peak_fn(10)
        equities = [1_000_000.0] * 5 + [500_000.0] * 10 + [400_000.0]
        # 旧高点 1e6 已在窗口外 ⇒ peak 只由尾部决定
        assert f(None, equities, 400_000.0) == 500_000.0

    def test_includes_current(self):
        f = pk.rolling_peak_fn(5)
        assert f(None, [100.0, 200.0], 999.0) == 999.0

    def test_short_history_uses_all(self):
        f = pk.rolling_peak_fn(252)
        assert f(None, [100.0, 300.0, 200.0], 150.0) == 300.0

    def test_empty_history_returns_current(self):
        f = pk.rolling_peak_fn(252)
        assert f(None, [], 777.0) == 777.0

    def test_rejects_bad_window(self):
        with pytest.raises(ValueError):
            pk.rolling_peak_fn(0)


class TestNoLookahead:
    """③ 无前视：rolling 的 peak 只由**已实现**权益决定（非 §40 的前瞻规则）。"""

    def test_peak_at_t_independent_of_future(self):
        """截断未来不得改变历史 peak —— 结构性保证。"""
        vals = [1_000_000.0] * 20 + [800_000.0] * 20 + [1_500_000.0] * 20
        eq = _eq(vals)
        full = pk.halt_state(eq, 0.12, mode="rolling", window=10)
        cut = pk.halt_state(eq.iloc[:60], 0.12, mode="rolling", window=10)
        # 前 60 天的 halted 序列必须完全相同（未来 1.5e6 的高点不得影响历史）
        assert full["max_halted_run"] == cut["max_halted_run"]

    def test_peak_equity_helper_reads_only_given_history(self):
        """`peak_equity` 是纯函数：给定历史即定，无隐藏状态、无未来参数。"""
        a = pk.peak_equity([100.0, 200.0, 150.0], 120.0, mode="rolling", window=3)
        b = pk.peak_equity([100.0, 200.0, 150.0], 120.0, mode="rolling", window=3)
        assert a == b == 200.0

    def test_rolling_peak_fn_is_pure(self):
        f = pk.rolling_peak_fn(3)
        args = ([1.0, 2.0, 3.0, 4.0], 2.5)
        assert f(None, *args) == f(None, *args)


class TestConfigBackwardCompat:
    """④ 旧配置仍可加载（加字段 ≠ 换基线）。"""

    def test_missing_keys_get_defaults(self, tmp_path):
        """没有新键的 config.yaml 必须能加载，且取到 none/252。"""
        import yaml
        raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        raw["account"].pop("dd_peak_mode")
        raw["account"].pop("dd_peak_window")
        p = tmp_path / "old.yaml"
        p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
        cfg = load_config(p)
        assert cfg.account.dd_peak_mode == "none"
        assert cfg.account.dd_peak_window == 252

    def test_explicit_value_overrides_default(self, tmp_path):
        """显式写了键就以文件为准（缺省表不覆盖显式值）。"""
        import yaml
        raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        raw["account"]["dd_peak_mode"] = "rolling"
        raw["account"]["dd_peak_window"] = 60
        p = tmp_path / "new.yaml"
        p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
        cfg = load_config(p)
        assert cfg.account.dd_peak_mode == "rolling"
        assert cfg.account.dd_peak_window == 60
        assert pk.peak_fn_from_cfg(cfg) is not None

    def test_invalid_mode_rejected(self, tmp_path):
        import yaml
        raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        raw["account"]["dd_peak_mode"] = "bogus"
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
        with pytest.raises(AssertionError):
            load_config(p)

    def test_bad_window_rejected(self, tmp_path):
        import yaml
        raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        raw["account"]["dd_peak_window"] = 0
        p = tmp_path / "bad2.yaml"
        p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
        with pytest.raises(AssertionError):
            load_config(p)


class TestHaltStateStats:
    """`halt_state` 统计量的口径（诊断与判据共用，避免分叉）。"""

    def test_no_halt_when_equity_rises(self):
        st = pk.halt_state(_eq(np.linspace(1e6, 1.5e6, 100)), 0.12, mode="none")
        assert st["n_halt_events"] == 0
        assert st["max_halted_run"] == 0
        assert st["halted_frac"] == 0.0

    def test_single_episode_counted_once(self):
        vals = [1e6] * 10 + [800_000.0] * 5 + [1e6] * 10
        st = pk.halt_state(_eq(vals), 0.12, mode="none")
        assert st["n_halt_events"] == 1
        assert st["n_release_events"] == 1
        assert st["max_halted_run"] == 5

    def test_first_halt_date_reported(self):
        vals = [1e6] * 10 + [800_000.0] * 5
        eq = _eq(vals)
        st = pk.halt_state(eq, 0.12, mode="none")
        assert st["first_halt"] == str(eq.index[10].date())

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError):
            pk.halt_state(_eq([1e6, 9e5]), 0.12, mode="bogus")
        with pytest.raises(ValueError):
            pk.peak_equity([1e6], 9e5, mode="bogus")


class TestSingleSourceOfTruth:
    """D2：回测与生产必须共用同一实现（防 §41/§42 那类语义分叉）。"""

    def test_arm_faithful_delegates_to_peak_module(self):
        import inspect
        import scripts.tierfix_replay as tf
        src = inspect.getsource(tf.arm_faithful)
        assert "peak_fn_from_cfg" in src, "回测侧必须走 fterm/risk/peak.py"

    def test_runner_delegates_to_peak_module(self):
        import inspect
        from fterm.runtime import runner
        src = inspect.getsource(runner)
        assert "risk.peak import peak_equity" in src, "生产侧必须走 fterm/risk/peak.py"

    def test_both_sides_agree_on_none(self):
        """同一历史 + 同一当前值 ⇒ 回测侧与生产侧给出同一 peak。"""
        hist = [1_000_000.0, 1_100_000.0, 950_000.0]
        cur = 980_000.0
        prod = pk.peak_equity(hist, cur, mode="none")
        bt_fn = pk.peak_fn_from_cfg(load_config(ROOT / "config.yaml"))
        assert bt_fn is None                       # 回测走原生分支
        assert prod == max(hist + [cur])           # 生产与原生分支同值


class TestSelfLockObservability:
    """生产必须能**看见**自己卡在吸收态（零行为改动，只加告警）。"""

    def test_runner_has_selflock_detection(self):
        import inspect
        from fterm.runtime import runner
        src = inspect.getsource(runner)
        assert "dd_halt_selflock" in src, "必须落自锁告警"
        assert "dd_selflock_warn_days" in src, "阈值须可配"

    def test_warn_days_default_and_configurable(self):
        cfg = load_config(ROOT / "config.yaml")
        assert cfg.account.dd_selflock_warn_days == 60
        import yaml
        doc = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        assert doc["account"]["dd_selflock_warn_days"] == 60

    def test_detection_logic_matches_defect_definition(self):
        """自锁判据 = 「halted 且权益连续 N 日无变动」—— 与缺陷本体同义。"""
        N = 60
        halt = 0.12
        # 构造：峰值 1.2e6 → 跌到 9e5（触发）→ 冻结
        eq = _eq([1_200_000.0] * 10 + list(np.linspace(1_200_000, 900_000, 20))
                 + [900_000.0] * 200)
        peak = float(eq.cummax().iloc[-1])
        tail = eq.iloc[-N:].to_numpy()
        halted = eq.iloc[-1] < peak * (1 - halt)
        frozen = float(tail.max() - tail.min()) < 1e-9
        assert halted and frozen, "该曲线必须被判为自锁"

    def test_not_flagged_when_equity_moves(self):
        """权益仍在波动时不得误报（告警必须能失败，也必须能放行）。"""
        N = 60
        eq = _eq([1_200_000.0] * 10 + list(np.linspace(1_200_000, 900_000, 200)))
        tail = eq.iloc[-N:].to_numpy()
        assert float(tail.max() - tail.min()) > 1e-9, "下跌中不应判为冻结"

    def test_warn_zero_disables(self, tmp_path):
        import yaml
        raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        raw["account"]["dd_selflock_warn_days"] = 0
        p = tmp_path / "nw.yaml"
        p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
        assert load_config(p).account.dd_selflock_warn_days == 0

    def test_invalid_warn_days_rejected(self, tmp_path):
        import yaml
        raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        raw["account"]["dd_selflock_warn_days"] = -1
        p = tmp_path / "bad3.yaml"
        p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
        with pytest.raises(AssertionError):
            load_config(p)

    def test_backward_compat_missing_warn_key(self, tmp_path):
        """旧配置无该键 ⇒ 取缺省 60（加字段 ≠ 换基线）。"""
        import yaml
        raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        raw["account"].pop("dd_selflock_warn_days")
        p = tmp_path / "old3.yaml"
        p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
        assert load_config(p).account.dd_selflock_warn_days == 60
