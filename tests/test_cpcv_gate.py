"""CPCV 稳定性门（预注册 §三）执行器与口径验收（2026-10-10 补入回归钉）。

## 背景（为什么需要这个测试）

预注册 `plans/model-upgrade-prereg-20261009.md §三` 的「CPCV 稳定性」门
（`P(IC>0) ≥ 0.9` 且路径间 IC 标准差低于 F0）**从未被计算过**：
`fterm/model/cpcv.py` 已实现净化划分并有 `tests/test_cpcv.py` 钉住，
但 grep 全部 `scripts/*.py` **零命中** —— 没有任何执行器调用它，
故该门对 §44 的臂 A/B/C 与 §45 的 D/E/E2 **都无读数**（docs/03 §45 已知缺口 #1）。
2026-10-10 把它接进 `scripts/stability_chain.py`。

## 本测试钉住的口径（四件，都是本函数实现时真实踩到的坑）

① **单流退化**：只吃一条预测流时，路径级 `path_ic` 逐位置都等于原序列
   ⇒ `path_ic_std` **恰为 0.0**（初版误按"划分 IC 的均值"算，得到 std=0.0065
   —— 那是**错的**，AFML 铺砖必须逐位置重组）。若把该退化读数当判据，
   C2 变成 `0.0 < 0.0` ⇒ 门对任何臂**无条件失败**（永不开的门 = 无信息）。
   故判据取**划分级**读数，且必须响亮标注退化。
② **铺砖完整性**：每个 (划分, 测试组) 对被恰好用一次，每条路径恰好覆盖全部 N 组
   各一次 ⇒ φ=C(N-1,k-1) 条路径（与 `n_paths` 的 C(N,k)·k/N 恒等，两条公式都钉）。
③ **组界同源**：`cpcv_groups` 必须与 `cpcv_date_splits` 用同一 `linspace` 边界，
   否则分组与净化划分会静默分叉。
④ **`split_daily_ic` 才是路径级的真正入口**：传入「每个划分各训一个模型」的
   全长 IC 序列时，路径级读数必须有分辨力（非退化）。
"""
from __future__ import annotations

import sys
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fterm.model import cpcv as cv  # noqa: E402
from fterm.model.cpcv import (  # noqa: E402
    cpcv_date_splits,
    cpcv_gate,
    cpcv_groups,
    cpcv_ic_stability,
    cpcv_path_assignments,
    n_paths,
)


class TestGroupsSameBoundaryAsSplits:
    """③ 组界必须与 `cpcv_date_splits` 同源（否则两条路径静默分叉）。"""

    def test_groups_match_linspace_bounds(self):
        T, N = 60, 6
        bounds = np.linspace(0, T, N + 1, dtype=int)
        gs = cpcv_groups(T, N)
        assert len(gs) == N
        for i, g in enumerate(gs):
            assert g.tolist() == list(range(bounds[i], bounds[i + 1]))

    def test_groups_tile_all_positions_once(self):
        gs = cpcv_groups(58, 6)          # 非整除也须无重不漏
        allpos = np.concatenate(gs)
        assert sorted(allpos.tolist()) == list(range(58))

    def test_test_sets_of_splits_are_exactly_group_unions(self):
        """划分的测试位置集必须恰是某些组的并集（与分组同源）。"""
        T, N, k = 60, 6, 2
        gs = cpcv_groups(T, N)
        got = sorted(tuple(sorted(te.tolist()))
                     for _tr, te in cpcv_date_splits(list(range(T)), N, k, 6, 0))
        want = sorted(tuple(sorted(np.concatenate([gs[g] for g in c]).tolist()))
                      for c in combinations(range(N), k))
        assert got == want

    def test_illegal_groups_raise(self):
        with pytest.raises(ValueError):
            cpcv_groups(30, 1)
        with pytest.raises(ValueError):
            cpcv_groups(3, 6)


class TestPathTiling:
    """② 铺砖完整性：每条路径覆盖全部 N 组各一次，每个 (划分,组) 对恰用一次。"""

    @pytest.mark.parametrize("N,k", [(6, 2), (6, 1), (8, 2), (5, 3)])
    def test_tiling_is_exact_cover(self, N, k):
        combos = list(combinations(range(N), k))
        paths = cpcv_path_assignments(N, k)
        # 每条路径：{组: 划分}，且该划分必须真的测试该组
        for a in paths:
            assert sorted(a) == list(range(N))
            for g, s in a.items():
                assert g in combos[s]
        # 每个 (划分, 测试组) 对恰好出现一次
        used = sorted((s, g) for a in paths for g, s in a.items())
        want = sorted((s, g) for s, c in enumerate(combos) for g in c)
        assert used == want

    @pytest.mark.parametrize("N,k", [(6, 2), (8, 2), (5, 3)])
    def test_path_count_matches_both_formulas(self, N, k):
        """φ = C(N-1,k-1)（铺砖公式）必须等于 n_paths 的 C(N,k)·k/N。"""
        assert len(cpcv_path_assignments(N, k)) == comb(N - 1, k - 1)
        assert float(comb(N - 1, k - 1)) == pytest.approx(n_paths(N, k))

    def test_illegal_params_raise(self):
        with pytest.raises(ValueError):
            cpcv_path_assignments(6, 6)
        with pytest.raises(ValueError):
            cpcv_path_assignments(1, 1)


class TestSingleStreamDegeneracy:
    """① 单流下路径级必然退化 —— 这是判据取划分级的**唯一理由**。"""

    def _ic(self, T=600, seed=0):
        rng = np.random.default_rng(seed)
        return rng.normal(0.05, 0.1, size=T)

    def test_path_std_is_exactly_zero(self):
        r = cpcv_ic_stability(self._ic(), n_groups=6, n_test_groups=2)
        assert r["path_ic_std"] == 0.0, "单流下 AFML 铺砖逐位置等于原序列 ⇒ std 恰为 0"
        assert r["path_std_degenerate"] is True
        assert r["n_paths"] == 5
        # 每条路径都等于全样本 IC
        assert all(v == pytest.approx(float(np.mean(self._ic()))) for v in r["path_ic"])

    def test_path_level_gate_would_fail_unconditionally(self):
        """反例守卫：照抄路径级读数 ⇒ C2 恒假，门永不开（故必须用划分级）。"""
        r = cpcv_ic_stability(self._ic(seed=1), n_groups=6, n_test_groups=2)
        g = cpcv_gate(r, r, p_min=0.9)          # 自比：std 与 f0 同为 0.0
        # 若判据取路径级，C2 = (0.0 < 0.0) = False
        assert not (r["path_ic_std"] < r["path_ic_std"])
        # 实际实现取划分级 ⇒ C2 在自比时为 False（不劣不算改善），但 C1 可为真
        assert g["c2_split_ic_std_below_f0"]["value"] == r["split_ic_std"]
        assert g["c2_split_ic_std_below_f0"]["f0"] == r["split_ic_std"]
        assert g["pass"] is False, "自比 std 相等 ⇒ 不构成改善 ⇒ 门不得通过"

    def test_split_std_is_not_degenerate(self):
        """划分级读数必须有分辨力（否则整门无意义）。"""
        r = cpcv_ic_stability(self._ic(seed=2), n_groups=6, n_test_groups=2)
        assert r["n_splits"] == 15
        assert r["split_ic_std"] > 0.0
        assert len(set(np.round(r["split_ic"], 12).tolist())) > 1

    def test_split_ic_matches_manual_group_means(self):
        """划分 IC 必须等于「该划分测试组并集内日频 IC 的均值」（手工复算）。"""
        v = self._ic(T=60, seed=3)
        r = cpcv_ic_stability(v, n_groups=6, n_test_groups=2)
        gs = cpcv_groups(60, 6)
        want = [float(v[np.concatenate([gs[g] for g in c])].mean())
                for c in combinations(range(6), 2)]
        assert r["split_ic"] == pytest.approx(want)


class TestSplitDailyIcPathInput:
    """④ `split_daily_ic` 是路径级的真正入口（多模型场景才有分辨力）。"""

    def test_non_degenerate_when_per_split_series_differ(self):
        T = 600
        rng = np.random.default_rng(7)
        base = rng.normal(0.05, 0.1, size=T)
        # 15 个划分，各给一条「略微不同」的全长序列
        per = {s: base + rng.normal(0, 0.02, size=T) for s in range(15)}
        r = cpcv_ic_stability(base, n_groups=6, n_test_groups=2, split_daily_ic=per)
        assert r["path_ic_std"] > 0.0, "划分各训一个模型时路径级必须有分辨力"
        assert r["path_std_degenerate"] is False

    def test_missing_split_raises(self):
        T = 600
        v = np.random.default_rng(0).normal(0.05, 0.1, size=T)
        with pytest.raises(ValueError):
            cpcv_ic_stability(v, n_groups=6, n_test_groups=2,
                              split_daily_ic={0: v})

    def test_wrong_length_raises(self):
        T = 600
        v = np.random.default_rng(0).normal(0.05, 0.1, size=T)
        with pytest.raises(ValueError):
            cpcv_ic_stability(v, n_groups=6, n_test_groups=2,
                              split_daily_ic={s: v[:100] for s in range(15)})


class TestGateJudgement:
    """门的两条判据：`P(IC>0) ≥ 0.9` 且 std 低于 F0。"""

    def _reading(self, p, std):
        return {"p_ic_pos_splits": p, "split_ic_std": std,
                "p_ic_pos_paths": 1.0, "path_ic_std": 0.0}

    def test_pass_requires_both(self):
        f0 = self._reading(0.93, 0.02)
        assert cpcv_gate(self._reading(1.0, 0.01), f0)["pass"] is True
        # C1 过、C2 不过（std 不低于 F0）
        assert cpcv_gate(self._reading(1.0, 0.03), f0)["pass"] is False
        # C2 过、C1 不过
        assert cpcv_gate(self._reading(0.80, 0.01), f0)["pass"] is False

    def test_p_min_boundary_is_inclusive(self):
        f0 = self._reading(0.5, 0.02)
        assert cpcv_gate(self._reading(0.9, 0.01), f0)["c1_p_ic_pos_ge"]["ok"] is True
        assert cpcv_gate(self._reading(0.899999, 0.01), f0)["c1_p_ic_pos_ge"]["ok"] is False

    def test_none_values_do_not_pass(self):
        """缺读数（None）不得静默通过。"""
        f0 = self._reading(None, None)
        g = cpcv_gate(self._reading(None, None), f0)
        assert g["c1_p_ic_pos_ge"]["ok"] is False
        assert g["c2_split_ic_std_below_f0"]["ok"] is False
        assert g["pass"] is False

    def test_gate_reports_split_level_and_flags_degeneracy(self):
        g = cpcv_gate(self._reading(1.0, 0.01), self._reading(1.0, 0.02))
        assert "退化" in g["note"]


class TestExecutorWiring:
    """门必须真的被 `scripts/stability_chain.py` 调用（缺口 #1 的核心）。"""

    def test_source_calls_cpcv(self):
        src = (ROOT / "scripts" / "stability_chain.py").read_text(encoding="utf-8")
        assert "from fterm.model import cpcv as cv" in src
        assert "cv.cpcv_ic_stability(" in src
        assert "cv.cpcv_gate(" in src
        assert '"cpcv"' in src and '"cpcv_gate"' in src

    def test_source_keeps_legacy_rolling_key(self):
        """§45 缺口 #3 的硬约束：旧键 `nw_t` 必须保留（既有读数逐位可复现）。"""
        src = (ROOT / "scripts" / "stability_chain.py").read_text(encoding="utf-8")
        assert '"nw_t": roll["nw_t_of_daily_ic"]' in src

    def test_cpcv_constants_match_prereg(self):
        """§三 门槛与划分参数须与预注册一致。"""
        import scripts.stability_chain as sc
        assert sc.CPCV_P_MIN == 0.9
        assert sc.CPCV_N_GROUPS == 6
        assert sc.CPCV_N_TEST_GROUPS == 2
        assert sc.ROLL_WINDOW == 60

    def test_reading_failure_is_failsafe_not_silent_pass(self):
        """读数不可算时必须留痕且**不得**报 pass=True（失败安全）。

        与 `arch_chain` 的 `tail` / `judgment_chain` 的 `regime_error` 同款处置：
        新增门既不能让既有调用崩溃，也不能静默报"通过"。
        """
        src = (ROOT / "scripts" / "stability_chain.py").read_text(encoding="utf-8")
        # 留痕分支
        assert '"error": cpcv_err[name]' in src or 'cpcv_err[name]' in src
        # 不可算 ⇒ pass 显式为 False
        assert '"pass": False, "error"' in src
        # 打印路径也必须容错（不得 KeyError 崩溃）
        assert "不可算" in src

    def test_too_short_stream_raises_loudly(self):
        """过短的流：`cpcv_groups` 必须响亮报错（由执行器留痕，不静默）。"""
        with pytest.raises(ValueError):
            cpcv_ic_stability(np.array([0.1, 0.2, 0.3]), n_groups=6, n_test_groups=2)
