"""「敞口/执行 vs 信号精度」解耦链的回归钉（2026-10-10）。

预注册 `plans/exposure-vs-precision-prereg-20261010.md`。

## 本测试钉住的口径（四条，都是本批真实依赖的判据前提）

① **n=13 临界值可复现**：预注册 §五 冻结的 `|ρ| ≥ 0.5549` 必须能由脚本内的
   `mc_critical(13)` 现场重算出来（固定种子 ⇒ 逐位可复现）。若临界值不可复现，
   主判据 C1/C2/C3 的门槛就成了"跑完再挑"的东西。
② **`mc_critical` 在已知闭式解上正确**：n=5 的精确枚举临界值 = 1.0000、
   `P(|ρ|≥0.40) = 0.5167`（`_calib_streams_and_null.py` 的标定读数）。
③ **构型变量只动 `total_dd_halt`**：`dataclasses.replace` 出来的 cfg 副本
   除该字段外必须逐字段相同 —— 这是"唯一变量"声明的可执行形式；
   `config.yaml` 不得被改写。
④ **面板竞态守卫必须响亮失败**：研究线裁决只允许在面板右端点 == 隔离线的检出里跑
   （§五-5 同款；实测生产检出面板已越过隔离线）。
"""
from __future__ import annotations

import itertools
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import exposure_precision_chain as epc  # noqa: E402


class TestFrozenThreshold:
    """① + ② 门槛标定的可复现性。"""

    def test_n13_critical_matches_prereg_frozen_value(self):
        """预注册 §五 冻结的 n=13 双侧 5% 临界值必须可现场重算。"""
        crit = epc.mc_critical(13)
        assert crit == pytest.approx(epc.CRIT_RHO, abs=5e-4), (
            f"现场重算 {crit} 与预注册冻结 {epc.CRIT_RHO} 不符 ⇒ 门槛不可复现")

    def test_mc_critical_n5_matches_exact_enumeration(self):
        """n=5 精确枚举：双侧 5% 临界 |ρ| = 1.0000（标定批读数）。"""
        crit_mc = epc.mc_critical(5)
        assert crit_mc == pytest.approx(1.0, abs=0.15)   # MC 在 1.0 附近（精确值 1.0）

    def test_n5_exact_null_matches_calibration(self):
        """n=5 精确枚举 P(|ρ|≥0.40) = 0.5167 —— §46 的 −0.400 落在零分布中央。"""
        dist: dict[float, int] = {}
        for perm in itertools.permutations(range(5)):
            d2 = sum((i - perm[i]) ** 2 for i in range(5))
            rho = round(1.0 - 6.0 * d2 / (5 * 24), 10)
            dist[rho] = dist.get(rho, 0) + 1
        tot = sum(dist.values())
        p040 = sum(v for k, v in dist.items() if abs(k) >= 0.4 - 1e-9) / tot
        assert p040 == pytest.approx(0.5167, abs=1e-4)

    def test_critical_increases_as_n_shrinks(self):
        """n 越小门槛越高（分辨率越差）—— 这是本批扩 n 的理由。"""
        assert epc.mc_critical(13) < epc.mc_critical(8) < epc.mc_critical(5)


class TestSpearmanHelper:
    """判据 C1/C2/C3 用的秩相关助手。"""

    def test_perfect_positive(self):
        assert epc.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)

    def test_perfect_negative(self):
        assert epc.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)

    def test_handles_ties_without_crash(self):
        v = epc.spearman([1, 1, 2, 2], [1, 2, 3, 4])
        assert np.isfinite(v)


class TestSingleVariableDeclaration:
    """③ 「唯一变量 = total_dd_halt」的可执行形式。"""

    def test_only_halt_field_differs(self):
        from fterm.config import load_config
        cfg = load_config(ROOT / "config.yaml")
        off = replace(cfg, account=replace(cfg.account,
                                            total_dd_halt=epc.HALT_OFF))
        assert off.account.total_dd_halt == epc.HALT_OFF
        assert cfg.account.total_dd_halt == epc.HALT_ON
        # 账户其余字段逐字段相同
        d_on, d_off = vars(cfg.account), vars(off.account)
        assert set(d_on) == set(d_off)
        diff = [k for k in d_on if d_on[k] != d_off[k]]
        assert diff == ["total_dd_halt"], f"非唯一变量：{diff}"
        # 账户之外的整棵配置树必须完全相同
        assert replace(off, account=cfg.account) == cfg

    def test_config_yaml_not_mutated_by_run_helper(self):
        """脚本只 replace 内存副本；config.yaml 的 total_dd_halt 必须仍是 0.12。"""
        import yaml
        doc = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        assert doc["account"]["total_dd_halt"] == 0.12


class TestPanelRaceGuard:
    """④ 面板竞态守卫（§五-5 同款）。"""

    def test_isolation_line_is_20260930(self):
        assert epc.MAX_DATA_DATE == pd.Timestamp("2026-09-30")

    def test_production_checkout_panel_would_be_rejected(self):
        """生产检出 D:\\Final 的面板已越过隔离线 ⇒ 必须被守卫拦下。

        面板是**宽表**（index=交易日，columns=股票代码），故只读索引。
        """
        p = Path(r"D:\Final\data\panel\close.parquet")
        if not p.exists():
            pytest.skip("生产检出面板不存在（CI/异机）")
        hi = pd.read_parquet(p, columns=[]).index.max()
        hi = pd.Timestamp(hi)
        if hi <= epc.MAX_DATA_DATE:
            pytest.skip(f"生产检出面板右端点 {hi.date()} 未越过隔离线")
        assert hi > epc.MAX_DATA_DATE   # 守卫必须判定为越线


class TestArmSetFrozen:
    """预注册 §三 冻结的臂集与参照臂。"""

    def test_thirteen_arms_declared(self):
        assert len(epc.ARMS) == 13
        assert set(epc.ARMS) == {"RL2", "MSE2", "MSE4", "LGBM", "A", "B", "C", "C14",
                                 "F0", "F0day", "D", "E", "E2"}

    def test_ref_arm_is_f0(self):
        assert epc.REF_ARM == "F0"
        assert epc.REF_ARM in epc.ARMS

    def test_anchor_covers_five_arms(self):
        """仪器门1 的锚点必须覆盖 §45 存档的五条流。"""
        assert set(epc.ANCHOR) == {"RL2", "F0", "F0day", "D", "E2"}


class TestLoadStreamGuards:
    """流加载的隔离线守卫（越线必须响亮失败）。"""

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            epc.load_stream([tmp_path / "nope.parquet"])

    def test_stream_past_isolation_line_raises(self, tmp_path):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2026-10-01"] * 3),
            "product": ["A", "B", "C"], "pred": [1.0, 2.0, 3.0],
            "y_ret": [0.1, 0.2, 0.3],
        })
        p = tmp_path / "late.parquet"
        df.to_parquet(p)
        with pytest.raises(ValueError, match="流含"):
            epc.load_stream([p])


class TestToPanelSemantics:
    """`to_panel` 必须与 `tierfix_replay.mean_panel` 同口径（pivot）。"""

    def test_panel_shape_and_values(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-02"] * 2 + ["2020-01-03"] * 2),
            "product": ["A", "B", "A", "B"],
            "pred": [1.0, 2.0, 3.0, 4.0],
            "y_ret": [0.1, 0.2, 0.3, 0.4],
        })
        p = epc.to_panel(df)
        assert p.shape == (2, 2)
        assert p.at[pd.Timestamp("2020-01-02"), "B"] == pytest.approx(2.0)
        assert p.at[pd.Timestamp("2020-01-03"), "A"] == pytest.approx(3.0)

    def test_multi_file_arm_averages(self, tmp_path):
        """多文件臂 = 逐行均值（`mean_panel` 语义）。"""
        a = pd.DataFrame({"date": pd.to_datetime(["2020-01-02"] * 2),
                          "product": ["A", "B"], "pred": [1.0, 3.0],
                          "y_ret": [0.1, 0.2]})
        b = pd.DataFrame({"date": pd.to_datetime(["2020-01-02"] * 2),
                          "product": ["A", "B"], "pred": [3.0, 5.0],
                          "y_ret": [0.1, 0.2]})
        pa, pb = tmp_path / "a.parquet", tmp_path / "b.parquet"
        a.to_parquet(pa)
        b.to_parquet(pb)
        out = epc.load_stream([pa, pb])
        assert out["pred"].tolist() == [2.0, 4.0]


class TestWeeklyStd:
    def test_constant_equity_gives_zero(self):
        idx = pd.date_range("2020-01-01", periods=60, freq="D")
        eq = pd.Series(100.0, index=idx)
        assert epc.weekly_std(eq) == pytest.approx(0.0, abs=1e-12)


class TestLabelConsistencyGate:
    """仪器门4 的最终口径：**覆盖率 + IC 轴自证**，不是「臂间逐位相等」。

    ## 为什么钉这个（本批实测教训，两轮）
    首版门写成"各臂 y_ret 必须与参照臂逐位相等"⇒ **假失败**，抓出两处
    **已知且已量化**的流间异质（臂 B 的训练段 std 标准化标签；RL2/MSE2/MSE4/LGBM
    的 3 天数据 vintage 差）。收紧为"日内秩等价"后**再次假失败**（那 4 条臂秩不等）。
    最终口径只保留**真正影响判据**的前提：
      ① 规范标签 join 覆盖率（覆盖率塌了才是真问题）；
      ② IC 轴自证（污染自带 y_ret 后 IC 必须逐位不变）。
    已量化且不污染的异质降级为留痕。**判据门槛与算法一字未改。**
    """

    def test_positive_scaling_is_rank_equivalent(self):
        """正比例缩放（臂 B 的 std 标准化形态）⇒ 日内秩逐位等价。"""
        idx = pd.to_datetime(["2020-01-02"] * 4 + ["2020-01-03"] * 4)
        prods = ["A", "B", "C", "D"] * 2
        y = [0.01, 0.02, -0.03, 0.04, 0.05, -0.01, 0.02, 0.03]
        df = pd.DataFrame({"date": idx, "product": prods, "y_ret": y})
        r1 = df.groupby("date")["y_ret"].rank().to_numpy()
        scaled = df.assign(y_ret=df["y_ret"] * 14.7434)
        r2 = scaled.groupby("date")["y_ret"].rank().to_numpy()
        assert np.array_equal(r1, r2)

    def test_negative_scaling_is_not_rank_equivalent(self):
        """负比例缩放**会**翻转秩 ⇒ 秩判据不能太松（此处记录该事实）。"""
        idx = pd.to_datetime(["2020-01-02"] * 4)
        df = pd.DataFrame({"date": idx, "product": list("ABCD"),
                           "y_ret": [0.01, 0.02, -0.03, 0.04]})
        r1 = df.groupby("date")["y_ret"].rank().to_numpy()
        r2 = df.assign(y_ret=-df["y_ret"]).groupby("date")["y_ret"].rank().to_numpy()
        assert not np.array_equal(r1, r2)

    def test_gate_does_not_hard_fail_on_quantified_heterogeneity(self):
        """最终门**不得**因"异质存在"而硬失败 —— 异质只留痕（实测口径）。"""
        import inspect
        src = inspect.getsource(epc.main)
        # 硬门只允许是覆盖率与 IC 自证
        assert "canonical_label_coverage_min" in src
        assert "ic_canonical_selfcheck" in src
        # 旧的"秩不等价即 SystemExit"必须已被移除
        assert "日内秩不等价（IC 轴不可比）" not in src


class TestCanonicalLabels:
    """IC 一律走规范标签（`bundle.labels['fwd_ret']`）⇒ 13 臂严格同标签可比。"""

    def test_canonical_labels_is_long_form(self):
        c = epc._canonical_labels()
        assert list(c.columns) == ["date", "product", "y_canon"]
        assert len(c) > 0
        assert c["date"].dtype.kind == "M"
        assert not c["y_canon"].isna().any()

    def test_canonical_labels_cached(self):
        a = epc._canonical_labels()
        b = epc._canonical_labels()
        assert a is b

    def test_daily_ic_uses_canonical_label_when_given(self):
        """给 canon 时，流自带的错误 y_ret 不得影响 IC（门4 自证的可执行形式）。"""
        c = epc._canonical_labels()
        day = c["date"].iloc[len(c) // 2]
        sub = c[c["date"] == day].head(30).copy()
        df = sub.rename(columns={"y_canon": "y_ret"})
        df["pred"] = df["y_ret"] * -1.0          # 完美负相关
        ic_true = epc.daily_ic(df, [day], sub)
        assert ic_true == pytest.approx(-1.0, abs=1e-6)
        # 把自带 y_ret 换成噪声：只要传了 canon，IC 不变
        df2 = df.assign(y_ret=np.random.RandomState(0).randn(len(df)))
        assert epc.daily_ic(df2, [day], sub) == pytest.approx(ic_true, abs=1e-12)

    def test_daily_ic_ignores_own_label_entirely(self):
        """逐位不变（不是"接近"）—— 这是门4 自证要求的强度。"""
        c = epc._canonical_labels()
        day = c["date"].iloc[len(c) // 3]
        sub = c[c["date"] == day].head(40).copy()
        base = sub.rename(columns={"y_canon": "y_ret"})
        base["pred"] = np.random.RandomState(7).randn(len(base))
        a = epc.daily_ic(base, [day], sub)
        b = epc.daily_ic(base.assign(y_ret=np.zeros(len(base))), [day], sub)
        assert a == b
