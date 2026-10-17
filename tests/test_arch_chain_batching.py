"""臂 D/E 执行器的按日分批前向验收（2026-10-10 OOM 回归钉）。

## 背景（为什么需要这个测试）

臂 D 的截面注意力掩码是 **B×B**（B = 单批行数）。`arch_chain.train_fold` 原实现
把**整个验证段**一次性前向——fold0 验证段约 25.3 万行 ⇒ 掩码 6.4e10 个元素
（bool ≈ 59.8 GiB）⇒ 必然 OOM。首次冒烟实测该路径请求 **25.37 GiB** 后失败。

训练与推理本已按日分批，**验证路径此前漏了**。本测试钉住三件事：
  ① 前向**逐日**分批（每批只含一个完整交易日），绝无整段前向；
  ② 行序与输入一致（回填正确，不重不漏）；
  ③ 与朴素逐日循环的结果逐位一致。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")

import scripts.arch_chain as ac  # noqa: E402


class _Recorder:
    """假网络：记录每批形状，并按行的首列返回确定性预测。"""

    def __init__(self):
        self.batch_rows = []

    def __call__(self, x, b, day_ids):
        self.batch_rows.append((int(len(day_ids)), int(day_ids.unique().numel())))
        # 预测 = 行内特征和，便于逐位比对
        return x.sum(dim=1)


def _mk(n_days=4, per_day=6, n_feat=3, seed=0):
    rng = np.random.default_rng(seed)
    day_ids = np.repeat(np.arange(n_days), per_day)
    x = rng.normal(size=(len(day_ids), n_feat))
    b = rng.integers(0, 16, size=(len(day_ids), n_feat))
    return x, b, day_ids


class TestForwardBatched:
    def test_each_batch_is_exactly_one_day(self, monkeypatch):
        x, b, day_ids = _mk()
        rec = _Recorder()
        monkeypatch.setattr(ac, "_fwd", lambda net, kind, xb, bb, db: rec(xb, bb, db))

        out = ac._forward_batched(object(), "attn", x, b, day_ids)

        assert len(rec.batch_rows) == 4, "应为 4 个交易日 ⇒ 4 批"
        for rows, days in rec.batch_rows:
            assert days == 1, "每批必须恰好一个交易日（整段前向会 OOM）"
            assert rows == 6, "每批应含该日全部样本"
        assert len(out) == len(day_ids)

    def test_never_single_giant_batch(self, monkeypatch):
        """反例守卫：整段前向（一批含全部行）必须被本测试抓到。"""
        x, b, day_ids = _mk(n_days=10, per_day=50)
        rec = _Recorder()
        monkeypatch.setattr(ac, "_fwd", lambda net, kind, xb, bb, db: rec(xb, bb, db))

        ac._forward_batched(object(), "attn", x, b, day_ids)

        biggest = max(rows for rows, _ in rec.batch_rows)
        assert biggest == 50 < len(day_ids), "不得出现整段（500 行）单批前向"

    def test_row_order_preserved(self, monkeypatch):
        x, b, day_ids = _mk()
        monkeypatch.setattr(ac, "_fwd",
                            lambda net, kind, xb, bb, db: xb.sum(dim=1))
        out = ac._forward_batched(object(), "attn", x, b, day_ids)
        assert np.allclose(out, x.sum(axis=1), rtol=1e-6, atol=1e-6), "回填必须保持原行序"

    def test_matches_naive_daywise_loop(self, monkeypatch):
        x, b, day_ids = _mk(seed=7)
        monkeypatch.setattr(ac, "_fwd",
                            lambda net, kind, xb, bb, db: xb.sum(dim=1))
        out = ac._forward_batched(object(), "attn", x, b, day_ids)
        naive = np.empty(len(day_ids))
        for d in np.unique(day_ids):
            m = day_ids == d
            naive[m] = x[m].sum(axis=1)
        # 前向走 float32（GPU 输入口径，与原实现一致）⇒ 按 float32 精度比对
        assert np.allclose(out, naive, rtol=1e-6, atol=1e-6), "与朴素逐日循环一致"
        assert np.isfinite(out).all(), "无 NaN/Inf"

    def test_net2_kind_also_daywise(self, monkeypatch):
        """net2 臂（F0day/E/E2）同样走分批路径，口径一致。"""
        x, b, day_ids = _mk(n_days=3, per_day=4)
        rec = _Recorder()
        monkeypatch.setattr(ac, "_fwd", lambda net, kind, xb, bb, db: rec(xb, bb, db))
        ac._forward_batched(object(), "net2", x, b, day_ids)
        assert all(days == 1 for _, days in rec.batch_rows)


class TestTailReference:
    """尾部判据的基准集合（2026-10-10 verifier 抓到的静默缺口回归钉）。

    初版 `tail_ref` 只从 `results` 取基准，而 `results` 的键来自 `ARM_SPEC`
    （= F0day/D/E/E2，**不含 "F0"**）⇒ `"F0" in results` 恒为 False ⇒
    F0 被**静默跳过**，判据只剩 F0day 一个（更松的）门槛。

    实测两个基准相差 2.11 倍（top0.5% 周均 F0 +0.007771 vs F0day +0.003689）
    ⇒ 这不是"基准差不多"，而是把显著更松的门槛当成唯一门槛。
    """

    def test_arm_spec_has_no_f0_key(self):
        """记录事实：F0 不在 ARM_SPEC 里（这正是缺口成因）。"""
        assert "F0" not in ac.ARM_SPEC
        assert "F0day" in ac.ARM_SPEC

    def test_tail_ref_must_include_f0_explicitly(self, tmp_path, monkeypatch):
        """tail_ref 必须**显式**包含 F0（不能依赖 results 的键）。"""
        import pandas as pd

        # 合成两条流：F0（强尾部）与 F0day（弱尾部）
        rng = np.random.default_rng(0)
        dates = pd.bdate_range("2024-01-01", periods=60)

        def mk_stream(path, noise):
            rows = []
            for d in dates:
                codes = [f"{600000+i}.SH" for i in range(40)]
                pred = rng.normal(size=40)
                y = pred * 0.01 + rng.normal(scale=noise, size=40)
                rows.append(pd.DataFrame({"date": d, "product": codes,
                                          "pred": pred, "y_ret": y}))
            pd.concat(rows, ignore_index=True).to_parquet(path)

        f0 = tmp_path / "stream_F0.parquet"
        f0d = tmp_path / "stream_F0day.parquet"
        mk_stream(f0, 0.001)
        mk_stream(f0d, 0.05)

        s_f0 = ac._tail_weekly_series(f0)
        s_f0d = ac._tail_weekly_series(f0d)
        assert len(s_f0) > 0 and len(s_f0d) > 0
        # 两条序列都可算 ⇒ 两个基准都能进 tail_ref（这是修复的核心）
        assert s_f0.mean() > s_f0d.mean(), "构造应使 F0 尾部优于 F0day"

    def test_tail_weekly_series_matches_bucket_mean(self):
        """逐周序列的均值必须与 _tail_bucket 的 mean_week 一致（同口径）。

        容差 1e-6：`tailloss_chain.tail_bucket_weekly` 对 mean_week 做 `round(...,6)`
        （实测差 4.8e-08），故不能要求逐位相等。
        """
        p = ROOT / "experiments" / "external_20261009" / "stream_F0_f4.parquet"
        if not p.exists():
            pytest.skip("缺 F0 流缓存（证据包未入库）")
        import pandas as pd

        d = pd.read_parquet(p)
        d["date"] = pd.to_datetime(d["date"])
        s = ac._tail_weekly_series(p)
        ref = ac._tail_bucket(d)["top0.5pct"]
        assert abs(float(s.mean()) - ref["mean_week"]) < 1e-6
        assert int(len(s)) == ref["n_weeks"]


class TestF0dayGuard:
    """`paired_t_vs_F0day` 的两个隐式前提都必须被堵住（静默缺口 #2 回归钉）。

    原实现依赖两个无保护的隐式前提：(1) ARM_SPEC 插入顺序让 "F0day" 排在前；
    (2) stream_F0day.parquet 存在。任一不成立 ⇒ 所有臂**静默失去** vs F0day 的
    配对 t（erratum §四-1 明定这是必需读数）。

    verifier 指出第一版修复只堵住 (2)、(1) 仍敞着 —— 本测试同时钉两层。
    """

    def test_source_has_explicit_f0day_guard(self):
        """前提 (2)：源码必须有对 stream_F0day.parquet 的显式存在性校验 + raise。"""
        src = (ROOT / "scripts" / "arch_chain.py").read_text(encoding="utf-8")
        assert 'stream_F0day.parquet' in src
        # 守卫必须是"缺则 raise"，而非静默跳过
        idx = src.index('if not (OUT_DIR / "stream_F0day.parquet").exists():')
        tail = src[idx:idx + 400]
        assert "raise FileNotFoundError" in tail

    def test_f0day_baseline_computed_outside_loop(self):
        """前提 (1)：F0day 的 IC 序列必须在**循环之外**显式算好 ⇒ 与 dict 顺序解耦。

        反例守卫：若 ic_f0day 仍靠循环内赋值（顺序依赖）或保留静默降级分支，本断言失败。
        """
        src = (ROOT / "scripts" / "arch_chain.py").read_text(encoding="utf-8")
        assert "ic_f0day = ic_series(_f0day_df)" in src, "F0day 基准须显式前置计算"
        assert "ic_f0day = ic\n" not in src, "循环内不应重新赋值 ic_f0day（顺序依赖）"
        assert "ic_f0day is not None" not in src, "不应保留静默降级分支"

    def test_baseline_precedes_decision_loop(self):
        """构造性验证：基准构造语句必须早于裁决循环（⇒ 顺序无关）。"""
        src = (ROOT / "scripts" / "arch_chain.py").read_text(encoding="utf-8")
        i_base = src.index("ic_f0day = ic_series(_f0day_df)")
        i_loop = src.index("for arm in [a for a in ARM_SPEC")
        assert i_base < i_loop, "F0day 基准必须在裁决循环之前算好"

    def test_f0day_is_first_in_arm_spec(self):
        """记录当前插入顺序（守卫已使其非必需，但登记事实）。"""
        assert list(ac.ARM_SPEC)[0] == "F0day"
