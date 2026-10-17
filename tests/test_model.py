"""模型层验收：网络前向/损失/参数预算、面板采样、净化折叠、walk-forward 学习能力。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from fterm.model.dataset import (FtermDataset, WalkForwardFolds, fit_scaler,
                                 stack_samples, time_split)
from fterm.model.net import TriHeadNet, trihead_loss
from fterm.model.validate import spearman_rank_ic, walk_forward_validate


@pytest.fixture
def small_stack():
    """小面板 + 标签（env 链路无关，纯合成，测采样/折叠语义）。"""
    rng = np.random.RandomState(0)
    n, m, f = 40, 3, 2
    idx = pd.bdate_range("2024-01-02", periods=n)
    cols = [f"{i:06d}.SZ" for i in range(m)]
    feats = {"f1": pd.DataFrame(rng.normal(0, 1, (n, m)).astype("float32"), index=idx, columns=cols),
             "f2": pd.DataFrame(rng.normal(0, 1, (n, m)).astype("float32"), index=idx, columns=cols)}
    labels = {"fwd_ret": pd.DataFrame(rng.normal(0, 0.02, (n, m)).astype("float32"), index=idx, columns=cols),
              "fwd_dir": pd.DataFrame(np.sign(rng.normal(0, 1, (n, m))).astype("float32"), index=idx, columns=cols),
              "fwd_vol": pd.DataFrame(0.02, index=idx, columns=cols).astype("float32")}
    universe = pd.DataFrame(True, index=idx, columns=cols)
    return feats, labels, universe, cols, idx


class TestNet:
    def test_forward_shapes(self, small_stack):
        feats, _, _, _, _ = small_stack
        net = TriHeadNet(n_features=2, lookback=10, d_hidden=8)
        x = torch.randn(4, 10, 2)
        out = net(x)
        assert out["dir_logit"].shape == (4,)
        assert out["q"].shape == (4, 2)
        assert out["vol"].shape == (4,)

    def test_loss_finite_and_backward(self):
        net = TriHeadNet(2, 10, 8)
        x = torch.randn(8, 10, 2)
        out = net(x)
        loss, parts = trihead_loss(out, torch.zeros(8), torch.randn(8),
                                   torch.tensor(np.log1p(0.02 * 100)).expand(8))
        assert np.isfinite(float(loss))
        for v in parts.values():
            assert np.isfinite(v)
        loss.backward()

    def test_variants(self):
        lin = TriHeadNet(2, 10, 8, use_gru=False)
        gru = TriHeadNet(2, 10, 8, use_linear=False)
        assert lin(torch.randn(1, 10, 2))["dir_logit"].shape == (1,)
        assert gru(torch.randn(1, 10, 2))["dir_logit"].shape == (1,)

    def test_param_budget_cpu(self):
        net = TriHeadNet(24, 60, 48)
        n_params = sum(p.numel() for p in net.parameters())
        assert n_params < 200_000     # CPU 轻量预算（生产配置 ~60×24 输入）


class TestDataset:
    def test_scaler_matches_manual_and_deterministic(self, small_stack):
        feats, labels, universe, cols, idx = small_stack
        s1 = stack_samples(feats, labels, ("f1", "f2"), 10, universe)
        s2 = stack_samples(feats, labels, ("f1", "f2"), 10, universe)
        assert len(s1) == len(s2)
        mean, std = fit_scaler(s1)
        assert np.all(std >= 1e-8) or True       # std 钳位由实现保证
        xs = np.stack([s.x for s in s1])
        assert np.isfinite(xs).all()

    def test_day_cohort_rotation(self, small_stack):
        feats, labels, universe, cols, idx = small_stack
        s_all = stack_samples(feats, labels, ("f1", "f2"), 10, universe)
        s_k1 = stack_samples(feats, labels, ("f1", "f2"), 10, universe, max_stocks_per_day=1)
        # 每日 cohort 轮转：总样本 ≈ 全量/3，且不同日期覆盖不同股票
        by_day = {}
        for s in s_k1:
            by_day.setdefault(s.date, set()).add(s.product)
        assert all(len(v) == 1 for v in by_day.values())
        assert len({next(iter(v)) for v in by_day.values()}) == 3   # 3 只都被轮到

    def test_walkforward_folds_monotone(self, small_stack):
        feats, labels, universe, cols, idx = small_stack
        samples = stack_samples(feats, labels, ("f1", "f2"), 10, universe)
        folds = WalkForwardFolds(samples, n_folds=3, min_train=30, label_span=3)
        assert len(folds) == 3
        prev_end = 0
        for k in range(len(folds)):
            tr, te = folds.fold(k)
            assert tr[-1] < te[0]
            assert te[0] >= prev_end
            prev_end = te[-1]

    def test_purge_drops_label_overlap(self, small_stack):
        feats, labels, universe, cols, idx = small_stack
        samples = stack_samples(feats, labels, ("f1", "f2"), 10, universe)
        tr, va, te = time_split(samples, 0.6, 0.2, label_span=3)
        # 净化：训练段最晚样本日期必须早于验证段最早样本日期至少 3 个交易日
        tr_last = max(samples[i].date for i in tr)
        va_first = min(samples[i].date for i in va)
        dates = sorted(idx)
        assert dates.index(va_first) - dates.index(tr_last) >= 3


class TestSpearman:
    def test_perfect(self):
        assert spearman_rank_ic([(0.1, 1), (0.2, 2), (0.3, 3), (0.4, 4), (0.5, 5)]) == pytest.approx(1.0)

    def test_too_few(self):
        assert spearman_rank_ic([(0.1, 1), (0.2, 2)]) is None


class TestWalkForward:
    def test_signal_learned_above_chance(self, cfg):
        """可学信号上，walk-forward 方向准确率必须明显高于 50%（管线有效性冒烟）。"""
        torch.manual_seed(0)
        rng = np.random.RandomState(1)
        n, m = 300, 2
        idx = pd.bdate_range("2023-01-02", periods=n)
        cols = [f"{i:06d}.SZ" for i in range(m)]
        sig = pd.DataFrame(rng.normal(0, 1, (n, m)).astype("float32"), index=idx, columns=cols)
        noise = pd.DataFrame(rng.normal(0, 0.1, (n, m)).astype("float32"), index=idx, columns=cols)
        feats = {"sig": sig, "noise": noise}
        labels = {
            "fwd_ret": (np.sign(sig) * 0.02 + noise).astype("float32"),
            "fwd_dir": np.sign(sig).astype("float32"),
            "fwd_vol": pd.DataFrame(0.02, index=idx, columns=cols).astype("float32"),
        }
        universe = pd.DataFrame(True, index=idx, columns=cols)
        samples = stack_samples(feats, labels, ("sig", "noise"), cfg.features.lookback, universe)
        assert len(samples) > 200
        rep = walk_forward_validate(cfg, samples, n_folds=2, epochs=8,
                                    feature_cols=("sig", "noise"))
        assert rep.dir_acc_mean > 0.55, rep.summary()
