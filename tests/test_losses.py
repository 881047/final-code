"""损失族验收（模型升级臂 E）。

钉：① listwise 对正确排序给低损失、对逆序给高损失；② 按日分组不跨日串扰；
③ 多任务不确定性加权有限且可反传；④ ranknet 与 train_production 口径一致。
"""
from __future__ import annotations

import torch

from fterm.model import losses as L


def _mk_day(n=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.randn(n, generator=g)
    out_perfect = y.clone()                 # 预测 = 标签 ⇒ 完美排序
    out_reversed = -y.clone()               # 完全逆序
    return y, out_perfect, out_reversed


class TestListNet:
    def test_perfect_lower_than_reversed(self):
        y, good, bad = _mk_day()
        days = torch.zeros(len(y), dtype=torch.long)
        assert L.listnet_loss(good, y, days) < L.listnet_loss(bad, y, days)

    def test_single_sample_day_skipped(self):
        y = torch.tensor([1.0])
        o = torch.tensor([0.5])
        days = torch.tensor([0])
        # 单样本组被跳过 ⇒ 返回 0
        assert float(L.listnet_loss(o, y, days)) == 0.0

    def test_groups_do_not_interact(self):
        y = torch.tensor([3.0, 1.0, 2.0, 0.0])
        days = torch.tensor([0, 0, 1, 1])
        o = y.clone()
        d0 = torch.tensor([0, 0])
        l_split = L.listnet_loss(o, y, days)
        l_g0 = L.listnet_loss(o[:2], y[:2], d0)
        l_g1 = L.listnet_loss(o[2:], y[2:], d0)
        assert torch.isclose(l_split, (l_g0 + l_g1) / 2, atol=1e-6)


class TestListMLE:
    def test_perfect_lower_than_reversed(self):
        y, good, bad = _mk_day()
        days = torch.zeros(len(y), dtype=torch.long)
        assert L.listmle_loss(good, y, days) < L.listmle_loss(bad, y, days)

    def test_equals_analytic_small(self):
        y = torch.tensor([2.0, 1.0, 0.0])
        o = torch.tensor([0.5, 0.2, 0.1])
        days = torch.zeros(3, dtype=torch.long)
        # 手工：-[(o0 - lse(o0,o1,o2)) + (o1 - lse(o1,o2)) + (o2 - o2)]
        lse = torch.logsumexp
        expect = -((o[0] - lse(o[0:3], 0)) + (o[1] - lse(o[1:3], 0)) + 0.0)
        assert torch.isclose(L.listmle_loss(o, y, days), expect, atol=1e-6)


class TestRankNet:
    def test_correct_order_zero_ish(self):
        y = torch.tensor([3.0, 2.0, 1.0])
        o = torch.tensor([3.0, 2.0, 1.0])
        days = torch.zeros(3, dtype=torch.long)
        # 相邻对间隔 1 ⇒ softplus(−1)=log(1+e⁻¹)=0.3133；正确排序应显著低于逆序
        v = float(L.ranknet_loss(o, y, days))
        assert abs(v - 0.3132617) < 1e-4
        assert v < float(L.ranknet_loss(-o, y, days))

    def test_reversed_positive(self):
        y = torch.tensor([3.0, 2.0, 1.0])
        o = torch.tensor([1.0, 2.0, 3.0])
        days = torch.zeros(3, dtype=torch.long)
        assert float(L.ranknet_loss(o, y, days)) > 1.0

    def test_no_valid_pairs_returns_zero(self):
        y = torch.tensor([1.0, 1.0])
        o = torch.tensor([0.1, 0.2])
        days = torch.zeros(2, dtype=torch.long)
        assert float(L.ranknet_loss(o, y, days)) == 0.0


class TestMultitask:
    def test_fixed_weights_sum(self):
        mt = L.MultitaskLoss(["a", "b"], base_loss="mse", fixed_weights={"a": 1.0, "b": 2.0})
        days = torch.zeros(4, dtype=torch.long)
        ya = torch.zeros(4); yb = torch.zeros(4)
        oa = torch.ones(4); ob = torch.ones(4)
        total, logs = mt({"a": (oa, ya), "b": (ob, yb)}, days)
        # mse(a)=1, mse(b)=1 ⇒ 1*1 + 2*1 = 3
        assert float(total) == 3.0
        assert set(logs) == {"a", "b"}

    def test_uncertainty_weighting_backward(self):
        mt = L.MultitaskLoss(["a", "b"], base_loss="mse")
        days = torch.zeros(4, dtype=torch.long)
        oa = torch.randn(4, requires_grad=True)
        ob = torch.randn(4, requires_grad=True)
        total, _ = mt({"a": (oa, torch.zeros(4)), "b": (ob, torch.ones(4))}, days)
        assert torch.isfinite(total)
        total.backward()
        assert oa.grad is not None and torch.isfinite(oa.grad).all()
        for t in mt.log_var:
            assert mt.log_var[t].grad is not None

    def test_listnet_base(self):
        mt = L.MultitaskLoss(["h5"], base_loss="listnet", fixed_weights={"h5": 1.0})
        y = torch.tensor([3.0, 1.0, 2.0])
        o = y.clone()
        days = torch.zeros(3, dtype=torch.long)
        # 预测=标签 ⇒ ListNet 损失 = 标签 softmax 分布的熵（非 0）
        import torch.nn.functional as F
        p = F.softmax(y, dim=0)
        expect = float(-(p * p.log()).sum())
        total, _ = mt({"h5": (o, y)}, days)
        assert abs(float(total) - expect) < 1e-5

    def test_unknown_base_raises(self):
        mt = L.MultitaskLoss(["x"], base_loss="nope")
        try:
            mt({"x": (torch.zeros(2), torch.zeros(2))}, torch.zeros(2, dtype=torch.long))
            raise AssertionError("应抛 ValueError")
        except ValueError:
            pass
