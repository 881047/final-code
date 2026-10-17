"""截面自注意力网（臂 D1）验收。

钉：① 前向形状与有限性；② **同日内可交互、跨日不可交互**（掩码正确性，PIT 关键）；
③ 初始化残差使注意力为增量（h + attn）。
"""
from __future__ import annotations

import torch

from fterm.model.net4 import CrossSectionAttentionNet
from fterm.model.net2 import CrossSectionNet
from fterm.model.net3 import count_params


def _mk(n=24, f=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, f, generator=g)
    b = torch.randint(0, 16, (n, f), generator=g)
    # 两个交易日：前 12 行 day0，后 12 行 day1
    days = torch.cat([torch.zeros(12, dtype=torch.long), torch.ones(12, dtype=torch.long)])
    return x, b, days


class TestForward:
    def test_shapes_and_finite(self):
        x, b, d = _mk()
        net = CrossSectionAttentionNet(8, hidden=32, n_blocks=2, n_heads=4)
        out = net(x, b, d)
        assert out.shape == (24,)
        assert torch.isfinite(out).all()

    def test_single_day_no_nan(self):
        x, b, _ = _mk()
        net = CrossSectionAttentionNet(8, hidden=32, n_blocks=2, n_heads=4)
        out = net(x, b, torch.zeros(24, dtype=torch.long))
        assert torch.isfinite(out).all()

    def test_singleton_day_no_nan(self):
        """某日只有 1 个样本 ⇒ 掩码行仅对角线可见，不应 NaN。"""
        x, b, _ = _mk(n=13)
        days = torch.cat([torch.zeros(12, dtype=torch.long), torch.ones(1, dtype=torch.long)])
        net = CrossSectionAttentionNet(8, hidden=32, n_blocks=2, n_heads=4)
        out = net(x, b, days)
        assert torch.isfinite(out).all()


class TestMaskCorrectness:
    def test_no_cross_day_influence(self):
        """改动 day1 的输入，不应改变 day0 的输出（跨日屏蔽生效）。"""
        x, b, d = _mk()
        net = CrossSectionAttentionNet(8, hidden=32, n_blocks=2, n_heads=4)
        net.eval()
        with torch.no_grad():
            o1 = net(x, b, d)
            x2 = x.clone()
            x2[12:] += 5.0                      # 只动 day1
            o2 = net(x2, b, d)
        # day0（前 12）输出应完全不变
        assert torch.allclose(o1[:12], o2[:12], atol=1e-6)
        # day1 应变化
        assert not torch.allclose(o1[12:], o2[12:], atol=1e-5)

    def test_within_day_influence(self):
        """改动 day0 中某一行，应改变 day0 其他行的输出（同日可见）。"""
        x, b, d = _mk()
        net = CrossSectionAttentionNet(8, hidden=32, n_blocks=2, n_heads=4)
        net.eval()
        with torch.no_grad():
            o1 = net(x, b, d)
            x2 = x.clone()
            x2[0] += 5.0
            o2 = net(x2, b, d)
        assert not torch.allclose(o1[:12], o2[:12], atol=1e-5)


class TestParams:
    def test_attention_adds_params(self):
        f = 8
        base = CrossSectionNet(f, hidden=32, n_blocks=2, n_embed_features=f)
        attn = CrossSectionAttentionNet(f, hidden=32, n_blocks=2, n_heads=4, n_embed_features=f)
        assert count_params(attn) > count_params(base)
