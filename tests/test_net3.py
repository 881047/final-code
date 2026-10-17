"""DL-v3 FiLM 条件化网验收（模型升级臂 A）。

钉三件事：① 前向形状；② **初始化时条件化 = 无条件**（γ≡1,β≡0，保证不劣于 net2）；
③ 扰动 FiLM 权重后输出确实随 z 变化（证明旁路通道真的接进了计算图）。
"""
from __future__ import annotations

import numpy as np
import torch

from fterm.model.net2 import CrossSectionNet, HIDDEN, N_BLOCKS
from fterm.model.net3 import RegimeCrossSectionNet, ZScaler, count_params


def _mk(n=200, f=12, z=5):
    torch.manual_seed(0)
    x = torch.randn(n, f)
    b = torch.randint(0, 16, (n, f))
    zz = torch.randn(n, z)
    return x, b, zz


class TestForward:
    def test_shapes(self):
        x, b, z = _mk()
        net = RegimeCrossSectionNet(12, 5, hidden=32, n_blocks=2)
        out = net(x, b, z)
        assert out.shape == (200,)
        assert torch.isfinite(out).all()


class TestInitEqualsUnconditional:
    def test_output_invariant_to_z_at_init(self):
        x, b, z = _mk()
        net = RegimeCrossSectionNet(12, 5, hidden=32, n_blocks=2)
        net.eval()
        with torch.no_grad():
            o1 = net(x, b, z)
            o2 = net(x, b, torch.zeros_like(z))
            o3 = net(x, b, torch.randn_like(z) * 5)
        # γ≡1, β≡0 ⇒ 初始化时 z 不影响输出
        assert torch.allclose(o1, o2, atol=1e-6)
        assert torch.allclose(o1, o3, atol=1e-6)

    def test_perturbed_film_uses_z(self):
        x, b, z = _mk()
        net = RegimeCrossSectionNet(12, 5, hidden=32, n_blocks=2)
        with torch.no_grad():
            net.blocks[0].to_gamma.weight.normal_(0, 0.5)
            net.blocks[0].to_beta.weight.normal_(0, 0.5)
        net.eval()
        with torch.no_grad():
            o1 = net(x, b, z)
            o2 = net(x, b, z + 3.0)
        assert not torch.allclose(o1, o2, atol=1e-5)   # z 确实进入计算


class TestParams:
    def test_param_delta_is_film_only(self):
        f, zd = 12, 5
        n2 = CrossSectionNet(f, hidden=32, n_blocks=2, n_embed_features=f)
        n3 = RegimeCrossSectionNet(f, zd, hidden=32, n_blocks=2, n_embed_features=f)
        delta = count_params(n3) - count_params(n2)
        # 每块：γ、β 各 (zd×hidden 权重 + hidden 偏置)，共 n_blocks 个块
        assert delta == 2 * 2 * (zd * 32 + 32)

    def test_default_hidden_blocks(self):
        net = RegimeCrossSectionNet(12, 5)
        assert len(net.blocks) == N_BLOCKS
        # 主干隐藏宽度 = HIDDEN
        assert net.inp.out_features == HIDDEN


class TestZScaler:
    def test_fit_transform(self):
        z = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
        s = ZScaler.fit(z)
        out = s.transform(z)
        assert np.allclose(out.mean(axis=0), 0.0, atol=1e-6)
        assert np.allclose(out.std(axis=0), 1.0, atol=1e-6)

    def test_nan_handling(self):
        z = np.array([[1.0, np.nan], [3.0, 2.0]])
        s = ZScaler.fit(z)
        out = s.transform(np.array([[np.nan, np.nan]]))
        assert np.isfinite(out).all()          # NaN → 0

    def test_zero_std_guard(self):
        z = np.array([[5.0], [5.0], [5.0]])
        s = ZScaler.fit(z)
        assert s.std[0] == 1.0                 # 零方差守卫
        assert np.isfinite(s.transform(z)).all()
