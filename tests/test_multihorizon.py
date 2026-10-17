"""多 horizon（臂 B）验收：标签口径、融合分、多头网。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from fterm.features.multihorizon import build_multihorizon_labels, fused_score
from fterm.model.heads import MultiHorizonNet
from fterm.model.net2 import CrossSectionNet
from fterm.model.net3 import count_params


class TestLabels:
    def test_horizons_and_pit(self, env):
        cfg, panels, cal = env
        labs = build_multihorizon_labels(cfg, panels, horizons=(1, 5))
        assert set(labs) == {"fwd_ret_h1", "fwd_ret_h5"}
        # 与 features.build.add_labels 的 h=5 口径一致
        from fterm.features.build import add_labels
        ref = add_labels(cfg, panels, horizon=5)["fwd_ret"]
        got = labs["fwd_ret_h5"]
        a = ref.to_numpy(dtype="float64")
        b = got.to_numpy(dtype="float64")
        m = np.isfinite(a) & np.isfinite(b)
        assert np.allclose(a[m], b[m], atol=1e-6)

    def test_tail_nan_grows_with_horizon(self, env):
        cfg, panels, cal = env
        labs = build_multihorizon_labels(cfg, panels, horizons=(1, 20))
        n1 = labs["fwd_ret_h1"].notna().to_numpy().sum()
        n20 = labs["fwd_ret_h20"].notna().to_numpy().sum()
        assert n20 <= n1                      # 长 horizon 截尾更多

    def test_illegal_horizon_raises(self, env):
        cfg, panels, cal = env
        try:
            build_multihorizon_labels(cfg, panels, horizons=(0,))
            raise AssertionError("应抛 ValueError")
        except ValueError:
            pass


class TestFusedScore:
    def test_equal_weights(self):
        s1 = pd.Series([1.0, 2.0, 3.0])
        s2 = pd.Series([3.0, 2.0, 1.0])
        f = fused_score({"h1": s1, "h5": s2})       # 标准化后相加 ⇒ 常为 0
        assert np.allclose(f.to_numpy(), 0.0, atol=1e-9)

    def test_weights_matter(self):
        s1 = pd.Series([1.0, 2.0, 3.0])
        s2 = pd.Series([3.0, 2.0, 1.0])
        f = fused_score({"h1": s1, "h5": s2}, weights={"h1": 1.0, "h5": 0.0})
        assert f.iloc[2] > f.iloc[0]                # 只保留 h1（递增）

    def test_empty_raises(self):
        try:
            fused_score({})
            raise AssertionError("应抛 ValueError")
        except ValueError:
            pass


class TestNet:
    def test_forward_all_heads(self):
        torch.manual_seed(0)
        x = torch.randn(64, 10)
        b = torch.randint(0, 16, (64, 10))
        net = MultiHorizonNet(10, horizons=(1, 3, 5), hidden=32, n_blocks=2)
        out = net(x, b)
        assert set(out) == {"h1", "h3", "h5"}
        for v in out.values():
            assert v.shape == (64,)
            assert torch.isfinite(v).all()

    def test_param_delta_is_heads_only(self):
        f = 10
        base = CrossSectionNet(f, hidden=32, n_blocks=2, n_embed_features=f)
        multi = MultiHorizonNet(f, horizons=(1, 5, 10), hidden=32, n_blocks=2, n_embed_features=f)
        # 单头 Linear(hidden,1) 被换成 3 个 ⇒ 净增 2×(32+1)
        assert count_params(multi) - count_params(base) == 2 * (32 + 1)
