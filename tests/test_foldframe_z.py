"""非秩旁路通道（z_cols）验收 —— 直接钉住审计 P0-4 与臂 A 的修复。

核心命题：
  1. 横截面常数特征经**秩化通道**后，同日截面内**零方差**（无法区分个股 ⇒ 对排序无贡献）；
  2. 同一特征经 **z_cols 旁路通道**保留原始值；
  3. 不传 z_cols 时 fold_frame 行为与旧口径逐位一致（返回 5 元组）。

⚠️ 实测更正（2026-10-09）：审计初稿写「常数特征白化后恒为 0」，**不准确**。
   live 管线（foldframe/infer）用的是 `√2·erfinv(p)`（非文档所写 `√2·erfinv(2p−1)`），
   故常数特征映射到**常数非零值**（如 p=0.5833 → 0.8122），而非 0。
   但结论不变：它在同日截面内无变异 ⇒ 对截面排序无贡献，市场状态信息仍进不了模型。
   详见 docs/audit-model-20261009.md §2.1 P0-4 的更正与新增发现 P0-6。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fterm.features.build import DatasetBundle
from fterm.model.foldframe import fold_frame

N_DAYS = 30
N_CODES = 6


def _bundle() -> DatasetBundle:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-01", periods=N_DAYS)
    codes = [f"{600000 + i}.SH" for i in range(N_CODES)]
    feat_a = pd.DataFrame(rng.normal(size=(N_DAYS, N_CODES)), index=dates, columns=codes)
    feat_b = pd.DataFrame(rng.normal(size=(N_DAYS, N_CODES)), index=dates, columns=codes)
    # 横截面常数特征：逐日递增（同日全市场同值）
    mkt = pd.DataFrame(
        np.repeat(np.linspace(1.0, 2.0, N_DAYS)[:, None], N_CODES, axis=1),
        index=dates, columns=codes)
    fwd = pd.DataFrame(rng.normal(size=(N_DAYS, N_CODES)), index=dates, columns=codes)
    fwd.iloc[-3:] = np.nan                       # 尾部无标签
    universe = pd.DataFrame(True, index=dates, columns=codes)
    feats = {"feat_a": feat_a, "feat_b": feat_b, "mkt_like": mkt}
    labels = {"fwd_ret": fwd}
    return DatasetBundle(feats, labels, universe, ("feat_a", "feat_b", "mkt_like"), {})


def _run(z_cols=None):
    b = _bundle()
    dates = b.features["feat_a"].index
    return fold_frame(b, b.universe, dates[0], dates[-1] + pd.Timedelta(days=1), 0, z_cols=z_cols), b


def test_rank_channel_zero_variance_within_day():
    """P0-4（更正版）：常数特征经秩化通道后，同日截面内零方差。"""
    (Xg, miss, y, meta, cols), b = _run()
    j = list(cols).index("mkt_like")
    s = pd.Series(Xg[:, j], index=[m[0] for m in meta])
    # 每日截面标准差应为 0（无法区分个股）
    assert np.allclose(s.groupby(level=0).std().fillna(0.0).to_numpy(), 0.0, atol=1e-6)
    # 且它并非恒为 0（live 管线用 √2·erfinv(p)，非常数 0）
    assert abs(float(np.unique(Xg[:, j])[0])) > 1e-3


def test_z_channel_preserves_raw_values():
    """旁路通道保留原始值（不做秩化）。"""
    out, b = _run(z_cols=["mkt_like"])
    assert len(out) == 6
    Xg, miss, y, meta, cols, Z = out
    assert Z.shape == (len(y), 1)
    dates = b.features["feat_a"].index
    vals = np.linspace(1.0, 2.0, N_DAYS)
    date_to_val = {d: v for d, v in zip(dates, vals)}
    for (d, _c), z in zip(meta, Z[:, 0]):
        assert abs(z - date_to_val[d]) < 1e-6
    # 旁路通道保留了日间变异（截面内仍常数，但跨日有变化）
    zs = pd.Series(Z[:, 0], index=[m[0] for m in meta])
    assert zs.groupby(level=0).first().nunique() > 1


def test_backward_compatible_when_no_zcols():
    """不传 z_cols ⇒ 返回 5 元组，且与传空列表时逐位一致。"""
    a, _ = _run()
    bb, _ = _run(z_cols=[])
    assert len(a) == 5 and len(bb) == 5
    for x, yy in zip(a, bb):
        if isinstance(x, np.ndarray):
            assert np.array_equal(x, yy)
        else:
            assert x == yy


def test_missing_zcol_raises():
    try:
        _run(z_cols=["no_such_col"])
        raise AssertionError("应抛 KeyError")
    except KeyError:
        pass

