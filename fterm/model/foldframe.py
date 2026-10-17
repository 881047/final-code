r"""walk-forward 单折样本长表构造（2026-10-08 性能改造 B2）。

原实现位于 `scripts/dlv2_chain.py`，其「逐日逐股 `.at` 标量取数」是全仓最热的瓶颈：
每个交易日 × 每只股票要两次 pandas `.at`（universe + fwd_ret），在
2367 日 × 5467 股规模下外推 **415–423 s（约 7 分钟）**；
改成布尔矩阵整数切片后同规模 **0.0014 s**（60 日实测），命中数完全一致（≈7000×）。

本模块保留 `fold_frame_reference`（旧实现）供 `tests/test_fold_frame_equivalence.py`
做**逐位对照**；生产路径走 `fold_frame`。

## 口径必须保持不变（改造前后逐位一致）

1. **逐日截面秩白化**（PIT）：每个交易日在当日 eligible 截面内排秩再过 erf⁻¹，
   **不是**全期秩（全期秩会前视）。
2. `len(elig) < 5` 的交易日**整天丢弃**。
3. `cap_per_day` 采样步进 `int(i * step)`，`step = len(elig) / cap_per_day`。
4. eligible 判定 = `universe[d, c]` 为真 **且** `fwd_ret[d, c]` 非 NaN。
5. eligible 顺序 = `universe.columns` 顺序。
6. `y` / `X` 一律 float64。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fterm.features.universe import UniverseIndex


def _align_panel_cols(panel: pd.DataFrame, codes: pd.Index, what: str) -> np.ndarray:
    """codes → panel 的列位置；有缺列则大声报错（原实现靠 .at/KeyError 触发）。"""
    pos = panel.columns.get_indexer(codes)
    if (pos < 0).any():
        missing = list(np.asarray(codes)[pos < 0][:5])
        raise KeyError(f"fold_frame: {what} 缺列 {missing}（面板未对齐）")
    return pos


def fold_frame(bundle, universe: pd.DataFrame, d0, d1, cap_per_day: int,
               z_cols: list[str] | None = None, y_key: str = "fwd_ret"):
    """[d0, d1) 的 (date, code) 样本长表：X 逐日截面秩白化、缺失指示、y、meta、cols。

    返回 (Xg, miss, y, meta, cols)，与旧实现逐位一致。

    z_cols（2026-10-09，模型升级臂 A）：**非秩旁路通道**的列名。这些列取自
    `bundle.features`，但**不做截面秩白化**（横截面常数特征若做秩会恒为 0，
    见 docs/audit-model-20261009.md P0-4）。给定 z_cols 时返回 6 元组，
    末尾追加 `Z`（(N, len(z_cols)) float64，原始量纲，标准化由训练端按训练段拟合）。

    y_key（2026-10-09，臂 B）：标签面板的键，默认 "fwd_ret"。多 horizon 训练时
    传 "fwd_ret_h{h}"（`features/multihorizon.py`），用于构造该 horizon 的 y。
    """
    feats, labels = bundle.features, bundle.labels
    cols = list(bundle.feature_cols)
    if y_key not in labels:
        raise KeyError(f"fold_frame: labels 中无 {y_key}（现有 {sorted(labels)}）")
    fwd = labels[y_key]
    master = feats[cols[0]].index

    # 一次性对齐校验：旧实现靠逐格 .at 触发 KeyError，这里显式检查，失败仍然大声
    for c in cols:
        if not feats[c].index.equals(master):
            raise ValueError(f"fold_frame: 特征 {c} 的 index 与 {cols[0]} 不一致")
    if not fwd.index.equals(master):
        raise ValueError("fold_frame: fwd_ret 的 index 与特征面板不一致")
    if not universe.index.equals(master):
        raise ValueError("fold_frame: universe 的 index 与特征面板不一致")

    uni = UniverseIndex(universe)
    codes = universe.columns
    # 各面板按 universe.columns 的列位置（一次算好，之后全走整数索引）
    feat_pos = [_align_panel_cols(feats[c], codes, f"特征 {c}") for c in cols]
    fwd_pos = _align_panel_cols(fwd, codes, "fwd_ret")
    feat_arr = [feats[c].to_numpy(dtype="float64") for c in cols]
    fwd_arr = fwd.to_numpy(dtype="float64")
    # 非秩旁路通道（可选）：单独取列，不做任何秩化
    z_pos = z_arr = None
    if z_cols:
        for c in z_cols:
            if c not in feats:
                raise KeyError(f"fold_frame: z_cols 中的 {c} 不在 bundle.features")
            if not feats[c].index.equals(master):
                raise ValueError(f"fold_frame: 旁路特征 {c} 的 index 与 {cols[0]} 不一致")
        z_pos = [_align_panel_cols(feats[c], codes, f"旁路 {c}") for c in z_cols]
        z_arr = [feats[c].to_numpy(dtype="float64") for c in z_cols]

    rows_X, rows_y = [], []
    rows_Z = []
    day_rows = []
    for ri, d in enumerate(master):
        if not (d0 <= d < d1):
            continue
        pos = uni.positions(d)                       # universe.columns 中的列位置（升序）
        if pos.size == 0:
            continue
        # eligible = universe 为真 且 fwd_ret 非 NaN（保持 universe.columns 顺序）
        keep = ~np.isnan(fwd_arr[ri, fwd_pos[pos]])
        pos = pos[keep]
        if pos.size < 5:
            continue
        if cap_per_day and pos.size > cap_per_day:
            step = pos.size / cap_per_day
            pos = pos[[int(i * step) for i in range(cap_per_day)]]
        elig = [codes[j] for j in pos]
        # X：(天数, F) —— 直接整数切片，不再逐日重建 DataFrame
        X = np.empty((pos.size, len(cols)), dtype="float64")
        for k, fp in enumerate(feat_pos):
            X[:, k] = feat_arr[k][ri, fp[pos]]
        y = fwd_arr[ri, fwd_pos[pos]]
        rows_X.append(X)
        rows_y.append(y)
        if z_arr is not None:
            Zb = np.empty((pos.size, len(z_arr)), dtype="float64")
            for k, zp in enumerate(z_pos):
                Zb[:, k] = z_arr[k][ri, zp[pos]]
            rows_Z.append(Zb)
        day_rows.append((d, elig, np.isnan(X)))

    X = np.concatenate(rows_X, axis=0)
    y = np.concatenate(rows_y)
    miss = np.concatenate([m for _, _, m in day_rows], axis=0)
    meta = [(d, c) for d, elig, _ in day_rows for c in elig]
    # 截面秩白化：**逐日**秩（PIT），不是全期秩
    Xg = np.empty_like(X)
    i0 = 0
    for (d, elig, _), k in zip(day_rows, [len(e) for _, e, _ in day_rows]):
        blk = X[i0:i0 + k]
        r = pd.DataFrame(blk).rank(pct=True, na_option="keep").fillna(0.5).to_numpy()
        Xg[i0:i0 + k] = np.sqrt(2.0) * _erfinv_vec(np.clip(r, 1e-4, 1 - 1e-4))
        i0 += k
    if z_arr is not None:
        Z = np.concatenate(rows_Z, axis=0)
        return Xg, miss.astype("float32"), y, meta, cols, Z
    return Xg, miss.astype("float32"), y, meta, cols


def _erfinv_vec(v: np.ndarray) -> np.ndarray:
    from fterm.model.net2 import _erfinv
    return _erfinv(v)


def fold_frame_reference(bundle, universe: pd.DataFrame, d0, d1, cap_per_day: int):
    """**旧实现**（逐日逐股 `.at` 标量取数）。

    仅供 `tests/test_fold_frame_equivalence.py` 做逐位对照；生产代码不得调用。
    """
    feats, labels = bundle.features, bundle.labels
    cols = list(bundle.feature_cols)
    fwd = labels["fwd_ret"]
    rows_X, rows_bin, rows_y, rows_meta = [], [], [], []
    day_rows = []
    for d in [d for d in feats[cols[0]].index if d0 <= d < d1]:
        elig = [c for c in universe.columns
                if bool(universe.at[d, c]) and pd.notna(fwd.at[d, c])]
        if len(elig) < 5:
            continue
        if cap_per_day and len(elig) > cap_per_day:
            step = len(elig) / cap_per_day
            elig = [elig[int(i * step)] for i in range(cap_per_day)]
        X = pd.DataFrame({c: feats[c].loc[d, elig] for c in cols})
        y = fwd.loc[d, elig].to_numpy(dtype="float64")
        rows_X.append(X.to_numpy(dtype="float64"))
        rows_y.append(y)
        day_rows.append((d, elig, X.isna().to_numpy()))
    X = np.concatenate(rows_X, axis=0)
    y = np.concatenate(rows_y)
    miss = np.concatenate([m for _, _, m in day_rows], axis=0)
    meta = [(d, c) for d, elig, _ in day_rows for c in elig]
    Xg = np.empty_like(X)
    i0 = 0
    for (d, elig, _), k in zip(day_rows, [len(e) for _, e, _ in day_rows]):
        blk = X[i0:i0 + k]
        r = pd.DataFrame(blk).rank(pct=True, na_option="keep").fillna(0.5).to_numpy()
        Xg[i0:i0 + k] = np.sqrt(2.0) * _erfinv_vec(np.clip(r, 1e-4, 1 - 1e-4))
        i0 += k
    return Xg, miss.astype("float32"), y, meta, cols
