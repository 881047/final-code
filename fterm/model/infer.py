"""DL-v2 生产推理 —— 工件加载 + 单日截面预测（20 网集成均值）。

工件由 scripts/train_production.py 产出：state_dicts(20) + feature_cols +
bin_edges + 指纹。推理日 t 的口径与训练完全一致：当日 eligible 截面内秩 →
erf⁻¹ 白化 → 桶索引 → 20 网前向均值。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .net2 import CrossSectionNet, to_bin_idx
from .net2 import _erfinv as _erfinv_np
from .net3 import RegimeCrossSectionNet

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dlv2Artifact:
    EXPECTED_KIND = "dlv2_production"

    def __init__(self, path: Path, payload: dict | None = None):
        if payload is None:
            payload = torch.load(path, weights_only=False, map_location=_DEVICE)
        assert payload.get("kind") == self.EXPECTED_KIND, "工件类型不符"
        self.payload = payload
        self.cols = list(payload["feature_cols"])
        self.edges = np.asarray(payload["bin_edges"])
        self.n_embed = payload["n_embed_features"]
        # 工件输入维守卫（2026-10-06）：网络输入 = 基础特征 + 缺失指示（2×cols 拼接），
        # 与 train_production 的 concat 顺序隐式耦合；特征改动时在此响亮失败而非静默错位。
        sd0 = payload["state_dicts"][0]["state"]
        n_in = int(sd0["inp.weight"].shape[1])
        assert n_in == 2 * len(self.cols), (
            f"工件输入维 {n_in} ≠ 2×特征数 {len(self.cols)}（基础+缺失指示）——"
            "feature_cols 与工件不匹配，检查特征清单或重训")
        self.nets = []
        for sd in payload["state_dicts"]:
            net = CrossSectionNet(len(self.cols) + len(self.cols),  # + 缺失指示列
                                  n_embed_features=self.n_embed)
            net.load_state_dict(sd["state"])
            net.to(_DEVICE).eval()
            self.nets.append(net)

    def predict_day(self, feats: dict[str, pd.DataFrame], universe_row: pd.Series,
                    d: pd.Timestamp) -> pd.Series:
        """d 日截面预测（Series: code → 集成均值 pred）。eligible = universe 当日行。"""
        codes = [c for c in universe_row.index if bool(universe_row[c])]
        if not codes:
            return pd.Series(dtype="float64")
        X = pd.DataFrame({c: feats[c].loc[d, codes] for c in self.cols})
        miss = X.isna().to_numpy().astype("float32")
        # 逐日截面秩 → Rank-Gauss（PIT：只用当日截面）
        r = X.rank(pct=True, na_option="keep").fillna(0.5).to_numpy()
        Xg = np.sqrt(2.0) * _erfinv_np(np.clip(r, 1e-4, 1 - 1e-4))
        Xin = np.concatenate([Xg, miss], axis=1)
        Xb = to_bin_idx(Xg, self.edges)
        x = torch.from_numpy(Xin.astype("float32")).to(_DEVICE)
        b = torch.from_numpy(Xb.astype("int64")).to(_DEVICE)
        with torch.no_grad():
            preds = np.zeros(len(Xin))
            for net in self.nets:
                preds += net(x, b).cpu().numpy() / len(self.nets)
        return pd.Series(preds, index=codes)


def file_sha256(path: Path) -> str:
    """工件文件内容哈希（生产位换版守卫用；账本 `model_version` kv 同口径）。"""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Dlv3Artifact:
    """DL-v3 生产推理：市场状态条件化（模型升级臂 A，FiLM 旁路通道）。

    与 Dlv2Artifact 的唯一差异：额外读取 `z_cols` 与训练段 `z_mean/z_std`，
    预测时把旁路通道标准化后喂给 `RegimeCrossSectionNet` 的 FiLM。
    """

    def __init__(self, path: Path, payload: dict | None = None):
        if payload is None:
            payload = torch.load(path, weights_only=False, map_location=_DEVICE)
        assert payload.get("kind") == "dlv3_production", "工件类型不符（期望 dlv3_production）"
        self.payload = payload
        self.cols = list(payload["feature_cols"])
        self.z_cols = list(payload["z_cols"])
        self.edges = np.asarray(payload["bin_edges"])
        self.n_embed = payload["n_embed_features"]
        self.z_mean = np.asarray(payload["z_mean"], dtype="float64")
        self.z_std = np.asarray(payload["z_std"], dtype="float64")
        sd0 = payload["state_dicts"][0]["state"]
        n_in = int(sd0["inp.weight"].shape[1])
        assert n_in == 2 * len(self.cols), (
            f"工件输入维 {n_in} ≠ 2×特征数 {len(self.cols)}（基础+缺失指示）——特征清单与工件不匹配")
        self.nets = []
        for sd in payload["state_dicts"]:
            net = RegimeCrossSectionNet(len(self.cols) + len(self.cols), len(self.z_cols),
                                        n_embed_features=self.n_embed)
            net.load_state_dict(sd["state"])
            net.to(_DEVICE).eval()
            self.nets.append(net)

    def _z(self, feats: dict[str, pd.DataFrame], d: pd.Timestamp, codes: list[str]) -> np.ndarray:
        Z = np.column_stack([feats[c].loc[d, codes].to_numpy(dtype="float64") for c in self.z_cols])
        Z = (Z - self.z_mean) / self.z_std
        return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")

    def predict_day(self, feats: dict[str, pd.DataFrame], universe_row: pd.Series,
                    d: pd.Timestamp) -> pd.Series:
        codes = [c for c in universe_row.index if bool(universe_row[c])]
        if not codes:
            return pd.Series(dtype="float64")
        X = pd.DataFrame({c: feats[c].loc[d, codes] for c in self.cols})
        miss = X.isna().to_numpy().astype("float32")
        r = X.rank(pct=True, na_option="keep").fillna(0.5).to_numpy()
        Xg = np.sqrt(2.0) * _erfinv_np(np.clip(r, 1e-4, 1 - 1e-4))
        Xin = np.concatenate([Xg, miss], axis=1)
        Xb = to_bin_idx(Xg, self.edges)
        Z = self._z(feats, d, codes)
        x = torch.from_numpy(Xin.astype("float32")).to(_DEVICE)
        b = torch.from_numpy(Xb.astype("int64")).to(_DEVICE)
        z = torch.from_numpy(Z).to(_DEVICE)
        with torch.no_grad():
            preds = np.zeros(len(Xin))
            for net in self.nets:
                preds += net(x, b, z).cpu().numpy() / len(self.nets)
        return pd.Series(preds, index=codes)


class A1Artifact(Dlv2Artifact):
    """A1 生产工件（判据链 #124 采纳对象的部署件，ListMLE 6 网）。

    与 Dlv2Artifact 的差异仅在 kind 断言 + 逐种子预测：采纳对象
    D = 0.5·rank_pct(Ā) + 0.5·rank_pct(dlv2) 的 Ā 用**逐种子日内秩均值**
    （预注册 §0 冻结口径），不用 raw-pred 均值后再取秩 ⇒ 逐种子前向。
    """

    EXPECTED_KIND = "a1_production"

    def __init__(self, path: Path, payload: dict | None = None):
        if payload is None:
            payload = torch.load(path, weights_only=False, map_location=_DEVICE)
        assert payload.get("kind") == self.EXPECTED_KIND, "工件类型不符（期望 a1_production）"
        # 复用父类装配逻辑（同 net2 结构、同输入构造），仅 kind 不同
        Dlv2Artifact.__init__(self, path, payload=payload)

    def predict_day_perseed(self, feats: dict[str, pd.DataFrame], universe_row: pd.Series,
                            d: pd.Timestamp) -> pd.DataFrame:
        """d 日逐种子截面预测（DataFrame: index=code, columns=seed）。"""
        codes = [c for c in universe_row.index if bool(universe_row[c])]
        if not codes:
            return pd.DataFrame()
        X = pd.DataFrame({c: feats[c].loc[d, codes] for c in self.cols})
        miss = X.isna().to_numpy().astype("float32")
        r = X.rank(pct=True, na_option="keep").fillna(0.5).to_numpy()
        Xg = np.sqrt(2.0) * _erfinv_np(np.clip(r, 1e-4, 1 - 1e-4))
        Xin = np.concatenate([Xg, miss], axis=1)
        Xb = to_bin_idx(Xg, self.edges)
        x = torch.from_numpy(Xin.astype("float32")).to(_DEVICE)
        b = torch.from_numpy(Xb.astype("int64")).to(_DEVICE)
        cols = {}
        with torch.no_grad():
            for sd in self.payload["state_dicts"]:
                net = self.nets_perseed().get(sd["seed"])
                cols[sd["seed"]] = net(x, b).detach().cpu().numpy()
        return pd.DataFrame(cols, index=codes)

    def nets_perseed(self) -> dict:
        """seed → net 映射（与 state_dicts 同序装配，惰性缓存）。"""
        if not hasattr(self, "_seed_nets"):
            self._seed_nets = {sd["seed"]: net
                               for sd, net in zip(self.payload["state_dicts"], self.nets)}
        return self._seed_nets


class RLArtifact(A1Artifact):
    """RL2 生产工件（T3 工程单步骤 1/3，2026-10-15）。

    与 A1Artifact 仅 kind 不同：A1 = ListMLE 排名损失；RL = MSE + RankNet 相邻对
    （rankloss_chain #14 冻结配方）。装配/逐种子前向逐字复用父类。
    """

    EXPECTED_KIND = "rl_production"


def load_artifact(path: Path):
    """按工件 `kind` 分派加载（v2 无条件 / v3 市场状态条件化 / a1 秩目标集成 / rl 秩邻对集成）。"""
    assert path.exists(), f"生产工件不存在: {path}（先跑 scripts/train_production.py）"
    payload = torch.load(path, weights_only=False, map_location=_DEVICE)
    kind = payload.get("kind")
    if kind == "dlv3_production":
        return Dlv3Artifact(path, payload=payload)
    if kind == "dlv2_production":
        return Dlv2Artifact(path, payload=payload)
    if kind == "a1_production":
        return A1Artifact(path, payload=payload)
    if kind == "rl_production":
        return RLArtifact(path, payload=payload)
    raise AssertionError(f"未知工件类型: {kind}")
