"""walk-forward 验证与经济性指标。

验证轴（不只看 MSE —— 调研结论的纪律）：
    1. 方向准确率（vs 50% 基线，按品种与时间分段）
    2. Rank IC（p_up 与 fwd_ret 的 Spearman，逐日截面）
    3. 分位覆盖校准：真实收益落在 [q_lo,q_hi] 外的比例 ≈ 20%?
walk-forward：expanding train，每个 fold 内 scaler 只 fit 训练段。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..config import Config
from .dataset import FtermDataset, WalkForwardFolds, fit_scaler
from .net import TriHeadNet, trihead_loss
from torch.utils.data import DataLoader


@dataclass
class FoldMetric:
    fold: int
    n_test: int
    dir_acc: float
    rank_ic: float | None
    q_coverage: float | None
    val_loss: float


@dataclass
class ValidationReport:
    folds: list[FoldMetric] = field(default_factory=list)

    @property
    def dir_acc_mean(self) -> float:
        return float(np.mean([f.dir_acc for f in self.folds]))

    @property
    def n_total(self) -> int:
        return sum(f.n_test for f in self.folds)

    def summary(self) -> str:
        lines = [f"walk-forward folds={len(self.folds)} samples={self.n_total}"]
        for f in self.folds:
            ic = f"{f.rank_ic:+.3f}" if f.rank_ic is not None else "n/a"
            cov = f"{f.q_coverage:.2f}" if f.q_coverage is not None else "n/a"
            lines.append(
                f"  fold{f.fold}: n={f.n_test} dir_acc={f.dir_acc:.3f} ic={ic} q_cov={cov} loss={f.val_loss:.4f}")
        lines.append(f"  mean dir_acc = {self.dir_acc_mean:.3f}")
        return "\n".join(lines)


def _eval_fold(net: TriHeadNet, samples, mean, std, revin_mask=None,
               batch_size: int = 512) -> FoldMetric:
    from torch.utils.data import DataLoader

    net.eval()
    dl = DataLoader(FtermDataset(samples, mean, std, revin_mask), batch_size=batch_size)
    dirs, rets, covered = [], [], []
    with torch.no_grad():
        for xb, ydir, yret, yvol, sw in dl:
            out = net(xb)
            dirs.extend(torch.sigmoid(out["dir_logit"]).tolist())
            q = out["q"]
            lo, hi = q[:, 0].tolist(), q[:, 1].tolist()
            rets.extend(float(v) for v in yret)
            covered.extend(lo_i <= r <= hi_i for lo_i, r, hi_i in zip(lo, rets[-len(lo):], hi))
    dirs_a, rets_a = np.array(dirs), np.array(rets)
    pred_sign = dirs_a >= 0.5
    true_sign = rets_a >= 0
    acc = float((pred_sign == true_sign).mean()) if len(dirs_a) else float("nan")
    ic = _daily_rank_ic(dirs, rets, samples)
    cov = float(np.mean(covered)) if covered else None
    return FoldMetric(0, len(samples), acc, ic, cov, 0.0)


def _daily_rank_ic(dirs: list[float], rets: list[float], samples) -> float | None:
    """按日期分组的截面 Spearman IC：每日截面算一个 IC，再取均值。

    单品种数据下每日截面只有 1 个样本（IC 无定义）→ 返回 None（合法）。
    """
    from collections import defaultdict

    by_day: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for p_up, ret, s in zip(dirs, rets, samples):
        by_day[s.date.strftime("%Y-%m-%d")].append((p_up, ret))
    daily_ics = [spearman_rank_ic(v) for v in by_day.values()]
    daily_ics = [x for x in daily_ics if x is not None]
    return float(np.mean(daily_ics)) if daily_ics else None


def spearman_rank_ic(pairs: list[tuple[float, float]]) -> float | None:
    """无 scipy 依赖的 Spearman（rank 后 Pearson）。有效样本 <5 或**退化输入**返回 None。

    ⚠️ 2026-10-09 修复（臂 B 事故）：原实现直接用 `np.argsort` 排 NaN——NaN 会被排到
    末尾并被赋予**非恒定秩**，于是「全 NaN 输入」也会返回一个看似正常的伪 IC
    （实际是"行序 vs 收益序"的相关）。现在先剔除任一侧非有限的样本对。

    ⚠️ 2026-10-10 修复（verifier 复核抓到的**同类残留**）：原守卫 `ra.std() == 0`
    **不可达**——`_ranks` 赋的是 1..n 的整数秩（并列时也如此），故 std 恒 > 0。
    ⇒ **常数预测**仍会产出伪 IC（实测：常数 pred 对 20 个不同 y 返回 +1.000000；
    整条 F0 流换成常数 pred 得到 |mean IC| ≈ 0.0018 的地板）。
    这与臂 B 的 NaN 事故是**同一失效模式**（退化输入静默产出"看似正常"的 IC）。
    现在显式检测"任一侧无变异"并返回 None。

    注：本函数用**整数秩**（`argsort(stable)`）而非标准 Spearman 的**并列平均秩**，
    故 y 并列较重时读数与标准 Spearman 有微差（实测 IC 均值差 ~1e-05 量级，
    远小于判据阈值，不改判决）。该口径差**已登记为已知限制**（docs/02），
    **本批不修**——改并列口径 = 改基线，须另立预注册 + 全量重算。
    """
    a = np.asarray([p[0] for p in pairs], dtype="float64")
    b = np.asarray([p[1] for p in pairs], dtype="float64")
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 5:
        return None
    # 退化输入守卫：任一侧为常数（无变异）⇒ 秩相关无定义，返回 None 而非伪 IC。
    # 用 ptp（极差）判定，对"全等值"稳健且与秩的具体赋法无关。
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    ra = _ranks(a)
    rb = _ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _ranks(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="stable")
    ranks = np.empty(len(a))
    ranks[order] = np.arange(1, len(a) + 1)
    return ranks


def walk_forward_validate(cfg: Config, samples, n_folds: int = 4,
                          revin: bool | None = None,
                          feature_cols: tuple[str, ...] | None = None,
                          epochs: int = 15) -> ValidationReport:
    """expanding-window walk-forward（净化折叠）：每 fold 重训（小模型 CPU 可承受）。

    revin 掩码由 feature_cols 决定；未提供列名时退化为全连续（验证路径可接受，
    正式训练走 train_ensemble（始终有列名））。
    """
    from .dataset import revin_mask_for

    if revin is None:
        revin = cfg.model.revin
    rmask = None
    if revin:
        rmask = (revin_mask_for(feature_cols) if feature_cols is not None
                 else np.ones(samples[0].x.shape[-1], dtype=bool))
    folds = WalkForwardFolds(samples, n_folds=n_folds,
                             label_span=cfg.features.label_horizon + 1)
    report = ValidationReport()
    for k in range(len(folds)):
        tr_idx, te_idx = folds.fold(k)
        tr_samples = [samples[i] for i in tr_idx]
        te_samples = [samples[i] for i in te_idx]
        mean, std = fit_scaler(tr_samples)
        torch.manual_seed(cfg.model.seeds[0])
        net = TriHeadNet(len(samples[0].x[0]), samples[0].x.shape[0],
                         cfg.model.d_hidden, cfg.model.n_layers, cfg.model.dropout)
        _quick_fit(net, cfg, tr_samples, mean, std, rmask, epochs=epochs)
        m = _eval_fold(net, te_samples, mean, std, rmask)
        m.fold = k
        report.folds.append(m)
    return report


def walk_forward_predictions(cfg: Config, samples, n_folds: int = 4,
                             feature_cols: tuple[str, ...] | None = None,
                             epochs: int = 15) -> list[dict]:
    """净化 walk-forward 的逐测试样本预测流（元标签评估/判据链用，docs/03 §5/§15）。

    返回 [{fold, date, product, p_up, q_lo, q_hi, y_ret, y_dir}]。与 walk_forward_validate
    同一折切分/净化/训练口径，只是暴露样本级输出而非聚合指标。
    """
    from .dataset import revin_mask_for

    rmask = None
    if cfg.model.revin:
        rmask = (revin_mask_for(feature_cols) if feature_cols is not None
                 else np.ones(samples[0].x.shape[-1], dtype=bool))
    folds = WalkForwardFolds(samples, n_folds=n_folds,
                             label_span=cfg.features.label_horizon + 1)
    out: list[dict] = []
    for k in range(len(folds)):
        tr_idx, te_idx = folds.fold(k)
        tr_samples = [samples[i] for i in tr_idx]
        te_samples = [samples[i] for i in te_idx]
        mean, std = fit_scaler(tr_samples)
        torch.manual_seed(cfg.model.seeds[0])
        net = TriHeadNet(len(samples[0].x[0]), samples[0].x.shape[0],
                         cfg.model.d_hidden, cfg.model.n_layers, cfg.model.dropout)
        _quick_fit(net, cfg, tr_samples, mean, std, rmask, epochs=epochs)
        dl = DataLoader(FtermDataset(te_samples, mean, std, rmask), batch_size=512)
        net.eval()
        with torch.no_grad():
            i0 = 0
            for xb, _, yret, _, _ in dl:
                o = net(xb)
                ps = torch.sigmoid(o["dir_logit"]).tolist()
                q = o["q"]
                los, his = q[:, 0].tolist(), q[:, 1].tolist()
                for j, p in enumerate(ps):
                    s = te_samples[i0 + j]
                    out.append({"fold": k, "date": s.date, "product": s.product,
                                "p_up": float(p), "q_lo": float(los[j]),
                                "q_hi": float(his[j]), "y_ret": float(s.y_ret),
                                "y_dir": float(s.y_dir)})
                i0 += len(ps)
    return out


def _quick_fit(net: TriHeadNet, cfg: Config, tr_samples, mean, std,
               rmask=None, epochs: int = 15) -> None:
    """验证用的精简训练（正式训练走 train.train_ensemble）。"""
    from torch.utils.data import DataLoader

    dl = DataLoader(FtermDataset(tr_samples, mean, std, rmask),
                    batch_size=cfg.model.batch_size, shuffle=True)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.model.lr)
    net.train()
    for _ in range(epochs):
        for xb, ydir, yret, yvol, sw in dl:
            opt.zero_grad()
            loss, _ = trihead_loss(net(xb), ydir, yret, yvol, sample_w=sw)
            loss.backward()
            opt.step()
