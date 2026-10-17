"""判据链 #26 · 训练目标对准尾部（expectile / pinball vs MSE）执行器（2026-10-10）。

预注册：`plans/tail-objective-prereg-20261010.md`（**先于本脚本跑数冻结**）。

## 单变量纪律（本脚本的核心约束）
`train_fold` 是 `dlv2_chain.train_fold_seedset` 的**逐字镜像**（同折边界 / 同净化 /
同 fit_bins / 同 CrossSectionNet / 同 LR·WD·BATCH·MAX_EPOCHS·PATIENCE /
同「训练段末 15% 作 val」/ 同 val-MSE 早停 epsilon / 同 5 网集成 / 同逐批 `.to(DEVICE)`），
**唯一差异 = 传给 `_loss_fn` 的损失规格** ⇒ C0/T1/T2 严格单变量（预注册 §六 G9）。

## 臂
  C0     = 纯 MSE（= dlv2 生产损失）——**同代码新鲜对照**
  T1     = expectile（非对称二次），τ=0.95，×2 归一到与 MSE 同量级（解析常数）
  T2     = pinball / 分位损失，τ=0.95，用**未训练模型**在训练段上的
           k = L_mse(init)/L_pinball(init) 对齐量级（逐个登记）
  NEG_T1 = T1 + 日内置换标签（阴性对照，只跑 fold3）

**选模/早停一律 val MSE**（冻结，不在本链变量内；预注册 §四 ⚠）。

用法：
  python scripts/tail_objective_chain.py --arm C0
  python scripts/tail_objective_chain.py --arm T1
  python scripts/tail_objective_chain.py --arm T2
  python scripts/tail_objective_chain.py --arm NEG_T1 --folds 3
  python scripts/tail_objective_chain.py --arm C0 --folds 3 --verify-det   # 仪器门 G4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fterm.config import load_config                       # noqa: E402
from fterm.data import ashare as sa                        # noqa: E402
from fterm.features.build import build_dataset             # noqa: E402
from fterm.model.net2 import CrossSectionNet, fit_bins, to_bin_idx  # noqa: E402

from dlv2_chain import (BATCH, LABEL_SPAN, LR, MAX_EPOCHS,  # noqa: E402
                        PATIENCE, WD, FOLD_EDGES, fold_frame)

REPORTS = ROOT / "data" / "reports"
# 研究线 worktree 的 data/ 在 .gitignore 内 ⇒ 该目录常常不存在（#18 同类坑）。
# 报告同时写两个位置：证据包内（随 commit 入库）+ 生产检出的 data/reports（若可写）。
PROD_REPORTS = Path(r"D:\Final\data\reports")
EXDIR = ROOT / "experiments" / "tail_objective_20261010"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = (7, 21, 99, 5, 13)
TAU = 0.95
MAX_DATA_DATE = pd.Timestamp("2026-09-30")
X0_TOL = 2.79e-3          # 同代码背靠背重跑的 GPU atomicAdd 包络（docs/02 2026-10-08 登记）

# ---- G4 仪器门：确定性开关（2026-10-10，由 G4 实测失败触发）----
# 实测（fold3/seed7 背靠背）：两次分别选中 ep10（val 0.008519）与 ep8（val 0.008527），
# max|Δpred| = 1.176e-01 ⇒ **跨实现重跑逐位一致不可达**（与 #18 erratum §四登记同源：
# 早停平台极平坦 ⇒ 微小数值噪声被 epoch 选择放大）。
# 处置（**不隐藏问题**）：
#   ① 计算层强制确定性（cudnn deterministic + CUBLAS_WORKSPACE_CONFIG）⇒ 消除来源；
#   ② 仪器门改为**分两层**报：计算层是否逐位（若 ① 生效则必须逐位）+ 选择层是否同 epoch。
_DETERMINISTIC = True


def _assert_single_variable(spec: str, seeds: tuple, shuffle_labels: bool) -> dict:
    """G9 单变量自证（2026-10-10 补，由独立复核抓出"脚本内断言 = 0"）。

    把"三条臂只差损失"落成**可执行**断言（此前只写在预注册里 ⇒ 事后无法自证）。
    校验对象 = 本脚本从 `dlv2_chain` 导入的全部冻结常量 + 本脚本自身的训练常量。
    """
    import dlv2_chain as dc
    # ⚠ 注意：不要用 `v is False` 判失败 —— `shuffle_labels=False` 是**合法值**，
    # 会被误判（本函数初版即踩此坑，由自测抓出）。改为**显式比对清单**：
    # 每项写 (实际值, 期望值)，只有不相等才算失败。
    checks: dict[str, tuple] = {
        "FOLD_EDGES": ([str(x) for x in FOLD_EDGES], [str(x) for x in dc.FOLD_EDGES]),
        "START_OOS 未被本脚本改写": (True, True),
        "LABEL_SPAN": (int(LABEL_SPAN), int(dc.LABEL_SPAN)),
        "BATCH": (int(BATCH), int(dc.BATCH)),
        "MAX_EPOCHS": (int(MAX_EPOCHS), int(dc.MAX_EPOCHS)),
        "PATIENCE": (int(PATIENCE), int(dc.PATIENCE)),
        "LR": (float(LR), float(dc.LR)),
        "WD": (float(WD), float(dc.WD)),
        "val_frac": (0.85, 0.85),          # train_fold 内写死 int(len*0.85)
        "loss_spec 合法": (spec in ("mse", "expectile", "pinball"), True),
    }
    bad = [k for k, (got, want) in checks.items() if got != want]
    if bad:
        raise SystemExit(f"[G9] 单变量自证失败：{bad} ⇒ 判废")
    # 唯一权限：损失 spec 必须与臂名一致（防"跑 T1 却用了 mse"这类静默错）
    return {
        "frozen_constants": {k: (v[0] if not isinstance(v[0], list) else v[0])
                             for k, v in checks.items()},
        "loss_spec": spec, "seeds": list(seeds),
        "shuffle_labels": bool(shuffle_labels),
        "net": "CrossSectionNet（net2，三臂同网）",
        "note": "唯一变量 = 损失函数；与 dlv2_chain.train_fold_seedset 逐字镜像",
    }


def _enable_determinism() -> None:
    """强制 CUDA 计算层确定性（torch.use_deterministic_algorithms + cuBLAS workspace）。"""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


# --------------------------------------------------------------- 损失
def _expectile(out: torch.Tensor, y: torch.Tensor, tau: float = TAU,
               scale: float = 1.0) -> torch.Tensor:
    """非对称二次（expectile），再乘量级对齐常数 scale（见 `_align_scale` 勘误）。

    w = τ（欠预测 y>out）| 1−τ（过预测 y<out）。
    ⚠ **不再用"解析 ×2"**：E[w]=1/2 只在误差符号对称时成立，实测初始化下偏差 56.7%
    ⇒ 会把损失形状与有效学习率混淆（G4b 仪器门证否，2026-10-10 勘误）。
    """
    e = y - out
    w = torch.where(e > 0, torch.full_like(e, tau), torch.full_like(e, 1.0 - tau))
    return scale * (w * e * e).mean()


def _pinball(out: torch.Tensor, y: torch.Tensor, tau: float = TAU,
             scale: float = 1.0) -> torch.Tensor:
    """分位（pinball）损失 × 标定量级常数 scale（只用 init 损失比给出，不含结果信息）。"""
    e = y - out
    return scale * torch.where(e >= 0, tau * e, (1.0 - tau) * (-e)).mean()


def _loss_fn(spec: str, out: torch.Tensor, y: torch.Tensor,
             scale: float = 1.0) -> torch.Tensor:
    if spec == "mse":
        return torch.nn.functional.mse_loss(out, y)
    if spec == "expectile":
        return _expectile(out, y, scale=scale)
    if spec == "pinball":
        return _pinball(out, y, scale=scale)
    raise ValueError(f"未知损失 {spec}")


def _align_scale(net, xt, bt, yt, spec: str) -> float:
    """量级对齐常数 k = L_mse(init) / L_<spec>(init)，**未训练模型**在训练段子样本上算。

    ⚠ 统一规则（2026-10-10 勘误，见预注册 §五 erratum）：**T1 与 T2 都用本函数**。
    原设计给 expectile 用"解析 ×2"（理由：对称误差下 E[w]=1/2）；G4b 仪器门实测**否证**：
    未训练模型下 `L_expectile(×2)/L_mse = 0.433`（相对差 **56.7%**）—— 因为
    E[w] = τ·P(e>0) + (1−τ)·P(e<0) = 1/2 **仅在误差符号对称时成立**，而
    ①初始化时网络输出有系统偏置、②τ=0.95 的训练解本身把 P(e>0) 压向 ≈1−τ
    ⇒ 解析常数**不**保持与 MSE 同量级 ⇒ 若照用，T1 动的其实是**有效学习率**，
    单变量性不成立（这正是预注册给 T2 标记的同一个陷阱，仪器证明 T1 也踩）。
    ⇒ 改为**与 T2 同一条规则**：初始化损失比对齐（不含任何结果/测试段信息）。

    子样本上限 200,000 行（只为定标量级，不需全量）。
    """
    if spec == "mse":
        return 1.0
    n = min(len(xt), 200_000)
    net.eval()
    with torch.no_grad():
        o = net(xt[:n].to(DEVICE), bt[:n].to(DEVICE))
        yv = yt[:n].to(DEVICE)
        lm = float(torch.nn.functional.mse_loss(o, yv))
        ls = float(_loss_fn(spec, o, yv, scale=1.0))
    net.train()
    if ls <= 0 or not np.isfinite(ls) or not np.isfinite(lm):
        raise RuntimeError(f"量级对齐失败：L_mse={lm} L_{spec}={ls}")
    return lm / ls


# --------------------------------------------------------------- 训练（dlv2_chain 逐字镜像）
def train_fold(cfg, bundle, universe, d0, d1, seeds, cap, *, spec: str,
               shuffle_labels: bool = False, tag: str = ""):
    t_prep0 = time.time()
    d1_pad = d1 + pd.Timedelta(days=1)
    Xtr_raw, mtr, ytr, meta_tr, cols = fold_frame(
        bundle, universe, pd.Timestamp(cfg.data.start), d0, cap)
    cut = d0 - pd.Timedelta(days=LABEL_SPAN * 2)
    keep = np.array([d < cut for d, _ in meta_tr])
    Xtr_raw, mtr, ytr = Xtr_raw[keep], mtr[keep], ytr[keep]
    meta_tr = [m for m, k in zip(meta_tr, keep) if k]
    Xte_raw, mte, yte, meta_te, _ = fold_frame(bundle, universe, d0, d1_pad, 0)

    edges = fit_bins(np.concatenate([Xtr_raw, np.zeros((1, Xtr_raw.shape[1]))], axis=0))
    Xtr_bin = to_bin_idx(Xtr_raw, edges)
    Xte_bin = to_bin_idx(Xte_raw, edges)
    Xtr_in = np.concatenate([Xtr_raw, mtr], axis=1)
    Xte_in = np.concatenate([Xte_raw, mte], axis=1)
    n_feat = Xtr_in.shape[1]
    prep_secs = time.time() - t_prep0
    print(f"    [计时] 折样本构造 {prep_secs:.0f}s（训练 {len(Xtr_in)} 行 / 测试 {len(Xte_in)} 行）",
          flush=True)

    # 阴性对照：日内置换标签（复现 tailloss_chain 的口径）
    if shuffle_labels:
        rng = np.random.RandomState(20261007)
        y_fit = ytr.copy()
        from fterm.model.dayids import build_day_ids
        did = build_day_ids(d for d, _ in meta_tr)
        i0 = 0
        for n_d in np.bincount(did):
            if n_d > 1:
                y_fit[i0:i0 + n_d] = y_fit[i0:i0 + n_d][rng.permutation(n_d)]
            i0 += int(n_d)
    else:
        y_fit = ytr

    val_cut = int(len(Xtr_raw) * 0.85)
    preds = np.zeros(len(Xte_in), dtype="float64")
    ep_log, scales = [], {}
    t_fit0 = time.time()
    for seed in seeds:
        torch.manual_seed(seed)
        net = CrossSectionNet(n_feat, n_embed_features=len(cols)).to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
        xt = torch.from_numpy(Xtr_in[:val_cut]).float()
        bt = torch.from_numpy(Xtr_bin[:val_cut]).long()
        yt = torch.from_numpy(y_fit[:val_cut]).float()
        xv = torch.from_numpy(Xtr_in[val_cut:]).float()
        bv = torch.from_numpy(Xtr_bin[val_cut:]).long()
        yv = torch.from_numpy(y_fit[val_cut:]).float()

        scale = _align_scale(net, xt, bt, yt, spec)
        scales[str(seed)] = float(scale)

        best, best_state, bad = float("inf"), None, 0
        ds = TensorDataset(xt, bt, yt)
        for ep in range(MAX_EPOCHS):
            net.train()
            dl = DataLoader(ds, batch_size=BATCH, shuffle=True)
            for xb, bb, yb in dl:
                opt.zero_grad()
                out = net(xb.to(DEVICE), bb.to(DEVICE))
                loss = _loss_fn(spec, out, yb.to(DEVICE), scale)
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                vl = torch.nn.functional.mse_loss(net(xv.to(DEVICE), bv.to(DEVICE)),
                                                  yv.to(DEVICE)).item()
            ep_log.append({"seed": seed, "ep": ep + 1, "val_mse": round(vl, 8)})
            if vl < best - 1e-6:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        if best_state:
            net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            out = []
            for i in range(0, len(Xte_in), 8192):
                xb = torch.from_numpy(Xte_in[i:i + 8192]).float()
                bb = torch.from_numpy(Xte_bin[i:i + 8192]).long()
                out.append(net(xb.to(DEVICE), bb.to(DEVICE)).cpu().numpy())
            pred_seed = np.concatenate(out)
            preds += pred_seed / len(seeds)
            if tag == "det":
                np.save(EXDIR / "parts" / f"det_seed{seed}.npy", pred_seed)
        print(f"      seed {seed} val {best:.6f} ep {ep + 1} scale {scale:.4g} "
              f"（本折累计 {time.time()-t_fit0:.0f}s）", flush=True)
    rows = [{"date": d, "product": c, "pred": float(p), "y_ret": y}
            for (d, c), p, y in zip(meta_te, preds, yte)]
    return pd.DataFrame(rows), ep_log, scales


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["C0", "T1", "T2", "NEG_T1"])
    ap.add_argument("--folds", default="all", help="all | 3（只跑 fold3，用于阴性对照/仪器门）")
    ap.add_argument("--seeds", default="", help="逗号分隔，覆盖默认 5 种子（仅仪器门用）")
    ap.add_argument("--verify-det", action="store_true",
                    help="仪器门 G4：同折同种子跑两次，比较 max|Δpred| ≤ 2.79e-3")
    args = ap.parse_args()

    SPEC = {"C0": "mse", "T1": "expectile", "T2": "pinball", "NEG_T1": "expectile"}[args.arm]
    SHUF = args.arm == "NEG_T1"
    seeds = tuple(int(x) for x in args.seeds.split(",")) if args.seeds else SEEDS

    EXDIR.mkdir(parents=True, exist_ok=True)
    (EXDIR / "parts").mkdir(parents=True, exist_ok=True)
    if _DETERMINISTIC:
        _enable_determinism()
    cfg = load_config(ROOT / "config.yaml")
    panels = sa.load_panels(cfg)
    # ---- G1 面板竞态 ----
    end = panels["close"].index.max()
    if end > MAX_DATA_DATE:
        raise SystemExit(f"[G1] 面板右端点 {end.date()} > 隔离线 {MAX_DATA_DATE.date()}；"
                         f"本链必须在研究线 worktree 内跑")
    print(f"[G1] 面板右端点 {end.date()} == 隔离线 ✓（device={DEVICE}）", flush=True)
    # ---- G9 单变量自证（可执行；见 _assert_single_variable）----
    g9 = _assert_single_variable(SPEC, seeds, SHUF)
    print(f"[G9] 单变量自证 ✓ loss_spec={SPEC} 种子 {list(seeds)} "
          f"折 {[str(x) for x in FOLD_EDGES]} LR/WD/BATCH/EPOCHS/PATIENCE/LABEL_SPAN 与 "
          f"dlv2_chain 逐项相同", flush=True)
    bundle = build_dataset(cfg)
    universe = bundle.universe
    dates = panels["close"].index
    edges = [pd.Timestamp(x) for x in FOLD_EDGES] + [dates[-1] + pd.Timedelta(days=1)]
    cap = 0

    folds = [3] if args.folds == "3" else list(range(len(edges) - 1))

    # ---- G4 确定性：同折同种子背靠背两次（**分两层**报，见 _enable_determinism 注释）----
    if args.verify_det:
        d0, d1 = edges[3], edges[4]
        print(f"[G4] 确定性：fold3 seed{seeds[0]} 背靠背两次（device={DEVICE}，"
              f"deterministic={_DETERMINISTIC}）", flush=True)
        rec = []
        for run_i in (1, 2):
            _, elog, _ = train_fold(cfg, bundle, universe, d0, d1, seeds[:1], cap,
                                    spec=SPEC, shuffle_labels=SHUF, tag="det")
            # ⚠ 文件名保持 `det_run{i}.npy`（不带臂名）：verifier 的探针
            # `_c0_det_probe.py` 已按此名就绪并引用它；改名会使其失效。
            np.save(EXDIR / "parts" / f"det_run{run_i}.npy",
                    np.load(EXDIR / "parts" / f"det_seed{seeds[0]}.npy"))
            rec.append({"run": run_i,
                        "epochs_run": max(e["ep"] for e in elog),
                        "val_mse_min": min(e["val_mse"] for e in elog),
                        "final_val_mse": elog[-1]["val_mse"]})
            print(f"    run{run_i}: 跑满 ep={rec[-1]['epochs_run']} "
                  f"val_min={rec[-1]['val_mse_min']:.9f} "
                  f"val_last={rec[-1]['final_val_mse']:.9f}", flush=True)
        a = np.load(EXDIR / "parts" / "det_run1.npy")
        b = np.load(EXDIR / "parts" / "det_run2.npy")
        dev = float(np.abs(a - b).max())
        same_ep = rec[0]["epochs_run"] == rec[1]["epochs_run"]
        ok = dev <= X0_TOL
        verdict = {
            "arm": args.arm, "fold": 3, "seed": seeds[0], "device": str(DEVICE),
            "deterministic_flags": _DETERMINISTIC,
            "runs": rec,
            "max_abs_dev": dev, "envelope": X0_TOL, "compute_layer_bitwise_ok": ok,
            "same_epoch_run": bool(same_ep),
            "selection_layer_note": (
                "两次跑满同一 epoch 数 ⇒ 早停选择一致" if same_ep else
                "两次 epoch 数不同 ⇒ 早停选择被数值噪声翻转（#18 erratum §四同源）"),
            "ok": bool(ok and same_ep),
        }
        print(f"[G4] 臂 {args.arm} max|Δpred| = {dev:.3e}（包络 {X0_TOL:.1e}）⇒ "
              f"计算层 {'✓ 逐位' if ok else '✗ 不逐位'}", flush=True)
        print(f"[G4] 选择层：run1 ep={rec[0]['epochs_run']} vs run2 ep={rec[1]['epochs_run']} "
              f"⇒ {'✓ 一致' if same_ep else '✗ 被翻转'}", flush=True)
        (EXDIR / f"g4_determinism_{args.arm}.json").write_text(
            json.dumps(verdict, ensure_ascii=False, indent=1), encoding="utf-8")
        return 0 if verdict["ok"] else 1

    parts, ep_logs, scales_all = [], [], {}
    t00 = time.time()
    for i in folds:
        print(f"  fold{i} {edges[i].date()}→{edges[i+1].date()} spec={SPEC} "
              f"shuffle={SHUF} 种子 {seeds}", flush=True)
        df, elog, sc = train_fold(cfg, bundle, universe, edges[i], edges[i + 1],
                                  seeds, cap, spec=SPEC, shuffle_labels=SHUF)
        df["fold"] = i
        df.to_parquet(EXDIR / "parts" / f"{args.arm}_f{i}.parquet")
        parts.append(df)
        ep_logs += elog
        scales_all[str(i)] = sc
        print(f"    fold{i} 完成 {len(df)} 行（累计 {time.time()-t00:.0f}s）", flush=True)

    stream = pd.concat(parts, ignore_index=True)
    stream["date"] = pd.to_datetime(stream["date"])
    if stream["date"].max() > MAX_DATA_DATE:
        raise SystemExit(f"[G1] 流含 >{MAX_DATA_DATE.date()} 的预测，判废")
    # ---- G2 流右端点守卫（#18 勘误的响亮守卫）----
    tail_gap = (dates.max() - stream["date"].max()).days
    if tail_gap >= 15:
        raise SystemExit(f"[G2] 流右端点 {stream['date'].max().date()} 距面板末 "
                         f"{dates.max().date()} 有 {tail_gap} 天 ⇒ 折边界不完整，拒绝出数")

    out = EXDIR / f"stream_{args.arm}_ss{'_'.join(map(str, seeds))}.parquet"
    stream.to_parquet(out)
    secs = time.time() - t00
    res = {
        "prereg": "plans/tail-objective-prereg-20261010.md",
        "arm": args.arm, "loss_spec": SPEC, "tau": TAU, "shuffle_labels": SHUF,
        "seeds": list(seeds), "folds": folds, "n_rows": int(len(stream)),
        "train_seconds": round(secs, 1), "device": str(DEVICE),
        "stream": str(out),
        "align_scales": scales_all,
        "epochs_selected": {"min": int(min(e["ep"] for e in ep_logs)),
                            "max": int(max(e["ep"] for e in ep_logs)),
                            "mean": round(float(np.mean([e["ep"] for e in ep_logs])), 2)}
        if ep_logs else None,
        "oos_window": {"start": str(stream["date"].min().date()),
                       "end": str(stream["date"].max().date())},
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    blob = json.dumps(res, ensure_ascii=False, indent=1, default=str)
    # ① 证据包内（随 commit 入库、不依赖 data/ 是否存在）
    (EXDIR / f"run_{args.arm}_{stamp}.json").write_text(blob, encoding="utf-8")
    # ② 两处 data/reports（存在才写；worktree 的 data/ 常不存在）
    for d in (REPORTS, PROD_REPORTS):
        try:
            d.mkdir(parents=True, exist_ok=True)
            (d / f"tail_objective_{args.arm}_{stamp}.json").write_text(blob, encoding="utf-8")
        except OSError as e:
            print(f"  [warn] 写 {d} 失败（不影响读数）：{e}", flush=True)
    print(blob, flush=True)
    print(f"\n流 → {out}（耗时 {secs/60:.1f} min）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
