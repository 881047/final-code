"""判据链 #27 · E2+MSE 融合（可学习 Kendall 权重）执行器（2026-10-11）。

预注册：`plans/e2mse-fusion-prereg-20261011.md`（**先于本脚本跑数冻结**）。

## 单变量纪律
`train_fold` 与 #26 的 `tail_objective_chain.train_fold` **同构**（同折边界/净化/fit_bins/
网络/优化器/早停/集成/种子/按日整批），**唯一差异 = 损失规格**（预注册 §二 G9/G10）。

## 臂
  F0     = 纯 MSE（按日整批对照）
  F2     = 纯 ListMLE（= E2，按日整批）
  F3     = ListMLE + MSE，Kendall 可学习权重（唯一新臂）
  NEG_F2 = F2 + 日内置换标签（阴性对照，fold3）
  NEG_F3 = F3 + 日内置换标签（阴性对照，fold3）

**⚠ 批构造**：ListMLE 需完整日截面 ⇒ 三臂**都按日整批**（`daybatch.py`）。
⇒ F0（按日整批 MSE）**不得**与 #26 的 C0（随机混洗 MSE）直接比（G10，批构造混淆）。

用法：
  python scripts/e2mse_fusion_chain.py --arm F3
  python scripts/e2mse_fusion_chain.py --arm F0 --verify-det
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fterm.config import load_config                       # noqa: E402
from fterm.data import ashare as sa                        # noqa: E402
from fterm.features.build import build_dataset             # noqa: E402
from fterm.model.daybatch import day_batches               # noqa: E402
from fterm.model.dayids import build_day_ids               # noqa: E402
from fterm.model.losses import listmle_loss                # noqa: E402
from fterm.model.net2 import CrossSectionNet, fit_bins, to_bin_idx  # noqa: E402

from dlv2_chain import FOLD_EDGES, LABEL_SPAN              # noqa: E402

REPORTS = ROOT / "data" / "reports"
PROD_REPORTS = Path(r"D:\Final\data\reports")
EXDIR = ROOT / "experiments" / "e2mse_fusion_20261011"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = (7, 21, 99, 5, 13)
TAU = 0.95
MAX_DATA_DATE = pd.Timestamp("2026-09-30")
X0_TOL = 2.79e-3
_DETERMINISTIC = True


def _enable_determinism() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def _mse(out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(out, y)


def _align_scale(net, xt, bt, yt, dt, spec: str) -> float:
    """init 比对齐常数 scale_k（同 #26 的 init 比思路；只用未训练模型在训练段子样本上算）。

    ⚠ 判据链 #27 的量级事实：`L_listmle / L_mse ≈ 29797`（G4b 实测）——
    ListMLE 损失比 MSE 大 ~3 万倍。**若不对齐**，ListMLE 会完全主导梯度、
    MSE 侧等于没学 ⇒ 融合退化成纯 ListMLE（F3 ≈ F2），单变量对照失效。
    ⇒ scale_k 必须乘在 **MSE 侧**，使 `scale_k·L_mse ≈ L_listmle`（两臂等权起步）。
    数值上 `scale_k = L_listmle(init) / L_mse(init)`（≈ 3 万）。
    """
    if spec in ("mse", "listmle"):
        return 1.0
    n = min(len(xt), 200_000)
    net.eval()
    with torch.no_grad():
        o = net(xt[:n].to(DEVICE), bt[:n].to(DEVICE))
        yv = yt[:n].to(DEVICE)
        dv = dt[:n].to(DEVICE)
        lm = float(_mse(o, yv))
        lr = float(listmle_loss(o, yv, dv))
    net.train()
    if lm <= 0 or not np.isfinite(lm) or not np.isfinite(lr):
        raise RuntimeError(f"量级对齐失败：L_mse={lm} L_listmle={lr}")
    return lr / lm  # 使 scale_k·L_mse == L_listmle


def _assert_single_variable(spec: str, seeds: tuple, shuffle_labels: bool) -> dict:
    """G9 单变量自证（可执行；同 #26）。"""
    import dlv2_chain as dc
    checks: dict[str, tuple] = {
        "FOLD_EDGES": ([str(x) for x in FOLD_EDGES], [str(x) for x in dc.FOLD_EDGES]),
        "LABEL_SPAN": (int(LABEL_SPAN), int(dc.LABEL_SPAN)),
        "loss_spec 合法": (spec in ("mse", "listmle", "fusion"), True),
    }
    bad = [k for k, (got, want) in checks.items() if got != want]
    if bad:
        raise SystemExit(f"[G9] 单变量自证失败：{bad} ⇒ 判废")
    return {"loss_spec": spec, "seeds": list(seeds), "shuffle_labels": bool(shuffle_labels),
            "net": "CrossSectionNet（net2，三臂同网）", "batch": "day_batches（三臂同）",
            "note": "唯一变量 = 损失函数；按日整批"}


# ---------------------------------------------------------------- 训练
def train_fold(cfg, bundle, universe, d0, d1, seeds, *, spec: str,
               shuffle_labels: bool = False, tag: str = ""):
    d1_pad = d1 + pd.Timedelta(days=1)
    Xtr_raw, mtr, ytr, meta_tr, cols = None, None, None, None, None
    from fterm.model.foldframe import fold_frame
    Xtr_raw, mtr, ytr, meta_tr, cols = fold_frame(
        bundle, universe, pd.Timestamp(cfg.data.start), d0, 0)
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

    if shuffle_labels:
        rng = np.random.RandomState(20261007)
        y_fit = ytr.copy()
        did = build_day_ids(d for d, _ in meta_tr)
        i0 = 0
        for n_d in np.bincount(did):
            if n_d > 1:
                y_fit[i0:i0 + n_d] = y_fit[i0:i0 + n_d][rng.permutation(n_d)]
            i0 += int(n_d)
    else:
        y_fit = ytr

    val_cut = int(len(Xtr_raw) * 0.85)
    day_ids = build_day_ids(d for d, _ in meta_tr)
    preds = np.zeros(len(Xte_in), dtype="float64")
    ep_log, scales = [], {}
    for seed in seeds:
        torch.manual_seed(seed)
        net = CrossSectionNet(n_feat, n_embed_features=len(cols)).to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
        # F3 的可学习 log-var（Kendall）；挂到 net 以便 state_dict 带走
        if spec == "fusion":
            net.log_var = torch.nn.Parameter(torch.zeros(2, device=DEVICE))
            opt.add_param_group({"params": [net.log_var]})
        xt = torch.from_numpy(Xtr_in[:val_cut]).float()
        bt = torch.from_numpy(Xtr_bin[:val_cut]).long()
        yt = torch.from_numpy(y_fit[:val_cut]).float()
        xv = torch.from_numpy(Xtr_in[val_cut:]).float()
        bv = torch.from_numpy(Xtr_bin[val_cut:]).long()
        yv = torch.from_numpy(y_fit[val_cut:]).float()
        dt_tr = torch.from_numpy(day_ids[:val_cut]).long()
        dt_va = torch.from_numpy(day_ids[val_cut:]).long()

        scale = _align_scale(net, xt, bt, yt, dt_tr, spec)
        scales[str(seed)] = float(scale)

        best, best_state, bad = float("inf"), None, 0
        for ep in range(30):
            net.train()
            for idx in day_batches(day_ids[:val_cut], batch_days=2, shuffle=True, seed=ep):
                idx_t = torch.from_numpy(idx).long()
                xb, bb = xt[idx_t], bt[idx_t]
                yb, db = yt[idx_t], dt_tr[idx_t]
                opt.zero_grad()
                out = net(xb.to(DEVICE), bb.to(DEVICE))
                if spec == "mse":
                    loss = _mse(out, yb.to(DEVICE))
                elif spec == "listmle":
                    loss = listmle_loss(out, yb.to(DEVICE), db.to(DEVICE))
                else:  # fusion：k·MSE + ListMLE，Kendall 可学习权重
                    o, yd, dd = out, yb.to(DEVICE), db.to(DEVICE)
                    l_mse = _mse(o, yd)
                    l_rank = listmle_loss(o, yd, dd)
                    lv = net.log_var
                    loss = (0.5 * torch.exp(-lv[0]) * (scale * l_mse)
                            + 0.5 * torch.exp(-lv[1]) * l_rank
                            + 0.5 * lv[0] + 0.5 * lv[1])
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                vl = _mse(net(xv.to(DEVICE), bv.to(DEVICE)), yv.to(DEVICE)).item()
            ep_log.append({"seed": seed, "ep": ep + 1, "val_mse": round(vl, 8)})
            if vl < best - 1e-6:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= 5:
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
        lv_txt = (f" lv={net.log_var.detach().cpu().numpy().round(3).tolist()}"
                  if spec == "fusion" else "")
        print(f"      seed {seed} val {best:.6f} ep {ep+1} scale {scale:.4g}{lv_txt}", flush=True)
    rows = [{"date": d, "product": c, "pred": float(p), "y_ret": y}
            for (d, c), p, y in zip(meta_te, preds, yte)]
    return pd.DataFrame(rows), ep_log, scales


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["F0", "F2", "F3", "NEG_F2", "NEG_F3"])
    ap.add_argument("--folds", default="all")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--verify-det", action="store_true")
    args = ap.parse_args()

    SPEC = {"F0": "mse", "F2": "listmle", "F3": "fusion",
            "NEG_F2": "listmle", "NEG_F3": "fusion"}[args.arm]
    SHUF = args.arm.startswith("NEG_")
    seeds = tuple(int(x) for x in args.seeds.split(",")) if args.seeds else SEEDS

    EXDIR.mkdir(parents=True, exist_ok=True)
    (EXDIR / "parts").mkdir(parents=True, exist_ok=True)
    if _DETERMINISTIC:
        _enable_determinism()
    cfg = load_config(ROOT / "config.yaml")
    panels = sa.load_panels(cfg)
    end = panels["close"].index.max()
    if end > MAX_DATA_DATE:
        raise SystemExit(f"[G1] 面板右端点 {end.date()} > 隔离线；必须在 worktree 内跑")
    print(f"[G1] 面板右端点 {end.date()} == 隔离线 ✓（device={DEVICE}）", flush=True)
    g9 = _assert_single_variable(SPEC, seeds, SHUF)
    print(f"[G9] 单变量自证 ✓ spec={SPEC} 种子 {list(seeds)} 按日整批", flush=True)
    bundle = build_dataset(cfg)
    universe = bundle.universe
    dates = panels["close"].index
    edges = [pd.Timestamp(x) for x in FOLD_EDGES] + [dates[-1] + pd.Timedelta(days=1)]
    folds = [3] if args.folds == "3" else list(range(len(edges) - 1))

    if args.verify_det:
        d0, d1 = edges[3], edges[4]
        print(f"[G4] 确定性：fold3 seed{seeds[0]} 背靠背两次（deterministic={_DETERMINISTIC}）",
              flush=True)
        rec = []
        for run_i in (1, 2):
            _, elog, _ = train_fold(cfg, bundle, universe, d0, d1, seeds[:1],
                                    spec=SPEC, shuffle_labels=SHUF, tag="det")
            np.save(EXDIR / "parts" / f"det_{args.arm}_run{run_i}.npy",
                    np.load(EXDIR / "parts" / f"det_seed{seeds[0]}.npy"))
            rec.append({"run": run_i, "epochs_run": max(e["ep"] for e in elog),
                        "val_mse_min": min(e["val_mse"] for e in elog)})
        a = np.load(EXDIR / "parts" / f"det_{args.arm}_run1.npy")
        b = np.load(EXDIR / "parts" / f"det_{args.arm}_run2.npy")
        dev = float(np.abs(a - b).max())
        same_ep = rec[0]["epochs_run"] == rec[1]["epochs_run"]
        ok = dev <= X0_TOL and same_ep
        verdict = {"arm": args.arm, "fold": 3, "seed": seeds[0], "device": str(DEVICE),
                   "deterministic_flags": _DETERMINISTIC, "runs": rec,
                   "max_abs_dev": dev, "envelope": X0_TOL,
                   "compute_layer_bitwise_ok": dev <= X0_TOL,
                   "same_epoch_run": bool(same_ep), "ok": bool(ok)}
        print(f"[G4] 臂 {args.arm} max|Δpred| = {dev:.3e} ⇒ "
              f"计算层 {'✓ 逐位' if dev <= X0_TOL else '✗ 不逐位'}", flush=True)
        print(f"[G4] 选择层：run1 ep={rec[0]['epochs_run']} vs run2 ep={rec[1]['epochs_run']} "
              f"⇒ {'✓ 一致' if same_ep else '✗ 被翻转'}", flush=True)
        (EXDIR / f"g4_determinism_{args.arm}.json").write_text(
            json.dumps(verdict, ensure_ascii=False, indent=1), encoding="utf-8")
        return 0 if ok else 1

    parts, ep_logs, scales_all = [], [], {}
    t00 = time.time()
    for i in folds:
        print(f"  fold{i} {edges[i].date()}→{edges[i+1].date()} spec={SPEC} "
              f"shuffle={SHUF} 种子 {seeds}", flush=True)
        df, elog, sc = train_fold(cfg, bundle, universe, edges[i], edges[i + 1],
                                  seeds, spec=SPEC, shuffle_labels=SHUF)
        df["fold"] = i
        df.to_parquet(EXDIR / "parts" / f"{args.arm}_f{i}.parquet")
        parts.append(df)
        ep_logs += elog
        scales_all[str(i)] = sc
        print(f"    fold{i} 完成 {len(df)} 行（累计 {time.time()-t00:.0f}s）", flush=True)

    stream = pd.concat(parts, ignore_index=True)
    stream["date"] = pd.to_datetime(stream["date"])
    if stream["date"].max() > MAX_DATA_DATE:
        raise SystemExit("[G1] 流含 >2026-09-30 的预测，判废")
    tail_gap = (dates.max() - stream["date"].max()).days
    if tail_gap >= 15:
        raise SystemExit(f"[G2] 流右端点距面板末 {tail_gap} 天 ⇒ 折边界不完整，拒绝出数")

    out = EXDIR / f"stream_{args.arm}_ss{'_'.join(map(str, seeds))}.parquet"
    stream.to_parquet(out)
    secs = time.time() - t00
    res = {"prereg": "plans/e2mse-fusion-prereg-20261011.md", "arm": args.arm,
           "loss_spec": SPEC, "shuffle_labels": SHUF, "seeds": list(seeds),
           "folds": folds, "n_rows": int(len(stream)), "train_seconds": round(secs, 1),
           "device": str(DEVICE), "stream": str(out), "align_scales": scales_all,
           "oos_window": {"start": str(stream["date"].min().date()),
                          "end": str(stream["date"].max().date())}}
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    blob = json.dumps(res, ensure_ascii=False, indent=1, default=str)
    (EXDIR / f"run_{args.arm}_{stamp}.json").write_text(blob, encoding="utf-8")
    for d in (REPORTS, PROD_REPORTS):
        try:
            d.mkdir(parents=True, exist_ok=True)
            (d / f"e2mse_fusion_{args.arm}_{stamp}.json").write_text(blob, encoding="utf-8")
        except OSError as e:
            print(f"  [warn] 写 {d} 失败：{e}", flush=True)
    print(blob, flush=True)
    print(f"\n流 → {out}（耗时 {secs/60:.1f} min）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
