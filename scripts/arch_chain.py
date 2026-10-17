"""判据链 · 臂 D（截面注意力）/ 臂 E（listwise 损失）执行器（2026-10-09）。

预注册：plans/model-upgrade-prereg-20261009.md §四 臂 D / 臂 E。
审计依据：P1-4（现有模型逐股独立打分、无股票间交互）。

## 为什么本脚本有三个臂而不是两个

臂 D 的注意力掩码与臂 E 的 listwise 损失**都要求同一交易日的股票同处一个 batch**，
而 F0 用的是跨日随机混洗（同批每日仅 1–2 只股票 ⇒ 两类改动都会失去意义）。
因此 D/E 必须改用**按日整批**（`fterm/model/daybatch.py`）。

但按日整批**本身会改变训练动态**（批内相关性、有效批大小、梯度噪声）。
若直接与 F0（随机混洗）比较，"批构造"与"架构/损失"两个变量混淆。故本脚本另设
**`F0day` 对照臂**：同样的按日整批、同样的 net2 + MSE。判据必须同时给出
「D/E 相对 F0」与「D/E 相对 F0day」两条配对 t。

## 用法
  python scripts/arch_chain.py --arm F0day     # 先跑对照
  python scripts/arch_chain.py --arm D
  python scripts/arch_chain.py --arm E
  python scripts/arch_chain.py --report        # 出裁决（对比所有已跑臂）
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from fterm.config import load_config  # noqa: E402
from fterm.data import ashare as sa  # noqa: E402
from fterm.features.build import build_dataset  # noqa: E402
from fterm.features.regime import market_regime_labels  # noqa: E402
from fterm.model import stats as st  # noqa: E402
from fterm.model.daybatch import day_batches  # noqa: E402
from fterm.model.dayids import build_day_ids  # noqa: E402
from fterm.model.foldframe import fold_frame  # noqa: E402
from fterm.model.losses import listmle_loss, listnet_loss  # noqa: E402
from fterm.model.net2 import (CrossSectionNet, N_BINS as N_BINS_DEFAULT,  # noqa: E402
                              fit_bins, to_bin_idx)
from fterm.model.net2_ln import CrossSectionNetLN              # noqa: E402  # 判据链 #42
from fterm.model.net2_variants import (CrossSectionNetAct,     # noqa: E402  # 判据链 #46
                                       CrossSectionNetBins)
from fterm.model.net4 import CrossSectionAttentionNet  # noqa: E402
from fterm.model.validate import spearman_rank_ic  # noqa: E402

OUT_DIR = ROOT / "experiments" / "arch_20261009"
PARTS = OUT_DIR / "parts"
FOLD_EDGES = ["2019-01-01", "2021-01-01", "2023-01-01", "2025-01-01"]
SEEDS = (7, 21)
LABEL_SPAN = 6
MAX_EPOCHS, PATIENCE, LR, WD = 30, 5, 1e-3, 1e-5
VAL_FRAC = 0.85
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 臂名 → (网络构造, 损失名, 是否按日整批)
ARM_SPEC = {
    "F0day": ("net2", "mse", True),
    "D": ("attn", "mse", True),
    "E": ("net2", "listnet", True),
    "E2": ("net2", "listmle", True),
    # 判据链 #42（2026-10-11）：BN→LN 单开关。**只加一个臂**，训练/评估口径与其余臂逐字相同。
    "L1": ("net2ln", "mse", True),
    # 判据链 #46（2026-10-11）：A2 分桶数扫描 + A3 激活函数。**各只改一个变量**。
    "B8": ("net2bins8", "mse", True),
    "B32": ("net2bins32", "mse", True),
    "B64": ("net2bins64", "mse", True),
    "PR": ("net2act", "mse", True),
}


def ic_series(df: pd.DataFrame) -> pd.Series:
    out = {}
    for d, g in df.groupby("date"):
        v = spearman_rank_ic(list(zip(g["pred"], g["y_ret"])))
        if v is not None:
            out[d] = v
    return pd.Series(out).sort_index()


def _tail_bucket(df: pd.DataFrame, q: float = 0.005) -> dict:
    """最优 q 分位篮子的周均收益（臂 E 预注册判据的第二项）。

    预注册 `plans/model-upgrade-prereg-20261009.md §四 臂 E` 判据 =
    「listwise IC 配对 t≥2 **且** 尾部（top0.5% 篮子周均收益）不劣」。
    口径逐字复用既有 `scripts/tailloss_chain.py::tail_bucket_weekly`
    （周五出信号、下周同批收益；k = max(3, round(N·q))），保证与 #18 可比。
    """
    import sys as _sys
    _scripts = str(Path(__file__).resolve().parent)
    if _scripts not in _sys.path:
        _sys.path.insert(0, _scripts)
    from tailloss_chain import tail_bucket_weekly  # noqa: E402
    return tail_bucket_weekly(df, q=q)


def _tail_weekly_series(path: Path, q: float = 0.005) -> pd.Series:
    """逐周 top0.5% 篮子收益序列（用于尾部判据的配对显著性检验）。

    与 `_tail_bucket` 同口径（同一 k 规则、同一 W-FRI 重采样），只是保留**逐周**值
    而不只给均值——这样"不劣"才能配一个配对 t，而不是只看点估计。
    """
    d = pd.read_parquet(path)
    d["date"] = pd.to_datetime(d["date"])
    rows = []
    for dt, g in d.groupby("date"):
        k = max(3, int(round(len(g) * q)))
        rows.append({"date": dt, "bucket_ret": float(g.nlargest(k, "pred")["y_ret"].mean())})
    s = pd.DataFrame(rows).set_index("date")["bucket_ret"]
    return s.resample("W-FRI").mean().dropna()


def _build_net(kind: str, n_feat: int, n_embed: int):
    if kind == "net2":
        return CrossSectionNet(n_feat, n_embed_features=n_embed).to(DEVICE)
    if kind == "net2ln":
        # 判据链 #42：BN→LN 单开关变体（预注册 plans/ln-prereg-20261011.md）。
        # ⚠ `elementwise_affine=False` ⇒ 参数差异恰为 2*HIDDEN*N_BLOCKS（仪器门 G4 可账平）。
        return CrossSectionNetLN(n_feat, n_embed_features=n_embed).to(DEVICE)
    if kind.startswith("net2bins"):
        # 判据链 #46 A2：**只**改分桶数（预注册 plans/a2a3-prereg-20261011.md）
        bins = int(kind[len("net2bins"):])
        return CrossSectionNetBins(n_feat, n_embed_features=n_embed, bins=bins).to(DEVICE)
    if kind == "net2act":
        # 判据链 #46 A3：**只**改激活（残差块内 ReLU→PReLU；输出头不变）
        return CrossSectionNetAct(n_feat, n_embed_features=n_embed).to(DEVICE)
    if kind == "attn":
        return CrossSectionAttentionNet(n_feat, n_embed_features=n_embed).to(DEVICE)
    raise ValueError(f"未知网络 {kind}")


def _fwd(net, kind, x, b, day_ids):
    """前向：attention 网需要 day_ids 构造同日掩码；net2 忽略之。"""
    if kind == "attn":
        return net(x, b, day_ids)
    return net(x, b)


def _loss(name, out, y, day_ids):
    if name == "mse":
        return torch.nn.functional.mse_loss(out, y)
    if name == "listnet":
        return listnet_loss(out, y, day_ids)
    if name == "listmle":
        return listmle_loss(out, y, day_ids)
    raise ValueError(f"未知损失 {name}")


def _forward_batched(net, kind: str, x: np.ndarray, b: np.ndarray,
                     day_ids: np.ndarray) -> np.ndarray:
    """按日整批前向（保持原行序），返回 float64 预测。

    ⚠️ **必须按日分批**（2026-10-10 修复）。臂 D 的注意力掩码是 B×B：若把整个
    验证段一次性前向（fold0 验证段 ~25.3 万行），掩码即 6.4e10 个元素
    （bool ≈ 59.8 GiB）⇒ 必然 OOM。首次冒烟实测该路径请求 25.37 GiB 后失败。
    训练（`day_batches`）与推理本已按日分批，**验证路径此前漏了**——三处必须同口径。
    """
    out = np.empty(len(day_ids), dtype="float64")
    with torch.no_grad():
        for idx in day_batches(day_ids, batch_days=1, shuffle=False):
            o = _fwd(net, kind,
                     torch.from_numpy(x[idx]).float().to(DEVICE),
                     torch.from_numpy(b[idx]).long().to(DEVICE),
                     torch.from_numpy(day_ids[idx]).to(DEVICE))
            out[idx] = o.detach().cpu().numpy()
    return out


def train_fold(cfg, bundle, arm, d0, d1, seeds) -> pd.DataFrame:
    kind, loss_name, daywise = ARM_SPEC[arm]
    d1_pad = d1 + pd.Timedelta(days=1)
    Xtr, mtr, ytr, meta_tr, cols = fold_frame(bundle, bundle.universe,
                                              pd.Timestamp(cfg.data.start), d0, 0)
    cut = d0 - pd.Timedelta(days=LABEL_SPAN * 2)
    keep = np.array([d < cut for d, _ in meta_tr])
    Xtr, mtr, ytr = Xtr[keep], mtr[keep], ytr[keep]
    meta_tr = [m for m, k in zip(meta_tr, keep) if k]
    Xte, mte, yte, meta_te, _ = fold_frame(bundle, bundle.universe, d0, d1_pad, 0)

    # ⚠⚠ 判据链 #46 A2 的**必要修正**（2026-10-11）：
    #   `bins` 同时决定 ①embedding 表行数 ②`fit_bins` 拟分的边界数。
    #   首版只改了 ① ⇒ bins=8 直接索引越界（CUDA device-side assert），
    #   bins=32/64 则**永远只有 16 个桶被用到** ⇒ 扫描从设计上无效。
    # ⇒ A2 变体的 `bins` 必须与 `fit_bins` 同源，否则不构成"单变量"。
    n_bins = int(kind[len("net2bins"):]) if kind.startswith("net2bins") else N_BINS_DEFAULT
    edges = fit_bins(np.concatenate([Xtr, np.zeros((1, Xtr.shape[1]))], axis=0),
                     n_bins=n_bins)
    Xtr_in = np.concatenate([Xtr, mtr], axis=1)
    Xte_in = np.concatenate([Xte, mte], axis=1)
    Xtr_bin, Xte_bin = to_bin_idx(Xtr, edges), to_bin_idx(Xte, edges)
    n_feat = Xtr_in.shape[1]
    day_tr = build_day_ids(d for d, _ in meta_tr)

    val_cut = int(len(Xtr) * VAL_FRAC)
    xt = torch.from_numpy(Xtr_in[:val_cut]).float()
    bt = torch.from_numpy(Xtr_bin[:val_cut]).long()
    yt = torch.from_numpy(ytr[:val_cut]).float()
    dt = torch.from_numpy(day_tr[:val_cut])

    # 训练批：按日整批（臂 D/E 的前提）；val 亦按日整批（见 _forward_batched）
    train_days = day_tr[:val_cut]

    preds = np.zeros(len(Xte_in))
    for seed in seeds:
        torch.manual_seed(seed)
        net = _build_net(kind, n_feat, len(cols))
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
        best, best_state, bad = float("inf"), None, 0
        for ep in range(MAX_EPOCHS):
            net.train()
            for idx in day_batches(train_days, batch_days=1, shuffle=True, seed=seed * 1000 + ep):
                ti = torch.from_numpy(idx)
                opt.zero_grad()
                o = _fwd(net, kind, xt[ti].to(DEVICE), bt[ti].to(DEVICE), dt[ti].to(DEVICE))
                loss = _loss(loss_name, o, yt[ti].to(DEVICE), dt[ti].to(DEVICE))
                if not torch.isfinite(loss):
                    raise ValueError(f"arch: {arm} 损失非有限（seed={seed} ep={ep}）")
                loss.backward(); opt.step()
            net.eval()
            # 验证也按日整批（臂 D 的注意力掩码为 B²，整段前向必然 OOM；见 _forward_batched）。
            # MSE 在 float64 上算：与批次划分无关、确定性、无 GPU 归约顺序差异
            # （口径仍是纯 val MSE，与 F0 一致；只是不再依赖单次整段前向）。
            ov = _forward_batched(net, kind, Xtr_in[val_cut:], Xtr_bin[val_cut:],
                                  day_tr[val_cut:])
            vl = float(np.mean((ov - ytr[val_cut:]) ** 2))
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
        # 推理也按**日整批**：臂 D 的注意力掩码按 day_id 分块，若把不同日混进同一批，
        # 会跨日互相注意（推理期泄漏）。故与训练同口径。
        day_te = build_day_ids(d for d, _ in meta_te)
        out_te = _forward_batched(net, kind, Xte_in, Xte_bin, day_te)
        preds += out_te / len(seeds)
        print(f"        seed {seed} val {best:.6f}", flush=True)
    if not np.isfinite(preds).all():
        raise ValueError(f"arch: {arm} 预测含 NaN（{int(np.isnan(preds).sum())}/{len(preds)}）")
    return pd.DataFrame({"date": [d for d, _ in meta_te], "product": [c for _, c in meta_te],
                         "pred": preds, "y_ret": yte})


def run_arm(cfg, bundle, arm, edges):
    parts = []
    for k in range(len(edges) - 1):
        p = PARTS / f"{arm}_f{k}.parquet"
        if p.exists():
            df = pd.read_parquet(p); df["date"] = pd.to_datetime(df["date"])
            print(f"      [{arm}] fold{k} 已缓存，跳过", flush=True)
        else:
            t0 = time.time()
            print(f"      [{arm}] fold{k} 训练中（{len(SEEDS)} 种子，{ARM_SPEC[arm]}）...", flush=True)
            df = train_fold(cfg, bundle, arm, edges[k], edges[k + 1], SEEDS)
            df.to_parquet(p)
            print(f"      [{arm}] fold{k} 完成 {len(df)} 条（{time.time()-t0:.0f}s）", flush=True)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def _f0_cache():
    for p in [ROOT / "experiments" / "external_20261009" / "stream_F0_f4.parquet",
              ROOT / "experiments" / "regime_20261009" / "stream_F0_f4.parquet"]:
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------- 仪器门（§五）
ISOLATION_MAX = pd.Timestamp("2026-09-30")   # 前瞻窗隔离线（沿用 #18 竞态处置）
RIGHT_GAP_MAX = 15                           # §五-3：流右端点距面板末须 < 15 天
TRIPWIRE_MAX = 0.025                         # §五-1：置换后 |mean IC| ≤ 0.025
CONN_WINDOWS = [("2019-01-01", "2019-04-01"), ("2021-01-01", "2021-04-01"),
                ("2023-01-01", "2023-04-01"), ("2025-01-01", "2025-04-01")]


def _instrument_gates(cfg, sf0, f0p, arm_names):
    """预注册 `plans/model-upgrade-prereg-20261009.md §五` 五道仪器门。

    ⚠️ **时点声明（2026-10-10，必读）**：本函数在四臂读数产出**之后**补入。
    原缺口见 `docs/03 §45` 已知缺口 #2（此前五门只由 verifier/Lead 在**外部**核验，
    裁决 json 内无留痕）。**本函数不改任何判据数值**，只把既有核验落成机器可读留痕；
    补入后重跑 `--report` 必须**复现原有全部读数**（IC / t / ICIR 逐位不变），
    否则视为实现缺陷。任一门失败 ⇒ 按 §五 该臂读数作废（此处 raise 响亮失败）。
    """
    import platform

    import torch as _torch

    from fterm.data import ashare as sa
    from fterm.features.build import build_dataset
    from fterm.model.foldframe import fold_frame

    gates: dict = {}

    # ---- §五-4 环境谱系登记（.venv 与 .venv312 读数不可互比）----
    gates["env_lineage"] = {"python": platform.python_version(),
                            "pandas": pd.__version__, "numpy": np.__version__,
                            "torch": _torch.__version__, "device": str(DEVICE)}

    panels = sa.load_panels(cfg)
    panel_end = pd.Timestamp(panels["close"].index[-1])

    # ---- §五-5 面板竞态守卫：研究线必须钉在隔离窗内 ----
    race_ok = bool(panel_end <= ISOLATION_MAX)
    gates["panel_race_guard"] = {"panel_end": str(panel_end.date()),
                                 "isolation_max": str(ISOLATION_MAX.date()),
                                 "ok": race_ok}
    if not race_ok:
        raise ValueError(f"仪器门 §五-5 失败：面板末 {panel_end.date()} 已越过隔离线 "
                         f"{ISOLATION_MAX.date()}，研究线读数作废")

    # ---- §五-1 阴性置换 tripwire：日内打乱标签后 |mean IC| 必须塌陷 ----
    # 固定种子保证可复现；对每个在场的臂流各做一次。
    trip = {}
    for name, df in [("F0", sf0)] + [(a, None) for a in arm_names]:
        if df is None:
            p = OUT_DIR / f"stream_{name}.parquet"
            if not p.exists():
                continue
            df = pd.read_parquet(p)
            df["date"] = pd.to_datetime(df["date"])
        rng = np.random.default_rng(20261010)
        perm = df.copy()
        perm["y_ret"] = (perm.groupby("date")["y_ret"]
                         .transform(lambda s: s.to_numpy()[rng.permutation(len(s))]))
        m = abs(float(ic_series(perm).mean()))
        trip[name] = {"abs_mean_ic_after_permutation": round(m, 6),
                      "threshold": TRIPWIRE_MAX, "ok": bool(m <= TRIPWIRE_MAX)}
    gates["permutation_tripwire"] = trip
    bad = [k for k, v in trip.items() if not v["ok"]]
    if bad:
        raise ValueError(f"仪器门 §五-1 失败：{bad} 置换后 |mean IC| > {TRIPWIRE_MAX}，读数作废")

    # ---- §五-3 流右端点守卫：OOS 流右端点距面板末须 < 15 天 ----
    right = {}
    for name in ["F0day"] + list(arm_names):
        p = OUT_DIR / f"stream_{name}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["date"])
        hi = pd.to_datetime(d["date"]).max()
        gap = int((panel_end - hi).days)
        right[name] = {"stream_last": str(hi.date()), "gap_days": gap,
                       "threshold": RIGHT_GAP_MAX, "ok": bool(gap < RIGHT_GAP_MAX)}
    gates["right_endpoint_guard"] = right
    bad = [k for k, v in right.items() if not v["ok"]]
    if bad:
        raise ValueError(f"仪器门 §五-3 失败：{bad} 流右端点距面板末 ≥ {RIGHT_GAP_MAX} 天，读数作废")

    # ---- §五-2 新鲜 F0 位连：用当前 HEAD 的 fold_frame 重算各折前 3 个月，
    #      与 F0 缓存流的 (date, product) 键集与 y_ret **逐位**比对（应为 0）----
    bundle = build_dataset(cfg, compute_fingerprints=False)
    conn = {}
    for k, (d0, d1) in enumerate(CONN_WINDOWS):
        _X, _m, y, meta, _cols = fold_frame(bundle, bundle.universe,
                                           pd.Timestamp(d0), pd.Timestamp(d1), 0)
        cur = pd.DataFrame({"date": pd.to_datetime([d for d, _ in meta]),
                            "product": [c for _, c in meta], "y_cur": y})
        # ⚠️ 口径：只重算**一个窗口** ⇒ F0 流里该窗口之外的 785 万行「只在流里」
        # 是**正常**的，不能算不一致。判据只有两条：
        #   (a) 重算出的每一行都必须能在流里找到（keys_only_in_recompute == 0）；
        #   (b) 交集行的 y_ret 必须逐位相等（max|Δy| == 0）。
        # 反方向（流里多出来的行）不作判据——那只是窗口未覆盖。
        j = cur.merge(sf0[["date", "product", "y_ret"]], on=["date", "product"], how="left",
                      indicator=True)
        only_cur = int((j["_merge"] == "left_only").sum())
        both = j[j["_merge"] == "both"]
        dy = float((both["y_cur"] - both["y_ret"]).abs().max()) if len(both) else float("nan")
        conn[f"fold{k}"] = {"window": f"{d0}..{d1}", "rows_recomputed": int(len(cur)),
                            "rows_matched_in_F0_stream": int(len(both)),
                            "keys_only_in_recompute": only_cur,
                            "max_abs_dy_ret": dy,
                            "ok": bool(only_cur == 0 and dy == 0.0)}
    gates["fresh_f0_bitconnect"] = conn
    bad = [k for k, v in conn.items() if not v["ok"]]
    if bad:
        raise ValueError(f"仪器门 §五-2 失败：{bad} 位连不一致（面板被改写或标签口径漂移），读数作废")
    return gates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARM_SPEC), default=None)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--no-gates", action="store_true",
                    help="跳过 §五 仪器门（仅供快速调试；正式裁决不得使用）")
    args = ap.parse_args()
    PARTS.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "config.yaml")

    if args.arm and not args.report:
        print(f"== 臂 {args.arm}（device={DEVICE}，{ARM_SPEC[args.arm]}）==", flush=True)
        panels = sa.load_panels(cfg)
        bundle = build_dataset(cfg)
        dates = panels["close"].index
        edges = [pd.Timestamp(x) for x in FOLD_EDGES] + [dates[-1] + pd.Timedelta(days=1)]
        sb = run_arm(cfg, bundle, args.arm, edges)
        sb.to_parquet(OUT_DIR / f"stream_{args.arm}.parquet")
        print(f"  {args.arm}: {len(sb)} 条 OOS 汇总完成", flush=True)
        return

    # ---- 裁决：每个已跑臂 vs F0（随机混洗）与 vs F0day（按日整批）----
    f0p = _f0_cache()
    if f0p is None:
        raise FileNotFoundError("缺 F0 缓存")
    sf0 = pd.read_parquet(f0p); sf0["date"] = pd.to_datetime(sf0["date"])
    ic_f0 = ic_series(sf0)
    reg = market_regime_labels(cfg).dropna()

    results, lines = {}, [f"臂 D/E 裁决 {datetime.now().strftime('%Y%m%d_%H%M')}（device={DEVICE}）",
                          f"F0（随机混洗）: IC {st.summarize_ic(ic_f0.to_numpy())['mean']:+.4f} "
                          f"ICIR {st.summarize_ic(ic_f0.to_numpy())['icir']:+.3f}"]
    # ⚠️ 2026-10-10 修复（verifier 复核抓到的静默缺口 #2）：`paired_t_vs_F0day` 原依赖
    # 两个**无保护的隐式前提**——(1) ARM_SPEC 的 dict 插入顺序让 "F0day" 排在 D/E/E2 之前；
    # (2) stream_F0day.parquet 必须存在。任一不成立 ⇒ 循环里 ic_f0day 恒为 None ⇒
    # **所有臂静默失去 vs F0day 的配对 t**，连 txt 里的提示也一并消失。
    # 而 erratum §四-1 明定「只报 vs F0 的读数不可解释」⇒ 这是**必需**读数。
    # 修复分两层（verifier 指出第一版只堵住前提 (2)，前提 (1) 仍敞着）：
    #   (a) 文件缺失 ⇒ 响亮失败（下）；
    #   (b) **F0day 基准显式前置**——不再依赖 dict 顺序，任何顺序下 D/E/E2 都拿得到。
    if not (OUT_DIR / "stream_F0day.parquet").exists():
        raise FileNotFoundError(
            f"缺 F0day 对照臂流 {OUT_DIR / 'stream_F0day.parquet'}——"
            "erratum §四-1 要求 D/E 必须同时给出 vs F0 与 vs F0day 两条配对 t；"
            "缺 F0day 会使该必需读数静默消失。先跑 `--arm F0day`。")
    # (b) 显式算 F0day 的 IC 序列（与循环顺序解耦）
    _f0day_df = pd.read_parquet(OUT_DIR / "stream_F0day.parquet")
    _f0day_df["date"] = pd.to_datetime(_f0day_df["date"])
    if _f0day_df["pred"].notna().sum() != len(_f0day_df):
        raise ValueError("仪器门失败：F0day 流含 NaN，读数作废")
    ic_f0day = ic_series(_f0day_df)
    del _f0day_df
    # 迭代顺序仍按 ARM_SPEC（F0day 自然在内），但 vs F0day 的比较已不依赖它。
    for arm in [a for a in ARM_SPEC if (OUT_DIR / f"stream_{a}.parquet").exists()]:
        sa_ = pd.read_parquet(OUT_DIR / f"stream_{arm}.parquet")
        sa_["date"] = pd.to_datetime(sa_["date"])
        if sa_["pred"].notna().sum() != len(sa_):
            raise ValueError(f"仪器门失败：{arm} 流含 NaN，读数作废")
        ic = ic_series(sa_)
        s = st.summarize_ic(ic.to_numpy())
        cm = ic.index.intersection(ic_f0.index)
        pt_f0 = st.paired_t(ic.loc[cm].to_numpy(), ic_f0.loc[cm].to_numpy())
        rec = {"ic": s, "paired_t_vs_F0": pt_f0}
        # F0day 自比无意义（t 恒 0）；基准已在循环外显式算好，不依赖本循环顺序。
        if arm != "F0day":
            cmd = ic.index.intersection(ic_f0day.index)
            rec["paired_t_vs_F0day"] = st.paired_t(ic.loc[cmd].to_numpy(),
                                                   ic_f0day.loc[cmd].to_numpy())
        cidx = ic.index.intersection(reg.index)
        strat = st.regime_stratified_ic(ic.loc[cidx].to_numpy(), reg.loc[cidx].to_numpy())
        rec["regime_min_ic"] = st.regime_min_ic(strat)
        # 臂 E 预注册判据第二项：尾部（top0.5% 篮子周均收益）不劣于 F0。
        # 口径复用 tailloss_chain.tail_bucket_weekly（与 #18 可比）。
        try:
            rec["tail"] = _tail_bucket(sa_)
        except Exception as e:  # 尾部读数失败不应作废主判据，但必须留痕
            rec["tail"] = {"error": f"{type(e).__name__}: {e}"}
        results[arm] = rec
        extra = ""
        if "paired_t_vs_F0day" in rec:
            extra = f" | vs F0day t={rec['paired_t_vs_F0day']['t']:+.2f}"
        tail_s = ""
        tv = rec.get("tail", {}).get("top0.5pct")
        if isinstance(tv, dict):
            tail_s = f" | 尾部周均 {tv['mean_week']:+.5f}"
        lines.append(f"{arm:6s} IC {s['mean']:+.4f} ICIR {s['icir']:+.3f} "
                     f"| vs F0 t={pt_f0['t']:+.2f}{extra} | regime最小IC {rec['regime_min_ic']}"
                     f"{tail_s}")

    # 尾部判据（臂 E）：listwise 臂的 top0.5% 篮子周均收益不得劣于 F0 与 F0day。
    #
    # ⚠️ 2026-10-10 修复（verifier 复核抓到的静默缺口）：初版写 `for base in ("F0","F0day")`
    # 但只从 `results` 取——而 `results` 的键来自 `ARM_SPEC`（= F0day/D/E/E2），
    # **不含 "F0"**（F0 只经 `_f0_cache()` 读入，从不进 results）。故 `"F0" in results`
    # 恒为 False ⇒ F0 被**静默跳过**，tail_ref 只剩 F0day，无任何告警。
    # 后果是判据被系统性放松：实测 top0.5% 周均 F0 = +0.007771 vs F0day = +0.003689
    # （相差 2.11 倍，且 F0day 尾部显著劣于 F0）——把更松的门槛当成了唯一门槛。
    # 现直接对 F0 流原件算尾部读数，使两个基准都在场。
    tail_ref = {}
    try:
        tail_ref["F0"] = _tail_bucket(sf0)["top0.5pct"]["mean_week"]
    except Exception as e:
        tail_ref["F0_error"] = f"{type(e).__name__}: {e}"
    for base in ("F0day",):
        if base in results:
            tv = results[base].get("tail", {}).get("top0.5pct")
            if isinstance(tv, dict):
                tail_ref[base] = tv["mean_week"]
    # 尾部点估计的显著性：对 F0 与 F0day 各做周度配对差（naive t 与 NW t）。
    # 预注册只要求"不劣"（点估计），但点估计无显著性时读者会误读 ⇒ 一并给出并留痕。
    try:
        tail_series = {b: _tail_weekly_series(p) for b, p in
                       (("F0", f0p), ("F0day", OUT_DIR / "stream_F0day.parquet"))}
        tail_pair = {}
        for b in ("F0", "F0day"):
            for arm in ("E", "E2"):
                if arm not in results:
                    continue
                s_arm = _tail_weekly_series(OUT_DIR / f"stream_{arm}.parquet")
                cmn = s_arm.index.intersection(tail_series[b].index)
                if len(cmn) >= 16:
                    diff = (s_arm.loc[cmn] - tail_series[b].loc[cmn]).to_numpy()
                    tail_pair[f"{arm}_minus_{b}"] = {
                        "n_weeks": int(len(diff)),
                        "mean_diff": float(diff.mean()),
                        "naive_t": float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))),
                        "nw_t": st.newey_west_t(diff)}
        if tail_pair:
            out_tail_pair = tail_pair
        else:
            out_tail_pair = {}
    except Exception as e:
        out_tail_pair = {"error": f"{type(e).__name__}: {e}"}
    for arm in ("E", "E2"):
        if arm not in results:
            continue
        tv = results[arm].get("tail", {}).get("top0.5pct")
        if isinstance(tv, dict) and tail_ref:
            results[arm]["tail_not_worse"] = {
                b: bool(tv["mean_week"] >= ref - 1e-9)
                for b, ref in tail_ref.items() if not b.endswith("_error")}
            results[arm]["tail_ref"] = tail_ref

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    # ---- §五 仪器门（任一门失败即响亮 raise，不产出裁决）----
    if args.no_gates:
        gates = {"skipped": True,
                 "note": "本次调用带 --no-gates ⇒ 无仪器门留痕，裁决不得引用"}
        print("⚠ --no-gates：本裁决无 §五 仪器门留痕，不得作为正式裁决引用", flush=True)
    else:
        print("== 仪器门（§五：置换 tripwire / 位连 / 右端点 / 环境谱系 / 面板竞态）==",
              flush=True)
        gates = _instrument_gates(cfg, sf0, f0p,
                                  [a for a in ARM_SPEC if a != "F0day"])
        for k, v in gates.items():
            ok = v.get("ok") if isinstance(v, dict) else None
            print(f"   {k}: {'✓' if ok else ('（留痕）' if ok is None else '✗')}", flush=True)

    out = {"prereg": "plans/model-upgrade-prereg-20261009.md#臂D-E", "device": str(DEVICE),
           "seeds": list(SEEDS), "folds": len(FOLD_EDGES), "results": results,
           "tail_ref": tail_ref, "tail_paired": out_tail_pair,
           "instrument_gates": gates,
           "verdict_word": ""}
    (OUT_DIR / f"verdict_{stamp}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    gate_line = ("仪器门：全部通过（见 json.instrument_gates）" if not args.no_gates
                 else "仪器门：**未运行**（--no-gates）")
    text = "\n".join(lines + [gate_line, "判决词：＿＿＿（由用户填写）"])
    (OUT_DIR / f"verdict_{stamp}.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
