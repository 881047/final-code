"""判据链 #14 · rank-loss 臂执行器 —— 判据冻结于 plans/rankloss-prereg-20261007.md。

只动一个变量（训练损失）：臂 B = #8 训练路径 + RankNet pairwise 秩项
（batch 内同日相邻对，λ=1，早停仍为 val MSE）；对照 = 已缓存 #8 MSE 流
（不重训）。仪器：X0 用原版 train_fold_seedset 重跑折 0 逐位自证；
阴性对照 = 日内 shuffle 标签（train+val 同源置换）。
主判据：日频 RankIC 配对 t（双种子集都过）+ 组合层月度对数配对 t。
判决词由用户填写。

smoke：RANKLOSS_SMOKE=1 → 微折/2 epoch/单种子，跳过 X0 与 S1 透明基线。
"""
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from fterm.config import load_config  # noqa: E402
from fterm.data import ashare as sa  # noqa: E402
from fterm.features.build import build_dataset  # noqa: E402
from fterm.model.dayids import build_day_ids  # noqa: E402
from fterm.model.net2 import CrossSectionNet, fit_bins, to_bin_idx  # noqa: E402
from fterm.model.validate import spearman_rank_ic  # noqa: E402
from dlv2_chain import (BATCH, CPU_CAP_PER_DAY, LABEL_SPAN, LR,  # noqa: E402
                        SEED_SETS as BASE_SEED_SETS, USE_GPU, WD, fold_frame)
from gate_chain import run_model_arm  # noqa: E402
from monthly_chain import monthly_log_paired  # noqa: E402

REPORTS = Path(__file__).resolve().parents[1] / "data" / "reports"
EXDIR = Path(__file__).resolve().parents[1] / "experiments" / "rankloss_20261007"
SMOKE = os.environ.get("RANKLOSS_SMOKE") == "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- 冻结（预注册 §一/§二） ----
LAMBDA_RANK = 1.0            # rank_mean 与 mse_mean 等权相加
LABEL_SHUFFLE_SEED = 20261007
TRIPWIRE_IC = 0.025          # 阴性对照 tripwire：|mean IC| 上限（真实信号 0.05 之半）
C2A_T = 2.0
X0_FOLD = 0                  # X0 自证折

if SMOKE:
    FOLD_EDGES = ["2024-06-01", "2024-09-01"]
    START_OOS = "2024-06-01"
    SEED_SETS = [(7,)]
    EPOCHS, PATIENCE_N = 2, 2
    CAP_TRAIN = 50
else:
    from dlv2_chain import FOLD_EDGES, MAX_EPOCHS, PATIENCE, START_OOS
    SEED_SETS = BASE_SEED_SETS
    EPOCHS, PATIENCE_N = MAX_EPOCHS, PATIENCE
    CAP_TRAIN = 0 if USE_GPU else CPU_CAP_PER_DAY


def train_fold_rankloss(cfg, bundle, universe, d0, d1, seeds, cap,
                        shuffle_labels: bool = False, lam: float | None = None):
    """#8 折训练路径的逐字镜像，仅损失加 rank 项（预注册 §一）。

    lam 覆盖 LAMBDA_RANK（阴性对照孪生臂 λ=0 用）。
    shuffle_labels=True（阴性对照专用）：train+val 同源做日内置换，
    早停 val 也用置换后标签——整个标签结构被毁，检验损失实现无泄漏。
    """
    Xtr_raw, mtr, ytr, meta_tr, cols = fold_frame(
        bundle, universe, pd.Timestamp(cfg.data.start), d0, cap)
    cut = d0 - pd.Timedelta(days=LABEL_SPAN * 2)
    keep = np.array([d < cut for d, _ in meta_tr])
    Xtr_raw, mtr, ytr, meta_tr = Xtr_raw[keep], mtr[keep], ytr[keep], \
        [m for m, k in zip(meta_tr, keep) if k]
    Xte_raw, mte, yte, meta_te, _ = fold_frame(bundle, universe, d0, d1, 0)

    edges = fit_bins(np.concatenate([Xtr_raw, np.zeros((1, Xtr_raw.shape[1]))], axis=0))
    Xtr_bin = to_bin_idx(Xtr_raw, edges)
    Xte_bin = to_bin_idx(Xte_raw, edges)
    Xtr_in = np.concatenate([Xtr_raw, mtr], axis=1)
    Xte_in = np.concatenate([Xte_raw, mte], axis=1)
    n_feat = Xtr_in.shape[1]

    # 训练行 → 日组 id（meta 按日期有序 ⇒ 同日块连续）
    # 2026-10-08 B9：逐行 dict.setdefault → pd.factorize（C 级）。两者编码规则同为
    # "按首次出现顺序"，输出**逐位一致**（1200 万行实测 11.4–11.8s → 0.07s，≈160×）。
    # 等价性由 tests/test_dayids_equivalence.py 钉住；实现见 fterm/model/dayids.py。
    day_ids = build_day_ids(d for d, _ in meta_tr)

    y_fit = ytr
    if shuffle_labels:
        rng = np.random.RandomState(LABEL_SHUFFLE_SEED)
        y_fit = ytr.copy()
        i0 = 0
        for n_d in np.bincount(day_ids):
            if n_d > 1:
                y_fit[i0:i0 + n_d] = y_fit[i0:i0 + n_d][rng.permutation(n_d)]
            i0 += int(n_d)

    preds = np.zeros(len(Xte_in), dtype="float64")
    val_cut = int(len(Xtr_raw) * 0.85)
    for seed in seeds:
        torch.manual_seed(seed)
        net = CrossSectionNet(n_feat, n_embed_features=len(cols)).to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
        xt = torch.from_numpy(Xtr_in[:val_cut]).float()
        bt = torch.from_numpy(Xtr_bin[:val_cut]).long()
        yt = torch.from_numpy(y_fit[:val_cut]).float()
        dt = torch.from_numpy(day_ids[:val_cut]).long()
        xv = torch.from_numpy(Xtr_in[val_cut:]).float()
        bv = torch.from_numpy(Xtr_bin[val_cut:]).long()
        yv = torch.from_numpy(y_fit[val_cut:]).float()
        best, best_state, bad = float("inf"), None, 0
        ds = TensorDataset(xt, bt, yt, dt)
        for _ep in range(EPOCHS):
            net.train()
            dl = DataLoader(ds, batch_size=BATCH, shuffle=True)
            for xb, bb, yb, db in dl:
                xb, bb = xb.to(DEVICE), bb.to(DEVICE)
                yb, db = yb.to(DEVICE), db.to(DEVICE)
                opt.zero_grad()
                out = net(xb, bb)
                mse = torch.nn.functional.mse_loss(out, yb)
                # RankNet 相邻对：batch 内按日稳定排序后取相邻元素，
                # 同日且 y 不等者构成对（决定于已种子化的 shuffle ⇒ 确定）
                order = torch.argsort(db, stable=True)
                si, sj = order[:-1], order[1:]
                m = (db[si] == db[sj]) & (yb[si] != yb[sj])
                if bool(m.any()):
                    i_, j_ = si[m], sj[m]
                    sign = torch.sign(yb[i_] - yb[j_])
                    rank = torch.nn.functional.softplus(
                        -(out[i_] - out[j_]) * sign).mean()
                else:
                    rank = out.sum() * 0.0
                loss = mse + (LAMBDA_RANK if lam is None else lam) * rank
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                vl = torch.nn.functional.mse_loss(
                    net(xv.to(DEVICE), bv.to(DEVICE)), yv.to(DEVICE)).item()
            if vl < best - 1e-6:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE_N:
                    break
        if best_state:
            net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            outs = []
            for i in range(0, len(Xte_in), 8192):
                xb = torch.from_numpy(Xte_in[i:i + 8192]).float()
                bb = torch.from_numpy(Xte_bin[i:i + 8192]).long()
                outs.append(net(xb.to(DEVICE), bb.to(DEVICE)).cpu().numpy())
            preds += np.concatenate(outs) / len(seeds)
        print(f"      seed {seed} val {best:.6f}", flush=True)
    rows = [{"date": d, "product": c, "pred": float(p), "y_ret": y}
            for (d, c), p, y in zip(meta_te, preds, yte)]
    return pd.DataFrame(rows)


def run_x0(cfg, bundle, universe) -> dict:
    """X0 仪器位连检查（2026-10-07 erratum 版，预注册 §二）。

    原判据（原版重跑 preds 与缓存逐位一致）实测不可达：GPU 训练路径非确定
    （同码同种背靠背两次 max|Δpred|=2.79e-03）。改为：缓存流折 0 的行集合与
    标签必须与今日 fold_frame 重算逐位一致；特征值层面由 2026-10-07 worktree
    探针（重构前 vs 现行代码逐位一致）覆盖，证据存档证据包。
    """
    seeds = BASE_SEED_SETS[0]
    d0 = pd.Timestamp("2019-01-01")
    d1 = pd.Timestamp("2021-01-01")
    Xg, miss, y, meta, _ = fold_frame(bundle, universe, d0, d1, 0)
    new = pd.DataFrame([{"date": d, "product": c, "y_new": yy}
                        for (d, c), yy in zip(meta, y)])
    cache = REPORTS / f"dlv2_ss{'_'.join(map(str, seeds))}.parquet"
    old = pd.read_parquet(cache)
    old["date"] = pd.to_datetime(old["date"])
    o = old[(old["date"] >= d0) & (old["date"] < d1)][["date", "product", "y_ret"]]
    j = new.merge(o, on=["date", "product"])
    dlab = float((j["y_new"] - j["y_ret"]).abs().max())
    rows_old = set(map(tuple, o[["date", "product"]].itertuples(index=False)))
    rows_new = set(map(tuple, new[["date", "product"]].itertuples(index=False)))
    return {"label_max_abs_diff": dlab, "rows_only_old": len(rows_old - rows_new),
            "rows_only_new": len(rows_new - rows_old), "n_rows": int(len(j)),
            "pass": bool(dlab == 0.0 and not (rows_old - rows_new)
                         and not (rows_new - rows_old)),
            "train_noise_envelope_max_abs_diff": 2.79e-03,
            "erratum": "GPU 训练非确定（BatchNorm/Embedding 反向 atomicAdd）；"
                       "原逐位判据不可达，改位连检查+worktree 探针（预注册 §二 erratum）",
            "cache": cache.name}


def daily_ic(df: pd.DataFrame) -> pd.Series:
    out = {}
    for d, g in df.groupby("date"):
        ic = spearman_rank_ic(list(zip(g["pred"], g["y_ret"])))
        if ic is not None:
            out[d] = ic
    return pd.Series(out)


def paired_t(diff: np.ndarray) -> dict:
    n = len(diff)
    if n < 3 or diff.std(ddof=1) == 0:
        return {"n": int(n), "mean": float(diff.mean()) if n else float("nan"),
                "t": float("nan")}
    return {"n": int(n), "mean": float(diff.mean()),
            "t": float(diff.mean() / (diff.std(ddof=1) / np.sqrt(n)))}


def mean_panel(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    base = None
    for df in dfs:
        p = df.pivot_table(index="date", columns="product", values="pred")
        base = p.copy() if base is None else base.add(p, fill_value=0)
    return base.div(len(dfs)).astype("float32")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mse_stream_path(seeds: tuple) -> Path:
    """MSE 对照流路径；smoke 单种子集映射到首个全量缓存（join 限重叠日期）。"""
    name = "_".join(map(str, BASE_SEED_SETS[0] if SMOKE else seeds))
    return REPORTS / f"dlv2_ss{name}.parquet"


def main():
    cfg = load_config(ROOT / "config.yaml")
    print(f"== #14 rank-loss（device={DEVICE}, smoke={SMOKE}）==", flush=True)
    EXDIR.mkdir(parents=True, exist_ok=True)
    print("== 面板/特征 ==", flush=True)
    panels = sa.load_panels(cfg)
    bundle = build_dataset(cfg)
    universe = bundle.universe
    dates = panels["close"].index
    start_ts = pd.Timestamp(START_OOS)
    edges = [pd.Timestamp(x) for x in FOLD_EDGES]
    if not SMOKE:
        edges = edges + [dates[-1] + pd.Timedelta(days=1)]

    result: dict = {"prereg": "plans/rankloss-prereg-20261007.md", "smoke": SMOKE,
                    "oos_window": {"start": START_OOS, "end": str(dates[-1].date())},
                    "frozen": {"LAMBDA_RANK": LAMBDA_RANK, "BATCH": BATCH,
                               "LR": LR, "WD": WD, "EPOCHS": EPOCHS,
                               "PATIENCE": PATIENCE_N,
                               "label_shuffle_seed": LABEL_SHUFFLE_SEED,
                               "tripwire_ic": TRIPWIRE_IC}}

    # ---- ②X0 确定性自证（smoke 跳过）----
    if not SMOKE:
        print("== X0 仪器位连检查（erratum 版，预注册 §二）==", flush=True)
        t0 = time.time()
        x0 = run_x0(cfg, bundle, universe)
        x0["seconds"] = round(time.time() - t0, 1)
        result["x0"] = x0
        print(f"  X0 位连检查: max|Δy|={x0['label_max_abs_diff']:.3e} "
              f"行差 old={x0['rows_only_old']} new={x0['rows_only_new']} "
              f"(n={x0['n_rows']}) pass={x0['pass']}", flush=True)
        if not x0["pass"]:
            result["status"] = "HALTED_X0"
            _write(result, [])
            sys.exit(2)

    # ---- ③阴性对照（erratum #3：λ1/λ0 孪生，tripwire）----
    print("== 阴性对照（日内 shuffle 标签，λ1/λ0 孪生）==", flush=True)
    d0, d1 = edges[X0_FOLD], edges[X0_FOLD + 1]
    neg1 = train_fold_rankloss(cfg, bundle, universe, d0, d1, (7,), CAP_TRAIN,
                               shuffle_labels=True)
    neg0 = train_fold_rankloss(cfg, bundle, universe, d0, d1, (7,), CAP_TRAIN,
                               shuffle_labels=True, lam=0.0)
    neg1.to_parquet(EXDIR / "negative_control_lam1.parquet")
    neg0.to_parquet(EXDIR / "negative_control_lam0.parquet")
    ic1, ic0 = daily_ic(neg1), daily_ic(neg0)
    common = ic1.index.intersection(ic0.index)
    pt = paired_t((ic1[common] - ic0[common]).to_numpy())
    m1, m0 = float(ic1.mean()), float(ic0.mean())
    trip_ok = bool(abs(m1) < TRIPWIRE_IC and abs(m0) < TRIPWIRE_IC)
    result["negative_control"] = {"lam1_mean_ic": m1, "lam0_mean_ic": m0,
                                  "paired": pt, "tripwire_ic": TRIPWIRE_IC,
                                  "pass": trip_ok,
                                  "note": "λ1-shuffle 读数=结构放大效应（处理性质，"
                                          "登记不判停，预注册 §二 erratum #3）"}
    print(f"  阴性 λ1={m1:+.5f} λ0={m0:+.5f} 配对 t={pt['t']:+.2f}(n={pt['n']}) "
          f"trip(|IC|<{TRIPWIRE_IC})={trip_ok}"
          f"{'（smoke 只报告不判停）' if SMOKE else ''}", flush=True)
    if not trip_ok and not SMOKE:
        result["status"] = "HALTED_NEGATIVE"
        _write(result, [])
        sys.exit(2)

    # ---- ④臂 B 全折训练（双种子集）----
    streams_b = {}
    for si, seeds in enumerate(SEED_SETS):
        tag = f"ss{'_'.join(map(str, seeds))}"
        parts = []
        for k in range(len(edges) - 1):
            d0, d1 = edges[k], edges[k + 1]
            t0 = time.time()
            dfk = train_fold_rankloss(cfg, bundle, universe, d0, d1, seeds, CAP_TRAIN)
            dfk["fold"] = k
            parts.append(dfk)
            print(f"  [{tag}] fold{k} 完成 ({time.time()-t0:.0f}s, {len(dfk)} 行)",
                  flush=True)
        df = pd.concat(parts, ignore_index=True)
        cache = REPORTS / f"rankloss_{tag}.parquet"
        df.to_parquet(cache)
        streams_b[tag] = cache
        print(f"  [{tag}] 流缓存 → {cache.name} ({len(df)} 行)", flush=True)

    # ---- ⑤P1：日频 RankIC 配对（B − MSE，同种子集）----
    print("== P1 日频 RankIC 配对 ==", flush=True)
    per_set, p1_all = {}, True
    for seeds in SEED_SETS:
        tag = f"ss{'_'.join(map(str, seeds))}"
        mse = pd.read_parquet(mse_stream_path(seeds))
        mse["date"] = pd.to_datetime(mse["date"])
        b = pd.read_parquet(streams_b[tag])
        j = b.merge(mse, on=["date", "product"], suffixes=("_b", "_m"))
        ic_b = daily_ic(j[["date", "product", "pred_b", "y_ret_b"]]
                        .rename(columns={"pred_b": "pred", "y_ret_b": "y_ret"}))
        ic_m = daily_ic(j[["date", "product", "pred_m", "y_ret_m"]]
                        .rename(columns={"pred_m": "pred", "y_ret_m": "y_ret"}))
        common = ic_b.index.intersection(ic_m.index)
        pt = paired_t((ic_b[common] - ic_m[common]).to_numpy())
        pt["pass"] = bool(pt["t"] >= C2A_T and pt["mean"] > 0)
        per_set[tag] = {"ic_b_mean": float(ic_b.mean()), "ic_mse_mean": float(ic_m.mean()),
                        "paired": pt, "ic_b_abs": float(ic_b.abs().mean())}
        p1_all &= pt["pass"]
        print(f"  [{tag}] IC B={ic_b.mean():+.5f} MSE={ic_m.mean():+.5f} "
              f"diff t={pt['t']:+.2f}(n={pt['n']}) pass={pt['pass']}", flush=True)
    result["per_set"] = per_set
    result["P1"] = bool(p1_all)

    # ---- ⑥P2：组合层月度对数配对（B 集成 − MSE 集成，同构 #13 组合层）----
    print("== P2 组合层（2 集成均值互配）==", flush=True)
    eqs, metrics = {}, {}
    for name, dfs in (("B", [pd.read_parquet(streams_b[f"ss{'_'.join(map(str, s))}"])
                            for s in SEED_SETS]),
                      ("MSE", [pd.read_parquet(mse_stream_path(s)) for s in SEED_SETS])):
        for df in dfs:
            df["date"] = pd.to_datetime(df["date"])
        eq, m = run_model_arm(-1e9, 10, 10, cfg, mean_panel(dfs), bundle,
                              panels, dates, start_ts)
        eqs[name], metrics[name] = eq, m
        print(f"  {name}: {m['total_return']:+.2%} / MDD {m['max_drawdown']:+.2%} / "
              f"成本 {m['total_costs']:.0f}", flush=True)
    s = pd.Series(dates, index=dates)
    month_ends = [d for d in s.groupby([dates.year, dates.month]).last() if d >= start_ts]
    p_m = monthly_log_paired(eqs["B"], eqs["MSE"], month_ends)
    p2_pass = bool(p_m["t"] >= C2A_T and p_m.get("mean_log", p_m.get("mean")) > 0)
    result["P2"] = {"paired": p_m, "pass": p2_pass}
    print(f"  P2 月度配对 t={p_m['t']:+.2f}(n={p_m['n']}) pass={p2_pass}", flush=True)

    # ---- ⑦透明读数：C1-C4 风格门 vs S1_20w_top2（不设门）----
    if not SMOKE:
        print("== 透明读数 vs S1_20w_top2 ==", flush=True)
        from capital_chain import S1_TIER, _rule_tier_arm
        cfg_s1 = replace(cfg,
                         account=replace(cfg.account, initial_equity=S1_TIER["equity"],
                                         max_symbols=S1_TIER["max_symbols"],
                                         max_lots_per_symbol=S1_TIER["max_lots"]),
                         signal=replace(cfg.signal, top_n=S1_TIER["top_n"]))
        _, m_s1 = _rule_tier_arm(cfg_s1, S1_TIER, bundle, panels, dates, start_ts)
        print(f"  S1_20w_top2: {m_s1['total_return']:+.2%} / "
              f"MDD {m_s1['max_drawdown']:+.2%} / 成本 {m_s1['total_costs']:.0f}",
              flush=True)
        result["transparency_s1_20w_top2"] = {
            "s1": {k: m_s1[k] for k in ("total_return", "max_drawdown", "total_costs")},
            "b_vs_s1": {"ret_gt": bool(metrics["B"]["total_return"] > m_s1["total_return"]),
                        "mdd_ok": bool(metrics["B"]["max_drawdown"]
                                       >= m_s1["max_drawdown"] - 0.02),
                        "cost_ok": bool(metrics["B"]["total_costs"]
                                        <= m_s1["total_costs"] * 1.25)},
            "note": "报告用，不设门（预注册 §三）"}

    result["portfolio"] = {k: {kk: float(vv) for kk, vv in m.items()
                               if isinstance(vv, (int, float, np.floating))}
                           for k, m in metrics.items()}
    result["adoption_ready"] = bool(result["P1"] and result["P2"]["pass"]) if not SMOKE else None
    result["verdict_word"] = ""   # 待用户填写
    _write(result, list(streams_b.values()))
    print("== 完成 ==", flush=True)


def _write(result: dict, stream_paths: list[Path]):
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    (REPORTS / f"rankloss_{stamp}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    lines = [f"判据链 #14 读数 {stamp}（判据冻结于 plans/rankloss-prereg-20261007.md）"]
    if result.get("smoke"):
        lines.append("[SMOKE 模式——读数不作为判据证据]")
    if result.get("x0") is not None:
        lines.append(f"X0 仪器位连检查: {'✓' if result['x0']['pass'] else '✗ 漂移'} "
                     f"(max|Δy|={result['x0']['label_max_abs_diff']:.3e}, "
                     f"行差 old={result['x0']['rows_only_old']} "
                     f"new={result['x0']['rows_only_new']}, n={result['x0']['n_rows']})")
    if result.get("negative_control"):
        nc = result["negative_control"]
        lines.append(f"阴性对照（λ1/λ0 孪生, tripwire<{nc.get('tripwire_ic')}）: "
                     f"λ1={nc['lam1_mean_ic']:+.5f} λ0={nc['lam0_mean_ic']:+.5f} "
                     f"配对 t={nc['paired']['t']:+.2f} → {'✓' if nc['pass'] else '✗ 泄漏'}")
    for tag, ps in result.get("per_set", {}).items():
        pt = ps["paired"]
        lines.append(f"[{tag}] IC B={ps['ic_b_mean']:+.5f} MSE={ps['ic_mse_mean']:+.5f} "
                     f"配对 t={pt['t']:+.2f}(n={pt['n']}) → {pt['pass']}")
    if result.get("portfolio"):
        for k, m in result["portfolio"].items():
            lines.append(f"{k} 集成: {m['total_return']:+.2%} / "
                         f"MDD {m['max_drawdown']:+.2%} / 成本 {m['total_costs']:.0f}")
    if result.get("P2"):
        lines.append(f"P2 组合层月度配对 t={result['P2']['paired']['t']:+.2f}"
                     f"(n={result['P2']['paired']['n']}) → {result['P2']['pass']}")
    if result.get("status"):
        lines.append(f"状态: {result['status']}（预注册 §二：停链归因）")
    elif "adoption_ready" in result:
        lines.append(f"P1={result.get('P1')} P2={result.get('P2', {}).get('pass')} "
                     f"→ adoption_ready={result.get('adoption_ready')}")
    lines.append("判决词：＿＿＿（由用户填写；过线≠自动换版：生产工件换版走 "
                 "train_production 重训+新哈希留痕，由用户拍板）")
    (REPORTS / f"rankloss_{stamp}.txt").write_text("\n".join(lines), encoding="utf-8")
    for sp in stream_paths:
        shutil.copy2(sp, EXDIR / sp.name)
    (EXDIR / "pred_stream_fingerprints.json").write_text(json.dumps(
        {p.name: sha256(p) for p in sorted(EXDIR.glob("*.parquet"))},
        indent=1), encoding="utf-8")
    print(f"报告 → data/reports/rankloss_{stamp}.json/.txt", flush=True)


if __name__ == "__main__":
    main()
