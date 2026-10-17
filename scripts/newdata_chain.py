"""判据链 · 新数据日频因子筛选（模型升级臂 C，2026-10-09）。

对 `data/external/*.parquet` 的每个日频因子，在 **fterm 自己的面板/标签**上重算
（不照抄老盘读数）：

  1. 逐日截面 Spearman IC（对 fwd_ret）→ 均值 / Newey-West t / ICIR / 正比例
  2. 与现役 25 特征的 **max|ρ|**（逐日截面 Spearman 的绝对值的最大，跨特征取最大）
  3. 制度分层 IC（趋势×波动 2×2）最小层

筛选门槛（沿用 alpha_pool_screen 口径，预注册冻结）：
  |IC| ≥ 0.02 且 |t| ≥ 2 且 max|ρ| vs 现役 ≤ 0.7

用法：python scripts/newdata_chain.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fterm.config import load_config                    # noqa: E402
from fterm.data import ashare as sa                     # noqa: E402
from fterm.data.external import load_external           # noqa: E402
from fterm.features.build import build_dataset          # noqa: E402
from fterm.features.regime import market_regime_labels  # noqa: E402
from fterm.model import stats as st                     # noqa: E402

OUT_DIR = ROOT / "experiments" / f"newdata_{datetime.now().strftime('%Y%m%d')}"
IC_MIN, T_MIN, RHO_MAX = 0.02, 2.0, 0.7


def row_corr(A: np.ndarray, B: np.ndarray, min_n: int = 5) -> np.ndarray:
    """逐行 Pearson（NaN 按有效对处理）。A/B 同形 (T, N)。"""
    valid = np.isfinite(A) & np.isfinite(B)
    n = valid.sum(axis=1)
    A0 = np.where(valid, A, 0.0)
    B0 = np.where(valid, B, 0.0)
    ma = A0.sum(1) / np.maximum(n, 1)
    mb = B0.sum(1) / np.maximum(n, 1)
    da = np.where(valid, A0 - ma[:, None], 0.0)
    db = np.where(valid, B0 - mb[:, None], 0.0)
    cov = (da * db).sum(1)
    va = np.sqrt((da * da).sum(1))
    vb = np.sqrt((db * db).sum(1))
    out = cov / (va * vb + 1e-12)
    out[n < min_n] = np.nan
    return out


def rank_ic_series(fac: pd.DataFrame, ret: pd.DataFrame, mask: pd.DataFrame) -> pd.Series:
    """逐日截面 Spearman IC（rank 后逐行 Pearson）。"""
    f = fac.where(mask).rank(axis=1, pct=True).to_numpy(dtype="float64")
    r = ret.where(mask).rank(axis=1, pct=True).to_numpy(dtype="float64")
    ic = row_corr(f, r)
    return pd.Series(ic, index=fac.index).dropna()


def main():
    cfg = load_config(ROOT / "config.yaml")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("== 载入面板/特征/外部因子 ==", flush=True)
    panels = sa.load_panels(cfg)
    bundle = build_dataset(cfg)
    # ⚠️ 各面板列集不一致（close/amount 5467 vs adj 5471）——统一对齐到 universe.columns
    #    （与 fold_frame 的 _align_panel_cols 同一口径）。
    idx = bundle.universe.index
    codes = bundle.universe.columns
    fwd = bundle.labels["fwd_ret"].reindex(index=idx, columns=codes)
    mask = bundle.universe.astype(bool) & fwd.notna()
    ext = load_external(cfg)
    print(f"外部因子 {len(ext)} 个；现役特征 {len(bundle.feature_cols)} 个；"
          f"面板 {len(idx)}日×{len(codes)}股", flush=True)

    # 现役特征秩（用于 max|ρ|），逐列一次性算
    print("计算现役特征截面秩...", flush=True)
    base_ranks = {}
    for c in bundle.feature_cols:
        panel = bundle.features[c].reindex(index=idx, columns=codes)
        base_ranks[c] = panel.where(mask).rank(axis=1, pct=True).to_numpy(dtype="float64")

    reg = market_regime_labels(cfg).dropna()
    ret_rank = fwd.where(mask).rank(axis=1, pct=True).to_numpy(dtype="float64")

    rows = {}
    for name, fac in ext.items():
        fac = fac.reindex(index=idx, columns=codes)
        fr = fac.where(mask).rank(axis=1, pct=True).to_numpy(dtype="float64")
        ic = pd.Series(row_corr(fr, ret_rank), index=fac.index).dropna()
        s = st.summarize_ic(ic.to_numpy())
        # max|ρ| vs 现役特征（逐日截面 Spearman，取绝对值的时间均值后再取最大）
        max_rho = 0.0
        for c, br in base_ranks.items():
            rho = row_corr(fr, br)
            m = np.nanmean(np.abs(rho))
            if np.isfinite(m):
                max_rho = max(max_rho, float(m))
        common = ic.index.intersection(reg.index)
        strat = st.regime_stratified_ic(ic.loc[common].to_numpy(), reg.loc[common].to_numpy())
        rows[name] = {
            "ic": s["mean"], "t": s["t"], "icir": s["icir"], "pos_frac": s["pos_frac"],
            "n_days": s["n"], "max_abs_rho_base": max_rho,
            "regime_min_ic": st.regime_min_ic(strat),
            "pass_ic": bool(s["mean"] is not None and abs(s["mean"]) >= IC_MIN),
            "pass_t": bool(s["t"] is not None and abs(s["t"]) >= T_MIN),
            "pass_rho": bool(max_rho <= RHO_MAX),
        }
        rows[name]["survivor"] = bool(rows[name]["pass_ic"] and rows[name]["pass_t"]
                                      and rows[name]["pass_rho"])

    surv = sorted([k for k, v in rows.items() if v["survivor"]],
                  key=lambda k: -abs(rows[k]["ic"]))
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out = {"generated_at": stamp, "gate": {"ic": IC_MIN, "t": T_MIN, "rho": RHO_MAX},
           "n_factors": len(rows), "n_survivors": len(surv), "survivors": surv, "detail": rows}
    (OUT_DIR / f"screen_{stamp}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    def fm(x, p=4):
        return f"{x:+.{p}f}" if isinstance(x, (int, float)) else "n/a"

    lines = [f"新数据日频因子筛选 {stamp}（门槛 |IC|≥{IC_MIN} & |t|≥{T_MIN} & max|ρ|≤{RHO_MAX}）",
             f"共 {len(rows)} 个因子，幸存 {len(surv)} 个：{surv}"]
    for k in sorted(rows, key=lambda x: -abs(rows[x]["ic"] or 0)):
        v = rows[k]
        flag = "★" if v["survivor"] else " "
        lines.append(f" {flag} {k:22s} IC {fm(v['ic'])} t {fm(v['t'], 2)} ICIR {fm(v['icir'], 3)} "
                     f"maxρ {v['max_abs_rho_base']:.2f} regime最小IC {fm(v['regime_min_ic'])}")
    text = "\n".join(lines)
    (OUT_DIR / f"screen_{stamp}.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f"报告 → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
