"""诊断：真 T3 忠实回测里"均仓位只有 0.11–0.23"到底是谁卡的？

候选三个，必须用拒绝原因分布 + 熔断天数占比 + 单手手数分布来分辨，不能靠推断
（§31 我曾断言"熔断后永久降级"，而 halted 会在权益回到峰值 88% 之上时自动解除——
那句话在未取证前不算结论）。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("D:/Final")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fterm.config import load_config                    # noqa: E402
from fterm.data import ashare as sa                     # noqa: E402
from fterm.features.build import build_dataset          # noqa: E402
from fterm.risk.sizer import size_position              # noqa: E402

import tierfix_replay as TF                             # noqa: E402


def main() -> int:
    cfg = load_config(ROOT / "config.yaml")
    panels = sa.load_panels(cfg)
    bundle = build_dataset(cfg, compute_fingerprints=False)
    dates = panels["close"].index
    start_ts = pd.Timestamp(TF.START_OOS)
    mse = TF.load_panel_map(TF.MSE_STREAMS)
    rl = TF.load_panel_map(TF.RL_STREAMS)
    rl2 = TF.mean_panel(list(rl.values()))

    stats: dict = {}
    t0 = time.time()
    eq, m = TF.arm_faithful(rl2, TF.T3, cfg, bundle, panels, dates, start_ts,
                            ema_step="daily", stats=stats)
    print(f"F_RL2_daily 复算: {m['total_return']:+.2%} / MDD {m['max_drawdown']:+.2%} "
          f"均仓位 {m.get('position_util_mean'):.4f} 拒绝 {m['n_governor_rejected']} "
          f"({time.time()-t0:.0f}s)", flush=True)
    print("拒绝原因分布:", stats.get("reject_by_reason"), flush=True)

    # 权益曲线的回撤分布：多少比例的调仓日处于 >12% 回撤（=halted 生效区）
    peak = eq.cummax()
    dd = (eq / peak - 1.0)
    reb = [d for d in eq.index if d.weekday() == cfg.signal.rebalance_weekday]
    dd_reb = dd.loc[[d for d in reb if d in dd.index]]
    print(f"调仓日回撤: 中位 {dd_reb.median():+.3f}  最差 {dd_reb.min():+.3f}  "
          f">12% 回撤占比 {(dd_reb < -0.12).mean():.1%}  >20% 占比 {(dd_reb < -0.20).mean():.1%}",
          flush=True)
    print(f"全程交易日 >12% 回撤占比 {(dd < -0.12).mean():.1%}", flush=True)

    # 单票手数：ATR 风险预算在 T3 档给多大？（看敞口是不是被 sizer 天然卡住）
    feats = bundle.features
    close = panels["close"]
    asof = rl2.index.max()
    row = rl2.loc[asof].dropna().sort_values(ascending=False).head(10)
    print("\n最新日 top10 的 sizer 预算（cfg risk_per_trade=2%, atr_k=1.2, max_lots=50）:",
          flush=True)
    tot = 0.0
    for c in row.index:
        px = float(close.at[asof, c]) if pd.notna(close.at[asof, c]) else float("nan")
        atr = float(feats["atr14"].at[asof, c]) if c in feats["atr14"].columns else float("nan")
        if not np.isfinite(px) or px <= 0:
            continue
        s = size_position(cfg, 1_000_000.0, px, atr if np.isfinite(atr) else 0.0)
        notional = s.lots * 100 * px
        tot += notional
        print(f"  {c} px={px:7.2f} atr={atr:6.3f} atr/px={atr/px:6.3%} "
              f"lots={s.lots:3d} 名义={notional:10,.0f} 占权益={notional/1e6:6.2%} "
              f"method={s.method_used}", flush=True)
    print(f"  top10 名义合计 {tot:,.0f} ⇒ 理论满配仓位 {tot/1e6:.1%}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
