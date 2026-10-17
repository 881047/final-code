"""诊断（不设门、不作采纳依据）：同一构型在**逐条流**上的离散度。

动机：#15 重放里 R1（4 流均值）= −39.47% / MDD −62.96% / β 0.482，
而 R3（2 流均值）= +29.43% / MDD −21.61% / β 0.176——同构型只换集成成员就
差出 69 pp。若逐流本身离散巨大，则"选哪几条流"在主导组合层读数，
任何单次配对 t 都只是在抽彩票；这是 #7"构造方差主导"的新落点。
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path("D:/Final")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fterm.config import load_config                      # noqa: E402
from fterm.data import ashare as sa                       # noqa: E402
from fterm.features.build import build_dataset            # noqa: E402

import dlv2_hurdle_chain as HC                            # noqa: E402
import tierfix_replay as TF                               # noqa: E402

STREAMS = {
    "MSE:ss7_21_99_5_13": TF.MSE_STREAMS["ss7_21_99_5_13"],
    "MSE:ss123_2024_42_314_2027": TF.MSE_STREAMS["ss123_2024_42_314_2027"],
    "MSE:ss101_202": TF.MSE_STREAMS["ss101_202"],
    "MSE:ss303_404": TF.MSE_STREAMS["ss303_404"],
    "RL:ss7_21_99_5_13": TF.RL_STREAMS["ss7_21_99_5_13"],
    "RL:ss123_2024_42_314_2027": TF.RL_STREAMS["ss123_2024_42_314_2027"],
}


def main() -> int:
    cfg = load_config(ROOT / "config.yaml")
    panels = sa.load_panels(cfg)
    bundle = build_dataset(cfg, compute_fingerprints=False)
    dates = panels["close"].index
    start_ts = pd.Timestamp(HC.START_OOS)
    out = {}
    for tag, rel in STREAMS.items():
        t0 = time.time()
        df = pd.read_parquet(ROOT / rel)
        df["date"] = pd.to_datetime(df["date"])
        panel = df.pivot_table(index="date", columns="product", values="pred")
        eq, m = HC.run_cema(0, HC.T3, panel, cfg, bundle, panels, dates, start_ts)
        out[tag] = {"total_return": round(float(m["total_return"]), 6),
                    "max_drawdown": round(float(m["max_drawdown"]), 6),
                    "total_costs": round(float(m["total_costs"]), 1),
                    "position_util_mean": round(float(m.get("position_util_mean", 0)), 4),
                    "n_targets": int(m.get("n_targets", 0)),
                    "beta_vs_000905": TF.beta_ir(eq, bench_series(cfg, start_ts))}
        print(f"  {tag}: {m['total_return']:+.2%} MDD {m['max_drawdown']:+.2%} "
              f"成本 {m['total_costs']:.0f} 均仓位 {out[tag]['position_util_mean']:.3f} "
              f"β {out[tag]['beta_vs_000905'].get('beta')} ({time.time()-t0:.0f}s)", flush=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    p = ROOT / "experiments" / "tierfix_20261008" / f"per_stream_dispersion_{stamp}.json"
    p.write_text(json.dumps({"note": "诊断读数，不设门、不作采纳依据", "tier": HC.T3,
                             "hurdle": 0, "results": out},
                            ensure_ascii=False, indent=1), encoding="utf-8")
    rets = [v["total_return"] for v in out.values()]
    print(f"极差: ret {min(rets):+.2%} ~ {max(rets):+.2%} 跨度 {max(rets)-min(rets):.2%}",
          flush=True)
    print(f"→ {p}", flush=True)
    return 0


_BENCH = {}


def bench_series(cfg, start_ts) -> pd.Series:
    if "b" not in _BENCH:
        import sqlite3
        con = sqlite3.connect(f"file:{Path(cfg.data.old_db).as_posix()}?mode=ro", uri=True)
        idx = pd.read_sql_query(
            "SELECT trade_date, close FROM index_daily WHERE ts_code='000905.SH' "
            "AND trade_date >= ? ORDER BY trade_date", con,
            params=[HC.START_OOS.replace("-", "")])
        con.close()
        _BENCH["b"] = pd.Series(idx["close"].to_numpy(dtype="float64"),
                                index=pd.to_datetime(idx["trade_date"].astype(str)))
    return _BENCH["b"]


if __name__ == "__main__":
    sys.exit(main())
