"""老库 index_daily → 本仓 data/index/*.parquet（只读老库，L1 契约同面板）。

指数序列是"市场状态/基准超额"两件事的唯一数据源（docs/03 §33 的 β/残差 IR、
#20 的前瞻敞口规则）。缺失时**直接报错**，不静默退回无基准口径。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fterm.config import load_config        # noqa: E402

CODES = ("000905.SH", "000906.SH", "000300.SH", "399001.SZ", "H00906.CSI", "000001.SH")


def index_dir(cfg) -> Path:
    d = Path(cfg.data.data_dir) / "index"
    d.mkdir(parents=True, exist_ok=True)
    return d


def sync(cfg, start: str = "20130101") -> dict[str, str]:
    import sqlite3
    out = {}
    con = sqlite3.connect(f"file:{Path(cfg.data.old_db).as_posix()}?mode=ro", uri=True)
    for code in CODES:
        df = pd.read_sql_query(
            "SELECT trade_date, open, high, low, close, vol, amount FROM index_daily "
            "WHERE ts_code = ? AND trade_date >= ? ORDER BY trade_date",
            con, params=[code, start])
        if df.empty:
            out[code] = "FAIL 老库无该指数数据"
            continue
        df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str))
        df = df.set_index("trade_date").astype("float64").sort_index()
        df.to_parquet(index_dir(cfg) / f"{code}.parquet")
        out[code] = f"ok rows={len(df)} {df.index.min().date()}→{df.index.max().date()}"
    con.close()
    return out


def load(cfg, code: str = "000905.SH") -> pd.DataFrame:
    p = index_dir(cfg) / f"{code}.parquet"
    assert p.exists(), f"指数面板不存在: {p}（先跑 scripts/sync_index.py）"
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


if __name__ == "__main__":
    cfg = load_config(ROOT / "config.yaml")
    r = sync(cfg)
    bad = 0
    for k, v in r.items():
        print(f"  {k:12s} {v}")
        bad += v.startswith("FAIL")
    sys.exit(1 if bad else 0)
