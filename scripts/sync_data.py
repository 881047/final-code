"""数据同步 CLI：python scripts/sync_data.py [--end 20261003]

老库 neweco.db → 本仓面板（data/panel/*.parquet + trade_cal），只读老库。
首次全量同步约 1-3 分钟；此后每日运行链会自动增量（整表覆盖，幂等）。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fterm.config import load_config
from fterm.data.ashare import sync_panels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default=None, help="截止日 YYYYMMDD（默认今天）")
    args = ap.parse_args()
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    print(f"== 老库面板同步（只读 {cfg.data.old_db}，start={cfg.data.start}）==")
    results = sync_panels(cfg, end=args.end)
    fails = 0
    for f, msg in results.items():
        print(f"  {f:12s} {msg}")
        if msg.startswith("FAIL"):
            fails += 1
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
