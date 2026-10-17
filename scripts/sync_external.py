"""外部日频因子提取 CLI：python scripts/sync_external.py [--end 20260930]

老库日频原始表 → 本仓 data/external/*.parquet（只读老库，幂等覆盖）。
须先跑 scripts/sync_data.py（提供交易日历与 circ_mv/amount/close 等参照面板）。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fterm.config import load_config          # noqa: E402
from fterm.data.external import sync_external  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default=None, help="截止日 YYYYMMDD（默认面板末日）")
    args = ap.parse_args()
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    print(f"== 外部日频因子提取（只读 {cfg.data.old_db}）==")
    results = sync_external(cfg, end=args.end)
    bad = 0
    for name, msg in sorted(results.items()):
        print(f"  {name:22s} {msg}")
        if msg.startswith("SKIP"):
            bad += 1
    print(f"-- 共 {len(results)} 个因子，{bad} 个为空 --")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
