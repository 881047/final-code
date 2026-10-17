"""每日运行 CLI：python scripts/run_daily.py [--asof 2026-10-09] [--skip-sync]

收盘后执行完整决策循环：面板同步 → 特征 → 推理 → 信号 → 风控 → 订单（Paper）。
A股现货只做多；实盘通道（券商程序化接口）接入时实现 exec.broker.Broker 即可。
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fterm.config import load_config
from fterm.data.ashare import sync_panels
from fterm.exec.broker import Broker
from fterm.exec.paper import PaperBroker
from fterm.runtime.runner import DailyRunner
from fterm.runtime.state import StateDB


def make_broker(name: str, cfg) -> Broker:
    if name == "paper":
        return PaperBroker(cfg)
    raise ValueError(f"未知 broker: {name}（A股版当前仅 paper；实盘通道实现 Broker 接口后注册）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None, help="特征截止日（默认最新）")
    ap.add_argument("--broker", default="paper", choices=["paper"])
    ap.add_argument("--skip-sync", action="store_true")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "config.yaml")

    if not args.skip_sync:
        print("== 数据同步（老库只读）==")
        for f, msg in sync_panels(cfg).items():
            print(f"{f:12s} {msg}")

    db = StateDB(cfg.runtime.state_db)
    broker = make_broker(args.broker, cfg)
    runner = DailyRunner(cfg, broker, db)
    asof = pd.Timestamp(args.asof) if args.asof else None
    summary = runner.run_once(asof=asof)
    print("== 决策摘要 ==")
    if summary.get("mode") == "dlv2":
        print(f"asof={summary['asof']}  mode=dlv2（双账户 paper 盘面）")
        for tr in summary["tracks"]:
            print(f"  轨道 {tr['track']}: equity={tr['equity']:.0f} "
                  f"持仓市值={tr['pos_value']:.0f} 挂单 {tr['orders']} 笔")
    else:
        print(f"asof={summary['asof']}  equity={summary['equity']:.0f}")
        for d in summary["decisions"]:
            flag = "✓" if d["approved"] else "✗"
            print(f"  {flag} {d['product']:10s} dir={d['direction']:+d} lots={d['lots']}  {d['reason']}")
        for o in summary["orders"]:
            print(f"  订单: {o['product']} {o['side']} {o['lots']}手 [{o['order_id']}]")


if __name__ == "__main__":
    main()
