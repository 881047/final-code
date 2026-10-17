# -*- coding: utf-8 -*-
r"""每日链自动对齐包装器（计划任务 fterm_daily 的入口，run_daily_task.bat 调用）。

A股版（2026-10-03 期货域移除后重写；2026-10-05 增数据就绪闸）：
  ① 先 sync_panels 从老库取数（只读，本地 SQLite，无网络依赖——
     老库自身的更新由 NEW ECO 侧每日链负责，本仓不重复建设）；
  ② 数据就绪闸：老库日链断更时 sync 会"成功地"拉到 T-1（无异常），而幂等闸只防
     "决策超前"不防"数据落后" ⇒ 纸面轨会静默缺一日决策。闸不过 = 响亮退出
     （exit 2，留痕 data/logs/run_daily.log 与任务结果），等老盘日链完成后重跑
     本脚本即可（幂等）；人工回放/复盘走 run_daily.py --asof，不经此闸；
  ③ asof = 面板最新交易日（决策语义 = 该日收盘状态，只用该日及以前数据，无前视）；
  ④ run_daily --asof 该日 --skip-sync（决策写入 INSERT OR REPLACE 幂等）。

用法：
    .venv\\Scripts\\python.exe -X utf8 scripts\\run_daily_auto.py
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from fterm.config import load_config            # noqa: E402
from fterm.data import ashare as sa             # noqa: E402


def main() -> int:
    cfg = load_config(BASE / "config.yaml")

    print("== 面板同步（老库只读）==")
    for f, msg in sa.sync_panels(cfg).items():
        print(f"  {f:12s} {msg}")

    print("== 数据就绪闸 ==")
    ok, why = sa.daily_chain_ready(cfg)
    print(f"  {why}")
    if not ok:
        print("✗ 数据未就绪 ⇒ 本日决策不落账（防静默跑在 T-1 上污染纸面轨）")
        print("  处置：等 NEW ECO 日链完成后重跑本脚本（幂等）；或人工核查老盘日链")
        return 2

    print("== asof 对齐 ==")
    asof = sa.load_panel(cfg, "close").index.max()
    if asof is None:
        print("面板为空，退出")
        return 1
    print(f"  → asof={asof.date()}")

    # 幂等：决策日志严格超前（have > asof）才退出；同日重跑允许（REPLACE 覆盖同日行）。
    # dlv2 双轨（2026-10-06 审查修复）：decisions 按轨落库（w2=主库、t3=独立库），
    # 取两轨 MAX——原实现只查主库（=w2 轨），t3 单独失败的重跑会被错误跳过。
    import sqlite3
    db_paths = [cfg.state_db_path]
    if cfg.signal.mode == "dlv2":
        db_paths += [Path(tr["state_db"]) for tr in cfg.raw["paper_tracks"].values()
                     if Path(tr["state_db"]) != cfg.state_db_path]
    have = None
    for p in db_paths:
        if not p.exists():
            continue
        con = sqlite3.connect(p)
        try:
            row = con.execute("SELECT MAX(date) FROM decisions").fetchone()
        finally:
            con.close()
        if row and row[0] and (have is None or pd.Timestamp(row[0]) > pd.Timestamp(have)):
            have = row[0]
    if have and pd.Timestamp(have) > asof:
        print(f"决策日志已到 {have}，asof={asof.date()} 无新交易日 ⇒ 幂等退出")
        return 0

    print("== run_daily（决策环）==")
    return subprocess.call(
        [sys.executable, "-X", "utf8", str(BASE / "scripts" / "run_daily.py"),
         "--asof", asof.strftime("%Y-%m-%d"), "--skip-sync"],
        cwd=str(BASE))


if __name__ == "__main__":
    sys.exit(main())
