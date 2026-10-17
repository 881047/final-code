"""进程外看门狗：探活 runner 心跳，超时告警（写库）。

⚠️ 未接线（2026-10-06 如实登记）：docstring 曾引用的 scripts/watchdog_loop.py
从未实现，本模块目前**无任何调用方**，check_once 仅可作手动探活入口。
现行保障 = 计划任务 fterm_daily 的任务结果 + run_daily_auto 数据就绪闸
exit 2 留痕（data/logs/run_daily.log）。若要启用，另写拉起脚本并注册计划任务。
心跳 3×heartbeat_minutes 未更新 → CRITICAL 告警；同 code 6 小时内去重。
"""

from __future__ import annotations

from datetime import datetime, timedelta

from ..config import Config
from .state import StateDB


def check_once(cfg: Config, now: datetime | None = None) -> list[str]:
    db = StateDB(cfg.runtime.state_db)
    now = now or datetime.now()
    hb = db.last_heartbeat()
    issues: list[str] = []
    if hb is None:
        issues.append("heartbeat: 从未心跳（runner 从未跑或库损坏）")
    else:
        ts = datetime.fromisoformat(hb[0])
        stale_min = (now - ts).total_seconds() / 60
        if stale_min > 3 * cfg.runtime.heartbeat_minutes:
            issues.append(f"heartbeat: 已 {stale_min:.0f} 分钟无心跳（阈值 {3 * cfg.runtime.heartbeat_minutes}）")
    for msg in issues:
        db.alert(now.isoformat(timespec="seconds"), "CRITICAL", "watchdog", msg)
    if not issues:
        db.alert(now.isoformat(timespec="seconds"), "INFO", "watchdog_ok", "探活正常")
    return issues
