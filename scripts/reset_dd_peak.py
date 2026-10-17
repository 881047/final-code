"""操作者复位回撤峰值锚（docs/03 §36 自锁缺陷的复位阀）。

用法：
    .venv\\Scripts\\python.exe -X utf8 scripts/reset_dd_peak.py --track all ^
        --reason "2026-11-03 人工批准重新入场，锚定当日权益"

不设锚时生产行为与修复前逐位一致（峰值取全历史）。设锚后 `RiskGovernor` 的
`total_dd_halt` 只在该日期之后的权益记录里取峰值 ⇒ 冻结账户得以重新承担敞口。
本脚本**不强平、不改历史行**，只写一个带原因的 kv 并落告警，全程可审计。
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fterm.config import load_config                 # noqa: E402
from fterm.runtime.state import StateDB              # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="all", help="w2 | t3 | all（默认全部纸面轨）")
    ap.add_argument("--date", default=None, help="锚定日 YYYY-MM-DD（默认今天）")
    ap.add_argument("--reason", required=True, help="复位原因（必填，进账本告警与 kv）")
    ap.add_argument("--dry-run", action="store_true", help="只打印前后对比，不写库")
    a = ap.parse_args()
    date = a.date or datetime.now().strftime("%Y-%m-%d")

    cfg = load_config(ROOT / "config.yaml")
    tracks = cfg.raw.get("paper_tracks") or {}
    if not tracks:
        print("config 无 paper_tracks（生产位非 dlv2？）⇒ 无需复位")
        return 1
    names = list(tracks) if a.track == "all" else [a.track]
    for n in names:
        if n not in tracks:
            print(f"未知轨道 {n}，可选 {list(tracks)}")
            return 1
    for n in names:
        db = StateDB(Path(tracks[n]["state_db"]))
        curve = db.equity_curve()
        if not curve:
            print(f"{n}: 账本无权益记录 ⇒ 拒绝锚定（无从计算峰值）")
            continue
        before = max(r[1] for r in curve)
        after_rows = [r[1] for r in curve if r[0] > date]
        after = max(after_rows) if after_rows else curve[-1][1]
        eq_at = next((r[1] for r in reversed(curve) if r[0] <= date), curve[-1][1])
        print(f"{n}: 锚 {date}  峰值 {before:,.0f} → {max(after, eq_at):,.0f} "
              f"（锚后记录 {len(after_rows)} 行，锚日权益 {eq_at:,.0f}）")
        halt = cfg.account.total_dd_halt
        print(f"    解除熔断所需涨幅：现 {(before/max(eq_at,1)-1)*100:.1f}% "
              f"→ 锚后 {max(0.0,(max(after,eq_at)/max(eq_at,1)-1-(halt))*100):.1f}%"
              f"（阈值 −{halt:.0%}）")
        if a.dry_run:
            print("    （dry-run，未写库）")
            continue
        db.set_dd_anchor(date, a.reason, equity=eq_at, track=n)
        db.alert(datetime.now().isoformat(timespec="seconds"), "warn", "dd_peak_reset",
                 f"{n}: 回撤峰值锚复位 → {date}，原因：{a.reason}")
        print(f"    已写入 kv `{StateDB.DD_ANCHOR_KEY}:{n}` 与告警 dd_peak_reset")
    return 0


if __name__ == "__main__":
    sys.exit(main())
