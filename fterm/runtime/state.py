"""SQLite 状态库 —— 权益曲线 / 成交 / 决策审计 / 心跳 / 告警。

单文件库（runtime.state_db），全部幂等写入，看板只读。
A股口径：无保证金，权益 = 现金 + 持仓市值（pos_value）。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS equity_curve (
    date TEXT PRIMARY KEY, equity REAL, pos_value REAL, cash REAL);
CREATE TABLE IF NOT EXISTS fills (
    ts TEXT, product TEXT, side TEXT, lots INTEGER,
    price REAL, fee REAL, order_id TEXT, PRIMARY KEY (ts, product, side, lots));
CREATE TABLE IF NOT EXISTS decisions (
    date TEXT, product TEXT, direction INTEGER, lots INTEGER, approved INTEGER,
    reason TEXT, PRIMARY KEY (date, product));
CREATE TABLE IF NOT EXISTS heartbeat (
    id INTEGER PRIMARY KEY CHECK (id = 1), ts TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS alerts (
    ts TEXT, level TEXT, code TEXT, message TEXT, PRIMARY KEY (ts, code));
CREATE TABLE IF NOT EXISTS ema_state (
    k TEXT PRIMARY KEY, val TEXT);
CREATE TABLE IF NOT EXISTS positions (
    code TEXT PRIMARY KEY, lots INTEGER, entry_price REAL, adj_entry REAL);
CREATE TABLE IF NOT EXISTS pending_orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT, code TEXT, side TEXT,
    lots INTEGER, reason TEXT, created TEXT);
"""


def fill_cash_delta(side: str, lots: int, price: float, fee: float) -> float:
    """成交对现金的影响 —— 与 `PaperBroker._fill` 严格同式。

    买：−(成交价×股数 + 费用)；卖：+成交价×股数 − 费用。
    `price` 是含滑点的成交价、`fee` 是 trade_cost 的全费用，二者都已落 fills 表，
    故同日重跑可由审计日志精确重放现金，而不依赖撮合是否重演。
    """
    from .. import contracts

    shares = contracts.shares_of(lots)
    return -(price * shares + fee) if side == "buy" else price * shares - fee


class StateDB:
    def __init__(self, path: str | Path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False：看板 API 在 FastAPI 线程池里只读访问。
        # 写入侧（runner/watchdog）约定单线程使用。
        self.conn = sqlite3.connect(p, check_same_thread=False)
        # WAL + busy_timeout：看板线程读与 runner 写并发时防 "database is locked"
        # （2026-10-06 审查修复；WAL 对该库文件持久生效，幂等）。
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._migrate_futures_legacy()
        # pending_orders 补 created 列（2026-10-05 同日重跑幂等 + 双轨隔离审计所需）；
        # 期货 legacy 主库的旧表无此列 ⇒ 就地 ADD COLUMN（旧行 created=NULL）。
        # positions 补 adj_entry 列（含权入账，2026-10-08 docs/03 §38）：旧库只有三列 ⇒
        # 就地 ADD COLUMN。本仓纸面轨当前零持仓 ⇒ 迁移无历史成本；若有持仓，缺列值按
        # 1.0 处理并在 runner 侧留告警（绝不猜因子）。
        pos_cols = {r[1] for r in self.conn.execute("PRAGMA table_info(positions)").fetchall()}
        if pos_cols and "adj_entry" not in pos_cols:
            self.conn.execute("ALTER TABLE positions ADD COLUMN adj_entry REAL")
        pend_cols = {r[1] for r in self.conn.execute("PRAGMA table_info(pending_orders)").fetchall()}
        if pend_cols and "created" not in pend_cols:
            self.conn.execute("ALTER TABLE pending_orders ADD COLUMN created TEXT")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def _migrate_futures_legacy(self) -> None:
        """期货版 schema 就地改名留档（不删任何历史行）。

        期货→A股改造（docs/03 §14）：fills 表去掉 contract 列、equity_curve 的
        margin_used 改 pos_value。检测到旧列时把旧表改名保留（历史审计留痕），
        再由 _SCHEMA 建新表。看板进程占用文件时同样可执行（SQLite 支持并发）。
        """
        fills_cols = {r[1] for r in self.conn.execute("PRAGMA table_info(fills)").fetchall()}
        if fills_cols and "contract" in fills_cols:
            self.conn.execute("ALTER TABLE fills RENAME TO fills_legacy_futures")
        eq_cols = {r[1] for r in self.conn.execute("PRAGMA table_info(equity_curve)").fetchall()}
        if eq_cols and "margin_used" in eq_cols:
            self.conn.execute("ALTER TABLE equity_curve RENAME TO equity_curve_legacy_futures")
        self.conn.commit()

    # ---- 写入 ----

    def record_equity(self, date: str, equity: float, pos_value: float, cash: float) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO equity_curve VALUES (?,?,?,?)", (date, equity, pos_value, cash))
        self.conn.commit()

    def record_fill(self, ts: str, product: str, side: str, lots: int,
                    price: float, fee: float, order_id: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO fills VALUES (?,?,?,?,?,?,?)",
            (ts, product, side, lots, price, fee, order_id))
        self.conn.commit()

    def record_decisions(self, date: str, decisions: list[tuple[str, int, int, bool, str]]) -> None:
        """(product, direction, lots, approved, reason)。"""
        self.conn.executemany(
            "INSERT OR REPLACE INTO decisions VALUES (?,?,?,?,?,?)",
            [(date, p, d, l, int(a), r) for p, d, l, a, r in decisions])
        self.conn.commit()

    def heartbeat(self, ts: str, note: str = "") -> None:
        self.conn.execute("INSERT OR REPLACE INTO heartbeat VALUES (1, ?, ?)", (ts, note))
        self.conn.commit()

    def alert(self, ts: str, level: str, code: str, message: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO alerts VALUES (?,?,?,?)",
                          (ts, level, code, message))
        self.conn.commit()

    # ---- 读取（看板/验证用）----

    def equity_curve(self) -> list[tuple]:
        return self.conn.execute("SELECT date, equity, pos_value, cash FROM equity_curve ORDER BY date").fetchall()

    def recent_fills(self, n: int = 50) -> list[tuple]:
        return self.conn.execute(
            "SELECT ts, product, side, lots, price, fee FROM fills ORDER BY ts DESC LIMIT ?", (n,)).fetchall()

    def recent_alerts(self, n: int = 20) -> list[tuple]:
        return self.conn.execute(
            "SELECT ts, level, code, message FROM alerts ORDER BY ts DESC LIMIT ?", (n,)).fetchall()

    def last_heartbeat(self) -> tuple | None:
        return self.conn.execute("SELECT ts, note FROM heartbeat WHERE id=1").fetchone()

    def get_kv(self, k: str) -> str | None:
        row = self.conn.execute("SELECT val FROM ema_state WHERE k=?", (k,)).fetchone()
        return row[0] if row else None

    def set_kv(self, k: str, val: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO ema_state VALUES (?,?)", (k, val))
        self.conn.commit()

    # ---- 回撤峰值锚（复位闸；2026-10-08 落地阻断项修复）----
    # 熔断 `equity < peak×(1−total_dd_halt)` 与"halted 只拒新开、不强平"组合起来会
    # 自锁：熔断 ⇒ 存量靠退出带自然清仓 ⇒ 账户变成纯现金 ⇒ 权益冻结 ⇒ 永远涨不回
    # 峰值的 88% ⇒ 熔断永不解除。实测真 T3 忠实回测里该状态只触发 2 次，却持续
    # 中位 946 个交易日（解除需权益上涨 25.3%，而阈值只要求回到 −12%）。
    # 原 governor 文档写"人工确认才重置"，但全仓没有任何复位入口（只能手改 SQLite）。
    # 这里提供**显式、留痕、默认不生效**的复位锚：不设锚 ⇒ 峰值仍取全历史（行为不变）。
    DD_ANCHOR_KEY = "dd_peak_anchor"

    def get_dd_anchor(self, track: str | None = None) -> dict | None:
        """读取复位锚。多轨共用一张 kv 表时用 `track:key` 区分。"""
        raw = self.get_kv(f"{self.DD_ANCHOR_KEY}:{track}" if track else self.DD_ANCHOR_KEY)
        if not raw:
            return None
        import json as _json
        try:
            return _json.loads(raw)
        except ValueError:
            return {"date": raw}      # 兼容裸日期串写法

    def set_dd_anchor(self, date: str, reason: str, equity: float | None = None,
                      track: str | None = None) -> None:
        import json as _json
        key = f"{self.DD_ANCHOR_KEY}:{track}" if track else self.DD_ANCHOR_KEY
        self.set_kv(key, _json.dumps({"date": date, "reason": reason,
                                      "equity_at_anchor": equity}, ensure_ascii=False))

    def load_positions(self) -> dict[str, dict]:
        rows = self.conn.execute(
            "SELECT code, lots, entry_price, adj_entry FROM positions").fetchall()
        return {c: {"lots": l, "entry_price": ep,
                    "adj_entry": (1.0 if adj is None or adj <= 0 else float(adj)),
                    "adj_missing": bool(adj is None or adj <= 0)}
                for c, l, ep, adj in rows}

    # ---- 现金恢复（2026-10-08 P0-A 修复）----
    def day_start_cash(self, before_date: str) -> float | None:
        """日初现金 = `date < before_date` 的最后一行 equity_curve.cash。

        原实现每日用 `tr["equity"]`（初始本金）重建 PaperBroker，而持仓从库里恢复
        ⇒ 现金与持仓双双计入，纸面权益自持仓成交次日起虚增约一个持仓市值
        （t3 实测口径 ≈1.95M / 1.0M 本金），60 日前瞻窗主判据失真。
        注：equity_curve.cash 写的一直是「当日撮合后现金」，正是次日的日初值。
        """
        row = self.conn.execute(
            "SELECT cash FROM equity_curve WHERE date < ? ORDER BY date DESC LIMIT 1",
            (before_date,)).fetchone()
        return float(row[0]) if row and row[0] is not None else None

    def fills_on(self, date: str) -> list[tuple]:
        """某日已记录的成交（价格/费用/手数），供同日重跑重放现金。"""
        return self.conn.execute(
            "SELECT product, side, lots, price, fee FROM fills WHERE ts = ?", (date,)).fetchall()

    def has_fill(self, date: str, product: str, side: str, lots: int) -> bool:
        """该 (日,票,方向,手数) 是否已撮合入库（防部分崩溃后重跑双成交）。"""
        return self.conn.execute(
            "SELECT 1 FROM fills WHERE ts = ? AND product = ? AND side = ? AND lots = ? LIMIT 1",
            (date, product, side, lots)).fetchone() is not None

    def save_positions(self, positions: dict[str, dict]) -> None:
        self.conn.execute("DELETE FROM positions")
        self.conn.executemany("INSERT OR REPLACE INTO positions VALUES (?,?,?,?)",
                              [(c, p["lots"], p["entry_price"], p.get("adj_entry", 1.0))
                               for c, p in positions.items()])
        self.conn.commit()

    def load_pending(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, code, side, lots, reason, created FROM pending_orders ORDER BY id").fetchall()
        return [{"id": i, "code": c, "side": s, "lots": l, "reason": r, "created": cr}
                for i, c, s, l, r, cr in rows]

    def save_pending(self, orders: list[dict]) -> None:
        self.conn.execute("DELETE FROM pending_orders")
        self.conn.executemany(
            "INSERT INTO pending_orders (code, side, lots, reason, created) VALUES (?,?,?,?,?)",
            [(o["code"], o["side"], o["lots"], o["reason"], o.get("created")) for o in orders])
        self.conn.commit()

    def stats(self) -> dict:
        def one(sql):
            return self.conn.execute(sql).fetchone()[0]
        return {
            "days": one("SELECT COUNT(*) FROM equity_curve"),
            "fills": one("SELECT COUNT(*) FROM fills"),
            "decisions": one("SELECT COUNT(*) FROM decisions"),
            "alerts": one("SELECT COUNT(*) FROM alerts"),
        }
