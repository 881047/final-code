"""统一报告元数据脚手架（防伪闸门 scripts/audit_backtest.py 的配套写入侧）。

规矩：链脚本出数时必须调 `report_meta()`，把复现契约、评测口（harness）、
含权口径、EMA 语义、数据窗右端点、自证块一起写进报告——
闸门查的就是这些字段，缺一项报告就不可采信。
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("D:/Final")


def content_sha(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(df.index.astype("int64").to_numpy().tobytes())
    h.update("|".join(map(str, df.columns)).encode())
    h.update(df.to_numpy(dtype="float64", na_value=np.nan).tobytes())
    return h.hexdigest()[:16]


def file_sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16] if Path(p).exists() else None


def report_meta(cfg, panels: dict, bundle, *, harness: str, dividend_basis: str,
                ema_step: str, stream_paths: list[str], window_max: pd.Timestamp,
                tier: dict, extra: dict | None = None) -> dict:
    """链报告的统一 `_meta`。

    harness: faithful（引擎循环内问决策、账本闭合）| two_pass（旧口，已知幻象持仓）
    dividend_basis: hfq_inclusive（含权）| unadjusted（不含权，读数系统性下偏）
    """
    try:
        # 脏树判定只看**代码/文档/配置**路径——实验输出（experiments/*.parquet、
        # data/reports/*.json）在运行时必然未跟踪/被改写，把它们算进脏树会让
        # A2 永远 WARN，失去"读数出自未提交代码"的信号价值（2026-10-08 审查批）。
        code_paths = ["fterm", "scripts", "tests", "docs", "plans", "config.yaml"]
        rev = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain", "--", *code_paths],
            capture_output=True, text=True).stdout.strip())
        dirty_any = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                                        capture_output=True, text=True).stdout.strip())
    except Exception as e:                                     # 取证失败必须留痕
        rev, dirty, dirty_any = f"unavailable:{e}", True, True
    import torch
    m = {"git_rev": rev, "worktree_dirty": dirty, "worktree_dirty_scope": "code",
         "worktree_dirty_any": dirty_any,
         "python": sys.version.split()[0], "pandas": pd.__version__,
         "numpy": np.__version__, "torch": torch.__version__,
         "panels_sha": {f: content_sha(panels[f]) for f in
                        ("close", "open", "high", "low", "up_limit", "down_limit", "adj")},
         "universe_sha": content_sha(bundle.universe),
         "atr14_sha": content_sha(bundle.features["atr14"]),
         "streams_sha": {Path(p).name: file_sha(ROOT / p) for p in stream_paths},
         "harness": harness, "dividend_basis": dividend_basis, "ema_step": ema_step,
         "data_window_max": str(pd.Timestamp(window_max).date()),
         "tier": dict(tier), "generated_at": datetime.now().isoformat(timespec="seconds")}
    m.update(extra or {})
    return m


def selfcheck(name: str, drift: dict, tolerance: float, ok: bool,
              provenance: str) -> dict:
    """统一自证块（闸门 A3 查的就是它）。drift 的键必须是各读数项的绝对偏差。"""
    return {name: {"abs_drift": {k: float(v) for k, v in drift.items()},
                   "tolerance": float(tolerance), "selfcheck_ok": bool(ok),
                   "provenance": provenance}}
