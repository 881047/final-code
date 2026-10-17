"""防伪闸门 scripts/audit_backtest.py 的 A3/自检行为钉（2026-10-08 审查批）。

A3 旧实现三处缺陷（假失败 ×2、假通过 ×1）已重写为按形状识别自证块；
本文件把四种报告形态的预期判定逐位钉住，防止闸门再次静默失效——
闸门自己必须能失败，也必须能放行（scripts/audit_backtest.py --self-test 同款断言）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_backtest import _a3_verdict, audit, main  # noqa: E402

META_OK = {"git_rev": "ca85798", "worktree_dirty": False, "python": "3.13.15",
           "pandas": "3.0.5", "panels_sha": {"close": "a"}, "universe_sha": "b",
           "streams_sha": {"s.parquet": "c"}, "harness": "faithful",
           "dividend_basis": "hfq_inclusive", "ema_step": "daily",
           "data_window_max": "2026-09-30"}


def _report(selfcheck_node: dict | None) -> dict:
    """最小合法报告骨架；selfcheck_node 挂在 results 下（scaffold.selfcheck 同形态）。"""
    results: dict = {
        "H0": {"total_return": 0.99, "max_drawdown": -0.19, "total_costs": 100.0,
               "n_blocked_cash": 0, "n_targets": 120,
               "negative_control": {"identical": False, "verdict": "有响应 ✓"}},
    }
    if selfcheck_node is not None:
        results.update(selfcheck_node)
    return {"prereg": "plans/x.md", "window_isolation": "prereg",
            "tier": {"equity": 1_000_000, "top_n": 10, "max_lots": 50},
            "_meta": META_OK, "results": results}


def _scaffold_block(ok: bool = True, drift: float = 0.0, tol: float = 1e-4) -> dict:
    return {"_selfcheck_H0_anchor": {"abs_drift": {"total_return": drift,
                                                   "max_drawdown": drift,
                                                   "total_costs": drift},
                                     "tolerance": tol, "selfcheck_ok": ok,
                                     "provenance": "锚点"}}


def test_a3_scaffold_selfcheck_block_passes():
    """scaffold.selfcheck() 形态（独立节点，不挂读数）必须被识别并 PASS。"""
    assert _a3_verdict(_report(_scaffold_block(ok=True, drift=0.0))) == "PASS"


def test_a3_attached_x0_block_verified_not_vacuous():
    """tierfix 形态（x0_determinism_ok 挂在读数节点上）：漂移必须真被核对。

    旧实现的双重前缀路径回查使该形态永远空心 PASS——漂移超差必须 FAIL。
    """
    doc = _report(None)
    doc["results"]["X0_anchor"] = {
        "total_return": 0.5, "max_drawdown": -0.2, "total_costs": 10.0,
        "x0_determinism_ok": True,
        "abs_drift": {"total_return": 0.5, "max_drawdown": 0.0, "total_costs": 0.0}}
    assert _a3_verdict(doc) == "FAIL"
    doc["results"]["X0_anchor"]["abs_drift"]["total_return"] = 1e-6
    assert _a3_verdict(doc) == "PASS"


def test_a3_ok_false_fails():
    assert _a3_verdict(_report(_scaffold_block(ok=False))) == "FAIL"


def test_a3_drift_over_tolerance_fails():
    assert _a3_verdict(_report(_scaffold_block(ok=True, drift=0.5))) == "FAIL"


def test_a3_missing_fails():
    assert _a3_verdict(_report(None)) == "FAIL"


def test_a3_ddhalt_scalar_drift_naming_passes():
    """ddhalt 命名形态（v0_selfcheck_ok + 标量 anchor_selfcheck_abs_drift）必须 PASS。"""
    doc = _report(None)
    doc["results"]["V0_RL2"] = {
        "total_return": 0.1, "max_drawdown": -0.2, "total_costs": 50.0,
        "v0_selfcheck_ok": True, "anchor_selfcheck_abs_drift": 0.0}
    assert _a3_verdict(doc) == "PASS"
    doc["results"]["V0_RL2"]["anchor_selfcheck_abs_drift"] = 1e-2
    assert _a3_verdict(doc) == "FAIL"


def test_a3_negative_control_ok_alone_is_not_selfcheck():
    """阴性对照的裸 ok 标志（无漂移）不得冒充自证块，也不得让 A3 空心通过。"""
    doc = _report(None)
    doc["results"]["NEG"] = {"total_return": 0.7, "max_drawdown": -0.3,
                             "total_costs": 90.0, "ok": True}
    assert _a3_verdict(doc) == "FAIL"


def test_a3_no_path_roundtrip_double_prefix():
    """回归钉：结果节点路径在 results 之下回查必须命中（旧实现 results/results 必空）。"""
    doc = _report(_scaffold_block(ok=True, drift=9e-3, tol=1e-2))
    v = _a3_verdict(doc)
    assert v == "PASS"


def test_doctored_report_fails_required_items():
    from audit_backtest import _doctored
    checks = dict((k, v) for k, v, _w in audit(_doctored(), json.dumps(_doctored())))
    for item in ("A1 复现契约", "A3 逐位自证", "A4 账本闭合", "A6 档位接线"):
        assert checks[item] == "FAIL", item


def test_self_test_exit_zero():
    """--self-test 必须 0：闸门能失败（做坏被抓）也能放行（合法全过 + A3 三形态）。"""
    sys.argv = ["audit_backtest.py", "--self-test"]
    assert main() == 0
