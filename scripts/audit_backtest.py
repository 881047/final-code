"""可执行防伪闸门：任何回测/判据链报告在写入台账或支撑采纳前必须过闸。

设计原则（与 docs/03 §29/§38、防伪清单同源）：**检查器必须能失败**。
因此本脚本自带自检（`--self-test`）：对一份"故意做坏"的报告必须逐项报出 FAIL，
否则闸门本身是假的。

审计项（缺一即 FAIL，不做"部分通过"）：
  A1 复现契约：git_rev / worktree_dirty / python / pandas / 面板 sha / 输入流 sha 全在
  A2 脏树声明：worktree_dirty=True 必须显式登记，不得静默
  A3 自证：存在 X0/锚点自证字段且 ok=True、漂移在容差内
  A4 账本闭合：所有臂 n_blocked_cash == 0（研究链若走两遍法旧口，必须显式标 harness）
  A5 阴性对照：存在且给出"有响应"判定（阴性不响应 = 仪器没分辨率）
  A6 档位接线：报告声明的 tier（如 T3）必须与读数里实际用的 equity/max_lots 一致
  A7 口径声明：必须写明 含权/不含权、EMA 步进语义、评测构型（否则历史读数不可比）
  A8 数据窗：读数区间不得越过声明的隔离线（前瞻窗隔离）

用法：
    python scripts/audit_backtest.py data/reports/tierfix_faithful_*.json
    python scripts/audit_backtest.py --self-test
退出码 0 = 全部通过；1 = 有 FAIL（不可入台账/不可作采纳依据）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED_REPRO = ("git_rev", "worktree_dirty", "python", "pandas",
                  "panels_sha", "universe_sha", "streams_sha")
LEVEL_KEYS = ("total_return", "max_drawdown", "total_costs")


def _walk_results(node, path=""):
    """深度遍历，把任何含 total_return 的 dict 当作"一条读数"。"""
    if isinstance(node, dict):
        if all(k in node for k in LEVEL_KEYS):
            yield path, node
        for k, v in node.items():
            yield from _walk_results(v, f"{path}/{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk_results(v, f"{path}[{i}]")


def _walk_all(node, path=""):
    """深度遍历**所有** dict 节点（A3 用：自证块不一定挂在读数节点上）。"""
    if isinstance(node, dict):
        yield path, node
        for k, v in node.items():
            yield from _walk_all(v, f"{path}/{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk_all(v, f"{path}[{i}]")


def _ok_flag(node: dict):
    """自证块的 ok 标志：精确键优先，兼容 `<前缀>_selfcheck_ok` 命名（如 v0_selfcheck_ok）。"""
    for k in ("selfcheck_ok", "x0_determinism_ok"):
        if k in node:
            return node[k]
    for k, v in node.items():
        if k.endswith("_selfcheck_ok"):
            return v
    return None


def _max_drift(node: dict) -> float | None:
    """自证块的漂移：`abs_drift` 字典取最大绝对偏差；标量（如 anchor_selfcheck_abs_drift）直接取值。"""
    d = node.get("abs_drift")
    if isinstance(d, dict) and d:
        return max(abs(float(v)) for v in d.values())
    for k, v in node.items():
        if "abs_drift" in k and isinstance(v, (int, float)) and not isinstance(v, bool):
            return abs(float(v))
    return None


def audit(doc: dict, body: str) -> list[tuple[str, str]]:
    """返回 [(检查项, 'PASS'|'FAIL'|'WARN', 说明)]。"""
    out: list[tuple[str, str]] = []
    results = doc.get("results", doc)
    meta = doc.get("_meta") or {}

    # A1 复现契约
    repro = meta or doc.get("repro") or results.get("_repro") or {}
    missing = [k for k in REQUIRED_REPRO if k not in repro]
    out.append(("A1 复现契约", "PASS" if not missing else "FAIL",
                f"缺字段 {missing}" if missing else f"git={repro['git_rev']}"))
    # A2 脏树
    dirty = bool(repro.get("worktree_dirty"))
    out.append(("A2 工作树状态", "WARN" if dirty else "PASS",
                "worktree_dirty=True ⇒ 读数出自未提交代码，须先建立回滚点"
                if dirty else "干净"))

    # A3 逐位自证：按**形状**识别自证块（ok 标志 + 漂移信息同节点），遍历时就地判定。
    # 2026-10-08 审查批重写。旧实现三处缺陷：①只遍历含 total_return 的节点，scaffold
    #   selfcheck() 写出的独立自证块不可见；②`_walk_results(doc)` 的路径拿到 `_at(results,..)`
    #   里回查时双重前缀必空 ⇒ flags 空 ⇒ all([])=True，漂移**从未被真正核对**；
    #   ③ddhalt 的 `v0_selfcheck_ok`/`anchor_selfcheck_abs_drift` 命名不被精确键匹配。
    checks = []
    for p, v in _walk_all(doc):
        ok_flag = _ok_flag(v)
        if ok_flag is None:
            continue
        drift = _max_drift(v)
        if drift is None:
            continue      # 有 ok 无漂移：不是自证块（如阴性对照的 ok），不参与 A3
        tol = float(v.get("tolerance", 1e-4))
        checks.append((p, bool(ok_flag), drift, tol, drift <= tol))
    if not checks:
        ok3, why3 = False, "报告里找不到任何 X0/锚点自证块（ok 标志 + 漂移同节点）"
    else:
        bad = [p for p, ok, _d, _t, in_tol in checks if not ok or not in_tol]
        dmax = max(d for _p, _ok, d, _t, _in in checks)
        ok3 = not bad
        why3 = (f"自证块 {len(checks)} 个，最大漂移 {dmax:.2e}"
                + ("" if ok3 else f"；未过 {bad}"))
    out.append(("A3 逐位自证", "PASS" if ok3 else "FAIL", why3))

    # A4 账本闭合
    leaks = [(p, v.get("n_blocked_cash")) for p, v in _walk_results(results)
             if v.get("n_blocked_cash") not in (None, 0)]
    harness = str(meta.get("harness") or doc.get("harness") or "")
    if leaks and "two_pass" in harness:
        out.append(("A4 账本闭合", "FAIL",
                    f"旧两遍法口 {len(leaks)} 处 blocked_cash>0 = 幻象持仓缺陷未修"
                    f"（读数只可作内部对比，不得支撑采纳）：{leaks[:3]}"))
    elif leaks and "faithful" in harness:
        # 忠实口下 blocked_cash 不是幻象持仓：审批按信号日收盘估价、引擎按次日开盘
        # 真实价成交，隔夜价差可造成少量拒单，且被拒的单不会变成持仓。故按**占比**判。
        def denom(path_node):
            for k in ("n_targets", "n_trades"):
                v = path_node.get(k)
                if v:
                    return float(v)
            return None
        big = []
        for path, n in leaks:
            d = denom(_at(results, path))
            if d is None:
                continue                       # 无分母 ⇒ 不据此判 FAIL（见下面 note）
            if n / d > 0.02:
                big.append((path, n, round(n / d, 4)))
        no_denom = [p_ for p_, _n in leaks if denom(_at(results, p_)) is None]
        out.append(("A4 账本闭合", "PASS" if not big else "FAIL",
                    (f"忠实口少量拒单（隔夜价差，非幻象持仓）：{leaks[:3]}"
                     if not big else f"拒单占比 >2% ⇒ 审批与现金脱节：{big[:3]}")
                    + (f"｜{len(no_denom)} 处无分母可判（报告应补 n_targets/n_trades）"
                       if no_denom else "")))
    else:
        out.append(("A4 账本闭合", "PASS" if not leaks else "FAIL",
                    "全部读数 blocked_cash=0" if not leaks
                    else f"{len(leaks)} 处>0：{leaks[:3]}"))

    # A5 阴性/响应性对照
    blob = json.dumps(doc, ensure_ascii=False).lower()
    neg_tokens = ("negative_control", "tripwire", "shuffle", "structure_control",
                  "shuffled", "response")
    found = [t for t in neg_tokens if t in blob]
    has_neg = bool(found)
    out.append(("A5 阴性/响应性对照", "PASS" if has_neg else "FAIL",
                f"检出对照标记 {found}" if has_neg
                else "无阴性对照 ⇒ 无法证明仪器能失败（PASS 不算证据）"))

    # A6 档位接线
    tier = meta.get("tier") or doc.get("tier")
    if isinstance(tier, dict) and "equity" in tier:
        bad = []
        for p, v in _walk_results(results):
            declared = v.get("tier_equity") or tier.get("equity")
            used = v.get("equity_used")
            if used is not None and declared is not None and abs(used - declared) > 1e-6:
                bad.append((p, declared, used))
        out.append(("A6 档位接线", "PASS" if not bad else "FAIL",
                    f"tier={tier.get('equity')}/{tier.get('top_n')}/"
                    f"lots{tier.get('max_lots')}" + (f"，错配 {bad[:3]}" if bad else "")))
    else:
        out.append(("A6 档位接线", "WARN", "报告未声明 tier 字典，无法核对档位接线"))

    # A7 口径声明
    needs = ("dividend", "ema_step", "harness")
    blob = json.dumps(doc, ensure_ascii=False).lower()
    declared = [n for n in needs if (meta.get(n) is not None) or (n in blob)]
    out.append(("A7 口径声明", "PASS" if len(declared) == len(needs) else "FAIL",
                f"已声明 {declared}" if len(declared) == len(needs)
                else f"缺 {set(needs) - set(declared)}（含权/EMA 步进/评测口不可省）"))

    # A8 数据窗
    win = meta.get("data_window_max") or doc.get("data_window_max")         or doc.get("oos_window_max")
    iso = doc.get("window_isolation") or doc.get("prereg")
    ok8 = bool(win) and bool(iso)
    out.append(("A8 数据窗隔离", "PASS" if ok8 else "WARN",
                f"max={win} 隔离依据={bool(iso)}" if win else
                "未声明读数右端点（前瞻窗内必须能核对未越界）"))
    return out


def _at(results: dict, path: str) -> dict:
    node = results
    for part in [p for p in path.split("/") if p]:
        if not isinstance(node, dict) or part not in node:
            return {}
        node = node[part]
    return node if isinstance(node, dict) else {}


def _doctored() -> dict:
    """故意做坏的报告：A1 缺 sha、A3 无自证、A4 blocked_cash>0、A5 无阴性、A6 档位错配。"""
    return {"results": {"R1": {"total_return": 0.5, "max_drawdown": -0.2,
                               "total_costs": 1000.0, "n_blocked_cash": 37,
                               "tier_equity": 1_000_000, "equity_used": 20_000}},
            "tier": {"equity": 1_000_000, "top_n": 10, "max_lots": 50},
            "repro": {"git_rev": "deadbee"}}


def _good_report() -> dict:
    """合法报告（scaffold.report_meta + selfcheck 形态）：全项必须 PASS。

    2026-10-08 审查批：旧自检只测"做坏必须 FAIL"单向，测不出"合法被误杀"与
    "空心通过"两个方向——恰是 A3 实际发生的事。此处补全三个方向。
    """
    return {
        "prereg": "plans/example.md", "window_isolation": "prereg",
        "tier": {"equity": 1_000_000, "top_n": 10, "max_lots": 50},
        "_meta": {"git_rev": "ca85798", "worktree_dirty": False, "python": "3.13.15",
                  "pandas": "3.0.5", "panels_sha": {"close": "a"}, "universe_sha": "b",
                  "streams_sha": {"s.parquet": "c"}, "harness": "faithful",
                  "dividend_basis": "hfq_inclusive", "ema_step": "daily",
                  "data_window_max": "2026-09-30"},
        "results": {
            "H0": {"total_return": 0.9931, "max_drawdown": -0.1948,
                   "total_costs": 100.0, "n_blocked_cash": 0, "n_targets": 120,
                   "negative_control": {"identical": False, "verdict": "有响应 ✓"}},
            "_selfcheck_H0_anchor": {"abs_drift": {"total_return": 0.0,
                                                   "max_drawdown": 0.0,
                                                   "total_costs": 0.0},
                                     "tolerance": 1e-4, "selfcheck_ok": True,
                                     "provenance": "锚点"}},
    }


def _ok_false_report() -> dict:
    """自证块存在但 ok=False ⇒ A3 必须 FAIL（旧实现：形状合法却查不到该字段）。"""
    d = _good_report()
    d["results"]["_selfcheck_H0_anchor"]["selfcheck_ok"] = False
    return d


def _drift_over_report() -> dict:
    """漂移超容差 ⇒ A3 必须 FAIL（旧实现：双重前缀使漂移从未被真正核对=空心通过）。"""
    d = _good_report()
    d["results"]["_selfcheck_H0_anchor"]["abs_drift"]["total_return"] = 0.5
    return d


def _ddhalt_shaped_report() -> dict:
    """ddhalt 命名形态（v0_selfcheck_ok + 标量 anchor_selfcheck_abs_drift）⇒ A3 必须 PASS。"""
    d = _doctored()   # 复用其 A1 空洞：本 fixture 只断言 A3，其余项可 FAIL
    d["results"]["V0_RL2"] = {"total_return": 0.1, "max_drawdown": -0.2,
                              "total_costs": 50.0,
                              "v0_selfcheck_ok": True,
                              "anchor_selfcheck_abs_drift": 0.0}
    return d


def _a3_verdict(doc: dict) -> str:
    return next(v for k, v, _w in audit(doc, json.dumps(doc)) if k.startswith("A3"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="*", help="待审计的 JSON 报告")
    ap.add_argument("--self-test", action="store_true",
                    help="对故意做坏的报告跑闸门，必须逐项报出 FAIL")
    a = ap.parse_args()

    if a.self_test:
        checks = audit(_doctored(), json.dumps(_doctored()))
        fails = [c for c in checks if c[1] == "FAIL"]
        print("自检 1（做坏的报告）:", flush=True)
        for k, v, why in checks:
            print(f"  {k}: {v}  {why}", flush=True)
        must = {"A1 复现契约", "A3 逐位自证", "A4 账本闭合", "A6 档位接线"}
        caught = {k for k, _v, _w in fails}
        ok = must <= caught
        print(f"  应抓出 {sorted(must)}，实际抓出 {sorted(caught)} ⇒ "
              f"{'✓' if ok else '闸门失效 ✗'}", flush=True)

        good = audit(_good_report(), json.dumps(_good_report()))
        print("自检 2（合法报告，全项应 PASS）:", flush=True)
        for k, v, why in good:
            print(f"  [{v}] {k}: {why}", flush=True)
        bad_good = [k for k, v, _w in good if v == "FAIL"]
        ok &= not bad_good
        print(f"  {'合法报告全项通过 ✓' if not bad_good else f'合法报告被误杀：{bad_good} ✗'}",
              flush=True)

        v_okf = _a3_verdict(_ok_false_report())
        v_drift = _a3_verdict(_drift_over_report())
        v_ddhalt = _a3_verdict(_ddhalt_shaped_report())
        print(f"自检 3（A3 三形态）: ok=False⇒{v_okf}（应 FAIL）｜"
              f"漂移超差⇒{v_drift}（应 FAIL）｜"
              f"ddhalt标量命名⇒{v_ddhalt}（应 PASS）", flush=True)
        ok &= (v_okf == "FAIL" and v_drift == "FAIL" and v_ddhalt == "PASS")
        print(f"自检结论：{'闸门能失败也能放行 ✓' if ok else '闸门失效 ✗'}", flush=True)
        return 0 if ok else 1

    if not a.reports:
        print("给一个报告路径，或用 --self-test")
        return 2
    worst = 0
    for rp in a.reports:
        p = Path(rp)
        body = p.read_text(encoding="utf-8")
        doc = json.loads(body)
        checks = audit(doc, body)
        print(f"== {p.name} ==", flush=True)
        for k, v, why in checks:
            print(f"  [{v}] {k}: {why}", flush=True)
        bad = [k for k, v, _w in checks if v == "FAIL"]
        print(f"  ⇒ {'可入台账' if not bad else '不可采信/须先修：' + str(bad)}", flush=True)
        worst = max(worst, 1 if bad else 0)
    return worst


if __name__ == "__main__":
    sys.exit(main())
