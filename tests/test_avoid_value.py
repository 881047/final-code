"""回归钉 · 判据链 #31（B 判弱资产化 + C 前瞻累积）。

钉住**仪器与口径**，不是读数：

  1. 预注册先于执行器冻结；成本覆盖门槛 = 1.0；
  2. **成本必须按实际单票金额算**（首版硬编码 1400 元 ⇒ 76.6bp，
     而 2 万档 top-2 实际 9500 元 ⇒ 17.1bp，**高估 4.5 倍会把结论判负**）；
  3. **只做减法**：不得产出买入名单（#30 已证判强侧仅捕获 47.8% 天花板）；
  4. **场景诚实性**：必须声明回避名单对"选股"无直接作用，真实场景是 EXIT；
  5. **只读生产**：生产 DB 必须 `mode=ro`；
  6. **幂等**：同 asof 重跑不重复追加；
  7. **无前视**：as-of 不得晚于今天；
  8. 基准 = 当日全市场跑赢率（同 #30）。

变异测试证明这些门能失败。
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAIN = ROOT / 'scripts' / 'avoid_value_chain.py'
PREREG = ROOT / 'plans' / 'avoid-value-prereg-20261011.md'


def _src() -> str:
    return CHAIN.read_text(encoding='utf-8')


def test_prereg_frozen_before_executor():
    assert PREREG.exists(), '预注册缺失'
    txt = PREREG.read_text(encoding='utf-8')
    assert 'H7' in txt and 'H8' in txt
    assert '≥ 1.0' in txt or '1.0' in txt, 'B2 门槛必须写死在预注册里'


def test_cost_uses_actual_notional():
    """成本必须按**实际单票金额**算（首版硬编码 1400 元 ⇒ 高估 4.5 倍）。"""
    src = _src()
    assert 'NOTIONAL_2W_TOP2' in src, '必须定义 2 万档单票预算'
    assert re.search(r'NOTIONAL_2W_TOP2\s*=\s*20000\s*\*\s*0\.95\s*/\s*2', src), \
        '单票预算必须 = initial_equity × max_cash_util / top_n'
    # 不得再出现硬编码 1 手
    assert not re.search(r'lots_per_name\s*[:=]\s*1', src), '不得硬编码 1 手'


def test_cost_uses_contracts():
    """成本口径唯一来源 = fterm/contracts.py。"""
    src = _src()
    assert 'contracts.trade_fee' in src
    assert not re.search(r'0\.00025', src), '不得硬编码佣金费率'


def test_only_subtraction_no_buy_list():
    """只做减法 —— 不得产出买入名单。"""
    src = _src()
    assert '不产出买入名单' in src or '只做减法' in src
    # 不应出现 buy 名单相关输出
    assert 'buy_list' not in src


def test_scenario_honesty_declared():
    """场景诚实性：必须声明回避名单对选股无直接作用、真实场景是 EXIT。"""
    src = _src()
    assert 'EXIT' in src
    assert '不相交' in src or '没有直接作用' in src


def test_production_db_readonly():
    """G1：生产 DB 必须只读打开。

    ⚠ 本钉首版太弱（变异测试抓到）：用 `"mode=ro" in src` 会被**注释或字符串**
    里的 `mode=ro` 满足 ⇒ 把真正的连接改成可写仍能通过。
    正解：用 AST 检查 `sqlite3.connect(...)` 的**实参**里含 mode=ro。
    """
    import ast
    src = _src()
    assert 'PROD_DB' in src
    tree = ast.parse(src)
    calls = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'connect'
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'sqlite3'):
            calls.append(node)
    assert calls, '找不到 sqlite3.connect 调用'
    # 至少一个 connect 必须显式带 mode=ro（且 uri=True）
    found = False
    for c in calls:
        seg = ast.get_source_segment(src, c) or ''
        if 'mode=ro' in seg and 'PROD_DB' in seg:
            found = True
            break
    assert found, '生产 DB 的连接必须显式 mode=ro（AST 实参级检查）'


def test_tracking_is_idempotent():
    """G5：同 asof 不重复追加。"""
    src = _src()
    assert 'avoid_tracking.jsonl' in src
    assert re.search(r"if any\(r\.get\('asof'\)\s*==\s*rec\['asof'\]", src), \
        '必须有同 asof 去重'


def test_no_lookahead_assertion():
    """G3：as-of 不得晚于今天。"""
    src = _src()
    assert re.search(r'if asof > today', src), '必须有前视断言'


def test_baseline_is_market_mean():
    """G7：基准 = 当日截面均值。"""
    src = _src()
    assert re.search(r'ex\s*=\s*yv\s*-\s*yv\.mean\(\)', src), '基准必须为当日截面均值'
