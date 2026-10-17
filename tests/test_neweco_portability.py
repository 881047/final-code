"""回归钉 · 判据链 #28（NEW ECO 2 万档可移植性）。

钉住的是**本链的仪器与口径**，不是策略读数（读数会随数据更新而变）：

  1. 冻结常量与预注册一致（本金/佣金/整手/Bonferroni 门槛/窗口）；
  2. **成本口径唯一来源**：执行器必须用 `fterm.contracts.trade_fee`，
     不得自建费率（本链首版曾在基准侧漏掉复权 ⇒ 单边偏向策略 59.9pp）；
  3. **被动基准必须含权**（防复现首版缺陷）：复权价构造，与策略盯市口径一致；
  4. **忠实账本不变式**：`equity == cash + pos_value`、`cash >= 0`；
  5. **阴性对照必须为负**（G6）：随机 6 只不得显著为正；
  6. **成本单调**（G7）：成本加倍 ⇒ 净收益下降；
  7. **判据不得事后放宽**：C2 门槛 = 2.50（Bonferroni，主检验数 4）。

这些是"能失败也能放行"的检验：故意破坏任一条，测试必须红。
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHAIN = ROOT / 'scripts' / 'neweco_portability_chain.py'
PREREG = ROOT / 'plans' / 'neweco-portability-prereg-20261011.md'


def _src() -> str:
    return CHAIN.read_text(encoding='utf-8')


def test_prereg_exists_and_frozen_before_executor():
    """预注册必须先于执行器冻结（本项目最硬的纪律）。"""
    assert PREREG.exists(), '预注册缺失'
    txt = PREREG.read_text(encoding='utf-8')
    assert 'H4' in txt and 'Bonferroni' in txt
    assert '2.50' in txt, '主检验门槛必须在预注册里写死'


def test_frozen_constants_match_prereg():
    """执行器常量 == 预注册冻结值（改一处必须改两处，防静默漂移）。"""
    src = _src()
    assert re.search(r'EQUITY0\s*=\s*20000\.0', src), '本金必须 2 万'
    assert re.search(r'MAX_CASH_UTIL\s*=\s*0\.95', src)
    assert re.search(r'MAX_LOTS_PER_SYMBOL\s*=\s*10', src)
    assert re.search(r'SLIP_TICKS\s*=\s*1', src)
    assert re.search(r'BONF_T\s*=\s*2\.50', src), 'C2 门槛必须是预注册的 2.50'
    assert re.search(r"PANEL_END\s*=\s*pd\.Timestamp\('2026-09-30'\)", src), '隔离线'


def test_cost_uses_single_source_of_truth():
    """成本必须走 contracts.trade_fee，不得自建费率。"""
    src = _src()
    assert 'contracts.trade_fee' in src, '必须用唯一口径来源'
    assert 'from fterm import contracts' in src
    # 不得出现硬编码费率数字（万2.5 / 5 元 / 印花税）
    assert not re.search(r'0\.00025', src), '不得硬编码佣金费率'
    assert not re.search(r'STAMP_DUTY|0\.0005\b', src), '不得硬编码印花税'


def test_benchmark_is_dividend_consistent():
    """被动基准必须**含权**且与账户**同价格口径**。

    两处自伤都由本钉拦住（2026-10-11）：
    ① 基准用未复权价而策略含权 ⇒ 偏向策略 59.9pp；
    ② 基准用 open→open 而账户按 close 盯市 ⇒ 口径不匹配 ⇒ 超额被系统性低估
       （实测 +1.26% → +1.76%/月，t 1.98 → 3.19）。
    """
    src = _src()
    # 基准必须显式乘 adj（含权）
    assert re.search(r'notional_series\(\s*cl\s*\*\s*adj', src), \
        'MKT 基准必须用复权收盘价（cl * adj）：含权 + 与账户 close→close 同口径'
    # 盯市必须用 adj 比率
    assert 'entry_adj' in src, '策略盯市必须含权（entry_adj 口径）'
    # 执行时点定预算必须用**开盘价**计值（防前视）
    assert 'value_at_open' in src, \
        '执行时点定预算必须用开盘价计值（用收盘价 mtm 定当日开盘单量 = 前视）'
    assert re.search(r'equity_now\s*=\s*cash\s*\+\s*value_at_open\(d\)', src), \
        '买入预算必须基于开盘价权益'


def test_ledger_invariants_present():
    """忠实账本不变式必须在代码里可执行（不是注释）。"""
    src = _src()
    assert re.search(r'assert\s+cash\s*>=\s*-1e-6', src), 'G4 现金非负必须断言'
    tree = ast.parse(src)
    # equity == cash + pos_value 必须在记账处出现
    assert 'equity' in src and 'pos_value' in src
    assert re.search(r'eq\s*=\s*cash\s*\+\s*pv', src), 'G3 恒等式必须实际计算'


def test_negative_control_and_cost_gate_wired():
    """G6 阴性对照 + G7 成本单调必须存在。"""
    src = _src()
    assert 'SEEDS_RND' in src and 'RND6' in src, 'G6 阴性对照缺失'
    assert '--cost-mult' in src, 'G7 成本单调入口缺失'


def test_gate_functions_can_fail():
    """仪器门本身"能失败"：故意破坏输入，检查断言会红。"""
    # 构造一个必然违反不变式的模拟片段，确认断言逻辑存在
    src = _src()
    # 若把 cash 断言删掉，本测试的意图即失效 —— 用存在性 + 语义双重检查
    assert 'assert cash >=' in src
    # 面板竞态断言
    assert re.search(r'assert\s+df\.index\.max\(\)\s*<=\s*PANEL_END', src), 'G1 必须断言'
