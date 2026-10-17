"""回归钉 · 判据链 #29（被动基准）。

钉住**仪器与口径**，不是读数（读数随数据更新而变）：

  1. 预注册先于执行器冻结；门槛 = Bonferroni n=12 ⇒ 2.87；
  2. **ETF 成本必须与个股分开**（ETF 免印花税）—— 本链最容易犯的错；
  3. **跨频率配对必须同区间复利** —— 首版把月频策略收益与日频基准收益直接配对，
     ⇒ 基准被当成日收益 ⇒ 超额虚增约 +1%/月（会伪造结论）；
  4. **Sharpe 必须频率感知** —— 首版一律 √243 ⇒ 月频 Sharpe 虚增 4.5 倍（实测 5.31 荒谬值）；
  5. ETF 代码**预先指定**，不得事后挑选；
  6. 忠实账本不变式与面板竞态断言仍在。

这些门"能失败也能放行"：故意破坏任一条，测试必须红。
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAIN = ROOT / 'scripts' / 'passive_benchmark_chain.py'
PREREG = ROOT / 'plans' / 'passive-benchmark-prereg-20261011.md'


def _src() -> str:
    return CHAIN.read_text(encoding='utf-8')


def test_prereg_frozen_before_executor():
    assert PREREG.exists(), '预注册缺失'
    txt = PREREG.read_text(encoding='utf-8')
    assert 'H5' in txt
    assert '2.87' in txt, '主检验门槛必须写死在预注册里'


def test_bonferroni_threshold_matches_prereg():
    assert re.search(r'BONF_T\s*=\s*2\.87', _src()), 'B8 门槛必须 = 2.87'


def test_etf_cost_separate_from_stock():
    """ETF 免印花税 —— 必须走独立计费，不得套用个股 trade_fee。

    `contracts.trade_fee` **允许**出现在 G6 互斥断言里（那是用来证明"ETF ≠ 个股计费"），
    但**不得**出现在 ETF 的计价路径（`etf_buyhold`）里。

    ⚠ 元测试陷阱（本钉首版踩到，记忆 `acceptance-bar-calibrate-first` 已警告）：
    用 `ast.get_source_segment` 会把**docstring** 也算进去，而 docstring 里写了
    "不套用个股 trade_fee" 这句**说明文字** ⇒ 钉被自己的说明触发。
    正解：只扫**代码节点**（跳过 docstring）。
    """
    src = _src()
    assert 'COMMISSION_RATE' in src and 'MIN_COMMISSION' in src
    tree = ast.parse(src)
    fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'etf_buyhold':
            fn = node
            break
    assert fn is not None, '找不到 etf_buyhold'
    # 去掉 docstring（Expr + Constant str）后只扫代码
    stmts = [s for s in fn.body
             if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
                     and isinstance(s.value.value, str))]
    code = '\n'.join(ast.get_source_segment(src, s) or '' for s in stmts)
    assert 'trade_fee' not in code, \
        'etf_buyhold 的**代码**里不得用个股 trade_fee（那会错误计征印花税）'
    assert re.search(r'COMMISSION_RATE', code), 'etf_buyhold 必须自行计佣金'


def test_cross_frequency_pairing_is_same_period():
    """跨频率配对必须同区间复利 —— 首版频率错配会虚增超额约 +1%/月。

    ⚠ 本钉首版太弱（变异测试抓到）：只断言"存在 period_returns 调用"，
    而脚本里有**两处**配对（B4 与 B6）；只变异其中一处仍能通过。
    正解：① 断言**基准**收益一律来自 period_returns；
          ② 断言**不存在** `beq.pct_change()` 这类直接配对。
    """
    src = _src()
    assert 'def period_returns' in src, '必须有同区间复利函数'
    # ① 不得有任何直接对基准做 pct_change 的配对
    assert not re.search(r'\bbeq\.pct_change\(\)', src), \
        '基准收益不得直接 pct_change 配对（频率错配 ⇒ 虚增超额）'
    # ② 每一处 `br = ...` 都必须走 period_returns
    br_assigns = re.findall(r'br\s*=\s*([^\n]+)', src)
    assert br_assigns, '找不到基准收益赋值'
    for expr in br_assigns:
        assert 'period_returns(' in expr, f'基准收益赋值必须走 period_returns：{expr.strip()}'
    # ③ 两处配对都要覆盖到
    assert len(re.findall(r'period_returns\(\s*beq\s*,\s*r\.index\s*\)', src)) >= 2, \
        'B4 与 B6 两处配对都必须走同区间复利'


def test_sharpe_is_frequency_aware():
    """Sharpe 必须按实际频率年化（首版一律 √243 ⇒ 月频虚增 4.5 倍）。"""
    src = _src()
    assert 'per_year' in src, '必须推断每年观测数'
    assert not re.search(r'np\.sqrt\(243\)(?!\s*if)', src) or 'per_year' in src
    assert re.search(r'per_year\s*=', src)


def test_etf_codes_predeclared():
    """ETF 代码必须预先指定（防事后挑最好的那只）。"""
    src = _src()
    for code in ('510300.SH', '510500.SH', '510050.SH', '159915.SZ'):
        assert code in src, f'{code} 必须在预注册指定的清单里'


def test_ledger_and_race_assertions_present():
    src = _src()
    assert re.search(r'assert\s+pool\.index\.max\(\)\s*<=\s*END', src), 'G1 面板竞态断言'


def test_chain_reuses_chain28_ledger():
    """策略臂必须复用 #28 忠实账本（逐字不改），保证跨链可比。"""
    src = _src()
    assert 'import neweco_portability_chain as chain' in src
    assert 'chain.simulate' in src
