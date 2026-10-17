"""回归钉 · 判据链 #30（强弱预测报告 + 纸面跟踪）。

钉住**仪器与口径**，不是读数：

  1. 预注册先于执行器冻结；Bonferroni 门槛 = 2.64；
  2. **"三不"声明必须在报告里**（预注册 §四 P6）—— 这是本链的存在理由；
  3. **基准 = 当日全市场跑赢率**（不是 0.5、不是 0.7）—— 标定批首版写错 0.7；
  4. **判弱口径** = `base_win − bot_win`（跑输率之差）—— 首版写成
     `bot_win − (1−base_win)` ⇒ 符号相反（−0.135 vs 真值 +0.033）；
  5. **分档均值按档位在时间上取均值** ⇒ shape (10,) —— 首版误取每日档间均值
     ⇒ 打印出 1871 个数；
  6. **P1 用汇总 ρ**（预注册本意），逐日 ρ 单列 —— 两者是不同量（0.98 vs 0.066）；
  7. **不输出名次**（只给档）—— 档内 IC 为负是实测事实；
  8. 面板竞态断言仍在。

变异测试证明这些门能失败。
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAIN = ROOT / 'scripts' / 'strength_report.py'
PREREG = ROOT / 'plans' / 'strength-report-prereg-20261011.md'


def _src() -> str:
    return CHAIN.read_text(encoding='utf-8')


def test_prereg_frozen_before_executor():
    assert PREREG.exists(), '预注册缺失'
    txt = PREREG.read_text(encoding='utf-8')
    assert 'H6' in txt
    assert '2.64' in txt, 'P8 门槛必须写死在预注册里'
    # 三不必须在预注册里
    for k in ('不预测绝对涨跌幅', '不预测大盘方向', '不输出'):
        assert k in txt, f'预注册缺少"三不"之一：{k}'


def test_bonferroni_threshold_matches_prereg():
    assert re.search(r'BONF_T\s*=\s*2\.64', _src()), 'P8 门槛必须 = 2.64'


def test_three_no_declaration_present():
    """P6：报告必须含"三不"声明 —— 这是本链的存在理由。"""
    src = _src()
    assert 'THREE_NO' in src
    for k in ('不预测绝对涨跌幅', '不预测大盘方向', '不输出'):
        assert k in src, f'缺少"三不"之一：{k}'
    # 必须真的被打印出来（不只是定义）
    assert re.search(r'for i, s in enumerate\(THREE_NO', src) or 'THREE_NO)' in src


def test_baseline_is_market_win_rate():
    """G5：基准必须是**当日全市场跑赢率**。

    标定批首版把基准写成 0.7（30/30 分组的随机命中率应为 0.50）⇒ 结论方向都错。
    """
    src = _src()
    assert re.search(r"base_win'\s*:\s*base_win", src) or "base_win = " in src, \
        '必须有 base_win（当日全市场跑赢率）'
    assert re.search(r"base_win\s*=\s*float\(\(ex\s*>\s*0\)\.mean\(\)\)", src), \
        'base_win 必须 = 当日 excess>0 的比例'


def test_weak_side_sign_convention():
    """判弱口径：base_win − bot_win（首版符号写反，实测 −0.135 vs 真值 +0.033）。"""
    src = _src()
    assert re.search(r"s_bot\s*=\s*\(D\['base_win'\]\s*-\s*D\['bot_win'\]\)", src), \
        '判弱必须是 base_win − bot_win'


def test_decile_mean_axis():
    """分档均值必须按**档位**聚合 ⇒ shape (10,)。"""
    src = _src()
    assert re.search(r'np\.nanmean\(dec_mat,\s*axis=0\)', src), \
        '分档均值必须 vstack 后按档位取均值（axis=0）'
    assert 'dec_mat = np.vstack' in src


def test_p1_uses_pooled_rho():
    """P1 判据用**汇总 ρ**；逐日 ρ 单列（两者是不同量：0.98 vs 0.066）。"""
    src = _src()
    assert 'p1_pooled_rho' in src, '必须有汇总 ρ'
    assert 'p1_daily_rho' in src, '逐日 ρ 必须单列（参考）'
    assert re.search(r'spearmanr\(np\.arange\(N_DECILE\),\s*decile_mean\)', src), \
        '汇总 ρ 必须在 10 个档均值上算'


def test_no_rank_output():
    """不得输出名次（只给档）—— 档内 IC 为负是实测事实。"""
    src = _src()
    assert '不输出名次' in src or '不提供名次' in src
    # 单日报告不得打印"第 N 名"式的个股排序
    assert 'head(' not in src or 'ranked.head' not in src


def test_panel_race_assertion():
    src = _src()
    assert re.search(r'assert\s+dates\.max\(\)\s*<=\s*PANEL_END', src), 'G1 面板竞态断言'


def test_tracking_output_exists():
    """纸面跟踪必须落盘（用户选 (iii) 的核心）。"""
    src = _src()
    assert 'TRACK_DIR' in src and 'tracking_' in src


def test_exit_check_integrated():
    """判据链 #31 的 EXIT 检查必须接进每日报告，且**只读生产**。"""
    import ast
    src = _src()
    assert 'avoid_section' in src, '必须有 EXIT 检查段'
    assert 'load_production_holdings' in src
    assert 'AVOID_10' in src and 'AVOID_30' in src
    # 场景诚实性：必须声明"不相交 / 只用于卖出"
    assert '不相交' in src, '必须声明回避名单与买入名单不相交'
    assert '只用于卖出' in src or '卖出决策' in src
    # 只读：AST 级检查生产 DB 连接带 mode=ro
    tree = ast.parse(src)
    ok = False
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'connect'
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'sqlite3'):
            seg = ast.get_source_segment(src, node) or ''
            if 'mode=ro' in seg:
                ok = True
    assert ok, '生产 DB 连接必须 mode=ro（AST 实参级）'


def test_pending_orders_uses_real_column():
    """列名必须是实际 schema 的 `code`（曾误写 `product` ⇒ 静默失败）。"""
    src = _src()
    assert 'select code from pending_orders' in src
    assert 'select product from pending_orders' not in src
