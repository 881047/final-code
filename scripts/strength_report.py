"""判据链 #30 执行器：**强弱预测报告 + 纸面跟踪**。

预注册：plans/strength-report-prereg-20261011.md（**先于本脚本跑数冻结**，commit 00bf717）

## 这是什么
用户 2026-10-11 给 Final 的最终定义：
「**每日**预测市场个股走势趋势；**不要绝对的涨跌幅**；要预测准这只股**是强是弱**」，
输出形态选 **(iii) 报告 + 纸面跟踪**。

**本脚本不是交易策略** —— 它不下单、不碰账户、不产出"按预测交易会赚多少"。
它只回答一个问题：**分档预测准不准，以及是否稳定。**

## 三个不可省的输出要件（预注册 §四）
1. 不预测绝对涨跌幅（架构不支持：逐日截面秩白化消灭市场级信息）
2. 不预测大盘方向（同上）
3. 不输出"第几名"（档内排序无分辨力：top-10 内 IC 为负）

## 用法
    python scripts/strength_report.py --backfill          # 全史标定 + 跟踪落盘
    python scripts/strength_report.py --asof 2026-09-30   # 单日报告
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from fterm.config import load_config                      # noqa: E402
from fterm.data import ashare as sa                       # noqa: E402
from fterm.features.build import build_dataset            # noqa: E402
from fterm.model.stats import newey_west_t                # noqa: E402
from scipy.stats import spearmanr                         # noqa: E402

PROD = Path(r'D:\Final\data\reports')
OUT_DIR = ROOT / 'experiments' / 'strength_report_20261011'
TRACK_DIR = ROOT / 'data' / 'strength_tracking'

START = pd.Timestamp('2019-01-02')
PANEL_END = pd.Timestamp('2026-09-30')       # G1 隔离线
N_DECILE = 10
Z2 = 1.959963984540054 + 0.8416212335729143
BONF_T = 2.64                                # 预注册 §三 P8（6 个主检验）
MAIN_H = 1                                   # 用户口径"每日"
ALT_H = 5                                    # 与生产 label_horizon 对齐

THREE_NO = [
    '本程序预测**相对强弱**，不预测绝对涨跌幅。',
    '本程序不预测大盘方向。',
    '本程序输出**分档**，不输出名次（档内排序无分辨力）。',
]


def nw(x):
    a = np.asarray(x, 'float64')
    a = a[np.isfinite(a)]
    if a.size < 5:
        return None
    try:
        v = newey_west_t(a)
    except Exception:
        return None
    return float(v) if v is not None and np.isfinite(v) else None


def mde(x):
    a = np.asarray(x, 'float64')
    a = a[np.isfinite(a)]
    if a.size < 2:
        return None
    return float(Z2 * a.std(ddof=1) / np.sqrt(a.size))


def load_streams():
    """生产谱系预测流（rankloss 双流平均）。"""
    d1 = pd.read_parquet(PROD / 'rankloss_ss7_21_99_5_13.parquet',
                         columns=['date', 'product', 'pred'])
    d2 = pd.read_parquet(PROD / 'rankloss_ss123_2024_42_314_2027.parquet',
                         columns=['date', 'product', 'pred'])
    for x in (d1, d2):
        x['date'] = pd.to_datetime(x['date'])
    P = (d1.pivot_table(index='date', columns='product', values='pred')
         .add(d2.pivot_table(index='date', columns='product', values='pred'), fill_value=0) / 2)
    return P.astype('float64').sort_index()


def build_forward(hfq: pd.DataFrame, h: int) -> pd.DataFrame:
    """fwd_ret = hfq_open[t+h+1]/hfq_open[t+1] − 1（与全链标签同口径）。"""
    return hfq.shift(-(h + 1)) / hfq.shift(-1) - 1.0


def decile_of(series: pd.Series) -> pd.Series:
    """按预测分 10 档（0=最弱，9=最强）。G2：每只恰属一档、档大小均衡。"""
    r = series.rank(method='first')
    return pd.qcut(r, N_DECILE, labels=False)


def eval_day(p_row: pd.Series, y_row: pd.Series, n_min: int = 500) -> dict | None:
    """单日分档评估。基准 = 当日全市场跑赢率（G5）。"""
    m = np.isfinite(p_row.to_numpy('float64')) & np.isfinite(y_row.to_numpy('float64'))
    if int(m.sum()) < n_min:
        return None
    p = p_row[m]
    y = y_row[m]
    mkt = float(y.mean())
    ex = y - mkt
    dec = decile_of(p)
    per = {}
    for d in range(N_DECILE):
        v = ex[dec == d]
        per[d] = {'n': int(len(v)), 'exc_mean': float(v.mean()) if len(v) else None,
                  'win': float((v > 0).mean()) if len(v) else None}
    dec_means = np.array([per[d]['exc_mean'] if per[d]['exc_mean'] is not None else np.nan
                          for d in range(N_DECILE)])
    base_win = float((ex > 0).mean())
    top = ex[dec == N_DECILE - 1]
    bot = ex[dec == 0]
    return {
        'n': int(m.sum()),
        'mkt': mkt,
        'base_win': base_win,
        'dec_mean': dec_means,
        'spread': float(dec_means[-1] - dec_means[0]),
        'rho': float(spearmanr(np.arange(N_DECILE), dec_means).statistic)
        if np.isfinite(dec_means).all() else None,
        'top_win': float((top > 0).mean()),
        'bot_win': float((bot > 0).mean()),
        'top_exc': float(top.mean()),
        'bot_exc': float(bot.mean()),
        'per_decile': per,
    }


def backfill(asof: pd.Timestamp | None = None) -> int:
    cfg = load_config(ROOT / 'config.yaml')
    panels = sa.load_panels(cfg)
    bundle = build_dataset(cfg, compute_fingerprints=False)
    dates = panels['close'].index
    assert dates.max() <= PANEL_END, f'G1 面板竞态：{dates.max()}'

    P = load_streams()
    uni = bundle.universe
    codes = [c for c in P.columns if c in bundle.labels['fwd_ret'].columns]
    o = panels['open'][codes].astype('float64')
    adj = panels['adj'][codes].astype('float64')
    hfq = (o * adj)

    idx = [d for d in dates if d >= START and d in P.index and d in hfq.index]
    if asof is not None:
        idx = [d for d in idx if d <= asof]
    mask = uni.reindex(index=idx, columns=codes).fillna(False)

    report = {'prereg': 'plans/strength-report-prereg-20261011.md',
              'definition': '每日相对强弱分档预测',
              'three_no': THREE_NO,
              'window': [str(idx[0].date()), str(idx[-1].date())],
              'n_days': len(idx), 'n_codes': len(codes),
              'horizons': {}}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TRACK_DIR.mkdir(parents=True, exist_ok=True)

    for H, tag in ((MAIN_H, 'h1'), (ALT_H, 'h5')):
        fwd = build_forward(hfq, H)
        Y = fwd.reindex(idx)
        Pm = P.reindex(index=idx, columns=codes).where(mask)
        Ym = Y.where(mask)

        rows = []
        for dt in idx:
            r = eval_day(Pm.loc[dt], Ym.loc[dt])
            if r is None:
                continue
            r['date'] = dt
            rows.append(r)
        D = pd.DataFrame(rows).set_index('date')
        if D.empty:
            continue

        # 逐日序列
        s_rho = D['rho'].dropna()
        s_spread = D['spread'].dropna()
        s_top = (D['top_win'] - D['base_win']).dropna()
        # ⚠ 判弱口径（首版写错，已修）：
        #   "判弱准" = 最弱档**跑输率** 高于 基准跑输率
        #            = (1 − bot_win) − (1 − base_win) = base_win − bot_win
        #   首版写成 `bot_win − (1 − base_win)` ⇒ 符号相反（实测 −0.135 而真值 +0.033）。
        s_bot = (D['base_win'] - D['bot_win']).dropna()

        # 分期（P4）
        yrs = pd.to_datetime(D.index).year
        pre, late = yrs <= 2020, yrs >= 2021

        def seg(x, m):
            v = x[np.asarray(m)[:len(x)]] if len(x) == len(m) else x
            return {'mean': float(v.mean()) if len(v) else None, 't': nw(v),
                    'n': int(len(v)), 'mde': mde(v)}

        # 各粒度 IC（P5 分辨率诚实）
        from scipy.stats import spearmanr as _sr
        gran = {}
        for lab, k in (('full', None), ('top1000', 1000), ('top100', 100),
                       ('top50', 50), ('top10', 10)):
            ics = []
            for dt in idx:
                p = Pm.loc[dt].dropna()
                y = Ym.loc[dt].reindex(p.index).dropna()
                cm = p.index.intersection(y.index)
                if len(cm) < (k or 0) + 20:
                    continue
                pv, yv = p[cm], y[cm]
                if k is not None:
                    oo = np.argsort(-pv.to_numpy())[:k]
                    pv, yv = pv.iloc[oo], yv.iloc[oo]
                if len(pv) < 10:
                    continue
                v = _sr(pv.to_numpy(), yv.to_numpy()).statistic
                if v is not None and np.isfinite(v):
                    ics.append(v)
            ics = np.array(ics)
            gran[lab] = {'ic': float(ics.mean()) if len(ics) else None,
                         't': nw(ics), 'n': int(len(ics))}

        # ⚠ 分档均值口径（首版写错，已修）：`D['dec_mean']` 每行是长度 10 的数组；
        #   首版用 `.apply(lambda a: np.nanmean(a))` ⇒ 得到"每日的档间均值"（1871 个数），
        #   打印出 1871 个值而不是 10 个档。正解：按**档位**在时间上取均值 ⇒ shape (10,)。
        dec_mat = np.vstack(D['dec_mean'].to_numpy())          # (n_days, 10)
        decile_mean = np.nanmean(dec_mat, axis=0)

        # ⚠ 单调性有两个**不同**的量，预注册只写了一个 ρ（erratum，见下）：
        #   ① **逐日 ρ**：当日 10 个档均值对档位的 Spearman —— 单日噪声极大
        #      （每档 ~400 只，档均值 SE ≈ 25bp，而档间真差只有几 bp）⇒ 接近 0；
        #   ② **汇总 ρ**：先对全期取各档均值（10 个数），再算 Spearman —— 这是
        #      立案前标定用的量（实测 +0.9758 / +1.0000）。
        #   预注册 §三 P1 写的「ρ ≥ +0.80」**指的是 ②**，但执行器首版只算了 ①。
        #   ⇒ 两个都报，并以 ② 对 P1 判定（erratum 登记，判据本身不改）。
        pooled_rho = float(spearmanr(np.arange(N_DECILE), decile_mean).statistic) \
            if np.isfinite(decile_mean).all() else None
        daily_rho = {'full': seg(s_rho, np.ones(len(s_rho), bool)),
                     'pre': seg(s_rho, pre), 'late': seg(s_rho, late)}

        report['horizons'][tag] = {
            'h': H,
            'n_days': int(len(D)),
            'mean_base_win': float(D['base_win'].mean()),
            'decile_mean': [float(x) for x in decile_mean],
            'p1_pooled_rho': pooled_rho,
            'p1_daily_rho': daily_rho,
            'p1_spread': {'full': seg(s_spread, np.ones(len(s_spread), bool)),
                          'pre': seg(s_spread, pre), 'late': seg(s_spread, late)},
            'p2_top': {'full': seg(s_top, np.ones(len(s_top), bool)),
                       'pre': seg(s_top, pre), 'late': seg(s_top, late)},
            'p3_bot': {'full': seg(s_bot, np.ones(len(s_bot), bool)),
                       'pre': seg(s_bot, pre), 'late': seg(s_bot, late)},
            'granularity_ic': gran,
            'last_decile': D['per_decile'].iloc[-1],
        }

        # 纸面跟踪落盘（逐日）
        tr = pd.DataFrame({
            'n': D['n'], 'base_win': D['base_win'],
            'top_win': D['top_win'], 'bot_win': D['bot_win'],
            'top_exc': D['top_exc'], 'bot_exc': D['bot_exc'],
            'spread': D['spread'], 'rho': D['rho'],
        })
        tr.to_parquet(TRACK_DIR / f'tracking_{tag}.parquet')

    (OUT_DIR / 'strength_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    _print_report(report)
    print(f'\n报告 → {OUT_DIR / "strength_report.json"}')
    print(f'跟踪 → {TRACK_DIR}/tracking_*.parquet')
    return 0


def _print_report(report: dict) -> None:
    print('=' * 104)
    print('Final · 每日相对强弱分档报告（判据链 #30）')
    print('=' * 104)
    print(f"窗口 {report['window'][0]} → {report['window'][1]}，"
          f"{report['n_days']} 交易日，{report['n_codes']} 只")
    print()
    print('【口径声明 · 三不】')
    for i, s in enumerate(report['three_no'], 1):
        print(f'  {i}. {s}')
    print()
    for tag, v in report['horizons'].items():
        print('=' * 104)
        print(f"持有期 {v['h']} 交易日（{'用户口径：每日' if v['h'] == 1 else '与生产 label_horizon 对齐'}）")
        print('=' * 104)
        dm = v['decile_mean']
        print('  分档平均超额（bp，0=最弱档，9=最强档）：')
        print('    ' + '  '.join(f'{d}:{x*1e4:+.1f}' for d, x in enumerate(dm)))
        print(f"  当日基准跑赢率（全市场均值）：{v['mean_base_win']:.4f}")
        print()
        hdr = f"  {'判据':>18s} | {'全样本':>18s} | {'前段19-20':>18s} | {'后段21-26':>18s}"
        print(hdr)
        print('  ' + '-' * (len(hdr) - 2))
        # P1 用**汇总 ρ**（预注册本意）；逐日 ρ 单列（噪声大，仅参考）
        pr = v['p1_pooled_rho']
        print(f"  {'P1 汇总ρ（P1判据）':>18s} | {pr if pr is None else format(pr, '+.4f'):>18s} "
              f"| {'—':>18s} | {'—':>18s}")
        for lab, key in (('P1 逐日ρ（参考）', 'p1_daily_rho'),
                         ('P1 首尾差', 'p1_spread'),
                         ('P2 判强超额', 'p2_top'), ('P3 判弱超额', 'p3_bot')):
            row = []
            for segname in ('full', 'pre', 'late'):
                s = v[key][segname]
                if s['mean'] is None:
                    row.append('n/a')
                else:
                    row.append(f"{s['mean']:+.4f} t={s['t'] or 0:+.2f}")
            print(f"  {lab:>18s} | {row[0]:>18s} | {row[1]:>18s} | {row[2]:>18s}")
        print()
        print('  P5 分辨率诚实声明（各粒度内 IC）—— 档内名次无效：')
        for lab, g in v['granularity_ic'].items():
            print(f"    {lab:>10s}: IC {g['ic'] if g['ic'] is None else format(g['ic'], '+.4f')}  "
                  f"t={g['t'] if g['t'] is None else format(g['t'], '+.2f')}")
        print()
        print(f"  Bonferroni（6 个主检验）临界 |t| ≥ {BONF_T}")


AVOID_10 = 0.10          # 判据链 #31：最弱 10% = 强回避（EXIT 用）
AVOID_30 = 0.30
WATCH = 0.50
PROD_DB = Path(r'D:\Final\data\state\fterm.db')   # 只读：生产持仓/pending


def load_production_holdings() -> tuple[set[str], set[str], str | None]:
    """只读生产持仓与挂单（判据链 #31 的 EXIT 检查用）。

    G1：**必须 `mode=ro`** —— 本脚本是只读报告，绝不可写生产 state。
    返回 (持仓代码, 挂单代码, as-of 说明)。
    """
    if not PROD_DB.exists():
        return set(), set(), None
    try:
        con = sqlite3.connect('file:' + str(PROD_DB) + '?mode=ro', uri=True)
        try:
            # 列名以实际 schema 为准（positions.code / pending_orders.code）
            pos = {r[0] for r in con.execute('select code from positions')}
            pend = {r[0] for r in con.execute('select code from pending_orders')}
        finally:
            con.close()
        return pos, pend, str(PROD_DB)
    except Exception as e:
        print(f'  ⚠ 读生产持仓失败（{e}）⇒ 跳过 EXIT 检查')
        return set(), set(), None


def avoid_section(p: pd.Series, holdings: set[str], pending: set[str]) -> None:
    """判据链 #31 的 EXIT 检查：**持仓是否跌入回避名单**。

    ⚠ 场景诚实性（#31 §一）：回避名单 = 最弱 10%，而买入名单 = 最强档
    ⇒ **两集合不相交** ⇒ 本节的用途**只有 EXIT**，不是选股。
    """
    rank = p.rank(pct=True)
    print()
    print('=' * 104)
    print('【判据链 #31 · EXIT 检查】持仓是否跌入回避名单')
    print('=' * 104)
    print('  ⚠ 回避名单（最弱 10%）与买入名单（最强档）**不相交** ⇒')
    print('    本节**只用于卖出决策**，不是选股建议。')
    print()
    n10 = int((rank <= AVOID_10).sum())
    n30 = int((rank <= AVOID_30).sum())
    print(f'  回避名单规模：AVOID-10 = {n10} 只；AVOID-30 = {n30} 只')
    print()
    if not holdings and not pending:
        print('  当前生产**无持仓、无挂单** ⇒ 无可检查项（首笔成交后本节生效）。')
        return
    for lab, codes in (('持仓', holdings), ('挂单', pending)):
        if not codes:
            continue
        print(f'  {lab}（{len(codes)} 只）：')
        for c in sorted(codes):
            if c not in rank.index:
                print(f'    {c}: 不在今日预测面（停牌/退市/不在池）⇒ 无法判档')
                continue
            r = float(rank[c])
            d = int(r * N_DECILE)
            d = min(d, N_DECILE - 1)
            if r <= AVOID_10:
                flag = '🔴 跌入 AVOID-10 ⇒ **卖出信号**（历史：后续 5 日 −80.0bp，t=−8.68）'
            elif r <= AVOID_30:
                flag = '🟠 跌入 AVOID-30（后续 5 日 −27.1bp）'
            elif r <= WATCH:
                flag = '🟡 WATCH（30–50%，无信号）'
            else:
                flag = '🟢 正常'
            print(f'    {c}: 预测秩 {r:.3f}（第 {d} 档）  {flag}')
    print()
    print('  ⚠ 卖出后资金去向会影响净效果：换成市场组合 ≈ 按 #31 的覆盖比；')
    print('     **闲置为现金**则还须减去市场收益（机会成本，见 #31 §六-2）。')


def single_day(asof: pd.Timestamp) -> int:
    """单日报告（--asof）。"""
    cfg = load_config(ROOT / 'config.yaml')
    panels = sa.load_panels(cfg)
    bundle = build_dataset(cfg, compute_fingerprints=False)
    P = load_streams()
    codes = [c for c in P.columns if c in bundle.labels['fwd_ret'].columns]
    uni = bundle.universe
    asof = pd.Timestamp(asof)
    assert asof <= PANEL_END, 'G1 面板竞态'
    if asof not in P.index:
        prev = [d for d in P.index if d <= asof]
        if not prev:
            print('无可用预测日')
            return 1
        asof = prev[-1]
    p = P.loc[asof].reindex(codes)
    mk = uni.loc[asof].reindex(codes).fillna(False) if asof in uni.index else None
    if mk is not None:
        p = p.where(mk)
    p = p.dropna()
    print('=' * 104)
    print(f'Final · 每日相对强弱报告   as-of {asof.date()}   合格股 {len(p)} 只')
    print('=' * 104)
    print()
    print('【口径声明 · 三不】')
    for i, s in enumerate(THREE_NO, 1):
        print(f'  {i}. {s}')
    print()
    dec = decile_of(p)
    print(f"  {'档':>3s} {'只数':>6s}  说明")
    print('  ' + '-' * 40)
    for d in range(N_DECILE - 1, -1, -1):
        n = int((dec == d).sum())
        note = '← 最强档' if d == N_DECILE - 1 else ('← 最弱档' if d == 0 else '')
        print(f'  {d:>3d} {n:>6d}  {note}')
    print()
    print('  ⚠ 档内**不提供名次**（top-10 内 IC 为负，名次无分辨力）。')
    print('  ⚠ 本报告不含"按此交易会赚多少"——那属组合层（判据链 #24 已证无功效）。')

    # ---- 判据链 #31：EXIT 检查（只读生产）----
    holdings, pending, _ = load_production_holdings()
    avoid_section(p, holdings, pending)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--backfill', action='store_true')
    ap.add_argument('--asof', type=str, default=None)
    args = ap.parse_args()
    if args.asof:
        return single_day(pd.Timestamp(args.asof))
    return backfill()


if __name__ == '__main__':
    raise SystemExit(main())
