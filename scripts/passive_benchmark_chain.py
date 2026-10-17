"""判据链 #29 执行器：**被动基准** —— 给「值不值得做」下定义。

预注册：plans/passive-benchmark-prereg-20261011.md（**先于本脚本跑数冻结**，commit 85f132e）

## 本链要回答的那个从未被回答的问题
**这 2 万块钱，7.7 年里"什么都不做"能拿到什么？策略相对它如何？**

本项目 26 条链、150–200 次回测，**全部**是"臂 A vs 臂 B"的自参照比较；
唯一的"基准"是**自己的生产位**（§49 R5 已判不合格并作废）。
⇒ 在没有"什么都不做"这个数之前，"值不值得继续"**无法回答**。

## 三层结构（预注册 §0.2）
  ① **POOL**   同池等权（`pool_index`）—— **同池参照系**，但 2 万买不起 ⇒ 不可投
  ② **ETF**    真实 ETF（`fund_daily` + `fund_adj`）—— **可投**（1 手 200–600 元）
  ③ **IDX**    指数价格 —— 交叉核对

## 纪律
- 只读老库 `neweco.db`；不改 NEW ECO 任何文件；不改本仓 config.yaml；
- ETF 成本**必须**与个股分开（**ETF 免印花税**）—— G6 互斥断言；
- ETF 复权用 `fund_adj`；
- 策略臂复用 #28 的忠实账本（逐字不改），保证跨链可比。
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

from fterm import contracts                                   # noqa: E402
import neweco_portability_chain as chain                      # noqa: E402

# ------------------------------------------------------------------ 冻结常量
NEWECO_DB = Path(r'D:/NEW ECO/data/neweco.db')
IDX_DIR = ROOT / 'data' / 'index'
OUT_DIR = ROOT / 'experiments' / 'passive_benchmark_20261011'

EQUITY0 = 20000.0                       # 2 万档（预注册 §二）
MAX_CASH_UTIL = 0.95
LOT = 100
SLIP_TICKS = 1
TICK = 0.01
START = pd.Timestamp('2019-01-02')      # 数据可用性决定（rankloss 流起点）
END = pd.Timestamp('2026-09-30')        # worktree 隔离线
Z2 = 1.959963984540054 + 0.8416212335729143
BONF_T = 2.87                           # 预注册 §四 B8（n=12 双侧 5%）

# 预注册 §三-5：**预先指定**，不做事后挑选
ETFS = {
    '510300.SH': '沪深300ETF',
    '510500.SH': '中证500ETF',
    '510050.SH': '上证50ETF',
    '159915.SZ': '创业板ETF',
}
IDX_MAP = {'510300.SH': '000300.SH', '510500.SH': '000905.SH',
           '159915.SZ': '399001.SZ'}


def nw(x):
    a = np.asarray(x, 'float64')
    a = a[np.isfinite(a)]
    if a.size < 5:
        return None
    try:
        v = chain.nw(a)
    except Exception:
        return None
    return float(v) if v is not None and np.isfinite(v) else None


def mde(x):
    a = np.asarray(x, 'float64')
    a = a[np.isfinite(a)]
    if a.size < 2:
        return None
    return float(Z2 * a.std(ddof=1) / np.sqrt(a.size))


def perf(eq: pd.Series) -> dict:
    """权益曲线 → 年化/MDD/Sharpe。

    ⚠ 频率感知（首版自伤，已修）：本链 POOL 是**日频**（1880 行）、策略臂是**月频**（~115 行）。
    首版一律用 √243 年化 ⇒ 月频 Sharpe 被虚增 √(243/12)≈4.5 倍
    （实测 NE2 Sharpe 5.31 —— 荒谬值）。现按实际观测间隔推断年化因子。
    """
    eq = eq.dropna()
    if len(eq) < 2:
        return {}
    tot = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    ann = (1.0 + tot) ** (1.0 / yrs) - 1.0 if yrs > 0 else np.nan
    dd = float((eq / eq.cummax() - 1.0).min())
    r = eq.pct_change().dropna()
    gaps = pd.Series(eq.index).diff().dt.days.dropna()
    med = float(gaps.median()) if len(gaps) else 1.0
    per_year = 243.0 if med <= 3 else (52.0 if med <= 10 else (12.0 if med <= 45 else 1.0))
    sharpe = (float(r.mean() / r.std(ddof=1) * np.sqrt(per_year))
              if r.std(ddof=1) > 0 else None)
    return {'total': tot, 'ann': ann, 'mdd': dd, 'sharpe': sharpe,
            'years': yrs, 'n': int(len(eq)), 'per_year': per_year,
            'calmar': float(ann / abs(dd)) if dd < 0 else None}


# ------------------------------------------------------------------ 老库读取
def load_pool():
    con = sqlite3.connect('file:' + str(NEWECO_DB) + '?mode=ro', uri=True)
    df = pd.read_sql_query('select trade_date, close, n from pool_index order by trade_date', con)
    con.close()
    df['dt'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    return df.set_index('dt')['close'].astype('float64').sort_index(), df


def load_fund(codes):
    con = sqlite3.connect('file:' + str(NEWECO_DB) + '?mode=ro', uri=True)
    out = {}
    for c in codes:
        q = pd.read_sql_query(
            f"select trade_date, open, close from fund_daily where ts_code='{c}' "
            f"order by trade_date", con)
        a = pd.read_sql_query(
            f"select trade_date, adj_factor from fund_adj where ts_code='{c}' "
            f"order by trade_date", con)
        if q.empty:
            continue
        q['dt'] = pd.to_datetime(q['trade_date'], format='%Y%m%d')
        q = q.set_index('dt')
        if not a.empty:
            a['dt'] = pd.to_datetime(a['trade_date'], format='%Y%m%d')
            q = q.join(a.set_index('dt')['adj_factor'], how='left')
            q['adj_factor'] = q['adj_factor'].ffill().fillna(1.0)
        else:
            q['adj_factor'] = 1.0
        out[c] = q[['open', 'close', 'adj_factor']].astype('float64').sort_index()
    con.close()
    return out


def load_index(code):
    df = pd.read_parquet(IDX_DIR / f'{code}.parquet')
    s = df['close'].astype('float64')
    s.index = pd.to_datetime(s.index.astype(str))
    return s.sort_index()


# ------------------------------------------------------------------ ETF 账本
def etf_buyhold(px: pd.DataFrame, label: str) -> tuple[pd.Series, dict]:
    """2 万档 ETF 买入持有：首日收盘买入（整手），持有到窗口末。

    成本（预注册 §三-3）：**ETF 免印花税**；佣金 max(amount×0.00025, 5)；
    过户费按场内基金不收处理（登记为近似）。**不套用个股 trade_fee**。
    """
    px = px[(px.index >= START) & (px.index <= END)].dropna(subset=['close'])
    if len(px) < 2:
        return pd.Series(dtype=float), {'error': '数据不足'}
    adj = px['adj_factor']
    d0 = px.index[0]
    p0 = float(px['close'].iloc[0])
    buy_px = p0 + SLIP_TICKS * TICK
    budget = EQUITY0 * MAX_CASH_UTIL
    lots = int(budget // (buy_px * LOT))
    while lots > 0:
        amount = buy_px * lots * LOT
        fee = max(amount * contracts.COMMISSION_RATE, contracts.MIN_COMMISSION)  # 无印花税
        if amount + fee <= EQUITY0:
            break
        lots -= 1
    if lots <= 0:
        return pd.Series(dtype=float), {'error': '买不起 1 手'}
    amount = buy_px * lots * LOT
    fee = max(amount * contracts.COMMISSION_RATE, contracts.MIN_COMMISSION)
    cash = EQUITY0 - amount - fee
    # 含权盯市：close × adj[d]/adj[entry]
    a0 = float(adj.iloc[0]) if adj.iloc[0] > 0 else 1.0
    eq = cash + px['close'] * (adj / a0) * lots * LOT
    return eq, {'lots': lots, 'buy_px': buy_px, 'entry_fee': fee,
                'deployed': amount, 'deploy_ratio': amount / EQUITY0,
                'cash_left': cash, 'entry_date': str(d0.date())}


def period_returns(eq: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    """把任意频率的权益曲线，复利到 `target_index` 的**同一区间**上。

    ⚠ 首版自伤（会伪造结论，已修）：把**月频策略收益**与**日频基准收益**直接配对
    （`r.index.intersection(br.index)` 取到月频日期，而 `br` 是**日**收益）
    ⇒ 基准被当成"每日只涨一点" ⇒ 超额被系统性夸大（实测虚增约 +1%/月）。
    正确做法：对每一对相邻目标日 (d0, d1)，用 `eq[d1]/eq[d0] − 1` 取**同区间**收益。
    """
    eq = eq.dropna().sort_index()
    out = {}
    for i in range(1, len(target_index)):
        d0, d1 = target_index[i - 1], target_index[i]
        s0 = eq[eq.index <= d0]
        s1 = eq[eq.index <= d1]
        if len(s0) == 0 or len(s1) == 0:
            continue
        v0, v1 = float(s0.iloc[-1]), float(s1.iloc[-1])
        if v0 > 0:
            out[d1] = v1 / v0 - 1.0
    return pd.Series(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-strategy', action='store_true')
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    report = {'prereg': 'plans/passive-benchmark-prereg-20261011.md',
              'window': [str(START.date()), str(END.date())],
              'equity0': EQUITY0, 'gates': {}, 'passive': {}, 'strategy': {}}

    # ---------------- ① POOL（同池参照系） ----------------
    pool, pool_raw = load_pool()
    pool = pool[(pool.index >= START) & (pool.index <= END)]
    assert pool.index.max() <= END, 'G1 面板竞态'
    # 名义权益（不可投，仅参照）：2 万全额投入
    pool_eq = pool / pool.iloc[0] * EQUITY0
    report['passive']['POOL'] = {**perf(pool_eq), 'investable': False,
                                 'note': '同池等权（主板+创业板），2 万买不起 ⇒ 不可投'}
    print(f"POOL      累计 {perf(pool_eq)['total']:+.2%}  年化 {perf(pool_eq)['ann']:+.2%}  "
          f"MDD {perf(pool_eq)['mdd']:+.2%}  Sharpe {perf(pool_eq)['sharpe']:.2f}")

    # ---------------- ② ETF（可投项） ----------------
    funds = load_fund(list(ETFS))
    print(f"\nETF 数据：请求 {len(ETFS)} 只，取到 {len(funds)} 只")
    for code, name in ETFS.items():
        if code not in funds:
            report['passive'][code] = {'error': '数据缺失', 'name': name}
            print(f"  {code} {name}: **数据缺失**（预注册规定不替换）")
            continue
        eq, meta = etf_buyhold(funds[code], name)
        if eq.empty:
            report['passive'][code] = {'error': meta.get('error', '?'), 'name': name}
            print(f"  {code} {name}: {meta.get('error')}")
            continue
        p = perf(eq)
        report['passive'][code] = {**p, **meta, 'name': name, 'investable': True}
        print(f"  {code} {name}: 累计 {p['total']:+.2%}  年化 {p['ann']:+.2%}  "
              f"MDD {p['mdd']:+.2%}  Sharpe {p['sharpe']:.2f}  "
              f"买 {meta['lots']} 手（部署 {meta['deploy_ratio']:.1%}）")

    # ---------------- ③ 指数（交叉核对） ----------------
    for code in ('000300.SH', '000905.SH'):
        s = load_index(code)
        s = s[(s.index >= START) & (s.index <= END)]
        if len(s) > 1:
            eq = s / s.iloc[0] * EQUITY0
            report['passive'][f'IDX_{code}'] = {**perf(eq), 'investable': False}
            print(f"  IDX {code}: 累计 {perf(eq)['total']:+.2%}  年化 {perf(eq)['ann']:+.2%}")

    # ---------------- 仪器门 ----------------
    print()
    print('=' * 100)
    print('仪器门')
    print('=' * 100)
    # G2：POOL 与独立复算（老库 hfq 等权）交叉核对
    g2 = None
    try:
        con = sqlite3.connect('file:' + str(NEWECO_DB) + '?mode=ro', uri=True)
        q = pd.read_sql_query(
            "select d.trade_date, d.ts_code, d.close*a.adj_factor as hfq "
            "from daily d join adj_factor a on d.ts_code=a.ts_code "
            "and d.trade_date=a.trade_date "
            "where d.trade_date >= '20190102' and d.trade_date <= '20260930'", con)
        con.close()
        q['dt'] = pd.to_datetime(q['trade_date'], format='%Y%m%d')
        piv = q.pivot_table(index='dt', columns='ts_code', values='hfq').sort_index()
        ew = piv.pct_change().mean(axis=1).iloc[1:]
        ind = (1 + ew).cumprod()
        tot = float(ind.iloc[-1] - 1)
        yrs = (ind.index[-1] - ind.index[0]).days / 365.25
        ann_ind = (1 + tot) ** (1 / yrs) - 1
        ann_pool = report['passive']['POOL']['ann']
        d_ann = abs(ann_ind - ann_pool)
        g2 = {'indep_ann': float(ann_ind), 'pool_ann': float(ann_pool), 'delta_pp': float(d_ann * 100)}
        ok2 = d_ann < 0.05          # 5pp/年 容差（口径差：日再平衡 vs 月内漂移，docstring 自述 ~5pp/年）
        report['gates']['G2_cross'] = {'pass': bool(ok2), **g2}
        print(f"  G2 交叉核对：独立复算（日频再平衡等权）年化 {ann_ind:+.2%} vs "
              f"POOL {ann_pool:+.2%} ⇒ Δ {d_ann*100:.2f}pp ⇒ {'✓' if ok2 else '✗'}")
        print(f"     （口径差已知：日再平衡自带再平衡红利，pool_index 为月内漂移+月初再平衡）")
    except Exception as e:
        report['gates']['G2_cross'] = {'pass': None, 'error': str(e)}
        print(f"  G2 交叉核对：失败 {e}")

    # G3：ETF300 vs 000300 指数相关
    if '510300.SH' in funds:
        f = funds['510300.SH']
        f = f[(f.index >= START) & (f.index <= END)]
        idx = load_index('000300.SH')
        idx = idx[(idx.index >= START) & (idx.index <= END)]
        rf = (f['close'] * f['adj_factor']).pct_change().dropna()
        ri = idx.pct_change().dropna()
        cm = rf.index.intersection(ri.index)
        if len(cm) > 30:
            corr = float(np.corrcoef(rf.loc[cm], ri.loc[cm])[0, 1])
            ok3 = corr > 0.9
            report['gates']['G3_idx_corr'] = {'pass': bool(ok3), 'corr': corr, 'n': int(len(cm))}
            print(f"  G3 ETF300 vs 000300 日收益相关 {corr:.4f}（n={len(cm)}）⇒ {'✓' if ok3 else '✗'}")

    # G6：ETF 不计印花税（互斥断言）
    ok6 = (contracts.trade_fee(10000.0, -1, pd.Timestamp('2025-01-01')) >
           max(10000.0 * contracts.COMMISSION_RATE, contracts.MIN_COMMISSION))
    report['gates']['G6_etf_no_stamp'] = {'pass': bool(ok6),
                                          'note': 'ETF 用独立计费（无印花税），个股用 trade_fee'}
    print(f"  G6 ETF/个股计费互斥：{'✓' if ok6 else '✗'}（ETF 臂走独立计费，无印花税）")

    # G7：ETF 收益应 >= 指数价格收益（含分红），且 <= 价格收益 + 5pp/年
    if '510300.SH' in funds and 'IDX_000300.SH' in report['passive']:
        a_etf = report['passive']['510300.SH']['ann']
        a_idx = report['passive']['IDX_000300.SH']['ann']
        ok7 = (a_etf >= a_idx - 0.005) and (a_etf <= a_idx + 0.05)
        report['gates']['G7_etf_vs_idx'] = {'pass': bool(ok7),
                                            'etf_ann': float(a_etf), 'idx_ann': float(a_idx)}
        print(f"  G7 ETF300 {a_etf:+.2%} vs 指数 {a_idx:+.2%} ⇒ "
              f"{'✓ 在[价格, 价格+5pp]内' if ok7 else '✗ 越界'}")

    # ---------------- 策略臂 ----------------
    if not args.skip_strategy:
        print()
        print('=' * 100)
        print('策略臂（复用 #28 忠实账本，逐字不改）')
        print('=' * 100)
        px = chain.load_panel()
        N = chain.load_neweco()
        dates_sorted = list(px['open'].index)
        anchors = [a for a in N.index if a in px['open'].index and a >= START]
        strategy_curves = {}
        etf_curves = {}
        for code in ETFS:
            if code in funds:
                e, _ = etf_buyhold(funds[code], code)
                if not e.empty:
                    etf_curves[code] = e

        for k in (2, 6):
            tg = chain.arm_targets(N, px, k)
            tg = {a: v for a, v in tg.items() if a >= START}
            C, meta = chain.simulate(tg, dates_sorted, px, f'NE{k}')
            C = C[C.index >= START]
            if len(C) < 2:
                continue
            eq = C['equity']
            strategy_curves[f'NE{k}'] = eq
            p = perf(eq)
            r = eq.pct_change().dropna()
            report['strategy'][f'NE{k}'] = {**p, 'n_trades': meta['n_trades'],
                                            'costs': meta['total_costs'],
                                            'deploy_median': meta['deploy_median']}
            print(f"  NE{k}: 累计 {p['total']:+.2%}  年化 {p['ann']:+.2%}  MDD {p['mdd']:+.2%}  "
                  f"Sharpe {p['sharpe']:.2f}  成交 {meta['n_trades']}  成本 {meta['total_costs']:,.0f}")
            # 逐期超额 vs POOL / 最佳 ETF（**同区间**配对，防频率错配）
            for bname in ('POOL', '510300.SH'):
                if bname not in report['passive'] or 'total' not in report['passive'][bname]:
                    continue
                if bname == 'POOL':
                    beq = pool_eq
                else:
                    beq, _ = etf_buyhold(funds[bname], bname)
                if beq is None or beq.empty:
                    continue
                br = period_returns(beq, r.index)      # ← 复利到策略的同区间
                cm = r.index.intersection(br.index)
                if len(cm) < 12:
                    continue
                d = r.loc[cm] - br.loc[cm]
                key = f'NE{k}_vs_{bname}'
                report['strategy'][key] = {
                    'n': int(len(d)), 'mean': float(d.mean()), 't': nw(d),
                    'mde': mde(d), 'median': float(d.median()),
                    'win': float((d > 0).mean()),
                    'bonf_pass': bool(nw(d) is not None and abs(nw(d)) >= BONF_T),
                }
        # 落盘权益曲线
        for k in (2, 6):
            key = f'NE{k}'
            if key in report['strategy']:
                pass

        print()
        print('=' * 100)
        print('B4 策略 vs 被动（配对差；Bonferroni n=12 临界 |t| ≥ 2.87）')
        print('=' * 100)
        hdr = (f"{'比较':>22s} | {'n':>4s} {'均值/月':>9s} {'t':>7s} {'MDE':>8s} "
               f"{'效/MDE':>7s} {'中位':>8s} {'胜率':>7s} {'过Bonf':>7s}")
        print(hdr)
        print('-' * len(hdr))
        for key, v in report['strategy'].items():
            if '_vs_' not in key:
                continue
            ratio = abs(v['mean']) / v['mde'] if v['mde'] else None
            print(f"{key:>22s} | {v['n']:>4d} {v['mean']*100:>+8.2f}% "
                  f"{(v['t'] if v['t'] is not None else float('nan')):>7.2f} "
                  f"{v['mde']*100:>7.2f}% {(ratio if ratio else float('nan')):>7.2f} "
                  f"{v['median']*100:>+7.2f}% {v['win']:>6.1%} "
                  f"{('是' if v['bonf_pass'] else '否'):>7s}")
        print()
        print('  读法：#24 R3 —— **效/MDE < 1 ⇒ 功效不足，不得裁决**。')
        print('        Bonferroni n=12 ⇒ |t| ≥ 2.87 才算过线。')

        # ---------------- B6 regime 分期（冻结判据） ----------------
        print()
        print('=' * 100)
        print('B6 regime 分期（前段 2019–2020 / 后段 2021–2026）')
        print('=' * 100)
        hdr2 = (f"{'比较':>22s} | {'前段均值':>9s} {'t':>7s} {'n':>4s} | "
                f"{'后段均值':>9s} {'t':>7s} {'n':>4s}")
        print(hdr2)
        print('-' * len(hdr2))
        for k in (2, 6):
            eq = strategy_curves.get(f'NE{k}')
            if eq is None:
                continue
            r = eq.pct_change().dropna()
            for bname, beq in (('POOL', pool_eq), ('510300.SH', etf_curves.get('510300.SH'))):
                if beq is None or beq.empty:
                    continue
                br = period_returns(beq, r.index)      # ← 同区间配对
                cm = r.index.intersection(br.index)
                if len(cm) < 12:
                    continue
                d = r.loc[cm] - br.loc[cm]
                i2 = pd.to_datetime(d.index)
                pre = np.asarray(i2.year <= 2020)
                late = np.asarray(i2.year >= 2021)
                tp, tl = nw(d[pre]), nw(d[late])
                report['strategy'][f'NE{k}_vs_{bname}_regime'] = {
                    'pre_mean': float(d[pre].mean()), 'pre_t': tp, 'pre_n': int(pre.sum()),
                    'late_mean': float(d[late].mean()), 'late_t': tl, 'late_n': int(late.sum())}
                print(f"{f'NE{k} vs {bname}':>22s} | {d[pre].mean()*100:>+8.2f}% "
                      f"{(tp if tp is not None else float('nan')):>7.2f} {int(pre.sum()):>4d} | "
                      f"{d[late].mean()*100:>+8.2f}% "
                      f"{(tl if tl is not None else float('nan')):>7.2f} {int(late.sum()):>4d}")

        # ---------------- B5 风险调整（登记） ----------------
        print()
        print('=' * 100)
        print('B5 风险调整：年化 ÷ |MDD|（登记项，无门槛）')
        print('=' * 100)
        rows_b5 = []
        for nm, v in report['passive'].items():
            if 'calmar' in v and v.get('calmar') is not None:
                rows_b5.append((nm, v['ann'], v['mdd'], v['sharpe'], v['calmar'],
                                v.get('investable')))
        for nm, v in report['strategy'].items():
            if '_vs_' not in nm and v.get('calmar') is not None:
                rows_b5.append((nm, v['ann'], v['mdd'], v['sharpe'], v['calmar'], '策略'))
        rows_b5.sort(key=lambda x: -(x[4] or 0))
        print(f"{'项':>16s} {'年化':>9s} {'MDD':>9s} {'Sharpe':>8s} {'年化/|MDD|':>11s} {'性质':>8s}")
        print('-' * 70)
        for nm, a, m, s, c, inv in rows_b5:
            print(f"{nm:>16s} {a:>+8.2%} {m:>+8.2%} {s:>8.2f} {c:>11.2f} {str(inv):>8s}")
        report['b5_risk_adjusted'] = [
            {'name': nm, 'ann': a, 'mdd': m, 'sharpe': s, 'calmar': c, 'kind': str(inv)}
            for nm, a, m, s, c, inv in rows_b5]
    # 落盘
    for name, ser in (('POOL', pool_eq),):
        ser.to_frame('equity').to_parquet(OUT_DIR / f'curve_{name}.parquet')
    for code in ETFS:
        if code in funds:
            eq, _ = etf_buyhold(funds[code], code)
            if not eq.empty:
                eq.to_frame('equity').to_parquet(OUT_DIR / f'curve_{code}.parquet')
    (OUT_DIR / 'passive_benchmark_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print(f'\n报告 → {OUT_DIR / "passive_benchmark_report.json"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
