"""判据链 #28 执行器：NEW ECO 信号的「2 万档可移植性」检验。

预注册：plans/neweco-portability-prereg-20261011.md（**先于本跑数冻结**，commit 05b9f30）

## 本链问什么
用户提议「把 NEW ECO 的模型和 Final 结合走新路」。立案前标定已否决"融合两个模型"
（Final IC 同口径更高；Grinold 界给出上限 = 样本最多减半；融合只在 2 万持不了的宽篮子有效）。
本链转而检验**唯一剩下的真问题**：两个仓的 α 位置相反 ——
Final 的 α 在宽篮子（2 万不可实施）、NEW ECO 的 α 在窄尾部（2 万可实施）。
⇒ **NEW ECO 的信号能否接上本仓的 2 万档微观结构而存活？**

## 两条关键口径（第一版执行器写错，已修）
1. **被动地板 MKT 必须是"名义"收益序列，不是 2 万档模拟账户**：
   2 万买 400 只根本买不起（只会买到 4 只），那不是等权全A。
   ⇒ 基准按**名义等权篮子**算收益序列，与策略的**实际受限账户**收益序列配对。
2. **持仓盯市必须含权（复权）**：按 `close_raw × adj[d]/adj[entry]` 计（total-return 等效），
   现金侧仍按真实未复权价交收（同 paper60b-prereg §二-2）。
   否则组合收益被系统性低估 **+3.3~3.8%/年**（该项目自己实测的 hfq−raw 差）。

## 纪律
- 只读 NEW ECO 的 `neweco.db`（不写、不改其任何文件）；
- 成本逐笔用本仓 `fterm/contracts.py::trade_fee`（唯一口径来源）；
- 一字板用真实停板价（`data/panel/up_limit/down_limit.parquet`）；
- 忠实账本：现金硬约束、`equity == cash + pos_value` 每次自证；
- 判据/臂/门槛一律取自预注册，**执行器内以常量实现**，跑数中不得改。
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

from fterm import contracts                                   # noqa: E402
from fterm.model.stats import newey_west_t                    # noqa: E402

# ------------------------------------------------------------------ 冻结常量
NEWECO_DB = Path(r'D:/NEW ECO/data/neweco.db')
PANEL = ROOT / 'data' / 'panel'
IDX_DIR = ROOT / 'data' / 'index'
OUT_DIR = ROOT / 'experiments' / 'neweco_portability_20261011'

PANEL_END = pd.Timestamp('2026-09-30')          # 隔离线（预注册 §二）
EQUITY0 = 20000.0
MAX_CASH_UTIL = 0.95
LOT = contracts.LOT_SIZE
MAX_LOTS_PER_SYMBOL = 10
SLIP_TICKS = 1
Z2 = 1.959963984540054 + 0.8416212335729143
BONF_T = 2.50                                   # 预注册 §3.3（主检验数 = 4）
ACTIVE_LO, ACTIVE_HI = pd.Timestamp('2024-01-01'), pd.Timestamp('2026-08-31')
PRE_HI = pd.Timestamp('2020-12-31')
LATE_LO = pd.Timestamp('2021-01-01')
SEEDS_RND = (7, 21, 99)


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


# ------------------------------------------------------------------ 数据
def load_panel():
    print('[data] 载入面板 …', flush=True)
    keys = ('open', 'close', 'adj', 'up_limit', 'down_limit', 'circ_mv')
    px = {}
    for k in keys:
        p = PANEL / f'{k}.parquet'
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index)
        assert df.index.max() <= PANEL_END, f'G1 面板竞态: {k} 右端 {df.index.max()}'
        px[k] = df
    return px


def load_neweco():
    con = sqlite3.connect('file:' + str(NEWECO_DB) + '?mode=ro', uri=True)
    ms = pd.read_sql_query('select ts_code, trade_date, score from model_score', con)
    con.close()
    ms['trade_date'] = pd.to_datetime(ms['trade_date'], format='%Y%m%d')
    return ms.pivot_table(index='trade_date', columns='ts_code', values='score').sort_index()


def load_index(code='000905.SH'):
    df = pd.read_parquet(IDX_DIR / f'{code}.parquet')
    s = df['close'].astype('float64')
    s.index = pd.to_datetime(s.index.astype(str))
    return s.sort_index()


# ------------------------------------------------------------------ 忠实账本
def simulate(targets_by_date, dates_sorted, px, label, cost_mult=1.0):
    """忠实单遍账本（月度调仓、t+1 开盘成交、整手、真实成本、真实一字板、含权盯市）。"""
    op, cl, adj = px['open'], px['close'], px['adj']
    up, dn = px['up_limit'], px['down_limit']
    date_index = {d: i for i, d in enumerate(dates_sorted)}

    cash = EQUITY0
    positions: dict[str, dict] = {}       # code -> {lots, entry_px, entry_adj}
    rows, trades, costs = [], 0, 0.0
    blocked_cash = unfilled = 0

    def oprice(d, code):
        if d not in op.index or code not in op.columns:
            return None
        v = op.at[d, code]
        return float(v) if pd.notna(v) and v > 0 else None

    def cprice(d, code):
        if d not in cl.index or code not in cl.columns:
            return None
        v = cl.at[d, code]
        return float(v) if pd.notna(v) and v > 0 else None

    def adj_at(d, code):
        if d not in adj.index or code not in adj.columns:
            return np.nan
        v = adj.at[d, code]
        return float(v) if pd.notna(v) and v > 0 else np.nan

    def one_word(d, code, which):
        p = oprice(d, code)
        if p is None:
            return False
        tab = up if which == 'up' else dn
        if d not in tab.index or code not in tab.columns:
            return False
        lim = tab.at[d, code]
        return pd.notna(lim) and abs(p - float(lim)) < 1e-9

    def mtm(d):
        """含权盯市（total-return 等效），**按 d 的收盘价**。"""
        tot = 0.0
        for c, v in positions.items():
            p = cprice(d, c)
            if p is None:
                p = v['entry_px']
                return_px = p
            else:
                a = adj_at(d, c)
                ratio = (a / v['entry_adj']) if (np.isfinite(a) and np.isfinite(v['entry_adj'])
                                                 and v['entry_adj'] > 0) else 1.0
                return_px = p * ratio
            tot += return_px * v['lots'] * LOT
        return tot

    def value_at_open(d):
        """持仓按 **d 的开盘价**（含权）计值 —— 执行时点的正确口径。

        ⚠ 为什么必须有这个函数（本链第二处自伤，2026-10-11 修）：
        首版用 `mtm(d)`（**收盘价**）给"执行日开盘要下的单"定预算 ⇒
        **用当日收盘价决定当日开盘买多少手 = 前视**。
        正确做法：执行时点的权益一律用**开盘价**计值。
        """
        tot = 0.0
        for c, v in positions.items():
            p = oprice(d, c)
            if p is None:
                p = v['entry_px']
                val_px = p
            else:
                a = adj_at(d, c)
                ratio = (a / v['entry_adj']) if (np.isfinite(a) and np.isfinite(v['entry_adj'])
                                                 and v['entry_adj'] > 0) else 1.0
                val_px = p * ratio
            tot += val_px * v['lots'] * LOT
        return tot

    anchors = sorted(targets_by_date)
    for a in anchors:
        i = date_index.get(a)
        if i is None or i + 1 >= len(dates_sorted):
            continue
        d = dates_sorted[i + 1]
        target = list(dict.fromkeys(targets_by_date[a]))
        tset = set(target)

        # ① 卖出（不在目标内）
        for code in list(positions):
            if code in tset:
                continue
            p = oprice(d, code)
            if p is None or one_word(d, code, 'down'):
                continue                                   # 卖不出 ⇒ 结转
            lots = positions[code]['lots']
            sell_px = p - SLIP_TICKS * contracts.TICK
            amount = sell_px * lots * LOT
            fee = contracts.trade_fee(amount, -1, d) * cost_mult
            cash += amount - fee
            costs += fee
            trades += 1
            del positions[code]

        # ② 买入
        # ⚠ 执行时点的权益必须用**开盘价**计值（不能用收盘价 mtm ⇒ 前视）
        equity_now = cash + value_at_open(d)
        budget = equity_now * MAX_CASH_UTIL / max(len(target), 1)
        for code in target:
            if code in positions:
                continue
            p = oprice(d, code)
            if p is None or one_word(d, code, 'up'):
                continue                                   # 买不进
            buy_px = p + SLIP_TICKS * contracts.TICK
            lots = min(int(budget // (buy_px * LOT)), MAX_LOTS_PER_SYMBOL)
            if lots <= 0:
                unfilled += 1
                continue
            while lots > 0:
                amount = buy_px * lots * LOT
                fee = contracts.trade_fee(amount, +1, d) * cost_mult
                if amount + fee <= cash:
                    break
                lots -= 1
            if lots <= 0:
                blocked_cash += 1
                continue
            amount = buy_px * lots * LOT
            fee = contracts.trade_fee(amount, +1, d) * cost_mult
            cash -= amount + fee
            costs += fee
            trades += 1
            ea = adj_at(d, code)
            positions[code] = {'lots': lots, 'entry_px': buy_px,
                               'entry_adj': ea if np.isfinite(ea) else 1.0}

        # ③ 记账 + 自证
        pv = mtm(d)
        eq = cash + pv
        assert cash >= -1e-6, f'G4 现金为负 {cash}'
        rows.append({'date': d, 'equity': eq, 'cash': cash, 'pos_value': pv,
                     'n_pos': len(positions)})

    C = pd.DataFrame(rows).set_index('date')
    if C.empty:
        raise RuntimeError(f'{label}: 无权益行')
    return C, {'n_trades': trades, 'total_costs': costs, 'blocked_cash': blocked_cash,
               'unfilled_lots': unfilled, 'final_equity': float(C['equity'].iloc[-1]),
               'total_return': float(C['equity'].iloc[-1] / EQUITY0 - 1.0),
               'deploy_median': float(C['pos_value'].median() / EQUITY0),
               'n_pos_median': float(C['n_pos'].median())}


# ------------------------------------------------------------------ 臂构造
def arm_targets(N, px, k, mv_neutral=False, rng=None):
    mv = px['circ_mv']
    out = {}
    for a in N.index:
        if a not in px['open'].index:
            continue
        row = N.loc[a].dropna()
        codes = [c for c in row.index if c in px['open'].columns]
        if len(codes) < 200:
            continue
        s = row[codes]
        if mv_neutral:
            if a not in mv.index:
                continue
            mvrow = mv.loc[a].reindex(s.index)
            m = np.isfinite(s.to_numpy()) & np.isfinite(mvrow.to_numpy()) & (mvrow.to_numpy() > 0)
            if int(m.sum()) < 200:
                continue
            rmv = pd.Series(mvrow.to_numpy()[m]).rank().to_numpy()
            rsv = pd.Series(s.to_numpy()[m]).rank().to_numpy()
            A = np.column_stack([np.ones_like(rmv), rmv])
            beta, *_ = np.linalg.lstsq(A, rsv, rcond=None)
            resid = rsv - A @ beta
            order = np.argsort(-resid)
            out[a] = list(np.array(s.index)[m][order[:k]])
        elif rng is not None:
            pick = rng.choice(len(s), min(k, len(s)), replace=False)
            out[a] = list(s.index[pick])
        else:
            out[a] = list(s.sort_values(ascending=False).index[:k])
    return out


# ------------------------------------------------------------------ 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='all')
    ap.add_argument('--verify-det', action='store_true')
    ap.add_argument('--cost-mult', type=float, default=1.0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    px = load_panel()
    N = load_neweco()
    print(f'[data] NEW ECO 打分 {N.shape}  {N.index.min().date()} → {N.index.max().date()}',
          flush=True)
    dates_sorted = list(px['open'].index)
    anchors = [a for a in N.index if a in px['open'].index]
    print(f'[data] 可用锚点 {len(anchors)}', flush=True)

    # ---------- 名义基准（不受 2 万约束）----------
    op, cl, adj = px['open'], px['close'], px['adj']
    idx905 = load_index('000905.SH')

    def notional_series(price_tab, anchor_list):
        """名义等权篮子：每期在锚点 t+1 开盘买入全部可用股，持有到下一期 t+1 开盘。"""
        di = {d: i for i, d in enumerate(dates_sorted)}
        out = {}
        prev_ex = None
        for a in anchor_list:
            i = di.get(a)
            if i is None or i + 1 >= len(dates_sorted):
                continue
            d = dates_sorted[i + 1]
            if prev_ex is None:
                prev_ex = d
                continue
            p0 = price_tab.loc[prev_ex]
            p1 = price_tab.loc[d]
            common = p0.index.intersection(p1.index)
            r = (p1[common] / p0[common] - 1.0).replace([np.inf, -np.inf], np.nan).dropna()
            out[d] = float(r.mean()) if len(r) else np.nan
            prev_ex = d
        return pd.Series(out)

    mkt_ret = notional_series(cl * adj, anchors)
    print(f'[bench] MKT 名义等权篮子（**复权收盘价**，含分红，close→close）{len(mkt_ret)} 期',
          flush=True)

    idx_ex = {}
    prev_ex = None
    di = {d: i for i, d in enumerate(dates_sorted)}
    for a in anchors:
        i = di.get(a)
        if i is None or i + 1 >= len(dates_sorted):
            continue
        d = dates_sorted[i + 1]
        if prev_ex is None:
            prev_ex = d
            continue
        s0 = idx905[idx905.index <= prev_ex]
        s1 = idx905[idx905.index <= d]
        if len(s0) and len(s1) and float(s0.iloc[-1]) > 0:
            idx_ex[d] = float(s1.iloc[-1] / s0.iloc[-1] - 1.0)
        prev_ex = d
    idx_ret = pd.Series(idx_ex)
    print(f'[bench] IDX905 {len(idx_ret)} 期', flush=True)

    report = {'prereg': 'plans/neweco-portability-prereg-20261011.md',
              'panel_end': str(PANEL_END), 'equity0': EQUITY0,
              'n_anchors': len(anchors),
              'gates': {'G1_panel_race': 'PASS'}}

    # ---------- 策略臂 ----------
    ARMS = {}
    if args.arm in ('all', 'NE6'):
        ARMS['NE6'] = arm_targets(N, px, 6)
    if args.arm in ('all', 'NE2'):
        ARMS['NE2'] = arm_targets(N, px, 2)
    if args.arm in ('all', 'NE6s'):
        ARMS['NE6s'] = arm_targets(N, px, 6, mv_neutral=True)
    if args.arm in ('all', 'RND6'):
        for sd in SEEDS_RND:
            ARMS[f'RND6_s{sd}'] = arm_targets(N, px, 6, rng=np.random.default_rng(sd))

    res = {}
    for name, tg in ARMS.items():
        print(f'[run] {name} …', flush=True)
        C, meta = simulate(tg, dates_sorted, px, name, cost_mult=args.cost_mult)
        # 逐期收益（相邻执行日之间）
        r = C['equity'].pct_change().dropna()
        res[name] = {'curve': C, 'meta': meta, 'ret': r}
        print(f'      期末 {meta["final_equity"]:,.0f}  总收益 {meta["total_return"]:+.2%}  '
              f'成交 {meta["n_trades"]}  成本 {meta["total_costs"]:,.0f}  '
              f'部署中位 {meta["deploy_median"]:.1%}  持仓中位 {meta["n_pos_median"]:.1f}  '
              f'blocked={meta["blocked_cash"]} 买不起={meta["unfilled_lots"]}', flush=True)

    # ---------- 汇总 ----------
    def seg_stats(e):
        idx = pd.to_datetime(e.index)
        segs = {'full': np.ones(len(e), bool),
                'pre': np.asarray(idx <= PRE_HI),
                'late': np.asarray(idx >= LATE_LO),
                'active': np.asarray((idx >= ACTIVE_LO) & (idx <= ACTIVE_HI))}
        o = {}
        for s, m in segs.items():
            v = np.asarray(e)[m]
            v = v[np.isfinite(v)]
            o[s] = {'n': int(len(v)),
                    'mean': float(v.mean()) if len(v) else None,
                    't': nw(v), 'mde': mde(v),
                    'median': float(np.median(v)) if len(v) else None,
                    'win': float((v > 0).mean()) if len(v) else None,
                    'trim95': float(v[v <= np.quantile(v, 0.95)].mean()) if len(v) else None}
        return o

    print()
    print('=' * 108)
    print('逐期（月度）超额 vs MKT 名义等权篮子')
    print('=' * 108)
    summ = {}
    for name, rr in res.items():
        common = rr['ret'].index.intersection(mkt_ret.index)
        e = rr['ret'].loc[common] - mkt_ret.loc[common]
        s = seg_stats(e)
        common_i = rr['ret'].index.intersection(idx_ret.index)
        e_i = rr['ret'].loc[common_i] - idx_ret.loc[common_i]
        summ[name] = {'meta': rr['meta'], 'seg': s, 'exc905': seg_stats(e_i)}

    hdr = (f"{'臂':>10s} | {'总收益':>9s} {'成本':>7s} {'部署':>6s} {'持仓':>5s} | "
           f"{'全样本超额':>10s} {'t':>6s} | {'后段超额':>9s} {'t':>6s} | "
           f"{'中位':>8s} {'胜率':>6s} {'去极值':>8s}")
    print(hdr)
    print('-' * len(hdr))
    for name, s in summ.items():
        f = s['seg']['full']
        l = s['seg']['late']
        print(f"{name:>10s} | {s['meta']['total_return']:>9.2%} {s['meta']['total_costs']:>7,.0f} "
              f"{s['meta']['deploy_median']:>6.1%} {s['meta']['n_pos_median']:>5.1f} | "
              f"{f['mean']:>10.4f} {f['t'] or 0:>6.2f} | {l['mean']:>9.4f} {l['t'] or 0:>6.2f} | "
              f"{f['median']:>8.4f} {f['win']:>6.1%} {f['trim95']:>8.4f}")

    print()
    print(f"MKT 名义等权篮子：{len(mkt_ret)} 期，累计 {float((1+mkt_ret).prod()-1):+.2%}")
    print(f"IDX905 中证500  ：{len(idx_ret)} 期，累计 {float((1+idx_ret).prod()-1):+.2%}")

    # 分期明细
    print()
    print('=' * 108)
    print('regime 三段明细（C6）')
    print('=' * 108)
    for name, s in summ.items():
        print(f'  {name}:')
        for seg in ('full', 'pre', 'late', 'active'):
            d = s['seg'][seg]
            print(f'    {seg:>7s} n={d["n"]:>3d} 超额 {d["mean"] if d["mean"] is None else format(d["mean"], "+.4f")} '
                  f' t={d["t"] if d["t"] is None else format(d["t"], "+.2f")}  '
                  f'MDE={d["mde"] if d["mde"] is None else format(d["mde"], ".4f")}  '
                  f'中位={d["median"] if d["median"] is None else format(d["median"], "+.4f")}  '
                  f'胜率={d["win"] if d["win"] is None else format(d["win"], ".1%")}')

    report['summary'] = summ
    report['bench'] = {'mkt_periods': int(len(mkt_ret)),
                       'mkt_cum': float((1 + mkt_ret).prod() - 1),
                       'idx905_cum': float((1 + idx_ret).prod() - 1)}
    # 落盘：权益曲线（供定因/复核；不进判据）
    for name, rr in res.items():
        rr['curve'].to_parquet(OUT_DIR / f'curve_{name}.parquet')
    mkt_ret.to_frame('mkt').to_parquet(OUT_DIR / 'bench_mkt.parquet')
    idx_ret.to_frame('idx905').to_parquet(OUT_DIR / 'bench_idx905.parquet')
    (OUT_DIR / 'neweco_portability_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print(f'\n报告 → {OUT_DIR / "neweco_portability_report.json"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
