"""判据链 #31 执行器：**B 判弱资产化 + C 前瞻累积**。

预注册：plans/avoid-value-prereg-20261011.md（**先于本脚本跑数冻结**，commit 5899c26）

## 两件事
- **B 判弱资产化**：把"这只不行"变成可用的**回避名单**，并量化**回避的净价值 vs 成本**。
- **C 前瞻累积**：接生产每日打分（`D:/Final/data/state/fterm.db` 的 `ema_state`），
  **逐日留痕**，为"滚动命中率是否稳定"提供真前瞻证据。

## 纪律
- **只做减法**：不产出买入名单（#30 已证判强侧仅捕获 47.8% 天花板、top-10 内 IC 为负）；
- **只读生产**：生产 DB 用 `mode=ro`；产物只落 worktree；
- 成本用 `fterm/contracts.py` 唯一口径；基准 = 当日全市场跑赢率（同 #30）。
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

from fterm import contracts                                # noqa: E402
from fterm.config import load_config                        # noqa: E402
from fterm.data import ashare as sa                         # noqa: E402
from fterm.features.build import build_dataset              # noqa: E402
from fterm.model.stats import newey_west_t                  # noqa: E402

PROD = Path(r'D:\Final\data\reports')
PROD_DB = Path(r'D:\Final\data\state\fterm.db')
OUT_DIR = ROOT / 'experiments' / 'avoid_value_20261011'
TRACK = ROOT / 'data' / 'avoid_tracking'
PANEL_END = pd.Timestamp('2026-09-30')
START = pd.Timestamp('2019-01-02')
Z2 = 1.959963984540054 + 0.8416212335729143
N_DECILE = 10

AVOID_10 = 0.10          # 最强回避：预测秩 ≤ 10%
AVOID_30 = 0.30
WATCH = 0.50


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
    d1 = pd.read_parquet(PROD / 'rankloss_ss7_21_99_5_13.parquet',
                         columns=['date', 'product', 'pred'])
    d2 = pd.read_parquet(PROD / 'rankloss_ss123_2024_42_314_2027.parquet',
                         columns=['date', 'product', 'pred'])
    for x in (d1, d2):
        x['date'] = pd.to_datetime(x['date'])
    P = (d1.pivot_table(index='date', columns='product', values='pred')
         .add(d2.pivot_table(index='date', columns='product', values='pred'), fill_value=0) / 2)
    return P.astype('float64').sort_index()


def cost_bp(notional: float, price: float = 14.0) -> float:
    """单票往返成本（bp of 该票金额）—— 用 contracts 唯一口径。

    ⚠ 首版自伤（已修）：原先硬编码"1 手 × 14 元 = 1400 元"⇒ 5 元佣金地板
    摊不开，往返 **76.6bp**。但 **2 万档 top-2 的单票预算 ≈ 9500 元（6 手）**
    ⇒ 实际往返仅 **17.1bp**。用 1400 元会把成本**高估 4.5 倍**、把结论**判负**。

    ⇒ 成本必须按**实际单票金额**算（金额越大，bp 越低）。
    """
    lots = max(1, int(notional // (price * 100)))
    amt = price * 100 * lots
    sell_fee = contracts.trade_fee(amt, -1, pd.Timestamp('2025-01-01'))
    buy_fee = contracts.trade_fee(amt, +1, pd.Timestamp('2025-01-01'))
    return float((sell_fee + buy_fee) / amt * 1e4)


# 2 万档 top-2 的实际单票预算（initial_equity × max_cash_util / top_n）
NOTIONAL_2W_TOP2 = 20000 * 0.95 / 2      # = 9,500 元


def backfill() -> int:
    """B：全史回避价值标定。"""
    cfg = load_config(ROOT / 'config.yaml')
    panels = sa.load_panels(cfg)
    bundle = build_dataset(cfg, compute_fingerprints=False)
    dates = panels['close'].index
    assert dates.max() <= PANEL_END, f'G2 面板竞态 {dates.max()}'

    P = load_streams()
    uni = bundle.universe
    codes = [c for c in P.columns if c in bundle.labels['fwd_ret'].columns]
    o = panels['open'][codes].astype('float64')
    adj = panels['adj'][codes].astype('float64')
    hfq = (o * adj)

    idx = [d for d in dates if d >= START and d in P.index and d in hfq.index]
    mask = uni.reindex(index=idx, columns=codes).fillna(False)
    Pm = P.reindex(index=idx, columns=codes).where(mask)

    report = {'prereg': 'plans/avoid-value-prereg-20261011.md',
              'window': [str(idx[0].date()), str(idx[-1].date())],
              'n_days': len(idx), 'horizons': {}, 'gates': {}}

    rng = np.random.default_rng(20261011)

    for H in (1, 5):
        fwd = hfq.shift(-(H + 1)) / hfq.shift(-1) - 1.0
        Ym = fwd.reindex(idx).where(mask)

        rows = []
        for dt in idx:
            p = Pm.loc[dt].dropna()
            y = Ym.loc[dt].reindex(p.index).dropna()
            cm = p.index.intersection(y.index)
            if len(cm) < 500:
                continue
            pv, yv = p[cm], y[cm]
            ex = yv - yv.mean()                     # G7 基准 = 当日截面均值
            base_win = float((ex > 0).mean())
            rank = pv.rank(pct=True)
            n = len(pv)

            rec = {'date': dt, 'n': n, 'base_win': base_win}
            for lab, thr in (('avoid10', AVOID_10), ('avoid30', AVOID_30),
                             ('watch', WATCH)):
                sel = rank <= thr
                if lab == 'watch':
                    sel = (rank > AVOID_30) & (rank <= WATCH)
                sub_ex = ex[sel]
                # 对照组：同规模随机组
                k = int(sel.sum())
                if k < 5:
                    continue
                ctrl = rng.choice(n, k, replace=False)
                rec[f'{lab}_exc'] = float(sub_ex.mean())
                rec[f'{lab}_win'] = float((sub_ex > 0).mean())
                rec[f'{lab}_ctrl_exc'] = float(ex.iloc[ctrl].mean())
                rec[f'{lab}_k'] = k
            rows.append(rec)

        D = pd.DataFrame(rows).set_index('date')
        if D.empty:
            continue
        D.index = pd.to_datetime(D.index)
        D = D.sort_index()

        seg = {}
        yrs = D.index.year
        for sname, m in (('full', np.ones(len(D), bool)),
                         ('pre', np.asarray(yrs <= 2020)),
                         ('late', np.asarray(yrs >= 2021))):
            e = {}
            for lab in ('avoid10', 'avoid30', 'watch'):
                col = f'{lab}_exc'
                if col not in D:
                    continue
                v = D[col].to_numpy()[m]
                v = v[np.isfinite(v)]
                e[lab] = {'mean': float(v.mean()) if len(v) else None,
                          't': nw(v), 'mde': mde(v), 'n': int(len(v)),
                          'win': float(D[f'{lab}_win'].to_numpy()[m].mean())}
            seg[sname] = e
        report['horizons'][f'h{H}'] = {'h': H, 'n_days': int(len(D)), 'seg': seg,
                                       'mean_base_win': float(D['base_win'].mean())}

        # 逐日序列落盘（供复核）
        D.to_parquet(OUT_DIR / f'avoid_daily_h{H}.parquet')

        # ---- 成本覆盖（B2）----
        # 用 **2 万档 top-2 的实际单票预算**（9500 元 ⇒ 往返 17.1bp）
        rt_bp = cost_bp(NOTIONAL_2W_TOP2)
        # ⚠ 场景诚实性（本链的核心限定）：
        #   回避名单 = 最弱 10%；而 top-2 策略买的是**最强 2 只** ⇒ 两集合**不相交**。
        #   ⇒ 回避名单对"选股"**没有直接作用**；唯一真实场景是 **EXIT（卖出规则）**。
        val10 = abs(seg['full']['avoid10']['mean'] or 0) * 1e4
        val30 = abs(seg['full']['avoid30']['mean'] or 0) * 1e4
        report.setdefault('cost_cover', {})[f'h{H}'] = {
            'notional_per_name': NOTIONAL_2W_TOP2,
            'roundtrip_cost_bp': rt_bp,
            'avoid10_value_bp': val10,
            'avoid10_cover_ratio': val10 / rt_bp if rt_bp > 0 else None,
            'avoid30_value_bp': val30,
            'avoid30_cover_ratio': val30 / rt_bp if rt_bp > 0 else None,
            'scenario': 'EXIT（跌入名单就卖）—— 非选股',
        }

    (OUT_DIR / 'avoid_value_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    _print_b(report)
    return 0


def _print_b(report: dict) -> None:
    print('=' * 100)
    print('B · 判弱资产化：**回避的净价值**（判据链 #31）')
    print('=' * 100)
    print(f"窗口 {report['window'][0]} → {report['window'][1]}，{report['n_days']} 交易日")
    print()
    print('  ⚠ 本链**只做减法**：不产出买入名单（#30 已证判强侧仅捕获 47.8% 天花板）。')
    for tag, v in report['horizons'].items():
        print()
        print('=' * 100)
        print(f"持有期 {v['h']} 交易日")
        print('=' * 100)
        hdr = f"  {'档':>12s} | {'全样本超额':>12s} {'t':>7s} | {'前段':>10s} | {'后段':>10s} | {'跑输率':>8s}"
        print(hdr)
        print('  ' + '-' * (len(hdr) - 2))
        for lab, name in (('avoid10', 'AVOID-10'), ('avoid30', 'AVOID-30'),
                          ('watch', 'WATCH 30-50')):
            f = v['seg']['full'].get(lab)
            p = v['seg']['pre'].get(lab)
            l = v['seg']['late'].get(lab)
            if not f:
                continue
            print(f"  {name:>12s} | {f['mean']*1e4:>+11.1f}bp {f['t'] or 0:>+7.2f} | "
                  f"{(p['mean']*1e4 if p else float('nan')):>+9.1f}bp | "
                  f"{(l['mean']*1e4 if l else float('nan')):>+9.1f}bp | "
                  f"{f['win']:>7.1%}")
    if 'cost_cover' in report:
        print()
        print('=' * 100)
        print('B2 成本覆盖（**≥1.0 才算可用**）—— 场景 = EXIT（跌入名单就卖）')
        print('=' * 100)
        print('  ⚠ 场景诚实性：回避名单 = 最弱 10%，而 top-2 买的是**最强 2 只**')
        print('    ⇒ 两集合**不相交** ⇒ 回避名单对"选股"**没有直接作用**；')
        print('      唯一真实场景是 **EXIT（持仓衰减进名单 ⇒ 卖出）**。')
        print()
        print(f"  {'持有期':>6s} {'单票金额':>9s} {'往返成本bp':>10s} "
              f"{'AVOID10价值':>11s} {'覆盖比':>8s} {'AVOID30价值':>11s} {'覆盖比':>8s}")
        print('  ' + '-' * 76)
        for k, c in report['cost_cover'].items():
            print(f"  {k:>6s} {c['notional_per_name']:>9,.0f} {c['roundtrip_cost_bp']:>10.1f} "
                  f"{c['avoid10_value_bp']:>11.1f} {c['avoid10_cover_ratio']:>8.2f} "
                  f"{c['avoid30_value_bp']:>11.1f} {c['avoid30_cover_ratio']:>8.2f}")
        print()
        print('  ⚠ 卖出后资金去哪？（本表按"换成市场组合"计）')
        print('    若闲置为现金，还须减去市场收益（机会成本）——见预注册 §六-2。')


def track() -> int:
    """C：接生产每日打分，追加一行前瞻留痕（幂等）。"""
    TRACK.mkdir(parents=True, exist_ok=True)
    f_track = TRACK / 'avoid_tracking.jsonl'

    if not PROD_DB.exists():
        print(f'生产 state 不存在：{PROD_DB}')
        return 1
    # G1 只读
    con = sqlite3.connect('file:' + str(PROD_DB) + '?mode=ro', uri=True)
    kv = dict(con.execute('select k, val from ema_state').fetchall())
    con.close()
    if 'ema' not in kv or 'ema_asof' not in kv:
        print('ema_state 缺少 ema / ema_asof')
        return 1
    scores = json.loads(kv['ema'])
    asof = pd.Timestamp(kv['ema_asof'])
    print(f'生产打分 as-of {asof.date()}，{len(scores)} 只')

    # G3 无前视：as-of 不得晚于"今天"
    today = pd.Timestamp.now().normalize()
    if asof > today:
        print(f'G3 失败：as-of {asof.date()} 晚于今天')
        return 1

    s = pd.Series(scores, dtype='float64')
    rank = s.rank(pct=True)
    rec = {
        'asof': str(asof.date()),
        'n_stocks': int(len(s)),
        'avoid10_n': int((rank <= AVOID_10).sum()),
        'avoid30_n': int((rank <= AVOID_30).sum()),
        'watch_n': int(((rank > AVOID_30) & (rank <= WATCH)).sum()),
        # 名单哈希（不存全名单，省空间；可复现）
        'avoid10_hash': __import__('hashlib').sha256(
            ','.join(sorted(s.index[rank <= AVOID_10])).encode()).hexdigest()[:16],
        'recorded_at': pd.Timestamp.now().isoformat(timespec='seconds'),
    }

    # G5 幂等：同 asof 不重复
    existing = []
    if f_track.exists():
        for line in f_track.read_text(encoding='utf-8').splitlines():
            if line.strip():
                try:
                    existing.append(json.loads(line))
                except Exception:
                    pass
    if any(r.get('asof') == rec['asof'] for r in existing):
        print(f'as-of {rec["asof"]} 已有记录 ⇒ 跳过（幂等）')
        return 0

    with f_track.open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f'已追加：{rec}')
    print(f'累计 {len(existing) + 1} 条 → {f_track}')
    print()
    print('  ⚠ 前瞻段从今天开始累积（生产 ema_state 只有 1 个 as-of，无法回溯）。')
    print('  ⚠ 实现值待结果可知后回填；本脚本只记"当时的预测"，防事后篡改。')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--backfill', action='store_true')
    ap.add_argument('--track', action='store_true')
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.track:
        return track()
    return backfill()


if __name__ == '__main__':
    raise SystemExit(main())
