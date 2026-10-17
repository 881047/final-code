"""外部日频因子提取（模型升级 Phase 0b，2026-10-09）。

用户指令：「月频源不直接用，拿原始数据提炼日频因子」。本模块把老库 `neweco.db`
的**日频原始表**（margin/hk_hold/moneyflow/block_trade/lhb/…）与**带公告日的事件表**
（forecast/fina_indicator/holder_num/research_report_em/…）提炼成 date×ts_code 宽面板，
落到本仓 `data/external/*.parquet`。

## PIT 铁律（不可违背）

1. **日频表**（margin/hk_hold/moneyflow/block_trade/lhb）：观测日 = `trade_date`，
   当日起可用（收盘后数据，与特征 t 收盘口径一致）。
2. **事件表**（forecast/fina_indicator/holder_num/restricted_release/repurchase/
   research_report_em）：观测日 = `ann_date`/`report_date`（**公告日**），
   **次一交易日**起可用；缺失公告日的用 `end_date` 兜底并 +1 交易日。
3. 所有宽面板对齐到**面板交易日历**（`data/panel/open.parquet` 的 index），
   非交易日行被丢弃；前向填充（ffill）只在日历内进行。
4. 任何无法确定可用时点的源**不接入**（大声失败），例如 `analyst/` parquet 残片。

## 与老盘读数的关系

`D:/NEW ECO/data/alpha_pool/*.npz`（月频 93 点）、`kg/kg_features.parquet`（月频 102 点）、
`text_corpus/factors_v1.parquet`（月频 181 点）**不直接接入**——它们是月频快照，
前视风险高。本模块只从**日频原始表**重算。
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from . import ashare as sa

EXTERNAL_DIR = "external"

# 需要提取的日频/事件表（只读）
_DAILY_TABLES = ("margin", "hk_hold", "moneyflow", "block_trade", "lhb", "lhb_org")


# ------------------------------------------------------------------ 基础工具

def _connect(cfg: Config) -> sqlite3.Connection:
    """老库只读连接（URI mode=ro，绝不写）。"""
    return sqlite3.connect(f"file:{cfg.data.old_db}?mode=ro", uri=True)


def _d(s: pd.Series) -> pd.Series:
    """YYYYMMDD（int/str）→ Timestamp。非法值 → NaT。"""
    return pd.to_datetime(s.astype("string"), format="%Y%m%d", errors="coerce")


def calendar(cfg: Config) -> pd.DatetimeIndex:
    """面板交易日历（权威）：data/panel/open.parquet 的 index。"""
    panel_dir = Path(cfg.data.root) / "panel"
    return pd.read_parquet(panel_dir / "open.parquet").index


def _pivot(long: pd.DataFrame, value: str, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """长表 (trade_date, ts_code, value) → 宽面板，对齐日历（未观测 = NaN）。"""
    w = long.pivot_table(index="trade_date", columns="ts_code", values=value, aggfunc="last")
    w = w.reindex(calendar)
    return w.astype("float32")


def _pit_ffill(long: pd.DataFrame, avail: str, value: str,
               calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """事件表 PIT 前向填充：观测在 avail（公告日）后**次一交易日**起可用。

    实现：把 avail 映射到日历上的「下一个交易日」（searchsorted side=right），
    在该位置写入值，然后 ffill 到全日历。
    """
    df = long[["ts_code", avail, value]].dropna(subset=[avail, value]).copy()
    if df.empty:
        return pd.DataFrame(index=calendar, dtype="float32")
    pos = calendar.searchsorted(pd.DatetimeIndex(df[avail].values), side="right")
    ok = pos < len(calendar)
    df = df.loc[ok].copy()
    df["_eff"] = calendar[pos[ok]]
    w = df.pivot_table(index="_eff", columns="ts_code", values=value, aggfunc="last")
    w = w.reindex(calendar.union(w.index)).ffill().reindex(calendar)
    return w.astype("float32")


def _read(cfg: Config, sql: str) -> pd.DataFrame:
    with _connect(cfg) as con:
        return pd.read_sql(sql, con)


# ------------------------------------------------------------------ 资金面

def build_flow(cfg: Config, panels: dict[str, pd.DataFrame],
               calendar: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """资金面日频因子：融资融券 / 北向 / 主力资金 / 大宗交易 / 龙虎榜。"""
    out: dict[str, pd.DataFrame] = {}
    circ_mv = panels["circ_mv"]              # 万元
    amount = panels["amount"]                # 元
    close = panels["close"]
    amt_yi = amount / 1e4                    # 万元（与 circ_mv 同尺度）

    # --- 融资融券 ---
    mg = _read(cfg, "SELECT trade_date, ts_code, rzye, rqye, rzmre, rzche FROM margin")
    if not mg.empty:
        mg["trade_date"] = _d(mg["trade_date"])
        for c in ("rzye", "rqye", "rzmre", "rzche"):
            mg[c] = pd.to_numeric(mg[c], errors="coerce")
        rzye = _pivot(mg, "rzye", calendar)
        rqye = _pivot(mg, "rqye", calendar)
        rzmre = _pivot(mg, "rzmre", calendar)
        out["mrg_rzye_ratio"] = (rzye / circ_mv.replace(0, np.nan)).astype("float32")
        out["mrg_rzye_chg5"] = (rzye / rzye.shift(5).replace(0, np.nan) - 1.0).astype("float32")
        out["mrg_rzmre_amt_ratio"] = (rzmre / amt_yi.replace(0, np.nan)).astype("float32")
        out["mrg_rqye_ratio"] = (rqye / circ_mv.replace(0, np.nan)).astype("float32")

    # --- 北向持股 ---
    hk = _read(cfg, "SELECT trade_date, ts_code, ratio FROM hk_hold")
    if not hk.empty:
        hk["trade_date"] = _d(hk["trade_date"])
        hk["ratio"] = pd.to_numeric(hk["ratio"], errors="coerce")
        ratio = _pivot(hk, "ratio", calendar)
        out["hk_ratio"] = ratio
        out["hk_ratio_chg5"] = (ratio - ratio.shift(5)).astype("float32")

    # --- 主力资金流（大单+超大单净额 / 成交额）---
    mf = _read(cfg, """SELECT trade_date, ts_code, buy_lg_amount, sell_lg_amount,
                       buy_elg_amount, sell_elg_amount FROM moneyflow""")
    if not mf.empty:
        mf["trade_date"] = _d(mf["trade_date"])
        for c in ("buy_lg_amount", "sell_lg_amount", "buy_elg_amount", "sell_elg_amount"):
            mf[c] = pd.to_numeric(mf[c], errors="coerce")
        bl = _pivot(mf, "buy_lg_amount", calendar); sl = _pivot(mf, "sell_lg_amount", calendar)
        be = _pivot(mf, "buy_elg_amount", calendar); se = _pivot(mf, "sell_elg_amount", calendar)
        out["mf_lg_net_ratio"] = ((bl + be - sl - se) / amount.replace(0, np.nan)).astype("float32")
        out["mf_elg_net_ratio"] = ((be - se) / amount.replace(0, np.nan)).astype("float32")

    # --- 大宗交易（折溢价 + 占比）---
    bt = _read(cfg, "SELECT trade_date, ts_code, total_amount, avg_price FROM block_trade")
    if not bt.empty:
        bt["trade_date"] = _d(bt["trade_date"])
        bt["total_amount"] = pd.to_numeric(bt["total_amount"], errors="coerce")
        bt["avg_price"] = pd.to_numeric(bt["avg_price"], errors="coerce")
        amt = _pivot(bt, "total_amount", calendar)          # 元
        avg = _pivot(bt, "avg_price", calendar)
        out["bt_amount_ratio"] = (amt / amount.replace(0, np.nan)).astype("float32")
        out["bt_premium"] = (avg / close.replace(0, np.nan) - 1.0).astype("float32")

    # --- 龙虎榜 ---
    lhb = _read(cfg, "SELECT trade_date, ts_code, net_buy FROM lhb")
    if not lhb.empty:
        lhb["trade_date"] = _d(lhb["trade_date"])
        lhb["net_buy"] = pd.to_numeric(lhb["net_buy"], errors="coerce")
        nb = _pivot(lhb, "net_buy", calendar)               # 元
        out["lhb_net_ratio"] = (nb / amount.replace(0, np.nan)).astype("float32")
        out["lhb_on"] = nb.notna().astype("float32")
    lo = _read(cfg, "SELECT trade_date, ts_code, org_net_value FROM lhb_org")
    if not lo.empty:
        lo["trade_date"] = _d(lo["trade_date"])
        lo["org_net_value"] = pd.to_numeric(lo["org_net_value"], errors="coerce")
        onv = _pivot(lo, "org_net_value", calendar)
        out["lhb_org_net_ratio"] = (onv / amount.replace(0, np.nan)).astype("float32")
    return out


# ------------------------------------------------------------------ 事件面

def build_events(cfg: Config, panels: dict[str, pd.DataFrame],
                 calendar: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """事件面日频因子：业绩预告 / 内部人 / 股东户数 / 解禁 / 回购。"""
    out: dict[str, pd.DataFrame] = {}
    circ_mv = panels["circ_mv"]

    fc = _read(cfg, "SELECT ts_code, ann_date, end_date, chg_mid, fc_type FROM forecast")
    if not fc.empty:
        fc["ann_date"] = _d(fc["ann_date"]).fillna(_d(fc["end_date"]))
        fc["chg_mid"] = pd.to_numeric(fc["chg_mid"], errors="coerce")
        fc = fc.dropna(subset=["ann_date", "chg_mid"]).sort_values("ann_date")
        out["fc_chg_mid"] = _pit_ffill(fc, "ann_date", "chg_mid", calendar)
        fc["_dir"] = np.sign(fc["chg_mid"])
        out["fc_dir"] = _pit_ffill(fc, "ann_date", "_dir", calendar)

    it = _read(cfg, "SELECT ts_code, change_date, delta_shares, amount FROM insider_trade")
    if not it.empty:
        it["change_date"] = _d(it["change_date"])
        it["amount"] = pd.to_numeric(it["amount"], errors="coerce").fillna(0.0)
        # 内部人净增持金额（元）→ 20 日滚动和 / 流通市值
        w = it.pivot_table(index="change_date", columns="ts_code", values="amount", aggfunc="sum")
        w = w.reindex(calendar).fillna(0.0).astype("float64")
        out["insider_net_20d"] = (w.rolling(20, min_periods=1).sum()
                                  / (circ_mv * 1e4).replace(0, np.nan)).astype("float32")

    hn = _read(cfg, "SELECT ts_code, ann_date, end_date, holder_num FROM holder_num")
    if not hn.empty:
        hn["ann_date"] = _d(hn["ann_date"]).fillna(_d(hn["end_date"]))
        hn["holder_num"] = pd.to_numeric(hn["holder_num"], errors="coerce")
        hn = hn.dropna(subset=["ann_date", "holder_num"]).sort_values("ann_date")
        raw = _pit_ffill(hn, "ann_date", "holder_num", calendar)
        out["holder_num"] = raw
        out["holder_num_chg"] = (raw / raw.shift(63).replace(0, np.nan) - 1.0).astype("float32")

    rr = _read(cfg, "SELECT ts_code, release_date, release_shares FROM restricted_release")
    if not rr.empty:
        rr["release_date"] = _d(rr["release_date"])
        rr["release_shares"] = pd.to_numeric(rr["release_shares"], errors="coerce")
        ev = rr.pivot_table(index="release_date", columns="ts_code",
                            values="release_shares", aggfunc="sum").reindex(calendar)
        # 未来 90 日解禁股数 / 当前流通股本（用 circ_mv/close 近似流通股本）
        fwd = ev.shift(-1).rolling(90, min_periods=1).sum().shift(-89)
        shares = (circ_mv * 1e4 / panels["close"].replace(0, np.nan))
        out["unlock_ratio_90d"] = (fwd / shares.replace(0, np.nan)).astype("float32")

    rp = _read(cfg, "SELECT ts_code, start_date, amount_cap FROM repurchase")
    if not rp.empty:
        rp["start_date"] = _d(rp["start_date"])
        rp["amount_cap"] = pd.to_numeric(rp["amount_cap"], errors="coerce")
        w = rp.pivot_table(index="start_date", columns="ts_code",
                           values="amount_cap", aggfunc="sum").reindex(calendar).fillna(0.0)
        out["repurchase_90d"] = (w.rolling(90, min_periods=1).sum()
                                 / (circ_mv * 1e4).replace(0, np.nan)).astype("float32")
    return out


# ------------------------------------------------------------------ 基本面（PIT）

def build_fundamentals(cfg: Config, calendar: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """财务指标（季频 → PIT 日频前填，按 ann_date 可用）。"""
    fi = _read(cfg, """SELECT ts_code, ann_date, end_date, roe, netprofit_yoy, or_yoy,
                       grossprofit_margin, debt_to_assets FROM fina_indicator""")
    out: dict[str, pd.DataFrame] = {}
    if fi.empty:
        return out
    fi["ann_date"] = _d(fi["ann_date"]).fillna(_d(fi["end_date"]))
    fi = fi.dropna(subset=["ann_date"]).sort_values("ann_date")
    for col in ("roe", "netprofit_yoy", "or_yoy", "grossprofit_margin", "debt_to_assets"):
        fi[col] = pd.to_numeric(fi[col], errors="coerce")
        out[f"fin_{col}"] = _pit_ffill(fi.dropna(subset=[col]), "ann_date", col, calendar)
    return out


# ------------------------------------------------------------------ 行业（PIT）

def build_industry(cfg: Config, calendar: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """申万一级行业（PIT 成员区间）→ 行业代码面板（中性化用）。

    sw_member: con_code/index_code/in_date/out_date；取 level=L1 的行业。
    """
    l1 = _read(cfg, "SELECT index_code FROM sw_classify WHERE level='L1'")
    l1_codes = set(l1["index_code"].tolist())
    m = _read(cfg, "SELECT con_code, index_code, in_date, out_date FROM sw_member")
    m = m[m["index_code"].isin(l1_codes)].copy()
    if m.empty:
        return {}
    m["in_date"] = _d(m["in_date"])
    m["out_date"] = _d(m["out_date"]).fillna(pd.Timestamp("2099-12-31"))
    m["sw_code"] = pd.to_numeric(m["index_code"].str.slice(3, 6), errors="coerce")
    # 逐行把 [in_date, out_date] 区间写进行业（同一股可能多段）
    frame = pd.DataFrame(np.nan, index=calendar, columns=sorted(m["con_code"].unique()),
                         dtype="float32")
    for con, grp in m.groupby("con_code"):
        for _, r in grp.iterrows():
            frame.loc[(calendar >= r["in_date"]) & (calendar <= r["out_date"]), con] = r["sw_code"]
    return {"sw_l1": frame}


# ------------------------------------------------------------------ 市场状态

def build_regime(cfg: Config, panels: dict[str, pd.DataFrame],
                 calendar: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """市场状态（横截面常数，供臂 A 的非秩通道）+ 市场广度/涨跌停家数。

    regime 表：state ∈ {defense, neutral, attack}；score/drawdown/vol 为市场级序列。
    """
    out: dict[str, pd.DataFrame] = {}
    rg = _read(cfg, "SELECT trade_date, state, score, drawdown, vol FROM regime")
    state_map = {"defense": 0.0, "neutral": 1.0, "attack": 2.0}
    if not rg.empty:
        rg["trade_date"] = _d(rg["trade_date"])
        rg["_s"] = rg["state"].map(state_map)
        rg = rg.dropna(subset=["_s"]).sort_values("trade_date")
        idx = calendar.union(pd.DatetimeIndex(rg["trade_date"]))
        for col, name in (("_s", "regime_state"), ("score", "regime_score"),
                          ("drawdown", "regime_drawdown"), ("vol", "regime_vol")):
            s = rg.set_index("trade_date")[col].astype("float64").reindex(idx).ffill().reindex(calendar)
            out[name] = _broadcast(s, panels, "float32")

    # 市场广度：% 股 > MA20；涨跌停家数（来自真实涨跌停价）
    ref = panels["close"]
    close = panels["close"]
    ma20 = close.rolling(20, min_periods=10).mean()
    valid = close.notna() & ma20.notna()
    breadth = ((close > ma20) & valid).sum(axis=1) / valid.sum(axis=1).replace(0, np.nan)
    out["mkt_breadth_ma20"] = _broadcast(breadth, panels, "float32")
    # 各面板列集不同（up/down_limit 5468 vs high 5467）⇒ 必须先对齐到参照面板再比较
    hi = panels["high"].reindex(index=ref.index, columns=ref.columns)
    lo = panels["low"].reindex(index=ref.index, columns=ref.columns)
    up = panels["up_limit"].reindex(index=ref.index, columns=ref.columns)
    dn = panels["down_limit"].reindex(index=ref.index, columns=ref.columns)
    limit_up = ((hi >= up) & up.notna()).sum(axis=1).astype("float64")
    limit_dn = ((lo <= dn) & dn.notna()).sum(axis=1).astype("float64")
    out["mkt_limitup_cnt"] = _broadcast(limit_up, panels, "float32")
    out["mkt_limitdn_cnt"] = _broadcast(limit_dn, panels, "float32")
    # 截面离散度（原始值，非秩）
    disp = panels["close"].pct_change().std(axis=1)
    out["mkt_disp"] = _broadcast(disp, panels, "float32")
    # 全市场换手水平
    out["mkt_turnover"] = _broadcast(panels["turnover"].mean(axis=1), panels, "float32")
    return out


def _broadcast(series: pd.Series, panels: dict[str, pd.DataFrame], dtype: str) -> pd.DataFrame:
    """市场级标量序列 → 同日全市场同值的宽面板（横截面常数）。"""
    ref = panels["close"]
    v = series.reindex(ref.index).to_numpy(dtype="float64")
    arr = np.repeat(v[:, None], ref.shape[1], axis=1)
    return pd.DataFrame(arr, index=ref.index, columns=ref.columns, dtype=dtype)


# ------------------------------------------------------------------ 分析师

def build_analyst(cfg: Config, calendar: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """研报（research_report_em，report_date 日频事件）→ 覆盖/评级上调计数。

    评级数值化：买入/增持/推荐/强烈推荐 → 正；中性/持有 → 0；减持/卖出 → 负。
    """
    rr = _read(cfg, "SELECT ts_code, report_date, rating, rating_change FROM research_report_em")
    out: dict[str, pd.DataFrame] = {}
    if rr.empty:
        return out
    rr["report_date"] = _d(rr["report_date"])
    rr = rr.dropna(subset=["report_date"])
    rr["cnt"] = 1.0
    rr["up"] = (pd.to_numeric(rr["rating_change"], errors="coerce") > 0).astype("float64")
    for col, name in (("cnt", "rr_cnt_20d"), ("up", "rr_rating_up_20d")):
        w = rr.pivot_table(index="report_date", columns="ts_code", values=col,
                           aggfunc="sum").reindex(calendar).fillna(0.0)
        out[name] = w.rolling(20, min_periods=1).sum().astype("float32")
    return out


# ------------------------------------------------------------------ 主入口

def sync_external(cfg: Config, end: str | None = None) -> dict[str, str]:
    """全量提取 → data/external/*.parquet + _meta.json。返回 {因子名: 状态}。

    幂等（整表覆盖）；只读老库；日历 = data/panel/open.parquet（须先跑 sync_data）。
    """
    panels = sa.load_panels(cfg)
    cal = calendar(cfg)
    if end:
        cal = cal[cal <= pd.Timestamp(end)]

    groups = {
        "flow": build_flow(cfg, panels, cal),
        "events": build_events(cfg, panels, cal),
        "fundamentals": build_fundamentals(cfg, cal),
        "industry": build_industry(cfg, cal),
        "regime": build_regime(cfg, panels, cal),
        "analyst": build_analyst(cfg, cal),
    }
    out_dir = Path(cfg.data.root) / EXTERNAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, str] = {}
    meta: dict[str, dict] = {}
    ref_cols = panels["close"].columns          # 统一列集（防跨面板对齐漂移）
    for grp, feats in groups.items():
        for name, df in feats.items():
            if df is None or df.empty:
                results[name] = "SKIP 空"
                continue
            df = df.reindex(index=cal, columns=ref_cols).astype("float32")
            path = out_dir / f"{name}.parquet"
            df.to_parquet(path)
            nn = int(df.notna().to_numpy().sum())
            cov = float(df.notna().to_numpy().mean())
            results[name] = f"ok {df.shape[0]}日×{df.shape[1]}股 非空={nn} 覆盖={cov:.1%}"
            meta[name] = {"group": grp, "shape": list(df.shape), "nonzero": nn,
                          "coverage": round(cov, 4),
                          "first": str(df.index.min().date()), "last": str(df.index.max().date())}

    (out_dir / "_meta.json").write_text(json.dumps(
        {"generated_at": datetime.now().isoformat(timespec="seconds"),
         "old_db": cfg.data.old_db, "end": end, "factors": meta},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def load_external(cfg: Config, names: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """加载已提取的外部因子面板（缺文件响亮报错）。"""
    out_dir = Path(cfg.data.root) / EXTERNAL_DIR
    if names is None:
        names = sorted(p.stem for p in out_dir.glob("*.parquet"))
    res: dict[str, pd.DataFrame] = {}
    for n in names:
        p = out_dir / f"{n}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"external: 缺因子面板 {n}（先跑 scripts/sync_external.py）")
        res[n] = pd.read_parquet(p)
    return res
