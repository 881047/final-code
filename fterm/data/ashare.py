"""A股数据层 —— 老库 neweco.db 只读契约（同 W2 的 L1：只读不写，本仓不 import 老仓代码）。

同步模式：把老库的日线事实表拉成本仓的宽面板（date × ts_code），落 data/panel/：
    open/high/low/close   未复权日线（元）
    adj                   复权因子（后复权价 = 未复权 × adj，分红送转全程正确）
    amount                成交额（元）   vol        成交量（手）
    turnover              换手率（%）   circ_mv    流通市值（万元）
    up_limit/down_limit   交易所真实涨跌停价（老库 2017 起全量；此前 NaN → 撮合层回退近似）
    st_flag               ST/退市风险 PIT 掩码（namechange 逐段展开）

为什么用面板而不是 per-stock 长表：特征全是"逐列滚动 + 逐行截面"两族，
宽表一次向量化算完全市场，且截面 rank/离散度天然免对齐。

PIT 纪律：面板本身是历史事实表（快照时点拉全史），特征层只用 t 及以前；
股票池过滤（上市天数/ST/流动性）逐交易日按当日可得信息判定。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config

# 面板字段 → 老库表.列（唯一口径来源）
PANEL_FIELDS = ("open", "high", "low", "close", "amount", "vol", "turnover",
                "circ_mv", "up_limit", "down_limit", "adj", "st_flag")

_DAILY_COLS = {"open", "high", "low", "close", "amount", "vol"}
_BASIC_COLS = {"turnover": "turnover_rate", "circ_mv": "circ_mv"}


def _connect_ro(cfg: Config) -> sqlite3.Connection:
    p = Path(cfg.data.old_db)
    assert p.exists(), f"老库不存在: {p}（只读契约，本仓不负责建库）"
    return sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True)


def _codes_at_sync(cfg: Config, conn: sqlite3.Connection) -> list[str]:
    """同步候选 = 全部沪深 A（含历史上退市的——防幸存者偏差）；BJ 按配置剔除。"""
    rows = conn.execute("SELECT ts_code, market FROM stock_basic").fetchall()
    out = []
    for code, _market in rows:
        if cfg.universe.exclude_bj and code.endswith(".BJ"):
            continue
        if not (code.endswith(".SZ") or code.endswith(".SH")):
            continue
        out.append(code)
    return sorted(set(out))


def sync_panels(cfg: Config, end: str | None = None) -> dict[str, str]:
    """老库 → 本仓面板全量同步（幂等覆盖）。返回 {field: 状态描述}。

    2026-10-08 P0-B：面板索引一律以老库 `trade_cal(is_open=1)` 为准。
    老库源表里存在**休市日脏行**（实测 `adj_factor` 有 689009.SH 在 20210920/21
    各 1 行，而 `daily`/`stk_limit` 当日 0 行）⇒ `adj` 面板曾长期比其余面板多
    2 行；`hfq()` 做索引并集后在 2021-09 中部插进 2 个全 NaN 幻象交易日，使
    hfq 派生特征的 rolling 窗与标签 `shift` 按"位置"而非交易日错位（实测该窗
    `pvol_20` 少 76,282 个合格单元格、`fwd_ret` 整行少 4,004 行），并触发
    `fold_frame` 的严格 index 断言 ⇒ 训练/研究链直接 ValueError。
    """
    conn = _connect_ro(cfg)
    try:
        start = cfg.data.start.replace("-", "")
        end = (end or pd.Timestamp.now().strftime("%Y%m%d")).replace("-", "")
        codes = _codes_at_sync(cfg, conn)
        code_ph = ",".join("?" * len(codes))
        results: dict[str, str] = {}
        drop_notes: list[str] = []

        # ---- 交易日历（老库 trade_cal）——先取，作为全部面板的索引权威 ----
        cal = pd.read_sql_query(
            "SELECT cal_date FROM trade_cal WHERE is_open = 1 AND cal_date >= ? AND cal_date <= ? "
            "ORDER BY cal_date", conn, params=[start, end])
        assert len(cal) > 0, f"老库 trade_cal 在 [{start},{end}] 内无任何开市日"
        cal_index = pd.DatetimeIndex(pd.to_datetime(cal["cal_date"].astype(str),
                                                    format="%Y%m%d"))
        _meta_dir(cfg).joinpath("trade_cal.txt").write_text(
            "\n".join(cal["cal_date"].astype(str)), encoding="utf-8")
        results["trade_cal"] = f"ok rows={len(cal)}"

        def align(field: str, wide: pd.DataFrame, price_source: bool = False) -> pd.DataFrame:
            return _align_to_cal(wide, cal_index, field, drop_notes, price_source)

        # ---- daily：未复权 OHLC + 量额 ----
        daily = pd.read_sql_query(
            f"SELECT ts_code, trade_date, open, high, low, close, amount, vol FROM daily "
            f"WHERE trade_date >= ? AND trade_date <= ? AND ts_code IN ({code_ph})",
            conn, params=[start, end, *codes])
        _assert_pulled("daily", daily, start)
        close_cal: pd.DatetimeIndex | None = None
        for f in _DAILY_COLS:
            wide = align(f, _pivot(daily, f), price_source=True)
            if f == "close":
                close_cal = wide.index
            _save_panel(cfg, f, wide, results)

        # ---- daily_basic：换手率 + 流通市值 ----
        basic = pd.read_sql_query(
            f"SELECT ts_code, trade_date, turnover_rate, circ_mv FROM daily_basic "
            f"WHERE trade_date >= ? AND trade_date <= ? AND ts_code IN ({code_ph})",
            conn, params=[start, end, *codes])
        _assert_pulled("daily_basic", basic, start)
        _save_panel(cfg, "turnover", align("turnover", _pivot(basic, "turnover_rate")), results)
        _save_panel(cfg, "circ_mv", align("circ_mv", _pivot(basic, "circ_mv")), results)

        # ---- stk_limit：真实涨跌停价 ----
        lim = pd.read_sql_query(
            f"SELECT ts_code, trade_date, up_limit, down_limit FROM stk_limit "
            f"WHERE trade_date >= ? AND trade_date <= ? AND ts_code IN ({code_ph})",
            conn, params=[start, end, *codes])
        _save_panel(cfg, "up_limit", align("up_limit", _pivot(lim, "up_limit")), results)
        _save_panel(cfg, "down_limit", align("down_limit", _pivot(lim, "down_limit")), results)
        n_lim = int(len(lim))
        results["up_limit"] += f"（{'全量覆盖' if n_lim > 0 else '空表：撮合层回退近似'}）"

        # ---- adj_factor ----
        adj = pd.read_sql_query(
            f"SELECT ts_code, trade_date, adj_factor FROM adj_factor "
            f"WHERE trade_date >= ? AND trade_date <= ? AND ts_code IN ({code_ph})",
            conn, params=[start, end, *codes])
        _save_panel(cfg, "adj", align("adj", _pivot(adj, "adj_factor")), results)

        # ---- ST/退市风险 PIT 掩码（namechange 逐段展开）----
        st = _st_mask(cfg, conn, close_cal)
        _save_panel(cfg, "st_flag", st, results)
        results["cal_align"] = ("；".join(drop_notes) if drop_notes
                                else "全部面板索引 ⊆ 交易日历（无丢弃）")
        return results
    finally:
        conn.close()


def _align_to_cal(wide: pd.DataFrame, cal_index: pd.DatetimeIndex, field: str,
                  notes: list[str], price_source: bool = False) -> pd.DataFrame:
    """面板索引 ∩ 交易日历开市日。只删不加（不引入全 NaN 的未来/休市日行）。

    两类"日历外日期"处置不同：
      · **价格源**（daily 的 OHLC/量额）——非开市日却有成交流水 ⇒ 要么老库 trade_cal
        断更、要么日线数据错位，两者都不能静默丢（丢了就等于改写历史）⇒ 响亮失败；
        真实事故里 `daily` 在 20210920/21 是 0 行，故不会误触此闸。
      · **辅助源**（adj_factor / stk_limit / daily_basic）——供应商在节假日插脏行是
        已知噪声（实测 adj_factor 有 689009.SH 两行），丢弃并在 results 留痕。
    """
    extra = wide.index.difference(cal_index)
    if len(extra) == 0:
        return wide
    sample = [str(d.date()) for d in extra[:3]]
    if price_source:
        raise AssertionError(
            f"{field}（价格源）有 {len(extra)} 个日期不在老库 trade_cal(is_open=1)：{sample}"
            f"⇒ 交易日历陈旧或日线数据错位，拒绝静默丢弃真实交易日")
    cells = int(wide.loc[extra].notna().to_numpy().sum())
    notes.append(f"{field}: 丢弃非开市日 {len(extra)} 行（{cells} 个有效单元格）")
    return wide.reindex(wide.index.intersection(cal_index))



def _assert_pulled(table: str, df: pd.DataFrame, start: str) -> None:
    assert len(df) > 0, f"老库 {table} 按 start={start} 拉取为空（检查 data.start 与老库覆盖）"


def _pivot(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """长表 → 宽面板（index=交易日, columns=ts_code, float32）。"""
    sub = df[["ts_code", "trade_date", col]].copy()
    sub[col] = pd.to_numeric(sub[col], errors="coerce").astype("float32")
    sub["trade_date"] = sub["trade_date"].astype(str)
    wide = sub.pivot_table(index="trade_date", columns="ts_code", values=col, aggfunc="last")
    wide.index = pd.to_datetime(wide.index, format="%Y%m%d")
    return wide.sort_index().rename_axis(index=None, columns=None)


def _st_mask(cfg: Config, conn: sqlite3.Connection, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """ST/退市风险逐日掩码。namechange 有记录的股票按 PIT 区间展开；
    无任何记录的股票用当前名单作常数掩码（名字从未变过 ⇒ PIT 等价）。"""
    rows = conn.execute(
        "SELECT n.ts_code, n.name, n.start_date, n.end_date FROM namechange n "
        "ORDER BY n.ts_code, n.start_date").fetchall()
    basic = dict(conn.execute("SELECT ts_code, name FROM stock_basic WHERE name LIKE '%ST%' OR name LIKE '%退%'").fetchall())
    by_code: dict[str, list[tuple[str, str, str]]] = {}
    for code, name, sd, ed in rows:
        by_code.setdefault(code, []).append((str(name), str(sd), None if ed is None else str(ed)))
    # 2026-10-08 性能改造：原为 `mask = pd.DataFrame(0, ..., dtype="int8")` 后**逐列**
    # `mask[code] = flag` —— 5467 次列插入会让 DataFrame 内部块碎片化。
    # 改为先累进预分配 ndarray、最后一次性构造。
    #
    # 同时修掉一个此前未被发现的 dtype 缺陷：原实现里走「无更名记录但当前是 ST/退」
    # 分支的列是 `mask[code] = 1`（Python 标量赋值），pandas 会把它**升位成 int64**，
    # 于是同一张掩码表里 int8 与 int64 混存（实测 7 列里 2 列是 int64）——
    # 那些列内存是 8 倍，且 `mask.to_numpy()` 的 dtype 变得不可预期。
    # 现在全部列统一 int8（0/1 掩码的语义不变，仅内存与 dtype 一致化）。
    cols = sorted({r[0] for r in rows} | set(basic))
    arr = np.zeros((len(dates), len(cols)), dtype="int8")
    col_pos = {c: j for j, c in enumerate(cols)}
    for code in cols:
        j = col_pos[code]
        recs = by_code.get(code)
        if not recs:
            if code in basic:          # 无更名记录但当前是 ST ⇒ 全程剔除（保守，登记于 docs/02）
                arr[:, j] = 1
            continue
        flag = arr[:, j]
        for name, sd, ed in recs:
            if not ("ST" in name.upper() or "退" in name):
                continue
            # ⚠️ 用 pandas 原生比较（asi8 的单位随 dtype us/ms 变化，与 Timestamp.value
            # 的纳秒不可直接比——2026-10-03 单测当场抓到）
            a = pd.Timestamp(sd) if sd and len(sd) == 8 else pd.Timestamp.min
            b = pd.Timestamp(ed) if ed and len(ed) == 8 else pd.Timestamp.max
            seg = np.asarray((dates >= a) & (dates <= b), dtype="int8")
            flag |= seg
    return pd.DataFrame(arr, index=dates, columns=cols, dtype="int8")


# ---------------------------------------------------------------------------
# 面板仓库（store）
# ---------------------------------------------------------------------------
def _panel_dir(cfg: Config) -> Path:
    d = cfg.data.data_dir / "panel"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _meta_dir(cfg: Config) -> Path:
    d = cfg.data.data_dir / "meta"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_panel(cfg: Config, field: str, wide: pd.DataFrame, results: dict[str, str]) -> None:
    wide.astype("float32").to_parquet(_panel_dir(cfg) / f"{field}.parquet")
    results[field] = f"ok {wide.shape[0]}日×{wide.shape[1]}股"


def load_panel(cfg: Config, field: str) -> pd.DataFrame:
    path = _panel_dir(cfg) / f"{field}.parquet"
    assert path.exists(), f"面板 {field} 不存在: {path}（先跑 scripts/sync_data.py）"
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def load_panels(cfg: Config, fields: tuple[str, ...] = PANEL_FIELDS) -> dict[str, pd.DataFrame]:
    return {f: load_panel(cfg, f) for f in fields}


def load_trade_cal(cfg: Config) -> pd.DatetimeIndex:
    p = _meta_dir(cfg) / "trade_cal.txt"
    assert p.exists(), f"交易日历不存在: {p}（先跑 scripts/sync_data.py）"
    return pd.DatetimeIndex([pd.Timestamp(x) for x in
                             p.read_text(encoding="utf-8").splitlines() if x.strip()])


def panel_index_consistency(cfg: Config) -> tuple[bool, str]:
    """跨面板索引一致性（就绪闸第 ⑤ 项的判据）。

    P0-B 教训：`adj` 面板多出 2 个休市日行时，全仓只有研究路径 `fold_frame` 的严格
    断言会报，而生产 runner 走 `predict_day` 根本不读 fold_frame ⇒ 索引漂移静默存在
    数年，直到污染训练样本才被发现。故把"12 张面板索引两两相等且 ⊆ 交易日历"变成
    日链上的显式检查。注：各面板**列集合**本就不同（st_flag 5911 列含已退市代码，
    close 5467 列），本检查只管日期索引；列差异由 `load_panel` 消费方的
    reindex/fillna 语义负责。
    """
    cal = set(load_trade_cal(cfg))
    idxs = {f: load_panel(cfg, f).index for f in PANEL_FIELDS}
    bad: list[str] = []
    for f, ix in idxs.items():
        off = [str(d.date()) for d in ix if d not in cal]
        if off:
            bad.append(f"{f} 含非开市日 {off[:3]}")
    ref = idxs["close"]
    for f, ix in idxs.items():
        if not ix.equals(ref):
            bad.append(f"{f} 索引与 close 不等（{len(ix)} 日 vs {len(ref)} 日）")
    if bad:
        return False, "；".join(bad)
    return True, f"{len(PANEL_FIELDS)} 张面板索引一致（{len(ref)} 日，均 ⊆ 交易日历）"


def daily_chain_ready(cfg: Config, now: pd.Timestamp | None = None,
                      strict_panel_align: bool = False) -> tuple[bool, str]:
    """每日链数据就绪闸（scripts/run_daily_auto.py 专用；人工 --asof 回放不走此处）。

    老库日链断更时 sync_panels 会"成功地"拉到 T-1（全程无异常），而下游幂等闸只防
    "决策超前"不防"数据落后" ⇒ 纸面轨会静默缺一日决策。本闸把这种落后变成响亮拒绝。
    判据（全部以本仓已同步面板 + 交易日历为源）：
      ① close 面板非空；
      ② 交易日历上 ≤now 的最后一个交易日必须被 close 覆盖（老库 trade_cal 预发布
         全年 ⇒ 开市日当晚数据没来时能正确识别，休市日重跑不误伤）；
      ③ up_limit 覆盖到 close 最新日——stk_limit 滞后/空表时撮合层会静默回退
         近似停板价（docs/02 已知限制），不允许在无感降级口径上落决策账；
      ④ 保鲜线：交易日历落后真实时钟 >10 天 ⇒ 老库 trade_cal 本身断更，拒绝
        （正常情况日历预发布到年末，此线只在长期断更时触发）；
      ⑤ 跨面板索引一致且 ⊆ 日历（见 panel_index_consistency）。默认**只警告不阻塞**
        （2026-10-08 落地：前瞻窗进行中，宁可留痕也不可因新加闸打断账本），
        `strict_panel_align=True` 才拒绝落账。
    返回 (ok, 说明)。
    """
    now = (pd.Timestamp(now) if now is not None else pd.Timestamp.now()).normalize()
    try:
        cal = load_trade_cal(cfg)
        close = load_panel(cfg, "close")
    except AssertionError as e:
        return False, f"数据源不可用: {e}"
    if close.empty:
        return False, "close 面板为空（先跑 sync）"
    if len(cal) == 0:
        return False, "交易日历为空"
    if now - cal.max() > pd.Timedelta(days=10):
        return False, (f"交易日历陈旧（最新 {cal.max().date()}，落后时钟 >10 天）"
                       f"⇒ 老库 trade_cal 断更，本日决策不落账")
    expected = cal[cal <= now].max()
    if pd.isna(expected):
        return False, f"交易日历在 {now.date()} 前无任何交易日（检查 data.start）"
    asof = close.index.max()
    if asof < expected:
        return False, (f"面板最新 {asof.date()} < 交易日历应达 {expected.date()} "
                       f"⇒ 老库日链断更/未完成，本日决策不落账")
    up = load_panel(cfg, "up_limit")
    if up.empty:
        return False, f"up_limit 面板为空 ⇒ 撮合将回退近似停板价，拒绝在降级口径上落账"
    if up.index.max() < asof:
        return False, (f"up_limit 最新 {up.index.max().date()} 落后 close {asof.date()} "
                       f"⇒ 撮合将回退近似停板价，拒绝在降级口径上落账")
    ok5, why5 = panel_index_consistency(cfg)
    if not ok5:
        if strict_panel_align:
            return False, f"面板索引不一致 ⇒ {why5}（本日决策不落账）"
        return True, f"就绪：asof={asof.date()} = 交易日历应达 {expected.date()}（⑤警告：{why5}）"
    return True, f"就绪：asof={asof.date()} = 交易日历应达 {expected.date()}（⑤{why5}）"


def df_fingerprint(df: pd.DataFrame) -> str:
    """内容哈希（训练数据集记录来源，复现实验校验输入未漂移）。"""
    import hashlib

    h = hashlib.sha256()
    h.update(str(df.index.astype("int64")).encode())
    for c in sorted(map(str, df.columns)):
        h.update(c.encode())
        h.update(df[c].to_csv(index=False).encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# PIT 股票池
# ---------------------------------------------------------------------------
def listed_days_panel(close: pd.DataFrame) -> pd.DataFrame:
    """上市以来交易日数（面板内第一根有效 bar 起计）。PIT：只用 t 及以前的信息。

    向量化（2026-10-06）：原实现逐列 first_valid_index + get_loc 的 Python 循环，
    全市场 5000+ 列在生产日链上白白多花秒级；向量化后与原实现逐位一致
    （首有效 bar 前与停牌 NaN 处均为 NaN，全 NaN 列全 NaN）。
    """
    notna = close.notna()
    has_any = notna.any().to_numpy()
    # 每列首个有效 bar 的整数位置（argmax 返回第一个 True；全 NaN 列由 has_any 屏蔽为 -1）
    first_i = np.where(has_any, notna.to_numpy().argmax(axis=0), -1)
    pos = np.arange(len(close), dtype="float32")[:, None]
    days = pos - first_i[None, :].astype("float32") + 1.0
    vals = np.where(notna.to_numpy() & (first_i[None, :] >= 0), days, np.nan)
    return pd.DataFrame(vals, index=close.index, columns=close.columns, dtype="float32")


def universe_mask(cfg: Config, panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """逐交易日 PIT 股票池掩码（True=当日可入池）。

    过滤链（全部只用当日可得信息）：上市满 min_list_days 个交易日 → 非 ST/退 →
    未复权价 ≥ min_price → 流通市值 ≥ min_circ_mv_wan → 当日有成交（可交易）。
    mode=fixed 时只保留显式清单（仍过滤上市/ST/当日可交易）。
    """
    close, adj = panels["close"], panels["adj"]
    raw_close = close
    days = listed_days_panel(close)
    ok = days.ge(cfg.universe.min_list_days)
    # ST 掩码面板可能缺列（无更名记录且当前非 ST 的股票）⇒ 缺列补 0 = 非 ST。
    # _st_mask 对已有列是全日期展开的 0/1（记录链覆盖全生命周期），列内不会出 NaN。
    st = panels["st_flag"].reindex(columns=close.columns).fillna(0).astype("int8")
    ok &= st.eq(0)
    ok &= raw_close.ge(cfg.universe.min_price)
    ok &= panels["circ_mv"].ge(cfg.universe.min_circ_mv_wan)
    ok &= panels["turnover"].notna()                             # 当日有交易事实（停牌剔除）
    if cfg.universe.mode == "fixed":
        keep = [c for c in close.columns if c in set(cfg.universe.codes)]
        ok = ok[keep]
    if cfg.universe.max_universe and cfg.universe.max_universe > 0:
        ordered = sorted(ok.columns)
        ok = ok[ordered[: cfg.universe.max_universe]]
    return ok.fillna(False)


def hfq(panels: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """后复权 OHLC（未复权 × 复权因子）：收益与分红送转全程正确，撮合价用未复权。"""
    adj = panels["adj"]
    return {f: (panels[f] * adj).astype("float32") for f in ("open", "high", "low", "close")}
