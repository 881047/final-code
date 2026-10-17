"""数据层验收：日历校验、老库面板同步（合成替身）、PIT 股票池、ST 掩码。"""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from fterm.data import ashare as sa
from fterm.data.calendar import assert_trading_days
from tests.conftest import CODES, write_synthetic_db


class TestCalendar:
    def test_valid_index_ok(self):
        idx = pd.bdate_range("2024-01-02", periods=5)
        assert_trading_days(idx)

    def test_weekend_rejected(self):
        idx = pd.DatetimeIndex(["2024-01-06", "2024-01-08"])   # 含周六
        with pytest.raises(AssertionError, match="周末"):
            assert_trading_days(idx)


class TestSyncPanels:
    def test_panels_shape_and_content(self, env):
        cfg, panels, cal = env
        close = panels["close"]
        assert set(CODES) <= set(close.columns)
        assert not any(c.endswith(".BJ") for c in close.columns)   # 北交所剔除
        assert close.shape[0] == len(cal)
        # 未复权 close 与老库一致；hfq = 未复权 × adj
        assert (panels["open"] <= panels["high"]).all().all()
        hfq_close = sa.hfq(panels)["close"]
        expect = panels["close"] * panels["adj"]
        pd.testing.assert_frame_equal(hfq_close, expect, check_dtype=False)

    def test_limit_prices_present(self, env):
        _, panels, _ = env
        up, dn = panels["up_limit"], panels["down_limit"]
        assert up.notna().all().all() and dn.notna().all().all()
        assert (up > dn).all().all()

    def test_trade_cal_persisted(self, env):
        cfg, _, cal = env
        cal2 = sa.load_trade_cal(cfg)
        assert list(cal) == list(cal2)


class TestUniverseMask:
    def test_st_pit_exclusion(self, env):
        cfg, panels, cal = env
        mask = sa.universe_mask(cfg, panels)
        st_code = "000002.SZ"
        mid = cal[len(cal) // 2]
        # ST 生效日前正常入池；生效后被剔除（PIT：历史前段不受"未来才 ST"影响）
        assert bool(mask.at[cal[10], st_code]) is True
        assert bool(mask.at[mid, st_code]) is False

    def test_liquidity_filters(self, tmp_path, cfg):
        write_synthetic_db(tmp_path / "old.db")
        results = sa.sync_panels(cfg)
        assert all(not v.startswith("FAIL") for v in results.values()), results
        panels, cal = sa.load_panels(cfg), sa.load_trade_cal(cfg)
        mask = sa.universe_mask(cfg, panels)
        # 合成数据：换手/市值全达标 ⇒ 上市暖机期(前 min_list_days-1 日)外每天都有股票可入池
        assert mask.any(axis=1).iloc[cfg.universe.min_list_days:].all()

    def test_no_price_excluded(self, env):
        cfg, panels, cal = env
        panels = dict(panels)
        panels["close"] = panels["close"].copy()
        panels["close"].iloc[20, 0] = 1.0            # 仙股价 < min_price
        mask = sa.universe_mask(cfg, panels)
        assert bool(mask.iloc[20, 0]) is False


class TestCalendarAlignment:
    """P0-B（2026-10-08）：面板索引以老库 trade_cal(is_open=1) 为唯一权威。

    真实事故：老库 `adj_factor` 里 `689009.SH` 在 20210920/21（中秋休市）各 1 行脏数据，
    而 `daily`/`stk_limit` 当日 0 行 ⇒ `adj` 面板比其余多 2 行 ⇒ `hfq()` 索引并集在
    2021-09 中部插进 2 个全 NaN 幻象交易日 ⇒ ①`fold_frame` 严格 index 断言直接
    ValueError（训练/研究链全灭），②即便不断言，hfq 派生特征的 rolling 窗与标签
    `shift` 也会按"位置"而非交易日错位。
    """

    @staticmethod
    def _inject_offcal_adj(path, dates, n_codes: int):
        """在紧接面板末日的**周末**（不在 trade_cal）给 n_codes 只股票插 adj 脏行。"""
        con = sqlite3.connect(path)
        d = dates[-1] + pd.Timedelta(days=6)          # 面板末日后第一个周末（必不在日历）
        assert d.dayofweek in (5, 6)
        assert d.strftime("%Y%m%d") not in {
            r[0] for r in con.execute("SELECT cal_date FROM trade_cal WHERE is_open=1")}
        for code in CODES[:n_codes]:
            con.execute("INSERT INTO adj_factor VALUES (?,?,?)",
                        (code, d.strftime("%Y%m%d"), 1.0))
        con.commit()
        con.close()
        return d

    def test_sync_drops_non_trading_day_rows(self, tmp_path, cfg):
        dates = write_synthetic_db(tmp_path / "old.db")
        off = self._inject_offcal_adj(tmp_path / "old.db", dates, n_codes=1)
        results = sa.sync_panels(cfg)
        assert "adj: 丢弃非开市日 1 行" in results["cal_align"], results["cal_align"]
        panels, cal = sa.load_panels(cfg), sa.load_trade_cal(cfg)
        assert off not in set(panels["adj"].index), "休市日脏行必须被丢弃"
        assert off not in set(cal)
        # 全 12 张面板索引必须一致（修前：adj=2369 vs 其余=2367）
        ref = panels["close"].index
        for f in sa.PANEL_FIELDS:
            assert panels[f].index.equals(ref), f"{f} 索引与 close 不等"

    def test_stale_calendar_raises_loudly_not_silent_drop(self, tmp_path, cfg):
        """反向对照：日历漏掉一个**真有数据**的交易日 ⇒ 不许静默丢弃，必须响亮失败。

        本仓纪律：修复不得改写历史。源脏行（单元格数 ≤ 容差）丢弃并留痕；
        真实交易日绝不因日历陈旧而被悄悄删掉。
        """
        dates = write_synthetic_db(tmp_path / "old.db")
        con = sqlite3.connect(tmp_path / "old.db")
        lost = dates[-1]
        con.execute("DELETE FROM trade_cal WHERE cal_date = ?",
                    (lost.strftime("%Y%m%d"),))              # 日历丢掉最后一个交易日
        con.commit()
        con.close()
        with pytest.raises(AssertionError, match="日历陈旧"):
            sa.sync_panels(cfg)

    def test_gate_reports_index_consistency(self, tmp_path, cfg):
        write_synthetic_db(tmp_path / "old.db")
        sa.sync_panels(cfg)
        ok, why = sa.panel_index_consistency(cfg)
        assert ok, why
        # 手工把幻象行塞回 adj 面板 ⇒ 第⑤项必须抓到
        adj = sa.load_panel(cfg, "adj")
        phantom = adj.index.max() + pd.Timedelta(days=3)     # 周一之后的休市日
        adj.loc[phantom] = 1.0
        adj.sort_index().to_parquet(sa._panel_dir(cfg) / "adj.parquet")
        ok2, why2 = sa.panel_index_consistency(cfg)
        assert not ok2 and "adj" in why2, why2
        ready, msg = sa.daily_chain_ready(cfg, now=adj.index.max() - pd.Timedelta(days=6))
        assert ready and "⑤警告" in msg, f"默认只警告不阻塞（窗内不加新闸）: {msg}"
        ready2, msg2 = sa.daily_chain_ready(cfg, now=adj.index.max() - pd.Timedelta(days=6),
                                            strict_panel_align=True)
        assert not ready2 and "面板索引不一致" in msg2, msg2


class TestFingerprint:
    def test_stable_and_sensitive(self, cfg, tmp_path):
        import pandas as pd
        df = pd.DataFrame({"a": [1.0, 2.0]})
        f1 = sa.df_fingerprint(df)
        assert f1 == sa.df_fingerprint(pd.DataFrame({"a": [1.0, 2.0]}))
        assert f1 != sa.df_fingerprint(pd.DataFrame({"a": [1.0, 3.0]}))


class TestDailyChainReady:
    """每日链就绪闸：老库断更/竞态造成的静默 T-1 落后必须变成响亮拒绝（2026-10-05）。"""

    @staticmethod
    def _extend_db(path, dates, with_bars: bool):
        """老库末尾追加 3 个日历日：+1/+2 休市、+3 开市（当日日线可选）。"""
        con = sqlite3.connect(path)
        d1, d2, d3 = [dates[-1] + pd.Timedelta(days=k) for k in (1, 2, 3)]
        for d, open_ in ((d1, 0), (d2, 0), (d3, 1)):
            con.execute("INSERT INTO trade_cal VALUES (?,?)", (d.strftime("%Y%m%d"), open_))
        if with_bars:
            for code in CODES:
                ds = d3.strftime("%Y%m%d")
                con.execute("INSERT INTO daily VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                            (code, ds, 10.0, 10.5, 9.8, 10.0, 10.0, 0.0, 0.0, 100000.0, 1e6))
                con.execute("INSERT INTO adj_factor VALUES (?,?,?)", (code, ds, 1.5))
                con.execute("INSERT INTO daily_basic VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                            (code, ds, 1.5, 1.6, 1.0, 8.0, 8.1, 1.0, 2.0, 2.0,
                             500000.0, 300000.0))
                con.execute("INSERT INTO stk_limit VALUES (?,?,?,?)", (ds, code, 11.0, 9.0))
        con.commit()
        con.close()
        return d3

    def test_ready_on_trading_day(self, tmp_path, cfg):
        dates = write_synthetic_db(tmp_path / "old.db")
        d3 = self._extend_db(tmp_path / "old.db", dates, with_bars=True)
        sa.sync_panels(cfg)
        ok, why = sa.daily_chain_ready(cfg, now=d3)
        assert ok, why

    def test_stale_when_open_day_has_no_bars(self, tmp_path, cfg):
        # 开市日当晚老库日链断更 ⇒ sync"成功"拉到 T-1 ⇒ 闸必须拦下（核心故障模式）
        dates = write_synthetic_db(tmp_path / "old.db")
        d3 = self._extend_db(tmp_path / "old.db", dates, with_bars=False)
        sa.sync_panels(cfg)
        ok, why = sa.daily_chain_ready(cfg, now=d3)
        assert not ok
        assert "不落账" in why

    def test_holiday_rerun_passes(self, tmp_path, cfg):
        # 休市日重跑（幂等路径）：expected=上一交易日，面板已覆盖 ⇒ 放行
        dates = write_synthetic_db(tmp_path / "old.db")
        d1 = dates[-1] + pd.Timedelta(days=1)
        con = sqlite3.connect(tmp_path / "old.db")
        con.execute("INSERT INTO trade_cal VALUES (?,0)", (d1.strftime("%Y%m%d"),))
        con.commit()
        con.close()
        sa.sync_panels(cfg)
        ok, why = sa.daily_chain_ready(cfg, now=d1)
        assert ok, why

    def test_up_limit_lag_rejected(self, tmp_path, cfg):
        # stk_limit 最后一日缺行 ⇒ 撮合将静默回退近似停板价 ⇒ 拒绝在降级口径上落账
        dates = write_synthetic_db(tmp_path / "old.db")
        con = sqlite3.connect(tmp_path / "old.db")
        con.execute("DELETE FROM stk_limit WHERE trade_date = ?",
                    (dates[-1].strftime("%Y%m%d"),))
        con.commit()
        con.close()
        sa.sync_panels(cfg)
        ok, why = sa.daily_chain_ready(cfg, now=dates[-1])
        assert not ok and "up_limit" in why

    def test_calendar_too_old(self, tmp_path, cfg):
        # 真实时钟 vs 合成日历（2024-02 止）差年级 ⇒ 保鲜线拒绝（长期断更兜底）
        write_synthetic_db(tmp_path / "old.db")
        sa.sync_panels(cfg)
        ok, why = sa.daily_chain_ready(cfg)
        assert not ok and "陈旧" in why

    def test_missing_panels(self, cfg):
        # 未同步过 ⇒ 数据源不可用，拒绝（不是异常逃逸）
        ok, why = sa.daily_chain_ready(cfg)
        assert not ok
