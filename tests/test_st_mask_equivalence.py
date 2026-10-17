r"""`_st_mask` 向量化改造的等价性回归（2026-10-08 B2 收尾的安全网）。

背景：`fterm/data/ashare.py::_st_mask` 原本**逐列**写 DataFrame（5467 列 × 每次列插入
⇒ 碎片化），改为先累进预分配 ndarray、最后一次性构造 DataFrame。

本文件把旧实现原样内联为 `_st_mask_reference`，对合成 sqlite 逐位对照。必须保住的语义：

1. **列集** = `sorted({namechange 的 ts_code} | {stock_basic 中 ST/退 的 ts_code})`（排序并集）；
2. **dtype 必须是 int8**（下游按位运算/内存敏感）；
3. **无更名记录但当前是 ST** ⇒ 整列置 1（保守口径，登记于 docs/02）；
4. **闭区间** `dates >= a & dates <= b`，且缺失端点用 `Timestamp.min` / `.max` 哨兵；
5. **必须用 pandas 原生 Timestamp 比较** —— asi8 的单位随 dtype（us/ms）变化，
   与 `Timestamp.value` 的纳秒不可直接比（2026-10-03 单测当场抓到过）。

`tests/test_data.py::TestUniverseMask` 覆盖的是**下游语义**（ST/PIT 是否正确），
本文件覆盖的是**实现替换的等价性**，两者互补。
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fterm.data import ashare as sa  # noqa: E402


def _make_conn():
    """内存 sqlite：namechange（含 ST 更名、非 ST 更名、开区间）+ stock_basic。"""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE namechange (ts_code TEXT, name TEXT, "
                 "start_date TEXT, end_date TEXT)")
    conn.execute("CREATE TABLE stock_basic (ts_code TEXT, name TEXT)")
    conn.executemany("INSERT INTO namechange VALUES (?,?,?,?)", [
        # 000001：先正常后 ST，闭区间
        ("000001.SZ", "平安银行", "19910403", "20150101"),
        ("000001.SZ", "ST平安", "20150101", "20180101"),
        # 000002：ST 后摘帽，第二段开区间（end_date 为 NULL）
        ("000002.SZ", "ST万科", "20100101", "20120101"),
        ("000002.SZ", "万科A", "20120101", None),
        # 000003：全程 ST（首段 end_date 为 NULL ⇒ 用 Timestamp.max 哨兵）
        ("000003.SZ", "*ST三", "20100101", None),
        # 000004：有更名记录但从未 ST ⇒ 应保持全 0
        ("000004.SZ", "国华网安", "20100101", None),
        # 000005：名字含「退」
        ("000005.SZ", "退市五", "20190101", None),
    ])
    conn.executemany("INSERT INTO stock_basic VALUES (?,?)", [
        ("000003.SZ", "*ST三"),     # 当前是 ST，但有记录 ⇒ 走记录分支
        ("000006.SZ", "ST恒久"),    # 无更名记录但当前 ST ⇒ 整列 1
        ("000007.SZ", "退市七"),    # 无记录 + 含「退」⇒ 整列 1
        ("000008.SZ", "平安银行"),  # 无记录且非 ST ⇒ 不进列集（覆盖"过滤后 basic"分支）
    ])
    conn.commit()
    return conn


def _st_mask_reference(conn, dates):
    """**旧实现**逐字复刻（逐列写 DataFrame）。"""
    rows = conn.execute(
        "SELECT n.ts_code, n.name, n.start_date, n.end_date FROM namechange n "
        "ORDER BY n.ts_code, n.start_date").fetchall()
    basic = dict(conn.execute(
        "SELECT ts_code, name FROM stock_basic WHERE name LIKE '%ST%' OR name LIKE '%退%'").fetchall())
    by_code: dict = {}
    for code, name, sd, ed in rows:
        by_code.setdefault(code, []).append((str(name), str(sd), None if ed is None else str(ed)))
    mask = pd.DataFrame(0, index=dates, columns=sorted({r[0] for r in rows} | set(basic)), dtype="int8")
    for code in mask.columns:
        recs = by_code.get(code)
        if not recs:
            if code in basic:
                mask[code] = 1
            continue
        flag = np.zeros(len(dates), dtype="int8")
        for name, sd, ed in recs:
            if not ("ST" in name.upper() or "退" in name):
                continue
            a = pd.Timestamp(sd) if sd and len(sd) == 8 else pd.Timestamp.min
            b = pd.Timestamp(ed) if ed and len(ed) == 8 else pd.Timestamp.max
            seg = np.asarray((dates >= a) & (dates <= b), dtype="int8")
            flag |= seg
        mask[code] = flag
    return mask


def _dates():
    return pd.bdate_range("2009-01-01", "2020-12-31")


def test_bitwise_identical():
    conn = _make_conn()
    dates = _dates()
    new = sa._st_mask(None, conn, dates)
    old = _st_mask_reference(conn, dates)
    # 列集与列序必须一致（下游按列取用）
    assert list(new.columns) == list(old.columns)
    assert new.index.equals(old.index)
    # 值逐位一致（dtype 无关比较 —— 旧实现本身 dtype 就不统一，见下一个测试）
    np.testing.assert_array_equal(new.to_numpy().astype("int8"),
                                  old.to_numpy().astype("int8"))


def test_dtype_is_uniform_int8():
    """本次顺带修掉的缺陷：旧实现 dtype 不统一（标量赋值分支被升位成 int64）。

    旧实现里「无更名记录但当前是 ST/退」的列走 `mask[code] = 1`（Python 标量赋值），
    pandas 会升位成 int64 ⇒ 同一张掩码表混存 int8/int64（实测 7 列里 2 列）。
    本测试钉住修复后的**统一 int8**。
    """
    conn = _make_conn()
    dates = _dates()
    new = sa._st_mask(None, conn, dates)
    assert new.dtypes.unique().tolist() == [np.dtype("int8")], "掩码必须全列统一 int8"

    # 反向证据：旧实现确实不统一（否则本测试没有存在意义）
    old = _st_mask_reference(conn, dates)
    assert len(old.dtypes.unique()) > 1, "旧实现应存在 dtype 混存（这是被修掉的缺陷）"
    bad = [c for c in old.columns if old[c].dtype != np.dtype("int8")]
    assert bad, "旧实现应至少有一列非 int8"
    for c in bad:
        assert old[c].dtype == np.dtype("int64")
        # 值本身没错，只是 dtype/内存不对
        np.testing.assert_array_equal(old[c].to_numpy(), new[c].to_numpy())


def test_st_semantics_preserved():
    """抽查每类分支的期望结果，防止"两边一起错"。"""
    conn = _make_conn()
    dates = _dates()
    m = sa._st_mask(None, conn, dates)

    # 000001：2015-01-01 起进入 ST（闭区间，端点当天即为 1）
    assert m.loc["2015-01-01", "000001.SZ"] == 1
    assert m.loc["2014-12-31", "000001.SZ"] == 0
    # 000002：2010-01-01 ~ 2012-01-01 为 ST
    assert m.loc["2011-06-01", "000002.SZ"] == 1
    assert m.loc["2013-06-03", "000002.SZ"] == 0
    # 000003：[2010-01-01, Timestamp.max) 开区间 ST ⇒ 起点前为 0、起点后全为 1
    assert m.loc["2009-12-31", "000003.SZ"] == 0
    assert m.loc["2010-01-01", "000003.SZ"] == 1
    assert m.loc["2010-01-01":, "000003.SZ"].min() == 1, "开区间 ST 起点后必须全为 1"
    # 000004：有记录但从未 ST ⇒ 全 0
    assert m["000004.SZ"].max() == 0
    # 000005：含「退」
    assert m.loc["2019-01-01", "000005.SZ"] == 1
    # 无记录但当前 ST / 含退 ⇒ 整列 1
    assert m["000006.SZ"].min() == 1
    assert m["000007.SZ"].min() == 1
    # 000008.SZ 名字非 ST/退且无更名记录 ⇒ **不进列集**（列集 = namechange ∪ 过滤后的 basic）
    assert "000008.SZ" not in m.columns


def test_column_set_is_sorted_union():
    conn = _make_conn()
    m = sa._st_mask(None, conn, _dates())
    assert list(m.columns) == sorted(m.columns)
    # 列集 = namechange 全部 code ∪ stock_basic 中名字含 ST/退 的 code
    assert set(m.columns) == {"000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ",
                              "000005.SZ", "000006.SZ", "000007.SZ"}


def test_empty_namechange_still_covers_basic():
    """namechange 空表时仍要按 stock_basic 的 ST/退 名单出列（原实现的分支）。"""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE namechange (ts_code TEXT, name TEXT, start_date TEXT, end_date TEXT)")
    conn.execute("CREATE TABLE stock_basic (ts_code TEXT, name TEXT)")
    conn.execute("INSERT INTO stock_basic VALUES ('600001.SH','ST测试')")
    conn.commit()
    m = sa._st_mask(None, conn, _dates())
    assert list(m.columns) == ["600001.SH"]
    assert m["600001.SH"].min() == 1
