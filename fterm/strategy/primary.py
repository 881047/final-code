"""主信号（规则层）—— 非机器学习的生产规则信号（继承期货版 primary 的定位纪律）。

主信号必须是规则（非 ML）、点时安全（只用 t 日及以前数据）、且**日志可复原**
（reason 必须携带判据原始读数，前瞻验证要从日志复原原始方向——期货版 H1 教训）。

rule_reversal（A股占位基线，**未证实 α**，采纳前必须预注册+组合层两轴，纪律同老盘）：
  score = rev_{rev_window} 的截面秩（旧盘 31 因子极性表 F70：rev 正极性、近 12 月增强）
  闸 1：静默波动制度——个股 pvol_20 ≤ 自身 252 日滚动中位数（点时口径，≥126 日暖机）
  闸 2：流动性/可交易性由 universe 掩码承担（数据层 PIT 过滤）
选股：top_n 入场；持仓跌出 top_n × exit_buffer 才清仓（退出带降换手）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def reversal_ranks(feats: dict[str, pd.DataFrame], asof: pd.Timestamp,
                   rev_window: int) -> pd.Series:
    """t 日截面反转秩（pct rank ∈ (0,1]，NaN 不参与排名）。"""
    col = f"rev_{rev_window}"
    assert col in feats, f"缺反转列 {col}（检查 config signal.rev_window 与特征构建）"
    row = feats[col].loc[asof]
    return row.rank(pct=True)


def rule_reversal_decision(feats: dict[str, pd.DataFrame], universe: pd.DataFrame,
                           asof: pd.Timestamp, rev_window: int = 5,
                           quiet_vol: bool = True,
                           pvol_med: pd.DataFrame | None = None) -> tuple[pd.Series, pd.Series]:
    """t 日截面决策（规则模式主入口）。

    返回 (score, eligible)：
      score     反转秩面板行（含全部股票，含 NaN）
      eligible  当日可交易掩码 = universe ∩ 反转秩非 NaN ∩ 静默波动
    pvol_med：pvol_20 的 252 日滚动中位数**面板**（调用方预计算一次复用——
    逐日重算是 O(天数×全市场) 的性能炸弹）。None 时现场算（测试/冒烟用）。
    """
    col = f"rev_{rev_window}"
    assert col in feats, f"缺反转列 {col}（检查 config signal.rev_window 与特征构建）"
    score = feats[col].loc[asof].rank(pct=True)
    eligible = universe.loc[asof].astype(bool) & score.notna()
    if quiet_vol:
        pvol = feats["pvol_20"].loc[asof]
        if pvol_med is None:
            pvol_med = feats["pvol_20"].rolling(252, min_periods=126).median()
        med = pvol_med.loc[asof]
        ok = (pvol <= med) & pvol.notna() & med.notna()
        eligible &= ok
    return score.astype("float64"), eligible
