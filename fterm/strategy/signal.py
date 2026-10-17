"""SignalProvider —— TriHeadNet 预测 → 目标方向（A股现货：只做多）。

门限设计（三头联用，继承期货版）：
  方向头 p_up：≥ 0.5 + gate 做多，其余=观望（现货无做空，-1 方向不存在）
  种子分歧惩罚：p_up_std > std_gate（集成不一致）→ 强制观望
  分位区间宽度（q_hi - q_lo）过宽 → 不确定性大 → 提高门槛（更保守）
方向为 0 或置信不足时不产生目标（该股票今日不动）。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Config


@dataclass(frozen=True)
class SignalDecision:
    product: str
    direction: int          # 0/+1（现货只做多；0=无新信号）
    confidence: float       # p_up 偏离 0.5 的幅度
    p_up: float
    p_up_std: float
    vol_pred_log: float
    reason: str


class SignalProvider:
    def __init__(self, cfg: Config, gate: float = 0.06, std_gate: float = 0.12,
                 q_width_gate: float = 0.06):
        self.cfg = cfg
        self.gate = gate
        self.std_gate = std_gate
        self.q_width_gate = q_width_gate

    def decide(self, product: str, pred: dict[str, float]) -> SignalDecision:
        """pred 来自 SeedEnsemble.predict()（p_up / p_up_std / q_lo / q_hi / vol_pred）。"""
        p = pred["p_up"]
        q_width = pred["q_hi"] - pred["q_lo"]
        # 分位区间过宽 → 提高 gate（更难触发）
        gate = self.gate + (self.q_width_gate if q_width > 0.04 else 0.0)
        if pred["p_up_std"] > self.std_gate:
            return SignalDecision(product, 0, 0.0, p, pred["p_up_std"], pred["vol_pred"],
                                  f"seed_disagreement({pred['p_up_std']:.2f})")
        d = 1 if p >= 0.5 + gate else 0
        reason = "long_gate" if d == 1 else "neutral"
        if q_width > 0.04:
            reason += "+wide_q"
        return SignalDecision(product, d, abs(p - 0.5), p, pred["p_up_std"], pred["vol_pred"], reason)
