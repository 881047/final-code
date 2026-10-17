"""每日调度 runner —— 数据→特征→推理→信号→风控→下单→落库 的总装配线（A股）。

两种运行模式：
  paper : 每日收盘后跑（默认），产出次日开盘单队列（PaperBroker 撮合）
  live  : 同一流程，Broker 换成券商程序化通道实现（接口不变，业务零改动）

本模块是业务侧唯一入口；回测引擎不在此路径（离线评估用 scripts/backtest_run.py）。
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .. import contracts
from ..backtest.engine import Target
from ..config import Config
from ..data import ashare as sa
from ..exec.broker import Broker, Order, Side
from ..features.build import build_dataset
from ..model.train import SeedEnsemble, load_ensemble
from ..risk.governor import AccountState, RiskGovernor
from ..risk.sizer import size_position
from ..strategy.primary import rule_reversal_decision
from ..strategy.signal import SignalProvider
from .state import StateDB, fill_cash_delta


class DailyRunner:
    def __init__(self, cfg: Config, broker: Broker, db: StateDB,
                 model_path: Path | None = None):
        self.cfg = cfg
        self.broker = broker
        self.db = db
        self.model_path = model_path or Path(cfg.data.root) / "models" / "trihead.pt"
        self.signals = SignalProvider(cfg)

    # ---------- 主流程 ----------

    def run_once(self, asof: pd.Timestamp | None = None) -> dict:
        """执行一个完整决策循环。返回摘要 dict（供看板/CLI 展示）。"""
        cfg = self.cfg
        if cfg.signal.mode == "dlv2":
            return self._run_dlv2(asof)
        if cfg.signal.mode == "halt":
            # 收口态（docs/02 §参数基线 2026-10-04）：生产信号位空缺时只跑骨架
            # ——数据链/权益快照/心跳照常，不产生任何开仓决策。
            asof = asof or pd.Timestamp.now().normalize()
            snap = self.broker.account() if hasattr(self.broker, "account") else None
            equity = snap.equity if snap else cfg.account.initial_equity
            snap2 = self.broker.account() if hasattr(self.broker, "account") else None
            if snap2 is not None:
                self.db.record_equity(str(asof.date()), snap2.equity, snap2.pos_value, snap2.available)
            self.db.heartbeat(datetime.now().isoformat(timespec="seconds"),
                              note=f"halt asof={asof.date()} equity={equity:.0f}")
            return {"asof": str(asof.date()), "equity": equity, "orders": [],
                    "decisions": [{"product": "-", "direction": 0, "lots": 0,
                                   "approved": True, "reason": "halt(生产信号位空缺)"}]}
        bundle = build_dataset(cfg, compute_fingerprints=False)
        feats, universe = bundle.features, bundle.universe
        rule_mode = cfg.signal.mode == "rule_reversal"
        ens = None
        if not rule_mode:
            ens = load_ensemble(self.model_path, cfg)

        # 1) 最新特征日（asof 之后的截断是防未来函数的关键闸门）
        asof = asof or feats["rev_%d" % cfg.signal.rev_window].index.max()
        snap = self.broker.account() if hasattr(self.broker, "account") else None
        raw_close = sa.load_panel(cfg, "close")
        prices = {c: float(raw_close.at[asof, c]) for c in universe.columns
                  if c in raw_close.columns and pd.notna(raw_close.at[asof, c])}
        equity = snap.equity if snap else cfg.account.initial_equity
        peak = max(equity, self._peak_equity(equity))
        marks = {c: prices.get(c, 0.0) for c in (snap.positions if snap else {})}

        # 2) 信号 → decisions: (code, direction, lots, has_target, reason)
        if rule_mode:
            decisions = self._rule_decisions(feats, universe, asof, equity, prices)
        else:
            decisions = self._model_decisions(ens, feats, universe, asof, equity, prices)

        # 3) 风控审批（开仓目标过 governor；清仓目标天然放行）
        gov = RiskGovernor(cfg)
        curve = self.db.equity_curve()
        day_start = curve[-1][1] if curve else equity
        # 审批估值用现价盯市（2026-10-08 审查批统一，docs/02 §参数基线）：旧实现
        # open_positions 直接用未标记快照（入场价口径），与回测忠实臂不一致。
        # 缺价（停牌等）退回快照自带估值，不按 0 计——0 会放松 cash_util 闸。
        open_pos: dict[str, dict] = {}
        if snap:
            for c, p in snap.positions.items():
                v = prices[c] * contracts.shares_of(p["lots"]) if c in prices else p["value"]
                open_pos[c] = {"direction": p["direction"], "lots": p["lots"], "value": v}
        state = AccountState(
            equity=equity, peak_equity=peak, day_pnl=equity - day_start,
            day_start_equity=day_start,
            open_positions=open_pos)
        open_targets = [Target(c, d, l, r) for (c, d, l, a, r) in decisions if d != 0]
        close_targets = [Target(c, 0, 0, r) for (c, d, l, a, r) in decisions
                         if d == 0 and a and c in state.open_positions]
        report = gov.approve(open_targets, state, prices=prices)
        approved = {(t.product, t.direction): t for t in report.approved}
        close_ok = {t.product for t in close_targets}

        # 4) 下单（清仓优先——卖出资金当日可用于买入）
        orders = []
        for (c, d, l, a, r) in decisions:
            if d == 0:
                if c in close_ok:
                    held = state.open_positions.get(c)
                    side = Side.SELL
                    lots = held["lots"] if held else l
                    oid = self.broker.submit(Order(c, side, lots, reason=r))
                    orders.append({"product": c, "side": side.value, "lots": lots,
                                   "order_id": oid})
                continue
            t = approved.get((c, d))
            lots = t.lots if t else l
            side = self._side_for(c, d, state.open_positions)
            if side is None:
                continue
            oid = self.broker.submit(Order(c, side, lots, reason=r))
            orders.append({"product": c, "side": side.value, "lots": lots, "order_id": oid})

        # 5) 审计落库（决策幂等 REPLACE + 每日权益快照）
        self.db.record_decisions(
            str(asof.date()),
            [(c, d, l, bool((c, d) in approved) if d != 0 else bool(c in close_ok), r)
             for (c, d, l, a, r) in decisions])
        snap2 = self.broker.account(mark_prices=marks) if hasattr(self.broker, "account") else None
        if snap2 is not None:
            self.db.record_equity(str(asof.date()), snap2.equity, snap2.pos_value, snap2.available)
        self.db.heartbeat(datetime.now().isoformat(timespec="seconds"),
                          note=f"asof={asof.date()} equity={equity:.0f}")
        return {"asof": str(asof.date()), "equity": equity, "orders": orders,
                "decisions": [{"product": c, "direction": d, "lots": l,
                               "approved": bool((c, d) in approved) if d != 0 else bool(c in close_ok),
                               "reason": r}
                              for (c, d, l, a, r) in decisions]}

    # ---------- 规则模式 ----------

    def _rule_decisions(self, feats, universe, asof, equity, prices) -> list[tuple]:
        cfg = self.cfg
        positions = {}
        if hasattr(self.broker, "account"):
            positions = self.broker.account().positions
        # 非信号日：只维持（记录 hold，不做任何调仓）
        if asof.weekday() != cfg.signal.rebalance_weekday:
            return [(c, 0, 0, False, "not_rebalance_day") for c in positions]
        score, eligible = rule_reversal_decision(
            feats, universe, asof, rev_window=cfg.signal.rev_window,
            quiet_vol=cfg.signal.quiet_vol)
        ranked = score[eligible].sort_values(ascending=False)
        keep_n = cfg.signal.top_n * cfg.signal.exit_buffer
        keep = list(ranked.head(keep_n).index)
        buy = [c for c in ranked.head(cfg.signal.top_n).index
               if c not in positions]
        atr_row = feats["atr14"].loc[asof]
        decisions = []
        for c in positions:
            if c not in keep:
                decisions.append((c, 0, 0, True,
                                  f"exit_band(rank_out_of_{keep_n})"))
        for c in buy:
            if not np.isfinite(atr_row.get(c, np.nan)) or c not in prices:
                decisions.append((c, 0, 0, True, "no_atr"))
                continue
            sz = size_position(cfg, equity, prices[c], float(atr_row[c]))
            if sz.lots == 0:
                decisions.append((c, 0, 0, True, f"zero_lots({sz.method_used})"))
                continue
            decisions.append((c, 1, sz.lots, True,
                              f"rule_reversal(rank={float(ranked[c]):.3f})"))
        return decisions

    # ---------- 模型模式 ----------

    def _model_decisions(self, ens: SeedEnsemble, feats, universe, asof, equity,
                         prices) -> list[tuple]:
        cfg = self.cfg
        atr_row = feats["atr14"].loc[asof]
        cands: list[tuple[str, dict[str, float]]] = []
        cols = [c for c in ens.feature_cols if c in feats]
        if len(cols) < len(ens.feature_cols):
            return [(c, 0, 0, True, "feature_cols_missing") for c in universe.columns]
        # 2026-10-08 B2：单日只取一次整行布尔掩码，再按列过滤
        # （原为遍历全部列 + 逐列 universe.at[asof, code] 标量判定）。
        # 单日场景不建全量 UniverseIndex —— O(T×N) 拷贝比这里的一次取行更贵。
        _urow = universe.loc[asof]
        for code in _urow.index[_urow.to_numpy(dtype=bool)]:
            win = np.stack([feats[c][code].loc[:asof].tail(ens.lookback).to_numpy()
                            for c in cols], axis=1)
            if len(win) < ens.lookback or np.isnan(win).any():
                continue
            pred = ens.predict(win)
            cands.append((code, pred))
        cands.sort(key=lambda x: x[1]["p_up"], reverse=True)
        decisions = []
        taken = 0
        for code, pred in cands:
            sig = self.signals.decide(code, pred)
            if sig.direction == 0 or taken >= cfg.account.max_symbols:
                decisions.append((code, 0, 0, True, sig.reason))
                continue
            sz = size_position(cfg, equity, prices[code], float(atr_row.get(code, np.nan)),
                               vol_pred_log=pred["vol_pred"])
            if sz.lots == 0:
                decisions.append((code, 0, 0, True, f"zero_lots({sz.method_used})"))
                continue
            decisions.append((code, 1, sz.lots, True,
                              sig.reason + f"(p={pred['p_up']:.2f})"))
            taken += 1
        return decisions

    # ---------- 辅助 ----------

    def _peak_equity(self, current: float, mode: str = "none",
                     window: int = 252) -> float:
        """回撤峰值（口径唯一来源：`fterm/risk/peak.py`）。

        `mode="none"` 时与原实现 `max([r[1] for r in curve] + [current])`
        **逐位等价**（默认行为不变）；`mode="rolling"` 时只取窗口内 ⇒
        消除"熔断永不解除"的吸收态（docs/03 §36 缺陷，§47 修复）。
        """
        from ..risk.peak import peak_equity as _pk

        curve = self.db.equity_curve()
        return _pk([r[1] for r in curve], current, mode=mode, window=window)

    def _side_for(self, code: str, direction: int, positions: dict) -> Side | None:
        """同向持仓不加仓（实盘保守口径：只做开新/清仓，加减仓由回测验证后再开）。"""
        cur = positions.get(code)
        if cur is None:
            return Side.BUY if direction == 1 else None
        if cur["direction"] == direction:
            return None
        return Side.SELL if cur["direction"] == 1 else None

    # ---------- dlv2 生产位（双账户 paper 盘面） ----------

    def _run_dlv2(self, asof: pd.Timestamp | None = None) -> dict:
        """候选生产位：DL-v2 20 网集成 + C_ema 平滑秩，双 paper 账户轨道。

        信号（预测 + EMA）轨道无关，算一次共享；sizer/governor/持仓按轨道隔离。
        执行语义与回测一致：t 收盘出信号挂单 → t+1 开盘按价撮合（挂单存 state 库）。
        """
        from ..model.infer import file_sha256, load_artifact
        from ..exec.paper import PaperBroker

        cfg = self.cfg
        tracks = cfg.raw["paper_tracks"]
        art_path = self.model_path.parent / "dlv2_production.pt"
        if not art_path.exists():
            # 工件未就绪（生产训练在途）⇒ 按 halt 语义骨架运行，保护每日计划任务
            asof = asof or pd.Timestamp.now().normalize()
            db0 = StateDB(cfg.state_db_path)
            db0.heartbeat(datetime.now().isoformat(timespec="seconds"),
                          note=f"dlv2 artifact pending, skeleton run asof={asof.date()}")
            return {"asof": str(asof.date()), "mode": "dlv2_skeleton", "tracks": [],
                    "decisions": [], "orders": []}
        artifact = load_artifact(art_path)
        bundle = build_dataset(cfg, compute_fingerprints=False)
        feats = bundle.features
        col = f"rev_{cfg.signal.rev_window}"
        asof = asof or feats[col].index.max()
        raw_close = sa.load_panel(cfg, "close")
        raw_open = sa.load_panel(cfg, "open")
        raw_low = sa.load_panel(cfg, "low")
        raw_high = sa.load_panel(cfg, "high")
        raw_up = sa.load_panel(cfg, "up_limit")
        raw_dn = sa.load_panel(cfg, "down_limit")
        # 复权因子面板：含权入账用（docs/03 §38）。现金/停板/整手/tick 仍全走真实价。
        raw_adj = sa.load_panel(cfg, "adj")
        # 调仓频率闸（2026-10-05 审计修复）：判据链组合层是**周五调仓**
        # （dlv2_chain.py 按 signal.rebalance_weekday 过滤；config 明文 rebalance_weekday: 4
        # "周五收盘出信号"），原实现每日重排 ⇒ 生产位实测的换手/成本结构不是被采纳的构型。
        is_rebalance = asof.weekday() == cfg.signal.rebalance_weekday
        eligible = bundle.universe.loc[asof].astype(bool)
        preds = artifact.predict_day(feats, eligible, asof)

        # ---- 秩平滑 EMA（2026-10-08 语义显式化 + 换版守卫）----
        # 状态库存主账本 runtime.state_db（原实现取 list(paper_tracks.values())[0]，
        # 依赖 YAML 键序 ⇒ 轨道顺序一变就悄悄换库）。
        # 步进频率由 signal.ema_step 决定：daily（生产位现用，半衰期≈1 交易日）
        # 或 rebalance（判据链 #11/#12/#13 被采纳构型，半衰期≈1 周）。
        # ⚠ 两者**不是同一构型**——docs 里"与判据链同款"的说法是错的，已按用户
        #   2026-10-08 拍板保留 daily 并另立前瞻预注册（plans/paper60b-prereg-*）。
        # 幂等：state 库存 ema_asof，同日重跑不二次更新。
        db0 = StateDB(cfg.state_db_path)
        art_sha = file_sha256(art_path)
        declared = db0.get_kv("model_version")
        if declared:
            d_sha = (json.loads(declared) or {}).get("sha256")
            if d_sha and d_sha != art_sha:
                db0.alert(datetime.now().isoformat(timespec="seconds"), "error",
                          "artifact_version_mismatch",
                          f"账本登记工件 sha={d_sha[:12]} ≠ 实际加载 {art_sha[:12]}（{art_path}）"
                          f"⇒ 生产位加载的不是登记对象，须人工核对")
        stored = db0.get_kv("ema_asof")
        ema: dict[str, float] = json.loads(db0.get_kv("ema") or "{}")
        if db0.get_kv("ema_model_version") != art_sha:
            # 换版 ⇒ 平滑状态必须清零：旧工件的残余秩会与新模型的秩混在一列里，
            # 而两条信号轨的秩尺度不可比（2026-10-08 换版时 EMA 是手工重置的，此处补自动化）。
            ema = {}
            db0.set_kv("ema", "{}")
            db0.set_kv("ema_asof", "")
            db0.set_kv("ema_model_version", art_sha)
            stored = None
            db0.alert(datetime.now().isoformat(timespec="seconds"), "info",
                      "ema_reset", f"工件换版 ⇒ 秩平滑清零（sha={art_sha[:12]}）")
        step_now = cfg.signal.ema_step == "daily" or is_rebalance
        if step_now and stored != str(asof.date()):
            a = cfg.signal.ema_alpha
            pct = preds.rank(pct=True)
            for c, v in pct.items():
                ema[c] = float(v) if c not in ema else a * float(v) + (1 - a) * ema[c]
            db0.set_kv("ema_asof", str(asof.date()))
            db0.set_kv("ema", json.dumps(ema))

        smooth = pd.Series(ema).reindex(preds.index).dropna().sort_values(ascending=False)

        # ---- #124 采纳影子轨（2026-10-14 用户拍板）：D = 0.5·rank_pct(Ā) + 0.5·rank_pct(dlv2) ----
        # 语义（预注册 §0/§4 冻结）：D 是**被判读验证的原对象**——无 EMA 平滑（判读口径即无 EMA），
        # 与主轨的 C_ema(α=0.5) **不是同一构型**；影子轨部署的是验证对象本身。
        # a1 工件未就绪 ⇒ 影子轨跳过（告警留痕，护住日任务主流程）。轨道经 paper_tracks.*.signal: blend 声明。
        smooth_blend: pd.Series | None = None
        a1_path = self.model_path.parent / "a1_production.pt"
        if a1_path.exists():
            try:
                a1_art = load_artifact(a1_path)
                pa = a1_art.predict_day_perseed(feats, eligible, asof)
                if len(pa):
                    a_bar = pa.rank(pct=True).mean(axis=1)      # Ā = 逐种子日内秩均值（§0 冻结）
                    smooth_blend = (0.5 * a_bar + 0.5 * preds.rank(pct=True)) \
                        .sort_values(ascending=False)
                a1_sha = file_sha256(a1_path)
                if db0.get_kv("a1_model_version") != a1_sha:
                    db0.set_kv("a1_model_version", a1_sha)
                    db0.alert(datetime.now().isoformat(timespec="seconds"), "info",
                              "a1_artifact_version", f"a1 工件登记 sha={a1_sha[:12]} asof={asof.date()}")
            except Exception as e:      # 影子轨任何异常不得拖垮主轨日任务
                smooth_blend = None
                db0.alert(datetime.now().isoformat(timespec="seconds"), "warn",
                          "blend_shadow_skip", f"a1 影子轨跳过：{type(e).__name__}: {e}")
        else:
            db0.alert(datetime.now().isoformat(timespec="seconds"), "warn",
                      "blend_shadow_skip", f"a1 工件不存在（{a1_path}）⇒ 影子轨跳过 asof={asof.date()}")

        # ---- #126 T3 三流影子轨（T3 工程单步骤 3，2026-10-15）：----
        # T3 = rank_pct( (rank_pct(Ā)+rank_pct(RL2̄)+rank_pct(dlv2)) / 3 )（#126 §0 冻结口径：
        # 组合后再日内 pct 秩）。rl_production 工件未就绪 ⇒ 轨跳过+告警（护主流程，同 blend 先例）。
        smooth_tri: pd.Series | None = None
        rl_path = self.model_path.parent / "rl_production.pt"
        if a1_path.exists() and rl_path.exists():
            try:
                a1_art_t = load_artifact(a1_path)
                pa_t = a1_art_t.predict_day_perseed(feats, eligible, asof)
                rl_art = load_artifact(rl_path)
                pr_t = rl_art.predict_day_perseed(feats, eligible, asof)
                if len(pa_t) and len(pr_t):
                    a_bar_t = pa_t.rank(pct=True).mean(axis=1)   # Ā = 逐种子日内秩均值
                    rl_bar_t = pr_t.rank(pct=True).mean(axis=1)  # RL2̄ 同口径
                    dlv_rank = preds.rank(pct=True)
                    smooth_tri = ((a_bar_t.rank(pct=True) + rl_bar_t.rank(pct=True)
                                   + dlv_rank) / 3.0).rank(pct=True) \
                        .sort_values(ascending=False)
                rl_sha = file_sha256(rl_path)
                if db0.get_kv("rl_model_version") != rl_sha:
                    db0.set_kv("rl_model_version", rl_sha)
                    db0.alert(datetime.now().isoformat(timespec="seconds"), "info",
                              "rl_artifact_version", f"rl 工件登记 sha={rl_sha[:12]} asof={asof.date()}")
            except Exception as e:      # 影子轨任何异常不得拖垮主轨日任务
                smooth_tri = None
                db0.alert(datetime.now().isoformat(timespec="seconds"), "warn",
                          "tri_shadow_skip", f"T3 影子轨跳过：{type(e).__name__}: {e}")
        elif not rl_path.exists():
            db0.alert(datetime.now().isoformat(timespec="seconds"), "warn",
                      "tri_shadow_skip", f"rl 工件不存在（{rl_path}）⇒ T3 影子轨跳过 asof={asof.date()}")

        summaries = []
        for name, tr in tracks.items():
            sig_kind = str(tr.get("signal", "dlv2"))
            if sig_kind == "blend" and smooth_blend is None:
                continue                # a1 工件未就绪（上方已告警）⇒ 影子轨本日跳过
            if sig_kind == "blend_tri" and smooth_tri is None:
                continue                # rl 工件未就绪（上方已告警）⇒ T3 影子轨本日跳过
            sig_use = (smooth_blend if sig_kind == "blend"
                       else smooth_tri if sig_kind == "blend_tri" else smooth)
            cfg_t = dataclasses.replace(
                cfg,
                account=dataclasses.replace(cfg.account, initial_equity=tr["equity"],
                                            max_symbols=tr["max_symbols"],
                                            max_lots_per_symbol=tr["max_lots"]),
                signal=dataclasses.replace(cfg.signal, top_n=tr["top_n"]),
                runtime=dataclasses.replace(cfg.runtime, state_db=tr["state_db"]))
            db = StateDB(cfg_t.state_db_path)
            # 日初现金（2026-10-08 P0-A 修复）：原实现每日把现金重置为 `tr["equity"]`
            # （初始本金），却从 state 库恢复持仓 ⇒ 现金与持仓市值**双计**，纸面权益自
            # 首笔成交次日起虚增约一个持仓市值（t3 口径 ≈1.95M / 1.0M 本金），
            # 前瞻窗主判据（收益>0、MDD<20%）、governor 的 peak/日停、sizer 的
            # 风险预算全部失真。equity_curve.cash 一直记的是「当日撮合后现金」，
            # 正是次日的日初值——只需读回。无历史行（首日）⇒ 退回初始本金。
            # 同日重跑：先按 fills 表重放本日已成交的现金影响，撮合循环用 has_fill 跳过
            # （防前次运行崩在 save_pending 之前造成双成交）。
            asof_d = str(asof.date())
            cash0 = db.day_start_cash(asof_d)
            if cash0 is None:
                cash0 = float(tr["equity"])
            else:
                for _code, _side, _lots, _px, _fee in db.fills_on(asof_d):
                    cash0 += fill_cash_delta(_side, _lots, _px, _fee)
            broker = PaperBroker(cfg_t, equity=cash0, now=asof)
            # 持仓恢复（state 库 → PaperBroker）
            from ..exec.paper import PaperPosition
            n_adj_miss = 0
            for c, p in db.load_positions().items():
                broker.positions[c] = PaperPosition(c, 1, p["lots"], p["entry_price"],
                                                    adj_entry=p.get("adj_entry", 1.0))
                n_adj_miss += int(bool(p.get("adj_missing")))
            if n_adj_miss:
                db.alert(datetime.now().isoformat(timespec="seconds"), "warn",
                         "adj_entry_missing",
                         f"{name}: {n_adj_miss} 个持仓缺 adj_entry（按 1.0 处理 ⇒ 含权可能低估）")

            # top/keep 按轨道配置取（2026-10-05 审计修复：原硬编码 head(10)/head(20)，
            # w2 轨的 top_n=2/keep_n=4 从未生效、实际行为与 docs/02 描述不符；
            # t3 轨 10/20 与硬编码恰好同值 ⇒ 行为不变）
            top = list(sig_use.head(tr["top_n"]).index)
            keep = list(sig_use.head(tr["keep_n"]).index)

            prices = {c: float(raw_close.at[asof, c]) for c in set(top) | set(broker.positions)
                      if c in raw_close.columns and pd.notna(raw_close.at[asof, c])}

            # ① 昨日挂单 → 今日开盘价撮合（2026-10-05 审计修复，对齐回测 _execute 语义）：
            #   成交价 = t+1 **开盘价**（原实现误用 close —— 被采纳构型是开盘价成交，
            #   docs/02 亦明文"次交易日开盘价"）；全天封板（low≥up / high≤dn，真实停板价
            #   + TICK/2 容差）⇒ blocked 弃单（回测同语义，下一调仓日重排）；停牌无行情
            #   ⇒ 弃单；created >= asof 的同日单不成交（同日重跑幂等，防双成交）。
            n_fill = n_block = n_reject = 0
            for o in db.load_pending():
                code = o["code"]
                if o.get("created") and o["created"] >= str(asof.date()):
                    continue                    # 同日重跑：当日新挂单不在此刻成交
                if db.has_fill(asof_d, code, o["side"], o["lots"]):
                    continue                    # 同日重跑：本日已撮合（现金已由 fills 重放）
                if code not in raw_open.columns or pd.isna(raw_open.at[asof, code]):
                    continue                    # 停牌/无行情 ⇒ 弃单（回测"目标死亡"同语义）
                up = (float(raw_up.at[asof, code])
                      if code in raw_up.columns and pd.notna(raw_up.at[asof, code]) else np.nan)
                dn = (float(raw_dn.at[asof, code])
                      if code in raw_dn.columns and pd.notna(raw_dn.at[asof, code]) else np.nan)
                lo = (float(raw_low.at[asof, code])
                      if code in raw_low.columns and pd.notna(raw_low.at[asof, code]) else np.nan)
                hi = (float(raw_high.at[asof, code])
                      if code in raw_high.columns and pd.notna(raw_high.at[asof, code]) else np.nan)
                side = Side.BUY if o["side"] == "buy" else Side.SELL
                if (side == Side.BUY and pd.notna(up) and pd.notna(lo)
                        and lo >= up - contracts.TICK / 2):
                    n_block += 1                # 全天封死涨停：买不进
                    continue
                if (side == Side.SELL and pd.notna(dn) and pd.notna(hi)
                        and hi <= dn + contracts.TICK / 2):
                    n_block += 1                # 全天封死跌停：卖不出，现状保持
                    continue
                _a = float(raw_adj.at[asof, code]) if code in raw_adj.columns \
                    and pd.notna(raw_adj.at[asof, code]) else 1.0
                rep = broker.fill_order_at(Order(code, side, o["lots"], o["reason"]),
                                           float(raw_open.at[asof, code]), adj=_a)
                if rep.ok:
                    db.record_fill(str(asof.date()), code, o["side"], o["lots"],
                                   rep.price, rep.fee, f"{name}-{n_fill}")
                    n_fill += 1
                else:
                    n_reject += 1     # blocked_cash / 无持仓超卖等硬拦（原实现连失败单也写 fills 表 = 幽灵成交，2026-10-06 修复）
            if n_block:
                db.alert(datetime.now().isoformat(timespec="seconds"), "warn",
                         "fill_blocked",
                         f"{name}: {n_block} 单全天封板未成交弃单 asof={asof.date()}")
            if n_reject:
                db.alert(datetime.now().isoformat(timespec="seconds"), "warn",
                         "fill_rejected",
                         f"{name}: {n_reject} 单撮合硬拦未成交（现金/持仓校验）asof={asof.date()}")
            db.save_pending([])                 # 挂单已决（成交/弃单）；调仓日在 ② 重写
            db.save_positions({c: {"lots": p.lots, "entry_price": p.entry_price,
                                   "adj_entry": p.adj_entry}
                               for c, p in broker.positions.items()})
            snap = broker.account(
                mark_prices={c: prices.get(c, p.entry_price)
                             for c, p in broker.positions.items()},
                adj_prices={c: (float(raw_adj.at[asof, c])
                                if c in raw_adj.columns and pd.notna(raw_adj.at[asof, c])
                                else p.adj_entry)
                            for c, p in broker.positions.items()})

            # ② 信号 → 挂单（仅调仓日重排；见上方"调仓频率闸"注释）
            pending = []
            rep = None
            if is_rebalance:
                gov = RiskGovernor(cfg_t)
                atr_row = feats["atr14"].loc[asof]
                tgts = []
                n_atr_miss = 0
                for c in list(broker.positions):
                    if c not in keep:
                        tgts.append(Target(c, 0, 0, "exit_band(keep_%d)" % tr["keep_n"]))
                for c in top:
                    if c in broker.positions or c not in prices:
                        continue
                    atr = float(atr_row.get(c, 0) or 0)
                    if not np.isfinite(atr) or atr <= 0:
                        n_atr_miss += 1    # sizer 对非有限 ATR 返 0 手；留痕使跳过可观察（2026-10-06）
                    sz = size_position(cfg_t, snap.equity, prices[c], atr)
                    if sz.lots > 0:
                        _why = (f"blendD(D={float(sig_use[c]):.3f})" if sig_kind == "blend"
                                else f"dlv2(ema={float(sig_use[c]):.3f})")
                        tgts.append(Target(c, 1, sz.lots, _why))
                if n_atr_miss:
                    db.alert(datetime.now().isoformat(timespec="seconds"), "warn",
                             "atr_missing",
                             f"{name}: {n_atr_miss} 只 top 候选 ATR 缺失/非正按 0 手跳过 asof={asof.date()}")
                curve = db.equity_curve()
                eq_hist = [r[1] for r in curve]
                # peak/day_pnl 用轨道真实权益曲线（原实现钉 initial equity / day_pnl=0，
                # governor 的回撤闸在整个验证窗内不会正确触发）。
                # day_pnl = 当前权益 − 日初（前一记录日）权益 = 今日已实现+浮动；
                # 日停分母同用日初权益。2026-10-06 审查修复两处口径：
                # ①原 day_pnl 取 eq_hist[-1]−eq_hist[-2] = **昨日**盈亏（当日权益此时尚未落库，
                #   周五调仓时滞后两天）；②原分母用 peak——回撤越深日停越松。
                #
                # 回撤峰值锚（2026-10-08，docs/03 §36 自锁缺陷的复位阀）：熔断解除要求
                # 权益涨回 peak×(1−halt)，而 halted 期间账户只剩现金、权益冻结 ⇒ 实测解除
                # 需上涨 25.3%（阈值只要求回到 −12%）⇒ 一旦触发就再也解不开。操作者复位
                # 后（scripts/reset_dd_peak.py）峰值只在该锚日期之后的记录里取。
                # 不设锚 ⇒ 取全历史（与修复前逐位一致，默认行为不变）。
                anchor = db.get_dd_anchor(name)
                if anchor and anchor.get("date"):
                    a_d = str(anchor["date"])
                    hist = [e for (dt, e) in zip([r[0] for r in curve], eq_hist) if dt > a_d]
                    db.alert(datetime.now().isoformat(timespec="seconds"), "info",
                             "dd_anchor_active",
                             f"{name}: 回撤峰值自 {a_d} 起算（原因：{anchor.get('reason', '?')}）")
                else:
                    hist = eq_hist
                # 回撤峰值口径（2026-10-10，docs/03 §47 自锁缺陷修复）：
                # dd_peak_mode=none（默认）⇒ max(hist+[eq])，与修复前**逐位一致**；
                # rolling ⇒ 只取窗口内，消除"halted 永不解除"的吸收态。
                # 设了 dd_peak_anchor 时仍走 hist（锚优先，操作者显式出口不变）。
                # 口径唯一来源 = fterm/risk/peak.py（回测侧走同一函数，防语义分叉）。
                from ..risk.peak import peak_equity as _pk

                _pk_mode = str(getattr(cfg.account, "dd_peak_mode", "none"))
                _pk_win = int(getattr(cfg.account, "dd_peak_window", 252))
                peak_eq = _pk(hist, snap.equity, mode=_pk_mode, window=_pk_win)
                if _pk_mode == "rolling" and not (anchor and anchor.get("date")):
                    db.alert(datetime.now().isoformat(timespec="seconds"), "info",
                             "dd_peak_rolling",
                             f"{name}: 回撤峰值按滚动 {_pk_win} 交易日计算"
                             f"（dd_peak_mode=rolling，无吸收态）")
                # 自锁检测（2026-10-10，docs/03 §36 缺陷的**可观察性**；零行为改动）：
                # 缺陷本体 = halted 且权益**冻结**（纯现金无变动）⇒ peak 不变 ⇒
                # 熔断永不解除。无论 dd_peak_mode 取何值，只要观测到该状态就报警
                # —— 生产系统必须能"看见"自己卡住了，而不是静默空转数年。
                _selflock_warn = int(getattr(cfg.account, "dd_selflock_warn_days", 60))
                if (_selflock_warn > 0 and snap.equity < peak_eq
                        * (1 - cfg.account.total_dd_halt) and len(eq_hist) >= _selflock_warn):
                    _tail = eq_hist[-_selflock_warn:]
                    if max(_tail) - min(_tail) < 1e-9:
                        db.alert(
                            datetime.now().isoformat(timespec="seconds"), "warn",
                            "dd_halt_selflock",
                            f"{name}: 熔断已触发且权益连续 {_selflock_warn} 日无变动"
                            f"（权益 {snap.equity:,.0f} < 峰值 {peak_eq:,.0f}×"
                            f"({1 - cfg.account.total_dd_halt:.0%})）⇒ 熔断处于**吸收态**，"
                            f"不会自行解除。处置：scripts/reset_dd_peak.py 设锚复位，"
                            f"或改用 dd_peak_mode=rolling（docs/03 §36/§47）")
                day_start = eq_hist[-1] if eq_hist else tr["equity"]
                state = AccountState(
                    equity=snap.equity, peak_equity=peak_eq,
                    day_pnl=snap.equity - day_start, day_start_equity=day_start,
                    # 审批估值用含权现价盯市（2026-10-08 审查批统一，docs/02 §参数基线）：
                    # 旧实现 entry_price×lots 与回测忠实臂（当日收盘盯市）口径不一致，
                    # 现金闸估计随行情偏离，回撤越深偏差越大。
                    open_positions={c: {"direction": 1, "lots": p.lots,
                                        "value": snap.positions[c]["value"]}
                                    for c, p in broker.positions.items()
                                    if c in snap.positions})
                rep = gov.approve(tgts, state, prices=prices)
                for dec in rep.decisions:
                    if not dec.approved:
                        continue
                    tgt = dec.target
                    pending.append({"code": tgt.product,
                                    "side": "sell" if tgt.direction == 0 else "buy",
                                    "lots": tgt.lots, "reason": tgt.reason,
                                    "created": str(asof.date())})
                db.save_pending(pending)

            db.record_equity(str(asof.date()), snap.equity, snap.pos_value, snap.available)
            if rep is not None:
                db.record_decisions(str(asof.date()),
                                    [(d.target.product, d.target.direction, d.target.lots,
                                      bool(d.approved), d.reason) for d in rep.decisions])
            summaries.append({"track": name, "equity": snap.equity,
                              "pos_value": snap.pos_value, "orders": len(pending)})

        db0.heartbeat(datetime.now().isoformat(timespec="seconds"),
                      note=f"dlv2 asof={asof.date()} tracks={len(tracks)}")
        return {"asof": str(asof.date()), "mode": "dlv2", "tracks": summaries,
                "decisions": [], "orders": []}

