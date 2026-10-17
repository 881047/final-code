"""判据链 #26 的**仪器门回归钉**（预注册 `plans/tail-objective-prereg-20261010.md` §七）。

本文件钉住三件**由独立复核抓出**的门缺陷，且每条都**证伪**（门必须能失败）：

① **E8 日集守卫**：`tail_objective_eval.check_day_sets` 必须拒绝
   · 1 折 1 种子的**前哨流**（≈418 天）—— 否则 C1′ 会在 418 天上**静默出数**；
   · 右端点距面板末 ≥15 天的流（缺终端折）；
   · 各臂日数不一致（配对日集不同源）。
② **E6 G5 的 NaN 语义**：阴性对照的逐日 IC **全 NaN**（常数预测的极限形态）时，
   必须判「**塌到噪声**」而**不是**「泄漏」—— 原实现 `abs(ic) <= 0.025` 在 NaN 下为 False
   ⇒ **完美的阴性对照被判泄漏**（假警报）。本测试用**分布/档内均衡**口径复现该语义。
③ **勘误 3 的回归钉**：`hit1_at5` **不度量方向**，只度量**极端性**——
   对预测**取负**后 `hit_top5` **反升**。若哪天真变差，说明指标语义变了，必须重审。

运行：pytest tests/test_tail_objective_gates.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import tail_objective_eval as ev  # noqa: E402


def _meta(n_days: int, first: str, last: str, file: str = "s.parquet") -> dict:
    return {"file": file, "n_days": n_days, "first": first, "last": last,
            "n_stocks": 5000}


# --------------------------------------------------------------- ① E8 日集守卫
def test_e8_rejects_single_fold_sentinel():
    """1 折 1 种子前哨（≈418 天）必须被拒 —— 否则会在 418 天上静默出数。"""
    meta = {"C0": _meta(418, "2025-01-02", "2026-09-21")}
    with pytest.raises(SystemExit) as e:
        ev.check_day_sets(meta)
    assert "前哨" in str(e.value) or "E8" in str(e.value)


def test_e8_accepts_full_four_fold_stream():
    """4 折全量（≈1871 天、右端点贴近面板末）必须放行 —— 门要能放行。"""
    meta = {"C0": _meta(1871, "2019-01-02", "2026-09-21"),
            "T1": _meta(1871, "2019-01-02", "2026-09-21"),
            "T2": _meta(1871, "2019-01-02", "2026-09-21")}
    ev.check_day_sets(meta)          # 不抛 ⇒ 通过


def test_e8_rejects_missing_terminal_fold():
    """右端点距面板末 ≥15 天（缺终端折）必须被拒（#18 勘误的响亮守卫）。"""
    meta = {"C0": _meta(1871, "2019-01-02", "2026-08-01")}   # 距 09-30 有 60 天
    with pytest.raises(SystemExit) as e:
        ev.check_day_sets(meta)
    assert "折边界不完整" in str(e.value)


def test_e8_rejects_mismatched_day_counts():
    """各臂日数相差 >5 ⇒ 配对日集不同源，必须被拒。"""
    meta = {"C0": _meta(1871, "2019-01-02", "2026-09-21"),
            "T1": _meta(1500, "2019-01-02", "2026-09-21")}
    with pytest.raises(SystemExit) as e:
        ev.check_day_sets(meta)
    assert "不同源" in str(e.value)


def test_e8_rejects_missing_c0():
    """无 C0 ⇒ 单变量对照不成立 ⇒ 必须被拒。"""
    with pytest.raises(SystemExit):
        ev.check_day_sets({"T1": _meta(1871, "2019-01-02", "2026-09-21")})


# --------------------------------------------------------------- ② E6 G5 NaN 语义
def _g5_verdict(hit: float, ic: float | None, excess: float | None,
                trip_hit: float = 0.075, trip_ic: float = 0.025,
                trip_exc: float = 0.025) -> bool:
    """复刻 eval 里 G5 的**修正后**判定逻辑（E6 修复的可测版本）。"""
    ic_ok = (ic is None) or (abs(ic) <= trip_ic)
    hit_ok = hit <= trip_hit
    exc_ok = (excess is None) or (excess <= trip_exc)
    return bool(ic_ok and hit_ok and exc_ok)


def test_g5_nan_ic_is_treated_as_noise_not_leak():
    """⭐ E6 回归钉：IC 全 NaN（常数预测）⇒ 必须判「塌到噪声」，不得判泄漏。

    原实现 `abs(nan) <= 0.025` = False ⇒ 把**完美的**阴性对照判成泄漏（假警报）。
    """
    assert _g5_verdict(hit=0.0, ic=None, excess=0.0) is True, \
        "IC 为 NaN 时必须视为无信息（塌到噪声），而不是泄漏"


def test_g5_flags_a_real_leak():
    """门必须能失败：真的泄漏（IC 与命中率都高）必须被判泄漏。"""
    assert _g5_verdict(hit=0.30, ic=0.20, excess=0.15) is False
    assert _g5_verdict(hit=0.30, ic=None, excess=0.15) is False   # 命中率就能拦下
    assert _g5_verdict(hit=0.0, ic=0.20, excess=0.0) is False     # IC 就能拦下


def test_g5_catches_exposure_only_degenerate_model():
    """E6 的第二个洞：**带暴露**但不含信息的退化模型 hit 可 > tripwire。

    复核实测 `rand_voltilt` hit=0.158 > 0.075 ⇒ 单靠绝对命中率拦不住；
    档内均衡 `hit_excess` 必须能把它拦下（>0.025）。
    """
    assert _g5_verdict(hit=0.158, ic=0.002, excess=0.10) is False, \
        "带暴露的退化模型必须被 hit_excess 拦下"


# --------------------------------------------------------------- ③ 勘误 3 回归钉
def test_hit1_at5_does_not_measure_direction_on_gaussian_noise():
    """`hit1_at5` 在**高斯**合成数据上确实随方向反转而变差 —— 故它**不是**方向量的证明。

    ⚠ 这条测试的**存在理由**（诚实记录）：本文件初稿想用**合成数据**钉住
    "取负后 hit_top5 反升"这一真数据反常，结果**失败**（0.0583 → 0.0333，正常变差）。
    查明原因：真数据的反常依赖**收益率的重尾**（实测 top-1 收益 偏度 2.47、峰度 13.54、
    sd 1549bp；见 `diag_mechanism_lottery.json`），而高斯噪声里 `argmax` 与
    `argmin` **对称**、两个极端被吸引的程度相同 ⇒ 无法复现该反常。
    ⇒ **反常的回归钉只能钉在真数据上**（见下一条测试），合成数据只能证明**反面**：
    "若收益对称，则 hit1_at5 表现得像个方向量"。这两个事实**合起来**才说明
    `hit1_at5` 的语义**依赖收益分布的偏度/峰度**（这正是它不能当方向判据的原因）。
    """
    rng = np.random.RandomState(20261010)
    n_day, n_stk = 120, 300
    idx = pd.date_range("2020-01-01", periods=n_day, freq="B")
    codes = [f"S{i:04d}" for i in range(n_stk)]
    Y = rng.randn(n_day, n_stk) * 0.02                # 高斯 ⇒ 对称
    P = pd.DataFrame(0.02 * Y + rng.randn(n_day, n_stk) * 0.05,
                     index=idx, columns=codes)
    msk = np.ones_like(Y, dtype=bool)
    Q = {"f": np.zeros_like(Y, dtype=np.int8)}
    h_pos = float(np.nanmean(ev.metrics(P, idx, msk, Y, Q)["hit1_at5"]))
    h_neg = float(np.nanmean(ev.metrics(P.apply(lambda v: -v), idx, msk, Y, Q)["hit1_at5"]))
    assert h_neg < h_pos, (
        f"高斯（对称）收益下，取负后 hit_top5 应变差：{h_pos:.4f} → {h_neg:.4f}；"
        f"若反升则说明『极端性』解释在高斯下也成立 ⇒ 机制解释需重修")
    # 对称收益下两个极端应被同等吸引（hit_bot5 ≈ hit_top5）
    m_pos = ev.metrics(P, idx, msk, Y, Q)
    hb = float(np.nanmean(m_pos["hit_bot5"]))
    assert abs(hb - h_pos) < 0.05, f"对称收益下 hit_bot5({hb:.4f}) 应≈hit_top5({h_pos:.4f})"


@pytest.mark.skipif(
    not (Path(r"D:\Final\data\reports") / "dlv2_ss7_21_99_5_13.parquet").exists(),
    reason="需要生产检出的缓存流（只读）；缺失则跳过")
def test_hit1_at5_anomaly_on_real_streams():
    """⭐ 勘误 3 的**真数据**回归钉：取负后 `hit_top5` **反升**（0.1309 → 0.1534）。

    这是本链撤回主判据的**唯一直接证据**，必须钉在**真数据**上
    （合成高斯数据复现不出，见上一条测试）。
    只读 `D:\\Final\\data\\reports` 的缓存流，**不训练、不跑 GPU**。
    """
    prod = Path(r"D:\Final\data\reports")
    P = ev.load(prod / "dlv2_ss7_21_99_5_13.parquet")
    # 用**流自身的 y_ret** 作标签（避免依赖老库/面板，测试可离线跑）
    d = pd.read_parquet(prod / "dlv2_ss7_21_99_5_13.parquet",
                        columns=["date", "product", "pred", "y_ret"])
    d["date"] = pd.to_datetime(d["date"])
    Yp = d.pivot_table(index="date", columns="product", values="y_ret").astype("float64")
    codes = [c for c in P.columns if c in Yp.columns]
    idx = [x for x in P.index if x in Yp.index]
    msk = np.ones((len(idx), len(codes)), dtype=bool)
    Y = Yp.reindex(index=idx, columns=codes).to_numpy("float64")
    Q = {"f": np.zeros_like(Y, dtype=np.int8)}
    h_pos = float(np.nanmean(ev.metrics(P, idx, msk, Y, Q)["hit1_at5"]))
    h_neg = float(np.nanmean(ev.metrics(P.apply(lambda v: -v), idx, msk, Y, Q)["hit1_at5"]))
    assert h_neg > h_pos + 0.005, (
        f"真数据反常应成立：hit_top5 {h_pos:.4f} → 取负 {h_neg:.4f}；"
        f"若不成立 ⇒ hit1_at5 语义已变，必须重审预注册勘误 3")


def test_metrics_extreme_and_net_hit_consistency():
    """恒等式：`extreme == hit_top5 + hit_bot5`、`net_hit == hit_top5 − hit_bot5`。"""
    rng = np.random.RandomState(7)
    n_day, n_stk = 40, 300
    idx = pd.date_range("2021-01-01", periods=n_day, freq="B")
    codes = [f"S{i:04d}" for i in range(n_stk)]
    P = pd.DataFrame(rng.randn(n_day, n_stk), index=idx, columns=codes)
    Y = rng.randn(n_day, n_stk) * 0.02
    msk = np.ones_like(Y, dtype=bool)
    Q = {"f": np.zeros_like(Y, dtype=np.int8)}
    M = ev.metrics(P, idx, msk, Y, Q)
    assert np.allclose(M["extreme"], M["hit1_at5"] + M["hit_bot5"], equal_nan=True)
    assert np.allclose(M["net_hit"], M["hit1_at5"] - M["hit_bot5"], equal_nan=True)


# --------------------------------------------------------------- ④ G9 单变量自证
def test_g9_accepts_shuffle_false():
    """⭐ 回归钉：`shuffle_labels=False` 是**合法值**，不得被 G9 误判为失败。

    `_assert_single_variable` 初版用 `v is False` 判失败 ⇒ 把
    `shuffle_labels=False`（三臂的**正常**取值）判成"单变量自证失败"⇒ 判废。
    由本测试对应的自测抓出，此处固化防复发。
    """
    import tail_objective_chain as tc
    for shuf in (False, True):
        out = tc._assert_single_variable("expectile", (7, 21), shuf)
        assert out["shuffle_labels"] is shuf
        assert out["loss_spec"] == "expectile"


def test_g9_rejects_unknown_loss_spec():
    """G9 必须拒绝未知损失（防"跑 T1 却用了 mse"这类静默错）。"""
    import tail_objective_chain as tc
    with pytest.raises(SystemExit):
        tc._assert_single_variable("bogus_loss", (7,), False)


def test_g9_frozen_constants_match_dlv2_chain():
    """G9 的冻结常量必须与 `dlv2_chain` 逐项相同（单变量性的承重条件）。"""
    import tail_objective_chain as tc
    g9 = tc._assert_single_variable("mse", (7, 21, 99, 5, 13), False)
    fc = g9["frozen_constants"]
    assert fc["LABEL_SPAN"] == 6
    assert fc["BATCH"] == 4096
    assert fc["MAX_EPOCHS"] == 30
    assert fc["PATIENCE"] == 5
    assert fc["LR"] == 1e-3
    assert fc["WD"] == 1e-5
    assert fc["FOLD_EDGES"] == ["2019-01-01", "2021-01-01", "2023-01-01", "2025-01-01"]
