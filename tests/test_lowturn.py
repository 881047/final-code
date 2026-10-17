# -*- coding: utf-8 -*-
"""#137 低换手链回归钉（2026-10-16）。

① arm_faithful.reb_mod 默认=1（零行为不变不变量——默认路径逐位不变的守卫）；
② 四臂定义单变量性（除注明变量外全键与 W2_TIER 相同；K4W2 只差 reb_mod）。

变异验证（开发时手动执行过）：
- ① `reb_mod: int = 1` 改为 `0` ⇒ test_reb_mod_default_is_one 必红；
- ② K8W1 的 keep_n=8 改为 6 ⇒ test_arm_definitions_single_variable 必红。
"""
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments" / "lowturn_20261016" / "scripts"))


def test_reb_mod_default_is_one():
    import tierfix_replay as TF
    sig = inspect.signature(TF.arm_faithful)
    assert "reb_mod" in sig.parameters, "arm_faithful 须有 reb_mod 参数（#137）"
    assert sig.parameters["reb_mod"].default == 1, (
        "reb_mod 默认必须为 1（>1 才取子序列；默认不变不变量）")


def test_arm_definitions_single_variable():
    from lowturn_chain import ARMS, W2_TIER
    base = dict(W2_TIER)
    # 锚：与 W2_TIER 全键等值 + reb_mod=1
    assert ARMS["K4W1"] == (base, 1)

    # K8W1 / K12W1：只差 keep_n
    for nm, keep in (("K8W1", 8), ("K12W1", 12)):
        tier, rm = ARMS[nm]
        diff = {k for k in set(base) | set(tier) if base.get(k) != tier.get(k)}
        assert diff == {"keep_n"}, f"{nm} 非单变量：{diff}"
        assert tier["keep_n"] == keep and rm == 1

    # K4W2：tier 完全同底座，只差 reb_mod=2
    tierW, rmW = ARMS["K4W2"]
    assert tierW == base and rmW == 2
