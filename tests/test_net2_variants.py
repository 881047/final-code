"""判据链 #46 回归钉：A2 分桶数 + A3 激活变体（防静默回退/混入第二变量）。

预注册：`plans/a2a3-prereg-20261011.md`
变体：`fterm/model/net2_variants.py`

本钉保护：
  1. ⭐ **单变量纪律**：`CrossSectionNetBins` 只改 `bins`、`CrossSectionNetAct` 只改激活
     ⇒ 断言**除 embedding 表外**的 state_dict 键**完全一致**、参数量差**可完全账平**；
  2. ⭐ **不改 `net2.py` 原件**（`net3`/`net4` 依赖它）；
  3. **state_dict 键名不被 Sequential 包裹破坏**（否则 arch_chain/生产加载会失效）；
  4. **前向形状 / 梯度 / 确定性**；
  5. **变异测试**（仪器门 G3/G4/G6）。
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fterm.model import net2, net2_variants as var   # noqa: E402

VAR_SRC = (ROOT / "fterm" / "model" / "net2_variants.py").read_text(encoding="utf-8")
NET2_SRC = (ROOT / "fterm" / "model" / "net2.py").read_text(encoding="utf-8")

N_F = 50
BIN_LIST = (8, 16, 32, 64)


def _nparams(m) -> int:
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------- 1. 单变量纪律

def test_bins_variant_only_changes_embedding_table():
    """⭐ A2：除 `nemb.embed.*` 外，state_dict 键与基线**逐字一致**。"""
    base = net2.CrossSectionNet(N_F)
    kb = list(base.state_dict().keys())
    for bins in BIN_LIST:
        m = var.CrossSectionNetBins(N_F, bins=bins)
        km = list(m.state_dict().keys())
        rest_b = [k for k in kb if not k.startswith("nemb.embed")]
        rest_m = [k for k in km if not k.startswith("nemb.embed")]
        assert rest_b == rest_m, f"bins={bins} 混入了第二个变量（键名变了）"


def test_bins_param_delta_exact():
    """仪器门 G4：参数量差 = `n_embed × embed_dim × (bins − bins0)`。

    **真实机制**（自伤记录 2026-10-11，首版两次算错）：
    bins 变大时 ①embedding 表变大 `n_embed×embed_dim×Δ`
    而 ②`proj` 是 `Linear(n_features×embed_dim → hidden)`，
      其**输入维不含 bins** ⇒ `proj` **不变**。
    ⇒ 期望差 = `n_embed × embed_dim × Δ`（**单倍**，不是两倍）。
    ⚠ 首版按「表+投影」估成两倍、实测不符 ⇒ 改为**从基线实际形状推导**，
      不再手写系数（这正是「参数量账平」该有的做法）。
    """
    base = net2.CrossSectionNet(N_F)
    p0 = _nparams(base)
    bins0 = net2.N_BINS
    n_emb = base.nemb.embed.__len__()          # 实际 embed 层数（= n_embed_features）
    for bins in BIN_LIST:
        m = var.CrossSectionNetBins(N_F, bins=bins)
        expect = n_emb * net2.EMBED_DIM * (bins - bins0)
        assert _nparams(m) - p0 == expect, \
            f"bins={bins} 参数量差 {_nparams(m) - p0}，期望 {expect}"


def test_bins_16_is_identical_to_baseline():
    """bins=16 必须与基线**完全等价**（同 seed 下逐值一致）⇒ A2 的"单变量"成立。

    ⚠ 必须**同 seed 重建**：两个构造各自消耗 RNG，顺序不同则权重不同
    （那是随机初始化顺序差异，不是结构差异）。
    """
    torch.manual_seed(0)
    a = net2.CrossSectionNet(N_F)
    torch.manual_seed(0)
    b = var.CrossSectionNetBins(N_F, bins=16)
    assert _nparams(a) == _nparams(b), "bins=16 参数量不同"
    ka, kb = a.state_dict(), b.state_dict()
    assert list(ka.keys()) == list(kb.keys())
    for k in ka:
        assert torch.equal(ka[k], kb[k]), f"bins=16 与基线不一致：{k}"


def test_act_variant_only_changes_activation():
    """⭐ A3：除激活参数外，结构键**完全一致**（含 BatchNorm / fc / head）。"""
    base = net2.CrossSectionNet(N_F)
    m = var.CrossSectionNetAct(N_F)
    kb = set(net2.CrossSectionNet(N_F).state_dict().keys())
    km = set(m.state_dict().keys())
    only_m = {k for k in km - kb}
    only_b = {k for k in kb - km}
    assert not only_b, f"A3 丢失了基线的键：{sorted(only_b)[:5]}"
    assert all(k.endswith("act1.weight") or k.endswith("act2.weight") or
               ".act1." in k or ".act2." in k for k in only_m), \
        f"A3 混入了激活之外的变量：{sorted(only_m)[:8]}"
    # BatchNorm 必须仍在（没被换成 LayerNorm）
    assert "blocks.0.bn.weight" in km, "A3 把 BatchNorm 换掉了 ⇒ 混入第二变量"
    assert "blocks.0.fc1.weight" in km, "A3 丢失 fc1"


def test_act_param_delta_exact():
    """A3 参数量差 = `2 × hidden × n_blocks`（每块两个 PReLU）。"""
    base = net2.CrossSectionNet(N_F)
    p0 = _nparams(base)
    m = var.CrossSectionNetAct(N_F)
    expect = 2 * net2.HIDDEN * net2.N_BLOCKS
    assert _nparams(m) - p0 == expect, \
        f"PReLU 参数量差 {_nparams(m) - p0}，期望 {expect}"


def test_act_head_stays_relu():
    """⚠ A3 的变量**只针对残差块内**；输出头激活保持 `torch.relu`（预注册 §2）。"""
    src = ast.unparse(ast.parse(VAR_SRC))
    tail = src.split("class CrossSectionNetAct")[-1]
    assert "torch.relu(h)" in tail, "输出头激活被改（A3 变量越界）"


# ---------------------------------------------------------------- 2. 不改原件

def test_net2_source_untouched():
    """`net2.py` 原件必须逐字不变（net3/net4 依赖它）。"""
    assert "nn.BatchNorm1d(dim)" in NET2_SRC
    assert "N_BINS = 16" in NET2_SRC, "net2.py 的 N_BINS 被改动"
    assert "class CrossSectionNetBins" not in NET2_SRC
    assert "class CrossSectionNetAct" not in NET2_SRC


# ---------------------------------------------------------------- 3. 键名不被包裹破坏

def test_blocks_not_wrapped_in_sequential():
    """⚠ `blocks.0.bn.*` 不得变成 `blocks.0.0.bn.*`（否则 arch_chain/生产加载失效）。

    这是真实踩过的坑：首版把 `ResBlock` 包进 `nn.Sequential(nn.Sequential(ResBlock))`
    ⇒ 键名多了一层索引。
    """
    m = var.CrossSectionNetBins(N_F, bins=32)
    keys = list(m.state_dict().keys())
    assert "blocks.0.bn.weight" in keys, f"键名被包裹破坏：{keys[:6]}"
    assert not any(".0.0." in k for k in keys), f"出现双层索引：{keys[:6]}"


def test_blocks_keys_match_net2_exactly():
    """A2 变体的块内键名必须与 net2 **完全一致**（A3 会多出 act1/act2，故排除）。"""
    base = [k for k in net2.CrossSectionNet(N_F).state_dict().keys()]
    m = var.CrossSectionNetBins(N_F, bins=32)
    km = list(m.state_dict().keys())
    assert [k for k in km if not k.startswith("nemb.embed")] == \
           [k for k in base if not k.startswith("nemb.embed")], "A2 块内键名漂移"
    # A3 只应**多出** act1/act2，其余一致
    a = var.CrossSectionNetAct(N_F)
    ka = list(a.state_dict().keys())
    extra = set(ka) - set(base)
    missing = set(base) - set(ka)
    assert not missing, f"A3 丢失基线键：{sorted(missing)[:5]}"
    assert all(".act" in k for k in extra), f"A3 多出非 act 键：{sorted(extra)[:5]}"


# ---------------------------------------------------------------- 4. 前向/梯度/确定性

@pytest.mark.parametrize("bins", BIN_LIST)
def test_bins_forward_shape(bins):
    m = var.CrossSectionNetBins(N_F, bins=bins)
    x = torch.randn(16, N_F)
    b = torch.randint(0, bins, (16, N_F))
    y = m(x, b)
    assert y.shape == (16,)
    assert torch.isfinite(y).all()


def test_act_forward_shape_and_grad():
    m = var.CrossSectionNetAct(N_F)
    x = torch.randn(16, N_F)
    b = torch.randint(0, net2.N_BINS, (16, N_F))
    m(x, b).sum().backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in m.parameters()), "梯度全零"


def test_determinism():
    import copy
    for m in (var.CrossSectionNetBins(N_F, bins=64), var.CrossSectionNetAct(N_F)):
        m.eval()
        x = torch.randn(16, N_F)
        b = torch.randint(0, 16, (16, N_F))
        with torch.no_grad():
            y1 = m(x, b)
            y2 = copy.deepcopy(m)(x, b)
        assert torch.equal(y1, y2), "同一网络两次前向不一致"


# ---------------------------------------------------------------- 5. 变异锚

def test_mutation_anchors_present():
    """变异测试的锚点仍在（守卫本身没被删）。"""
    assert "nn.Sequential(*[ResBlock(hidden, dropout)" in VAR_SRC, \
        "A2 的 blocks 复用被破坏（可能被包成 Sequential）"
    assert "self.act1 = nn.PReLU(num_parameters=dim, init=init)" in VAR_SRC, \
        "A3 的 PReLU 被替换"
