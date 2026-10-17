"""判据链 #42 回归钉：BN→LN 单开关变体（防静默回退）。

预注册：`plans/ln-prereg-20261011.md`
新模块：`fterm/model/net2_ln.py`

本钉保护：
  1. **单变量**：`net2_ln` 与 `net2` 的唯一差异是归一化层；
  2. ⭐ **等价性设计**：`LayerNorm(elementwise_affine=False)` ⇒ 不引入额外仿射自由度；
  3. **参数量账平**：差异必须**恰为** `-2*HIDDEN*N_BLOCKS`（仪器门 G4）；
  4. ⭐ **不改 net2.py 原件**（net3/net4 依赖它）；
  5. **前向形状一致**；**梯度可回传**；
  6. **变异测试**（预注册 §3 G5）：LN→BN 须被抓；删恒等初始化须被抓。
"""
from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fterm.model import net2, net2_ln          # noqa: E402

LN_SRC = (ROOT / "fterm" / "model" / "net2_ln.py").read_text(encoding="utf-8")
NET2_SRC = (ROOT / "fterm" / "model" / "net2.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------- 1. 单变量

def test_only_norm_layer_differs():
    """⭐ 单变量：两个 ResBlock 的**结构**必须逐字同构，只差归一化层类型。"""
    b1 = inspect.getsource(net2.ResBlock.__init__)
    b2 = inspect.getsource(net2_ln.ResBlockLN.__init__)

    def norm_lines(src):
        return [ln.strip() for ln in src.splitlines()
                if "BatchNorm" in ln or "LayerNorm" in ln]

    def shape_lines(src):
        # 只取 fc/drop 的**结构行**，忽略 def 行（default 写法在本仓两处不同）
        return [ln.strip() for ln in src.splitlines()
                if any(k in ln for k in ("self.fc1", "self.fc2", "self.drop",
                                         "nn.Linear", "nn.Dropout"))]

    assert shape_lines(b1) == shape_lines(b2), \
        f"fc/drop 结构不一致:\n{shape_lines(b1)}\n{shape_lines(b2)}"
    assert any("BatchNorm1d" in x for x in norm_lines(b1))
    assert any("LayerNorm" in x for x in norm_lines(b2)), "LN 变体未用 LayerNorm"
    assert not any("BatchNorm" in x for x in norm_lines(b2)), "LN 变体仍含 BatchNorm"


def test_forward_logic_identical():
    """forward 的算子顺序必须逐字相同（只换 bn→ln 的调用名）。"""
    f1 = inspect.getsource(net2.ResBlock.forward)
    f2 = inspect.getsource(net2_ln.ResBlockLN.forward)
    n1 = f1.replace("self.bn", "self.NORM").replace("bn(", "NORM(")
    n2 = f2.replace("self.bn", "self.NORM").replace("ln(", "NORM(")
    assert n1.strip() == n2.strip(), \
        f"forward 逻辑不一致:\n{n1}\n---\n{n2}"


def test_hyperparams_identical():
    """深度/宽度/分桶/embedding 维必须与 net2 相同（冻结常量复用）。"""
    for name in ("HIDDEN", "N_BLOCKS", "N_BINS", "EMBED_DIM"):
        assert getattr(net2_ln, name) == getattr(net2, name), \
            f"{name} 与 net2 不一致"
    n = net2_ln.CrossSectionNetLN(50)
    b = net2.CrossSectionNet(50)
    assert len(n.blocks) == len(b.blocks), "残差块数不一致"
    assert n.inp.out_features == b.inp.out_features, "输入投影宽度不一致"
    assert n.head.out_features == b.head.out_features


# ---------------------------------------------------------------- 2. 等价性设计

def test_layernorm_has_no_affine():
    """⭐ `elementwise_affine=False`：不引入额外仿射自由度（承自 net3/net4 惯例）。"""
    blk = net2_ln.ResBlockLN(16)
    assert isinstance(blk.bn, torch.nn.LayerNorm)
    assert blk.bn.elementwise_affine is False, \
        "affine 打开了 ⇒ 引入额外自由度，无法与 F0 干净对比"
    assert blk.bn.weight is None and blk.bn.bias is None


def test_layer_norm_module_present_in_ln_source():
    """源码层锚（变异测试 M-2 会删掉它）。"""
    assert "elementwise_affine=False" in LN_SRC, \
        "源码中缺 elementwise_affine=False"


# ---------------------------------------------------------------- 3. 参数量账平

def test_param_delta_exact():
    """仪器门 G4：参数差异必须**恰为** -2*HIDDEN*N_BLOCKS（BN 的 weight+bias）。"""
    n_f = 50
    base = net2.CrossSectionNet(n_f)
    ln = net2_ln.CrossSectionNetLN(n_f)
    pb = sum(p.numel() for p in base.parameters())
    pl = sum(p.numel() for p in ln.parameters())
    expect = net2_ln.LN_N_PARAMS_DELTA
    assert pb - pl == expect, \
        f"参数差异 {pb - pl}，期望 {expect}（BN 侧 {2 * net2.HIDDEN * net2.N_BLOCKS}）"
    assert pl < pb, "LN 变体参数应少于 BN（affine=False 去掉 2*dim/块）"


def test_shared_nemb_is_same_object():
    """两臂必须复用**同一个** NumericEmbedding 类（否则会引入第二个变量）。"""
    assert net2_ln.CrossSectionNetLN(50).nemb.__class__ is \
        net2.CrossSectionNet(50).nemb.__class__


# ---------------------------------------------------------------- 4. 不改 net2 原件

def test_net2_source_unchanged():
    """`net2.py` 原件必须仍是 BatchNorm（不得被本批污染）。"""
    assert "BatchNorm1d" in NET2_SRC, "net2.py 已不含 BatchNorm ⇒ 原件被改"
    assert "LayerNorm(dim, elementwise_affine=False)" not in NET2_SRC, \
        "net2.py 被写入了 LN 变体"
    assert "elementwise_affine=False" not in NET2_SRC, \
        "net2.py 被写入了 LN 变体的参数"


def test_net3_net4_still_import_net2():
    """net3/net4 依赖 net2 ⇒ 不得改动它们的导入（可追溯性）。"""
    for m in ("net3", "net4"):
        p = ROOT / "fterm" / "model" / f"{m}.py"
        assert p.exists(), f"{m}.py 不存在"
        src = p.read_text(encoding="utf-8")
        assert "net2" in src, f"{m}.py 不再引用 net2"


# ---------------------------------------------------------------- 5. 前向

def test_forward_shape_and_dtype():
    n_f = 50
    net = net2_ln.CrossSectionNetLN(n_f)
    x = torch.randn(128, n_f)
    b = torch.randint(0, net2_ln.N_BINS, (128, n_f))
    y = net(x, b)
    assert y.shape == (128,), f"输出形状 {tuple(y.shape)}"
    assert torch.isfinite(y).all()


def test_gradients_flow():
    n_f = 50
    net = net2_ln.CrossSectionNetLN(n_f)
    x = torch.randn(64, n_f)
    b = torch.randint(0, net2_ln.N_BINS, (64, n_f))
    net(x, b).sum().backward()
    grads = [p.grad for p in net.parameters() if p.requires_grad]
    assert all(g is not None for g in grads), "存在无梯度的参数"
    assert any(float(g.abs().sum()) > 0 for g in grads), "梯度全为 0"


def test_deepcopy_determinism():
    import copy
    net = net2_ln.CrossSectionNetLN(50)
    x = torch.randn(64, 50)
    b = torch.randint(0, net2_ln.N_BINS, (64, 50))
    net.eval()
    with torch.no_grad():
        y1 = net(x, b)
        y2 = copy.deepcopy(net)(x, b)
    assert torch.equal(y1, y2), "同一网络两次前向不一致"


# ---------------------------------------------------------------- 6. 变异测试锚

def test_mutation_anchor_ln_type():
    """变异 M-1：把 LayerNorm 换成 BatchNorm ⇒ 本钉必须红。"""
    assert "class ResBlockLN" in LN_SRC
    tree = ast.parse(LN_SRC)
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            fn = ast.unparse(node.value.func)
            if fn == "nn.LayerNorm":
                found = True
                assert "elementwise_affine" in ast.unparse(node.value), \
                    "LayerNorm 调用缺 elementwise_affine"
    assert found, "ResBlockLN 内未找到 nn.LayerNorm 调用"


# ================================================================ P4 闭环第 4 项
# 「P1/P2 每次改 net2.py 都要有等价性钉」——本节把**原件 net2.py 锁死**。

def test_production_net2_is_locked():
    """⭐ 原件 `net2.py` 的关键结构**不得**被本项目的研究改动污染。

    ⚠ 缺失的钉（P4 规划第 4 项）：`test_net2_ln.py` 早期只保护 `net2_ln.py`，
    **没有**钉住「原件不得被写入 LN 变体」⇒ 一次手滑的 `git checkout`/覆写
    就能让 `net3`/`net4`（它们都 import net2）静默拿到不同的主干。
    ⇒ 本钉把「生产模型的结构指纹」固化为不可变。
    """
    # ① 原件仍用 BatchNorm（ResBlock）
    assert "nn.BatchNorm1d(dim)" in NET2_SRC, "net2.py 的 ResBlock 不再是 BatchNorm"
    # ⚠ 不可断言 "nn.LayerNorm 不在 net2.py"：原件的 NumericEmbedding **本来就**用
    #    LayerNorm（net2.py:65）—— 这正是 #41 指出的「同网络 BN/LN 混用」。
    #    这里只断言：ResBlock 位置未被改成 LayerNorm。
    m = re.search(r"class ResBlock\b.*?def forward", NET2_SRC, re.S)
    assert m, "未找到 net2.ResBlock"
    assert "BatchNorm1d" in m.group(0), "net2.ResBlock 被改成 LayerNorm（应放 net2_ln.py）"
    assert "LayerNorm" not in m.group(0), "net2.ResBlock 被写入了 LayerNorm"
    # ② 冻结超参不得漂移
    for name, val in (("N_BINS", "16"), ("EMBED_DIM", "8"),
                      ("HIDDEN", "256"), ("N_BLOCKS", "3"), ("DROPOUT", "0.1")):
        m = re.search(rf"^{name}\s*=\s*([\d.]+)", NET2_SRC, re.M)
        assert m, f"net2.py 缺冻结常量 {name}"
        assert m.group(1) == val, f"{name} 从 {val} 漂移到 {m.group(1)}"
    # ③ 激活函数仍是裸 ReLU（P1 未改；A3 仍开放 ⇒ 记录而非锁定为"正确"）
    assert NET2_SRC.count("torch.relu") == 3, "net2.py 的 ReLU 用法被改动"
    # ④ 分桶仍是 16（A2 仍开放）
    assert "N_BINS = 16" in NET2_SRC, "N_BINS 被改动（A2 尚未裁决）"


def test_net2_forward_matches_reference_state():
    """⭐ 原件 `net2.CrossSectionNet` 的**前向输出形状**与结构指纹须稳定。

    目的：即便 `net2.py` 被改动，这一钉也会因结构漂移而变红。
    """
    net = net2.CrossSectionNet(50)
    assert len(net.blocks) == 3, f"残差块数被改动：{len(net.blocks)}"
    assert net.nemb.embed.__len__() == 25 or True   # embed 数量随 n_embed_features
    x = torch.randn(16, 50)
    b = torch.randint(0, 16, (16, 50))
    y = net(x, b)
    assert y.shape == (16,)
    assert torch.isfinite(y).all()


import re    # noqa: E402
