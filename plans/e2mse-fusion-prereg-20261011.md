# 预注册 · 判据链 #27：E2+MSE 融合 ——「方向 + 极端性」能否两头都保（2026-10-11 冻结）

> **触发**：用户选方向三（E2+MSE 融合重审），融合方式选**可学习权重（Kendall）**。
> **标定批**（先于本预注册，`_calib_e2_fusion_preactivation.py`）的承重读数：
>
> | 量 | E2 | MSE | E2−MSE |
> |---|---|---|---|
> | 全域 IC（full） | **0.0975** | 0.0580 | 配对 t=**+5.92**（显著） |
> | 全域 IC（late） | 0.1014 | 0.0668 | 配对 t=**+4.33** |
> | 宽 k=50 篮子超额 | 11.4bp | **31.1bp** | **−19.7bp**（t=−1.75） |
> | 宽 k=100 | 11.9bp | **28.7bp** | **−16.7bp** |
> | 宽 k=450 | 14.1bp | 24.0bp | **−9.9bp** |
>
> ⇒ E2 的 IC 增益**显著为正**（动机成立），但**宽篮子更薄**（顶端 50/100/450 名把"极端性"丢了）。
> **融合要回答的问题**：能否让全域 IC 保住 E2 水平、同时顶端篮子保住 MSE 水平？

---

## §一 待检验假设（本预注册的唯一对象）

> **H3**：E2+MSE 融合（可学习 Kendall 权重）相对两个父臂（E2 单臂 / MSE 单臂），
> 能同时满足 ① 全域 IC **不劣于** E2（保住"方向"），且 ② 宽 k 篮子超额 **优于** E2（把"极端性"拿回来）。

**为什么"互补"是合理先验**：两臂短板恰是对方强项 —— E2 条件方向性 0.836（#26）但极端性最低；
MSE 极端性厚（顶端篮子 31.1bp）但全域 IC 低。Kendall 可学习权重让模型自己分配两者的量。

---

## §二 臂（单变量；其余逐字冻结）

| 臂 | 训练目标 | 说明 |
|---|---|---|
| **F0** | 纯 MSE | 同代码新鲜对照（`dlv2_chain` 损失）|
| **F2** | 纯 ListMLE（=E2） | 重跑，保证同种子/同面板/同路径 |
| **F3** | **ListMLE + MSE，Kendall 可学习权重** | 唯一新臂 |

**冻结项**（与 `dlv2_chain` 逐字对齐，同 #26）：`FOLD_EDGES` / `START_OOS` / 净化 `LABEL_SPAN*2` /
`CrossSectionNet(50)` / `LR=1e-3` / `WD=1e-5` / `BATCH=4096` / `MAX_EPOCHS=30` / `PATIENCE=5` /
val = 训练段末 15% / 选模 val MSE / 5 网集成 / 种子集 `(7,21,99,5,13)` / 面板钉 2026-09-30。

**⚠ 与 #26 的两点关键区别**：
1. **ListMLE 需要完整日截面** ⇒ 用**按日整批**（`daybatch.py`，同 #26 臂 D/E 的口径）；
   F0（纯 MSE）也走按日整批作对照（否则批构造混淆）。**F3 同样按日整批。**
2. **ListMLE 的 Kendall 权重**：`L = Σ_t exp(−s_t)·L_t + s_t`，`s_t` 可学习（同 `fterm/model/losses.py::MultitaskLoss` 的 log_var 机制）。
   初始化 `s = 0`（两任务等权起步）。

**归一化（吸取 #26 教训）**：F3 的 MSE 与 ListMLE 量级不同 ⇒ 用与 #26 相同的 **init 比对齐**
（未训练模型在训练段子样本上算 `k = L_mse(init)/L_listmle(init)`，登记进报告）。
**不再用"解析常数"**（#26 已证不成立）。

---

## §三 判据（**冻结**；全部建立在 #24 R4 的有功效口径上，**不碰**组合层收益、**不碰**尾部指标）

### 3.1 判据的设计原则

#26 的教训：尾部指标（hit 率/极端性）度量**暴露**不度量**能力**；
#24 的教训：组合层收益在本项目规模下**无功效**。
⇒ 本链**只用两个有功效的量**：**全域逐日 RankIC**（t≈14 量级）与 **宽 k 篮子超额**（k=50/100/450）。

### 3.2 冻结判据

| 编号 | 内容 | 通过条件 |
|---|---|---|
| **C1 方向保住** | F3 − F2（E2 单臂）的全域逐日 IC 配对差 | **NW-t ≥ −2**（即 F3 不比 E2 显著差）**且** 全样本 + late 两 regime 均满足 |
| **C2 极端性拿回** | F3 − F2 的宽 k 篮子超额配对差（k=50/100/450）| **至少 2 个 k** 的配对 NW-t ≥ 2 且为正 |
| **C3 不劣于 MSE** | F3 − F0（MSE 单臂）的全域 IC 配对差 | NW-t ≥ −2（F3 的 IC 不比 MSE 差）|
| **C4 功效声明** | 每个判据的效应/MDE 必报 | 若 C2 的某 k 效应 < MDE ⇒ 该 k 判"功效不足"，不得裁决 |
| **C5 合并裁决** | C1 ∧ C2 ∧ C3 | 全过 ⇒ 融合**有价值**；任一不过 ⇒ 判负，不采纳 |
| **C6 regime** | 全样本 + late（2021-01-01 起）各报一遍 | 结论不得只报一段 |
| **C7 组合层** | 忠实账本 w2/t3 × none/rolling | **仅登记，不作采纳依据**（#24 + 用户"47 年不现实"）|

**多重比较声明**：C1/C3 各 1 检验、C2 是 3 个 k ⇒ 主检验数 = 2 + 3 = **5** ⇒ Bonferroni
双侧临界 |t| ≥ **2.58**（比 #26 的 2.24 更严）。**C1/C3 用 −2.58（单边不劣）**、**C2 用 +2.58（单边优）**。

---

## §四 仪器门（**必须能失败**；吸取 #26 全部教训）

| 门 | 内容 | 失败处置 |
|---|---|---|
| G1 面板竞态 | 右端点 == 2026-09-30；worktree 内跑 | 拒绝出数 |
| G2 流右端点 | 流末距面板末 < 15 天 | 拒绝出数 |
| G3 行/标签位连 | 三臂 `(date, product, y_ret)` 对今日面板 `fold_frame` 重算逐位一致 | 判废 |
| **G4 确定性（两层）** | 同折同种子背靠背：① 计算层 max\|Δpred\| ≤ 2.79e-3；② 选择层同 epoch。**按日整批 + Kendall 权重是新路径 ⇒ 三臂各跑一次 `--verify-det`** | 判废 |
| **G4b 损失确定性/归一化** | 三臂损失各自：同 batch 同初始权重 ⇒ loss+梯度 `torch.equal`；且 `k·L_spec(init) == L_mse(init)` 相对差 <1e-6 | 判废（单变量/归一化不成立）|
| G5 阴性对照 | 每主臂一次日内置换标签（fold3）：\|全域 IC\| ≤ 0.025 且命中 lift ≤ 1.50 | 未塌 ⇒ 泄漏 ⇒ 全链判废 |
| G6 账本闭合 | `blocked_cash == 0`（C7 登记项） | 判废 |
| G7 环境谱系 | `.venv312`（py3.12/pandas 3.0.6/torch cu126）+ 面板 sha | 登记 |
| G8 谱系登记（非门） | vs 旧缓存流偏差（#18 erratum §四 的结构性不可达） | 只留档 |
| G9 单变量自证 | 三臂的折边界/种子/val 划分/fit_bins/网络构造逐项相同（`_assert_single_variable` 可执行）| 判废 |
| G10 **单变量附加（按日整批）** | F0/F2/F3 都按日整批 ⇒ 批构造不混淆；F0 的按日整批 MSE 对照**单独登记**（不与 #26 的随机混洗 C0 比）| 不同 ⇒ 判废 |

**⚠ 与 #26 的 C0 不通用**：#26 的 C0 是**随机混洗**批；本链 F0 是**按日整批** MSE。
两者批构造不同 ⇒ 本链的 F0 必须**自己重跑**，不得复用 #26 的 C0 流。这是 G10 的承重约束。

### §四-bis 标定与仪器的**实测结果**（2026-10-11，**先于整环跑数**）

**① 立案前标定**（`_calib_e2_fusion_preactivation.py`，零 GPU）：见本文件开头的承重表
⇒ E2 的 IC 增益**显著**（配对 t=+5.92 full / +4.33 late），**动机成立**。

**② ⚠⭐ 关键量级事实（G4b 抓出，若不对齐会毁掉单变量性）**：

| 量 | 未训练模型实测 |
|---|---|
| `L_mse(init)` | 0.0906 |
| `L_listmle(init)` | **2698.67** |
| 比值 | **≈ 29,797**（真实训练数据上 ≈ **109,446**） |

⇒ **ListMLE 损失比 MSE 大 ~3–11 万倍**。若**不对齐**，ListMLE 会**完全主导梯度**、
MSE 侧等于没学 ⇒ **F3 退化成纯 ListMLE（F3 ≈ F2）**，单变量对照失效。
⇒ **冻结的归一化**：`scale_k` 乘在 **MSE 侧**，`scale_k = L_listmle(init)/L_mse(init)`
（只用未训练模型在训练段子样本上算，不含结果信息），使 `scale_k·L_mse(init) == L_listmle(init)`。
实测 `scale_k`：G4b 玩具数据 2.98e4 / 真实 fold3 数据 **1.09e5**（逐折逐种子登记）。

**③ G4b（损失确定性 + Kendall 可学习性）**：三臂损失各自 loss 与**全部梯度** `torch.equal`
逐位一致（`max|Δgrad| = 0.00e+00`）；`∂loss/∂log_var = [-1348.8, -1348.8]`
**两项都非零** ⇒ 两个任务都真的被学到（不是单一任务主导）✓

**④ G4（整环确定性，三臂各自背靠背）**：

| 臂 | max\|Δpred\| | 选择层（epoch） | 判定 |
|---|---|---|---|
| F0 | **0.000e+00** | ep8 vs ep8 ✓ | ✓ |
| F2 | **0.000e+00** | ep9 vs ep9 ✓ | ✓ |
| F3 | **0.000e+00** | ep8 vs ep8 ✓ | ✓ |

**⑤ F3 的 Kendall 权重确实在学习**（G4 冒烟实测）：`log_var` 从初值 `[0, 0]`
学到 **`[0.339, 1.642]`** ⇒ 模型把权重偏向了 rank 项（`exp(-0.339) vs exp(-1.642)`）。

**⑥ 冒烟已清理**：单折单种子产物（`stream_F3_ss7.parquet` / `run_F3_*.json`）**已删除**，
防止被误当作臂读数（#26 的 E8 门教训）。

---

## §五 损失实现（要点）

- `L_mse = mean((out − y)²)`
- `L_listmle = listmle_loss(out, y, day_ids)`（`fterm/model/losses.py`，#26 已验证）
- `L_fusion = exp(−s)·(k·L_mse) + exp(−s')·L_listmle + s + s'` —— Kendall 可学习 `s, s'`；
  `k` 是 init 比常数（登记）；`s, s'` 是**标量可学习参数**（非逐任务，简化）。
- 按日整批：`day_batches(day_ids, batch_days=2, shuffle=True)`（同 #26 臂 D/E）。

---

## §六 禁止事项

1. 不得因本链结果改 `config.yaml` 任何默认值；
2. 不得直接换生产工件 —— 采纳须另立前瞻预注册，且 paper60b 窗内禁止；
3. 不得只报全样本、不报 late；
4. 不得用 IC 当"模型准确率"的代理去优化 1–2 只组合（#25/#26 已证方向反/无解）；
5. 不得用尾部指标（hit 率/极端性）作判据（#26 已立碑）；
6. 不得用组合层收益差异作判据（#24 已立碑）；
7. 不得在本链结果出来后调整臂定义/τ/判据。

---

## §七 算力预算

- 3 臂 × 4 折 × 5 种子 + 3 个阴性对照（单折）+ 3 个 `--verify-det`。
- 参照：#26 同结构实测 ~2.2 h/臂（含按日整批可能略慢）；Kendall 权重几乎不增耗时。
- 预算 **≈ 7–8 h** GPU。实测逐臂登记。

---

## §八 复现

```powershell
cd C:\Users\27817\WorkBuddy\Worktrees\Final\master-d5a41331
$env:PYTHONIOENCODING='utf-8'
# 标定（先于预注册）
.\.venv312\Scripts\python.exe experiments\tail_objective_20261010\_calib_e2_fusion_preactivation.py
# 仪器门（G4b 损失/归一化 + 三臂 G4）
.\.venv312\Scripts\python.exe experiments\e2mse_fusion_20261011\_g4b_fusion.py
.\.venv312\Scripts\python.exe scripts\e2mse_fusion_chain.py --arm F0 --verify-det
.\.venv312\Scripts\python.exe scripts\e2mse_fusion_chain.py --arm F2 --verify-det
.\.venv312\Scripts\python.exe scripts\e2mse_fusion_chain.py --arm F3 --verify-det
# 三臂 + 阴性对照
.\.venv312\Scripts\python.exe scripts\e2mse_fusion_chain.py --arm F0
.\.venv312\Scripts\python.exe scripts\e2mse_fusion_chain.py --arm F2
.\.venv312\Scripts\python.exe scripts\e2mse_fusion_chain.py --arm F3
.\.venv312\Scripts\python.exe scripts\e2mse_fusion_chain.py --arm NEG_F2 --folds 3
.\.venv312\Scripts\python.exe scripts\e2mse_fusion_chain.py --arm NEG_F3 --folds 3
# 裁决
.\.venv312\Scripts\python.exe scripts\e2mse_fusion_eval.py
```

---

## §九 判决词（**用户填写**）

| 项 | 判决词 |
|---|---|
| C1 方向保住（F3 IC ≥ E2） | |
| C2 极端性拿回（F3 宽篮子 > E2） | |
| C3 不劣于 MSE | |
| C4 功效声明 | |
| **C5 合并裁决（融合是否有价值）** | |
| C6 regime | |
| C7 组合层（仅登记） | |
