# 2026-10-09 工作日志

> **⚠️ 本文是 WorkBuddy AI 会话的原始工作日志**（2026-10-10 由 Lead 自
> `.workbuddy-ai/memory/2026-10-09.md` 收编入库，原文未改，保留会话当时的认识轨迹）。
> **已知勘误一处**（2026-10-10 Lead 复核）：文中称臂 C 是"唯一全指标同向改善"的干预，
> 系沿用审计初稿的取整假象——`stability_F0C.json` 原始值显示滚动负窗占比
> **0.66% → 1.32%（翻倍劣化）**，故臂 C 并非全部指标同向。
> 权威记载见 `docs/audit-model-20261009.md §7.4 erratum` 与 `docs/03 §44`。
> 本文其余读数与结论经复核无误。

## fterm 深度学习信号模型升级（Final worktree: master-d5a41331）

### 背景
用户要求：全面深度审计 Final 的问题 → 上网搜索解决方案（不照搬、靠创新）→ 打造成 A 股计算与股票排序上表现突出的深度学习模型。
计划文件：`C:\Users\27817\.workbuddy-ai\plans\toasty-forging-tesla-MBmJvaSa.md`（已批准）。

### 用户拍板的四项决策
1. 立即换版 + 重置 60 日前瞻窗（**接受破坏项目自身预注册纪律的代价**）。
2. 数据全面放开；**月频源不直接用，从原始数据提炼日频因子**。
3. 主判据 = 信息层 RankIC/ICIR + 时间稳定性。
4. 架构不设限；坚持 DL，允许混合模型；允许装 GPU 版 torch。

### 审计核心结论（写进 docs/audit-model-20261009.md）
- 瓶颈**不是模型容量/损失函数**（§40/§43 已 5 次验证）；出路在信号质量。
- **P0-4 最关键**：`foldframe.py:95-103`/`infer.py:52-54` 对每列做逐日截面 Rank-Gauss，
  横截面常数特征（市场状态/指数收益/广度）秩恒 0.5 → 白化后恒 0 ⇒ **市场状态信息在进网前被消灭**。
  这是「市场状态非秩通道」必须旁路 rank 的机制性证据。
- 老库 `neweco.db` **264 表**含大量日频原始数据（margin 6.8M/hk_hold 5.6M/lhb/block_trade/
  insider/forecast/fina_indicator/kg_edge_event/sw_member/regime/research_report_em）。
- 纠正：`alpha_pool`(93 月)/`kg_features`(102 月)/`text_corpus`(181 月) 均**月频快照**，不可当日频用；
  `analyst/` parquet 无 PIT 时间轴，不可用（改用 `research_report_em`）。

### 已完成
- **Phase 0 环境/数据基线**：
  - `.venv`（py3.13.15/pandas3.0.5/numpy2.4.6/torch2.13.0+cpu/lightgbm4.7.0/pytest9.1.1）
  - `.venv312`（uv py3.12.14/pandas3.0.6/numpy2.5.3/scipy1.18.1/**torch2.14.1+cu126**，RTX4060 8GB 实测可用；
    SAC 未拦截 wheel）—— 谱系与 docs/02 登记一致
  - `data/panel` 12 张（2367 日 × 5467 股）+ `data/index` 6 指数
  - 基线：**213 passed**（183 原有 + 23 stats/cpcv + 7 external）
  - `train_production.py --smoke` GPU 通过（3.2M 样本/2 网/53s）
- **修复既有 bug**：`scripts/train_production.py` smoke 路径局部 `universe` 未随切片更新 →
  `fold_frame` 索引校验失败。该 bug 使 `--smoke` 自加入以来一直不可用（不影响生产全量路径）。
- **Phase 0b 外部日频因子**：新建 `fterm/data/external.py` + `scripts/sync_external.py`，
  从老库日频原始表提炼 **37 个日频因子** → `data/external/*.parquet` + `_meta.json`。
  PIT 铁律：日频表按 trade_date；事件表按 ann_date/report_date **次一交易日**起可用；对齐面板日历。
- **Phase 1 审计报告**：`docs/audit-model-20261009.md`（P0×5 / P1×9 / P2×5 + 根因 + 仪器可信度）。
- **总预注册**：`plans/model-upgrade-prereg-20261009.md`。
- **Phase 3 验证栈（部分）**：新建 `fterm/model/stats.py`（ICIR/Newey-West/regime 分层/Deflated Sharpe/PBO）
  与 `fterm/model/cpcv.py`（组合净化交叉验证）；配套 `tests/test_stats.py`、`tests/test_cpcv.py`。

### 环境事实（本机）
- GPU：RTX 4060 Laptop 8GB，driver 591.74；`nvidia-smi` 可用。
- 老库 `D:/NEW ECO/data/neweco.db` = 13.6GB，只读 URI。
- 仓库已设 `git config gc.auto 0` + `maintenance.auto false`（防自动 gc 误删，用户 2026-09-18 事故教训）。

### 待续
- Phase 3 接线：扩展 `validate.py`、`judgment_chain.info_layer_metrics`；新建 `scripts/cpcv_chain.py`、`stability_chain.py`。
- Phase 2 五臂：A 市场状态非秩通道（FiLM，先做）/ B 多 horizon / C 新数据 / D 架构 / E 损失+混合。
- Phase 4-5：组合层校验（T3/w2）、重训 dlv3、换版、前瞻窗重置。

---

## 追加（同日，臂 A 与实证）

### 新增发现（重要，已写入 docs/audit-model-20261009.md）
- **P0-6（新）**：`net2.rank_gauss`（文档所述 `√2·erfinv(2p−1)`）是**死代码**（仅被 import，无调用）；
  live 训练/推理路径（`foldframe.py:123,168`、`infer.py:56,122`）实际算 `√2·erfinv(p)`（**漏 `2p−1`**）。
  实测 `_erfinv(0.5833)=0.5743` vs 正确 `0.2104`。**非功能性故障**（train/infer 自洽），
  但属文档-实现漂移，且是「常数特征归零」直觉失准之因。**未擅自修改**（改白化=改基线，须走参数基线登记）。
- **P0-4 更正**：常数特征经秩化后**同日截面零方差**（非「恒为 0」）；结论不变（对排序零贡献）。

### 实证
- `scripts/newdata_chain.py` 对 37 个外部日频因子做单变量截面 IC 筛查（`experiments/newdata_20261009/`）：
  **`mkt_*` 与 `regime_*` 的 IC 恰为 +0.0000、max|ρ|=0.00** —— P0-4 的直接实证。
  7 个幸存：`lhb_on`(−0.043,t−29.5) / `lhb_org_net_ratio`(+0.027) / `mrg_rqye_ratio`(−0.025) /
  `bt_amount_ratio`(+0.024) / `unlock_ratio_90d`(−0.023) / `hk_ratio_chg5`(+0.021,t+8.9,ICIR 0.394) /
  `mrg_rzye_chg5`(−0.021,t−8.4)。最强者 `hk_ratio_chg5` ICIR 0.394 ≈ 模型级。
- **臂 A 单折冒烟（folds=1）**：F0 IC +0.0156/ICIR 0.388 → A（FiLM 条件化）IC **+0.0391**/ICIR 0.643，
  配对 t **+8.10**，4 层 regime 分层 IC 全正（min 0.0308），三门全过 → adoption_ready=True。
  ⚠️ 仅单折、训练窗仅 2017-2019，**非正式读数**；四折正式跑进行中。

### 新增文件
`fterm/model/{stats,cpcv,net3}.py`、`fterm/features/regime.py`、`fterm/data/external.py`、
`scripts/{sync_external,regime_chain,stability_chain,newdata_chain}.py`、
`tests/{test_stats,test_cpcv,test_external,test_net3,test_foldframe_z}.py`、
`docs/audit-model-20261009.md`、`plans/model-upgrade-prereg-20261009.md`。
改动：`foldframe.py`（z_cols 旁路）、`infer.py`（Dlv3Artifact + 分派）、`train_production.py`（修 smoke bug）。

### 测试基线
**225 passed**（183 原有 + 42 新增）。

### 待续
- 四折臂 A 正式读数（进行中）。
- 臂 B/C/D/E；Phase 4-5。

---

## 追加二（同日，臂 B/E 基础设施 + Phase 3 接线）

- `fterm/model/losses.py`：listwise 损失族（ListNet/Plackett-Luce、ListMLE）、
  ranknet（与 train_production 同款，供对照）、多任务不确定性加权（Kendall 2018）。
  依据：#18 暴露 pairwise 只用相邻对（同日覆盖 ~0.24%/epoch），listwise 是机制上不同的一类。
- `fterm/features/multihorizon.py` + `fterm/model/heads.py`：多 horizon 标签（h∈{1,3,5,10,20}）、
  融合分、共享主干多头网。依据 #7「IC 随 h 单调升」。
- Phase 3 接线：`scripts/judgment_chain.py::info_layer_metrics` 扩展出
  `icir / rolling_nw_t / frac_negative_windows`（+ 可选 `regime_stratified/regime_min_ic`），
  旧键全部保留（向后兼容）。已实测。
- **测试基线：245 passed**（183 原有 + 62 新增）。
- 环境：`.venv` CPU 跑测试；`.venv312` GPU 跑训练（RTX4060 实测可用）。

### 仍未开始
臂 C（幸存因子入模+增量 IC 检验）、臂 D（截面注意力/FT-Transformer）、
Phase 4-5（组合层校验、dlv3 换版、前瞻窗重置 paper60c）。

---

## 追加三（同日，臂 C 基础设施 + 用户决定）

- **用户 2026-10-09 拍板：P0-6 白化口径先保持现状**（live `√2·erfinv(p)` 不动），
  作为独立变量留待臂 A 之后单独检验。已登记 docs/audit-model-20261009.md P0-6。
- `fterm/features/external_features.py`：幸存外部因子入模（共线 max|ρ|≤0.7 剔除、
  不翻转符号、不插补）+ `load_screen_survivors` 从证据包读回。
- `scripts/external_chain.py`：臂 C 执行器（F0 vs C 增量 IC 配对 t），**已就绪待跑**
  （等臂 A 四折腾出 GPU）。
- **测试基线：250 passed**（183 原有 + 67 新增）。
- 臂 A 四折进行中（F0 fold2/4，预计约 12:35 完成）。

---

## 追加四（同日，臂 D1 截面注意力）

- `fterm/model/net4.py`：`CrossSectionAttentionNet` —— 主干（同 net2）+ 一层截面自注意力，
  用 `day_id` 构造**块对角掩码**（同日可见、跨日屏蔽，PIT 安全）；残差+LayerNorm 使注意力为增量。
  依据 P1-4（现有模型逐股独立打分，无股票间交互）。
- `tests/test_net4.py` 6 项全过，含**关键 PIT 钉**：改动 day1 输入不改变 day0 输出
  （证明跨日屏蔽生效）；单日/单例不 NaN。
- **测试基线：256 passed**（183 原有 + 73 新增）。
- 臂 A 四折：F0 已到 fold3（末折），随后跑 arm A。

### 五臂基础设施已全部就绪
A（net3 FiLM + z_cols 旁路 + regime_chain）/ B（multihorizon + heads + losses）/
C（external_features + external_chain）/ D（net4 注意力）/ E（losses listwise + 多任务）。
待跑：A 四折（进行中）、C（等 GPU）、B/D/E 的执行器。

---

## 追加五（同日，臂 A 四折正式裁决 —— 判负）

证据包 `experiments/regime_20261009/verdict_20261009_1248.*`（4 折/2 种子/GPU/805 万条 OOS，
耗时 102 分钟）。**结果与单折冒烟相反**：

| 臂 | IC | NW-t | ICIR | regime 最小分层 IC |
|---|---|---|---|---|
| F0 无条件 | +0.0447 | +14.11 | **+0.652** | +0.0308 |
| A FiLM 条件化 | +0.0491 | +13.40 | **+0.578** | +0.0319 |

- 逐日配对 t（A−F0）= **+1.03** → 未达预注册门槛 t≥2。
- **ICIR 下降**（0.652→0.578）→ 滚动稳定性门不通过。
- `adoption_ready = False`。

**关键方法论教训**：单折冒烟曾给 ΔIC +0.0235 / 配对 t +8.10 的**假阳性**——训练窗仅 2 年
把基线 IC 压到 0.0156，条件化通道"补"的是本应由更长训练窗提供的信息。四折基线 IC 0.0447
（与文档 dlv2 的 0.0485–0.0579 同量级）后增益缩水到不显著。**再次证明单折不可作裁决依据。**

**F0 四折基线读数**（stability_F0_f4.json）：IC +0.0447 / NW-t +14.11 / ICIR +0.652 /
正比例 0.75 / 滚动负窗 1% / regime 最小分层 IC +0.0308 —— 与项目文档口径一致，复现正确。

**方向未完全关闭但不宜过度期待**：只测了 FiLM + 固定 11 特征 + 2 种子；
条件 LayerNorm/MoE/其他 Z 集/14 种子均未测。按 §43 教训，不应把"再换注入方式"当主要杠杆。

**测试基线 256 passed**。已写入 docs/audit-model-20261009.md §7.3。

---

## 追加六（同日，臂 C 运行 + 臂 B 执行器就绪）

- **臂 C 四折运行中**（`experiments/external_4fold.log`）：特征 25→32，7 个幸存因子
  max|ρ| 全部 ≤0.641（过共线门）。**复用臂 A 的 F0 流缓存**（配置逐字相同，节省 63 分钟），
  只训 C 臂。已到 fold2/4。
- `foldframe.py` 新增 `y_key` 参数（默认 "fwd_ret"，向后兼容，等价性测试仍过）——
  供多 horizon 标签使用。
- `scripts/multihorizon_chain.py`：臂 B 执行器就绪。设计：F0 复用缓存；
  B = MultiHorizonNet(5 头 h∈{1,3,5,10,20}) + MultitaskLoss 不确定性加权；
  样本集以最长 horizon(h=20) eligibility 为准（保证多头标签全非 NaN）；
  判据 = h5 头不劣 + ≥2 个其他头 IC>0 + 融合分 ICIR > F0。
- **测试基线 256 passed**。

### 复用 F0 流的合法性与收益
`regime_chain` 与 `external_chain`/`multihorizon_chain` 的 F0 臂配置逐字相同
（同 25 特征 / 同折界 / 同种子 / MSE），故 F0 流可跨链复用 —— 既省时（每链省 ~63 分钟），
又让配对比较基于**同一份 F0**（更严谨）。

---

## 追加七（同日，臂 C 四折正式裁决 —— 方向正确但未达门槛）

证据包 `experiments/external_20261009/verdict_20261009_1331.*` + `stability_F0C.json`。
耗时 40 分钟（复用 F0 缓存）。

| 读数 | F0(25特征) | C(32特征) | 方向 |
|---|---|---|---|
| IC | +0.0447 | **+0.0514** | ↑ +15% |
| NW-t | +14.11 | +14.48 | ↑ |
| **ICIR（主判据）** | +0.652 | **+0.660** | **↑** |
| IC 正比例 | 0.75 | 0.78 | ↑ |
| regime 最小分层 IC | +0.0308 | **+0.0441** | ↑ |

- 逐日配对 t（C−F0）= **+1.66**（95% CI [−0.001,+0.015] 含 0）→ 未达门槛 t≥2。
- **PBO = 0.514**（70 划分）→ 无可靠选择优势。
- `adoption_ready = False`。

**与臂 A 的关键差异**：臂 C **全部六项读数同向为正**且**未削弱 ICIR**
（臂 A 把 ICIR 从 0.652 降到 0.578）。是本次唯一"全指标同向改善"的干预。

**如实结论**：方向正确、量级可观（IC +15%）但**统计不显著**，PBO 无优势，**不予采纳**。

**待预注册的后续假设**：本臂仅 2 种子；生产位 14 网。集成方差随种子数下降，
配对 t 可能越过 2 —— 但这属"再试直到显著"，**必须跑之前预注册**（固定 14 种子一次性判定，
不得事后挑种子集）。

**累计**：模型层/特征层干预 5 次（历史）+ 臂 A（ICIR 下降）+ 臂 C（t=1.66）＝ 7 次，
无一达预注册门槛。§40/§43「α 时间结构是瓶颈」第 4 次被独立确认。

已写入 docs/audit-model-20261009.md §7.4。测试基线 256 passed。

---

## 追加八（同日，臂 B 首跑事故 —— 全 NaN 预测 + 两个真实缺陷）

臂 B 首跑（`verdict_20261009_1350.*`）报 adoption_ready=True、配对 t=+2.74 —— **作废**。
核查：预测列在流中**非空数 = 0**（797 万行全 NaN）。IC 算出 +0.0578 是因为
`spearman_rank_ic` 对全 NaN 输入把 NaN 排末尾并赋予非恒定秩 → 退化为"行序 vs 收益序"的伪相关。
**五个头 IC 完全相同（+0.0578）是该退化的指纹**。已写 `experiments/multihorizon_20261009/INVALID.md`。

### 根因链（4 项）
1. **`build_multi_y` 列对齐错误（新代码）**：标签面板 `fwd_ret_h*` 由 open×adj 派生（**5471 列**），
   特征参照面板 **5467 列**；直接 `arr[ri,ci]` 索引错位取到 NaN → MSE 变 NaN → 网络发散 → 全 NaN 预测。
   修复：先 `reindex(index=master, columns=codes)`。
   ⚠️ **这是本会话第 3 次同类"面板列集异构"陷阱**（前两次 newdata_chain、external.py 的 up/down_limit）。
   只有 `fold_frame._align_panel_cols` 做了显式对齐 —— **系统性风险**。
2. **`spearman_rank_ic` 静默接受 NaN（既有核心缺陷）**：已修复（先剔除任一侧非有限对），
   新增 `tests/test_spearman_nan.py` 6 项回归钉。复核臂 A/C 流无 NaN，**其读数不受影响**。
3. **样本集口径不完整**：仅按 h=20 过滤不足（个股 t+2 停牌而 t+1/t+21 正常 ⇒ h=1 NaN，实测 1018 行）。
   改为**所有 horizon 交集**。
4. **Kendall 不确定性加权退化**：L→0 时 `0.5·exp(−s)·L+0.5·s` 关于 s **无下界**，train_loss
   单调降到 −12。改用「标签按训练段 std 标准化 + 固定等权」，登记偏离。

### 修复后单折复验（有效）
F0(f1) IC +0.0156/ICIR 0.388 → B(f1) IC +0.0038/ICIR 0.031，配对 t **−2.68**，损失稳定（train~4.2）。
**四折正式跑进行中**。

### 测试基线
**262 passed**（183 原有 + 79 新增）。

### 教训（写入台账）
本项目「面板列集异构」是系统性陷阱；任何跨面板索引都必须先 reindex 对齐。
另：核心指标函数对 NaN 必须显式处理，否则伪读数会静默通过全部闸门。

---

## 追加九（同日，臂 B 四折有效裁决 —— 判负）

证据包 `experiments/multihorizon_20261009/verdict_20261009_1422.*`（流校验 100% 非空）。

| 臂 | IC | NW-t | ICIR |
|---|---|---|---|
| F0 单头h5 | +0.0447 | +14.11 | **+0.652** |
| B h5头 | +0.0400 | +7.45 | **+0.354** |
| B 融合分 | +0.0431 | +7.32 | +0.353 |

- 配对 t（B_h5−F0）= **−1.00**；（融合−F0）= −0.32 → 无改善。
- **ICIR 近乎腰斩**（0.652→0.354）。
- 各头 IC：h1 .0371 / h3 .0404 / h5 .0400 / h10 **.0441** / h20 .0410
  —— 长 horizon 略优（方向与 #7 一致，但幅度远小于 #7 的树臂读数）。
- adoption_ready = False。

**解释**：h=5 信噪比较好；把 5 个 horizon 塞进共享主干等于给表示层加了 4 个噪声更大的目标，
稀释了 h=5 的排序质量。融合分也未能挽回。

---

## 三臂小结（截至今日）

| 臂 | 干预 | ΔIC | ΔICIR | 配对 t | 判定 |
|---|---|---|---|---|---|
| A | 市场状态非秩通道(FiLM) | +0.0045 | **−0.074** | +1.03 | 判负 |
| B | 多 horizon 融合 | −0.0047 | **−0.298** | −1.00 | 判负 |
| C | 外部日频因子入模 | **+0.0067** | **+0.008** | +1.66 | 判负（未达 t≥2） |

**唯一全指标同向改善的是臂 C**（IC +15%、ICIR +1.2%、正比例 +3pp、regime 最小分层 IC +43%），
但未达门槛，PBO 0.514 无优势。

**累计干预 8 次（历史 5 + A + B + C），无一达门槛**。§40/§43「α 时间结构是瓶颈，
模型层/特征层边际干预不是杠杆」**第 5 次被独立确认**。
新增证据：三次干预中两次（A、B）**损害了稳定性主判据** —— 在现有信号上加"结构复杂度"
往往**削弱**而非增强排序稳定性。

### 剩余未跑
臂 D（截面注意力 net4，执行器未写）、臂 E（listwise 损失，执行器未写）。
**Phase 4-5（组合层校验、dlv3 换版、前瞻窗重置）未开始。**

### 测试基线
**262 passed**（183 原有 + 79 新增）。

---

## 追加十（同日，用户拍板：预注册并一次性跑 14 种子版臂 C）

**用户决定**：选"预注册 + 一次性跑 14 种子版臂 C"这条路径。

### 预注册
`plans/armC-14seed-prereg-20261009.md` —— **先于实验冻结判据与种子池**：
- 种子池 = `(7,21,99,5,13,123,2024,42,314,2027,101,202,303,404)` = **14 个**，
  与 `train_production.py::SEED_SETS` **逐字相同**（生产位口径）。
- **关键纪律**：对照 F0 **必须同为 14 种子**（否则"特征"与"集成规模"混淆）
  ⇒ 需同时训练 F0_14 与 C_14。
- 主判据：逐日配对 t（C_14 − F0_14）**≥ 2 且 ΔIC>0**（一次性判定）。
- 辅判据：ICIR(C)>ICIR(F0)；regime 最小分层 IC>0；PBO<0.5。
- 一次性纪律：禁止跑后改种子/判据/门槛；中途读数**仅作诊断**不构成判定。
- **如实登记局限**：本实验不直接对比生产 dlv2 工件（差异在损失 RankNet 与 val 比例）；
  这是臂 C 的第 2 次检验，Bonferroni 下门槛应≈t≥2.24，但沿用项目 t≥2 并在结论中报告该放宽效应。

### 执行
`scripts/armC14_chain.py`（新建）：112 次训练（2 臂 × 4 折 × 14 网），
**逐折增量落盘** `experiments/armC14_20261009/parts/{arm}_f{k}.parquet`，
中断可从断点续跑同一配置。含仪器门（流必须 100% 非空）与 NaN 守卫。
预计 **8–12 小时**（RTX 4060）。已启动。

### 判负处置（预注册 §三）
若主判据不通过 ⇒ **臂 C 判负并结案**，不再做"加更多种子/换种子集"的尝试。

---

## 追加十一（同日，臂 D/E 执行器就绪 —— 只写不跑）

用户指示"做1"：编写臂 D/E 执行器但不运行（避免与 14 种子实验争抢 GPU）。

### 关键设计问题（已处理）
臂 D（截面注意力）与臂 E（listwise）**都要求同一交易日的股票同处一个 batch**，
而 F0 用跨日随机混洗（同批每日仅 1–2 只 ⇒ 两类改动都失去意义）。
但**按日整批本身会改变训练动态**，若直接对比 F0 则"批构造"与"架构/损失"混淆。
⇒ 新增 **`F0day` 对照臂**（同样按日整批 + net2 + MSE），判据须同时给出
「D/E 相对 F0」与「D/E 相对 F0day」两条配对 t。

### 新增文件
- `fterm/model/daybatch.py`：按日整批采样器（每批含整数个完整交易日）。
  ⚠️ 首版用 `np.searchsorted` 有缺陷（要求全局有序，而 day_ids 只按日块连续），
  已改为**块边界检测**（`np.diff` + `flatnonzero`）。`tests/test_daybatch.py` 8 项全过。
- `scripts/arch_chain.py`：统一执行器，臂 `F0day / D(attn) / E(listnet) / E2(listmle)`，
  逐臂逐折增量落盘 `experiments/arch_20261009/parts/`。
  **修正一处推理期泄漏**：臂 D 推理原用 `day_ids=0`（全部同日）⇒ 会跨日互相注意；
  已改为与训练同口径的**按日整批推理**。
- 选模统一用 MSE（与 F0 一致），避免"选模口径"成为额外变量。

### 测试基线
**270 passed**（183 原有 + 87 新增）。

### 14 种子实验进度
F0 fold0 训练中（3/14 种子完成，约 1.7 分钟/种子）。预计总时长 ~8 小时。

---

## 追加十二（同日，进程检查 + 每日链竞态评估）

### 14 种子进程（14:36 检查）
- F0 fold0 **4/14 种子**完成（seed 7/21/99/5），~1.75 分钟/种子。
- GPU：util 39%、显存 **1.87 GB / 8 GB**、温度 50°C —— 健康。
- 进程 PID 738 自 14:29 运行。
- ETA：fold0 ~24 min、fold1 ~49、fold2 ~73、fold3 ~98 ⇒ F0 约 18:34 完成；
  C 臂（32 特征，约 +15%）⇒ 全部约 **23:20** 完成。

### 每日链 `fterm_daily`（20:05）竞态评估 —— **风险低**
关键事实（实读核实）：
1. `scripts/run_daily_task.bat` 执行 **`cd /d D:\Final`** ⇒ 每日链跑在
   **`D:\Final`（独立生产检出）**，**不是**本工作树
   `C:\Users\27817\WorkBuddy\Worktrees\Final\master-d5a41331` ⇒ **文件系统隔离**，
   不会写本工作树的 `data/panel`。
2. `run_daily_auto.py` 只调 `run_daily.py --asof ... --skip-sync`（**决策环，纯推理**），
   **不触发训练**（无 train 调用）。
3. bat 用 `D:\Final\.venv` ⇒ 虚拟环境也不共享。
4. 唯一共享资源 = 同机 **GPU/CPU/RAM**；我占用 GPU 1.9GB/8GB，推理占用很小 ⇒ 余量充足。

⇒ **结论：无需干预**。20:05 每日链与本实验可并行。
（注：docs/02 记载 `fterm_daily` 曾自 10-03 起因 bat 内 CR 字节 nightly 失败，
10-06 已修；本评估基于修复后的 bat。）

---

## 追加十三（同日，14 种子实验监控 —— 10 分钟一轮）

用户要求"每十分钟检查一次进程"。**能力边界**：本环境自动化调度器**最小粒度 1 小时**
（`HOURLY` 的 INTERVAL 以小时计），**无法表达 10 分钟** ⇒ 只能在**会话内**做 10 分钟检查。

### 实测时间线（供后续估算参考）
| 折 | 训练窗 | 耗时 | 产出 |
|---|---|---|---|
| F0 fold0 | 2017-2019（2y） | **935 s（15.6 min）** | 1,688,732 条 |
| F0 fold1 | 2017-2021（4y） | **4437 s（74 min）** | 2,006,044 条 |
| F0 fold2 | 2017-2023（6y） | 进行中 | — |
| F0 fold3 | 2017-2025（8y） | 未开始 | — |

⇒ 耗时**远超训练窗线性**（fold1 是 fold0 的 4.7×，而窗仅 2×）——因数据量增大后早停更晚。
**修正 ETA：F0 约 20:20 完成，C 臂再约 6.7 h ⇒ 约次日 03:00。总计 ~12.5 h。**

### 新发现：GPU 显存逼近上限（可优化项）
- fold1 时显存 4.6 GB；**fold2 跃升至 7.7 GB / 8 GB（96%）**。fold3 可能触顶。
- **根因**：`Xtr_bin` 用 **int64** 存桶索引（取值仅 0–15），fold2 下约 **2 GB 纯属浪费**；
  改 `int8` 可省约 1.8 GB。同 `Xtr_in`(float32) + torch 副本合计约 6–7 GB。
- **本次不动**（预注册已冻结配置，中途改代码破坏"一次性判定"纪律）。**记入台账供后续优化。**
- 风险判断：同样四折在臂 A/C 已跑通（含 fold3）⇒ 大概率可存活（Windows CUDA 可溢出共享内存，
  最坏是变慢而非崩溃），但无法排除 OOM，故每轮紧盯。

---

## 追加十四（同日 18:03，用户报告"内存大量占用" —— 实测确认）

用户观察正确。实测（PowerShell `Win32_OperatingSystem`）：

| 指标 | 读数 |
|---|---|
| 物理内存 | 总 **31.69 GB** / 可用 **3.91 GB** / 已用 27.78 GB（**87.7%**） |
| 我的训练进程（python PID 35616） | **5.44 GB** |
| Memory Compression | **3.18 GB**（Windows 压缩内存 = **压力指示**） |
| WorkBuddyAI | 1.71 GB |
| 页面文件 | 分配 24010 MB，峰值 5473 MB，当前 **3153 MB** |

### 根因（三层叠加）
1. **CUDA 共享内存溢出**：显存用满 7.6/8 GB，Windows 上 CUDA 溢出部分**从系统 RAM 借**
   （"shared GPU memory"）⇒ GPU 显存压力直接转化为系统 RAM 压力。
2. **大 numpy/torch 数组**：fold3（8 年窗）下 `Xtr_in`(float32) + `Xtr_bin`(**int64**，浪费)
   + torch 副本，合计数 GB。
3. **`meta` 列表**：`meta_tr` 是 ~1000 万条 `(Timestamp, code)` 元组的 Python list，
   保守估计 **≥1 GB**（纯 Python 对象开销）。

### 风险评估
- **可用 3.91 GB** —— 尚可运转，但 Memory Compression 已启动，说明已在承压。
- 同样四折此前在臂 A / 臂 C 已成功跑通 ⇒ **大概率可存活**；但若继续下降至 ~1 GB 将开始换页抖动。
- **不建议中途改代码**（预注册已冻结配置，改代码破坏"一次性判定"纪律）。
- 优化项已记录供后续实验：`Xtr_bin` 改 int8（省 ~1.8 GB）、`meta` 改 numpy 结构化数组或
  只在需要时物化（省 ~1 GB）。

---

## 追加十五（同日 23:04，核查：F0_14 流完整性 + 一个重要发现）

### 三项核查全部通过
1. **流完整性**：8,055,261 行；`pred`/`y_ret` **100% 非空**；2019-01-02 → 2026-09-21，1874 交易日。
   （与臂 A/B/C 所用 F0 缓存行数**完全一致**）
2. **增量落盘一致性**：4 个 parts 合计 = stream 行数，且 **pred 逐位相同（True）** ⇒ 续跑机制无错位。
3. **IC 水平**（重要发现）：

| 版本 | IC | NW-t | ICIR |
|---|---|---|---|
| F0（**2 种子**） | +0.0447 | +14.11 | **+0.652** |
| **F0_14（14 种子）** | **+0.0595** | +13.94 | **+0.623** |

**两个含义**：
- **集成规模显著抬高 IC**（+33%）⇒ 用户"加种子"的直觉方向正确。
- **但 ICIR 略降**（−4.5%）⇒ 更多种子 ⇒ IC 均值更高但稳定性略降。

**对臂 C 裁决的直接影响**：臂 C 要打败的对照**已从 2 种子的 0.0447 提高到 14 种子的 0.0595**，
门槛显著提高。2 种子时臂 C 的 +15%（0.0447→0.0514）若同比例外推约 0.068，
能否显著超 0.0595 **尚不可知**。

**谨慎标注**：F0_14 的 0.0595 **略高于文档所载生产 dlv2 的 0.0485–0.0579 上沿**，
但口径不同（生产位含 RankNet、val 0.10）⇒ **不可据此断言复现优于生产工件**。

---

## 追加十六（同日 23:58，臂 C 14 种子最终裁决 —— **判负，假设被证伪**）

证据包 `experiments/armC14_20261009/verdict_20261009_2358.*`
（14 种子 = 生产位种子池逐字相同 / 4 折 / 两臂各 8,055,261 条 OOS / 总耗时 9.5 小时）。

| 臂 | IC | NW-t | ICIR |
|---|---|---|---|
| F0_14（25 特征） | +0.0595 | +13.94 | +0.623 |
| **C_14（32 特征）** | **+0.0629** | **+15.05** | **+0.665** |

- 配对 t（C_14 − F0_14）= **+1.61**（ΔIC +0.0034）→ **未达 t≥2**
- regime 4 层分层 IC 全正（最小 +0.0523）✓
- **PBO = 0.800**（劣于 2 种子的 0.514）
- `adoption_ready = False`

### 关键：**"加种子"假设被证伪**

| | 2 种子 | **14 种子** |
|---|---|---|
| ΔIC 绝对 | +0.0067 | **+0.0034** |
| ΔIC 相对 | **+15.0%** | **+5.7%** |
| **配对 t** | +1.66 | **+1.61** |
| PBO | 0.514 | **0.800** |

1. 配对 t **几乎未变**（1.66→1.61）⇒ 集成规模扩大**未带来显著性**。
2. **相对增益缩水**（+15%→+5.7%）⇒ 2 种子时的 +15% 中**相当部分是集成噪声**。
3. **PBO 恶化**（0.514→0.800）⇒ C 相对 F0 的排序优势**时间上不稳定**。

**正面项**：ICIR 改善（0.623→0.665，+6.7%），regime 各层 IC 全正。

**依预注册 §三结案**：主判据不通过 ⇒ 臂 C 判负并结案，不再加种子/换种子集。
（Bonferroni 门槛≈2.24 的说明在此不影响结论 —— t=1.61 连未校正的 2 都不通过。）

### 九次干预全部判负（历史 5 + A + B + C2 + C14）

**最终判断**：瓶颈确证为 **α 的时间结构**，非模型容量/损失/架构/特征数量。
**不具备换版依据 ⇒ 现有 `dlv2_production.pt` 保持不动。**

### 本次会话净产出（即使全判负仍有实质价值）
1. 建立项目此前**完全缺失**的验证栈（ICIR/CPCV/制度分层/PBO/DSR）。
2. 修正 2 个真实缺陷（`spearman_rank_ic` 静默接受 NaN；smoke 路径 universe 未同步）。
3. 发现并登记 P0-6（文档所述 Rank-Gauss 是死代码）。
4. 从老库提炼 **37 个日频因子**，筛出 7 个幸存者（`hk_ratio_chg5` 单因子 ICIR 0.394）。
5. **证伪"加种子提升显著性"假设**，避免一次基于噪声的换版决策。
6. 测试基线 262 → **270 passed**。
