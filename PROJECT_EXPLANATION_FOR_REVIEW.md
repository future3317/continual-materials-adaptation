# FR-PhyTCA 项目解释文档（供外部评审）

> 本文档基于仓库中实际代码、报告与评审意见写成，供 GPT / 审稿人快速定位：数据来源、处理流程、数据表示、模型实现、实验结果、当前问题与下一步。所有 `file:line` 引用均来自仓库当前版本；若结果在仓库中不存在，会明确说明。

---

## 1. 项目概述与核心贡献

**FR-PhyTCA**（Fidelity-Residual Physics-Structured Tensor Component Adaptation）是一个面向“持续演化的材料数据库”的持续学习框架。核心思路写在 `README.md:1`、`FR-PhyTCA_Technical_Documentation.md:9`、`CLAUDE.md:1`：

1. 先在低成本、数据丰富的低保真度（JARVIS 的 OptB88vdW，简称 OPT）上训练一个**父模型**；
2. 训练结束后**永久冻结父路径**（encoder + OPT 适配器 + OPT head）；
3. 每个后续高保真度（TB-mBJ，简称 MBJ）被建模为父模型之上的一个**结构化残差**；
4. 因此父 fidelity 的预测**精确不变**，遗忘严格为零。

当前代码实现了两种形态：

- **旧版对称 PhyTCA 基线**（`phytca.py:197`）：所有任务共享 Tucker 因子，通过 `zero_frozen_gradients` 软冻结切片，实验已证明其不如顺序微调，属于被否定的基线。
- **新版 FR-PhyTCA**（`models.py:109` `ContinualCrystalModel`）：通过**结构隔离**实现精确保留——每个 `(property, fidelity)` 任务拥有独立的 adapter bank 与 head，旧任务通过 `requires_grad=False` 冻结，并且不被加入后续优化器，不使用 gradient hook。

核心贡献可概括为：

- 在 JARVIS Protocol B（OPT → MBJ band gap）上，FR-PhyTCA 实现了 **exact zero forgetting**（`opt_route_drift = 0.00`）；
- 用约 4k 增量参数（旧实现）显著优于参数匹配的 MLP/low-rank 残差基线；
- 提供了从数据审计、周期图构建、持续训练到诊断实验的完整 pipeline。

但当前版本仍存在明显限制：只验证了一个 property（band gap）、两个 fidelity、一个数据库；backbone 是随机初始化的 EGNN；Tucker 多轴共享在当前单-child 设定下实际退化为低秩映射。这些问题在第 8 节详细展开。

---

## 2. 数据来源与预处理

### 2.1 JARVIS 数据源

全部真实数据来自 **JARVIS**：

| 数据集键 | 文件名 | 原始记录数 | 唯一 JID 数 |
|---------|--------|-----------|------------|
| `dft_3d_2021` | `jdft_3d-8-18-2021.json.zip` | 55,723 | 55,712 |
| `dft_3d` | `jdft_3d-12-12-2022.json.zip` | 75,993 | 75,993 |

缓存位置：`data_cache/jarvis/`。

加载函数 `data.load_jarvis_dataset(name, cache_dir)`（`data.py:46`）：
- 优先读取本地 zip 中的 JSON；
- 若本地缓存不存在或损坏，回退到 `jarvis.db.figshare.data` 官方下载器。

### 2.2 数据清洗与目标解析

目标解析器 `data.parse_target(value)`（`data.py:113`）拒绝 `None`、`NaN`、`inf`、字符串 `"na"` / `"n/a"` / `"none"` / `"nan"` / `"inf"` / `"-inf"`、空字符串及不可解析类型。有效目标才会保留。

晶体结构转换 `data.jarvis_record_to_structure(record)`（`data.py:96`）把 JARVIS 的 `atoms` 字典转成 `pymatgen.Structure`，处理笛卡尔/分数坐标，并检查体积大于 `1e-6`。

### 2.3 Protocol A：数据库演化（data-incremental）

构建函数 `data.build_protocol_a()`（`data.py:205`）。任务顺序：

1. A1：JARVIS-2021 formation energy / OptB88vdW
2. A2：JARVIS-2022 formation energy / OptB88vdW（仅 2022 相对 2021 新增的 JID）
3. A3：JARVIS-2021 band gap / OptB88vdW
4. A4：JARVIS-2022 band gap / OptB88vdW（仅新增 JID）

关键处理：
- A2 / A4 仅保留 `added_jids`（`data.py:247-252`），保证 A1↔A2、A3↔A4 的 JID 不相交；
- 同一 `(property, fidelity)` 共享 adapter route、embedding 与 prediction head，仅在最后一次出现后才冻结（`train_phytca.py:52` `_last_occurrences`）；
- 每个任务按 **formula-disjoint** 划分 train/val/test（`data.py:176` `_assign_splits`）。

审计结果 `reports/audit_protocol_a.md`：

| Task | train | val | test | total |
|------|------:|----:|-----:|------:|
| task_a1 | 38,863 | 8,368 | 8,492 | 55,723 |
| task_a2 | 14,267 | 3,026 | 2,988 | 20,281 |
| task_a3 | 38,946 | 8,393 | 8,384 | 55,723 |
| task_a4 | 14,235 | 3,017 | 3,029 | 20,281 |

### 2.4 Protocol B：多保真 band gap

构建函数 `data.build_protocol_b()`（`data.py:319`）。任务顺序：

1. B1：JARVIS-2021 band gap / OptB88vdW
2. B2：JARVIS-2021 band gap / TB-mBJ
3. B3：JARVIS-2022 band gap / OptB88vdW
4. B4：JARVIS-2022 band gap / TB-mBJ

关键处理：
- 仅保留同时具有 `optb88vdw_bandgap` 与 `mbj_bandgap` 的记录（`data.py:347-365` `pair_bandgaps`）；
- 同一结构的 OPT 与 MBJ 记录共享 train/val/test 划分（`data.py:371` `assign_paired_splits`）；
- 不同 fidelity 共享 property embedding 与 prediction head，仅 fidelity embedding 不同（旧实现；新实现中每个任务独立 head）。

审计结果 `reports/audit_protocol_b.md`：

| Task | train | val | test | total |
|------|------:|----:|-----:|------:|
| task_b1 | 12,675 | 2,789 | 2,708 | 18,172 |
| task_b2 | 12,675 | 2,789 | 2,708 | 18,172 |
| task_b3 | 13,871 | 2,917 | 3,017 | 19,805 |
| task_b4 | 13,871 | 2,917 | 3,017 | 19,805 |

### 2.5 数据审计 GO/NO-GO Gate

脚本 `data_audit.py`（配置在 `data_audit.py:42` `AUDIT_CONFIG`）检查：
- 每任务样本数 ≥ 1,000；
- 所有目标值有限；
- Protocol A：数据增量快照 JID 不相交；
- Protocol B：OPT/MBJ 配对记录共享划分；
- 周期图构建器在随机样本上成功运行。

当前两个协议均通过 Gate（`reports/audit_protocol_a.md:4`、`reports/audit_protocol_b.md:4`）。

---

## 3. 数据表示

### 3.1 超胞周期图（默认路径）

类 `data.PeriodicGraphBuilder`（`data.py:521`）默认将原胞扩展为 `2×2×2` 超胞：
- 使用 `pymatgen.Structure * supercell_matrix` 生成超胞（`data.py:625`）；
- 通过逆晶格矩阵将超胞原子坐标映射回原胞原子索引和整数格点偏移（`data.py:633-642`）；
- 返回 `node_feats`（元素 one-hot，dim=92）、`coords`、
  `original_mask`（仅标记原胞原子）、`image_offsets`、
  `original_indices`。

池化时只使用 `original_mask` 对应的原子。`JARVISCrystalDataset`（`data.py:666`）按 split 过滤并做 z-score 归一化；`collate_crystals`（`data.py:721`）将变长样本 pad 成 dense batch。

### 3.2 显式周期边（稀疏图）

`data.PeriodicEdgeGraphBuilder`（`data.py:566`）与 `periodic_graph.build_periodic_edge_graph`（`periodic_graph.py:86`）提供另一种表示：
- 节点只保存原胞原子；
- 边通过 `Structure.get_neighbor_list(r=cutoff)` 按实空间截断构建；
- 每条边保存整数晶格偏移 `edge_shifts`，相对位移为 `r_ij = x_j + L @ n_ij - x_i`（`periodic_graph.py:44-55`）；
- 可选 `max_neighbors` 对每原子最近邻做上限裁剪（`periodic_graph.py:58-83`）。

`periodic_graph.collate_periodic_graphs`（`periodic_graph.py:228`）将稀疏图 batch 化，并同时生成 dense padded 张量 `dense_node_feats` / `dense_coords` / `dense_mask` / `dense_original_mask` 以兼容旧模型接口。

### 3.3 dense vs sparse

- **默认训练路径**使用 dense padded 超胞张量 + `egnn_pytorch.EGNN`（kNN 图）。这种方式**没有显式周期边**，EGNN 在超胞内部用 kNN 近似周期邻居，可能重复选择同一原子的多个镜像或漏掉跨边界邻居（`反馈_2.md` 6.4 指出）。
- **稀疏路径**是为主干替换（MatGL / ALIGNN）准备的，`periodic_graph.to_dense_tensors`（`periodic_graph.py:172`）明确警告：dense 回退会丢弃 `edge_shifts`，不能正确编码周期边界条件。

---

## 4. 模型架构

### 4.1 输入特征

- 节点特征：元素 one-hot，维度 92（`node_dim=92`）；
- 坐标：3D 笛卡尔坐标；
- Mask：`mask`（padding）、`original_mask`（仅原胞原子）；
- 任务索引：`prop_id`、`fid_id`（推理时已知目标 property / fidelity，但不知道数据库版本或任务 ID）。

### 4.2 ContinualCrystalModel

核心类 `models.ContinualCrystalModel`（`models.py:109`）。初始化参数见 `models.py:128-140`：

```python
ContinualCrystalModel(
    node_dim=92,
    hidden_dim=64,
    n_properties=...,
    n_fidelities=...,
    adapter_name="single_child_tucker",
    adapter_rank=8,
    n_layers=3,
    num_nearest_neighbors=8,
    update_coors=False,   # 默认 False，依据 反馈_2.md 6.4
    encoder=None,         # 可传入 MatGLBackbone / ALIGNNBackbone
)
```

生命周期方法：
- `add_task(prop_id, fid_id)`（`models.py:175`）：若该 `(p,f)` 已存在则复用（data-incremental），否则新建 adapter bank + head；
- `freeze_task(prop_id, fid_id)`（`models.py:198`）：把该任务的 head 与 adapter bank 全部 `requires_grad=False`；
- `current_trainable_parameters()`（`models.py:213`）：只返回 `requires_grad=True` 的参数，因此旧任务参数天然不会被后续优化器更新。

### 4.3 Exact retention via structural isolation

新版不再使用 gradient zeroing，而是**物理隔离**：

- 父任务训练完成后调用 `freeze_task`；
- 子任务 `add_task` 会创建**新的** adapter bank 与 head；
- `train_task`（`train_phytca.py:89`）构造 `AdamW(model.current_trainable_parameters(), ...)`，优化器只包含当前任务参数；
- 因此父路径的任何参数都不会被 momentum / weight decay 触碰，实现 exact zero drift。

`diagnostics.d6_progressive_tucker`（`diagnostics.py:1407`）在 Task 2 前后显式 snapshot OPT 预测并计算 `opt_route_drift = max|y_after - y_before|`（`diagnostics.py:1376` `_snapshot_opt_predictions`），报告中该值恒为 `0.00`。

### 4.4 Adapter 类型

所有 adapter 实现统一接口 `ResidualAdapter`（`adapters.py:34`），通过 `make_adapter_bank`（`adapters.py:282`）按 `ADAPTER_REGISTRY` 创建。

| Adapter | 类 | 前向形式 | 备注 |
|---------|-----|----------|------|
| LoRA-AB | `LoRAABAdapter` `adapters.py:54` | `x @ U_in @ U_out^T` | 最简低秩残差 |
| LoRA-ABA | `LoRAABAAdapter` `adapters.py:80` | `x @ U_in @ M^T @ U_out^T` | 可训练中间矩阵，与单-child Tucker 等价 |
| Single-child Tucker | `SingleChildTuckerAdapter` `adapters.py:112` | 继承 `LoRAABAAdapter` | 当前代码有意将其作为 LoRA-ABA 的语义别名，因为单 fidelity 时 4D Tucker 的 property/fidelity 轴无共享 |
| Multi-axis Tucker | `MultiAxisTuckerAdapter` `adapters.py:129` | `U_out @ (G ×₃ e_p ×₄ e_f) @ U_in^T` | 完整 4D Tucker，带 property/fidelity embedding；仅在 `n_properties≥2` 或 `n_fidelities≥3` 时才有跨任务共享 |
| Bottleneck | `BottleneckAdapter` `adapters.py:228` | 两层 MLP | 基线 |
| Zero | `ZeroAdapter` `adapters.py:257` | 返回零 | 父路径占位 |

**注意**：`SingleChildTuckerAdapter` 当前是 `LoRAABAAdapter` 的别名，这意味着默认 FR-PhyTCA 的增量参数为：每层 `64·8 + 8·8 + 64·8 = 1088`，3 层共 3264，加上 head `64+1 = 65`，总计 **3329**。但 `reports/phase0_b_screening/diagnostic_experiments.json` 与 `reports/phase2_b_scaling/scaling_aggregate.json` 仍显示 **3923**，对应早期实现中完整 4D Tucker child（含 `rank_prop=2, rank_fid=2` 的 embedding）。若用当前 `adapters.py` 复跑，增量参数数字会变为 3329。

### 4.5 PredictionResidualHead

`models.PredictionResidualHead`（`models.py:318`）用于显式“父预测 + 物理残差”基线：
- 将父预测从父归一化空间反归一化到物理单位；
- 通过 MLP 学习物理残差；
- 再归一化到子（MBJ）空间输出。

这避免了 `反馈_2.md` 5.1 指出的跨 fidelity 归一化错误：不能直接把 `y_L^norm` 与 `δ^norm` 相加。

### 4.6 旧版对称 PhyTCA 基线

`phytca.py` 保留用于对比：
- `Tucker4DAdapter`（`phytca.py:21`）实现完整 4D Tucker；
- `PhyTCAModel`（`phytca.py:197`）使用共享 Tucker 因子；
- 通过 `freeze_slice` + `zero_frozen_gradients`（`phytca.py:129`）软冻结切片；
- 提供 `stability_loss(mu, anchor)`（`phytca.py:311`）。

Phase 0 实验证明该对称设计失败（见第 6 节），已被降级为基线。

---

## 5. 代码实现细节

### 5.1 关键类与函数索引

| 模块 | 类/函数 | 行号 | 作用 |
|------|---------|------|------|
| `data.py` | `load_jarvis_dataset` | 46 | 加载 JARVIS zip / 官方下载 |
| `data.py` | `parse_target` | 113 | 目标解析与清洗 |
| `data.py` | `jarvis_record_to_structure` | 96 | JARVIS → pymatgen Structure |
| `data.py` | `build_protocol_a` | 205 | Protocol A 构建 |
| `data.py` | `build_protocol_b` | 319 | Protocol B 构建 |
| `data.py` | `_assign_splits` | 176 | formula-disjoint 划分 |
| `data.py` | `assign_paired_splits` | 371 | Protocol B 配对共享划分 |
| `data.py` | `PeriodicGraphBuilder` | 521 | 超胞周期图构建器 |
| `data.py` | `build_periodic_graph` | 609 | 超胞展开与张量生成 |
| `data.py` | `JARVISCrystalDataset` | 666 | PyTorch Dataset |
| `data.py` | `collate_crystals` | 721 | dense pad 后的 batch 拼接 |
| `periodic_graph.py` | `build_periodic_edge_graph` | 86 | 显式周期边稀疏图 |
| `periodic_graph.py` | `collate_periodic_graphs` | 228 | 稀疏图 batch 化 |
| `models.py` | `CrystalEncoder` | 31 | 冻结 EGNN 编码器 |
| `models.py` | `ContinualCrystalModel` | 109 | 主模型 |
| `models.py` | `add_task` | 175 | 新增任务 bank/head |
| `models.py` | `freeze_task` | 198 | 冻结旧任务 |
| `models.py` | `encode` | 225 | 编码 + 原胞池化 |
| `models.py` | `forward` | 242 | 输出子任务归一化预测 |
| `models.py` | `count_task_parameters` | 268 | 单任务参数量 |
| `models.py` | `PredictionResidualHead` | 318 | 物理单位残差头 |
| `adapters.py` | `ResidualAdapter` | 34 | adapter 抽象基类 |
| `adapters.py` | `LoRAABAdapter` | 54 | LoRA-AB |
| `adapters.py` | `LoRAABAAdapter` | 80 | LoRA-ABA |
| `adapters.py` | `SingleChildTuckerAdapter` | 112 | 单-child Tucker（当前=LoRA-ABA） |
| `adapters.py` | `MultiAxisTuckerAdapter` | 129 | 完整多轴 Tucker |
| `adapters.py` | `make_adapter_bank` | 282 | 按名称创建 adapter bank |
| `train_phytca.py` | `_name_to_id` | 40 | property/fidelity 名 → ID |
| `train_phytca.py` | `_last_occurrences` | 52 | 决定何时冻结 |
| `train_phytca.py` | `evaluate_loader` | 61 | nMAE 评估 |
| `train_phytca.py` | `train_task` | 89 | 单任务训练 + 早停 |
| `train_phytca.py` | `continual_experiment` | 162 | 持续训练主循环 |
| `diagnostics.py` | `train_opt_parent` | 281 | 训练共享 OPT parent |
| `diagnostics.py` | `d1_full_joint` | 435 | D1 联合训练上界 |
| `diagnostics.py` | `d2_joint_phytca` | 503 | D2 联合 PhyTCA |
| `diagnostics.py` | `d3_sequential_phytca` | 569 | D3 顺序 PhyTCA |
| `diagnostics.py` | `FrozenOptCorrectionModel` | 664 | D4/D5 冻结父路径修正模型 |
| `diagnostics.py` | `d6_progressive_tucker` | 1407 | D6 FR-PhyTCA |
| `diagnostics.py` | `_snapshot_opt_predictions` | 1376 | OPT 漂移 snapshot |
| `diagnostics.py` | `_fr_phytca_incremental_params` | 1242 | 增量参数计算公式 |
| `baselines.py` | `BASELINE_REGISTRY` | 969 | 基线方法注册表 |

### 5.2 Forward 流程

以 `ContinualCrystalModel` 为例：

1. `node_embed` 将 `(B, N, 92)` 映射到 `(B, N, hidden_dim)`（`models.py:90`）；
2. 对每层 EGNN：`h, coords = layer(h, coords, mask=mask)`（`models.py:92`）；
3. 若传入当前任务的 `adapter_bank`，则 `h = h + adapter(h)`（`models.py:95`）；
4. `encode` 用 `original_mask` 做 mean pooling 得到 `(B, hidden_dim)`（`models.py:238-240`）；
5. `forward` 用对应 `(p,f)` 的 head 输出 `(B,)` 的归一化预测（`models.py:257-259`）。

注意：EGNN 的 `update_coors` 默认设为 `False`（`models.py:138`、`train_phytca.py:176`），因为 dense 路径下的坐标更新会破坏周期一致性（`反馈_2.md` 6.4）。

### 5.3 参数统计

`ContinualCrystalModel.count_task_parameters`（`models.py:268`）统计“head + 该任务 adapter bank”。当前默认配置下：

```
per_layer = d_in*r + r*r + d_out*r   # LoRA-ABA / single_child_tucker
          = 64*8 + 8*8 + 64*8 = 1088
adapter_params = 3 * 1088 = 3264
head_params    = 64 + 1 = 65
incremental    = 3329
```

完整编码器（CrystalEncoder）约 285k 参数，冻结且不随任务增长。

`diagnostics.py` 中旧实现给出的 3923 对应完整 4D Tucker child：

```
per_layer = d_in*r + d_out*r + r²*R_p*R_f + N_p*R_p + N_f*R_f
          = 512 + 512 + 64*2*2 + 1*2 + 2*2 = 1286
3 layers + head = 3*1286 + 65 = 3923
```

---

## 6. 实验

### 6.1 Phase 0：Protocol B 单种子筛选（2k train / 500 val / 1000 test）

**注意**：仓库中存在两份 Phase 0 筛选痕迹：
- `reports/phase0_b_screening/report_round1.md` 是一份较早报告，显示 GO；
- `reports/phase0_b_screening/screening_results.json` 与 `README.md:143` 是**修正后**的结果，显示 `NO_GO_PHYTCA_NO_ADVANTAGE_AT_2K`。以下表格取自修正后的 JSON。

| Method | T1@T1 | T1@T2 | T2 final | abs forgetting | BWT | avg final nMAE | trainable params | stored params |
|---|---|---|---|---|---|---|---|---|
| Sequential FT | 0.791 | 1.421 | 0.885 | 0.630 | -0.630 | 1.153 | 285,283 | 285,283 |
| + replay (1%) | 0.782 | 2.873 | 0.848 | 2.091 | -2.091 | 1.860 | 3,988 | 285,283 |
| Shared LoRA bank | 0.812 | 2.197 | 0.818 | 1.385 | -1.385 | 1.507 | 5,132 | 286,427 |
| PhyTCA (μ=0) | 0.782 | 2.940 | 0.794 | 2.159 | -2.159 | 1.867 | 3,858 | 285,283 |
| PhyTCA (μ=0.01) | 0.782 | 2.584 | 0.798 | 1.803 | -1.803 | 1.691 | 3,858 | 285,283 |
| Joint (upper bound) | 0.356 | 0.356 | 0.261 | 0.000 | 0.000 | 0.309 | 285,283 | 285,283 |

结论：对称 PhyTCA 的遗忘甚至大于顺序微调，因此进入 FR-PhyTCA 重新设计。

### 6.2 D1–D6 诊断实验（2k / seed 42，continual_dev）

结果来自 `reports/phase0_b_screening/diagnostic_experiments.json`。D4–D6 共享同一个 `train_opt_parent` 训练出的 OPT parent，确保 T1@T1 与预测 hash 完全一致。

| Experiment | T1@T1 | T1@T2 | T2 final | abs forgetting | avg final nMAE | raw MAE (eV) | incr. params |
|---|---|---|---|---|---|---|---|
| D1 Full joint | 0.339 | 0.339 | 0.248 | 0.000 | 0.294 | 0.441 | 0 |
| D2 Joint PhyTCA | 0.471 | 0.471 | 0.357 | 0.000 | 0.414 | 0.621 | 0 |
| D3 Sequential PhyTCA | 1.406 | 2.511 | 2.106 | 1.105 | 2.309 | 1.818 | 165 |
| D4 Frozen OPT + affine MBJ | 0.881 | 0.881 | 0.835 | 0.000 | 0.858 | 1.288 | 8,450 |
| D5 Frozen OPT + residual MBJ | 0.881 | 0.881 | 0.838 | 0.000 | 0.859 | 1.290 | 4,225 |
| D6 FR-PhyTCA | 0.881 | 0.881 | 0.446 | 0.000 | 0.664 | 0.996 | 3,923 |

诊断结论（`diagnostics.py` 输出 / `README.md:158-174`）：
- `DIAGNOSIS_ADAPTER_ON_RANDOM_BACKBONE`：编码器随机初始化；
- `DIAGNOSIS_SEQUENTIAL_OPTIMIZATION_FAILURE`：D3 在 Task 1 就已经失败，不完全是 Task 2 干扰；
- `GO_TO_FIDELITY_RESIDUAL_PHYTCA`：冻结父路径 + Tucker 残差有效。

### 6.3 D6 消融

同样来自 `reports/phase0_b_screening/diagnostic_experiments.json`：

| Experiment | T1@T1 | T1@T2 | T2 final | avg final nMAE | raw MAE (eV) | incr. params | opt_route_drift |
|---|---|---|---|---|---|---|---|
| D6-a FR-PhyTCA, no distill | 0.881 | 0.881 | 0.446 | 0.664 | 0.996 | 3,923 | 0.00 |
| D6-b + distill λ=1.0 | 0.881 | 0.881 | 0.446 | 0.664 | 0.996 | 3,923 | 0.00 |
| D6-c Independent low-rank | 0.881 | 0.881 | 0.905 | 0.893 | 1.341 | 520 | 0.00 |
| D6-d Parameter-matched MLP | 0.881 | 0.881 | 0.908 | 0.895 | 1.343 | 1,057 | 0.00 |
| D6-e Orthogonal Tucker | 0.881 | 0.881 | 0.446 | 0.664 | 0.996 | 3,923 | 0.00 |
| D6-f Shared factors + top layer | 0.881 | 1.180 | 0.342 | 0.761 | 1.142 | 92,632 | 2.72 |

蒸馏 λ∈{0,0.1,1.0,10.0} 结果不变，说明父路径确实冻结；D6-f 因更新顶层 EGNN 导致 OPT 漂移。

### 6.4 2k × 3-seed 复现

来自 `reports/phase0_b_screening/repro_2k3seed_summary.json`（seeds 42, 43, 44）：

| Experiment | T1@T1 | T1@T2 | T2 final | forgetting | avg final nMAE | raw MAE (eV) | incr. params |
|---|---|---|---|---|---|---|---|
| Full joint | 0.337 ± 0.036 | 0.337 ± 0.036 | 0.225 ± 0.027 | 0.000 | 0.281 ± 0.031 | 0.424 ± 0.043 | 0 |
| Joint PhyTCA | 0.481 ± 0.048 | 0.481 ± 0.048 | 0.348 ± 0.032 | 0.000 | 0.415 ± 0.040 | 0.626 ± 0.052 | 0 |
| Sequential PhyTCA | 1.482 ± 0.069 | 1.882 ± 0.517 | 2.039 ± 0.240 | 0.400 ± 0.511 | 1.961 ± 0.269 | 1.527 ± 0.210 | 165 |
| Frozen OPT + affine | 0.824 ± 0.044 | 0.824 ± 0.044 | 0.840 ± 0.032 | 0.000 | 0.832 ± 0.036 | 1.258 ± 0.061 | 8,450 |
| Frozen OPT + residual | 0.824 ± 0.044 | 0.824 ± 0.044 | 0.841 ± 0.034 | 0.000 | 0.832 ± 0.036 | 1.258 ± 0.062 | 4,225 |
| **FR-PhyTCA** | **0.824 ± 0.044** | **0.824 ± 0.044** | **0.382 ± 0.038** | **0.000** | **0.603 ± 0.004** | **0.912 ± 0.009** | **3,923** |
| Orthogonal FR-PhyTCA | 0.824 ± 0.044 | 0.824 ± 0.044 | 0.382 ± 0.038 | 0.000 | 0.603 ± 0.004 | 0.912 ± 0.010 | 3,923 |
| Shared factor + top layer | 0.824 ± 0.044 | 0.911 ± 0.035 | 0.324 ± 0.031 | 0.086 ± 0.047 | 0.617 ± 0.004 | 0.933 ± 0.018 | 92,632 |

父路径不变性在所有种子上严格成立。

### 6.5 Stage 2：5k × 3-seed 扩展验证

来自 `reports/phase2_b_scaling/scaling_aggregate.json`，gate 为 `GO_TO_REALISTIC_FIDELITY_SCALING`（`reports/phase2_b_scaling/scaling_gates.json`）：

| Method | T1@T1 | T1@T2 | T2 final | avg final nMAE | raw MAE (eV) | incr. params |
|---|---|---|---|---|---|---|
| Full joint | 0.293 ± 0.034 | 0.293 ± 0.034 | 0.188 ± 0.009 | 0.241 ± 0.017 | 0.367 ± 0.025 | 0 |
| Joint PhyTCA | 0.471 ± 0.022 | 0.471 ± 0.022 | 0.310 ± 0.024 | 0.390 ± 0.017 | 0.595 ± 0.027 | 0 |
| MBJ-only | 1.056 ± 0.228 | 1.056 ± 0.228 | 0.758 ± 0.037 | 0.907 ± 0.133 | 1.630 ± 0.240 | 3,988 |
| OPT pretrain → MBJ full FT | 0.839 ± 0.030 | 1.793 ± 0.812 | 0.895 ± 0.045 | 1.344 ± 0.404 | 2.048 ± 0.607 | 285,283 |
| Frozen OPT + affine | 0.839 ± 0.030 | 0.839 ± 0.030 | 0.842 ± 0.017 | 0.840 ± 0.022 | 1.281 ± 0.038 | 8,450 |
| Frozen OPT + matched MLP | 0.839 ± 0.030 | 0.839 ± 0.030 | 0.829 ± 0.033 | 0.834 ± 0.028 | 1.271 ± 0.048 | 3,895 |
| Frozen OPT + matched low-rank | 0.839 ± 0.030 | 0.839 ± 0.030 | 0.831 ± 0.017 | 0.835 ± 0.015 | 1.273 ± 0.029 | 3,900 |
| **FR-PhyTCA** | **0.839 ± 0.030** | **0.839 ± 0.030** | **0.342 ± 0.022** | **0.590 ± 0.026** | **0.900 ± 0.040** | **3,923** |
| Orthogonal FR-PhyTCA | 0.839 ± 0.030 | 0.839 ± 0.030 | 0.342 ± 0.021 | 0.590 ± 0.025 | 0.900 ± 0.039 | 3,923 |
| Feature transfer | 0.839 ± 0.030 | 0.839 ± 0.030 | 0.823 ± 0.017 | 0.831 ± 0.013 | 1.267 ± 0.018 | 8,321 |

FR-PhyTCA 的 OPT route drift 在三个种子上均为 0.00，T2 nMAE 比参数匹配的 MLP/low-rank 残差低 10% 以上。

### 6.6 基线方法

`baselines.py` 实现了以下基线（注册表见 `baselines.py:969`）：

- Joint training（上界）
- Independent models（每任务独立模型）
- Sequential fine-tuning（顺序微调）
- Frozen encoder + independent heads
- EWC（对角 Fisher 正则，`baselines.py:406` `EWCLearner`）
- Experience replay（1% buffer）
- Independent LoRA（`baselines.py:610` `LoRALinear`）
- Shared LoRA bank（带任务门控）
- Symmetric PhyTCA（`phytca.py:197`）
- FR-PhyTCA 适配器变体：`fr_lora_ab`、`fr_lora_aba`、`fr_single_child_tucker`、`fr_multi_axis_tucker`（`baselines.py:840-962`）

### 6.7 日志文件

`logs/` 目录下所有 `training_*.log` 文件大小均为 **0 字节**，没有可用训练日志。所有定量结果均来自 `reports/` 下的 JSON/Markdown。

---

## 7. 强 backbone 集成

`backbones.py` 提供了两个可选的强 backbone，可作为 `ContinualCrystalModel` 的 `encoder` 参数传入（`models.py:150`）。

### 7.1 MatGL

`MatGLBackbone`（`backbones.py:44`）：
- 封装 MatGL `M3GNet`；
- 将 one-hot 元素向量通过 `argmax` 转回原子序数，再映射到 `element_types` 索引（`backbones.py:103` `_node_feats_to_node_type`）；
- 将 `periodic_graph.py` 的稀疏图字典转成 PyG `Data`，并传入 `pbc_offshift`（`backbones.py:131` `_graph_dict_to_pyg`）；
- 取 MatGL 最后一个 graph convolution block 的 node feat，投影到 `hidden_dim`（`backbones.py:151` `_run_matgl`）。

`build_matgl_backbone`（`backbones.py:259`）：若 `model_name=None`，则构造一个 tiny random M3GNet 用于快速测试；若传入预训练模型名或路径，则加载真实权重。

### 7.2 ALIGNN

`ALIGNNBackbone`（`backbones.py:400`）：
- 基于纯 PyTorch 的 `ALIGNNAtomWisePure`，不需要 DGL；
- 子类 `ALIGNNAtomWisePureEncoder`（`backbones.py:327`）暴露 `encode` 方法，返回 ALIGNN + GCN 层后的节点特征；
- 直接消费 `periodic_graph.py` 的 `edge_index`、`edge_shifts`、`lattice`；
- 默认冻结 backbone。

### 7.3 当前使用情况

`scripts/run_phase2_b_scaling.py` 与 `train_phytca.py` 默认使用 `CrystalEncoder`（随机初始化 EGNN）。强 backbone 目前只是可插拔模块，**尚未在主要实验结果中使用**。

---

## 8. 当前问题与下一步（基于 `反馈_2.md` 与 `升级计划.md`）

`反馈_2.md` 与 `升级计划.md` 指出了项目当前最关键的问题与升级路线，本节做系统整理。

### 8.1 理论层面的问题

1. **Fidelity-error decomposition 与训练目标不一致**（`反馈_2.md` 2.1）
   - 当前模型学的是 `δ(x) ≈ y_MBJ - ŷ_OPT`，而非物理意义上的 `Δy = y_MBJ - y_OPT`。
   - 因此“即使 residual 完美，高保真误差也不能低于 parent error”的三角不等式解释不成立；应改称 **prediction-residual learning**，并给出残差算子低秩近似的误差界。

2. **参数增长命题与实际参数数不一致**（`反馈_2.md` 2.2）
   - 旧命题给出 `ΔP ≤ L(2dr + r²) + d = 3328`，但实际 3923。
   - 当前 `adapters.py:112` 把 `single_child_tucker` 实现为 LoRA-ABA，确实把增量降到 3329，但仓库报告仍停留在 3923。需要在论文中统一公式或统一实现。

3. **当前 Tucker 实际退化为普通低秩映射**（`反馈_2.md` 2.3）
   - 主要实验只有 1 个 property、1 个新增 fidelity，property/fidelity 两个 Tucker 轴没有可验证的跨任务共享。
   - 若要证明“多轴 Tucker 结构优于普通低秩结构”，需要 `n_properties ≥ 2` 且 `n_fidelities ≥ 3`，并增加架构完全匹配的 LoRA-AB / LoRA-ABA / Tucker 对比。

4. **Parent-route invariance 理论贡献较弱**（`反馈_2.md` 2.4）
   - “冻结旧参数就不遗忘”本质上是结构隔离的直接推论。
   - 更强的理论应分析：固定预算下哪个 parent 最合适、residual 有效秩与误差关系、rank 如何随任务增长、多路径一致性等。

### 8.2 实验层面的问题

5. **D3 更像 sequential optimization failure**（`反馈_1.md` 一）
   - 旧表中 D3 的 `T1@T1` 已经很差，不能证明 Task-2 interference。新版 `diagnostics.py:569` 已通过结构隔离重新实现 D3，结果中已体现这一点。

6. **需要严格配对父 checkpoint**（`反馈_1.md` 二）
   - 当前 `diagnostics.py` 已统一使用 `train_opt_parent`（`diagnostics.py:281`）训练一次 OPT parent，D4–D6 加载同一 bundle，并检查 state_dict hash 与 prediction hash。报告中 D4/D5/D6 的 T1@T1 已完全一致。

7. **蒸馏不应改变结果**（`反馈_1.md` 三）
   - D6 消融显示 λ=0/0.1/1.0/10.0 结果相同，`opt_route_drift=0.00`，与父路径真正冻结一致。

8. **Protocol A 没有覆盖真实数据库演化**（`反馈_2.md` 5.3）
   - 当前 A2/A4 只使用新增 JID，未覆盖同一 JID 的 label 修正、结构重新弛豫、元数据变化等。需要把数据库 diff 细分为：new material、revised structure、revised target、new property、changed fidelity、deleted record。

9. **数据划分可能仍有泄漏**（`反馈_2.md` 5.2）
   - formula-disjoint 不能排除同化学式不同多晶型或近重复结构。
   - 建议建立全局 canonical material group（composition + StructureMatcher/fingerprint cluster），一次性分配 split，所有年份、property、fidelity 继承该 split。

10. **未使用真实多保真比例**（`反馈_2.md` 5.7）
    - 当前 2k/5k 是等大小设置。现实场景应是“大量 OPT + 少量 MBJ”，例如 10k OPT → 2k MBJ、30k OPT → 5k MBJ、full OPT → full MBJ。

### 8.3 代码实现层面的问题

11. **周期图处理不够严谨**（`反馈_2.md` 6.4）
    - `2×2×2` 超胞 + dense kNN EGNN 未显式编码周期边；`update_coors=True` 会破坏周期一致性。
    - 应改用显式周期边（节点 = 原胞原子，边 = (src, dst, lattice_shift)），并优先 `update_coors=False`。
    - `periodic_graph.py` 已经提供该实现，但主训练流程尚未迁移。

12. **dense padding 效率低**（`反馈_2.md` 6.5）
    - 建议迁移到 PyG `Data/Batch` 或 DGL graph batch，避免按最大原子数 padding，并缓存预处理后的 periodic graph。

13. **跨 fidelity 归一化风险**（`反馈_2.md` 5.1）
    - 父预测与子残差必须在同一物理单位下相加。`PredictionResidualHead`（`models.py:318`）已处理该问题，但需要逐行审计 `diagnostics.py` 中所有 correction baseline。

14. **Tucker forward 可优化**（`反馈_2.md` 6.1）
    - 当前 `phytca.py:84` 会物化完整 `d_out × d_in` 的 `delta_w`；应使用 `F.linear` 链式计算，复杂度从 `O(d²r + Nd²)` 降到 `O(Ndr + Nr²)`。当前 `adapters.py:23-31` 的 `_apply_linear_chain` 已在新 adapter 中实现。

### 8.4 下一步：从 FR-PhyTCA 到 Continual Fidelity Graph Adaptation

`升级计划.md` 与 `反馈_2.md` 第 4 节建议将方法升级为 **Continual Fidelity Graph Adaptation (CFGA)**：

- 把 fidelity 建模为有向无环图（DAG），节点是 `(property, fidelity)`，边是 residual correction；
- 新 fidelity 到达时自动选择父节点（`fidelity_graph.py:110` `ParentSelector`）；
- 根据 residual 算子谱尾部自适应分配 rank（`fidelity_graph.py:156` `AdaptiveRankAllocator`）；
- 多路径一致性 loss（`fidelity_graph.py:202` `path_consistency_loss`）；
- 旧节点与旧 edge 永久冻结，保证 exact retention。

`fidelity_graph.py` 已提供骨架类 `FidelityGraph`、`ParentSelector`、`AdaptiveRankAllocator`、`path_consistency_loss`、`FidelityGraphPredictor`，但尚未与训练流程集成。

### 8.5 近期可执行的优先级

根据 `反馈_2.md` P0 审计清单，下一步应：

1. 用当前 `adapters.py` 重新跑 2k×3 seeds，确认增量参数变为 3329 后结果是否仍优于 LoRA-ABA / LoRA-AB / Bottleneck；
2. 加入架构完全匹配的 adapter 基线（同位置、同 budget、同 parent checkpoint、同初始化尺度）；
3. 把主训练流程迁移到稀疏周期图 + `update_coors=False`；
4. 在 MatGL/ALIGNN 预训练 backbone 上验证 FR-PhyTCA；
5. 测试真实不平衡 fidelity 比例（10k/30k OPT → 2k/5k MBJ）；
6. 扩展 Protocol B 到 2021 OPT → 2021 MBJ → 2022 OPT → 2022 MBJ 的完整序列，或引入第三个 fidelity；
7. 统一论文中的参数公式与代码实现；
8. 修正理论表述，从“零遗忘定理”转向“残差算子低秩近似 +  fidelity DAG 路径一致性”。

---

## 9. 结语

FR-PhyTCA 当前已在 JARVIS Protocol B 上形成一条相对完整的证据链：数据审计 → Phase 0 NO-GO → D1–D6 诊断 → 2k×3 seed 复现 → 5k×3 seed 扩展。核心结论——**在严格冻结父路径的前提下，用少量参数学习结构化 fidelity 残差，可实现精确零遗忘并显著优于参数匹配的简单残差基线**——是成立的。

但项目距离可投 ICLR 主会的强度仍有差距：当前只是一个数据库、一个 property、两个 fidelity；backbone 随机初始化；Tucker 多轴共享未得到验证；理论贡献偏薄弱。最有前景的升级方向是 `升级计划.md` 与 `反馈_2.md` 提出的 **Continual Fidelity Graph Adaptation**：把 fidelity 扩展为 DAG，加入自动父节点选择、自适应 rank、多路径一致性，并在强 backbone 与真实不平衡数据上验证。
