# FR-PhyTCA 技术说明文档（供架构审核与 ICLR 投稿规划）

> 论文路径：`E:\PAPER\Parameter-Efficient Continual Learning via Structured Tensor Decomposition\example_paper.tex`  
> 代码路径：`E:\CODE\Continual Learning`  
> 文档目的：系统说明数据来源、处理流程、模型实现、实验设计及结果，便于 GPT/Codex 判断哪些模块可用现有库替代、哪些实现可能不正确、哪些架构可优化、哪些 idea 可升级。

---

## 1. 研究背景与核心方法

### 1.1 问题定义

材料数据库（JARVIS、Materials Project）持续演化：
- 数据库版本更新（JARVIS-2021 → JARVIS-2022）；
- 新性质被加入（formation energy、band gap、elastic、dielectric 等）；
- 更高精度的计算方法替代旧估计（OptB88vdW → TB-mBJ）。

本文研究**持续学习（Continual Learning）**在这一场景下的应用，核心挑战：
1. **推理时无任务身份**：部署模型只看到晶体结构，不知道它来自哪个数据库版本或 fidelity；
2. **多轴同时演化**：材料、性质、 fidelity 同时变化；
3. **物理约束**：预测需满足晶体对称性。

### 1.2 FR-PhyTCA 核心思想

**FR-PhyTCA**（Fidelity-Residual Physics-Structured Tensor Component Adaptation）：
- 先在低成本、数据丰富的低保真度（OptB88vdW, OPT）上训练一个**父模型**；
- 训练完成后**永久冻结父路径**（encoder + OPT Tucker slice + OPT head）；
- 每个后续高保真度（TB-mBJ, MBJ）被建模为父模型之上的一个**结构化残差**；
- 因此父 fidelity 的预测**精确不变**，遗忘严格为零。

对称的 **PhyTCA 基线**（共享 Tucker 因子被每个任务更新）作为对比，实验显示其存在严重顺序干扰。

---

## 2. 数据来源与处理

### 2.1 数据来源

全部真实数据来自 **JARVIS**（Joint Automated Repository for Various Integrated Simulations）：

| 数据集键 | 文件名 | 原始记录数 | 唯一 JID 数 |
|---------|--------|-----------|------------|
| `dft_3d_2021` | `jdft_3d-8-18-2021.json.zip` | 55,723 | 55,712 |
| `dft_3d` | `jdft_3d-12-12-2022.json.zip` | 75,993 | 75,993 |

缓存位置：`E:/CODE/Continual Learning/data_cache/jarvis/`。

加载函数：`data.load_jarvis_dataset(name, cache_dir)`（`data.py:44`）。实现：
- 优先读取本地 zip 中的 JSON；
- 若本地缓存不存在或损坏，回退到 `jarvis.db.figshare.data` 官方下载器。

### 2.2 数据清洗与目标解析

目标解析器：`data.parse_target(value)`（`data.py:111`）。

拒绝以下无效值：
- `None`
- 字符串 `"na"`、`"n/a"`、`"none"`、`"nan"`、`"inf"`、空字符串
- `NaN`、`inf`、`-inf`
- 不可解析类型

有效目标才被保留。

### 2.3 晶体结构转换

函数：`data.jarvis_record_to_structure(record)`（`data.py:94`）。

将 JARVIS 的 `atoms` 字典转换为 `pymatgen.Structure`：
- 读取 `lattice_mat` 构造 `Lattice`；
- 读取元素列表和坐标；
- 处理笛卡尔/分数坐标；
- 检查体积大于 `1e-6`。

### 2.4 任务协议

#### Protocol A：数据库演化（data-incremental）

构建函数：`data.build_protocol_a()`（`data.py:203`）。

任务顺序：
1. A1：JARVIS-2021 formation energy / OptB88vdW
2. A2：JARVIS-2022 formation energy / OptB88vdW（仅新增材料）
3. A3：JARVIS-2021 band gap / OptB88vdW
4. A4：JARVIS-2022 band gap / OptB88vdW（仅新增材料）

关键处理：
- A2 仅保留 2022 相对于 2021 新增的 JID；
- A1↔A2 共享同一 (property, fidelity)，因此共享 adapter route、embedding、prediction head；
- 每个任务按 **formula-disjoint** 划分为 train/val/test（`data._assign_splits`，`data.py:174`）。

最终数量（来自 `reports/audit_protocol_a.md`）：

| Task | train | val | test | total |
|------|------:|----:|-----:|------:|
| A1 | 38,863 | 8,368 | 8,492 | 55,723 |
| A2 | 14,267 | 3,026 | 2,988 | 20,281 |
| A3 | 38,946 | 8,393 | 8,384 | 55,723 |
| A4 | 14,235 | 3,017 | 3,029 | 20,281 |

#### Protocol B：多保真 band gap

构建函数：`data.build_protocol_b()`（`data.py:317`）。

任务顺序：
1. B1：JARVIS-2021 band gap / OptB88vdW
2. B2：JARVIS-2021 band gap / TB-mBJ
3. B3：JARVIS-2022 band gap / OptB88vdW
4. B4：JARVIS-2022 band gap / TB-mBJ

关键处理：
- 仅保留同时具有 `optb88vdw_bandgap` 和 `mbj_bandgap` 的结构；
- 同一结构的 OPT 和 MBJ 记录共享 train/val/test 划分（`data.assign_paired_splits`，`data.py:369`）；
- 不同 fidelity 共享 property embedding 和 prediction head，仅 fidelity embedding 不同。

最终数量（来自 `reports/audit_protocol_b.md`）：

| Task | train | val | test | total |
|------|------:|----:|-----:|------:|
| B1 | 12,675 | 2,789 | 2,708 | 18,172 |
| B2 | 12,675 | 2,789 | 2,708 | 18,172 |
| B3 | 13,871 | 2,917 | 3,017 | 19,805 |
| B4 | 13,871 | 2,917 | 3,017 | 19,805 |

### 2.5 周期性图构建

类：`data.PeriodicGraphBuilder`（`data.py:519`）。

实现细节：
- 默认将原胞扩展为 `2×2×2` 超胞（`supercell_matrix` 可配置）；
- 使用 `pymatgen.Structure * supercell_matrix` 生成超胞；
- 通过逆晶格矩阵将超胞原子坐标映射回原胞原子索引和整数格点偏移（`data.py:598-607`）；
- 生成：
  - `node_feats`: 元素 one-hot（dim=92）
  - `coords`: 笛卡尔坐标
  - `original_mask`: 标记原胞原子
  - `image_offsets`: 整数超胞偏移
  - `original_indices`: 每个超胞原子对应的原胞原子索引

池化时只使用 `original_mask` 对应的原子。

### 2.6 数据审计 GO/NO-GO Gate

脚本：`data_audit.py`。

审计标准（`data_audit.AUDIT_CONFIG`，`data_audit.py:42`）：
- 每任务样本数 ≥ 1,000
- 所有目标值有限
- Protocol A：数据增量快照 JID 不相交
- Protocol B：OPT/MBJ 配对记录共享划分
- 周期性图构建器在随机样本上成功运行

运行方式：
```bash
python data_audit.py --protocol a --report-dir reports
python data_audit.py --protocol b --report-dir reports
```

当前两个协议均通过 Gate。

---

## 3. 模型架构与特征

### 3.1 输入特征

- 节点特征：元素 one-hot，维度 92（`node_feature_dim=92`）；
- 坐标特征：3D 笛卡尔坐标；
- Mask：padding mask 和 original_mask；
- 任务索引：property_id 和 fidelity_id（推理时已知目标 property/fidelity，但**不**知道数据库版本/task id）。

### 3.2 核心网络：EGNN + Tucker Adapter

#### 3.2.1 晶体图编码器

使用 `egnn_pytorch.EGNN`（`phytca.py:18`）。

`AdapterCrystalGraphLayer`（`phytca.py:153`）结构：
```python
new_feats, new_coords = self.encoder(feats, coords, mask=mask)
delta = self.adapter(new_feats, prop_id, fid_id)
return new_feats + delta, new_coords
```

配置：
- `dim=hidden_dim`（默认 64）
- `edge_dim=0`
- `m_dim=max(16, dim)`
- `num_nearest_neighbors=8` 或 16
- `update_coors=True`
- `update_feats=True`

#### 3.2.2 Tucker4DAdapter：对称基线

类：`phytca.Tucker4DAdapter`（`phytca.py:21`）。

对 4D 权重更新张量进行 Tucker 分解：

```
A = G ×₁ U_out ×₂ U_in ×₃ E_prop ×₄ E_fid
```

张量形状：
- `G`: `(R_out, R_in, R_p, R_f)`
- `U_out`: `(d_out, R_out)`
- `U_in`: `(d_in, R_in)`
- `E_prop`: `(N_p, R_p)` embedding
- `E_fid`: `(N_f, R_f)` embedding

对于特定 (p, f)，前向传播：
```python
e_p = E_prop(p)  # (R_p,)
e_f = E_fid(f)   # (R_f,)
core_slice = torch.einsum("oipf,p,f->oi", G, e_p, e_f)  # (R_out, R_in)
delta_w = U_out @ core_slice @ U_in.t()  # (d_out, d_in)
return x @ delta_w.t()
```

参数：
- `rank_out=rank_in=adapter_rank`（默认 8）
- `rank_prop=max(2, n_properties)`
- `rank_fid=max(2, n_fidelities)`

#### 3.2.3 PhyTCAModel：对称基线完整模型

类：`phytca.PhyTCAModel`（`phytca.py:197`）。

结构：
1. `node_embed`: Linear(node_dim → hidden_dim)
2. `layers`: `n_layers` 个 `AdapterCrystalGraphLayer`
3. `heads`: 每个 (property, fidelity) 一个 `Linear(hidden_dim → 1)`

默认 `freeze_encoder_weights=True`，即 EGNN 编码器和 node_embed 被冻结，只训练 adapters 和 heads。

关键方法：
- `freeze_task(prop_id, fid_id)`：冻结已完成任务的 adapter slice 和 head（`phytca.py:295`）；
- `stability_loss(mu, anchor)`：L2 锚定损失（`phytca.py:311`）；
- `zero_frozen_gradients()`：零化已冻结 slice 的梯度（`phytca.py:129`）。

### 3.3 FR-PhyTCA：Fidelity-Residual 版本

实现位置：`diagnostics.py`（主要类 `ProgressivePhyTCAModel`、`ProgressiveTuckerAdapter`）。

#### 3.3.1 ProgressiveTuckerAdapter

类：`diagnostics.ProgressiveTuckerAdapter`（`diagnostics.py:1126`）。

结构：
- `parent`: 父 Tucker4DAdapter（训练 OPT 后冻结）；
- `child`: 子 Tucker4DAdapter（随机初始化残差）；
- 前向传播：`parent(x, p, f) + child(x, p, f)`。

关键机制：
- `zero_and_freeze_child_slice(prop_id, fid_id)`：将子 adapter 中父 fidelity（OPT）的 slice 永久置零（`diagnostics.py:1160`）；
- `zero_child_gradients_for_parent()`：每个 backward 后零化子 adapter 中父 fidelity slice 的梯度（`diagnostics.py:1174`）；
- 父 adapter 的参数被 `requires_grad=False` 冻结。

这保证了 OPT 路径完全不受影响。

#### 3.3.2 ProgressivePhyTCAModel

类：`diagnostics.ProgressivePhyTCAModel`（`diagnostics.py:1217`）。

与 PhyTCAModel 的区别：
- 每层使用 `ProgressiveAdapterCrystalGraphLayer`；
- `freeze_parent_task(prop_id, fid_id)`：冻结父 adapter slice 和 head，同时冻结父 adapter 所有参数（`diagnostics.py:1287`）。

#### 3.3.3 父-子状态迁移

函数：`diagnostics._remap_phytca_to_progressive_state()`（`diagnostics.py:1307`）。

将标准 PhyTCAModel 的 state_dict 映射到 ProgressivePhyTCAModel 的 parent adapter：
```python
if k.startswith("layers.") and ".adapter." in k:
    remapped[k.replace(".adapter.", ".adapter.parent.", 1)] = v
```

### 3.4 输出头与物理约束

当前实现的 Protocol A 和 B 都是**标量回归**：
- formation energy per atom；
- band gap。

损失函数：MSE（目标先按训练集 mean/std 归一化）。

论文方法部分提到张量性质的对称投影：
```
T_hat = P_G(x)(T_raw)
```
但当前代码中**未实现**具体的对称投影器。这是 ICLR 升级可扩展的方向。

---

## 4. 训练流程

### 4.1 目标归一化

每个任务独立计算训练集目标值的 mean 和 std，对 target 进行 z-score 归一化：
```python
target_norm = (target - mean) / std
```

评估时反归一化：
```python
pred = pred_norm * std + mean
```

### 4.2 评估指标

- **nMAE**（normalized MAE）：`MAE / MAD(target)`（`phytca.normalized_mae`）
- **Forgetting**：每个旧任务 best nMAE 与 final nMAE 之差平均
- **Backward Transfer (BWT)**：对误差指标，正数表示旧任务改善，负数表示遗忘
- **Raw MAE (eV)**：用于 band gap

### 4.3 优化器与超参数

- 优化器：AdamW
- 学习率：默认 `1e-3`
- weight decay：`1e-4`
- 学习率调度：CosineAnnealingLR
- 早停：patience=3 或 5
- 默认 batch size：32
- 默认 hidden_dim：64
- 默认 adapter_rank：8
- 默认 n_layers：3

### 4.4 持续训练流程

`train_phytca.continual_experiment()`（`train_phytca.py:190`）：
1. 构建模型，冻结 encoder；
2. 对每个任务：
   - 构造 train/val loader；
   - 调用 `train_task()` 训练；
   - 若当前任务是某个 (property, fidelity) 的最后一次出现，则 `freeze_task()`；
   - 保存 `anchor_state()` 作为 stability loss 的锚点；
   - 评估所有已见任务的 test set。

### 4.5 FR-PhyTCA 训练流程

`diagnostics.d6_progressive_tucker()`（`diagnostics.py:1320`）：

1. 构建 `ProgressivePhyTCAModel`；
2. 将子 adapter 的 OPT slice 置零；
3. **Task 1（OPT）**：仅训练父路径（子 adapter 冻结）；
4. 保存共享 OPT parent checkpoint；
5. 冻结父路径；
6. **Task 2（MBJ）**：仅训练子 adapter 和 MBJ head；
7. 每个 step 后调用 `zero_child_gradients_for_parent()`；
8. 可选蒸馏损失（lambda_distill），实验显示 lambda 变化不影响结果，证明父路径确实冻结；
9. 计算 OPT route drift：`max|y_OPT_after - y_OPT_before|`。

---

## 5. 基线方法

实现位置：`baselines.py`。

| 方法 | 说明 |
|------|------|
| Joint training | 所有任务联合训练，上界 |
| Independent models | 每个任务独立模型 |
| Sequential fine-tuning | 顺序微调，全部参数更新 |
| Frozen encoder + independent heads | 冻结编码器，每个任务独立 head |
| EWC | 对角 Fisher 正则 |
| Experience replay | 1% 回放 buffer |
| Independent LoRA | 每个任务独立 LoRA adapter |
| Shared LoRA bank | 共享 LoRA + 任务特定 gate |
| Symmetric PhyTCA | 共享 Tucker 因子 + stability loss |

---

## 6. 实验与结果

### 6.1 Phase 0：Protocol B 两任务筛选（2k/seed 42）

脚本：`scripts/run_phase0_b_screening.py`。

设置：
- train_cap=2000, val_cap=500, test_cap=1000
- test 进一步划分为 continual_dev（500）和 final_test（500）
- hidden_dim=64, adapter_rank=8, 10 epochs, patience=3
- 所有方法从同一 canonical base checkpoint 开始

结果（来自论文 Table 2 和 README）：

| Method | T1@T1 | T1@T2 | T2 final | Abs. forgetting | BWT | Avg. final nMAE | Train. params | Stored params |
|---|---|---|---|---|---|---|---|---|
| Sequential FT | 0.791 | 1.421 | 0.885 | 0.630 | -0.630 | 1.153 | 285,283 | 285,283 |
| + replay (1%) | 0.782 | 2.873 | 0.848 | 2.091 | -2.091 | 1.860 | 3,988 | 285,283 |
| Shared LoRA bank | 0.812 | 2.197 | 0.818 | 1.385 | -1.385 | 1.507 | 5,132 | 286,427 |
| PhyTCA (μ=0) | 0.782 | 2.940 | 0.794 | 2.159 | -2.159 | 1.867 | 3,858 | 285,283 |
| PhyTCA (μ=0.01) | 0.782 | 2.584 | 0.798 | 1.803 | -1.803 | 1.691 | 3,858 | 285,283 |
| Joint (upper bound) | 0.356 | 0.356 | 0.261 | 0.000 | 0.000 | 0.309 | 285,283 | 285,283 |

Gate 决策：`NO_GO_PHYTCA_NO_ADVANTAGE_AT_2K`。对称 PhyTCA 甚至不如 sequential fine-tuning。

### 6.2 诊断实验 D1–D6（2k/seed 42）

脚本：`scripts/run_phase0_diagnostics.py`。

D4/D5/D6 共享同一个训练好的 OPT parent checkpoint，确保公平比较。

结果（来自论文 Table 3）：

| Experiment | T1@T1 | T1@T2 | T2 final | Forgetting | BWT | Avg. nMAE | Raw MAE (eV) | Incr. params |
|---|---|---|---|---|---|---|---|---|
| D1 Full joint | 0.339 | 0.339 | 0.248 | 0.000 | 0.000 | 0.294 | 0.441 | 0 |
| D2 Joint PhyTCA | 0.471 | 0.471 | 0.357 | 0.000 | 0.000 | 0.414 | 0.621 | 0 |
| D3 Sequential PhyTCA | 1.406 | 2.511 | 2.106 | 1.105 | -1.105 | 2.309 | 1.818 | 165 |
| D4 Frozen OPT + affine MBJ | 0.881 | 0.881 | 0.835 | 0.000 | 0.000 | 0.858 | 1.288 | 8,450 |
| D5 Frozen OPT + residual MBJ | 0.881 | 0.881 | 0.838 | 0.000 | 0.000 | 0.859 | 1.290 | 4,225 |
| D6 FR-PhyTCA | 0.881 | 0.881 | 0.446 | 0.000 | 0.000 | 0.664 | 0.996 | 3,923 |

诊断结论：
- `DIAGNOSIS_ADAPTER_ON_RANDOM_BACKBONE`：encoder 随机初始化；
- `DIAGNOSIS_SEQUENTIAL_OPTIMIZATION_FAILURE`：D3 Task-1 已失败；
- `GO_TO_FIDELITY_RESIDUAL_PHYTCA`：冻结父路径 + Tucker 残差有效。

D6 ablations（来自论文 Table 4）：

| Experiment | T1@T1 | T1@T2 | T2 final | Avg. nMAE | Raw MAE (eV) | Incr. params |
|---|---|---|---|---|---|---|
| D6-a FR-PhyTCA | 0.881 | 0.881 | 0.446 | 0.664 | 0.996 | 3,923 |
| D6-b + distillation λ=1.0 | 0.881 | 0.881 | 0.446 | 0.664 | 0.996 | 3,923 |
| D6-c Independent low-rank | 0.881 | 0.881 | 0.905 | 0.893 | 1.341 | 520 |
| D6-d Parameter-matched MLP | 0.881 | 0.881 | 0.908 | 0.895 | 1.343 | 1,057 |
| D6-e Orthogonal Tucker | 0.881 | 0.881 | 0.446 | 0.664 | 0.996 | 3,923 |
| D6-f Shared factors + top layer | 0.881 | 1.180 | 0.342 | 0.761 | 1.142 | 92,632 |

### 6.3 2k × 3-seed 复现

来自 `reports/phase0_b_screening/repro_2k3seed_summary.json`：

| Method | T1@T1 | T1@T2 | T2 final | Avg. final nMAE | Raw MAE (eV) | Incr. params |
|---|---|---|---|---|---|---|
| Full joint | 0.337±0.036 | 0.337±0.036 | 0.225±0.027 | 0.281±0.031 | 0.424±0.043 | 0 |
| Joint PhyTCA | 0.481±0.048 | 0.481±0.048 | 0.348±0.032 | 0.415±0.040 | 0.626±0.052 | 0 |
| Sequential PhyTCA | 1.482±0.069 | 1.882±0.517 | 2.039±0.240 | 1.961±0.269 | 1.527±0.210 | 165 |
| Frozen OPT + affine | 0.824±0.044 | 0.824±0.044 | 0.840±0.032 | 0.832±0.036 | 1.258±0.061 | 8,450 |
| Frozen OPT + residual | 0.824±0.044 | 0.824±0.044 | 0.841±0.034 | 0.832±0.036 | 1.258±0.062 | 4,225 |
| **FR-PhyTCA** | **0.824±0.044** | **0.824±0.044** | **0.382±0.038** | **0.603±0.004** | **0.912±0.009** | **3,923** |
| Orthogonal FR-PhyTCA | 0.824±0.044 | 0.824±0.044 | 0.382±0.038 | 0.603±0.004 | 0.912±0.010 | 3,923 |
| Shared factor + top layer | 0.824±0.044 | 0.911±0.035 | 0.324±0.031 | 0.617±0.004 | 0.933±0.018 | 92,632 |

### 6.4 Stage 2：5k × 3-seed 扩展验证

脚本：`scripts/run_phase2_b_scaling.py`。

来自 `reports/phase2_b_scaling/scaling_aggregate.json`：

| Method | T1@T1 | T1@T2 | T2 final | Avg. final nMAE | Raw MAE (eV) | Incr. params |
|---|---|---|---|---|---|---|
| Full joint | 0.293±0.034 | 0.293±0.034 | 0.188±0.009 | 0.241±0.017 | 0.367±0.025 | 0 |
| Joint PhyTCA | 0.471±0.022 | 0.471±0.022 | 0.310±0.024 | 0.390±0.017 | 0.595±0.027 | 0 |
| MBJ-only | 1.056±0.228 | 1.056±0.228 | 0.758±0.037 | 0.907±0.133 | 1.630±0.240 | 3,988 |
| OPT pretrain → MBJ full FT | 0.839±0.030 | 1.793±0.812 | 0.895±0.045 | 1.344±0.404 | 2.048±0.607 | 285,283 |
| Frozen OPT + affine | 0.839±0.030 | 0.839±0.030 | 0.842±0.017 | 0.840±0.022 | 1.281±0.038 | 8,450 |
| Frozen OPT + matched MLP | 0.839±0.030 | 0.839±0.030 | 0.829±0.033 | 0.834±0.028 | 1.271±0.048 | 3,895 |
| Frozen OPT + matched low-rank | 0.839±0.030 | 0.839±0.030 | 0.831±0.017 | 0.835±0.015 | 1.273±0.029 | 3,900 |
| **FR-PhyTCA** | **0.839±0.030** | **0.839±0.030** | **0.342±0.022** | **0.590±0.026** | **0.900±0.040** | **3,923** |
| Orthogonal FR-PhyTCA | 0.839±0.030 | 0.839±0.030 | 0.342±0.021 | 0.590±0.025 | 0.900±0.039 | 3,923 |
| Feature transfer | 0.839±0.030 | 0.839±0.030 | 0.823±0.017 | 0.831±0.013 | 1.267±0.018 | 8,321 |

Gate 决策：`GO_TO_REALISTIC_FIDELITY_SCALING`。

---

## 7. 理论结果

论文中提出的三个理论结果：

### 7.1 Theorem 1：Parent-route invariance

冻结父路径参数 Θ_P，Task 2 优化器只更新 Θ_P 外的参数，且子 adapter 的父 fidelity slice 保持为零，则：
```
f_OPT(x; Θ^(2)) = f_OPT(x; Θ^(1))
```
绝对遗忘为零。

### 7.2 Proposition 2：Residual parameter growth

增加一个新 fidelity 的参数增长上界：
```
ΔP_fid ≤ L(2dr + r²) + d
```
其中 L 是层数，d 是 hidden_dim，r 是 child Tucker rank。增长与之前 fidelity 数量无关。

### 7.3 Proposition 3：Fidelity-error decomposition

```
|y_MBJ - ŷ_MBJ| ≤ |y_OPT - ŷ_OPT| + |Δy - Δ̂y|
```

说明 MBJ 误差受限于父 OPT 误差加上残差预测误差。因此好的父模型很重要。

---

## 8. 代码结构

```
Continual Learning/
├── data.py                        # 数据加载、协议构建、周期图构建
├── data_audit.py                  # 数据审计与 GO/NO-GO Gate
├── phytca.py                      # 对称 PhyTCA 基线（Tucker4DAdapter, PhyTCAModel）
├── diagnostics.py                 # FR-PhyTCA 实现与诊断实验
├── train_phytca.py                # 持续训练主入口
├── baselines.py                   # 基线方法
├── scripts/
│   ├── run_phase0.py              # Phase 0 多方法比较
│   ├── run_phase0_b_screening.py  # Protocol B 两任务筛选
│   ├── run_phase0_diagnostics.py  # D1-D6 诊断实验
│   └── run_phase2_b_scaling.py    # 5k×3-seed 扩展验证
├── tests/
│   ├── test_periodic_graph.py     # PBC 正确性、协议划分测试
│   ├── test_protocol_semantics.py # 协议语义不变量
│   ├── test_diagnostics.py        # FR-PhyTCA 父路径不变性
│   └── test_metrics.py            # 持续学习指标单元测试
├── configs/
│   ├── jarvis_protocol_a.yaml
│   └── jarvis_protocol_b.yaml
└── reports/                       # 审计报告、实验结果、Gate 文件
```

---

## 9. 测试覆盖

运行：
```bash
python -m pytest tests/ -v
```

测试内容：
- 目标解析鲁棒性；
- JARVIS → pymatgen 结构转换；
- 超胞大小、original_mask 一致性；
- 周期图平移不变性；
- PBC 不变性（depth 1/2/4）；
- halo 收敛性（2×2×2 vs 3×3×3 vs 4×4×4）；
- 原胞 vs 超胞等价性；
- Protocol A JID 不相交；
- Protocol B OPT/MBJ 配对共享划分；
- 公式不相交 split；
- FR-PhyTCA 子 adapter OPT slice 永久为零；
- 父路径预测精确保留；
- 指标计算正确性。

---

## 10. 已知的实现选择与潜在问题（供 GPT 审核）

### 10.1 可能可用现有库替代的部分

1. **EGNN 实现**：当前用 `egnn_pytorch`。可考虑换成：
   - `torch_geometric` + SchNet/DimeNet/ALIGNN；
   - `dgl`；
   - `fairchem` / `matgl` 中的预训练晶体图编码器。

2. **Tucker 分解**：当前手动实现 einsum。可考虑：
   - `tensorly` 库的 `tucker_to_tensor`；
   - 但当前实现高度定制化（冻结 slice、zero gradient 等），直接替换需谨慎。

3. **LoRA 实现**：`baselines.py` 中的 `LoRALinear` 是手动实现的。可用：
   - `peft` 库（Hugging Face）；
   - 但当前只用于 node_embed 和 heads，规模小，手动实现可控。

4. **数据加载**：当前用 `torch.utils.data.Dataset` + 自定义 `collate_crystals`。可考虑：
   - `torch_geometric.data.Data` 和 `DataLoader`；
   - 可简化 batching 和 padding 逻辑。

5. **对称投影器**：当前未实现。实现时可考虑：
   - `pymatgen.symmetry.analyzer.SpacegroupAnalyzer` 获取点群；
   - `torch` 手动构建投影矩阵。

### 10.2 可能实现不正确或需改进的地方

1. **EGNN 的周期性处理**：
   - 当前通过 `2×2×2` 超胞近似周期边界，但 EGNN 本身使用 k-NN 图，未显式考虑周期性边；
   - 若两个原胞原子在周期镜像中接近，k-NN 可能选取同一镜像多次或漏掉跨边界邻居；
   - 更正确的做法：显式构建周期邻居列表（如 ASE、DGL-LifeSci、alignn 的 periodic graph）。

2. **超胞映射的鲁棒性**：
   - `data.py:604` 使用 `np.rint` 映射超胞原子到原胞原子，依赖浮点精度；
   - 对复杂晶胞或有畸变的结构，可能映射错误；
   - 建议增加断言检查每个超胞原子恰好找到一个原胞原子的距离小于阈值。

3. **EGNN update_coors=True 与周期图**：
   - EGNN 会更新坐标，但当前未将更新后的坐标限制在原胞或周期镜像；
   - 后续层若依赖坐标，可能破坏周期一致性；
   - 测试中 `update_coors=False`，但实际模型 `update_coors=True`。

4. **Tucker4DAdapter 的 gradient zeroing**：
   - `zero_frozen_gradients()` 在 `G.grad` 上按 slice 置零；
   - 若 optimizer 使用 momentum 或 Adam 的二阶矩，历史动量仍会影响被冻结 slice；
   - 更严格的做法：使用 `param.register_hook(lambda grad: ...)` 或将被冻结 slice 的 parameter 完全移出 optimizer。

5. **ProgressiveTuckerAdapter 的 child OPT slice**：
   - 当前在初始化时 `zero_and_freeze_child_slice`，并在每次 backward 后 zero gradient；
   - 但若 optimizer 使用了 weight decay 或 momentum，子 OPT slice 仍可能被微小更新；
   - 实测 `opt_route_drift=0.00`，但理论上应更严格：将子 OPT slice 从 child.G 中排除，或使 child.G 的该 slice 不是 leaf parameter。

6. **Protocol A 的数据增量语义**：
   - A1 和 A2 是同一 (property, fidelity) 的数据增量快照，共享 head；
   - 但当前在 A1 后不冻结，在 A2 后才冻结；
   - 然而 A1/A2 是 JID 不相交的，模型在 A2 训练时实际是在新数据上继续训练同一 head，这等价于联合训练 A1+A2；
   - 这个设计在持续学习意义上是否能体现“遗忘”存疑，因为 A1 数据在 A2 中不可见，但 head 共享。

7. **对称 PhyTCA 基线的稳定性损失**：
   - `stability_loss` 对当前所有可训练参数做 L2 锚定；
   - 这会抑制新任务学习，但无法真正保证旧任务输出不变；
   - 实验也证明其效果不佳。

8. **张量性质与物理约束未实现**：
   - 论文方法部分提到对称投影，但代码中未实现；
   - 这是当前工作与材料物理结合最深、但实现最少的部分。

### 10.3 架构可优化方向

1. **更强的 backbone**：
   - 当前 encoder 随机初始化，容量和表征质量有限；
   - 可替换为 ALIGNN、Matformer、CrystalTransformer、JMP 等预训练模型；
   - 预训练 backbone 可能显著缩小 D2-D1 的 architecture gap。

2. **真正的周期图神经网络**：
   - 使用显式周期边（periodic edges）而非超胞近似；
   - 参考 alignn、CHGNet、MACE 的周期图构建方式。

3. **多轴 adapter 的更高效实现**：
   - 当前 4D Tucker 对每个 (p,f) 都做一次 einsum；
   - 可预计算所有 (p,f) 的 core slice 缓存；
   - 可用低秩近似替代完整 Tucker core。

4. **自适应 rank 扩展**：
   - 当前 rank 固定；
   - 可根据新任务在旧子空间上的投影残差决定是否扩展 rank。

5. **Fidelity Graph / DAG 扩展**：
   - 当前只有 OPT→MBJ 两节点；
   - 可扩展为 fidelity DAG：PBE→GLLB-SC→SCAN→HSE→Experiment；
   - 每个边是一个 residual，可组合多条路径；
   - 这能显著提升方法的 generalization 和 ICLR 新颖性。

6. **多性质与张量性质**：
   - 当前只做了 band gap（标量）和 formation energy（标量，Protocol A）；
   - 增加 elastic、dielectric、piezoelectric 等张量性质；
   - 实现点群对称投影器。

7. **样本效率与不平衡 fidelity**：
   - 真实多保真学习中低保真数据远多于高保真；
   - 当前 2k/5k 是等大小的；
   - 应测试 10k OPT → 2k MBJ、30k OPT → 5k MBJ 等比例。

### 10.4 Idea 升级方向（针对 ICLR 2027）

1. **Continual Fidelity Graph Adaptation (CFGA)**：
   - 将 fidelity 建模为有向无环图；
   - 新 fidelity 到达时自动选择父节点；
   - 学习节点间的结构化 residual edge；
   - 保证旧节点精确不变；
   - 可证明路径一致性、误差传播上界。

2. **Physics-Guided Tensor Component Reuse**：
   - 分析哪些 Tucker 组件被哪些性质/fidelity 复用；
   - 与已知物理关联对比（如 bulk/shear modulus 共享、dielectric/piezoelectric 部分共享）；
   - 提供可解释性。

3. **Exact-Retention Parameter-Efficient Continual Learning 理论框架**：
   - 不仅针对材料；
   - 抽象为：冻结旧子空间 + 在新子空间学习残差；
   - 给出零遗忘、参数增长、误差传播的通用界。

4. **跨数据库迁移**：
   - 同时利用 MP 和 JARVIS；
   - 数据库作为 fidelity/域的一个轴。

5. **与 SOTA 多保真方法对比**：
   - MFGNet；
   - Node Transfer；
   - Δ-learning；
   - Denoising multi-fidelity learning。

---

## 11. 运行命令速查

```bash
# 数据审计
python data_audit.py --protocol a --report-dir reports
python data_audit.py --protocol b --report-dir reports

# 完整训练（对称 PhyTCA）
python train_phytca.py --protocol b --hidden-dim 64 --adapter-rank 8 --epochs 20 --device cuda

# Protocol B 两任务筛选
python scripts/run_phase0_b_screening.py \
  --train-cap 2000 --val-cap 500 --test-cap 1000 \
  --seed 42 --epochs 10 --patience 3 --batch-size 32 \
  --hidden-dim 64 --adapter-rank 8 --device cuda --with-joint

# D1-D6 诊断实验
python scripts/run_phase0_diagnostics.py --device cuda

# 5k×3-seed 扩展验证
python scripts/run_phase2_b_scaling.py \
  --train-cap 5000 --val-cap 500 --test-cap 1000 \
  --seeds 42 43 44 --epochs 10 --patience 3 --batch-size 32 \
  --hidden-dim 64 --adapter-rank 8 --device cuda

# 测试
python -m pytest tests/ -v
```

---

## 12. 依赖环境

- Python 3.11+
- PyTorch
- `egnn-pytorch`
- `pymatgen`
- `jarvis-tools`
- `numpy`
- `pytest`

Conda 环境名：`EGNN`。

---

## 13. 总结

FR-PhyTCA 当前已在 JARVIS Protocol B（OPT→MBJ band gap）上通过多阶段验证：
- 对称 PhyTCA 基线失败；
- 冻结父路径 + Tucker 残差实现精确零遗忘；
- 在 2k 和 5k 设置下均显著优于参数匹配的 MLP/low-rank 残差基线；
- 父路径漂移严格为 0。

但当前版本仍存在 backbone 较弱、周期图处理较简单、张量性质未实现、fidelity 图仅两节点等问题。若目标是 ICLR 2027 主会，建议升级为 **Continual Fidelity Graph Adaptation** 框架，扩展至多 fidelity DAG、多数据源、多性质，并与 MFGNet、Node Transfer 等 SOTA 方法系统对比。
