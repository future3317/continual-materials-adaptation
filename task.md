## 总判断

**这个工作可以转成 AI4Materials，但不能只是把 Split CIFAR-100 换成 Materials Project 和 JARVIS。**

当前论文的核心是“低秩组件复用 + 稀疏门控 + 持续学习”，但材料领域真正有价值的问题不是图像分类任务不断增加，而是：

> **材料数据库会持续加入新材料、新性质和新计算精度；模型怎样在不重新训练全部历史数据的情况下吸收新知识，同时保留旧性质预测能力？**

我最推荐把它改造成：

# Continual Tensor Adaptation for Evolving Multi-Property and Multi-Fidelity Materials Databases

中文可以叫：

> **面向演化材料数据库的多性质、多保真持续张量适配**

这个方向比“在 JARVIS 上做持续学习”强得多，也能让 Tucker 分解真正具有科学含义。

---

# 一、先审查当前 TCCL：现在还不能直接变成材料论文

## 1. 当前所谓 Tucker 分解，本质上只是低秩矩阵分解

论文只对最终分类层的二维矩阵做

[
W=UGV^\top.
]

对于没有附加约束的二维矩阵，中心矩阵 (G) 可以被吸收到 (U) 或 (V) 中，因此这并不是特别有辨识度的“Tucker 组件”。而且论文目前只替换了 ResNet-18 的最后全连接层，backbone 基本冻结，测试时还提供 task identity。

这会带来一个严重问题：

> 它目前证明的主要是“怎样给冻结特征加一个参数较少的任务头”，而不是“怎样持续积累材料表征知识”。

如果只是把 ResNet 换成 ALIGNN，再在最后一层预测 formation energy、band gap，这个问题仍然存在。

---

## 2. 模型增长问题实际上没有解决

每个任务都增加一个

[
\Delta G_t\in\mathbb R^{R\times R},
]

所以任务特定参数仍然是

[
O(TR^2).
]

当任务数增加时仍然线性增长。论文声称控制模型增长，但实验中的参数量只统计了 decomposed head，没有统计完整 backbone，也没有和“每个任务一个低秩 adapter”做公平比较。

材料领域可能有几十种性质、多个数据库、多个 DFT functional，这个问题会更加明显。

---

## 3. 理论部分存在实质性错误

最明显的是 Corollary 3.4。

原证明得到：

[
n_t\lambda_{\rm share}\leq n_t\Delta_{\max},
]

然后声称可以推出：

[
n_t\leq \frac{\Delta_{\max}}{\lambda_{\rm share}}.
]

这是推不出来的，因为 (n_t) 在两边会直接消掉。正确的条件应当是：

> 新增全部组件所能带来的**总损失下降**至多为 (B_t)，每个组件至少产生 (\lambda) 的正则代价，才能推出 (n_t\leq B_t/\lambda)。

另外还有两个问题：

* Theorem 3.3 的正文使用了 (\nabla L_t(\Theta^{t-1}))，附录推导却使用 (\nabla L_t(\Theta^t))，两者不一致。
* 它约束的是旧参数漂移，但旧任务损失是否只依赖这些参数并未严格证明；新扩展组件、共享 core 或 normalization 的变化也可能影响旧任务。

这些理论在改写材料论文前需要全部重做。

---

## 4. 当前实验不足以支撑高水平论文

目前只有：

* Split CIFAR-100；
* 单一任务顺序；
* 冻结 backbone；
* FineTune、EWC、Replay 三个简单基线；
* 没有多 seed、顺序敏感性、梯度冲突、表示漂移；
* 没有 LoRA、独立 adapter、动态低秩、Mixture-of-Experts 等更直接的竞争方法。

因此我的评价是：

| 版本                    |             研究潜力 |
| --------------------- | ---------------: |
| 当前 TCCL               |             4/10 |
| 仅将 CIFAR 换成 MP/JARVIS |             5/10 |
| 改成多性质、多保真、数据库演化持续学习   |             8/10 |
| 再加入晶体对称性和张量性质         | 8.5/10，但实现难度明显增加 |

---

# 二、为什么 MP 和 JARVIS 非常适合这个问题

MP 和 JARVIS 不是两个普通数据集，而是两个具有不同计算协议、数据分布和性质覆盖的材料知识源。

ALIGNN 的研究已经指出，MP 和 JARVIS 即使预测相同的 formation energy 或 band gap，也会受到不同 functional、DFT+U 设置、k-point、smearing 和材料分布的影响。MP 主要包含 PBE/GGA 系列结果，而 JARVIS 大量使用 OptB88vdW，并对部分性质提供 TB-mBJ 等更高精度结果。JARVIS 的经典 ALIGNN 基准包含约 55,722 个材料和 29 个回归性质。([Nature][1])

更重要的是，材料数据库本身确实在不断增长。一项针对数据库版本变化的研究使用了：

* JARVIS18：约 53k；
* JARVIS22：约 76k；
* MP18：约 68k；
* MP21：约 146k；

并将旧版本之外的新材料作为时间分布外测试集。这天然就是一个真实的 continual learning 场景，而不是人为把类别切成十份。

---

# 三、现有材料工作已经做到了哪里

必须注意：以下几个简单方向已经比较拥挤。

### 1. 跨性质迁移已经有人做

已有工作用大规模性质预训练，再迁移到小规模材料性质，并在 39 个计算性质和两个实验性质上进行了验证。([Nature][2])

2024 年的研究进一步比较了单性质预训练、多性质联合预训练和 fine-tuning，在 7 个数据集上表明，多性质预训练有时优于成对迁移。([Nature][3])

### 2. 多模型、Mixture-of-Experts 也已经有人做

已有材料 MoE 工作联合多个预训练模型和数据集，在 19 个材料性质任务上优于多数成对 transfer learning。([Nature][4])

因此，“给每个性质一个 gate，然后共享若干组件”本身不够新。

### 3. 材料模型的 catastrophic forgetting 也开始有人研究

在机器学习原子势领域，reEWC 已经把经验回放与 EWC 结合，用于保持预训练 MLIP 的泛化能力。([arXiv][5])

2026 年一项更系统的研究已经比较了 full fine-tuning、冻结层、LoRA、multi-head replay、pseudo-label replay 和 LoRA+replay，并发现 replay 对维持预训练分布鲁棒性非常重要。([arXiv][6])

所以仅仅做：

> ALIGNN + LoRA/EWC + MP/JARVIS

现在已经很难成为有竞争力的创新。

---

# 四、我最推荐的核心研究问题

## 研究问题

> 给定一个持续演化的晶体预测模型，新的材料、性质和 DFT fidelity 会异步到达。模型在不能重新访问全部历史数据、不能为每个新任务保存完整模型的条件下，如何复用已有物理组件、吸收新知识，并控制旧性质遗忘和模型增长？

这里包含三个互补维度。

### 维度一：数据库时间演化

同一个性质、同一个预测头：

[
\text{MP18}\rightarrow \text{MP21},
\qquad
\text{JARVIS18}\rightarrow \text{JARVIS22}.
]

这是真正的 domain/data incremental learning，不需要 task ID，也不能靠“每个任务单独一个 head”逃避遗忘问题。

### 维度二：计算保真度演化

例如 band gap：

[
\text{MP-PBE}
\rightarrow
\text{JARVIS-OptB88vdW}
\rightarrow
\text{JARVIS-TBmBJ}.
]

这里不是简单 label shift，而是不同电子结构理论给出的系统性偏差。跨 functional 迁移中，能量参考尺度和原子参考能都可能显著不同。([arXiv][7])

### 维度三：性质持续增加

例如：

[
E_f
\rightarrow E_g
\rightarrow K
\rightarrow G
\rightarrow \varepsilon
\rightarrow e_{ijk}.
]

它们分别对应：

* 热力学；
* 电子结构；
* 弹性；
* 介电响应；
* 压电响应。

这样能够测试组件复用是否真的对应物理关联，而不是仅仅压缩参数。

---

# 五、方法应当怎样改：从 TCCL 变成真正的材料张量适配

## 1. 对“任务轴”进行真正的 Tucker 分解

不要只分解一个二维权重矩阵，而是定义多维适配张量：

[
\mathcal A^{(\ell)}
\in
\mathbb R^{d_{\rm out}\times d_{\rm in}\times N_p\times N_f},
]

其中：

* 第一个维度：输出通道；
* 第二个维度：输入通道；
* 第三个维度：材料性质；
* 第四个维度：数据库或 DFT fidelity。

进行 Tucker 分解：

[
\mathcal A^{(\ell)}
===================

\mathcal G^{(\ell)}
\times_1 U^{(\ell)}
\times_2 V^{(\ell)}
\times_3 E_{\rm prop}
\times_4 E_{\rm fidelity}.
]

对于性质 (p)、保真度 (f)，收缩后得到当前 adapter：

[
\Delta W^{(\ell)}_{p,f}
=======================

\mathcal A^{(\ell)}
\times_3 e_p
\times_4 e_f.
]

这比当前 (UGV^\top) 强很多，因为它明确分离了：

* 通用材料表征；
* 性质相关知识；
* functional 或数据库相关偏差。

这才是真正能够在论文中解释的 structured tensor decomposition。

---

## 2. Adapter 必须进入晶体编码器，而不只是最后一层

建议首先使用 ALIGNN：

* JARVIS 官方实现成熟；
* 多性质基准完整；
* 单卡 4090 可以承受；
* 容易在 node、edge 和 line-graph 更新层插入低秩 adapter。

至少应在以下位置加入 adapter：

* atom embedding projection；
* edge/bond update；
* angle/line-graph update；
* graph readout 前的 projection。

必须比较：

[
\text{head-only}
\quad \text{vs.}\quad
\text{all-layer tensor adapters}.
]

如果 head-only 与完整方法差不多，就说明所谓持续材料知识复用只是任务头复用，没有学习新的材料表示。

---

## 3. 用自适应扩展替代手工 rank schedule

当前论文使用预先规定的

[
R_t=4+2\lfloor t/2\rfloor,
]

缺乏依据。建议根据当前任务梯度在已有组件空间中的残差来决定是否扩展：

[
\rho_t =
\frac{|(I-P_{\rm old})g_t|}
{|g_t|}.
]

* (\rho_t) 小：新任务主要可由旧组件解释，只学习 gate；
* (\rho_t) 大：存在新的物理方向，增加少量 tensor components。

这样能够提出更有意义的解释：

> rank expansion 表示模型遇到了不能由已有热力学、电子或力学组件表示的新知识。

---

## 4. 将 task-specific residual 改成低秩稀疏 residual

不要继续使用完整的 (R\times R) residual core。

可以改成：

[
\Delta G_t = a_tb_t^\top,
]

或者少量 rank-(r_t) 残差：

[
\Delta G_t=A_tB_t^\top,\qquad r_t\ll R.
]

并对 residual 施加 group sparsity，使任务特定参数保持在 (O(Rr_t))，而不是 (O(R^2))。

---

## 5. 加入材料物理约束

这是从“普通持续学习”变为“AI4Materials”的关键。

对于不同性质，应使用不同输出约束：

* formation energy、band gap：旋转不变标量；
* dielectric tensor：二阶对称张量；
* elastic tensor：满足 minor/major symmetry；
* piezoelectric tensor：满足晶体点群约束，在中心对称晶体中应为零。

JARVIS 中已经包含 elastic、dielectric、piezoelectric、mBJ band gap 等多类性质，且部分张量性质仍然具有较大预测空间。([Nature][1])

可以采用：

[
\widehat T=P_{\mathcal G(x)}\big(T_{\rm raw}\big),
]

其中 (P_{\mathcal G(x)}) 是由材料点群决定的对称性投影器。

这样模型组件的复用不只是“参数共享”，还可以研究：

> 标量热力学知识是否帮助弹性性质？
> 弹性组件是否被介电和压电任务复用？
> OptB88vdW 与 mBJ 的差异主要发生在哪些组件？

这会成为材料论文中最重要的科学分析。

---

# 六、推荐的数据和任务协议

## 主协议 A：真实数据库演化

| 阶段 | 数据            | 性质               |
| -- | ------------- | ---------------- |
| 1  | JARVIS18      | formation energy |
| 2  | JARVIS22 新增材料 | formation energy |
| 3  | MP18          | formation energy |
| 4  | MP21 新增材料     | formation energy |

同一个输出头，不能使用 task-specific head。

目标是同时衡量：

* 新版本适应；
* 旧版本遗忘；
* 跨数据库迁移；
* 参数增长。

---

## 主协议 B：多保真 band-gap 学习

[
\text{MP-PBE}
\rightarrow
\text{JARVIS-OPT}
\rightarrow
\text{JARVIS-mBJ}.
]

应对数据库间重叠材料进行匹配，单独分析：

[
\Delta E_g^{\rm PBE\rightarrow mBJ}.
]

这个协议的材料意义最强，因为不同 functional 给出的 band gap 存在清晰的精度和系统偏差区别。

---

## 辅协议 C：性质增量

建议从数据充足到数据稀缺：

1. formation energy；
2. band gap；
3. bulk modulus；
4. shear modulus；
5. dielectric constant；
6. piezoelectric response。

Matbench 本身包含 13 个不同规模和来源的任务，规模从 312 到约 132k，适合补充跨性质实验。([Nature][8])

---

# 七、实验必须怎样设计

## 数据划分

不能只做 random split。

材料数据库中有大量成分、结构原型和近重复材料，随机划分容易高估模型性能。材料 OOD 基准研究已经表明，结构分布外测试会使现有 GNN 性能显著下降。([arXiv][9])

至少需要：

* formula-disjoint split；
* composition-system-disjoint split；
* structure-prototype-disjoint split；
* old-version/new-version temporal split；
* MP/JARVIS 重复结构去泄露。

---

## 基线

至少包括：

| 类型     | 方法                                    |
| ------ | ------------------------------------- |
| 上界     | Joint training on all historical data |
| 参数上界   | 每个性质独立 ALIGNN                         |
| 简单持续学习 | Sequential full fine-tuning           |
| 冻结表示   | Frozen ALIGNN + independent heads     |
| 正则化    | EWC、LwF                               |
| 回放     | Reservoir replay、prototype replay     |
| PEFT   | Independent LoRA、shared LoRA          |
| 模块化    | Adapter bank、Mixture-of-Experts       |
| 本方法    | Multi-axis Tucker continual adapter   |

如果做等变 backbone，还应加入 ELoRA 类方法。

---

## 指标

不同材料性质单位不同，不能直接平均 MAE。建议使用：

[
\mathrm{nMAE}_p =
\frac{\mathrm{MAE}_p}
{\mathrm{MAD}(y_p)}.
]

持续学习指标包括：

* Final Average nMAE；
* Average Forgetting；
* Backward Transfer；
* Forward Transfer；
* 新增参数量；
* 训练时间与峰值显存；
* 每个任务新增 rank。

对于稳定性筛选，不应只报告 formation-energy MAE。Matbench Discovery 已指出，回归误差与真正的稳定材料筛选效果可能并不一致，因此还应报告稳定材料 precision、recall、F1 和 discovery acceleration factor。([Nature][10])

---

# 八、必须先做的 Phase 0 审计

在实现完整方法前，先做四个非常便宜的实验。

### Phase 0.1：确认材料任务真的会遗忘

训练顺序：

[
\text{MP formation energy}
\rightarrow
\text{MP band gap}
\rightarrow
\text{JARVIS formation energy}.
]

比较旧任务 fine-tuning 前后的 MAE。

如果旧任务 MAE 几乎不变，说明当前设定不需要 continual learning。

### Phase 0.2：确认共享表示真的有正迁移

比较：

* scratch；
* frozen encoder；
* full fine-tuning；
* LoRA；
* joint training。

如果新性质用 frozen encoder 已经达到最好结果，复杂的组件扩展没有必要。

### Phase 0.3：计算任务冲突

记录不同性质之间的：

* gradient cosine；
* representation CKA；
* adapter subspace overlap。

理想情况应当看到：

* formation energy 与 (E_{\rm hull}) 高共享；
* bulk 与 shear modulus 高共享；
* dielectric 与 piezoelectric 有部分共享；
* PBE 与 mBJ band gap 共享基础电子组件，但需要 fidelity-specific residual。

### Phase 0.4：数据库偏差审计

对 MP 与 JARVIS 中匹配的材料分析：

* formation energy 差值；
* band-gap 差值；
* 元素和晶系分布；
* 2D/3D 比例；
* functional-dependent systematic shift。

这部分本身就可以成为论文的一张重要 motivation figure。

---

# 九、最后可以形成的论文贡献

最终论文最好不是只写“我们把 TCCL 用到了材料数据”，而是形成以下四项贡献：

1. **建立演化材料数据库持续学习协议**
   覆盖数据库版本、材料性质和 DFT fidelity。

2. **提出多轴 Tucker 材料适配器**
   在通道、性质、fidelity 三个维度分解和复用知识。

3. **实现自适应组件扩展**
   根据新任务是否超出已有物理子空间决定是否扩 rank。

4. **给出材料知识复用的科学解释**
   分析哪些性质共享组件、哪些 functional 需要专属组件，以及共享关系是否符合已知物理联系。

我没有在检索中找到一个同时处理：

> **MP/JARVIS 数据库演化 + 多性质 + 多 functional + 参数受限持续学习 + 可解释组件复用**

的完全同构公开工作。现有研究更多分别研究跨性质迁移、多任务预训练、MoE、数据库 OOD，或 MLIP fine-tuning。因此这个交叉点仍然有空间，但投稿前仍需对 2026 年后续工作再做一次系统检索。

---

## 最终建议

**不要直接做“TCCL-Mat”。**

建议重新命名和重写为：

> **PhyTCA: Physics-Structured Tensor Component Adaptation for Continually Evolving Materials Databases**

先以 **ALIGNN + MP/JARVIS 标量性质**完成 Phase 0 和主方法，保证单卡 4090 能跑通；确认有效后，再加入 dielectric、elastic、piezoelectric 等张量性质和晶体对称性约束。这样既控制工程成本，也能逐步把一篇普通持续学习工作提升为真正有材料问题、材料数据和材料解释的 AI4Materials 工作。

[1]: https://www.nature.com/articles/s41524-021-00650-1 "Atomistic Line Graph Neural Network for improved materials property predictions | npj Computational Materials"
[2]: https://www.nature.com/articles/s41467-021-26921-5 "Cross-property deep transfer learning framework for enhanced predictive analytics on small materials data | Nature Communications"
[3]: https://www.nature.com/articles/s41524-024-01486-1 "Optimal pre-train/fine-tune strategies for accurate material property predictions | npj Computational Materials"
[4]: https://www.nature.com/articles/s41524-022-00929-x "Towards overcoming data scarcity in materials science: unifying models and datasets with a mixture of experts framework | npj Computational Materials"
[5]: https://arxiv.org/html/2506.15223v1 "An efficient forgetting-aware fine-tuning framework for pretrained universal machine-learning interatomic potentials"
[6]: https://arxiv.org/html/2606.12704v1 "Fine-tuning MLIP foundation models: strategies for accuracy and transferability"
[7]: https://arxiv.org/html/2504.05565v1 "Cross-functional transferability in universal machine learning interatomic potentials"
[8]: https://www.nature.com/articles/s41524-020-00406-3 "Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm | npj Computational Materials"
[9]: https://arxiv.org/abs/2401.08032?utm_source=chatgpt.com "Structure-based out-of-distribution (OOD) materials property prediction: a benchmark study"
[10]: https://www.nature.com/articles/s42256-025-01055-1 "A framework to evaluate machine learning crystal stability predictions | Nature Machine Intelligence"
