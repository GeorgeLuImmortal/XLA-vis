# 小米 XLA 认知大模型：技术全景与各工作串讲

## 一、XLA 架构总览

XLA（X-L-A）是小米具身智能认知大模型的顶层架构，其核心理念可以概括为三个字母：

- **X — 多模态认知输入（Multi-modal Inputs）**：支持雷达、麦克风、摄像头、LiDAR、GPS、互联网信息、机器人本体感知等多种输入模态。
- **L — 高效的原生语言（Native Language）**：以潜空间思维链（Latent CoT）和行为 Token（Action Token）作为核心"语言"，基座模型为 Xiaomi MiMo-Embodied。
- **A — 拟人化的行为（Human-like Action）**：输出层涵盖行为（Action）、世界模型（World Model）、行为专家（Expert）、奖励函数（Reward Model）、状态估计（States）、强化学习（RL），通过闭环训练（Closed-loop Training）持续迭代。

XLA 的设计哲学是：**更多模态、更高效、更可控**。围绕这一目标，团队在输入感知、语言表征、行为生成三个维度分别展开了深入的研究工作。以下五项工作各自解决了 XLA 链路中的关键技术瓶颈，彼此形成上下游协同。

---

## 二、五项工作的定位与关联

| 工作 | 在 XLA 中的定位 | 核心解决的问题 |
|------|-------------|----------|
| **SpatioLM** | X 层 — 空间感知增强 | 如何让 VLM 获得物理空间智能，而不依赖额外 3D 先验 |
| **GDA** | X 层 — 深度融合保真 | 如何注入 3D 几何信息而不破坏预训练的视觉-语言对齐 |
| **OneVL** | L 层 — Latent CoT | 如何将显式推理链压缩为潜空间表征，同时保留因果动力学信息 |
| **LTA** | L 层 — Action Token 的生成范式 | 如何突破自回归瓶颈，实现高质量并行 token 生成 |
| **NanoVLA** | A 层 — 边缘端部署 | 如何将完整 VLA 能力部署到资源受限的边缘设备 |

五项工作在 XLA 链路中的位置：

```
        ┌─────────── 感知输入 (X) ───────────┐     ┌── 语言/推理 (L) ──┐     ┌── 行为输出 (A) ──┐
        │                                    │     │                   │     │                  │
  SpatioLM ── 空间智能增强 ──┐                │     │                   │     │                  │
        │                    ├→ 3D感知token ──→ OneVL → Latent推理 ──→ NanoVLA → 动作执行
  GDA ────── 深度融合保真 ──┘                │     │                   │     │                  │
        │                                    │     │                   │     │                  │
        └────────────────────────────────────┘     └─ LTA: 并行生成 ──┘     └──────────────────┘
                                                      Action Token
```

SpatioLM 和 GDA 构成 X 层的"双保险"：SpatioLM 侧重从 VLM 内部激发空间智能（纯视觉路线），GDA 侧重从外部教师安全注入 3D 几何（门控融合路线）。两者从不同角度解决同一个问题——如何让视觉表征"懂"三维空间——且都遵循"不破坏预训练对齐"的原则。

---

## 三、各工作技术详解

### 3.1 SpatioLM — 赋予视觉语言模型物理空间智能

**问题**：现有 VLM 在常识推理上表现出色，但对物理空间（距离估计、方位判断、相对位置）的理解极为薄弱。传统方案依赖外部 3D 先验（深度图、点云、相机参数），导致部署成本高、泛化差。

**核心方案**：SpatioLM 提出"冻结骨干 + 即插即用 Spatio-Vision Module"的范式：

1. **Spatio-Vision Module (SV-Module)**：由多个 Spatio-Vision Block 堆叠组成，从 Vision Encoder 的中间层提取几何感知 token，通过零初始化投影注入冻结的 LLM Block。
2. **Frame Attention + Global Attention 交替**：帧内注意力捕捉单帧细粒度几何，全局注意力聚合多视角/时序一致性。
3. **Dual DPT Head**：联合预测伪深度图和相机射线图，作为辅助监督信号，让注入的几何特征编码真实的 3D 先验。
4. **蒸馏损失**：用 Depth Anything V3 作为 teacher，通过 Gram 矩阵匹配对齐 token 结构。

**关键结果**：在 VSI-Bench 上首个突破 70 分（71.6），在 LIBERO 机器人操作 benchmark 上取得竞争性表现，且全程仅需 RGB 输入。

**在 XLA 中的角色**：SpatioLM 为 X 层提供空间增强的视觉表征。当这些具备几何理解能力的 token 输入到 L 层时，后续的推理和动作规划就拥有了物理世界的"尺度感"和"方位感"——这是具身智能的基础能力。

---

### 3.2 GDA (Gated Depth Attention) — 保真式深度融合

**问题**：将 3D 几何信息注入 VLM 已成为共识，但现有方法存在一个被忽视的失败模式——无条件注入几何信息（无论是拼接深度 token 还是对齐到几何教师表征）会**破坏预训练的视觉-语言对齐**，导致 VLM 在通用任务（VQA、OCR、幻觉检测）上性能下降。即：3D 监督越强，不代表 VLM 越好用。

**核心方案**：GDA 将几何信息视为 RGB token 之上的**条件残差**，而非替代或无条件融合：

1. **Latent Depth Branch**：轻量投影器将 Vision Encoder 特征映射为几何 token D̂，训练时由冻结的 DA3 教师蒸馏监督，推理时直接从 RGB 特征预测（不需要深度传感器）。
2. **Cross-Attention 检索**：以 RGB token 为 Query、几何 token 为 Key/Value 做交叉注意力，为每个视觉 token 检索相关的几何信息，产生残差 Δ。
3. **Image-conditioned Gate**：关键创新——一个基于图像内容的逐 token 门控 g ∈ [0,1]。当几何修正无益时（如纯文字识别场景），门控趋近于 0，RGB 路径原样保留；当深度信息有用时（如空间定位），门控打开，残差注入。
4. **零初始化启动**：门控网络最后一层初始化为零权重和负偏置（σ(−2)≈0.12），确保训练初期模型行为近似原始 RGB-only VLM，几何信息逐步"被信任"。

**关键设计洞察**：
- 门控在 image-text grounding **之前**进行路由（基于图像而非指令条件），保持 GDA 作为纯视觉适配器的角色，将指令条件的利用留给下游 LLM。
- Cross-attention 选择"检索什么几何信息"，gate 决定"是否使用"，两个机制互补分工。

**关键结果**：在 Spatial Grounding、通用 VLM、鲁棒性三类 benchmark 上，GDA 全面优于 alignment-based 和 concatenation-based 方法。在 VLA 下游任务（LIBERO、RoboTwin 2.0、真实机器人）上，GDA 同样提升操作成功率——证明 VLM 侧的对齐保真直接传导到具身控制质量。

**在 XLA 中的角色**：GDA 与 SpatioLM 互为补充，共同构成 X 层的空间感知方案。SpatioLM 从 VLM 内部层间特征激发隐含的空间结构（侧入注入 + DPT 监督），GDA 从外部 3D 教师安全地融入几何知识（残差门控 + 蒸馏）。两者共享一个核心原则：**增强空间能力的前提是不破坏已有的视觉-语言对齐**。GDA 的发现也为整个 XLA 体系提供了一个重要警示——3D 增强不是越多越好，选择性融合优于强制融合。

#### SpatioLM 与 GDA 的对比与互补

| 维度 | SpatioLM | GDA |
|------|----------|-----|
| 几何信息来源 | VLM 内部中间层特征 | 外部 3D 教师（DA3）蒸馏 |
| 注入方式 | 侧通道：SV-Block 平行于 LLM Block | 残差：门控调制 RGB token |
| 保真机制 | 零初始化投影 + 骨干全冻结 | Image-conditioned gate + 零初始化 |
| 推理依赖 | 仅 RGB | 仅 RGB（教师训练时用，推理时移除） |
| 适用场景 | 深度空间推理（距离/方位/路径规划） | 需在保真 VLM 能力的同时增强 grounding |
| 参数开销 | SV-Module（多 block 堆叠） | 轻量 cross-attention + 门控网络 |

---

### 3.3 OneVL — 一步潜空间推理与规划

**问题**：Chain-of-Thought (CoT) 是自动驾驶 VLA 性能提升的关键驱动力，但显式 CoT 的自回归生成带来了不可接受的延迟。现有 Latent CoT 方法（COCONUT、CODI、SIM-CoT）将推理压缩为纯语言潜向量，丢失了驾驶场景中的因果动力学信息，导致性能始终不如显式 CoT。

**核心方案**：OneVL 提出双模态辅助解码器监督的潜空间 CoT 框架：

1. **Latent Token 设计**：将潜空间 token 分为视觉潜变量（Z_v）和语言潜变量（Z_l）两类。
2. **Visual Auxiliary Decoder**：从视觉潜变量重建未来帧，相当于世界模型，迫使潜空间编码因果场景动力学（道路几何、智能体运动、环境变化）。
3. **Language Auxiliary Decoder**：从语言潜变量重建 CoT 文本，保持语义可解释性。
4. **三阶段训练**：Main Model Warmup → Auxiliary Decoder Warmup → Joint End-to-End Fine-tuning，渐进式对齐。
5. **Prefill 推理**：推理时丢弃辅助解码器，所有潜 token 作为 prompt 前缀一次性 prefill，达到 answer-only 的速度。

**关键结果**：在 NAVSIM、ROADWork、Impromptu、Alpamayo-R1 四个 benchmark 上，OneVL 是首个超越显式 CoT 的 Latent CoT 方法，同时延迟仅为显式 CoT 的 5.4%（实车部署场景）。

**在 XLA 中的角色**：OneVL 正是 XLA 架构 L 层中"Latent CoT"的技术实现。它证明了一个重要结论——压缩不是妥协，而是更有效推理的驱动力。当世界模型监督迫使潜空间编码因果结构而非符号摘要时，推理质量反而超越了冗长的显式推理链。这为 XLA 的"高效原生语言"理念提供了核心技术支撑。

---

### 3.4 LTA (Learning to Anchor) — 推进连续扩散语言模型

**问题**：XLA 架构中 Action Token 的生成需要高并行度（动作序列短但多维度需同时决策），自回归范式天然受限。Continuous Diffusion Language Model（如 Flow Map LM）具备并行生成的潜力，但现有方法存在两个根本缺陷：(1) 中间状态脱离概率单纯形（可能出现负值），(2) 训练-推理不一致（训练只见 noise→gold，推理却遇到错误 token 的状态）。

**核心方案**：LTA 提出两个关键改动：

1. **状态约束在概率单纯形上**：
   - 用 Dirichlet 分布替代 Gaussian 噪声，保证所有中间状态始终是合法概率分布（非负、和为 1）。
   - 输出层天然匹配 softmax：模型学的是"从一个概率分布到另一个概率分布"的映射，而非跨空间映射。
   - Euler 步永远合法：两个 simplex 上的点的凸组合仍在 simplex 上，不会越界。

2. **Wrong-token Semantic Corruption**：
   - 训练时以一定概率用错误 token 替换 gold endpoint，显式教会模型在"被误导的上下文"中恢复正确答案。
   - 解决了 FLM 训练只见 noise→gold、推理却遇到前一步预测错误状态的分布偏移问题。

**关键结果**：在 LM1B 上，LTA 以 128 步 NFE 达到 GenPPL 37.53（FLM 1024 步才 96.91），1024 步进一步降至 20.12，大幅超越所有 diffusion/flow LM 方法，且尚未训练完毕。

**在 XLA 中的角色**：LTA 是 XLA 架构中 Action Token 生成范式的底层技术探索。它的核心优势——高并行度、天然适配世界模型（同时预测多元素状态）、支持 test-time scaling（更多推理步数换更高质量）——与 XLA 的需求高度契合：

- **世界模型**：需要同时预测一帧中多个元素的位置和状态，元素间有依赖关系。AR 模型引入人为顺序偏置，而 LTA 的双向注意力让所有位置互相可见、同时更新。
- **VLA 动作生成**：动作序列短但每步多维度需同时决策（如机械臂的多关节协调），diffusion 的并行生成比逐 token 自回归更自然。
- **不确定性表达**：LTA 的 Dirichlet bridge 可多次采样给出不同未来轨迹，比 temperature sampling 更结构化。

---

### 3.5 NanoVLA — 边缘端通用机器人策略

**问题**：大规模 VLA 模型（如 OpenVLA 7B）在数据中心 GPU 上表现优异，但无法部署到 Jetson Orin Nano 等边缘设备。简单压缩参数量并不能解决推理架构本身的低效问题。

**核心方案**：NanoVLA 从推理架构层面重新设计，围绕三个互补思想：

1. **Vision-Language Decoupling with Caching**：
   - 将视觉和语言编码器独立处理，延迟到最后阶段才通过轻量 cross-attention 融合。
   - 指令（instruction）特征只需编码一次即可跨时间步复用，仅视觉特征逐帧更新。
   - 消除了传统 VLA 中重复的跨模态计算。

2. **Long-Short Action Chunking (LSAC)**：
   - 训练时在长时间窗上优化，学习时序依赖和平滑性；
   - 推理时只执行预测 chunk 的前几步，然后用最新观测重新规划。
   - 兼顾长程连贯性与实时响应性。

3. **Dynamic Routing**：
   - 引入贝叶斯成功率建模的路由器，简单任务走轻量骨干，复杂任务升级到大骨干。
   - 基于不确定性感知的成对胜率概率进行模型选择，保持低平均延迟的同时不丢复杂任务性能。

**关键结果**：仅用 OpenVLA 2% 的参数（约 140M），在标准 benchmark 和真实机器人上达到或超越 SOTA 性能，推理速度提升 52 倍。

**在 XLA 中的角色**：NanoVLA 是 XLA 架构中 A 层"行为专家（Expert）"落地的关键技术。它解决了从实验室到产品的最后一公里——如何在手机/机器人/车载等资源受限设备上，以实时帧率运行完整的"感知→推理→动作"流水线。其 Dynamic Routing 机制也呼应了 XLA 架构中对"可控"的追求：按需分配计算，简单动作快速响应，复杂场景调动更多能力。

---

## 四、整体技术叙事

将五项工作放回 XLA 的全局视角，一条清晰的技术主线浮现：

**从感知到行动的全链路效率革命。**

1. **感知侧（SpatioLM + GDA）**：两条互补路线解决同一核心问题——如何让 VLM "看懂"三维世界。SpatioLM 从 VLM 内部中间层激发隐含的空间结构，走的是"内生激发"路线；GDA 从外部 3D 教师安全注入几何知识，走的是"外部蒸馏"路线。两者共享"不破坏预训练对齐"的核心原则，GDA 的门控机制更进一步揭示了一条设计准则——3D 增强应该是选择性的精细化，而非无差别的强制替换。

2. **推理侧（OneVL + LTA）**：两条互补的技术路线共同服务于 XLA 的 L 层。
   - OneVL 解决"思考"的效率：将冗长的语言推理链压缩为紧凑的潜空间表征，通过世界模型监督保留因果结构，实现一步推理。
   - LTA 解决"表达"的范式：突破自回归逐 token 的串行瓶颈，在概率单纯形上做并行扩散生成，为 Action Token 和世界模型提供更原生的生成机制。

3. **执行侧（NanoVLA）**：在架构层面重新思考模态融合、动作执行和计算分配，让完整的 VLA 能力在边缘端实时运行——让模型"做得出"精准动作。

这五项工作共同勾勒了 XLA 的技术蓝图：一个能够接收多模态输入、在高效的原生语言空间中进行因果推理、最终输出拟人化行为的认知大模型——部署在从数据中心到手持设备的全谱系硬件上。

---

## 五、未来展望

基于当前各工作的推进方向，XLA 架构的后续演进可能聚焦于：

1. **SpatioLM + GDA 的统一感知前端**：SpatioLM 的 SV-Module 侧入注入和 GDA 的门控残差融合可以级联使用——SV-Module 在 LLM 层间注入粗粒度空间结构，GDA 在 Vision Encoder 输出端做细粒度几何修正，形成"粗调 + 精修"的双层空间增强。

2. **GDA + OneVL 的端到端驾驶流水线**：GDA 增强后的空间感知 token 输入 OneVL 的潜空间推理，让 Latent CoT 编码的因果结构既包含语义信息又包含精确的 3D 空间信息。

3. **LTA 作为统一的 Action/World Model 生成引擎**：利用 LTA 的并行生成和 test-time scaling 特性，统一 Action Token 生成和世界模型预测——两者都需要同时决策多个相互依赖的元素。

4. **NanoVLA 的 LTA 化**：将 NanoVLA 的 action chunking 机制替换为 LTA 的扩散生成，在边缘端实现"简单动作 32 步快速解码、复杂任务 512 步精细推理"的 compute 动态分配。

5. **闭环训练**：将 OneVL 的世界模型预测与真实环境反馈对齐，通过 RL 和奖励模型持续校准，实现 XLA 架构图中的"闭环训练"目标。

---

*文档更新日期：2026-05-15*
*基于 XLA 架构概念图及相关论文：SpatioLM, GDA, OneVL, LTA, NanoVLA*
