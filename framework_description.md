# AO-FRL 框架完整陈述

## 系统概述

**AO-FRL (Agent-Orchestrated Federated Representation Learning)** 是一个基于多智能体架构的隐私保护联邦学习框架。系统由 22 个自主 Agent 协同工作，通过 A2A (Agent-to-Agent) 通信协议完成分布式模型训练。

## 核心组件

### 1. 数据与模型层

#### 1.1 数据集
- **CIFAR-100**: 50,000 训练样本 + 10,000 测试样本
- **Non-IID 划分**: 使用 Dirichlet(α=0.3) 分布将训练数据分配给 20 个客户端
- **训练/验证分割**: 每个客户端内部 90% 训练，10% 验证

#### 1.2 模型架构
- **编码器**: ResNet-18 (ImageNet 预训练，冻结参数)
  - 输入: 3×32×32 图像
  - 输出: 512 维特征嵌入
- **分类头**: MLPHead (可训练)
  - 结构: Linear(512→256) → BatchNorm → ReLU → Dropout(0.2) → Linear(256→100)
  - 参数量: ~157K

### 2. Agent 层

#### 2.1 ServerAgent (协调者)
**职责**:
- 收集客户端上传的嵌入和状态摘要
- 分析全局类别分布，识别稀缺类别
- 动态生成每个客户端的个性化指令
- 在收集的嵌入上训练全局分类头
- 管理 replay buffer (历史嵌入缓存)

**核心算法**:
- `orchestrate()`: 计算 label_gap，生成自适应 budget/sigma/augmentation 指令
- `train_head()`: 在嵌入 + replay buffer 上训练 MLPHead (3 epochs/round)
- `fedavg_aggregate()`: FedAvg 基线的参数聚合

**数据结构**:
- `self.head`: 提出方法的全局分类头
- `self.fedavg_head`: FedAvg 基线的全局分类头
- `self.replay_buffer`: 历史嵌入缓存 (最多 50,000)

#### 2.2 ClientAgent × 20 (执行者)
**职责**:
- 提取本地数据的特征嵌入（使用冻结的 ResNet-18）
- 应用三重隐私保护: L2 clipping → Gaussian noise → Privacy Gate
- 生成多视图嵌入 (n_views=2) 增强数据多样性
- 计算类别原型 (class prototypes) 用于 Privacy Gate 判断
- 上报状态摘要 (label_histogram, reject_ratio, sigma 等)
- 接收并执行服务器指令

**核心算法**:
- `extract_gated_embeddings()`:
  1. 计算类别原型 (每类嵌入的 L2 归一化平均)
  2. 生成多视图嵌入 (原始 + 小噪声扰动)
  3. L2 clipping (||z|| ≤ C=1.0)
  4. 添加高斯噪声 (σ=0.02)
  5. Privacy Gate: 拒绝与原型相似度 > threshold 的样本
  6. 采样至 upload_budget 上限
- `apply_server_instructions()`: 更新 budget, sigma, augmentation_mode
- `_apply_hooks()`: 执行三个自适应钩子

**三个自适应钩子**:
1. **low_data_hook**: 若任何类别样本 < k (=10) → 切换到保守增强
2. **high_risk_hook**: 若 reject_ratio > r (=0.30) → sigma×1.5, budget÷2
3. **drift_hook**: 若验证准确率连续下降 → budget×1.3

**数据结构**:
- `self.encoder`: 冻结的 ResNet-18 (共享)
- `self.train_embs/labels`: 预计算的训练嵌入缓存
- `self.upload_budget`: 当前上传额度 (初始 500)
- `self.sigma`: 当前噪声水平 (初始 0.02, 最大 0.5)
- `self.augmentation_mode`: "normal" 或 "conservative"

#### 2.3 EvaluatorAgent (监控者)
**职责**:
- 在全局测试集上评估模型性能
- 记录每轮的 accuracy, macro-F1, 通信成本
- 保存最佳模型和最终结果

**数据结构**:
- `self.test_embs`: 预计算的测试集嵌入 (10,000×512)
- `self.results`: 每轮结果记录

### 3. 通信层

#### 3.1 A2A Bus (消息总线)
**职责**:
- 注册所有 Agent 身份卡片 (AgentCard)
- 路由 Agent 间的任务消息
- 记录完整的通信审计日志

**核心方法**:
- `register_agent()`: 注册 Agent
- `send_task()`: 创建并发送任务
- `complete_task()`: 标记任务完成，记录 artifacts
- `save_log()`: 保存 JSON 格式的审计日志

**协议组件**:
- **AgentCard**: Agent 身份 (id, name, skills)
- **Task**: 任务实例 (task_id, sender, receiver, state)
- **Message**: 消息体 (Parts: text/json/data)
- **Artifact**: 数据产物 (embeddings, parameters, size_bytes)

### 4. 隐私保护层

#### 4.1 差分隐私机制
- **L2 Clipping**: 限制嵌入的 L2 范数 ≤ C (=1.0)
- **高斯噪声**: 添加 N(0, σ²I) 噪声，σ 自适应调整 (0.02~0.5)

#### 4.2 Privacy Gate
**原理**: 拒绝与类别原型过于相似的样本，防止成员推理攻击

**计算流程**:
1. 计算类别原型: `proto[c] = normalize(mean(embeddings[class==c]))`
2. 计算相似度: `sim = cosine_similarity(noised_embedding, proto[c])`
3. 自适应阈值: `threshold = max(percentile(sims, 85%), tau_min=0.5)`
4. 决策: 若 `sim > threshold` → 拒绝，否则接受

**关键参数**:
- `tau_percentile=0.15`: 拒绝最相似的 15% 样本
- `tau_min=0.5`: 最低阈值保护

## 完整训练流程

### 初始化阶段

1. **数据准备**
   - 加载 CIFAR-100 数据集
   - Dirichlet 分割到 20 个客户端 (α=0.3)
   - 每个客户端内部 90/10 分割训练/验证集

2. **模型准备**
   - 构建冻结的 ResNet-18 编码器
   - **预计算所有嵌入** (一次性操作):
     - 训练集: 50,000 → 50,000×512 嵌入矩阵
     - 测试集: 10,000 → 10,000×512 嵌入矩阵
   - 初始化 MLPHead (两个: proposed 和 fedavg)

3. **Agent 注册**
   - 注册 ServerAgent, EvaluatorAgent
   - 注册 20 个 ClientAgent
   - 所有 Agent 注册到 A2A Bus

### 每轮训练 (Round R = 1~100)

#### Phase 1: 客户端数据收集

**A2A 消息**: Server → Client_i: `"extract_embeddings"`

对于每个 ClientAgent (i=0~19):

1. **提取本地嵌入**
   - 从预计算缓存中读取训练嵌入
   - 根据 `augmentation_mode` 生成增强嵌入

2. **计算类别原型**
   ```
   for each class c in local_data:
       proto[c] = normalize(mean(embeddings[class==c]))
   ```

3. **生成多视图嵌入**
   ```
   for view in [1, 2]:
       view_noise = randn(N, 512) * 0.01 * view
       z_view = embeddings + view_noise
       z_view = clip(z_view, C=1.0)
       z_tilde = z_view + randn(N, 512) * sigma
       candidates.append(z_tilde)
   ```

4. **Privacy Gate 过滤**
   ```
   for each candidate (z_tilde, label):
       sim = cosine_similarity(z_tilde, proto[label])
       if sim > threshold[label]:
           reject
       else:
           accept
   ```

5. **采样至 Budget**
   - 从接受的嵌入中随机采样 `upload_budget` 个

6. **生成状态摘要**
   ```
   summary = {
       "client_id": i,
       "label_histogram": [上传的各类别数量],
       "reject_ratio": 拒绝率,
       "sigma": 当前噪声水平,
       "n_uploaded": 实际上传数量,
       "augmentation_mode": 当前增强模式
   }
   ```

7. **上传数据**
   - **A2A 消息**: Client_i → Server:
     - Artifacts: `gated_embeddings` (size_bytes)
     - Response: `summary` (JSON)

#### Phase 2: 服务器 Orchestration

**输入**: 20 个客户端的 summaries

1. **计算全局类别分布**
   ```
   global_hist = sum(summary["label_histogram"] for all clients)
   # 示例: global_hist[c] = 本轮第 c 类收到的嵌入总数
   ```

2. **计算目标分布**
   ```
   total = sum(global_hist)
   target_per_class = total / 100  # 均匀分布目标
   ```

3. **识别稀缺类别**
   ```
   label_gap = max(target_per_class - global_hist, 0)
   # label_gap[c] > 0 表示第 c 类不足

   label_gap_normalized = label_gap / sum(label_gap)
   # 归一化为权重分布
   ```

4. **为每个客户端计算稀缺性得分**
   ```
   for each client i:
       client_classes = [该客户端上传的类别列表]
       rarity_score = sum(label_gap_normalized[c] for c in client_classes)
       # 持有稀缺类别的客户端得分高
   ```

5. **生成个性化指令**
   ```
   for each client i:
       # 基础 budget 调整
       budget = base_budget * (1 + rarity_score)

       # High-risk hook
       if summary["reject_ratio"] > 0.30:
           sigma = min(summary["sigma"] * 1.5, 0.5)
           budget = max(budget // 2, 50)
           augmentation = "conservative"

       # Low-data hook
       if any(class_count < 10):
           budget = budget * 1.2
           augmentation = "conservative"

       instructions[i] = {
           "client_id": i,
           "upload_budget": budget,
           "sigma": sigma,
           "augmentation_mode": augmentation
       }
   ```

6. **广播指令**
   - **A2A 消息**: Server → Client_i: `"apply_instructions"`

#### Phase 3: 服务器模型训练

1. **合并嵌入**
   ```
   merged_embs = concat(all_client_embeddings)
   merged_labels = concat(all_client_labels)
   # 本轮收到约 10,000~20,000 个嵌入
   ```

2. **更新 Replay Buffer**
   ```
   replay_buffer.add(merged_embs, merged_labels)
   replay_buffer.apply_decay(factor=0.995)
   # 保持最多 50,000 个历史嵌入
   ```

3. **训练全局头**
   ```
   for epoch in [1, 2, 3]:
       # 从 merged + replay 中采样 mini-batch
       batch = sample(merged_embs + replay_buffer)

       # 标准监督学习
       logits = head(batch_embs)
       loss = CrossEntropyLoss(logits, batch_labels)
       optimizer.step()

   # 学习率衰减
   lr = max(lr * 0.98, lr_min=1e-4)
   ```

4. **广播更新后的头**
   - **A2A 消息**: Server → All Clients: 新的 head 参数

#### Phase 4: 全局评估

**A2A 消息**: Server → Evaluator: `"evaluate"`

1. **测试集推理**
   ```
   for batch in test_embeddings:
       logits = head(batch)
       predictions = argmax(logits)
   ```

2. **计算指标**
   - Accuracy: 整体准确率
   - Macro-F1: 各类别 F1 分数的平均

3. **记录结果**
   ```
   result = {
       "round": R,
       "accuracy": acc,
       "macro_f1": f1,
       "round_comm_bytes": 本轮通信量,
       "cumulative_comm_bytes": 累计通信量
   }
   ```

**A2A 消息**: Evaluator → Server: `result`

#### Phase 5: 客户端本地评估

每个 ClientAgent:
1. 使用更新后的 head 在本地验证集上评估
2. 记录 validation accuracy (用于 drift_hook)

### 终止条件

- 达到预设轮数 (rounds=100)
- 或提前停止条件触发

### 输出结果

1. **性能指标**
   - `results/Centralized.csv`: 集中式基线结果
   - `results/FedAvg.csv`: FedAvg 基线结果
   - `results/AO-FRL.csv`: 提出方法结果
   - `results/final_summary.json`: 最终汇总

2. **可视化**
   - `results/acc_vs_rounds.png`: 准确率曲线对比
   - `results/comm_vs_acc.png`: 通信效率对比

3. **审计日志**
   - `results/a2a_communication.json`: 完整的 A2A 通信记录

## 关键创新点

1. **Agent-Orchestrated 架构**
   - 中心化决策 + 分布式执行
   - 自适应资源分配
   - 标准化 A2A 通信协议

2. **三重隐私保护**
   - L2 Clipping (有界敏感度)
   - Gaussian Noise (差分隐私)
   - Privacy Gate (经验隐私增强)

3. **动态自适应机制**
   - 稀缺类别识别与补偿
   - 三个自适应钩子
   - 学习率和权重衰减

4. **高效表示学习**
   - 冻结编码器 + 预计算嵌入
   - 只传输嵌入，不传输参数
   - Replay buffer 平滑训练

## 对比基线

### FedAvg
- 客户端训练 MLP (3 local_epochs)
- 服务器参数聚合 (加权平均)
- 固定通信量 (25.2 MB/round)
- 无隐私保护

### Centralized (上界)
- 在全部数据上训练 (50 epochs)
- 无通信成本
- 无隐私问题

### AO-FRL (提出方法)
- 客户端只提取嵌入
- 服务器集中训练 (3 epochs)
- 可变通信量 (取决于 privacy gate)
- 三重隐私保护

## 参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rounds` | 100 | 通信轮次 |
| `n_clients` | 20 | 客户端数量 |
| `alpha` | 0.3 | Dirichlet 浓度 (non-IID 程度) |
| `upload_budget` | 500 | 初始上传额度 |
| `sigma` | 0.02 | 初始噪声水平 |
| `clip_C` | 1.0 | L2 clipping 上界 |
| `tau_percentile` | 0.15 | Privacy gate 拒绝比例 |
| `tau_min` | 0.5 | 最低相似度阈值 |
| `local_epochs` | 3 | FedAvg 本地训练轮数 |
| `server_train_epochs` | 3 | AO-FRL 服务器训练轮数 |
| `server_lr` | 0.001 | 服务器学习率 |
| `server_lr_decay` | 0.98 | 学习率衰减因子 |

## 实验结果 (CIFAR-100, 100 rounds)

| 方法 | Best Acc | Final Acc | Comm (GB) |
|------|----------|-----------|-----------|
| Centralized | 67.39% | 65.54% | 0 |
| FedAvg | 51.45% | 51.45% | 2.52 |
| AO-FRL | 63.12% | 61.91% | 5.22 |

**关键发现**:
- AO-FRL 比 FedAvg 高 9.67 个百分点
- 接近 Centralized 性能 (差距 3.63 个百分点)
- 通信量增加 2倍，但性能提升显著
