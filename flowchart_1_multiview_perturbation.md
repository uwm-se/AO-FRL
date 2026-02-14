# Flowchart 1: Multi-View Augmentation + Representation Perturbation

## 概述

这张流程图展示了客户端如何将原始数据转换为隐私保护的嵌入表示，包括：
- **Multi-View Augmentation**: 生成多个增强视图
- **Representation Perturbation**: 三重隐私保护机制

---

## Mermaid 流程图

```mermaid
flowchart TB
    subgraph Input["输入层"]
        RAW[原始图像 x]
        LABEL[标签 y]
    end

    subgraph Encoder["特征提取"]
        FROZEN[冻结的 ResNet-18<br/>pretrained on ImageNet]
        EMB[原始嵌入 z₀<br/>512-dim vector]
    end

    subgraph MultiView["Multi-View Augmentation"]
        direction TB
        VIEWS[生成 n_views=2 个视图]

        subgraph View1["View 1"]
            AUG1[数据增强<br/>RandomCrop + Flip]
            ENCODE1[ResNet-18 编码]
            Z1[z₁ = encoder(aug₁(x))]
            NOISE1[添加视图噪声<br/>z₁ + N(0, 0.01²×1)]
        end

        subgraph View2["View 2"]
            AUG2[数据增强<br/>RandomCrop + Flip]
            ENCODE2[ResNet-18 编码]
            Z2[z₂ = encoder(aug₂(x))]
            NOISE2[添加视图噪声<br/>z₂ + N(0, 0.01²×2)]
        end

        CONCAT[合并所有视图<br/>embeddings = [z₁, z₂, ...]]
    end

    subgraph Perturbation["Representation Perturbation (三重隐私保护)"]
        direction TB

        subgraph Stage1["Stage 1: L2 Clipping"]
            CLIP_CHECK{||z|| > C?}
            CLIP_ACTION[z ← z × C / ||z||]
            CLIP_PASS[保持不变]
        end

        subgraph Stage2["Stage 2: Gaussian Noise"]
            NOISE_ADD[z_tilde ← z + N(0, σ²I)<br/>σ = 0.02]
            NOISE_NOTE[差分隐私保证]
        end

        subgraph Stage3["Stage 3: Privacy Gate"]
            PROTO_CALC[计算类别原型<br/>proto_c = mean(embs | y=c)]
            SIM_CALC[计算相似度<br/>sim = cos(z_tilde, proto_c)]
            GATE_CHECK{sim > threshold?}
            REJECT[拒绝上传<br/>隐私风险高]
            ACCEPT[接受上传<br/>隐私安全]
        end
    end

    subgraph Sampling["采样与上传"]
        SAMPLE[随机采样至 budget<br/>初始: 500 个嵌入]
        UPLOAD[上传到 Server]
        SUMMARY[生成 summary:<br/>- label_histogram<br/>- reject_ratio<br/>- n_uploaded]
    end

    subgraph Output["输出"]
        GATED[通过门控的嵌入<br/>Gated Embeddings]
        META[元数据 Summary]
    end

    %% 连接流程
    RAW --> FROZEN
    FROZEN --> EMB

    EMB --> VIEWS
    VIEWS --> View1
    VIEWS --> View2

    View1 --> AUG1 --> ENCODE1 --> Z1 --> NOISE1
    View2 --> AUG2 --> ENCODE2 --> Z2 --> NOISE2

    NOISE1 --> CONCAT
    NOISE2 --> CONCAT

    CONCAT --> Stage1

    Stage1 --> CLIP_CHECK
    CLIP_CHECK -->|Yes| CLIP_ACTION --> Stage2
    CLIP_CHECK -->|No| CLIP_PASS --> Stage2

    Stage2 --> NOISE_ADD --> NOISE_NOTE --> Stage3

    Stage3 --> PROTO_CALC --> SIM_CALC --> GATE_CHECK
    GATE_CHECK -->|Yes| REJECT
    GATE_CHECK -->|No| ACCEPT --> SAMPLE

    SAMPLE --> UPLOAD
    SAMPLE --> SUMMARY

    UPLOAD --> GATED
    SUMMARY --> META

    %% 样式定义
    style RAW fill:#ffe6e6
    style FROZEN fill:#e6f3ff
    style MultiView fill:#fff9e6
    style Perturbation fill:#e6ffe6
    style REJECT fill:#ffcccc
    style ACCEPT fill:#ccffcc
    style GATED fill:#cce6ff
    style META fill:#f0e6ff
```

---

## 详细步骤说明

### 1️⃣ Multi-View Augmentation (多视图增强)

#### 目的
- 增加数据多样性
- 提高模型泛化能力
- 为每个样本生成多个表示

#### 过程

**Step 1: 生成多个视图**
```python
n_views = 2  # 默认生成2个视图
for view_idx in range(n_views):
    # 对同一张图像应用不同的数据增强
    augmented_image = augment(original_image)
```

**Step 2: 数据增强操作**
- RandomCrop(32, padding=4)
- RandomHorizontalFlip(p=0.5)
- (可选) ColorJitter, Rotation 等

**Step 3: 编码为嵌入**
```python
embedding = resnet18_frozen(augmented_image)
# 输出: 512维向量
```

**Step 4: 添加视图噪声**
```python
view_noise = torch.randn_like(embedding) * 0.01 * view_idx
embedding_with_noise = embedding + view_noise
```

**为什么添加视图噪声？**
- 让不同视图的嵌入有轻微差异
- view_idx 越大，噪声越大
- 有助于模型学习更鲁棒的表示

**Step 5: 合并所有视图**
```python
all_embeddings = torch.cat([view1_embs, view2_embs], dim=0)
# 如果原始样本 N=1000，n_views=2
# 则 all_embeddings.shape = (2000, 512)
```

---

### 2️⃣ Representation Perturbation (表示扰动)

三重隐私保护机制，确保上传的嵌入不会泄露敏感信息。

---

#### Stage 1: L2 Clipping (L2裁剪)

**目的**: 限制嵌入的最大范数，防止极端值泄露信息

**数学公式**:
```
如果 ||z|| > C:
    z ← z × (C / ||z||)
否则:
    z 保持不变
```

**参数**:
- C = 1.0 (裁剪阈值)

**代码**:
```python
norms = torch.norm(embeddings, dim=1, keepdim=True)
clip_mask = norms > C
embeddings[clip_mask] = embeddings[clip_mask] * (C / norms[clip_mask])
```

**效果**:
- 保证 ||z|| ≤ 1.0
- 为后续差分隐私提供敏感度保证

---

#### Stage 2: Gaussian Noise (高斯噪声)

**目的**: 添加随机噪声，实现差分隐私

**数学公式**:
```
z_tilde = z + N(0, σ²I)
```

**参数**:
- σ = 0.02 (噪声标准差)
- 自适应调整: high_risk_hook 触发时 σ *= 1.5

**代码**:
```python
noise = torch.randn_like(embeddings) * sigma
embeddings_noisy = embeddings + noise
```

**隐私保证**:
- 结合 L2 Clipping，提供 (ε, δ)-差分隐私
- 较大的 σ → 更强的隐私保护，但准确率下降
- 较小的 σ → 较弱的隐私保护，但准确率提高

---

#### Stage 3: Privacy Gate (隐私门控)

**目的**: 拒绝与类别原型过于相似的嵌入，防止成员推断攻击

**原理**:
如果一个嵌入与某个类别的原型非常相似，说明它很"典型"，容易被攻击者识别。

**详细步骤**:

**3.1 计算类别原型**
```python
prototypes = {}
for c in range(n_classes):
    class_embeddings = embeddings[labels == c]
    prototypes[c] = class_embeddings.mean(dim=0)
```

**3.2 计算余弦相似度**
```python
for i, emb in enumerate(embeddings):
    label = labels[i]
    sim = cosine_similarity(emb, prototypes[label])
```

**3.3 动态阈值计算**
```python
# 方法1: 百分位数阈值
tau = np.percentile(similarities, (1 - tau_percentile) * 100)
tau = max(tau, tau_min)  # 保证最低阈值

# 参数:
# - tau_percentile = 0.15 (拒绝最相似的15%)
# - tau_min = 0.5 (最低阈值)
```

**3.4 门控决策**
```python
if sim > tau:
    reject_embedding()  # 隐私风险高
else:
    accept_embedding()  # 可以上传
```

**拒绝率统计**:
```python
reject_ratio = n_rejected / n_total
# 实验中: reject_ratio ≈ 0.16 (16%)
```

**为什么不直接用固定阈值？**
- 固定阈值无法适应数据分布变化
- 动态阈值根据当前轮次的相似度分布自动调整
- 保证拒绝率相对稳定

---

### 3️⃣ 采样与上传

**Step 1: 随机采样**
```python
accepted_indices = [i for i, sim in enumerate(similarities) if sim <= tau]
n_accepted = len(accepted_indices)

if n_accepted > upload_budget:
    # 随机采样至 budget
    sampled_indices = random.sample(accepted_indices, upload_budget)
else:
    # 全部上传
    sampled_indices = accepted_indices
```

**Step 2: 生成 Summary**
```python
summary = {
    "label_histogram": compute_histogram(sampled_labels),
    "reject_ratio": n_rejected / n_total,
    "sigma": current_sigma,
    "n_uploaded": len(sampled_indices)
}
```

**Step 3: 上传**
- Gated Embeddings → Server (via A2A Bus)
- Summary → Server (用于 Orchestration)

---

## 参数总结

| 参数 | 默认值 | 说明 | 调整时机 |
|------|--------|------|----------|
| **Multi-View** | | | |
| n_views | 2 | 视图数量 | 固定 |
| view_noise_scale | 0.01 | 视图噪声强度 | 固定 |
| **L2 Clipping** | | | |
| C | 1.0 | 裁剪阈值 | 固定 |
| **Gaussian Noise** | | | |
| σ | 0.02 | 噪声标准差 | high_risk_hook: ×1.5 |
| **Privacy Gate** | | | |
| tau_percentile | 0.15 | 拒绝百分位 | 固定 |
| tau_min | 0.5 | 最低阈值 | 固定 |
| **Upload Budget** | | | |
| base_budget | 500 | 基础上传额度 | orchestrate: ×(1+rarity_score) |
| | | | drift_hook: ×1.3 |
| | | | high_risk_hook: ÷2 |

---

## 自适应调整机制

### Hook 1: Low-Data Hook
**触发条件**: 任何类别样本 < 10
```python
if any(class_counts < low_data_k):
    augmentation_mode = "conservative"
```

**效果**:
- 使用更保守的数据增强
- 防止稀有类过拟合

### Hook 2: High-Risk Hook
**触发条件**: reject_ratio > 0.30
```python
if reject_ratio > 0.30:
    sigma *= 1.5
    upload_budget //= 2
    augmentation_mode = "conservative"
```

**效果**:
- 增加噪声强度
- 减少上传量
- 降低隐私风险

### Hook 3: Drift Hook
**触发条件**: 验证准确率连续下降
```python
if val_acc[t] < val_acc[t-1] < val_acc[t-2]:
    upload_budget *= 1.3
```

**效果**:
- 增加上传预算
- 提供更多数据
- 帮助模型恢复

---

## 实验结果

### 拒绝率统计
```
reject_ratio = 0.16 (16%)
目标: 0.15 (15%)
状态: ✅ 接近目标
```

### 上传量统计
```
平均上传量: ~950 个嵌入/轮/客户端
初始 budget: 500
实际 budget: 937~979 (根据稀缺性调整)
```

### 隐私保护效果
```
✅ L2 Clipping: 所有嵌入 ||z|| ≤ 1.0
✅ Gaussian Noise: σ = 0.02 (未触发 high_risk_hook)
✅ Privacy Gate: 拒绝 16% 高风险嵌入
```

### 准确率表现
```
AO-FRL: 61.91%
FedAvg: 51.45%
提升: +10.46%
```

---

## 绘图建议

### 推荐工具
1. **Mermaid Live**: https://mermaid.live/
   - 复制上面的 Mermaid 代码
   - 实时预览和导出

2. **Draw.io**: https://app.diagrams.net/
   - 手动绘制更精美的版本
   - 自定义颜色和布局

3. **PPT / Keynote**
   - 适合答辩演示
   - 可以添加动画效果

### 关键点高亮

在绘图时，建议突出以下几点：

1. **Multi-View 的并行性**
   - 用并行的框图表示多个视图同时生成

2. **三重隐私保护的顺序性**
   - 用清晰的箭头表示 Clipping → Noise → Gate

3. **Privacy Gate 的决策点**
   - 用菱形表示判断节点
   - 清晰标注接受/拒绝分支

4. **参数的自适应调整**
   - 用不同颜色标注会动态调整的参数
   - 如 σ, budget

---

## 数学符号说明

| 符号 | 含义 |
|------|------|
| x | 原始图像 |
| y | 标签 |
| z | 嵌入向量 |
| z_tilde | 添加噪声后的嵌入 |
| σ | 高斯噪声标准差 |
| C | L2 裁剪阈值 |
| τ (tau) | Privacy Gate 阈值 |
| proto_c | 类别 c 的原型向量 |
| sim | 余弦相似度 |
| N(0, σ²I) | 高斯分布 |
| ||·|| | L2 范数 |

---

## 总结

这张流程图展示了 AO-FRL 的核心创新：

✅ **Multi-View Augmentation**: 增加数据多样性，提高泛化
✅ **L2 Clipping**: 限制敏感度，为 DP 提供保证
✅ **Gaussian Noise**: 实现差分隐私保护
✅ **Privacy Gate**: 拒绝高风险嵌入，防止成员推断
✅ **自适应调整**: 三个 Hooks 动态优化参数

**隐私与准确率的平衡**: 在保护隐私的前提下，达到比 FedAvg 高 10.46% 的准确率！
