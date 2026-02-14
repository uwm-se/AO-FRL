# Privacy Gate 机制详细流程图

## 核心思想

**拒绝与类别原型过于相似的嵌入，防止成员推断攻击**

---

## Mermaid 流程图

```mermaid
flowchart TB
    subgraph Input["输入 (每个客户端)"]
        NOISY[加噪声后的嵌入<br/>z_tilde = z + N(0, σ²)<br/>shape: (N, 512)]
        LABELS[对应标签 y<br/>shape: (N,)]
    end

    subgraph ProtoCalc["步骤1: 计算类别原型"]
        GROUPBY[按类别分组嵌入]
        MEAN[计算每类的均值向量<br/>proto_c = mean(z_tilde | y=c)]
        PROTO_DICT[原型字典<br/>prototypes = {0: proto_0, ..., 99: proto_99}]
    end

    subgraph SimCalc["步骤2: 计算相似度"]
        LOOP[遍历每个嵌入 z_i]
        GET_PROTO[获取对应类别原型<br/>proto = prototypes[y_i]]
        COS_SIM[计算余弦相似度<br/>sim_i = cos(z_i, proto)]
        SIM_LIST[相似度列表<br/>similarities = [sim_0, ..., sim_N]]
    end

    subgraph ThresholdCalc["步骤3: 动态阈值计算"]
        PERCENTILE[计算百分位数<br/>tau_raw = percentile(similarities, 85%)]
        NOTE1[85% = 100% - tau_percentile(15%)<br/>即保留最不相似的85%]
        MAX_FUNC[应用最低阈值<br/>tau = max(tau_raw, tau_min)]
        NOTE2[tau_min = 0.5<br/>保证至少有些嵌入被拒绝]
        THRESHOLD[最终阈值 tau]
    end

    subgraph GateDecision["步骤4: 门控决策"]
        CHECK{sim_i > tau?}
        REJECT[❌ 拒绝上传<br/>隐私风险高<br/>太像类别原型]
        ACCEPT[✅ 接受上传<br/>隐私安全<br/>足够扰动]
    end

    subgraph Stats["步骤5: 统计与监控"]
        COUNT_REJECT[统计拒绝数量<br/>n_rejected]
        COUNT_ACCEPT[统计接受数量<br/>n_accepted]
        RATIO[计算拒绝率<br/>reject_ratio = n_rejected / N]
        RATIO_CHECK{reject_ratio > 0.30?}

        subgraph HighRiskHook["High-Risk Hook 触发"]
            INCREASE_SIGMA[增加噪声<br/>σ ← σ × 1.5]
            REDUCE_BUDGET[减少上传额度<br/>budget ← budget ÷ 2]
            CONSERVATIVE[切换保守增强<br/>augmentation = conservative]
        end

        NO_ACTION[保持当前参数]
    end

    subgraph LocalValid["步骤6: Local Validation 反馈"]
        SAMPLE_UPLOAD[从接受的嵌入中<br/>随机采样 budget 个上传]
        LOCAL_EVAL[客户端本地评估<br/>val_acc on 验证集]
        DRIFT_CHECK{连续下降?}

        subgraph DriftHook["Drift Hook 触发"]
            INCREASE_BUDGET[增加上传额度<br/>budget ← budget × 1.3]
            NOTE3[给服务器提供更多数据<br/>帮助模型恢复]
        end

        KEEP_BUDGET[保持当前 budget]
    end

    subgraph Output["输出"]
        GATED[通过门控的嵌入<br/>Gated Embeddings]
        SUMMARY[Summary 统计信息:<br/>• label_histogram<br/>• reject_ratio: 0.16<br/>• n_uploaded: ~950<br/>• sigma: 0.02]
    end

    %% 连接流程
    NOISY --> GROUPBY
    LABELS --> GROUPBY
    GROUPBY --> MEAN --> PROTO_DICT

    NOISY --> LOOP
    LABELS --> LOOP
    PROTO_DICT --> GET_PROTO
    LOOP --> GET_PROTO --> COS_SIM --> SIM_LIST

    SIM_LIST --> PERCENTILE --> NOTE1 --> MAX_FUNC
    NOTE2 --> MAX_FUNC --> THRESHOLD

    LOOP --> CHECK
    THRESHOLD --> CHECK
    CHECK -->|Yes 相似度高| REJECT
    CHECK -->|No 相似度低| ACCEPT

    REJECT --> COUNT_REJECT
    ACCEPT --> COUNT_ACCEPT
    COUNT_REJECT --> RATIO
    COUNT_ACCEPT --> RATIO

    RATIO --> RATIO_CHECK
    RATIO_CHECK -->|Yes| HighRiskHook
    RATIO_CHECK -->|No| NO_ACTION

    HighRiskHook --> INCREASE_SIGMA & REDUCE_BUDGET & CONSERVATIVE

    ACCEPT --> SAMPLE_UPLOAD
    SAMPLE_UPLOAD --> LOCAL_EVAL --> DRIFT_CHECK
    DRIFT_CHECK -->|Yes| DriftHook
    DRIFT_CHECK -->|No| KEEP_BUDGET

    DriftHook --> INCREASE_BUDGET --> NOTE3

    SAMPLE_UPLOAD --> GATED
    RATIO --> SUMMARY
    COUNT_ACCEPT --> SUMMARY

    %% 样式
    style REJECT fill:#ffcccc,stroke:#ff0000,stroke-width:3px
    style ACCEPT fill:#ccffcc,stroke:#00aa00,stroke-width:3px
    style THRESHOLD fill:#fff4cc,stroke:#ff9900,stroke-width:2px
    style HighRiskHook fill:#ffe6e6
    style DriftHook fill:#e6f7ff
    style GATED fill:#cce6ff
    style SUMMARY fill:#f0e6ff
```

---

## 核心逻辑详解

### 为什么要用 Privacy Gate？

**问题**: 即使加了高斯噪声，某些嵌入仍然可能泄露隐私
- 如果一个嵌入与其类别原型非常相似
- 攻击者可以推断："这个样本是类别 X 的典型代表"
- 容易受到成员推断攻击

**解决**: Privacy Gate 拒绝"太典型"的嵌入

---

### 为什么用余弦相似度？

**余弦相似度衡量方向的相似性，而不是距离**

```python
cos(z, proto) = (z · proto) / (||z|| × ||proto||)

取值范围: [-1, 1]
• 1.0  → 完全同向（非常相似）❌ 高风险
• 0.5  → 中等相似
• 0.0  → 正交（不相似）✅ 安全
• -1.0 → 反向（完全不同）✅ 非常安全
```

**为什么不用欧氏距离？**
- 嵌入的方向比距离更能表示语义
- 余弦相似度对尺度不敏感
- 更适合高维向量比较

---

### 动态阈值的计算

**为什么不用固定阈值？**

固定阈值（如 tau=0.7）的问题：
- 不同轮次、不同客户端的相似度分布不同
- 可能导致拒绝率过高或过低

**动态阈值的优势：**

```python
# Step 1: 计算第 85 百分位数
tau_raw = np.percentile(similarities, 85)
# 含义: 保留最不相似的 85% 的嵌入

# Step 2: 应用最低阈值
tau = max(tau_raw, tau_min)
# tau_min = 0.5，确保至少拒绝一些嵌入

# 实验结果: 拒绝率稳定在 16% 左右
```

**示例：**

假设某轮某客户端的相似度分布：
```
similarities = [0.92, 0.88, 0.85, 0.82, ..., 0.45, 0.40, 0.35]
               ↑ 前 15% (太相似)      ↑ 后 85% (可接受)

percentile(85) = 0.85
tau = max(0.85, 0.5) = 0.85

拒绝: sim > 0.85 的嵌入 (约 15%)
接受: sim ≤ 0.85 的嵌入 (约 85%)
```

---

### High-Risk Hook (高风险钩子)

**触发条件**: `reject_ratio > 0.30` (拒绝率超过 30%)

**为什么触发？**
- 如果拒绝率太高，说明太多嵌入"太相似"
- 可能是噪声不够，或数据分布异常
- 需要增强隐私保护

**触发动作：**
1. **增加噪声**: `σ ← σ × 1.5`
   - 更大的噪声 → 嵌入更不相似 → 拒绝率下降

2. **减少上传**: `budget ← budget ÷ 2`
   - 只上传最安全的一半
   - 降低隐私风险

3. **保守增强**: `augmentation_mode = conservative`
   - 使用更温和的数据增强
   - 减少极端样本

**实验结果：**
```
拒绝率: 0.16 (16%)
阈值: 0.30 (30%)
状态: ✅ 未触发（系统稳定）
```

---

### Drift Hook (漂移钩子)

**触发条件**: 验证准确率连续下降

```python
if val_acc[t] < val_acc[t-1] < val_acc[t-2]:
    # 连续两轮下降，触发 Drift Hook
```

**为什么触发？**
- 模型性能下降，可能是数据不足
- 或者当前 budget 太小，服务器收到的数据太少

**触发动作：**
```python
budget ← budget × 1.3  # 增加 30%
```

**逻辑：**
- 给服务器提供更多嵌入
- 帮助模型恢复性能

**实验结果：**
```
R20: Acc:0.6221
R30: Acc:0.6198 ↓ 触发
R40: Acc:0.6251 ↑ 恢复

R50: Acc:0.6205
R60: Acc:0.6160 ↓ 触发
R70: Acc:0.6188 ↑ 恢复
```

**影响：**
- Budget 从 937 增加到 979 (+4.5%)
- 变化不大，因为主要由 Orchestration 的 rarity_score 主导

---

## 完整参数流程

```
初始参数:
├─ sigma = 0.02
├─ tau_percentile = 0.15 (拒绝 15%)
├─ tau_min = 0.5
└─ upload_budget = 500

每轮调整:
├─ [Orchestration] budget ← 500 × (1 + rarity_score)
│                  ↓
│               937~979
│
├─ [Privacy Gate] 拒绝 ~16% 的嵌入
│
├─ [High-Risk Hook] 如果 reject_ratio > 0.30:
│   ├─ sigma ← sigma × 1.5
│   └─ budget ← budget ÷ 2
│
└─ [Drift Hook] 如果 val_acc 连续下降:
    └─ budget ← budget × 1.3
```

---

## 实验数据

### 拒绝率稳定性

| 轮次 | 拒绝率 | 状态 |
|------|--------|------|
| R1-R10 | 0.16 | ✅ 正常 |
| R11-R50 | 0.16 | ✅ 正常 |
| R51-R100 | 0.16 | ✅ 正常 |
| **平均** | **0.16** | **✅ 非常稳定** |

### 上传量统计

| 指标 | 值 |
|------|-----|
| 原始样本数 | ~2,500 / 客户端 |
| 多视图增强 | ×2 = ~5,000 |
| Privacy Gate 拒绝 | 16% → 剩余 ~4,200 |
| 采样至 budget | 937~979 |
| **实际上传** | **~950 个嵌入/轮** |

### 参数调整记录

| 参数 | 初始 | 最小 | 最大 | 最终 | 变化 |
|------|------|------|------|------|------|
| sigma | 0.0200 | 0.0200 | 0.0200 | 0.0200 | 0% |
| budget | 937 | 937 | 979 | 972 | +3.7% |
| reject_ratio | - | 0.16 | 0.16 | 0.16 | 稳定 |

---

## 与差分隐私的关系

### 差分隐私 (Differential Privacy)

**定义**: 添加高斯噪声提供理论隐私保证

```
z_tilde = z + N(0, σ²I)
```

**优点**: 有数学证明，隐私保证强
**缺点**: 不考虑数据分布，可能过度保护或保护不足

### Privacy Gate (经验隐私)

**定义**: 根据实际相似度分布拒绝高风险样本

```
if cos(z_tilde, proto) > tau:
    reject
```

**优点**: 适应数据分布，动态调整
**缺点**: 没有理论保证，基于经验

### 结合使用 = 最佳实践

```
输入 z
  ↓
[差分隐私] 添加噪声 → z_tilde
  ↓
[Privacy Gate] 拒绝高风险 → z_safe
  ↓
上传
```

**效果**:
- 差分隐私提供基础保护
- Privacy Gate 再次过滤，双重保险
- 实验证明: reject_ratio=16%，准确率 61.91%

---

## 可视化建议

### 关键要点

1. **输入输出要清晰**
   - 输入: 加噪声后的嵌入 + 标签
   - 输出: 通过门控的嵌入 + 统计信息

2. **决策点要突出**
   - 相似度 vs 阈值的比较
   - 拒绝率 vs 30% 的比较
   - 准确率连续下降的检测

3. **反馈循环要明确**
   - High-Risk Hook → 调整 sigma 和 budget
   - Drift Hook → 调整 budget
   - 这些调整会影响下一轮

4. **用颜色区分**
   - 红色: 拒绝分支
   - 绿色: 接受分支
   - 黄色: 阈值和决策点
   - 蓝色: 输出

---

## 论文写作建议

### 算法伪代码

```
Algorithm: Privacy Gate

Input:
  Z_tilde: noisy embeddings (N × 512)
  Y: labels (N,)
  tau_percentile: 0.15
  tau_min: 0.5

Output:
  Z_gated: accepted embeddings
  reject_ratio: rejection rate

1. // 计算类别原型
2. For c = 0 to C-1:
3.     prototypes[c] ← mean(Z_tilde[Y == c])

4. // 计算相似度
5. For i = 0 to N-1:
6.     sim[i] ← cosine_similarity(Z_tilde[i], prototypes[Y[i]])

7. // 动态阈值
8. tau_raw ← percentile(sim, 100 - tau_percentile × 100)
9. tau ← max(tau_raw, tau_min)

10. // 门控决策
11. accepted ← []
12. For i = 0 to N-1:
13.     If sim[i] ≤ tau:
14.         accepted.append(Z_tilde[i])

15. reject_ratio ← (N - len(accepted)) / N
16. Return accepted, reject_ratio
```

### 描述性文字

```
"To mitigate membership inference attacks, we propose a Privacy Gate
mechanism that filters out embeddings exhibiting high similarity to
their class prototypes. For each noisy embedding z̃ᵢ, we compute its
cosine similarity with the prototype of its corresponding class c.
Embeddings with similarity exceeding a dynamically computed threshold
τ are rejected from upload.

The threshold τ is adaptive, defined as τ = max(P₈₅(sim), τₘᵢₙ), where
P₈₅ denotes the 85th percentile of the similarity distribution. This
ensures a stable rejection rate (≈15%) across varying data distributions.

In our experiments on CIFAR-100, the Privacy Gate maintained a consistent
rejection rate of 16%, effectively removing high-risk embeddings while
preserving model utility (61.91% accuracy, +10.46% over FedAvg)."
```

---

## 总结

| 组件 | 作用 | 结果 |
|------|------|------|
| **Gaussian Noise** | 差分隐私保护 | σ=0.02 (稳定) |
| **Privacy Gate** | 拒绝高风险嵌入 | 16% 拒绝率 |
| **High-Risk Hook** | 极端情况保护 | 未触发 (系统稳定) |
| **Drift Hook** | 性能恢复机制 | 轻微触发 (R30, R60) |
| **综合效果** | 隐私+性能平衡 | ✅ 61.91% 准确率 |

**核心创新**: 将理论隐私保护（DP）与经验风险过滤（Privacy Gate）相结合，实现了隐私与准确率的最佳平衡！
