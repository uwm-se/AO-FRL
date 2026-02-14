# Server Orchestration with Adaptive Hooks

## English Version (2 Paragraphs)

### 3.X Server Orchestration and Adaptive Adjustment Mechanisms

The server orchestration mechanism analyzes client summaries (label histograms, rejection rates, noise levels) after each round to generate per-client instructions that dynamically adjust privacy parameters and resource allocation. **Rarity-based budget allocation** forms the core strategy: the server computes a global label distribution gap by comparing the aggregated histogram against a uniform target, then assigns higher upload budgets to clients holding underrepresented classes via a rarity score $\rho = \sum_{c \in \mathcal{C}_i} \Delta_c$ (where $\mathcal{C}_i$ are classes present at client $i$ and $\Delta_c$ is the normalized gap for class $c$), resulting in $B_i = B_{\text{base}} \times (1 + \rho)$. This ensures minority classes receive proportionally more representation in the global model, improving fairness across heterogeneous data distributions. On top of this base allocation, two **adaptive hooks** further refine instructions based on observed risks: the **High-Risk Hook** monitors each client's privacy gate rejection ratio and, when it exceeds $r=30\%$ (indicating many samples are dangerously similar to class prototypes), increases noise scale ($\sigma \leftarrow \min(1.5\sigma, 0.5)$), halves the upload budget ($B \leftarrow \max(B/2, 50)$), and switches to conservative augmentation to mitigate exposure; the **Low-Data Hook** detects when any class has fewer than $k=10$ samples and responds by increasing the budget by 20% ($B \leftarrow 1.2B$) and enabling conservative augmentation to prevent over-distortion of rare-class features.

These mechanisms work synergistically within a single orchestration pass: for each client, the server first computes a rarity-adjusted budget, then applies the High-Risk Hook (which may reduce it for safety), and finally applies the Low-Data Hook (which may increase it for data-starved classes). **Priority is given to privacy over utility**: if a client qualifies for a high rarity budget (e.g., 800) but simultaneously triggers the High-Risk Hook, the final budget is halved (400), ensuring that privacy concerns override fairness objectives. An ablation study where orchestration is disabled (fixed budget=500, σ=0.02, augmentation=normal) shows that this adaptive mechanism contributes **+2.80pp accuracy improvement** (61.91% vs 59.11%) and maintains a manageable rejection rate of 15.73%, demonstrating that intelligent resource allocation is critical for handling data heterogeneity in federated learning. The orchestration's effectiveness stems from its ability to balance three competing objectives—**fairness** (via rarity scoring), **privacy** (via High-Risk Hook), and **quality** (via Low-Data Hook)—through a single unified framework that requires no manual hyperparameter tuning per client.

---

## 中文版本（2段）

### 3.X 服务器编排与自适应调整机制

服务器编排机制在每轮后分析客户端摘要（标签直方图、拒绝率、噪声水平），生成针对每个客户端的指令，动态调整隐私参数和资源分配。**基于稀缺性的预算分配**构成核心策略：服务器通过将聚合直方图与均匀目标进行比较来计算全局标签分布差距，然后通过稀缺性得分$\rho = \sum_{c \in \mathcal{C}_i} \Delta_c$（其中$\mathcal{C}_i$是客户端$i$存在的类别，$\Delta_c$是类别$c$的归一化差距）为持有代表性不足类别的客户端分配更高的上传预算，结果为$B_i = B_{\text{base}} \times (1 + \rho)$。这确保了少数类在全局模型中获得相应更多的表示，改善了异构数据分布的公平性。在此基础分配之上，两个**自适应钩子**根据观察到的风险进一步细化指令：**高风险钩子**监控每个客户端的隐私门拒绝率，当其超过$r=30\%$时（表明许多样本与类原型危险地相似），增加噪声尺度（$\sigma \leftarrow \min(1.5\sigma, 0.5)$）、将上传预算减半（$B \leftarrow \max(B/2, 50)$）并切换到保守增强以缓解暴露；**低数据钩子**检测任何类别样本数少于$k=10$的情况，响应方式是将预算增加20%（$B \leftarrow 1.2B$）并启用保守增强，以防止稀有类特征的过度失真。

这些机制在单次编排过程中协同工作：对于每个客户端，服务器首先计算稀缺性调整后的预算，然后应用高风险钩子（可能为安全性降低预算），最后应用低数据钩子（可能为数据匮乏的类别增加预算）。**优先考虑隐私而非效用**：如果客户端符合高稀缺性预算（例如800）但同时触发高风险钩子，最终预算减半（400），确保隐私关注优先于公平性目标。禁用编排的消融研究（固定budget=500, σ=0.02, augmentation=normal）显示，这种自适应机制贡献了**+2.80pp的准确率提升**（61.91% vs 59.11%）并维持了15.73%的可管理拒绝率，证明了智能资源分配对于处理联邦学习中的数据异构性至关重要。编排的有效性源于其通过单一统一框架平衡三个竞争目标的能力——**公平性**（通过稀缺性评分）、**隐私性**（通过高风险钩子）和**质量**（通过低数据钩子）——无需为每个客户端手动调整超参数。

---

## Summary Table

| Mechanism | Trigger | Action | Purpose |
|-----------|---------|--------|---------|
| **Rarity-based Budget** | Global label gap | Increase budget: $B_i = B_{\text{base}} \times (1 + \rho)$ | Fairness for minority classes |
| **High-Risk Hook** | Rejection rate > 30% | Increase noise ×1.5, reduce budget ÷2, conservative aug | Emergency privacy protection |
| **Low-Data Hook** | Class count < 10 | Increase budget ×1.2, conservative augmentation | Prevent rare-class distortion |

**Priority:** Privacy > Fairness > Utility (High-Risk Hook can override rarity-based budget allocation)

**Empirical Impact (Ablation):**
- **w/o Orchestration:** 59.11% accuracy (fixed budget=500)
- **w/ Orchestration:** 61.91% accuracy (+2.80pp)
- **Rejection Rate:** 15.73% (well-controlled)

---

## Orchestration Algorithm (Simplified)

```python
def orchestrate(summaries):
    # 1. Compute global label distribution gap
    global_hist = sum([s["label_histogram"] for s in summaries])
    target = global_hist.sum() / n_classes
    label_gap = normalized(target - global_hist)

    instructions = []
    for client_summary in summaries:
        # 2. Base budget from rarity score
        rarity_score = sum(label_gap[c] for c in client_classes)
        budget = base_budget * (1 + rarity_score)

        # 3. Apply High-Risk Hook
        if client_summary["reject_ratio"] > 0.30:
            sigma = min(sigma * 1.5, 0.5)
            budget = max(budget // 2, 50)
            aug_mode = "conservative"

        # 4. Apply Low-Data Hook
        if any(class_count < 10 for class_count in client_hist):
            budget = int(budget * 1.2)
            aug_mode = "conservative"

        instructions.append({
            "client_id": cid,
            "upload_budget": budget,
            "sigma": sigma,
            "augmentation_mode": aug_mode,
        })

    return instructions
```

---

**Document Created:** 2026-02-11
**Format:** Brief 2-paragraph description for Method section (server orchestration focus)
**Word Count:** ~200 words per language
**Key Insight:** Server-side orchestration with rarity scoring + two adaptive hooks (High-Risk, Low-Data) for dynamic privacy-utility-fairness balance
