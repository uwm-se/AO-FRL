# Limitations and Future Work

## For Paper (Concise Version)

### Limitations

Despite the significant accuracy improvements, AO-FRL faces two primary limitations that warrant careful consideration. **First and most critically, the communication cost is approximately 2× that of FedAvg** (5.22 GB vs 2.52 GB over 100 rounds). This overhead stems from dynamic budget allocation (500-1000 embeddings per client) and multi-view augmentation, making AO-FRL less suitable for bandwidth-constrained environments such as mobile networks or IoT deployments. While this cost is justifiable in high-stakes applications where accuracy is paramount (e.g., medical diagnosis), practitioners must weigh the 20.33% accuracy gain against the doubled communication burden.

**Second, our privacy protection mechanism lacks formal guarantees.** Although we employ Gaussian noise (σ=0.02) and filter the top 15% most similar embeddings, we cannot provide a rigorous (ε, δ)-differential privacy certificate. The adaptive nature of the percentile-based privacy gate—where the threshold depends on data-dependent similarity distributions—violates standard DP composition assumptions. Moreover, the relatively small noise scale (σ=0.02) and modest filtering rate (15%) may be insufficient against sophisticated attacks such as membership inference or gradient inversion, particularly if the attacker has knowledge of class prototypes. For deployment in highly sensitive domains, stronger privacy mechanisms (larger σ, stricter filtering, or secure aggregation) would be necessary, albeit at further accuracy cost.

### Future Work

Several promising directions could address these limitations and extend AO-FRL's applicability. **Communication efficiency** can be improved through gradient compression techniques (quantization, sparsification), adaptive participation strategies (sampling clients per round rather than full participation), or reducing augmentation views in later training stages. **Formal privacy analysis** should employ Rényi DP accounting or switch to non-adaptive gate thresholds to enable rigorous privacy budgeting across multiple rounds.

Beyond technical improvements, **extending AO-FRL to more complex and sensitive domains** represents an important frontier. Current experiments focus on image classification (CIFAR-100), but many real-world federated learning scenarios involve richer tasks: **multi-modal learning** (image-text models for medical report generation, video captioning for surveillance systems), **generative modeling** (federated training of diffusion models or large language models on private documents), and **time-series forecasting** (financial market prediction, patient vital sign monitoring). These domains often involve **highly sensitive data**—medical records under HIPAA, financial transactions under PCI-DSS, personal communications under GDPR—where both strong privacy guarantees and high utility are critical. Adapting AO-FRL's orchestration and privacy mechanisms to these settings, particularly for non-IID sequential data or heterogeneous modalities, would significantly broaden its impact and enable privacy-preserving collaboration in currently underserved application areas.

---

## Alternative Compact Version (Even Shorter)

### Limitations and Future Work

Despite achieving 20.33% accuracy improvement over FedAvg, AO-FRL has two primary limitations. **Communication cost is 2× higher** (5.22 GB vs 2.52 GB), limiting applicability in bandwidth-constrained settings. **Privacy guarantees are not formal**: Gaussian noise (σ=0.02) and 15% filtering lack rigorous (ε, δ)-DP certification due to adaptive mechanisms, and may be insufficient against sophisticated attacks with prototype knowledge.

Future work should address these limitations through **communication compression** (quantization, client sampling) and **formal privacy analysis** (Rényi DP accounting, non-adaptive thresholds). More importantly, extending AO-FRL beyond image classification to **complex sensitive domains**—such as multi-modal medical report generation (combining CT scans and clinical notes), federated video understanding, or financial fraud detection on transaction sequences—would demonstrate its practical value. These applications require both strong privacy (HIPAA, GDPR compliance) and high utility, making them ideal testbeds for privacy-preserving federated learning at scale.

---

## For Conclusion Section (Embedded Version)

Insert this into the Conclusion section:

### 5.3 Limitations and Path Forward

While AO-FRL demonstrates substantial improvements in accuracy (+20.33%) and fairness (+24.3% macro F1), two key limitations must be acknowledged. **Communication overhead is 2× that of FedAvg** (5.22 GB vs 2.52 GB), primarily due to dynamic budget scaling and multi-view augmentation. This makes the method less suitable for bandwidth-limited scenarios such as mobile or IoT deployments, though the cost may be acceptable in high-stakes applications where accuracy justifies the overhead. **Additionally, our privacy mechanism lacks formal (ε, δ)-DP guarantees.** The combination of Gaussian noise (σ=0.02) and percentile-based filtering (top 15%) provides empirical privacy protection, but the adaptive nature of the privacy gate prevents rigorous composition analysis, and the modest noise/filtering levels may be vulnerable to sophisticated attacks.

Future research should prioritize **communication efficiency** (through compression, client sampling, or reduced augmentation) and **formal privacy analysis** (via Rényi DP or non-adaptive thresholds). Beyond these technical improvements, extending AO-FRL to **complex, sensitive domains** represents a critical next step: multi-modal models (medical image-report generation), video understanding (surveillance with privacy), sequential prediction (financial time-series), and other HIPAA/GDPR-regulated applications. These domains demand both strong privacy and high utility, making them ideal venues for demonstrating federated learning's real-world impact.

---

## 中文版本

### 局限性与未来工作

尽管AO-FRL实现了20.33%的准确率提升,但存在两个主要局限性。**首先,通信成本约为FedAvg的2倍**(5.22 GB vs 2.52 GB),这源于动态预算分配(每客户端500-1000个嵌入)和多视图增强,使得该方法在带宽受限环境(如移动网络或物联网部署)中适用性降低。虽然在高风险应用(如医学诊断)中这一成本是合理的,但从业者必须权衡20.33%的准确率收益与双倍的通信负担。

**其次,我们的隐私保护机制缺乏形式化保证。**虽然采用了高斯噪声(σ=0.02)并过滤了相似度最高的前15%嵌入,但无法提供严格的(ε, δ)-差分隐私证明。基于百分位的隐私门的自适应特性——阈值依赖于数据相关的相似度分布——违反了标准差分隐私组合假设。此外,相对较小的噪声尺度(σ=0.02)和适度的过滤率(15%)可能不足以抵御成员推断或梯度反演等复杂攻击,特别是当攻击者掌握类原型知识时。对于高度敏感领域的部署,需要更强的隐私机制(更大的σ、更严格的过滤或安全聚合),尽管这会带来进一步的准确率损失。

未来工作应通过**通信压缩**(量化、稀疏化)、**客户端采样**(而非全员参与)或**减少后期增强视图**来改善通信效率,并通过**Rényi差分隐私核算**或**非自适应门阈值**实现形式化隐私分析。更重要的是,将AO-FRL扩展到**更复杂和敏感的领域**是一个重要前沿:多模态学习(医学报告生成结合CT扫描和临床笔记)、视频理解(监控系统的隐私保护)、时间序列预测(金融市场或患者生命体征监测)以及其他受HIPAA/GDPR监管的应用。这些领域既需要强隐私保证又需要高效用,是展示联邦学习实际影响力的理想场景。

---

## Key Points Summary

**Limitations (2 main points):**
1. ⚠️ **Communication Cost = 2× FedAvg** (5.22 GB vs 2.52 GB)
   - Dynamic budgets + multi-view augmentation
   - Problematic for bandwidth-constrained settings
   - Trade accuracy (+20.33%) for communication overhead

2. ⚠️ **No Formal Privacy Guarantee**
   - Gaussian noise (σ=0.02) + 15% filtering lacks (ε, δ)-DP proof
   - Adaptive gate violates DP composition
   - May be vulnerable to sophisticated attacks
   - Need stronger mechanisms for sensitive domains

**Future Work (2 main directions):**
1. 🔧 **Technical Improvements**
   - Communication: Compression, client sampling, reduced views
   - Privacy: Rényi DP accounting, non-adaptive thresholds

2. 🌟 **Domain Extensions** (Most Important!)
   - Multi-modal models (image-text, video-text)
   - Medical: Federated medical report generation (imaging + notes)
   - Finance: Transaction fraud detection, market forecasting
   - Video: Surveillance, action recognition with privacy
   - All require HIPAA/GDPR compliance + high utility

---

## Usage Recommendation

**For journal papers:** Use the full 3-paragraph version (first section)

**For conference papers (page limit):** Use the compact 2-paragraph version

**For thesis:** Expand with more technical details

**Emphasis in oral presentation:**
- "Communication cost is our primary limitation—2× FedAvg"
- "Privacy is not formally guaranteed—important for future work"
- "Excited about extensions to medical imaging, financial data"
