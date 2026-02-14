# Three Adaptive Hooks (Simple Version)

## English Version (2 Paragraphs)

### 3.X Adaptive Hooks

To dynamically balance privacy, utility, and fairness during training, AO-FRL employs **three adaptive hooks** that automatically adjust client parameters based on real-time observations. The **Low-Data Hook** activates when any class has fewer than $k=10$ samples, switching to conservative augmentation (weaker crops, smaller rotations) to prevent over-distortion of rare-class features. The **High-Risk Hook** monitors the privacy gate's rejection rate and triggers when it exceeds $r=30\%$, indicating that many embeddings are dangerously similar to class prototypes; it responds by increasing noise scale ($\sigma \leftarrow \min(1.5\sigma, 0.5)$) and reducing upload budget by half ($B \leftarrow \max(B/2, 50)$) to mitigate privacy exposure. The **Drift Hook** tracks validation accuracy trends and activates when performance declines for consecutive rounds, increasing the upload budget by 30% ($B \leftarrow 1.3B$) to inject more client-specific data and recover from model drift.

These hooks operate without manual intervention and follow a priority order where **privacy overrides utility**: if both rarity-based orchestration (which increases budget for minority classes) and the High-Risk Hook (which reduces budget for safety) apply simultaneously, the privacy constraint takes precedence. An ablation study shows that disabling server orchestration (which includes the Low-Data and High-Risk Hooks) reduces accuracy by 2.80pp (from 61.91% to 59.11%), while disabling only the privacy gate improves accuracy by 0.67pp but eliminates critical privacy protection, demonstrating the effectiveness of these adaptive mechanisms in achieving a practical privacy-utility-fairness balance.

---

## 中文版本（2段）

### 3.X 自适应钩子

为了在训练过程中动态平衡隐私、效用和公平性，AO-FRL采用**三个自适应钩子**，根据实时观察自动调整客户端参数。**低数据钩子**在任何类别样本数少于$k=10$时激活，切换到保守增强（更弱的裁剪、更小的旋转），以防止稀有类特征的过度失真。**高风险钩子**监控隐私门的拒绝率，当其超过$r=30\%$时触发，表明许多嵌入与类原型危险地相似；它通过增加噪声尺度（$\sigma \leftarrow \min(1.5\sigma, 0.5)$）并将上传预算减半（$B \leftarrow \max(B/2, 50)$）来缓解隐私暴露。**漂移钩子**跟踪验证准确率趋势，当性能连续多轮下降时激活，将上传预算增加30%（$B \leftarrow 1.3B$），注入更多客户端特定数据以从模型漂移中恢复。

这些钩子无需人工干预即可运行，并遵循**隐私优先于效用**的优先级顺序：如果基于稀缺性的编排（为少数类增加预算）和高风险钩子（为安全性减少预算）同时适用，则隐私约束优先。消融研究显示，禁用服务器编排（包括低数据和高风险钩子）将准确率降低2.80pp（从61.91%降至59.11%），而仅禁用隐私门则将准确率提高0.67pp但消除了关键的隐私保护，证明了这些自适应机制在实现实用的隐私-效用-公平性平衡方面的有效性。

---

## Summary Table

| Hook | Trigger | Action |
|------|---------|--------|
| **Low-Data** | Class count < 10 | Conservative augmentation |
| **High-Risk** | Rejection rate > 30% | Increase noise ×1.5, reduce budget ÷2 |
| **Drift** | Val accuracy ↓ for 2-3 rounds | Increase budget ×1.3 |

**Priority:** Privacy > Fairness > Utility

---

**Document Created:** 2026-02-11
**Format:** Simple 2-paragraph description of three adaptive hooks only
