# Adaptive Hooks: Client-Side Self-Adjustment Mechanisms

## English Version (2 Paragraphs)

### 3.X Adaptive Hooks for Dynamic Privacy-Utility Balance

In addition to server-side orchestration, AO-FRL employs **three client-side adaptive hooks** that dynamically adjust privacy and augmentation parameters in response to local conditions without requiring coordination with the server. These hooks execute after each round based on observable metrics and modify client behavior for the subsequent round. The **Low-Data Hook** monitors local class distribution and switches to conservative augmentation mode (weaker crops, smaller rotations) when any class has fewer than $k=10$ samples, preventing over-augmentation from distorting rare-class features. The **High-Risk Hook** responds to elevated privacy threats by monitoring the privacy gate's rejection rate: when rejection exceeds $r=30\%$, indicating many samples are too similar to class prototypes, the hook increases noise scale ($\sigma \leftarrow \min(1.5\sigma, 0.5)$) and reduces upload budget by half ($B \leftarrow \max(B/2, 50)$) to mitigate exposure risk.

The **Drift Hook** addresses performance degradation by tracking validation accuracy trends: when accuracy declines for consecutive rounds, suggesting the global model is drifting away from the client's local distribution, the hook increases upload budget by 30% ($B \leftarrow 1.3B$) to inject more client-specific data into the aggregation. These hooks operate independently of server orchestration but follow a conflict resolution policy where **privacy concerns override utility optimization**—for example, if the server allocates a budget of 800 but the High-Risk Hook triggers, the final budget is 400. An ablation study with hooks disabled shows they contribute **+1.46pp accuracy improvement** and reduce the rejection rate by 6.68pp (from 22.41% to 15.73%), demonstrating their effectiveness in balancing privacy protection and model utility without manual hyperparameter tuning.

---

## 中文版本（2段）

### 3.X 自适应钩子：动态隐私-效用平衡机制

除服务器端编排外，AO-FRL采用**三个客户端自适应钩子**，根据本地条件动态调整隐私和增强参数，无需与服务器协调。这些钩子在每轮后基于可观测指标执行，并修改下一轮的客户端行为。**低数据钩子**监控本地类别分布，当任何类别样本数少于$k=10$时切换到保守增强模式（更弱的裁剪、更小的旋转），防止过度增强导致稀有类特征失真。**高风险钩子**通过监控隐私门的拒绝率来响应隐私威胁：当拒绝率超过$r=30\%$时（表明许多样本与类原型过于相似），该钩子增加噪声尺度（$\sigma \leftarrow \min(1.5\sigma, 0.5)$）并将上传预算减半（$B \leftarrow \max(B/2, 50)$），以降低暴露风险。

**漂移钩子**通过跟踪验证准确率趋势来解决性能退化问题：当准确率连续多轮下降时（表明全局模型偏离客户端本地分布），该钩子将上传预算增加30%（$B \leftarrow 1.3B$），向聚合中注入更多客户端特定数据。这些钩子独立于服务器编排运行，但遵循冲突解决策略，其中**隐私关注优先于效用优化**——例如，如果服务器分配预算800但高风险钩子触发，最终预算为400。禁用钩子的消融研究显示，钩子贡献了**+1.46pp的准确率提升**，并将拒绝率降低6.68pp（从22.41%降至15.73%），证明了它们在无需手动超参数调优的情况下平衡隐私保护和模型效用的有效性。

---

## Summary Table

| Hook | Trigger | Action | Purpose |
|------|---------|--------|---------|
| **Low-Data** | Class count < 10 | Conservative augmentation | Prevent rare-class distortion |
| **High-Risk** | Rejection rate > 30% | Increase noise ×1.5, reduce budget ÷2 | Emergency privacy protection |
| **Drift** | Val accuracy ↓ for 2-3 rounds | Increase budget ×1.3 | Recover from model drift |

**Empirical Impact:** +1.46pp accuracy, -6.68pp rejection rate, +0.39 GB communication

**Conflict Resolution:** Privacy (hooks) overrides utility (server orchestration)

---

**Document Created:** 2026-02-11
**Format:** Brief 2-paragraph description for Method section
**Word Count:** ~200 words per language
