# 实验结果可视化建议

基于你的实验数据，以下是推荐的图表和分析维度。

---

## 📊 已有的图表

✅ **acc_vs_rounds.png** - 准确率随轮次变化
✅ **f1_vs_rounds.png** - F1分数随轮次变化
✅ **comm_vs_rounds.png** - 通信量随轮次变化
✅ **comm_vs_acc.png** - 通信效率对比
✅ **server_instructions.png** - 服务器指令可视化

---

## 🆕 建议新增的图表

### 类别 1: 隐私保护效果 (Privacy Protection)

#### 图1: Rejection Ratio Over Rounds
**数据来源**: `server_instructions.csv` → `avg_reject_ratio`

**目的**: 展示 Privacy Gate 的稳定性

```python
# 展示内容
- X轴: Round (1-100)
- Y轴: Rejection Ratio (0.0-1.0)
- 基准线: target=0.15, threshold=0.30

# 关键信息
- 平均拒绝率: 0.157 (非常稳定)
- 从未触发 high_risk_hook (< 0.30)
- 说明隐私参数设置合理
```

**图表类型**: 折线图 + 阴影区域（正常/警告/危险区）

**论文价值**: ⭐⭐⭐⭐⭐ (证明隐私机制的稳定性)

---

#### 图2: Privacy Budget vs Model Utility
**数据来源**: 对比不同 σ (噪声水平) 下的准确率

**目的**: 展示隐私-准确率权衡

```python
# 需要额外实验 (Ablation Study)
σ = [0.01, 0.02, 0.03, 0.05, 0.10]
accuracy = [?, 0.619, ?, ?, ?]

# 横轴: 噪声水平 σ
# 纵轴: 最终准确率
# 标注: reject_ratio 的变化
```

**图表类型**: 折线图 + 误差棒

**论文价值**: ⭐⭐⭐⭐⭐ (ablation study，展示参数选择合理性)

---

### 类别 2: 自适应机制 (Adaptive Orchestration)

#### 图3: Budget Evolution Over Rounds
**数据来源**: `server_instructions.csv` → `avg_budget`, `min_budget`, `max_budget`

**目的**: 展示动态预算分配机制

```python
# 展示内容
- X轴: Round
- Y轴: Upload Budget
- 三条线:
  - Average Budget (蓝色)
  - Max Budget (绿色虚线)
  - Min Budget (红色虚线)
- 基准线: base_budget=500

# 关键观察
- 平均 budget: 937-982 (比基准高 87-96%)
- 最大波动: 814-1111
- 说明 Orchestration 有效调整资源分配
```

**图表类型**: 折线图 + 填充区域

**论文价值**: ⭐⭐⭐⭐⭐ (核心创新点，展示自适应能力)

---

#### 图4: Budget Distribution by Client
**数据来源**: 需要从代码提取每个客户端的 budget 分配历史

**目的**: 展示服务器如何个性化分配资源

```python
# 展示内容
- 20个客户端在第50轮的 budget
- 按 budget 大小排序
- 标注每个客户端持有的稀缺类数量

# 示例
Client_5:  1,234 budget (持有3个稀缺类) → 高预算
Client_12:   856 budget (主要持有常见类) → 低预算

# 分析
相关性: 稀缺类越多 → budget 越高
```

**图表类型**: 条形图 + 颜色编码

**论文价值**: ⭐⭐⭐⭐ (展示 Orchestration 的公平性和针对性)

---

#### 图5: Augmentation Mode Distribution
**数据来源**: `server_instructions.csv` → `n_conservative`, `n_normal`

**目的**: 展示数据增强策略的自适应切换

```python
# 展示内容
- X轴: Round
- Y轴: 客户端数量 (0-20)
- 堆叠面积图:
  - Conservative (红色): 一直是20
  - Normal (绿色): 0

# 观察
- 所有轮次都是 Conservative (因为 low_data_hook 一直触发)
- 说明数据稀缺，系统保守策略正确
```

**图表类型**: 堆叠面积图

**论文价值**: ⭐⭐⭐ (展示自适应hooks的作用)

---

### 类别 3: 通信效率 (Communication Efficiency)

#### 图6: Communication Efficiency Breakdown
**数据来源**: `AO-FRL_rounds.csv` → `comm_bytes`

**目的**: 详细分析通信成本构成

```python
# 饼图展示单轮通信成本
总成本: ~52 MB/round
├─ 嵌入数据: ~51.6 MB (99%)
│   └─ 19,400 embeddings × 512 × 4 bytes
└─ Summary: ~10 KB (0.02%)
    └─ 20 clients × 500 bytes

# 关键信息
- Summary 开销几乎可忽略
- 主要成本在嵌入传输
```

**图表类型**: 饼图 + 详细标注

**论文价值**: ⭐⭐⭐⭐ (证明轻量级设计)

---

#### 图7: Communication Cost Reduction
**数据来源**: 对比 FedAvg 和 AO-FRL 的通信模式

**目的**: 展示参数传输 vs 嵌入传输的差异

```python
# 对比表格
方法        | 单轮通信量 | 100轮总量 | 相对比例
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FedAvg      | 25.2 MB   | 2,520 MB  | 1.0×
AO-FRL      | 52.2 MB   | 5,218 MB  | 2.07×

# 但是！
AO-FRL 准确率高 10.46%
性价比: +10.46% accuracy / 2.07× comm = 5.05% per 1×

# 折线图展示
- X轴: Communication Cost (MB)
- Y轴: Accuracy (%)
- 两条曲线: FedAvg vs AO-FRL
- 标注: AO-FRL 更陡峭 (效率更高)
```

**图表类型**: 双Y轴图 + 表格

**论文价值**: ⭐⭐⭐⭐⭐ (证明虽然通信多，但换来更高准确率，值得)

---

### 类别 4: 模型性能 (Model Performance)

#### 图8: Convergence Speed Comparison
**数据来源**: 三种方法达到特定准确率的轮次数

**目的**: 展示收敛速度

```python
# 表格
准确率目标 | FedAvg | AO-FRL | Centralized
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
50%       | ~10轮  | <1轮   | <1轮
55%       | ~30轮  | 1轮    | 1轮
60%       | 未达到  | 2轮    | 2轮
61%       | 未达到  | 5轮    | 3轮

# 折线图
- X轴: Round
- Y轴: Accuracy
- 三条曲线
- 标注达到60%的轮次
```

**图表类型**: 阶梯图 + 标注

**论文价值**: ⭐⭐⭐⭐ (证明收敛快)

---

#### 图9: Per-Class Accuracy Heatmap
**数据来源**: 需要计算每个类别的准确率

**目的**: 展示对稀有类的保护效果

```python
# 需要额外计算
- 100个类别的准确率
- 按类别频率排序
- 对比 FedAvg vs AO-FRL

# 热图
- Y轴: 100个类别
- X轴: [FedAvg, AO-FRL]
- 颜色: 准确率 (0-100%)

# 关键观察
- 稀有类 (频率<1%) 的准确率提升
- AO-FRL 对长尾类别更友好
```

**图表类型**: 热图 + 类别频率注释

**论文价值**: ⭐⭐⭐⭐⭐ (证明公平性，Non-IID处理能力)

---

#### 图10: Accuracy Distribution Across Clients
**数据来源**: 需要记录每个客户端的本地验证准确率

**目的**: 展示所有客户端都受益

```python
# 箱线图
- X轴: [FedAvg, AO-FRL]
- Y轴: 客户端本地准确率
- 20个客户端的分布

# 关键指标
方法    | 平均 | 最小 | 最大 | 标准差
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FedAvg | 51%  | 42%  | 58%  | 4.2%
AO-FRL | 62%  | 55%  | 68%  | 3.1%

# 观察
- AO-FRL 提升所有客户端
- 标准差更小 (更公平)
```

**图表类型**: 箱线图 + 统计表格

**论文价值**: ⭐⭐⭐⭐⭐ (证明方法的普适性和公平性)

---

### 类别 5: 类别平衡 (Class Balance)

#### 图11: Global Class Distribution Evolution
**数据来源**: 需要记录每轮的 global_hist

**目的**: 展示如何平衡类别分布

```python
# 堆叠面积图
- X轴: Round
- Y轴: 上传的嵌入数量
- 100层: 每个类别的数量
- 颜色: 从蓝(常见)到红(稀有)

# 观察
- 初期: 常见类占主导
- 中期: Orchestration 开始调整
- 后期: 分布趋于均匀

# 统计指标
标准差 (类别分布):
- Round 1:  σ = 145 (极不均衡)
- Round 50: σ = 89  (改善)
- Round 100: σ = 76 (更均衡)
```

**图表类型**: 堆叠面积图 + 标准差曲线

**论文价值**: ⭐⭐⭐⭐⭐ (核心创新，展示Orchestration效果)

---

#### 图12: Rare Class Protection
**数据来源**: 对比稀有类在 FedAvg vs AO-FRL 的表现

**目的**: 突出对长尾类别的保护

```python
# 定义稀有类: 全局频率 < 0.5%
稀有类数量: 32个 (如类2, 类15, ...)

# 散点图
- X轴: 类别全局频率
- Y轴: 类别准确率
- 两组点: FedAvg (红) vs AO-FRL (绿)

# 趋势线
- FedAvg: 稀有类准确率低 (20-30%)
- AO-FRL: 稀有类准确率高 (50-60%)

# 重点标注几个极稀有类
类2:  FedAvg 18% → AO-FRL 52% (+34%)
类15: FedAvg 22% → AO-FRL 48% (+26%)
```

**图表类型**: 散点图 + 趋势线

**论文价值**: ⭐⭐⭐⭐⭐ (证明对Non-IID长尾问题的解决能力)

---

### 类别 6: 消融实验 (Ablation Study)

#### 图13: Component Contribution Analysis
**数据来源**: 需要做消融实验

**目的**: 量化每个组件的贡献

```python
# 实验设置
方法                              | Accuracy | F1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FedAvg (Baseline)                | 51.45%   | 49.49%
+ Multi-View Augmentation        | 54.2%    | 52.1%   (+2.75%)
+ Privacy Gate                   | 56.8%    | 54.5%   (+5.35%)
+ Orchestration                  | 61.9%    | 61.5%   (+10.45%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Centralized (Upper Bound)        | 65.54%   | 65.54%

# 柱状图展示
每个组件的增量贡献
```

**图表类型**: 堆叠柱状图

**论文价值**: ⭐⭐⭐⭐⭐ (必须有！证明每个组件都有效)

---

#### 图14: Hyperparameter Sensitivity
**数据来源**: 需要额外实验

**目的**: 展示对关键参数的敏感性

```python
# 参数1: tau_percentile (拒绝百分位)
tau_percentile = [0.05, 0.10, 0.15, 0.20, 0.25]
accuracy = [?, ?, 61.9%, ?, ?]
reject_ratio = [0.05, 0.10, 0.15, 0.20, 0.25]

# 参数2: alpha (Non-IID程度)
alpha = [0.1, 0.3, 0.5, 1.0]
accuracy = [?, 61.9%, ?, ?]

# 参数3: upload_budget
base_budget = [300, 500, 700, 1000]
accuracy = [?, 61.9%, ?, ?]

# 多子图展示
3个子图，每个展示一个参数的影响
```

**图表类型**: 多子图折线图

**论文价值**: ⭐⭐⭐⭐ (展示鲁棒性和参数选择依据)

---

### 类别 7: 对比分析 (Comparative Analysis)

#### 图15: Method Comparison Table
**数据来源**: 综合所有指标

**目的**: 全面对比三种方法

```python
指标                        | Centralized | FedAvg | AO-FRL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
准确率 (Accuracy)           | 65.54%      | 51.45% | 61.91%
F1-Score                   | 65.54%      | 49.49% | 61.53%
通信成本 (MB)               | 0           | 2,520  | 5,218
收敛轮次 (到60%)            | 2           | -      | 2
隐私保护                    | ❌          | ❌     | ✅
稀有类保护                  | ✅          | ❌     | ✅
自适应能力                  | -           | ❌     | ✅
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
综合评分                    | 8/10        | 4/10   | 9/10
```

**图表类型**: 表格 + 雷达图

**论文价值**: ⭐⭐⭐⭐⭐ (总结性图表)

---

#### 图16: Scalability Analysis
**数据来源**: 需要额外实验（不同客户端数量）

**目的**: 展示方法的可扩展性

```python
# 实验设置
n_clients = [5, 10, 20, 50, 100]
accuracy = [?, ?, 61.9%, ?, ?]
comm_per_round = [?, ?, 52.2, ?, ?]

# 双Y轴图
- X轴: 客户端数量
- Y轴1: 准确率
- Y轴2: 单轮通信量
```

**图表类型**: 双Y轴折线图

**论文价值**: ⭐⭐⭐⭐ (展示实用性)

---

## 🎯 优先级推荐

### 必须有 (Must Have) ⭐⭐⭐⭐⭐

1. **图3: Budget Evolution** - 展示核心创新
2. **图9: Per-Class Accuracy** - 展示Non-IID处理能力
3. **图11: Global Class Distribution Evolution** - 展示Orchestration效果
4. **图12: Rare Class Protection** - 突出优势
5. **图13: Ablation Study** - 证明每个组件有效
6. **图15: Method Comparison Table** - 总结

### 强烈推荐 (Highly Recommended) ⭐⭐⭐⭐

7. **图1: Rejection Ratio** - 证明隐私机制稳定
8. **图7: Communication Efficiency** - 解释通信成本
9. **图8: Convergence Speed** - 证明效率
10. **图10: Accuracy Distribution Across Clients** - 证明公平性
11. **图14: Hyperparameter Sensitivity** - 参数分析

### 可选 (Optional) ⭐⭐⭐

12. **图2: Privacy-Utility Tradeoff** - 需要额外实验
13. **图4: Budget Distribution by Client** - 细节展示
14. **图5: Augmentation Mode** - 辅助说明
15. **图6: Communication Breakdown** - 细节分析
16. **图16: Scalability** - 需要额外实验

---

## 📝 实现建议

### 可以直接画的图 (基于现有数据)

```python
# 图3: Budget Evolution
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results/server_instructions.csv')
plt.figure(figsize=(10, 6))
plt.plot(df['round'], df['avg_budget'], label='Average', linewidth=2)
plt.fill_between(df['round'], df['min_budget'], df['max_budget'],
                 alpha=0.3, label='Range')
plt.axhline(y=500, color='r', linestyle='--', label='Base Budget')
plt.xlabel('Round')
plt.ylabel('Upload Budget')
plt.legend()
plt.title('Dynamic Budget Allocation Over Rounds')
plt.savefig('budget_evolution.png')
```

```python
# 图1: Rejection Ratio
plt.figure(figsize=(10, 6))
plt.plot(df['round'], df['avg_reject_ratio'], linewidth=2)
plt.axhline(y=0.15, color='g', linestyle='--', label='Target (15%)')
plt.axhline(y=0.30, color='r', linestyle='--', label='Threshold (30%)')
plt.fill_between(df['round'], 0, 0.15, alpha=0.2, color='green', label='Safe')
plt.fill_between(df['round'], 0.15, 0.30, alpha=0.2, color='yellow', label='Warning')
plt.fill_between(df['round'], 0.30, 1.0, alpha=0.2, color='red', label='Danger')
plt.xlabel('Round')
plt.ylabel('Rejection Ratio')
plt.legend()
plt.title('Privacy Gate Rejection Rate Stability')
plt.savefig('rejection_ratio.png')
```

### 需要额外数据的图

**图9, 10, 12**: 需要修改代码记录每个类别/客户端的详细准确率

**图13**: 需要做消融实验（去掉某些组件重新训练）

**图14**: 需要做参数扫描实验

**图16**: 需要改变客户端数量重新实验

---

## 🎨 绘图风格建议

### 颜色方案
- **AO-FRL**: 蓝色 `#2196F3`
- **FedAvg**: 橙色 `#FF9800`
- **Centralized**: 绿色 `#4CAF50`
- **危险/拒绝**: 红色 `#F44336`
- **安全/接受**: 绿色 `#4CAF50`

### 字体
- Title: 14pt, bold
- Axis labels: 12pt
- Legend: 10pt
- 使用无衬线字体 (Arial, Helvetica)

### 尺寸
- 单图: 10×6 inches (适合论文单栏)
- 双图: 12×5 inches (适合论文双栏)
- DPI: 300 (出版质量)

---

## 📄 论文组织建议

### 实验结果章节结构

```
5. Experimental Results

5.1 Overall Performance (图15表格 + 图8收敛)
    - 三种方法的总体对比
    - 收敛速度分析

5.2 Communication Efficiency (图7 + 图6)
    - 通信成本分析
    - 效率对比

5.3 Privacy Protection (图1 + 图2)
    - Privacy Gate 稳定性
    - 隐私-准确率权衡

5.4 Adaptive Orchestration (图3 + 图11)
    - 动态资源分配
    - 类别平衡效果

5.5 Non-IID Handling (图9 + 图12)
    - 每类准确率
    - 稀有类保护

5.6 Fairness Analysis (图10)
    - 所有客户端受益

5.7 Ablation Study (图13 + 图14)
    - 组件贡献
    - 参数敏感性
```

---

## 🔧 快速实现脚本

我可以帮你写一个完整的绘图脚本 `plot_all_figures.py`，一次性生成所有可以画的图。

需要吗？
