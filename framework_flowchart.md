# AO-FRL 框架流程图 (Mermaid 格式)

## 1. 系统架构图

```mermaid
graph TB
    subgraph "数据层"
        D1[CIFAR-100 训练集<br/>50,000 samples]
        D2[CIFAR-100 测试集<br/>10,000 samples]
        D3[Dirichlet 分割<br/>α=0.3]
    end

    subgraph "模型层"
        M1[ResNet-18 编码器<br/>冻结参数<br/>512维输出]
        M2[MLPHead<br/>可训练<br/>512→256→100]
    end

    subgraph "Agent 层"
        A1[ServerAgent<br/>Orchestrator]
        A2[ClientAgent × 20<br/>Workers]
        A3[EvaluatorAgent<br/>Monitor]
    end

    subgraph "通信层"
        C1[A2A Bus<br/>消息路由 + 审计日志]
    end

    subgraph "隐私层"
        P1[L2 Clipping<br/>||z|| ≤ 1.0]
        P2[Gaussian Noise<br/>σ = 0.02~0.5]
        P3[Privacy Gate<br/>拒绝相似样本]
    end

    D1 --> D3
    D3 --> A2
    D2 --> A3
    M1 --> A2
    A2 --> P1
    P1 --> P2
    P2 --> P3
    P3 --> C1
    C1 --> A1
    A1 --> M2
    M2 --> A3
    A1 -.指令.-> C1
    C1 -.指令.-> A2
```

## 2. 单轮训练完整流程

```mermaid
sequenceDiagram
    participant S as ServerAgent
    participant B as A2A Bus
    participant C1 as ClientAgent 1
    participant C2 as ClientAgent 2
    participant CN as ClientAgent 20
    participant E as EvaluatorAgent

    Note over S,E: Round R 开始

    %% Phase 1: 数据收集
    rect rgb(200, 220, 255)
        Note over S,CN: Phase 1: 客户端数据收集
        S->>B: send_task("extract_embeddings")
        B->>C1: 转发任务
        B->>C2: 转发任务
        B->>CN: 转发任务

        Note over C1: 1. 计算类别原型<br/>2. 生成多视图嵌入<br/>3. L2 Clipping<br/>4. 添加高斯噪声<br/>5. Privacy Gate 过滤<br/>6. 采样至 budget
        Note over C2: 同上处理
        Note over CN: 同上处理

        C1->>B: 上传嵌入 + summary
        C2->>B: 上传嵌入 + summary
        CN->>B: 上传嵌入 + summary
        B->>S: 收集所有数据
    end

    %% Phase 2: Orchestration
    rect rgb(200, 255, 220)
        Note over S: Phase 2: 服务器 Orchestration
        Note over S: 1. 统计 global_hist<br/>2. 计算 label_gap<br/>3. 识别稀缺类别<br/>4. 计算 rarity_score<br/>5. 生成个性化指令
        S->>B: send_task("apply_instructions")
        B->>C1: 指令: budget=690, sigma=0.02
        B->>C2: 指令: budget=550, sigma=0.03
        B->>CN: 指令: budget=720, sigma=0.02
        C1->>C1: 更新本地参数
        C2->>C2: 更新本地参数
        CN->>CN: 更新本地参数
    end

    %% Phase 3: 训练
    rect rgb(255, 220, 200)
        Note over S: Phase 3: 服务器模型训练
        Note over S: 1. 合并嵌入<br/>2. 更新 replay buffer<br/>3. 训练 MLPHead (3 epochs)<br/>4. 学习率衰减
        S->>C1: 广播新的 head
        S->>C2: 广播新的 head
        S->>CN: 广播新的 head
    end

    %% Phase 4: 评估
    rect rgb(255, 255, 200)
        Note over S,E: Phase 4: 全局评估
        S->>B: send_task("evaluate")
        B->>E: 转发评估任务
        Note over E: 1. 测试集推理<br/>2. 计算 accuracy, F1<br/>3. 记录通信成本
        E->>B: 返回结果
        B->>S: acc=0.6240, f1=0.6190
    end

    %% Phase 5: 本地评估
    rect rgb(230, 230, 250)
        Note over C1,CN: Phase 5: 客户端本地评估
        Note over C1: 验证集评估<br/>记录用于 drift_hook
        Note over C2: 验证集评估<br/>记录用于 drift_hook
        Note over CN: 验证集评估<br/>记录用于 drift_hook
    end

    Note over S,E: Round R 结束
```

## 3. 客户端嵌入提取详细流程

```mermaid
flowchart TD
    Start([开始: extract_gated_embeddings]) --> A1[从缓存读取训练嵌入]
    A1 --> A2{检查 augmentation_mode}

    A2 -->|normal| A3[正常增强]
    A2 -->|conservative| A4[保守增强<br/>更少变换]

    A3 --> B1[计算类别原型]
    A4 --> B1

    B1 --> B2["对每个类别 c:<br/>proto[c] = normalize(mean(embs[y==c]))"]
    B2 --> C1[生成多视图嵌入]

    C1 --> C2["view 1: embs + randn*0.01"]
    C1 --> C3["view 2: embs + randn*0.02"]

    C2 --> D1[L2 Clipping]
    C3 --> D1

    D1 --> D2["||z|| > C ? z = z * C/||z|| : z"]
    D2 --> E1[添加高斯噪声]

    E1 --> E2["z_tilde = z + N(0, σ²I)"]
    E2 --> F1[计算与原型相似度]

    F1 --> F2["对每个样本:<br/>sim = cosine_similarity(z_tilde, proto[label])"]
    F2 --> G1[Privacy Gate 判断]

    G1 --> G2["按类别计算 threshold:<br/>threshold = max(percentile(sims, 85%), 0.5)"]
    G2 --> G3{sim > threshold?}

    G3 -->|是| H1[拒绝样本]
    G3 -->|否| H2[接受样本]

    H1 --> I1[计算 reject_ratio]
    H2 --> I1

    I1 --> J1[从接受的样本中采样]
    J1 --> J2["采样至 upload_budget 个"]

    J2 --> K1[应用自适应钩子]

    K1 --> K2{low_data_hook<br/>任何类 < 10?}
    K2 -->|是| K3[augmentation_mode = conservative]
    K2 -->|否| K4[保持不变]

    K3 --> L1{high_risk_hook<br/>reject_ratio > 0.30?}
    K4 --> L1

    L1 -->|是| L2["sigma *= 1.5 (max 0.5)<br/>budget //= 2 (min 50)"]
    L1 -->|否| L3[保持不变]

    L2 --> M1{drift_hook<br/>验证准确率下降?}
    L3 --> M1

    M1 -->|是| M2[budget *= 1.3]
    M1 -->|否| M3[保持不变]

    M2 --> N1[生成 summary]
    M3 --> N1

    N1 --> N2["summary = {<br/>  label_histogram,<br/>  reject_ratio,<br/>  sigma,<br/>  n_uploaded<br/>}"]

    N2 --> End([返回: embeddings, labels, summary])

    style Start fill:#e1f5e1
    style End fill:#ffe1e1
    style G3 fill:#fff4e1
    style K2 fill:#fff4e1
    style L1 fill:#fff4e1
    style M1 fill:#fff4e1
```

## 4. 服务器 Orchestration 详细流程

```mermaid
flowchart TD
    Start([开始: orchestrate]) --> A1[收到 20 个客户端 summaries]

    A1 --> B1[统计全局类别分布]
    B1 --> B2["global_hist = Σ summary[i]['label_histogram']"]
    B2 --> B3["global_hist[c] = 本轮第 c 类收到的嵌入总数"]

    B3 --> C1[计算目标均匀分布]
    C1 --> C2["total = sum(global_hist)<br/>target = total / 100"]

    C2 --> D1[识别稀缺类别]
    D1 --> D2["label_gap[c] = max(target - global_hist[c], 0)"]
    D2 --> D3["label_gap_normalized = label_gap / sum(label_gap)"]

    D3 --> E1[为每个客户端计算稀缺性得分]
    E1 --> E2["对客户端 i:<br/>client_classes = summary[i] 中的类别"]
    E2 --> E3["rarity_score[i] = Σ label_gap_normalized[c]<br/>for c in client_classes"]

    E3 --> F1[生成基础 budget]
    F1 --> F2["budget[i] = base_budget * (1 + rarity_score[i])"]

    F2 --> G1{检查 high_risk_hook}
    G1 -->|reject_ratio > 0.30| G2["sigma *= 1.5 (max 0.5)<br/>budget //= 2 (min 50)<br/>augmentation = conservative"]
    G1 -->|否| G3[保持不变]

    G2 --> H1{检查 low_data_hook}
    G3 --> H1

    H1 -->|任何类 < 10| H2["budget *= 1.2<br/>augmentation = conservative"]
    H1 -->|否| H3[保持不变]

    H2 --> I1[构建指令列表]
    H3 --> I1

    I1 --> I2["instructions[i] = {<br/>  client_id: i,<br/>  upload_budget: budget[i],<br/>  sigma: sigma[i],<br/>  augmentation_mode: mode[i]<br/>}"]

    I2 --> End([返回: instructions])

    style Start fill:#e1f5e1
    style End fill:#ffe1e1
    style G1 fill:#fff4e1
    style H1 fill:#fff4e1
```

## 5. Privacy Gate 详细机制

```mermaid
flowchart TB
    subgraph "1. 计算类别原型"
        A1[本地训练嵌入] --> A2[按类别分组]
        A2 --> A3["proto[c] = normalize(mean(embs[y==c]))"]
    end

    subgraph "2. 生成加噪嵌入"
        B1[原始嵌入] --> B2[L2 Clipping]
        B2 --> B3["||z|| ≤ 1.0"]
        B3 --> B4[添加高斯噪声]
        B4 --> B5["z_tilde = z + N(0, σ²I)"]
    end

    subgraph "3. 相似度计算"
        C1[加噪嵌入 z_tilde] --> C2[获取对应原型]
        C2 --> C3["proto[label]"]
        C3 --> C4["sim = cosine_similarity(z_tilde, proto)"]
    end

    subgraph "4. 自适应阈值"
        D1[收集所有相似度] --> D2[按类别分组]
        D2 --> D3["对每个类别:<br/>sims_c = [sim for samples in class c]"]
        D3 --> D4["threshold_c = max(<br/>  percentile(sims_c, 85%),<br/>  0.5<br/>)"]
    end

    subgraph "5. 过滤决策"
        E1{sim > threshold?}
        E1 -->|是| E2[拒绝<br/>隐私风险高]
        E1 -->|否| E3[接受<br/>可安全上传]

        E2 --> F1[reject_count++]
        E3 --> F2[accepted_samples.add]
    end

    A3 --> C2
    B5 --> C1
    C4 --> D1
    D4 --> E1

    style E1 fill:#fff4e1
    style E2 fill:#ffe1e1
    style E3 fill:#e1f5e1
```

## 6. 三个自适应钩子

```mermaid
flowchart LR
    subgraph "Low-Data Hook"
        L1{任何类别<br/>样本数 < 10?}
        L1 -->|是| L2[切换到保守增强<br/>减少过拟合风险]
        L1 -->|否| L3[保持正常模式]
    end

    subgraph "High-Risk Hook"
        H1{reject_ratio<br/>> 0.30?}
        H1 -->|是| H2["增加噪声: σ × 1.5<br/>减少 budget: ÷ 2<br/>保守增强"]
        H1 -->|否| H3[保持当前设置]
    end

    subgraph "Drift Hook"
        D1{验证准确率<br/>连续下降?}
        D1 -->|是| D2[增加 budget: × 1.3<br/>补充更多数据]
        D1 -->|否| D3[保持当前 budget]
    end

    Start([客户端状态]) --> L1
    L2 --> H1
    L3 --> H1
    H2 --> D1
    H3 --> D1
    D2 --> End([更新参数])
    D3 --> End

    style L1 fill:#fff4e1
    style H1 fill:#fff4e1
    style D1 fill:#fff4e1
    style L2 fill:#ffe1e1
    style H2 fill:#ffe1e1
    style D2 fill:#e1f5e1
```

## 7. 对比三种方法的流程

```mermaid
graph TB
    subgraph "Centralized (上界)"
        C1[全部 50K 样本] --> C2[训练 MLP]
        C2 --> C3[50 epochs]
        C3 --> C4[准确率: 65.54%]
        C5[通信: 0 GB]
        C6[隐私: 无保护]
    end

    subgraph "FedAvg (基线)"
        F1[20 个客户端] --> F2[本地训练 MLP]
        F2 --> F3[每轮 3 local_epochs]
        F3 --> F4[上传参数 ~0.63MB]
        F4 --> F5[服务器聚合参数]
        F5 --> F6[100 rounds]
        F6 --> F7[准确率: 51.45%]
        F8[通信: 2.52 GB]
        F9[隐私: 无保护]
    end

    subgraph "AO-FRL (提出方法)"
        A1[20 个客户端] --> A2[提取嵌入]
        A2 --> A3[三重隐私保护]
        A3 --> A4[Privacy Gate 过滤]
        A4 --> A5[上传嵌入 ~1MB]
        A5 --> A6[服务器 Orchestration]
        A6 --> A7[训练 MLP 3 epochs]
        A7 --> A8[100 rounds]
        A8 --> A9[准确率: 61.91%]
        A10[通信: 5.22 GB]
        A11[隐私: 三重保护]
    end

    style C4 fill:#90EE90
    style F7 fill:#FFB6C6
    style A9 fill:#87CEEB
```

## 8. 数据流向图

```mermaid
flowchart LR
    subgraph "数据源"
        D1[(CIFAR-100<br/>60K images)]
    end

    subgraph "预处理"
        P1[Dirichlet 分割<br/>α=0.3]
        P2[ResNet-18 编码<br/>冻结]
        P3[预计算嵌入<br/>60K×512]
    end

    subgraph "客户端"
        C1[Client 1<br/>~2.5K embs]
        C2[Client 2<br/>~2.5K embs]
        C3[...<br/>...]
        C4[Client 20<br/>~2.5K embs]
    end

    subgraph "隐私处理"
        PR1[L2 Clip]
        PR2[+ Noise]
        PR3[Gate Filter]
    end

    subgraph "服务器"
        S1[收集嵌入<br/>~10-20K/round]
        S2[Orchestration<br/>决策]
        S3[训练 MLPHead]
        S4[Replay Buffer<br/>最多 50K]
    end

    subgraph "评估"
        E1[Test Set<br/>10K embs]
        E2[Accuracy<br/>Macro-F1]
    end

    D1 --> P1
    P1 --> P2
    P2 --> P3

    P3 --> C1
    P3 --> C2
    P3 --> C3
    P3 --> C4

    C1 --> PR1
    C2 --> PR1
    C3 --> PR1
    C4 --> PR1

    PR1 --> PR2
    PR2 --> PR3

    PR3 --> S1
    S1 --> S2
    S2 -.指令.-> C1
    S2 -.指令.-> C2
    S2 -.指令.-> C3
    S2 -.指令.-> C4

    S1 --> S3
    S3 --> S4
    S4 --> S3

    S3 --> E1
    E1 --> E2

    style D1 fill:#E6F3FF
    style PR3 fill:#FFE6E6
    style S2 fill:#E6FFE6
    style E2 fill:#FFF9E6
```

## 使用说明

### 在线工具
1. **Mermaid Live Editor**: https://mermaid.live/
   - 复制上面的代码块
   - 粘贴到编辑器
   - 自动生成流程图
   - 可导出 PNG/SVG

2. **VS Code** (安装 Mermaid 插件):
   - 安装 "Markdown Preview Mermaid Support"
   - 预览 .md 文件即可看到图表

3. **Notion/GitHub**:
   - 直接粘贴 mermaid 代码块
   - 自动渲染

### 建议
- **图 1 (系统架构)**: 用于论文的 Overview/Framework 部分
- **图 2 (单轮流程)**: 用于详细解释算法执行
- **图 3 (嵌入提取)**: 用于解释客户端隐私保护
- **图 4 (Orchestration)**: 用于解释服务器决策
- **图 5 (Privacy Gate)**: 用于解释隐私机制
- **图 7 (三方法对比)**: 用于 Related Work 或 Experiments
- **图 8 (数据流)**: 用于补充说明数据处理流程
