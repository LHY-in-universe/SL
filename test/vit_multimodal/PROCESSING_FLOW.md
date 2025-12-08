# ViT 多模态模型处理流程详解

本文档详细说明 Vision Transformer (ViT) 多模态模型在每一步处理中信息的变化。

## 目录

1. [输入处理阶段](#1-输入处理阶段)
2. [Embedding 层](#2-embedding-层)
3. [Transformer Encoder 层](#3-transformer-encoder-层)
4. [输出处理](#4-输出处理)
5. [多模态融合](#5-多模态融合)
6. [完整数据流示例](#6-完整数据流示例)

---

## 1. 输入处理阶段

### 1.1 图像输入处理

#### 步骤 1.1.1: 原始图像输入
```
输入: 原始图像
维度: [H, W, C]
说明: 
  - H: 图像高度（例如 224）
  - W: 图像宽度（例如 224）
  - C: 通道数（RGB 图像为 3）
```

#### 步骤 1.1.2: 图像预处理
```
操作: 归一化、resize
输入: [H, W, C]
输出: [224, 224, 3] (标准尺寸)
信息变化: 图像被调整到固定尺寸，像素值归一化到 [0, 1] 或标准化
```

#### 步骤 1.1.3: Patch 分割
```
操作: 将图像分割成固定大小的 patches
输入: [224, 224, 3]
参数: Patch 大小 P = 16
输出: N 个 patches，每个 patch: [P, P, C] = [16, 16, 3]
计算: N = (224 × 224) / (16 × 16) = 196 个 patches

维度变化:
  [224, 224, 3] 
  → reshape → 
  [14, 14, 16, 16, 3] 
  → flatten spatial dims → 
  [196, 16, 16, 3]
  → flatten patch → 
  [196, 768]  (196 个 patches，每个 patch 有 16×16×3=768 个值)

信息变化: 
  - 空间结构信息被转换为序列信息
  - 每个 patch 代表图像的局部区域
  - 原始 2D 结构变为 1D 序列
```

### 1.2 文本输入处理（多模态场景）

#### 步骤 1.2.1: 文本 Token 化
```
输入: 原始文本字符串
示例: "a photo of a cat"

操作: 使用 tokenizer 进行分词
输出: Token IDs
维度: [seq_len]
示例: [101, 1037, 2264, 1997, 1037, 4937, 102] (假设 seq_len = 7)

信息变化: 文本转换为数字序列
```

#### 步骤 1.2.2: Token 嵌入
```
输入: Token IDs [seq_len]
操作: 查找嵌入表 (Embedding Lookup)
输出: Token Embeddings
维度: [seq_len, D]
说明: 
  - D: 嵌入维度（例如 768）
  - 每个 token ID 映射到一个 D 维向量

信息变化: 离散的 token 转换为连续的向量表示
```

---

## 2. Embedding 层

### 2.1 Patch Embedding

#### 步骤 2.1.1: Patch 线性投影
```
输入: Patches [batch_size, N, P²×C]
      = [batch_size, 196, 768]
操作: 线性投影 (Linear Projection)
      z₀ = [x_p¹E; x_p²E; ...; x_pᴺE]
      E ∈ ℝ^(P²·C × D)
输出: Patch Embeddings
维度: [batch_size, N, D]
      = [batch_size, 196, 768]

信息变化:
  - 每个 patch 的原始像素值被投影到高维空间
  - 维度从 768 (16×16×3) 投影到 D (768)
  - 保留了 patch 的语义信息
```

#### 步骤 2.1.2: 位置编码添加
```
输入: Patch Embeddings [batch_size, N, D]
操作: 添加位置编码
      z₀ = z₀ + E_pos
      E_pos ∈ ℝ^((N+1) × D)
输出: 带位置信息的 Patch Embeddings
维度: [batch_size, N, D]

位置编码类型:
  - 可学习的位置编码 (Learnable Positional Embedding)
  - 固定位置编码 (Fixed Sinusoidal)
  
信息变化:
  - 为每个 patch 添加位置信息
  - 模型可以学习空间关系
```

### 2.2 Class Token

#### 步骤 2.2.1: Class Token 创建
```
操作: 创建可学习的 CLS token
输入: 无（可学习参数）
维度: [1, D] = [1, 768]

说明:
  - CLS token 是用于分类的特殊 token
  - 可学习参数，随训练更新
  - 聚合全局信息
```

#### 步骤 2.2.2: Class Token 拼接
```
输入: 
  - Patch Embeddings: [batch_size, N, D]
  - CLS Token: [batch_size, 1, D]
操作: 在序列开头拼接 CLS token
      z₀ = [x_class; x_p¹; x_p²; ...; x_pᴺ]
输出: 完整的输入序列
维度: [batch_size, N+1, D]
      = [batch_size, 197, 768] (196 patches + 1 CLS token)

信息变化:
  - 序列长度从 N 变为 N+1
  - CLS token 位于位置 0
  - 为后续的全局信息聚合做准备
```

---

## 3. Transformer Encoder 层

ViT 模型包含 L 层 Transformer Encoder，每层的处理流程相同。下面详细说明单层的处理过程。

### 3.1 单层 Transformer Block 概览

```
输入: z_{l-1} [batch_size, N+1, D]
      ↓
┌─────────────────────────────────┐
│  Layer Normalization (Pre-Norm) │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│ Multi-Head Self-Attention (MHSA)│
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│      Residual Connection (+)    │
│      z'_l = z_{l-1} + MHSA(z)   │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│  Layer Normalization (Pre-Norm) │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│   Feed-Forward Network (FFN)    │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│      Residual Connection (+)    │
│      z_l = z'_l + FFN(z')       │
└──────────────┬──────────────────┘
               ↓
输出: z_l [batch_size, N+1, D]
```

### 3.2 Multi-Head Self-Attention (MHSA)

#### 步骤 3.2.1: Query, Key, Value 计算
```
输入: z [batch_size, N+1, D]
操作: 线性投影得到 Q, K, V
      Q = zW_Q, K = zW_K, V = zW_V
      W_Q, W_K, W_V ∈ ℝ^(D × D_k)
      D_k = D / num_heads (例如 768 / 12 = 64)

输出: 
  - Q: [batch_size, N+1, D_k]
  - K: [batch_size, N+1, D_k]
  - V: [batch_size, N+1, D_k]

信息变化: 将输入映射到查询、键、值空间
```

#### 步骤 3.2.2: 多头分割
```
输入: Q, K, V [batch_size, N+1, D_k]
操作: 分割为 num_heads 个头
      Q → [batch_size, num_heads, N+1, D_k]
      K → [batch_size, num_heads, N+1, D_k]
      V → [batch_size, num_heads, N+1, D_k]

示例: num_heads = 12, D_k = 64
输出维度: [batch_size, 12, 197, 64]

信息变化: 每个头关注不同的特征子空间
```

#### 步骤 3.2.3: 注意力分数计算
```
输入: 
  - Q: [batch_size, num_heads, N+1, D_k]
  - K: [batch_size, num_heads, N+1, D_k]
操作: 计算注意力分数
      Attention(Q, K, V) = softmax(QK^T / √D_k) V
      scores = QK^T / √D_k
      attention_weights = softmax(scores)

中间结果:
  - scores: [batch_size, num_heads, N+1, N+1]
  - attention_weights: [batch_size, num_heads, N+1, N+1]

信息变化:
  - 计算每个位置与其他位置的相似度
  - attention_weights[i, j] 表示位置 i 对位置 j 的关注程度
  - 建立了全局依赖关系
```

#### 步骤 3.2.4: 加权求和
```
输入: 
  - attention_weights: [batch_size, num_heads, N+1, N+1]
  - V: [batch_size, num_heads, N+1, D_k]
操作: 加权求和
      output = attention_weights @ V
输出: [batch_size, num_heads, N+1, D_k]

信息变化:
  - 每个位置的信息由所有位置的加权组合得到
  - 权重由相似度决定
  - 实现了全局信息聚合
```

#### 步骤 3.2.5: 多头拼接
```
输入: [batch_size, num_heads, N+1, D_k]
操作: 拼接所有头
      Concatenate(head_1, head_2, ..., head_h)
输出: [batch_size, N+1, D] (其中 D = num_heads × D_k)

信息变化: 融合多个注意力头的信息
```

#### 步骤 3.2.6: 输出投影
```
输入: [batch_size, N+1, D]
操作: 线性投影
      output = Concatenate(...) W_O
      W_O ∈ ℝ^(D × D)
输出: [batch_size, N+1, D]

信息变化: 最终的特征表示
```

#### 步骤 3.2.7: 残差连接
```
输入: 
  - MHSA 输出: [batch_size, N+1, D]
  - 原始输入 z_{l-1}: [batch_size, N+1, D]
操作: 残差连接
      z'_l = z_{l-1} + MHSA_output
输出: [batch_size, N+1, D]

信息变化:
  - 保留原始信息
  - 允许梯度直接传播
  - 稳定训练
```

### 3.3 Feed-Forward Network (FFN)

#### 步骤 3.3.1: 第一个线性层（扩展）
```
输入: z'_l [batch_size, N+1, D]
操作: 线性投影（扩展维度）
      W_1 ∈ ℝ^(D × 4D)
      hidden = z'_l W_1
输出: [batch_size, N+1, 4D]
      = [batch_size, 197, 3072] (假设 D=768)

信息变化: 将特征维度扩展到 4 倍，增加模型容量
```

#### 步骤 3.3.2: 激活函数
```
输入: [batch_size, N+1, 4D]
操作: GELU 激活
      output = GELU(hidden)
输出: [batch_size, N+1, 4D]

信息变化: 引入非线性变换
```

#### 步骤 3.3.3: 第二个线性层（压缩）
```
输入: [batch_size, N+1, 4D]
操作: 线性投影（压缩维度）
      W_2 ∈ ℝ^(4D × D)
      output = GELU(hidden) W_2
输出: [batch_size, N+1, D]

信息变化: 将特征维度压缩回原始大小
```

#### 步骤 3.3.4: 残差连接
```
输入: 
  - FFN 输出: [batch_size, N+1, D]
  - z'_l: [batch_size, N+1, D]
操作: 残差连接
      z_l = z'_l + FFN_output
输出: [batch_size, N+1, D]

信息变化: 保留 MHSA 的输出信息
```

### 3.4 多层堆叠

```
Layer 0 (输入): z_0 [batch_size, 197, 768]
      ↓
Layer 1: z_1 [batch_size, 197, 768]
      ↓
Layer 2: z_2 [batch_size, 197, 768]
      ↓
...
      ↓
Layer L: z_L [batch_size, 197, 768]

信息变化:
  - 每一层都建立更复杂的特征表示
  - 浅层关注局部特征
  - 深层关注全局语义
  - 维度保持不变，但特征表示不断抽象
```

---

## 4. 输出处理

### 4.1 CLS Token 提取

#### 步骤 4.1.1: 提取 CLS Token
```
输入: z_L [batch_size, N+1, D]
操作: 提取位置 0 的 CLS token
      y = z_L[:, 0, :]
输出: [batch_size, D]
      = [batch_size, 768]

信息变化:
  - 从序列表示 [N+1, D] 到全局表示 [D]
  - CLS token 聚合了所有 patch 的信息
  - 包含图像的全局语义表示
```

### 4.2 分类头 / 任务头

#### 步骤 4.2.1: 分类头（用于图像分类）
```
输入: y [batch_size, D]
操作: 线性分类层
      logits = y W_head + b
      W_head ∈ ℝ^(D × num_classes)
输出: [batch_size, num_classes]

示例: num_classes = 1000 (ImageNet)
输出维度: [batch_size, 1000]

信息变化:
  - 从特征表示到类别概率
  - 每个类别对应一个分数
```

#### 步骤 4.2.2: Softmax（可选）
```
输入: logits [batch_size, num_classes]
操作: Softmax 归一化
      probabilities = softmax(logits)
输出: [batch_size, num_classes]

信息变化: 将分数转换为概率分布
```

---

## 5. 多模态融合

### 5.1 图像和文本特征对齐

#### 步骤 5.1.1: 图像特征提取
```
输入: 图像 patches
处理: ViT Encoder (如上所述)
输出: 
  - 图像 CLS token: [batch_size, D_img]
  - 图像 patch features: [batch_size, N_img, D_img]
```

#### 步骤 5.1.2: 文本特征提取
```
输入: 文本 tokens
处理: Text Encoder (类似 ViT 或 BERT)
输出:
  - 文本 CLS token: [batch_size, D_text]
  - 文本 token features: [batch_size, N_text, D_text]
```

#### 步骤 5.1.3: 特征维度对齐
```
操作: 投影到统一维度
      image_feat = Linear(image_feat)  → [batch_size, D]
      text_feat = Linear(text_feat)    → [batch_size, D]
      
输出: [batch_size, D] (统一维度)

信息变化: 不同模态的特征映射到相同空间
```

### 5.2 跨模态注意力

#### 步骤 5.2.1: 图像-文本注意力
```
输入:
  - Query (来自图像): [batch_size, N_img, D]
  - Key, Value (来自文本): [batch_size, N_text, D]
操作: Cross-Attention
      cross_attn = Attention(Q_img, K_text, V_text)
输出: [batch_size, N_img, D]

信息变化: 图像特征根据文本信息进行加权
```

#### 步骤 5.2.2: 文本-图像注意力
```
输入:
  - Query (来自文本): [batch_size, N_text, D]
  - Key, Value (来自图像): [batch_size, N_img, D]
操作: Cross-Attention
      cross_attn = Attention(Q_text, K_img, V_img)
输出: [batch_size, N_text, D]

信息变化: 文本特征根据图像信息进行加权
```

### 5.3 特征融合策略

#### 策略 1: 简单拼接
```
输入:
  - 图像特征: [batch_size, D]
  - 文本特征: [batch_size, D]
操作: 拼接
      fused = Concat([img_feat, text_feat])
输出: [batch_size, 2D]

信息变化: 直接拼接两个模态的特征
```

#### 策略 2: 加权融合
```
输入:
  - 图像特征: [batch_size, D]
  - 文本特征: [batch_size, D]
操作: 加权求和
      fused = α · img_feat + (1-α) · text_feat
输出: [batch_size, D]

信息变化: 按权重融合两个模态
```

#### 策略 3: 多层融合
```
输入: 多模态特征
操作: 通过多层 Transformer 融合
      fused = MultiModalTransformer([img_feat, text_feat])
输出: [batch_size, D]

信息变化: 通过深层交互建模多模态关系
```

---

## 6. 完整数据流示例

### 示例: ViT-Base 图像分类（单模态）

假设配置:
- 图像尺寸: 224 × 224
- Patch 大小: 16 × 16
- 嵌入维度: D = 768
- 层数: L = 12
- 注意力头数: 12
- Batch 大小: 32

```
[输入] 原始图像
[batch=32, H=224, W=224, C=3]
      ↓
[预处理] 归一化、resize
[batch=32, 224, 224, 3]
      ↓
[Patch 分割] 
[batch=32, 196, 768]  (196 patches × 768)
      ↓
[Patch Embedding]
[batch=32, 196, 768]
      ↓
[位置编码]
[batch=32, 196, 768]
      ↓
[添加 CLS Token]
[batch=32, 197, 768]  (196 + 1)
      ↓
┌─────────────────────┐
│ Transformer Layer 1 │
│  [batch=32, 197, 768]│
└─────────────────────┘
      ↓
┌─────────────────────┐
│ Transformer Layer 2 │
│  [batch=32, 197, 768]│
└─────────────────────┘
      ↓
      ...
      ↓
┌─────────────────────┐
│ Transformer Layer 12│
│  [batch=32, 197, 768]│
└─────────────────────┘
      ↓
[提取 CLS Token]
[batch=32, 768]
      ↓
[分类头]
[batch=32, 1000]  (ImageNet 类别数)
      ↓
[输出] 类别概率
[batch=32, 1000]
```

### 示例: ViT 多模态（图像+文本）

```
[图像输入]           [文本输入]
[batch=32, 224, 224, 3]  "a photo of a cat"
      ↓                        ↓
[图像处理]                [文本处理]
[batch=32, 197, 768]     [batch=32, 77, 768]
      ↓                        ↓
[图像 Encoder]           [文本 Encoder]
[batch=32, 768]          [batch=32, 768]
      ↓                        ↓
      └──────────┬─────────────┘
                 ↓
          [特征对齐]
          [batch=32, 768]
                 ↓
          [跨模态注意力]
          [batch=32, 768]
                 ↓
          [特征融合]
          [batch=32, 768]
                 ↓
          [任务头]
          [batch=32, num_classes]
                 ↓
          [输出] 预测结果
```

---

## qwen3vl 2B 拆分策略（ViT → Transformer 交界）

### 拆分点位置
- **位置**：ViT 前端完成 Patch Embedding + Position Embedding + CLS Token 拼接之后，与后续 Transformer Encoder 之间。
- **含义**：前端负责把图像切分为序列并加上位置信息；后端 Transformer 专注于序列建模。

### 拆分前张量形状
- 形状：`[batch, num_patches + 1, D]`
  - 示例：`[B, 197, D]`（224×224 图像，16×16 patch，N=196，含 1 个 CLS）
  - D：嵌入维度（qwen3vl 2B 典型设置按模型配置，可与 ViT Base/D 对齐）
- 内容：已完成线性投影的 patch 序列 + CLS + 位置编码

### 拆分后接口要求（发送到 Transformer 端）
- **张量传输**：
  - 发送：`[B, N+1, D]`（含 CLS 的序列张量）
  - dtype/精度：按模型配置（fp16/bf16/fp32）一致
  - 维度约定：batch 在维度 0，序列在维度 1，特征在维度 2
- **对齐**：
  - 位置编码与 patch 序列的顺序保持一致（CLS 在 index 0，后跟 196 个 patch）
  - 如存在 mask（可变长度或 padding），需一并传递且与序列长度对齐
- **后端输入**：
  - 直接作为 Transformer Encoder 的输入序列，不再重复位置编码/CLS 拼接

### 传输规模估算（示例）
- 若 B=32，N+1=197，D=768，fp16：
  - 元素数：32 × 197 × 768 ≈ 4.84M
  - 大小：约 9.7 MB 每次前向（4.84M × 2 bytes）
  - 可按实际 batch / D / 精度调整估算

### 小结
- 拆分点选在“序列化完成、Transformer 尚未开始”的边界，职责清晰。
- 前端专注图像分块与位置编码，后端专注序列建模。
- 传输接口简单：单一序列张量（可选 mask），避免重复处理和对齐问题。

---

## 总结

ViT 多模态模型的信息变化可以总结为：

1. **维度变化**: 
   - 输入: [H, W, C] → Patches: [N, P²C] → Embeddings: [N+1, D]
   - 经过 L 层 Transformer: 维度保持 [N+1, D]
   - 输出: [D] → [num_classes]

2. **信息抽象层次**:
   - 像素级 → Patch 级 → 局部特征 → 全局特征 → 语义表示

3. **多模态融合**:
   - 特征对齐 → 跨模态交互 → 融合表示 → 任务输出

4. **关键机制**:
   - Self-Attention: 建立全局依赖关系
   - Multi-Head: 关注不同的特征子空间
   - Residual Connection: 保留原始信息，稳定训练
   - Layer Normalization: 稳定训练过程
