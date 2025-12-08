# ViT 多模态模型处理流程说明

## 简介

本文档详细说明 Vision Transformer (ViT) 多模态模型在处理图像和文本输入时，每一步中信息的变化。文档涵盖从输入预处理到最终输出的完整流程，包括维度变化、数据流转换和处理操作。

## 文档结构

- **README.md** (本文件): 文档概览和索引
- **PROCESSING_FLOW.md**: 详细的处理流程说明，包括每一步的信息变化
- **run_vit_info.py**: 可运行的 ViT 模型代码，实时显示每一步的张量形状和大小
- **qwen3vl 2B 拆分策略**: 参见下方概览或 `PROCESSING_FLOW.md` 中的拆分章节
- **QWEN3VL_SPLIT_CORE.md**: 使用 SplitLearnCore API 进行 qwen3vl 2B 拆分的示例说明

## 快速开始

### 运行 ViT 模型查看信息变化

```bash
cd /Users/lhy/Desktop/Git/SL
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 test/vit_multimodal/run_vit_info.py
```

这个脚本会：
- 创建完整的 ViT 模型
- 执行前向传播
- **在每一步打印详细的张量信息**：
  - 形状 (Shape)
  - 元素数量 (Num Elements)
  - 数据类型 (Dtype)
  - 内存占用 (Memory)
  - 维度说明

### 输出示例

脚本会输出类似以下的信息：

```
【输入图像 (Input Image)】
形状 (Shape): torch.Size([2, 3, 224, 224])
元素数量 (Num Elements): 301,056
内存占用 (Memory): 1.1484 MB

【Patch Embedding 后 (After Conv2d)】
形状 (Shape): torch.Size([2, 768, 14, 14])
元素数量 (Num Elements): 301,056
内存占用 (Memory): 1.1484 MB
...
```

## 主要内容

### 1. 输入处理阶段
- 图像输入：Patch 分割和嵌入
- 文本输入：Token 化和嵌入
- 多模态输入对齐

### 2. Embedding 层
- Patch Embedding 处理
- Class Token 添加
- 位置编码

### 3. Transformer Encoder 层
- Multi-Head Self-Attention
- Feed-Forward Network
- Layer Normalization 和残差连接

### 4. 输出处理
- CLS Token 提取
- 分类头/任务头
- 多模态融合策略

## 文档导航

- **理论说明**: [PROCESSING_FLOW.md](./PROCESSING_FLOW.md) - 详细的处理流程理论说明
- **实际运行**: [run_vit_info.py](./run_vit_info.py) - 可运行代码查看实际张量信息
- **qwen3vl 2B 拆分策略**: [PROCESSING_FLOW.md](./PROCESSING_FLOW.md#qwen3vl-2b-拆分策略vit-→-transformer-交界) - 拆分点与张量形状说明
- **使用核心 API 拆分**: [QWEN3VL_SPLIT_CORE.md](./QWEN3VL_SPLIT_CORE.md) - 使用 SplitLearnCore API 在 ViT 与 Transformer 交界拆分 qwen3vl 2B 的示例

## 使用说明

本文档适用于：
- 理解 ViT 多模态模型的内部工作机制
- 分析模型各层的信息变化
- 调试和优化模型性能
- 学习 Transformer 架构在视觉任务中的应用
- **通过实际运行代码验证理论理解**

## 配置说明

可以在 `run_vit_info.py` 中修改以下配置：

```python
BATCH_SIZE = 2          # 批次大小
IMAGE_SIZE = 224        # 图像尺寸
PATCH_SIZE = 16         # Patch 大小
EMBED_DIM = 768         # 嵌入维度
NUM_HEADS = 12          # 注意力头数
NUM_LAYERS = 12         # Transformer 层数
NUM_CLASSES = 1000      # 分类类别数
```

