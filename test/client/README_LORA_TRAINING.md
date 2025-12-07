# LoRA 微调测试说明

## 概述

这是一个简化的 LoRA 微调测试脚本，用于证明在 Split Learning 架构下可以使用 PEFT LoRA 进行模型微调。

## 特性

- ✅ 使用 HuggingFace PEFT 库（标准库，无需自实现）
- ✅ 支持合成数据集和 HuggingFace datasets
- ✅ 简化实现：服务器只做前向传播，客户端做反向传播
- ✅ 自动保存 LoRA 权重
- ✅ 详细的训练进度显示

## 依赖要求

### 必需依赖
```bash
pip install peft
```

### 可选依赖
```bash
# 如果使用 HuggingFace datasets（可选，不使用会自动使用合成数据集）
pip install datasets
```

## 使用方法

### 1. 启动服务器

首先确保 Trunk 服务器正在运行：

```bash
cd /Users/lhy/Desktop/Git/SL
bash test/start_all.sh
```

### 2. 运行训练脚本

基本使用（使用合成数据集，快速测试）：

```bash
cd /Users/lhy/Desktop/Git/SL
python test/client/train_lora_simple.py
```

使用 HuggingFace 数据集：

```bash
python test/client/train_lora_simple.py --dataset wikitext --samples 50
```

自定义参数：

```bash
python test/client/train_lora_simple.py \
    --server localhost:50052 \
    --dataset synthetic \
    --samples 20 \
    --batch-size 2 \
    --epochs 1 \
    --lora-rank 8 \
    --lr 1e-4 \
    --save-dir ./lora_checkpoints
```

### 3. 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--server` | `localhost:50052` | Trunk 服务器地址 |
| `--dataset` | `synthetic` | 数据集类型 (`synthetic` 或 `wikitext`) |
| `--samples` | `20` | 数据集样本数 |
| `--batch-size` | `2` | 批次大小 |
| `--max-length` | `128` | 最大序列长度 |
| `--epochs` | `1` | 训练轮数 |
| `--lora-rank` | `8` | LoRA 秩 |
| `--lr` | `1e-4` | 学习率 |
| `--save-dir` | `./lora_checkpoints` | LoRA 权重保存目录 |

## 训练流程

### 简化版本（当前实现）

```
输入数据
    ↓
Bottom 模型 (本地，保留梯度) → hidden_1
    ↓
Trunk 模型 (远程服务器，断开梯度) → hidden_2
    ↓
Top 模型 (本地，保留梯度) → logits
    ↓
计算损失
    ↓
反向传播 (只在客户端)
    ↓
更新 Bottom 和 Top 的 LoRA 参数
```

**注意**：
- Trunk 模型的参数不会更新（简化版本）
- 只有 Bottom 和 Top 的 LoRA 参数会被更新
- 这足以证明 LoRA 微调的可行性

## 输出示例

```
======================================================================
Split Learning LoRA 微调测试
======================================================================

配置:
  服务器: localhost:50052
  数据集: synthetic (20 样本)
  批次大小: 2
  训练轮数: 1
  LoRA 秩: 8
  学习率: 0.0001

======================================================================
依赖检查
======================================================================
✅ 所有必需依赖已安装

[1] 加载模型
--------------------------------------------------
加载 Bottom 模型...
  ✓ Bottom 模型加载成功 (Layers 0-2)
加载 Top 模型...
  ✓ Top 模型加载成功 (Layers 10+)

[2] 应用 PEFT LoRA
--------------------------------------------------
LoRA 配置: rank=8, alpha=16, dropout=0.1
目标模块: c_attn, c_fc, c_proj

应用到 Bottom 模型...
Bottom 模型参数:
trainable params: 12,288 || all params: 53,559,552 || trainable%: 0.02

应用到 Top 模型...
Top 模型参数:
trainable params: 12,288 || all params: 52,774,656 || trainable%: 0.02

[3] 连接到 Trunk 服务器
--------------------------------------------------
✓ 已连接到服务器: localhost:50052

[4] 加载数据集
--------------------------------------------------
正在处理 20 条文本...
✓ 数据集加载成功，共 10 个批次

======================================================================
开始训练
======================================================================
训练配置:
  总批次: 10
  训练轮数: 1
  总训练步骤: 10

======================================================================
Epoch 1/1
======================================================================
  Step 2/10: loss = 10.2345, avg_loss = 10.1234
  Step 4/10: loss = 9.8765, avg_loss = 10.0123
  ...

Epoch 1 完成:
  平均损失: 9.8765
  最小损失: 9.5678
  最大损失: 10.2345

======================================================================
保存 LoRA 权重
======================================================================
✓ Bottom LoRA 权重已保存: ./lora_checkpoints/bottom_lora
✓ Top LoRA 权重已保存: ./lora_checkpoints/top_lora

======================================================================
训练总结
======================================================================
总训练步骤: 10
初始损失: 10.2345
最终损失: 9.5678
最低损失: 9.5678
平均损失: 9.8765

✅ 损失下降 0.6667，训练成功！

======================================================================
测试完成
======================================================================
```

## 验证训练成功

训练成功的标志：

1. ✅ **损失可以计算** - 没有报错，损失值正常显示
2. ✅ **参数可以更新** - 损失值有变化（通常下降）
3. ✅ **LoRA 权重可以保存** - 保存目录中有权重文件

## 文件结构

训练完成后，会生成以下文件：

```
lora_checkpoints/
  ├── bottom_lora/
  │   ├── adapter_config.json
  │   └── adapter_model.bin
  └── top_lora/
      ├── adapter_config.json
      └── adapter_model.bin
```

## 加载训练好的 LoRA 权重

```python
from peft import PeftModel

# 加载基础模型
bottom = GPT2BottomModel(config, end_layer=2)
bottom.load_state_dict(torch.load("models/bottom/gpt2_2-10_bottom.pt"))

# 加载 LoRA 权重
bottom_with_lora = PeftModel.from_pretrained(
    bottom,
    "./lora_checkpoints/bottom_lora"
)
```

## 常见问题

### Q: 训练速度很慢？
A: 这是正常的，因为使用 CPU 训练。可以使用更小的批次大小和更少的样本。

### Q: 损失没有下降？
A: 可能的原因：
- 数据集太小或质量不高
- 学习率不合适
- 训练步数太少

可以尝试：
- 增加训练样本数
- 调整学习率（如 5e-5 或 2e-4）
- 增加训练轮数

### Q: Trunk 模型的参数为什么不更新？
A: 这是简化版本的预期。当前的实现中，服务器只做前向传播，不做反向传播。

如果要更新 Trunk 模型，需要：
1. 实现完整的反向传播协议（扩展 gRPC）
2. 实现梯度传递机制

但当前的简化版本足以证明 LoRA 微调的可行性。

## 下一步

如果这个简化版本运行成功，可以：

1. **扩展功能**：
   - 实现完整的反向传播协议
   - 支持服务器端参数更新

2. **优化性能**：
   - 使用 GPU 加速
   - 增加批次大小
   - 使用混合精度训练

3. **改进训练**：
   - 添加验证集
   - 实现早停机制
   - 添加学习率调度

## 参考

- [PEFT 库文档](https://huggingface.co/docs/peft)
- [LoRA 微调指南](./LORA_TRAINING_GUIDE.md)
- [PEFT 集成指南](./PEFT_INTEGRATION_GUIDE.md)
