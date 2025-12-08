# 模型输出重复问题修复

## 问题描述

模型每次输出都是相同的词（如 "stairs stairs stairs..."），这是因为使用了 `argmax()` 总是选择概率最高的 token。

## 原因分析

### 原始代码问题
```python
# 问题代码：总是选择最高概率的 token
next_token_id = scaled_logits.argmax().item()
```

**问题**：
1. `argmax()` 总是选择概率最高的 token
2. 输出完全确定性，没有随机性
3. 一旦模型陷入某个模式，就会一直重复
4. 无法生成多样化的回复

### 为什么会出现重复？

1. **确定性选择**：`argmax()` 总是选择同一个 token
2. **模式循环**：如果 "stairs" 是最高概率，就会一直生成 "stairs"
3. **缺乏随机性**：没有采样机制，输出完全可预测

## 解决方案

### 使用 Top-K 采样

```python
# 修复后的代码：使用 top-k 采样
top_k = 50  # 从前 50 个最可能的 token 中选择
top_k_logits, top_k_indices = torch.topk(scaled_logits, top_k)
top_k_probs = torch.softmax(top_k_logits, dim=-1)

# 使用 numpy 随机采样（避免 Bus error）
sampled_idx = np.random.choice(len(top_k_probs), p=top_k_probs.numpy())
next_token_id = top_k_indices[sampled_idx].item()
```

**优势**：
1. ✅ 有随机性，输出多样化
2. ✅ 只从前 k 个最可能的 token 中选择，保证质量
3. ✅ 使用 numpy 采样，避免 `torch.multinomial()` 可能的 Bus error
4. ✅ 可以控制多样性（通过调整 top_k 和 temperature）

## 参数说明

### Temperature（温度）
- **默认**: 0.8
- **作用**: 控制输出的随机性
  - 较低（0.1-0.5）：更确定性，更保守
  - 中等（0.7-1.0）：平衡随机性和质量
  - 较高（1.0-2.0）：更随机，更有创意

### Top-K
- **默认**: 50
- **作用**: 限制采样范围
  - 较小（10-20）：更保守，质量更高
  - 中等（50-100）：平衡多样性和质量
  - 较大（200+）：更随机，但可能质量下降

## 使用建议

1. **想要更保守的输出**：降低 temperature (0.5-0.7) 和 top_k (20-30)
2. **想要更多样化的输出**：提高 temperature (0.9-1.2) 和 top_k (100+)
3. **遇到重复问题**：确保使用了 top-k 采样，而不是 argmax

## 对比

| 方法 | 随机性 | 质量 | 多样性 | Bus Error 风险 |
|------|--------|------|--------|---------------|
| `argmax()` | ❌ 无 | ✅ 高 | ❌ 低（重复） | ✅ 无 |
| `torch.multinomial()` | ✅ 有 | ✅ 高 | ✅ 高 | ⚠️ 可能 |
| `top-k + numpy` | ✅ 有 | ✅ 高 | ✅ 高 | ✅ 无 |

## 修复状态

✅ **已修复**：代码已更新为使用 top-k 采样，输出应该更加多样化。

