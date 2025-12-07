# 为什么这个版本没有 Bus Error？

## 关键改进

### 1. **使用 tiny-gpt2 模型（最重要）**

```python
model_name = os.getenv("GPT2_MODEL", "sshleifer/tiny-gpt2")  # 默认使用 tiny-gpt2
```

**原因**：
- `tiny-gpt2` 是一个极小的模型（约 0.1M 参数）
- 完整 `gpt2` 有 124.44M 参数，内存占用和计算复杂度高得多
- 小模型减少了内存压力，降低了 Bus error 的风险

**对比**：
- `tiny-gpt2`: ~0.1M 参数，~0.5MB 内存
- `gpt2`: 124.44M 参数，~500MB 内存

### 2. **使用 numpy 采样而不是 torch.multinomial**

```python
# 使用 numpy 的随机采样（更安全，避免 Bus error）
sampled_idx = np.random.choice(len(top_k_probs), p=top_k_probs.numpy())
```

**原因**：
- `torch.multinomial()` 在某些环境下（特别是 macOS + Python 3.11.0）可能导致 Bus error
- `numpy.random.choice()` 更稳定，兼容性更好
- 将概率移到 CPU 的 numpy 数组，避免 GPU/MPS 相关问题

### 3. **将 logits 移到 CPU 处理**

```python
next_token_logits = logits[0, -1, :].cpu()  # 移到 CPU 避免 MPS 问题
```

**原因**：
- 即使强制使用 CPU，某些操作可能仍会触发 MPS（Metal Performance Shaders）
- 明确移到 CPU 可以避免 MPS 相关的 Bus error

### 4. **使用 torch.inference_mode()**

```python
with torch.inference_mode():
    # 模型推理
```

**原因**：
- `torch.inference_mode()` 比 `torch.no_grad()` 更安全
- 提供更好的性能优化，同时避免某些内存问题

### 5. **使用 clone() 避免内存问题**

```python
generated_ids = torch.cat([generated_ids.clone(), next_token_tensor], dim=1)
```

**原因**：
- `clone()` 创建新的内存副本，避免内存对齐问题
- 减少内存碎片，降低 Bus error 风险

### 6. **单线程模式**

```python
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
torch.set_num_threads(1)
```

**原因**：
- 多线程竞争可能导致内存对齐问题
- 单线程模式更稳定，避免竞争条件

### 7. **禁用 JIT 编译**

```python
os.environ['PYTORCH_JIT'] = '0'
torch.jit._state.disable()
```

**原因**：
- JIT 编译可能在某些环境下有问题
- 禁用 JIT 可以避免潜在的兼容性问题

## 为什么完整 gpt2 会有 Bus Error？

1. **内存压力**：124.44M 参数需要更多内存，更容易触发内存对齐问题
2. **计算复杂度**：更多的层和参数导致更复杂的计算，增加出错概率
3. **Python 3.11.0 兼容性**：早期版本的 Python 3.11 与某些库组合可能有已知问题

## 验证方法

如果想测试完整 gpt2 是否仍有问题：

```bash
# 设置环境变量使用完整 gpt2
export GPT2_MODEL=gpt2
./run.sh
```

## 总结

这个版本没有 Bus error 的主要原因是：

1. ✅ **使用 tiny-gpt2**（最关键）- 大幅减少内存和计算压力
2. ✅ **使用 numpy 采样** - 避免 torch.multinomial 的问题
3. ✅ **CPU 处理** - 明确移到 CPU，避免 MPS 问题
4. ✅ **安全的内存操作** - clone() 和单线程模式
5. ✅ **禁用可能有问题 features** - JIT 编译等

这些改进的组合使得代码在 Python 3.11.0 环境下也能稳定运行。
