# Bus Error 原因分析

## 问题现象

- ✅ 模型加载成功（3.75秒，124.44M参数）
- ✅ 输入编码成功
- ❌ 模型推理时出现 Bus error (10)

## 可能原因

### 1. **Python 3.11.0 与 PyTorch 2.9.1 的兼容性问题**
   - Python 3.11.0 是早期版本，可能存在兼容性问题
   - PyTorch 2.9.1 可能与 Python 3.11.0 存在已知问题

### 2. **MPS (Metal Performance Shaders) 相关问题**
   - 虽然强制使用 CPU，但 MPS 后端可能仍然被初始化
   - macOS 上的 MPS 与某些操作不兼容

### 3. **内存对齐问题**
   - `torch.cat` 操作可能导致内存对齐问题
   - 模型推理时的内存分配可能有问题

### 4. **多线程竞争**
   - 即使设置了单线程，某些底层库可能仍使用多线程
   - BLAS/MKL 库的线程设置可能不生效

### 5. **模型推理时的底层操作**
   - GPT-2 的某些操作（如 attention）可能在特定环境下有问题
   - `torch.inference_mode()` 可能与某些操作不兼容

## 解决方案

### 方案1: 使用更稳定的 Python 版本
```bash
# 使用 Python 3.11.5+ 或 3.12
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m venv venv
```

### 方案2: 降级 PyTorch 版本
```bash
pip install torch==2.0.1 transformers==4.30.0
```

### 方案3: 使用更安全的生成方式
- 避免使用 `torch.cat`，改用列表收集后一次性转换
- 使用 `torch.no_grad()` 而不是 `torch.inference_mode()`
- 限制序列长度，避免内存问题

### 方案4: 完全禁用 MPS
```python
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
# 在导入 torch 之前设置
```

### 方案5: 使用 tiny-gpt2 进行测试
```python
model_name = "sshleifer/tiny-gpt2"  # 更小的模型，更稳定
```

## 推荐方案

1. **短期**: 使用 `sshleifer/tiny-gpt2` 进行测试
2. **中期**: 升级 Python 到 3.11.5+ 或 3.12
3. **长期**: 考虑使用更稳定的 PyTorch 版本（2.0.x）

