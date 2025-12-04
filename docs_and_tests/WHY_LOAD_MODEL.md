# 为什么需要加载模型？

## 📁 .pt 文件是什么？

`.pt` 文件是**保存在磁盘上的模型文件**，它包含：
- 模型的权重（weights）
- 模型的结构（architecture）
- 其他元数据

**但是**，`.pt` 文件只是磁盘上的数据，**不能直接用于计算**。

## 🔄 加载模型的过程

即使 `.pt` 文件已经存在，要使用模型也必须执行以下步骤：

```python
# 1. 从磁盘读取文件到内存
model = torch.load("model.pt", map_location='cpu')

# 2. 设置为评估模式
model.eval()

# 3. 现在模型在内存中，可以用于计算了
output = model(input_tensor)
```

## 💡 类比理解

| 概念 | 类比 |
|------|------|
| `.pt` 文件 | 书在书架上（不能直接读） |
| `torch.load()` | 把书从书架拿到桌子上 |
| 使用模型 | 阅读书的内容 |

## ✅ 总结

- **`.pt` 文件** = 磁盘上的文件（不能直接计算）
- **`torch.load()`** = 加载到内存（可以计算）
- **即使文件存在，也必须加载才能使用！**

## 🎯 在服务器中的使用

在 `server_async_with_model.py` 中：

```python
# 1. 检查文件是否存在
if not os.path.exists(MODEL_PATH):
    print("模型文件不存在")
    return

# 2. 加载模型到内存（这一步是必须的！）
model = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
model.eval()

# 3. 现在模型可以在服务器中使用了
compute_fn = AsyncModelComputeFunction(model=model, ...)
```

**所以，即使有现成的 `.pt` 文件，也需要执行 `torch.load()` 将其加载到内存中才能使用！**

