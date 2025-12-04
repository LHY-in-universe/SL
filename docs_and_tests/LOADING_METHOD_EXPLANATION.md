# 模型加载方式详解

## 当前使用的加载方式

### **传统加载方式（Traditional Loading）**

这是当前代码使用的默认加载方式。

#### 特点

1. **默认参数**
   ```python
   low_memory = False  # 默认值
   verbose = False     # 默认值
   ```

2. **加载策略选择逻辑**
   ```python
   if is_sharded and low_memory:
       # 增量加载（低内存模式）
       return ModelFactory._create_split_models_incremental(...)
   else:
       # 传统加载（默认）
       return ModelFactory._create_split_models_traditional(...)
   ```

3. **对于 GPT-2 模型**
   - GPT-2 通常**不是分片模型**（`is_sharded = False`）
   - 即使 `low_memory=False`，也会使用传统加载
   - **结论：GPT-2 使用传统加载方式**

---

## 传统加载方式详细流程

### 步骤 1: 加载完整模型 ⏱️ **最耗时**

```python
full_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
```

**这个过程包括：**
- 从 HuggingFace Hub 下载模型（如果不在缓存中）
- 加载所有模型权重到内存
- 构建完整的模型结构

**耗时：**
- 首次下载：1-5 分钟（取决于网络速度）
- 已缓存：10-30 秒

**内存占用：**
- 完整模型的所有参数都在内存中
- GPT-2: 约 500MB

### 步骤 2: 获取状态字典

```python
full_state_dict = full_model.state_dict()
```

**作用：**
- 提取所有层的权重
- 准备用于拆分

### 步骤 3: 创建 Bottom 模型

```python
bottom_model = BottomCls.from_pretrained_split(
    full_state_dict, config, end_layer=split_point_1
)
```

**包含的层：**
- 层 0 到 `split_point_1-1`
- 对于 `split_point_1=2`：包含层 0 和 1

### 步骤 4: 创建 Trunk 模型

```python
trunk_model = TrunkCls.from_pretrained_split(
    full_state_dict, config,
    start_layer=split_point_1,
    end_layer=split_point_2
)
```

**包含的层：**
- 层 `split_point_1` 到 `split_point_2-1`
- 对于 `split_point_1=2, split_point_2=10`：包含层 2 到 9

### 步骤 5: 创建 Top 模型

```python
top_model = TopCls.from_pretrained_split(
    full_state_dict, config, start_layer=split_point_2
)
```

**包含的层：**
- 层 `split_point_2` 到最后一层
- 对于 `split_point_2=10`：包含层 10 和 11

### 步骤 6: 移动到设备

```python
bottom_model = bottom_model.to(device).eval()
trunk_model = trunk_model.to(device).eval()
top_model = top_model.to(device).eval()
```

### 步骤 7: 清理完整模型

```python
del full_model
gc.collect()
```

**目的：**
- 释放完整模型占用的内存
- 只保留拆分后的三个子模型

---

## 两种加载方式对比

### 传统加载（Traditional Loading）

**优点：**
- ✅ 简单直接
- ✅ 适用于小到中等模型（如 GPT-2）
- ✅ 不需要分片模型
- ✅ 加载速度快（对于已缓存的模型）

**缺点：**
- ❌ 需要同时加载完整模型到内存
- ❌ 内存占用峰值高
- ❌ 不适合超大模型（如 7B+ 参数）

**适用场景：**
- GPT-2（124M 参数）
- 其他小到中等模型
- 非分片模型

### 增量加载（Incremental Loading）

**优点：**
- ✅ 内存占用低
- ✅ 适合超大模型
- ✅ 可以按需加载分片

**缺点：**
- ❌ 只适用于分片模型
- ❌ 实现更复杂
- ❌ 加载时间可能更长

**适用场景：**
- 大模型（7B+ 参数）
- 分片模型（如 Qwen2-7B）
- 内存受限环境

---

## 当前配置总结

### 你的测试配置

```python
model_config = ModelConfig(
    model_id="gpt2_trunk_test",
    model_path="gpt2",
    model_type="gpt2",
    device="cpu",
    config={
        "component": "trunk",
        "split_points": [2, 10],
        "cache_dir": "./models"
    }
)
```

### 实际使用的加载方式

1. **加载方式：传统加载（Traditional Loading）**
   - 原因：GPT-2 不是分片模型，且 `low_memory=False`（默认）

2. **加载流程：**
   ```
   AutoModelForCausalLM.from_pretrained("gpt2")
     ↓
   加载完整 GPT-2 模型（约 500MB）
     ↓
   拆分为 Bottom(层0-1) / Trunk(层2-9) / Top(层10-11)
     ↓
   返回 Trunk 模型（你配置的 component）
     ↓
   删除完整模型，释放内存
   ```

3. **耗时分布：**
   - 下载/加载完整模型：10-30 秒（已缓存）或 1-5 分钟（首次）
   - 拆分模型：5-15 秒
   - 移动到设备：1-5 秒
   - **总计：约 20-50 秒（已缓存）或 2-6 分钟（首次）**

---

## 如何切换到增量加载？

如果你想使用增量加载（对于大模型），需要：

```python
# 在调用 load_split_model 时
bottom, trunk, top = load_split_model(
    model_type="qwen2",
    split_points=[8, 24],
    model_name_or_path="Qwen/Qwen2-7B",
    low_memory=True,  # 启用低内存模式
    verbose=True      # 显示详细进度
)
```

**注意：**
- 增量加载只对**分片模型**有效
- GPT-2 不是分片模型，即使设置 `low_memory=True` 也会使用传统加载


