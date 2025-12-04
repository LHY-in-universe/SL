# 模型加载完整流程详解

## 整体架构

```
用户调用
  ↓
AsyncModelManager.load_model()
  ↓
ModelLoader.load_from_config()
  ↓
SplitLearnCore.load_split_model()
  ↓
ModelFactory.create_split_models()
  ↓
HuggingFace.from_pretrained()
  ↓
模型拆分和返回
```

## 详细流程

### 阶段 1: 入口 - AsyncModelManager.load_model()

**位置**: `SplitLearnManager/src/splitlearn_manager/core/async_model_manager.py`

**步骤**:

1. **验证配置** (锁外，快速)
   ```python
   config.validate()  # 检查配置是否有效
   ```

2. **短暂持锁，添加占位符** (<1ms)
   ```python
   async with self.lock:
       # 检查是否已加载
       if config.model_id in self.models:
           raise ValueError("Model already loaded")
       
       # 检查资源
       if not resource_manager.check_available_resources():
           raise RuntimeError("Insufficient resources")
       
       # 创建占位符
       placeholder = LoadingPlaceholder(model_id, config)
       self.models[model_id] = placeholder
   ```
   **目的**: 防止重复加载，标记模型正在加载中

3. **锁外加载模型** (耗时操作，几秒到几分钟)
   ```python
   # 在 ThreadPoolExecutor 中执行（不阻塞事件循环）
   loop = asyncio.get_event_loop()
   model = await loop.run_in_executor(
       self.executor,
       self.loader.load_from_config,  # 调用 ModelLoader
       config
   )
   ```
   **关键**: 使用线程池执行，不阻塞 asyncio 事件循环

4. **短暂持锁，更新为真实模型** (<1ms)
   ```python
   async with self.lock:
       managed_model = AsyncManagedModel(model_id, model, config)
       self.models[model_id] = managed_model  # 替换占位符
   ```

---

### 阶段 2: ModelLoader.load_from_config()

**位置**: `SplitLearnManager/src/splitlearn_manager/core/model_loader.py`

**步骤**:

1. **配置 PyTorch 线程数**
   ```python
   configure_pytorch_threads(single_threaded=True)
   # 设置 OMP_NUM_THREADS=1, MKL_NUM_THREADS=1
   # 设置 torch.set_num_threads(1)
   ```

2. **根据 model_type 选择加载方式**
   ```python
   if config.model_type == "pytorch":
       # 从 .pt/.pth 文件加载
       model = ModelLoader.load_pytorch_checkpoint(...)
   
   elif config.model_type == "huggingface":
       # 直接加载 HuggingFace 模型
       model = ModelLoader._load_huggingface_model(...)
   
   elif config.model_type in ["gpt2", "qwen2", "gemma"]:
       # 使用 SplitLearnCore 加载分割模型
       model = ModelLoader._load_split_model(...)
   ```

3. **Warmup（可选）**
   ```python
   if config.warmup:
       ModelLoader._warmup_model(model, config)
       # 运行几次前向传播来初始化 CUDA kernels
   ```

---

### 阶段 3: ModelLoader._load_split_model()

**位置**: `SplitLearnManager/src/splitlearn_manager/core/model_loader.py`

**步骤**:

1. **调用 SplitLearnCore**
   ```python
   from splitlearn_core.quickstart import load_split_model
   
   bottom, trunk, top = load_split_model(
       model_type=config.model_type,      # "gpt2"
       split_points=split_points,          # [2, 10]
       cache_dir=cache_dir,                # "./models"
       device=config.device                # "cpu" or "cuda"
   )
   ```

2. **根据 component 返回对应模型**
   ```python
   if component == "bottom":
       return bottom
   elif component == "trunk":
       return trunk  # 服务器端使用
   elif component == "top":
       return top
   ```

---

### 阶段 4: SplitLearnCore.load_split_model()

**位置**: `SplitLearnCore/src/splitlearn_core/quickstart.py`

**步骤**:

1. **配置 PyTorch 线程数**（再次确保）
   ```python
   os.environ.setdefault('OMP_NUM_THREADS', '1')
   torch.set_num_threads(1)
   ```

2. **调用 ModelFactory**
   ```python
   bottom, trunk, top = ModelFactory.create_split_models(
       model_type="gpt2",
       model_name_or_path="gpt2",
       split_point_1=2,
       split_point_2=10,
       device="cpu"
   )
   ```

---

### 阶段 5: ModelFactory.create_split_models()

**位置**: `SplitLearnCore/src/splitlearn_core/factory.py`

**步骤**:

1. **验证模型类型**
   ```python
   if not ModelRegistry.is_model_registered(model_type):
       raise KeyError("Model type not registered")
   ```

2. **配置 PyTorch 线程数**（在 HuggingFace 加载之前）
   ```python
   _configure_pytorch_threads_for_loading()
   ```

3. **加载模型配置**（轻量级）
   ```python
   config = AutoConfig.from_pretrained(model_name_or_path)
   # 只加载配置，不加载权重
   ```

4. **检测模型是否分片**
   ```python
   is_sharded = ShardLoader.is_sharded_model(model_name_or_path)
   ```

5. **选择加载策略**
   ```python
   if is_sharded and low_memory:
       # 增量加载（低内存模式）
       return ModelFactory._create_split_models_incremental(...)
   else:
       # 传统加载（加载完整模型然后拆分）
       return ModelFactory._create_split_models_traditional(...)
   ```

---

### 阶段 6: ModelFactory._create_split_models_traditional()

**位置**: `SplitLearnCore/src/splitlearn_core/factory.py`

**步骤**（这是最耗时的部分）:

1. **加载完整模型** ⏱️ **最耗时**
   ```python
   full_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
   # HuggingFace 会：
   # - 从缓存或网络下载模型
   # - 加载所有权重到内存
   # - 构建完整的模型结构
   ```
   **耗时**: 几秒到几分钟（取决于模型大小和网络速度）

2. **获取模型状态字典**
   ```python
   full_state_dict = full_model.state_dict()
   ```

3. **获取层数**
   ```python
   num_layers = config.n_layer  # GPT-2 通常是 12
   ```

4. **验证拆分点**
   ```python
   if not (0 < split_point_1 < split_point_2 < num_layers):
       raise ValueError("Invalid split points")
   ```

5. **从注册表获取模型类**
   ```python
   BottomCls = ModelRegistry.get_model_class("gpt2", "bottom")
   TrunkCls = ModelRegistry.get_model_class("gpt2", "trunk")
   TopCls = ModelRegistry.get_model_class("gpt2", "top")
   ```

6. **创建 Bottom 模型**
   ```python
   bottom_model = BottomCls.from_pretrained_split(
       full_state_dict, config, end_layer=split_point_1
   )
   # 只包含层 0 到 split_point_1-1
   ```

7. **创建 Trunk 模型**
   ```python
   trunk_model = TrunkCls.from_pretrained_split(
       full_state_dict, config,
       start_layer=split_point_1, end_layer=split_point_2
   )
   # 包含层 split_point_1 到 split_point_2-1
   ```

8. **创建 Top 模型**
   ```python
   top_model = TopCls.from_pretrained_split(
       full_state_dict, config, start_layer=split_point_2
   )
   # 包含层 split_point_2 到 num_layers-1
   ```

9. **移动到设备并设置为评估模式**
   ```python
   bottom_model = bottom_model.to(device).eval()
   trunk_model = trunk_model.to(device).eval()
   top_model = top_model.to(device).eval()
   ```

10. **保存（可选）**
    ```python
    if auto_save and storage_path:
        torch.save(trunk_model, f"{storage_path}/trunk.pt")
    ```

11. **清理完整模型**（释放内存）
    ```python
    del full_model
    gc.collect()
    ```

---

## 关键时间点

### 耗时操作（按顺序）

1. **HuggingFace.from_pretrained()** - 最耗时 ⏱️⏱️⏱️
   - 下载模型（如果不在缓存中）: 几十秒到几分钟
   - 加载权重到内存: 几秒到几十秒
   - 总耗时: **1-5 分钟**（首次）或 **10-30 秒**（已缓存）

2. **模型拆分** - 中等耗时 ⏱️⏱️
   - 创建三个子模型: **5-15 秒**
   - 移动权重到设备: **1-5 秒**

3. **其他操作** - 快速 ⏱️
   - 配置验证: <1ms
   - 占位符创建: <1ms
   - 锁操作: <1ms

---

## 为什么看不到模型？

可能的原因：

1. **模型还在加载中**
   - HuggingFace 下载/加载需要时间
   - 首次加载可能需要 1-5 分钟
   - 检查日志中的 "Loading pretrained model..." 消息

2. **加载卡住了**
   - 网络问题（下载模型）
   - 内存不足
   - 线程数设置问题

3. **加载失败但没有显示错误**
   - 检查异常处理
   - 查看完整日志

4. **模型加载成功但不在预期位置**
   - HuggingFace 默认缓存: `~/.cache/huggingface/`
   - 本地缓存: `./models/`（如果设置了 cache_dir）

---

## 调试建议

1. **增加日志级别**
   ```python
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **检查加载进度**
   - 查看 "Loading pretrained model..." 消息
   - 查看 "Splitting model..." 消息
   - 查看 "Creating Bottom/Trunk/Top model..." 消息

3. **检查缓存**
   ```bash
   ls -lh ~/.cache/huggingface/hub/models--gpt2/
   ls -lh ./models/
   ```

4. **增加超时时间**
   ```python
   result = await asyncio.wait_for(
       manager.load_model(config),
       timeout=600.0  # 10 分钟
   )
   ```


