# SplitLearn 剩余实施计划

**当前状态**: Phase 3 已完成 ✅
**更新日期**: 2025-12-09
**剩余阶段**: Phase 4 和 Phase 5

---

## 📊 整体进度概览

| 阶段 | 状态 | 完成度 | 预计时间 |
|------|------|--------|---------|
| Phase 1: Protocol Foundation | ✅ 完成 | 100% | 已完成 |
| Phase 2: LoRA Integration | ✅ 完成 | 100% | 已完成 |
| Phase 3: End-to-End Training | ✅ 完成 | 100% | 已完成 |
| **Phase 4: KV-Cache & Optimization** | ⏳ 待开始 | 0% | 2-3周 |
| **Phase 5: Production Hardening** | ⏳ 待开始 | 0% | 2-3周 |

---

## ⏳ Phase 4: KV-Cache & Optimization (第7-8周)

**目标**: 实现快速推理和硬件加速，大幅提升性能

### 优先级排序

#### 🔴 高优先级：KV-Cache 分离（CRITICAL for inference）

**为什么重要**:
- 自回归生成（文本生成、对话）必需
- 没有 KV-cache，生成速度极慢（100x 慢）
- **30x 推理加速**
- **97% 带宽减少**（生成 100 token: 30 MB → 4 MB）

**任务列表**:

1. **实现 KV-Cache 在 Trunk 模型中** ⏳
   - **文件**: `SplitLearnCore/src/splitlearn_core/core/trunk.py`
   - **修改位置**: `SplitTrunkModel.forward()` 方法（lines 36-78）
   - **需要添加**:
     ```python
     def forward(
         self,
         hidden_states: torch.Tensor,
         attention_mask: Optional[torch.Tensor] = None,
         past_key_values: Optional[Tuple] = None,  # NEW
         use_cache: bool = False  # NEW
     ) -> Union[torch.Tensor, Tuple]:
         # Process through layers with KV-cache
         present_key_values = []
         for i, layer in enumerate(self.layers):
             past_kv = past_key_values[i] if past_key_values else None
             layer_output = layer(
                 hidden_states,
                 attention_mask=attention_mask,
                 past_key_value=past_kv,
                 use_cache=use_cache
             )
             hidden_states = layer_output[0]
             if use_cache:
                 present_key_values.append(layer_output[1])

         if use_cache:
             return hidden_states, tuple(present_key_values)
         return hidden_states
     ```

2. **在 gRPC Servicer 中传输 KV-Cache** ⏳
   - **文件**: `SplitLearnComm/src/splitlearn_comm/server/servicer.py`
   - **修改位置**: `Compute` RPC handler（lines 120-232）
   - **需要添加**:
     - 从 `ComputeRequest` 解码 `past_key_values`
     - 调用 `compute_fn` 时传递 KV-cache
     - 将 `present_key_values` 编码到 `ComputeResponse`
   - **协议已定义**: `compute_service.proto` 中的 `KVCacheEntry` 消息

3. **在客户端支持 KV-Cache** ⏳
   - **文件**: `SplitLearnComm/src/splitlearn_comm/client/grpc_client.py`
   - **修改位置**: `compute()` 方法（lines 128-208）
   - **需要添加**:
     ```python
     def compute(
         self,
         input_tensor: torch.Tensor,
         past_key_values: Optional[Tuple] = None,  # NEW
         use_cache: bool = False  # NEW
     ) -> Union[torch.Tensor, Tuple]:
         # Encode past_key_values to protobuf
         # Call RPC with KV-cache
         # Decode and return present_key_values
     ```

4. **实现增量生成示例** ⏳
   - **新文件**: `SplitLearnCore/examples/generate_with_cache.py`
   - **功能**:
     - Token-by-token 生成
     - 重用 KV-cache
     - 性能对比（有/无 cache）
   - **示例代码**:
     ```python
     def generate_with_cache(prompt_ids, max_new_tokens=100):
         # First token (no cache)
         hidden_1 = bottom_model(prompt_ids)
         response = trunk_client.compute(hidden_1, use_cache=True)
         hidden_2, trunk_kv_cache = response

         output = top_model(hidden_2, use_cache=True)
         top_kv_cache = output.past_key_values
         generated_ids = [output.logits.argmax()]

         # Remaining tokens (with cache)
         for _ in range(max_new_tokens - 1):
             new_hidden_1 = bottom_model(generated_ids[-1:])
             response = trunk_client.compute(
                 new_hidden_1,
                 past_key_values=trunk_kv_cache,
                 use_cache=True
             )
             hidden_2, trunk_kv_cache = response

             output = top_model(
                 hidden_2,
                 past_key_values=top_kv_cache,
                 use_cache=True
             )
             top_kv_cache = output.past_key_values
             generated_ids.append(output.logits.argmax())

         return generated_ids
     ```

5. **KV-Cache 性能测试** ⏳
   - 对比生成速度（有/无 cache）
   - 测量网络带宽使用
   - 验证 30x 加速目标

**预期结果**:
- ✅ 生成 100 token: 从 120 MB → 4 MB 带宽
- ✅ 推理速度提升 30x
- ✅ 实用的文本生成功能

---

#### 🟡 中优先级：硬件加速

##### 4.1 启用 FlashAttention/SDPA

**为什么重要**:
- **1.5-2x 训练加速**
- **25-44% 内存减少**
- 完全支持 autograd（训练和推理都能用）

**任务列表**:

1. **在基础模型配置中启用 SDPA** ⏳
   - **文件**: `SplitLearnCore/src/splitlearn_core/core/base.py`
   - **修改位置**: 模型初始化（line 46 附近）
   - **需要添加**:
     ```python
     # In model config
     config._attn_implementation = "flash_attention_2"  # If available
     # Fallback to SDPA if FlashAttention not installed
     if not is_flash_attn_available():
         config._attn_implementation = "sdpa"
     ```

2. **在 ModelFactory 中添加配置选项** ⏳
   - **文件**: `SplitLearnCore/src/splitlearn_core/factory.py`
   - **添加参数**: `use_flash_attention: bool = True`
   - **自动检测**: Flash Attention 2 是否可用
   - **降级策略**: Flash Attention 2 → SDPA → Eager

3. **验证加速效果** ⏳
   - 对比训练速度（Eager vs SDPA vs Flash Attention 2）
   - 测量内存使用
   - 确认梯度正确性

**预期结果**:
- ✅ 训练速度提升 1.5-2x
- ✅ VRAM 使用减少 25-44%
- ✅ 梯度流正确

##### 4.2 torch.compile 支持

**为什么重要**:
- PyTorch 2.0+ 的编译优化
- 额外 10-30% 加速
- 减少 Python 开销

**任务列表**:

1. **为 Bottom/Top 模型添加 compile** ⏳
   - **文件**: `SplitLearnCore/src/splitlearn_core/training/trainer.py`
   - **修改位置**: `__init__` 方法
   - **需要添加**:
     ```python
     if self.config.use_torch_compile:
         logger.info("Compiling models with torch.compile...")
         self.bottom_model = torch.compile(
             self.bottom_model,
             mode="reduce-overhead"  # Client side
         )
         self.top_model = torch.compile(
             self.top_model,
             mode="reduce-overhead"
         )
     ```

2. **为 Trunk 模型添加 compile** ⏳
   - **文件**: `SplitLearnComm/src/splitlearn_comm/server/servicer.py`
   - **修改位置**: `__init__` 方法
   - **编译模式**: `mode="max-autotune"` (服务器有更多优化时间)

3. **处理编译限制** ⏳
   - 避免在编译函数中使用动态控制流
   - 处理不支持的操作（fallback）
   - 添加编译缓存预热

**预期结果**:
- ✅ 额外 10-30% 加速
- ✅ 首次运行后稳定性能
- ✅ 与 SDPA 叠加效果

---

#### 🟢 低优先级：高级训练功能

##### 4.3 服务器端梯度累积

**任务列表**:

1. **在 Trunk Servicer 中实现累积** ⏳
   - **文件**: `SplitLearnComm/src/splitlearn_comm/server/servicer.py`
   - **功能**:
     - 累积多个 backward 请求的梯度
     - 在 `accumulation_step` 达到阈值时更新
     - 返回是否执行了更新的标志

2. **客户端协调** ⏳
   - **文件**: `SplitLearnCore/src/splitlearn_core/training/trainer.py`
   - **功能**:
     - 在 backward RPC 中传递 `accumulation_step`
     - 同步客户端和服务器的累积步数

**预期结果**:
- ✅ 支持更大的有效 batch size
- ✅ 训练更稳定

##### 4.4 混合精度训练验证

**任务列表**:

1. **端到端测试混合精度** ⏳
   - 验证 GradScaler 正确工作
   - 测试 FP16/BF16 梯度通信
   - 确认没有精度损失导致的训练问题

2. **自动精度选择** ⏳
   - 检测 GPU 支持（BF16 vs FP16）
   - Ampere+ GPU 优先使用 BF16
   - 旧 GPU 使用 FP16

**预期结果**:
- ✅ 混合精度稳定训练
- ✅ 内存使用减半

---

### Phase 4 验收标准

完成 Phase 4 需满足：

1. ✅ **KV-Cache 分离实现**
   - 生成 100 token 带宽从 30 MB → 4 MB
   - 推理速度提升 >25x（目标 30x）
   - 支持增量生成

2. ✅ **SDPA/FlashAttention 启用**
   - 训练速度提升 >1.3x（目标 1.5-2x）
   - 内存使用减少 >20%（目标 25-44%）
   - 梯度正确性验证通过

3. ✅ **torch.compile 集成**
   - 额外 >5% 加速（目标 10-30%）
   - 稳定运行无崩溃

4. ✅ **完整示例和文档**
   - 生成示例（`generate_with_cache.py`）
   - 性能对比文档
   - 配置指南

---

## ⏳ Phase 5: Production Hardening (第9-10周)

**目标**: 生产环境部署准备，提升鲁棒性和可用性

### 优先级排序

#### 🔴 高优先级：错误处理和恢复

##### 5.1 网络故障处理

**任务列表**:

1. **增强 Retry 策略** ⏳
   - **文件**: `SplitLearnComm/src/splitlearn_comm/client/grpc_client.py`
   - **功能**:
     - 指数退避重试
     - 区分瞬时错误和永久错误
     - 超时配置（forward vs backward）
     - 重试次数限制

2. **中途恢复机制** ⏳
   - **文件**: `SplitLearnCore/src/splitlearn_core/training/trainer.py`
   - **功能**:
     - 检测网络中断
     - 自动保存检查点
     - 重新连接后恢复训练
     - 跳过失败的 batch（可选）

3. **激活缓存过期处理** ⏳
   - **文件**: `SplitLearnComm/src/splitlearn_comm/server/activation_cache.py`
   - **功能**:
     - 客户端检测过期错误
     - 自动重新执行 forward pass
     - 日志记录过期事件

**预期结果**:
- ✅ 训练可从网络中断中恢复
- ✅ 减少因瞬时错误导致的失败

##### 5.2 梯度异常检测

**任务列表**:

1. **梯度爆炸/消失检测** ⏳
   - **文件**: `SplitLearnCore/src/splitlearn_core/training/trainer.py`
   - **功能**:
     - 监控梯度范数
     - 检测 NaN/Inf
     - 自动调整学习率或跳过 batch

2. **梯度过时检测** ⏳
   - **功能**:
     - 时间戳验证
     - 模型版本追踪
     - 拒绝过时梯度

**预期结果**:
- ✅ 训练不会因梯度异常而崩溃
- ✅ 自动异常恢复

---

#### 🟡 中优先级：多客户端训练

##### 5.3 多客户端协调

**架构变更**:

```
多客户端架构:
┌─────────────┐
│  Client 1   │ ──┐
│  (Bottom+Top)  │  │
└─────────────┘  │
                 ├─→ ┌─────────────┐
┌─────────────┐  │   │   Server    │
│  Client 2   │ ──┤   │   (Trunk)   │
│  (Bottom+Top)  │  │   │ + Aggregator│
└─────────────┘  │   └─────────────┘
                 │
┌─────────────┐  │
│  Client 3   │ ──┘
│  (Bottom+Top)  │
└─────────────┘
```

**任务列表**:

1. **梯度聚合器** ⏳
   - **新文件**: `SplitLearnComm/src/splitlearn_comm/server/gradient_aggregator.py`
   - **功能**:
     - 收集多个客户端的梯度
     - 平均或加权聚合
     - 同步/异步更新策略
     - 支持 Federated Learning 算法

2. **客户端标识和状态管理** ⏳
   - **文件**: `SplitLearnComm/src/splitlearn_comm/server/servicer.py`
   - **功能**:
     - 为每个客户端分配唯一 ID
     - 追踪客户端训练状态
     - 处理客户端加入/离开

3. **数据分片策略** ⏳
   - **文件**: `SplitLearnCore/src/splitlearn_core/utils/shard_loader.py` (已存在)
   - **功能**:
     - 为不同客户端分配不同数据分片
     - 避免数据重叠
     - 支持 IID/Non-IID 分布

4. **同步策略** ⏳
   - **同步 SGD**: 等待所有客户端完成 batch
   - **异步 SGD**: 立即更新，不等待其他客户端
   - **半异步**: 等待部分客户端（quorum）

**预期结果**:
- ✅ 支持 2-10 个客户端同时训练
- ✅ 梯度正确聚合
- ✅ 训练收敛性验证

---

#### 🟢 低优先级：监控和日志

##### 5.4 分布式监控

**任务列表**:

1. **服务器端监控** ⏳
   - **新文件**: `SplitLearnComm/src/splitlearn_comm/server/monitor.py`
   - **指标**:
     - 请求吞吐量（requests/sec）
     - 延迟分布（forward/backward）
     - 激活缓存命中率
     - GPU 利用率
     - 内存使用

2. **客户端监控** ⏳
   - **文件**: `SplitLearnCore/src/splitlearn_core/training/trainer.py`
   - **指标**:
     - 训练速度（samples/sec）
     - 损失曲线
     - 梯度范数
     - 学习率变化
     - 网络延迟

3. **集成监控系统** ⏳
   - **Prometheus**: 指标收集
   - **Grafana**: 可视化仪表板
   - **TensorBoard**: 训练曲线

**预期结果**:
- ✅ 实时监控训练状态
- ✅ 性能瓶颈识别
- ✅ 异常告警

##### 5.5 结构化日志

**任务列表**:

1. **统一日志格式** ⏳
   - JSON 格式日志
   - 包含上下文（client_id, forward_id, step）
   - 日志级别分层（DEBUG/INFO/WARNING/ERROR）

2. **日志聚合** ⏳
   - 客户端和服务器日志统一收集
   - 支持 ELK Stack (Elasticsearch, Logstash, Kibana)
   - 日志搜索和分析

**预期结果**:
- ✅ 问题快速定位
- ✅ 训练过程可追溯

---

#### 🟢 低优先级：性能优化

##### 5.6 网络传输优化

**任务列表**:

1. **梯度压缩** ⏳
   - **新文件**: `SplitLearnComm/src/splitlearn_comm/core/compression.py`
   - **方法**:
     - Top-K 稀疏化（只传输最大的 K 个梯度）
     - 量化（FP32 → INT8/INT16）
     - 误差补偿（Error Feedback）
   - **预期**: 带宽减少 5-10x

2. **异步通信** ⏳
   - **文件**: `SplitLearnCore/src/splitlearn_core/training/trainer.py`
   - **功能**:
     - Pipeline 并行（forward 和 backward 重叠）
     - Prefetch 下一个 batch
     - 异步 optimizer step

3. **批量请求** ⏳
   - 多个小 batch 合并成一个大 batch
   - 减少 RPC 往返次数
   - 提高吞吐量

**预期结果**:
- ✅ 训练速度提升 20-50%
- ✅ 网络带宽使用减少 5-10x
- ✅ GPU 利用率提升

##### 5.7 内存优化

**任务列表**:

1. **激活检查点 (Activation Checkpointing)** ⏳
   - **文件**: `SplitLearnCore/src/splitlearn_core/core/trunk.py`
   - **功能**:
     - 不保存所有层的激活
     - 反向传播时重新计算
     - 节省 50% 内存，增加 20% 计算

2. **梯度检查点 (Gradient Checkpointing)** ⏳
   - 对部分层应用检查点
   - 平衡内存和计算

**预期结果**:
- ✅ 支持更大的模型
- ✅ 内存使用减少 30-50%

---

### Phase 5 验收标准

完成 Phase 5 需满足：

1. ✅ **鲁棒性**
   - 网络中断后自动恢复
   - 梯度异常自动处理
   - 48小时连续训练无崩溃

2. ✅ **多客户端支持**
   - 支持至少 5 个客户端同时训练
   - 梯度正确聚合
   - 收敛性不劣于单客户端

3. ✅ **性能指标**
   - 吞吐量 >100 requests/sec
   - 端到端延迟 <100ms (LAN)
   - GPU 利用率 >80%

4. ✅ **监控和日志**
   - 实时监控仪表板
   - 结构化日志
   - 异常告警机制

5. ✅ **文档完整**
   - 部署指南
   - 故障排查手册
   - API 参考文档
   - 最佳实践

---

## 📁 预期新增文件

### Phase 4 新增文件

```
SplitLearnCore/
├── examples/
│   ├── generate_with_cache.py          # KV-cache 生成示例
│   └── benchmark_acceleration.py       # 硬件加速性能测试
└── docs/
    ├── KV_CACHE_GUIDE.md               # KV-cache 使用指南
    └── ACCELERATION_GUIDE.md           # 硬件加速配置指南

SplitLearnComm/
└── src/splitlearn_comm/
    └── core/
        └── kv_cache_codec.py           # KV-cache 序列化
```

### Phase 5 新增文件

```
SplitLearnCore/
├── src/splitlearn_core/
│   └── training/
│       ├── gradient_aggregator.py      # 梯度聚合器
│       └── monitor.py                  # 训练监控
├── examples/
│   ├── multi_client_training.py        # 多客户端训练示例
│   └── federated_learning.py          # 联邦学习示例
└── docs/
    ├── MULTI_CLIENT_GUIDE.md           # 多客户端训练指南
    ├── DEPLOYMENT_GUIDE.md             # 部署指南
    ├── TROUBLESHOOTING.md              # 故障排查
    └── API_REFERENCE.md                # API 参考

SplitLearnComm/
└── src/splitlearn_comm/
    ├── server/
    │   ├── gradient_aggregator.py      # 服务器端聚合器
    │   └── client_manager.py           # 客户端管理器
    └── core/
        └── compression.py              # 梯度压缩
```

---

## 🎯 关键里程碑

| 里程碑 | 目标日期 | 验收标准 |
|--------|---------|---------|
| Phase 4 KV-Cache 完成 | +2周 | 推理加速 30x，带宽减少 97% |
| Phase 4 硬件加速完成 | +3周 | 训练加速 1.5x，内存减少 25% |
| Phase 4 完整验收 | +3周 | 所有 Phase 4 标准满足 |
| Phase 5 错误处理完成 | +4周 | 48小时稳定训练 |
| Phase 5 多客户端完成 | +5周 | 5客户端稳定训练 |
| Phase 5 完整验收 | +5周 | 所有 Phase 5 标准满足 |
| **项目完成** | **+5周** | **生产就绪** |

---

## 📊 预期性能提升总结

完成所有阶段后的性能对比：

| 指标 | 当前 (Phase 3) | Phase 4 | Phase 5 | 总提升 |
|------|---------------|---------|---------|--------|
| **训练速度** | 基准 | 1.5-2x | 1.3x | **2-2.6x** |
| **推理速度** | 基准 | 30x | 1.2x | **36x** |
| **内存使用** | 基准 | -30% | -20% | **-50%** |
| **网络带宽 (训练)** | 4.8 MB/step | 持平 | -5x | **-5x** |
| **网络带宽 (推理)** | 30 MB/100 token | -7.5x | -1.5x | **-11x** |
| **稳定性** | 基准 | 持平 | 48h+ | **生产级** |

---

## 🔄 迭代策略

建议按以下顺序实施：

### 短期（未来 1-2周）
1. **Phase 4.1**: KV-Cache 分离（高优先级）
   - Trunk 模型 KV-cache 支持
   - gRPC 传输 KV-cache
   - 生成示例

### 中期（未来 3-4周）
2. **Phase 4.2**: 硬件加速
   - SDPA/FlashAttention
   - torch.compile

3. **Phase 5.1**: 基础错误处理
   - Retry 策略增强
   - 网络故障恢复

### 长期（未来 5周+）
4. **Phase 5.2**: 多客户端训练
   - 梯度聚合
   - 客户端协调

5. **Phase 5.3**: 监控和优化
   - 监控系统
   - 性能优化

---

## 💡 实施建议

1. **优先级原则**: 先实现高价值功能（KV-cache > 硬件加速 > 多客户端）
2. **增量开发**: 每个功能完成后立即测试和验证
3. **向后兼容**: 新功能应该可选，不破坏现有功能
4. **文档同步**: 每个功能完成后立即更新文档
5. **性能基准**: 每个优化后测量实际提升

---

## 📞 决策点

在开始 Phase 4/5 之前，需要确认：

1. **Phase 3 测试**: 是否需要先测试当前的训练实现？
2. **优先级**: KV-cache（推理）vs 硬件加速（训练）哪个更重要？
3. **多客户端**: 是否需要多客户端训练？还是单客户端就够？
4. **部署环境**: 本地测试 vs 生产环境部署？
5. **资源限制**: 可用的 GPU 资源和时间？

---

**下一步行动**:
- [ ] 确认优先级和时间表
- [ ] 开始实施 Phase 4.1 (KV-Cache)
- [ ] 或者先测试 Phase 3 实现

**创建日期**: 2025-12-09
**最后更新**: 2025-12-09
