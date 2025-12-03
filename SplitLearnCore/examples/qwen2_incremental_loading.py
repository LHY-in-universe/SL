"""
Qwen2-7B 增量加载示例

这个示例展示如何使用增量加载功能来显著降低大型分片模型的内存使用。

对于 Qwen2-7B 模型：
- 传统加载方式：峰值内存约 84GB
- 增量加载方式：峰值内存约 32GB (节省 62%)

增量加载特别适用于：
1. 内存有限的环境
2. 多个模型同时运行的场景
3. 边缘设备或消费级硬件
"""

import torch
from splitlearn_core import ModelFactory

# ============================================================================
# 示例 1: 基础增量加载
# ============================================================================

def example_basic_incremental_loading():
    """
    最简单的增量加载示例
    """
    print("\n" + "="*70)
    print("示例 1: 基础增量加载")
    print("="*70 + "\n")

    # 创建分片模型，启用低内存模式
    bottom, trunk, top = ModelFactory.create_split_models(
        model_type='qwen2',
        model_name_or_path='Qwen/Qwen2-7B',
        split_point_1=8,      # 前 8 层作为 Bottom
        split_point_2=24,     # 中间 16 层作为 Trunk
        device='cpu',
        low_memory=True,      # 启用增量加载
        verbose=False,        # 简洁输出
    )

    print(f"✓ Bottom 模型已加载 (层 0-7)")
    print(f"✓ Trunk 模型已加载 (层 8-23)")
    print(f"✓ Top 模型已加载 (层 24-27 + LM Head)")

    # 使用模型进行推理
    batch_size, seq_len = 1, 10
    vocab_size = bottom.model.embed_tokens.weight.shape[0]

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Forward pass through all components
    hidden = bottom(input_ids)
    hidden = trunk(hidden)
    output = top(hidden)

    print(f"\n推理完成:")
    print(f"  输入形状: {input_ids.shape}")
    print(f"  输出形状: {output.shape}")


# ============================================================================
# 示例 2: 详细内存监控
# ============================================================================

def example_memory_monitoring():
    """
    使用 verbose=True 查看详细的内存使用情况
    """
    print("\n" + "="*70)
    print("示例 2: 详细内存监控")
    print("="*70 + "\n")

    bottom, trunk, top = ModelFactory.create_split_models(
        model_type='qwen2',
        model_name_or_path='Qwen/Qwen2-7B',
        split_point_1=8,
        split_point_2=24,
        device='cpu',
        low_memory=True,
        verbose=True,         # 启用详细输出，显示内存使用
    )

    # verbose=True 会自动显示：
    # - 每个组件加载前后的内存使用
    # - 内存增量
    # - 峰值内存使用
    # - 完整的时间线

    print("\n查看上面的内存报告，了解每个阶段的内存使用情况")


# ============================================================================
# 示例 3: 多设备分配
# ============================================================================

def example_device_mapping():
    """
    将不同组件分配到不同设备
    """
    print("\n" + "="*70)
    print("示例 3: 多设备分配")
    print("="*70 + "\n")

    # 检查可用设备
    if torch.cuda.device_count() >= 2:
        # 手动指定设备映射
        device_map = {
            'bottom': 'cpu',         # Bottom 放在 CPU
            'trunk': 'cuda:0',       # Trunk 放在第一个 GPU
            'top': 'cuda:1',         # Top 放在第二个 GPU
        }

        print("检测到多个 GPU，使用手动设备映射:")
        print(f"  Bottom -> CPU")
        print(f"  Trunk  -> GPU 0")
        print(f"  Top    -> GPU 1")

    elif torch.cuda.is_available():
        # 只有一个 GPU，使用自动映射
        device_map = 'auto'  # 会自动将组件分配到 CPU 和 GPU

        print("检测到单个 GPU，使用自动设备映射 (device_map='auto')")

    else:
        # 没有 GPU
        device_map = None
        print("未检测到 GPU，所有组件将在 CPU 上运行")

    bottom, trunk, top = ModelFactory.create_split_models(
        model_type='qwen2',
        model_name_or_path='Qwen/Qwen2-7B',
        split_point_1=8,
        split_point_2=24,
        device='cpu',        # 默认设备
        device_map=device_map,  # 覆盖特定组件的设备
        low_memory=True,
        verbose=False,
    )

    print(f"\n实际设备分配:")
    print(f"  Bottom: {next(bottom.parameters()).device}")
    print(f"  Trunk:  {next(trunk.parameters()).device}")
    print(f"  Top:    {next(top.parameters()).device}")


# ============================================================================
# 示例 4: 传统 vs 增量加载对比
# ============================================================================

def example_comparison():
    """
    对比传统加载和增量加载的内存使用

    注意：这个示例需要足够的内存来运行传统加载方式！
    如果内存不足，请跳过此示例。
    """
    print("\n" + "="*70)
    print("示例 4: 传统 vs 增量加载对比")
    print("="*70 + "\n")

    import gc
    from splitlearn_core.utils import MemoryTracker

    tracker = MemoryTracker()

    print("⚠️  注意：这个对比需要约 84GB 内存来运行传统加载")
    print("如果内存不足，建议跳过此示例\n")

    user_input = input("是否继续？(y/N): ").strip().lower()
    if user_input != 'y':
        print("已跳过对比示例")
        return

    # 方式 1: 传统加载
    print("\n--- 传统加载方式 ---")
    tracker.snapshot("开始")

    try:
        bottom, trunk, top = ModelFactory.create_split_models(
            model_type='qwen2',
            model_name_or_path='Qwen/Qwen2-7B',
            split_point_1=8,
            split_point_2=24,
            device='cpu',
            low_memory=False,  # 使用传统加载
        )

        tracker.snapshot("传统加载完成")
        tracker.report()

        # 清理
        del bottom, trunk, top
        gc.collect()

    except Exception as e:
        print(f"传统加载失败（可能由于内存不足）: {e}")

    # 方式 2: 增量加载
    print("\n--- 增量加载方式 ---")
    tracker.snapshot("开始")

    bottom, trunk, top = ModelFactory.create_split_models(
        model_type='qwen2',
        model_name_or_path='Qwen/Qwen2-7B',
        split_point_1=8,
        split_point_2=24,
        device='cpu',
        low_memory=True,   # 使用增量加载
        verbose=True,      # 显示详细信息
    )

    tracker.snapshot("增量加载完成")
    tracker.report()

    # 显示总结
    print("\n--- 内存使用总结 ---")
    tracker.summary()


# ============================================================================
# 示例 5: 保存和加载分片模型
# ============================================================================

def example_save_and_load():
    """
    创建、保存和重新加载分片模型
    """
    print("\n" + "="*70)
    print("示例 5: 保存和加载分片模型")
    print("="*70 + "\n")

    # 创建模型
    print("正在创建模型...")
    bottom, trunk, top = ModelFactory.create_split_models(
        model_type='qwen2',
        model_name_or_path='Qwen/Qwen2-7B',
        split_point_1=8,
        split_point_2=24,
        device='cpu',
        low_memory=True,
        storage_path='./saved_models/qwen2',  # 指定保存路径
        auto_save=True,  # 自动保存
    )

    print("\n模型已自动保存到: ./saved_models/qwen2/")
    print("  - bottom.pt")
    print("  - trunk.pt")
    print("  - top.pt")

    # 稍后可以重新加载
    print("\n重新加载已保存的模型:")

    from splitlearn_core import SplitModel

    bottom_loaded = SplitModel.load('./saved_models/qwen2/bottom.pt', device='cpu')
    trunk_loaded = SplitModel.load('./saved_models/qwen2/trunk.pt', device='cpu')
    top_loaded = SplitModel.load('./saved_models/qwen2/top.pt', device='cpu')

    print("✓ 模型重新加载成功")

    # 验证
    batch_size, seq_len = 1, 5
    vocab_size = bottom_loaded.model.embed_tokens.weight.shape[0]
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    hidden = bottom_loaded(input_ids)
    hidden = trunk_loaded(hidden)
    output = top_loaded(hidden)

    print(f"推理验证通过: 输出形状 {output.shape}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """
    运行所有示例
    """
    print("\n" + "="*70)
    print("Qwen2-7B 增量加载示例")
    print("="*70)

    print("\n可用示例:")
    print("  1. 基础增量加载")
    print("  2. 详细内存监控")
    print("  3. 多设备分配")
    print("  4. 传统 vs 增量加载对比 (需要大内存)")
    print("  5. 保存和加载分片模型")
    print("  0. 运行所有示例")

    choice = input("\n请选择要运行的示例 (0-5): ").strip()

    if choice == '1':
        example_basic_incremental_loading()
    elif choice == '2':
        example_memory_monitoring()
    elif choice == '3':
        example_device_mapping()
    elif choice == '4':
        example_comparison()
    elif choice == '5':
        example_save_and_load()
    elif choice == '0':
        example_basic_incremental_loading()
        example_memory_monitoring()
        example_device_mapping()
        example_comparison()
        example_save_and_load()
    else:
        print("无效选择")
        return

    print("\n" + "="*70)
    print("示例运行完成")
    print("="*70)


if __name__ == "__main__":
    main()
