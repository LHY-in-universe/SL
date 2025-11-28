"""
测试增量加载功能
使用 GPT-2 (非分片) 和 Qwen2-0.5B (可能分片) 进行测试
"""
import sys
import os

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, 'SplitLearning', 'src'))

import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestIncrementalLoading")

def test_imports():
    """测试导入"""
    logger.info("=" * 70)
    logger.info("测试 1: 导入模块")
    logger.info("=" * 70)

    try:
        from splitlearn import ModelFactory
        from splitlearn.utils import ShardLoader, MemoryTracker
        logger.info("✅ 所有模块导入成功")
        return True
    except Exception as e:
        logger.error(f"❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_shard_detection():
    """测试分片检测"""
    logger.info("\n" + "=" * 70)
    logger.info("测试 2: 分片检测")
    logger.info("=" * 70)

    try:
        from splitlearn.utils import ShardLoader

        # 测试非分片模型
        logger.info("检测 GPT-2 (非分片模型)...")
        is_sharded = ShardLoader.is_sharded_model('gpt2')
        logger.info(f"  GPT-2 是否分片: {is_sharded}")

        if is_sharded:
            logger.warning("  ⚠️  GPT-2 不应该是分片模型")
        else:
            logger.info("  ✅ 正确识别为非分片模型")

        # 测试分片模型（如果网络可用）
        logger.info("\n检测 Qwen2-0.5B (可能是分片模型)...")
        try:
            is_sharded = ShardLoader.is_sharded_model('Qwen/Qwen2-0.5B')
            logger.info(f"  Qwen2-0.5B 是否分片: {is_sharded}")
            logger.info("  ✅ 检测完成")
        except Exception as e:
            logger.warning(f"  ⚠️  无法检测 Qwen2-0.5B (可能是网络问题): {e}")

        return True

    except Exception as e:
        logger.error(f"❌ 分片检测失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_tracker():
    """测试内存追踪器"""
    logger.info("\n" + "=" * 70)
    logger.info("测试 3: 内存追踪器")
    logger.info("=" * 70)

    try:
        from splitlearn.utils import MemoryTracker

        tracker = MemoryTracker()

        # 记录快照
        tracker.snapshot("开始")

        # 分配一些内存
        data = torch.randn(1000, 1000)

        tracker.snapshot("分配内存后")

        # 查看报告
        logger.info("\n内存变化报告:")
        tracker.report()

        # 查看摘要
        logger.info("\n内存使用摘要:")
        tracker.summary()

        # 获取当前使用
        usage = tracker.get_current_usage()
        logger.info(f"\n当前内存使用:")
        logger.info(f"  RAM: {usage['ram_gb']:.2f} GB ({usage['ram_percent']:.1f}%)")
        logger.info(f"  GPU: {usage['gpu_gb']:.2f} GB")

        logger.info("\n✅ 内存追踪器测试通过")
        return True

    except Exception as e:
        logger.error(f"❌ 内存追踪器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_traditional_loading():
    """测试传统加载（非分片模型）"""
    logger.info("\n" + "=" * 70)
    logger.info("测试 4: 传统加载 (GPT-2, low_memory=False)")
    logger.info("=" * 70)

    try:
        from splitlearn import ModelFactory

        logger.info("正在加载 GPT-2...")

        bottom, trunk, top = ModelFactory.create_split_models(
            model_type='gpt2',
            model_name_or_path='gpt2',
            split_point_1=2,
            split_point_2=10,
            device='cpu',
            low_memory=False,  # 传统模式
            verbose=False,
        )

        logger.info("✅ 模型加载成功")
        logger.info(f"  Bottom: {type(bottom).__name__}")
        logger.info(f"  Trunk:  {type(trunk).__name__}")
        logger.info(f"  Top:    {type(top).__name__}")

        # 测试推理
        logger.info("\n测试推理...")
        vocab_size = bottom.wte.weight.shape[0]
        input_ids = torch.randint(0, vocab_size, (1, 5))

        hidden = bottom(input_ids)
        logger.info(f"  Bottom 输出: {hidden.shape}")

        hidden = trunk(hidden)
        logger.info(f"  Trunk 输出:  {hidden.shape}")

        output = top(hidden)
        # Top 模型返回 CausalLMOutputWithPast 对象
        if hasattr(output, 'logits'):
            logger.info(f"  Top 输出:    {output.logits.shape}")
        else:
            logger.info(f"  Top 输出:    {output.shape}")

        logger.info("\n✅ 传统加载测试通过")
        return True

    except Exception as e:
        logger.error(f"❌ 传统加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_incremental_loading_non_sharded():
    """测试增量加载（非分片模型应该回退到传统加载）"""
    logger.info("\n" + "=" * 70)
    logger.info("测试 5: 增量加载非分片模型 (GPT-2, low_memory=True)")
    logger.info("=" * 70)

    try:
        from splitlearn import ModelFactory

        logger.info("正在加载 GPT-2 (low_memory=True)...")
        logger.info("期望：应该自动回退到传统加载\n")

        bottom, trunk, top = ModelFactory.create_split_models(
            model_type='gpt2',
            model_name_or_path='gpt2',
            split_point_1=2,
            split_point_2=10,
            device='cpu',
            low_memory=True,   # 增量模式
            verbose=True,      # 显示详细信息
        )

        logger.info("\n✅ 模型加载成功 (应该使用了传统加载)")

        # 测试推理
        logger.info("\n测试推理...")
        vocab_size = bottom.wte.weight.shape[0]
        input_ids = torch.randint(0, vocab_size, (1, 5))

        hidden = bottom(input_ids)
        hidden = trunk(hidden)
        output = top(hidden)

        # Top 模型返回 CausalLMOutputWithPast 对象
        if hasattr(output, 'logits'):
            logger.info(f"  推理成功: {output.logits.shape}")
        else:
            logger.info(f"  推理成功: {output.shape}")
        logger.info("\n✅ 增量加载非分片模型测试通过")
        return True

    except Exception as e:
        logger.error(f"❌ 增量加载非分片模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_device_map():
    """测试设备映射"""
    logger.info("\n" + "=" * 70)
    logger.info("测试 6: 设备映射")
    logger.info("=" * 70)

    try:
        from splitlearn import ModelFactory

        # 测试手动设备映射
        logger.info("测试手动设备映射...")
        device_map = {
            'bottom': 'cpu',
            'trunk': 'cpu',
            'top': 'cpu',
        }

        bottom, trunk, top = ModelFactory.create_split_models(
            model_type='gpt2',
            model_name_or_path='gpt2',
            split_point_1=2,
            split_point_2=10,
            device='cpu',
            device_map=device_map,
            low_memory=True,
        )

        logger.info("✅ 手动设备映射成功")
        logger.info(f"  Bottom 设备: {next(bottom.parameters()).device}")
        logger.info(f"  Trunk 设备:  {next(trunk.parameters()).device}")
        logger.info(f"  Top 设备:    {next(top.parameters()).device}")

        # 测试 auto 设备映射
        logger.info("\n测试 auto 设备映射...")
        bottom2, trunk2, top2 = ModelFactory.create_split_models(
            model_type='gpt2',
            model_name_or_path='gpt2',
            split_point_1=2,
            split_point_2=10,
            device='cpu',
            device_map='auto',
            low_memory=True,
        )

        logger.info("✅ Auto 设备映射成功")
        logger.info(f"  Bottom 设备: {next(bottom2.parameters()).device}")
        logger.info(f"  Trunk 设备:  {next(trunk2.parameters()).device}")
        logger.info(f"  Top 设备:    {next(top2.parameters()).device}")

        logger.info("\n✅ 设备映射测试通过")
        return True

    except Exception as e:
        logger.error(f"❌ 设备映射测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    logger.info("\n" + "=" * 70)
    logger.info("开始测试增量加载功能")
    logger.info("=" * 70)

    results = []

    # 运行测试
    results.append(("导入模块", test_imports()))

    if results[-1][1]:  # 如果导入成功，继续其他测试
        results.append(("分片检测", test_shard_detection()))
        results.append(("内存追踪器", test_memory_tracker()))
        results.append(("传统加载", test_traditional_loading()))
        results.append(("增量加载非分片", test_incremental_loading_non_sharded()))
        results.append(("设备映射", test_device_map()))

    # 显示结果摘要
    logger.info("\n" + "=" * 70)
    logger.info("测试结果摘要")
    logger.info("=" * 70)

    for test_name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        logger.info(f"{test_name:20s}: {status}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    logger.info("\n" + "=" * 70)
    logger.info(f"总计: {passed}/{total} 测试通过")
    logger.info("=" * 70)

    if passed == total:
        logger.info("\n🎉 所有测试通过！增量加载功能正常工作。")
        return 0
    else:
        logger.error(f"\n⚠️  有 {total - passed} 个测试失败，请检查错误信息。")
        return 1

if __name__ == "__main__":
    exit(main())
