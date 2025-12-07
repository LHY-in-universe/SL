#!/usr/bin/env python3
"""
验证 PEFT 库是否可以直接应用到拆分的模型

这个脚本证明：标准的微调库（PEFT）可以直接使用！
"""

import os
import sys
import torch

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# 检查 PEFT 是否已安装
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
    print("✅ PEFT 库已安装")
except ImportError:
    PEFT_AVAILABLE = False
    print("❌ PEFT 库未安装，请运行: pip install peft")
    sys.exit(1)

from transformers import GPT2Config, AutoConfig
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from splitlearn_core.models.gpt2 import GPT2BottomModel, GPT2TopModel


def test_single_block():
    """测试 1: PEFT 是否可以应用到单个 GPT2Block"""
    print("\n" + "=" * 70)
    print("测试 1: 单个 GPT2Block")
    print("=" * 70)
    
    config = GPT2Config()
    block = GPT2Block(config, layer_idx=0)
    
    print(f"原始 Block 参数数量: {sum(p.numel() for p in block.parameters()):,}")
    
    # 应用 PEFT
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        target_modules=["c_attn", "c_fc", "c_proj"]
    )
    
    try:
        peft_block = get_peft_model(block, lora_config)
        print("✅ PEFT 成功应用到单个 Block")
        peft_block.print_trainable_parameters()
        return True
    except Exception as e:
        print(f"❌ 失败: {e}")
        return False


def test_bottom_model():
    """测试 2: PEFT 是否可以应用到 Bottom 模型"""
    print("\n" + "=" * 70)
    print("测试 2: Bottom 模型")
    print("=" * 70)
    
    config = GPT2Config()
    bottom = GPT2BottomModel(config, end_layer=2)
    
    total_params = sum(p.numel() for p in bottom.parameters())
    print(f"原始 Bottom 模型参数数量: {total_params:,}")
    
    # 应用 PEFT
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        target_modules=["c_attn", "c_fc", "c_proj"]
    )
    
    try:
        bottom_peft = get_peft_model(bottom, lora_config)
        print("✅ PEFT 成功应用到 Bottom 模型")
        print("\n参数统计:")
        bottom_peft.print_trainable_parameters()
        
        # 测试前向传播
        input_ids = torch.randint(0, config.vocab_size, (1, 10))
        with torch.no_grad():
            output = bottom_peft(input_ids)
        print(f"✅ 前向传播成功，输出形状: {output.shape}")
        
        return True
    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_top_model():
    """测试 3: PEFT 是否可以应用到 Top 模型"""
    print("\n" + "=" * 70)
    print("测试 3: Top 模型")
    print("=" * 70)
    
    config = GPT2Config()
    top = GPT2TopModel(config, start_layer=10)
    
    total_params = sum(p.numel() for p in top.parameters())
    print(f"原始 Top 模型参数数量: {total_params:,}")
    
    # 应用 PEFT
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        target_modules=["c_attn", "c_fc", "c_proj"]
    )
    
    try:
        top_peft = get_peft_model(top, lora_config)
        print("✅ PEFT 成功应用到 Top 模型")
        print("\n参数统计:")
        top_peft.print_trainable_parameters()
        
        # 测试前向传播
        hidden_states = torch.randn(1, 10, config.n_embd)
        with torch.no_grad():
            output = top_peft(hidden_states)
            logits = output.logits if hasattr(output, 'logits') else output
        print(f"✅ 前向传播成功，输出形状: {logits.shape}")
        
        return True
    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_mode():
    """测试 4: PEFT 模型是否可以正常训练"""
    print("\n" + "=" * 70)
    print("测试 4: 训练模式测试")
    print("=" * 70)
    
    config = GPT2Config()
    bottom = GPT2BottomModel(config, end_layer=2)
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        target_modules=["c_attn", "c_fc", "c_proj"]
    )
    
    bottom_peft = get_peft_model(bottom, lora_config)
    bottom_peft.train()
    
    # 获取可训练参数
    trainable_params = [p for p in bottom_peft.parameters() if p.requires_grad]
    frozen_params = [p for p in bottom_peft.parameters() if not p.requires_grad]
    
    print(f"可训练参数数量: {sum(p.numel() for p in trainable_params):,}")
    print(f"冻结参数数量: {sum(p.numel() for p in frozen_params):,}")
    
    # 创建优化器
    optimizer = torch.optim.Adam(trainable_params, lr=1e-4)
    
    # 模拟训练步骤
    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    output = bottom_peft(input_ids)
    
    # 创建一个简单的损失
    target = torch.randn_like(output)
    loss = torch.nn.MSELoss()(output, target)
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    has_grad = any(p.grad is not None for p in trainable_params if p.requires_grad)
    
    if has_grad:
        print("✅ 梯度计算成功")
        
        # 更新参数
        optimizer.step()
        optimizer.zero_grad()
        print("✅ 参数更新成功")
        
        return True
    else:
        print("❌ 未检测到梯度")
        return False


def test_save_and_load():
    """测试 5: PEFT 模型是否可以保存和加载"""
    print("\n" + "=" * 70)
    print("测试 5: 保存和加载")
    print("=" * 70)
    
    from pathlib import Path
    import tempfile
    
    config = GPT2Config()
    bottom = GPT2BottomModel(config, end_layer=2)
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        target_modules=["c_attn", "c_fc", "c_proj"]
    )
    
    bottom_peft = get_peft_model(bottom, lora_config)
    
    # 保存
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "lora_weights"
        bottom_peft.save_pretrained(str(save_path))
        print(f"✅ LoRA 权重已保存到: {save_path}")
        
        # 检查文件
        adapter_config_file = save_path / "adapter_config.json"
        adapter_model_file = save_path / "adapter_model.bin"
        
        if adapter_config_file.exists() and adapter_model_file.exists():
            print(f"✅ 配置文件存在: {adapter_config_file.exists()}")
            print(f"✅ 模型文件存在: {adapter_model_file.exists()}")
            
            # 加载
            from peft import PeftModel
            bottom_loaded = PeftModel.from_pretrained(bottom, str(save_path))
            print("✅ LoRA 权重加载成功")
            
            return True
        else:
            print(f"❌ 文件不存在")
            return False


def main():
    """运行所有测试"""
    print("=" * 70)
    print("PEFT 兼容性验证")
    print("=" * 70)
    print("\n这个脚本验证：标准的 PEFT 库可以直接应用到你的拆分模型！")
    
    results = []
    
    # 运行所有测试
    results.append(("单个 Block", test_single_block()))
    results.append(("Bottom 模型", test_bottom_model()))
    results.append(("Top 模型", test_top_model()))
    results.append(("训练模式", test_training_mode()))
    results.append(("保存/加载", test_save_and_load()))
    
    # 总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！")
        print("✅ 结论: PEFT 库可以完全兼容你的拆分模型！")
        print("✅ 建议: 直接使用 PEFT 库，无需自实现 LoRA！")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
