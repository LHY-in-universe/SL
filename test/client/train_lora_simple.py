#!/usr/bin/env python3
"""
简化的 LoRA 微调测试脚本

使用 PEFT LoRA 库在 Split Learning 架构下进行模型微调
简化版本：服务器只做前向传播，客户端做反向传播和参数更新

使用方法:
    python test/client/train_lora_simple.py [--server SERVER] [--dataset TYPE] [--samples N]
"""

import os
import sys
import json
import time
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# 添加 test 目录到路径（用于导入 data 模块）
test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, test_dir)

from transformers import AutoTokenizer, AutoConfig

# 检查 PEFT 库
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("❌ PEFT 库未安装")
    print("   请运行: pip install peft")
    sys.exit(1)

from splitlearn_comm.quickstart import Client
from splitlearn_core.models.gpt2 import GPT2BottomModel, GPT2TopModel

# 添加 test 目录到路径以导入数据集加载器
test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, test_dir)
from data.dataset_loader import load_test_dataset


def print_section(title: str):
    """打印分节标题"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_step(step_num: int, title: str):
    """打印步骤标题"""
    print(f"\n[{step_num}] {title}")
    print("-" * 50)


def check_dependencies():
    """检查依赖"""
    print_section("依赖检查")
    
    issues = []
    
    if not PEFT_AVAILABLE:
        issues.append("PEFT 库未安装 (pip install peft)")
    
    try:
        import datasets
        print("✅ datasets 库已安装")
    except ImportError:
        print("⚠️  datasets 库未安装 (pip install datasets) - 将使用合成数据集")
    
    if issues:
        print("\n❌ 缺少依赖:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    
    print("✅ 所有必需依赖已安装")
    return True


def load_models(models_dir: Path):
    """加载模型"""
    print_step(1, "加载模型")
    
    bottom_path = models_dir / "bottom" / "gpt2_2-10_bottom.pt"
    top_path = models_dir / "top" / "gpt2_2-10_top.pt"
    bottom_metadata_path = models_dir / "bottom" / "gpt2_2-10_bottom_metadata.json"
    top_metadata_path = models_dir / "top" / "gpt2_2-10_top_metadata.json"
    
    # 检查文件
    if not all([bottom_path.exists(), top_path.exists(), 
                bottom_metadata_path.exists(), top_metadata_path.exists()]):
        print("❌ 模型文件不存在")
        print(f"   请确保模型文件在: {models_dir}")
        return None, None, None
    
    # 加载元数据
    with open(bottom_metadata_path, 'r') as f:
        bottom_metadata = json.load(f)
    with open(top_metadata_path, 'r') as f:
        top_metadata = json.load(f)
    
    # 加载配置
    config = AutoConfig.from_pretrained("gpt2")
    
    # 加载模型
    print("加载 Bottom 模型...")
    bottom = GPT2BottomModel(config, end_layer=bottom_metadata['end_layer'])
    bottom.load_state_dict(
        torch.load(bottom_path, map_location='cpu', weights_only=True)
    )
    print(f"  ✓ Bottom 模型加载成功 (Layers 0-{bottom_metadata['end_layer']})")
    
    print("加载 Top 模型...")
    top = GPT2TopModel(config, start_layer=top_metadata['start_layer'])
    top.load_state_dict(
        torch.load(top_path, map_location='cpu', weights_only=True)
    )
    print(f"  ✓ Top 模型加载成功 (Layers {top_metadata['start_layer']}+)")
    
    return bottom, top, config


def apply_peft_lora(bottom, top, lora_rank: int = 8):
    """应用 PEFT LoRA 到模型"""
    print_step(2, "应用 PEFT LoRA")
    
    # LoRA 配置 - 使用 FEATURE_EXTRACTION 因为我们不是完整的模型
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,  # 使用特征提取任务类型，更适合拆分模型
        inference_mode=False,
        r=lora_rank,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_fc", "c_proj"]
    )
    
    print(f"LoRA 配置: rank={lora_rank}, alpha=16, dropout=0.1")
    print(f"目标模块: c_attn, c_fc, c_proj")
    
    # 应用到 Bottom 模型
    print("\n应用到 Bottom 模型...")
    bottom_peft = get_peft_model(bottom, lora_config)
    print("Bottom 模型参数:")
    bottom_peft.print_trainable_parameters()
    
    # 应用到 Top 模型
    print("\n应用到 Top 模型...")
    top_peft = get_peft_model(top, lora_config)
    print("Top 模型参数:")
    top_peft.print_trainable_parameters()
    
    return bottom_peft, top_peft


def create_optimizers(bottom_peft, top_peft, learning_rate: float = 1e-4):
    """创建优化器（只优化 LoRA 参数）"""
    # 获取可训练参数
    trainable_params_bottom = [
        p for p in bottom_peft.parameters() if p.requires_grad
    ]
    trainable_params_top = [
        p for p in top_peft.parameters() if p.requires_grad
    ]
    
    optimizer_bottom = torch.optim.Adam(trainable_params_bottom, lr=learning_rate)
    optimizer_top = torch.optim.Adam(trainable_params_top, lr=learning_rate)
    
    print(f"\n优化器创建成功 (学习率: {learning_rate})")
    print(f"  Bottom 可训练参数: {sum(p.numel() for p in trainable_params_bottom):,}")
    print(f"  Top 可训练参数: {sum(p.numel() for p in trainable_params_top):,}")
    
    return optimizer_bottom, optimizer_top


def train_step(
    bottom_peft,
    top_peft,
    trunk_client,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    optimizer_bottom,
    optimizer_top,
    criterion
) -> Dict[str, float]:
    """执行一个训练步骤"""
    # 清零梯度
    optimizer_bottom.zero_grad()
    optimizer_top.zero_grad()
    
    # 前向传播
    # Bottom 模型（本地，保留梯度）
    # 直接调用 base_model，因为我们的模型不支持 inputs_embeds 参数
    hidden_1 = bottom_peft.base_model(input_ids)
    
    # Trunk 模型（远程服务器，断开梯度 - 简化版本）
    hidden_2 = trunk_client.compute(hidden_1.detach())
    hidden_2 = hidden_2.requires_grad_(True)  # 重新启用梯度用于反向传播
    
    # Top 模型（本地，保留梯度）
    # 直接调用 base_model
    output = top_peft.base_model(hidden_2)
    logits = output.logits if hasattr(output, 'logits') else output
    
    # 计算损失
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = criterion(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    
    # 反向传播
    loss.backward()
    
    # 参数更新（只更新 Bottom 和 Top 的 LoRA 参数）
    optimizer_bottom.step()
    optimizer_top.step()
    
    return {
        'loss': loss.item(),
        'hidden_1_norm': hidden_1.norm().item(),
        'hidden_2_norm': hidden_2.norm().item()
    }


def main():
    parser = argparse.ArgumentParser(description='简化的 LoRA 微调测试')
    parser.add_argument('--server', type=str, default='localhost:50052',
                       help='Trunk 服务器地址 (默认: localhost:50052)')
    parser.add_argument('--dataset', type=str, default='synthetic',
                       choices=['synthetic', 'wikitext'],
                       help='数据集类型 (默认: synthetic)')
    parser.add_argument('--samples', type=int, default=20,
                       help='数据集样本数 (默认: 20)')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='批次大小 (默认: 2)')
    parser.add_argument('--max-length', type=int, default=128,
                       help='最大序列长度 (默认: 128)')
    parser.add_argument('--epochs', type=int, default=1,
                       help='训练轮数 (默认: 1)')
    parser.add_argument('--lora-rank', type=int, default=8,
                       help='LoRA 秩 (默认: 8)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率 (默认: 1e-4)')
    parser.add_argument('--save-dir', type=str, default='./lora_checkpoints',
                       help='保存目录 (默认: ./lora_checkpoints)')
    
    args = parser.parse_args()
    
    print_section("Split Learning LoRA 微调测试")
    print(f"\n配置:")
    print(f"  服务器: {args.server}")
    print(f"  数据集: {args.dataset} ({args.samples} 样本)")
    print(f"  批次大小: {args.batch_size}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  LoRA 秩: {args.lora_rank}")
    print(f"  学习率: {args.lr}")
    
    # 检查依赖
    if not check_dependencies():
        return 1
    
    # 加载模型
    models_dir = Path(project_root) / "models"
    bottom, top, config = load_models(models_dir)
    if bottom is None:
        return 1
    
    # 应用 PEFT LoRA
    bottom_peft, top_peft = apply_peft_lora(bottom, top, lora_rank=args.lora_rank)
    
    # 设置为训练模式
    bottom_peft.train()
    top_peft.train()
    
    # 创建优化器
    optimizer_bottom, optimizer_top = create_optimizers(
        bottom_peft, top_peft, learning_rate=args.lr
    )
    
    # 损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # 连接到服务器
    print_step(3, "连接到 Trunk 服务器")
    try:
        trunk_client = Client(args.server)
        print(f"✓ 已连接到服务器: {args.server}")
    except Exception as e:
        print(f"❌ 连接服务器失败: {e}")
        print(f"\n请确保 Trunk 服务器正在运行:")
        print(f"  bash test/start_all.sh")
        return 1
    
    # 加载数据集
    print_step(4, "加载数据集")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    try:
        dataloader = load_test_dataset(
            dataset_type=args.dataset,
            num_samples=args.samples,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length
        )
        print(f"✓ 数据集加载成功，共 {len(dataloader)} 个批次")
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 训练循环
    print_section("开始训练")
    print(f"训练配置:")
    print(f"  总批次: {len(dataloader)}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  总训练步骤: {len(dataloader) * args.epochs}")
    print()
    
    all_losses = []
    
    try:
        for epoch in range(args.epochs):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1}/{args.epochs}")
            print(f"{'='*70}")
            
            epoch_losses = []
            
            for step, batch in enumerate(dataloader):
                input_ids = batch['input_ids']
                labels = batch['labels']
                
                # 训练步骤
                metrics = train_step(
                    bottom_peft,
                    top_peft,
                    trunk_client,
                    input_ids,
                    labels,
                    optimizer_bottom,
                    optimizer_top,
                    criterion
                )
                
                loss = metrics['loss']
                epoch_losses.append(loss)
                all_losses.append(loss)
                
                # 显示进度
                if (step + 1) % max(1, len(dataloader) // 5) == 0 or (step + 1) == len(dataloader):
                    avg_loss = sum(epoch_losses) / len(epoch_losses)
                    print(f"  Step {step + 1}/{len(dataloader)}: "
                          f"loss = {loss:.4f}, avg_loss = {avg_loss:.4f}")
            
            # Epoch 总结
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"\nEpoch {epoch + 1} 完成:")
            print(f"  平均损失: {avg_epoch_loss:.4f}")
            print(f"  最小损失: {min(epoch_losses):.4f}")
            print(f"  最大损失: {max(epoch_losses):.4f}")
    
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # 保存 LoRA 权重
        print_section("保存 LoRA 权重")
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            bottom_lora_path = save_dir / "bottom_lora"
            bottom_peft.save_pretrained(str(bottom_lora_path))
            print(f"✓ Bottom LoRA 权重已保存: {bottom_lora_path}")
            
            top_lora_path = save_dir / "top_lora"
            top_peft.save_pretrained(str(top_lora_path))
            print(f"✓ Top LoRA 权重已保存: {top_lora_path}")
        except Exception as e:
            print(f"⚠️  保存 LoRA 权重失败: {e}")
        
        # 关闭连接
        trunk_client.close()
    
    # 训练总结
    if all_losses:
        print_section("训练总结")
        print(f"总训练步骤: {len(all_losses)}")
        print(f"初始损失: {all_losses[0]:.4f}")
        print(f"最终损失: {all_losses[-1]:.4f}")
        print(f"最低损失: {min(all_losses):.4f}")
        print(f"平均损失: {sum(all_losses) / len(all_losses):.4f}")
        
        if len(all_losses) > 1:
            loss_change = all_losses[-1] - all_losses[0]
            if loss_change < 0:
                print(f"\n✅ 损失下降 {abs(loss_change):.4f}，训练成功！")
            else:
                print(f"\n⚠️  损失上升 {loss_change:.4f}，可能需要调整学习率")
    
    print_section("测试完成")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code if exit_code is not None else 0)
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
