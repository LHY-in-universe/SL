#!/usr/bin/env python3
"""
分析 GPT-2 Split Learning 模型中 Top 和 Bottom 的性能差异

探索为什么参数量相近但运行时间差别很大的原因
"""

import os
import sys
import json
import time
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from splitlearn_core.models.gpt2 import GPT2BottomModel, GPT2TopModel


def count_parameters(model):
    """统计模型参数数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'total_m': total / 1e6,
        'trainable_m': trainable / 1e6
    }


def analyze_model_structure(model, model_name, config):
    """分析模型结构"""
    print(f"\n{'='*70}")
    print(f"{model_name} 模型结构分析")
    print(f"{'='*70}\n")
    
    # 统计参数
    params = count_parameters(model)
    print(f"参数统计:")
    print(f"  总参数: {params['total']:,} ({params['total_m']:.2f}M)")
    print(f"  可训练参数: {params['trainable']:,} ({params['trainable_m']:.2f}M)")
    
    # 分析各组件
    print(f"\n模型组件详细分析:")
    
    if hasattr(model, 'wte'):
        # Bottom 模型
        wte_params = sum(p.numel() for p in model.wte.parameters())
        wpe_params = sum(p.numel() for p in model.wpe.parameters())
        h_params = sum(p.numel() for p in model.h.parameters())
        
        print(f"  1. Token Embedding (wte):")
        print(f"     参数: {wte_params:,} ({wte_params/1e6:.2f}M)")
        print(f"     维度: {config.vocab_size} × {config.n_embd}")
        print(f"     操作: 查找表 (O(seq_len))")
        
        print(f"  2. Position Embedding (wpe):")
        print(f"     参数: {wpe_params:,} ({wpe_params/1e6:.2f}M)")
        print(f"     维度: {config.n_positions} × {config.n_embd}")
        print(f"     操作: 查找表 (O(seq_len))")
        
        print(f"  3. Transformer Blocks (h):")
        print(f"     层数: {len(model.h)}")
        print(f"     参数: {h_params:,} ({h_params/1e6:.2f}M)")
        print(f"     平均每层: {h_params/len(model.h):,} ({h_params/len(model.h)/1e6:.2f}M)")
        
        # 分析每层transformer block
        if len(model.h) > 0:
            first_block = model.h[0]
            block_params = sum(p.numel() for p in first_block.parameters())
            print(f"     单个Block参数: {block_params:,} ({block_params/1e6:.2f}M)")
        
    elif hasattr(model, 'lm_head'):
        # Top 模型
        h_params = sum(p.numel() for p in model.h.parameters())
        ln_f_params = sum(p.numel() for p in model.ln_f.parameters())
        lm_head_params = sum(p.numel() for p in model.lm_head.parameters())
        
        print(f"  1. Transformer Blocks (h):")
        print(f"     层数: {len(model.h)}")
        print(f"     参数: {h_params:,} ({h_params/1e6:.2f}M)")
        print(f"     平均每层: {h_params/len(model.h):,} ({h_params/len(model.h)/1e6:.2f}M)")
        
        if len(model.h) > 0:
            first_block = model.h[0]
            block_params = sum(p.numel() for p in first_block.parameters())
            print(f"     单个Block参数: {block_params:,} ({block_params/1e6:.2f}M)")
        
        print(f"  2. Final Layer Norm (ln_f):")
        print(f"     参数: {ln_f_params:,} ({ln_f_params/1e6:.2f}M)")
        print(f"     操作: Layer Normalization (O(seq_len × hidden_size))")
        
        print(f"  3. Language Modeling Head (lm_head):")
        print(f"     参数: {lm_head_params:,} ({lm_head_params/1e6:.2f}M)")
        print(f"     维度: {config.n_embd} × {config.vocab_size}")
        print(f"     计算复杂度: O(seq_len × hidden_size × vocab_size)")
        print(f"     矩阵乘法大小: {config.n_embd} × {config.vocab_size} = {config.n_embd * config.vocab_size:,}")
        print(f"     每次前向传播的计算量: seq_len × {config.n_embd * config.vocab_size:,} FLOPS")
    
    return params


def estimate_computation_cost(model, model_name, input_shape, config):
    """估算计算成本"""
    print(f"\n{'='*70}")
    print(f"{model_name} 计算成本估算")
    print(f"{'='*70}\n")
    
    batch_size, seq_len = input_shape[0], input_shape[1]
    hidden_size = config.n_embd
    vocab_size = config.vocab_size
    
    print(f"输入形状: {input_shape}")
    print(f"配置: hidden_size={hidden_size}, vocab_size={vocab_size}\n")
    
    total_flops = 0
    operations = []
    
    if hasattr(model, 'wte'):
        # Bottom 模型计算
        print("1. Token Embedding:")
        wte_flops = seq_len * hidden_size  # 查找操作，可忽略计算量
        print(f"   查找表操作 (可忽略)")
        
        print("2. Position Embedding:")
        wpe_flops = seq_len * hidden_size  # 查找操作
        print(f"   查找表操作 (可忽略)")
        
        print(f"3. Transformer Blocks ({len(model.h)} 层):")
        block_flops_per_layer = 0
        
        # 每个transformer block包含：
        # - Self-attention: O(seq_len^2 * hidden_size)
        # - Feed-forward: O(seq_len * hidden_size^2)
        
        # Self-attention 计算
        # Q, K, V 投影: 3 * seq_len * hidden_size^2
        # Attention scores: seq_len^2 * hidden_size
        # Output projection: seq_len * hidden_size^2
        attention_flops = (3 * seq_len * hidden_size * hidden_size + 
                          seq_len * seq_len * hidden_size + 
                          seq_len * hidden_size * hidden_size)
        
        # Feed-forward 计算 (通常有4倍扩展)
        ff_flops = 2 * seq_len * hidden_size * (hidden_size * 4)
        
        block_flops_per_layer = attention_flops + ff_flops
        total_block_flops = block_flops_per_layer * len(model.h)
        
        print(f"   每层Block: ~{block_flops_per_layer/1e9:.2f} GFLOPS")
        print(f"   总计: {total_block_flops/1e9:.2f} GFLOPS")
        total_flops = total_block_flops
        
        operations.append(("Embedding + Transformer Blocks", total_flops))
        
    elif hasattr(model, 'lm_head'):
        # Top 模型计算
        print(f"1. Transformer Blocks ({len(model.h)} 层):")
        
        # 同bottom的transformer block计算
        attention_flops = (3 * seq_len * hidden_size * hidden_size + 
                          seq_len * seq_len * hidden_size + 
                          seq_len * hidden_size * hidden_size)
        ff_flops = 2 * seq_len * hidden_size * (hidden_size * 4)
        block_flops_per_layer = attention_flops + ff_flops
        total_block_flops = block_flops_per_layer * len(model.h)
        
        print(f"   每层Block: ~{block_flops_per_layer/1e9:.2f} GFLOPS")
        print(f"   总计: {total_block_flops/1e9:.2f} GFLOPS")
        
        print(f"\n2. Final Layer Norm:")
        ln_flops = seq_len * hidden_size * 2  # mean + variance + normalization
        print(f"   ~{ln_flops/1e6:.2f} MFLOPS (可忽略)")
        
        print(f"\n3. Language Modeling Head (关键部分):")
        # 矩阵乘法: [batch, seq_len, hidden_size] × [hidden_size, vocab_size]
        lm_head_flops = batch_size * seq_len * hidden_size * vocab_size
        print(f"   矩阵乘法: {batch_size} × {seq_len} × {hidden_size} × {vocab_size}")
        print(f"   计算量: {lm_head_flops/1e9:.2f} GFLOPS")
        print(f"   这是最大的计算瓶颈！")
        
        total_flops = total_block_flops + lm_head_flops
        operations.append(("Transformer Blocks", total_block_flops))
        operations.append(("LM Head", lm_head_flops))
    
    print(f"\n总计算量估算: {total_flops/1e9:.2f} GFLOPS")
    
    return total_flops, operations


def benchmark_model_forward(model, model_name, input_tensor, num_runs=10):
    """基准测试模型前向传播"""
    print(f"\n{'='*70}")
    print(f"{model_name} 基准测试 ({num_runs} 次运行)")
    print(f"{'='*70}\n")
    
    model.eval()
    
    # 预热
    with torch.no_grad():
        _ = model(input_tensor) if hasattr(model, 'wte') else model(input_tensor)
    
    # 测量时间
    times = []
    with torch.no_grad():
        for i in range(num_runs):
            start = time.time()
            output = model(input_tensor) if hasattr(model, 'wte') else model(input_tensor)
            end = time.time()
            times.append((end - start) * 1000)  # 转换为毫秒
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"平均时间: {avg_time:.2f} ms")
    print(f"最短时间: {min_time:.2f} ms")
    print(f"最长时间: {max_time:.2f} ms")
    print(f"标准差: {(sum((t - avg_time)**2 for t in times) / len(times))**0.5:.2f} ms")
    
    if hasattr(output, 'logits'):
        print(f"输出形状: {output.logits.shape}")
    else:
        print(f"输出形状: {output.shape}")
    
    return avg_time, times


def analyze_layer_by_layer(model, model_name, input_tensor):
    """逐层分析性能"""
    print(f"\n{'='*70}")
    print(f"{model_name} 逐层性能分析")
    print(f"{'='*70}\n")
    
    model.eval()
    
    with torch.no_grad():
        x = input_tensor
        layer_times = []
        
        if hasattr(model, 'wte'):
            # Bottom 模型
            start = time.time()
            inputs_embeds = model.wte(x)
            x = model.apply_position_encoding(inputs_embeds, None)
            layer_time = (time.time() - start) * 1000
            layer_times.append(("Embedding", layer_time))
            print(f"1. Embedding: {layer_time:.2f} ms")
            
            for i, block in enumerate(model.h):
                start = time.time()
                x = block(x)[0]
                layer_time = (time.time() - start) * 1000
                layer_times.append((f"Block {i}", layer_time))
                print(f"{i+2}. Transformer Block {i}: {layer_time:.2f} ms")
        
        elif hasattr(model, 'lm_head'):
            # Top 模型
            for i, block in enumerate(model.h):
                start = time.time()
                x = block(x)[0]
                layer_time = (time.time() - start) * 1000
                layer_times.append((f"Block {i}", layer_time))
                print(f"{i+1}. Transformer Block {i}: {layer_time:.2f} ms")
            
            start = time.time()
            x = model.ln_f(x)
            layer_time = (time.time() - start) * 1000
            layer_times.append(("Layer Norm", layer_time))
            print(f"{len(model.h)+1}. Final Layer Norm: {layer_time:.2f} ms")
            
            start = time.time()
            logits = model.lm_head(x)
            layer_time = (time.time() - start) * 1000
            layer_times.append(("LM Head", layer_time))
            print(f"{len(model.h)+2}. LM Head: {layer_time:.2f} ms")
            x = type('obj', (object,), {'logits': logits})()
    
    return layer_times


def main():
    print("=" * 70)
    print("GPT-2 Split Learning 模型性能分析")
    print("探索 Top 和 Bottom 模型性能差异的原因")
    print("=" * 70)
    
    # 加载配置 - 模型文件在项目根目录的models文件夹
    models_dir = Path(project_root) / "models"
    bottom_path = models_dir / "bottom" / "gpt2_2-10_bottom.pt"
    top_path = models_dir / "top" / "gpt2_2-10_top.pt"
    bottom_metadata_path = models_dir / "bottom" / "gpt2_2-10_bottom_metadata.json"
    top_metadata_path = models_dir / "top" / "gpt2_2-10_top_metadata.json"
    
    # 打印路径检查
    print(f"\n模型文件路径检查:")
    print(f"  Bottom: {bottom_path} - {'存在' if bottom_path.exists() else '不存在'}")
    print(f"  Top: {top_path} - {'存在' if top_path.exists() else '不存在'}")
    print(f"  Bottom metadata: {bottom_metadata_path} - {'存在' if bottom_metadata_path.exists() else '不存在'}")
    print(f"  Top metadata: {top_metadata_path} - {'存在' if top_metadata_path.exists() else '不存在'}")
    
    # 检查文件
    if not all([bottom_path.exists(), top_path.exists(), 
                bottom_metadata_path.exists(), top_metadata_path.exists()]):
        print("❌ 模型文件不存在，请先准备模型文件")
        return 1
    
    # 加载配置
    config = AutoConfig.from_pretrained("gpt2")
    print(f"\nGPT-2 配置:")
    print(f"  Hidden size: {config.n_embd}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Number of layers: {config.n_layer}")
    print(f"  Number of attention heads: {config.n_head}")
    
    # 加载元数据
    with open(bottom_metadata_path, 'r') as f:
        bottom_metadata = json.load(f)
    with open(top_metadata_path, 'r') as f:
        top_metadata = json.load(f)
    
    print(f"\n拆分配置:")
    print(f"  Bottom layers: 0-{bottom_metadata['end_layer']} ({bottom_metadata['end_layer']} 层)")
    print(f"  Top layers: {top_metadata['start_layer']}+ ({config.n_layer - top_metadata['start_layer']} 层)")
    
    # 加载模型
    print("\n" + "=" * 70)
    print("加载模型...")
    print("=" * 70)
    
    bottom = GPT2BottomModel(config, end_layer=bottom_metadata['end_layer'])
    bottom.load_state_dict(torch.load(bottom_path, map_location='cpu', weights_only=True))
    bottom.eval()
    
    top = GPT2TopModel(config, start_layer=top_metadata['start_layer'])
    top.load_state_dict(torch.load(top_path, map_location='cpu', weights_only=True))
    top.eval()
    
    print("✓ 模型加载完成")
    
    # 准备测试输入
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    test_text = "The future of artificial intelligence is"
    input_ids = tokenizer.encode(test_text, return_tensors="pt")
    seq_len = input_ids.shape[1]
    
    print(f"\n测试输入: '{test_text}'")
    print(f"序列长度: {seq_len}")
    
    # 1. 分析模型结构
    bottom_params = analyze_model_structure(bottom, "Bottom", config)
    top_params = analyze_model_structure(top, "Top", config)
    
    # 2. 估算计算成本
    bottom_flops, bottom_ops = estimate_computation_cost(bottom, "Bottom", input_ids.shape, config)
    top_flops, top_ops = estimate_computation_cost(top, "Top", (1, seq_len), config)
    
    # 3. 逐层性能分析
    print("\n" + "=" * 70)
    print("逐层性能分析")
    print("=" * 70)
    
    bottom_layer_times = analyze_layer_by_layer(bottom, "Bottom", input_ids)
    bottom_output = bottom(input_ids)
    top_layer_times = analyze_layer_by_layer(top, "Top", bottom_output)
    
    # 4. 基准测试
    bottom_time, bottom_times = benchmark_model_forward(bottom, "Bottom", input_ids)
    bottom_output = bottom(input_ids)
    top_time, top_times = benchmark_model_forward(top, "Top", bottom_output)
    
    # 5. 总结对比
    print("\n" + "=" * 70)
    print("性能对比总结")
    print("=" * 70)
    
    print(f"\n参数数量对比:")
    print(f"  Bottom: {bottom_params['total_m']:.2f}M 参数")
    print(f"  Top:    {top_params['total_m']:.2f}M 参数")
    print(f"  差异:   {abs(bottom_params['total'] - top_params['total'])/1e6:.2f}M 参数")
    
    print(f"\n计算量对比:")
    print(f"  Bottom: {bottom_flops/1e9:.2f} GFLOPS")
    print(f"  Top:    {top_flops/1e9:.2f} GFLOPS")
    print(f"  比率:   {top_flops/bottom_flops:.2f}x")
    
    print(f"\n运行时间对比:")
    print(f"  Bottom: {bottom_time:.2f} ms")
    print(f"  Top:    {top_time:.2f} ms")
    print(f"  比率:   {top_time/bottom_time:.2f}x")
    
    print(f"\n关键发现:")
    print(f"  1. Top模型的LM Head计算量: {top_ops[-1][1]/1e9:.2f} GFLOPS")
    print(f"     这是导致Top模型慢的主要原因")
    print(f"  2. LM Head矩阵乘法维度: {config.n_embd} × {config.vocab_size} = {config.n_embd * config.vocab_size:,}")
    print(f"  3. 即使参数量相近，Top模型的FLOPS是Bottom的 {top_flops/bottom_flops:.2f} 倍")
    
    # 分析逐层时间
    print(f"\nBottom模型逐层时间占比:")
    bottom_total = sum(t for _, t in bottom_layer_times)
    for name, t in bottom_layer_times:
        print(f"  {name}: {t:.2f} ms ({t/bottom_total*100:.1f}%)")
    
    print(f"\nTop模型逐层时间占比:")
    top_total = sum(t for _, t in top_layer_times)
    for name, t in top_layer_times:
        print(f"  {name}: {t:.2f} ms ({t/top_total*100:.1f}%)")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code if exit_code is not None else 0)
    except KeyboardInterrupt:
        print("\n\n分析被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
