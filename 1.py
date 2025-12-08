#!/usr/bin/env python3
"""
Qwen2-VL-2B详细架构分析
"""

from huggingface_hub import hf_hub_download
import json

def main():
    # 获取配置文件
    config_path = hf_hub_download(
        repo_id="Qwen/Qwen2-VL-2B-Instruct",
        filename="config.json"
    )
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    vision_config = config.get('vision_config', {})
    
    # 计算参数
    hidden_size = config['hidden_size']
    num_layers = config['num_hidden_layers']
    num_heads = config['num_attention_heads']
    kv_heads = config['num_key_value_heads']
    vocab_size = config['vocab_size']
    
    # 单层参数估算
    layer_params = (
        # 注意力
        4 * hidden_size * hidden_size +  # QKV+O投影
        # MLP (SwiGLU)
        3 * hidden_size * config['intermediate_size'] +  # gate/up/down
        # 归一化 (忽略)
        0
    )
    
    # 总参数
    total_params = (
        vocab_size * hidden_size +  # 词嵌入
        layer_params * num_layers +  # Transformer层
        hidden_size * vocab_size     # 输出层
    )
    
    print("="*70)
    print("Qwen2-VL-2B 详细架构分析")
    print("="*70)
    
    print(f"\n📊 总参数: ~{total_params/1e9:.2f}B")
    print(f"  视觉编码器: 675M (官方)")
    print(f"  语言模型: ~{total_params/1e9 - 0.675:.2f}B")
    print(f"  总计: ~2.175B")
    
    print("\n🏗️  语言模型架构:")
    print(f"  • 类型: Decoder-only Transformer")
    print(f"  • 隐藏维度: {hidden_size}")
    print(f"  • 层数: {num_layers}")
    print(f"  • 注意力: GQA {num_heads}:{kv_heads}")
    print(f"  • MLP扩展: {hidden_size} → {config['intermediate_size']} (×{config['intermediate_size']/hidden_size:.1f})")
    print(f"  • 激活函数: {config['hidden_act']}")
    print(f"  • 归一化: RMSNorm (ε={config['rms_norm_eps']})")
    print(f"  • 位置编码: RoPE (θ={config.get('rope_theta', 1000000)})")
    
    print("\n👁️  视觉编码器:")
    print(f"  • 类型: Vision Transformer")
    print(f"  • 隐藏维度: {vision_config.get('hidden_size', hidden_size)}")
    print(f"  • Patch大小: {vision_config.get('patch_size', 14)}")
    print(f"  • 动态分辨率: 是")
    print(f"  • 位置编码: M-ROPE (多模态)")
    
    print("\n🔗 多模态融合:")
    print(f"  • 方式: 共享维度 ({vision_config.get('hidden_size', hidden_size)} = {hidden_size})")
    print(f"  • 融合: 视觉特征作为语言模型输入前缀")
    print(f"  • Token数: 动态 (根据图像分辨率)")
    
    print("\n⚡ 优化特性:")
    print(f"  • GQA优化: KV缓存减少 {(1 - kv_heads/num_heads)*100:.0f}%")
    print(f"  • 滑动窗口: {config.get('sliding_window', 32768)} tokens")
    print(f"  • 长上下文: {config['max_position_embeddings']:,} tokens")
    
    print("\n💾 内存需求 (估算):")
    dtypes = {
        'FP16': 2,
        'INT8': 1,
        'INT4': 0.5
    }
    
    for name, bytes_per_param in dtypes.items():
        param_memory = 2_175_000_000 * bytes_per_param / (1024**3)
        total_memory = param_memory * 2.5  # 包含激活值等
        print(f"  • {name}: {param_memory:.1f}GB / {total_memory:.1f}GB")
    
    print("="*70)

if __name__ == "__main__":
    main()