#!/usr/bin/env python3
"""
ViT 模型信息追踪脚本
运行此脚本可以看到 ViT 模型在每一步处理中张量的形状和大小变化
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

# 配置
BATCH_SIZE = 2
IMAGE_SIZE = 224
PATCH_SIZE = 16
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 196
EMBED_DIM = 768
NUM_HEADS = 12
NUM_LAYERS = 12
NUM_CLASSES = 1000


def print_tensor_info(name: str, tensor: torch.Tensor, show_values: bool = False):
    """打印张量的详细信息"""
    shape = tensor.shape
    numel = tensor.numel()
    dtype = tensor.dtype
    memory_mb = numel * tensor.element_size() / (1024 * 1024)
    
    print(f"\n{'='*70}")
    print(f"【{name}】")
    print(f"{'='*70}")
    print(f"形状 (Shape): {shape}")
    print(f"元素数量 (Num Elements): {numel:,}")
    print(f"数据类型 (Dtype): {dtype}")
    print(f"内存占用 (Memory): {memory_mb:.4f} MB")
    
    if len(shape) > 0:
        print(f"维度说明:")
        for i, dim in enumerate(shape):
            print(f"  维度 {i}: {dim}")
    
    if show_values and numel <= 20:
        print(f"值: {tensor}")
    print()


class PatchEmbedding(nn.Module):
    """Patch Embedding 层"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x, verbose=True):
        if verbose:
            print_tensor_info("输入图像 (Input Image)", x)
        
        # Conv2d 会自动进行 patch 分割和投影
        x = self.proj(x)  # [B, C, H, W] -> [B, embed_dim, H', W']
        if verbose:
            print_tensor_info("Patch Embedding 后 (After Conv2d)", x)
        
        # Flatten: [B, embed_dim, H', W'] -> [B, embed_dim, H'*W']
        B, C, H, W = x.shape
        x = x.flatten(2)  # [B, embed_dim, num_patches]
        if verbose:
            print_tensor_info("Flatten 后 (After Flatten)", x)
        
        # Transpose: [B, embed_dim, num_patches] -> [B, num_patches, embed_dim]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        if verbose:
            print_tensor_info("转置后 (After Transpose) - 最终 Patch Embeddings", x)
        
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention"""
    def __init__(self, embed_dim=768, num_heads=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x, verbose=True, detailed=False):
        if verbose and detailed:
            print_tensor_info("MHSA 输入 (MHSA Input)", x)
        
        B, N, C = x.shape
        
        # QKV 计算
        qkv = self.qkv(x)  # [B, N, 3*embed_dim]
        
        # 分割为 Q, K, V
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 注意力分数计算
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn = attn.softmax(dim=-1)
        
        # 加权求和
        x = (attn @ v)  # [B, num_heads, N, head_dim]
        
        # 拼接多头
        x = x.transpose(1, 2)  # [B, N, num_heads, head_dim]
        x = x.reshape(B, N, C)  # [B, N, embed_dim]
        
        # 输出投影
        x = self.proj(x)
        
        if verbose:
            print_tensor_info("MHSA 输出 (MHSA Output)", x)
        
        return x


class FeedForward(nn.Module):
    """Feed-Forward Network"""
    def __init__(self, embed_dim=768, mlp_ratio=4):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim * mlp_ratio)
        self.fc2 = nn.Linear(embed_dim * mlp_ratio, embed_dim)
        self.act = nn.GELU()
        
    def forward(self, x, verbose=True, detailed=False):
        # 扩展维度
        x = self.fc1(x)  # [B, N, embed_dim] -> [B, N, 4*embed_dim]
        x = self.act(x)
        # 压缩维度
        x = self.fc2(x)  # [B, N, 4*embed_dim] -> [B, N, embed_dim]
        
        if verbose:
            print_tensor_info("FFN 输出 (FFN Output)", x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer Encoder Block"""
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, mlp_ratio)
        
    def forward(self, x, layer_idx=0, verbose=True, detailed=False):
        # Pre-norm + Attention + Residual
        residual = x
        x = self.norm1(x)
        x = self.attn(x, verbose=False, detailed=False)
        x = x + residual
        
        # Pre-norm + FFN + Residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x, verbose=False, detailed=False)
        x = x + residual
        
        return x


class VisionTransformer(nn.Module):
    """完整的 Vision Transformer 模型"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, 
                 embed_dim=768, num_heads=12, num_layers=12, num_classes=1000):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch Embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # Class Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        print_tensor_info("Class Token (可学习参数)", self.cls_token, show_values=False)
        
        # Position Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        print_tensor_info("Position Embedding (可学习参数)", self.pos_embed, show_values=False)
        
        # Transformer Layers
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) 
            for _ in range(num_layers)
        ])
        
        # Layer Norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification Head
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x, verbose=True):
        if verbose:
            print(f"\n{'='*70}")
            print(f"Vision Transformer 前向传播")
            print(f"{'='*70}")
        
        B = x.shape[0]
        
        # Patch Embedding
        x = self.patch_embed(x, verbose=verbose)
        
        # 添加 Class Token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        if verbose:
            print_tensor_info("扩展后的 Class Token", cls_tokens)
        
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches+1, embed_dim]
        if verbose:
            print_tensor_info("拼接 Class Token 后", x)
        
        # 添加 Position Embedding
        x = x + self.pos_embed
        if verbose:
            print_tensor_info("添加 Position Embedding 后", x)
        
        # Transformer Layers (不显示详细信息)
        if verbose:
            print(f"\n{'='*70}")
            print(f"通过 {len(self.blocks)} 层 Transformer Encoder...")
            print(f"{'='*70}")
        
        for i, block in enumerate(self.blocks):
            x = block(x, layer_idx=i, verbose=False, detailed=False)
        
        # Final Layer Norm
        x = self.norm(x)
        if verbose:
            print_tensor_info("Transformer 输出 (After All Layers)", x)
        
        # 提取 CLS Token
        cls_token_final = x[:, 0]  # [B, embed_dim]
        if verbose:
            print_tensor_info("提取 CLS Token", cls_token_final)
        
        # Classification Head
        logits = self.head(cls_token_final)  # [B, num_classes]
        if verbose:
            print_tensor_info("分类头输出 (Classification Head Output)", logits)
        
        return logits


def main():
    print("="*70)
    print("ViT 模型信息追踪")
    print("="*70)
    print(f"\n配置:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Image Size: {IMAGE_SIZE} × {IMAGE_SIZE}")
    print(f"  Patch Size: {PATCH_SIZE} × {PATCH_SIZE}")
    print(f"  Number of Patches: {NUM_PATCHES}")
    print(f"  Embedding Dimension: {EMBED_DIM}")
    print(f"  Number of Heads: {NUM_HEADS}")
    print(f"  Number of Layers: {NUM_LAYERS}")
    print(f"  Number of Classes: {NUM_CLASSES}")
    print()
    
    # 创建模型
    model = VisionTransformer(
        img_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES
    )
    
    # 创建随机输入
    x = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    
    # 前向传播
    print("\n" + "="*70)
    print("开始前向传播...")
    print("="*70)
    
    with torch.no_grad():
        output = model(x, verbose=True)
    
    print("\n" + "="*70)
    print("前向传播完成！")
    print("="*70)
    
    # 统计总参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型统计:")
    print(f"  总参数数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  参数大小: {total_params * 4 / (1024**2):.2f} MB (float32)")


if __name__ == "__main__":
    main()
