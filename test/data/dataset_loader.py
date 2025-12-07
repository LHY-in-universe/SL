#!/usr/bin/env python3
"""
简单的数据集加载工具

支持从 HuggingFace datasets 加载小数据集，用于微调测试
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Dict, Any
from transformers import AutoTokenizer


class SimpleTextDataset(Dataset):
    """简单的文本数据集"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        """
        初始化数据集
        
        Args:
            texts: 文本列表
            tokenizer: tokenizer 实例
            max_length: 最大序列长度
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Tokenize 所有文本
        print(f"正在处理 {len(texts)} 条文本...")
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
    
    def __len__(self):
        return len(self.encodings['input_ids'])
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.encodings['input_ids'][idx].clone()  # 用于语言建模
        }


def load_simple_synthetic_dataset(num_samples: int = 50, max_length: int = 128) -> List[str]:
    """
    创建简单的合成数据集（用于快速测试）
    
    Args:
        num_samples: 样本数量
        max_length: 最大序列长度
    
    Returns:
        文本列表
    """
    texts = [
        "The future of artificial intelligence is",
        "Machine learning has revolutionized",
        "Deep neural networks can process",
        "Natural language processing enables",
        "Computer vision systems can recognize",
        "Reinforcement learning algorithms learn",
        "Transformer models have transformed",
        "Large language models demonstrate",
        "Distributed systems enable scalable",
        "Cloud computing provides flexible",
        "The development of algorithms",
        "Data science combines statistics",
        "Software engineering principles guide",
        "Human computer interaction focuses",
        "Information technology connects",
    ]
    
    # 扩展数据集
    expanded_texts = []
    for i in range(num_samples):
        expanded_texts.append(texts[i % len(texts)])
    
    return expanded_texts


def load_huggingface_dataset(
    dataset_name: str = "wikitext",
    subset: str = "wikitext-2-raw-v1",
    split: str = "train",
    max_samples: int = 100,
    text_column: str = "text"
) -> List[str]:
    """
    从 HuggingFace datasets 加载数据集
    
    Args:
        dataset_name: 数据集名称
        subset: 数据集子集
        split: 数据分割（train/validation/test）
        max_samples: 最大样本数
        text_column: 文本列名称
    
    Returns:
        文本列表
    """
    try:
        from datasets import load_dataset
        
        print(f"正在从 HuggingFace 加载数据集: {dataset_name}/{subset}")
        
        # 加载数据集
        dataset = load_dataset(dataset_name, subset, split=f"{split}[:{max_samples}]")
        
        # 提取文本
        texts = []
        for item in dataset:
            text = item.get(text_column, "")
            if text and len(text.strip()) > 10:  # 过滤太短的文本
                texts.append(text.strip())
        
        print(f"成功加载 {len(texts)} 条文本")
        return texts
        
    except ImportError:
        print("⚠️  datasets 库未安装，使用合成数据集")
        print("   安装命令: pip install datasets")
        return load_simple_synthetic_dataset(max_samples)
    except Exception as e:
        print(f"⚠️  加载 HuggingFace 数据集失败: {e}")
        print("   使用合成数据集")
        return load_simple_synthetic_dataset(max_samples)


def create_dataloader(
    texts: List[str],
    tokenizer,
    batch_size: int = 2,
    max_length: int = 128,
    shuffle: bool = True
) -> DataLoader:
    """
    创建数据加载器
    
    Args:
        texts: 文本列表
        tokenizer: tokenizer 实例
        batch_size: 批次大小
        max_length: 最大序列长度
        shuffle: 是否打乱数据
    
    Returns:
        DataLoader 实例
    """
    dataset = SimpleTextDataset(texts, tokenizer, max_length=max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_test_dataset(
    dataset_type: str = "synthetic",
    num_samples: int = 50,
    tokenizer=None,
    batch_size: int = 2,
    max_length: int = 128
) -> DataLoader:
    """
    加载测试数据集（便捷函数）
    
    Args:
        dataset_type: 数据集类型 ("synthetic" 或 "wikitext")
        num_samples: 样本数量
        tokenizer: tokenizer 实例（如果为 None，会创建默认的）
        batch_size: 批次大小
        max_length: 最大序列长度
    
    Returns:
        DataLoader 实例
    """
    # 如果没有提供 tokenizer，创建默认的
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    # 加载文本
    if dataset_type == "synthetic":
        texts = load_simple_synthetic_dataset(num_samples=num_samples)
    elif dataset_type == "wikitext":
        texts = load_huggingface_dataset(max_samples=num_samples)
    else:
        raise ValueError(f"未知的数据集类型: {dataset_type}")
    
    # 创建 DataLoader
    return create_dataloader(texts, tokenizer, batch_size=batch_size, max_length=max_length)


if __name__ == "__main__":
    # 测试数据集加载
    from transformers import AutoTokenizer
    
    print("=" * 70)
    print("数据集加载工具测试")
    print("=" * 70)
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 测试合成数据集
    print("\n1. 测试合成数据集")
    dataloader = load_test_dataset("synthetic", num_samples=10, batch_size=2)
    print(f"   DataLoader 创建成功，共 {len(dataloader)} 个批次")
    
    # 测试一个批次
    batch = next(iter(dataloader))
    print(f"   批次键: {batch.keys()}")
    print(f"   input_ids 形状: {batch['input_ids'].shape}")
    print(f"   labels 形状: {batch['labels'].shape}")
    
    # 测试 HuggingFace 数据集（如果可用）
    print("\n2. 测试 HuggingFace 数据集")
    try:
        dataloader_hf = load_test_dataset("wikitext", num_samples=10, batch_size=2)
        print(f"   DataLoader 创建成功，共 {len(dataloader_hf)} 个批次")
    except Exception as e:
        print(f"   ⚠️  HuggingFace 数据集不可用: {e}")
