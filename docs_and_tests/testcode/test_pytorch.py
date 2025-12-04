#!/usr/bin/env python3
"""
PyTorch 功能测试脚本

测试 PyTorch 的基本功能：
- 版本信息
- 设备检查
- 张量操作
- 基本运算
- 模型创建和推理
- 内存使用
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# 设置环境变量（如果需要）
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')


def print_separator(title=""):
    """打印分隔线"""
    if title:
        print("\n" + "=" * 70)
        print(f"  {title}")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)


def test_version_info():
    """测试 1: PyTorch 版本信息"""
    print_separator("测试 1: PyTorch 版本信息")
    
    print(f"\nPyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("  (使用 CPU)")
    
    print(f"\n线程配置:")
    print(f"  计算线程数: {torch.get_num_threads()}")
    print(f"  互操作线程数: {torch.get_num_interop_threads()}")


def test_device():
    """测试 2: 设备检查"""
    print_separator("测试 2: 设备检查")
    
    print(f"\n可用设备:")
    print(f"  CPU: ✅ 可用")
    
    if torch.cuda.is_available():
        print(f"  CUDA: ✅ 可用")
        print(f"  当前 CUDA 设备: {torch.cuda.current_device()}")
    else:
        print(f"  CUDA: ❌ 不可用")
    
    # 测试设备创建
    print(f"\n设备创建测试:")
    try:
        device_cpu = torch.device('cpu')
        print(f"  CPU 设备: {device_cpu} ✅")
        
        if torch.cuda.is_available():
            device_cuda = torch.device('cuda:0')
            print(f"  CUDA 设备: {device_cuda} ✅")
    except Exception as e:
        print(f"  ❌ 设备创建失败: {e}")


def test_tensor_creation():
    """测试 3: 张量创建"""
    print_separator("测试 3: 张量创建")
    
    print(f"\n创建不同类型的张量:")
    
    # 随机张量
    tensor1 = torch.randn(3, 4)
    print(f"  随机张量 (3x4):")
    print(f"    形状: {tensor1.shape}")
    print(f"    数据类型: {tensor1.dtype}")
    print(f"    设备: {tensor1.device}")
    print(f"    前3个值: {tensor1.flatten()[:3].tolist()}")
    
    # 全零张量
    tensor2 = torch.zeros(2, 3)
    print(f"\n  全零张量 (2x3):")
    print(f"    形状: {tensor2.shape}")
    print(f"    所有值: {tensor2.tolist()}")
    
    # 全一张量
    tensor3 = torch.ones(2, 3)
    print(f"\n  全一张量 (2x3):")
    print(f"    形状: {tensor3.shape}")
    print(f"    所有值: {tensor3.tolist()}")
    
    # 从列表创建
    tensor4 = torch.tensor([1.0, 2.0, 3.0, 4.0])
    print(f"\n  从列表创建:")
    print(f"    形状: {tensor4.shape}")
    print(f"    值: {tensor4.tolist()}")


def test_tensor_operations():
    """测试 4: 张量运算"""
    print_separator("测试 4: 张量运算")
    
    a = torch.randn(2, 3)
    b = torch.randn(2, 3)
    
    print(f"\n输入张量:")
    print(f"  a 形状: {a.shape}")
    print(f"  b 形状: {b.shape}")
    
    # 加法
    c = a + b
    print(f"\n运算结果:")
    print(f"  a + b:")
    print(f"    形状: {c.shape}")
    print(f"    前3个值: {c.flatten()[:3].tolist()}")
    
    # 乘法
    d = a * b
    print(f"\n  a * b:")
    print(f"    形状: {d.shape}")
    print(f"    前3个值: {d.flatten()[:3].tolist()}")
    
    # 矩阵乘法
    e = torch.randn(3, 4)
    f = torch.matmul(a, e)
    print(f"\n  a @ e (矩阵乘法):")
    print(f"    a 形状: {a.shape}, e 形状: {e.shape}")
    print(f"    结果形状: {f.shape}")
    
    # 求和
    sum_a = torch.sum(a)
    print(f"\n  sum(a):")
    print(f"    结果: {sum_a.item():.6f}")
    
    # 平均值
    mean_a = torch.mean(a)
    print(f"\n  mean(a):")
    print(f"    结果: {mean_a.item():.6f}")


def test_model_creation():
    """测试 5: 模型创建"""
    print_separator("测试 5: 模型创建和推理")
    
    # 创建简单模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 20)
            self.linear2 = nn.Linear(20, 5)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.linear1(x))
            x = self.linear2(x)
            return x
    
    model = SimpleModel()
    model.eval()
    
    print(f"\n模型结构:")
    print(f"  类型: {type(model).__name__}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 测试推理
    print(f"\n推理测试:")
    input_tensor = torch.randn(1, 10)
    print(f"  输入形状: {input_tensor.shape}")
    
    with torch.no_grad():
        start_time = time.time()
        output = model(input_tensor)
        inference_time = (time.time() - start_time) * 1000
    
    print(f"  输出形状: {output.shape}")
    print(f"  推理时间: {inference_time:.3f} ms")
    print(f"  输出前3个值: {output.flatten()[:3].tolist()}")


def test_device_transfer():
    """测试 6: 设备传输"""
    print_separator("测试 6: 设备传输")
    
    tensor = torch.randn(3, 4)
    print(f"\n原始张量:")
    print(f"  设备: {tensor.device}")
    print(f"  形状: {tensor.shape}")
    
    # CPU 操作
    tensor_cpu = tensor.cpu()
    print(f"\nCPU 张量:")
    print(f"  设备: {tensor_cpu.device}")
    print(f"  数据相同: {torch.allclose(tensor, tensor_cpu)}")
    
    # CUDA 操作（如果可用）
    if torch.cuda.is_available():
        tensor_cuda = tensor.cuda()
        print(f"\nCUDA 张量:")
        print(f"  设备: {tensor_cuda.device}")
        print(f"  数据相同: {torch.allclose(tensor, tensor_cuda.cpu())}")
    else:
        print(f"\nCUDA 不可用，跳过 CUDA 测试")


def test_memory():
    """测试 7: 内存使用"""
    print_separator("测试 7: 内存使用")
    
    print(f"\n内存信息:")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"\n  GPU {i} ({torch.cuda.get_device_name(i)}):")
            print(f"    总内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"    已分配: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"    已缓存: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
    else:
        print(f"  CUDA 不可用，无法显示 GPU 内存信息")
    
    # 测试内存分配
    print(f"\n内存分配测试:")
    large_tensor = torch.randn(1000, 1000)
    memory_mb = large_tensor.numel() * 4 / 1024 / 1024
    print(f"  创建大张量 (1000x1000):")
    print(f"    大小: {memory_mb:.2f} MB")
    print(f"    形状: {large_tensor.shape}")
    
    del large_tensor
    print(f"  ✓ 张量已释放")


def test_threading():
    """测试 8: 线程配置"""
    print_separator("测试 8: 线程配置")
    
    print(f"\n当前线程配置:")
    print(f"  计算线程数: {torch.get_num_threads()}")
    print(f"  互操作线程数: {torch.get_num_interop_threads()}")
    
    # 尝试设置线程
    print(f"\n线程设置测试:")
    try:
        torch.set_num_threads(2)
        print(f"  ✓ 设置计算线程数为 2")
        print(f"    当前值: {torch.get_num_threads()}")
    except Exception as e:
        print(f"  ⚠️  设置失败: {e}")
    
    try:
        torch.set_num_interop_threads(1)
        print(f"  ✓ 设置互操作线程数为 1")
        print(f"    当前值: {torch.get_num_interop_threads()}")
    except Exception as e:
        print(f"  ⚠️  设置失败: {e}")


def test_performance():
    """测试 9: 性能测试"""
    print_separator("测试 9: 性能测试")
    
    # 矩阵乘法性能
    print(f"\n矩阵乘法性能测试:")
    sizes = [(100, 100), (500, 500), (1000, 1000)]
    
    for size in sizes:
        a = torch.randn(size[0], size[1])
        b = torch.randn(size[1], size[0])
        
        # 预热
        _ = torch.matmul(a, b)
        
        # 测试
        start_time = time.time()
        for _ in range(10):
            c = torch.matmul(a, b)
        elapsed = (time.time() - start_time) / 10 * 1000
        
        print(f"  {size[0]}x{size[1]} @ {size[1]}x{size[0]}: {elapsed:.3f} ms")


def main():
    """主测试函数"""
    print("\n" + "=" * 70)
    print("PyTorch 功能测试")
    print("=" * 70)
    
    try:
        # 运行所有测试
        test_version_info()
        test_device()
        test_tensor_creation()
        test_tensor_operations()
        test_model_creation()
        test_device_transfer()
        test_memory()
        test_threading()
        test_performance()
        
        print_separator("测试总结")
        print("\n✅ 所有测试完成！")
        print("\nPyTorch 功能正常，可以正常使用。")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

