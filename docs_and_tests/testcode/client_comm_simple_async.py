#!/usr/bin/env python3
"""
gRPC 客户端测试脚本 - 异步版本

连接异步服务器，测试通信功能
使用异步客户端（使用 grpc.aio）
"""

import os
import sys
import asyncio
import time
import torch
import logging

# 设置环境变量（必须在导入 grpc 之前）
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, 'SplitLearnComm', 'src'))

# 配置详细日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

import grpc.aio
from splitlearn_comm.protocol import compute_service_pb2, compute_service_pb2_grpc
from splitlearn_comm.core import TensorCodec

# 测试配置
SERVER_ADDRESS = "localhost:50056"  # 异步服务器端口
TIMEOUT = 30.0


def print_tensor_info(tensor, name, prefix="   "):
    """打印张量详细信息"""
    print(f"{prefix}形状: {tensor.shape}")
    print(f"{prefix}数据类型: {tensor.dtype}")
    print(f"{prefix}数据大小: {tensor.numel() * 4 / 1024:.2f} KB")
    print(f"{prefix}最小值: {tensor.min().item():.6f}")
    print(f"{prefix}最大值: {tensor.max().item():.6f}")
    print(f"{prefix}平均值: {tensor.mean().item():.6f}")


class AsyncGRPCClient:
    """简单的异步 gRPC 客户端"""
    
    def __init__(self, server_address: str, timeout: float = 30.0):
        self.server_address = server_address
        self.timeout = timeout
        self.channel = None
        self.stub = None
        self.codec = TensorCodec()
        self.request_count = 0
    
    async def connect(self):
        """异步连接服务器"""
        print(f"\n📡 连接服务器: {self.server_address}")
        print("   正在连接...")
        
        try:
            self.channel = grpc.aio.insecure_channel(
                self.server_address,
                options=[
                    ("grpc.keepalive_time_ms", 30000),
                    ("grpc.keepalive_timeout_ms", 10000),
                ]
            )
            self.stub = compute_service_pb2_grpc.ComputeServiceStub(self.channel)
            
            # 健康检查
            try:
                response = await asyncio.wait_for(
                    self.stub.HealthCheck(compute_service_pb2.HealthRequest()),
                    timeout=5.0
                )
                print("   ✓ 连接成功！")
                print(f"   服务器状态: {response.status}")
                return True
            except asyncio.TimeoutError:
                print("   ⚠️  连接超时")
                return False
            except Exception as e:
                print(f"   ⚠️  健康检查失败: {e}")
                return True  # 连接可能成功，只是健康检查失败
                
        except Exception as e:
            print(f"   ❌ 连接失败: {e}")
            return False
    
    async def compute(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """异步发送计算请求"""
        self.request_count += 1
        
        # 编码输入
        data, shape = self.codec.encode(input_tensor)
        
        # 创建请求
        request = compute_service_pb2.ComputeRequest(
            data=data,
            shape=list(shape)
        )
        
        # 发送请求
        try:
            response = await asyncio.wait_for(
                self.stub.Compute(request),
                timeout=self.timeout
            )
            
            # 解码输出
            output = self.codec.decode(
                data=response.data,
                shape=tuple(response.shape)
            )
            
            return output
            
        except asyncio.TimeoutError:
            raise TimeoutError(f"请求超时（{self.timeout} 秒）")
        except Exception as e:
            raise RuntimeError(f"计算请求失败: {e}")
    
    async def get_service_info(self):
        """获取服务器信息"""
        if self.stub is None:
            return None
        
        try:
            response = await asyncio.wait_for(
                self.stub.GetServiceInfo(compute_service_pb2.ServiceInfoRequest()),
                timeout=5.0
            )
            
            return {
                "service_name": response.service_name,
                "version": response.version,
                "device": response.device,
                "total_requests": response.total_requests,
                "uptime_seconds": response.uptime_seconds,
            }
        except Exception as e:
            print(f"   ⚠️  获取服务器信息失败: {e}")
            return None
    
    async def close(self):
        """关闭连接"""
        if self.channel:
            await self.channel.close()
            print("   ✓ 连接已关闭")


async def test_connection(client):
    """测试连接"""
    print("\n" + "=" * 70)
    print("🔌 连接测试")
    print("=" * 70)
    
    if await client.connect():
        return True
    else:
        print("   ❌ 连接失败！")
        print(f"\n💡 请确保服务器正在运行:")
        print(f"   python testcode/server_comm_simple.py")
        return False


async def test_compute(client, request_num=1):
    """测试计算并显示详细信息"""
    print("\n" + "=" * 70)
    print(f"📤 发送请求 #{request_num}")
    print("=" * 70)
    
    # 创建测试输入
    test_input = torch.randn(1, 10, 768)
    
    print(f"\n📊 准备发送的数据:")
    print_tensor_info(test_input, "输入数据")
    
    # 计算数据大小
    input_size_kb = test_input.numel() * 4 / 1024
    
    # 发送请求
    print(f"\n🚀 发送计算请求...")
    print(f"   数据大小: {input_size_kb:.2f} KB")
    print(f"   正在传输...")
    
    start_time = time.time()
    
    try:
        output = await client.compute(test_input)
        
        total_time = (time.time() - start_time) * 1000
        
        print(f"\n📥 收到响应")
        print("=" * 70)
        
        # 显示输出数据信息
        print(f"\n📊 接收到的数据:")
        print_tensor_info(output, "输出数据")
        
        # 验证计算结果（应该是 input * 2 + 1）
        expected = test_input * 2 + 1
        if torch.allclose(output, expected, atol=1e-5):
            print(f"\n✅ 计算结果正确: output = input * 2 + 1")
        else:
            print(f"\n⚠️  计算结果不符合预期")
        
        # 计算传输统计
        output_size_kb = output.numel() * 4 / 1024
        total_size_kb = input_size_kb + output_size_kb
        
        print(f"\n📡 传输统计:")
        print(f"   发送数据: {input_size_kb:.2f} KB")
        print(f"   接收数据: {output_size_kb:.2f} KB")
        print(f"   总传输: {total_size_kb:.2f} KB")
        print(f"   总耗时: {total_time:.2f} ms")
        if total_time > 0:
            print(f"   吞吐量: {total_size_kb / (total_time / 1000):.2f} KB/s")
        
        # 验证输出形状
        if output.shape == test_input.shape:
            print(f"\n✅ 输出形状正确: {output.shape}")
        else:
            print(f"\n⚠️  输出形状不符合预期: {output.shape} (期望: {test_input.shape})")
        
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 请求失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_multiple_requests(client, num_requests=5):
    """测试多次请求"""
    print("\n" + "=" * 70)
    print(f"🔄 多次请求测试 ({num_requests} 次)")
    print("=" * 70)
    
    successes = 0
    total_time = 0.0
    total_data = 0.0
    
    for i in range(num_requests):
        print(f"\n--- 请求 {i+1}/{num_requests} ---")
        
        test_input = torch.randn(1, 5, 768)
        input_size_kb = test_input.numel() * 4 / 1024
        
        start_time = time.time()
        try:
            output = await client.compute(test_input)
            elapsed = (time.time() - start_time) * 1000
            
            # 验证结果
            expected = test_input * 2 + 1
            if torch.allclose(output, expected, atol=1e-5):
                output_size_kb = output.numel() * 4 / 1024
                request_data = input_size_kb + output_size_kb
                
                total_time += elapsed
                total_data += request_data
                successes += 1
                
                print(f"   ✓ 成功 (耗时: {elapsed:.2f} ms, 数据: {request_data:.2f} KB)")
            else:
                print(f"   ⚠️  结果不正确")
            
        except Exception as e:
            print(f"   ❌ 失败: {e}")
    
    print(f"\n📊 总结:")
    print(f"   成功: {successes}/{num_requests}")
    if successes > 0:
        print(f"   总耗时: {total_time:.2f} ms")
        print(f"   平均耗时: {total_time/successes:.2f} ms")
        print(f"   总传输: {total_data:.2f} KB")
        if total_time > 0:
            print(f"   平均吞吐量: {total_data / (total_time / 1000):.2f} KB/s")
    
    return successes == num_requests


async def test_service_info(client):
    """测试服务信息"""
    print("\n" + "=" * 70)
    print("ℹ️  服务信息查询")
    print("=" * 70)
    
    info = await client.get_service_info()
    
    if info:
        print(f"\n📋 服务器信息:")
        print(f"   服务名: {info.get('service_name', 'N/A')}")
        print(f"   版本: {info.get('version', 'N/A')}")
        print(f"   设备: {info.get('device', 'N/A')}")
        print(f"   总请求数: {info.get('total_requests', 0)}")
        print(f"   运行时间: {info.get('uptime_seconds', 0):.1f} 秒")
        return True
    else:
        print("   ❌ 无法获取服务器信息")
        return False


async def async_main():
    """异步主函数"""
    print("\n" + "=" * 70)
    print("💻 gRPC 客户端测试（异步版本 - 测试通信功能）")
    print("=" * 70)
    print(f"\n📡 服务器地址: {SERVER_ADDRESS}")
    print(f"⏱️  超时时间: {TIMEOUT} 秒")
    print(f"💡 服务器执行: output = input * 2 + 1")
    print(f"✅ 使用异步版本（无线程竞争）")
    print()
    
    # 创建客户端
    client = AsyncGRPCClient(
        server_address=SERVER_ADDRESS,
        timeout=TIMEOUT
    )
    
    # 连接服务器
    if not await test_connection(client):
        return 1
    
    try:
        # 测试服务信息
        await test_service_info(client)
        
        # 测试单次计算
        await test_compute(client, request_num=1)
        
        # 测试多次请求
        await test_multiple_requests(client, num_requests=5)
        
        print("\n" + "=" * 70)
        print("✅ 所有测试完成")
        print("=" * 70)
        
        print(f"\n📊 客户端统计:")
        print(f"   总请求数: {client.request_count}")
        
    finally:
        print("\n🔌 关闭连接...")
        await client.close()
    
    return 0


def main():
    """主函数"""
    try:
        return asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\n\n⏹️  测试被用户中断")
        return 1
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

