#!/usr/bin/env python3
"""
Split Learning 客户端测试脚本
客户端本地加载 Bottom 和 Top 模型，连接到 Trunk 服务器执行完整的推理流程
"""

import os
import sys
import time
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from splitlearn_comm.quickstart import Client
from splitlearn_core.models.gpt2 import GPT2BottomModel, GPT2TopModel


def print_section(title):
    """打印分节标题"""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    print()


def print_step(step_num, title):
    """打印步骤标题"""
    print(f"\n[{step_num}] {title}")
    print("-" * 50)


def main():
    print_section("Split Learning 客户端 - 完整推理测试")

    # 配置 - 支持命令行参数或环境变量指定服务器地址
    import sys
    if len(sys.argv) > 1:
        TRUNK_SERVER = sys.argv[1]
    else:
        TRUNK_SERVER = os.getenv("TRUNK_SERVER", "localhost:50052")
    TEST_TEXT = "The future of artificial intelligence is"
    
    # 模型路径
    models_dir = Path(project_root) / "models"
    bottom_path = models_dir / "bottom" / "gpt2_2-10_bottom.pt"
    top_path = models_dir / "top" / "gpt2_2-10_top.pt"
    bottom_metadata_path = models_dir / "bottom" / "gpt2_2-10_bottom_metadata.json"
    top_metadata_path = models_dir / "top" / "gpt2_2-10_top_metadata.json"

    print("配置信息:")
    print(f"  Trunk Server:  {TRUNK_SERVER}")
    print(f"  Bottom 模型:   本地加载 ({bottom_path})")
    print(f"  Top 模型:      本地加载 ({top_path})")
    print(f"  测试文本: '{TEST_TEXT}'")

    # 步骤 1: 加载本地模型
    print_step(1, "加载本地模型 (Bottom 和 Top)")

    # 检查模型文件是否存在
    if not bottom_path.exists():
        print(f"\n❌ Bottom 模型文件不存在: {bottom_path}")
        return 1
    if not top_path.exists():
        print(f"\n❌ Top 模型文件不存在: {top_path}")
        return 1

    try:
        # 加载元数据
        with open(bottom_metadata_path, 'r') as f:
            bottom_metadata = json.load(f)
        with open(top_metadata_path, 'r') as f:
            top_metadata = json.load(f)

        # 加载 GPT-2 配置
        config = AutoConfig.from_pretrained("gpt2")

        # 加载 Bottom 模型
        print("加载 Bottom 模型...")
        bottom = GPT2BottomModel(config, end_layer=bottom_metadata['end_layer'])
        bottom.load_state_dict(torch.load(bottom_path, map_location='cpu', weights_only=True))
        bottom.eval()
        print(f"  ✓ Bottom 模型加载成功 (Layers 0-{bottom_metadata['end_layer']})")

        # 加载 Top 模型
        print("加载 Top 模型...")
        top = GPT2TopModel(config, start_layer=top_metadata['start_layer'])
        top.load_state_dict(torch.load(top_path, map_location='cpu', weights_only=True))
        top.eval()
        print(f"  ✓ Top 模型加载成功 (Layers {top_metadata['start_layer']}+)")

    except Exception as e:
        print(f"\n❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # 步骤 2: 连接到 Trunk 服务器
    print_step(2, "连接到 Trunk 服务器")

    try:
        print("连接 Trunk Server...")
        trunk_client = Client(TRUNK_SERVER)
        print("  ✓ Trunk Server 连接成功")

    except ConnectionRefusedError as e:
        print(f"\n❌ 连接失败: {e}")
        print("\n请确保 Trunk 服务器正在运行:")
        print("  python test/server/trunk_server.py")
        print("  或运行: bash test/start_all.sh")
        return 1

    # 步骤 3: 准备输入
    print_step(3, "准备输入数据")

    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print(f"原始文本: '{TEST_TEXT}'")
    input_ids = tokenizer.encode(TEST_TEXT, return_tensors="pt")
    print(f"Token IDs: {input_ids[0].tolist()}")
    print(f"输入形状: {input_ids.shape}")
    print(f"数据大小: {input_ids.numel() * 4 / 1024:.2f} KB")

    # 步骤 4: 执行推理
    print_step(4, "执行分布式推理")

    start_time = time.time()

    try:
        # 4.1 Bottom 模型 (本地)
        print("\n[4.1] Bottom 模型处理 (本地)...")
        print(f"  输入: {input_ids.shape}")
        print(f"  开始时间: {time.strftime('%H:%M:%S.%f')[:-3]}")
        bottom_start = time.time()
        with torch.no_grad():
            hidden_1 = bottom(input_ids)
        bottom_end = time.time()
        bottom_time = bottom_end - bottom_start
        print(f"  结束时间: {time.strftime('%H:%M:%S.%f')[:-3]}")
        print(f"  输出: {hidden_1.shape}")
        print(f"  耗时: {bottom_time:.3f} 秒 ({bottom_time*1000:.1f} ms)")
        print(f"  数据: {hidden_1.numel() * 4 / 1024:.2f} KB")
        print(f"  [本地处理] Bottom 模型执行时间: {bottom_time*1000:.1f}ms")

        # 4.2 Trunk 模型 (远程服务器)
        print("\n[4.2] Trunk 模型处理 (远程服务器)...")
        print(f"  输入: {hidden_1.shape}")
        print(f"  开始时间: {time.strftime('%H:%M:%S.%f')[:-3]}")
        trunk_start = time.time()
        hidden_2 = trunk_client.compute(hidden_1)
        trunk_end = time.time()
        trunk_time = trunk_end - trunk_start
        print(f"  结束时间: {time.strftime('%H:%M:%S.%f')[:-3]}")
        print(f"  输出: {hidden_2.shape}")
        print(f"  耗时: {trunk_time:.3f} 秒 ({trunk_time*1000:.1f} ms)")
        print(f"  数据: {hidden_2.numel() * 4 / 1024:.2f} KB")
        print(f"  [传输统计]")
        print(f"    总往返时间: {trunk_time*1000:.1f}ms")
        print(f"    (包含: 编码 + 网络传输 + 服务器计算 + 网络传输 + 解码)")
        # 尝试获取客户端统计信息
        try:
            stats = trunk_client._client.get_statistics()
            if stats:
                print(f"    平均网络时间: {stats.get('avg_network_time_ms', 0):.1f}ms")
                print(f"    平均服务器计算: {stats.get('avg_compute_time_ms', 0):.1f}ms")
        except:
            pass

        # 4.3 Top 模型 (本地)
        print("\n[4.3] Top 模型处理 (本地)...")
        print(f"  输入: {hidden_2.shape}")
        print(f"  开始时间: {time.strftime('%H:%M:%S.%f')[:-3]}")
        top_start = time.time()
        with torch.no_grad():
            output = top(hidden_2)
            logits = output.logits
        top_end = time.time()
        top_time = top_end - top_start
        print(f"  结束时间: {time.strftime('%H:%M:%S.%f')[:-3]}")
        print(f"  输出: {logits.shape}")
        print(f"  耗时: {top_time:.3f} 秒 ({top_time*1000:.1f} ms)")
        print(f"  数据: {logits.numel() * 4 / 1024:.2f} KB")
        print(f"  [本地处理] Top 模型执行时间: {top_time*1000:.1f}ms")

    except Exception as e:
        print(f"\n❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    total_time = time.time() - start_time

    # 步骤 5: 解析结果
    print_step(5, "解析预测结果")

    # 获取最后一个 token 的预测
    next_token_id = logits[0, -1].argmax().item()
    next_token = tokenizer.decode([next_token_id])

    print(f"输入文本: '{TEST_TEXT}'")
    print(f"预测的下一个词: '{next_token}' (ID: {next_token_id})")

    # 获取 top-5 预测
    print("\nTop-5 预测:")
    top5_ids = logits[0, -1].topk(5).indices.tolist()
    for i, token_id in enumerate(top5_ids, 1):
        token = tokenizer.decode([token_id])
        prob = torch.softmax(logits[0, -1], dim=0)[token_id].item()
        print(f"  {i}. '{token}' (概率: {prob:.4f})")

    # 步骤 6: 性能统计
    print_step(6, "详细性能统计")

    print(f"总耗时: {total_time:.3f} 秒 ({total_time*1000:.1f} ms)")
    print(f"\n[时间分解]")
    print(f"  Bottom (本地): {bottom_time:.3f} 秒 ({bottom_time*1000:.1f} ms) - {bottom_time/total_time*100:.1f}%")
    print(f"  Trunk (远程):  {trunk_time:.3f} 秒 ({trunk_time*1000:.1f} ms) - {trunk_time/total_time*100:.1f}%")
    print(f"    └─ 包含: 编码 + 网络传输(发送) + 服务器计算 + 网络传输(接收) + 解码")
    print(f"    └─ 注: 网络传输时间 = 总时间 - 服务器计算时间 (需从服务器响应获取)")
    print(f"  Top (本地):    {top_time:.3f} 秒 ({top_time*1000:.1f} ms) - {top_time/total_time*100:.1f}%")
    
    print(f"\n[本地处理总计]")
    local_total = bottom_time + top_time
    print(f"  本地处理时间: {local_total:.3f} 秒 ({local_total*1000:.1f} ms) - {local_total/total_time*100:.1f}%")
    print(f"  远程处理时间: {trunk_time:.3f} 秒 ({trunk_time*1000:.1f} ms) - {trunk_time/total_time*100:.1f}%")

    # 步骤 7: 清理
    print_step(7, "清理资源")

    trunk_client.close()
    print("  ✓ Trunk Client 已关闭")
    print("  ✓ 本地模型已释放")

    print_section("✓ 测试完成！")
    print("Split Learning 系统工作正常！")
    print()

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code if exit_code is not None else 0)
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
