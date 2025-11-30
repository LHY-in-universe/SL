"""
测试集成到 splitlearn-comm 的客户端 UI 功能
"""
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, 'SplitLearning', 'src'))
sys.path.insert(0, os.path.join(project_root, 'splitlearn-comm', 'src'))

import torch
from transformers import AutoTokenizer
from splitlearn_comm import GRPCComputeClient

def main():
    print("=" * 70)
    print("测试集成的客户端 UI (splitlearn-comm)")
    print("=" * 70)

    # 检查模型文件
    bottom_path = os.path.join(current_dir, "gpt2_bottom_cached.pt")
    top_path = os.path.join(current_dir, "gpt2_top_cached.pt")

    if not os.path.exists(bottom_path) or not os.path.exists(top_path):
        print("\n❌ 模型文件不存在！")
        print(f"   Bottom: {bottom_path}")
        print(f"   Top: {top_path}")
        print("\n请运行: python testcode/prepare_models.py")
        return 1

    print("\n✓ 模型文件存在")
    print(f"  Bottom: {os.path.getsize(bottom_path) / (1024**2):.1f} MB")
    print(f"  Top: {os.path.getsize(top_path) / (1024**2):.1f} MB")

    # 加载模型
    print("\n加载模型...")
    bottom_model = torch.load(bottom_path, map_location='cpu', weights_only=False)
    top_model = torch.load(top_path, map_location='cpu', weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    print("✓ 模型加载完成")

    # 连接服务器
    print("\n连接服务器 (localhost:50053)...")
    client = GRPCComputeClient("localhost:50053", timeout=10.0)

    if not client.connect():
        print("❌ 无法连接到服务器！")
        print("\n请先启动服务器:")
        print("  python testcode/start_server.py")
        return 1

    print("✓ 已连接到服务器")

    # 获取客户端统计
    stats = client.get_statistics()
    print(f"\n客户端统计:")
    print(f"  总请求数: {stats['total_requests']}")

    # 启动 UI
    print("\n" + "=" * 70)
    print("启动 Gradio UI")
    print("=" * 70)
    print("\n🚀 使用集成的 launch_ui() 方法...")
    print("   访问地址: http://127.0.0.1:7860")
    print("\n✨ 这是集成到 splitlearn-comm 包中的新 UI 功能")
    print("   代码从 366 行减少到 15 行 (96% 减少)")
    print("\nPress Ctrl+C 停止\n")

    try:
        # 使用新的集成 UI 方法
        client.launch_ui(
            bottom_model=bottom_model,
            top_model=top_model,
            tokenizer=tokenizer,
            theme="default",  # 可选: "default", "dark", "light"
            share=False,
            server_port=7860
        )
    except KeyboardInterrupt:
        print("\n\n✓ UI 已停止")
    except ImportError as e:
        print(f"\n❌ 导入错误: {e}")
        print("\n请安装 UI 依赖:")
        print("  pip install gradio pandas")
        print("或:")
        print("  pip install -e splitlearn-comm[ui]")
        return 1
    finally:
        client.close()
        print("✓ 连接已关闭")

    return 0


if __name__ == "__main__":
    exit(main())
