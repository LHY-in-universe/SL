"""
一键演示：集成的 Gradio UI 功能

这个脚本展示如何使用集成到 splitlearn-comm 的 UI 功能。
它会在单个进程中启动服务器和客户端UI（分别在不同的线程中）。
"""
import sys
import os
import time
import threading

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, 'SplitLearning', 'src'))
sys.path.insert(0, os.path.join(project_root, 'splitlearn-comm', 'src'))
sys.path.insert(0, os.path.join(project_root, 'splitlearn-manager', 'src'))

import torch
from transformers import AutoTokenizer
from splitlearn_comm import GRPCComputeClient
from splitlearn_manager import ManagedServer, ServerConfig, ModelConfig


def start_server():
    """在后台线程启动服务器和监控UI"""
    print("\n[服务器线程] 正在启动...")

    # 检查模型
    trunk_path = os.path.join(current_dir, "gpt2_trunk_full.pt")
    if not os.path.exists(trunk_path):
        print(f"[服务器线程] ❌ Trunk 模型不存在: {trunk_path}")
        return

    # 配置服务器
    server_config = ServerConfig(
        host="localhost",
        port=50053,
        metrics_port=8002,
        log_level="WARNING"  # 减少日志输出
    )

    server = ManagedServer(config=server_config)

    # 加载模型
    print("[服务器线程] 加载 Trunk 模型...")
    model_config = ModelConfig(
        model_id="gpt2-trunk",
        model_path=trunk_path,
        model_type="pytorch",
        device="cpu",
        warmup=False,
        config={"input_shape": (1, 10, 768)}
    )

    server.load_model(model_config)
    print("[服务器线程] ✓ 模型加载完成")

    # 启动服务器
    server.start()
    print("[服务器线程] ✓ gRPC 服务器已启动 (localhost:50053)")

    # 等待服务器完全启动
    time.sleep(2)

    # 启动监控 UI（在后台）
    print("[服务器线程] 启动监控 UI...")
    grpc_server = server.grpc_server

    try:
        grpc_server.launch_monitoring_ui(
            theme="default",
            refresh_interval=2,
            share=False,
            server_port=7861,
            blocking=False  # 在后台运行
        )
        print("[服务器线程] ✓ 监控 UI 已启动 (http://127.0.0.1:7861)")
    except ImportError as e:
        print(f"[服务器线程] ⚠️  监控 UI 未启动: {e}")
        print("[服务器线程] 安装依赖: pip install gradio pandas")

    # 保持服务器运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[服务器线程] 正在关闭...")
    finally:
        server.stop()
        print("[服务器线程] ✓ 服务器已停止")


def start_client():
    """在主线程启动客户端UI"""
    print("\n[客户端线程] 正在启动...")

    # 检查模型
    bottom_path = os.path.join(current_dir, "gpt2_bottom_cached.pt")
    top_path = os.path.join(current_dir, "gpt2_top_cached.pt")

    if not os.path.exists(bottom_path) or not os.path.exists(top_path):
        print("[客户端线程] ❌ 模型文件不存在！")
        print(f"   Bottom: {bottom_path}")
        print(f"   Top: {top_path}")
        print("\n请运行: python testcode/prepare_models.py")
        return

    # 加载模型
    print("[客户端线程] 加载 Bottom 和 Top 模型...")
    bottom_model = torch.load(bottom_path, map_location='cpu', weights_only=False)
    top_model = torch.load(top_path, map_location='cpu', weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    print("[客户端线程] ✓ 模型加载完成")

    # 等待服务器启动
    print("[客户端线程] 等待服务器启动...")
    time.sleep(5)

    # 连接服务器
    print("[客户端线程] 连接到服务器...")
    client = GRPCComputeClient("localhost:50053", timeout=10.0)

    max_retries = 5
    for i in range(max_retries):
        if client.connect():
            print("[客户端线程] ✓ 已连接到服务器")
            break
        else:
            if i < max_retries - 1:
                print(f"[客户端线程] 重试连接 ({i+1}/{max_retries})...")
                time.sleep(2)
            else:
                print("[客户端线程] ❌ 无法连接到服务器")
                return

    # 启动客户端 UI
    print("[客户端线程] 启动客户端 UI...")

    try:
        # 🚀 这里是关键：使用集成的 launch_ui() 方法
        client.launch_ui(
            bottom_model=bottom_model,
            top_model=top_model,
            tokenizer=tokenizer,
            theme="default",
            share=False,
            server_port=7860
        )
    except ImportError as e:
        print(f"[客户端线程] ❌ 导入错误: {e}")
        print("\n请安装 UI 依赖:")
        print("  pip install gradio pandas")
    except KeyboardInterrupt:
        print("\n[客户端线程] UI 已停止")
    finally:
        client.close()
        print("[客户端线程] ✓ 连接已关闭")


def main():
    """主函数：启动演示"""
    print("=" * 70)
    print("集成 UI 功能演示")
    print("=" * 70)
    print("\n这个演示展示了如何使用集成到 splitlearn-comm 的 Gradio UI")
    print("\n将启动:")
    print("  1. gRPC 服务器 (localhost:50053)")
    print("  2. 服务器监控 UI (http://127.0.0.1:7861)")
    print("  3. 客户端生成 UI (http://127.0.0.1:7860)")
    print("\n🎯 代码对比:")
    print("  旧方法: 366 行代码")
    print("  新方法: 15 行代码")
    print("  减少: 96%")
    print("\n" + "=" * 70)

    # 检查模型文件
    required_files = [
        ("gpt2_bottom_cached.pt", "Bottom 模型"),
        ("gpt2_top_cached.pt", "Top 模型"),
        ("gpt2_trunk_full.pt", "Trunk 模型"),
    ]

    print("\n检查模型文件...")
    all_exist = True
    for filename, desc in required_files:
        path = os.path.join(current_dir, filename)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  ✓ {desc}: {size_mb:.1f} MB")
        else:
            print(f"  ✗ {desc}: 不存在")
            all_exist = False

    if not all_exist:
        print("\n❌ 缺少模型文件")
        print("请运行: python testcode/prepare_models.py")
        return 1

    # 启动服务器线程
    print("\n启动服务器线程...")
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # 在主线程启动客户端 UI (blocking)
    try:
        start_client()
    except KeyboardInterrupt:
        print("\n\n正在关闭...")

    print("\n" + "=" * 70)
    print("演示结束")
    print("=" * 70)
    print("\n✨ 体验如何？")
    print("\n使用集成的 UI 功能，你可以:")
    print("  • 用 1 行代码启动完整的 UI")
    print("  • 享受专业的界面设计")
    print("  • 获得实时的性能监控")
    print("  • 减少 96% 的样板代码")
    print("\n查看更多:")
    print("  • 文档: splitlearn-comm/examples/UI_README.md")
    print("  • 示例: splitlearn-comm/examples/")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
