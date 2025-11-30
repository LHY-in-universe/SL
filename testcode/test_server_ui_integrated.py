"""
测试集成到 splitlearn-comm 的服务器监控 UI 功能
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
from splitlearn_manager import ManagedServer, ServerConfig, ModelConfig

def main():
    print("=" * 70)
    print("测试集成的服务器监控 UI (splitlearn-comm)")
    print("=" * 70)

    # 检查模型文件
    trunk_path = os.path.join(current_dir, "gpt2_trunk_full.pt")

    if not os.path.exists(trunk_path):
        print("\n❌ Trunk 模型文件不存在！")
        print(f"   Path: {trunk_path}")
        print("\n请运行: python testcode/prepare_models.py")
        return 1

    print("\n✓ 模型文件存在")
    print(f"  Trunk: {os.path.getsize(trunk_path) / (1024**2):.1f} MB")

    # 配置服务器
    print("\n配置服务器...")
    server_config = ServerConfig(
        host="localhost",
        port=50053,
        metrics_port=8002,
        log_level="INFO"
    )

    server = ManagedServer(config=server_config)

    # 加载模型
    print("加载 Trunk 模型...")
    model_config = ModelConfig(
        model_id="gpt2-trunk",
        model_path=trunk_path,
        model_type="pytorch",
        device="cpu",
        warmup=False,
        config={"input_shape": (1, 10, 768)}
    )

    server.load_model(model_config)
    print("✓ 模型加载完成")

    # 启动服务器
    print("\n启动服务器...")
    server.start()
    print("✓ 服务器已启动 (localhost:50053)")

    # 等待服务器完全启动
    time.sleep(2)

    print("\n" + "=" * 70)
    print("启动监控 UI")
    print("=" * 70)
    print("\n🚀 使用集成的 launch_monitoring_ui() 方法...")
    print("   监控地址: http://127.0.0.1:7861")
    print("   服务器地址: localhost:50053")
    print("\n✨ 这是集成到 splitlearn-comm 包中的新功能")
    print("   特性:")
    print("     • 实时服务器统计")
    print("     • 请求历史记录")
    print("     • 计算时间趋势图")
    print("     • 自动刷新 (每 2 秒)")
    print("\nPress Ctrl+C 停止\n")

    try:
        # 获取底层 gRPC 服务器
        grpc_server = server.grpc_server

        # 使用新的集成监控 UI 方法（在后台线程运行）
        grpc_server.launch_monitoring_ui(
            theme="default",       # 可选: "default", "dark", "light"
            refresh_interval=2,    # 每 2 秒刷新
            share=False,
            server_port=7861,
            blocking=False         # 在后台运行
        )

        print("✓ 监控 UI 已在后台启动")
        print("\n服务器继续运行，等待请求...")
        print("提示: 在另一个终端运行客户端来生成一些请求:")
        print("  python testcode/test_client_ui_integrated.py")
        print("\n按 Ctrl+C 停止服务器和 UI\n")

        # 保持服务器运行
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\n✓ 正在关闭...")
    except ImportError as e:
        print(f"\n❌ 导入错误: {e}")
        print("\n请安装 UI 依赖:")
        print("  pip install gradio pandas")
        print("或:")
        print("  pip install -e splitlearn-comm[ui]")
        return 1
    finally:
        server.stop()
        print("✓ 服务器已停止")

    return 0


if __name__ == "__main__":
    exit(main())
