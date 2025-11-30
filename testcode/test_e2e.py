"""
端到端测试：启动服务器和客户端，验证基本通信
"""
import sys
import os
import time
import multiprocessing
import signal

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, 'SplitLearning', 'src'))
sys.path.insert(0, os.path.join(project_root, 'splitlearn-comm', 'src'))
sys.path.insert(0, os.path.join(project_root, 'splitlearn-manager', 'src'))

def run_server(stop_event):
    """在独立进程中运行服务器"""
    try:
        import torch
        from splitlearn_manager import ManagedServer, ServerConfig, ModelConfig

        # 检查模型文件
        model_path = os.path.join(current_dir, "gpt2_trunk_full.pt")
        if not os.path.exists(model_path):
            print("❌ Trunk 模型文件不存在")
            return

        print("🚀 启动服务器...")

        # 配置服务器
        server_config = ServerConfig(
            host="localhost",
            port=50053,
            metrics_port=8002,
            log_level="WARNING"  # 减少日志输出
        )

        server = ManagedServer(config=server_config)

        # 加载模型
        model_config = ModelConfig(
            model_id="gpt2-trunk",
            model_path=model_path,
            model_type="pytorch",
            device="cpu",
            warmup=False,  # 跳过预热加快启动
            config={"input_shape": (1, 10, 768)}
        )

        server.load_model(model_config)
        print("✓ 服务器模型加载完成")

        # 启动服务
        server.start()
        print("✓ 服务器已启动 (port 50053)")

        # 等待停止信号
        while not stop_event.is_set():
            time.sleep(0.5)

        server.stop()
        print("✓ 服务器已停止")

    except Exception as e:
        print(f"❌ 服务器错误: {e}")
        import traceback
        traceback.print_exc()

def test_client_connection():
    """测试客户端连接和基本功能"""
    try:
        import torch
        from splitlearn_comm import GRPCComputeClient

        print("\n🔌 测试客户端连接...")

        # 加载模型
        bottom_path = os.path.join(current_dir, "gpt2_bottom_cached.pt")
        top_path = os.path.join(current_dir, "gpt2_top_cached.pt")

        if not os.path.exists(bottom_path) or not os.path.exists(top_path):
            print("❌ 客户端模型文件不存在")
            return False

        print("  加载 Bottom 模型...")
        bottom_model = torch.load(bottom_path, map_location='cpu', weights_only=False)

        print("  加载 Top 模型...")
        top_model = torch.load(top_path, map_location='cpu', weights_only=False)

        print("  连接服务器...")
        client = GRPCComputeClient("localhost:50053", timeout=5.0)

        if not client.connect():
            print("❌ 无法连接到服务器")
            return False

        print("✓ 客户端已连接")

        # 测试推理
        print("\n🧪 测试推理流程...")
        input_ids = torch.randint(0, 50257, (1, 5))

        print("  1. Bottom 前向传播...")
        with torch.no_grad():
            hidden_bottom = bottom_model(input_ids)
        print(f"     输出形状: {hidden_bottom.shape}")

        print("  2. 发送到 Trunk (远程)...")
        hidden_trunk = client.compute(hidden_bottom)
        print(f"     输出形状: {hidden_trunk.shape}")

        print("  3. Top 前向传播...")
        with torch.no_grad():
            output = top_model(hidden_trunk)
            logits = output.logits
        print(f"     输出形状: {logits.shape}")

        # 验证形状
        expected_shape = (1, 5, 50257)
        if logits.shape == expected_shape:
            print(f"\n✅ 推理成功！输出形状正确: {logits.shape}")
        else:
            print(f"\n❌ 输出形状错误: 期望 {expected_shape}, 实际 {logits.shape}")
            return False

        client.close()
        print("✓ 客户端已断开")

        return True

    except Exception as e:
        print(f"❌ 客户端测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行端到端测试"""
    print("=" * 70)
    print("端到端测试：服务器 + 客户端通信")
    print("=" * 70)

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

    # 启动服务器
    print("\n" + "=" * 70)
    print("步骤 1: 启动服务器")
    print("=" * 70)

    multiprocessing.set_start_method('spawn', force=True)
    stop_event = multiprocessing.Event()

    server_process = multiprocessing.Process(target=run_server, args=(stop_event,))
    server_process.start()

    # 等待服务器启动
    print("\n等待服务器启动...")
    time.sleep(5)

    if not server_process.is_alive():
        print("❌ 服务器启动失败")
        return 1

    # 测试客户端
    print("\n" + "=" * 70)
    print("步骤 2: 测试客户端通信")
    print("=" * 70)

    success = test_client_connection()

    # 停止服务器
    print("\n" + "=" * 70)
    print("步骤 3: 清理")
    print("=" * 70)
    print("停止服务器...")
    stop_event.set()
    server_process.join(timeout=5)

    if server_process.is_alive():
        print("⚠️  服务器未正常停止，强制终止...")
        server_process.terminate()
        server_process.join()

    # 总结
    print("\n" + "=" * 70)
    print("测试结果")
    print("=" * 70)

    if success:
        print("✅ 端到端测试通过！")
        print("\n所有组件工作正常:")
        print("  ✓ 服务器可以启动和停止")
        print("  ✓ 客户端可以连接服务器")
        print("  ✓ Bottom → Trunk → Top 推理流程正常")
        print("\nGradio 应用可以正常使用了！")
        print("启动命令: python testcode/client_with_gradio.py")
        return 0
    else:
        print("❌ 端到端测试失败")
        return 1

if __name__ == "__main__":
    exit(main())
