"""
Quickstart Server Example

This example demonstrates the simplest way to start a SplitLearn server.
It loads a model and serves it via gRPC.

Usage:
    python quickstart_server.py

Then run quickstart_client.py in another terminal to test the server.
"""

import os
import warnings

# 抑制 gRPC 和 PyTorch 的内部 mutex 警告（这些警告不影响功能）
# 这些警告来自 gRPC/PyTorch 的内部实现，是正常的
os.environ['GRPC_VERBOSITY'] = 'ERROR'  # 只显示错误，不显示警告
os.environ['GLOG_minloglevel'] = '2'    # 抑制 C++ 日志

# 在导入任何 PyTorch 相关模块之前设置线程数
# 这必须在导入 torch 或任何使用 torch 的模块之前
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

# 抑制 Python 警告
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*coroutine.*')
warnings.filterwarnings('ignore', message='.*mutex.*')

# 现在导入模块（此时环境变量已设置）
from splitlearn_manager.quickstart import ManagedServer

def main():
    print("=== SplitLearn Quickstart Server ===\n")

    print("Starting server with GPT-2 Trunk model...")
    print("  Model: gpt2")
    print("  Component: trunk")
    print("  Port: 50051")
    print("  Device: auto-detect (CUDA if available, else CPU)\n")

    print("Press Ctrl+C to stop the server\n")
    print("-" * 50)

    # Create and start server (blocking)
    # max_workers=1 表示单线程模式，可以避免 mutex 警告
    # 如果需要更高并发，可以设置为更大的值（如 10）
    server = ManagedServer(
        model_type="gpt2",
        component="trunk",
        port=50051,
        max_workers=1  # 单线程模式，避免 mutex 警告
    )

    # This will block until Ctrl+C
    server.start()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
