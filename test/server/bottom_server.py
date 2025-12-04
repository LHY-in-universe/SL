#!/usr/bin/env python3
"""
Bottom Server - GPT-2 模型的 Bottom 部分
处理输入 token IDs，输出隐藏状态

端口: 50051
"""

import os
import sys
import warnings

# 必须在导入 torch 之前设置环境变量
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

# 抑制警告
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*mutex.*')

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from splitlearn_manager.quickstart import ManagedServer


def main():
    print("=" * 70)
    print("Bottom Server - GPT-2 Bottom Model")
    print("=" * 70)
    print()
    print("配置信息:")
    print("  模型: GPT-2")
    print("  组件: Bottom (layers 0-1)")
    print("  端口: 50051")
    print("  设备: CPU")
    print("  拆分点: [2, 10]")
    print()
    print("功能: 接收 token IDs → 输出隐藏状态")
    print()
    print("按 Ctrl+C 停止服务器")
    print("-" * 70)
    print()

    # 创建并启动服务器
    server = ManagedServer(
        model_type="gpt2",
        component="bottom",
        port=50051,
        split_points=[2, 10],
        device="cpu",
        max_workers=1  # 单线程模式
    )

    server.start()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✓ Bottom Server 已停止")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
