#!/usr/bin/env python3
"""
Top Server - GPT-2 模型的 Top 部分
处理隐藏状态，输出最终的 logits

端口: 50053
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
    print("Top Server - GPT-2 Top Model")
    print("=" * 70)
    print()
    print("配置信息:")
    print("  模型: GPT-2")
    print("  组件: Top (layers 10-11)")
    print("  端口: 50053")
    print("  设备: CPU")
    print("  拆分点: [2, 10]")
    print()
    print("功能: 接收隐藏状态 → 输出 logits (词汇表概率)")
    print()
    print("按 Ctrl+C 停止服务器")
    print("-" * 70)
    print()

    # 创建并启动服务器
    server = ManagedServer(
        model_type="gpt2",
        component="top",
        port=50053,
        split_points=[2, 10],
        device="cpu",
        max_workers=1  # 单线程模式
    )

    server.start()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✓ Top Server 已停止")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
