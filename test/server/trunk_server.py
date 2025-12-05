#!/usr/bin/env python3
"""
Trunk Server - GPT-2 模型的 Trunk 部分
处理隐藏状态，输出变换后的隐藏状态

端口: 50052
"""

import os
import sys
import warnings
import signal
import atexit

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

# 尝试导入 SplitLearnMonitor（可选）
try:
    from splitlearn_monitor import ServerMonitor
    MONITOR_AVAILABLE = True
except ImportError:
    MONITOR_AVAILABLE = False
    ServerMonitor = None

# 全局 monitor 变量
monitor = None


def cleanup_monitor():
    """清理监控资源"""
    global monitor
    if monitor:
        try:
            monitor.stop()
            # 保存监控报告
            try:
                report_path = monitor.save_report(format="html")
                print(f"\n[监控] 监控报告已保存: {report_path}")
            except Exception as e:
                print(f"\n[监控] 保存监控报告失败: {e}")
        except Exception as e:
            print(f"\n[监控] 停止监控失败: {e}")


def signal_handler(signum, frame):
    """信号处理器"""
    print("\n\n收到停止信号，正在清理...")
    cleanup_monitor()
    sys.exit(0)


def main():
    global monitor
    
    print("=" * 70)
    print("Trunk Server - GPT-2 Trunk Model")
    print("=" * 70)
    print()
    print("配置信息:")
    print("  模型: GPT-2")
    print("  组件: Trunk (layers 2-9)")
    print("  端口: 50052")
    print("  设备: CPU")
    print("  拆分点: [2, 10]")
    print()
    print("功能: 接收隐藏状态 → 输出变换后的隐藏状态")
    print()
    
    # 初始化 SplitLearnMonitor（如果可用）
    if MONITOR_AVAILABLE:
        try:
            from datetime import datetime
            server_name = f"trunk_server_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            monitor = ServerMonitor(
                server_name=server_name,
                sampling_interval=0.1,  # 每0.1秒采样一次
                enable_gpu=False,  # 当前使用CPU
                auto_start=True
            )
            print(f"[监控] SplitLearnMonitor 已启动 (会话: {server_name})")
            
            # 注册清理函数
            atexit.register(cleanup_monitor)
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        except Exception as e:
            print(f"[监控] SplitLearnMonitor 启动失败: {e}，将使用基础监控")
            monitor = None
    else:
        print("[监控] SplitLearnMonitor 不可用，将使用基础监控")
        monitor = None
    
    print()
    print("按 Ctrl+C 停止服务器")
    print("-" * 70)
    print()

    # 创建并启动服务器
    server = ManagedServer(
        model_type="gpt2",
        component="trunk",
        port=50052,
        split_points=[2, 10],
        device="cpu",
        max_workers=1  # 单线程模式
    )

    try:
        server.start()
    except KeyboardInterrupt:
        print("\n\n收到停止信号...")
    finally:
        cleanup_monitor()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✓ Trunk Server 已停止")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
