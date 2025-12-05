#!/usr/bin/env python3
"""
完整模型监控示例

演示如何监控单服务器运行完整模型的场景
"""
import time
import torch
from splitlearn_monitor import FullModelMonitor


def simulate_model_inference(batch_size=1, seq_length=128):
    """模拟模型推理"""
    # 模拟推理耗时
    time.sleep(0.05 + torch.rand(1).item() * 0.03)
    return torch.randn(batch_size, seq_length, 768)


def main():
    print("=" * 80)
    print("完整模型监控示例")
    print("=" * 80)
    print()

    # 方式一：使用上下文管理器
    print("方式一：使用上下文管理器")
    with FullModelMonitor(
        model_name="gpt2_full",
        sampling_interval=0.1,
        enable_gpu=False
    ) as monitor:

        # 运行多次推理
        print("运行 50 次推理...")
        for i in range(50):
            with monitor.track_inference(metadata={"step": i+1, "batch_size": 1}):
                output = simulate_model_inference()

            if (i + 1) % 10 == 0:
                print(f"  完成 {i+1}/50 次推理")

        print("\n推理完成！")

    # 打印摘要
    monitor.print_summary()

    # 保存报告
    report_path = monitor.save_report(format="html")
    print(f"报告已保存: {report_path}")

    print("\n" + "=" * 80)

    # 方式二：手动控制
    print("\n方式二：手动控制")
    monitor2 = FullModelMonitor(
        model_name="gpt2_full_manual",
        sampling_interval=0.1,
        enable_gpu=False
    )

    monitor2.start()
    print("监控已启动")

    print("运行 30 次推理...")
    for i in range(30):
        start = time.time()
        output = simulate_model_inference()
        duration_ms = (time.time() - start) * 1000

        # 手动记录
        monitor2.record_inference(duration_ms, metadata={"step": i+1})

        if (i + 1) % 10 == 0:
            print(f"  完成 {i+1}/30 次推理")

    monitor2.stop()
    print("\n推理完成！监控已停止")

    # 打印统计
    stats = monitor2.get_statistics()
    print(f"\n统计信息:")
    print(f"  总推理次数: {stats['total_inferences']}")
    print(f"  平均耗时: {stats['inference_stats']['mean_ms']:.1f} ms")

    # 保存报告
    report_path2 = monitor2.save_report(format="html")
    print(f"报告已保存: {report_path2}")

    print("\n" + "=" * 80)
    print("示例完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
