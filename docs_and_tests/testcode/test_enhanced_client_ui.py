"""
测试增强的客户端UI

这个脚本演示了客户端新增的监控功能：
1. 实时日志显示
2. 服务器通信延迟统计（P95/P99等）
3. 延迟分布和趋势图

注意：需要先启动服务端才能运行此测试
"""

import torch
from splitlearn_comm import GRPCComputeClient
from splitlearn_comm.ui import ClientUI


def create_mock_models():
    """创建测试用的模型"""
    # 简单的 bottom 模型
    bottom_model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU()
    )
    bottom_model.eval()

    # 简单的 top 模型（带 logits 属性）
    class TopModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(20, 100)  # 假设词汇表大小为100

        def forward(self, x):
            logits = self.linear(x)
            # 返回一个带 logits 属性的对象
            class Output:
                def __init__(self, logits):
                    self.logits = logits
            return Output(logits)

    top_model = TopModel()
    top_model.eval()

    return bottom_model, top_model


def create_mock_tokenizer():
    """创建测试用的 tokenizer"""
    class MockTokenizer:
        def __init__(self):
            self.eos_token_id = 2
            self.vocab = ["<pad>", "<unk>", "<eos>"] + [f"word_{i}" for i in range(97)]

        def encode(self, text, return_tensors=None):
            # 简单地返回一些随机 token ids
            if return_tensors == "pt":
                return torch.randint(3, 100, (1, 5))
            return [3, 4, 5, 6, 7]

        def decode(self, token_ids):
            # 简单地返回一个单词
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()
            if isinstance(token_ids, list):
                token_id = token_ids[0] if len(token_ids) > 0 else 3
            else:
                token_id = token_ids
            return self.vocab[token_id % len(self.vocab)] + " "

    return MockTokenizer()


def main():
    print("=== 测试增强的客户端UI ===\n")

    # 1. 创建测试模型
    print("1. 创建测试模型...")
    bottom_model, top_model = create_mock_models()
    tokenizer = create_mock_tokenizer()
    print("   ✓ 模型创建完成\n")

    # 2. 连接到服务器
    print("2. 连接到gRPC服务器...")
    print("   服务器地址: localhost:50051")
    try:
        client = GRPCComputeClient("localhost:50051")
        client.connect()
        print("   ✓ 连接成功\n")
    except Exception as e:
        print(f"   ❌ 连接失败: {e}")
        print("\n   请确保服务端正在运行:")
        print("   python testcode/test_enhanced_monitoring.py")
        print("\n   或者使用快速启动脚本:")
        print("   python examples/quickstart_server.py")
        return

    # 3. 创建增强的客户端UI
    print("3. 创建增强的客户端UI...")
    ui = ClientUI(
        client=client,
        bottom_model=bottom_model,
        top_model=top_model,
        tokenizer=tokenizer,
        theme="default"
    )
    print("   ✓ UI 创建完成")
    print("   ✓ 监控管理器已初始化\n")

    # 4. 启动UI
    print("4. 启动客户端UI...")
    print("\n" + "="*60)
    print("📊 客户端监控 UI 包含 3 个标签页:")
    print("   1. 📝 文本生成 - 交互式文本生成界面")
    print("   2. 📝 实时日志 - 客户端操作日志")
    print("   3. ⏱️ 延迟分析 - 服务器通信延迟统计")
    print("="*60)
    print("\n🌐 UI 将在浏览器中打开: http://127.0.0.1:7860")
    print("\n⚡ 功能特性:")
    print("   • 实时记录每次服务器通信的延迟")
    print("   • 日志级别过滤 (DEBUG/INFO/WARNING/ERROR)")
    print("   • 延迟百分位统计 (P50/P95/P99)")
    print("   • 交互式 Plotly 图表")
    print("   • 每 2 秒自动刷新监控数据")
    print("\n💡 使用提示:")
    print("   • 在'文本生成'Tab生成文本，会自动记录延迟")
    print("   • 切换到'实时日志'Tab查看操作日志")
    print("   • 切换到'延迟分析'Tab查看通信性能")
    print("\n按 Ctrl+C 停止\n")

    try:
        ui.launch(
            share=False,
            server_port=7860,
            inbrowser=True,
            blocking=True
        )
    except KeyboardInterrupt:
        print("\n\n✓ 测试完成！")
    finally:
        try:
            client.disconnect()
            print("✓ 已断开与服务器的连接")
        except:
            pass


if __name__ == "__main__":
    main()
