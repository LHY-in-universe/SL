"""
自动化测试 Gradio 接口
"""
from gradio_client import Client

def test_gradio_api():
    print("正在连接 Gradio API (http://127.0.0.1:7777)...")
    try:
        client = Client("http://127.0.0.1:7777")
        
        # 1. 调用初始化
        print("1. 测试初始化...")
        result_init = client.predict(fn_index=0) # 假设第一个按钮是 fn_index=0
        print(f"初始化结果: {result_init}")
        
        if "成功" not in result_init:
            print("❌ 初始化失败")
            return

        # 2. 调用生成
        print("\n2. 测试文本生成...")
        result_gen = client.predict(
            "The future of AI is", # prompt
            20,                    # max_length
            fn_index=1             # 假设生成按钮是 fn_index=1
        )
        print(f"生成结果: {result_gen}")
        print("\n✅ Gradio 接口测试通过！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print("提示: 可能是 Gradio 版本兼容性问题或 API 未暴露")

if __name__ == "__main__":
    test_gradio_api()
