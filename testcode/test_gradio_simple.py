"""
最简单的 Gradio 测试
只有一个文本框和按钮，测试基本功能
"""
import gradio as gr

def greet(name):
    return f"Hello {name}!"

# 创建最简单的界面
demo = gr.Interface(
    fn=greet,
    inputs=gr.Textbox(label="输入你的名字"),
    outputs=gr.Textbox(label="问候语"),
    title="Gradio 基础测试"
)

if __name__ == "__main__":
    print("启动最简单的 Gradio 测试...")
    print("=" * 50)
    print("请在浏览器中访问: http://127.0.0.1:7777")
    print("=" * 50)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7777,
        share=False,
        inbrowser=True  # 自动打开浏览器
    )
