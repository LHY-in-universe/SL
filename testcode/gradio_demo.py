"""
Split Learning Gradio 演示 (使用 Gradio 3.x)
"""
import sys
import os
import torch
from transformers import AutoTokenizer
import gradio as gr

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, 'SplitLearning', 'src'))
sys.path.append(os.path.join(project_root, 'splitlearn-comm', 'src'))

from splitlearn_comm import GRPCComputeClient

# 全局变量
bottom_model = None
top_model = None
tokenizer = None
client = None

def load_models():
    """加载本地模型和连接服务器"""
    global bottom_model, top_model, tokenizer, client
    
    try:
        # 加载模型
        bottom_path = os.path.join(current_dir, "gpt2_bottom_cached.pt")
        top_path = os.path.join(current_dir, "gpt2_top_cached.pt")
        
        if not os.path.exists(bottom_path) or not os.path.exists(top_path):
            return "❌ 模型文件不存在！\n请先运行: python testcode/prepare_models.py"
        
        bottom_model = torch.load(bottom_path, map_location='cpu')
        top_model = torch.load(top_path, map_location='cpu')
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        
        # 连接服务器
        client = GRPCComputeClient("127.0.0.1:50053", timeout=20.0)
        if not client.connect():
            return "❌ 无法连接到服务器！\n请确保服务器正在运行:\npython testcode/start_server.py"
            
        return "✅ 初始化成功！\n\n- Bottom 模型已加载\n- Top 模型已加载\n- 服务器已连接 (127.0.0.1:50053)\n\n现在可以开始生成文本了！"
    except Exception as e:
        import traceback
        return f"❌ 初始化失败:\n{str(e)}\n\n{traceback.format_exc()}"

def generate_text(prompt, max_length=20):
    """生成文本"""
    global bottom_model, top_model, tokenizer, client
    
    if client is None:
        return "请先点击'初始化'按钮！"
    
    try:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        generated_text = prompt
        
        for _ in range(max_length):
            # Bottom (本地)
            with torch.no_grad():
                hidden_bottom = bottom_model(input_ids)
            
            # Trunk (远程)
            hidden_trunk = client.compute(hidden_bottom, model_id="gpt2-trunk")
            
            # Top (本地)
            with torch.no_grad():
                output = top_model(hidden_trunk)
                logits = output.logits
            
            # 采样
            next_token_id = logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            
            generated_text += tokenizer.decode(next_token_id[0])
        
        # 获取统计信息
        stats = client.get_statistics()
        stats_text = f"\n\n统计信息:\n- 总请求: {stats['total_requests']}\n- 平均延迟: {stats['avg_network_time_ms']:.2f}ms\n- 平均计算: {stats['avg_compute_time_ms']:.2f}ms"
        
        return generated_text + stats_text
        
    except Exception as e:
        return f"❌ 生成失败:\n{str(e)}"

def get_system_status():
    """获取系统状态（服务器资源 + 客户端统计）"""
    global client
    
    if client is None:
        return "请先初始化模型"
        
    try:
        # 1. 获取服务器信息
        server_info = client.get_service_info()
        if not server_info:
            server_status = "无法获取服务器信息"
        else:
            custom = server_info.get("custom_info", {})
            
            # 处理嵌套的字符串字典
            import ast
            if "custom_info" in custom and isinstance(custom["custom_info"], str):
                try:
                    nested_custom = ast.literal_eval(custom["custom_info"])
                    if isinstance(nested_custom, dict):
                        custom = nested_custom
                except:
                    pass
            
            cpu = custom.get("cpu_percent", "N/A")
            mem = custom.get("memory_mb", "N/A")
            mem_pct = custom.get("memory_percent", "N/A")
            reqs = server_info.get("total_requests", 0)
            uptime = server_info.get("uptime_seconds", 0)
            
            server_status = (
                f"🌍 服务器状态 (Trunk)\n"
                f"-------------------\n"
                f"CPU 使用率: {cpu}%\n"
                f"内存使用: {mem} MB ({mem_pct}%)\n"
                f"总处理请求: {reqs}\n"
                f"运行时间: {int(uptime)}秒"
            )

        # 2. 获取客户端统计
        stats = client.get_statistics()
        client_status = (
            f"🚀 客户端性能\n"
            f"-------------------\n"
            f"本地已发请求: {stats.get('total_requests', 0)}\n"
            f"平均网络延迟: {stats.get('avg_network_time_ms', 0):.2f} ms\n"
            f"平均计算耗时: {stats.get('avg_compute_time_ms', 0):.2f} ms"
        )
        
        return server_status + "\n\n" + client_status
        
    except Exception as e:
        return f"获取状态失败: {str(e)}"

# 创建 Gradio 界面
with gr.Blocks(title="Split Learning Demo") as demo:
    gr.Markdown("# 🚀 Split Learning 分布式推理演示")
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("**架构**: Bottom(本地) → Trunk(远程服务器) → Top(本地)")
            with gr.Row():
                init_btn = gr.Button("初始化模型并连接服务器", variant="primary")
            
            status_box = gr.Textbox(label="初始化状态", value="未初始化", lines=3)
            
            gr.Markdown("---")
            
            with gr.Row():
                prompt_box = gr.Textbox(
                    label="输入 Prompt", 
                    placeholder="例如: The future of AI is...",
                    value="The future of AI is"
                )
            
            with gr.Row():
                max_length = gr.Slider(
                    minimum=5, 
                    maximum=50, 
                    value=20, 
                    step=1, 
                    label="生成长度 (tokens)"
                )
            
            generate_btn = gr.Button("开始生成", variant="primary")
            output_box = gr.Textbox(label="生成结果", lines=8)

        with gr.Column(scale=1):
            gr.Markdown("### 📊 实时监控")
            monitor_box = gr.Textbox(label="系统状态", lines=15, value="等待连接...")
            refresh_btn = gr.Button("刷新状态")

    # 事件绑定
    init_btn.click(fn=load_models, outputs=status_box)
    generate_btn.click(fn=generate_text, inputs=[prompt_box, max_length], outputs=output_box)
    
    # 监控刷新
    refresh_btn.click(fn=get_system_status, outputs=monitor_box)
    # 自动刷新 (每 2 秒) - 注意：Gradio 3.x 使用 every 参数
    demo.load(fn=get_system_status, inputs=None, outputs=monitor_box, every=2.0)

if __name__ == "__main__":
    print("=" * 70)
    print("Split Learning Gradio 客户端")
    print("=" * 70)
    print("请在浏览器中访问: http://127.0.0.1:7788")
    print("=" * 70)
    demo.queue()  # 显式启用队列
    demo.launch(
        server_name="127.0.0.1",
        server_port=7788,
        share=False,
        inbrowser=True
    )
