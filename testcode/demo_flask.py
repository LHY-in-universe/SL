"""
使用 Flask 的简单 Web 界面
比 Gradio 更轻量，更容易访问
"""
from flask import Flask, render_template_string, request, jsonify
import sys
import os
import torch
from transformers import AutoTokenizer

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, 'SplitLearning', 'src'))
sys.path.append(os.path.join(project_root, 'splitlearn-comm', 'src'))

from splitlearn_comm import GRPCComputeClient

app = Flask(__name__)

# 全局变量
bottom_model = None
top_model = None
tokenizer = None
client = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Split Learning Demo</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .status {
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
            font-weight: bold;
        }
        .status.success {
            background: #d4edda;
            color: #155724;
        }
        .status.error {
            background: #f8d7da;
            color: #721c24;
        }
        .status.info {
            background: #d1ecf1;
            color: #0c5460;
        }
        input[type="text"] {
            width: 100%;
            padding: 12px;
            margin: 10px 0;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
        }
        button {
            background: #007bff;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin: 5px;
        }
        button:hover {
            background: #0056b3;
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        #output {
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
            min-height: 100px;
            white-space: pre-wrap;
            font-family: monospace;
        }
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Split Learning 演示</h1>
        <p style="text-align: center; color: #666;">
            Bottom(本地) → Trunk(远程服务器) → Top(本地)
        </p>
        
        <div id="status" class="status info">
            点击"初始化"按钮开始
        </div>
        
        <button onclick="initialize()" id="initBtn">初始化模型并连接服务器</button>
        
        <div style="margin-top: 30px;">
            <label><strong>输入 Prompt:</strong></label>
            <input type="text" id="prompt" placeholder="例如: The future of AI is..." value="The future of AI is">
            <button onclick="generate()" id="genBtn" disabled>开始生成</button>
        </div>
        
        <div class="loading" id="loading">⏳ 生成中...</div>
        
        <div id="output"></div>
    </div>
    
    <script>
        function setStatus(message, type) {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = 'status ' + type;
        }
        
        async function initialize() {
            const btn = document.getElementById('initBtn');
            btn.disabled = true;
            setStatus('正在初始化...', 'info');
            
            try {
                const response = await fetch('/init');
                const data = await response.json();
                
                if (data.success) {
                    setStatus(data.message, 'success');
                    document.getElementById('genBtn').disabled = false;
                } else {
                    setStatus('初始化失败: ' + data.message, 'error');
                    btn.disabled = false;
                }
            } catch (error) {
                setStatus('错误: ' + error, 'error');
                btn.disabled = false;
            }
        }
        
        async function generate() {
            const prompt = document.getElementById('prompt').value;
            const output = document.getElementById('output');
            const loading = document.getElementById('loading');
            const btn = document.getElementById('genBtn');
            
            btn.disabled = true;
            loading.style.display = 'block';
            output.textContent = '';
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({prompt: prompt})
                });
                
                const data = await response.json();
                
                if (data.success) {
                    output.textContent = data.result;
                    setStatus('生成完成！', 'success');
                } else {
                    output.textContent = '错误: ' + data.message;
                    setStatus('生成失败', 'error');
                }
            } catch (error) {
                output.textContent = '错误: ' + error;
                setStatus('生成失败', 'error');
            } finally {
                loading.style.display = 'none';
                btn.disabled = false;
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/init')
def init():
    global bottom_model, top_model, tokenizer, client
    
    try:
        # 加载模型
        bottom_path = os.path.join(current_dir, "gpt2_bottom_cached.pt")
        top_path = os.path.join(current_dir, "gpt2_top_cached.pt")
        
        bottom_model = torch.load(bottom_path, map_location='cpu')
        top_model = torch.load(top_path, map_location='cpu')
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        
        # 连接服务器
        client = GRPCComputeClient("127.0.0.1:50053", timeout=20.0)
        if not client.connect():
            return jsonify({'success': False, 'message': '无法连接到服务器'})
        
        return jsonify({'success': True, 'message': '✅ 初始化成功！模型已加载，服务器已连接'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/generate', methods=['POST'])
def generate():
    global bottom_model, top_model, tokenizer, client
    
    try:
        data = request.json
        prompt = data.get('prompt', '')
        
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        result = prompt
        
        for _ in range(20):
            with torch.no_grad():
                hidden_bottom = bottom_model(input_ids)
            
            hidden_trunk = client.compute(hidden_bottom, model_id="gpt2-trunk")
            
            with torch.no_grad():
                output = top_model(hidden_trunk)
                logits = output.logits
            
            next_token_id = logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            
            result += tokenizer.decode(next_token_id[0])
        
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    print("=" * 70)
    print("Split Learning Flask 服务器")
    print("=" * 70)
    print("\n请在浏览器中访问: http://localhost:5000")
    print("=" * 70)
    app.run(host='0.0.0.0', port=5000, debug=False)
