import sys
import os
import torch
from transformers import AutoTokenizer

# 添加 SplitLearning/src 到路径，这样即使没有 pip install 也可以运行
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
splitlearn_path = os.path.join(project_root, 'SplitLearning', 'src')
sys.path.append(splitlearn_path)

try:
    from splitlearn import ModelFactory
except ImportError:
    print("无法导入 splitlearn。请确保路径正确或已安装该包。")
    print(f"尝试加载路径: {splitlearn_path}")
    sys.exit(1)

def test_split_model():
    print("=== 开始测试 SplitLearning 库 ===")

    # 配置
    model_type = 'gpt2'
    model_name = 'gpt2' # 这将尝试从 HuggingFace 下载模型
    
    print(f"正在加载并拆分模型: {model_name}...")
    try:
        # 创建拆分模型
        # Split point 1: 第 2 层之后 (Bottom 包含层 0, 1)
        # Split point 2: 第 10 层之后 (Trunk 包含层 2-9)
        # Top 包含层 10-11 (GPT-2 base 总共 12 层)
        bottom, trunk, top = ModelFactory.create_split_models(
            model_type=model_type,
            model_name_or_path=model_name,
            split_point_1=2,
            split_point_2=10,
            device='cpu'
        )
        print("✅ 模型拆分成功!")
        
        # 测试推理流程
        print("\n准备输入数据...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        text = "Hello, this is a test run for SplitLearning."
        input_ids = tokenizer.encode(text, return_tensors="pt")
        print(f"输入文本: '{text}'")
        print(f"Input IDs shape: {input_ids.shape}")
        
        print("\n开始分段推理...")
        
        # 1. Bottom Model
        print("1. 运行 Bottom Model...")
        hidden_1 = bottom(input_ids)
        print(f"   -> Bottom 输出 shape: {hidden_1.shape}")
        
        # 2. Trunk Model
        print("2. 运行 Trunk Model...")
        hidden_2 = trunk(hidden_1)
        print(f"   -> Trunk 输出 shape: {hidden_2.shape}")
        
        # 3. Top Model
        print("3. 运行 Top Model...")
        output = top(hidden_2)
        logits = output.logits
        print(f"   -> Top 输出 Logits shape: {logits.shape}")
        
        # 结果解码
        predicted_id = logits[0, -1].argmax()
        predicted_token = tokenizer.decode(predicted_id)
        print(f"\n预测的下一个 Token: '{predicted_token}'")
        print("=== 测试完成 ===")
        
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_split_model()
