#!/bin/bash
# 测试交互式客户端（非交互模式）

echo "======================================================================"
echo "测试交互式客户端"
echo "======================================================================"
echo ""

# 使用 Python 脚本模拟交互式输入
python3 << 'PYTHON_SCRIPT'
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

# 导入交互式客户端的主要函数
from test.client.interactive_client import load_models, generate_text
from splitlearn_comm.quickstart import Client

print("1. 加载模型...")
try:
    bottom, top, tokenizer = load_models()
    print("   ✓ 模型加载成功\n")
except Exception as e:
    print(f"   ✗ 模型加载失败: {e}")
    sys.exit(1)

print("2. 连接到服务器...")
try:
    trunk_client = Client("localhost:50052")
    print("   ✓ 连接成功\n")
except Exception as e:
    print(f"   ✗ 连接失败: {e}")
    print("   请确保服务器正在运行: bash test/start_all.sh")
    sys.exit(1)

print("3. 测试文本生成...")
test_prompts = [
    "The future of AI is",
    "Machine learning is",
    "Hello, how are you?"
]

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n   测试 {i}/{len(test_prompts)}: '{prompt}'")
    try:
        generated = generate_text(
            bottom, top, tokenizer, trunk_client,
            prompt,
            max_length=20,
            temperature=0.8
        )
        # 只显示新生成的部分
        if generated.startswith(prompt):
            response = generated[len(prompt):].strip()
        else:
            response = generated
        print(f"   AI 回复: {response}")
    except Exception as e:
        print(f"   ✗ 生成失败: {e}")
        import traceback
        traceback.print_exc()

trunk_client.close()
print("\n✓ 测试完成！")
PYTHON_SCRIPT


