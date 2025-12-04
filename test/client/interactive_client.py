#!/usr/bin/env python3
"""
Split Learning 交互式客户端
在终端输入文本，获得 AI 回复
"""

import os
import sys
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from splitlearn_comm.quickstart import Client
from splitlearn_core.models.gpt2 import GPT2BottomModel, GPT2TopModel


def load_models():
    """加载本地模型"""
    print("正在加载模型...")
    
    # 模型路径
    models_dir = Path(project_root) / "models"
    bottom_path = models_dir / "bottom" / "gpt2_2-10_bottom.pt"
    top_path = models_dir / "top" / "gpt2_2-10_top.pt"
    bottom_metadata_path = models_dir / "bottom" / "gpt2_2-10_bottom_metadata.json"
    top_metadata_path = models_dir / "top" / "gpt2_2-10_top_metadata.json"

    # 检查模型文件
    if not bottom_path.exists() or not top_path.exists():
        print("❌ 模型文件不存在！")
        print(f"请确保以下文件存在：")
        print(f"  - {bottom_path}")
        print(f"  - {top_path}")
        sys.exit(1)

    # 加载元数据
    with open(bottom_metadata_path, 'r') as f:
        bottom_metadata = json.load(f)
    with open(top_metadata_path, 'r') as f:
        top_metadata = json.load(f)

    # 加载 GPT-2 配置
    config = AutoConfig.from_pretrained("gpt2")

    # 加载 Bottom 模型
    print("  加载 Bottom 模型...", end=" ", flush=True)
    bottom = GPT2BottomModel(config, end_layer=bottom_metadata['end_layer'])
    bottom.load_state_dict(torch.load(bottom_path, map_location='cpu', weights_only=True))
    bottom.eval()
    print("✓")

    # 加载 Top 模型
    print("  加载 Top 模型...", end=" ", flush=True)
    top = GPT2TopModel(config, start_layer=top_metadata['start_layer'])
    top.load_state_dict(torch.load(top_path, map_location='cpu', weights_only=True))
    top.eval()
    print("✓")

    # 加载 tokenizer
    print("  加载 Tokenizer...", end=" ", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓")

    print("✓ 所有模型加载完成！\n")
    return bottom, top, tokenizer


def generate_text(bottom, top, tokenizer, trunk_client, prompt, max_length=50, temperature=0.7):
    """
    生成文本
    
    Args:
        bottom: Bottom 模型
        top: Top 模型
        tokenizer: Tokenizer
        trunk_client: Trunk 服务器客户端
        prompt: 输入文本
        max_length: 最大生成长度
        temperature: 温度参数（控制随机性）
    """
    # 编码输入
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    generated_ids = input_ids.clone()
    input_length = input_ids.shape[1]

    with torch.no_grad():
        for _ in range(max_length):
            # Bottom 模型处理
            hidden_1 = bottom(generated_ids)

            # Trunk 模型处理（远程）
            hidden_2 = trunk_client.compute(hidden_1)

            # Top 模型处理
            output = top(hidden_2)
            logits = output.logits

            # 获取最后一个 token 的 logits
            next_token_logits = logits[0, -1, :] / temperature

            # 应用 softmax 并采样
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

            # 添加到生成的序列 (generated_ids 是 [1, seq_len], next_token_id 是 [1])
            # 需要 reshape 为 [1, 1] 才能正确 concat
            generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=1)

            # 如果生成了结束符，停止
            if next_token_id.item() == tokenizer.eos_token_id:
                break

    return generated_ids, input_length


def main():
    print("=" * 70)
    print("Split Learning 交互式客户端")
    print("=" * 70)
    print()

    # 配置
    TRUNK_SERVER = "localhost:50052"

    # 加载模型
    try:
        bottom, top, tokenizer = load_models()
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 连接到服务器
    print(f"连接到 Trunk 服务器 ({TRUNK_SERVER})...", end=" ", flush=True)
    try:
        trunk_client = Client(TRUNK_SERVER)
        print("✓\n")
    except Exception as e:
        print(f"❌\n连接失败: {e}")
        print("\n请确保 Trunk 服务器正在运行:")
        print("  bash test/start_all.sh")
        sys.exit(1)

    print("=" * 70)
    print("准备就绪！输入文本开始对话（输入 'quit' 或 'exit' 退出）")
    print("=" * 70)
    print()

    try:
        while True:
            # 获取用户输入
            try:
                user_input = input("你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n再见！")
                break

            # 检查退出命令
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("再见！")
                break

            if not user_input:
                continue

            # 生成回复
            print("AI: ", end="", flush=True)
            try:
                # 生成文本（限制长度，避免太长）
                generated_ids, input_length = generate_text(
                    bottom, top, tokenizer, trunk_client,
                    user_input,
                    max_length=30,  # 限制生成长度
                    temperature=0.8
                )
                
                # 只显示新生成的部分（去掉原始输入）
                response_tokens = generated_ids[0, input_length:]
                response = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
                
                print(response)
                print()
                
            except Exception as e:
                print(f"❌ 生成失败: {e}")
                import traceback
                traceback.print_exc()
                print()

    finally:
        trunk_client.close()
        print("\n连接已关闭")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

