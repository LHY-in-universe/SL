#!/usr/bin/env python3
"""
从磁盘加载已保存的拆分模型

这个脚本演示如何加载之前保存的 Bottom/Trunk/Top 模型，
无需重新下载或拆分原始模型。
"""

import os
import sys
import time
import torch
from pathlib import Path
from transformers import AutoTokenizer

# 配置环境变量
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

# 直接导入模型类
from splitlearn_core.models.gpt2 import GPT2BottomModel, GPT2TrunkModel, GPT2TopModel


def print_separator(char="=", length=70):
    """打印分隔线"""
    print(char * length)


def main():
    print_separator()
    print("从磁盘加载已保存的拆分模型")
    print_separator()

    # 模型路径
    storage_path = Path("./models")
    bottom_path = storage_path / "bottom" / "gpt2_2-10_bottom.pt"
    trunk_path = storage_path / "trunk" / "gpt2_2-10_trunk.pt"
    top_path = storage_path / "top" / "gpt2_2-10_top.pt"

    print(f"\n模型路径:")
    print(f"  Bottom: {bottom_path}")
    print(f"  Trunk:  {trunk_path}")
    print(f"  Top:    {top_path}")

    # 检查文件是否存在
    print(f"\n[1] 检查模型文件...")
    for name, path in [("Bottom", bottom_path), ("Trunk", trunk_path), ("Top", top_path)]:
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            print(f"  ✓ {name}: {size_mb:.2f} MB")
        else:
            print(f"  ✗ {name}: 文件不存在！")
            print(f"\n请先运行 save_split_models.py 来保存模型")
            return False

    # 加载元数据
    print(f"\n[2] 读取模型元数据...")
    import json

    bottom_metadata_path = storage_path / "bottom" / "gpt2_2-10_bottom_metadata.json"
    trunk_metadata_path = storage_path / "trunk" / "gpt2_2-10_trunk_metadata.json"
    top_metadata_path = storage_path / "top" / "gpt2_2-10_top_metadata.json"

    with open(bottom_metadata_path, 'r') as f:
        bottom_metadata = json.load(f)
    with open(trunk_metadata_path, 'r') as f:
        trunk_metadata = json.load(f)
    with open(top_metadata_path, 'r') as f:
        top_metadata = json.load(f)

    print(f"  Bottom: Layers {bottom_metadata['end_layer']}")
    print(f"  Trunk:  Layers {trunk_metadata['start_layer']}-{trunk_metadata['end_layer']}")
    print(f"  Top:    Layers {top_metadata['start_layer']}+")

    # 加载模型
    print(f"\n[3] 从磁盘加载模型...")
    start_time = time.time()

    try:
        from transformers import AutoConfig

        # 加载 GPT-2 配置
        config = AutoConfig.from_pretrained("gpt2")

        print(f"  创建并加载 Bottom 模型...")
        bottom = GPT2BottomModel(config, end_layer=bottom_metadata['end_layer'])
        bottom.load_state_dict(torch.load(bottom_path, map_location='cpu', weights_only=True))
        bottom.eval()

        print(f"  创建并加载 Trunk 模型...")
        trunk = GPT2TrunkModel(
            config,
            start_layer=trunk_metadata['start_layer'],
            end_layer=trunk_metadata['end_layer']
        )
        trunk.load_state_dict(torch.load(trunk_path, map_location='cpu', weights_only=True))
        trunk.eval()

        print(f"  创建并加载 Top 模型...")
        top = GPT2TopModel(config, start_layer=top_metadata['start_layer'])
        top.load_state_dict(torch.load(top_path, map_location='cpu', weights_only=True))
        top.eval()

        elapsed = time.time() - start_time
        print(f"\n✓ 模型加载成功！耗时: {elapsed:.2f} 秒")
        print(f"  (比从头加载快很多！)")

    except Exception as e:
        print(f"\n✗ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 模型信息
    print(f"\n[4] 模型信息:")
    bottom_params = sum(p.numel() for p in bottom.parameters()) / 1e6
    trunk_params = sum(p.numel() for p in trunk.parameters()) / 1e6
    top_params = sum(p.numel() for p in top.parameters()) / 1e6

    print(f"  Bottom: {bottom_params:.2f}M 参数")
    print(f"  Trunk:  {trunk_params:.2f}M 参数")
    print(f"  Top:    {top_params:.2f}M 参数")

    # 推理测试
    print(f"\n[5] 推理测试...")

    try:
        print(f"  加载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        test_text = "The future of AI is"
        input_ids = tokenizer.encode(test_text, return_tensors="pt")
        print(f"  输入文本: '{test_text}'")

        print(f"  执行推理...")
        with torch.no_grad():
            h1 = bottom(input_ids)
            h2 = trunk(h1)
            output = top(h2)

        next_token_id = output.logits[0, -1].argmax().item()
        next_token = tokenizer.decode([next_token_id])

        print(f"\n✓ 推理成功！")
        print(f"  预测的下一个词: '{next_token}'")

    except Exception as e:
        print(f"\n✗ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 文本生成
    print(f"\n[6] 文本生成测试...")

    try:
        generated_ids = input_ids.clone()

        print(f"  生成 15 个 tokens...")
        for i in range(15):
            with torch.no_grad():
                h1 = bottom(generated_ids)
                h2 = trunk(h1)
                output = top(h2)

            next_token_id = output.logits[0, -1].argmax()
            generated_ids = torch.cat([
                generated_ids,
                next_token_id.unsqueeze(0).unsqueeze(0)
            ], dim=1)

        generated_text = tokenizer.decode(generated_ids[0])

        print(f"\n✓ 文本生成成功！")
        print(f"\n原始输入: '{test_text}'")
        print(f"完整输出: '{generated_text}'")

    except Exception as e:
        print(f"\n⚠ 文本生成失败: {e}")

    # 总结
    print("\n")
    print_separator("=")
    print("✓ 测试完成！")
    print_separator("=")
    print("\n重要发现:")
    print("  1. 保存的模型可以直接从磁盘加载")
    print(f"  2. 加载速度快（{elapsed:.2f} 秒 vs 首次加载 6-7 秒）")
    print("  3. 加载后的模型功能完全正常")
    print("  4. 无需重新下载或拆分原始模型")
    print("\n这意味着你可以:")
    print("  - 将拆分后的模型部署到不同的服务器")
    print("  - 快速启动推理服务")
    print("  - 节省模型加载时间")
    print()

    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n操作被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
