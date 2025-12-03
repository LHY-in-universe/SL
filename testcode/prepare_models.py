"""
预先准备 Split Learning 所需的所有模型文件
运行此脚本将下载 GPT-2 并拆分为三个部分
"""
import sys
import os
import torch

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, 'SplitLearnCore', 'src'))

from splitlearn_core import ModelFactory

def prepare_models():
    print("=" * 70)
    print("准备 Split Learning 模型文件")
    print("=" * 70)
    
    # 1. 下载并拆分 GPT-2
    print("\n[1/3] 正在下载并拆分 GPT-2 模型...")
    print("这可能需要几分钟，请耐心等待...\n")
    
    bottom, trunk, top = ModelFactory.create_split_models(
        model_type='gpt2',
        model_name_or_path='gpt2',
        split_point_1=2,   # Bottom: 层 0-1
        split_point_2=10,  # Trunk: 层 2-9, Top: 层 10-11
        device='cpu'
    )
    
    # 2. 保存服务器端模型 (Trunk)
    print("\n[2/3] 保存服务器端模型...")
    server_model_path = os.path.join(current_dir, "gpt2_trunk_full.pt")
    torch.save(trunk, server_model_path)
    file_size_mb = os.path.getsize(server_model_path) / (1024 * 1024)
    print(f"✓ 服务器模型已保存: {server_model_path}")
    print(f"  大小: {file_size_mb:.1f} MB")
    
    # 3. 保存客户端模型 (Bottom + Top)
    print("\n[3/3] 保存客户端模型...")
    
    bottom_path = os.path.join(current_dir, "gpt2_bottom_cached.pt")
    torch.save(bottom, bottom_path)
    bottom_size_mb = os.path.getsize(bottom_path) / (1024 * 1024)
    print(f"✓ Bottom 模型已保存: {bottom_path}")
    print(f"  大小: {bottom_size_mb:.1f} MB")
    
    top_path = os.path.join(current_dir, "gpt2_top_cached.pt")
    torch.save(top, top_path)
    top_size_mb = os.path.getsize(top_path) / (1024 * 1024)
    print(f"✓ Top 模型已保存: {top_path}")
    print(f"  大小: {top_size_mb:.1f} MB")
    
    # 4. 总结
    print("\n" + "=" * 70)
    print("✅ 所有模型文件准备完成！")
    print("=" * 70)
    print(f"\n总大小: {file_size_mb + bottom_size_mb + top_size_mb:.1f} MB")
    print("\n文件列表:")
    print(f"  1. {server_model_path} (服务器端)")
    print(f"  2. {bottom_path} (客户端)")
    print(f"  3. {top_path} (客户端)")
    print("\n下一步:")
    print("  1. 启动服务器: python testcode/start_server.py")
    print("  2. 启动客户端: python testcode/client_with_gradio.py")
    print("  3. 在 Gradio 界面点击'初始化模型并连接服务器'")
    print("=" * 70)

if __name__ == "__main__":
    prepare_models()
