#!/usr/bin/env python3
"""
检查 LoRA 微调测试的依赖环境

只检查，不安装任何包
"""

import sys

print("=" * 70)
print("LoRA 微调测试 - 依赖环境检查")
print("=" * 70)
print()

all_ok = True

# 检查 PEFT 库（必需）
print("1. 检查 PEFT 库（必需）...")
try:
    import peft
    print(f"   ✅ PEFT 已安装 (版本: {peft.__version__})")
except ImportError:
    print("   ❌ PEFT 未安装")
    print("   📝 安装命令: pip install peft")
    all_ok = False

# 检查 datasets 库（可选）
print("\n2. 检查 datasets 库（可选）...")
try:
    import datasets
    print(f"   ✅ datasets 已安装 (版本: {datasets.__version__})")
    print("   ℹ️  可以使用 HuggingFace datasets")
except ImportError:
    print("   ⚠️  datasets 未安装（可以使用合成数据集）")
    print("   📝 安装命令: pip install datasets")

# 检查 transformers
print("\n3. 检查 transformers 库...")
try:
    import transformers
    print(f"   ✅ transformers 已安装 (版本: {transformers.__version__})")
except ImportError:
    print("   ❌ transformers 未安装")
    all_ok = False

# 检查 torch
print("\n4. 检查 PyTorch...")
try:
    import torch
    print(f"   ✅ PyTorch 已安装 (版本: {torch.__version__})")
except ImportError:
    print("   ❌ PyTorch 未安装")
    all_ok = False

# 检查服务器状态
print("\n5. 检查 Trunk 服务器...")
import os
from pathlib import Path

pid_file = Path(__file__).parent / ".trunk.pid"
if pid_file.exists():
    try:
        pid = int(pid_file.read_text().strip())
        import subprocess
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "pid,command"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"   ✅ Trunk 服务器正在运行 (PID: {pid})")
        else:
            print(f"   ⚠️  服务器 PID 文件存在但进程未运行")
            all_ok = False
    except Exception as e:
        print(f"   ⚠️  无法检查服务器状态: {e}")
else:
    print("   ⚠️  服务器未运行")
    print("   📝 启动命令: bash test/start_all.sh")
    all_ok = False

# 检查模型文件
print("\n6. 检查模型文件...")
project_root = Path(__file__).parent.parent
models_dir = project_root / "models"

required_files = [
    "bottom/gpt2_2-10_bottom.pt",
    "top/gpt2_2-10_top.pt",
    "bottom/gpt2_2-10_bottom_metadata.json",
    "top/gpt2_2-10_top_metadata.json"
]

all_files_exist = True
for file_path in required_files:
    full_path = models_dir / file_path
    if full_path.exists():
        print(f"   ✅ {file_path}")
    else:
        print(f"   ❌ {file_path} 不存在")
        all_files_exist = False

if not all_files_exist:
    all_ok = False

# 总结
print("\n" + "=" * 70)
if all_ok:
    print("✅ 所有依赖已满足，可以运行测试！")
    print("\n运行测试:")
    print("  python test/client/train_lora_simple.py")
    print("  或")
    print("  bash test/run_lora_training.sh")
else:
    print("❌ 部分依赖缺失，请先安装缺失的依赖")
    print("\n必需的依赖:")
    print("  pip install peft")
    print("\n可选的依赖:")
    print("  pip install datasets  # 如果要使用 HuggingFace datasets")
print("=" * 70)

sys.exit(0 if all_ok else 1)
