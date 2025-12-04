#!/usr/bin/env python3
"""
保存拆分后的模型到本地

这个脚本会：
1. 加载并拆分 GPT-2 模型
2. 将 Bottom/Trunk/Top 三个模型保存到 ./models 目录
3. 显示保存的文件和大小
"""

import os
import sys
from pathlib import Path
import torch

# 配置环境变量
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

from splitlearn_core import ModelFactory
from splitlearn_core.utils import StorageManager


def main():
    print("=" * 70)
    print("保存拆分后的 GPT-2 模型")
    print("=" * 70)

    # 配置
    model_type = "gpt2"
    split_point_1 = 2
    split_point_2 = 10
    device = "cpu"
    storage_path = "./models"

    print(f"\n配置:")
    print(f"  模型类型: {model_type}")
    print(f"  拆分点: [{split_point_1}, {split_point_2}]")
    print(f"  保存路径: {storage_path}")

    # 创建存储目录
    print(f"\n[1] 创建存储目录...")
    dirs = StorageManager.create_storage_directories(storage_path)
    print(f"✓ 目录创建完成:")
    for dir_path in dirs.values():
        print(f"  - {dir_path}")

    # 加载和拆分模型
    print(f"\n[2] 加载并拆分模型...")
    bottom, trunk, top = ModelFactory.create_split_models(
        model_type=model_type,
        model_name_or_path=model_type,
        split_point_1=split_point_1,
        split_point_2=split_point_2,
        device=device
    )
    print("✓ 模型拆分完成")

    # 生成保存路径
    split_config = f"{split_point_1}-{split_point_2}"

    bottom_path = StorageManager.get_split_model_path(
        storage_path, model_type, "bottom", split_config
    )
    trunk_path = StorageManager.get_split_model_path(
        storage_path, model_type, "trunk", split_config
    )
    top_path = StorageManager.get_split_model_path(
        storage_path, model_type, "top", split_config
    )

    # 保存模型
    print(f"\n[3] 保存模型...")

    print(f"  保存 Bottom 模型...")
    bottom.save_split_model(bottom_path)
    print(f"    ✓ {bottom_path}")

    print(f"  保存 Trunk 模型...")
    trunk.save_split_model(trunk_path)
    print(f"    ✓ {trunk_path}")

    print(f"  保存 Top 模型...")
    top.save_split_model(top_path)
    print(f"    ✓ {top_path}")

    # 检查保存的文件
    print(f"\n[4] 验证保存的文件...")
    print_separator("-")

    def show_files(directory):
        """显示目录中的文件"""
        dir_path = Path(directory)
        if dir_path.exists():
            files = list(dir_path.glob("*"))
            if files:
                for file in sorted(files):
                    size_mb = file.stat().st_size / 1024 / 1024
                    print(f"  {file.name:50s} {size_mb:8.2f} MB")
            else:
                print(f"  (空)")
        else:
            print(f"  (目录不存在)")

    print(f"\nBottom 模型目录: {dirs['bottom']}")
    show_files(dirs['bottom'])

    print(f"\nTrunk 模型目录: {dirs['trunk']}")
    show_files(dirs['trunk'])

    print(f"\nTop 模型目录: {dirs['top']}")
    show_files(dirs['top'])

    # 计算总大小
    print(f"\n")
    print_separator("-")
    total_size = 0
    for dir_name, dir_path in dirs.items():
        dir_path_obj = Path(dir_path)
        if dir_path_obj.exists():
            dir_size = sum(f.stat().st_size for f in dir_path_obj.glob("*") if f.is_file())
            total_size += dir_size
            print(f"{dir_name.capitalize():10s} 目录: {dir_size/1024/1024:8.2f} MB")

    print(f"{'总计':10s}        {total_size/1024/1024:8.2f} MB")

    print("\n")
    print_separator("=")
    print("✓ 模型保存完成！")
    print_separator("=")
    print(f"\n你现在可以在以下位置找到拆分后的模型:")
    print(f"  {Path(storage_path).absolute()}/")
    print()


def print_separator(char="=", length=70):
    """打印分隔线"""
    print(char * length)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n操作被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
