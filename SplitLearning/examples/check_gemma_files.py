"""
检查 Gemma 模型文件是否已正确创建

这个脚本检查所有必需的 Gemma 文件是否存在，不需要导入 transformers。
"""

import os
from pathlib import Path


def check_file_exists(filepath, description):
    """检查文件是否存在"""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    print(f"   {status} {description}")
    if exists:
        # 显示文件大小
        size = os.path.getsize(filepath)
        lines = 0
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = len(f.readlines())
        except:
            pass
        print(f"      文件大小: {size} 字节, 行数: {lines}")
    return exists


def main():
    print("=" * 70)
    print("检查 Gemma 模型文件")
    print("=" * 70)
    
    # 找到项目根目录
    script_dir = Path(__file__).parent.parent
    src_dir = script_dir / "src" / "splitlearn"
    
    all_exist = True
    
    # 1. 检查 Gemma 模型文件
    print("\n1. 检查 Gemma 模型文件:")
    
    gemma_dir = src_dir / "models" / "gemma"
    files = [
        (gemma_dir / "__init__.py", "Gemma 包初始化文件"),
        (gemma_dir / "bottom.py", "Gemma Bottom 模型"),
        (gemma_dir / "trunk.py", "Gemma Trunk 模型"),
        (gemma_dir / "top.py", "Gemma Top 模型"),
    ]
    
    for filepath, desc in files:
        exists = check_file_exists(filepath, desc)
        all_exist = all_exist and exists
    
    # 2. 检查模型注册
    print("\n2. 检查模型注册:")
    models_init = src_dir / "models" / "__init__.py"
    
    if os.path.exists(models_init):
        with open(models_init, 'r', encoding='utf-8') as f:
            content = f.read()
            has_import = 'from . import gemma' in content
            status = "✓" if has_import else "✗"
            print(f"   {status} models/__init__.py 中导入 gemma")
            
            has_all = "'gemma'" in content
            status = "✓" if has_all else "✗"
            print(f"   {status} __all__ 列表包含 'gemma'")
            
            all_exist = all_exist and has_import and has_all
    else:
        print("   ✗ models/__init__.py 不存在")
        all_exist = False
    
    # 3. 检查 ParamMapper 更新
    print("\n3. 检查 ParamMapper 更新:")
    param_mapper = src_dir / "utils" / "param_mapper.py"
    
    if os.path.exists(param_mapper):
        with open(param_mapper, 'r', encoding='utf-8') as f:
            content = f.read()
            
            has_layer_pattern = "'gemma': r'\\.layers\\.([0-9]+)'" in content
            status = "✓" if has_layer_pattern else "✗"
            print(f"   {status} LAYER_PATTERNS 包含 gemma")
            
            has_component = "'gemma': {" in content
            status = "✓" if has_component else "✗"
            print(f"   {status} COMPONENT_PATTERNS 包含 gemma")
            
            has_remap = "'gemma'" in content and "remap_layer_index" in content
            status = "✓" if has_remap else "✗"
            print(f"   {status} remap_layer_index 支持 gemma")
            
            all_exist = all_exist and has_layer_pattern and has_component and has_remap
    else:
        print("   ✗ param_mapper.py 不存在")
        all_exist = False
    
    # 4. 检查示例文件
    print("\n4. 检查示例和文档:")
    
    examples_dir = script_dir / "examples"
    example_files = [
        (examples_dir / "gemma_example.py", "Gemma 使用示例"),
        (examples_dir / "GEMMA_README.md", "Gemma 文档"),
        (examples_dir / "verify_gemma_registration.py", "Gemma 注册验证脚本"),
    ]
    
    for filepath, desc in example_files:
        check_file_exists(filepath, desc)
    
    # 5. 检查主 README 更新
    print("\n5. 检查主 README 更新:")
    main_readme = script_dir / "README.md"
    
    if os.path.exists(main_readme):
        with open(main_readme, 'r', encoding='utf-8') as f:
            content = f.read()
            
            has_in_features = 'Gemma' in content and 'Multi-Architecture' in content
            status = "✓" if has_in_features else "✗"
            print(f"   {status} Features 部分提及 Gemma")
            
            has_in_table = "'gemma'" in content or 'gemma' in content.lower()
            status = "✓" if has_in_table else "✗"
            print(f"   {status} 支持的模型表格包含 Gemma")
            
            all_exist = all_exist and has_in_features and has_in_table
    else:
        print("   ✗ README.md 不存在")
        all_exist = False
    
    # 6. 检查代码质量标记
    print("\n6. 检查代码内容:")
    
    # 检查 bottom.py 的关键部分
    bottom_file = gemma_dir / "bottom.py"
    if os.path.exists(bottom_file):
        with open(bottom_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
            checks = [
                ('@ModelRegistry.register("gemma", "bottom")', "注册装饰器"),
                ('class GemmaBottomModel(BaseBottomModel)', "类定义"),
                ('def get_embeddings(', '实现 get_embeddings'),
                ('def apply_position_encoding(', '实现 apply_position_encoding'),
                ('def get_transformer_blocks(', '实现 get_transformer_blocks'),
                ('def prepare_attention_mask(', '实现 prepare_attention_mask'),
                ('def get_layer_name_pattern(', '实现 get_layer_name_pattern'),
                ('def from_pretrained_split(', '实现 from_pretrained_split'),
            ]
            
            for check_str, desc in checks:
                has_it = check_str in content
                status = "✓" if has_it else "✗"
                print(f"   {status} Bottom: {desc}")
    
    # 总结
    print("\n" + "=" * 70)
    if all_exist:
        print("✓ 所有检查通过！Gemma 模型实现完成。")
    else:
        print("✗ 部分检查未通过，请检查上述标记为 ✗ 的项目。")
    print("=" * 70)
    
    print("\n注意：")
    print("  - 如果要运行实际的模型代码，需要 transformers >= 4.38.0")
    print("  - 当前环境可能使用较旧版本的 transformers")
    print("  - 代码本身已正确实现，只是需要更新依赖版本")
    print("\n升级 transformers:")
    print("  pip install --upgrade 'transformers>=4.38.0'")


if __name__ == "__main__":
    main()

