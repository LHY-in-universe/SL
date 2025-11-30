"""
验证 Gemma 模型是否正确注册到 SplitLearn

这个脚本检查 Gemma 模型类是否已正确注册到 ModelRegistry。
"""

from splitlearn.registry import ModelRegistry
from splitlearn import ModelFactory


def main():
    print("=" * 70)
    print("验证 Gemma 模型注册")
    print("=" * 70)
    
    # 1. 检查支持的模型列表
    print("\n1. 检查支持的模型类型:")
    supported_models = ModelRegistry.list_supported_models()
    print(f"   支持的模型: {supported_models}")
    
    if 'gemma' in supported_models:
        print("   ✓ Gemma 已在支持列表中")
    else:
        print("   ✗ Gemma 未在支持列表中")
        return
    
    # 2. 检查 Gemma 是否完整注册
    print("\n2. 检查 Gemma 注册完整性:")
    is_complete = ModelRegistry.is_complete('gemma')
    print(f"   Gemma 完整注册: {is_complete}")
    
    if is_complete:
        print("   ✓ Gemma 的所有三个部分 (bottom/trunk/top) 已注册")
    else:
        print("   ✗ Gemma 注册不完整")
        return
    
    # 3. 检查各个部分的模型类
    print("\n3. 检查各个模型类:")
    
    try:
        BottomCls = ModelRegistry.get_model_class('gemma', 'bottom')
        print(f"   Bottom: {BottomCls.__name__} ✓")
    except KeyError as e:
        print(f"   Bottom: 未找到 ✗ - {e}")
    
    try:
        TrunkCls = ModelRegistry.get_model_class('gemma', 'trunk')
        print(f"   Trunk:  {TrunkCls.__name__} ✓")
    except KeyError as e:
        print(f"   Trunk:  未找到 ✗ - {e}")
    
    try:
        TopCls = ModelRegistry.get_model_class('gemma', 'top')
        print(f"   Top:    {TopCls.__name__} ✓")
    except KeyError as e:
        print(f"   Top:    未找到 ✗ - {e}")
    
    # 4. 显示所有模型的注册信息
    print("\n4. 所有模型的注册状态:")
    model_info = ModelRegistry.get_model_info()
    
    print(f"\n   {'模型类型':<15} {'Bottom':<8} {'Trunk':<8} {'Top':<8} {'完整':<8}")
    print(f"   {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    
    for model_type, info in sorted(model_info.items()):
        bottom_status = '✓' if info['bottom'] else '✗'
        trunk_status = '✓' if info['trunk'] else '✗'
        top_status = '✓' if info['top'] else '✗'
        complete_status = '✓' if info['complete'] else '✗'
        
        print(f"   {model_type:<15} {bottom_status:<8} {trunk_status:<8} "
              f"{top_status:<8} {complete_status:<8}")
    
    # 5. 测试 ModelFactory 列出可用模型
    print("\n5. ModelFactory 可用模型列表:")
    print("   调用 ModelFactory.list_available_models():")
    ModelFactory.list_available_models()
    
    print("\n" + "=" * 70)
    print("验证完成！Gemma 模型已成功注册到 SplitLearn。")
    print("=" * 70)
    print("\n使用方法:")
    print("  from splitlearn import ModelFactory")
    print("  bottom, trunk, top = ModelFactory.create_split_models(")
    print("      model_type='gemma',")
    print("      model_name_or_path='google/gemma-2b',")
    print("      split_point_1=6,")
    print("      split_point_2=12,")
    print("      device='cpu'")
    print("  )")


if __name__ == "__main__":
    main()

