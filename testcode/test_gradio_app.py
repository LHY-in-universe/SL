"""
测试改进后的 Gradio 应用
"""
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, 'SplitLearning', 'src'))
sys.path.insert(0, os.path.join(project_root, 'splitlearn-comm', 'src'))

def test_imports():
    """测试所有导入"""
    print("=" * 70)
    print("测试 1: 导入模块")
    print("=" * 70)
    try:
        import torch
        import gradio as gr
        from transformers import AutoTokenizer
        print(f"✓ torch: {torch.__version__}")
        print(f"✓ gradio: {gr.__version__}")
        print(f"✓ transformers: 已安装")
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_gradio_interface():
    """测试 Gradio 界面构建"""
    print("\n" + "=" * 70)
    print("测试 2: Gradio 界面构建")
    print("=" * 70)

    try:
        # 动态导入以避免全局执行
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "client_app",
            os.path.join(current_dir, "client_with_gradio.py")
        )
        module = importlib.util.module_from_spec(spec)

        # 不执行 if __name__ == "__main__" 部分
        # 只检查能否导入
        print("✓ 模块可以被导入")

        # 检查关键函数是否存在
        expected_functions = [
            'load_models',
            'check_models_loaded',
            'stop_generation_fn',
            'generate_text'
        ]

        spec.loader.exec_module(module)

        for func_name in expected_functions:
            if hasattr(module, func_name):
                print(f"✓ 函数 {func_name} 存在")
            else:
                print(f"✗ 函数 {func_name} 不存在")
                return False

        # 检查 demo 对象
        if hasattr(module, 'demo'):
            print(f"✓ Gradio demo 对象已创建")
            print(f"  类型: {type(module.demo)}")
        else:
            print("✗ demo 对象不存在")
            return False

        return True

    except Exception as e:
        print(f"✗ 界面构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_helper_functions():
    """测试辅助函数"""
    print("\n" + "=" * 70)
    print("测试 3: 辅助函数")
    print("=" * 70)

    try:
        # 导入模块
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "client_app",
            os.path.join(current_dir, "client_with_gradio.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # 测试 check_models_loaded (应该返回 False，因为模型未加载)
        is_ready, error_msg = module.check_models_loaded()
        if not is_ready and "请先点击" in error_msg:
            print("✓ check_models_loaded() 正常工作")
        else:
            print("✗ check_models_loaded() 行为异常")
            return False

        # 测试 stop_generation_fn
        result = module.stop_generation_fn()
        if "停止" in result:
            print("✓ stop_generation_fn() 正常工作")
        else:
            print("✗ stop_generation_fn() 返回异常")
            return False

        return True

    except Exception as e:
        print(f"✗ 辅助函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_files_exist():
    """检查模型文件是否存在"""
    print("\n" + "=" * 70)
    print("测试 4: 检查模型文件")
    print("=" * 70)

    bottom_path = os.path.join(current_dir, "gpt2_bottom_cached.pt")
    top_path = os.path.join(current_dir, "gpt2_top_cached.pt")
    trunk_path = os.path.join(current_dir, "gpt2_trunk_full.pt")

    files_status = []

    if os.path.exists(bottom_path):
        size_mb = os.path.getsize(bottom_path) / (1024 * 1024)
        print(f"✓ Bottom 模型存在: {size_mb:.1f} MB")
        files_status.append(True)
    else:
        print(f"⚠ Bottom 模型不存在: {bottom_path}")
        files_status.append(False)

    if os.path.exists(top_path):
        size_mb = os.path.getsize(top_path) / (1024 * 1024)
        print(f"✓ Top 模型存在: {size_mb:.1f} MB")
        files_status.append(True)
    else:
        print(f"⚠ Top 模型不存在: {top_path}")
        files_status.append(False)

    if os.path.exists(trunk_path):
        size_mb = os.path.getsize(trunk_path) / (1024 * 1024)
        print(f"✓ Trunk 模型存在 (服务器用): {size_mb:.1f} MB")
        files_status.append(True)
    else:
        print(f"⚠ Trunk 模型不存在: {trunk_path}")
        files_status.append(False)

    if not any(files_status):
        print("\n💡 提示: 首次运行需要下载模型")
        print("   运行: python testcode/prepare_models.py")

    return True  # 这不是错误，只是提示

def test_environment_variables():
    """测试环境变量配置"""
    print("\n" + "=" * 70)
    print("测试 5: 环境变量配置")
    print("=" * 70)

    # 测试 GRADIO_SHARE
    share = os.environ.get("GRADIO_SHARE", "false").lower() == "true"
    print(f"✓ GRADIO_SHARE: {share} (默认: False)")

    # 测试 GRADIO_PORT
    port = int(os.environ.get("GRADIO_PORT", "7861"))
    print(f"✓ GRADIO_PORT: {port} (默认: 7861)")

    print("\n使用方法:")
    print("  GRADIO_SHARE=true python testcode/client_with_gradio.py")
    print("  GRADIO_PORT=8080 python testcode/client_with_gradio.py")

    return True

def main():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("Gradio 应用测试套件")
    print("=" * 70)

    results = []

    results.append(("导入模块", test_imports()))
    results.append(("Gradio 界面构建", test_gradio_interface()))
    results.append(("辅助函数", test_helper_functions()))
    results.append(("模型文件检查", test_model_files_exist()))
    results.append(("环境变量配置", test_environment_variables()))

    # 总结
    print("\n" + "=" * 70)
    print("测试结果摘要")
    print("=" * 70)

    for test_name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name:20s}: {status}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    print("\n" + "=" * 70)
    print(f"总计: {passed}/{total} 测试通过")
    print("=" * 70)

    if passed == total:
        print("\n✅ 所有测试通过！")
        print("\n下一步:")
        print("  1. 启动服务器: python testcode/start_server.py")
        print("  2. 启动客户端: python testcode/client_with_gradio.py")
        print("  3. 访问: http://127.0.0.1:7861")
        return 0
    else:
        print(f"\n⚠️ 有 {total - passed} 个测试失败")
        return 1

if __name__ == "__main__":
    exit(main())
