"""
测试 splitlearn-comm UI 集成 - 完整性检查

这个脚本验证所有UI组件都正确集成到 splitlearn-comm 包中
"""
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, 'splitlearn-comm', 'src'))

def test_imports():
    """测试 1: 导入检查"""
    print("=" * 70)
    print("测试 1: 检查导入")
    print("=" * 70)

    try:
        # 测试核心导入
        from splitlearn_comm import GRPCComputeClient, GRPCComputeServer
        print("✓ 核心模块导入成功")

        # 测试 UI 导入（如果安装了 gradio）
        try:
            from splitlearn_comm import ClientUI, ServerMonitoringUI
            print("✓ UI 类导入成功 (ClientUI, ServerMonitoringUI)")
            has_ui = True
        except ImportError as e:
            print(f"⚠️  UI 类未安装: {e}")
            print("   运行以下命令安装:")
            print("   pip install gradio pandas")
            has_ui = False

        # 测试 UI 组件导入
        if has_ui:
            from splitlearn_comm.ui import ClientUI, ServerMonitoringUI
            from splitlearn_comm.ui.components import get_theme, StatsPanel, DEFAULT_CSS
            print("✓ UI 组件导入成功 (themes, StatsPanel)")

        return has_ui

    except Exception as e:
        print(f"✗ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_client_methods():
    """测试 2: 检查客户端方法"""
    print("\n" + "=" * 70)
    print("测试 2: 检查客户端方法")
    print("=" * 70)

    try:
        from splitlearn_comm import GRPCComputeClient

        # 检查 launch_ui 方法是否存在
        if hasattr(GRPCComputeClient, 'launch_ui'):
            print("✓ GRPCComputeClient.launch_ui() 方法存在")

            # 检查方法签名
            import inspect
            sig = inspect.signature(GRPCComputeClient.launch_ui)
            params = list(sig.parameters.keys())
            expected = ['self', 'bottom_model', 'top_model', 'tokenizer', 'theme', 'share', 'server_port', 'kwargs']

            print(f"  参数: {params}")

            if all(p in params for p in ['bottom_model', 'top_model', 'tokenizer']):
                print("✓ 方法签名正确")
            else:
                print("✗ 方法签名不完整")
                return False
        else:
            print("✗ GRPCComputeClient.launch_ui() 方法不存在")
            return False

        return True

    except Exception as e:
        print(f"✗ 客户端方法检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_server_methods():
    """测试 3: 检查服务器方法"""
    print("\n" + "=" * 70)
    print("测试 3: 检查服务器方法")
    print("=" * 70)

    try:
        from splitlearn_comm import GRPCComputeServer

        # 检查 launch_monitoring_ui 方法是否存在
        if hasattr(GRPCComputeServer, 'launch_monitoring_ui'):
            print("✓ GRPCComputeServer.launch_monitoring_ui() 方法存在")

            # 检查方法签名
            import inspect
            sig = inspect.signature(GRPCComputeServer.launch_monitoring_ui)
            params = list(sig.parameters.keys())

            print(f"  参数: {params}")

            if all(p in params for p in ['theme', 'refresh_interval', 'share', 'server_port']):
                print("✓ 方法签名正确")
            else:
                print("✗ 方法签名不完整")
                return False
        else:
            print("✗ GRPCComputeServer.launch_monitoring_ui() 方法不存在")
            return False

        return True

    except Exception as e:
        print(f"✗ 服务器方法检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_servicer_metrics():
    """测试 4: 检查 Servicer 指标收集"""
    print("\n" + "=" * 70)
    print("测试 4: 检查 Servicer 指标收集")
    print("=" * 70)

    try:
        from splitlearn_comm.server import ComputeServicer

        # 检查 get_metrics 方法
        if hasattr(ComputeServicer, 'get_metrics'):
            print("✓ ComputeServicer.get_metrics() 方法存在")
        else:
            print("✗ ComputeServicer.get_metrics() 方法不存在")
            return False

        # 检查初始化参数
        import inspect
        sig = inspect.signature(ComputeServicer.__init__)
        params = list(sig.parameters.keys())

        if 'history_size' in params:
            print("✓ ComputeServicer 支持 history_size 参数")
        else:
            print("✗ ComputeServicer 缺少 history_size 参数")
            return False

        return True

    except Exception as e:
        print(f"✗ Servicer 指标检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ui_components():
    """测试 5: 检查 UI 组件"""
    print("\n" + "=" * 70)
    print("测试 5: 检查 UI 组件")
    print("=" * 70)

    try:
        from splitlearn_comm.ui.components import get_theme, StatsPanel, DEFAULT_CSS

        # 测试 get_theme
        for variant in ["default", "dark", "light"]:
            theme = get_theme(variant)
            print(f"✓ get_theme('{variant}') 工作正常")

        # 测试 StatsPanel
        stats_methods = [
            'format_generation_stats',
            'format_server_stats',
            'format_client_stats',
            'format_connection_status'
        ]

        for method in stats_methods:
            if hasattr(StatsPanel, method):
                print(f"✓ StatsPanel.{method}() 存在")
            else:
                print(f"✗ StatsPanel.{method}() 不存在")
                return False

        # 测试 DEFAULT_CSS
        if isinstance(DEFAULT_CSS, str) and len(DEFAULT_CSS) > 0:
            print("✓ DEFAULT_CSS 已定义")
        else:
            print("✗ DEFAULT_CSS 未正确定义")
            return False

        return True

    except ImportError as e:
        print(f"⚠️  UI 组件未安装: {e}")
        print("   这是预期的，如果没有安装 gradio")
        return True  # 不算失败
    except Exception as e:
        print(f"✗ UI 组件检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dependencies():
    """测试 6: 检查依赖配置"""
    print("\n" + "=" * 70)
    print("测试 6: 检查依赖配置")
    print("=" * 70)

    try:
        import toml
        pyproject_path = os.path.join(project_root, 'splitlearn-comm', 'pyproject.toml')

        if not os.path.exists(pyproject_path):
            print(f"✗ pyproject.toml 不存在: {pyproject_path}")
            return False

        with open(pyproject_path, 'r') as f:
            config = toml.load(f)

        # 检查 UI 依赖
        optional_deps = config.get('project', {}).get('optional-dependencies', {})

        if 'ui' in optional_deps:
            ui_deps = optional_deps['ui']
            print(f"✓ UI 依赖已配置: {ui_deps}")

            # 检查必要的包
            has_gradio = any('gradio' in dep for dep in ui_deps)
            has_pandas = any('pandas' in dep for dep in ui_deps)

            if has_gradio:
                print("  ✓ gradio 依赖已配置")
            else:
                print("  ✗ gradio 依赖缺失")
                return False

            if has_pandas:
                print("  ✓ pandas 依赖已配置")
            else:
                print("  ✗ pandas 依赖缺失")
                return False
        else:
            print("✗ UI 可选依赖未配置")
            return False

        return True

    except ImportError:
        print("⚠️  toml 模块未安装，跳过依赖检查")
        print("   pip install toml")
        return True  # 不算失败
    except Exception as e:
        print(f"✗ 依赖检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("splitlearn-comm UI 集成测试")
    print("=" * 70)
    print()

    results = []

    # 运行所有测试
    results.append(("导入检查", test_imports()))
    results.append(("客户端方法", test_client_methods()))
    results.append(("服务器方法", test_server_methods()))
    results.append(("Servicer 指标", test_servicer_metrics()))
    results.append(("UI 组件", test_ui_components()))
    results.append(("依赖配置", test_dependencies()))

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
        print("\n✅ 所有集成测试通过！")
        print("\n下一步:")
        print("  1. 测试服务器监控 UI:")
        print("     python testcode/test_server_ui_integrated.py")
        print("  2. 测试客户端 UI:")
        print("     python testcode/test_client_ui_integrated.py")
        print("\n📦 安装 UI 依赖 (如果还没有):")
        print("     pip install gradio pandas")
        return 0
    else:
        print(f"\n⚠️ 有 {total - passed} 个测试失败")
        return 1


if __name__ == "__main__":
    exit(main())
