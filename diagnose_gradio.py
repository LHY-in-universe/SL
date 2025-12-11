#!/usr/bin/env python3
"""
Gradio 按钮问题诊断脚本
用于检查按钮绑定和函数调用是否正常工作
"""

import sys
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/diagnose.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_code_structure():
    """检查代码结构"""
    logger.info("=" * 70)
    logger.info("步骤 1: 检查代码结构")
    logger.info("=" * 70)
    
    with open("gpt2_full_model_gradio.py", "r") as f:
        content = f.read()
    
    checks = {
        "with gr.Blocks": "with gr.Blocks" in content,
        "generate_btn.click": "generate_btn.click" in content,
        "[UI] 正在绑定按钮事件": "[UI] 正在绑定按钮事件" in content,
        "[GENERATE]": "[GENERATE]" in content,
        "def generate_with_kv_cache": "def generate_with_kv_cache" in content,
        "demo.queue()": "demo.queue()" in content,
        "demo.launch()": "demo.launch()" in content,
    }
    
    for check, result in checks.items():
        status = "✓" if result else "✗"
        logger.info(f"{status} {check}: {result}")
    
    return all(checks.values())

def check_running_process():
    """检查运行中的进程"""
    logger.info("=" * 70)
    logger.info("步骤 2: 检查运行中的进程")
    logger.info("=" * 70)
    
    import subprocess
    result = subprocess.run(
        ["ps", "aux"],
        capture_output=True,
        text=True
    )
    
    gpt2_processes = [line for line in result.stdout.split("\n") if "gpt2_full" in line]
    
    if gpt2_processes:
        logger.info(f"✓ 找到 {len(gpt2_processes)} 个相关进程:")
        for proc in gpt2_processes:
            logger.info(f"  {proc}")
    else:
        logger.warning("✗ 未找到运行中的 gpt2_full 进程")
    
    return len(gpt2_processes) > 0

def check_port():
    """检查端口占用"""
    logger.info("=" * 70)
    logger.info("步骤 3: 检查端口占用")
    logger.info("=" * 70)
    
    import subprocess
    result = subprocess.run(
        ["lsof", "-i", ":7891"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0 and result.stdout:
        logger.info("✓ 端口 7891 被占用:")
        logger.info(result.stdout)
        return True
    else:
        logger.warning("✗ 端口 7891 未被占用")
        return False

def check_server_response():
    """检查服务器响应"""
    logger.info("=" * 70)
    logger.info("步骤 4: 检查服务器响应")
    logger.info("=" * 70)
    
    import urllib.request
    import urllib.error
    
    try:
        response = urllib.request.urlopen("http://127.0.0.1:7891/", timeout=5)
        logger.info(f"✓ 服务器响应正常: {response.status}")
        return True
    except urllib.error.URLError as e:
        logger.error(f"✗ 服务器响应失败: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ 检查失败: {e}")
        return False

def check_logs():
    """检查日志文件"""
    logger.info("=" * 70)
    logger.info("步骤 5: 检查日志文件")
    logger.info("=" * 70)
    
    log_file = Path("logs/gpt2_full.log")
    if not log_file.exists():
        logger.warning("✗ 日志文件不存在")
        return False
    
    with open(log_file, "r") as f:
        lines = f.readlines()
    
    logger.info(f"✓ 日志文件存在，共 {len(lines)} 行")
    
    # 检查关键日志
    key_logs = {
        "[UI] 正在绑定按钮事件": False,
        "[UI] ✓ 按钮事件绑定完成": False,
        "[GENERATE]": False,
        "[TEST]": False,
        "Gradio 本地地址": False,
    }
    
    for line in lines[-100:]:  # 检查最后 100 行
        for key in key_logs:
            if key in line:
                key_logs[key] = True
    
    for key, found in key_logs.items():
        status = "✓" if found else "✗"
        logger.info(f"{status} {key}: {'找到' if found else '未找到'}")
    
    return any(key_logs.values())

def main():
    logger.info("开始诊断 Gradio 按钮问题...")
    logger.info("")
    
    results = {
        "代码结构": check_code_structure(),
        "运行进程": check_running_process(),
        "端口占用": check_port(),
        "服务器响应": check_server_response(),
        "日志检查": check_logs(),
    }
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("诊断结果汇总")
    logger.info("=" * 70)
    
    for check, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        logger.info(f"{check}: {status}")
    
    all_passed = all(results.values())
    
    logger.info("")
    if all_passed:
        logger.info("✓ 所有检查通过，但按钮仍然不工作，可能是前端问题")
        logger.info("建议:")
        logger.info("  1. 检查浏览器控制台是否有 JavaScript 错误")
        logger.info("  2. 确认访问的 URL 是日志中显示的地址")
        logger.info("  3. 清除浏览器缓存并强制刷新 (Cmd+Shift+R)")
        logger.info("  4. 尝试使用不同的浏览器")
    else:
        logger.warning("✗ 部分检查失败，请根据上述结果修复问题")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
