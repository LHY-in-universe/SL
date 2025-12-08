#!/usr/bin/env python3
"""
测试监控功能是否正常工作
"""

import sys
from pathlib import Path

# 添加路径
splitcore_path = Path(__file__).parent.parent / "SplitLearnCore" / "src"
if splitcore_path.exists():
    sys.path.insert(0, str(splitcore_path))

monitor_path = Path(__file__).parent.parent / "SplitLearnMonitor" / "src"
if monitor_path.exists():
    sys.path.insert(0, str(monitor_path))

from splitlearn_core.quickstart import load_full_model
from token_monitor import TokenMonitor

# 测试 SplitLearnMonitor
try:
    from splitlearn_monitor.integrations.full_model_monitor import FullModelMonitor
    MONITOR_AVAILABLE = True
    print("✓ SplitLearnMonitor 可用")
except (ImportError, ModuleNotFoundError):
    MONITOR_AVAILABLE = False
    FullModelMonitor = None
    print("✗ SplitLearnMonitor 不可用（将只使用 TokenMonitor）")

print("\n" + "=" * 60)
print("监控功能测试")
print("=" * 60 + "\n")

# 加载模型
print("1. 加载模型...")
model, tokenizer = load_full_model("sshleifer/tiny-gpt2", device="cpu")
print("   ✓ 模型加载成功\n")

# 初始化监控器
print("2. 初始化监控器...")
token_monitor = TokenMonitor(session_name="test_session")
print("   ✓ TokenMonitor 初始化成功")

if MONITOR_AVAILABLE:
    try:
        full_model_monitor = FullModelMonitor(
            model_name="tiny-gpt2",
            sampling_interval=0.1,
            enable_gpu=False,
            auto_start=True
        )
        print("   ✓ FullModelMonitor 初始化成功")
    except Exception as e:
        print(f"   ✗ FullModelMonitor 初始化失败: {e}")
        full_model_monitor = None
else:
    full_model_monitor = None

print()

# 测试生成和监控
print("3. 测试生成和监控...")
test_input = "Hello"
token_monitor.start_interaction(test_input, 1)

import torch
input_ids = tokenizer.encode(test_input, return_tensors="pt")

import time

if full_model_monitor:
    with full_model_monitor.track_inference():
        with torch.inference_mode():
            for step in range(3):
                token_start = time.time()
                outputs = model(input_ids)
                logits = outputs.logits
                next_token_id = logits[0, -1, :].argmax().item()
                token_time_ms = (time.time() - token_start) * 1000
                
                token_text = tokenizer.decode([next_token_id], skip_special_tokens=True)
                token_monitor.record_token(
                    token_id=next_token_id,
                    token_text=token_text,
                    generation_time_ms=token_time_ms,
                    step=step + 1
                )
                
                next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long)
                input_ids = torch.cat([input_ids, next_token_tensor], dim=1)
else:
    with torch.inference_mode():
        for step in range(3):
            token_start = time.time()
            outputs = model(input_ids)
            logits = outputs.logits
            next_token_id = logits[0, -1, :].argmax().item()
            token_time_ms = (time.time() - token_start) * 1000
            
            token_text = tokenizer.decode([next_token_id], skip_special_tokens=True)
            token_monitor.record_token(
                token_id=next_token_id,
                token_text=token_text,
                generation_time_ms=token_time_ms,
                step=step + 1
            )
            
            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long)
            input_ids = torch.cat([input_ids, next_token_tensor], dim=1)

response = tokenizer.decode(input_ids[0], skip_special_tokens=True)
token_monitor.end_interaction(response)
print("   ✓ 生成和监控测试成功\n")

# 显示统计
print("4. 监控统计:")
token_monitor.print_summary()

# 保存报告
print("\n5. 保存报告...")
try:
    report_path = token_monitor.save_report(format="json")
    print(f"   ✓ TokenMonitor 报告已保存: {report_path}")
except Exception as e:
    print(f"   ✗ TokenMonitor 报告保存失败: {e}")

if full_model_monitor:
    full_model_monitor.stop()
    try:
        json_report = full_model_monitor.save_report(format='json')
        print(f"   ✓ FullModelMonitor JSON 报告已保存: {json_report}")
    except Exception as e:
        print(f"   ✗ FullModelMonitor 报告保存失败: {e}")

print("\n" + "=" * 60)
print("✓ 所有测试完成！")
print("=" * 60)

