#!/usr/bin/python3
"""
使用 SplitLearnCore 的 load_full_model 加载 GPT-2 模型的交互式对话脚本
使用 SplitLearnCore 库提供的统一接口，避免直接使用 transformers 可能导致的 Bus error
"""

import sys
import os
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# 添加 SplitLearnCore 到路径
splitcore_path = Path(__file__).parent.parent / "SplitLearnCore" / "src"
if splitcore_path.exists():
    sys.path.insert(0, str(splitcore_path))

# 添加 SplitLearnMonitor 到路径（如果存在）
monitor_path = Path(__file__).parent.parent / "SplitLearnMonitor" / "src"
if monitor_path.exists():
    sys.path.insert(0, str(monitor_path))

from splitlearn_core.quickstart import load_full_model

# 尝试导入 SplitLearnMonitor（直接导入核心功能，避免可视化依赖）
try:
    from splitlearn_monitor.integrations.full_model_monitor import FullModelMonitor
    MONITOR_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    MONITOR_AVAILABLE = False
    FullModelMonitor = None
    # 不打印错误，因为这是可选的

# 导入本地 TokenMonitor（如果没有 SplitLearnMonitor）
from token_monitor import TokenMonitor

# 强制使用 CPU，避免 MPS 相关的 Bus error
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
# 禁用 JIT 编译，可能避免某些 Bus error
os.environ['PYTORCH_JIT'] = '0'
try:
    torch.jit._state.disable()
except:
    pass

device = torch.device('cpu')

# 设置 PyTorch 线程数（必须在导入 torch 后立即设置）
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except:
    pass

def load_model(model_name="gpt2"):
    """使用 SplitLearnCore 的 load_full_model 加载模型和分词器"""
    print(f"正在加载模型（使用 SplitLearnCore load_full_model）...")
    print(f"模型名称: {model_name}")
    start_time = time.time()
    
    try:
        # 使用 SplitLearnCore 的统一接口加载模型
        model, tokenizer = load_full_model(
            model_name_or_path=model_name,
            device="cpu",
            dtype=torch.float32,  # 使用 float32 避免精度问题
            low_cpu_mem_usage=True,  # 低内存模式
        )
        
        load_time = time.time() - start_time
        print(f"模型加载完成，耗时: {load_time:.2f} 秒")
        print(f"使用设备: {device}")
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M\n")
        
        return model, tokenizer
    except Exception as e:
        print(f"模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        raise

def generate_response_manual(model, tokenizer, user_input, max_new_tokens=50, monitor=None):
    """
    手动生成回复，避免使用 model.generate() 可能导致的 Bus error
    
    Args:
        model: 模型
        tokenizer: 分词器
        user_input: 用户输入
        max_new_tokens: 最大生成 token 数
        monitor: TokenMonitor 实例（可选）
    """
    # 编码输入
    encode_start = time.time()
    input_ids = tokenizer.encode(user_input, return_tensors="pt").to(device)
    encode_time = time.time() - encode_start
    encode_time_ms = encode_time * 1000
    
    if monitor:
        monitor.record_encode(encode_time_ms)
    
    # 模型推理（手动循环生成，更安全）
    inference_start = time.time()
    generated_ids = input_ids.clone()
    token_times = []  # 存储每个 token 的生成时间
    
    try:
        # 使用 torch.inference_mode() 而不是 no_grad()，更安全
        with torch.inference_mode():
            for step in range(max_new_tokens):
                # 记录每个 token 生成开始时间
                token_start = time.time()
                
                # 前向传播 - 只使用最后一个位置
                try:
                    outputs = model(generated_ids)
                    logits = outputs.logits
                    
                    # 获取最后一个 token 的 logits
                    next_token_logits = logits[0, -1, :].cpu()  # 移到 CPU 避免 MPS 问题
                    
                    # 应用温度缩放
                    temperature = 0.8
                    scaled_logits = next_token_logits / temperature
                    
                    # 使用 top-k 采样，避免总是选择最高概率的 token（导致重复）
                    # 这样可以有随机性，同时避免 multinomial 可能导致的 Bus error
                    top_k = 50  # 只从前 50 个最可能的 token 中选择
                    if top_k > 0 and top_k < len(scaled_logits):
                        # 获取 top-k
                        top_k_logits, top_k_indices = torch.topk(scaled_logits, top_k)
                        # 应用 softmax 得到概率分布
                        top_k_probs = torch.softmax(top_k_logits, dim=-1)
                        
                        # 使用 numpy 的随机采样（更安全，避免 Bus error）
                        sampled_idx = np.random.choice(len(top_k_probs), p=top_k_probs.numpy())
                        next_token_id = top_k_indices[sampled_idx].item()
                    else:
                        # 回退到 argmax（如果 top_k 无效）
                        next_token_id = scaled_logits.argmax().item()
                    
                    # 检查是否到达结束符
                    if next_token_id == tokenizer.eos_token_id:
                        break
                    
                    # 记录 token 生成时间
                    token_time_ms = (time.time() - token_start) * 1000
                    token_times.append(token_time_ms)
                    
                    # 获取 token 文本（用于监控）
                    token_text = tokenizer.decode([next_token_id], skip_special_tokens=True)
                    
                    # 记录到 monitor
                    if monitor:
                        monitor.record_token(
                            token_id=next_token_id,
                            token_text=token_text,
                            generation_time_ms=token_time_ms,
                            step=step + 1,
                            metadata={'top_k': top_k, 'temperature': temperature}
                        )
                    
                    # 转换为 tensor 并添加到序列（使用更安全的方式）
                    next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
                    # 使用 clone() 避免内存问题
                    generated_ids = torch.cat([generated_ids.clone(), next_token_tensor], dim=1)
                    
                    # 限制最大长度
                    if generated_ids.shape[1] >= input_ids.shape[1] + max_new_tokens:
                        break
                        
                except RuntimeError as e:
                    if "Bus error" in str(e) or "bus error" in str(e).lower():
                        print(f"警告: 在第 {step} 步遇到 Bus error，使用 argmax 继续...")
                        # 回退到简单的 argmax
                        outputs = model(generated_ids)
                        logits = outputs.logits
                        next_token_logits = logits[0, -1, :].cpu()
                        # 使用 top-k 采样
                        top_k = 50
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, min(top_k, len(next_token_logits)))
                        top_k_probs = torch.softmax(top_k_logits, dim=-1)
                        sampled_idx = np.random.choice(len(top_k_probs), p=top_k_probs.numpy())
                        next_token_id = top_k_indices[sampled_idx].item()
                        if next_token_id == tokenizer.eos_token_id:
                            break
                        next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
                        generated_ids = torch.cat([generated_ids.clone(), next_token_tensor], dim=1)
                    else:
                        raise
                    
    except Exception as e:
        print(f"生成错误: {e}")
        import traceback
        traceback.print_exc()
        # 如果出错，至少返回输入
        generated_ids = input_ids
        raise
    
    inference_time = time.time() - inference_start
    
    # 解码输出
    decode_start = time.time()
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    decode_time = time.time() - decode_start
    decode_time_ms = decode_time * 1000
    
    if monitor:
        monitor.record_decode(decode_time_ms)
    
    # 移除原始输入部分，只返回生成的部分
    if response.startswith(user_input):
        response = response[len(user_input):].strip()
    
    # 返回回复和时间统计
    time_stats = {
        'encode_time': encode_time,
        'inference_time': inference_time,  # 这是模型实际运行时间
        'decode_time': decode_time,
        'total_time': encode_time + inference_time + decode_time,
        'token_times': token_times,  # 每个 token 的生成时间
        'num_tokens': len(token_times),
    }
    
    return response, time_stats

def main():
    """主函数"""
    print("=" * 60)
    print("GPT-2 交互式对话 (使用 SplitLearnCore load_full_model)")
    print("=" * 60)
    print("输入 'quit' 或 'exit' 退出程序\n")
    
    # 加载模型（可以修改这里使用不同的模型）
    # 注意: 如果遇到 Bus error，可以尝试使用 tiny-gpt2 进行测试
    model_name = os.getenv("GPT2_MODEL", "sshleifer/tiny-gpt2")  # 默认使用 tiny-gpt2 避免 Bus error
    # model_name = "gpt2"  # 完整 GPT-2，可能在 Python 3.11.0 上出现 Bus error
    model, tokenizer = load_model(model_name)
    
    # 初始化监控器
    session_name = f"gpt2_interactive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    token_monitor = TokenMonitor(session_name=session_name)
    
    # 如果 SplitLearnMonitor 可用，也初始化它
    full_model_monitor = None
    if MONITOR_AVAILABLE:
        try:
            full_model_monitor = FullModelMonitor(
                model_name=model_name,
                sampling_interval=0.1,
                enable_gpu=False,  # 使用 CPU
                auto_start=True
            )
            print(f"[监控] SplitLearnMonitor 已启动\n")
        except Exception as e:
            print(f"[监控] SplitLearnMonitor 启动失败: {e}\n")
            full_model_monitor = None
    
    # 交互循环
    session_start = time.time()
    interaction_count = 0
    total_inference_time = 0.0  # 累计模型推理时间
    
    while True:
        try:
            # 获取用户输入
            user_input = input("你: ").strip()
            
            if not user_input:
                continue
            
            # 检查退出命令
            if user_input.lower() in ['quit', 'exit', '退出']:
                break
            
            # 记录交互开始时间
            interaction_start = time.time()
            interaction_count += 1
            
            # 开始监控这次交互
            token_monitor.start_interaction(user_input, interaction_count)
            
            # 生成回复（使用手动生成方式）
            if full_model_monitor:
                with full_model_monitor.track_inference(metadata={'interaction_id': interaction_count}):
                    response, time_stats = generate_response_manual(model, tokenizer, user_input, monitor=token_monitor)
            else:
                response, time_stats = generate_response_manual(model, tokenizer, user_input, monitor=token_monitor)
            
            # 结束监控这次交互
            token_monitor.end_interaction(response)
            
            # 累计模型推理时间
            total_inference_time += time_stats['inference_time']
            
            # 计算总交互时间（包括输入等待时间）
            interaction_time = time.time() - interaction_start
            
            # 显示回复和时间
            print(f"GPT-2: {response}")
            print(f"[交互 #{interaction_count} 时间统计:]")
            print(f"  编码时间: {time_stats['encode_time']*1000:.2f} ms")
            print(f"  模型推理时间: {time_stats['inference_time']*1000:.2f} ms ⭐")
            print(f"  解码时间: {time_stats['decode_time']*1000:.2f} ms")
            print(f"  总处理时间: {time_stats['total_time']*1000:.2f} ms")
            print(f"  生成 Token 数: {time_stats.get('num_tokens', 0)}")
            
            # 显示每个 token 的时间（如果有）
            if time_stats.get('token_times'):
                token_times = time_stats['token_times']
                avg_token_time = sum(token_times) / len(token_times)
                print(f"  平均每个 Token 时间: {avg_token_time:.2f} ms")
                if len(token_times) <= 10:  # 如果 token 数不多，显示每个
                    print(f"  每个 Token 时间: {[f'{t:.1f}' for t in token_times]} ms")
            print()
            
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            break
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 停止监控
    if full_model_monitor:
        full_model_monitor.stop()
    
    # 显示会话统计
    session_time = time.time() - session_start
    print("\n" + "=" * 60)
    print("会话统计:")
    print(f"  总交互次数: {interaction_count}")
    print(f"  总会话时间: {session_time:.2f} 秒")
    if interaction_count > 0:
        print(f"  平均交互时间: {session_time / interaction_count:.3f} 秒")
        print(f"  累计模型推理时间: {total_inference_time:.3f} 秒")
        print(f"  平均模型推理时间: {total_inference_time / interaction_count:.3f} 秒")
        print(f"  模型推理时间占比: {total_inference_time / session_time * 100:.1f}%")
    print("=" * 60)
    
    # 显示 Token 监控统计
    if interaction_count > 0:
        print("\n")
        token_monitor.print_summary()
        
        # 保存监控报告
        try:
            report_path = token_monitor.save_report(format="json")
            print(f"\n[监控] Token 监控报告已保存: {report_path}")
        except Exception as e:
            print(f"\n[监控] 保存报告失败: {e}")
    
    # 如果使用了 FullModelMonitor，也保存它的报告
    if full_model_monitor:
        try:
            # 优先保存 JSON 报告（更可靠）
            json_report = full_model_monitor.save_report(format='json')
            print(f"[监控] SplitLearnMonitor JSON 报告已保存: {json_report}")
            
            # 尝试保存 HTML 报告（可能需要 matplotlib）
            try:
                html_report = full_model_monitor.save_report(format='html')
                print(f"[监控] SplitLearnMonitor HTML 报告已保存: {html_report}")
            except Exception as e:
                print(f"[监控] HTML 报告生成失败（使用 JSON 报告）: {e}")
        except Exception as e:
            print(f"[监控] SplitLearnMonitor 报告保存失败: {e}")

if __name__ == "__main__":
    main()
