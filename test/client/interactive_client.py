#!/usr/bin/env python3
"""
Split Learning 交互式客户端
在终端输入文本，获得 AI 回复
"""

import os
import sys
import json
import time
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoConfig

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# 添加 SplitLearnMonitor 路径（监控模块在本地）
monitor_src_path = os.path.join(project_root, 'SplitLearnMonitor', 'src')
if os.path.exists(monitor_src_path):
    sys.path.insert(0, monitor_src_path)

# 尝试导入 psutil（可选）
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# 尝试导入 SplitLearnMonitor（可选）
try:
    from splitlearn_monitor import ClientMonitor
    MONITOR_AVAILABLE = True
except ImportError as e:
    MONITOR_AVAILABLE = False
    ClientMonitor = None
    # 不在控制台输出错误，因为这是可选的模块

from splitlearn_comm.quickstart import Client
from splitlearn_core.models.gpt2 import GPT2BottomModel, GPT2TopModel


def load_models():
    """加载本地模型"""
    print("正在加载模型...")
    
    # 模型路径
    models_dir = Path(project_root) / "models"
    bottom_path = models_dir / "bottom" / "gpt2_2-10_bottom.pt"
    top_path = models_dir / "top" / "gpt2_2-10_top.pt"
    bottom_metadata_path = models_dir / "bottom" / "gpt2_2-10_bottom_metadata.json"
    top_metadata_path = models_dir / "top" / "gpt2_2-10_top_metadata.json"

    # 检查模型文件
    if not bottom_path.exists() or not top_path.exists():
        print("❌ 模型文件不存在！")
        print(f"请确保以下文件存在：")
        print(f"  - {bottom_path}")
        print(f"  - {top_path}")
        sys.exit(1)

    # 加载元数据
    with open(bottom_metadata_path, 'r') as f:
        bottom_metadata = json.load(f)
    with open(top_metadata_path, 'r') as f:
        top_metadata = json.load(f)

    # 加载 GPT-2 配置
    config = AutoConfig.from_pretrained("gpt2")

    # 加载 Bottom 模型
    print("  加载 Bottom 模型...", end=" ", flush=True)
    bottom = GPT2BottomModel(config, end_layer=bottom_metadata['end_layer'])
    bottom.load_state_dict(torch.load(bottom_path, map_location='cpu', weights_only=True))
    bottom.eval()
    print("✓")

    # 加载 Top 模型
    print("  加载 Top 模型...", end=" ", flush=True)
    top = GPT2TopModel(config, start_layer=top_metadata['start_layer'])
    top.load_state_dict(torch.load(top_path, map_location='cpu', weights_only=True))
    top.eval()
    print("✓")

    # 加载 tokenizer
    print("  加载 Tokenizer...", end=" ", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓")

    print("✓ 所有模型加载完成！\n")
    return bottom, top, tokenizer


def generate_text(bottom, top, tokenizer, trunk_client, prompt, max_length=50, temperature=0.7, show_timing=False, monitor=None):
    """
    生成文本
    
    Args:
        bottom: Bottom 模型
        top: Top 模型
        tokenizer: Tokenizer
        trunk_client: Trunk 服务器客户端
        prompt: 输入文本
        max_length: 最大生成长度
        temperature: 温度参数（控制随机性）
        show_timing: 是否显示时间统计
    
    Returns:
        (generated_ids, input_length, timing_info): 生成的 token IDs、输入长度和时间统计
    """
    import time
    
    # 编码输入
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    generated_ids = input_ids.clone()
    input_length = input_ids.shape[1]
    
    # 时间统计
    timing_info = {
        'total_time': 0,
        'bottom_times': [],
        'trunk_times': [],
        'top_times': [],
        'trunk_round_trips': [],  # 每次远程调用的往返时间
        'encode_times': [],  # 编码时间
        'decode_times': [],  # 解码时间
        'network_times': [],  # 网络传输时间（从客户端获取）
        'server_compute_times': [],  # 服务器计算时间
        'step_details': []  # 每步的详细信息
    }
    
    total_start = time.time()

    with torch.no_grad():
        for step in range(max_length):
            step_start = time.time()
            step_detail = {'step': step + 1}
            
            # Bottom 模型处理（使用 monitor 跟踪）
            bottom_start = time.time()
            if monitor:
                with monitor.track_phase("bottom_model", metadata={"step": step + 1}):
                    hidden_1 = bottom(generated_ids)
            else:
                hidden_1 = bottom(generated_ids)
            bottom_time = time.time() - bottom_start
            timing_info['bottom_times'].append(bottom_time)
            step_detail['bottom_time'] = bottom_time

            # Trunk 模型处理（远程）- 使用 monitor 跟踪
            trunk_start = time.time()
            if monitor:
                with monitor.track_phase("trunk_remote", metadata={"step": step + 1}):
                    hidden_2 = trunk_client.compute(hidden_1)
            else:
                hidden_2 = trunk_client.compute(hidden_1)
            trunk_time = time.time() - trunk_start
            timing_info['trunk_times'].append(trunk_time)
            
            # 获取客户端统计信息（如果可用）
            try:
                stats = trunk_client._client.get_statistics()
                network_time = trunk_time  # 总时间
            except:
                network_time = trunk_time  # 总时间包括编码+网络+解码+服务器计算
            
            timing_info['network_times'].append(network_time)
            
            timing_info['trunk_round_trips'].append({
                'step': step + 1,
                'time': trunk_time,
                'timestamp': time.time()
            })
            step_detail['trunk_time'] = trunk_time
            step_detail['network_time'] = network_time

            # Top 模型处理（使用 monitor 跟踪）
            top_start = time.time()
            if monitor:
                with monitor.track_phase("top_model", metadata={"step": step + 1}):
                    output = top(hidden_2)
                    logits = output.logits
            else:
                output = top(hidden_2)
                logits = output.logits
            top_time = time.time() - top_start
            timing_info['top_times'].append(top_time)
            step_detail['top_time'] = top_time

            # 获取最后一个 token 的 logits
            next_token_logits = logits[0, -1, :] / temperature

            # 应用 softmax 并采样
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

            # 添加到生成的序列 (generated_ids 是 [1, seq_len], next_token_id 是 [1])
            # 需要 reshape 为 [1, 1] 才能正确 concat
            generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=1)

            # 如果生成了结束符，停止
            if next_token_id.item() == tokenizer.eos_token_id:
                break
            
            step_time = time.time() - step_start
            step_detail['step_time'] = step_time
            step_detail['total_time'] = step_time
            timing_info['step_details'].append(step_detail)
            
            if show_timing:
                print(f"  [步骤 {step+1}] "
                      f"Bottom: {bottom_time*1000:.1f}ms | "
                      f"网络传输: {network_time*1000:.1f}ms | "
                      f"Top: {top_time*1000:.1f}ms | "
                      f"总计: {step_time*1000:.1f}ms", flush=True)
        
        timing_info['total_time'] = time.time() - total_start

    return generated_ids, input_length, timing_info


def main():
    print("=" * 70)
    print("Split Learning 交互式客户端")
    print("=" * 70)
    print()

    # 配置 - 支持命令行参数或环境变量指定服务器地址
    import sys
    if len(sys.argv) > 1:
        TRUNK_SERVER = sys.argv[1]
    else:
        TRUNK_SERVER = os.getenv("TRUNK_SERVER", "localhost:50052")

    # 加载模型
    try:
        bottom, top, tokenizer = load_models()
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 连接到服务器
    print(f"连接到 Trunk 服务器 ({TRUNK_SERVER})...", end=" ", flush=True)
    try:
        trunk_client = Client(TRUNK_SERVER)
        print("✓\n")
    except Exception as e:
        print(f"❌\n连接失败: {e}")
        print("\n请确保 Trunk 服务器正在运行:")
        print("  bash test/start_all.sh")
        sys.exit(1)

    print("=" * 70)
    print("准备就绪！输入文本开始对话（输入 'quit' 或 'exit' 退出）")
    print("=" * 70)
    print()

    # 初始化 SplitLearnMonitor（如果可用）
    monitor = None
    if MONITOR_AVAILABLE:
        try:
            session_name = f"interactive_client_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            monitor = ClientMonitor(
                session_name=session_name,
                sampling_interval=0.1,  # 每0.1秒采样一次
                enable_gpu=False,  # 当前使用CPU
                auto_start=True
            )
            print(f"[监控] SplitLearnMonitor 已启动 (会话: {session_name})\n")
        except Exception as e:
            print(f"[监控] SplitLearnMonitor 启动失败: {e}")
            print(f"[监控] 将使用基础监控（不会生成 HTML 报告）\n")
            import traceback
            traceback.print_exc()
            monitor = None

    # 系统资源监控（在模型加载之前测量初始内存）
    if PSUTIL_AVAILABLE:
        process = psutil.Process(os.getpid())
        initial_cpu_percent = process.cpu_percent(interval=0.1)
        initial_memory = process.memory_info()
        print(f"[初始状态] 内存: {initial_memory.rss / 1024 / 1024:.1f} MB")
    else:
        process = None
        initial_memory = None
    
    # 模型加载后，显示加载后的内存
    if PSUTIL_AVAILABLE and process:
        # 等待一下，让内存稳定
        time.sleep(0.2)
        after_load_memory = process.memory_info()
        load_memory_growth = (after_load_memory.rss - initial_memory.rss) / 1024 / 1024
        print(f"[模型加载后] 内存: {after_load_memory.rss / 1024 / 1024:.1f} MB "
              f"(增长: {load_memory_growth:+.1f} MB)\n")
        # 使用加载后的内存作为基准（这样退出时显示的是运行过程中的增长）
        baseline_memory = after_load_memory
    else:
        baseline_memory = initial_memory

    # 全局统计信息
    session_stats = {
        'start_time': time.time(),
        'total_requests': 0,
        'total_tokens_generated': 0,
        'all_trunk_times': [],  # 所有远程调用时间
        'all_bottom_times': [],
        'all_top_times': [],
        'total_generation_time': 0,
        'requests': [],  # 每次请求的详细信息
        'memory_samples': []  # 每次请求时的内存快照
    }

    try:
        while True:
            # 获取用户输入
            try:
                user_input = input("你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n再见！")
                break

            # 检查退出命令
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("再见！")
                break

            if not user_input:
                continue

            # 生成回复
            print("AI: ", end="", flush=True)
            try:
                generation_start = time.time()
                
                # 生成文本（传递 monitor 用于内部阶段跟踪）
                generated_ids, input_length, timing_info = generate_text(
                    bottom, top, tokenizer, trunk_client,
                    user_input,
                    max_length=200,  # 限制生成长度
                    temperature=0.8,
                    show_timing=False,  # 不显示每步详情，只显示汇总
                    monitor=monitor  # 传递 monitor 用于内部阶段跟踪
                )
                
                generation_time = time.time() - generation_start
                
                # 记录当前内存（如果可用）
                if PSUTIL_AVAILABLE and process:
                    current_memory = process.memory_info()
                    session_stats['memory_samples'].append({
                        'request_num': session_stats['total_requests'] + 1,
                        'memory_mb': current_memory.rss / 1024 / 1024,
                        'timestamp': time.time()
                    })
                
                # 只显示新生成的部分（去掉原始输入）
                response_tokens = generated_ids[0, input_length:]
                response = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
                tokens_generated = len(response_tokens)
                
                print(response)
                
                # 更新会话统计
                session_stats['total_requests'] += 1
                session_stats['total_tokens_generated'] += tokens_generated
                session_stats['total_generation_time'] += generation_time
                session_stats['all_trunk_times'].extend(timing_info['trunk_times'])
                session_stats['all_bottom_times'].extend(timing_info['bottom_times'])
                session_stats['all_top_times'].extend(timing_info['top_times'])
                session_stats['all_network_times'] = session_stats.get('all_network_times', [])
                session_stats['all_network_times'].extend(timing_info['network_times'])
                
                # 记录本次请求详情
                request_info = {
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'input': user_input[:50],  # 只记录前50个字符
                    'tokens_generated': tokens_generated,
                    'generation_time': generation_time,
                    'trunk_calls': len(timing_info['trunk_round_trips']),
                    'trunk_times': timing_info['trunk_times'],
                    'bottom_times': timing_info['bottom_times'],
                    'top_times': timing_info['top_times'],
                    'network_times': timing_info['network_times'],
                    'step_details': timing_info['step_details']
                }
                session_stats['requests'].append(request_info)
                
                # 显示本次详细时间统计
                if timing_info['trunk_round_trips']:
                    trunk_times = timing_info['trunk_times']
                    bottom_times = timing_info['bottom_times']
                    top_times = timing_info['top_times']
                    network_times = timing_info['network_times']
                    
                    avg_bottom = sum(bottom_times) / len(bottom_times) if bottom_times else 0
                    avg_trunk = sum(trunk_times) / len(trunk_times) if trunk_times else 0
                    avg_network = sum(network_times) / len(network_times) if network_times else 0
                    avg_top = sum(top_times) / len(top_times) if top_times else 0
                    
                    total_bottom = sum(bottom_times) * 1000
                    total_network = sum(network_times) * 1000
                    total_top = sum(top_times) * 1000
                    
                    print(f"\n[本次详细统计]")
                    print(f"  总生成时间: {generation_time*1000:.1f}ms")
                    print(f"  生成步骤数: {len(trunk_times)}")
                    print(f"  ┌─ 本地处理:")
                    print(f"  │  Bottom 模型: {len(bottom_times)}次, "
                          f"总计 {total_bottom:.1f}ms, 平均 {avg_bottom*1000:.1f}ms/次")
                    print(f"  │  Top 模型:    {len(top_times)}次, "
                          f"总计 {total_top:.1f}ms, 平均 {avg_top*1000:.1f}ms/次")
                    print(f"  └─ 远程调用 (每次包含编码+网络+服务器计算+解码):")
                    print(f"    总往返时间:  {sum(trunk_times)*1000:.1f}ms, "
                          f"平均 {avg_trunk*1000:.1f}ms/次")
                    print(f"    (注: 包含编码/解码时间, 网络传输时间, 服务器计算时间)")
                    print(f"  生成 tokens: {tokens_generated}")
                print()
                
            except Exception as e:
                print(f"❌ 生成失败: {e}")
                import traceback
                traceback.print_exc()
                print()

    finally:
        # 在关闭连接前，先尝试获取服务器监控数据（如果可用）
        server_monitoring_data = None
        try:
            server_monitoring_data = trunk_client._client.get_server_monitoring_data()
            if server_monitoring_data:
                print(f"\n[监控] 获取到 {len(server_monitoring_data)} 个服务器监控快照")
        except Exception as e:
            # 服务器监控数据不可用是正常的
            pass
        
        trunk_client.close()
        
        # 显示最终统计信息
        print("\n" + "=" * 70)
        print("会话统计信息")
        print("=" * 70)
        
        session_duration = time.time() - session_stats['start_time']
        
        # 时间统计
        print("\n[时间统计]")
        print(f"  会话时长: {session_duration:.1f} 秒 ({session_duration/60:.1f} 分钟)")
        print(f"  总请求数: {session_stats['total_requests']}")
        print(f"  总生成时间: {session_stats['total_generation_time']:.2f} 秒")
        print(f"  平均每次生成: {session_stats['total_generation_time']/session_stats['total_requests']*1000:.1f}ms" 
              if session_stats['total_requests'] > 0 else "  平均每次生成: N/A")
        
        # 远程调用统计
        if session_stats['all_trunk_times']:
            trunk_times = session_stats['all_trunk_times']
            print(f"\n[远程调用统计]")
            print(f"  总调用次数: {len(trunk_times)}")
            print(f"  总耗时: {sum(trunk_times)*1000:.1f}ms")
            print(f"  平均: {sum(trunk_times)/len(trunk_times)*1000:.1f}ms")
            print(f"  最短: {min(trunk_times)*1000:.1f}ms")
            print(f"  最长: {max(trunk_times)*1000:.1f}ms")
            print(f"  中位数: {sorted(trunk_times)[len(trunk_times)//2]*1000:.1f}ms")
        
        # 本地处理统计（详细）
        if session_stats['all_bottom_times']:
            bottom_times = session_stats['all_bottom_times']
            print(f"\n[本地处理统计]")
            print(f"  Bottom 模型:")
            print(f"    总调用: {len(bottom_times)}次")
            print(f"    总耗时: {sum(bottom_times)*1000:.1f}ms")
            print(f"    平均: {sum(bottom_times)/len(bottom_times)*1000:.1f}ms/次")
            print(f"    最短: {min(bottom_times)*1000:.1f}ms")
            print(f"    最长: {max(bottom_times)*1000:.1f}ms")
            print(f"    占比: {sum(bottom_times)/session_stats['total_generation_time']*100:.1f}%")
        
        if session_stats['all_top_times']:
            top_times = session_stats['all_top_times']
            print(f"  Top 模型:")
            print(f"    总调用: {len(top_times)}次")
            print(f"    总耗时: {sum(top_times)*1000:.1f}ms")
            print(f"    平均: {sum(top_times)/len(top_times)*1000:.1f}ms/次")
            print(f"    最短: {min(top_times)*1000:.1f}ms")
            print(f"    最长: {max(top_times)*1000:.1f}ms")
            print(f"    占比: {sum(top_times)/session_stats['total_generation_time']*100:.1f}%")
        
        # 远程调用统计（详细）
        if session_stats.get('all_network_times'):
            network_times = session_stats['all_network_times']
            print(f"\n[远程调用统计]")
            print(f"  总调用次数: {len(network_times)}")
            print(f"  总调用时间: {sum(network_times)*1000:.1f}ms")
            print(f"  平均调用时间: {sum(network_times)/len(network_times)*1000:.1f}ms/次")
            print(f"  最短调用: {min(network_times)*1000:.1f}ms")
            print(f"  最长调用: {max(network_times)*1000:.1f}ms")
            if session_stats['total_generation_time'] > 0:
                print(f"  占比: {sum(network_times)/session_stats['total_generation_time']*100:.1f}%")
            print(f"\n  说明:")
            print(f"    - 每次调用时间包括: 编码 + 网络传输 + 服务器计算 + 网络传输 + 解码")
            print(f"    - 网络传输时间 = 调用时间 - 服务器计算时间 (需从服务器响应获取)")
            print(f"    - 编码/解码时间通常 < 1ms，可忽略")
        
        # Token 统计
        print(f"\n[生成统计]")
        print(f"  总生成 tokens: {session_stats['total_tokens_generated']}")
        print(f"  平均每次: {session_stats['total_tokens_generated']/session_stats['total_requests']:.1f} tokens"
              if session_stats['total_requests'] > 0 else "  平均每次: N/A")
        
        # SplitLearnMonitor 统计（如果可用）
        if monitor:
            print(f"\n{'='*70}")
            print("[SplitLearnMonitor 详细统计]")
            print(f"{'='*70}")
            try:
                monitor.stop()
                
                # 获取详细统计信息
                summary = monitor.get_summary()
                
                # 资源统计
                if 'resource_stats' in summary:
                    resource_stats = summary['resource_stats']
                    print(f"\n[系统资源统计]")
                    print(f"  CPU:")
                    if 'cpu_mean' in resource_stats:
                        print(f"    平均使用率: {resource_stats['cpu_mean']:.1f}%")
                    if 'cpu_max' in resource_stats:
                        print(f"    峰值使用率: {resource_stats['cpu_max']:.1f}%")
                    print(f"  内存:")
                    if 'memory_mean_mb' in resource_stats:
                        print(f"    平均使用: {resource_stats['memory_mean_mb']:.1f} MB")
                    if 'memory_max_mb' in resource_stats:
                        print(f"    峰值使用: {resource_stats['memory_max_mb']:.1f} MB")
                    if 'memory_min_mb' in resource_stats:
                        print(f"    最低使用: {resource_stats['memory_min_mb']:.1f} MB")
                
                # 性能阶段统计
                if 'phase_stats' in summary:
                    phase_stats = summary['phase_stats']
                    print(f"\n[性能阶段统计]")
                    total_time = summary.get('total_time_ms', 0)
                    
                    # 按阶段显示统计
                    for phase_name, stats in phase_stats.items():
                        if isinstance(stats, dict):
                            count = stats.get('count', 0)
                            mean_ms = stats.get('mean_ms', 0)
                            total_ms = stats.get('total_ms', 0)
                            min_ms = stats.get('min_ms', 0)
                            max_ms = stats.get('max_ms', 0)
                            
                            # 计算占比
                            percentage = (total_ms / total_time * 100) if total_time > 0 else 0
                            
                            # 格式化阶段名称
                            phase_display = {
                                'bottom_model': 'Bottom 模型',
                                'trunk_remote': 'Trunk 远程调用',
                                'top_model': 'Top 模型',
                                'text_generation': '文本生成'
                            }.get(phase_name, phase_name)
                            
                            print(f"  {phase_display}:")
                            print(f"    调用次数: {count}")
                            print(f"    总耗时: {total_ms:.1f}ms")
                            print(f"    平均耗时: {mean_ms:.1f}ms/次")
                            print(f"    最短: {min_ms:.1f}ms")
                            print(f"    最长: {max_ms:.1f}ms")
                            if total_time > 0:
                                print(f"    占比: {percentage:.1f}%")
                    
                    if total_time > 0:
                        print(f"\n  总监控时间: {total_time:.1f}ms")
                
                # 打印标准摘要
                print(f"\n[监控摘要]")
                monitor.print_summary()
                
                # 保存监控报告
                try:
                    # 创建报告目录
                    reports_dir = os.path.join(project_root, 'test', 'reports')
                    os.makedirs(reports_dir, exist_ok=True)
                    
                    # 生成报告文件名（包含时间戳）
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    html_report_path = os.path.join(reports_dir, f"{monitor.session_name}_{timestamp}_report.html")
                    json_report_path = os.path.join(reports_dir, f"{monitor.session_name}_{timestamp}_report.json")
                    
                    # 使用之前在 finally 块开头获取的服务器监控数据
                    # 如果有服务器监控数据，生成合并报告
                    if server_monitoring_data and len(server_monitoring_data) > 0:
                        try:
                            merged_report_path = os.path.join(reports_dir, f"{monitor.session_name}_{timestamp}_merged_report.html")
                            merged_path = monitor.save_merged_report(
                                server_monitoring_snapshots=server_monitoring_data,
                                output_path=merged_report_path
                            )
                            print(f"\n[监控报告]")
                            print(f"  ✓ 合并报告已保存（包含客户端+服务器统计）: {merged_path}")
                        except Exception as merged_e:
                            print(f"  ⚠️  合并报告生成失败: {merged_e}，将生成客户端报告")
                            # 如果合并报告失败，回退到普通报告
                            server_monitoring_data = None
                    
                    # 如果没有服务器监控数据或合并失败，生成普通客户端报告
                    if not server_monitoring_data or len(server_monitoring_data) == 0:
                    report_path = monitor.save_report(output_path=html_report_path, format="html")
                    print(f"\n[监控报告]")
                        print(f"  HTML 报告已保存（仅客户端统计）: {report_path}")
                    
                    # 同时保存 JSON 格式
                    try:
                        json_path = monitor.save_report(output_path=json_report_path, format="json")
                        print(f"  JSON 数据已保存: {json_path}")
                    except Exception as json_e:
                        print(f"  [警告] JSON 报告保存失败: {json_e}")
                        import traceback
                        traceback.print_exc()
                except Exception as e:
                    print(f"\n[监控报告] 保存失败: {e}")
                    import traceback
                    traceback.print_exc()
                    
            except Exception as e:
                print(f"\n[SplitLearnMonitor] 获取统计失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            # 监控模块未启动时的提示
            if not MONITOR_AVAILABLE:
                print(f"\n{'='*70}")
                print("[监控报告]")
                print(f"{'='*70}")
                print("  ⚠️  监控模块不可用，无法生成 HTML 统计报告")
                print("  原因: splitlearn_monitor 模块未找到或导入失败")
                print("  提示: 确保 SplitLearnMonitor 模块已正确配置路径")
            elif monitor is None:
                print(f"\n{'='*70}")
                print("[监控报告]")
                print(f"{'='*70}")
                print("  ⚠️  监控模块启动失败，无法生成 HTML 统计报告")
                print("  请查看上方的错误信息了解具体原因")
        
        # 系统资源统计
        print(f"\n[系统资源使用]")
        if PSUTIL_AVAILABLE and process:
            try:
                # 当前资源使用
                current_cpu = process.cpu_percent(interval=0.1)
                current_memory = process.memory_info()
                
                # 系统总体资源
                system_cpu = psutil.cpu_percent(interval=0.1)
                system_memory = psutil.virtual_memory()
                
                print(f"  本进程 CPU 使用率: {current_cpu:.1f}%")
                print(f"  本进程内存使用: {current_memory.rss / 1024 / 1024:.1f} MB")
                if hasattr(current_memory, 'peak_wss'):
                    print(f"  内存峰值: {current_memory.peak_wss / 1024 / 1024:.1f} MB")
                print(f"  系统 CPU 使用率: {system_cpu:.1f}%")
                print(f"  系统内存使用: {system_memory.used / 1024 / 1024 / 1024:.1f} GB / "
                      f"{system_memory.total / 1024 / 1024 / 1024:.1f} GB "
                      f"({system_memory.percent:.1f}%)")
                
                # 内存增长
                if initial_memory:
                    memory_growth = (current_memory.rss - initial_memory.rss) / 1024 / 1024
                    print(f"  内存增长: {memory_growth:+.1f} MB")
            except Exception as e:
                print(f"  无法获取系统资源信息: {e}")
        else:
            print(f"  (需要安装 psutil 来显示系统资源: pip install psutil)")
        
        # 请求详情（可选，显示最近几次，包含详细信息）
        if session_stats['requests']:
            print(f"\n[最近请求详情] (显示最近5次)")
            for req in session_stats['requests'][-5:]:
                avg_bottom = sum(req['bottom_times'])/len(req['bottom_times'])*1000 if req['bottom_times'] else 0
                avg_network = sum(req['network_times'])/len(req['network_times'])*1000 if req['network_times'] else 0
                avg_top = sum(req['top_times'])/len(req['top_times'])*1000 if req['top_times'] else 0
                avg_trunk = sum(req['trunk_times'])/len(req['trunk_times'])*1000 if req['trunk_times'] else 0
                
                print(f"  [{req['timestamp']}] {req['input']}...")
                print(f"    └─ {req['tokens_generated']} tokens, {req['generation_time']*1000:.0f}ms, "
                      f"{req['trunk_calls']}步")
                print(f"       Bottom: {avg_bottom:.1f}ms/步 | "
                      f"网络: {avg_network:.1f}ms/步 | "
                      f"Top: {avg_top:.1f}ms/步 | "
                      f"总远程: {avg_trunk:.1f}ms/步")
        
        print("\n" + "=" * 70)
        print("连接已关闭")
        print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

