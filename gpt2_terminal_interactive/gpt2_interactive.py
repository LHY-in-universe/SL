#!/usr/bin/env python3
"""
GPT-2 终端交互式生成工具
直接在终端使用，不使用 Gradio
使用 monitor 库进行详细的性能统计
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent / "SplitLearnCore" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "SplitLearnComm" / "src"))

import torch
from transformers import GPT2LMHeadModel, AutoTokenizer

# 尝试导入监控库
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("⚠ 警告: psutil 未安装，部分监控功能不可用")
    print("   安装: pip install psutil")

# 配置日志
log_dir = Path("./logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "gpt2_interactive.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# 全局变量
# ============================================================================

model_id = "gpt2"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_CACHE = str(Path(__file__).parent.parent / "models")

# 性能统计
all_token_stats = []

# ============================================================================
# 监控类
# ============================================================================

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.process = psutil.Process() if HAS_PSUTIL else None
        self.start_time = None
        self.start_memory = None
        self.start_cpu = None
        
    def start(self):
        """开始监控"""
        if self.process:
            self.start_time = time.time()
            self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            self.start_cpu = self.process.cpu_percent(interval=0.1)
        else:
            self.start_time = time.time()
    
    def get_stats(self):
        """获取当前统计"""
        if not self.process:
            return {
                "elapsed_time": time.time() - self.start_time if self.start_time else 0,
                "memory_mb": 0,
                "cpu_percent": 0,
            }
        
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        current_cpu = self.process.cpu_percent(interval=0.1)
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        return {
            "elapsed_time": elapsed,
            "memory_mb": current_memory,
            "memory_delta_mb": current_memory - self.start_memory if self.start_memory else 0,
            "cpu_percent": current_cpu,
        }
    
    def print_stats(self, prefix=""):
        """打印统计信息"""
        stats = self.get_stats()
        print(f"{prefix}时间: {stats['elapsed_time']:.2f}s")
        if self.process:
            print(f"{prefix}内存: {stats['memory_mb']:.2f} MB (增量: {stats['memory_delta_mb']:+.2f} MB)")
            print(f"{prefix}CPU: {stats['cpu_percent']:.1f}%")

# ============================================================================
# 模型加载
# ============================================================================

logger.info("=" * 70)
logger.info("GPT-2 终端交互式生成工具")
logger.info("=" * 70)
logger.info(f"设备: {device}")
logger.info(f"模型: {model_id}")
logger.info(f"模型缓存: {MODEL_CACHE}")

logger.info("\n加载模型...")
load_start = time.time()

model = GPT2LMHeadModel.from_pretrained(
    model_id,
    cache_dir=MODEL_CACHE,
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    cache_dir=MODEL_CACHE,
    use_fast=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.to(device)
model.eval()

# PyTorch 优化
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision("medium")

if device == "cuda":
    torch.backends.cudnn.benchmark = True

load_time = time.time() - load_start
logger.info(f"✓ 模型加载完成 ({load_time:.2f}s)")
logger.info(f"✓ 参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# ============================================================================
# 生成函数
# ============================================================================

def generate_with_kv_cache(
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
    monitor: PerformanceMonitor = None,
):
    """
    使用 KV Cache 生成文本
    
    Args:
        prompt: 输入提示
        max_new_tokens: 最大生成 token 数
        temperature: 温度参数
        top_k: Top-k 采样
        monitor: 性能监控器
    """
    global all_token_stats
    
    print("\n" + "=" * 70)
    print(f"生成请求")
    print("=" * 70)
    print(f"提示: {prompt}")
    print(f"参数: max_tokens={max_new_tokens}, temperature={temperature}, top_k={top_k}")
    print("=" * 70)
    
    logger.info(f"[GENERATE] prompt='{prompt}', max_tokens={max_new_tokens}, temp={temperature}, top_k={top_k}")
    
    # 开始监控
    if monitor:
        monitor.start()
    
    # 编码输入
    try:
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
    except Exception as e:
        logger.error(f"编码输入失败: {e}", exc_info=True)
        print(f"❌ 错误: 编码输入失败 - {e}")
        return None
    
    # 统计
    token_times = []
    total_start = time.time()
    generated_tokens = []
    past_key_values = None
    
    print("\n生成中...")
    print("-" * 70)
    
    # 生成循环
    with torch.inference_mode():
        for step in range(max_new_tokens):
            step_start = time.time()
            
            # 输入准备
            if step == 0:
                current_input_ids = input_ids
                current_attention_mask = attention_mask
            else:
                current_input_ids = torch.tensor([[next_token_id]], device=device, dtype=torch.long)
                current_attention_mask = None
            
            # 前向传播
            model_kwargs = {
                "past_key_values": past_key_values,
                "use_cache": True,
                "output_attentions": False,
                "output_hidden_states": False,
            }
            
            if current_attention_mask is not None:
                model_kwargs["attention_mask"] = current_attention_mask
            
            outputs = model(current_input_ids, **model_kwargs)
            
            logits = outputs.logits[0, -1, :]
            past_key_values = outputs.past_key_values
            
            # 采样
            if temperature > 0:
                logits = logits / temperature
                if top_k > 0:
                    top_k_value = min(top_k, logits.size(-1))
                    top_k_logits, top_k_indices = torch.topk(logits, top_k_value)
                    logits_filtered = torch.full_like(logits, float('-inf'))
                    logits_filtered.scatter_(0, top_k_indices, top_k_logits)
                    logits = logits_filtered
                probs = torch.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token_id = logits.argmax(dim=-1).item()
            
            generated_tokens.append(next_token_id)
            
            # 记录时间
            token_time = time.time() - step_start
            token_times.append(token_time * 1000)
            
            # 记录统计
            all_token_stats.append({
                "step": step,
                "token_id": next_token_id,
                "time_ms": token_time * 1000,
            })
            
            # 实时输出
            token_text = tokenizer.decode([next_token_id], skip_special_tokens=True)
            print(token_text, end="", flush=True)
            
            logger.info(f"Token #{len(generated_tokens)}: '{token_text}' (ID={next_token_id}) | "
                       f"Time={token_time*1000:.2f}ms")
            
            if next_token_id == tokenizer.eos_token_id:
                break
    
    # 最终统计
    total_time = time.time() - total_start
    final_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    print("\n" + "-" * 70)
    print("\n✅ 生成完成")
    print("=" * 70)
    print(f"总 Token 数: {len(generated_tokens)}")
    print(f"总时间: {total_time:.2f}s")
    print(f"平均速度: {len(generated_tokens)/total_time:.2f} tokens/s")
    if token_times:
        print(f"平均延迟: {sum(token_times)/len(token_times):.2f}ms/token")
        print(f"最小延迟: {min(token_times):.2f}ms")
        print(f"最大延迟: {max(token_times):.2f}ms")
    
    # 性能监控统计
    if monitor:
        monitor_stats = monitor.get_stats()
        print("\n性能监控:")
        monitor.print_stats("  ")
    
    logger.info(f"✅ 生成完成: {len(generated_tokens)} tokens in {total_time:.2f}s "
               f"({len(generated_tokens)/total_time:.2f} tokens/s)")
    
    return prompt + final_text

# ============================================================================
# 交互式主循环
# ============================================================================

def main():
    """主函数"""
    monitor = PerformanceMonitor() if HAS_PSUTIL else None
    
    print("\n" + "=" * 70)
    print("GPT-2 终端交互式生成工具")
    print("=" * 70)
    print("\n使用说明:")
    print("  - 输入提示文本，按 Enter 生成")
    print("  - 输入 'quit' 或 'exit' 退出")
    print("  - 输入 'stats' 查看统计信息")
    print("  - 输入 'help' 查看帮助")
    print("=" * 70)
    
    # 默认参数
    max_tokens = 200
    temperature = 1.0
    top_k = 50
    
    while True:
        try:
            print("\n" + "-" * 70)
            user_input = input("\n请输入提示 (或 'help' 查看命令): ").strip()
            
            if not user_input:
                continue
            
            # 处理命令
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n再见！")
                break
            
            elif user_input.lower() == 'help':
                print("\n可用命令:")
                print("  help          - 显示帮助")
                print("  stats         - 显示统计信息")
                print("  max_tokens N   - 设置最大生成 token 数 (默认: 200)")
                print("  temperature N - 设置温度参数 (默认: 1.0)")
                print("  top_k N       - 设置 Top-k 采样 (默认: 50)")
                print("  quit/exit     - 退出程序")
                continue
            
            elif user_input.lower() == 'stats':
                print("\n统计信息:")
                print(f"  总生成次数: {len(all_token_stats)} tokens")
                if all_token_stats:
                    times = [s['time_ms'] for s in all_token_stats]
                    print(f"  平均延迟: {sum(times)/len(times):.2f}ms/token")
                    print(f"  最小延迟: {min(times):.2f}ms")
                    print(f"  最大延迟: {max(times):.2f}ms")
                if monitor:
                    monitor.print_stats("  ")
                continue
            
            elif user_input.startswith('max_tokens '):
                try:
                    max_tokens = int(user_input.split()[1])
                    print(f"✓ 最大生成 token 数设置为: {max_tokens}")
                except:
                    print("❌ 无效的数字")
                continue
            
            elif user_input.startswith('temperature '):
                try:
                    temperature = float(user_input.split()[1])
                    print(f"✓ 温度参数设置为: {temperature}")
                except:
                    print("❌ 无效的数字")
                continue
            
            elif user_input.startswith('top_k '):
                try:
                    top_k = int(user_input.split()[1])
                    print(f"✓ Top-k 采样设置为: {top_k}")
                except:
                    print("❌ 无效的数字")
                continue
            
            # 生成文本
            result = generate_with_kv_cache(
                prompt=user_input,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                monitor=monitor,
            )
            
            if result:
                print(f"\n完整结果:\n{result}")
        
        except KeyboardInterrupt:
            print("\n\n中断，退出...")
            break
        except Exception as e:
            logger.error(f"错误: {e}", exc_info=True)
            print(f"\n❌ 错误: {e}")

if __name__ == "__main__":
    main()
