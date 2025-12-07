"""
Token 级别监控器
用于统计每个生成 token 的时间
"""

import time
from typing import Dict, List, Optional
from collections import defaultdict
from datetime import datetime
import json


class TokenMonitor:
    """监控每个 token 生成时间的监控器"""
    
    def __init__(self, session_name: Optional[str] = None):
        """
        初始化监控器
        
        Args:
            session_name: 会话名称
        """
        if session_name is None:
            session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_name = session_name
        
        # 存储每次交互的 token 时间
        self.interactions: List[Dict] = []
        self.current_interaction: Optional[Dict] = None
        
    def start_interaction(self, user_input: str, interaction_id: int):
        """开始一次交互"""
        self.current_interaction = {
            'interaction_id': interaction_id,
            'user_input': user_input,
            'start_time': time.time(),
            'tokens': [],
            'total_tokens': 0,
            'total_time_ms': 0.0,
            'encode_time_ms': 0.0,
            'decode_time_ms': 0.0,
        }
    
    def record_token(self, token_id: int, token_text: str, generation_time_ms: float, 
                    step: int, metadata: Optional[Dict] = None):
        """
        记录一个 token 的生成时间
        
        Args:
            token_id: Token ID
            token_text: Token 文本
            generation_time_ms: 生成时间（毫秒）
            step: 生成步骤（第几个 token）
            metadata: 可选的元数据
        """
        if self.current_interaction is None:
            return
        
        token_info = {
            'step': step,
            'token_id': token_id,
            'token_text': token_text,
            'generation_time_ms': generation_time_ms,
            'timestamp': time.time(),
        }
        if metadata:
            token_info['metadata'] = metadata
        
        self.current_interaction['tokens'].append(token_info)
        self.current_interaction['total_tokens'] += 1
        self.current_interaction['total_time_ms'] += generation_time_ms
    
    def record_encode(self, encode_time_ms: float):
        """记录编码时间"""
        if self.current_interaction:
            self.current_interaction['encode_time_ms'] = encode_time_ms
    
    def record_decode(self, decode_time_ms: float):
        """记录解码时间"""
        if self.current_interaction:
            self.current_interaction['decode_time_ms'] = decode_time_ms
    
    def end_interaction(self, response: str):
        """结束一次交互"""
        if self.current_interaction:
            self.current_interaction['end_time'] = time.time()
            self.current_interaction['response'] = response
            self.current_interaction['total_time_ms'] = (
                (self.current_interaction['end_time'] - self.current_interaction['start_time']) * 1000
            )
            self.interactions.append(self.current_interaction)
            self.current_interaction = None
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        if not self.interactions:
            return {}
        
        all_token_times = []
        all_total_times = []
        all_encode_times = []
        all_decode_times = []
        total_tokens = 0
        
        for interaction in self.interactions:
            all_total_times.append(interaction['total_time_ms'])
            all_encode_times.append(interaction['encode_time_ms'])
            all_decode_times.append(interaction['decode_time_ms'])
            total_tokens += interaction['total_tokens']
            
            for token in interaction['tokens']:
                all_token_times.append(token['generation_time_ms'])
        
        def calc_stats(values):
            if not values:
                return {}
            import numpy as np
            arr = np.array(values)
            return {
                'mean': float(np.mean(arr)),
                'median': float(np.median(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'std': float(np.std(arr)),
                'p50': float(np.percentile(arr, 50)),
                'p95': float(np.percentile(arr, 95)),
                'p99': float(np.percentile(arr, 99)),
            }
        
        return {
            'session_name': self.session_name,
            'total_interactions': len(self.interactions),
            'total_tokens': total_tokens,
            'token_generation': calc_stats(all_token_times),
            'total_time': calc_stats(all_total_times),
            'encode_time': calc_stats(all_encode_times),
            'decode_time': calc_stats(all_decode_times),
            'avg_tokens_per_interaction': total_tokens / len(self.interactions) if self.interactions else 0,
        }
    
    def print_summary(self):
        """打印统计摘要"""
        stats = self.get_statistics()
        if not stats:
            print("没有统计数据")
            return
        
        print("\n" + "=" * 60)
        print("监控统计摘要")
        print("=" * 60)
        print(f"会话名称: {stats['session_name']}")
        print(f"总交互次数: {stats['total_interactions']}")
        print(f"总生成 Token 数: {stats['total_tokens']}")
        print(f"平均每次交互 Token 数: {stats['avg_tokens_per_interaction']:.1f}")
        print()
        
        if stats['token_generation']:
            print("每个 Token 生成时间统计 (ms):")
            tg = stats['token_generation']
            print(f"  平均: {tg['mean']:.2f} ms")
            print(f"  中位数: {tg['median']:.2f} ms")
            print(f"  最小: {tg['min']:.2f} ms")
            print(f"  最大: {tg['max']:.2f} ms")
            print(f"  P95: {tg['p95']:.2f} ms")
            print(f"  P99: {tg['p99']:.2f} ms")
            print()
        
        if stats['total_time']:
            print("每次交互总时间统计 (ms):")
            tt = stats['total_time']
            print(f"  平均: {tt['mean']:.2f} ms")
            print(f"  中位数: {tt['median']:.2f} ms")
            print()
        
        if stats['encode_time']:
            print("编码时间统计 (ms):")
            et = stats['encode_time']
            print(f"  平均: {et['mean']:.2f} ms")
            print()
        
        if stats['decode_time']:
            print("解码时间统计 (ms):")
            dt = stats['decode_time']
            print(f"  平均: {dt['mean']:.2f} ms")
            print()
        
        print("=" * 60)
    
    def save_report(self, output_path: Optional[str] = None, format: str = "json"):
        """
        保存监控报告
        
        Args:
            output_path: 输出路径
            format: 格式 ("json", "txt")
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{self.session_name}_{timestamp}_report.{format}"
        
        if format == "json":
            report = {
                'session_name': self.session_name,
                'statistics': self.get_statistics(),
                'interactions': self.interactions,
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        elif format == "txt":
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Token Monitor Report - {self.session_name}\n")
                f.write("=" * 60 + "\n\n")
                stats = self.get_statistics()
                f.write(f"总交互次数: {stats.get('total_interactions', 0)}\n")
                f.write(f"总生成 Token 数: {stats.get('total_tokens', 0)}\n\n")
                
                for interaction in self.interactions:
                    f.write(f"交互 #{interaction['interaction_id']}\n")
                    f.write(f"  输入: {interaction['user_input']}\n")
                    f.write(f"  生成 Token 数: {interaction['total_tokens']}\n")
                    f.write(f"  总时间: {interaction['total_time_ms']:.2f} ms\n")
                    f.write(f"  Token 详情:\n")
                    for token in interaction['tokens']:
                        f.write(f"    Step {token['step']}: '{token['token_text']}' "
                               f"({token['generation_time_ms']:.2f} ms)\n")
                    f.write("\n")
        
        return output_path
