"""
激活值缓存管理器
用于分布式训练时在服务端缓存前向传播的激活值，供反向传播使用
"""

import time
import threading
import logging
from typing import Dict, Optional, Any
from collections import OrderedDict

import torch

logger = logging.getLogger(__name__)


class ActivationCache:
    """
    线程安全的激活值缓存，支持 TTL 和 LRU 驱逐策略

    用于训练场景：
    1. 前向传播时，服务端缓存中间激活值
    2. 反向传播时，客户端发送梯度，服务端检索激活值进行反向计算
    3. 自动清理过期和最久未使用的激活值

    Example:
        >>> cache = ActivationCache(max_size=100, ttl_seconds=60)
        >>> # 前向传播时缓存
        >>> forward_id = "abc123"
        >>> activation = hidden_states.detach().requires_grad_(True)
        >>> cache.store(forward_id, activation)
        >>>
        >>> # 反向传播时检索
        >>> cached_activation = cache.retrieve(forward_id)
        >>> cached_activation.backward(grad_output)
    """

    def __init__(
        self,
        max_size: int = 100,
        ttl_seconds: float = 60.0,
        enable_stats: bool = True
    ):
        """
        Args:
            max_size: 最大缓存条目数（达到后使用 LRU 驱逐）
            ttl_seconds: 缓存条目存活时间（秒）
            enable_stats: 是否启用统计信息收集
        """
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.enable_stats = enable_stats

        # 使用 OrderedDict 实现 LRU
        self.cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.timestamps: Dict[str, float] = {}
        self.lock = threading.Lock()

        # 统计信息
        self.stats = {
            'total_stores': 0,
            'total_retrievals': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'expired_evictions': 0,
            'lru_evictions': 0,
        }

        logger.info(
            f"Initialized ActivationCache: max_size={max_size}, "
            f"ttl_seconds={ttl_seconds}"
        )

    def store(self, forward_id: str, activation: torch.Tensor) -> None:
        """
        存储激活值

        Args:
            forward_id: 前向传播唯一 ID
            activation: 激活值张量（需要支持梯度计算）

        注意：
            - 激活值会被 detach 并重新设置 requires_grad=True
            - 这样可以避免保留整个前向计算图
        """
        with self.lock:
            # 清理过期条目
            self._cleanup_expired()

            # LRU 驱逐（如果缓存已满）
            if len(self.cache) >= self.max_size:
                self._evict_oldest()

            # 存储激活值（detach 后重新启用梯度）
            cached_activation = activation.detach().requires_grad_(True)
            self.cache[forward_id] = cached_activation
            self.cache.move_to_end(forward_id)  # 移到最新位置

            # 记录时间戳
            self.timestamps[forward_id] = time.time()

            # 更新统计
            if self.enable_stats:
                self.stats['total_stores'] += 1

            logger.debug(
                f"Stored activation {forward_id}: "
                f"shape={tuple(activation.shape)}, "
                f"cache_size={len(self.cache)}/{self.max_size}"
            )

    def retrieve(self, forward_id: str) -> torch.Tensor:
        """
        检索并移除激活值

        Args:
            forward_id: 前向传播唯一 ID

        Returns:
            激活值张量（启用了梯度）

        Raises:
            KeyError: 如果 forward_id 不存在或已过期
        """
        with self.lock:
            # 更新统计
            if self.enable_stats:
                self.stats['total_retrievals'] += 1

            # 检查是否存在
            if forward_id not in self.cache:
                if self.enable_stats:
                    self.stats['cache_misses'] += 1
                raise KeyError(
                    f"Activation {forward_id} not found in cache "
                    "(may have expired or never cached)"
                )

            # 检查是否过期
            timestamp = self.timestamps[forward_id]
            age = time.time() - timestamp
            if age > self.ttl:
                # 过期，移除
                self.cache.pop(forward_id)
                self.timestamps.pop(forward_id)
                if self.enable_stats:
                    self.stats['cache_misses'] += 1
                    self.stats['expired_evictions'] += 1
                raise KeyError(
                    f"Activation {forward_id} expired "
                    f"(age={age:.1f}s > ttl={self.ttl}s)"
                )

            # 检索并移除
            activation = self.cache.pop(forward_id)
            self.timestamps.pop(forward_id)

            if self.enable_stats:
                self.stats['cache_hits'] += 1

            logger.debug(
                f"Retrieved activation {forward_id}: "
                f"shape={tuple(activation.shape)}, age={age:.2f}s"
            )

            return activation

    def _cleanup_expired(self) -> int:
        """
        清理过期的缓存条目

        Returns:
            清理的条目数
        """
        current_time = time.time()
        expired_ids = [
            fid for fid, ts in self.timestamps.items()
            if current_time - ts > self.ttl
        ]

        for fid in expired_ids:
            self.cache.pop(fid, None)
            self.timestamps.pop(fid, None)
            if self.enable_stats:
                self.stats['expired_evictions'] += 1

        if expired_ids:
            logger.debug(f"Cleaned up {len(expired_ids)} expired activations")

        return len(expired_ids)

    def _evict_oldest(self) -> Optional[str]:
        """
        驱逐最久未使用的缓存条目（LRU）

        Returns:
            被驱逐的 forward_id，如果缓存为空则返回 None
        """
        if not self.cache:
            return None

        # OrderedDict.popitem(last=False) 弹出最旧的条目
        oldest_id, _ = self.cache.popitem(last=False)
        self.timestamps.pop(oldest_id, None)

        if self.enable_stats:
            self.stats['lru_evictions'] += 1

        logger.debug(f"LRU evicted activation {oldest_id}")

        return oldest_id

    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            统计信息字典，包括:
            - num_entries: 当前缓存条目数
            - memory_mb: 估计内存使用（MB）
            - avg_size_mb: 平均每个条目大小（MB）
            - hit_rate: 缓存命中率
            - total_stores: 总存储次数
            - total_retrievals: 总检索次数
            - cache_hits: 缓存命中次数
            - cache_misses: 缓存未命中次数
            - expired_evictions: 过期驱逐次数
            - lru_evictions: LRU 驱逐次数
        """
        with self.lock:
            total_memory = sum(
                a.element_size() * a.nelement()
                for a in self.cache.values()
            )
            memory_mb = total_memory / 1e6
            avg_size_mb = memory_mb / len(self.cache) if self.cache else 0

            total_retrievals = self.stats['total_retrievals']
            hit_rate = (
                self.stats['cache_hits'] / total_retrievals
                if total_retrievals > 0 else 0.0
            )

            return {
                'num_entries': len(self.cache),
                'memory_mb': round(memory_mb, 2),
                'avg_size_mb': round(avg_size_mb, 2),
                'hit_rate': round(hit_rate, 3),
                **self.stats
            }

    def clear(self) -> None:
        """清空缓存"""
        with self.lock:
            num_cleared = len(self.cache)
            self.cache.clear()
            self.timestamps.clear()
            logger.info(f"Cleared {num_cleared} cached activations")

    def __len__(self) -> int:
        """返回当前缓存条目数"""
        with self.lock:
            return len(self.cache)

    def __repr__(self) -> str:
        with self.lock:
            return (
                f"ActivationCache(size={len(self.cache)}/{self.max_size}, "
                f"ttl={self.ttl}s)"
            )
