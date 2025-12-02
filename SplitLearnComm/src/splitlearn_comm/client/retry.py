"""
重试策略实现
"""

import logging
import random
import time
from typing import Callable, TypeVar, List

import grpc

logger = logging.getLogger(__name__)

T = TypeVar('T')

# 不可重试的错误代码
NON_RETRIABLE_CODES = [
    grpc.StatusCode.INVALID_ARGUMENT,
    grpc.StatusCode.UNAUTHENTICATED,
    grpc.StatusCode.PERMISSION_DENIED,
    grpc.StatusCode.NOT_FOUND,
    grpc.StatusCode.UNIMPLEMENTED,
]


class RetryStrategy:
    """重试策略基类"""

    def execute(self, func: Callable[[], T]) -> T:
        """
        执行函数，失败时根据策略重试

        Args:
            func: 要执行的函数

        Returns:
            函数返回值

        Raises:
            最后一次异常（如果所有重试都失败）
        """
        raise NotImplementedError


class ExponentialBackoff(RetryStrategy):
    """
    指数退避重试策略

    使用指数退避 + 随机抖动来避免雷鸣群效应。

    Example:
        >>> retry = ExponentialBackoff(max_retries=3, initial_delay=1.0)
        >>> result = retry.execute(lambda: risky_function())
    """

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
        jitter: float = 0.25
    ):
        """
        Args:
            max_retries: 最大重试次数
            initial_delay: 初始延迟（秒）
            max_delay: 最大延迟上限（秒）
            jitter: 随机抖动比例 (0.0-1.0)
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.jitter = jitter

    def execute(self, func: Callable[[], T]) -> T:
        """执行函数，失败时重试"""
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return func()

            except grpc.RpcError as e:
                last_exception = e

                # 检查是否为不可重试的错误
                if e.code() in NON_RETRIABLE_CODES:
                    logger.error(f"Non-retriable error: {e.code()}")
                    raise

                # 如果还有重试机会
                if attempt < self.max_retries - 1:
                    # 计算延迟：指数退避 + 最大延迟限制
                    base_delay = self.initial_delay * (2 ** attempt)
                    delay = min(base_delay, self.max_delay)

                    # 添加随机抖动
                    if self.jitter > 0:
                        jitter_amount = delay * self.jitter * (2 * random.random() - 1)
                        delay = max(0.1, delay + jitter_amount)

                    logger.warning(
                        f"Request failed ({e.code()}), "
                        f"retrying {attempt + 1}/{self.max_retries - 1} "
                        f"after {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Max retries ({self.max_retries}) reached")

        # 所有重试都失败
        if last_exception:
            raise last_exception
        raise RuntimeError("Retry failed without exception")


class FixedDelay(RetryStrategy):
    """
    固定延迟重试策略

    每次重试使用相同的延迟时间。

    Example:
        >>> retry = FixedDelay(max_retries=3, delay=2.0)
        >>> result = retry.execute(lambda: risky_function())
    """

    def __init__(self, max_retries: int = 3, delay: float = 1.0):
        """
        Args:
            max_retries: 最大重试次数
            delay: 重试延迟（秒）
        """
        self.max_retries = max_retries
        self.delay = delay

    def execute(self, func: Callable[[], T]) -> T:
        """执行函数，失败时重试"""
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return func()

            except grpc.RpcError as e:
                last_exception = e

                if e.code() in NON_RETRIABLE_CODES:
                    logger.error(f"Non-retriable error: {e.code()}")
                    raise

                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Request failed ({e.code()}), "
                        f"retrying {attempt + 1}/{self.max_retries - 1} "
                        f"after {self.delay:.1f}s..."
                    )
                    time.sleep(self.delay)

        if last_exception:
            raise last_exception
        raise RuntimeError("Retry failed without exception")
