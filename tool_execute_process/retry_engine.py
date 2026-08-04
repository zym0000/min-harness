
import asyncio
import random
import time
from dataclasses import dataclass
from typing import Optional
import inspect

from tool_execute_process.error_classify import ErrorClassify, ToolErrorType

@dataclass
class RetryPolicy:
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0       # 最大延迟
    backoff_factor: float = 2.0   # 退避基数
    jitter_ratio: float = 0.1     # 抖动比例
    rate_limit_delay: float = 5.0  # 限流特殊延迟


class RetryEngine:
    def __init__(self, policy: Optional[RetryPolicy] = None):
        self.policy = policy or RetryPolicy()

    def execute(self, func, *args, **kwargs):
        """
        同步执行带重试的函数（注意：sleep 会阻塞调用线程，async 环境请用 execute_async）。

        Returns: (result, error_message, retry_count)
        """
        last_error = None
        for attempt in range(self.policy.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                return result, None, attempt
            except Exception as e:
                error_msg = str(e)
                error_type = ErrorClassify.classify(error_msg)
                last_error = error_msg

                if attempt < self.policy.max_retries and ErrorClassify.is_retryable(error_type):
                    time.sleep(self.calc_delay(attempt, error_type))
                else:
                    return None, last_error, attempt
        return None, last_error, attempt

    async def execute_async(self, func, *args, **kwargs):
        """async 版本：不阻塞事件循环。func 可为协程函数或普通函数。"""
        last_error = None
        for attempt in range(self.policy.max_retries + 1):
            try:
                if inspect.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = await asyncio.to_thread(func, *args, **kwargs)
                return result, None, attempt
            except asyncio.CancelledError:
                raise
            except Exception as e:
                error_msg = str(e)
                error_type = ErrorClassify.classify(error_msg)
                last_error = error_msg

                if attempt < self.policy.max_retries and ErrorClassify.is_retryable(error_type):
                    await asyncio.sleep(self.calc_delay(attempt, error_type))
                else:
                    return None, last_error, attempt
        return None, last_error, attempt

    def calc_delay(self, attempt, error_type):
        if error_type == ToolErrorType.RATE_LIMITED:
            delay = self.policy.rate_limit_delay
        else:
            # 指数退避：1s -> 2s -> 4s ...
            delay = self.policy.base_delay * (self.policy.backoff_factor ** attempt)
            delay = min(delay, self.policy.max_delay)

        # 对称抖动 
        jitter = delay * random.uniform(-self.policy.jitter_ratio, self.policy.jitter_ratio)
        return max(0.0, delay + jitter)

