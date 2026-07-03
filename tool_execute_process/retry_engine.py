
from dataclasses import dataclass
from typing import Optional
from tool_execute_process.error_classify import LLMErrorClassify,ToolErrorType
import random
import time

@dataclass
class RetryPolicy:
    max_retries: int = 3
    base_delay:float = 1.0
    max_delay: float = 60.0 # 最大延迟
    backoff_factor: float = 2.0 #退避基数
    jitter_ratio: float = 0.1 #抖动比例
    rate_limit_delay: float = 5.0  # 限流特殊延迟

class RetryEngine:
    def __init__(self, Policy:Optional[RetryPolicy] = None):
        self.policy = Policy or RetryPolicy()


    def execute(self, func, *args, **kwargs):
        """
        执行带重试的函数

        Returns:
            (result, error_message, retry_count)
        """

        last_error = None

        for attempt in range(self.policy.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                return result, None, attempt
            
            except Exception as e:
                error_msg = str(e)
                error_type = LLMErrorClassify(error_msg)

                if attempt < self.policy.max_retries and  LLMErrorClassify.is_retryable(error_type):
                    delay = self.calc_delay(attempt, error_type)

                    time.sleep(delay)
                    last_error = error_msg
                else:
                    return None, last_error,attempt
                
        return None,last_error, attempt
    
    def calc_delay(self, attempt, error_type):

        if error_type == ToolErrorType.RATE_LIMITED:
            delay = self.policy.rate_limit_delay
        else:
            #指数退避
            # 第一次 1s
            # 第二次 2s
            # 第三次 4s
            delay = self.policy.base_delay * (self.policy.backoff_factor **attempt)
            delay = min(delay,self.policy.max_delay)

        #抖动，目的是减轻服务器压力
        #如果有1000个重试， 那么通过指数退避，到1s的时候，服务器还是要处理1000请求
        #如果服务器压力通过指数退避的时候，到达下一次的时候，还是抗住请求高峰
        #抖动的目标把1000个请求，通过抖动进行减流，例如在同样的是1s请求，划分成0.9~1 1~1.1,...来解决请求的问题
        jitter = random.uniform(delay,self.policy.jitter_ratio)

        delay += jitter
        return delay
            

