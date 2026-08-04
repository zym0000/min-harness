

from collections import defaultdict,deque
from typing import Optional,Any,Dict
import asyncio
import json
import hashlib

class AntioscillationWatchDog:
    '''
    防止大模型调用工具,循环调用,watchdog 就是用来防止大模型循环调用，采取的熔断措施
    '''
    def __init__(self,
                 max_repeat_threshold: int = 3,
                 window_size: int = 6,
                 warning_template: Optional[str] = None,
                 critical_template: Optional[str] = None):

        self.max_repeat_threshold = max_repeat_threshold
        self.window_size = window_size

        self.warning_template = warning_template or (
            "Detected high-frequency repeated tool calls to `{tool_name}`. "
            "You may be stuck in a logic loop. Please review observations and change strategy."
        )

        self.critical_template = critical_template or (
            "Tool {tool_name} triggered continuous loop oscillation."
        )

        # deque(maxlen) 自动丢弃最旧记录，滑窗语义天然正确
        self.history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.window_size))
        self.locks = defaultdict(asyncio.Lock)

    def _generate_action_hash(self, tool_name: str, arguments: Optional[dict]) -> str:
        """hash = sha256(tool_name + 规范化 arguments)"""
        try:
            serialized_args = json.dumps(arguments or {}, sort_keys=True,
                                         ensure_ascii=False, default=repr)
        except TypeError:
            serialized_args = repr(arguments)
        raw_text = f"{tool_name}:{serialized_args}"
        return hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

    async def record_and_check(self, task_id: str, tool_call: Any):
        async with self.locks[task_id]:
            hash_text = self._generate_action_hash(
                tool_call.tool_name, tool_call.arguments)

            task_history = self.history[task_id]
            task_history.append(hash_text)      # 超出 maxlen 自动挤掉最旧一条

            repeat_count = sum(1 for h in task_history if h == hash_text)

            if repeat_count > self.max_repeat_threshold:
                return "CRITICAL", self.critical_template.format(
                    tool_name=tool_call.tool_name)

            if repeat_count == self.max_repeat_threshold:
                return "WARNING", self.warning_template.format(
                    tool_name=tool_call.tool_name)

            return "SAFE", ""

    def clear_task(self, task_id: str) -> None:
        self.history.pop(task_id, None)
        self.locks.pop(task_id, None)

            


