
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

import time

class TaskStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class ContinueResult(Enum):
    FAILED = auto()
    QUEUED = auto()
    ACTIVATED = auto()


@dataclass
class TaskState:
    task_id: str
    user_input: str
    system_prompt: str
    current_step: int
    task_status: TaskStatus
    messages: List[Dict[str, str]]  # role, content
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    total_tokens_used: int = 0
    pending_input: deque = field(default_factory=deque)

    # 记忆相关
    task_summary: str = ""                              # 当前任务目标
    key_facts: List[str] = field(default_factory=list)  # 关键事实和发现
    memory_segment: Optional[str] = None                # 增量结构化记忆
    memory_cursor: int = 0                              # 已总结到 messages 的第几条

    def to_checkpoint(self) -> Dict:
        """全量快照"""
        return {
            "task_id": self.task_id,
            "user_input": self.user_input,
            "system_prompt": self.system_prompt,
            "current_step": self.current_step,
            "messages": self.messages,
            "status": self.task_status.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "total_tokens_used": self.total_tokens_used,
            "pending_input": list(self.pending_input),
            "task_summary": self.task_summary,
            "key_facts": list(self.key_facts),
            "memory_segment": self.memory_segment,
            "memory_cursor": self.memory_cursor,
        }

    @classmethod
    def from_checkpoint(cls, data: Dict) -> "TaskState":
        state = cls(
            task_id=data["task_id"],
            user_input=data.get("user_input", ""),
            system_prompt=data.get("system_prompt", ""),
            current_step=data.get("current_step", 0),
            task_status=TaskStatus[data.get("status", "PENDING")],
            messages=list(data.get("messages", [])),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            total_tokens_used=data.get("total_tokens_used", 0),
            pending_input=deque(data.get("pending_input", [])),
            task_summary=data.get("task_summary", ""),
            key_facts=list(data.get("key_facts", [])),
            memory_segment=data.get("memory_segment"),
            memory_cursor=data.get("memory_cursor", 0),
        )
        # 进程重启后，RUNNING/PAUSED 执行者已经没了，
        # 统一降为 PAUSED，等 continue_task 重新激活
        if state.task_status in (TaskStatus.RUNNING,):
            state.task_status = TaskStatus.PAUSED
        return state

