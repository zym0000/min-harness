
from enum import Enum,auto
from dataclasses import dataclass,field
from typing import List, Dict,Optional

import time
from collections import deque

class TaskStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    PAUSED  = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()

class ContinueResult(Enum):
    FAILED = auto()
    QUEUED = auto()
    ACTIVATED = auto()

@dataclass
class TaskState:
    task_id:str
    user_input:str
    system_prompt:str
    current_step:int
    task_status:TaskStatus
    messages:List[Dict[str,str]]  #role, 
    created_at:float = field(default_factory=time.time)
    updated_at:float = field(default_factory=time.time)
    total_tokens_used: int = 0
    pending_input: deque = field(default_factory=deque)

    #记忆相关
    task_summary:str = "" #当前任务目标
    key_facts:List[str] = field(default_factory = list) #关键事实和发现
    memory_segment:Optional[str] = None #增量结构化记忆

    def to_checkpoint(self):
        return { 
            "task_id":self.task_id,
            "user_input":self.user_input,
            "current_step":self.current_step,
            "messages":self.messages,
            "status":self.task_status.name,
            "created_at":self.created_at,
            "updated_at":self.updated_at,
        }
    

