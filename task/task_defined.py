
from enum import Enum,auto
from dataclasses import dataclass,field
from typing import List, Dict
from collections import defaultdict

import time

class TaskStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    PAUSED  = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()

@dataclass
class TaskState:
    task_id:str
    user_input:str
    current_step:int
    task_status:TaskStatus
    messages:List[Dict[str,str]]  #role, 
    created_at:float = field(default_factory=time.time)
    updated_at:float = field(default_factory=time.time)

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
    

