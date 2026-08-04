from enum import Enum,auto
from dataclasses import dataclass
from typing import Optional,Dict,Any


class EventType(Enum):
    TASK_STARTED = auto()
    TASK_CANCELLED = auto()
    TASK_FAILED = auto()
    TASK_COMPLETED = auto()
    THINKING_STARTED = auto()
    THINKING_DELTA = auto() 
    THINKING_COMPLETED = auto()
    TOOL_CALL_PARSED = auto()
    TOOL_VALIDATION_FAILED = auto()
    TOOL_VALIDATION_PASSED = auto()
    TOOL_EXECUTION_STARTED = auto()
    TOOL_EXECUTION_FAILED = auto()
    TOOL_EXECUTION_COMPLETED = auto()
    NEED_APPROVAL = auto()
    APPROVAL_DENIED = auto()
    APPROVAL_GRANTED = auto()
    FINAL_ANSWER = auto()
    PROGRESS_UPDATE = auto()

@dataclass
class LoopEvent:
    event_type: EventType
    task_id: str
    timestamp: float
    step_num: int = 0
    content: Any = None
    tool_name: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    error: Any = None
    trace_id: str = ""
    
