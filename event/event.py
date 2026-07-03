from enum import Enum,auto
from dataclasses import dataclass
from typing import Optional,Dict,Any


class EventType(Enum):
    TASK_STARTED = auto()
    TASK_COMPLETED = auto()
    TASK_CANCELLED = auto()
    TASK_FAILED = auto()

    THINKNING_STARTED = auto()
    THINKING_COMPLETED = auto()
    TOOL_CALL_PARSED = auto()
    TOOL_VALIDATION_PASSED = auto()
    TOOL_VALIDATION_FAILED = auto()
    TOOL_EXECUTION_STARTED = auto()
    TOOL_EXECUTION_PROGRESS = auto()
    TOOL_EXECUTION_COMPLETED = auto()
    TOOL_EXECUTION_FAILED = auto()
    TOOL_RETRY_SCHEDULED = auto()
    TOOL_FALLBACK_TRIGGERED = auto()

    FEEDBACK_GENERATED = auto()
    LLM_CALL_STARTED = auto()
    LLM_CALL_COMPLETED = auto()
    LLM_STREAM_CHUNK = auto()

    CONTEXT_COMPRESSION = auto()
    MEMORY_RECALLED = auto()
    STATE_SNAPSHOT = auto()

    NEED_USER_INPUT = auto()
    NEED_APPROVAL = auto()
    PROGRESS_UPDATE = auto()
    FINAL_ANSWER = auto()


@dataclass
class LoopEvent:
    event_type:EventType
    task_id:str
    timestamp: float = 0.0
    tool_name:Optional[str] = None
    step_num:Optional[int] = None
    content:Optional[str] = None
    data:Optional[Dict[str,Any]] = None

    def to_dict(self):
        return {
            "event_type":self.event_type.name,
            "task_id":self.task_id,
            "timestamp":self.timestamp,
            "step_num": self.step_num,
            "tool_name":self.tool_name,
            "content":self.content,
            "data":self.data
        }
    
