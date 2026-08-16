"""Sub-agent 运行状态。独立于主 TaskManager,不污染主任务列表,不写 SQLite。"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class SubAgentState:
    """单个 sub-agent 实例的运行快照。

    由 run_sub_agent 创建,运行结束后填充 finished_at / final_state /
    final_answer / total_steps。状态仅在内存,重启后丢失——sub-agent 是
    "临时工具调用"语义,不是持久任务。
    """
    subagent_id: str
    parent_task_id: str
    created_at: float
    finished_at: Optional[float] = None
    final_answer: str = ""
    final_state: str = ""           # "COMPLETED" / "MAX_STEPS_REACHED" / "CANCELLED" / "ERROR"
    total_steps: int = 0
    depth: int = 0