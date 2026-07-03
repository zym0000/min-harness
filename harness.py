from llm_client import LLMClient
from tool_manger import ToolRegistry
from input_gateway import InputGateway
from agent_loop import RunResult,AgentLoop
from tool_execute_process.tool_metrics import MetricsControll
from tool_execute_process.error_classify import ToolErrorType
from task.task_manager import TaskManager
from async_execution_engine import AsyncExecutionEngine
from event.event import LoopEvent,EventType
from typing import Dict
import asyncio
import time

class Harness:
    def __init__(self, registry: ToolRegistry, llm: LLMClient,max_concurrent = 10):
        self.registry = registry
        self.llm = llm
        self.gateway = InputGateway()
        self.metices = MetricsControll()
        self.task_manager = TaskManager(max_history= 100)
        self.engine = AsyncExecutionEngine()
        self.cancel_event:Dict[str,asyncio.Event]= {}
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def submit_task(self, user_input: str) -> RunResult:
        tags = self.gateway.process(user_input)

        if tags:
            filtered_tools = self.registry.filter_by_tags(tags)
        else:   
            filtered_tools = self.registry.list_tools()

        task_id = await self.task_manager.create_task(user_input)
        cancel_event = asyncio.Event()
        self.cancel_event[task_id] = cancel_event

        loop = AgentLoop(self.registry, self.llm, 
                         self.task_manager,
                         self.engine,
                         cancel_event,
                         max_steps=5)
        
        async def _task_wrapper():
            async with self.semaphore:
                start_time = time.time()
                try:
                    async for event in loop.run(task_id,user_input, filtered_tools):
                        self._record_metrics(event)
                        yield event
                finally:
                    total_latency_ms = (time.time() - start_time) *1000
                    self.metices.record_task(total_latency_ms)
                    self.cancel_event.pop(task_id,None)

        return task_id, _task_wrapper()

    def _record_metrics(self,event:LoopEvent):
        if event.event_type == EventType.TOOL_EXECUTION_COMPLETED:
            latency_ms = event.data.get("latency_ms",0) if event.data else 0
            retry_count = event.data.get("retry_count",0) if event.data else 0

            self.metices.record_tool_call(
                event.tool_name,
                latency_ms=latency_ms,
                success=True,
                retry_count=retry_count)
        elif event.event_type == EventType.TOOL_EXECUTION_FAILED:
            latency_ms = event.data.get("latency_ms",0) if event.data else 0
            retry_count = event.data.get("retry_count",0) if event.data else 0
            error_type_name = event.data.get("event_type") if event.data else None
            error_type = ToolErrorType[error_type_name] if error_type_name else None

            self.metices.record_tool_call(
                tool_name=event.tool_name,
                latency_ms=latency_ms,
                success=False,
                retry_count=retry_count,
                error_type=error_type)

    def print_metrics(self):
        self.metices.print_summary()