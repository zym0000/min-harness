from llm_client import LLMClient
from tool_manger import ToolRegistry
from input_gateway import InputGateway
from agent_loop import RunResult,AgentLoop
from tool_execute_process.tool_metrics import MetricsControll
from tool_execute_process.error_classify import ToolErrorType
from task.task_manager import TaskManager,ContinueResult
from async_execution_engine import AsyncExecutionEngine
from event.event import LoopEvent,EventType
from typing import Dict
from context_message import ContextManager
from approval_gate import ApprovalGate
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
        self.context_manager = ContextManager(max_tokens=8000,reserve_tokens=2000)
        self.approval_grant: Dict[str, ApprovalGate] = {}

    async def submit_task(self, user_input: str) -> RunResult:
        tags = self.gateway.process(user_input)

        if tags:
            filtered_tools = self.registry.filter_by_tags(tags)
        else:   
            filtered_tools = self.registry.list_tools()
        
        system_prompt = self._build_system_prompt(filtered_tools)
        task_id = await self.task_manager.create_task(user_input,system_prompt)
        cancel_event = asyncio.Event()
        self.cancel_event[task_id] = cancel_event

        approval_gate = ApprovalGate()
        self.approval_grant[task_id] = approval_gate
        return task_id, await self._create_generator(task_id,user_input,filtered_tools,cancel_event,approval_gate)
    
    async def continue_task(self,task_id, user_input:str):
        result = await self.task_manager.continue_task(task_id,user_input)
        
        if result == ContinueResult.FAILED:
            return False, self._empty_generator()
        
        if result == ContinueResult.QUEUED:
            def _queue_gen():
                yield LoopEvent(
                    event_type=EventType.PROGRESS_UPDATE,
                    task_id=task_id,
                    timestamp=time.time(),
                    content="消息已加入排队，等待当前操作完成",
                )

            return True, await _queue_gen()
        
        if task_id not in self.cancel_event:
            self.cancel_event[task_id] = asyncio.Event()
        cancel_event = self.cancel_event[task_id]

        if task_id not in self.approval_grant:
            self.approval_grant[task_id] = ApprovalGate()
        approval_gate = self.approval_grant[task_id]

        tags = self.gateway.process(user_input)
        if tags:
            filter_tools = self.registry.filter_by_tags(tags)
        else:
            filter_tools = self.registry.list_tools()

        return True,await self._create_generator(task_id,user_input,filter_tools,cancel_event,approval_gate)
    
    async def _create_generator(self,
                                task_id:str,
                                user_input,
                                filtered_tools,
                                cancel_event:asyncio.Event,
                                approval_gate:asyncio.Event
                                ):
        
        loop = AgentLoop(self.registry, 
                         self.llm, 
                         self.task_manager,
                         self.engine,
                         self.context_manager,
                         approval_gate,
                         cancel_event,
                         max_steps=5)
        
        async def _task_wrapper():
            async with self.semaphore:
                start_time = time.time()
                try:
                    async for event in loop.run(task_id, filtered_tools):
                        self._record_metrics(event)
                        yield event
                finally:
                    total_latency_ms = (time.time() - start_time) *1000
                    self.metices.record_task(total_latency_ms)
                    self.cancel_event.pop(task_id,None)
                    self.approval_grant.pop(task_id,None)
        
        return _task_wrapper()
    
    def _empty_generator(self):
        def _gen():
            return
            yield
        return _gen()
    
    async def grant_approval(self, task_id):
        if task_id in self.approval_grant:
            self.approval_grant[task_id].approval()
    
    async def reject_approval(self,task_id):
        if task_id in self.approval_grant:
            self.approval_grant[task_id].reject()

    async def cancel_task(self,task_id):
        if task_id not in self.cancel_event:
            return False
        
        self.cancel_event[task_id].set()
        self.engine.cancel_execution(task_id)
        return True

    def _record_metrics(self,event:LoopEvent):
        if event.event_type == EventType.TOOL_EXECUTION_COMPLETED:
            latency_ms = event.data.get("latency_ms",0) if event.data else 0
            retry_count = event.data.get("retry_count",0) if event.data else 0

            self.metices.record_tool_call(
                event.tool_name or "UNKNOW",
                latency_ms=latency_ms,
                success=True,
                retry_count=retry_count)
        elif event.event_type == EventType.TOOL_EXECUTION_FAILED:
            latency_ms = event.data.get("latency_ms",0) if event.data else 0
            retry_count = event.data.get("retry_count",0) if event.data else 0
            error_type_name = event.data.get("event_type") if event.data else None
            error_type = ToolErrorType[error_type_name] if error_type_name else None

            self.metices.record_tool_call(
                tool_name=event.tool_name or "UNKNOW",
                latency_ms=latency_ms,
                success=False,
                retry_count=retry_count,
                error_type=error_type)

    def print_metrics(self):
        self.metices.print_summary()

    async def get_all_task_states(self):
        return await self.task_manager.get_all_tasks()

    def _build_system_prompt(self, filtered_tools) -> str:
        tools_desc = "\n\n".join([t.to_react_description() for t in filtered_tools])
        return f"""你是一个智能助手，可以使用以下工具来完成用户的任务。

可用工具：
{tools_desc}

---

输出规则：
1. 如果需要使用工具，请严格按以下格式输出：

Thought: 你的思考过程（为什么需要这个工具）
Action: 工具名
Action Input: {{"参数名": "参数值"}}

2. 如果不需要工具，直接回答用户问题。

3. 工具执行结果会返回给你，请基于结果给出最终回答。

4. 如果工具执行失败，请分析错误原因并尝试修正参数后重新调用。
"""