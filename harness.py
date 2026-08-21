import asyncio
import time
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional

from agent_loop import AgentLoop, LoopRunResult
from antioscillation_watchdog import AntioscillationWatchDog
from interaction.approval_gate import ApprovalGate
from async_execution_engine import AsyncExecutionEngine
from context_message import ContextManager
from event.event import EventType, LoopEvent
from input_gateway import InputGateway
from llm_client import LLMClient
from task.task_manager import ContinueResult, TaskManager
from task.task_store import SQLiteTaskStore, TaskStore
from tool_execute_process.error_classify import ToolErrorType
from tool_execute_process.tool_metrics import MetricsControll
from tool_manger import Tool, ToolRegistry

@dataclass
class HarnessConfig:
    """Harness 全部可调参数的统一入口。"""
    max_concurrent: int = 10                 # 全局并发任务数（信号量）
    max_steps: int = 200                      # 单任务 ReAct 最大步数
    llm_timeout: float = 120.0               # 单次 LLM 调用超时（秒，流式为逐帧竞速的总预算）
    approval_timeout: float = 300.0          # 审批等待超时（秒）
    max_consecutive_parse_errors: int = 3    # 连续解析/LLM 失败上限，超限终止任务
    enable_watchdog: bool = True             # 反振荡看门狗
    use_embeddings: bool = False             # 意图识别用向量模型（需下载权重）
    context_max_tokens: int = 12000           # 上下文窗口预算
    context_reserve_tokens: int = 2000       # 给回答预留的 token
    recent_messages_token: int = 4000       # recent 区 token 水位线
    min_old_token_for_extract: int = 1500   # 旧区累计触发提取的 token 阈值
    task_max_history: int = 100              # 单任务消息历史上限
    store_path: Optional[str] = None         # SQLite 检查点路径；None=纯内存
    #sub agent
    max_subagent_depth:int = 3               #sub agent 最大深度
    #max_subagent_prompt_chars: int = 0      #sub agent prompt 最大字符数
    subagent_heartbeat_interval: float = 2.0  #sub agent 心跳时常
    max_subagent_events: int = 5000          #sub agent 事件最大数量
    max_subagent_states: int = 1000          # sub agent 最大状态
    subagent_tool_timeout: float = 1800.0    # sub_agent 工具调用超时秒数(默认 30min,覆盖 engine 默认 30s)
    
class Harness:
    def __init__(self,
                 registry: ToolRegistry,
                 llm: LLMClient,
                 config: Optional[HarnessConfig] = None,
                 context_manager: Optional[ContextManager] = None,
                 store: Optional[TaskStore] = None):
        self.registry = registry
        self.llm = llm

        # config 为基线，显式传入的旧 kwargs 覆盖对应字段。
        # replace 复制，不原地修改调用方共享的 config 对象
        cfg = replace(config) if config is not None else HarnessConfig()
        self.config = cfg

        self.gateway = InputGateway()
        self.metrics = MetricsControll()

        #持久化接线。显式 store 优先，如果为空 使用store_path 创建建 SQLite。
        self.store: Optional[TaskStore] = store
        if self.store is None and cfg.store_path:
            self.store = SQLiteTaskStore(cfg.store_path)
        self.task_manager = TaskManager(
            max_history=cfg.task_max_history, store=self.store)

        self.engine = AsyncExecutionEngine()
        self.cancel_events: Dict[str, asyncio.Event] = {}
        self.semaphore = asyncio.Semaphore(cfg.max_concurrent)
        self.context_manager = context_manager or ContextManager(
            max_tokens=cfg.context_max_tokens,
            reserve_tokens=cfg.context_reserve_tokens,
            recent_messages_token=cfg.recent_messages_token,
            min_old_token_for_extract=cfg.min_old_token_for_extract)
        self.approval_gates: Dict[str, ApprovalGate] = {}
        self.watch_dog = AntioscillationWatchDog() if cfg.enable_watchdog else None

        #运行结果缓存（任务结束后仍可查）
        self._loops: Dict[str, AgentLoop] = {}
        self._run_results: Dict[str, LoopRunResult] = {}

        # sub-agent 集成所需字段(被 subagent/runner.py 读取)
        # _loop_tools: task_id -> 当前 AgentLoop 看到的工具集,sub-agent 用此继承父的工具范围
        self._loop_tools: Dict[str, List[Tool]] = {}
        # _subagent_states: subagent_id -> 状态对象
        self._subagent_states: Dict[str, Any] = {}

    #任务提交
    async def submit_task(self, user_input: str):
        """返回 (task_id, LoopEvent 异步事件流)。"""
        filtered_tools = self._filter_tools(user_input)
        system_prompt = self._build_system_prompt(filtered_tools)
        task_id = await self.task_manager.create_task(user_input, system_prompt)

        cancel_event = asyncio.Event()
        self.cancel_events[task_id] = cancel_event
        approval_gate = ApprovalGate()
        self.approval_gates[task_id] = approval_gate

        return task_id, self._create_generator(
            task_id, user_input, filtered_tools, cancel_event, approval_gate)

    async def continue_task(self, task_id: str, user_input: str):
        result = await self.task_manager.continue_task(task_id, user_input)

        if result == ContinueResult.FAILED:
            return False, self._empty_generator()

        if result == ContinueResult.QUEUED:
            return True, self._queue_generator(task_id)

        # ACTIVATED
        cancel_event = self.cancel_events.setdefault(task_id, asyncio.Event())
        cancel_event.clear()
        approval_gate = self.approval_gates.setdefault(task_id, ApprovalGate())

        filtered_tools = self._filter_tools(user_input)
        return True, self._create_generator(
            task_id, user_input, filtered_tools, cancel_event, approval_gate)

    #生成器
    def _create_generator(self,
                          task_id: str,
                          user_input: str,
                          filtered_tools: List[Tool],
                          cancel_event: asyncio.Event,
                          approval_gate: ApprovalGate):
        loop = AgentLoop(
            self.registry,
            self.llm,
            self.task_manager,
            self.engine,
            self.context_manager,
            approval_gate,
            cancel_event,
            max_steps=self.config.max_steps,
            watch_dog=self.watch_dog,
            llm_timeout=self.config.llm_timeout,
            approval_timeout=self.config.approval_timeout,
            max_consecutive_parse_errors=self.config.max_consecutive_parse_errors,
        )
        self._loops[task_id] = loop
        # sub-agent 通过 harness._loop_tools[parent_task_id] 拿到父任务的工具集
        self._loop_tools[task_id] = list(filtered_tools)
        trace_id = task_id

        async def _task_wrapper():
            # 手动管理信号量——审批等待期间释放并发额度
            start_time = time.time()
            holds_sem = False
            try:
                await self.semaphore.acquire()
                holds_sem = True
                async for event in loop.run(task_id, filtered_tools):
                    if holds_sem and event.event_type == EventType.NEED_APPROVAL:
                        self.semaphore.release()          # 审批等待不占额度
                        holds_sem = False
                    elif not holds_sem and event.event_type in (
                            EventType.APPROVAL_GRANTED, EventType.APPROVAL_DENIED,
                            EventType.TOOL_EXECUTION_FAILED):
                        # TOOL_EXECUTION_FAILED 在此分支只可能来自审批系统异常路径
                        await self.semaphore.acquire()    # 审批结束恢复执行，重新占位
                        holds_sem = True
                    # 统一回填 trace_id（事件自带则尊重自带值）
                    if not event.trace_id:
                        event.trace_id = trace_id
                    self._record_metrics(event)
                    yield event
            finally:
                if holds_sem:
                    self.semaphore.release()
                total_latency_ms = (time.time() - start_time) * 1000
                self.metrics.record_task(total_latency_ms)
                # 缓存 RunResult 供事后查询
                self._run_results[task_id] = loop.get_run_result(user_input)
                self._loops.pop(task_id, None)
                self._loop_tools.pop(task_id, None)
                self.cancel_events.pop(task_id, None)
                self.approval_gates.pop(task_id, None)

        return _task_wrapper()

    async def _empty_generator(self):
        return
        yield

    async def _queue_generator(self, task_id: str):
        yield LoopEvent(
            event_type=EventType.PROGRESS_UPDATE,
            task_id=task_id,
            timestamp=time.time(),
            content="消息已加入排队，等待当前操作完成",
            trace_id=task_id,
        )

    # 意图过滤
    def _filter_tools(self, user_input: str) -> List[Tool]:
        """tags 命中 general 通用工具；并集为空兜底全量。"""
        tags = self.gateway.process(user_input)
        if not tags:
            return self.registry.list_tools()
        matched = {t.name: t for t in self.registry.filter_by_tags(tags)}
        for t in self.registry.filter_by_tags(["general"]):
            matched.setdefault(t.name, t)
        return list(matched.values()) or self.registry.list_tools()

    # 持久化
    async def restore_tasks(self) -> int:
        """重启后从 store 恢复全部任务，返回恢复数量。"""
        return await self.task_manager.restore()

    #结果查询
    def get_run_result(self, task_id: str) -> Optional[LoopRunResult]:
        """运行结束后的汇总结果；任务不存在或未结束返回 None。"""
        return self._run_results.get(task_id)

    def get_subagent_states_for_parent(self, parent_task_id: str) -> List[Any]:
        """返回指定父任务派生的所有 sub-agent 状态(按创建时间排序)。"""
        states = [
            s for s in self._subagent_states.values()
            if s.parent_task_id == parent_task_id
        ]
        states.sort(key=lambda s: s.created_at)
        return states

    # 控制面
    async def grant_approval(self, task_id: str):
        gate = self.approval_gates.get(task_id)
        if gate:
            gate.approve()

    async def reject_approval(self, task_id: str):
        gate = self.approval_gates.get(task_id)
        if gate:
            gate.reject()

    async def cancel_task(self, task_id: str) -> bool:
        event = self.cancel_events.get(task_id)
        if event is None:
            return False
        event.set()
        self.engine.cancel_execution(task_id)
        return True

    async def clear_task(self, task_id: str) -> bool:
        """清空 task 全部状态(含 SQLite + harness 级缓存 + 派生 sub-agent)。

        若 loop 仍在跑,先 cancel 并等待退出再清理。
        返回 True 表示清理发生,False 表示 task 不存在。
        """
        if task_id not in self.task_manager.tasks:
            return False

        await self.cancel_task(task_id)

        for _ in range(50):
            if task_id not in self._loops:
                break
            await asyncio.sleep(0.1)

        await self.task_manager.delete_task(task_id)

        self.cancel_events.pop(task_id, None)
        self.approval_gates.pop(task_id, None)
        self._loops.pop(task_id, None)
        self._loop_tools.pop(task_id, None)
        self._run_results.pop(task_id, None)

        stale = [
            sid for sid, s in self._subagent_states.items()
            if getattr(s, "parent_task_id", None) == task_id
        ]
        for sid in stale:
            self._subagent_states.pop(sid, None)

        return True

    async def shutdown(self) -> None:
        """统一关停:engine -> LLM client -> store。幂等。"""
        await self.engine.shutdown()

        aclose = getattr(self.llm, "aclose", None)
        if callable(aclose):
            try:
                await aclose()
            except Exception:
                pass

        if self.store is not None and hasattr(self.store, "close"):
            try:
                await asyncio.to_thread(self.store.close)
            except Exception:
                pass

    #指标
    def _record_metrics(self, event: LoopEvent):
        # sub-agent 事件不计入主任务 metrics——避免子代理的工具调用 / LLM 计数污染父任务
        # 父 LLM 看到的最终 ToolResult 走标准路径(已计入),中间事件不计
        if event.data and event.data.get("subagent_id"):
            return

        if event.event_type == EventType.TOOL_EXECUTION_COMPLETED:
            latency_ms = event.data.get("latency_ms", 0) if event.data else 0
            retry_count = event.data.get("retry_count", 0) if event.data else 0
            self.metrics.record_tool_call(
                event.tool_name or "UNKNOWN",
                latency_ms=latency_ms,
                success=True,
                retry_count=retry_count)

        elif event.event_type == EventType.TOOL_EXECUTION_FAILED:
            latency_ms = event.data.get("latency_ms", 0) if event.data else 0
            retry_count = event.data.get("retry_count", 0) if event.data else 0
            raw = event.data.get("error_type") if event.data else None
            if isinstance(raw, ToolErrorType):
                error_type = raw
            elif isinstance(raw, str) and raw in ToolErrorType.__members__:
                error_type = ToolErrorType[raw]
            else:
                error_type = None
            self.metrics.record_tool_call(
                event.tool_name or "UNKNOWN",
                latency_ms=latency_ms,
                success=False,
                retry_count=retry_count,
                error_type=error_type)

        elif event.event_type == EventType.THINKING_COMPLETED:
            if event.data:
                self.metrics.record_llm_call(
                    event.data.get("input_tokens", 0) + event.data.get("output_tokens", 0))
            else:
                self.metrics.record_llm_call(0)

    def get_metrics_summary(self) -> Dict[str, Any]:
        return self.metrics.get_summary()

    def print_metrics(self):
        self.metrics.print_summary()

    async def get_all_task_states(self):
        return await self.task_manager.get_all_tasks()

    #prompt

    def _build_system_prompt(self, filtered_tools: List[Tool]) -> str:
        # 走 OpenAI 原生 tools 协议:工具的 schema 通过 API tools 参数传入,
        # 不再让 LLM 在文本里手写 Thought/Action/Action Input,也不嵌入工具描述。
        # 这样 LLM 直接返回结构化 tool_calls,parser 路径不再误判"final answer"。
        tool_names = ", ".join(t.name for t in filtered_tools)
        return f"""你是一个智能助手,可以使用工具帮助用户完成任务。

工作方式:
- 任务需要信息或操作时,直接调用工具(工具的 schema 已经在 tools 参数中提供)
- 一次只能调用一个工具,需要多个就多轮调用
- 工具调用失败时,根据错误信息调整参数或换工具重试
- 任务完成后,用自然语言简洁地总结结果给用户
- 不需要工具的问题(如闲聊、解释概念、问答),直接回答即可

当前任务可用工具:{tool_names}
"""