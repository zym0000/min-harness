import asyncio
import contextvars
import json
import time
from dataclasses import dataclass
from enum import auto, Enum
from typing import Any, Callable, Dict, List, Optional

from antioscillation_watchdog import AntioscillationWatchDog
#from interaction.approval_gate import ApprovalGate
from async_execution_engine import AsyncExecutionEngine
from event.event import EventType, LoopEvent
from llm_client import LLMClient
from task.task_manager import ContextManager, TaskManager, TaskStatus
from tool_execute_process.retry_engine import RetryEngine
from tool_manger import ToolCall, ToolCallParser

# Sub-agent sink:由 AgentLoop.run() 在 run 期间 set,subagent/runner.py 读。
# 放在 agent_loop.py 是因为 owner 是 AgentLoop;依赖方向是 runner → agent_loop,
# 不会形成循环 import。
EventSink = Callable[[LoopEvent], None]
_current_event_sink: contextvars.ContextVar[Optional[EventSink]] = contextvars.ContextVar(
    "agent_loop_subagent_event_sink",
    default=None,
)

class LoopState(Enum):
    IDLE = auto()
    THINKING = auto()
    PARSING = auto()
    VALIDATING = auto()
    ACTIVE = auto()
    OBSERVING = auto()
    APPROVAL_WAITING = auto()
    FINISHED = auto()
    MAX_STEPS_REACHED = auto()
    ERROR = auto()

@dataclass
class StepRecord:
    step_num: int
    state: LoopState
    timestamp: float
    llm_out: str = ""
    tool_call: Optional[Any] = None
    tool_result: Optional[Any] = None
    final_answer: str = ""

@dataclass
class LoopRunResult:
    """微观 ReAct 循环一次运行的结果"""
    input: str
    final_answer: str
    total_steps: int
    steps: List[StepRecord]
    final_state: LoopState
    execution_time_ms: float = 0.0
    total_cost_estimate: float = 0.0

RunResult = LoopRunResult

class AgentLoop:
    def __init__(self,
                 tool_registry: Any,
                 llm: Any,
                 task_manager: Any,
                 execution_engine: Any,
                 context_manager: Any,
                 approval: Any,
                 cancel_event: asyncio.Event,
                 max_steps: int = 10,
                 retry_engine: Optional[Any] = None,
                 watch_dog: Optional[AntioscillationWatchDog] = None,
                 llm_timeout: float = 120.0,
                 approval_timeout: float = 300.0,
                 max_consecutive_parse_errors: int = 3):

        self.tool_registry = tool_registry
        self.llm = llm
        self.cancel_event = cancel_event
        self.max_step = max_steps
        self.parser = ToolCallParser()
        self.retry_engine = retry_engine if retry_engine else RetryEngine()
        self.execute_engine = execution_engine
        self.task_manager = task_manager
        self.context_manager = context_manager
        self.approval = approval
        self.watch_dog = watch_dog
        self.llm_timeout = llm_timeout
        self.approval_timeout = approval_timeout
        self.max_consecutive_parse_errors = max_consecutive_parse_errors

        # 运行期状态
        self.state = LoopState.IDLE
        self._steps: List[StepRecord] = []
        self._run_started_at: float = 0.0
        self._final_answer: str = ""
        # 工具 schema 缓存
        self._schema_cache_key: Optional[tuple] = None
        self._schema_cache_value: Any = None
        # sub-agent 通过 _current_event_sink 把实时事件 push 到这里,run() 在迭代间隙 drain
        self._external_events: asyncio.Queue = asyncio.Queue()

    # 基础
    async def _check_cancelled(self):
        if self.cancel_event.is_set():
            raise asyncio.CancelledError()

    def _drain_external_events(self) -> List[LoopEvent]:
        """非阻塞清空 sub-agent 推过来的事件队列。空队列返回 []。"""
        out: List[LoopEvent] = []
        while True:
            try:
                out.append(self._external_events.get_nowait())
            except asyncio.QueueEmpty:
                break
        return out

    def _record_step(self, step_num: int, llm_out: str = "",
                     tool_call: Optional[Any] = None, tool_result: Optional[Any] = None,
                     final_answer: str = "") -> None:
        self._steps.append(StepRecord(
            step_num=step_num, state=self.state, timestamp=time.time(),
            llm_out=llm_out, tool_call=tool_call,
            tool_result=tool_result, final_answer=final_answer))

    def get_run_result(self, input_text: str = "") -> LoopRunResult:
        return LoopRunResult(
            input=input_text,
            final_answer=self._final_answer,
            total_steps=len(self._steps),
            steps=list(self._steps),
            final_state=self.state,
            execution_time_ms=(time.time() - self._run_started_at) * 1000
            if self._run_started_at else 0.0,
        )

    def _cached_tool_schema(self, tools: List[Any]) -> Any:
        key = tuple(getattr(t, "name", str(t)) for t in tools)
        if key != self._schema_cache_key:
            self._schema_cache_key = key
            self._schema_cache_value = self.tool_registry.to_openai_schema(tools)
        return self._schema_cache_value

    @staticmethod
    def _is_response_frame(item: Any) -> bool:
        """openai 流式返回最后一帧,才返回usage,tool_calls,如果对于其他模型,可能返回不一样，"""
        return hasattr(item, "text") and (
            hasattr(item, "usage") or hasattr(item, "tool_calls") or getattr(item, "reasoning_content", None))

    @staticmethod
    def _assistant_msg(llm_output: str, tool_calls: Optional[List[Dict]] = None,reasoning_content:Optional[str] = None) -> Dict:
        """原生协议下 assistant 消息必须携带 tool_calls(OpenAI 协议要求)。"""
        msg: Dict[str, Any] = {"role": "assistant", "content": llm_output}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if reasoning_content:
            msg["reasoning_content"] = reasoning_content
        return msg

    @staticmethod
    def _feedback_msg(content: str, tool_call_id: Optional[str]) -> Dict:
        """反馈消息：原生协议下 assistant 带 tool_calls 后
        必须紧跟 role="tool" 消息，否则下一轮请求被 OpenAI 拒绝（400）。"""
        if tool_call_id:
            return {"role": "tool", "tool_call_id": tool_call_id, "content": content}
        return {"role": "user", "content": content}

    async def run(self, task_id: str, tools: List[Any]):
        await self.task_manager.update_state(task_id, status=TaskStatus.RUNNING)
        self.state = LoopState.THINKING
        self._run_started_at = time.time()
        self._steps = []
        self._final_answer = ""
        yield LoopEvent(
            event_type=EventType.TASK_STARTED,
            task_id=task_id,
            timestamp=time.time(),
        )

        # 安装 sink:子代理在工具调用内部运行时,实时事件推到 self._external_events
        sink_token = _current_event_sink.set(self._external_events.put_nowait)
        final_status = TaskStatus.FAILED
        try:
            async for event in self._execute_steps(task_id, tools):
                # 先把子代理在上一段时间里产生的事件 yield 出去,再 yield 当前事件
                # 这样 CLI 看到的是"子代理做了一堆事 → sub_agent 完成"
                for ext in self._drain_external_events():
                    yield ext
                yield event

            # own loop 自然结束后,再 drain 一次残余事件(子代理最后一次的工具调用等)
            for ext in self._drain_external_events():
                yield ext

            final_status = TaskStatus.COMPLETED
        except asyncio.CancelledError:
            final_status = TaskStatus.CANCELLED
            self.state = LoopState.ERROR
            try:
                yield LoopEvent(
                    event_type=EventType.TASK_CANCELLED,
                    task_id=task_id,
                    timestamp=time.time()
                )
            finally:
                try:
                    await self.task_manager.update_state(task_id, status=TaskStatus.CANCELLED)
                except Exception:
                    pass
            raise
        except Exception as e:
            final_status = TaskStatus.FAILED
            self.state = LoopState.ERROR
            try:
                yield LoopEvent(
                    event_type=EventType.TASK_FAILED,
                    task_id=task_id,
                    timestamp=time.time(),
                    content=str(e)
                )
            finally:
                try:
                    await self.task_manager.update_state(task_id, status=TaskStatus.FAILED)
                except Exception:
                    pass
            raise
        finally:
            try:
                _current_event_sink.reset(sink_token)
            except Exception:
                pass

            if self.watch_dog:
                self.watch_dog.clear_task(task_id)

            state = await self.task_manager.get_state(task_id)
            if state is not None and state.task_status == TaskStatus.RUNNING:
                await self.task_manager.update_state(task_id=task_id, status=final_status)

    #步骤循环
    async def _next_chunk(self, agen: Any, deadline: float) -> Any:
        """逐帧竞速：下一帧 / 取消 / 超时。返回 None 表示流结束。"""
        next_task = asyncio.ensure_future(agen.__anext__())
        cancel_task = asyncio.ensure_future(self.cancel_event.wait())
        try:
            done, pending = await asyncio.wait(
                {next_task, cancel_task},
                timeout=max(0.01, deadline - time.monotonic()),
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            if cancel_task in done:
                raise asyncio.CancelledError()
            if next_task not in done:
                next_task.cancel()
                raise TimeoutError("llm stream 整体超时")
            try:
                return next_task.result()
            except StopAsyncIteration:
                return None
        finally:
            for t in (next_task, cancel_task):
                if not t.done():
                    t.cancel()

    async def _execute_steps(self, task_id: str, tools: List[Any]):
        consecutive_parse_errors = 0
        consecutive_llm_failures = 0

        for step_num in range(1, self.max_step + 1):
            await self.task_manager.update_state(task_id=task_id, current_step=step_num)
            await self._check_cancelled()

            # 处理挂起的用户输入
            pending = await self.task_manager.drain_pending_input(task_id)
            for user_input in pending:
                await self.task_manager.append_msg(task_id=task_id, messages=[
                    {"role": "user", "content": user_input}
                ])

            # 渐进式披露
            if hasattr(self.tool_registry, "get_disclosed_tools_for_task"):
                current_visible_tools = await self.tool_registry.get_disclosed_tools_for_task(
                    task_id
                )
            else:
                current_visible_tools = tools

            # 上下文压缩与 Token 统计 (进入 THINKING 状态)
            messages = await self.task_manager.compress_messages(
                task_id, self.llm, self.context_manager)
            state = await self.task_manager.get_state(task_id)

            # 口径：累计"每步实际发送给 LLM 的 prompt tokens 之和"（≈计费口径）；
            # 若 LLM 返回真实 usage，随后会用真实值修正
            step_input_tokens = self.context_manager.token_estimate.estimate_message(messages)
            await self.task_manager.update_state(
                task_id,
                total_tokens_used=state.total_tokens_used + step_input_tokens
            )

            self.state = LoopState.THINKING
            yield LoopEvent(
                event_type=EventType.THINKING_STARTED,
                task_id=task_id,
                timestamp=time.time(),
                step_num=step_num
            )

            #LLM 调用：流式优先，逐帧 timeout/cancel 竞速
            tool_schema = self._cached_tool_schema(current_visible_tools)
            llm_start_time = time.time()
            chunks: List[str] = []
            resp_usage: Optional[Dict[str, int]] = None
            resp_tool_calls: Optional[List[Dict]] = None
            llm_timed_out = False
            llm_error: Optional[str] = None
            reasoning_content:Optional[str] = ""

            if hasattr(self.llm, "chat_stream"):
                agen = None
                deadline = time.monotonic() + self.llm_timeout
                try:
                    agen = self.llm.chat_stream(messages, tool_schema).__aiter__()
                    while True:
                        try:
                            item = await self._next_chunk(agen, deadline)
                        except TimeoutError:
                            llm_timed_out = True
                            break
                        except Exception as e:
                            llm_error = repr(e)
                            break
                        if item is None:
                            break
                        #OpenAI 流式响应,最后一个chunk finish_reason 才会携带usage or tool_calls
                        if self._is_response_frame(item):
                            resp_usage = getattr(item, "usage", None)
                            resp_tool_calls = getattr(item, "tool_calls", None)
                            if getattr(item, "text", "") and not chunks:
                                chunks.append(item.text)
                            if getattr(item, "reasoning_content",None):
                                    reasoning_content += item.reasoning_content
                        else:
                            #中间过程
                            chunks.append(str(item))
                            yield LoopEvent(
                                event_type=EventType.THINKING_DELTA,
                                task_id=task_id,
                                timestamp=time.time(),
                                step_num=step_num,
                                content=str(item),
                            )
                finally:
                    aclose = getattr(agen, "aclose", None) if agen is not None else None
                    if aclose:
                        try:
                            await aclose()
                        except Exception:
                            pass
            else:
                # 客户端兜底：一次性 chat + 整体超时/取消竞速
                chat_task = asyncio.ensure_future(self.llm.chat(messages, tool_schema))
                cancel_task = asyncio.ensure_future(self.cancel_event.wait())
                done, pending = await asyncio.wait(
                    {chat_task, cancel_task},
                    timeout=self.llm_timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
                if cancel_task in done:
                    raise asyncio.CancelledError()
                if chat_task not in done:
                    llm_timed_out = True
                else:
                    try:
                        result = chat_task.result()
                    except Exception as e:
                        llm_error = repr(e)
                        result = None
                    #chat_task.result 返回 不一定是字符串，可能是一个对象
                    #这里统一使用chunks 保证整个LLM 流式或者非流式提取是自洽的
                    if result is not None:
                        if self._is_response_frame(result):
                            resp_usage = getattr(result, "usage", None)
                            resp_tool_calls = getattr(result, "tool_calls", None)
                            chunks.append(getattr(result, "text", ""))
                            reasoning_content = getattr(result,"reasoning_content",None)
                        else:
                            chunks.append(str(result))

            llm_latency = (time.time() - llm_start_time) * 1000

            if llm_timed_out or llm_error:
                consecutive_llm_failures += 1
                reason = (f"超时（>{self.llm_timeout}s）" if llm_timed_out
                          else f"异常：{llm_error}")
                await self.task_manager.append_msg(task_id=task_id, messages=[
                    {"role": "user", "content":
                     f"[System]: LLM 调用{reason}，请重试。"}
                ])
                if consecutive_llm_failures >= self.max_consecutive_parse_errors:
                    yield LoopEvent(
                        event_type=EventType.TASK_FAILED,
                        task_id=task_id,
                        timestamp=time.time(),
                        step_num=step_num,
                        content=f"LLM 连续 {consecutive_llm_failures} 次调用失败/超时",
                        data={"latency_ms": round(llm_latency, 2)},
                    )
                    await self.task_manager.update_state(task_id, status=TaskStatus.FAILED)
                    self.state = LoopState.ERROR
                    return
                continue
            consecutive_llm_failures = 0

            #所有输出拼接
            llm_output = "".join(chunks)

            #token 统计：真实 usage 优先
            #真实LLM 会返回输出和输出字符数，那么在计算的时候，不能直接使用prompt_tokens + completion_tokens
            #因为输入在上面已经计算了
            if resp_usage:
                real_input = resp_usage.get("prompt_tokens", step_input_tokens)
                step_output_tokens = resp_usage.get("completion_tokens", 0)
                #估算误差，少算加，多算减，减去之前算的
                adjust = real_input - step_input_tokens
            else:
                step_output_tokens = self.context_manager.token_estimate.estimate(llm_output)
                adjust = 0
            state = await self.task_manager.get_state(task_id)
            await self.task_manager.update_state(
                task_id,
                total_tokens_used=state.total_tokens_used + step_output_tokens + adjust
            )

            yield LoopEvent(
                event_type=EventType.THINKING_COMPLETED,
                task_id=task_id,
                timestamp=time.time(),
                step_num=step_num,
                content=llm_output,
                data={"latency_ms": round(llm_latency, 2),
                      "input_tokens": step_input_tokens + adjust,
                      "output_tokens": step_output_tokens,
                      "usage_source": "real" if resp_usage else "estimate"},
            )

            self.state = LoopState.PARSING
            tool_call_id: Optional[str] = None
            extra_ignored = 0
            #如果是走结构返回的tool 调用，进行规则解析
            if resp_tool_calls:
                #本循环一次只执行一个工具调用，多余截断并在事件中标注
                extra_ignored = max(0, len(resp_tool_calls) - 1)
                resp_tool_calls = resp_tool_calls[:1]
                first = resp_tool_calls[0]
                function = first.get("function") or {}
                tool_call_id = first.get("id")
                # 真实 OpenAI 的 arguments 是 JSON 字符串，需解析兜底
                raw_args = function.get("arguments") or {}
                parse_error = None

                if isinstance(raw_args, str):
                    try:
                        raw_args = json.loads(raw_args) if raw_args.strip() else {}
                    except Exception as e:
                        raw_args = {}
                        parse_error = f"原生 tool_calls 参数 JSON 解析失败: {e}"
                
                if parse_error is None and not isinstance(raw_args, dict):
                    parse_error = f"原生 tool_calls 参数不是对象: {type(raw_args).__name__}"
                    raw_args = {}
                
                if parse_error is None and not function.get("name"):
                    parse_error = "原生 tool_calls 缺少工具名"
                
                if parse_error is None:
                    tool_call = ToolCall(
                        tool_name=function["name"],
                        arguments=raw_args,
                        raw_text=llm_output,
                        thought="",
                    )
                else:
                    tool_call = None
            else:
                #如果不是规则返回的，走ReAct 文本解析
                tool_call, parse_error = self.parser.parse(llm_output)

            if parse_error:
                consecutive_parse_errors += 1
                yield LoopEvent(
                    event_type=EventType.TOOL_VALIDATION_FAILED,
                    task_id=task_id,
                    timestamp=time.time(),
                    step_num=step_num,
                    content=parse_error
                )

                await self.task_manager.append_msg(task_id=task_id, messages=[
                    self._assistant_msg(llm_output, resp_tool_calls,reasoning_content=reasoning_content),
                    self._feedback_msg(f"[Parse Error]: {parse_error}", tool_call_id)
                ])
                if consecutive_parse_errors >= self.max_consecutive_parse_errors:
                    yield LoopEvent(
                        event_type=EventType.TASK_FAILED,
                        task_id=task_id,
                        timestamp=time.time(),
                        step_num=step_num,
                        content=f"连续 {consecutive_parse_errors} 次解析失败，终止任务",
                    )
                    await self.task_manager.update_state(task_id, status=TaskStatus.FAILED)
                    self.state = LoopState.ERROR
                    self._record_step(step_num, llm_out=llm_output)
                    return
                self._record_step(step_num, llm_out=llm_output)
                continue
            consecutive_parse_errors = 0

            # 如果没有工具调用，说明是最终回答 (FINISHED)
            if tool_call is None:
                self.state = LoopState.FINISHED
                self._final_answer = llm_output
                yield LoopEvent(
                    event_type=EventType.FINAL_ANSWER,
                    task_id=task_id,
                    timestamp=time.time(),
                    step_num=step_num,
                    content=llm_output
                )
                yield LoopEvent(
                    event_type=EventType.TASK_COMPLETED,
                    task_id=task_id,
                    timestamp=time.time()
                )
                await self.task_manager.update_state(task_id, status=TaskStatus.COMPLETED)
                await self.task_manager.append_msg(task_id=task_id, messages=[
                    self._assistant_msg(llm_output,reasoning_content=reasoning_content),
                ])
                self._record_step(step_num, llm_out=llm_output, final_answer=llm_output)
                return

            #反振荡看门狗（在验证与执行之前记录
            if self.watch_dog:
                level, msg = await self.watch_dog.record_and_check(task_id, tool_call)
                if level == "WARNING":
                    warning_msg = f"[SYSTEM  INTERVENTION] {msg}"
                    await self.task_manager.append_msg(task_id=task_id, messages=[
                        self._assistant_msg(llm_output, resp_tool_calls,reasoning_content=reasoning_content),
                        self._feedback_msg(warning_msg, tool_call_id),
                    ])
                    self._record_step(step_num, llm_out=llm_output, tool_call=tool_call)
                    continue
                elif level == "CRITICAL":
                    yield LoopEvent(
                        event_type=EventType.TASK_FAILED,
                        task_id=task_id,
                        tool_name=tool_call.tool_name,
                        timestamp=time.time(),
                        content=msg
                    )
                    await self.task_manager.update_state(task_id, status=TaskStatus.FAILED)
                    self.state = LoopState.ERROR
                    self._record_step(step_num, llm_out=llm_output, tool_call=tool_call)
                    return

            # 进入工具验证阶段 (VALIDATING)
            self.state = LoopState.VALIDATING
            yield LoopEvent(
                event_type=EventType.TOOL_CALL_PARSED,
                task_id=task_id,
                timestamp=time.time(),
                step_num=step_num,
                tool_name=tool_call.tool_name,
                data={"arguments": tool_call.arguments,
                      "ignored_extra_tool_calls": extra_ignored}
            )

            validate_error = self.tool_registry.validate_call(tool_call)
            if validate_error:
                yield LoopEvent(
                    event_type=EventType.TOOL_VALIDATION_FAILED,
                    task_id=task_id,
                    timestamp=time.time(),
                    step_num=step_num,
                    tool_name=tool_call.tool_name,
                    content=validate_error,
                )

                await self.task_manager.append_msg(task_id=task_id, messages=[
                    self._assistant_msg(llm_output, resp_tool_calls,reasoning_content=reasoning_content),
                    self._feedback_msg(f"Observation [Tool Error]: {validate_error}", tool_call_id)
                ])
                self._record_step(step_num, llm_out=llm_output, tool_call=tool_call)
                continue

            yield LoopEvent(
                event_type=EventType.TOOL_VALIDATION_PASSED,
                task_id=task_id,
                timestamp=time.time(),
                step_num=step_num,
                tool_name=tool_call.tool_name
            )

            # 不同 registry 的 get 语义不同（返回 None 或抛 KeyError），两者都防御
            try:
                tool = self.tool_registry.get(tool_call.tool_name)
            except KeyError:
                tool = None
            if tool is None:
                yield LoopEvent(
                    event_type=EventType.TOOL_VALIDATION_FAILED,
                    task_id=task_id,
                    timestamp=time.time(),
                    step_num=step_num,
                    tool_name=tool_call.tool_name,
                    content=f"Tool '{tool_call.tool_name}' not found in registry",
                )
                await self.task_manager.append_msg(task_id=task_id, messages=[
                    self._assistant_msg(llm_output, resp_tool_calls,reasoning_content=reasoning_content),
self._feedback_msg(f"Observation [Tool Error]: 工具 {tool_call.tool_name} 不存在", tool_call_id)
                ])
                self._record_step(step_num, llm_out=llm_output, tool_call=tool_call)
                continue

            # 人工干预卡点管理 (Human-in-the-loop)
            if getattr(tool, "dangerous", False):
                await self.task_manager.update_state(task_id, status=TaskStatus.PAUSED)
                self.state = LoopState.APPROVAL_WAITING

                # reset 必须在 yield 之前，否则事件与恢复之间的批准会被抹掉
                self.approval.reset()

                yield LoopEvent(
                    event_type=EventType.NEED_APPROVAL,
                    task_id=task_id,
                    tool_name=tool.name,
                    timestamp=time.time(),
                    content=f"Whether to execute the tool {tool.name}",
                    data={"arguments": tool_call.arguments},
                )

                # 等待审批（可取消 + 超时视为拒绝）
                cancel_task = asyncio.create_task(self.cancel_event.wait())
                approval_task = asyncio.create_task(
                    asyncio.wait_for(self.approval.wait(), timeout=self.approval_timeout)
                )

                done, pending_tasks = await asyncio.wait(
                    [cancel_task, approval_task],
                    return_when=asyncio.FIRST_COMPLETED
                )

                for t in pending_tasks:
                    t.cancel()
                    try:
                        await t
                    except (asyncio.CancelledError, Exception):
                        pass

                await self.task_manager.update_state(task_id, status=TaskStatus.RUNNING)

                # 取消优先于审批
                if cancel_task in done:
                    raise asyncio.CancelledError()

                approved = False
                try:
                    approved = approval_task.result()
                except asyncio.TimeoutError:
                    approved = False        # 审批超时 = 拒绝
                except Exception as e:
                    yield LoopEvent(
                        event_type=EventType.TOOL_EXECUTION_FAILED,
                        task_id=task_id,
                        timestamp=time.time(),
                        content=f"Approval system error: {str(e)}"
                    )
                    self._record_step(step_num, llm_out=llm_output, tool_call=tool_call)
                    continue

                if not approved:
                    yield LoopEvent(
                        event_type=EventType.APPROVAL_DENIED,
                        task_id=task_id,
                        timestamp=time.time(),
                        content="User refused to execute or timed out",
                    )

                    await self.task_manager.append_msg(task_id=task_id, messages=[
                        self._assistant_msg(llm_output, resp_tool_calls,reasoning_content=reasoning_content),
                        self._feedback_msg(f"[Approval Denied] {tool.name} rejected", tool_call_id)
                    ])
                    self._record_step(step_num, llm_out=llm_output, tool_call=tool_call)
                    continue

                yield LoopEvent(
                    event_type=EventType.APPROVAL_GRANTED,
                    task_id=task_id,
                    timestamp=time.time(),
                    tool_name=tool.name
                )

            # 工具执行 (ACTIVE)
            self.state = LoopState.ACTIVE
            yield LoopEvent(
                event_type=EventType.TOOL_EXECUTION_STARTED,
                task_id=task_id,
                tool_name=tool.name,
                timestamp=time.time(),
                data={"arguments": tool_call.arguments}
            )

            result, tool_latency_ms = await self._execute_with_retry(task_id, tool, tool_call)

            if result.is_error:
                yield LoopEvent(
                    event_type=EventType.TOOL_EXECUTION_FAILED,
                    task_id=task_id,
                    tool_name=tool.name,
                    timestamp=time.time(),
                    error=result.error,
                    data={
                        "error_type": result.error_type,
                        "retry_count": result.retry_count,
                        "latency_ms": round(tool_latency_ms, 2),
                    }
                )
            else:
                yield LoopEvent(
                    event_type=EventType.TOOL_EXECUTION_COMPLETED,
                    task_id=task_id,
                    timestamp=time.time(),
                    tool_name=tool.name,
                    content=result.to_text()[:500],   # 截断预览，全量在任务消息里
                    data={
                        "latency_ms": round(tool_latency_ms, 2),
                        "retry_count": result.retry_count,
                    },
                )

            # sub-agent 事件回放:子代理返回的 ToolResult.data["events"] 包含完整子代理事件流
            # 这里 post-hoc yield 出去,让父的 event stream / CLI 能看到子代理完整执行过程
            # 实时错误/心跳已通过 sink 路径推到 _external_events(drain 时已 yield)
            # 这里只补齐 sink 没覆盖的部分(THINKING_*, TOOL_EXECUTION_*, FINAL_ANSWER 等)
            replayed = (result.data or {}).get("events") if result.data else None
            if replayed:
                for sub_event in replayed:
                    # 不重 yield sub_agent 工具自己的 COMPLETED(就是上面那条),
                    # 避免重复。其它子代理事件都 forward
                    if sub_event.tool_name == "sub_agent":
                        continue
                    yield sub_event

            # 状态转移（仅"转移被拒绝"才跳过 Observation）
            if not result.is_error and hasattr(self.tool_registry, "transit_task_skill_state"):
                error_msg = await self.tool_registry.transit_task_skill_state(
                    task_id=task_id,
                    target_stage=tool_call.tool_name
                )
                if error_msg:
                    await self.task_manager.append_msg(
                        task_id=task_id,
                        messages=[
                            self._assistant_msg(llm_output, resp_tool_calls,reasoning_content=reasoning_content),
                            self._feedback_msg(f"Observation [stage transit Rejected]: {error_msg}", tool_call_id)
                        ]
                    )
                    self._record_step(step_num, llm_out=llm_output,
                                      tool_call=tool_call, tool_result=result)
                    continue

            self.state = LoopState.OBSERVING
            # 原生协议：role="tool" + tool_call_id；ReAct：user 角色的 Observation
            if tool_call_id:
                observation_msg = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result.to_text(),
                }
            else:
                observation_msg = {
                    "role": "user",
                    "content": f"Observation: {result.to_text()}",
                }
            await self.task_manager.append_msg(task_id=task_id, messages=[
                self._assistant_msg(llm_output, resp_tool_calls,reasoning_content=reasoning_content),
                observation_msg,
            ])
            self._record_step(step_num, llm_out=llm_output,
                              tool_call=tool_call, tool_result=result)

        self.state = LoopState.MAX_STEPS_REACHED
        yield LoopEvent(
            event_type=EventType.TASK_FAILED,
            task_id=task_id,
            timestamp=time.time(),
            content="The maximum step limit has been reached",
        )
        await self.task_manager.update_state(task_id, status=TaskStatus.FAILED)
        return

    async def _execute_with_retry(self, task_id: str, tool: Any, call: Any):
        last_result = None
        total_latency_ms = 0.0

        for attempt in range(self.retry_engine.policy.max_retries + 1):
            await self._check_cancelled()

            start_time = time.perf_counter()
            # 透传 tool.timeout(None 表示走 engine 默认 30s)
            tool_timeout = getattr(tool, "timeout", None)
            result = await self.execute_engine.execute(
                task_id, tool, call.arguments, timeout=tool_timeout)
            end_time = time.perf_counter()

            total_latency_ms += (end_time - start_time) * 1000

            if not result.is_error:
                result.retry_count = attempt
                return result, total_latency_ms

            last_result = result

            # 只要错误不可重试，立即熔断打破循环，不浪费后面的尝试
            if not result.is_retryable:
                break

            if attempt == self.retry_engine.policy.max_retries:
                break

            delay = self.retry_engine.calc_delay(attempt, result.error_type)
            # 可取消的 backoff：cancel_event 置位时立即醒来
            try:
                await asyncio.wait_for(self.cancel_event.wait(), timeout=delay)
                raise asyncio.CancelledError()      # event 被置位 -> 取消
            except asyncio.TimeoutError:
                pass                                # 正常睡满 delay，进入下一次尝试

        return last_result, total_latency_ms