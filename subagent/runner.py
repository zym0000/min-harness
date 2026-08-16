"""
Sub-agent runner: 实现 sub_agent 工具的 Tool.func,内部启动独立 AgentLoop。
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Set

from agent_loop import AgentLoop, EventSink, _current_event_sink
from event.event import EventType, LoopEvent
from subagent._contextvars import _current_context, _current_depth
from subagent.state import SubAgentState
from task.task_manager import TaskManager, TaskStatus
from tool_manger import Tool, ToolParameter, ToolResult

if TYPE_CHECKING:
    from harness import Harness

logger = logging.getLogger(__name__)

_SUB_AGENT_TOOL_NAME = "sub_agent"

@dataclass
class _SubAgentContext:
    """
    当前 sub-agent 上下文
    """

    task_id: str
    subagent_id: str
    depth: int
    tools: List[Tool]
    approval_gate: Any

def _build_realtime_event_types() -> Set[Any]:
    """
    构造需要实时转发的事件类型集合。

    不同版本 EventType 命名可能不同，这里做兼容，避免模块导入时 AttributeError。
    """
    candidate_names = {
        # 审批
        "APPROVAL_REQUIRED",
        "APPROVAL_REQUEST",
        "APPROVAL_GRANTED",
        "APPROVAL_DENIED",
        "APPROVAL_RESPONSE",
        "APPROVAL_RESULT",

        # 进度 / 心跳
        "PROGRESS_UPDATE",

        # 关键错误
        "TOOL_EXECUTION_FAILED",
        "LOOP_ERROR",
        "ERROR",
    }

    result: Set[Any] = set()
    for name in candidate_names:
        member = getattr(EventType, name, None)
        if member is not None:
            result.add(member)

    return result


_REALTIME_EVENT_TYPES: Set[Any] = _build_realtime_event_types()

_PROGRESS_UPDATE_EVENT = getattr(EventType, "PROGRESS_UPDATE", None)

_ERROR_EVENT_TYPE = (
    getattr(EventType, "LOOP_ERROR", None)
    or getattr(EventType, "TOOL_EXECUTION_FAILED", None)
    or getattr(EventType, "ERROR", None)
    or _PROGRESS_UPDATE_EVENT
)


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def _cfg(harness: "Harness", name: str, default: Any) -> Any:
    """从 harness.config 读取配置，容忍缺失。"""
    config = getattr(harness, "config", None)
    if config is None:
        return default
    return getattr(config, name, default)

def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default

def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default

def _normalize_max_steps(value: Any, default: int) -> int:
    if value is None:
        return default

    try:
        v = int(value)
    except Exception:
        return default

    if v <= 0:
        return default

    return v

def _safe_emit(sink: Optional[EventSink], event: LoopEvent) -> None:
    """
    安全调用 event sink。
    """
    if sink is None:
        return

    try:
        sink(event)
    except Exception:
        logger.exception("sub-agent event sink failed")


async def _cancel_task(task: Optional[asyncio.Task]) -> None:
    """安全取消并等待 task。"""
    if task is None:
        return

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception("failed to cancel sub-agent background task")


def _should_forward_realtime(event: LoopEvent) -> bool:
    """是否需要实时转发。"""
    return event.event_type in _REALTIME_EVENT_TYPES

def _describe_tool(tool: Tool) -> str:
    """安全生成工具描述。"""
    name = getattr(tool, "name", "unknown_tool")

    desc_func = getattr(tool, "to_react_description", None)
    if callable(desc_func):
        try:
            return str(desc_func())
        except Exception:
            logger.exception("Tool.to_react_description failed: %s", name)
            return f"Tool: {name}"

    return f"Tool: {name}"

# ---------------------------------------------------------------------------
# AgentLoop / Harness 私有字段适配层
# ---------------------------------------------------------------------------
#
# 这些函数集中访问私有属性。等 AgentLoop / Harness 提供公开接口后，只改这里。

def _get_final_answer(loop: Optional[AgentLoop]) -> str:
    if loop is None:
        return ""
    value = getattr(loop, "_final_answer", "")
    return value or ""

def _get_total_steps(loop: Optional[AgentLoop]) -> int:
    if loop is None:
        return 0

    steps = getattr(loop, "_steps", None)
    if steps is None:
        return 0

    try:
        return len(steps)
    except Exception:
        return 0

def _get_loop_state_name(loop: Optional[AgentLoop]) -> str:
    if loop is None:
        return "UNKNOWN"

    state = getattr(loop, "state", None)
    if state is None:
        return "UNKNOWN"

    name = getattr(state, "name", None)
    if isinstance(name, str) and name:
        return name

    if isinstance(state, str):
        return state

    try:
        return str(state)
    except Exception:
        return "UNKNOWN"

def _get_current_step(loop: Optional[AgentLoop]) -> int:
    if loop is None:
        return 0

    value = getattr(loop, "_current_step", 0)
    try:
        return int(value)
    except Exception:
        return 0


def _resolve_parent_context(
    harness: "Harness",
    parent_task_id: str,
) -> tuple[List[Tool], Any]:
    """
    解析父任务上下文，返回 (tools, approval_gate)。

    优先使用 contextvar 中的当前 sub-agent 上下文；否则 fallback 到 harness 字典。
    取消事件统一通过 harness.cancel_events 获取，不在此函数返回。
    """
    ctx: Optional[_SubAgentContext] = _current_context.get()

    if ctx is not None and ctx.task_id == parent_task_id:
        return list(ctx.tools), ctx.approval_gate

    default_tools: List[Tool] = []
    try:
        default_tools = list(harness.registry.list_tools())
    except Exception:
        logger.exception("failed to list default tools from registry")

    tools: List[Tool] = default_tools

    tools_map = getattr(harness, "_loop_tools", None)
    if isinstance(tools_map, dict):
        tools = tools_map.get(parent_task_id, default_tools)

    gate = None
    gates = getattr(harness, "approval_gates", None)
    if isinstance(gates, dict):
        gate = gates.get(parent_task_id)

    if gate is None:
        gate = getattr(harness, "default_approval_gate", None)

    return list(tools or []), gate

def _register_child_context_maps(
    harness: "Harness",
    child_task_id: str,
    tools: List[Tool],
    approval_gate: Any,
    cancel_event: asyncio.Event,
) -> List[tuple[str, str, str]]:
    """
    将子代理上下文注册到 harness 的 task_id -> context 字典中。

    只注册不覆盖，避免污染已有任务。
    返回已注册的 (attr_name, key, kind) 列表，用于后续清理。
    """
    registered: List[tuple[str, str, str]] = []

    tools_map = getattr(harness, "_loop_tools", None)
    if isinstance(tools_map, dict) and child_task_id not in tools_map:
        tools_map[child_task_id] = list(tools)
        registered.append(("_loop_tools", child_task_id, "tools"))

    gates_map = getattr(harness, "approval_gates", None)
    if isinstance(gates_map, dict) and child_task_id not in gates_map:
        gates_map[child_task_id] = approval_gate
        registered.append(("approval_gates", child_task_id, "approval_gate"))

    cancel_map = getattr(harness, "cancel_events", None)
    if isinstance(cancel_map, dict) and child_task_id not in cancel_map:
        cancel_map[child_task_id] = cancel_event
        registered.append(("cancel_events", child_task_id, "cancel_event"))

    return registered

def _unregister_child_context_maps(
    harness: "Harness",
    registered: List[tuple[str, str, str]],
    child_cancel_event: asyncio.Event,
) -> None:
    """清理通过 _register_child_context_maps 注册的上下文。"""
    for attr_name, key, kind in reversed(registered):
        try:
            d = getattr(harness, attr_name, None)
            if not isinstance(d, dict):
                continue

            if kind == "cancel_event":
                if d.get(key) is child_cancel_event:
                    d.pop(key, None)
            else:
                d.pop(key, None)
        except Exception:
            logger.exception(
                "failed to cleanup child context map: %s.%s",
                type(harness).__name__,
                attr_name,
            )

def _store_subagent_state(
    harness: "Harness",
    subagent_id: str,
    state: SubAgentState,
) -> None:
    """
    保存 sub-agent state,并做简单 GC。

    如果 harness._subagent_states 不存在则忽略。
    """
    states = getattr(harness, "_subagent_states", None)
    if not isinstance(states, dict):
        return

    states[subagent_id] = state

    max_states = _as_int(_cfg(harness, "max_subagent_states", 1000), 1000)
    if max_states <= 0 or len(states) <= max_states:
        return

    # 优先删除已经结束的旧 state
    finished: List[tuple[float, str]] = []
    for sid, s in states.items():
        finished_at = getattr(s, "finished_at", None)
        if finished_at is not None:
            created_at = _as_float(getattr(s, "created_at", 0.0), 0.0)
            finished.append((created_at, sid))

    finished.sort(key=lambda x: x[0])

    overflow = len(states) - max_states
    for _, sid in finished[:overflow]:
        states.pop(sid, None)

def _wrap_child_event(
    child: LoopEvent,
    subagent_id: str,
    depth: int,
    parent_task_id: str,
) -> LoopEvent:
    """
    给子事件附加 subagent 标识。

    注意:
    - 如果事件已经带有 subagent_id,说明它可能来自更深层 sub-agent;
      此时不覆盖原始归属，只补充 parent 链路。
    - 每个事件补充 event_id,方便父层 / CLI 去重。
    """
    data = dict(child.data or {})

    if not data.get("event_id"):
        data["event_id"] = uuid.uuid4().hex

    if data.get("subagent_id"):
        parent_ids = list(data.get("parent_subagent_ids", []))
        if subagent_id not in parent_ids:
            parent_ids.insert(0, subagent_id)
        data["parent_subagent_ids"] = parent_ids
        data.setdefault("parent_task_id", parent_task_id)
        data.setdefault("depth", depth)
    else:
        data["subagent_id"] = subagent_id
        data["depth"] = depth
        data["parent_task_id"] = parent_task_id

    return LoopEvent(
        event_type=child.event_type,
        task_id=child.task_id,
        timestamp=child.timestamp,
        step_num=child.step_num,
        content=child.content,
        tool_name=child.tool_name,
        data=data,
        error=child.error,
        trace_id=getattr(child, "trace_id", None) or subagent_id,
    )

def _make_internal_event(
    event_type: Any,
    task_id: str,
    content: str,
    subagent_id: str,
    depth: int,
    parent_task_id: str,
    error: Optional[str] = None,
) -> LoopEvent:
    """构造 runner 内部事件。"""
    return _wrap_child_event(
        LoopEvent(
            event_type=event_type,
            task_id=task_id,
            timestamp=time.time(),
            step_num=None,
            content=content,
            tool_name="sub_agent",
            data={},
            error=error,
            trace_id=subagent_id,
        ),
        subagent_id=subagent_id,
        depth=depth,
        parent_task_id=parent_task_id,
    )

def _build_child_system_prompt(
    harness: "Harness",
    prompt: str,
    parent_filtered_tools: List[Tool],
) -> str:
    """
    子代理 system_prompt。

    强调:
    - 没有父任务历史；
    - 只看 prompt;
    - 使用父任务工具集；
    - ReAct 输出格式。
    """
    tools_desc = "\n\n".join(_describe_tool(t) for t in parent_filtered_tools)

    skill_patch_func = getattr(harness.registry, "get_discovery_prompt_patch", None)
    skill_patch = ""
    if callable(skill_patch_func):
        try:
            skill_patch = str(skill_patch_func() or "")
        except Exception:
            logger.exception("registry.get_discovery_prompt_patch failed")
            skill_patch = ""

    return f"""你是一个子代理，正在执行父代理委派的一个独立子任务。
你没有父任务的对话历史——只看下面的 prompt 作为你的目标。

[Sub-task]
{prompt}

---

可用工具：
{tools_desc}
{skill_patch}

---

输出规则:
1. 如果需要使用工具，请严格按以下格式输出:

Thought: 你的思考过程(为什么需要这个工具)
Action: 工具名
Action Input: {{"参数名": "参数值"}}

2. 如果不需要工具，直接以 "Final Answer: ..." 开头回答。

3. 工具执行结果会以 Observation: 的形式返回。

4. 完成后，你的整段最终回答会作为结果返回给父代理。
"""

def make_sub_agent_tool(harness: "Harness") -> Tool:
    """工厂: 用 harness 闭包构造 sub_agent 工具。"""

    async def run_sub_agent(
        task_id: str,
        prompt: str,
        max_steps: Optional[int] = None,
    ) -> ToolResult:
        return await _run_sub_agent(harness, task_id, prompt, max_steps)

    return Tool(
        name=_SUB_AGENT_TOOL_NAME,
        description=(
            "启动一个全新上下文的子代理来完成任务。"
            "子代理看不到父任务的对话历史，只看 prompt 和父任务当前可用的工具集。"
            "返回子代理的最终答案。"
        ),
        parameters=[
            ToolParameter(
                name="prompt",
                type="string",
                description="子代理要完成的任务描述",
                required=True,
            ),
            ToolParameter(
                name="max_steps",
                type="integer",
                description="限制子代理最大步数，默认使用父任务或系统配置",
                required=False,
            ),
        ],
        func=run_sub_agent,
        tags=["general"],
        executor_type="async",
        dangerous=False,
        timeout=_as_float(_cfg(harness, "subagent_tool_timeout", 1800.0), 1800.0),
    )

async def _run_sub_agent(
    harness: "Harness",
    parent_task_id: str,
    prompt: str,
    max_steps: Optional[int],
) -> ToolResult:
    
    prompt = str(prompt or "").strip()
    if not prompt:
        return ToolResult(
            tool_name="sub_agent",
            arguments={"prompt": prompt, "max_steps": max_steps},
            output=None,
            error="[Error] prompt 不能为空",
            data={"events": []},
        )

    # max_prompt_chars = _as_int(_cfg(harness, "max_subagent_prompt_chars", 0), 0)
    # if max_prompt_chars > 0 and len(prompt) > max_prompt_chars:
    #     prompt = prompt[:max_prompt_chars] + "\n\n[prompt truncated]"

    parent_depth = _as_int(_current_depth.get(), 0)
    depth = parent_depth + 1
    max_depth = _as_int(_cfg(harness, "max_subagent_depth", 1), 1)

    if depth > max_depth:
        return ToolResult(
            tool_name="sub_agent",
            arguments={"prompt": prompt, "max_steps": max_steps},
            output=None,
            error=f"[Error] 递归深度超限 (depth={depth}, max={max_depth})",
            data={"events": []},
        )

    depth_token = _current_depth.set(depth)

    subagent_id = (
        f"{str(parent_task_id)[:8]}_d{depth}_"
        f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
    )

    state = SubAgentState(
        subagent_id=subagent_id,
        parent_task_id=parent_task_id,
        created_at=time.time(),
        depth=depth,
    )
    _store_subagent_state(harness, subagent_id, state)

    child_task_id: Optional[str] = None
    child_loop: Optional[AgentLoop] = None
    child_cancel_event = asyncio.Event()

    heartbeat_task: Optional[asyncio.Task] = None
    parent_cancel_task: Optional[asyncio.Task] = None

    ctx_token: Optional[contextvars.Token] = None
    registered_maps: List[tuple[str, str, str]] = []

    collected_events: List[LoopEvent] = []
    dropped_events = 0
    events_truncated = False

    final_answer = ""
    final_state = "ERROR"
    normalized_max_steps: Optional[int] = None

    try:
        parent_tools, parent_gate = _resolve_parent_context(
            harness,
            parent_task_id,
        )
        parent_tools = list(parent_tools or [])

        if parent_gate is None:
            final_state = "ERROR"
            return ToolResult(
                tool_name="sub_agent",
                arguments={"prompt": prompt, "max_steps": max_steps},
                output=None,
                error="[Error] 父任务 ApprovalGate 不存在",
                data={"events": []},
            )

        # 如果当前已经是最大深度，子代理内部不应再看到 sub_agent 工具。
        if depth >= max_depth:
            parent_tools = [
                t for t in parent_tools
                if getattr(t, "name", None) != _SUB_AGENT_TOOL_NAME
            ]

        # 获取父任务的取消事件
        parent_cancel_event = None
        cancel_events = getattr(harness, "cancel_events", None)
        if isinstance(cancel_events, dict):
            parent_cancel_event = cancel_events.get(parent_task_id)

        max_history = _as_int(
            getattr(getattr(harness, "task_manager", None), "max_history", 100),
            100,
        )

        child_tm = TaskManager(
            max_history=max_history,
            store=None,
        )

        system_prompt = _build_child_system_prompt(
            harness=harness,
            prompt=prompt,
            parent_filtered_tools=parent_tools,
        )

        child_task_id = str(await child_tm.create_task(prompt, system_prompt))

        child_ctx = _SubAgentContext(
            task_id=child_task_id,
            subagent_id=subagent_id,
            depth=depth,
            tools=parent_tools,
            approval_gate=parent_gate,
        )
        ctx_token = _current_context.set(child_ctx)

        registered_maps = _register_child_context_maps(
            harness=harness,
            child_task_id=child_task_id,
            tools=parent_tools,
            approval_gate=parent_gate,
            cancel_event=child_cancel_event,
        )

        default_steps = _as_int(_cfg(harness, "max_steps", 200), 200)

        normalized_max_steps = _normalize_max_steps(
            value=max_steps,
            default=default_steps,
        )

        child_loop = AgentLoop(
            tool_registry=harness.registry,
            llm=harness.llm,
            task_manager=child_tm,
            execution_engine=harness.engine,
            context_manager=harness.context_manager,
            approval=parent_gate,
            cancel_event=child_cancel_event,
            max_steps=normalized_max_steps,
            watch_dog=harness.watch_dog,
            llm_timeout=_cfg(harness, "llm_timeout", None),
            approval_timeout=_cfg(harness, "approval_timeout", None),
            max_consecutive_parse_errors=_cfg(
                harness,
                "max_consecutive_parse_errors",
                3,
            ),
        )

        if (
            parent_cancel_event is not None
            and parent_cancel_event is not child_cancel_event
        ):
            async def _watch_parent_cancel() -> None:
                await parent_cancel_event.wait()
                child_cancel_event.set()

            parent_cancel_task = asyncio.create_task(_watch_parent_cancel())

        heartbeat_interval = _as_float(
            _cfg(harness, "subagent_heartbeat_interval", 2.0),
            2.0,
        )

        if heartbeat_interval > 0 and _PROGRESS_UPDATE_EVENT is not None:
            async def _heartbeat() -> None:
                try:
                    while True:
                        await asyncio.sleep(heartbeat_interval)

                        sink = _current_event_sink.get()
                        if sink is None:
                            continue

                        current_step = _get_current_step(child_loop)
                        event = _make_internal_event(
                            event_type=_PROGRESS_UPDATE_EVENT,
                            task_id=child_task_id or subagent_id,
                            content=(
                                f"sub-agent running... "
                                f"step {current_step} (depth={depth})"
                            ),
                            subagent_id=subagent_id,
                            depth=depth,
                            parent_task_id=parent_task_id,
                        )
                        event.data["realtime_forwarded"] = True
                        _safe_emit(sink, event)

                except asyncio.CancelledError:
                    return
                except Exception:
                    logger.exception("sub-agent heartbeat failed")

            heartbeat_task = asyncio.create_task(_heartbeat())

        max_events = _as_int(_cfg(harness, "max_subagent_events", 5000), 5000)
        if max_events <= 0:
            max_events = 5000

        run_stream = child_loop.run(child_task_id, parent_tools)

        try:
            async for child_event in run_stream:
                wrapped = _wrap_child_event(
                    child_event,
                    subagent_id=subagent_id,
                    depth=depth,
                    parent_task_id=parent_task_id,
                )

                if len(collected_events) < max_events:
                    collected_events.append(wrapped)
                else:
                    dropped_events += 1
                    events_truncated = True

                # 实时转发关键事件。
                # 如果事件来自更深层 sub-agent，并且已经被实时转发过，
                # 则这里不要重复转发。
                if (
                    _should_forward_realtime(child_event)
                    and not wrapped.data.get("realtime_forwarded")
                ):
                    wrapped.data["realtime_forwarded"] = True
                    _safe_emit(_current_event_sink.get(), wrapped)

        finally:
            # 尽量确保 async generator 被安全关闭。
            aclose = getattr(run_stream, "aclose", None)
            if callable(aclose):
                try:
                    await aclose()
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.exception("failed to close child AgentLoop stream")

        final_answer = _get_final_answer(child_loop)

        child_state = await child_tm.get_state(child_task_id)
        if child_state is None:
            final_state = "ERROR"

            error_event = _make_internal_event(
                event_type=_ERROR_EVENT_TYPE,
                task_id=child_task_id or subagent_id,
                content="sub-agent TaskState 丢失",
                subagent_id=subagent_id,
                depth=depth,
                parent_task_id=parent_task_id,
                error="TaskState missing",
            )
            error_event.data["realtime_forwarded"] = True

            collected_events.append(error_event)
            _safe_emit(_current_event_sink.get(), error_event)

        else:
            task_status = getattr(child_state, "task_status", None)

            if task_status == TaskStatus.COMPLETED:
                final_state = "COMPLETED"
            elif task_status == TaskStatus.CANCELLED:
                final_state = "CANCELLED"
            elif task_status == TaskStatus.FAILED:
                final_state = "ERROR"
            elif _get_loop_state_name(child_loop) == "MAX_STEPS_REACHED":
                final_state = "MAX_STEPS_REACHED"
            else:
                final_state = "UNKNOWN"
    except asyncio.CancelledError:
        final_state = "CANCELLED"
        child_cancel_event.set()
        raise

    except Exception as e:
        logger.exception("sub-agent internal error: %s", subagent_id)

        final_state = "ERROR"
        final_answer = _get_final_answer(child_loop)

        error_event = _make_internal_event(
            event_type=_ERROR_EVENT_TYPE,
            task_id=child_task_id or subagent_id,
            content=f"sub-agent 内部异常: {e}",
            subagent_id=subagent_id,
            depth=depth,
            parent_task_id=parent_task_id,
            error=str(e),
        )
        error_event.data["realtime_forwarded"] = True

        collected_events.append(error_event)
        _safe_emit(_current_event_sink.get(), error_event)

    finally:
        await _cancel_task(heartbeat_task)
        await _cancel_task(parent_cancel_task)

        if ctx_token is not None:
            try:
                _current_context.reset(ctx_token)
            except Exception:
                logger.exception("failed to reset sub-agent context token")

        _unregister_child_context_maps(
            harness=harness,
            registered=registered_maps,
            child_cancel_event=child_cancel_event,
        )

        try:
            _current_depth.reset(depth_token)
        except Exception:
            logger.exception("failed to reset sub-agent depth token")

        state.finished_at = time.time()
        state.final_answer = final_answer
        state.final_state = final_state
        state.total_steps = _get_total_steps(child_loop)
        _store_subagent_state(harness, subagent_id, state)

    result_data = {
        "events": collected_events,
        "dropped_events": dropped_events,
        "events_truncated": events_truncated,
        "subagent_id": subagent_id,
        "depth": depth,
        "steps": state.total_steps,
        "final_state": final_state,
        "max_steps": normalized_max_steps,
    }

    if final_state == "CANCELLED":
        return ToolResult(
            tool_name="sub_agent",
            arguments={"prompt": prompt, "max_steps": max_steps},
            output=None,
            error="[Cancelled] 子代理被取消",
            data=result_data,
        )

    if final_state == "ERROR":
        partial = final_answer[:200] if final_answer else ""
        return ToolResult(
            tool_name="sub_agent",
            arguments={"prompt": prompt, "max_steps": max_steps},
            output=None,
            error=f"[Failed] 子代理执行失败，部分结果: {partial}",
            data=result_data,
        )

    if final_state == "UNKNOWN":
        partial = final_answer[:200] if final_answer else ""
        return ToolResult(
            tool_name="sub_agent",
            arguments={"prompt": prompt, "max_steps": max_steps},
            output=None,
            error=f"[Unknown] 子代理终止状态未知，部分结果: {partial}",
            data=result_data,
        )

    if final_state == "MAX_STEPS_REACHED":
        return ToolResult(
            tool_name="sub_agent",
            arguments={"prompt": prompt, "max_steps": max_steps},
            output=(
                "[Warning] sub-agent reached max steps before completion.\n"
                f"{final_answer or '(no answer)'}"
            ),
            data=result_data,
        )

    # COMPLETED
    return ToolResult(
        tool_name="sub_agent",
        arguments={"prompt": prompt, "max_steps": max_steps},
        output=final_answer or "(子代理未返回最终答案)",
        data=result_data,
    )