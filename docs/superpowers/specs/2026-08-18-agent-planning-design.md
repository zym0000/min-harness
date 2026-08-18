# Agent Planning Design

**Date:** 2026-08-18
**Status:** Approved (pending user spec review)

## Goal

Add persistent in-execution TodoList capability to the existing harness so the agent can track multi-step work and stay coherent across long tasks, while keeping prompt cache hit rate high.

## Non-Goals

- Multi-agent role collaboration (Planner / Executor / Reviewer as separate LLM instances)
- Plan-and-execute mode (forced pre-execution planning phase)
- Re-planning triggered by tool failure (LLM is free to call `update_plan` whenever it wants)
- Hierarchical decomposition via sub-agents (sub-agents remain a separate "divide and conquer" mechanism)

## Scope Confirmation

This design adds planning capability **inside** the existing single-agent ReAct loop of the harness. Sub-agents are untouched and remain available for "divide and conquer" subtask delegation — they are not part of the planning mechanism.

## Design

### 1. Architecture

```
Harness.submit_task(user_input)
  ↓
InputGateway.process() → tags=["general", "plan"]?
  ↓
UnifiedToolRegistry.filter_by_tags(["general", "plan"])
  ↓
filtered_tools (contains update_plan if "plan" tag present)
  ↓
AgentLoop._execute_steps
  ↓ per step:
  ├─ compress_messages() (history)
  ├─ PlanInjector.inject(messages)         ← NEW
  │     └─ read TaskState.plan → render → insert at [1]
  ├─ llm.chat_stream(messages)
  │     ↓
  │     if LLM calls update_plan(steps):
  │       ├─ UpdatePlanTool.execute()
  │       ├─ TaskManager.update_plan(task_id, steps)
  │       ├─ emit PLAN_UPDATED event
  │       └─ return success
  └─ next step: inject reflects new plan (if LLM updated it)
```

### 2. Components

| Component | Responsibility | File |
|-----------|----------------|------|
| `PlanStep` | dataclass: `id` / `content` / `status` | `plan/plan_step.py` (NEW) |
| `PlanManager` | render + summarize | `plan/plan_manager.py` (NEW) |
| `UpdatePlanTool` | built-in tool, persists plan | `plan/update_plan_tool.py` (NEW) |
| `PlanInjector` | injects plan message each step | `plan/plan_injector.py` (NEW) |
| `TaskState.plan` | persistent field | `task/task_defined.py` (MODIFY) |
| `TaskManager.update_plan` | locked write + validation | `task/task_manager.py` (MODIFY) |
| `InputGateway` | keyword triggers "plan" tag | `input_gateway.py` (MODIFY) |
| `EventType.PLAN_UPDATED` | new event type | `event/event.py` (MODIFY) |
| `Harness` | wires UpdatePlanTool + PlanInjector into AgentLoop | `harness.py` (MODIFY) |
| `bootstrap.py` | constructs UpdatePlanTool | `config/bootstrap.py` (MODIFY) |

### 3. Data Model

#### `PlanStep`

```python
# plan/plan_step.py
from dataclasses import dataclass
from typing import Literal

PlanStatus = Literal["pending", "in_progress", "done", "blocked"]

@dataclass(frozen=True)
class PlanStep:
    id: str
    content: str
    status: PlanStatus

    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> "PlanStep": ...
```

`frozen=True` so updates go through replacement. `Literal` so the tool's JSON schema is clean. `PlanStep` is used at the boundaries (tool argument parsing, validation, render), but `TaskState.plan` stores `List[Dict]` for SQLite round-trip parity with the existing `messages` field — see below.

#### `TaskState.plan`

```python
# task/task_defined.py
@dataclass
class TaskState:
    # ... existing fields ...
    plan: List[Dict[str, str]] = field(default_factory=list)
    # element: {"id": "s1", "content": "...", "status": "in_progress"}

    def to_checkpoint(self) -> Dict:
        # add "plan": list(self.plan)
    @classmethod
    def from_checkpoint(cls, data: Dict) -> "TaskState":
        # state.plan = list(data.get("plan", []))
```

Stored as `List[Dict]` (not `List[PlanStep]`) for SQLite round-trip parity with `messages`.

#### `TaskManager.update_plan`

```python
# task/task_manager.py
async def update_plan(self, task_id: str, steps: List[Dict]) -> Optional[str]:
    """Validate and replace plan; return error message or None."""
    # Acquire task lock
    # Validate each step has id/content/status; status in legal set
    # state.plan = list(steps)
    # Checkpoint auto-written
    # Return None on success
```

Empty `steps` is rejected — explicit empty plan is not allowed (LLM self-corrects).

### 4. `PlanManager.render` (cache-critical)

```python
# plan/plan_manager.py
class PlanManager:
    @staticmethod
    def render(plan: List[Dict]) -> str:
        """Deterministic render → LLM-visible text."""
        if not plan:
            return "[当前计划: 暂无]"
        lines = ["[当前计划]"]
        marker = {"pending": "[ ]", "in_progress": "[→]",
                  "done": "[x]", "blocked": "[!]"}
        for s in plan:
            m = marker.get(s.get("status", "pending"), "[ ]")
            lines.append(f"- {m} {s.get('id', '?')}: {s.get('content', '')}")
        return "\n".join(lines)

    @staticmethod
    def summarize(plan: List[Dict]) -> str:
        """Used in update_plan ToolResult."""
        if not plan:
            return "0/0 steps."
        done = sum(1 for s in plan if s.get("status") == "done")
        total = len(plan)
        if done == total:
            return f"All {total} steps completed. You can now provide the Final Answer or continue with related work."
        elif done > 0:
            return f"{done}/{total} steps done. Next: continue with pending steps."
        return f"{total} steps pending."
```

Determinism is required for cache hit rate. Same plan → byte-identical output → cache hit.

### 5. `UpdatePlanTool`

Constructed as a `Tool` instance whose `func` is an async function. `Tool.execute` calls `func(**kwargs)` — the framework does not subclass `Tool`, so the implementation follows the existing pattern (see `subagent/runner.py:483`).

```python
# plan/update_plan_tool.py
async def _update_plan_func(
    task_manager,            # injected via closure / partial
    plan_manager,            # same
    event_emitter,           # same
    task_id: str,
    steps: list,
) -> str:
    """`func` body — signature matches ToolParameter names + task_id."""
    if not isinstance(steps, list) or not steps:
        raise ValueError("steps must be a non-empty array")
    err = await task_manager.update_plan(task_id, steps)
    if err:
        raise ValueError(err)
    # emit PLAN_UPDATED for UI
    await event_emitter(LoopEvent(
        event_type=EventType.PLAN_UPDATED,
        task_id=task_id,
        timestamp=time.time(),
        data={"step_count": len(steps),
              "done_count": sum(1 for s in steps if s.get("status") == "done")},
    ))
    return f"Plan updated. {plan_manager.summarize(steps)}"

def build_update_plan_tool(task_manager, plan_manager, event_emitter) -> Tool:
    bound = functools.partial(
        _update_plan_func, task_manager, plan_manager, event_emitter)
    return Tool(
        name="update_plan",
        description=(
            "Update the current task plan. Replaces the entire plan (not incremental). "
            "Each step must have id (short string), content (description), "
            "and status (pending|in_progress|done|blocked). "
            "Use to track multi-step work: create plan early, mark in_progress when starting, "
            "mark done when complete. Plan is visible to you on every step."
        ),
        parameters=[
            ToolParameter("steps", "array", required=True,
                         description="List of step objects: {id, content, status}"),
        ],
        func=bound,
        tags=["plan"],
        dangerous=False,
        executor_type="async",
    )
```

`ToolResult` is the string return value (existing convention). Errors raise `ValueError`, which the execution engine surfaces to the LLM as a failed tool result.

### 6. `PlanInjector`

```python
# plan/plan_injector.py
class PlanInjector:
    def __init__(self, task_manager, plan_manager):
        self.task_manager = task_manager
        self.plan_manager = plan_manager

    async def inject(self, task_id: str, messages: List[Dict]) -> None:
        """Mutate messages in place: insert plan message at position 1 (after system)."""
        state = await self.task_manager.get_state(task_id)
        plan = state.plan if state else []
        plan_text = self.plan_manager.render(plan)
        insert_pos = 0
        if messages and messages[0].get("role") == "system":
            insert_pos = 1
        messages.insert(insert_pos, {"role": "system", "content": plan_text})
```

Mutates the existing list (matches the calling convention in `agent_loop._execute_steps`). If `messages` is empty, no-op; if no system message present, plan is inserted at index 0.

### 7. `AgentLoop` integration

```python
# agent_loop.py _execute_steps (modify)
# After compress_messages, before LLM call:
messages = await self.task_manager.compress_messages(...)
messages = await self.plan_injector.inject(task_id, messages)
# Then continue with existing LLM call
```

`PlanInjector` is constructed in `Harness._create_generator` and passed into `AgentLoop` constructor (similar to how `context_manager` is passed).

### 8. `InputGateway` keyword gate

```python
# input_gateway.py
_PLAN_KEYWORDS = ["分步", "步骤", "先做", "计划", "分解", "todo", "plan"]

def process(self, user_input: str) -> List[str]:
    tags = super_existing_logic()
    if any(kw in user_input.lower() for kw in self._PLAN_KEYWORDS):
        if "plan" not in tags:
            tags.append("plan")
    return tags
```

Existing `Harness._filter_tools` already intersects tags with `["general"]`, so `update_plan` becomes visible only when the keyword matches.

### 9. `EventType.PLAN_UPDATED`

```python
# event/event.py
class EventType(Enum):
    # ... existing ...
    PLAN_UPDATED = auto()
```

Emitted by `UpdatePlanTool.execute` after successful write. Consumed by display layer (CLI), **not** pushed into the LLM message stream.

## Cache Hit Rate Strategy

| Lever | Mechanism |
|-------|-----------|
| Plan lives at `messages[1]`, system stays `messages[0]` | Stable prefix preserved |
| `render()` is deterministic | Same plan → byte-identical string |
| Empty plan = `"[当前计划: 暂无]"` | Fixed placeholder; doesn't break prefix |
| Plan only changes when LLM calls `update_plan` | Most steps: plan unchanged → cache hit |
| Tool schema for `update_plan` is static | Tool description cached at registration |
| `summarize()` returns deterministic text | ToolResult content also cache-stable |

## Error Handling

| Scenario | Detection | Behavior |
|----------|-----------|----------|
| `steps` not array | `UpdatePlanTool.execute` | `ToolResult(is_error=True, error="steps must be array")` |
| `steps` empty | same | `ToolResult(is_error=True, error="steps must be non-empty")` |
| Step missing field | `TaskManager.update_plan` | `ToolResult(is_error=True, error="Step missing required field")` |
| Invalid status | same | `ToolResult(is_error=True, error="Invalid status 'foo'")` |
| Task not found | same | `ToolResult(is_error=True, error="Task not found")` |
| Plan render fails (corrupt data) | `PlanManager.render` fallback | Return `"[当前计划: 格式异常]"`, no exception |
| Empty messages list | `PlanInjector.inject` | Skip inject, return messages unchanged |
| Concurrent update_plan | task-level lock | Serialized; last write wins |
| Update on FINISHED task | no extra check | Plan persists; task status unaffected (allowed for post-hoc planning) |

No errors are fatal — LLM receives feedback and self-corrects.

## Testing

### Unit

| File | Coverage |
|------|----------|
| `tests/test_plan_step.py` | to_dict / from_dict roundtrip, frozen invariant |
| `tests/test_plan_manager.py` | render (empty, all statuses), summarize (3 cases) |
| `tests/test_update_plan_tool.py` | execute (valid / empty / missing fields / bad status / task not found) |
| `tests/test_task_manager_plan.py` | update_plan lock, validation, checkpoint write |
| `tests/test_input_gateway_plan.py` | keyword match ("分步", "plan", etc.) → "plan" tag |
| `tests/test_plan_injector.py` | insert position, empty plan, no system message fallback |

### Integration

| File | Coverage |
|------|----------|
| `tests/test_plan_e2e.py` | End-to-end: user says "分步 X" → gateway tags → LLM sees update_plan → LLM calls it → next step injects plan → cache-friendly trace |

### Manual

- `/status` shows plan
- Multi-step trace with stable plan shows consistent token usage (cache hit indicator)

## Implementation Order

1. Add `PlanStep` dataclass
2. Extend `TaskState` with `plan` field + checkpoint roundtrip
3. Add `TaskManager.update_plan` with validation + lock
4. Add `PlanManager.render` + `summarize`
5. Add `EventType.PLAN_UPDATED`
6. Add `UpdatePlanTool`
7. Modify `InputGateway` keyword set
8. Add `PlanInjector`
9. Wire into `Harness._create_generator` and `AgentLoop.__init__` + `_execute_steps`
10. Update `config/bootstrap.py` to construct + register `UpdatePlanTool` (via `build_update_plan_tool` factory)
11. Tests (unit + integration)
12. Manual `/status` verification

## Risks & Open Questions

1. **`update_plan` tool visible without keyword match?** — `tags=["plan"]` ensures it only appears via gateway match. If user wants plan for every task, remove keyword gate.
2. **Plan grows unbounded?** — LLM-controlled; if LLM adds 1000 steps, plan message bloats. Trust LLM to keep it short.
3. **Compression interaction?** — `compress_messages` may strip the plan message. Need to verify plan injection happens **after** compression, so plan is always fresh.
4. **Tool description verbosity?** — `update_plan` description is moderately long. Adds ~150 tokens to system prompt. Acceptable trade-off.

## Decision Log

| Decision | Rationale |
|----------|-----------|
| Single-agent + TodoList (not multi-agent) | Minimal scope, cache-friendly, sub-agents remain separate |
| Persistent TodoList (not plan-then-execute) | Matches actual coding workflows, no forced phase |
| Dedicated `update_plan` tool (not text protocol) | Reuses existing tool ecosystem, no extra parsing |
| Intent-gated via keywords (not always-on) | Saves prompt tokens for simple queries |
| `messages[1]` injection (not system or tail) | Preserves system-message cache prefix |
| `List[Dict]` storage (not `List[PlanStep]`) | SQLite parity with `messages` field |
| Strict validation (no auto-fix) | LLM self-corrects; framework shouldn't second-guess |
| Empty plan rejected (not allowed) | Forces LLM to be explicit about plan state |