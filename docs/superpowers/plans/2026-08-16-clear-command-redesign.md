# `/clear` Command Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modify the `/clear` CLI command so it both clears the terminal screen AND resets the current task's context (messages, LLM-extracted memory, SQLite checkpoint, harness caches, derived sub-agent states).

**Architecture:** Add a new `Harness.clear_task(task_id)` method that encapsulates cancel + wait + cleanup logic. The CLI's `/clear` branch becomes a thin wrapper that calls it. All cleanup is centralized in one method, making it unit-testable and reusable.

**Tech Stack:** Python 3.13, asyncio, unittest (project convention — see `tests/subagent/test_runner.py`), existing Harness / TaskManager.

**Reference Spec:** `docs/superpowers/specs/2026-08-16-clear-command-redesign-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `harness.py` | Modify | Add `Harness.clear_task()` method (~30 lines) after the existing `cancel_task`. |
| `interaction/cli.py` | Modify | Replace the `/clear` branch (~10 lines) and update the help-text line. |
| `tests/test_harness_clear_task.py` | Create | Unit tests for `Harness.clear_task`. New top-level test file. |

No new directories needed. The project already has `tests/__init__.py`.

---

## Task 1: Add unit tests for `Harness.clear_task` (TDD — failing first)

**Files:**
- Create: `tests/test_harness_clear_task.py`

**Context:** The project uses `unittest` + `asyncio.run(self._run())` for async tests. See `tests/subagent/test_runner.py:101-115` for the canonical pattern. We don't need the sub-agent tool for these tests; use a minimal harness constructor.

- [ ] **Step 1: Create the test file with imports and a minimal harness factory**

Create `tests/test_harness_clear_task.py`:

```python
"""Tests for Harness.clear_task — covers cancel + cleanup of all task-related state."""
import asyncio
import unittest

from harness import Harness, HarnessConfig
from interaction.approval_gate import ApprovalGate
from llm_client import ChatResponse
from task.task_defined import TaskState, TaskStatus
from tool_manger import ToolRegistry


class _StubLLM:
    """Minimal LLM stub — never invoked by clear_task, but Harness requires one."""

    async def chat(self, messages, tool_schema=None) -> ChatResponse:
        return ChatResponse(text="Final Answer: stub")

    async def chat_stream(self, messages, tool_schema=None):
        resp = await self.chat(messages, tool_schema)
        yield resp

    async def aclose(self):
        pass


def _make_harness() -> Harness:
    cfg = HarnessConfig(
        max_concurrent=1,
        max_steps=10,
        enable_watchdog=False,
        store_path=None,  # in-memory only — no SQLite for unit tests
    )
    return Harness(registry=ToolRegistry(), llm=_StubLLM(), config=cfg)


def _seed_task(harness: Harness, task_id: str = "test_task_1") -> None:
    """Populate harness-level dicts as if a task were live, BUT not actively running.

    Does NOT add to `_loops` — that entry only exists while a wrapper is iterating.
    Without it, clear_task's wait loop exits on iteration 0 (fast tests).
    """
    harness.task_manager.tasks[task_id] = TaskState(
        task_id=task_id,
        user_input="hello",
        system_prompt="sys",
        current_step=0,
        task_status=TaskStatus.RUNNING,
        messages=[{"role": "user", "content": "hello"}],
        task_summary="some summary",
        key_facts=["fact one"],
        memory_segment="memory text",
        memory_cursor=2,
    )
    harness.cancel_events[task_id] = asyncio.Event()
    harness.approval_gates[task_id] = ApprovalGate()
    harness._loop_tools[task_id] = []
    harness._run_results[task_id] = None


def _seed_running_loop(harness: Harness, task_id: str) -> None:
    """Same as _seed_task plus _loops[task_id] — simulates an active AgentLoop.

    Tests using this MUST also override `harness.cancel_task` to clear `_loops`
    synchronously, otherwise clear_task's wait loop burns 5 s.
    """
    _seed_task(harness, task_id)
    harness._loops[task_id] = object()


class TestClearTaskUnknown(unittest.TestCase):
    """Calling clear_task on a non-existent task_id returns False and is a no-op."""

    def test_returns_false_for_unknown_task_id(self):
        async def _run():
            harness = _make_harness()
            ok = await harness.clear_task("does_not_exist")
            self.assertFalse(ok)
        asyncio.run(_run())


class TestClearTaskCleanup(unittest.TestCase):
    """clear_task removes all traces of the task from harness state."""

    def test_removes_task_from_task_manager(self):
        async def _run():
            harness = _make_harness()
            _seed_task(harness, "t1")
            self.assertIn("t1", harness.task_manager.tasks)

            ok = await harness.clear_task("t1")

            self.assertTrue(ok)
            self.assertNotIn("t1", harness.task_manager.tasks)

        asyncio.run(_run())

    def test_pops_all_harness_level_dicts(self):
        async def _run():
            harness = _make_harness()
            _seed_task(harness, "t1")

            await harness.clear_task("t1")

            self.assertNotIn("t1", harness.cancel_events)
            self.assertNotIn("t1", harness.approval_gates)
            self.assertNotIn("t1", harness._loops)
            self.assertNotIn("t1", harness._loop_tools)
            self.assertNotIn("t1", harness._run_results)

        asyncio.run(_run())

    def test_clears_active_loop_entry(self):
        """When _loops has the task_id, clear_task still pops it (via the
        defensive pop in step 4 of clear_task, not the wait loop)."""
        async def _run():
            harness = _make_harness()
            _seed_running_loop(harness, "t1")
            # Replace cancel_task so the wait loop doesn't burn 5 s — simulate
            # the wrapper's finally block having already popped _loops.
            async def fast_cancel(tid):
                harness._loops.pop(tid, None)
                event = harness.cancel_events.get(tid)
                if event:
                    event.set()
            harness.cancel_task = fast_cancel

            await harness.clear_task("t1")

            self.assertNotIn("t1", harness._loops)
        asyncio.run(_run())

    def test_memory_fields_were_cleared_via_task_manager_delete(self):
        async def _run():
            # clear_task delegates to task_manager.delete_task, which already
            # removes the TaskState (and therefore its memory_* fields). Verify
            # the contract: after clear_task, the task is fully gone from memory.
            harness = _make_harness()
            _seed_task(harness, "t1")
            # Confirm seed worked
            state = harness.task_manager.tasks["t1"]
            self.assertEqual(state.memory_segment, "memory text")
            self.assertEqual(state.key_facts, ["fact one"])

            await harness.clear_task("t1")

            self.assertNotIn("t1", harness.task_manager.tasks)

        asyncio.run(_run())


class TestClearTaskSubAgents(unittest.TestCase):
    """clear_task removes sub-agent states for the cleared parent only."""

    def _make_state(self, parent_id: str, sub_id: str):
        # Build a minimal object with .parent_task_id attribute.
        # Use a simple namespace to avoid importing subagent.state and risking
        # a circular import.
        from types import SimpleNamespace
        return SimpleNamespace(
            subagent_id=sub_id,
            parent_task_id=parent_id,
            final_state="COMPLETED",
            final_answer="x",
        )

    def test_removes_subagents_for_target_parent(self):
        async def _run():
            harness = _make_harness()
            _seed_task(harness, "parent_a")
            _seed_task(harness, "parent_b")
            harness._subagent_states["sub_a1"] = self._make_state("parent_a", "sub_a1")
            harness._subagent_states["sub_a2"] = self._make_state("parent_a", "sub_a2")
            harness._subagent_states["sub_b1"] = self._make_state("parent_b", "sub_b1")

            await harness.clear_task("parent_a")

            self.assertNotIn("sub_a1", harness._subagent_states)
            self.assertNotIn("sub_a2", harness._subagent_states)
            self.assertIn("sub_b1", harness._subagent_states)

        asyncio.run(_run())


class TestClearTaskMetricsPreserved(unittest.TestCase):
    """clear_task does not touch MetricsControll."""

    def test_metrics_unchanged(self):
        async def _run():
            harness = _make_harness()
            _seed_task(harness, "t1")
            # Pre-record some metrics
            harness.metrics.record_llm_call(100)
            snapshot_before = harness.metrics.get_summary()

            await harness.clear_task("t1")

            snapshot_after = harness.metrics.get_summary()
            self.assertEqual(snapshot_before, snapshot_after)

        asyncio.run(_run())


class TestClearTaskIdempotent(unittest.TestCase):
    """Second clear_task call on the same task_id returns False without raising."""

    def test_second_call_returns_false(self):
        async def _run():
            harness = _make_harness()
            _seed_task(harness, "t1")

            first = await harness.clear_task("t1")
            second = await harness.clear_task("t1")

            self.assertTrue(first)
            self.assertFalse(second)

        asyncio.run(_run())
```

- [ ] **Step 2: Run the tests and verify they all fail with `AttributeError`**

Run from project root:

```bash
cd "E:/project/harness" && python -m unittest tests.test_harness_clear_task -v
```

Expected: every test errors out with `AttributeError: 'Harness' object has no attribute 'clear_task'`. This confirms the tests are failing for the right reason.

- [ ] **Step 3: Commit the failing tests**

```bash
cd "E:/project/harness" && git add tests/test_harness_clear_task.py && git commit -m "$(cat <<'EOF'
test: add failing tests for Harness.clear_task

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Implement `Harness.clear_task`

**Files:**
- Modify: `harness.py:236` (insert after the existing `cancel_task` method)

- [ ] **Step 1: Add the `clear_task` method to `Harness`**

Insert directly after `cancel_task` (which ends at line 242: `return True`):

```python
    async def clear_task(self, task_id: str) -> bool:
        """清空 task 全部状态(含 SQLite + harness 级缓存 + 派生 sub-agent)。

        若 loop 仍在跑,先 cancel 并等待退出再清理。
        返回 True 表示清理发生,False 表示 task 不存在。
        """
        if task_id not in self.task_manager.tasks:
            return False

        # 1. Cancel running loop (幂等;已退出则 no-op)
        await self.cancel_task(task_id)

        # 2. 等 loop 退出——wrapper 的 finally 会 pop _loops[task_id],
        #    这是最自然的"已结束"信号。若 task 从未进入 _loops(PENDING),
        #    第一次循环即退出。
        for _ in range(50):
            if task_id not in self._loops:
                break
            await asyncio.sleep(0.1)

        # 3. 删 SQLite row + 内存 TaskState
        await self.task_manager.delete_task(task_id)

        # 4. 防御性清理 harness 级 dict
        #    (wrapper finally 已 pop 一部分;此处兜底幂等)
        self.cancel_events.pop(task_id, None)
        self.approval_gates.pop(task_id, None)
        self._loops.pop(task_id, None)
        self._loop_tools.pop(task_id, None)
        self._run_results.pop(task_id, None)

        # 5. 清派生 sub-agent(仅 parent 匹配的;不做级联)
        stale = [
            sid for sid, s in self._subagent_states.items()
            if getattr(s, "parent_task_id", None) == task_id
        ]
        for sid in stale:
            self._subagent_states.pop(sid, None)

        return True
```

Note: `asyncio` is already imported at the top of `harness.py` (line 8). No new imports needed.

- [ ] **Step 2: Run the tests and verify they pass**

Run from project root:

```bash
cd "E:/project/harness" && python -m unittest tests.test_harness_clear_task -v
```

Expected: all 8 tests pass (1 unknown + 4 cleanup + 1 sub-agent + 1 metrics + 1 idempotent).

If any fail, common causes:
- `TaskManager.delete_task` mutating something unexpectedly → check `task/task_manager.py:65-71`
- `_subagent_states` items not exposing `parent_task_id` → check `subagent/state.py`

- [ ] **Step 3: Commit the implementation**

```bash
cd "E:/project/harness" && git add harness.py && git commit -m "$(cat <<'EOF'
feat(harness): add clear_task for full task state cleanup

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Update CLI `/clear` handler and help text

**Files:**
- Modify: `interaction/cli.py:32` (help text line)
- Modify: `interaction/cli.py:226-227` (`/clear` branch)

- [ ] **Step 1: Update the help text line**

In `interaction/cli.py`, change line 32 from:

```
  /clear      clear screen
```

to:

```
  /clear      clear screen + reset current task memory
```

- [ ] **Step 2: Replace the `/clear` branch**

In `interaction/cli.py`, change lines 226-227 from:

```python
        elif c == "/clear":
            os.system("clear" if os.name != "nt" else "cls")
```

to:

```python
        elif c == "/clear":
            os.system("clear" if os.name != "nt" else "cls")
            tid = self.task_id
            if not tid:
                print(f"  {D}no active task — screen cleared{R}")
            else:
                cleared = await self.harness.clear_task(tid)
                self.task_id = None
                if cleared:
                    print(f"  {YEL}task cleared: {tid[:12]}…{R}")
                else:
                    print(f"  {D}task already gone — screen cleared{R}")
```

- [ ] **Step 3: Verify the CLI module imports without syntax errors**

```bash
cd "E:/project/harness" && python -c "import interaction.cli; print('OK')"
```

Expected output: `OK`. If any `SyntaxError` or `NameError`, fix and re-run.

- [ ] **Step 4: Re-run the unit tests to make sure nothing regressed**

```bash
cd "E:/project/harness" && python -m unittest tests.test_harness_clear_task -v
```

Expected: all 8 tests still pass.

- [ ] **Step 5: Commit the CLI changes**

```bash
cd "E:/project/harness" && git add interaction/cli.py && git commit -m "$(cat <<'EOF'
feat(cli): /clear now resets current task memory

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Manual smoke test

**Files:** None (verification only).

- [ ] **Step 1: Run the smoke test from the spec**

Run the following commands and observe the output:

```bash
cd "E:/project/harness" && python main.py
```

Then interactively:

1. Type any task, e.g. `say hello`. Wait for the agent to finish (you should see a task ID like `task: 1701234567890_…`).
2. Type a follow-up that triggers a tool call (e.g. `read file CLAUDE.md`). This adds to the current task's history.
3. Type `/clear`.
4. **Expected:**
   - Terminal screen clears.
   - Line `task cleared: <prefix>…` printed.
5. Type `say hello again`.
6. **Expected:** a *new* task ID is printed (different prefix from step 1).
7. Type `/quit` to exit.
8. Restart `python main.py`. Type `/status`.

**Expected:** the cleared task from step 1 does NOT appear.

- [ ] **Step 2: Verify the no-task case**

```bash
cd "E:/project/harness" && python main.py
```

1. Without typing anything else, immediately type `/clear`.
2. **Expected:** screen clears; line `no active task — screen cleared` printed.
3. Type `/quit`.

- [ ] **Step 3: Final commit (if any stray changes)**

If the smoke test surfaced nothing to fix, no commit. If anything was tweaked:

```bash
cd "E:/project/harness" && git status --short
```

For each modified tracked file:

```bash
cd "E:/project/harness" && git add <file> && git commit -m "fix: address smoke test findings"
```

---

## Verification Checklist

Before declaring done, confirm:

- [ ] `python -m unittest tests.test_harness_clear_task -v` → 8/8 pass
- [ ] `python -m unittest discover tests -v` → all existing tests still pass
- [ ] `python -c "import interaction.cli"` → no errors
- [ ] Manual smoke test (Task 4) → new task ID after `/clear`, task gone after restart
- [ ] No stray TODOs in modified files
- [ ] All commits use `Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>`