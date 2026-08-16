# `/clear` Command Redesign — Design Spec

**Date:** 2026-08-16
**Status:** Approved (verbal approval from user)
**Scope:** Modify the existing `/clear` CLI command to clear screen AND clear current task context memory.

## 1. Motivation

The current `/clear` command (`interaction/cli.py:226-227`) only calls `os.system('cls')`. It does not affect any agent state — task messages, LLM-extracted cumulative memory, persisted SQLite checkpoints, or harness-level caches all remain intact. The user wants `/clear` to behave like Claude Code's `/clear`: a fresh start within the same CLI session.

The design decision was reached through brainstorming with the user, who selected:
- Scope: **only the current task** (messages + cumulative memory), not all tasks.
- Running-task handling: **cancel first, then clear** (no race window by user choice).
- SQLite: **delete the row entirely**, next user input starts a fresh `task_id`.

## 2. Behavior

When the user types `/clear` at the CLI prompt:

1. **Always:** Clear the terminal screen (`os.system('cls')` on Windows, `clear` elsewhere).
2. **If `self.task_id is None`:** Print `no active task — screen cleared` and stop.
3. **Otherwise:**
   - Set cancel event for the running loop (`Harness.cancel_task`).
   - Wait up to 5 seconds for the loop to exit, polling `_loops` dict.
   - Delete the SQLite checkpoint row and the in-memory `TaskState`.
   - Pop entries from `cancel_events`, `approval_gates`, `_loops`, `_loop_tools`, `_run_results`.
   - Drop all sub-agent states whose `parent_task_id == task_id`.
   - Set `self.task_id = None`.
   - Print `task cleared: <task_id_prefix>…`.

## 3. Architecture

A new method `Harness.clear_task(task_id)` encapsulates all cleanup logic. The CLI command becomes a thin wrapper.

### 3.1 New Method: `Harness.clear_task`

**Location:** `harness.py`, near the existing `cancel_task` method.

**Signature:**
```python
async def clear_task(self, task_id: str) -> bool:
```

**Returns:** `True` if cleanup happened; `False` if `task_id` was unknown.

**Algorithm:**
1. **Early exit:** If `task_id not in self.task_manager.tasks` and `task_id not in self.cancel_events`, return `False`.
2. **Cancel:** `await self.cancel_task(task_id)` — sets the cancel event and cancels in-flight tool executions via the engine. Idempotent.
3. **Wait for loop exit:** Loop up to 50 iterations × 100 ms (`asyncio.sleep(0.1)`). Exit early when `task_id not in self._loops`. The wrapper's `finally` block pops `_loops[task_id]` when the loop terminates, so this is the natural signal.
4. **Delete state:** `await self.task_manager.delete_task(task_id)` — existing method; deletes the SQLite row and pops `tasks[task_id]`.
5. **Defensive cleanup:** `pop(task_id, None)` on `cancel_events`, `approval_gates`, `_loops`, `_loop_tools`, `_run_results`. These are independent of step 4 because `_task_wrapper`'s `finally` block only pops some of them, and may have already done so before our timeout fired.
6. **Sub-agent cleanup:** Iterate `self._subagent_states`; for every state whose `parent_task_id == task_id`, pop it from the dict. No cascade — orphaned grandchildren are out of scope.

### 3.2 CLI Wrapper: `/clear`

**Location:** `interaction/cli.py:_cmd`, replacing the existing `/clear` branch (line 226).

```python
elif c == "/clear":
    os.system("clear" if os.name != "nt" else "cls")
    tid = self.task_id
    if not tid:
        print(f"  {D}no active task — screen cleared{R}")
        continue
    cleared = await self.harness.clear_task(tid)
    self.task_id = None
    if cleared:
        print(f"  {YEL}task cleared: {tid[:12]}…{R}")
    else:
        print(f"  {D}task already gone — screen cleared{R}")
```

### 3.3 Help Text Update

`interaction/cli.py:32`:

```
/clear      clear screen + reset current task memory
```

## 4. Error Handling

| Scenario | Behavior |
|---|---|
| `task_id is None` | Only screen clears; no error. |
| Loop exits before 5 s | Cleanup proceeds immediately. |
| Loop still running after 5 s | Cleanup proceeds anyway. `pop(..., None)` is idempotent, so the wrapper's eventual `finally` is harmless even if it runs after step 5. Worst case: a stale `cancel_event` or `_loop_tools` entry persists until the wrapper's finally runs (seconds at most). |
| `TaskManager.delete_task` raises (e.g., SQLite I/O error) | Exception propagates up through `_cmd`. The user sees a Python traceback. Future work: catch and print a friendly error. Not in scope. |
| `/clear` invoked twice in a row | First call clears. Second call sees `self.task_id is None` and only clears the screen. |
| `/clear` while no chat iteration is active (task is RUNNING but CLI is at prompt) | `_chat` holds the generator; cancel event propagates; loop's finally block runs in the background; our wait loop catches the exit. |

## 5. Concurrency

The harness runs loops as async generators consumed by `_chat`. The `clear_task` flow does NOT call `gen.aclose()` on the generator owned by `_chat`. Instead, it relies on the cancel event propagating to the `AgentLoop`, which exits its state machine and triggers `_task_wrapper`'s `finally` block. This is the same mechanism `cancel_task` already uses.

The 5-second wait loop polls `_loops` for absence of `task_id`. Because `_task_wrapper`'s `finally` block does `self._loops.pop(task_id, None)`, the wait loop terminates as soon as the wrapper has begun cleanup. This is correct under the harness's existing invariant: `_loops` always reflects "loops that haven't yet finalized."

## 6. Testing

### 6.1 Unit Test: `Harness.clear_task`

**Location:** `tests/test_harness_clear_task.py` (new file).

Test cases:
1. **Unknown task_id** → returns `False`, no exception, no state mutation.
2. **Task in PENDING (never ran)** → returns `True`, state dicts cleaned, SQLite row deleted.
3. **Task RUNNING** → cancel event set before sleep, loop finalizes within 5 s, all cleanup happens.
4. **Sub-agent states for parent** → removed; sub-agent states for OTHER parents untouched.
5. **Idempotent re-call** → second call with same `task_id` returns `False`.
6. **Metrics NOT cleared** — call `clear_task`, assert `harness.metrics` unchanged.
7. **`prompt_toolkit` input history untouched** — out of scope; assert no code path touches the prompt session.

### 6.2 Integration Test: CLI `/clear`

**Location:** `tests/interaction/test_cli_clear.py` (new).

Spawn the CLI as a subprocess (or use `prompt_toolkit`'s testing utilities if available). Send `/clear` via stdin. Assert:
- Screen-clearing escape sequence was emitted.
- When no task: `no active task — screen cleared` printed.
- When task present: `task cleared: <prefix>…` printed and `agent_state.db` no longer contains the row.

### 6.3 Manual Smoke Test

`python main.py` → run a task → type `/clear` → confirm:
- Screen clears.
- `task cleared:` line printed.
- Type a follow-up question → new `task_id` printed (different from before).
- Restart the CLI → previous task does NOT appear in `/status`.

## 7. Out of Scope

- Clearing global `MetricsControll`.
- Clearing `prompt_toolkit` input history (that's input history, not agent memory).
- Restarting the harness or resetting `subagent/runner.py` global state.
- Adding a confirmation prompt (Claude Code's `/clear` doesn't ask either).
- A new `/reset` or `/clearall` command for nuclear wipe.
- Error UI for `TaskManager.delete_task` failures.

## 8. Files Touched

| File | Change |
|---|---|
| `harness.py` | Add `Harness.clear_task` method (~30 lines). |
| `interaction/cli.py` | Replace `/clear` branch (~10 lines); update help text. |
| `tests/test_harness_clear_task.py` | New unit tests. |
| `tests/interaction/test_cli_clear.py` | New CLI integration tests. |