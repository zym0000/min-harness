from harness import Harness,asyncio
from event.event import EventType,LoopEvent

from typing import Callable,Any

async def _to_thread(func: Callable, *args) -> Any:
    """显式同步桥接：标准库阻塞 I/O 的异步包装。
    生产环境应优先使用原生异步库（如 aiofiles、aiosqlite、httpx）。 """
    if hasattr(asyncio, "to_thread"):
        return await asyncio.to_thread(func, *args)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, *args)

class UserInterface:
    def __init__(self, harness: 'Harness'):
        self.harness = harness
        self._running = True

    async def start_interactive(self):
        print("=" * 60)
        print("Async Harness")
        print("命令: /cancel <task_id> | /status | /quit")
        print("=" * 60)

        while self._running:
            try:
                user_input = await _to_thread(input, "\nUser: ")
                user_input = user_input.strip()

                if not user_input:
                    continue

                if user_input.startswith("/"):
                    await self._handle_command(user_input)
                else:
                    await self._submit_task(user_input)

            except EOFError:
                break
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break

    async def _submit_task(self, user_input: str):
        task_id, event_generator = await self.harness.submit_task(user_input)
        print(f"\n 任务已提交: {task_id}")
        print("执行中...（可输入 /cancel {task_id} 取消）")

        try:
            async for event in event_generator:
                self._render_event(event)
        except asyncio.CancelledError:
            print(f"\n 任务 {task_id} 已被取消")
        except Exception as e:
            print(f"\n 任务异常: {e}")

    def _render_event(self, event: LoopEvent):
        et = event.event_type
        print(et.name)
        if et == EventType.THINKNING_STARTED:
            print(f"   Step {event.step_num}: 思考中...")
        elif et == EventType.TOOL_CALL_PARSED:
            print(f"   调用工具: {event.tool_name}")
        elif et == EventType.TOOL_EXECUTION_STARTED:
            print(f"    执行中...")
        elif et == EventType.TOOL_EXECUTION_COMPLETED:
            latency = event.data.get("latency_ms", "unknown") if event.data else "unknown"
            print(f"   执行完成 ({latency}ms)")
        elif et == EventType.TOOL_EXECUTION_FAILED:
            print(f"   执行失败: {event.content}")
        elif et == EventType.TOOL_RETRY_SCHEDULED:
            attempt = event.data.get("attempt", "?") if event.data else "?"
            print(f"   计划重试... (第{attempt}次)")
        elif et == EventType.TOOL_FALLBACK_TRIGGERED:
            print(f"   触发备用工具")
        elif et == EventType.FINAL_ANSWER:
            print(f"\n 最终答案:\n{event.content}")
        elif et == EventType.TASK_COMPLETED:
            print(f"\n 任务完成")
        elif et == EventType.TASK_FAILED:
            print(f"\n 任务失败: {event.content}")
        elif et == EventType.TASK_CANCELLED:
            print(f"\n 任务已取消")
        elif et == EventType.NEED_APPROVAL:
            print(f"\n  需要您的确认: {event.content}")

    async def _handle_command(self, command: str):
        parts = command.split()
        cmd = parts[0].lower()

        if cmd == "/quit" or cmd == "/q":
            self._running = False
            print("👋 再见！")

        elif cmd == "/status":
            states = await self.harness.get_all_task_states()
            print(f"\n 当前任务: {len(states)}")
            for tid, state in states.items():
                print(f"  {tid}: {state.status.name} (Step {state.current_step})")

        elif cmd == "/cancel" and len(parts) > 1:
            task_id = parts[1]
            success = await self.harness.cancel_task(task_id)
            print(f"\n{'' if success else ' '} 取消任务 {task_id}")

        else:
            print(f"\n 未知命令: {cmd}")