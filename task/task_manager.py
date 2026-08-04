import asyncio
import random
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from context_message import ContextManager
from task.task_defined import ContinueResult, TaskState, TaskStatus
from task.task_store import TaskStore

__all__ = ["TaskManager", "TaskStatus", "ContinueResult", "TaskState", "ContextManager"]

class TaskManager:
    def __init__(self, max_history: int = 100, store: Optional[TaskStore] = None):
        self.tasks: Dict[str, TaskState] = {}
        self.async_tasks: Dict[str, asyncio.Task] = {}
        self.max_history = max_history
        self.store = store
        self._guard = asyncio.Lock()
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    # ------------------------------------------------------------ 基础操作

    async def create_task(self, user_input: str, system_prompt: str):
        task_id = f"{time.time() * 1000}_{random.randint(1000, 9999)}"
        state = TaskState(
            task_id=task_id,
            user_input=user_input,
            task_status=TaskStatus.PENDING,
            current_step=0,
            messages=[{"role": "user", "content": user_input}],
            system_prompt=system_prompt,
        )
        async with self._guard:
            self.tasks[task_id] = state
        await self._persist(state)
        return task_id

    async def get_state(self, task_id) -> Optional[TaskState]:
        return self.tasks.get(task_id)

    async def register_task(self, task_id, async_task: asyncio.Task):
        async with self._guard:
            self.async_tasks[task_id] = async_task

    async def unregister_task(self, task_id):
        async with self._guard:
            self.async_tasks.pop(task_id, None)

    async def update_state(self, task_id, **kwargs):
        async with self._locks[task_id]:
            state = self.tasks.get(task_id)
            if not state:
                return
            for key, value in kwargs.items():
                if key == "status":
                    key = "task_status"  
                if not hasattr(state, key):
                    raise TypeError(f"update_state 收到未知字段: {key}")
                setattr(state, key, value)
            state.updated_at = time.time()
            snapshot = state.to_checkpoint()
        await self._persist_snapshot(snapshot)

    async def delete_task(self, task_id):
        async with self._guard:
            self.async_tasks.pop(task_id, None)
            existed = self.tasks.pop(task_id, None) is not None
        if self.store:
            await self.store.delete(task_id)
        return existed

    async def get_all_tasks(self):
        return dict(self.tasks)

    async def append_msg(self, task_id: str, messages: List[Dict[str, str]]):
        async with self._locks[task_id]:
            state = self.tasks.get(task_id)
            if not state:
                return
            state.messages.extend(messages)
            state.updated_at = time.time()
            snapshot = state.to_checkpoint()
        await self._persist_snapshot(snapshot)

    async def compress_messages(self,
                                task_id: str,
                                llm: Optional[Any],
                                context_message: ContextManager):
        """非破坏性压缩：返回"发给 LLM 的视图"，不改写 canonical 历史。"""
        async with self._locks[task_id]:
            state = self.tasks.get(task_id)
            if not state:
                return []
            system_prompt = state.system_prompt
            history_message = list(state.messages)          # 快照

        # prepare_message 可能触发记忆提取（写 task_state 的记忆字段），
        # 在锁外执行（LLM 调用耗时长，持锁会阻塞同任务的其他写入）
        messages = await context_message.prepare_message(
            system_prompt=system_prompt,
            history=history_message,
            llm_client=llm,
            task_state=state)
        return messages

    async def drain_pending_input(self, task_id: str):
        async with self._locks[task_id]:
            state = self.tasks.get(task_id)
            if not state or not state.pending_input:
                return []
            inputs = list(state.pending_input)
            state.pending_input.clear()
            return inputs

    async def continue_task(self, task_id: str, user_input: str):
        async with self._locks[task_id]:
            state = self.tasks.get(task_id)
            if not state:
                return ContinueResult.FAILED

            if state.task_status in {TaskStatus.COMPLETED, TaskStatus.FAILED,
                                     TaskStatus.CANCELLED}:
                state.task_status = TaskStatus.RUNNING
                state.messages.append({"role": "user", "content": user_input})
                state.updated_at = time.time()
                snapshot = state.to_checkpoint()
                result = ContinueResult.ACTIVATED

            elif state.task_status in {TaskStatus.RUNNING, TaskStatus.PAUSED}:
                if task_id not in self.async_tasks:
                    # 进程重启后的孤儿任务：没有活着的执行者，重新激活而非排队
                    state.task_status = TaskStatus.RUNNING
                    state.messages.append({"role": "user", "content": user_input})
                    state.updated_at = time.time()
                    snapshot = state.to_checkpoint()
                    result = ContinueResult.ACTIVATED
                else:
                    state.pending_input.append(user_input)
                    state.updated_at = time.time()
                    snapshot = state.to_checkpoint()
                    result = ContinueResult.QUEUED
            else:
                return ContinueResult.FAILED

        await self._persist_snapshot(snapshot)
        return result

    async def _persist(self, state: TaskState) -> None:
        await self._persist_snapshot(state.to_checkpoint())

    async def _persist_snapshot(self, snapshot: Dict[str, Any]) -> None:
        if self.store:
            await self.store.save(snapshot)

    async def restore(self) -> int:
        """从 store 恢复全部任务。返回恢复的任务数。"""
        if not self.store:
            return 0
        checkpoints = await self.store.load_all()
        async with self._guard:
            for data in checkpoints:
                state = TaskState.from_checkpoint(data)
                self.tasks[state.task_id] = state
        return len(checkpoints)